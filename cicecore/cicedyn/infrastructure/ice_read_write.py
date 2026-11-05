"""Routines for opening, reading and writing external files

author for JAX version: Xinyuan Zhu
XZ: converted from F90 to Python

2007: netcdf versions added by Alison McLaren & Ann Keen, Met Office
2006: Converted to free source form (F90) by Elizabeth Hunke
2004: Block structure added by William Lipscomb, LANL
author: Tony Craig, NCAR

Direct-access records (``ida4``, ``ida8``, ``rda4``, ``rda8``) have no
sequential record markers. One record is ``nx * ny`` values, i index
fastest. The byte order is chosen from the first record of that file.
``ruf8`` is one sequential unformatted record. NetCDF variables are read
with the file's first dimension as i, matching ``ice_read_global_nc``.
"""

import netCDF4
import numpy as np

from cicecore.cicedyn.general import ice_init
from cicecore.shared import ice_fileunits
from cicecore.shared import ice_domain_size
from cicecore.shared.ice_constants import spval_dbl
from cicecore.cicedyn.infrastructure.comm.serial.ice_boundary import (
    scatter_global,
    scatter_global_ext,
)
from cicecore.shared.ice_kinds_mod import dbl_kind

_endian = {}

# format for input array
# (real/integer, 4-byte/8-byte)
# nbits indicates whether the file is sequential or direct access.
_DIRECT = {
    "ida4": ("i4", 4, "kmt"),
    "ida8": ("i8", 8, "kmt"),
    "rda4": ("f4", 4, "real"),
    "rda8": ("f8", 8, "real"),
}


def ice_read_global(path, nrec, atype, diag=True, shape=None):
    """Read an unformatted file
    Just like ice_read except that it returns a global array.
    work_g is a real array, atype indicates the format of the data

    Adapted by William Lipscomb, LANL, from ice_read

    Read one global record. Default shape is (nx_global, ny_global).
    """
    # nu: unit number
    # nrec: record number (0 for sequential access)
    # work_g: output array (real, 8-byte)
    # atype: format for input array
    #        (real/integer, 4-byte/8-byte)
    # diag: if true, write diagnostic output
    if shape is None:
        # output array (real, 8-byte)
        nx, ny = int(ice_domain_size.nx_global), int(ice_domain_size.ny_global)
    else:
        nx, ny = int(shape[0]), int(shape[1])
    count = nx * ny
    #-------------------------------------------------------------------
    # Read global array according to format atype
    #-------------------------------------------------------------------
    if atype == "ruf8":
        # sequential access
        # record number (0 for sequential access)
        # no. of bits per variable (0 for sequential access)
        nbytes = count * 8
        with open(path, "rb") as stream:
            payload = None
            for rec in range(1, nrec + 1):
                mark = stream.read(4)
                if len(mark) < 4:
                    raise ValueError(f"(ice_read) short sequential record {rec} in {path}")
                length = None
                marker = 4
                for trial in (">", "<"):
                    size = int(np.frombuffer(mark, dtype=np.dtype(trial + "i4"))[0])
                    if size == nbytes:
                        length = size
                        break
                if length is None:
                    extra = stream.read(4)
                    mark = mark + extra
                    marker = 8
                    for trial in (">", "<"):
                        size = int(np.frombuffer(mark, dtype=np.dtype(trial + "i8"))[0])
                        if size == nbytes:
                            length = size
                            break
                if length != nbytes:
                    raise ValueError(f"(ice_read) bad sequential marker in {path}")
                data = stream.read(nbytes)
                stream.read(marker)
                if rec == nrec:
                    payload = data
            if payload is None or len(payload) != nbytes:
                raise ValueError(f"(ice_read) missing sequential record {nrec} in {path}")
            raw = payload
        if path in _endian:
            endian = _endian[path]
        else:
            endian = None
            for trial in (">", "<"):
                sample = np.frombuffer(raw, dtype=np.dtype(trial + "f8"))
                if (
                    sample.size > 0
                    and np.isfinite(sample).all()
                    and np.max(np.abs(sample)) <= 1.0e10
                    and endian is None
                ):
                    endian = trial
            if endian is None:
                raise ValueError(f"(ice_read) could not decode {path}")
            _endian[path] = endian
        values = np.frombuffer(raw, dtype=np.dtype(endian + "f8"))
    elif atype in _DIRECT:
        # direct access
        # record number (0 for sequential access)
        # use nbytes to compute RecSize.
        # this prevents integer overflow with large global grids using nbits
        # nx*ny*nbits > 2^31 -1 (i.e., global grid 9000x7054x64)
        code, width, kind = _DIRECT[atype]
        nbytes = count * width
        with open(path, "rb") as stream:
            stream.seek((nrec - 1) * nbytes)
            raw = stream.read(nbytes)
        if len(raw) != nbytes:
            raise ValueError(
                f"(ice_read) short record {nrec} in {path}: "
                f"got {len(raw)} bytes, expected {nbytes}"
            )
        if path in _endian:
            endian = _endian[path]
        else:
            endian = None
            for trial in (">", "<"):
                sample = np.frombuffer(raw, dtype=np.dtype(trial + code))
                if kind == "kmt":
                    ok = (
                        sample.size > 0
                        and np.all(sample >= 0)
                        and np.all(sample <= 500)
                        and np.any(sample > 0)
                    )
                else:
                    ok = (
                        sample.size > 0
                        and np.isfinite(sample).all()
                        and np.max(np.abs(sample)) <= 1.0e10
                    )
                if ok and endian is None:
                    endian = trial
            if endian is None:
                raise ValueError(f"(ice_read) could not decode {path}")
            _endian[path] = endian
        values = np.frombuffer(raw, dtype=np.dtype(endian + code))
    else:
        raise ValueError(f"(ice_read_global) unknown atype {atype}")
    work = np.array(values, dtype=dbl_kind).reshape((nx, ny), order="F")
    #-------------------------------------------------------------------
    # optional diagnostics
    #-------------------------------------------------------------------
    if diag:
        # min, max values and sum of input array
        kept = work[work != spval_dbl]
        ice_fileunits.nu_diag.write(
            f" (ice_read_global) read_global {path} {nrec} "
            f"{np.min(work)} {np.max(kept) if kept.size else np.max(work)} "
            f"{np.sum(kept)}\n"
        )
        ice_fileunits.nu_diag.flush()
    return work


def ice_read(path, nrec, array, atype, diag, field_loc, field_type):
    """Read an unformatted file and scatter to processors.
    work is a real array, atype indicates the format of the data.
    If the optional variables field_loc and field_type are present,
    the ghost cells are filled using values from the global array.
    This prevents them from being filled with zeroes in land cells
    (subroutine ice_HaloUpdate need not be called).

    author: Tony Craig, NCAR

    Read one direct-access record and scatter it onto a halo array.
    """
    # nrec: record number (0 for sequential access)
    # work: output array (real, 8-byte)
    # atype: format for input array
    #        (real/integer, 4-byte/8-byte)
    # diag: if true, write diagnostic output
    # field_loc: location of field on staggered grid
    # field_type: type of field (scalar, vector, angle)
    work = ice_read_global(path, nrec, atype, diag)
    #-------------------------------------------------------------------
    # Scatter data to individual processors.
    # NOTE: Ghost cells are not updated unless field_loc is present.
    #-------------------------------------------------------------------
    scatter_global(array, work, field_loc, field_type)
    return array


def ice_read_global_nc(path, varname, diag=True, shape=None):
    """Read a netcdf file.
    Just like ice_read_nc except that it returns a global array.
    work_g is a real array

    Adapted by William Lipscomb, LANL, from ice_read
    Adapted by Ann Keen, Met Office, to read from a netcdf file

    Read one 2D netCDF variable. The file's first dimension is i.
    """
    # fid: file id
    # nrec: record number
    # varname: field name in netcdf file
    # work_g: output array (real, 8-byte)
    # diag: if true, write diagnostic output
    if shape is None:
        # output array (real, 8-byte)
        nx, ny = int(ice_domain_size.nx_global), int(ice_domain_size.ny_global)
    else:
        nx, ny = int(shape[0]), int(shape[1])
    # Opens a netCDF file for reading
    # Adapted by Alison McLaren, Met Office from ice_open
    # filename: netCDF filename
    # fid: unit number
    # status variable from netCDF routine
    dataset = netCDF4.Dataset(path)
    try:
        #-------------------------------------------------------------
        # Find out ID of required variable
        #-------------------------------------------------------------
        # varid: netcdf id for field
        if varname not in dataset.variables:
            raise KeyError(f"(ice_read_global_nc) {path} has no variable {varname}")
        var = dataset.variables[varname]
        #--------------------------------------------------------------
        # Read global array
        #--------------------------------------------------------------
        # Get the data
        data = np.array(var[:], dtype=dbl_kind)
    finally:
        dataset.close()
    # Check var size : is var 2d ?
    # number of variable dimensions
    # Is work_g the same size as the variable?
    # size of dimension
    if data.ndim != 2 or tuple(data.shape) != (nx, ny):
        raise ValueError(
            f"(ice_read_global_nc) {varname} shape {getattr(data, 'shape', None)}; "
            f"expected ({nx}, {ny}) with i as the first dimension"
        )
    #-------------------------------------------------------------------
    # optional diagnostics
    #-------------------------------------------------------------------
    if diag:
        # min, max values and sum of input array
        kept = data[data != spval_dbl]
        ice_fileunits.nu_diag.write(
            f" (ice_read_global_nc) read_global {path} {varname} "
            f"{np.min(data)} {np.max(kept) if kept.size else np.max(data)} "
            f"{np.sum(kept)}\n"
        )
        ice_fileunits.nu_diag.flush()
    return data


def nc_has_var(path, varname):
    # Find out ID of required variable
    # varid: netcdf id for field
    # Opens a netCDF file for reading
    # Adapted by Alison McLaren, Met Office from ice_open
    # filename: netCDF filename
    # fid: unit number
    # status variable from netCDF routine
    dataset = netCDF4.Dataset(path)
    try:
        return varname in dataset.variables
    finally:
        dataset.close()


def mask_fieldname(path):
    """POP mask variable, ``kmt`` when that name exists, otherwise ``mask``."""
    # mask variable name might be kmt or mask, check both
    if nc_has_var(path, "kmt"):
        return "kmt"
    if nc_has_var(path, "mask"):
        return "mask"
    raise KeyError(f"(kmtmask) {path} contains neither kmt nor mask")


def ice_read_nc(path, varname, array, diag, field_loc, field_type, extended=False):
    """Read a netCDF file and scatter to processors.
    If the optional variables field_loc and field_type are present,
    the ghost cells are filled using values from the global array.
    This prevents them from being filled with zeroes in land cells
    (subroutine ice_HaloUpdate need not be called).

    Adapted by Alison McLaren, Met Office from ice_read

    Read a 2D netCDF field and scatter it. ``extended`` includes ghosts.
    """
    # fid: file id
    # nrec: record number
    # varname: field name in netcdf file
    # work: output array (real, 8-byte)
    # diag: if true, write diagnostic output
    # field_loc: location of field on staggered grid
    # field_type: type of field (scalar, vector, angle)
    # restart_ext: if true, read extended grid
    #-------------------------------------------------------------------
    # Scatter data to individual processors.
    # NOTE: Ghost cells are not updated unless field_loc is present.
    #-------------------------------------------------------------------
    if extended:
        # grid dimensions including ghost cells
        ghost = int(ice_domain_size.nghost)
        shape = (
            int(ice_domain_size.nx_global) + 2 * ghost,
            int(ice_domain_size.ny_global) + 2 * ghost,
        )
        work = ice_read_global_nc(path, varname, diag, shape)
        scatter_global_ext(array, work)
    else:
        work = ice_read_global_nc(path, varname, diag)
        scatter_global(array, work, field_loc, field_type)
    return array
