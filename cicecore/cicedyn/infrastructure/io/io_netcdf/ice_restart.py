"""NetCDF restart read and write for a single-process CICE run.

author for JAX version: Xinyuan Zhu
XZ: converted from F90 to Python

author David A. Bailey, NCAR

USE_NETCDF is the only path. MPI broadcast is one process. The unit number
passed to ``read_restart_field`` is unused, matching the Fortran netCDF
reader, which always uses the file opened by ``init_restart_read``.
Record index 0 means the first record when a time dimension is present.
Axis 0 of a 2D field is i, the same convention as ``ice_read_global_nc``.
"""

import netCDF4
import numpy as np

from cicecore.cicedyn.general import ice_init
from cicecore.cicedyn.infrastructure.comm.serial.ice_boundary import (
    scatter_global,
    scatter_global_ext,
)
from cicecore.shared import ice_calendar
from cicecore.shared import ice_domain_size
from cicecore.shared import ice_fileunits
from cicecore.shared.ice_constants import (
    field_loc_noupdate,
    field_type_noupdate,
    spval_dbl,
)
from cicecore.shared.ice_kinds_mod import dbl_kind

# open restart dataset. None until init_restart_read.
_dataset = None
_filename = ""
_write_filename = None
_write_fields = None


def init_restart_read(ice_ic=None):
    """Open a restart file and, when requested, replace the calendar.

    author David A. Bailey, NCAR
    """
    global _dataset, _filename

    if ice_ic is not None and str(ice_ic).strip() != "":
        filename = str(ice_ic).strip()
    else:
        pointer = str(ice_init.pointer_file).strip()
        with open(pointer, encoding="utf-8") as handle:
            filename = handle.readline().strip()
        ice_fileunits.nu_diag.write(f"Read {pointer}\n")

    _filename = filename
    ice_fileunits.nu_diag.write(f"Using restart dump= {filename}\n")
    if _dataset is not None:
        _dataset.close()
    _dataset = netCDF4.Dataset(filename)

    # for backwards compatibility, check nyr, month, and sec as well
    if bool(ice_init.use_restart_time):
        ice_calendar.istep0 = int(_attr("istep1"))
        ice_calendar.myear = int(_attr_fallback("myear", "nyr"))
        ice_calendar.mmonth = int(_attr_fallback("mmonth", "month"))
        ice_calendar.mday = int(_attr("mday"))
        ice_calendar.msec = int(_attr_fallback("msec", "sec"))
    # use namelist values if use_restart_time = F

    ice_calendar.istep1 = ice_calendar.istep0

    # if runid is bering then need to correct npt for istep0
    if str(ice_init.runid).strip() == "bering":
        ice_calendar.npt = int(ice_calendar.npt) - int(ice_calendar.istep0)
    ice_fileunits.nu_diag.flush()


def query_field(vname):
    """Inquire field existance.

    author T. Craig

    The unit number is unused. The open restart dataset is searched.
    """
    if _dataset is None:
        raise RuntimeError("(query_field) restart file is not open")
    return vname in _dataset.variables


def read_restart_field(
    work,
    vname,
    ndim3,
    diag,
    field_loc=None,
    field_type=None,
    nrec=0,
    atype="ruf8",
):
    """Read a single restart field.

    author David A. Bailey, NCAR

    ``work`` is written in place. ``ndim3 == 1`` is a horizontal field.
    ``ndim3 == ncat`` is one field per category. ``atype`` is accepted and
    ignored; values are stored as ``dbl_kind``.
    """
    subname = "(read_restart_field)"
    _ = atype
    if _dataset is None:
        raise RuntimeError(subname + " restart file is not open")
    if vname not in _dataset.variables:
        raise KeyError(subname + f" ERROR: Cannot find variable {vname}")

    ncat = int(ice_domain_size.ncat)
    extended = bool(ice_init.restart_ext)
    ghost = int(ice_domain_size.nghost)
    if extended:
        nx = int(ice_domain_size.nx_global) + 2 * ghost
        ny = int(ice_domain_size.ny_global) + 2 * ghost
    else:
        nx = int(ice_domain_size.nx_global)
        ny = int(ice_domain_size.ny_global)

    var = _dataset.variables[vname]
    data = np.array(var[:], dtype=dbl_kind)
    # Check nrec axis size. A 4th dimension is the record.
    rec = 0 if int(nrec) < 1 else int(nrec) - 1
    if data.ndim > 3:
        if rec >= data.shape[-1] and data.shape[0] != nx:
            raise RuntimeError(
                subname + f" ERROR: not enough records {vname} {int(nrec)} {data.shape[-1]}"
            )
        if data.shape[0] == nx and data.shape[1] == ny:
            data = data[:, :, :, rec]
        elif data.shape[-1] == nx and data.shape[-2] == ny:
            # (record, ncat or 1, ny, nx)
            slab = data[rec]
            data = np.moveaxis(slab, -1, 0)
            if data.ndim == 3:
                data = np.moveaxis(data, -1, 1)
        else:
            raise ValueError(subname + f" ERROR: Cannot get variable {vname} shape {data.shape}")

    if int(ndim3) == 1:
        planes = (_as_horizontal(data, nx, ny, subname, vname),)
    elif int(ndim3) == ncat:
        cube = _as_categories(data, nx, ny, ncat, subname, vname)
        planes = tuple(cube[:, :, n] for n in range(ncat))
    else:
        ice_fileunits.nu_diag.write(f"ndim3 not supported {int(ndim3)}\n")
        raise ValueError(subname + f" ndim3 not supported {int(ndim3)}")

    #-------------------------------------------------------------------
    # optional diagnostics
    #-------------------------------------------------------------------
    # Global array only. Ghost cells are filled by the scatter below.
    if diag:
        ice_fileunits.nu_diag.write(
            f"{subname} fid= 0, lnrec = {int(nrec):8d}, varname = {vname}\n"
        )
        for plane in planes:
            _write_range(plane, vname)
        ice_fileunits.nu_diag.flush()

    #-------------------------------------------------------------------
    # Scatter data to individual processors.
    # NOTE: Ghost cells are not updated unless field_loc is present.
    #-------------------------------------------------------------------
    if int(ndim3) == 1:
        _scatter_one(work, planes[0], extended, field_loc, field_type)
    else:
        for n, plane in enumerate(planes):
            _scatter_one(work[:, :, n], plane, extended, field_loc, field_type)
    return work


def final_restart():
    """Close the restart dataset.

    Fortran leaves the file open until a later ``final_restart``. Initialization
    calls this after the tracer reads so the handle is not kept for the run.
    """
    global _dataset
    if _dataset is not None:
        _dataset.close()
        _dataset = None


def init_restart_write(filename_spec=None):
    """Prepare the single-process netCDF restart and its pointer filename."""
    global _write_filename, _write_fields
    if filename_spec is None:
        date = (f"{int(ice_calendar.myear):04d}-{int(ice_calendar.mmonth):02d}-"
                f"{int(ice_calendar.mday):02d}-{int(ice_calendar.msec):05d}")
        stem = str(ice_init.restart_dir).strip() + str(ice_init.restart_file).strip() + "." + date
    else:
        stem = str(filename_spec).strip()
    _write_filename = stem + ".nc"
    _write_fields = {}


def write_restart_field(work, vname, diag=True):
    """Stage one horizontal or category field, preserving Fortran's i,j axes."""
    if _write_fields is None:
        raise RuntimeError("(write_restart_field) call init_restart_write first")
    data = np.asarray(work, dtype=dbl_kind)
    if data.ndim not in (2, 3):
        raise ValueError(f"(write_restart_field) unsupported shape for {vname}: {data.shape}")
    if bool(ice_init.restart_ext):
        slab = data.copy()
    else:
        ilo, ihi, jlo, jhi = ice_domain_size.physical_bounds()
        slab = data[ilo:ihi + 1, jlo:jhi + 1].copy()
    _write_fields[vname] = slab
    if diag:
        _write_range(slab, vname)


def final_restart_write():
    """Write the staged fields and close the restart before updating pointer."""
    global _write_filename, _write_fields
    if _write_filename is None or _write_fields is None:
        raise RuntimeError("(final_restart_write) no restart write is open")
    formats = {
        "cdf1": "NETCDF3_CLASSIC", "cdf2": "NETCDF3_64BIT_OFFSET",
        "cdf5": "NETCDF3_64BIT_DATA", "hdf5": "NETCDF4",
    }
    kind = str(ice_init.restart_format).strip()
    if kind not in formats:
        raise ValueError("(init_restart_write) restart_format not allowed: " + kind)
    nx = int(ice_domain_size.nx_global)
    ny = int(ice_domain_size.ny_global)
    if bool(ice_init.restart_ext):
        ghost = int(ice_domain_size.nghost)
        nx += 2 * ghost
        ny += 2 * ghost
    with netCDF4.Dataset(_write_filename, "w", format=formats[kind]) as nc:
        nc.setncattr("istep1", int(ice_calendar.istep1))
        nc.setncattr("myear", int(ice_calendar.myear))
        nc.setncattr("mmonth", int(ice_calendar.mmonth))
        nc.setncattr("mday", int(ice_calendar.mday))
        nc.setncattr("msec", int(ice_calendar.msec))
        nc.createDimension("ni", nx)
        nc.createDimension("nj", ny)
        nc.createDimension("ncat", int(ice_domain_size.ncat))
        for name, slab in _write_fields.items():
            # nf90 reverses dimension order in the on-disk C interface.
            dims = ("ncat", "nj", "ni") if slab.ndim == 3 else ("nj", "ni")
            options = {}
            if kind == "hdf5":
                level = int(ice_init.restart_deflate)
                if level > 0:
                    options.update(zlib=True, complevel=level, shuffle=False)
                chunks = ice_init.restart_chunksize
                if int(chunks[0]) > 0 and int(chunks[1]) > 0:
                    sizes = (min(int(chunks[1]), ny), min(int(chunks[0]), nx))
                    options["chunksizes"] = ((int(ice_domain_size.ncat),) + sizes
                                             if slab.ndim == 3 else sizes)
            var = nc.createVariable(name, "f8", dims, **options)
            var[:] = np.transpose(slab, (2, 1, 0)) if slab.ndim == 3 else slab.T
    pointer = str(ice_init.pointer_file).strip()
    # The standalone driver leaves ice_restart_shared.pointer_date false.
    with open(pointer, "w", encoding="utf-8") as stream:
        stream.write(_write_filename + "\n")
    ice_fileunits.nu_diag.write(
        f"Restart read/written {int(ice_calendar.istep1):8d} "
        f"{int(ice_calendar.myear):04d}-{int(ice_calendar.mmonth):02d}-"
        f"{int(ice_calendar.mday):02d}-{int(ice_calendar.msec):05d}\n"
    )
    _write_filename = None
    _write_fields = None


def _attr(name):
    if name not in _dataset.ncattrs():
        raise RuntimeError(f"(init_restart_read) ERROR: reading restart attribute {name}")
    return _dataset.getncattr(name)


def _attr_fallback(primary, fallback):
    if primary in _dataset.ncattrs():
        return _dataset.getncattr(primary)
    return _attr(fallback)


def _as_horizontal(data, nx, ny, subname, vname):
    if data.ndim == 3 and data.shape[-1] == 1 and data.shape[0] == nx:
        data = data[:, :, 0]
    if data.ndim == 3 and data.shape[0] == 1 and data.shape[-1] == nx:
        data = np.moveaxis(data[0], -1, 0)
    if tuple(data.shape) == (ny, nx):
        data = data.T
    if tuple(data.shape) != (nx, ny):
        raise ValueError(
            f"{subname} ERROR: Cannot get variable {vname} shape {getattr(data, 'shape', None)}; "
            f"expected ({nx}, {ny})"
        )
    return data


def _as_categories(data, nx, ny, ncat, subname, vname):
    if data.ndim == 3 and data.shape[0] == nx and data.shape[1] == ny and data.shape[2] == ncat:
        return data
    if data.ndim == 3 and data.shape[0] == ncat and data.shape[1] == ny and data.shape[2] == nx:
        return np.moveaxis(np.moveaxis(data, 0, -1), 0, 1)
    if data.ndim == 3 and data.shape[0] == ncat and data.shape[-1] == ny and data.shape[-2] == nx:
        return np.moveaxis(data, 0, -1)
    raise ValueError(
        f"{subname} ERROR: Cannot get variable {vname} shape {getattr(data, 'shape', None)}; "
        f"expected ({nx}, {ny}, {ncat})"
    )


def _scatter_one(dest, slab, extended, field_loc, field_type):
    if extended:
        scatter_global_ext(dest, slab)
        return
    if field_loc is None:
        scatter_global(dest, slab, field_loc_noupdate, field_type_noupdate)
    else:
        scatter_global(dest, slab, field_loc, field_type)


def _write_range(slab, vname):
    # optional
    # missing value. NaNs are excluded the same way as a missing-value mask.
    finite = np.isfinite(slab) & (slab != spval_dbl)
    if not np.any(finite):
        amin = amax = asum = spval_dbl
    else:
        kept = slab[finite]
        amin = np.min(kept)
        amax = np.max(kept)
        asum = np.sum(kept)
    ice_fileunits.nu_diag.write(
        f"(read_restart_field)  min, max, sum = {amin} {amax} {asum} {vname}\n"
    )
