"""Reads and interpolates forcing data for atmosphere and ocean quantities.

author for JAX version: Xinyuan Zhu
XZ: converted from F90 to Python

authors: Elizabeth C. Hunke and William H. Lipscomb, LANL

2004 WHL: Block structure added
2005 WHL: ECMWF option added
2006 ECH: LY option added
2006 WHL: Module name changed from ice_flux_in
2006 ECH: Fixed bugs, rearranged routines, edited comments, etc.
          Added NCAR ocean forcing file
          Converted to free source form (F90)
2007: netcdf version of read_data added by Alison McLaren, Met Office

One grid, no block index. ``restore_bgc`` is copied onto ``ice_init`` by
``input_zbgc``. It stays false until that namelist is read.

Ported: ``alloc_forcing``, ``init_forcing_atmo``, ``init_forcing_ocn``,
``get_forcing_atmo``, ``get_forcing_ocn``, ``ocn_freezing_temperature``,
and the remaining atmosphere, ocean, wave, and snow-table routines in
``ice_forcing.F90``. Record-indexed netCDF reads live in this module.
``icepack_init_wave`` is not a Python function; ``_init_wave_bins`` applies
its frequency table. ``get_forcing_ocn`` leaves the hycom time update
commented out, matching the Fortran.
"""

import math
import os

import netCDF4
import numpy as np

from cicecore.cicedyn.general import ice_flux
from cicecore.cicedyn.general import ice_init
from cicecore.cicedyn.general import ice_state
from cicecore.cicedyn.infrastructure import ice_grid
from cicecore.cicedyn.infrastructure import ice_read_write
from cicecore.cicedyn.infrastructure.comm.serial.ice_boundary import (
    global_indices,
    ice_halo_update,
    scatter_global,
)
from cicecore.shared import ice_arrays_column
from cicecore.shared import ice_calendar
from cicecore.shared import ice_domain_size
from cicecore.shared import ice_fileunits
from cicecore.shared.ice_constants import (
    c0,
    c1,
    c2,
    c3,
    c4,
    c5,
    c8,
    c10,
    c3600,
    c12,
    c15,
    c180,
    c360,
    c365,
    c1000,
    cm_to_m,
    field_loc_NEcorner,
    field_loc_center,
    field_type_scalar,
    field_type_vector,
    p001,
    p01,
    p1,
    p2,
    p5,
    p6,
    spval_dbl,
)
from cicecore.shared.ice_kinds_mod import dbl_kind
from icepack.columnphysics import icepack_parameters
from icepack.columnphysics import icepack_tracers
from icepack.columnphysics.icepack_therm_shared import icepack_sea_freezing_temperature

# PRIVATE:
mixed_layer_depth_default = dbl_kind(20.0)  # default mixed layer depth in m
local_debug = False  # local debug flag

nfld = 8  # number of fields to search for in forcing file

# as in the dummy atm (latm)
frcvdr = dbl_kind(0.28)  # frac of incoming sw in vis direct band
frcvdf = dbl_kind(0.24)  # frac of incoming sw in vis diffuse band
frcidr = dbl_kind(0.31)  # frac of incoming sw in near IR direct band
frcidf = dbl_kind(0.17)  # frac of incoming sw in near IR diffuse band

ocn_frc_m = None  # ocn data for 12 months

trest = None  # restoring time scale (sec)

cldf = None
fsw_data = None
cldf_data = None
fsnow_data = None
Tair_data = None
uatm_data = None
vatm_data = None
wind_data = None
strax_data = None
stray_data = None
Qa_data = None
rhoa_data = None
flw_data = None
sst_data = None
sss_data = None
uocn_data = None
vocn_data = None
sublim_data = None
frain_data = None
topmelt_data = None
botmelt_data = None
wave_spectrum_data = None  # field values at 2 temporal data points
topmelt_file = None
botmelt_file = None

sss_file = None
sst_file = None

# current year in forcing cycle, varying during the run
fyear = 0
fyear_final = 0  # last year in cycle, computed at init

c1intp = c0  # interpolation coefficients
c2intp = c0

oldrecnum = 0  # old record number (save between steps)
oldrecnum4X = 0

# prints forcing debugging output if true. Starts from the namelist and
# can be turned on inside the read routines.
debug_forcing = False

uwind_file = ""  # this is also used a generic file containing all fields for JRA55
vwind_file = ""
wind_file = ""
strax_file = ""
stray_file = ""
tair_file = ""
humid_file = ""
rhoa_file = ""
fsw_file = ""
flw_file = ""
rain_file = ""
sublim_file = ""
snow_file = ""

# remember prior JRA55 values to reduce reading
# first dim is yr, recnum; second dim is data1, data2
frec_info = np.full((2, 2), -99, dtype=np.int32)
_jra_dataset = None
_jra_dataset_path = None

jday_atm = None  # jday time vector from atm forcing files
Njday_atm = 0  # Number of atm forcing timesteps

# ISPOL point values saved between steps
Tair_data_p = np.zeros(2, dtype=dbl_kind)
Qa_data_p = np.zeros(2, dtype=dbl_kind)
fsnow_data_p = np.zeros(2, dtype=dbl_kind)
fsw_data_p = np.zeros(2, dtype=dbl_kind)
flw_data_p = np.zeros(2, dtype=dbl_kind)
uatm_data_p = np.zeros(2, dtype=dbl_kind)
vatm_data_p = np.zeros(2, dtype=dbl_kind)


def _zero(shape):
    return np.zeros(shape, dtype=dbl_kind)


def alloc_forcing():
    """Allocate space for all variables."""
    global cldf, fsw_data, cldf_data, fsnow_data, Tair_data, uatm_data, vatm_data
    global wind_data, strax_data, stray_data, Qa_data, rhoa_data, flw_data
    global sst_data, sss_data, uocn_data, vocn_data, sublim_data, frain_data
    global topmelt_data, botmelt_data, ocn_frc_m, topmelt_file, botmelt_file
    global wave_spectrum_data
    subname = "(alloc_forcing)"
    if local_debug:
        ice_fileunits.nu_diag.write(f" {subname} fdbg start\n")

    close_forcing()
    horiz = ice_domain_size.horizontal_shape()
    ncat = int(ice_domain_size.ncat)
    nfreq = int(ice_init.nfreq)
    pair = horiz + (2,)
    cldf = _zero(horiz)  # cloud fraction
    fsw_data = _zero(pair)  # field values at 2 temporal data points
    cldf_data = _zero(pair)
    fsnow_data = _zero(pair)
    Tair_data = _zero(pair)
    uatm_data = _zero(pair)
    vatm_data = _zero(pair)
    wind_data = _zero(pair)
    strax_data = _zero(pair)
    stray_data = _zero(pair)
    Qa_data = _zero(pair)
    rhoa_data = _zero(pair)
    flw_data = _zero(pair)
    sst_data = _zero(pair)
    sss_data = _zero(pair)
    uocn_data = _zero(pair)
    vocn_data = _zero(pair)
    sublim_data = _zero(pair)
    frain_data = _zero(pair)
    topmelt_data = _zero(horiz + (2, ncat))
    botmelt_data = _zero(horiz + (2, ncat))
    ocn_frc_m = _zero(horiz + (nfld, 12))  # ocn data for 12 months
    topmelt_file = [""] * ncat
    botmelt_file = [""] * ncat
    wave_spectrum_data = _zero(horiz + (nfreq, 2))


def init_forcing_ocn(dt):
    """Set sea surface salinity and freezing temperature to annual mean value
    using a 12-month climatology.
    Read sst data for current month, and adjust sst based on freezing
    temperature.  No interpolation in time.

    Note: SST is subsequently prognosed if CICE is run
    with a mixed layer ocean (oceanmixed_ice = T), and can be
    restored to data (restore_ocn = T).
    """
    global trest, sss_file, sst_file, debug_forcing
    subname = "(init_forcing_ocn)"
    if local_debug:
        ice_fileunits.nu_diag.write(f" {subname} fdbg start\n")

    # secday is the module parameter. The query only reads it.
    secday = icepack_parameters.secday
    # call icepack_warnings_flush / abort_ice are not ported.

    alloc_forcing()

    sst_data[...] = c0
    sss_data[...] = c0
    uocn_data[...] = c0
    vocn_data[...] = c0

    nbits = 64  # double precision data
    # ice_open's nbits is the direct-access width. The reader encodes that as 'rda8'.
    _ = nbits

    # restore_bgc is set in input_zbgc
    restore_bgc = bool(getattr(ice_init, "restore_bgc", False))
    if bool(ice_init.restore_ocn) or restore_bgc:
        if int(ice_init.trestore) == 0:
            trest = dbl_kind(dt)  # use data instantaneously
        else:
            trest = dbl_kind(ice_init.trestore) * secday  # seconds

    #-------------------------------------------------------------------
    # Sea surface salinity (SSS)
    # initialize to annual climatology created from monthly data
    #-------------------------------------------------------------------
    ocn_data_type = str(ice_init.ocn_data_type).strip()
    ocn_data_dir = str(ice_init.ocn_data_dir).strip()
    nu = ice_fileunits.nu_diag
    debug_forcing = bool(ice_init.debug_forcing)

    if ocn_data_type == "clim":
        sss_file = ocn_data_dir + "/sss.mm.100x116.da"  # gx3 only
        # if (my_task == master_task): one process.
        nu.write(" \n")
        nu.write(" SSS climatology computed from:\n")
        nu.write(f" {sss_file}\n")
        nu.flush()

        ice_flux.sss[...] = c0
        work1 = _zero(ice_domain_size.horizontal_shape())
        for k in range(1, 13):  # loop over 12 months
            ice_read_write.ice_read(
                sss_file,
                k,
                work1,
                "rda8",
                debug_forcing,
                field_loc_center,
                field_type_scalar,
            )
            ice_flux.sss[...] = ice_flux.sss + work1

        ice_flux.sss[...] = ice_flux.sss / c12  # annual average
        ice_flux.sss[...] = np.maximum(ice_flux.sss, c0)

        ocn_freezing_temperature()

        #-------------------------------------------------------------------
        # Sea surface temperature (SST)
        # initialize to data for current month
        #-------------------------------------------------------------------
        if int(ice_domain_size.nx_global) == 320:  # gx1
            sst_file = ocn_data_dir + "/sst_clim_hurrell.dat"
        else:  # gx3
            sst_file = ocn_data_dir + "/sst.mm.100x116.da"

        nu.write(" \n")
        nu.write(f" Initial SST file: {sst_file}\n")
        nu.flush()

        ice_read_write.ice_read(
            sst_file,
            int(ice_calendar.mmonth),
            ice_flux.sst,
            "rda8",
            debug_forcing,
            field_loc_center,
            field_type_scalar,
        )

        # Make sure sst is not less than freezing temperature Tf
        ice_flux.sst[...] = np.maximum(ice_flux.sst, ice_flux.Tf)

    elif ocn_data_type in ("hadgem_sst", "hadgem_sst_uvocn"):
        diag = True  # write diagnostic information
        sst_file = ocn_data_dir + "/MONTHLY/sst.1997.nc"
        nu.write(" \n")
        nu.write(f" Initial SST file: {sst_file}\n")
        nu.flush()

        fieldname = "sst"
        _read_nc_xy(
            sst_file,
            int(ice_calendar.mmonth),
            fieldname,
            ice_flux.sst,
            diag,
            field_loc_center,
            field_type_scalar,
        )

        # Make sure sst is not less than freezing temperature Tf
        ice_flux.sst[...] = np.maximum(ice_flux.sst, ice_flux.Tf)

    elif ocn_data_type == "ncar":
        ocn_data_ncar_init()
        # call ocn_data_ncar_init_3D

    elif ocn_data_type == "hycom":
        ocn_data_hycom_init()

    elif ocn_data_type == "box2001":
        box2001_data_ocn()

    # uniform forcing options
    elif ocn_data_type == "uniform_northeast":
        uniform_data_ocn("NE", p1)
    elif ocn_data_type == "uniform_east":
        uniform_data_ocn("E", p1)
    elif ocn_data_type == "uniform_north":
        uniform_data_ocn("N", p1)
    elif ocn_data_type == "calm":
        uniform_data_ocn("N", c0)  # directon does not matter for c0
    elif ocn_data_type == "default":
        # don't need to do anything more
        pass
    else:
        raise ValueError(subname + " ERROR ocn_data_type unknown = " + ocn_data_type)


def ocn_freezing_temperature():
    """Compute ocean freezing temperature Tf based on tfrz_option
    'minus1p8'         Tf = -1.8 C (default)
    'linear_salt'      Tf = -depressT * sss
    'mushy'            Tf conforms with mushy layer thermo (ktherm=2)
    """
    subname = "(ocn_freezing_temperature)"
    if local_debug:
        ice_fileunits.nu_diag.write(f" {subname} fdbg start\n")
    ice_flux.Tf[...] = icepack_sea_freezing_temperature(ice_flux.sss)
    # call icepack_warnings_flush / abort_ice are not ported.


def ocn_data_hycom_init():
    """Read SSS+SST from a HYCOM file converted to NetCDF format.
    HYCOM binary2NetCDF: hcdata2ncdf2d (or hcdata2ncdf3z)
      + rename/link file
    """
    global sss_file, sst_file
    subname = "(ocn_data_hycom_init)"
    if local_debug:
        ice_fileunits.nu_diag.write(f" {subname} fdbg start\n")

    if str(ice_init.ocn_data_type).strip() == "hycom":
        ocn_data_dir = str(ice_init.ocn_data_dir).strip()
        sss_file = ocn_data_dir + "ice.restart.surf.nc"
        nu = ice_fileunits.nu_diag
        nu.write(" \n")
        nu.write(f" Initial SSS file: {sss_file}\n")
        nu.flush()

        fieldname = "sss"
        _read_nc_xy(
            sss_file,
            1,
            fieldname,
            ice_flux.sss,
            bool(ice_init.debug_forcing),
            field_loc_center,
            field_type_scalar,
        )

        ocn_freezing_temperature()

        sst_file = ocn_data_dir + "ice.restart.surf.nc"
        nu.write(" \n")
        nu.write(f" Initial SST file: {sst_file}\n")
        nu.flush()

        fieldname = "sst"
        _read_nc_xy(
            sst_file,
            1,
            fieldname,
            ice_flux.sst,
            bool(ice_init.debug_forcing),
            field_loc_center,
            field_type_scalar,
        )

        # Make sure sst is not less than freezing temperature Tf
        ice_flux.sst[...] = np.maximum(ice_flux.sst, ice_flux.Tf)


def box2001_data_ocn():
    """current fields as in Hunke, JCP 2001
    these are defined at the u point
    authors: Elizabeth Hunke, LANL
    """
    subname = "(box2001_data_ocn)"
    if local_debug:
        ice_fileunits.nu_diag.write(f" {subname} fdbg start\n")

    ig, jg = global_indices()
    nx = dbl_kind(int(ice_domain_size.nx_global))
    ny = dbl_kind(int(ice_domain_size.ny_global))
    # ocean current
    # constant in time, could be initialized in ice_flux.F90
    # uocn depends only on global j, vocn only on global i.
    ice_flux.uocn[...] = p2 * dbl_kind(jg)[np.newaxis, :] / ny - p1
    ice_flux.vocn[...] = -p2 * dbl_kind(ig)[:, np.newaxis] / nx + p1


def uniform_data_ocn(dir, spd=None):
    """uniform current fields in some direction

    dir: 'NE', 'N', or 'E'
    spd: velocity. Default is p1 when omitted, matching the Fortran optional.
    """
    subname = "(uniform_data_ocn)"
    if local_debug:
        ice_fileunits.nu_diag.write(f" {subname} fdbg start\n")

    if spd is not None:
        ocn_val = spd
    else:
        ocn_val = p1  # default

    # ocn components
    if dir == "NE":
        ice_flux.uocn[...] = ocn_val
        ice_flux.vocn[...] = ocn_val
    elif dir == "N":
        ice_flux.uocn[...] = c0
        ice_flux.vocn[...] = ocn_val
    elif dir == "E":
        ice_flux.uocn[...] = ocn_val
        ice_flux.vocn[...] = c0
    else:
        raise ValueError(subname + "ERROR: dir unknown, dir = " + dir)


def _abort(message):
    ice_fileunits.nu_diag.write(message + "\n")
    ice_fileunits.nu_diag.flush()
    raise RuntimeError(message)


def _fdiv(num, den):
    """Fortran integer division, toward zero."""
    return int(num / den)


def _fmod(num, den):
    """Fortran MOD for integers."""
    return num - _fdiv(num, den) * den


def _atm_type():
    return str(ice_init.atm_data_type).strip()


def _ocn_type():
    return str(ice_init.ocn_data_type).strip()


def _params(*names):
    return icepack_parameters.icepack_query_parameters(*names)


def _note_debug():
    """Turn diagnostics on after debug_model_step, as the read routines do."""
    global debug_forcing
    if int(ice_calendar.istep1) > int(ice_init.debug_model_step):
        debug_forcing = True  # debugging
    return debug_forcing


def _forcing_year(myear=None):
    """Current year inside the forcing cycle."""
    if myear is None:
        myear = int(ice_calendar.myear)
    fyear_init = int(ice_init.fyear_init)
    ycycle = int(ice_init.ycycle)
    span = myear - fyear_init
    # adjustment for mod function
    modadj = abs((_fdiv(min(0, span), ycycle) + 1) * ycycle)
    return fyear_init + _fmod(span + modadj, ycycle)


def _mask_range(field, mask):
    """Min and max of field where mask is true. One grid, no reduction."""
    values = np.asarray(field)[np.asarray(mask)]
    if values.size == 0:
        return np.min(field), np.max(field)
    return np.min(values), np.max(values)


def _nc_open(path):
    try:
        return netCDF4.Dataset(path)
    except OSError as exc:
        raise RuntimeError("(ice_open_nc) cannot open " + path) from exc


def _load_var(dataset, varname, subname):
    if varname not in dataset.variables:
        raise RuntimeError(subname + " ERROR: Cannot find variable " + varname)
    return np.array(dataset.variables[varname][:], dtype=dbl_kind)


def _as_ij_first(data):
    """Put i on axis 0 and j on axis 1.

    netCDF4 arrays are C order, so a horizontal field arrives as
    (..., ny, nx). Fortran nf90_get_var uses (nx, ny, ...).
    """
    nx = int(ice_domain_size.nx_global)
    ny = int(ice_domain_size.ny_global)
    data = np.asarray(data)
    if data.ndim >= 2 and data.shape[0] == nx and data.shape[1] == ny:
        return data
    if data.ndim >= 2 and data.shape[-1] == nx and data.shape[-2] == ny:
        return np.transpose(data)
    return data


def _check_ij(data, subname, varname):
    nx = int(ice_domain_size.nx_global)
    ny = int(ice_domain_size.ny_global)
    if data.ndim < 2 or data.shape[0] != nx or data.shape[1] != ny:
        raise RuntimeError(
            subname
            + " "
            + varname
            + " shape "
            + str(getattr(data, "shape", None))
            + "; expected ("
            + str(nx)
            + ", "
            + str(ny)
            + ", ...) with i as the first dimension"
        )


def _scatter_slab(array, slab, diag, field_loc, field_type, subname, varname, nrec, path):
    if diag:
        kept = slab[slab != spval_dbl]
        ice_fileunits.nu_diag.write(
            f" {subname} {path} lnrec = {nrec}, varname = {varname}"
            f" min, max, sum = {np.min(slab)} "
            f"{np.max(kept) if kept.size else np.max(slab)} {np.sum(kept)}\n"
        )
        ice_fileunits.nu_diag.flush()
    scatter_global(array, slab, field_loc, field_type)
    # echmod:  this should not be necessary if fill/missing are only on land
    array[array > dbl_kind(1.0e30)] = c0


def _dataset_xy(dataset, nrec, varname, array, diag, field_loc, field_type, path=""):
    """One horizontal record. nrec is 1-based. i is axis 0, then j, then time."""
    subname = "(ice_read_nc_xy)"
    if varname not in dataset.variables:
        raise RuntimeError(subname + " ERROR: Cannot find variable " + varname)
    variable = dataset.variables[varname]
    nx = int(ice_domain_size.nx_global)
    ny = int(ice_domain_size.ny_global)
    # Match _as_ij_first's layout precedence without loading the time series.
    if variable.shape[:2] == (nx, ny):
        transpose = False
    elif variable.ndim >= 2 and variable.shape[-2:] == (ny, nx):
        transpose = True
    else:
        _check_ij(variable, subname, varname)
    nrec = int(nrec)
    if variable.ndim == 2:
        ntime = 1
        if nrec != 1:
            ice_fileunits.nu_diag.write(
                f" {subname} ERROR: not enough records, {varname} {nrec} {ntime}\n"
            )
            raise RuntimeError(subname + " ERROR: not enough records " + varname)
        slab = np.array(variable[:, :], dtype=dbl_kind)
    elif variable.ndim == 3:
        ntime = int(variable.shape[0] if transpose else variable.shape[2])
        if nrec < 1 or nrec > ntime:
            ice_fileunits.nu_diag.write(
                f" {subname} ERROR: not enough records, {varname} {nrec} {ntime}\n"
            )
            raise RuntimeError(subname + " ERROR: not enough records " + varname)
        # Slice at the NetCDF boundary so only one horizontal record is read
        # and converted, rather than the full annual forcing variable.
        selection = ((nrec - 1, slice(None), slice(None)) if transpose
                     else (slice(None), slice(None), nrec - 1))
        slab = np.array(variable[selection], dtype=dbl_kind)
    else:
        raise RuntimeError(subname + " ERROR: Cannot get variable " + varname)
    if transpose:
        slab = slab.T
    _check_ij(slab, subname, varname)
    _scatter_slab(array, slab, diag, field_loc, field_type, subname, varname, nrec, path)
    return array


def _read_nc_xy(path, nrec, varname, array, diag, field_loc, field_type):
    dataset = _nc_open(path)
    try:
        return _dataset_xy(dataset, nrec, varname, array, diag, field_loc, field_type, path)
    finally:
        dataset.close()


def _read_nc_uv(path, nrec, nzlev, varname, array, diag, field_loc, field_type):
    """i, j, z, time. nzlev and nrec are 1-based."""
    subname = "(ice_read_nc_uv)"
    dataset = _nc_open(path)
    try:
        data = _as_ij_first(_load_var(dataset, varname, subname))
    finally:
        dataset.close()
    if data.ndim != 4:
        raise RuntimeError(subname + " ERROR: Cannot get variable " + varname)
    _check_ij(data, subname, varname)
    nrec = int(nrec)
    nzlev = int(nzlev)
    if nzlev < 1 or nzlev > data.shape[2] or nrec < 1 or nrec > data.shape[3]:
        raise RuntimeError(subname + " ERROR: Cannot get variable " + varname)
    slab = np.array(data[:, :, nzlev - 1, nrec - 1], dtype=dbl_kind)
    _scatter_slab(array, slab, diag, field_loc, field_type, subname, varname, nrec, path)
    return array


def _read_nc_xyf(path, nrec, varname, dest, diag, field_loc, field_type):
    """Wave spectrum record. dest is (i, j, nfreq). File axes are i, j, freq, time."""
    subname = "(ice_read_nc_xyf)"
    dataset = _nc_open(path)
    try:
        data = _as_ij_first(_load_var(dataset, varname, subname))
    finally:
        dataset.close()
    nfreq = int(dest.shape[2])
    nrec = int(nrec)
    _check_ij(data, subname, varname)
    if data.ndim == 3:
        if nrec != 1 or data.shape[2] < nfreq:
            ice_fileunits.nu_diag.write(
                f" {subname} ERROR: not enough records, {varname} {nrec} {data.shape}\n"
            )
            raise RuntimeError(subname + " ERROR: not enough records " + varname)
        cube = data[:, :, :nfreq]
    elif data.ndim == 4:
        ntime = int(data.shape[3])
        if data.shape[2] < nfreq or nrec < 1 or nrec > ntime:
            ice_fileunits.nu_diag.write(
                f" {subname} ERROR: not enough records, {varname} {nrec} {ntime}\n"
            )
            raise RuntimeError(subname + " ERROR: not enough records " + varname)
        cube = data[:, :, :nfreq, nrec - 1]
    else:
        raise RuntimeError(subname + " ERROR: Cannot get variable " + varname)
    for freq in range(nfreq):
        plane = _zero(dest.shape[:2])
        slab = np.array(cube[:, :, freq], dtype=dbl_kind)
        _scatter_slab(plane, slab, diag, field_loc, field_type, subname, varname, nrec, path)
        dest[:, :, freq] = plane
    return dest


def _read_nc_point(path, nrec, varname, diag):
    subname = "(ice_read_nc_point)"
    dataset = _nc_open(path)
    try:
        data = np.array(_load_var(dataset, varname, subname)).reshape(-1)
    finally:
        dataset.close()
    nrec = int(nrec)
    if nrec < 1 or nrec > data.size:
        ice_fileunits.nu_diag.write(
            f" {subname} ERROR: not enough records, {varname} {nrec} {data.size}\n"
        )
        raise RuntimeError(subname + " ERROR: not enough records " + varname)
    if diag:
        ice_fileunits.nu_diag.write(
            f" {subname} lnrec = {nrec}, varname = {varname}\n"
        )
        ice_fileunits.nu_diag.flush()
    return dbl_kind(data[nrec - 1])


def _read_nc_vec(path, nrec, varname, diag):
    subname = "(ice_read_vec_nc)"
    dataset = _nc_open(path)
    try:
        data = np.array(_load_var(dataset, varname, subname)).reshape(-1)
    finally:
        dataset.close()
    nrec = int(nrec)
    if data.size < nrec:
        raise RuntimeError(subname + " ERROR: Cannot get variable " + varname)
    work = np.array(data[:nrec], dtype=dbl_kind, copy=True)
    if diag:
        ice_fileunits.nu_diag.write(
            f" {subname} min, max, nrec = {np.min(work)} {np.max(work)} {nrec}\n"
        )
        ice_fileunits.nu_diag.flush()
    return work


def _nc_dimension_length(path, dimname):
    """ice_get_ncvarsize: length of the dimension whose name is dimname."""
    subname = "(ice_get_ncvarsize)"
    dataset = _nc_open(path)
    try:
        for name, dim in dataset.dimensions.items():
            if str(name).strip() == str(dimname).strip():
                return int(len(dim))
    finally:
        dataset.close()
    raise RuntimeError(subname + " ERROR: Did not find variable " + dimname)


def _read_nc_1d(path, varname, length, diag):
    subname = "(ice_read_nc_1D)"
    dataset = _nc_open(path)
    try:
        data = np.array(_load_var(dataset, varname, subname)).reshape(-1)
    finally:
        dataset.close()
    length = int(length)
    if data.size < length:
        ice_fileunits.nu_diag.write(f" {subname} work, dim=1 {data.size} {length}\n")
        raise RuntimeError(subname + " ERROR: work array wrong size " + varname)
    if diag:
        ice_fileunits.nu_diag.write(
            f" {subname} xdim = {length} varname = {varname}\n"
        )
    return np.array(data[:length], dtype=dbl_kind, copy=True)


def _read_nc_3d(path, varname, shape, diag):
    subname = "(ice_read_nc_3D)"
    dataset = _nc_open(path)
    try:
        data = _load_var(dataset, varname, subname)
    finally:
        dataset.close()
    wanted = tuple(int(n) for n in shape)
    # Fortran reads (x, y, z). netCDF4 returns that order reversed.
    if tuple(int(n) for n in data.shape) != wanted and tuple(int(n) for n in data.shape[::-1]) == wanted:
        data = np.transpose(data)
    if tuple(int(n) for n in data.shape) != wanted:
        ice_fileunits.nu_diag.write(
            f" {subname} work shape {data.shape} expected {wanted} {varname}\n"
        )
        raise RuntimeError(subname + " ERROR: work array wrong size " + varname)
    if diag:
        ice_fileunits.nu_diag.write(
            f" {subname} xdim = {wanted[0]} ydim = {wanted[1]} zdim = {wanted[2]}"
            f" varname = {varname}\n"
        )
    return np.array(data, dtype=dbl_kind, copy=True)


def _bin_record(path, nrec, array, field_loc, field_type):
    ice_read_write.ice_read(
        path,
        int(nrec),
        array,
        "rda8",
        debug_forcing,
        field_loc,
        field_type,
    )


def init_forcing_atmo():
    """Determine the current and final year of the forcing cycle based on
    namelist input; initialize the atmospheric forcing data filenames.
    """
    global fyear, fyear_final, debug_forcing
    subname = "(init_forcing_atmo)"
    if local_debug:
        ice_fileunits.nu_diag.write(f" {subname} fdbg start\n")

    debug_forcing = bool(ice_init.debug_forcing)
    # adjustment for mod function
    fyear = _forcing_year()
    fyear_final = int(ice_init.fyear_init) + int(ice_init.ycycle) - 1  # last year in forcing cycle

    nu = ice_fileunits.nu_diag
    if local_debug:
        nu.write(f" {subname} fdbg fyear = {fyear} {int(ice_init.fyear_init)} {fyear_final}\n")
        nu.write(f" {subname} fdbg atm_data_type = {_atm_type()}\n")

    atm = _atm_type()
    if atm != "default":
        nu.write(f"  Initial forcing data year = {int(ice_init.fyear_init)}\n")
        nu.write(f"  Final   forcing data year = {fyear_final}\n")
        nu.flush()

    if atm == "hadgem" and str(ice_init.precip_units).strip() != "mks":
        nu.write(" WARNING: HadGEM atmospheric data chosen with wrong precip_units\n")
        nu.write(" WARNING:   Changing precip_units to mks (i.e. kg/m2 s).\n")
        nu.flush()
        _abort(subname + " HadGEM precip_units error")

    if bool(ice_calendar.use_leap_years) and (
        "JRA55" not in atm and atm != "hycom" and atm != "box2001"
    ):
        nu.write(" use_leap_years option is currently only supported for\n")
        nu.write(" JRA55, JRA55do, default , and box2001 atmospheric data\n")
        nu.flush()
        _abort(subname)

    #-------------------------------------------------------------------
    # Get filenames for input forcing data
    #-------------------------------------------------------------------

    # default forcing values from init_flux_atm
    if atm == "ncar":
        ncar_files(fyear)
    elif "JRA55" in atm:
        JRA55_files(fyear)
    elif atm == "hadgem":
        hadgem_files(fyear)
    elif atm == "monthly":
        monthly_files(fyear)
    elif atm == "oned":
        oned_files()
    elif atm == "ISPOL":
        ISPOL_files()
    elif atm == "box2001":
        box2001_data_atm()
    elif atm == "uniform_northeast":
        uniform_data_atm("NE")
    elif atm == "uniform_north":
        uniform_data_atm("N")
    elif atm == "uniform_east":
        uniform_data_atm("E")
    elif atm == "uniform_south":
        uniform_data_atm("S")
    elif atm == "uniform_west":
        uniform_data_atm("W")
    elif atm == "calm":
        uniform_data_atm("N", c0)  # direction does not matter when c0
    elif atm == "hycom":
        hycom_atm_files()
    elif atm == "default":
        # don't need to do anything more
        pass
    else:
        _abort(subname + " ERROR atm_data_type unknown = " + atm)


def get_forcing_atmo(defer_ice_correction=False):
    """Read atmospheric inputs; the device driver can apply the JRA55 ice cap.

    With defer_ice_correction, the caller must correct Tair and potT using
    current ice area. The cap is above freezing and does not change rain/snow.
    """
    global fyear
    subname = "(get_forcing_atmo)"
    if local_debug:
        ice_fileunits.nu_diag.write(f" {subname} fdbg start\n")

    fyear_old = fyear
    fyear = _forcing_year()
    atm = _atm_type()
    if atm != "default" and (int(ice_calendar.istep) <= 1 or fyear != fyear_old):
        ice_fileunits.nu_diag.write(f"  Set current forcing data year = {fyear}\n")
        ice_fileunits.nu_diag.flush()

    nt_Tsfc = int(icepack_tracers.nt_Tsfc)
    if nt_Tsfc < 0:
        _abort(subname)

    #-------------------------------------------------------------------
    # Read and interpolate atmospheric data
    #-------------------------------------------------------------------

    if local_debug:
        ice_fileunits.nu_diag.write(f" {subname} fdbg fyear = {fyear}\n")
        ice_fileunits.nu_diag.write(f" {subname} fdbg atm_data_type = {atm}\n")

    if atm == "ncar":
        ncar_data()
    elif "JRA55" in atm:
        JRA55_data(defer_ice_correction=defer_ice_correction)
    elif atm == "hadgem":
        hadgem_data()
    elif atm == "monthly":
        monthly_data()
    elif atm == "oned":
        oned_data()
    elif atm == "box2001":
        box2001_data_atm()
    elif atm == "uniform_northeast":
        uniform_data_atm("NE")
    elif atm == "uniform_north":
        uniform_data_atm("N")
    elif atm == "uniform_east":
        uniform_data_atm("E")
    elif atm == "uniform_south":
        uniform_data_atm("S")
    elif atm == "uniform_west":
        uniform_data_atm("W")
    elif atm == "calm":
        uniform_data_atm("N", c0)  # direction does not matter when c0
    elif atm == "hycom":
        hycom_atm_data()
    # elif atm == "uniform_northeast":
    # elif atm == "uniform_east":
    # elif atm == "uniform_north":
    else:  # default values set in init_flux
        return

    #-------------------------------------------------------------------
    # Convert forcing data to fields needed by ice model
    #-------------------------------------------------------------------

    ilo, ihi, jlo, jhi = ice_domain_size.physical_bounds()
    nx_block, ny_block = ice_domain_size.horizontal_shape()
    prepare_forcing(
        nx_block,
        ny_block,
        ilo,
        ihi,
        jlo,
        jhi,
        ice_grid.hm,
        ice_flux.Tair,
        ice_flux.fsw,
        cldf,
        ice_flux.flw,
        ice_flux.frain,
        ice_flux.fsnow,
        ice_flux.Qa,
        ice_flux.rhoa,
        ice_flux.uatm,
        ice_flux.vatm,
        ice_flux.strax,
        ice_flux.stray,
        ice_flux.zlvl,
        ice_flux.wind,
        ice_flux.swvdr,
        ice_flux.swvdf,
        ice_flux.swidr,
        ice_flux.swidf,
        ice_flux.potT,
        ice_grid.ANGLET,
        ice_state.trcr[:, :, nt_Tsfc],
        ice_flux.sst,
        ice_state.aice,
    )

    ice_halo_update(ice_flux.swvdr, field_loc_center, field_type_scalar, fill_value=c0)
    ice_halo_update(ice_flux.swvdf, field_loc_center, field_type_scalar, fill_value=c0)
    ice_halo_update(ice_flux.swidr, field_loc_center, field_type_scalar, fill_value=c0)
    ice_halo_update(ice_flux.swidf, field_loc_center, field_type_scalar, fill_value=c0)


def get_forcing_ocn(dt):
    """Read and interpolate annual climatologies of SSS and SST.
    Restore model SST to data if desired.
    Interpolate ocean fields to U grid if necessary.
    """
    subname = "(get_forcing_ocn)"
    if local_debug:
        ice_fileunits.nu_diag.write(f" {subname} fdbg start\n")
        ice_fileunits.nu_diag.write(f" {subname} fdbg fyear = {fyear}\n")
        ice_fileunits.nu_diag.write(f" {subname} fdbg ocn_data_type = {_ocn_type()}\n")

    ocn = _ocn_type()
    if ocn == "clim":
        ocn_data_clim(dt)
    elif ocn in ("ncar", "ISPOL"):
        ocn_data_ncar(dt)
    elif ocn in ("hadgem_sst", "hadgem_sst_uvocn"):
        ocn_data_hadgem(dt)
    elif ocn == "oned":
        ocn_data_oned()
    elif ocn == "hycom":
        # call ocn_data_hycom(dt)
        # MHRI: NOT IMPLEMENTED YET
        pass
    elif ocn == "box2001":
        box2001_data_ocn()
    # uniform forcing options
    elif ocn == "uniform_northeast":
        # tcraig, not time varying
        uniform_data_ocn("NE", p1)
    elif ocn == "uniform_east":
        uniform_data_ocn("E", p1)
    elif ocn == "uniform_north":
        uniform_data_ocn("N", p1)
    elif ocn == "calm":
        uniform_data_ocn("N", c0)  # directon does not matter for c0


def read_data(flag, recd, yr, ixm, ixx, ixp, maxrec, data_file, field_data, field_loc, field_type):
    """If data is at the beginning of a one-year record, get data from
     the previous year.
    If data is at the end of a one-year record, get data from the
     following year.
    If no earlier data exists (beginning of fyear_init), then
     (1) For monthly data, get data from the end of fyear_final.
     (2) For more frequent data, let the ixm value equal the
         first value of the year.
    If no later data exists (end of fyear_final), then
     (1) For monthly data, get data from the beginning of fyear_init.
     (2) For more frequent data, let the ixp value
         equal the last value of the year.
    In other words, we assume persistence when daily or 6-hourly
      data is missing, and we assume periodicity when monthly data
      is missing.
    """
    subname = "(read_data)"
    if local_debug:
        ice_fileunits.nu_diag.write(f" {subname} fdbg start\n")

    # nbits = 64              ! double precision data
    _note_debug()
    if debug_forcing:
        ice_fileunits.nu_diag.write(f"   {data_file}\n")
        ice_fileunits.nu_diag.flush()

    if flag:
        #-----------------------------------------------------------------
        # Initialize record counters
        # (n2, n4 will change only at the very beginning or end of
        #  a forcing cycle.)
        #-----------------------------------------------------------------
        n2 = ixm
        n4 = ixp
        arg = 0

        #-----------------------------------------------------------------
        # read data
        #-----------------------------------------------------------------

        if ixm != -99:
            # currently in first half of data interval
            if ixx <= 1:
                if yr > int(ice_init.fyear_init):  # get data from previous year
                    data_file = file_year(data_file, yr - 1)
                else:  # yr = fyear_init, no prior data exists
                    if maxrec > 12:  # extrapolate from first record
                        if ixx == 1:
                            n2 = ixx
                    else:  # go to end of fyear_final
                        data_file = file_year(data_file, fyear_final)
            arg = 1
            nrec = recd + n2
            _bin_record(data_file, nrec, field_data[:, :, arg - 1], field_loc, field_type)

        # always read ixx data from data file for current year
        data_file = file_year(data_file, yr)
        arg = arg + 1
        nrec = recd + ixx
        _bin_record(data_file, nrec, field_data[:, :, arg - 1], field_loc, field_type)

        if ixp != -99:
            # currently in latter half of data interval
            if ixx == maxrec:
                if yr < fyear_final:  # get data from following year
                    data_file = file_year(data_file, yr + 1)
                else:  # yr = fyear_final, no more data exists
                    if maxrec > 12:  # extrapolate from ixx
                        n4 = ixx
                    else:  # go to beginning of fyear_init
                        data_file = file_year(data_file, int(ice_init.fyear_init))
            arg = arg + 1
            nrec = recd + n4
            _bin_record(data_file, nrec, field_data[:, :, arg - 1], field_loc, field_type)

    return data_file


def read_data_nc(
    flag, recd, yr, ixm, ixx, ixp, maxrec, data_file, fieldname, field_data, field_loc, field_type
):
    """If data is at the beginning of a one-year record, get data from
     the previous year.
    If data is at the end of a one-year record, get data from the
     following year.
    If no earlier data exists (beginning of fyear_init), then
     (1) For monthly data, get data from the end of fyear_final.
     (2) For more frequent data, let the ixm value equal the
         first value of the year.
    If no later data exists (end of fyear_final), then
     (1) For monthly data, get data from the beginning of fyear_init.
     (2) For more frequent data, let the ixp value
         equal the last value of the year.
    In other words, we assume persistence when daily or 6-hourly
      data is missing, and we assume periodicity when monthly data
      is missing.

    Adapted by Alison McLaren, Met Office from read_data
    """
    subname = "(read_data_nc)"
    if local_debug:
        ice_fileunits.nu_diag.write(f" {subname} fdbg start\n")

    _note_debug()
    if debug_forcing:
        ice_fileunits.nu_diag.write(f"   {data_file}\n")
        ice_fileunits.nu_diag.flush()

    if flag:
        #-----------------------------------------------------------------
        # Initialize record counters
        # (n2, n4 will change only at the very beginning or end of
        #  a forcing cycle.)
        #-----------------------------------------------------------------
        n2 = ixm
        n4 = ixp
        arg = 0

        #-----------------------------------------------------------------
        # read data
        #-----------------------------------------------------------------

        if ixm != -99:
            # currently in first half of data interval
            if ixx <= 1:
                if yr > int(ice_init.fyear_init):  # get data from previous year
                    data_file = file_year(data_file, yr - 1)
                else:  # yr = fyear_init, no prior data exists
                    if maxrec > 12:  # extrapolate from first record
                        if ixx == 1:
                            n2 = ixx
                    else:  # go to end of fyear_final
                        data_file = file_year(data_file, fyear_final)
            arg = 1
            nrec = recd + n2
            _read_nc_xy(
                data_file,
                nrec,
                fieldname,
                field_data[:, :, arg - 1],
                debug_forcing,
                field_loc,
                field_type,
            )

        # always read ixx data from data file for current year
        data_file = file_year(data_file, yr)
        arg = arg + 1
        nrec = recd + ixx
        _read_nc_xy(
            data_file,
            nrec,
            fieldname,
            field_data[:, :, arg - 1],
            debug_forcing,
            field_loc,
            field_type,
        )

        if ixp != -99:
            # currently in latter half of data interval
            if ixx == maxrec:
                if yr < fyear_final:  # get data from following year
                    data_file = file_year(data_file, yr + 1)
                else:  # yr = fyear_final, no more data exists
                    if maxrec > 12:  # extrapolate from ixx
                        n4 = ixx
                    else:  # go to beginning of fyear_init
                        data_file = file_year(data_file, int(ice_init.fyear_init))
            arg = arg + 1
            nrec = recd + n4
            _read_nc_xy(
                data_file,
                nrec,
                fieldname,
                field_data[:, :, arg - 1],
                debug_forcing,
                field_loc,
                field_type,
            )

    return data_file


def read_data_nc_hycom(flag, recd, data_file, fieldname, field_data, field_loc, field_type):
    """Data is assumed to cover the entire time period of simulation.
    It is not bounded by start of year nor end of year
    Data must be accesible both before and after (or on) the point in time
      Assume increasing timeaxis within the forcing files, but they do not
        have to be equal spaced. Read time vector from "MT" in "init_hycom"

    Adapted by Mads Hvid Ribergaard, DMI from read_data_nc
    """
    subname = "(read_data_nc_hycom)"
    if local_debug:
        ice_fileunits.nu_diag.write(f" {subname} fdbg start\n")

    _note_debug()
    if debug_forcing:
        ice_fileunits.nu_diag.write(f"   {data_file}\n")
        ice_fileunits.nu_diag.flush()

    if flag:
        #-----------------------------------------------------------------
        # read data
        #-----------------------------------------------------------------
        _read_nc_xy(
            data_file,
            recd,
            fieldname,
            field_data[:, :, 0],
            debug_forcing,
            field_loc,
            field_type,
        )
        _read_nc_xy(
            data_file,
            recd + 1,
            fieldname,
            field_data[:, :, 1],
            debug_forcing,
            field_loc,
            field_type,
        )
    return data_file


def read_clim_data(readflag, recd, ixm, ixx, ixp, data_file, field_data, field_loc, field_type):
    """Read data needed for interpolation, as in read_data.
    Assume a one-year cycle of climatological data, so that there is
     no need to get data from other years or to extrapolate data beyond
     the forcing time period.
    """
    subname = "(read_clim_data)"
    if local_debug:
        ice_fileunits.nu_diag.write(f" {subname} fdbg start\n")

    # nbits = 64                ! double precision data
    _note_debug()
    if debug_forcing:
        ice_fileunits.nu_diag.write(f"   {data_file}\n")
        ice_fileunits.nu_diag.flush()

    if readflag:
        #-----------------------------------------------------------------
        # read data
        #-----------------------------------------------------------------
        arg = 0
        if ixm != -99:
            arg = 1
            nrec = recd + ixm
            _bin_record(data_file, nrec, field_data[:, :, arg - 1], field_loc, field_type)
        arg = arg + 1
        nrec = recd + ixx
        _bin_record(data_file, nrec, field_data[:, :, arg - 1], field_loc, field_type)
        if ixp != -99:
            arg = arg + 1
            nrec = recd + ixp
            _bin_record(data_file, nrec, field_data[:, :, arg - 1], field_loc, field_type)


def read_clim_data_nc(
    readflag, recd, ixm, ixx, ixp, data_file, fieldname, field_data, field_loc, field_type
):
    """Read data needed for interpolation, as in read_data.
    Assume a one-year cycle of climatological data, so that there is
     no need to get data from other years or to extrapolate data beyond
     the forcing time period.
    """
    subname = "(read_clim_data_nc)"
    if local_debug:
        ice_fileunits.nu_diag.write(f" {subname} fdbg start\n")

    _note_debug()
    if debug_forcing:
        ice_fileunits.nu_diag.write(f"   {data_file}\n")
        ice_fileunits.nu_diag.flush()

    if readflag:
        #-----------------------------------------------------------------
        # read data
        #-----------------------------------------------------------------
        arg = 0
        if ixm != -99:
            arg = 1
            nrec = recd + ixm
            _read_nc_xy(
                data_file,
                nrec,
                fieldname,
                field_data[:, :, arg - 1],
                debug_forcing,
                field_loc,
                field_type,
            )
        arg = arg + 1
        nrec = recd + ixx
        _read_nc_xy(
            data_file,
            nrec,
            fieldname,
            field_data[:, :, arg - 1],
            debug_forcing,
            field_loc,
            field_type,
        )
        if ixp != -99:
            arg = arg + 1
            nrec = recd + ixp
            _read_nc_xy(
                data_file,
                nrec,
                fieldname,
                field_data[:, :, arg - 1],
                debug_forcing,
                field_loc,
                field_type,
            )


def interp_coeff_monthly(recslot):
    """Compute coefficients for interpolating monthly data to current time step."""
    global c1intp, c2intp
    subname = "(interp_coeff_monthly)"
    if local_debug:
        ice_fileunits.nu_diag.write(f" {subname} fdbg start\n")

    secday = _params("secday")["secday"]
    daymid = [dbl_kind(14.0)] * 14  # month mid-points, indices 0:13
    daymid[1:14] = [dbl_kind(14.0)] * 13  # time frame ends 0 sec into day 15
    daymid[0] = dbl_kind(14.0) - dbl_kind(ice_calendar.daymo[11])  # Dec 15, 0 sec

    # compute days since Jan 1, 00h, yday is the day counter for the year
    tt = dbl_kind(float(ice_calendar.yday) - 1) + dbl_kind(ice_calendar.msec) / secday

    # Find neighboring times
    # Fortran daycal(mmonth) is days before that month: Python daycal[mmonth - 1]
    mm = int(ice_calendar.mmonth)
    daycal = ice_calendar.daycal
    if recslot == 2:  # first half of month
        t2 = dbl_kind(daycal[mm - 1]) + daymid[mm]  # midpoint, current month
        if mm == 1:
            t1 = daymid[0]  # Dec 15 (0 sec)
        else:
            t1 = dbl_kind(daycal[mm - 2]) + daymid[mm - 1]  # midpoint, previous month
    else:  # second half of month
        t1 = dbl_kind(daycal[mm - 1]) + daymid[mm]  # midpoint, current month
        t2 = dbl_kind(daycal[mm]) + daymid[mm + 1]  # day 15 of next month (0 sec)

    if tt < t1 or tt > t2:
        ice_fileunits.nu_diag.write(f" {subname} ERROR in tt {tt} {t1} {t2}\n")
        _abort(subname + " ERROR in tt")

    # Compute coefficients
    c1intp = (t2 - tt) / (t2 - t1)
    c2intp = c1 - c1intp


def interp_coeff(recnum, recslot, secint, dataloc):
    """Compute coefficients for interpolating data to current time step.
    Works for any data interval that divides evenly into a
     year (daily, 6-hourly, etc.)
    Use interp_coef_monthly for monthly data.
    """
    global c1intp, c2intp
    subname = "(interp_coeff)"
    if local_debug:
        ice_fileunits.nu_diag.write(f" {subname} fdbg start\n")

    secday = _params("secday")["secday"]
    # compute seconds since Jan 1, 00h, yday is the day counter for the year
    tt = dbl_kind(float(ice_calendar.yday) - 1) * secday + dbl_kind(ice_calendar.msec)

    # Find neighboring times
    rcnum = dbl_kind(recnum)
    if recslot == 2:  # current record goes in slot 2
        if dataloc == 1:  # data located at middle of interval
            t2 = (rcnum - p5) * secint
        else:  # data located at end of interval
            t2 = rcnum * secint
        t1 = t2 - secint  # - 1 interval
    else:  # recslot = 1
        if dataloc == 1:  # data located at middle of interval
            t1 = (rcnum - p5) * secint
        else:
            t1 = rcnum * secint  # data located at end of interval
        t2 = t1 + secint  # + 1 interval

    # Compute coefficients
    c1intp = dbl_kind(abs((t2 - tt) / (t2 - t1)))
    c2intp = c1 - c1intp

    if local_debug:
        ice_fileunits.nu_diag.write(f" {subname} fdbg yday,sec = {ice_calendar.yday} {ice_calendar.msec}\n")
        ice_fileunits.nu_diag.write(f" {subname} fdbg tt = {tt}\n")
        ice_fileunits.nu_diag.write(f" {subname} fdbg c12intp = {c1intp} {c2intp}\n")


def interp_coeff2(tt, t1, t2):
    """Compute coefficients for interpolating data to current time step.
    Works for any data interval using decimal daynumbers
    """
    global c1intp, c2intp
    subname = "(interp_coeff2)"
    if local_debug:
        ice_fileunits.nu_diag.write(f" {subname} fdbg start\n")

    # Compute coefficients
    c1intp = dbl_kind(abs((t2 - tt) / (t2 - t1)))
    c2intp = c1 - c1intp


def interpolate_data(field_data, field):
    """Linear interpolation

    author: Elizabeth C. Hunke, LANL
    """
    subname = "(interpolate data)"
    if local_debug:
        ice_fileunits.nu_diag.write(f" {subname} fdbg start\n")
    field[...] = c1intp * field_data[:, :, 0] + c2intp * field_data[:, :, 1]


def interpolate_wavespec_data(field_data, field):
    """Linear interpolation

    author: Elizabeth C. Hunke, LANL
    """
    field[...] = c1intp * field_data[:, :, :, 0] + c2intp * field_data[:, :, :, 1]


def file_year(data_file, yr):
    """Construct the correct name of the atmospheric data file
    to be read, given the year and assuming the naming convention
    that filenames end with 'yyyy.dat' or 'yyyy.r' or 'yyyy.nc'.
    """
    subname = "(file_year)"
    if local_debug:
        ice_fileunits.nu_diag.write(f" {subname} fdbg start\n")

    atm = _atm_type()
    if atm == "hadgem":  # netcdf
        ext = ".nc"
    elif "JRA55" in atm:  # netcdf
        ext = ".nc"
    else:  # LANL/NCAR naming convention
        ext = ".dat"
    pos = data_file.find(ext)
    if pos < 0:
        _abort(subname + " " + ext + " not found in " + data_file)
    # Fortran index is 1-based; i = index(ext) - 5 keeps the text before yyyy
    i = (pos + 1) - 5
    if i < 0:
        _abort(subname + " year field missing in " + data_file)
    return data_file[:i] + f"{int(yr):04d}" + ext


def _month_neighbors(midmonth):
    """Record numbers and recslot for data given on day midmonth."""
    mm = int(ice_calendar.mmonth)
    mday = int(ice_calendar.mday)
    maxrec = 12
    ixm = (mm + maxrec - 2) % maxrec + 1
    ixp = mm % maxrec + 1
    if mday >= midmonth:
        ixm = -99  # other two points will be used
    if mday < midmonth:
        ixp = -99
    # recslot = 2 means we use values 1:2, with the current value (2)
    #  in the second slot
    # recslot = 1 means we use values 2:3, with the current value (2)
    #  in the first slot
    recslot = 1  # latter half of month
    if mday < midmonth:
        recslot = 2  # first half of month
    readm = False
    if int(ice_calendar.istep) == 1 or (mday == midmonth and int(ice_calendar.msec) == 0):
        readm = True
    return ixm, ixp, maxrec, recslot, readm


def prepare_forcing(
    nx_block,
    ny_block,
    ilo,
    ihi,
    jlo,
    jhi,
    hm,
    Tair,
    fsw,
    cldf_field,
    flw,
    frain,
    fsnow,
    Qa,
    rhoa,
    uatm,
    vatm,
    strax,
    stray,
    zlvl,
    wind,
    swvdr,
    swvdf,
    swidr,
    swidf,
    potT,
    ANGLET,
    Tsfc,
    sst,
    aice,
):
    subname = "(prepare_forcing)"
    if local_debug:
        ice_fileunits.nu_diag.write(f" {subname} fdbg start\n")
    # nx_block, ny_block, hm, Tsfc, sst, aice: block shape and fields used by
    # the commented Rosati-Miyakoda longwave option
    _ = (nx_block, ny_block)

    queried = _params("Tffresh", "puny", "secday", "calc_strair")
    Tffresh = queried["Tffresh"]
    _ = queried["puny"]
    secday = queried["secday"]
    calc_strair = bool(queried["calc_strair"])

    slc = (slice(ilo, ihi + 1), slice(jlo, jhi + 1))

    #-----------------------------------------------------------------
    # make sure interpolated values are physically realistic
    #-----------------------------------------------------------------
    cldf_field[slc] = np.maximum(np.minimum(cldf_field[slc], c1), c0)
    fsw[slc] = np.maximum(fsw[slc], c0)
    fsnow[slc] = np.maximum(fsnow[slc], c0)
    rhoa[slc] = np.maximum(rhoa[slc], c0)
    Qa[slc] = np.maximum(Qa[slc], c0)

    # if (rhoa .lt. puny) rhoa = 1.3
    # if (Tair .lt. puny) Tair = Tffresh
    # if (Qa .lt. puny) Qa = 0.0035

    #-----------------------------------------------------------------
    # calculations specific to datasets
    #-----------------------------------------------------------------

    zlvl0 = c10  # default
    atm = _atm_type()
    if atm == "ncar":
        # precip is in mm/month
        zlvl0 = c10
        # correct known biases in NCAR data (as in CESM latm)
        Qa[slc] = Qa[slc] * dbl_kind(0.94)
        fsw[slc] = fsw[slc] * dbl_kind(0.92)
        # downward longwave as in Parkinson and Washington (1979)
        flw[slc] = longwave_parkinson_washington(Tair[slc], cldf_field[slc])
    elif atm == "oned":  # rectangular grid
        # precip is in kg/m^2/s
        zlvl0 = c10
        #-----------------------------------------------------------------
        # compute downward longwave as in Parkinson and Washington (1979)
        #-----------------------------------------------------------------
        # downward longwave as in Parkinson and Washington (1979)
        flw[slc] = longwave_parkinson_washington(Tair[slc], cldf_field[slc])
        # longwave based on Rosati and Miyakoda, JPO 18, p. 1607 (1988)
        # longwave_rosati_miyakoda(cldf, Tsfc, aice, sst, Qa, Tair, hm, flw)
        _ = (hm, Tsfc, sst, aice)

    #-----------------------------------------------------------------
    # Compute other fields needed by model
    #-----------------------------------------------------------------

    # convert precipitation units to kg/m^2 s
    precip_units = str(ice_init.precip_units).strip()
    if precip_units == "mm_per_month":
        precip_factor = c12 / (secday * dbl_kind(ice_calendar.days_per_year))
    elif precip_units == "mm_per_day":
        precip_factor = c1 / secday
    elif precip_units in ("mm_per_sec", "mks"):
        precip_factor = c1  # mm/sec = kg/m^2 s
    elif precip_units == "m_per_sec":
        precip_factor = c1000
    else:
        _abort(subname + " unknown precip_units = " + precip_units)

    zlvl[slc] = zlvl0
    potT[slc] = Tair[slc]

    # divide shortwave into spectral bands
    swvdr[slc] = fsw[slc] * frcvdr  # visible direct
    swvdf[slc] = fsw[slc] * frcvdf  # visible diffuse
    swidr[slc] = fsw[slc] * frcidr  # near IR direct
    swidf[slc] = fsw[slc] * frcidf  # near IR diffuse

    # convert precipitation units to kg/m^2 s
    fsnow[slc] = fsnow[slc] * precip_factor

    # determine whether precip is rain or snow
    # HadGEM forcing provides separate snowfall and rainfall rather
    # than total precipitation
    if atm != "hadgem":
        rain = Tair[slc] >= Tffresh
        frain[slc] = np.where(rain, fsnow[slc], c0)
        fsnow[slc] = np.where(rain, c0, fsnow[slc])

    if calc_strair:
        if bool(ice_init.rotate_wind):
            wind[slc] = np.sqrt(uatm[slc] ** 2 + vatm[slc] ** 2)
            #-----------------------------------------------------------------
            # Rotate zonal/meridional vectors to local coordinates.
            # Velocity comes in on T grid, but is oriented geographically ---
            # need to rotate to pop-grid FIRST using ANGLET
            # then interpolate to the U-cell centers  (otherwise we
            # interpolate across the pole).
            # Use ANGLET which is on the T grid !
            # Atmo variables are needed in T cell centers in subroutine
            # atmo_boundary_layer, and are interpolated to the U grid later as
            # necessary.
            #-----------------------------------------------------------------
            workx = np.array(uatm[slc], copy=True)  # wind velocity, m/s
            worky = np.array(vatm[slc], copy=True)
            angle = ANGLET[slc]
            uatm[slc] = workx * np.cos(angle) + worky * np.sin(angle)  # convert to POP grid
            vatm[slc] = worky * np.cos(angle) - workx * np.sin(angle)
        else:  # not rotated
            wind[slc] = np.sqrt(uatm[slc] ** 2 + vatm[slc] ** 2)
    else:  # strax, stray, wind are read from files
        if bool(ice_init.rotate_wind):
            workx = np.array(strax[slc], copy=True)  # wind stress
            worky = np.array(stray[slc], copy=True)
            angle = ANGLET[slc]
            strax[slc] = workx * np.cos(angle) + worky * np.sin(angle)  # convert to POP grid
            stray[slc] = worky * np.cos(angle) - workx * np.sin(angle)
        # else: wind (speed) is already read from file, so all is in place


def longwave_parkinson_washington(Tair, cldf_field):
    """compute downward longwave as in Parkinson and Washington (1979)
    (for now)
    Parkinson, C. L. and W. M. Washington (1979),
    Large-scale numerical-model of sea ice,
    JGR, 84, 311-337, doi:10.1029/JC084iC01p00311
    """
    subname = "(longwave_parkinson_washington)"
    if local_debug:
        ice_fileunits.nu_diag.write(f" {subname} fdbg start\n")
    queried = _params("Tffresh", "stefan_boltzmann")
    Tffresh = queried["Tffresh"]
    stefan_boltzmann = queried["stefan_boltzmann"]
    return (
        stefan_boltzmann
        * Tair ** 4
        * (c1 - dbl_kind(0.261) * np.exp(dbl_kind(-7.77e-4) * (Tffresh - Tair) ** 2))
        * (c1 + dbl_kind(0.275) * cldf_field)
    )


def longwave_rosati_miyakoda(cldf_field, Tsfc, aice, sst, Qa, Tair, hm, flw):
    """based on
    Rosati, A. and K. Miyakoda (1988),
    A general-circulation model for upper ocean simulation,
    J. Physical Oceanography, 18, 1601-1626,
    doi:10.1175/1520-0485(1988)018<1601:AGCMFU>2.0.CO;2
    """
    subname = "(longwave_rosati_miyakoda)"
    if local_debug:
        ice_fileunits.nu_diag.write(f" {subname} fdbg start\n")
    queried = _params("Tffresh", "stefan_boltzmann", "emissivity")
    Tffresh = queried["Tffresh"]
    stefan_boltzmann = queried["stefan_boltzmann"]
    emissivity = queried["emissivity"]
    fcc = c1 - dbl_kind(0.8) * cldf_field
    sstk = (Tsfc * aice + sst * (c1 - aice)) + Tffresh
    rtea = np.sqrt(c1000 * Qa / (dbl_kind(0.622) + dbl_kind(0.378) * Qa))
    ptem = Tair  # get this from stability?
    qlwm = ptem * ptem * ptem * (
        ptem * (dbl_kind(0.39) - dbl_kind(0.05) * rtea) * fcc + c4 * (sstk - ptem)
    )
    flw[...] = emissivity * stefan_boltzmann * (sstk ** 4 - qlwm)
    flw[...] = flw * hm  # land mask
    return flw


def ncar_files(yr):
    """Construct filenames based on the LANL naming conventions for NCAR data.
    Edit for other directory structures or filenames.
    Note: The year number in these filenames does not matter, because
          subroutine file_year will insert the correct year.
    Note: atm_data_dir may have NCAR_bulk or not

    atm_data_type should be 'ncar'
    atm_dat_dir should be ${CICE_DATA_root}/forcing/$grid/[NCAR_bulk,'']
       atm_data_dir should be set to ${CICE_DATA_root}/forcing/$grid/[JRA55,JRA55do,'']
          NCAR_bulk at the end of the atm_data_dir is optional to provide backwards
          compatibility and if not included, will be appended automaticaly.
          The grid is typically gx1, gx3, tx1, or similar.
    """
    global fsw_file, flw_file, rain_file, uwind_file, vwind_file, tair_file, humid_file, rhoa_file
    subname = "(ncar_files)"
    if local_debug:
        ice_fileunits.nu_diag.write(f" {subname} fdbg start\n")

    atm_data_dir = str(ice_init.atm_data_dir).strip()
    # decide whether NCAR_bulk is part of atm_data_dir and set atm_data_dir_extra
    atm_data_dir_extra = "/NCAR_bulk"
    if "NCAR_bulk" in atm_data_dir:
        atm_data_dir_extra = ""

    fsw_file = atm_data_dir + atm_data_dir_extra + "/MONTHLY/swdn.1996.dat"
    fsw_file = file_year(fsw_file, yr)
    flw_file = atm_data_dir + atm_data_dir_extra + "/MONTHLY/cldf.1996.dat"
    flw_file = file_year(flw_file, yr)
    rain_file = atm_data_dir + atm_data_dir_extra + "/MONTHLY/prec.1996.dat"
    rain_file = file_year(rain_file, yr)
    uwind_file = atm_data_dir + atm_data_dir_extra + "/4XDAILY/u_10.1996.dat"
    uwind_file = file_year(uwind_file, yr)
    vwind_file = atm_data_dir + atm_data_dir_extra + "/4XDAILY/v_10.1996.dat"
    vwind_file = file_year(vwind_file, yr)
    tair_file = atm_data_dir + atm_data_dir_extra + "/4XDAILY/t_10.1996.dat"
    tair_file = file_year(tair_file, yr)
    humid_file = atm_data_dir + atm_data_dir_extra + "/4XDAILY/q_10.1996.dat"
    humid_file = file_year(humid_file, yr)
    rhoa_file = atm_data_dir + atm_data_dir_extra + "/4XDAILY/dn10.1996.dat"
    rhoa_file = file_year(rhoa_file, yr)

    nu = ice_fileunits.nu_diag
    nu.write(" \n")
    nu.write(f" Forcing data year = {fyear}\n")
    nu.write(" Atmospheric data files:\n")
    for name in (fsw_file, flw_file, rain_file, uwind_file, vwind_file, tair_file, humid_file, rhoa_file):
        nu.write(f" {name}\n")
    nu.flush()


def ncar_data():
    global fsw_file, flw_file, rain_file, tair_file, uwind_file, vwind_file, rhoa_file, humid_file
    global oldrecnum
    subname = "(ncar_data)"
    if local_debug:
        ice_fileunits.nu_diag.write(f" {subname} fdbg start\n")

    secday = _params("secday")["secday"]

    #-------------------------------------------------------------------
    # monthly data
    #
    # Assume that monthly data values are located in the middle of the
    # month.
    #-------------------------------------------------------------------

    midmonth = 15  # data is given on 15th of every month
    # midmonth = fix(p5 * real(daymo(mmonth)))  ! exact middle
    ixm, ixp, maxrec, recslot, readm = _month_neighbors(midmonth)
    # Find interpolation coefficients
    interp_coeff_monthly(recslot)

    # Read 2 monthly values
    if str(ice_init.atm_data_format).strip() == "bin":
        fsw_file = read_data(
            readm, 0, fyear, ixm, int(ice_calendar.mmonth), ixp, maxrec, fsw_file, fsw_data,
            field_loc_center, field_type_scalar,
        )
        flw_file = read_data(
            readm, 0, fyear, ixm, int(ice_calendar.mmonth), ixp, maxrec, flw_file, cldf_data,
            field_loc_center, field_type_scalar,
        )
        rain_file = read_data(
            readm, 0, fyear, ixm, int(ice_calendar.mmonth), ixp, maxrec, rain_file, fsnow_data,
            field_loc_center, field_type_scalar,
        )
    else:
        _abort(subname + "nonbinary atm_data_format unavailable")
        # The routine exists, for example:
        # read_data_nc(readm, 0, fyear, ixm, mmonth, ixp, maxrec, fsw_file, 'fsw', fsw_data, ...)
        # read_data_nc(..., flw_file, 'cldf', cldf_data, ...)
        # read_data_nc(..., rain_file, 'prec', fsnow_data, ...)

    # Interpolate to current time step
    interpolate_data(fsw_data, ice_flux.fsw)
    interpolate_data(cldf_data, cldf)
    interpolate_data(fsnow_data, ice_flux.fsnow)

    #-------------------------------------------------------------------
    # 6-hourly data
    #
    # Assume that the 6-hourly value is located at the end of the
    #  6-hour period.  This is the convention for NCEP reanalysis data.
    #  E.g. record 1 gives conditions at 6 am GMT on 1 January.
    #-------------------------------------------------------------------

    dataloc = 2  # data located at end of interval
    sec6hr = secday / c4  # seconds in 6 hours
    maxrec = 1460  # 365*4

    # current record number
    recnum = 4 * int(ice_calendar.yday) - 3 + int(float(ice_calendar.msec) / float(sec6hr))

    # Compute record numbers for surrounding data
    ixm = (recnum + maxrec - 2) % maxrec + 1
    ixx = (recnum - 1) % maxrec + 1
    # ixp = mod(recnum, maxrec) + 1

    # Compute interpolation coefficients
    # If data is located at the end of the time interval, then the
    #  data value for the current record always goes in slot 2.
    recslot = 2
    ixp = -99
    interp_coeff(recnum, recslot, sec6hr, dataloc)

    # Read
    read6 = False
    if int(ice_calendar.istep) == 1 or oldrecnum != recnum:
        read6 = True

    if str(ice_init.atm_data_format).strip() == "bin":
        tair_file = read_data(
            read6, 0, fyear, ixm, ixx, ixp, maxrec, tair_file, Tair_data,
            field_loc_center, field_type_scalar,
        )
        uwind_file = read_data(
            read6, 0, fyear, ixm, ixx, ixp, maxrec, uwind_file, uatm_data,
            field_loc_center, field_type_vector,
        )
        vwind_file = read_data(
            read6, 0, fyear, ixm, ixx, ixp, maxrec, vwind_file, vatm_data,
            field_loc_center, field_type_vector,
        )
        rhoa_file = read_data(
            read6, 0, fyear, ixm, ixx, ixp, maxrec, rhoa_file, rhoa_data,
            field_loc_center, field_type_scalar,
        )
        humid_file = read_data(
            read6, 0, fyear, ixm, ixx, ixp, maxrec, humid_file, Qa_data,
            field_loc_center, field_type_scalar,
        )
    else:
        _abort(subname + "nonbinary atm_data_format unavailable")

    # Interpolate
    interpolate_data(Tair_data, ice_flux.Tair)
    interpolate_data(uatm_data, ice_flux.uatm)
    interpolate_data(vatm_data, ice_flux.vatm)
    interpolate_data(rhoa_data, ice_flux.rhoa)
    interpolate_data(Qa_data, ice_flux.Qa)

    # Save record number for next time step
    oldrecnum = recnum


def JRA55_files(yr):
    """find the JRA55 files:
    This subroutine finds the JRA55 atm forcing files based on settings
    in atm_data_type and atm_data_dir.  Because the filenames are not
    entirely consistent, we need a flexible method.

    atm_data_type could be JRA55 or JRA55do with/without _grid appended
    atm_data_dir could contain JRA55 or JRA55do or not
    actual files could have grid in name in two location or not at all

    The files will generally be of the format
       $atm_data_type/[JRA55,JRA55do,'']/8XDAILY/[JRA55,JRA55do][_$grid,'']_03hr_forcing[_$grid,'']_$year.nc
    The options defined by cnt try several versions of paths/filenames
    As a user,
       atm_data_type should be set to JRA55, JRA55do, JRA55_xxx, or JRA55do_xxx
          where xxx can be any set of characters.  The _xxx if included will be ignored.
          Historically, these were set to JRA55_gx1 and so forth but the _gx1 is no longer needed
          but this is still allowed for backwards compatibility.  atm_data_type_prefix
          is atm_data_type with _ and everything after _ removed.
       atm_data_dir should be set to ${CICE_DATA_root}/forcing/$grid/[JRA55,JRA55do,'']
          The [JRA55,JRA55do] at the end of the atm_data_dir is optional to provide backwards
          compatibility and if not included, will be appended automaticaly using
          the atm_data_type_prefix value.  The grid is typically gx1, gx3, tx1, or similar.
    In general, we recommend using the following format
       atm_data_type = [JRA55,JRA55do]
       atm_data_dir = ${CICE_DATA_root}/forcing/$grid
    """
    global uwind_file
    subname = "(JRA55_files)"
    if local_debug:
        ice_fileunits.nu_diag.write(f" {subname} fdbg start\n")

    atm = _atm_type()
    # this could be JRA55[do] or JRA55[do]_grid, drop the _grid if set
    atm_data_type_prefix = atm
    strind = atm.find("_")
    if strind >= 0:
        atm_data_type_prefix = atm[:strind]

    atm_data_dir = str(ice_init.atm_data_dir).strip()
    # check for grid version using fortran INDEX intrinsic
    if "gx1" in atm_data_dir:
        grd = "gx1"
    elif "gx3" in atm_data_dir:
        grd = "gx3"
    elif "tx1" in atm_data_dir:
        grd = "tx1"
    else:
        _abort(subname + " unknown grid type")

    version = str(ice_init.atm_data_version).strip()
    # cnt represents the possible file format options and steps thru them until one is found
    exists = False
    cnt = 1
    while (not exists) and cnt <= 6:
        if cnt == 1:
            uwind_file = (
                atm_data_dir + "/" + atm_data_type_prefix
                + "/8XDAILY/" + atm_data_type_prefix + "_" + grd
                + "_03hr_forcing" + version + "_2005.nc"
            )
        if cnt == 2:
            uwind_file = (
                atm_data_dir + "/" + atm_data_type_prefix
                + "/8XDAILY/" + atm_data_type_prefix + "_03hr_forcing_" + grd
                + version + "_2005.nc"
            )
        if cnt == 3:
            uwind_file = (
                atm_data_dir + "/" + atm_data_type_prefix
                + "/8XDAILY/" + atm_data_type_prefix
                + "_03hr_forcing" + version + "_2005.nc"
            )
        if cnt == 4:
            uwind_file = (
                atm_data_dir
                + "/8XDAILY/" + atm_data_type_prefix + "_" + grd
                + "_03hr_forcing" + version + "_2005.nc"
            )
        if cnt == 5:
            uwind_file = (
                atm_data_dir
                + "/8XDAILY/" + atm_data_type_prefix + "_03hr_forcing_" + grd
                + version + "_2005.nc"
            )
        if cnt == 6:
            uwind_file = (
                atm_data_dir
                + "/8XDAILY/" + atm_data_type_prefix
                + "_03hr_forcing" + version + "_2005.nc"
            )
        uwind_file = file_year(uwind_file, yr)
        exists = os.path.isfile(uwind_file)
        if debug_forcing:
            ice_fileunits.nu_diag.write(f" {subname} {cnt} {exists} {uwind_file}\n")
        cnt = cnt + 1

    if not exists:
        nu = ice_fileunits.nu_diag
        nu.write(f" {subname} atm_data_dir = {atm_data_dir}\n")
        nu.write(f" {subname} atm_data_type_prefix = {atm_data_type_prefix}\n")
        nu.write(f" {subname} atm_data_version = {version}\n")
        _abort(subname + " could not find forcing file")

    nu = ice_fileunits.nu_diag
    nu.write(" \n")
    nu.write(f" {subname}Atmospheric data files:\n")
    nu.write(f" {subname}{uwind_file}\n")
    nu.flush()


def close_forcing():
    """Release the single retained JRA55 input file at reset or model exit."""
    global _jra_dataset, _jra_dataset_path
    dataset = _jra_dataset
    _jra_dataset = None
    _jra_dataset_path = None
    if dataset is not None:
        dataset.close()


def _jra_copy_slot():
    Tair_data[:, :, 0] = Tair_data[:, :, 1]
    uatm_data[:, :, 0] = uatm_data[:, :, 1]
    vatm_data[:, :, 0] = vatm_data[:, :, 1]
    Qa_data[:, :, 0] = Qa_data[:, :, 1]
    fsw_data[:, :, 0] = fsw_data[:, :, 1]
    flw_data[:, :, 0] = flw_data[:, :, 1]
    fsnow_data[:, :, 0] = fsnow_data[:, :, 1]


def _jra_read_slot(dataset, recnum, slot):
    names = (
        ("airtmp", Tair_data),
        ("wndewd", uatm_data),
        ("wndnwd", vatm_data),
        ("spchmd", Qa_data),
        ("glbrad", fsw_data),
        ("dlwsfc", flw_data),
        ("ttlpcp", fsnow_data),
    )
    for fieldname, store in names:
        _dataset_xy(
            dataset,
            recnum,
            fieldname,
            store[:, :, slot],
            local_debug,
            field_loc_center,
            field_type_scalar,
            uwind_file,
        )


def JRA55_data(defer_ice_correction=False):
    global uwind_file, c1intp, c2intp, frec_info
    global _jra_dataset, _jra_dataset_path
    subname = "(JRA55_data)"
    if local_debug:
        ice_fileunits.nu_diag.write(f" {subname} fdbg start\n")

    queried = _params("Tffresh", "secday")
    Tffresh = queried["Tffresh"]
    secday = queried["secday"]
    sec3hr = secday / c8  # seconds in 3 hours
    maxrec = int(ice_calendar.days_per_year) * 8

    if local_debug:
        ice_fileunits.nu_diag.write(
            f" {subname} fdbg dpy, maxrec = {ice_calendar.days_per_year} {maxrec}\n"
        )

    #-------------------------------------------------------------------
    # 3-hourly data
    # states are instantaneous, 1st record is 00z Jan 1
    # fluxes are 3 hour averages, 1st record is 00z-03z Jan 1
    # interpolate states, do not interpolate fluxes
    #-------------------------------------------------------------------
    # File is NETCDF with winds in NORTH and EAST direction
    # file variable names are:
    # glbrad   (shortwave W/m^2), 3 hr average
    # dlwsfc   (longwave W/m^2), 3 hr average
    # wndewd   (eastward wind m/s), instantaneous
    # wndnwd   (northward wind m/s), instantaneous
    # airtmp   (air temperature K), instantaneous
    # spchmd   (specific humidity kg/kg), instantaneous
    # ttlpcp   (precipitation kg/m s-1), 3 hr average
    #-------------------------------------------------------------------

    try:
        for n1 in (1, 2):
            lfyear = fyear
            uwind_file = file_year(uwind_file, lfyear)
            if n1 == 1:
                recnum = (
                    8 * int(ice_calendar.yday)
                    - 7
                    + int(float(ice_calendar.msec) / float(sec3hr))
                )
                if recnum <= 2 or recnum >= maxrec - 1:
                    ice_fileunits.nu_diag.write(
                        f" {subname} reading forcing file 1st ts = {uwind_file}\n"
                    )
            else:
                recnum = (
                    8 * int(ice_calendar.yday)
                    - 7
                    + int(float(ice_calendar.msec) / float(sec3hr))
                    + 1
                )
                if recnum > maxrec:
                    lfyear = fyear + 1  # next year
                    if lfyear > fyear_final:
                        lfyear = int(ice_init.fyear_init)
                    recnum = 1
                    uwind_file = file_year(uwind_file, lfyear)
                    ice_fileunits.nu_diag.write(
                        f" {subname} reading forcing file 2nd ts = {uwind_file}\n"
                    )

            if local_debug:
                ice_fileunits.nu_diag.write(f" {subname} fdbg read recnum = {recnum} {n1}\n")

            # to reduce reading, check whether it's the same data as last read
            if lfyear != int(frec_info[0, n1 - 1]) or recnum != int(frec_info[1, n1 - 1]):
                # check whether we can copy values from 2 to 1, should be faster than reading
                # can only do this from 2 to 1 or 1 to 2 without setting up a temporary
                # it's more likely that the values from data2 when time advances are needed in data1
                # compare n1=1 year/record with data from last timestep at n1=2
                if (
                    n1 == 1
                    and lfyear == int(frec_info[0, 1])
                    and recnum == int(frec_info[1, 1])
                ):
                    _jra_copy_slot()
                else:
                    # Reuse the annual file across steps; retain at most one handle.
                    if _jra_dataset_path != uwind_file:
                        close_forcing()
                        _jra_dataset = _nc_open(uwind_file)
                        _jra_dataset_path = uwind_file
                    _jra_read_slot(_jra_dataset, recnum, n1 - 1)
            frec_info[0, n1 - 1] = lfyear
            frec_info[1, n1 - 1] = recnum
    except BaseException:
        close_forcing()
        raise
    finally:
        # Restore the current forcing year even if a record read fails.
        uwind_file = file_year(uwind_file, fyear)

    # Compute interpolation coefficients
    eps = dbl_kind(1.0e-6)
    tt = dbl_kind(_fmod(int(ice_calendar.msec), ice_calendar._nint(sec3hr)))
    c2intp = tt / sec3hr
    if c2intp < c0 and c2intp > c0 - eps:
        c2intp = c0
    if c2intp > c1 and c2intp < c1 + eps:
        c2intp = c1
    c1intp = dbl_kind(1.0) - c2intp
    if c2intp < c0 or c2intp > c1:
        ice_fileunits.nu_diag.write(f" {subname} ERROR: c2intp = {c2intp}\n")
        _abort(subname + " ERROR: c2intp out of range")
    if local_debug:
        ice_fileunits.nu_diag.write(f" {subname} fdbg c12intp = {c1intp} {c2intp}\n")

    # Interpolate
    interpolate_data(Tair_data, ice_flux.Tair)
    interpolate_data(uatm_data, ice_flux.uatm)
    interpolate_data(vatm_data, ice_flux.vatm)
    interpolate_data(Qa_data, ice_flux.Qa)
    # use 3 hr average for heat flux and precip fields, no interpolation
    # interpolate_data(fsw_data, fsw)
    # interpolate_data(flw_data, flw)
    # interpolate_data(fsnow_data, fsnow)
    ice_flux.fsw[...] = fsw_data[:, :, 0]
    ice_flux.flw[...] = flw_data[:, :, 0]
    ice_flux.fsnow[...] = fsnow_data[:, :, 0]

    # limit summer Tair values where ice is present
    if not defer_ice_correction:
        iced = ice_state.aice > p1
        ice_flux.Tair[iced] = np.minimum(ice_flux.Tair[iced], Tffresh + p1)
    hm = ice_grid.hm
    ice_flux.Qa[...] = ice_flux.Qa * hm
    ice_flux.Tair[...] = ice_flux.Tair * hm
    ice_flux.uatm[...] = ice_flux.uatm * hm
    ice_flux.vatm[...] = ice_flux.vatm * hm
    ice_flux.fsw[...] = ice_flux.fsw * hm
    ice_flux.flw[...] = ice_flux.flw * hm
    ice_flux.fsnow[...] = ice_flux.fsnow * hm

    if debug_forcing or local_debug:
        nu = ice_fileunits.nu_diag
        nu.write(f" {subname} fdbg JRA55_bulk_data\n")
        for label, field, mask in (
            ("fsw", ice_flux.fsw, ice_grid.tmask),
            ("flw", ice_flux.flw, ice_grid.tmask),
            ("fsnow", ice_flux.fsnow, ice_grid.tmask),
            ("Tair", ice_flux.Tair, ice_grid.tmask),
            ("uatm", ice_flux.uatm, ice_grid.umask),
            ("vatm", ice_flux.vatm, ice_grid.umask),
            ("Qa", ice_flux.Qa, ice_grid.tmask),
        ):
            vmin, vmax = _mask_range(field, mask)
            nu.write(f" {subname} fdbg {label} {vmin} {vmax}\n")
        nu.flush()


def compute_shortwave(nx_block, ny_block, ilo, ihi, jlo, jhi, TLON, TLAT, hm, Qa, cldf_field, fsw):
    """AOMIP shortwave forcing
    standard calculation using solar declination angle
    then shortwave is reduced using a function of cloud fraction
    """
    subname = "(compute_shortwave)"
    if local_debug:
        ice_fileunits.nu_diag.write(f" {subname} fdbg start\n")
    _ = (nx_block, ny_block)
    queried = _params("secday", "pi")
    secday = queried["secday"]
    pi = queried["pi"]
    slc = (slice(ilo, ihi + 1), slice(jlo, jhi + 1))
    deg2rad = pi / c180
    # solar_time = mod(real(msec),secday)/c3600 + c12*sin(p5*TLON)
    # Convert longitude to range of -180 to 180 for LST calculation
    lontmp = np.mod(TLON[slc] / deg2rad, c360)
    lontmp = np.where(lontmp > c180, lontmp - c360, lontmp)
    lontmp = np.where(lontmp < -c180, lontmp + c360, lontmp)
    solar_time = np.mod(dbl_kind(ice_calendar.msec), secday) / c3600 + lontmp / c15
    solar_time = np.where(solar_time >= dbl_kind(24.0), solar_time - dbl_kind(24.0), solar_time)
    hour_angle = (c12 - solar_time) * pi / c12
    declin = (
        dbl_kind(23.44)
        * np.cos((dbl_kind(172.0) - dbl_kind(ice_calendar.yday)) * c2 * pi / c365)
        * deg2rad
    )  # use dayyr instead of c365???
    cosZ = (
        np.sin(TLAT[slc]) * np.sin(declin)
        + np.cos(TLAT[slc]) * np.cos(declin) * np.cos(hour_angle)
    )
    cosZ = np.maximum(cosZ, c0)
    e = dbl_kind(1.0e5) * Qa[slc] / (dbl_kind(0.622) + dbl_kind(0.378) * Qa[slc])
    d = (cosZ + dbl_kind(2.7)) * e * dbl_kind(1.0e-5) + dbl_kind(1.085) * cosZ + p1
    sw0 = dbl_kind(1353.0) * cosZ ** 2 / d
    sw0 = np.maximum(sw0, c0)
    # total downward shortwave for cice
    fsw[slc] = sw0 * (c1 - p6 * cldf_field[slc] ** 3)
    fsw[slc] = fsw[slc] * hm[slc]


def Qa_fixLY(nx_block, ny_block, Tair, Qa):
    """prevents humidity from being super-saturated"""
    subname = "(Qa_fixLY)"
    if local_debug:
        ice_fileunits.nu_diag.write(f" {subname} fdbg start\n")
    _ = (nx_block, ny_block)
    queried = _params("Tffresh", "puny")
    Tffresh = queried["Tffresh"]
    puny = queried["puny"]
    worka = Tair - Tffresh
    worka = (
        c2
        + (dbl_kind(0.7859) + dbl_kind(0.03477) * worka) / (c1 + dbl_kind(0.00412) * worka)
        # 2+ converts ea mb -> Pa
        + dbl_kind(0.00422) * worka  # for ice
    )
    # vapor pressure
    worka = c10 ** worka  # saturated
    worka = np.maximum(worka, puny)  # puny over land to prevent division by zero
    # specific humidity
    worka = dbl_kind(0.622) * worka / (dbl_kind(1.0e5) - dbl_kind(0.378) * worka)
    Qa[...] = np.minimum(Qa, worka)


def hadgem_files(yr):
    """Construct filenames based on selected model options

    Note: The year number in these filenames does not matter, because
          subroutine file_year will insert the correct year.

    author: Alison McLaren, Met Office
    """
    global snow_file, rain_file, uwind_file, vwind_file, strax_file, stray_file, wind_file
    global fsw_file, flw_file, tair_file, humid_file, rhoa_file, sublim_file
    subname = "(hadgem_files)"
    if local_debug:
        ice_fileunits.nu_diag.write(f" {subname} fdbg start\n")

    queried = _params("calc_strair", "calc_Tsfc")
    calc_strair = bool(queried["calc_strair"])
    calc_Tsfc = bool(queried["calc_Tsfc"])
    atm_data_dir = str(ice_init.atm_data_dir).strip()
    nu = ice_fileunits.nu_diag

    # -----------------------------------------------------------
    # Rainfall and snowfall
    # -----------------------------------------------------------
    snow_file = atm_data_dir + "/MONTHLY/snowfall.1996.nc"
    snow_file = file_year(snow_file, yr)
    rain_file = atm_data_dir + "/MONTHLY/rainfall.1996.nc"
    rain_file = file_year(rain_file, yr)
    nu.write(" \n")
    nu.write(" Atmospheric data files:\n")
    nu.write(f" {rain_file}\n")
    nu.write(f" {snow_file}\n")

    if calc_strair:
        # --------------------------------------------------------
        # Wind velocity
        # --------------------------------------------------------
        uwind_file = atm_data_dir + "/MONTHLY/u_10.1996.nc"
        uwind_file = file_year(uwind_file, yr)
        vwind_file = atm_data_dir + "/MONTHLY/v_10.1996.nc"
        vwind_file = file_year(vwind_file, yr)
        nu.write(f" {uwind_file}\n")
        nu.write(f" {vwind_file}\n")
    else:
        # --------------------------------------------------------
        # Wind stress
        # --------------------------------------------------------
        strax_file = atm_data_dir + "/MONTHLY/taux.1996.nc"
        strax_file = file_year(strax_file, yr)
        stray_file = atm_data_dir + "/MONTHLY/tauy.1996.nc"
        stray_file = file_year(stray_file, yr)
        nu.write(f" {strax_file}\n")
        nu.write(f" {stray_file}\n")
        if calc_Tsfc or bool(ice_init.oceanmixed_ice):
            # --------------------------------------------------
            # Wind speed
            # --------------------------------------------------
            wind_file = atm_data_dir + "/MONTHLY/wind_10.1996.nc"
            wind_file = file_year(wind_file, yr)
            nu.write(f" {wind_file}\n")

    # --------------------------------------------------------------
    # Atmosphere properties.  Even if these fields are not
    # being used to force the ice (i.e. calc_Tsfc=.false.), they
    # are still needed to generate forcing for mixed layer model or
    # to calculate wind stress
    # --------------------------------------------------------------
    if calc_Tsfc or bool(ice_init.oceanmixed_ice) or calc_strair:
        fsw_file = atm_data_dir + "/MONTHLY/SW_incoming.1996.nc"
        fsw_file = file_year(fsw_file, yr)
        flw_file = atm_data_dir + "/MONTHLY/LW_incoming.1996.nc"
        flw_file = file_year(flw_file, yr)
        tair_file = atm_data_dir + "/MONTHLY/t_10.1996.nc"
        tair_file = file_year(tair_file, yr)
        humid_file = atm_data_dir + "/MONTHLY/q_10.1996.nc"
        humid_file = file_year(humid_file, yr)
        rhoa_file = atm_data_dir + "/MONTHLY/rho_10.1996.nc"
        rhoa_file = file_year(rhoa_file, yr)
        for name in (fsw_file, flw_file, tair_file, humid_file, rhoa_file):
            nu.write(f" {name}\n")

    if not calc_Tsfc:
        # ------------------------------------------------------
        # Sublimation, topmelt and botmelt
        # ------------------------------------------------------
        ncat = int(ice_domain_size.ncat)
        for n in range(1, ncat + 1):
            # 'topmelt' = fsurf - fcondtop.
            topmelt_file[n - 1] = atm_data_dir + "/MONTHLY/topmeltn" + str(n) + ".1996.nc"
            topmelt_file[n - 1] = file_year(topmelt_file[n - 1], yr)
            # 'botmelt' = fcondtop.
            botmelt_file[n - 1] = atm_data_dir + "/MONTHLY/botmeltn" + str(n) + ".1996.nc"
            botmelt_file[n - 1] = file_year(botmelt_file[n - 1], yr)
        # 'sublim' = - flat / Lsub.
        sublim_file = atm_data_dir + "/MONTHLY/sublim.1996.nc"
        sublim_file = file_year(sublim_file, yr)
        for n in range(ncat):
            nu.write(f" {topmelt_file[n]}\n")
            nu.write(f" {botmelt_file[n]}\n")
        nu.write(f" {sublim_file}\n")
    nu.flush()


def hadgem_data():
    """read HadGEM or HadGAM atmospheric data

    authors: Alison McLaren, Met Office
    """
    global rain_file, snow_file, uwind_file, vwind_file, strax_file, stray_file, wind_file
    global fsw_file, flw_file, tair_file, rhoa_file, humid_file, sublim_file
    subname = "(hadgem_data)"
    if local_debug:
        ice_fileunits.nu_diag.write(f" {subname} fdbg start\n")

    queried = _params("Lsub", "calc_strair", "calc_Tsfc")
    Lsub = queried["Lsub"]
    calc_strair = bool(queried["calc_strair"])
    calc_Tsfc = bool(queried["calc_Tsfc"])

    #-------------------------------------------------------------------
    # monthly data
    #
    # Assume that monthly data values are located in the middle of the
    # month.
    #-------------------------------------------------------------------
    midmonth = 15  # data is given on 15th of every month
    # midmonth = fix(p5 * real(daymo(mmonth)))  ! exact middle
    ixm, ixp, maxrec, recslot, readm = _month_neighbors(midmonth)
    interp_coeff_monthly(recslot)
    mm = int(ice_calendar.mmonth)

    # -----------------------------------------------------------
    # Rainfall and snowfall
    # -----------------------------------------------------------
    rain_file = read_data_nc(
        readm, 0, fyear, ixm, mm, ixp, maxrec, rain_file, "rainfall", frain_data,
        field_loc_center, field_type_scalar,
    )
    snow_file = read_data_nc(
        readm, 0, fyear, ixm, mm, ixp, maxrec, snow_file, "snowfall", fsnow_data,
        field_loc_center, field_type_scalar,
    )
    interpolate_data(fsnow_data, ice_flux.fsnow)
    interpolate_data(frain_data, ice_flux.frain)

    if calc_strair:
        # --------------------------------------------------------
        # Wind velocity
        # --------------------------------------------------------
        uwind_file = read_data_nc(
            readm, 0, fyear, ixm, mm, ixp, maxrec, uwind_file, "u_10", uatm_data,
            field_loc_center, field_type_vector,
        )
        vwind_file = read_data_nc(
            readm, 0, fyear, ixm, mm, ixp, maxrec, vwind_file, "v_10", vatm_data,
            field_loc_center, field_type_vector,
        )
        interpolate_data(uatm_data, ice_flux.uatm)
        interpolate_data(vatm_data, ice_flux.vatm)
    else:
        # --------------------------------------------------------
        # Wind stress
        # --------------------------------------------------------
        strax_file = read_data_nc(
            readm, 0, fyear, ixm, mm, ixp, maxrec, strax_file, "taux", strax_data,
            field_loc_center, field_type_vector,
        )
        stray_file = read_data_nc(
            readm, 0, fyear, ixm, mm, ixp, maxrec, stray_file, "tauy", stray_data,
            field_loc_center, field_type_vector,
        )
        interpolate_data(strax_data, ice_flux.strax)
        interpolate_data(stray_data, ice_flux.stray)
        if calc_Tsfc or bool(ice_init.oceanmixed_ice):
            # --------------------------------------------------
            # Wind speed
            # --------------------------------------------------
            wind_file = read_data_nc(
                readm, 0, fyear, ixm, mm, ixp, maxrec, wind_file, "wind_10", wind_data,
                field_loc_center, field_type_scalar,
            )
            interpolate_data(wind_data, ice_flux.wind)

    # -----------------------------------------------------------
    # SW incoming, LW incoming, air temperature, density and
    # humidity at 10m.
    #
    # Even if these fields are not being used to force the ice
    # (i.e. calc_Tsfc=.false.), they are still needed to generate
    # forcing for mixed layer model or to calculate wind stress
    # -----------------------------------------------------------
    if calc_Tsfc or bool(ice_init.oceanmixed_ice) or calc_strair:
        fsw_file = read_data_nc(
            readm, 0, fyear, ixm, mm, ixp, maxrec, fsw_file, "SW_incoming", fsw_data,
            field_loc_center, field_type_scalar,
        )
        flw_file = read_data_nc(
            readm, 0, fyear, ixm, mm, ixp, maxrec, flw_file, "LW_incoming", flw_data,
            field_loc_center, field_type_scalar,
        )
        tair_file = read_data_nc(
            readm, 0, fyear, ixm, mm, ixp, maxrec, tair_file, "t_10", Tair_data,
            field_loc_center, field_type_scalar,
        )
        rhoa_file = read_data_nc(
            readm, 0, fyear, ixm, mm, ixp, maxrec, rhoa_file, "rho_10", rhoa_data,
            field_loc_center, field_type_scalar,
        )
        humid_file = read_data_nc(
            readm, 0, fyear, ixm, mm, ixp, maxrec, humid_file, "q_10", Qa_data,
            field_loc_center, field_type_scalar,
        )
        interpolate_data(fsw_data, ice_flux.fsw)
        interpolate_data(flw_data, ice_flux.flw)
        interpolate_data(Tair_data, ice_flux.Tair)
        interpolate_data(rhoa_data, ice_flux.rhoa)
        interpolate_data(Qa_data, ice_flux.Qa)

    if not calc_Tsfc:
        # ------------------------------------------------------
        # Sublimation, topmelt and botmelt
        # ------------------------------------------------------
        sublim_file = read_data_nc(
            readm, 0, fyear, ixm, mm, ixp, maxrec, sublim_file, "sublim", sublim_data,
            field_loc_center, field_type_scalar,
        )
        sublim = _zero(ice_domain_size.horizontal_shape())
        topmelt = _zero(sublim.shape)
        botmelt = _zero(sublim.shape)
        interpolate_data(sublim_data, sublim)
        ncat = int(ice_domain_size.ncat)
        for n in range(1, ncat + 1):
            topmelt_file[n - 1] = read_data_nc(
                readm, 0, fyear, ixm, mm, ixp, maxrec, topmelt_file[n - 1],
                "topmeltn" + str(n), topmelt_data[:, :, :, n - 1],
                field_loc_center, field_type_scalar,
            )
            botmelt_file[n - 1] = read_data_nc(
                readm, 0, fyear, ixm, mm, ixp, maxrec, botmelt_file[n - 1],
                "botmeltn" + str(n), botmelt_data[:, :, :, n - 1],
                field_loc_center, field_type_scalar,
            )
            interpolate_data(topmelt_data[:, :, :, n - 1], topmelt)
            interpolate_data(botmelt_data[:, :, :, n - 1], botmelt)
            #--------------------------------------------------------
            # Convert from UM variables to CICE variables
            #  topmelt = fsurf - fcondtop
            #  botmelt = fcondtop  (as zero layer)
            #
            # Convert UM sublimation data into CICE LH flux
            # (sublim = - flatn / Lsub) and have same value for all
            # categories
            #--------------------------------------------------------
            ice_flux.fcondtopn_f[:, :, n - 1] = botmelt
            ice_flux.fsurfn_f[:, :, n - 1] = topmelt + botmelt
            ice_flux.flatn_f[:, :, n - 1] = -sublim * Lsub


def monthly_files(yr):
    """Construct filenames based on the LANL naming conventions for NCAR data.
    Edit for other directory structures or filenames.
    Note: The year number in these filenames does not matter, because
          subroutine file_year will insert the correct year.

    author: Elizabeth C. Hunke, LANL
    """
    global flw_file, rain_file, tair_file, humid_file, strax_file, stray_file, wind_file
    subname = "(monthly_files)"
    if local_debug:
        ice_fileunits.nu_diag.write(f" {subname} fdbg start\n")
    atm_data_dir = str(ice_init.atm_data_dir).strip()
    flw_file = atm_data_dir + "/MONTHLY/cldf.omip.dat"
    rain_file = atm_data_dir + "/MONTHLY/prec.nmyr.dat"
    tair_file = atm_data_dir + "/MONTHLY/t_10.1996.dat"
    tair_file = file_year(tair_file, yr)
    humid_file = atm_data_dir + "/MONTHLY/q_10.1996.dat"
    humid_file = file_year(humid_file, yr)
    # stress/speed is used instead of wind components
    strax_file = atm_data_dir + "/MONTHLY/strx.1996.dat"
    strax_file = file_year(strax_file, yr)
    stray_file = atm_data_dir + "/MONTHLY/stry.1996.dat"
    stray_file = file_year(stray_file, yr)
    wind_file = atm_data_dir + "/MONTHLY/wind.1996.dat"
    wind_file = file_year(wind_file, yr)
    nu = ice_fileunits.nu_diag
    nu.write(" \n")
    nu.write(f" Forcing data year = {fyear}\n")
    nu.write(" Atmospheric data files:\n")
    for name in (flw_file, rain_file, tair_file, humid_file, uwind_file, vwind_file):
        nu.write(f" {name}\n")
    nu.flush()


def monthly_data():
    subname = "(monthly_data)"
    if local_debug:
        ice_fileunits.nu_diag.write(f" {subname} fdbg start\n")

    #-------------------------------------------------------------------
    # monthly data
    #
    # Assume that monthly data values are located in the middle of the
    # month.
    #-------------------------------------------------------------------
    midmonth = 15  # data is given on 15th of every month
    # midmonth = fix(p5 * real(daymo(mmonth)))  ! exact middle
    ixm, ixp, _maxrec, recslot, readm = _month_neighbors(midmonth)
    interp_coeff_monthly(recslot)
    mm = int(ice_calendar.mmonth)
    read_clim_data(readm, 0, ixm, mm, ixp, flw_file, cldf_data, field_loc_center, field_type_scalar)
    read_clim_data(readm, 0, ixm, mm, ixp, rain_file, fsnow_data, field_loc_center, field_type_scalar)
    read_clim_data(readm, 0, ixm, mm, ixp, tair_file, Tair_data, field_loc_center, field_type_scalar)
    read_clim_data(readm, 0, ixm, mm, ixp, humid_file, Qa_data, field_loc_center, field_type_scalar)
    read_clim_data(readm, 0, ixm, mm, ixp, wind_file, wind_data, field_loc_center, field_type_scalar)
    read_clim_data(readm, 0, ixm, mm, ixp, strax_file, strax_data, field_loc_center, field_type_vector)
    read_clim_data(readm, 0, ixm, mm, ixp, stray_file, stray_data, field_loc_center, field_type_vector)

    interpolate_data(cldf_data, cldf)
    interpolate_data(fsnow_data, ice_flux.fsnow)  # units mm/s = kg/m^2/s
    interpolate_data(Tair_data, ice_flux.Tair)
    interpolate_data(Qa_data, ice_flux.Qa)
    interpolate_data(wind_data, ice_flux.wind)
    interpolate_data(strax_data, ice_flux.strax)
    interpolate_data(stray_data, ice_flux.stray)

    nx_block, ny_block = ice_domain_size.horizontal_shape()
    Qa_fixLY(nx_block, ny_block, ice_flux.Tair, ice_flux.Qa)
    hm = ice_grid.hm
    ice_flux.Qa[...] = ice_flux.Qa * hm
    ice_flux.Tair[...] = ice_flux.Tair * hm
    ice_flux.wind[...] = ice_flux.wind * hm
    ice_flux.strax[...] = ice_flux.strax * hm
    ice_flux.stray[...] = ice_flux.stray * hm

    # AOMIP
    ilo, ihi, jlo, jhi = ice_domain_size.physical_bounds()
    compute_shortwave(
        nx_block, ny_block, ilo, ihi, jlo, jhi,
        ice_grid.TLON, ice_grid.TLAT, hm, ice_flux.Qa, cldf, ice_flux.fsw,
    )

    if debug_forcing:
        nu = ice_fileunits.nu_diag
        nu.write(" LY_bulk_data\n")
        for label, field, mask in (
            ("fsw", ice_flux.fsw, ice_grid.tmask),
            ("cldf", cldf, ice_grid.tmask),
            ("fsnow", ice_flux.fsnow, ice_grid.tmask),
            ("Tair", ice_flux.Tair, ice_grid.tmask),
            ("wind", ice_flux.wind, ice_grid.umask),
            ("strax", ice_flux.strax, ice_grid.umask),
            ("stray", ice_flux.stray, ice_grid.umask),
            ("Qa", ice_flux.Qa, ice_grid.tmask),
        ):
            vmin, vmax = _mask_range(field, mask)
            nu.write(f" {label} {vmin} {vmax}\n")
        nu.flush()


def oned_data():
    subname = "(oned_data)"
    if local_debug:
        ice_fileunits.nu_diag.write(f" {subname} fdbg start\n")

    diag = False  # write diagnostic information
    # coefficients for Hyland-Wexler Qa
    ps1 = dbl_kind(0.58002206e4)  # (K)
    ps2 = dbl_kind(1.3914993)
    ps3 = dbl_kind(0.48640239e-1)  # (K^-1)
    ps4 = dbl_kind(0.41764768e-4)  # (K^-2)
    ps5 = dbl_kind(0.14452093e-7)  # (K^-3)
    ps6 = dbl_kind(6.5459673)
    ws1 = dbl_kind(621.97)  # for saturation mixing ratio
    Pair = dbl_kind(1020.0)  # Sea level pressure (hPa)

    if str(ice_init.atm_data_format).strip() == "nc":  # read nc file
        # hourly data beginning Jan 1, 1989, 01:00
        # HARDWIRED for dt = 1 hour!
        istep1 = int(ice_calendar.istep1)
        work = _read_nc_point(uwind_file, istep1, "Uatm", diag)
        ice_flux.uatm[...] = work
        work = _read_nc_point(uwind_file, istep1, "Vatm", diag)
        ice_flux.vatm[...] = work
        Temp = _read_nc_point(uwind_file, istep1, "Tair", diag)
        ice_flux.Tair[...] = Temp

        # hourly solar data beginning Jan 1, 1989, 01:00
        work = _read_nc_point(fsw_file, istep1, "fsw", diag)
        ice_flux.fsw[...] = work

        # hourly interpolated monthly  data beginning Jan 1, 1989, 01:00
        rh = _read_nc_point(humid_file, istep1, "rh", diag)
        work = _read_nc_point(humid_file, istep1, "fsnow", diag)
        ice_flux.fsnow[...] = work

        #-------------------------------------------------------------------
        # Find specific humidity using Hyland-Wexler formulation
        # Hyland, R.W. and A. Wexler, Formulations for the Thermodynamic
        # Properties of the saturated phases of H20 from 173.15K to 473.15K,
        # ASHRAE Trans, 89(2A), 500-519, 1983
        #-------------------------------------------------------------------
        Psat = np.exp(
            -ps1 / Temp + ps2 - ps3 * Temp + ps4 * Temp ** 2 - ps5 * Temp ** 3 + ps6 * np.log(Temp)
        ) * p01  # saturation vapour pressure
        ws = ws1 * Psat / (Pair - Psat)  # saturation mixing ratio
        ice_flux.Qa[...] = rh * ws * p01 / (c1 + rh * ws * p01) * p001
        # specific humidity (kg/kg)

    # flw calculated in prepare_forcing
    ice_flux.rhoa[...] = dbl_kind(1.3)  # air density (kg/m^3)
    cldf[...] = p25  # cloud fraction
    ice_flux.frain[...] = c0  # this is available in hourlymet_rh file


def oned_files():
    global fsw_file, rain_file, uwind_file, vwind_file, tair_file, humid_file
    subname = "(oned_files)"
    if local_debug:
        ice_fileunits.nu_diag.write(f" {subname} fdbg start\n")
    atm_data_dir = str(ice_init.atm_data_dir).strip()
    fsw_file = atm_data_dir + "/hourlysolar_brw1989_5yr.nc"
    rain_file = atm_data_dir + "/hourlymet_rh_5yr.nc"
    uwind_file = atm_data_dir + "/hourlymet_brw1989_5yr.nc"
    vwind_file = atm_data_dir + "/hourlymet_brw1989_5yr.nc"
    tair_file = atm_data_dir + "/hourlymet_brw1989_5yr.nc"
    humid_file = atm_data_dir + "/hourlymet_rh_5yr.nc"
    nu = ice_fileunits.nu_diag
    nu.write(" \n")
    nu.write(" Atmospheric data files:\n")
    for name in (fsw_file, rain_file, uwind_file, vwind_file, tair_file, humid_file):
        nu.write(f" {name}\n")
    nu.flush()


def ocn_data_clim(dt):
    """Interpolate monthly sss, sst data to timestep.
    Restore prognostic sst to data.
    Interpolate fields from U grid to T grid if necessary.

    author: Elizabeth C. Hunke and William H. Lipscomb, LANL
    """
    subname = "(ocn_data_clim)"
    if local_debug:
        ice_fileunits.nu_diag.write(f" {subname} fdbg start\n")

    nu = ice_fileunits.nu_diag
    if int(ice_calendar.istep) == 1 and _ocn_type() == "clim":
        nu.write(" \n")
        nu.write(" SSS data interpolated to timestep:\n")
        nu.write(f" {sss_file}\n")
        nu.write(" \n")
        nu.write(" SST data interpolated to timestep:\n")
        nu.write(f" {sst_file}\n")
        if bool(ice_init.restore_ocn):
            nu.write(f" SST restoring timescale (days) = {ice_init.trestore}\n")
        nu.flush()

    #-------------------------------------------------------------------
    # monthly data
    #
    # Assume that monthly data values are located in the middle of the
    # month.
    #-------------------------------------------------------------------
    if _ocn_type() == "clim":
        midmonth = 15  # data is given on 15th of every month
        # midmonth = fix(p5 * real(daymo(mmonth)))  ! exact middle
        ixm, ixp, _maxrec, recslot, readm = _month_neighbors(midmonth)
        interp_coeff_monthly(recslot)
        mm = int(ice_calendar.mmonth)

        #-------------------------------------------------------------------
        # Read two monthly SSS values and interpolate.
        # Note: SSS is restored instantaneously to data.
        #-------------------------------------------------------------------
        read_clim_data(
            readm, 0, ixm, mm, ixp, sss_file, sss_data, field_loc_center, field_type_scalar
        )
        interpolate_data(sss_data, ice_flux.sss)
        ice_flux.sss[...] = np.maximum(ice_flux.sss, c0)
        ocn_freezing_temperature()

        #-------------------------------------------------------------------
        # Read two monthly SST values and interpolate.
        # Restore toward interpolated value.
        #-------------------------------------------------------------------
        read_clim_data(
            readm, 0, ixm, mm, ixp, sst_file, sst_data, field_loc_center, field_type_scalar
        )
        sstdat = _zero(ice_domain_size.horizontal_shape())
        interpolate_data(sst_data, sstdat)
        if bool(ice_init.restore_ocn):
            ice_flux.sst[...] = ice_flux.sst + (sstdat - ice_flux.sst) * dbl_kind(dt) / trest


def _ocn_frc_names():
    # order is important
    return ("T", "S", "hblt", "U", "V", "dhdx", "dhdy", "qdp")


def _check_ocn_dims(path, subname):
    dataset = _nc_open(path)
    try:
        if "ni" not in dataset.dimensions:
            _abort(subname + " ERROR: inq dimid ni")
        if "nj" not in dataset.dimensions:
            _abort(subname + " ERROR: inq dimid nj")
        nlon = len(dataset.dimensions["ni"])
        nlat = len(dataset.dimensions["nj"])
    finally:
        dataset.close()
    if nlon != int(ice_domain_size.nx_global):
        _abort(subname + "ice: ocn frc file nlon ne nx_global")
    if nlat != int(ice_domain_size.ny_global):
        _abort(subname + "ice: ocn frc file nlat ne ny_global")


def ocn_data_ncar_init():
    """Reads NCAR pop ocean forcing data set 'pop_frc_gx1v3_010815.nc'

    List of ocean forcing fields: Note that order is important!
    (order is determined by field list in vname).

    For ocean mixed layer-----------------------------units

    1  sst------temperature---------------------------(C)
    2  sss------salinity------------------------------(ppt)
    3  hbl------depth---------------------------------(m)
    4  u--------surface u current---------------------(m/s)
    5  v--------surface v current---------------------(m/s)
    6  dhdx-----surface tilt x direction--------------(m/m)
    7  dhdy-----surface tilt y direction--------------(m/m)
    8  qdp------ocean sub-mixed layer heat flux-------(W/m2)

    Fields 4, 5, 6, 7 are on the U-grid; 1, 2, 3, and 8 are
    on the T-grid.

    authors: Bruce Briegleb, NCAR
             Elizabeth Hunke, LANL
    """
    global sst_file
    subname = "(ocn_data_ncar_init)"
    if local_debug:
        ice_fileunits.nu_diag.write(f" {subname} fdbg start\n")

    nu = ice_fileunits.nu_diag
    nu.write(" WARNING: evp_prep calculates surface tilt\n")
    nu.write(" WARNING: stress from geostrophic currents,\n")
    nu.write(" WARNING: not data from ocean forcing file.\n")
    nu.write(" WARNING: Alter ice_dyn_evp.F90 if desired.\n")
    if bool(ice_init.restore_ocn):
        nu.write(f" SST restoring timescale = {ice_init.trestore} days\n")
    sst_file = str(ice_init.ocn_data_dir).strip() + "/" + str(ice_init.oceanmixed_file).strip()
    #---------------------------------------------------------------
    # Read in ocean forcing data from an existing file
    #---------------------------------------------------------------
    nu.write(f" ocean mixed layer forcing data file = {sst_file}\n")
    nu.flush()

    vname = _ocn_frc_names()
    work1 = _zero(ice_domain_size.horizontal_shape())
    if str(ice_init.ocn_data_format).strip() == "nc":
        _check_ocn_dims(sst_file, subname)
        # Read in ocean forcing data for all 12 months
        for n in range(1, nfld + 1):
            for m in range(1, 13):
                # Note: netCDF does single to double conversion if necessary
                # if n >= 4 and n <= 7: NEcorner vector (commented in the Fortran)
                _read_nc_xy(
                    sst_file, m, vname[n - 1], work1, debug_forcing,
                    field_loc_center, field_type_scalar,
                )
                ocn_frc_m[:, :, n - 1, m - 1] = work1
    else:  # binary format
        # nbits = 64
        nrec = 0
        for n in range(1, nfld + 1):
            for m in range(1, 13):
                nrec = nrec + 1
                if 4 <= n <= 7:
                    _bin_record(sst_file, nrec, work1, field_loc_NEcorner, field_type_vector)
                else:
                    _bin_record(sst_file, nrec, work1, field_loc_center, field_type_scalar)
                ocn_frc_m[:, :, n - 1, m - 1] = work1

    # echmod - currents cause Fram outflow to be too large
    #             ocn_frc_m(:,:,:,4,:) = c0
    #             ocn_frc_m(:,:,:,5,:) = c0
    # echmod


def ocn_data_ncar_init_3D():
    """Reads NCAR pop ocean forcing data set 'oceanmixed_ice_depth.nc'

    List of ocean forcing fields: Note that order is important!
    (order is determined by field list in vname).

    For ocean mixed layer-----------------------------units

    1  sst------temperature---------------------------(C)
    2  sss------salinity------------------------------(ppt)
    3  hbl------depth---------------------------------(m)
    4  u--------surface u current---------------------(m/s)
    5  v--------surface v current---------------------(m/s)
    6  dhdx-----surface tilt x direction--------------(m/m)
    7  dhdy-----surface tilt y direction--------------(m/m)
    8  qdp------ocean sub-mixed layer heat flux-------(W/m2)

    All fields are on the T-grid.

    authors: Bruce Briegleb, NCAR
             Elizabeth Hunke, LANL
    """
    global sst_file
    subname = "(ocn_data_ncar_init_3D)"
    if local_debug:
        ice_fileunits.nu_diag.write(f" {subname} fdbg start\n")

    nu = ice_fileunits.nu_diag
    nu.write(" WARNING: evp_prep calculates surface tilt\n")
    nu.write(" WARNING: stress from geostrophic currents,\n")
    nu.write(" WARNING: not data from ocean forcing file.\n")
    nu.write(" WARNING: Alter ice_dyn_evp.F if desired.\n")
    if bool(ice_init.restore_ocn):
        nu.write(f" SST restoring timescale = {ice_init.trestore} days\n")
    sst_file = str(ice_init.ocn_data_dir).strip() + "/" + str(ice_init.oceanmixed_file).strip()
    #---------------------------------------------------------------
    # Read in ocean forcing data from an existing file
    #---------------------------------------------------------------
    nu.write(f" ocean mixed layer forcing data file = {sst_file}\n")
    nu.write("\n")
    nu.flush()

    if str(ice_init.ocn_data_format).strip() != "nc":
        _abort(subname + "new ocean forcing is netcdf only")

    _check_ocn_dims(sst_file, subname)
    vname = _ocn_frc_names()
    work1 = _zero(ice_domain_size.horizontal_shape())
    # Read in ocean forcing data for all 12 months
    for n in range(1, nfld + 1):
        for m in range(1, 13):
            # Note: netCDF does single to double conversion if necessary
            if n == 4 or n == 5:  # 3D currents
                nzlev = 1  # surface currents
                _read_nc_uv(
                    sst_file, m, nzlev, vname[n - 1], work1, debug_forcing,
                    field_loc_center, field_type_scalar,
                )
            else:
                _read_nc_xy(
                    sst_file, m, vname[n - 1], work1, debug_forcing,
                    field_loc_center, field_type_scalar,
                )
            # the land mask used in ocean_mixed_depth.nc does not
            # match our gx1v3 mask (hm)
            filled = np.where(work1 < dbl_kind(-900.0), c0, work1)
            ocn_frc_m[:, :, n - 1, m - 1] = filled

    # Rotate vector quantities and shift to U-grid
    for n in range(4, 7, 2):
        for m in range(1, 13):
            east = np.array(ocn_frc_m[:, :, n - 1, m - 1], copy=True)
            north = np.array(ocn_frc_m[:, :, n, m - 1], copy=True)
            angle = ice_grid.ANGLET
            ocn_frc_m[:, :, n - 1, m - 1] = east * np.cos(angle) + north * np.sin(angle)
            ocn_frc_m[:, :, n, m - 1] = north * np.cos(angle) - east * np.sin(angle)
            shifted = _zero(east.shape)
            ice_grid.grid_average_X2Y("A", ocn_frc_m[:, :, n - 1, m - 1], "T", shifted, "U")
            ocn_frc_m[:, :, n - 1, m - 1] = shifted
            shifted = _zero(east.shape)
            ice_grid.grid_average_X2Y("A", ocn_frc_m[:, :, n, m - 1], "T", shifted, "U")
            ocn_frc_m[:, :, n, m - 1] = shifted


def ocn_data_ncar(dt):
    """Interpolate monthly ocean data to timestep.
    Restore sst if desired. sst is updated with surface fluxes in ice_ocean.F.
    """
    subname = "(ocn_data_ncar)"
    if local_debug:
        ice_fileunits.nu_diag.write(f" {subname} fdbg start\n")

    #-------------------------------------------------------------------
    # monthly data
    #
    # Assume that monthly data values are located in the middle of the
    # month.
    #-------------------------------------------------------------------
    midmonth = 15  # data is given on 15th of every month
    # midmonth = fix(p5 * real(daymo(mmonth)))  ! exact middle
    ixm, ixp, _maxrec, recslot, _readm = _month_neighbors(midmonth)
    interp_coeff_monthly(recslot)
    mm = int(ice_calendar.mmonth)

    sst_data[...] = c0
    work1 = _zero(ice_domain_size.horizontal_shape())
    ocean = ice_grid.hm == c1
    for n in range(nfld, 0, -1):
        # use sst_data arrays as temporary work space until n=1
        if ixm != -99:  # first half of month
            sst_data[:, :, 0] = ocn_frc_m[:, :, n - 1, ixm - 1]
            sst_data[:, :, 1] = ocn_frc_m[:, :, n - 1, mm - 1]
        else:  # second half of month
            sst_data[:, :, 0] = ocn_frc_m[:, :, n - 1, mm - 1]
            sst_data[:, :, 1] = ocn_frc_m[:, :, n - 1, ixp - 1]
        interpolate_data(sst_data, work1)
        # masking by hm is necessary due to NaNs in the data file
        if n == 2:
            ice_flux.sss[...] = c0
            ice_flux.sss[ocean] = work1[ocean]
        elif n == 3:
            ice_flux.hmix[...] = c0
            ice_flux.hmix[ocean] = np.maximum(mixed_layer_depth_default, work1[ocean])
        elif n == 4:
            ice_flux.uocn[...] = c0
            ice_flux.uocn[ocean] = work1[ocean]
        elif n == 5:
            ice_flux.vocn[...] = c0
            ice_flux.vocn[ocean] = work1[ocean]
        elif n == 6:
            ice_flux.ss_tltx[...] = c0
            ice_flux.ss_tltx[ocean] = work1[ocean]
        elif n == 7:
            ice_flux.ss_tlty[...] = c0
            ice_flux.ss_tlty[ocean] = work1[ocean]
        elif n == 8:
            ice_flux.qdp[...] = c0
            ice_flux.qdp[ocean] = work1[ocean]

    ice_flux.sss[...] = np.maximum(ice_flux.sss, c0)
    ice_flux.hmix[...] = np.maximum(ice_flux.hmix, c0)
    ocn_freezing_temperature()

    if bool(ice_init.restore_ocn):
        ice_flux.sst[...] = ice_flux.sst + (work1 - ice_flux.sst) * dbl_kind(dt) / trest
    # else sst is only updated in ice_ocean.F

    # initialize sst properly on first step
    if int(ice_calendar.istep1) <= 1 and not bool(ice_init.restart):
        interpolate_data(sst_data, ice_flux.sst)
        ice_flux.sst[...] = np.where(ocean, np.maximum(ice_flux.sst, ice_flux.Tf), c0)

    if debug_forcing:
        nu = ice_fileunits.nu_diag
        nu.write(" ocn_data_ncar\n")
        for label, field, mask in (
            ("Tf", ice_flux.Tf, ice_grid.tmask),
            ("sst", ice_flux.sst, ice_grid.tmask),
            ("sss", ice_flux.sss, ice_grid.tmask),
            ("hmix", ice_flux.hmix, ice_grid.tmask),
            ("uocn", ice_flux.uocn, ice_grid.umask),
            ("vocn", ice_flux.vocn, ice_grid.umask),
            ("ss_tltx", ice_flux.ss_tltx, ice_grid.umask),
            ("ss_tlty", ice_flux.ss_tlty, ice_grid.umask),
            ("qdp", ice_flux.qdp, ice_grid.tmask),
        ):
            vmin, vmax = _mask_range(field, mask)
            nu.write(f" {label} {vmin} {vmax}\n")
        nu.flush()


def ocn_data_oned():
    """ocean data for oned configuration
    Current (released) values are the same as the defaults (ice_flux.F90)
    """
    subname = "(ocn_data_oned)"
    if local_debug:
        ice_fileunits.nu_diag.write(f" {subname} fdbg start\n")
    ice_flux.sss[...] = dbl_kind(34.0)  # sea surface salinity (ppt)
    ocn_freezing_temperature()
    ice_flux.sst[...] = ice_flux.Tf  # sea surface temp (C)
    ice_flux.uocn[...] = c0  # surface ocean currents (m/s)
    ice_flux.vocn[...] = c0
    ice_flux.ss_tltx[...] = c0  # sea surface tilt (m/m)
    ice_flux.ss_tlty[...] = c0
    ice_flux.frzmlt[...] = c0  # freezing/melting potential (W/m^2)
    ice_flux.qdp[...] = c0  # deep ocean heat flux (W/m^2)
    ice_flux.hmix[...] = mixed_layer_depth_default  # ocean mixed layer depth


def ocn_data_hadgem(dt):
    """Reads in HadGEM ocean forcing data as required from netCDF files
    Current options (selected by ocn_data_type)
    hadgem_sst: read in sst only
    hadgem_sst_uvocn: read in sst plus uocn and vocn

    authors: Ann Keen, Met Office
    """
    global sst_file
    subname = "(ocn_data_hadgem)"
    if local_debug:
        ice_fileunits.nu_diag.write(f" {subname} fdbg start\n")

    #-------------------------------------------------------------------
    # monthly data
    #
    # Assume that monthly data values are located in the middle of the
    # month.
    #-------------------------------------------------------------------
    midmonth = 15  # data is given on 15th of every month
    # midmonth = fix(p5 * real(daymo(mmonth)))  ! exact middle
    ixm, ixp, maxrec, recslot, readm = _month_neighbors(midmonth)
    interp_coeff_monthly(recslot)
    mm = int(ice_calendar.mmonth)
    ocn_data_dir = str(ice_init.ocn_data_dir).strip()

    if int(ice_calendar.istep) == 1:
        nu = ice_fileunits.nu_diag
        nu.write(" \n")
        nu.write(" SST data interpolated to timestep:\n")
        nu.write(f" {ocn_data_dir}/MONTHLY/sst.1997.nc\n")
        if bool(ice_init.restore_ocn):
            nu.write(f" SST restoring timescale (days) = {ice_init.trestore}\n")
        if _ocn_type() == "hadgem_sst_uvocn":
            nu.write(" \n")
            nu.write(" uocn and vocn interpolated to timestep:\n")
            nu.write(f" {ocn_data_dir}/MONTHLY/uocn.1997.nc\n")
            nu.write(f" {ocn_data_dir}/MONTHLY/vocn.1997.nc\n")
        nu.flush()

    # -----------------------------------------------------------
    # SST
    # -----------------------------------------------------------
    sst_file = ocn_data_dir + "/MONTHLY/sst.1997.nc"
    sst_file = read_data_nc(
        readm, 0, fyear, ixm, mm, ixp, maxrec, sst_file, "sst", sst_data,
        field_loc_center, field_type_scalar,
    )
    sstdat = _zero(ice_domain_size.horizontal_shape())
    interpolate_data(sst_data, sstdat)
    # Restore SSTs if required
    if bool(ice_init.restore_ocn):
        ice_flux.sst[...] = ice_flux.sst + (sstdat - ice_flux.sst) * dbl_kind(dt) / trest

    # -----------------------------------------------------------
    # Ocean currents
    # --------------
    # Values read in are on T grid and oriented geographically, hence
    # vectors need to be rotated to model grid and then interpolated
    # to U grid.
    # Also need to be converted from cm s-1 (UM) to m s-1 (CICE)
    # -----------------------------------------------------------
    if _ocn_type() == "hadgem_sst_uvocn":
        filename = ocn_data_dir + "/MONTHLY/uocn.1997.nc"
        read_data_nc(
            readm, 0, fyear, ixm, mm, ixp, maxrec, filename, "uocn", uocn_data,
            field_loc_center, field_type_vector,
        )
        interpolate_data(uocn_data, ice_flux.uocn)
        filename = ocn_data_dir + "/MONTHLY/vocn.1997.nc"
        read_data_nc(
            readm, 0, fyear, ixm, mm, ixp, maxrec, filename, "vocn", vocn_data,
            field_loc_center, field_type_vector,
        )
        interpolate_data(vocn_data, ice_flux.vocn)
        #-----------------------------------------------------------------
        # Rotate zonal/meridional vectors to local coordinates,
        # and change  units
        #-----------------------------------------------------------------
        workx = np.array(ice_flux.uocn, copy=True)
        worky = np.array(ice_flux.vocn, copy=True)
        angle = ice_grid.ANGLET
        ice_flux.uocn[...] = workx * np.cos(angle) + worky * np.sin(angle)
        ice_flux.vocn[...] = worky * np.cos(angle) - workx * np.sin(angle)
        ice_flux.uocn[...] = ice_flux.uocn * cm_to_m
        ice_flux.vocn[...] = ice_flux.vocn * cm_to_m
        #-----------------------------------------------------------------
        # Interpolate to U grid
        #-----------------------------------------------------------------
        # tcraig, this is now computed in dynamics for consistency


def hycom_atm_files():
    global fsw_file, flw_file, rain_file, uwind_file, vwind_file, tair_file, humid_file
    global jday_atm, Njday_atm
    subname = "(hycom_atm_files)"
    if local_debug:
        ice_fileunits.nu_diag.write(f" {subname} fdbg start\n")
    atm_data_dir = str(ice_init.atm_data_dir).strip()
    fsw_file = atm_data_dir + "/forcing.shwflx.nc"
    flw_file = atm_data_dir + "/forcing.radflx.nc"
    rain_file = atm_data_dir + "/forcing.precip.nc"
    uwind_file = atm_data_dir + "/forcing.wndewd.nc"
    vwind_file = atm_data_dir + "/forcing.wndnwd.nc"
    tair_file = atm_data_dir + "/forcing.airtmp.nc"
    humid_file = atm_data_dir + "/forcing.vapmix.nc"

    # Read time vector from "tair_file"
    varname = "MT"
    Njday_atm = _nc_dimension_length(tair_file, varname)
    jday_atm = _read_nc_vec(tair_file, Njday_atm, varname, True)

    nu = ice_fileunits.nu_diag
    nu.write(" \n")
    nu.write(f" CICE: Atm. (hycomdate) Start = {jday_atm[0]}\n")
    nu.write(f" CICE: Atm. (hycomdate) End   = {jday_atm[Njday_atm - 1]}\n")
    nu.write(f" CICE: Total Atm timesteps    = {Njday_atm}\n")
    nu.write(" CICE: Atmospheric forcing files:\n")
    for name in (fsw_file, flw_file, rain_file, uwind_file, vwind_file, tair_file, humid_file):
        nu.write(f" {name}\n")
    nu.flush()


def hycom_atm_data():
    global oldrecnum
    subname = "(hycom_atm_data)"
    if local_debug:
        ice_fileunits.nu_diag.write(f" {subname} fdbg start\n")

    queried = _params("Tffresh", "secday")
    Tffresh = queried["Tffresh"]
    secday = queried["secday"]

    # current time in HYCOM jday units (HYCOM ref year: 1900,12,31,000000)
    hcdate = dbl_kind(
        ice_calendar.compute_days_between(
            1900, 12, 31,
            int(ice_calendar.myear), int(ice_calendar.mmonth), int(ice_calendar.mday),
        )
    ) + dbl_kind(ice_calendar.msec) / secday

    # Init recnum try
    recnum = min(max(oldrecnum, 1), Njday_atm - 1)

    # Find correct time in ATM data ... assume cont. incr. time-axis
    while recnum < Njday_atm:
        left = jday_atm[recnum - 1]
        right = jday_atm[recnum]
        if left <= hcdate <= right:
            break
        if abs(hcdate - left) < p001:  # Accept within tolerance = 0.001 days
            break
        if abs(hcdate - right) < p001:
            break
        recnum = recnum + 1

    # Last Atm date might be the same as last CICE date.
    recnum = min(recnum, Njday_atm - 1)

    # Check if current time do not exceed last forcing time
    # + check forcing is available before (or at) current forcing time
    if hcdate > jday_atm[recnum] + p001 or hcdate < jday_atm[recnum - 1] - p001:
        nu = ice_fileunits.nu_diag
        nu.write(f" ERROR: CICE: Atm forcing not available at hcdate = {hcdate}\n")
        nu.write(
            f" ERROR: CICE: myear, yday ,msec = {ice_calendar.myear} {ice_calendar.yday} {ice_calendar.msec}\n"
        )
        _abort("ERROR: CICE stopped")

    # Compute interpolation coefficients
    interp_coeff2(hcdate, jday_atm[recnum - 1], jday_atm[recnum])

    # Read
    read6 = False
    if int(ice_calendar.istep) == 1 or oldrecnum != recnum:
        read6 = True

    if str(ice_init.atm_data_format).strip() == "nc":
        if read6:
            ice_fileunits.nu_diag.write(
                f" CICE: Atm. read: = {jday_atm[recnum - 1]} {jday_atm[recnum]}\n"
            )
        read_data_nc_hycom(
            read6, recnum, tair_file, "airtmp", Tair_data, field_loc_center, field_type_scalar
        )
        read_data_nc_hycom(
            read6, recnum, uwind_file, "wndewd", uatm_data, field_loc_center, field_type_vector
        )
        read_data_nc_hycom(
            read6, recnum, vwind_file, "wndnwd", vatm_data, field_loc_center, field_type_vector
        )
        read_data_nc_hycom(
            read6, recnum, humid_file, "vapmix", Qa_data, field_loc_center, field_type_scalar
        )
        read_data_nc_hycom(
            read6, recnum, fsw_file, "shwflx", fsw_data, field_loc_center, field_type_scalar
        )
        read_data_nc_hycom(
            read6, recnum, flw_file, "radflx", flw_data, field_loc_center, field_type_scalar
        )
        read_data_nc_hycom(
            read6, recnum, rain_file, "precip", fsnow_data, field_loc_center, field_type_scalar
        )
    else:
        _abort(subname + "ERROR: atm_data_format unavailable for hycom")

    # Interpolate
    if debug_forcing:
        ice_fileunits.nu_diag.write(f" CICE: Atm. interpolate: = {hcdate} {c1intp} {c2intp}\n")
    interpolate_data(Tair_data, ice_flux.Tair)
    interpolate_data(uatm_data, ice_flux.uatm)
    interpolate_data(vatm_data, ice_flux.vatm)
    interpolate_data(fsw_data, ice_flux.fsw)
    interpolate_data(flw_data, ice_flux.flw)
    interpolate_data(Qa_data, ice_flux.Qa)
    interpolate_data(fsnow_data, ice_flux.fsnow)

    # Adjust data forcing to CICE units
    # Air temperature: Degrees --> Kelvin
    ice_flux.Tair[...] = ice_flux.Tair + Tffresh

    # Save record number for next time step
    oldrecnum = recnum


def read_data_nc_point(
    flag, recd, yr, ixm, ixx, ixp, maxrec, data_file, fieldname, field_data, field_loc, field_type
):
    """If data is at the beginning of a one-year record, get data from
     the previous year.
    If data is at the end of a one-year record, get data from the
     following year.
    If no earlier data exists (beginning of fyear_init), then
     (1) For monthly data, get data from the end of fyear_final.
     (2) For more frequent data, let the ixm value equal the
         first value of the year.
    If no later data exists (end of fyear_final), then
     (1) For monthly data, get data from the beginning of fyear_init.
     (2) For more frequent data, let the ixp value
         equal the last value of the year.
    In other words, we assume persistence when daily or 6-hourly
      data is missing, and we assume periodicity when monthly data
      is missing.
    """
    subname = "(read_data_nc_point)"
    if local_debug:
        ice_fileunits.nu_diag.write(f" {subname} fdbg start\n")
    _ = (field_loc, field_type)

    field_data[...] = c0  # to satisfy intent(out) attribute
    _note_debug()
    if debug_forcing:
        ice_fileunits.nu_diag.write(f"   {data_file}\n")

    if flag:
        #-----------------------------------------------------------------
        # Initialize record counters
        # (n2, n4 will change only at the very beginning or end of
        #  a forcing cycle.)
        #-----------------------------------------------------------------
        n2 = ixm
        n4 = ixp
        arg = 0

        #-----------------------------------------------------------------
        # read data
        #-----------------------------------------------------------------
        if ixm != -99:
            # currently in first half of data interval
            if ixx <= 1:
                if yr > int(ice_init.fyear_init):  # get data from previous year
                    # file_year(data_file, yr - 1)
                    pass
                else:  # yr = fyear_init, no prior data exists
                    if maxrec > 12:  # extrapolate from first record
                        if ixx == 1:
                            n2 = ixx
                    # else: file_year(data_file, fyear_final)
            # write(nu_diag,*) '!! read_data_nc !!!', trim(data_file)
            arg = 1
            field_data[arg - 1] = _read_nc_point(data_file, recd + n2, fieldname, debug_forcing)

        # always read ixx data from data file for current year
        # file_year(data_file, yr)
        arg = arg + 1
        field_data[arg - 1] = _read_nc_point(data_file, recd + ixx, fieldname, debug_forcing)

        if ixp != -99:
            # currently in latter half of data interval
            if ixx == maxrec:
                if yr < fyear_final:  # get data from following year
                    # file_year(data_file, yr + 1)
                    pass
                else:  # yr = fyear_final, no more data exists
                    if maxrec > 12:  # extrapolate from ixx
                        n4 = ixx
                    # else: file_year(data_file, fyear_init)
            arg = arg + 1
            field_data[arg - 1] = _read_nc_point(data_file, recd + n4, fieldname, debug_forcing)
    return data_file


def ISPOL_files():
    global fsw_file, flw_file, rain_file, uwind_file, vwind_file, tair_file, humid_file
    subname = "(ISPOL_files)"
    if local_debug:
        ice_fileunits.nu_diag.write(f" {subname} fdbg start\n")
    atm_data_dir = str(ice_init.atm_data_dir).strip()
    fsw_file = atm_data_dir + "/fsw_sfc_4Xdaily.nc"
    flw_file = atm_data_dir + "/flw_sfc_4Xdaily.nc"
    rain_file = atm_data_dir + "/fsnow_sfc_daily_mod3.nc"
    uwind_file = atm_data_dir + "/uatm_10m_daily.nc"
    vwind_file = atm_data_dir + "/vatm_10m_daily.nc"
    tair_file = atm_data_dir + "/Tair_2m_daily.nc"
    humid_file = atm_data_dir + "/Qa_2m_daily.nc"
    nu = ice_fileunits.nu_diag
    nu.write(" \n")
    nu.write(" Atmospheric data files:\n")
    for name in (fsw_file, flw_file, rain_file, uwind_file, vwind_file, tair_file, humid_file):
        nu.write(f" {name}\n")
    nu.flush()


def ISPOL_data():
    """Defines atmospheric data fields for Antarctic Weddell sea location

    authors: Nicole Jeffery, LANL
    """
    global oldrecnum, oldrecnum4X
    global Tair_data_p, Qa_data_p, fsnow_data_p, fsw_data_p, flw_data_p, uatm_data_p, vatm_data_p
    subname = "(ISPOL_data)"
    if local_debug:
        ice_fileunits.nu_diag.write(f" {subname} fdbg start\n")

    secday = _params("secday")["secday"]
    # coefficients for Hyland-Wexler Qa are unused here; Qa is read directly
    lapse_rate = dbl_kind(0.0065)  # (K/m) lapse rate over sea level
    recnum = oldrecnum
    recnum4X = oldrecnum4X

    if str(ice_init.atm_data_format).strip() == "nc":  # read nc file
        #-------------------------------------------------------------------
        # data from NCEP_DOE Reanalysis 2 and Bareiss et al 2008
        # daily data located at the end of the 24-hour period.
        #-------------------------------------------------------------------
        dataloc = 2  # data located at end of interval
        sec1hr = secday  # seconds in day
        maxrec = 366
        # current record number
        recnum = int(ice_calendar.yday)
        # Compute record numbers for surrounding data (2 on each side)
        ixm = (recnum + maxrec - 2) % maxrec + 1
        ixx = (recnum - 1) % maxrec + 1
        # ixp = mod(recnum, maxrec) + 1
        # Compute interpolation coefficients
        # If data is located at the end of the time interval, then the
        #  data value for the current record goes in slot 2
        recslot = 2
        ixp = -99
        interp_coeff(recnum, recslot, sec1hr, dataloc)
        read1 = int(ice_calendar.istep) == 1 or oldrecnum != recnum

        # Daily 2m Air temperature 1991
        read_data_nc_point(
            read1, 0, fyear, ixm, ixx, ixp, maxrec, tair_file, "Tair", Tair_data_p,
            field_loc_center, field_type_scalar,
        )
        ice_flux.Tair[...] = c1intp * Tair_data_p[0] + c2intp * Tair_data_p[1] - lapse_rate * dbl_kind(8.0)

        read_data_nc_point(
            read1, 0, fyear, ixm, ixx, ixp, maxrec, humid_file, "Qa", Qa_data_p,
            field_loc_center, field_type_scalar,
        )
        Qa_pnt = c1intp * Qa_data_p[0] + c2intp * Qa_data_p[1]
        ice_flux.Qa[...] = Qa_pnt

        read_data_nc_point(
            read1, 0, fyear, ixm, ixx, ixp, maxrec, uwind_file, "uatm", uatm_data_p,
            field_loc_center, field_type_scalar,
        )
        ice_flux.uatm[...] = c1intp * uatm_data_p[0] + c2intp * uatm_data_p[1]

        read_data_nc_point(
            read1, 0, fyear, ixm, ixx, ixp, maxrec, vwind_file, "vatm", vatm_data_p,
            field_loc_center, field_type_scalar,
        )
        ice_flux.vatm[...] = c1intp * vatm_data_p[0] + c2intp * vatm_data_p[1]

        read_data_nc_point(
            read1, 0, fyear, ixm, ixx, ixp, maxrec, rain_file, "fsnow", fsnow_data_p,
            field_loc_center, field_type_scalar,
        )
        ice_flux.fsnow[...] = c1intp * fsnow_data_p[0] + c2intp * fsnow_data_p[1]

        #-----------------------------
        # fsw and flw are every 6 hours
        #------------------------------
        dataloc = 2  # data located at end of interval
        sec1hr = secday / c4  # seconds in 6 hours
        maxrec = 1460  # 366*4
        # current record number
        recnum4X = 4 * int(ice_calendar.yday) - 3 + int(float(ice_calendar.msec) / float(sec1hr))
        # Compute record numbers for surrounding data (2 on each side)
        ixm = (recnum4X + maxrec - 2) % maxrec + 1
        ixx = (recnum4X - 1) % maxrec + 1
        # Compute interpolation coefficients
        # If data is located at the end of the time interval, then the
        #  data value for the current record goes in slot 2
        recslot = 2
        ixp = -99
        interp_coeff(recnum4X, recslot, sec1hr, dataloc)
        read1 = int(ice_calendar.istep) == 1 or oldrecnum4X != recnum4X

        read_data_nc_point(
            read1, 0, fyear, ixm, ixx, ixp, maxrec, fsw_file, "fsw", fsw_data_p,
            field_loc_center, field_type_scalar,
        )
        ice_flux.fsw[...] = c1intp * fsw_data_p[0] + c2intp * fsw_data_p[1]

        read_data_nc_point(
            read1, 0, fyear, ixm, ixx, ixp, maxrec, flw_file, "flw", flw_data_p,
            field_loc_center, field_type_scalar,
        )
        ice_flux.flw[...] = c1intp * flw_data_p[0] + c2intp * flw_data_p[1]

    # flw given cldf and Tair calculated in prepare_forcing
    #-----------------------------
    # fixed data
    # May not be needed
    #-----------------------------
    ice_flux.rhoa[...] = dbl_kind(1.3)  # air density (kg/m^3)
    cldf[...] = c1  # 0.25_dbl_kind ! cloud fraction
    ice_flux.frain[...] = c0  # this is available in hourlymet_rh file

    # Save record number for next time step
    oldrecnum = recnum
    oldrecnum4X = recnum4X


def ocn_data_ispol_init():
    """Reads NCAR pop ocean forcing data set 'pop_frc_gx1v3_010815.nc'
    at the ISPOL location -67.4677N, 310.4375E

    Fields 4, 5, 6, 7 are on the U-grid; 1, 2, 3, and 8 are
    on the T-grid.

    authors: Nicole Jeffery, LANL
    """
    global sst_file
    subname = "(ocn_data_ispol_init)"
    if local_debug:
        ice_fileunits.nu_diag.write(f" {subname} fdbg start\n")

    nu = ice_fileunits.nu_diag
    if bool(ice_init.restore_ocn):
        nu.write(f" SST restoring timescale = {ice_init.trestore} days\n")
    sst_file = str(ice_init.ocn_data_dir).strip() + "/" + str(ice_init.oceanmixed_file).strip()
    #---------------------------------------------------------------
    # Read in ocean forcing data from an existing file
    #---------------------------------------------------------------
    nu.write(f" ocean mixed layer forcing data file = {sst_file}\n")
    nu.flush()

    if str(ice_init.ocn_data_format).strip() != "nc":
        _abort(subname + "new ocean forcing is netcdf only")

    vname = _ocn_frc_names()
    # Read in ocean forcing data for all 12 months
    for n in range(1, nfld + 1):
        for m in range(1, 13):
            # Note: netCDF does single to double conversion if necessary
            # A scalar target selects ice_read_nc_point. The value is then
            # broadcast over the horizontal field.
            if 4 <= n <= 7:
                _ = (field_loc_NEcorner, field_type_vector)
            work = _read_nc_point(sst_file, m, vname[n - 1], debug_forcing)
            ocn_frc_m[:, :, n - 1, m - 1] = work

    # echmod - currents cause Fram outflow to be too large
    ocn_frc_m[:, :, 3, :] = c0
    ocn_frc_m[:, :, 4, :] = c0
    # echmod


def box2001_data_atm():
    """wind fields as in Hunke, JCP 2001
    these are defined at the u point
    authors: Elizabeth Hunke, LANL
    """
    subname = "(box2001_data_atm)"
    if local_debug:
        ice_fileunits.nu_diag.write(f" {subname} fdbg start\n")

    queried = _params("pi", "pi2", "puny", "secday")
    pi = queried["pi"]
    pi2 = queried["pi2"]
    _ = queried["puny"]
    secday = queried["secday"]
    period = c4 * secday

    ig, jg = global_indices()
    nx = dbl_kind(int(ice_domain_size.nx_global))
    ny = dbl_kind(int(ice_domain_size.ny_global))
    gi = dbl_kind(ig)[:, np.newaxis]
    gj = dbl_kind(jg)[np.newaxis, :]
    # wind components
    ice_flux.uatm[...] = c5 + (np.sin(pi2 * ice_calendar.timesecs / period) - c3) * np.sin(
        pi2 * gi / nx
    ) * np.sin(pi * gj / ny)
    ice_flux.vatm[...] = c5 + (np.sin(pi2 * ice_calendar.timesecs / period) - c3) * np.sin(
        pi * gi / nx
    ) * np.sin(pi2 * gj / ny)
    # wind stress
    ice_flux.wind[...] = np.sqrt(ice_flux.uatm ** 2 + ice_flux.vatm ** 2)
    tau = ice_flux.rhoa * dbl_kind(0.0012) * ice_flux.wind
    ice_flux.strax[...] = ice_state.aice * tau * ice_flux.uatm
    ice_flux.stray[...] = ice_state.aice * tau * ice_flux.vatm

    # initialization test
    # Diagonal wind vectors 1
    # uatm = c1 * real(j-nghost) / real(ny_global)
    # vatm = c1 * real(j-nghost) / real(ny_global)
    # Diagonal wind vectors 2
    # uatm = c1 * real(i-nghost) / real(nx_global)
    # vatm = -c1 * real(i-nghost) / real(nx_global)
    # Wind in x direction
    # uatm = c1 * real(i-nghost) / real(nx_global)
    # vatm = c0
    # Wind in y direction
    # uatm = c0
    # vatm = c1 * real(j-nghost) / real(ny_global)
    # initialization test


def uniform_data_atm(dir, spd=None):
    """uniform wind fields in some direction"""
    subname = "(uniform_data_atm)"
    if local_debug:
        ice_fileunits.nu_diag.write(f" {subname} fdbg start\n")

    # check for optional spd
    if spd is not None:
        atm_val = spd
    else:
        atm_val = c5  # default

    # wind components
    if dir == "NE":
        ice_flux.uatm[...] = atm_val
        ice_flux.vatm[...] = atm_val
    elif dir == "N":
        ice_flux.uatm[...] = c0
        ice_flux.vatm[...] = atm_val
    elif dir == "E":
        ice_flux.uatm[...] = atm_val
        ice_flux.vatm[...] = c0
    elif dir == "S":
        ice_flux.uatm[...] = c0
        ice_flux.vatm[...] = -atm_val
    elif dir == "W":
        ice_flux.uatm[...] = -atm_val
        ice_flux.vatm[...] = c0
    else:
        _abort(subname + "ERROR: dir unknown, dir = " + dir)

    # wind stress
    ice_flux.wind[...] = np.sqrt(ice_flux.uatm ** 2 + ice_flux.vatm ** 2)
    tau = ice_flux.rhoa * dbl_kind(0.0012) * ice_flux.wind
    ice_flux.strax[...] = ice_state.aice * tau * ice_flux.uatm
    ice_flux.stray[...] = ice_state.aice * tau * ice_flux.vatm


def _init_wave_bins(nfreq):
    """Frequency bins from icepack_init_wave (icepack_wavefracspec.F90).

    That routine is not in the Python column package. The values below are
    its hardwired NIWA/Wavewatch table (C = 1.1, 25 frequencies).
    """
    # set for 25 frequencies
    profile_data = np.zeros(100, dtype=dbl_kind)
    # FOR TESTING ONLY - do not use for actual runs!!
    profile_data[0] = 0.00015429197810590267
    profile_data[1] = 0.002913531381636858
    profile_data[2] = 0.02312942035496235
    profile_data[3] = 0.07201970368623734
    profile_data[4] = 0.06766948103904724
    profile_data[5] = 0.005527883302420378
    profile_data[6] = 3.326293881400488e-05
    profile_data[7] = 6.815936703929992e-10
    profile_data[8] = 2.419401186610744e-20

    # hardwired for wave coupling with NIWA version of Wavewatch
    # From Wavewatch, f(n+1) = C*f(n) where C is a constant set by the user
    # These freq are for C = 1.1
    wavefreq = np.array(
        [
            0.04118, 0.045298, 0.0498278, 0.05481058, 0.06029164,
            0.06632081, 0.07295289, 0.08024818, 0.08827299, 0.09710029,
            0.10681032, 0.11749136, 0.1292405, 0.14216454, 0.15638101,
            0.17201911, 0.18922101, 0.20814312, 0.22895744, 0.25185317,
            0.27703848, 0.30474234, 0.33521661, 0.36873826, 0.40561208,
        ],
        dtype=dbl_kind,
    )
    if int(nfreq) != wavefreq.size:
        _abort(
            "(icepack_init_wave) wavefreq is hardwired for 25 frequencies, nfreq="
            + str(nfreq)
        )
    # boundaries of bin n are at f(n)*sqrt(1/C) and f(n)*sqrt(C)
    dwavefreq = wavefreq * (math.sqrt(1.1) - math.sqrt(1.0 / 1.1))
    ice_arrays_column.wavefreq[...] = wavefreq
    ice_arrays_column.dwavefreq[...] = np.array(dwavefreq, dtype=dbl_kind)
    return np.array(profile_data[: int(nfreq)], dtype=dbl_kind)


def get_wave_spec():
    global debug_forcing
    subname = "(get_wave_spec)"
    if local_debug:
        ice_fileunits.nu_diag.write(f" {subname} fdbg start\n")

    queried = _params("wave_spec", "wave_spec_type")
    wave_spec = bool(queried["wave_spec"])
    wave_spec_type = str(queried["wave_spec_type"]).strip()

    # if no wave data is provided, wave_spectrum is zero everywhere
    ice_arrays_column.wave_spectrum[...] = c0
    debug_forcing = False

    # wave spectrum and frequencies
    if wave_spec:
        # get hardwired frequency bin info and a dummy wave spectrum profile
        # the latter is used if wave_spec_type == profile
        nfreq = int(ice_init.nfreq)
        _init_wave_bins(nfreq)
        # read more realistic data from a file
        if wave_spec_type in ("constant", "random"):
            spec_name = str(ice_init.wave_spec_file)
            if spec_name[:4] == "unkn":
                _abort(subname + "ERROR: wave_spec_file " + spec_name.strip())
            else:
                wave_spec_data()


def wave_spec_data():
    """Read in wave spectrum forcing as a function of time. 6 hourly
    LR started working from JRA55_data routine
    Changed fields, and changed 3 hourly to 6 hourly
    """
    global oldrecnum
    subname = "(wave_spec_data)"
    debug_n_d = False  # usually false
    secday = _params("secday")["secday"]
    nfreq = int(ice_init.nfreq)
    _init_wave_bins(nfreq)

    # spec_file = trim(ocn_data_dir)//'/'//trim(wave_spec_file)
    spec_file = str(ice_init.wave_spec_file).strip()
    wave_spectrum_data[...] = c0
    ice_arrays_column.wave_spectrum[...] = c0
    yr = fyear  # current year
    #-------------------------------------------------------------------
    # 6-hourly data
    #
    # Assume that the 6-hourly value is located at the end of the
    #  6-hour period.  This is the convention for NCEP reanalysis data.
    #  E.g. record 1 gives conditions at 6 am GMT on 1 January.
    #-------------------------------------------------------------------
    dataloc = 2  # data located at end of interval
    sec6hr = secday / c4  # seconds in 6 hours
    # maxrec = 2920            ! 365*8; for leap years = 366*8
    if bool(ice_calendar.use_leap_years):
        ice_calendar.days_per_year = 366  # overrides setting of 365 in ice_calendar
    maxrec = int(ice_calendar.days_per_year) * 4
    if int(ice_calendar.days_per_year) == 365 and _fmod(yr, 4) == 0:
        _abort("days_per_year should be set to 366 for leap years")

    # current record number
    recnum = 4 * int(ice_calendar.yday) - 3 + int(float(ice_calendar.msec) / float(sec6hr))
    # Compute record numbers for surrounding data (2 on each side)
    _ixm = (recnum + maxrec - 2) % maxrec + 1
    _ixx = (recnum - 1) % maxrec + 1
    # Compute interpolation coefficients
    # If data is located at the end of the time interval, then the
    #  data value for the current record goes in slot 2
    recslot = 2
    _ixp = -99
    interp_coeff(recnum, recslot, sec6hr, dataloc)
    _ = (_ixm, _ixx, _ixp)

    # Read
    _read6 = int(ice_calendar.istep) == 1 or oldrecnum != recnum
    _ = _read6
    #-------------------------------------------------------------------
    # File is NETCDF
    # file variable names are:
    # efreq   (wave spectrum, energy as a function of wave frequency UNITS)
    #-------------------------------------------------------------------
    _read_nc_xyf(
        spec_file, recnum, "efreq", wave_spectrum_data[:, :, :, 0], debug_n_d,
        field_loc_center, field_type_scalar,
    )
    _read_nc_xyf(
        spec_file, recnum, "efreq", wave_spectrum_data[:, :, :, 1], debug_n_d,
        field_loc_center, field_type_scalar,
    )

    # Interpolate
    interpolate_wavespec_data(wave_spectrum_data, ice_arrays_column.wave_spectrum)

    # Save record number
    oldrecnum = recnum

    if local_debug:
        ice_fileunits.nu_diag.write(f" wave_spec_data {spec_file}\n")
        ice_fileunits.nu_diag.write(f" maxrec {maxrec}\n")
        ice_fileunits.nu_diag.write(f" days_per_year {ice_calendar.days_per_year}\n")
        ice_fileunits.nu_diag.flush()


def init_snowtable():
    """initial snow aging lookup table

    Dry snow metamorphism table
    snicar_drdt_bst_fit_60_c070416.nc
    Flanner (file metadata units mislabelled)
    drdsdt0 (10^-6 m/hr) tau (10^-6 m)
    """
    subname = "(init_snowtable)"
    #-----------------------------------------------------------------
    # read table of snow aging parameters
    #-----------------------------------------------------------------
    queried = _params("snw_aging_table", "isnw_rhos", "isnw_Tgrd", "isnw_T")
    snw_aging_table = str(queried["snw_aging_table"]).strip()
    idx_rhos_max = int(queried["isnw_rhos"])
    idx_Tgrd_max = int(queried["isnw_Tgrd"])
    idx_T_max = int(queried["isnw_T"])
    diag = False
    snw_filename = str(ice_init.snw_filename).strip()

    nu = ice_fileunits.nu_diag
    nu.write(" \n")
    nu.write(f" Snow aging file: {snw_filename}\n")
    nu.flush()

    if snw_aging_table == "snicar":
        # just read the 3d data and pass it in
        shape = (idx_rhos_max, idx_Tgrd_max, idx_T_max)
        snowage_tau = _read_nc_3d(snw_filename, str(ice_init.snw_tau_fname).strip(), shape, diag)
        snowage_kappa = _read_nc_3d(snw_filename, str(ice_init.snw_kappa_fname).strip(), shape, diag)
        snowage_drdt0 = _read_nc_3d(snw_filename, str(ice_init.snw_drdt0_fname).strip(), shape, diag)
        nu.write(f" {subname}  \n")
        nu.write(f" {subname} Successfully read snow aging properties:\n")
        nu.write(f" {subname} snw_aging_table = {snw_aging_table}\n")
        nu.write(f" {subname} idx_rhos_max = {idx_rhos_max}\n")
        nu.write(f" {subname} idx_Tgrd_max = {idx_Tgrd_max}\n")
        nu.write(f" {subname} idx_T_max    = {idx_T_max}\n")
        nu.write(f" {subname} Data at rhos, Tgrd, T at first index \n")
        nu.write(f" {subname} snoage_tau (1,1,1)         = {snowage_tau[0, 0, 0]}\n")
        nu.write(f" {subname} snoage_kappa (1,1,1)       = {snowage_kappa[0, 0, 0]}\n")
        nu.write(f" {subname} snoage_drdt0 (1,1,1)       = {snowage_drdt0[0, 0, 0]}\n")
        nu.write(f" {subname} Data at rhos, Tgrd, T at max index\n")
        nu.write(
            f" {subname} snoage_tau (max,max,max)   = "
            f"{snowage_tau[idx_rhos_max - 1, idx_Tgrd_max - 1, idx_T_max - 1]}\n"
        )
        nu.write(
            f" {subname} snoage_kappa (max,max,max) = "
            f"{snowage_kappa[idx_rhos_max - 1, idx_Tgrd_max - 1, idx_T_max - 1]}\n"
        )
        nu.write(
            f" {subname} snoage_drdt0 (max,max,max) = "
            f"{snowage_drdt0[idx_rhos_max - 1, idx_Tgrd_max - 1, idx_T_max - 1]}\n"
        )
        nu.flush()
        icepack_parameters.icepack_init_parameters(
            snowage_tau_in=snowage_tau,
            snowage_kappa_in=snowage_kappa,
            snowage_drdt0_in=snowage_drdt0,
        )
    else:
        # read everything and pass it in
        idx_rhos_max = _nc_dimension_length(snw_filename, str(ice_init.snw_rhos_fname).strip())
        idx_Tgrd_max = _nc_dimension_length(snw_filename, str(ice_init.snw_Tgrd_fname).strip())
        idx_T_max = _nc_dimension_length(snw_filename, str(ice_init.snw_T_fname).strip())
        snowage_rhos = _read_nc_1d(snw_filename, str(ice_init.snw_rhos_fname).strip(), idx_rhos_max, diag)
        snowage_Tgrd = _read_nc_1d(snw_filename, str(ice_init.snw_Tgrd_fname).strip(), idx_Tgrd_max, diag)
        snowage_T = _read_nc_1d(snw_filename, str(ice_init.snw_T_fname).strip(), idx_T_max, diag)
        shape = (idx_rhos_max, idx_Tgrd_max, idx_T_max)
        snowage_tau = _read_nc_3d(snw_filename, str(ice_init.snw_tau_fname).strip(), shape, diag)
        snowage_kappa = _read_nc_3d(snw_filename, str(ice_init.snw_kappa_fname).strip(), shape, diag)
        snowage_drdt0 = _read_nc_3d(snw_filename, str(ice_init.snw_drdt0_fname).strip(), shape, diag)
        nu.write(f" {subname}  \n")
        nu.write(f" {subname} Successfully read snow aging properties:\n")
        nu.write(f" {subname} idx_rhos_max = {idx_rhos_max}\n")
        nu.write(f" {subname} idx_Tgrd_max = {idx_Tgrd_max}\n")
        nu.write(f" {subname} idx_T_max    = {idx_T_max}\n")
        nu.write(
            f" {subname} Data at rhos, Tgrd, T = {snowage_rhos[0]} {snowage_Tgrd[0]} {snowage_T[0]}\n"
        )
        nu.write(f" {subname} snoage_tau (1,1,1)         = {snowage_tau[0, 0, 0]}\n")
        nu.write(f" {subname} snoage_kappa (1,1,1)       = {snowage_kappa[0, 0, 0]}\n")
        nu.write(f" {subname} snoage_drdt0 (1,1,1)       = {snowage_drdt0[0, 0, 0]}\n")
        nu.write(
            f" {subname} Data at rhos, Tgrd, T = {snowage_rhos[idx_rhos_max - 1]} "
            f"{snowage_Tgrd[idx_Tgrd_max - 1]} {snowage_T[idx_T_max - 1]}\n"
        )
        nu.write(
            f" {subname} snoage_tau (max,max,max)   = "
            f"{snowage_tau[idx_rhos_max - 1, idx_Tgrd_max - 1, idx_T_max - 1]}\n"
        )
        nu.write(
            f" {subname} snoage_kappa (max,max,max) = "
            f"{snowage_kappa[idx_rhos_max - 1, idx_Tgrd_max - 1, idx_T_max - 1]}\n"
        )
        nu.write(
            f" {subname} snoage_drdt0 (max,max,max) = "
            f"{snowage_drdt0[idx_rhos_max - 1, idx_Tgrd_max - 1, idx_T_max - 1]}\n"
        )
        nu.flush()
        # Fortran keyword isnw_t_in is the same symbol as isnw_T
        icepack_parameters.icepack_init_parameters(
            isnw_T_in=idx_T_max,
            isnw_Tgrd_in=idx_Tgrd_max,
            isnw_rhos_in=idx_rhos_max,
            snowage_rhos_in=snowage_rhos,
            snowage_Tgrd_in=snowage_Tgrd,
            snowage_T_in=snowage_T,
            snowage_tau_in=snowage_tau,
            snowage_kappa_in=snowage_kappa,
            snowage_drdt0_in=snowage_drdt0,
        )
