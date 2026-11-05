"""Reads and interpolates forcing data for biogeochemistry

author for JAX version: Xinyuan Zhu
XZ: converted from F90 to Python

authors:  Nicole Jeffery, LANL
         Elizabeth C. Hunke, LANL

One grid, no block index. ``faero_data`` and ``fzaero_data`` are not
called from standalone ``cice_init`` (those calls are commented out);
``faero_default`` is the live aerosol path. Climatology reads use the
helpers already in ``ice_forcing``.
"""

import netCDF4
import numpy as np

from cicecore.cicedyn.general import ice_flux_bgc
from cicecore.cicedyn.general import ice_forcing
from cicecore.cicedyn.general import ice_init
from cicecore.cicedyn.infrastructure.comm.serial.ice_boundary import scatter_global
from cicecore.shared import ice_arrays_column
from cicecore.shared import ice_calendar
from cicecore.shared import ice_domain_size
from cicecore.shared import ice_fileunits
from cicecore.shared.ice_constants import c0, c1, field_loc_center, field_type_scalar, p1
from cicecore.shared.ice_kinds_mod import dbl_kind, int_kind
from icepack.columnphysics import icepack_parameters
from icepack.columnphysics import icepack_tracers

bgcrecnum = int_kind(0)  # old record number (save between steps)

nitdat = None       # data value toward which nitrate is restored
sildat = None       # data value toward which silicate is restored
nit_data = None     # field values at 2 temporal data points
sil_data = None

# field values at 2 temporal data points (ISPOL point series)
sil_data_p = np.zeros(2, dtype=dbl_kind)
nit_data_p = np.zeros(2, dtype=dbl_kind)


def alloc_forcing_bgc():
    """Allocate space for forcing_bgc variables."""
    global nitdat, sildat, nit_data, sil_data
    horiz = ice_domain_size.horizontal_shape()
    nitdat = np.zeros(horiz, dtype=dbl_kind)
    sildat = np.zeros(horiz, dtype=dbl_kind)
    nit_data = np.zeros(horiz + (2,), dtype=dbl_kind)
    sil_data = np.zeros(horiz + (2,), dtype=dbl_kind)
    nitdat[...] = c0
    sildat[...] = c0
    nit_data[...] = c0
    sil_data[...] = c0


def _physical():
    ilo, ihi, jlo, jhi = ice_domain_size.physical_bounds()
    return slice(ilo, ihi + 1), slice(jlo, jhi + 1)


def _ocean_slot(ks):
    """Fortran 1-based bio index into ``ocean_bio_all``."""
    return int(ks) - 1


def get_forcing_bgc():
    """Read and interpolate annual climatologies of silicate and nitrate.
    Restore model quantities to data if desired.

    author: Elizabeth C. Hunke, LANL
    """
    global bgcrecnum, sil_data_p, nit_data_p

    subname = "(get_forcing_bgc)"
    secday = dbl_kind(icepack_parameters.secday)
    tr_bgc_Nit = bool(icepack_tracers.tr_bgc_Nit)
    tr_bgc_Sil = bool(icepack_tracers.tr_bgc_Sil)
    bgc_data_type = str(ice_init.bgc_data_type).strip()
    bgc_data_dir = str(ice_init.bgc_data_dir).strip()
    restore_bgc = bool(ice_init.restore_bgc)
    max_algae = int(icepack_tracers.max_algae)
    max_doc = int(icepack_tracers.max_doc)
    max_dic = int(icepack_tracers.max_dic)
    islice, jslice = _physical()
    sil = ice_flux_bgc.sil
    nit = ice_flux_bgc.nit
    ocean_bio_all = ice_arrays_column.ocean_bio_all
    nu = ice_fileunits.nu_diag

    if bgc_data_type == "clim":
        nit_file = "nitrate_climatologyWOA_gx1v6f_20150107.nc"
        # 'nitrate_WOA2005_surface_monthly'  ! gx1 only
        sil_file = "silicate_climatologyWOA_gx1v6f_20150107.nc"
        # 'silicate_WOA2005_surface_monthly' ! gx1 only
        nit_file = bgc_data_dir + "/" + nit_file
        sil_file = bgc_data_dir + "/" + sil_file

        if int(ice_calendar.istep) == 1:
            if tr_bgc_Sil:
                nu.write(" \n")
                nu.write(" silicate data interpolated to timestep:\n")
                nu.write(f" {sil_file}\n")
            if tr_bgc_Nit:
                nu.write(" \n")
                nu.write(" nitrate data interpolated to timestep:\n")
                nu.write(f" {nit_file}\n")
                if restore_bgc:
                    nu.write(
                        f" bgc restoring timescale (days) = {ice_forcing.trestore}\n"
                    )

        #-------------------------------------------------------------------
        # monthly data
        #
        # Assume that monthly data values are located in the middle of the
        # month.
        #-------------------------------------------------------------------
        midmonth = 15          # data is given on 15th of every month
        # midmonth = fix(p5 * real(daymo(mmonth)))  ! exact middle
        mmonth = int(ice_calendar.mmonth)
        mday = int(ice_calendar.mday)
        maxrec = 12
        ixm = (mmonth + maxrec - 2) % maxrec + 1
        ixp = (mmonth % maxrec) + 1
        if mday >= midmonth:
            ixm = -99  # other two points will be used
        if mday < midmonth:
            ixp = -99
        # recslot = 2 means we use values 1:2, with the current value (2)
        #  in the second slot
        # recslot = 1 means we use values 2:3, with the current value (2)
        #  in the first slot
        recslot = 1            # latter half of month
        if mday < midmonth:
            recslot = 2        # first half of month
        ice_forcing.interp_coeff_monthly(recslot)
        readm = False
        if int(ice_calendar.istep) == 1 or (
            mday == midmonth and int(ice_calendar.msec) == 0
        ):
            readm = True

    #-------------------------------------------------------------------
    # Read two monthly silicate values and interpolate.
    # Restore toward interpolated value.
    #-------------------------------------------------------------------
    if bgc_data_type == "clim" and tr_bgc_Sil:
        # call read_clim_data (readm, 0, ixm, mmonth, ixp, &
        #                      sil_file,  sil_data, &
        #                      field_loc_center, field_type_scalar)
        fieldname = "silicate"
        ice_forcing.read_clim_data_nc(
            readm, 0, ixm, mmonth, ixp, sil_file, fieldname, sil_data,
            field_loc_center, field_type_scalar,
        )
        ice_forcing.interpolate_data(sil_data, sildat)
        ks = 2 * max_algae + max_doc + 3 + max_dic
        if int(ice_calendar.istep) == 1 or not restore_bgc:
            sil[islice, jslice] = sildat[islice, jslice]
            ocean_bio_all[islice, jslice, _ocean_slot(ks)] = sil[islice, jslice]
        elif restore_bgc:
            if ice_forcing.trest is None:
                raise RuntimeError(subname + " trest is not set")
            sil[islice, jslice] = sil[islice, jslice] + (
                sildat[islice, jslice] - sil[islice, jslice]
            ) * dbl_kind(ice_calendar.dt) / dbl_kind(ice_forcing.trest)
            ocean_bio_all[islice, jslice, _ocean_slot(ks)] = sil[islice, jslice]
    elif tr_bgc_Sil:  # bgc_data_type /= 'clim'
        ks = 2 * max_algae + max_doc + 3 + max_dic
        sil[islice, jslice] = dbl_kind(25.0)
        ocean_bio_all[islice, jslice, _ocean_slot(ks)] = sil[islice, jslice]

    #-------------------------------------------------------------------
    # Read two monthly nitrate values and interpolate.
    # Restore toward interpolated value.
    #-------------------------------------------------------------------
    if bgc_data_type == "clim" and tr_bgc_Nit:
        # call read_clim_data (readm, 0, ixm, mmonth, ixp, &
        #                      nit_file, nit_data, &
        #                      field_loc_center, field_type_scalar)
        fieldname = "nitrate"
        ice_forcing.read_clim_data_nc(
            readm, 0, ixm, mmonth, ixp, nit_file, fieldname, nit_data,
            field_loc_center, field_type_scalar,
        )
        ice_forcing.interpolate_data(nit_data, nitdat)
        ks_nit = max_algae + 1
        ks_pon = 2 * max_algae + max_doc + 7 + max_dic
        if int(ice_calendar.istep) == 1 or not restore_bgc:
            nit[islice, jslice] = nitdat[islice, jslice]
        elif restore_bgc:
            if ice_forcing.trest is None:
                raise RuntimeError(subname + " trest is not set")
            nit[islice, jslice] = nit[islice, jslice] + (
                nitdat[islice, jslice] - nit[islice, jslice]
            ) * dbl_kind(ice_calendar.dt) / dbl_kind(ice_forcing.trest)
        ocean_bio_all[islice, jslice, _ocean_slot(ks_nit)] = nit[islice, jslice]
        ocean_bio_all[islice, jslice, _ocean_slot(ks_pon)] = nit[islice, jslice]
    # elseif (trim(nit_data_type) == 'sss'  .AND.  tr_bgc_Nit) then
    #    nit(i,j,iblk) = sss(i,j,iblk)
    elif tr_bgc_Nit:  # bgc_data_type /= 'clim'
        ks_nit = max_algae + 1
        ks_pon = 2 * max_algae + max_doc + 7 + max_dic
        nit[islice, jslice] = dbl_kind(12.0)
        ocean_bio_all[islice, jslice, _ocean_slot(ks_nit)] = nit[islice, jslice]
        ocean_bio_all[islice, jslice, _ocean_slot(ks_pon)] = nit[islice, jslice]

    #-------------------------------------------------------------------
    # Data from Papdimitrious et al., 2007, Limnol. Oceanogr.
    # and WOA at 68oS, 304.5oE :
    # daily data located at the end of the 24-hour period.
    #-------------------------------------------------------------------
    if bgc_data_type == "ISPOL":
        nit_file = bgc_data_dir + "nutrients_daily_ISPOL_WOA_field3.nc"
        sil_file = bgc_data_dir + "nutrients_daily_ISPOL_WOA_field3.nc"
        if int(ice_calendar.istep) == 1:
            if tr_bgc_Sil:
                nu.write(" \n")
                nu.write(" silicate data interpolated to timestep:\n")
                nu.write(f" {sil_file}\n")
            if tr_bgc_Nit:
                nu.write(" \n")
                nu.write(" nitrate data interpolated to timestep:\n")
                nu.write(f" {nit_file}\n")
                if restore_bgc:
                    nu.write(
                        f" bgc restoring timescale (days) = {ice_forcing.trestore}\n"
                    )
        dataloc = 2                          # data located at end of interval
        sec1hr = secday                      # seconds in day
        maxrec = 365
        recnum = int(ice_calendar.yday)
        ixm = (recnum + maxrec - 2) % maxrec + 1
        ixx = (recnum - 1) % maxrec + 1
        recslot = 2
        ixp = -99
        ice_forcing.interp_coeff(recnum, recslot, sec1hr, dataloc)
        read1 = False
        if int(ice_calendar.istep) == 1 or int(bgcrecnum) != recnum:
            read1 = True
        if tr_bgc_Sil:
            ice_forcing.read_data_nc_point(
                read1, 0, ice_forcing.fyear, ixm, ixx, ixp, maxrec,
                sil_file, "silicate", sil_data_p, field_loc_center, field_type_scalar,
            )
            sil[...] = (
                ice_forcing.c1intp * sil_data_p[0] + ice_forcing.c2intp * sil_data_p[1]
            )
        if tr_bgc_Nit:
            ice_forcing.read_data_nc_point(
                read1, 0, ice_forcing.fyear, ixm, ixx, ixp, maxrec,
                nit_file, "nitrate", nit_data_p, field_loc_center, field_type_scalar,
            )
            nit[...] = (
                ice_forcing.c1intp * nit_data_p[0] + ice_forcing.c2intp * nit_data_p[1]
            )
        ks_sil = 2 * max_algae + max_doc + 3 + max_dic
        ks_nit = max_algae + 1
        ks_pon = 2 * max_algae + max_doc + 7 + max_dic
        ocean_bio_all[islice, jslice, _ocean_slot(ks_sil)] = sil[islice, jslice]
        ocean_bio_all[islice, jslice, _ocean_slot(ks_nit)] = nit[islice, jslice]
        ocean_bio_all[islice, jslice, _ocean_slot(ks_pon)] = nit[islice, jslice]
        bgcrecnum = int_kind(recnum)


def get_atm_bgc():
    """author: Nicole Jeffery, LANL"""
    #-----------------------------------------------------------------
    # initialize
    #-----------------------------------------------------------------
    tr_zaero = bool(icepack_tracers.tr_zaero)
    nlt_zaero = icepack_tracers.nlt_zaero
    ice_flux_bgc.flux_bio_atm[...] = c0
    if tr_zaero:
        ilo, ihi, jlo, jhi = ice_domain_size.physical_bounds()
        islice = slice(ilo, ihi + 1)
        jslice = slice(jlo, jhi + 1)
        for nn in range(int(icepack_tracers.n_zaero)):
            slot = int(nlt_zaero[nn]) - 1
            ice_flux_bgc.flux_bio_atm[islice, jslice, slot] = (
                ice_flux_bgc.faero_atm[islice, jslice, nn]
            )


def fiso_default():
    """constant values for atmospheric water isotopes

    authors: David Bailey, NCAR
    """
    ice_flux_bgc.fiso_atm[...] = dbl_kind(1.0e-14)  # kg/m^2 s


def faero_default():
    """constant values for atmospheric aerosols

    authors: Elizabeth Hunke, LANL
    """
    faero = ice_flux_bgc.faero_atm
    faero[:, :, 0] = dbl_kind(1.0e-12)  # kg/m^2 s
    faero[:, :, 1] = dbl_kind(1.0e-13)
    faero[:, :, 2] = dbl_kind(1.0e-14)
    faero[:, :, 3] = dbl_kind(1.0e-14)
    faero[:, :, 4] = dbl_kind(1.0e-14)
    faero[:, :, 5] = dbl_kind(1.0e-14)


def init_bgc_data(fed1, fep1):
    """Initialize ocean iron from file

    authors: Nicole Jeffery, LANL

    ``fed1`` and ``fep1`` are the first dissolved and particulate pools.
    """
    subname = "(init_bgc_data)"
    #-------------------------------------------------------------------
    # Annual average data from Tagliabue, 2012 (top 50 m average
    # poisson grid filled on gx1v6
    #-------------------------------------------------------------------
    if str(ice_init.fe_data_type).strip() != "clim":
        return
    diag = True
    bgc_data_dir = str(ice_init.bgc_data_dir).strip()
    iron_file = bgc_data_dir + "dFe_50m_annual_Tagliabue_gx1.nc"
    nu = ice_fileunits.nu_diag
    nu.write(" \n")
    nu.write(" Dissolved iron ocean concentrations from:\n")
    nu.write(f" {iron_file}\n")
    _read_iron(iron_file, "dFe", fed1, diag)
    fed1[fed1 > dbl_kind(1.0e20)] = p1

    iron_file = bgc_data_dir + "pFe_bathy_gx1.nc"
    nu.write(" \n")
    nu.write(" Particulate iron ocean concentrations from:\n")
    nu.write(f" {iron_file}\n")
    _read_iron(iron_file, "pFe", fep1, diag)
    fep1[fep1 > dbl_kind(1.0e20)] = p1
    _ = subname


def _read_iron(path, fieldname, array, diag):
    """Record 1 of a 2D iron field. First dimension is i; record is last."""
    dataset = netCDF4.Dataset(path)
    try:
        if fieldname not in dataset.variables:
            raise KeyError(f"(init_bgc_data) {path} has no variable {fieldname}")
        data = np.array(dataset.variables[fieldname][:], dtype=dbl_kind)
    finally:
        dataset.close()
    if data.ndim == 3:
        data = data[:, :, 0]
    nx = int(ice_domain_size.nx_global)
    ny = int(ice_domain_size.ny_global)
    if tuple(data.shape) != (nx, ny):
        raise ValueError(
            f"(init_bgc_data) {fieldname} shape {data.shape}; expected ({nx}, {ny})"
        )
    if diag:
        ice_fileunits.nu_diag.write(
            f" (init_bgc_data) {path} {fieldname} {np.min(data)} {np.max(data)}\n"
        )
    scatter_global(array, data, field_loc_center, field_type_scalar)
