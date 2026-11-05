"""Read a CICE restart dump into the ice state.

author for JAX version: Xinyuan Zhu
XZ: converted from F90 to Python

authors Elizabeth C. Hunke, LANL
        William H. Lipscomb, LANL

``restartfile`` reads and ``dumpfile`` writes the NetCDF restart.
The v4 binary reader ``restartfile_v4`` is commented out in
``init_restart`` and is not ported. ``direct_adjust_aice`` runs when
``restart_mod`` requests a concentration adjustment and then calls the
initialization path through ``cleanup_itd``.
One grid, no block index.
"""

import numpy as np

from cicecore.cicedyn.dynamics import ice_dyn_shared
from cicecore.cicedyn.general import ice_flux
from cicecore.cicedyn.general import ice_flux_bgc
from cicecore.cicedyn.general import ice_init
from cicecore.cicedyn.general import ice_state
from cicecore.cicedyn.infrastructure import ice_grid
from cicecore.cicedyn.infrastructure import ice_read_write
from cicecore.cicedyn.infrastructure.comm.serial import ice_boundary
from cicecore.cicedyn.infrastructure.comm.serial.ice_boundary import ice_HaloUpdate_stress
from cicecore.cicedyn.infrastructure.io.io_netcdf import ice_restart
from cicecore.shared import ice_calendar
from cicecore.shared import ice_arrays_column
from cicecore.shared import ice_domain_size
from cicecore.shared import ice_fileunits
from cicecore.shared.ice_constants import (
    c0,
    field_loc_Eface,
    field_loc_NEcorner,
    field_loc_Nface,
    field_loc_center,
    field_type_scalar,
    field_type_vector,
    p5,
)
from icepack.columnphysics import icepack_itd
from icepack.columnphysics import icepack_parameters
from icepack.columnphysics import icepack_therm_shared
from icepack.columnphysics import icepack_tracers


def dumpfile(filename_spec=None):
    """Stage core fields for the netCDF restart, in Fortran write order."""
    t = icepack_tracers
    trcrn = ice_state.trcrn
    ice_restart.init_restart_write(filename_spec)

    # The dump resets tracer values on land outside the optional ocean mask.
    land = (~ice_grid.tmask) & (~ice_grid.opmask)
    ice_state.trcrn = np.where(land[:, :, None, None], c0, trcrn)
    trcrn = ice_state.trcrn
    for name in ("aicen", "vicen", "vsnon"):
        ice_restart.write_restart_field(getattr(ice_state, name), name)
    ice_restart.write_restart_field(trcrn[:, :, int(t.nt_Tsfc), :], "Tsfcn")
    for stem, index, count in (
        ("sice", t.nt_sice, ice_domain_size.nilyr),
        ("qice", t.nt_qice, ice_domain_size.nilyr),
        ("qsno", t.nt_qsno, ice_domain_size.nslyr),
    ):
        for k in range(int(count)):
            ice_restart.write_restart_field(
                trcrn[:, :, int(index) + k, :], f"{stem}{k + 1:03d}"
            )

    for name in ("uvel", "vvel"):
        ice_restart.write_restart_field(getattr(ice_state, name), name)
    grid_ice = str(ice_init.grid_ice).strip()
    if grid_ice == "CD":
        for name in ("uvelE", "vvelE", "uvelN", "vvelN"):
            ice_restart.write_restart_field(getattr(ice_state, name), name)
    elif grid_ice == "C":
        for name in ("uvelE", "vvelN"):
            ice_restart.write_restart_field(getattr(ice_state, name), name)

    if bool(ice_init.restart_coszen):
        ice_restart.write_restart_field(ice_flux.coszen, "coszen")
    for name in ("scale_factor", "swvdr", "swvdf", "swidr", "swidf"):
        ice_restart.write_restart_field(getattr(ice_flux, name), name)
    ice_restart.write_restart_field(ice_flux.strocnxT_iavg, "strocnxT")
    ice_restart.write_restart_field(ice_flux.strocnyT_iavg, "strocnyT")
    for stem in ("stressp", "stressm", "stress12"):
        for k in (1, 3, 2, 4):
            name = f"{stem}_{k}"
            ice_restart.write_restart_field(getattr(ice_flux, name), name)
    if grid_ice == "CD":
        stress_names = ("stresspT", "stressmT", "stress12T",
                        "stresspU", "stressmU", "stress12U")
    elif grid_ice == "C":
        stress_names = ("stresspT", "stressmT", "stress12U")
    else:
        stress_names = ()
    for name in stress_names:
        ice_restart.write_restart_field(getattr(ice_flux, name), name)

    mask = ice_dyn_shared.iceUmask if int(ice_init.kdyn) > 0 else np.zeros_like(ice_grid.tmask)
    ice_restart.write_restart_field(np.asarray(mask, dtype=np.float64), "iceumask")
    if grid_ice in ("C", "CD"):
        for field, name in (("iceNmask", "icenmask"), ("iceEmask", "iceemask")):
            mask = (getattr(ice_dyn_shared, field) if int(ice_init.kdyn) > 0
                    else np.zeros_like(ice_grid.tmask))
            ice_restart.write_restart_field(np.asarray(mask, dtype=np.float64), name)
    if bool(ice_init.oceanmixed_ice):
        for name in ("sst", "frzmlt"):
            ice_restart.write_restart_field(getattr(ice_flux, name), name)


def restartfile(ice_ic=None):
    """Read the core restart file.

    ``ice_ic`` is the path. Omit it to read the path stored in ``pointer_file``.
    """
    subname = "(restartfile)"
    ntrcr = int(icepack_tracers.ntrcr)
    nt_Tsfc = int(icepack_tracers.nt_Tsfc)
    nt_sice = int(icepack_tracers.nt_sice)
    nt_qice = int(icepack_tracers.nt_qice)
    nt_qsno = int(icepack_tracers.nt_qsno)
    if nt_Tsfc < 0:
        raise RuntimeError(subname + " nt_Tsfc is not set")

    tr_aero = bool(icepack_tracers.tr_aero)
    tr_pond_topo = bool(icepack_tracers.tr_pond_topo)
    _ = (ntrcr, tr_aero, tr_pond_topo)

    ice_restart.init_restart_read(ice_ic)
    ice_calendar.calendar()

    diag = True
    nu = ice_fileunits.nu_diag
    nilyr = int(ice_domain_size.nilyr)
    nslyr = int(ice_domain_size.nslyr)
    ncat = int(ice_domain_size.ncat)
    center = field_loc_center
    scalar = field_type_scalar

    #-----------------------------------------------------------------
    # state variables
    # Tsfc is the only tracer read in this file.  All other
    # tracers are in their own dump/restart files.
    #-----------------------------------------------------------------
    nu.write(" min/max area, vol ice, vol snow, Tsfc\n")
    _cat(ice_state.aicen, "aicen", ncat, diag, center, scalar)
    _cat(ice_state.vicen, "vicen", ncat, diag, center, scalar)
    _cat(ice_state.vsnon, "vsnon", ncat, diag, center, scalar)
    _cat(ice_state.trcrn[:, :, nt_Tsfc, :], "Tsfcn", ncat, diag, center, scalar)

    nu.write("min/max sice for each layer\n")
    for k in range(1, nilyr + 1):
        nchar = f"{k:03d}"
        _cat(
            ice_state.trcrn[:, :, nt_sice + k - 1, :],
            "sice" + nchar,
            ncat,
            diag,
            center,
            scalar,
        )

    nu.write("min/max qice for each layer\n")
    for k in range(1, nilyr + 1):
        nchar = f"{k:03d}"
        _cat(
            ice_state.trcrn[:, :, nt_qice + k - 1, :],
            "qice" + nchar,
            ncat,
            diag,
            center,
            scalar,
        )

    nu.write("min/max qsno for each layer\n")
    for k in range(1, nslyr + 1):
        nchar = f"{k:03d}"
        _cat(
            ice_state.trcrn[:, :, nt_qsno + k - 1, :],
            "qsno" + nchar,
            ncat,
            diag,
            center,
            scalar,
        )

    #-----------------------------------------------------------------
    # velocity
    #-----------------------------------------------------------------
    nu.write("min/max velocity components\n")
    _horiz(ice_state.uvel, "uvel", diag, field_loc_NEcorner, field_type_vector)
    _horiz(ice_state.vvel, "vvel", diag, field_loc_NEcorner, field_type_vector)

    grid_ice = str(ice_init.grid_ice).strip()
    if grid_ice == "CD":
        _optional(ice_state.uvelE, "uvelE", diag, field_loc_Eface, field_type_vector)
        _optional(ice_state.vvelE, "vvelE", diag, field_loc_Eface, field_type_vector)
        _optional(ice_state.uvelN, "uvelN", diag, field_loc_Nface, field_type_vector)
        _optional(ice_state.vvelN, "vvelN", diag, field_loc_Nface, field_type_vector)
    if grid_ice == "C":
        _optional(ice_state.uvelE, "uvelE", diag, field_loc_Eface, field_type_vector)
        _optional(ice_state.vvelN, "vvelN", diag, field_loc_Nface, field_type_vector)

    #-----------------------------------------------------------------
    # radiation fields
    #-----------------------------------------------------------------
    nu.write("radiation fields\n")
    if bool(ice_init.restart_coszen):
        _horiz(ice_flux.coszen, "coszen", diag, None, None)
    _horiz(ice_flux.scale_factor, "scale_factor", diag, center, scalar)
    _horiz(ice_flux.swvdr, "swvdr", diag, center, scalar)
    _horiz(ice_flux.swvdf, "swvdf", diag, center, scalar)
    _horiz(ice_flux.swidr, "swidr", diag, center, scalar)
    _horiz(ice_flux.swidf, "swidf", diag, center, scalar)

    #-----------------------------------------------------------------
    # ocean stress
    #-----------------------------------------------------------------
    nu.write("min/max ocean stress components\n")
    _horiz(ice_flux.strocnxT_iavg, "strocnxT", diag, center, field_type_vector)
    _horiz(ice_flux.strocnyT_iavg, "strocnyT", diag, center, field_type_vector)

    #-----------------------------------------------------------------
    # internal stress
    # The stress tensor must be read and scattered in pairs in order
    # to properly match corner values across a tripole grid cut.
    #-----------------------------------------------------------------
    nu.write("internal stress components\n")
    _horiz(ice_flux.stressp_1, "stressp_1", diag, center, scalar)
    _horiz(ice_flux.stressp_3, "stressp_3", diag, center, scalar)
    _horiz(ice_flux.stressp_2, "stressp_2", diag, center, scalar)
    _horiz(ice_flux.stressp_4, "stressp_4", diag, center, scalar)
    _horiz(ice_flux.stressm_1, "stressm_1", diag, center, scalar)
    _horiz(ice_flux.stressm_3, "stressm_3", diag, center, scalar)
    _horiz(ice_flux.stressm_2, "stressm_2", diag, center, scalar)
    _horiz(ice_flux.stressm_4, "stressm_4", diag, center, scalar)
    _horiz(ice_flux.stress12_1, "stress12_1", diag, center, scalar)
    _horiz(ice_flux.stress12_3, "stress12_3", diag, center, scalar)
    _horiz(ice_flux.stress12_2, "stress12_2", diag, center, scalar)
    _horiz(ice_flux.stress12_4, "stress12_4", diag, center, scalar)

    if grid_ice in ("CD", "C"):
        _optional(ice_flux.stresspT, "stresspT", diag, center, scalar)
        _optional(ice_flux.stressmT, "stressmT", diag, center, scalar)
        _optional(ice_flux.stress12T, "stress12T", diag, center, scalar)
        _optional(ice_flux.stresspU, "stresspU", diag, field_loc_NEcorner, scalar)
        _optional(ice_flux.stressmU, "stressmU", diag, field_loc_NEcorner, scalar)
        _optional(ice_flux.stress12U, "stress12U", diag, field_loc_NEcorner, scalar)

    ns = str(ice_boundary.ns_boundary_type).strip()
    if ns in ("tripole", "tripoleT"):
        _stress_pair(ice_flux.stressp_1, ice_flux.stressp_3)
        _stress_pair(ice_flux.stressp_3, ice_flux.stressp_1)
        _stress_pair(ice_flux.stressp_2, ice_flux.stressp_4)
        _stress_pair(ice_flux.stressp_4, ice_flux.stressp_2)
        _stress_pair(ice_flux.stressm_1, ice_flux.stressm_3)
        _stress_pair(ice_flux.stressm_3, ice_flux.stressm_1)
        _stress_pair(ice_flux.stressm_2, ice_flux.stressm_4)
        _stress_pair(ice_flux.stressm_4, ice_flux.stressm_2)
        _stress_pair(ice_flux.stress12_1, ice_flux.stress12_3)
        _stress_pair(ice_flux.stress12_3, ice_flux.stress12_1)
        _stress_pair(ice_flux.stress12_2, ice_flux.stress12_4)
        _stress_pair(ice_flux.stress12_4, ice_flux.stress12_2)
        # TODO: CD-grid

    #-----------------------------------------------------------------
    # ice mask for dynamics
    #-----------------------------------------------------------------
    kdyn = int(ice_init.kdyn)
    if kdyn > 0:
        nu.write("ice mask for dynamics\n")
        _read_mask("iceumask", ice_dyn_shared.iceUmask, diag)
        if grid_ice in ("CD", "C"):
            _read_mask("icenmask", ice_dyn_shared.iceNmask, diag)
            _read_mask("iceemask", ice_dyn_shared.iceEmask, diag)
    else:
        nu.write(
            "ice mask for dynamics - not used, however mandatory to read in binary files\n"
        )
        if ice_restart.query_field("iceumask"):
            _discard("iceumask", diag)
        if grid_ice in ("CD", "C"):
            if ice_restart.query_field("icenmask"):
                _discard("icenmask", diag)
            if ice_restart.query_field("iceemask"):
                _discard("iceemask", diag)

    # set Tsfcn to c0 on land
    land = (~ice_grid.tmask) & (~ice_grid.opmask)
    ice_state.trcrn[:, :, nt_Tsfc, :] = np.where(
        land[:, :, None], c0, ice_state.trcrn[:, :, nt_Tsfc, :]
    )

    # for mixed layer model
    if bool(ice_init.oceanmixed_ice):
        nu.write("min/max sst, frzmlt\n")
        _horiz(ice_flux.sst, "sst", diag, center, scalar)
        _horiz(ice_flux.frzmlt, "frzmlt", diag, center, scalar)

    #-----------------------------------------------------------------
    # Ensure unused stress values in west and south ghost cells are 0
    #-----------------------------------------------------------------
    ghost = int(ice_domain_size.nghost)
    for name in (
        "stressp_1", "stressp_2", "stressp_3", "stressp_4",
        "stressm_1", "stressm_2", "stressm_3", "stressm_4",
        "stress12_1", "stress12_2", "stress12_3", "stress12_4",
    ):
        field = getattr(ice_flux, name)
        field[:, :ghost] = c0
        field[:ghost, :] = c0
    # TODO: CD-grid ?

    #-----------------------------------------------------------------
    # Ensure ice is binned in correct categories
    # (should not be necessary unless restarting from a run with
    #  different category boundaries).
    #
    # If called, this subroutine does not give exact restart.
    #-----------------------------------------------------------------
    # call cleanup_itd

    #-----------------------------------------------------------------
    # compute aggregate ice state and open water area
    #-----------------------------------------------------------------
    _aggregate()

    # if runid is bering then need to correct npt for istep0
    # init_restart_read already subtracted istep0 once. The Fortran
    # subtracts it again here.
    if str(ice_init.runid).strip() == "bering":
        ice_calendar.npt = int(ice_calendar.npt) - int(ice_calendar.istep0)

    #-----------------------------------------------------------------
    # update concentration from a file
    #-----------------------------------------------------------------
    restart_mod = str(ice_init.restart_mod).strip()
    if restart_mod != "none":
        if restart_mod in ("adjust_aice", "adjust_aice_test"):
            direct_adjust_aice(test=restart_mod == "adjust_aice_test")
        else:
            raise RuntimeError(subname + "ERROR: unsupported restart_mod=" + restart_mod)

        # Ensure ice is binned in the correct thickness categories.
        depend = ice_state.trcr_depend
        base = ice_state.trcr_base
        strata_n = ice_state.n_trcr_strata
        strata = ice_state.nt_strata
        nx, ny = ice_state.aice.shape
        for j in range(ny):
            for i in range(nx):
                if not ice_grid.tmask[i, j]:
                    continue
                (
                    _, _, dfpond, dfresh, dfsalt, dfhocn,
                    dfaero_ocn, dfiso_ocn, dflux_bio,
                ) = icepack_itd.cleanup_itd(
                    ice_calendar.dt, ice_arrays_column.hin_max,
                    ice_state.aicen[i, j, :], ice_state.trcrn[i, j, :, :],
                    ice_state.vicen[i, j, :], ice_state.vsnon[i, j, :],
                    bool(icepack_tracers.tr_aero), bool(icepack_tracers.tr_pond_topo),
                    ice_arrays_column.first_ice[i, j, :],
                    depend, base, strata_n, strata, ice_flux.Tf[i, j],
                    limit_aice=True,
                )
                ice_flux.fpond[i, j] += dfpond
                ice_flux.fresh[i, j] += dfresh
                ice_flux.fsalt[i, j] += dfsalt
                ice_flux.fhocn[i, j] += dfhocn
                ice_flux_bgc.faero_ocn[i, j, :len(dfaero_ocn)] += dfaero_ocn
                ice_flux_bgc.fiso_ocn[i, j, :len(dfiso_ocn)] += dfiso_ocn
                ice_flux_bgc.flux_bio[i, j, :len(dflux_bio)] += dflux_bio
                (
                    ice_state.aice[i, j], ice_state.trcr[i, j, :],
                    ice_state.vice[i, j], ice_state.vsno[i, j],
                    ice_state.aice0[i, j],
                ) = icepack_itd.icepack_aggregate(
                    aicen=ice_state.aicen[i, j, :],
                    trcrn=ice_state.trcrn[i, j, :, :],
                    vicen=ice_state.vicen[i, j, :],
                    vsnon=ice_state.vsnon[i, j, :],
                    trcr_depend=depend, trcr_base=base,
                    n_trcr_strata=strata_n, nt_strata=strata,
                    Tf=ice_flux.Tf[i, j],
                )
                ice_state.aice_init[i, j] = ice_state.aice[i, j]


def direct_adjust_aice(test=False):
    """Adjust restart ice concentration to a target field.

    Alan J. Wallcraft, COAPS/FSU, Nov 2024
    """
    subname = "(direct_adjust_aice)"
    ncat = int(ice_domain_size.ncat)
    nilyr = int(ice_domain_size.nilyr)
    nslyr = int(ice_domain_size.nslyr)
    nt_Tsfc = int(icepack_tracers.nt_Tsfc)
    nt_qice = int(icepack_tracers.nt_qice)
    nt_sice = int(icepack_tracers.nt_sice)
    nt_qsno = int(icepack_tracers.nt_qsno)
    puny = icepack_parameters.puny

    if test:
        ice_fileunits.nu_diag.write(subname + " direct_adjust_aice rounding to nearest 1/20th\n")
        # Fortran NINT rounds positive half cases away from zero.
        work1 = np.floor(ice_state.aice * 20 + 0.5) / 20
    else:
        import os

        aice_filename = os.path.join(str(ice_init.restart_dir).strip(), "sic.nc")
        ice_fileunits.nu_diag.write(subname + " direct_adjust_aice from " + aice_filename + "\n")
        work1 = np.zeros_like(ice_state.aice)
        ice_read_write.ice_read_nc(
            aice_filename, "sic", work1, True, field_loc_center, field_type_scalar
        )

    edge_om = 0.2    # nominal ice edge zone
    diff_om = 0.1    # allowed model vs obs difference
    hin_om = ice_arrays_column.hin_max[1] * 0.9    # new ice thickness
    nx, ny = ice_state.aice.shape
    for j in range(ny):
        for i in range(nx):
            aice_o = work1[i, j]            # observed ice concentration
            aice_m = ice_state.aice[i, j]   # model ice concentration
            if not ice_grid.tmask[i, j]:
                continue                    # land
            if aice_o > 0.01 and abs(aice_o - aice_m) <= 0.01:
                continue                    # model and observation are very close
            if min(aice_o, aice_m) >= edge_om and abs(aice_o - aice_m) <= diff_om:
                continue                    # model and observation are close enough
            if aice_o == aice_m:
                continue

            reset_cats = []
            if aice_o < aice_m:
                if aice_o < 0.01:
                    # Remove all ice; warm SST so ice will not grow immediately.
                    ice_flux.sst[i, j] += 0.2
                    ice_state.aicen[i, j, :] = c0
                    ice_state.vicen[i, j, :] = c0
                    ice_state.vsnon[i, j, :] = c0
                    reset_cats = range(ncat)
                else:
                    if aice_o < edge_om:
                        aice_t = aice_o
                    else:
                        aice_t = aice_o + diff_om
                    # Exhaust categories in order until the target area is reached.
                    aice_i = aice_m - aice_t
                    for n in range(ncat):
                        if aice_i <= 0.001:
                            break
                        if aice_i >= ice_state.aicen[i, j, n]:
                            aice_i -= ice_state.aicen[i, j, n]
                            ice_state.aicen[i, j, n] = c0
                            ice_state.vicen[i, j, n] = c0
                            ice_state.vsnon[i, j, n] = c0
                            reset_cats.append(n)
                        else:
                            # Reduce volume and snow with area, preserving thickness.
                            q = (ice_state.aicen[i, j, n] - aice_i) / ice_state.aicen[i, j, n]
                            aice_i = c0
                            ice_state.aicen[i, j, n] *= q
                            ice_state.vicen[i, j, n] *= q
                            ice_state.vsnon[i, j, n] *= q
            elif aice_o > 0.01:
                if aice_m < edge_om:
                    aice_t = aice_o
                else:
                    aice_t = aice_o - diff_om
                q = aice_t - aice_m
                # Add new ice to the first category and cool SST to inhibit melt.
                ice_flux.sst[i, j] -= q
                aicen_old = ice_state.aicen[i, j, 0]
                vsnon_old = ice_state.vsnon[i, j, 0]
                ice_state.aicen[i, j, 0] += q
                ice_state.vicen[i, j, 0] += q * hin_om
                ice_state.vsnon[i, j, 0] += q * hin_om * 0.2
                if vsnon_old < puny:
                    # Existing ice may have had zero snow: define snow enthalpy.
                    Tsfc = min(icepack_parameters.Tsmelt,
                               ice_flux.Tair[i, j] - icepack_parameters.Tffresh)
                    ice_state.trcrn[i, j, nt_Tsfc, 0] = Tsfc
                    Ti = min(c0, Tsfc)
                    ice_state.trcrn[i, j, nt_qsno:nt_qsno + nslyr, 0] = (
                        -icepack_parameters.rhos
                        * (icepack_parameters.Lfresh - icepack_parameters.cp_ice * Ti)
                    )
                if aicen_old < puny:
                    # Newly created category ice needs enthalpy and salinity.
                    reset_cats = (0,)

            for n in reset_cats:
                Tsfc, qin, qsn = icepack_therm_shared.icepack_init_trcr(
                    ice_flux.Tair[i, j], ice_flux.Tf[i, j],
                    ice_flux.salinz[i, j, :], ice_flux.Tmltz[i, j, :],
                )
                ice_state.trcrn[i, j, nt_Tsfc, n] = Tsfc
                ice_state.trcrn[i, j, nt_qice:nt_qice + nilyr, n] = qin
                ice_state.trcrn[i, j, nt_sice:nt_sice + nilyr, n] = (
                    ice_flux.salinz[i, j, :nilyr]
                )
                ice_state.trcrn[i, j, nt_qsno:nt_qsno + nslyr, n] = qsn


def _cat(work, vname, ncat, diag, field_loc, field_type):
    ice_restart.read_restart_field(
        work, vname, ncat, diag, field_loc=field_loc, field_type=field_type
    )


def _horiz(work, vname, diag, field_loc, field_type):
    ice_restart.read_restart_field(
        work, vname, 1, diag, field_loc=field_loc, field_type=field_type
    )


def _optional(work, vname, diag, field_loc, field_type):
    if ice_restart.query_field(vname):
        _horiz(work, vname, diag, field_loc, field_type)


def _stress_pair(array1, array2):
    ice_HaloUpdate_stress(array1, array2, field_loc_center, field_type_scalar)


def _read_mask(vname, mask, diag):
    if not ice_restart.query_field(vname) or mask is None:
        return
    work = np.zeros(mask.shape, dtype=ice_flux.sst.dtype)
    _horiz(work, vname, diag, field_loc_center, field_type_scalar)
    mask[...] = False
    mask[work > p5] = True


def _discard(vname, diag):
    work = np.zeros(ice_domain_size.horizontal_shape(), dtype=ice_flux.sst.dtype)
    _horiz(work, vname, diag, field_loc_center, field_type_scalar)


def _aggregate():
    tmask = ice_grid.tmask
    opmask = ice_grid.opmask
    depend = ice_state.trcr_depend
    base = ice_state.trcr_base
    strata_n = ice_state.n_trcr_strata
    strata = ice_state.nt_strata
    ny = ice_state.aice.shape[1]
    nx = ice_state.aice.shape[0]
    for j in range(ny):
        for i in range(nx):
            if tmask[i, j] or opmask[i, j]:
                (
                    ice_state.aice[i, j],
                    ice_state.trcr[i, j, :],
                    ice_state.vice[i, j],
                    ice_state.vsno[i, j],
                    ice_state.aice0[i, j],
                ) = icepack_itd.icepack_aggregate(
                    aicen=ice_state.aicen[i, j, :],
                    trcrn=ice_state.trcrn[i, j, :, :],
                    vicen=ice_state.vicen[i, j, :],
                    vsnon=ice_state.vsnon[i, j, :],
                    trcr_depend=depend,
                    trcr_base=base,
                    n_trcr_strata=strata_n,
                    nt_strata=strata,
                    Tf=ice_flux.Tf[i, j],
                )
            ice_state.aice_init[i, j] = ice_state.aice[i, j]
