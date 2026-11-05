"""Restart routines for the column package.

author for JAX version: Xinyuan Zhu
XZ: converted from F90 to Python

author: Elizabeth C. Hunke, LANL

2014: Moved subroutines from column package modules

Read paths are used by ``init_restart``. Write paths are used by the time
loop's restart dump. The non-BGC and BGC tracers share the same netCDF
dataset. The Fortran unit number is unused for netCDF.
"""

import numpy as np

from cicecore.cicedyn.general import ice_flux
from cicecore.cicedyn.general import ice_flux_bgc
from cicecore.cicedyn.general import ice_init
from cicecore.cicedyn.general import ice_state
from cicecore.cicedyn.infrastructure import ice_grid
from cicecore.cicedyn.infrastructure.io.io_netcdf import ice_restart
from cicecore.shared import ice_arrays_column
from cicecore.shared import ice_domain_size
from cicecore.shared import ice_fileunits
from cicecore.shared.ice_constants import field_loc_center, field_type_scalar, p5
from icepack.columnphysics import icepack_parameters
from icepack.columnphysics import icepack_tracers

# if .true., read age tracer restart file
restart_age = False
# if .true., read FY tracer restart file
restart_FY = False
# if .true., read lvl tracer restart file
restart_lvl = False
# if .true., read meltponds restart file
restart_pond_lvl = False
# if .true., read meltponds restart file
restart_pond_sealvl = False
# if .true., read meltponds restart file
restart_pond_topo = False
# if .true., read snow tracer restart file
restart_snow = False
# if .true., read floe size restart file
restart_fsd = False
# if .true., read isotope tracer restart file
restart_iso = False
# if .true., read aerosol tracer restart file
restart_aero = False
# if .true., read hbrine from restart file
restart_hbrine = False
# if .true., read bgc restart file
restart_bgc = False


def _sync_flags():
    """Namelist copies live on ice_init. continue-runs overwrite them in init_restart."""
    global restart_age, restart_FY, restart_lvl, restart_pond_lvl, restart_pond_sealvl
    global restart_pond_topo, restart_snow, restart_fsd, restart_iso, restart_aero
    restart_age = bool(ice_init.restart_age)
    restart_FY = bool(ice_init.restart_FY)
    restart_lvl = bool(ice_init.restart_lvl)
    restart_pond_lvl = bool(ice_init.restart_pond_lvl)
    restart_pond_sealvl = bool(ice_init.restart_pond_sealvl)
    restart_pond_topo = bool(ice_init.restart_pond_topo)
    restart_snow = bool(ice_init.restart_snow)
    restart_fsd = bool(ice_init.restart_fsd)
    restart_iso = bool(ice_init.restart_iso)
    restart_aero = bool(ice_init.restart_aero)


def write_restart_age():
    ice_restart.write_restart_field(
        ice_state.trcrn[:, :, int(icepack_tracers.nt_iage), :], "iage"
    )


def write_restart_FY():
    ice_restart.write_restart_field(
        ice_state.trcrn[:, :, int(icepack_tracers.nt_FY), :], "FY"
    )
    ice_restart.write_restart_field(ice_flux.frz_onset, "frz_onset")


def write_restart_lvl():
    for name, index in (("alvl", icepack_tracers.nt_alvl),
                        ("vlvl", icepack_tracers.nt_vlvl)):
        ice_restart.write_restart_field(ice_state.trcrn[:, :, int(index), :], name)


def write_restart_pond_lvl():
    for name, index in (("apnd", icepack_tracers.nt_apnd),
                        ("hpnd", icepack_tracers.nt_hpnd),
                        ("ipnd", icepack_tracers.nt_ipnd)):
        ice_restart.write_restart_field(ice_state.trcrn[:, :, int(index), :], name)
    ice_restart.write_restart_field(ice_flux.fsnow, "fsnow")
    ice_restart.write_restart_field(ice_arrays_column.dhsn, "dhs")
    ice_restart.write_restart_field(ice_arrays_column.ffracn, "ffrac")


def write_restart_pond_topo():
    for name, index in (("apnd", icepack_tracers.nt_apnd),
                        ("hpnd", icepack_tracers.nt_hpnd),
                        ("ipnd", icepack_tracers.nt_ipnd)):
        ice_restart.write_restart_field(ice_state.trcrn[:, :, int(index), :], name)


def write_restart_pond_sealvl():
    for name, index in (("apnd", icepack_tracers.nt_apnd),
                        ("hpnd", icepack_tracers.nt_hpnd),
                        ("ipnd", icepack_tracers.nt_ipnd)):
        ice_restart.write_restart_field(ice_state.trcrn[:, :, int(index), :], name)
    ice_restart.write_restart_field(ice_flux.fsnow, "fsnow")
    ice_restart.write_restart_field(ice_arrays_column.dhsn, "dhs")
    ice_restart.write_restart_field(ice_arrays_column.ffracn, "ffrac")


def write_restart_snow():
    for k in range(int(ice_domain_size.nslyr)):
        for stem, index in (("smice", icepack_tracers.nt_smice),
                            ("smliq", icepack_tracers.nt_smliq),
                            ("rhos", icepack_tracers.nt_rhos),
                            ("rsnw", icepack_tracers.nt_rsnw)):
            ice_restart.write_restart_field(
                ice_state.trcrn[:, :, int(index) + k, :], f"{stem}{k + 1:03d}"
            )


def write_restart_fsd():
    for k in range(int(ice_domain_size.nfsd)):
        ice_restart.write_restart_field(
            ice_state.trcrn[:, :, int(icepack_tracers.nt_fsd) + k, :],
            f"fsd{k + 1:03d}",
        )


def write_restart_iso():
    for stem, index in (("isosno", icepack_tracers.nt_isosno),
                        ("isoice", icepack_tracers.nt_isoice)):
        for k in range(int(icepack_tracers.n_iso)):
            ice_restart.write_restart_field(
                ice_state.trcrn[:, :, int(index) + k, :], f"{stem}{k + 1:03d}"
            )


def write_restart_aero():
    for k in range(int(icepack_tracers.n_aero)):
        for offset, stem in enumerate(("aerosnossl", "aerosnoint", "aeroicessl", "aeroiceint")):
            index = int(icepack_tracers.nt_aero) + 4 * k + offset
            ice_restart.write_restart_field(
                ice_state.trcrn[:, :, index, :], f"{stem}{k + 1:03d}"
            )


def write_restart_hbrine():
    ice_restart.write_restart_field(
        ice_state.trcrn[:, :, int(icepack_tracers.nt_fbri), :], "fbrn"
    )
    ice_restart.write_restart_field(
        np.where(ice_grid.tmask[:, :, None], ice_arrays_column.first_ice, False).astype(float),
        "first_ice",
    )


def write_restart_bgc():
    """Write skeletal or vertical ice BGC tracers and ocean BGC fields."""
    t = icepack_tracers
    trcrn = ice_state.trcrn
    # Skeletal layer BGC; the algal-carbon loop is commented out in Fortran.
    if bool(icepack_parameters.skl_bgc):
        for mm in range(int(t.n_algae)):
            ice_restart.write_restart_field(
                trcrn[:, :, int(t.nt_bgc_N[mm]), :], f"bgc_N{mm + 1:03d}"
            )
            if bool(t.tr_bgc_chl):
                ice_restart.write_restart_field(
                    trcrn[:, :, int(t.nt_bgc_chl[mm]), :], f"bgc_chl{mm + 1:03d}"
                )
        if bool(t.tr_bgc_C):
            for stem, index, count in (("bgc_DOC", t.nt_bgc_DOC, t.n_doc),
                                       ("bgc_DIC", t.nt_bgc_DIC, t.n_dic)):
                for mm in range(int(count)):
                    ice_restart.write_restart_field(
                        trcrn[:, :, int(index[mm]), :], f"{stem}{mm + 1:03d}"
                    )
        for flag, index, name in (
            (t.tr_bgc_Nit, t.nt_bgc_Nit, "bgc_Nit"),
            (t.tr_bgc_Am, t.nt_bgc_Am, "bgc_Am"),
            (t.tr_bgc_Sil, t.nt_bgc_Sil, "bgc_Sil"),
            (t.tr_bgc_hum, t.nt_bgc_hum, "bgc_hum"),
            (t.tr_bgc_PON, t.nt_bgc_PON, "bgc_PON"),
            (t.tr_bgc_DMS, t.nt_bgc_DMSPp, "bgc_DMSPp"),
            (t.tr_bgc_DMS, t.nt_bgc_DMSPd, "bgc_DMSPd"),
            (t.tr_bgc_DMS, t.nt_bgc_DMS, "bgc_DMS"),
        ):
            if bool(flag):
                ice_restart.write_restart_field(trcrn[:, :, int(index), :], name)
        for flag, stem, index, count in (
            (t.tr_bgc_DON, "bgc_DON", t.nt_bgc_DON, t.n_don),
            (t.tr_bgc_Fe, "bgc_Fed", t.nt_bgc_Fed, t.n_fed),
            (t.tr_bgc_Fe, "bgc_Fep", t.nt_bgc_Fep, t.n_fep),
        ):
            if bool(flag):
                for mm in range(int(count)):
                    ice_restart.write_restart_field(
                        trcrn[:, :, int(index[mm]), :], f"{stem}{mm + 1:03d}"
                    )
    else:
        # The z-layer files use a separate field for each species and depth.
        layers = int(t.nblyr) + 3
        for flag, index, stem in (
            (t.tr_bgc_Nit, t.nt_bgc_Nit, "bgc_Nit"),
            (t.tr_bgc_Am, t.nt_bgc_Am, "bgc_Am"),
            (t.tr_bgc_Sil, t.nt_bgc_Sil, "bgc_Sil"),
            (t.tr_bgc_hum, t.nt_bgc_hum, "bgc_hum"),
            (t.tr_bgc_PON, t.nt_bgc_PON, "bgc_PON"),
            (t.tr_bgc_DMS, t.nt_bgc_DMSPp, "bgc_DMSPp"),
            (t.tr_bgc_DMS, t.nt_bgc_DMSPd, "bgc_DMSPd"),
            (t.tr_bgc_DMS, t.nt_bgc_DMS, "bgc_DMS"),
        ):
            if bool(flag):
                for k in range(layers):
                    ice_restart.write_restart_field(
                        trcrn[:, :, int(index) + k, :], f"{stem}{k + 1:03d}"
                    )
        for flag, stem, index, count in (
            (t.tr_bgc_N, "bgc_N", t.nt_bgc_N, t.n_algae),
            (bool(t.tr_bgc_N) and bool(t.tr_bgc_chl), "bgc_chl", t.nt_bgc_chl, t.n_algae),
            (t.tr_bgc_C, "bgc_DOC", t.nt_bgc_DOC, t.n_doc),
            (t.tr_bgc_C, "bgc_DIC", t.nt_bgc_DIC, t.n_dic),
            (t.tr_bgc_DON, "bgc_DON", t.nt_bgc_DON, t.n_don),
            (t.tr_bgc_Fe, "bgc_Fed", t.nt_bgc_Fed, t.n_fed),
            (t.tr_bgc_Fe, "bgc_Fep", t.nt_bgc_Fep, t.n_fep),
            (t.tr_zaero, "zaero", t.nt_zaero, t.n_zaero),
        ):
            if bool(flag):
                for mm in range(int(count)):
                    for k in range(layers):
                        ice_restart.write_restart_field(
                            trcrn[:, :, int(index[mm]) + k, :],
                            f"{stem}{mm + 1:03d}{k + 1:03d}",
                        )
        for mm in range(int(t.nbtrcr)):
            ice_restart.write_restart_field(
                trcrn[:, :, int(t.nt_zbgc_frac) + mm, :], f"zbgc_frac{mm + 1:03d}"
            )

    # Fortran clears ocean BGC over land before writing (except zaeros).
    ocean = ice_grid.tmask
    for flag, name, stem, count in (
        (t.tr_bgc_N, "algalN", "algalN", t.n_algae),
        (t.tr_bgc_C, "doc", "doc", t.n_doc),
        (t.tr_bgc_C, "dic", "dic", t.n_dic),
        (t.tr_bgc_DON, "don", "don", t.n_don),
        (t.tr_bgc_Fe, "fed", "fed", t.n_fed),
        (t.tr_bgc_Fe, "fep", "fep", t.n_fep),
        (t.tr_zaero, "zaeros", "zaeros", t.n_zaero),
    ):
        if bool(flag):
            field = getattr(ice_flux_bgc, name)
            for mm in range(int(count)):
                plane = field[:, :, mm]
                if name != "zaeros":
                    plane = np.where(ocean, plane, 0.0)
                ice_restart.write_restart_field(plane, f"{stem}{mm + 1:03d}")
    for flag, name in (
        (t.tr_bgc_Nit, "nit"), (t.tr_bgc_Am, "amm"),
        (t.tr_bgc_Sil, "sil"), (t.tr_bgc_hum, "hum"),
        (t.tr_bgc_DMS, "dmsp"), (t.tr_bgc_DMS, "dms"),
    ):
        if bool(flag):
            ice_restart.write_restart_field(
                np.where(ocean, getattr(ice_flux_bgc, name), 0.0), name
            )


def read_restart_age():
    """Reads all values needed for an ice age restart.

    author Elizabeth C. Hunke, LANL
    """
    subname = "(read_restart_age)"
    nt_iage = int(icepack_tracers.nt_iage)
    ice_fileunits.nu_diag.write(f" {subname} min/max age (s)\n")
    _cat(ice_state.trcrn[:, :, nt_iage, :], "iage")


def read_restart_FY():
    """Reads all values needed for a first-year ice restart.

    author Elizabeth C. Hunke, LANL
    """
    subname = "(read_restart_FY)"
    nt_FY = int(icepack_tracers.nt_FY)
    nu = ice_fileunits.nu_diag
    nu.write(f" {subname} min/max first-year ice area\n")
    _cat(ice_state.trcrn[:, :, nt_FY, :], "FY")
    nu.write(f" {subname} min/max frz_onset\n")
    _horiz(ice_flux.frz_onset, "frz_onset")


def read_restart_lvl():
    """Reads all values needed for a level-ice restart.

    author Elizabeth C. Hunke, LANL
    """
    subname = "(read_restart_lvl)"
    ice_fileunits.nu_diag.write(f" {subname} min/max level ice area, volume\n")
    _cat(ice_state.trcrn[:, :, int(icepack_tracers.nt_alvl), :], "alvl")
    _cat(ice_state.trcrn[:, :, int(icepack_tracers.nt_vlvl), :], "vlvl")


def read_restart_pond_lvl():
    """Reads level-ice melt ponds.

    author Elizabeth C. Hunke, LANL
    """
    subname = "(read_restart_pond_lvl)"
    ice_fileunits.nu_diag.write(f" {subname} min/max level-ice ponds\n")
    _pond_tracers()
    _horiz(ice_flux.fsnow, "fsnow")
    _cat(ice_arrays_column.dhsn, "dhs")
    _cat(ice_arrays_column.ffracn, "ffrac")


def read_restart_pond_topo():
    """Reads topographic melt ponds.

    author Elizabeth C. Hunke, LANL
    """
    subname = "(read_restart_pond_topo)"
    ice_fileunits.nu_diag.write(f" {subname} min/max topo ponds\n")
    _pond_tracers()


def read_restart_pond_sealvl():
    """Reads sea-level melt ponds.

    author Elizabeth C. Hunke, LANL
    """
    subname = "(read_restart_pond_sealvl)"
    ice_fileunits.nu_diag.write(f" {subname} min/max sealvl ponds\n")
    _pond_tracers()
    _horiz(ice_flux.fsnow, "fsnow")
    _cat(ice_arrays_column.dhsn, "dhs")
    _cat(ice_arrays_column.ffracn, "ffrac")


def read_restart_snow():
    """Reads snow redistribution tracers.

    author Elizabeth C. Hunke, LANL
    """
    subname = "(read_restart_snow)"
    ice_fileunits.nu_diag.write(f" {subname} min/max snow tracers\n")
    nt_smice = int(icepack_tracers.nt_smice)
    nt_smliq = int(icepack_tracers.nt_smliq)
    nt_rhos = int(icepack_tracers.nt_rhos)
    nt_rsnw = int(icepack_tracers.nt_rsnw)
    for k in range(1, int(ice_domain_size.nslyr) + 1):
        ck = f"{k:03d}"
        _cat(ice_state.trcrn[:, :, nt_smice + k - 1, :], "smice" + ck)
        _cat(ice_state.trcrn[:, :, nt_smliq + k - 1, :], "smliq" + ck)
        _cat(ice_state.trcrn[:, :, nt_rhos + k - 1, :], "rhos" + ck)
        _cat(ice_state.trcrn[:, :, nt_rsnw + k - 1, :], "rsnw" + ck)


def read_restart_fsd():
    """Reads all values needed for an ice fsd restart.

    author Elizabeth C. Hunke, LANL
    """
    subname = "(read_restart_fsd)"
    ice_fileunits.nu_diag.write(f" {subname} min/max fsd (s)\n")
    nt_fsd = int(icepack_tracers.nt_fsd)
    for k in range(1, int(ice_domain_size.nfsd) + 1):
        ck = f"{k:03d}"
        _cat(ice_state.trcrn[:, :, nt_fsd + k - 1, :], "fsd" + ck)


def read_restart_iso():
    """Reads isotope tracers.

    author Elizabeth C. Hunke, LANL
    """
    nt_isosno = int(icepack_tracers.nt_isosno)
    nt_isoice = int(icepack_tracers.nt_isoice)
    n_iso = int(icepack_tracers.n_iso)
    for k in range(1, n_iso + 1):
        ck = f"{k:03d}"
        _cat(ice_state.trcrn[:, :, nt_isosno + k - 1, :], "isosno" + ck)
    for k in range(1, n_iso + 1):
        ck = f"{k:03d}"
        _cat(ice_state.trcrn[:, :, nt_isoice + k - 1, :], "isoice" + ck)


def read_restart_aero():
    """Reads all values needed for an ice aerosol restart.

    authors Elizabeth Hunke, LANL (original version)
            David Bailey, NCAR
            Marika Holland, NCAR
    """
    subname = "(read_restart_aero)"
    ice_fileunits.nu_diag.write(f" {subname} aerosols\n")
    nt_aero = int(icepack_tracers.nt_aero)
    # Driver n_aero is zero when the tracer is off. The namelist count is on icepack.
    n_aero = int(ice_init.n_aero) if int(ice_init.n_aero) > 0 else int(icepack_tracers.n_aero)
    for k in range(1, n_aero + 1):
        nchar = f"{k:03d}"
        base = nt_aero + (k - 1) * 4
        _cat(ice_state.trcrn[:, :, base, :], "aerosnossl" + nchar)
        _cat(ice_state.trcrn[:, :, base + 1, :], "aerosnoint" + nchar)
        _cat(ice_state.trcrn[:, :, base + 2, :], "aeroicessl" + nchar)
        _cat(ice_state.trcrn[:, :, base + 3, :], "aeroiceint" + nchar)


def read_restart_hbrine():
    """Reads all values needed for hbrine.

    author Elizabeth C. Hunke, LANL
    """
    subname = "(read_restart_hbrine)"
    ice_fileunits.nu_diag.write(f" {subname} brine restart\n")
    nt_fbri = int(icepack_tracers.nt_fbri)
    _cat(ice_state.trcrn[:, :, nt_fbri, :], "fbrn")
    _cat(ice_arrays_column.first_ice_real, "first_ice")
    # physical domain, one grid
    ilo, ihi, jlo, jhi = ice_domain_size.physical_bounds()
    real = ice_arrays_column.first_ice_real
    flag = ice_arrays_column.first_ice
    flag[ilo:ihi + 1, jlo:jhi + 1, :] = real[ilo:ihi + 1, jlo:jhi + 1, :] >= p5


def read_restart_bgc():
    """Read skeletal or vertical ice BGC tracers and mixed-layer ocean BGC.

    author Elizabeth C. Hunke, LANL
    """
    subname = "(read_restart_bgc)"
    t = icepack_tracers
    trcrn = ice_state.trcrn
    nblyr = int(t.nblyr)
    layers = nblyr + 3

    #-----------------------------------------------------------------
    # Skeletal Layer BGC
    #-----------------------------------------------------------------
    if bool(restart_bgc):
        if bool(icepack_parameters.skl_bgc):
            ice_fileunits.nu_diag.write(f" {subname} skl bgc restart\n")
            for k in range(int(t.n_algae)):
                nchar = f"{k + 1:03d}"
                _cat(trcrn[:, :, int(t.nt_bgc_N[k]), :], "bgc_N" + nchar,
                     update_halo=False)
                if bool(t.tr_bgc_chl):
                    _cat(trcrn[:, :, int(t.nt_bgc_chl[k]), :], "bgc_chl" + nchar,
                         update_halo=False)
            if bool(t.tr_bgc_C):
                # The Fortran algal-carbon restart loop is commented out.
                for k in range(int(t.n_doc)):
                    _cat(trcrn[:, :, int(t.nt_bgc_DOC[k]), :], f"bgc_DOC{k + 1:03d}",
                         update_halo=False)
                for k in range(int(t.n_dic)):
                    _cat(trcrn[:, :, int(t.nt_bgc_DIC[k]), :], f"bgc_DIC{k + 1:03d}",
                         update_halo=False)
            for flag, index, name in (
                (t.tr_bgc_Nit, t.nt_bgc_Nit, "bgc_Nit"),
                (t.tr_bgc_Am, t.nt_bgc_Am, "bgc_Am"),
                (t.tr_bgc_Sil, t.nt_bgc_Sil, "bgc_Sil"),
                (t.tr_bgc_hum, t.nt_bgc_hum, "bgc_hum"),
                (t.tr_bgc_PON, t.nt_bgc_PON, "bgc_PON"),
            ):
                if bool(flag):
                    _cat(trcrn[:, :, int(index), :], name, update_halo=False)
            if bool(t.tr_bgc_DMS):
                for index, name in (
                    (t.nt_bgc_DMSPp, "bgc_DMSPp"),
                    (t.nt_bgc_DMSPd, "bgc_DMSPd"),
                    (t.nt_bgc_DMS, "bgc_DMS"),
                ):
                    _cat(trcrn[:, :, int(index), :], name, update_halo=False)
            if bool(t.tr_bgc_DON):
                for k in range(int(t.n_don)):
                    _cat(trcrn[:, :, int(t.nt_bgc_DON[k]), :], f"bgc_DON{k + 1:03d}",
                         update_halo=False)
            if bool(t.tr_bgc_Fe):
                for k in range(int(t.n_fed)):
                    _cat(trcrn[:, :, int(t.nt_bgc_Fed[k]), :], f"bgc_Fed{k + 1:03d}",
                         update_halo=False)
                for k in range(int(t.n_fep)):
                    _cat(trcrn[:, :, int(t.nt_bgc_Fep[k]), :], f"bgc_Fep{k + 1:03d}",
                         update_halo=False)
        else:
            #-----------------------------------------------------------------
            # Z Layer BGC
            #-----------------------------------------------------------------
            for flag, index, name in (
                (t.tr_bgc_Nit, t.nt_bgc_Nit, "bgc_Nit"),
                (t.tr_bgc_Am, t.nt_bgc_Am, "bgc_Am"),
                (t.tr_bgc_Sil, t.nt_bgc_Sil, "bgc_Sil"),
                (t.tr_bgc_hum, t.nt_bgc_hum, "bgc_hum"),
                (t.tr_bgc_PON, t.nt_bgc_PON, "bgc_PON"),
            ):
                if bool(flag):
                    for k in range(layers):
                        _cat(trcrn[:, :, int(index) + k, :], f"{name}{k + 1:03d}")
            if bool(t.tr_bgc_N):
                for mm in range(int(t.n_algae)):
                    for k in range(layers):
                        suffix = f"{mm + 1:03d}{k + 1:03d}"
                        _cat(trcrn[:, :, int(t.nt_bgc_N[mm]) + k, :], "bgc_N" + suffix)
                        if bool(t.tr_bgc_chl):
                            _cat(trcrn[:, :, int(t.nt_bgc_chl[mm]) + k, :], "bgc_chl" + suffix)
            if bool(t.tr_bgc_C):
                # The Fortran algal-carbon restart loop is commented out.
                for mm in range(int(t.n_doc)):
                    for k in range(layers):
                        _cat(trcrn[:, :, int(t.nt_bgc_DOC[mm]) + k, :],
                             f"bgc_DOC{mm + 1:03d}{k + 1:03d}")
                for mm in range(int(t.n_dic)):
                    for k in range(layers):
                        _cat(trcrn[:, :, int(t.nt_bgc_DIC[mm]) + k, :],
                             f"bgc_DIC{mm + 1:03d}{k + 1:03d}")
            if bool(t.tr_bgc_DMS):
                for index, name in (
                    (t.nt_bgc_DMSPp, "bgc_DMSPp"),
                    (t.nt_bgc_DMSPd, "bgc_DMSPd"),
                    (t.nt_bgc_DMS, "bgc_DMS"),
                ):
                    for k in range(layers):
                        _cat(trcrn[:, :, int(index) + k, :], f"{name}{k + 1:03d}")
            if bool(t.tr_bgc_DON):
                for mm in range(int(t.n_don)):
                    for k in range(layers):
                        _cat(trcrn[:, :, int(t.nt_bgc_DON[mm]) + k, :],
                             f"bgc_DON{mm + 1:03d}{k + 1:03d}")
            if bool(t.tr_bgc_Fe):
                for mm in range(int(t.n_fed)):
                    for k in range(layers):
                        _cat(trcrn[:, :, int(t.nt_bgc_Fed[mm]) + k, :],
                             f"bgc_Fed{mm + 1:03d}{k + 1:03d}")
                for mm in range(int(t.n_fep)):
                    for k in range(layers):
                        _cat(trcrn[:, :, int(t.nt_bgc_Fep[mm]) + k, :],
                             f"bgc_Fep{mm + 1:03d}{k + 1:03d}")
            if bool(t.tr_zaero):
                for mm in range(int(t.n_zaero)):
                    for k in range(layers):
                        _cat(trcrn[:, :, int(t.nt_zaero[mm]) + k, :],
                             f"zaero{mm + 1:03d}{k + 1:03d}")
            for mm in range(int(t.nbtrcr)):
                _cat(trcrn[:, :, int(t.nt_zbgc_frac) + mm, :],
                     f"zbgc_frac{mm + 1:03d}", update_halo=False)

        #-----------------------------------------------------------------
        # Ocean BGC
        #-----------------------------------------------------------------
        ice_fileunits.nu_diag.write(f" {subname} mixed layer ocean bgc restart\n")
        if bool(t.tr_bgc_N):
            for k in range(int(t.n_algae)):
                _horiz(ice_flux_bgc.algalN[:, :, k], f"algalN{k + 1:03d}",
                       update_halo=False)
        if bool(t.tr_bgc_C):
            for k in range(int(t.n_doc)):
                _horiz(ice_flux_bgc.doc[:, :, k], f"doc{k + 1:03d}",
                       update_halo=False)
            for k in range(int(t.n_dic)):
                _horiz(ice_flux_bgc.dic[:, :, k], f"dic{k + 1:03d}",
                       update_halo=False)
        for flag, field, name in (
            (t.tr_bgc_Nit, ice_flux_bgc.nit, "nit"),
            (t.tr_bgc_Am, ice_flux_bgc.amm, "amm"),
            (t.tr_bgc_Sil, ice_flux_bgc.sil, "sil"),
            (t.tr_bgc_hum, ice_flux_bgc.hum, "hum"),
            (t.tr_bgc_DMS, ice_flux_bgc.dmsp, "dmsp"),
            (t.tr_bgc_DMS, ice_flux_bgc.dms, "dms"),
        ):
            if bool(flag):
                _horiz(field, name, update_halo=name in ("nit", "amm", "sil", "hum"))
        if bool(t.tr_bgc_DON):
            for k in range(int(t.n_don)):
                _horiz(ice_flux_bgc.don[:, :, k], f"don{k + 1:03d}",
                       update_halo=False)
        if bool(t.tr_bgc_Fe):
            for k in range(int(t.n_fed)):
                _horiz(ice_flux_bgc.fed[:, :, k], f"fed{k + 1:03d}",
                       update_halo=False)
            for k in range(int(t.n_fep)):
                _horiz(ice_flux_bgc.fep[:, :, k], f"fep{k + 1:03d}",
                       update_halo=False)
        if bool(t.tr_zaero):
            for k in range(int(t.n_zaero)):
                _horiz(ice_flux_bgc.zaeros[:, :, k], f"zaeros{k + 1:03d}",
                       update_halo=False)


def _pond_tracers():
    _cat(ice_state.trcrn[:, :, int(icepack_tracers.nt_apnd), :], "apnd")
    _cat(ice_state.trcrn[:, :, int(icepack_tracers.nt_hpnd), :], "hpnd")
    _cat(ice_state.trcrn[:, :, int(icepack_tracers.nt_ipnd), :], "ipnd")


def _cat(work, vname, update_halo=True):
    # Fortran omits field_loc on skeletal BGC fields and zbgc_frac.
    ice_restart.read_restart_field(
        work,
        vname,
        int(ice_domain_size.ncat),
        True,
        field_loc=field_loc_center if update_halo else None,
        field_type=field_type_scalar if update_halo else None,
    )


def _horiz(work, vname, update_halo=True):
    ice_restart.read_restart_field(
        work, vname, 1, True,
        field_loc=field_loc_center if update_halo else None,
        field_type=field_type_scalar if update_halo else None,
    )
