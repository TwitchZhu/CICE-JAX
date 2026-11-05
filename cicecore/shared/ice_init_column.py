"""Initialization routines for the column package.

author for JAX version: Xinyuan Zhu
XZ: converted from F90 to Python

author: Elizabeth C. Hunke, LANL

Ported: ``init_thermo_vertical``, ``init_shortwave``, ``init_age``, ``init_FY``,
``init_lvl``, ``init_meltponds_lvl``, ``init_meltponds_topo``,
``init_meltponds_sealvl``, ``init_snowtracers``, ``init_fsd``,
``init_isotope``, ``init_aerosol``, ``init_bgc``, ``init_hbrine``,
``input_zbgc``, ``init_zbgc``, and ``init_bgc_trcr``.
``count_tracers`` in this file is not the driver copy; icepack has its own.
``icepack_init_bgc`` writes a local tracer copy that Fortran never copies
back onto ``trcrn``.
"""

import re

import numpy as np

from cicecore.cicedyn.general import ice_flux
from cicecore.cicedyn.general import ice_flux_bgc
from cicecore.cicedyn.general import ice_forcing_bgc
from cicecore.cicedyn.general import ice_init
from cicecore.cicedyn.general import ice_state
from cicecore.cicedyn.infrastructure import ice_grid
from cicecore.shared import ice_arrays_column
from cicecore.shared import ice_calendar
from cicecore.shared import ice_domain_size
from cicecore.shared import ice_fileunits
from cicecore.shared import ice_restart_column
from icepack.columnphysics import icepack_fsd
from icepack.columnphysics import icepack_orbital
from icepack.columnphysics import icepack_parameters
from icepack.columnphysics import icepack_shortwave
from icepack.columnphysics import icepack_therm_shared
from icepack.columnphysics import icepack_tracers
from icepack.columnphysics.icepack_kinds import dbl_kind, int_kind, log_kind
from icepack.columnphysics.icepack_parameters import (
    c0,
    c1,
    c1p5,
    c2,
    p1,
    p25,
    p5,
    rhoi,
    rhos,
)


def init_thermo_vertical():
    """Initialize the vertical profile of ice salinity and melting temperature.

    authors: C. M. Bitz, UW
             William H. Lipscomb, LANL
    """
    nilyr = int(ice_domain_size.nilyr)
    # vertical salinity profile
    sprofile = np.zeros(nilyr + 1, dtype=dbl_kind)

    #-----------------------------------------------------------------
    # initialize
    #-----------------------------------------------------------------
    # depressT is the module parameter. icepack_query_parameters only reads it.
    depressT = icepack_parameters.depressT
    icepack_therm_shared.icepack_init_thermo(sprofile=sprofile)
    # call icepack_warnings_flush(nu_diag)
    # if (icepack_warnings_aborted()) call abort_ice(...)
    # Column-physics failures raise. The warning flush is not ported.

    #-----------------------------------------------------------------
    # Prescibe vertical profile of salinity and melting temperature.
    # Note this profile is only used for BL99 thermodynamics.
    #-----------------------------------------------------------------
    ice_flux.salinz[...] = sprofile
    ice_flux.Tmltz[...] = -ice_flux.salinz * depressT


def init_age(iage):
    """Initialize ice age tracer (call prior to reading restart data)."""
    # iage(:,:,:) = c0
    iage[...] = 0


def init_FY(firstyear):
    """Initialize ice FY tracer (call prior to reading restart data)."""
    firstyear[...] = 0


def init_lvl(alvl, vlvl):
    """Initialize ice lvl tracers (call prior to reading restart data).

    The Fortran ``iblk`` argument is omitted. One grid, so ``ffracn`` and
    ``dhsn`` are the whole arrays.
    """
    alvl[...] = c1  # level ice area fraction
    vlvl[...] = c1  # level ice volume
    ice_arrays_column.ffracn[...] = c0
    ice_arrays_column.dhsn[...] = c0


def init_meltponds_lvl(apnd, hpnd, ipnd, dhsn):
    """Initialize melt ponds."""
    apnd[...] = 0
    hpnd[...] = 0
    ipnd[...] = 0
    dhsn[...] = 0


def init_meltponds_sealvl(apnd, hpnd, ipnd, dhsn):
    """Initialize melt ponds."""
    apnd[...] = 0
    hpnd[...] = 0
    ipnd[...] = 0
    dhsn[...] = 0


def init_meltponds_topo(apnd, hpnd, ipnd):
    """Initialize melt ponds."""
    apnd[...] = 0
    hpnd[...] = 0
    ipnd[...] = 0


def init_snowtracers(smice, smliq, rhos_cmp, rsnw):
    """Initialize snow redistribution/metamorphosis tracers (call prior to reading restart data)."""
    # snow grain radius of new fallen snow  (10^-6 m)
    rsnw_fall = icepack_parameters.rsnw_fall
    rhos = icepack_parameters.rhos  # snow density (kg/m^3)
    rsnw[...] = rsnw_fall
    rhos_cmp[...] = rhos
    smice[...] = rhos
    smliq[...] = 0


def init_isotope(isosno, isoice):
    """Initialize isotope tracers (call prior to reading restart data)."""
    isosno[...] = 0
    isoice[...] = 0


def init_aerosol(aero):
    """Initialize ice aerosol tracer (call prior to reading restart data)."""
    aero[...] = 0


def init_fsd(floesize):
    """Initialize floe size distribution tracer (call prior to reading restart data).

    ``floesize`` has shape ``(i, j, nfsd, ncat)``. One grid, so there is no block index.
    """
    subname = "(init_fsd)"
    puny = icepack_parameters.puny
    col = ice_arrays_column
    col.wavefreq[...] = 0
    col.dwavefreq[...] = 0
    col.wave_sig_ht[...] = 0
    col.wave_spectrum[...] = 0
    col.d_afsd_newi[...] = 0
    col.d_afsd_latg[...] = 0
    col.d_afsd_latm[...] = 0
    col.d_afsd_wave[...] = 0
    col.d_afsd_weld[...] = 0

    nfsd = int(ice_domain_size.nfsd)
    ncat = int(ice_domain_size.ncat)
    # default: floes occupy the smallest size category in all thickness categories
    floesize[...] = 0
    floesize[:, :, 0, :] = 1

    tr_fsd = bool(icepack_tracers.tr_fsd)
    # call icepack_warnings_flush(nu_diag)
    # Column-physics failures raise.
    _ = subname
    if tr_fsd:
        # initialize floe size distribution the same in every column and category
        afsd = np.zeros(nfsd, dtype=dbl_kind)
        icepack_fsd.icepack_init_fsd(ice_ic=str(ice_init.ice_ic).strip(), afsd=afsd)
        aicen = ice_state.aicen
        afsdn = np.zeros((nfsd, ncat), dtype=dbl_kind)
        ny = floesize.shape[1]
        nx = floesize.shape[0]
        for j in range(ny):
            for i in range(nx):
                afsdn[:, :] = 0
                afsdn[0, :] = 1
                for n in range(ncat):
                    for k in range(nfsd):
                        if aicen[i, j, n] > puny:
                            afsdn[k, n] = afsd[k]
                icepack_fsd.icepack_cleanup_fsd(afsdn=afsdn)  # renormalize
                for n in range(ncat):
                    for k in range(nfsd):
                        floesize[i, j, k, n] = afsdn[k, n]
        ice_fileunits.nu_diag.flush()


def init_shortwave():
    """Initialize shortwave radiation using the current or restart shortwave.

    authors: William H. Lipscomb, LANL
             Elizabeth C. Hunke, LANL

    One grid. ``icepack_init_orbit`` does not depend on i or j. The Fortran
    calls it inside the i,j loop; one call before the loop is the same result.
    CESMCOUPLED and GEOSCOUPLED are not defined, so the standalone orbit call runs.
    """
    puny = icepack_parameters.puny
    shortwave = str(icepack_parameters.shortwave).strip()
    dEdd_algae = bool(icepack_parameters.dEdd_algae)
    snwgrain = bool(icepack_parameters.snwgrain)
    tr_brine = bool(icepack_tracers.tr_brine)
    ncat = int(ice_domain_size.ncat)
    nslyr = int(ice_domain_size.nslyr)
    nbtrcr_sw = int(icepack_tracers.nbtrcr_sw)
    nt_alvl = int(icepack_tracers.nt_alvl)
    nt_apnd = int(icepack_tracers.nt_apnd)
    nt_hpnd = int(icepack_tracers.nt_hpnd)
    nt_ipnd = int(icepack_tracers.nt_ipnd)
    nt_aero = int(icepack_tracers.nt_aero)
    nt_fbri = int(icepack_tracers.nt_fbri)
    nt_Tsfc = int(icepack_tracers.nt_Tsfc)
    nt_rsnw = int(icepack_tracers.nt_rsnw)
    n_aero = int(icepack_tracers.n_aero)
    n_algae = int(icepack_tracers.n_algae)
    nblyr = int(ice_domain_size.nblyr)
    col = ice_arrays_column
    flux = ice_flux

    # Initialize
    col.fswpenln[...] = c0
    col.Iswabsn[...] = c0
    col.Sswabsn[...] = c0
    flux.alvdf[...] = c0
    flux.alidf[...] = c0
    flux.alvdr[...] = c0
    flux.alidr[...] = c0
    flux.alvdr_ai[...] = c0
    flux.alidr_ai[...] = c0
    flux.alvdf_ai[...] = c0
    flux.alidf_ai[...] = c0
    flux.albice[...] = c0
    flux.albsno[...] = c0
    flux.albpnd[...] = c0
    flux.snowfrac[...] = c0
    flux.apeff_ai[...] = c0
    for name in (
        "alvdrn", "alidrn", "alvdfn", "alidfn", "albpndn", "albicen", "albsnon",
        "apeffn", "snowfracn", "fswsfcn", "fswintn", "fswthrun",
        "fswthrun_vdr", "fswthrun_vdf", "fswthrun_idr", "fswthrun_idf",
        "fswthrun_uvrdr", "fswthrun_uvrdf", "fswthrun_pardr", "fswthrun_pardf",
    ):
        getattr(col, name)[...] = c0

    if shortwave[:4] == "dEdd":  # delta Eddington
        # initialize orbital parameters
        # These come from the driver in the coupled model.
        icepack_orbital.icepack_init_orbit()

    ilo, ihi, jlo, jhi = ice_domain_size.physical_bounds()
    ztrcr_sw = np.zeros((max(nbtrcr_sw, 1), ncat), dtype=dbl_kind)
    rsnow = np.zeros((nslyr, ncat), dtype=dbl_kind)
    fbri = np.zeros(ncat, dtype=dbl_kind)
    trcrn = ice_state.trcrn
    aicen = ice_state.aicen
    for j in range(jlo, jhi + 1):
        for i in range(ilo, ihi + 1):
            l_print_point = False
            fbri[...] = c0
            ztrcr_sw[...] = c0
            rsnow[...] = c0
            for n in range(ncat):
                if tr_brine:
                    fbri[n] = trcrn[i, j, nt_fbri, n]
                if snwgrain:
                    for k in range(nslyr):
                        rsnow[k, n] = trcrn[i, j, nt_rsnw + k, n]
            if ice_grid.tmask[i, j] or ice_grid.opmask[i, j]:
                aero_n = n_aero * 4
                bgc_n = n_algae * (nblyr + 3)
                n_zaero = int(getattr(icepack_tracers, "n_zaero", 0))
                zaero_n = n_zaero * (nblyr + 3)
                aeron = _tracer_block(trcrn, i, j, nt_aero, aero_n, ncat)
                # Fortran nt_bgc_N(1), nt_zaero(1). Both are length max_algae / max_aero.
                nt_bgc_n = int(icepack_tracers.nt_bgc_N[0])
                bgcNn = _tracer_block(trcrn, i, j, nt_bgc_n, bgc_n, ncat)
                nt_zaero = int(icepack_tracers.nt_zaero[0])
                zaeron = _tracer_block(trcrn, i, j, nt_zaero, zaero_n, ncat)
                # coszen is intent(inout). A grid value is a scalar, so the
                # updated cosine comes back as the return value.
                flux.coszen[i, j] = icepack_shortwave.icepack_step_radiation(
                    dt=ice_calendar.dt,
                    fbri=fbri,
                    aicen=aicen[i, j, :],
                    vicen=ice_state.vicen[i, j, :],
                    vsnon=ice_state.vsnon[i, j, :],
                    Tsfcn=trcrn[i, j, nt_Tsfc, :],
                    alvln=trcrn[i, j, nt_alvl, :],
                    apndn=trcrn[i, j, nt_apnd, :],
                    hpndn=trcrn[i, j, nt_hpnd, :],
                    ipndn=trcrn[i, j, nt_ipnd, :],
                    aeron=aeron,
                    bgcNn=bgcNn,
                    zaeron=zaeron,
                    trcrn_bgcsw=ztrcr_sw,
                    TLAT=ice_grid.TLAT[i, j],
                    TLON=ice_grid.TLON[i, j],
                    calendar_type=ice_calendar.calendar_type,
                    days_per_year=ice_calendar.days_per_year,
                    nextsw_cday=ice_calendar.nextsw_cday,
                    yday=ice_calendar.yday,
                    sec=ice_calendar.msec,
                    swvdr=flux.swvdr[i, j],
                    swvdf=flux.swvdf[i, j],
                    swidr=flux.swidr[i, j],
                    swidf=flux.swidf[i, j],
                    swuvrdr=flux.swuvrdr[i, j],
                    swuvrdf=flux.swuvrdf[i, j],
                    swpardr=flux.swpardr[i, j],
                    swpardf=flux.swpardf[i, j],
                    coszen=flux.coszen[i, j],
                    fsnow=flux.fsnow[i, j],
                    alvdrn=col.alvdrn[i, j, :],
                    alvdfn=col.alvdfn[i, j, :],
                    alidrn=col.alidrn[i, j, :],
                    alidfn=col.alidfn[i, j, :],
                    fswsfcn=col.fswsfcn[i, j, :],
                    fswintn=col.fswintn[i, j, :],
                    fswthrun=col.fswthrun[i, j, :],
                    fswthrun_vdr=col.fswthrun_vdr[i, j, :],
                    fswthrun_vdf=col.fswthrun_vdf[i, j, :],
                    fswthrun_idr=col.fswthrun_idr[i, j, :],
                    fswthrun_idf=col.fswthrun_idf[i, j, :],
                    fswthrun_uvrdr=col.fswthrun_uvrdr[i, j, :],
                    fswthrun_uvrdf=col.fswthrun_uvrdf[i, j, :],
                    fswthrun_pardr=col.fswthrun_pardr[i, j, :],
                    fswthrun_pardf=col.fswthrun_pardf[i, j, :],
                    fswpenln=col.fswpenln[i, j, :, :],
                    Sswabsn=col.Sswabsn[i, j, :, :],
                    Iswabsn=col.Iswabsn[i, j, :, :],
                    albicen=col.albicen[i, j, :],
                    albsnon=col.albsnon[i, j, :],
                    albpndn=col.albpndn[i, j, :],
                    apeffn=col.apeffn[i, j, :],
                    snowfracn=col.snowfracn[i, j, :],
                    dhsn=col.dhsn[i, j, :],
                    ffracn=col.ffracn[i, j, :],
                    rsnow=rsnow,
                    l_print_point=l_print_point,
                    initonly=True,
                )
            #-----------------------------------------------------------------
            # Define aerosol tracer on shortwave grid
            #-----------------------------------------------------------------
            if dEdd_algae and (
                bool(getattr(icepack_tracers, "tr_zaero", False))
                or bool(getattr(icepack_tracers, "tr_bgc_N", False))
            ):
                if col.trcrn_sw is not None:
                    for n in range(ncat):
                        for k in range(nbtrcr_sw):
                            col.trcrn_sw[i, j, k, n] = ztrcr_sw[k, n]

    #-----------------------------------------------------------------
    # Aggregate albedos
    # Match loop order in coupling_prep for same order of operations
    #-----------------------------------------------------------------
    for n in range(ncat):
        for j in range(jlo, jhi + 1):
            for i in range(ilo, ihi + 1):
                if aicen[i, j, n] > puny:
                    flux.alvdf[i, j] = flux.alvdf[i, j] + col.alvdfn[i, j, n] * aicen[i, j, n]
                    flux.alidf[i, j] = flux.alidf[i, j] + col.alidfn[i, j, n] * aicen[i, j, n]
                    flux.alvdr[i, j] = flux.alvdr[i, j] + col.alvdrn[i, j, n] * aicen[i, j, n]
                    flux.alidr[i, j] = flux.alidr[i, j] + col.alidrn[i, j, n] * aicen[i, j, n]
                    netsw = (
                        flux.swvdr[i, j] + flux.swidr[i, j]
                        + flux.swvdf[i, j] + flux.swidf[i, j]
                    )
                    if netsw > puny:  # sun above horizon
                        flux.albice[i, j] = flux.albice[i, j] + col.albicen[i, j, n] * aicen[i, j, n]
                        flux.albsno[i, j] = flux.albsno[i, j] + col.albsnon[i, j, n] * aicen[i, j, n]
                        flux.albpnd[i, j] = flux.albpnd[i, j] + col.albpndn[i, j, n] * aicen[i, j, n]
                    flux.apeff_ai[i, j] = flux.apeff_ai[i, j] + col.apeffn[i, j, n] * aicen[i, j, n]
                    flux.snowfrac[i, j] = flux.snowfrac[i, j] + col.snowfracn[i, j, n] * aicen[i, j, n]

    #----------------------------------------------------------------
    # Store grid box mean albedos and fluxes before scaling by aice
    #----------------------------------------------------------------
    flux.alvdf_ai[...] = flux.alvdf
    flux.alidf_ai[...] = flux.alidf
    flux.alvdr_ai[...] = flux.alvdr
    flux.alidr_ai[...] = flux.alidr

    # for history averaging
    # echmod?            cszn = c0
    # echmod            if (coszen(i,j,iblk) > puny) cszn = c1
    # echmod            do n = 1, nstreams
    # echmod               albcnt(i,j,iblk,n) = albcnt(i,j,iblk,n) + cszn
    # echmod            enddo

    #----------------------------------------------------------------
    # Save net shortwave for scaling factor in scale_factor
    #----------------------------------------------------------------
    runtype = str(ice_init.runtype).strip()
    if runtype == "initial" and not bool(ice_init.restart):
        flux.scale_factor[...] = (
            flux.swvdr * (c1 - flux.alvdr_ai)
            + flux.swvdf * (c1 - flux.alvdf_ai)
            + flux.swidr * (c1 - flux.alidr_ai)
            + flux.swidf * (c1 - flux.alidf_ai)
        )


def _tracer_block(trcrn, i, j, nt, count, ncat):
    """Category slice of ``count`` tracers starting at 0-based index ``nt``.

    A zero count or an unset index (-1) returns an empty ``(0, ncat)`` array.
    """
    if count < 1 or nt < 0:
        return np.zeros((0, ncat), dtype=dbl_kind)
    return trcrn[i, j, nt:nt + count, :]


# Namelist parameters needed locally. input_zbgc fills this from zbgc_nml.
# grid_oS and l_skS are deprecated with zsalinity and have no Fortran default.
zbgc_state = {
    "tr_brine": False,
    "tr_zaero": False,
    "modal_aero": False,
    "restore_bgc": False,
    "solve_zsal": False,
    "restart_bgc": False,
    "restart_zsal": False,
    "restart_hbrine": False,
    "scale_bgc": False,
    "skl_bgc": False,
    "z_tracers": False,
    "dEdd_algae": False,
    "solve_zbgc": False,
    "tr_bgc_PON": False,
    "tr_bgc_Nit": False,
    "tr_bgc_C": False,
    "tr_bgc_chl": False,
    "tr_bgc_Sil": False,
    "tr_bgc_Am": False,
    "tr_bgc_DMS": False,
    "tr_bgc_DON": False,
    "tr_bgc_hum": False,
    "tr_bgc_Fe": False,
    "tr_bgc_N": True,
    "phi_snow": -1.0,
    "bgc_flux_type": "Jin2006",
    "grid_o": 0.006,
    "grid_o_t": 0.006,
    "l_sk": 2.0,
    "initbio_frac": 1.0,
    "frazil_scav": 0.8,
    "ratio_Si2N_diatoms": 1.8,
    "ratio_Si2N_sp": 0.0,
    "ratio_Si2N_phaeo": 0.0,
    "ratio_S2N_diatoms": 0.03,
    "ratio_S2N_sp": 0.03,
    "ratio_S2N_phaeo": 0.03,
    "ratio_Fe2C_diatoms": 0.0033,
    "ratio_Fe2C_sp": 0.0033,
    "ratio_Fe2C_phaeo": 0.1,
    "ratio_Fe2N_diatoms": 0.023,
    "ratio_Fe2N_sp": 0.023,
    "ratio_Fe2N_phaeo": 0.7,
    "ratio_Fe2DON": 0.023,
    "ratio_Fe2DOC_s": 0.1,
    "ratio_Fe2DOC_l": 0.033,
    "fr_resp": 0.05,
    "tau_min": 3600.0,
    "tau_max": 604800.0,
    "algal_vel": 1.0e-7,
    "R_dFe2dust": 0.035,
    "dustFe_sol": 0.005,
    "chlabs_diatoms": 0.03,
    "chlabs_sp": 0.01,
    "chlabs_phaeo": 0.05,
    "alpha2max_low_diatoms": 0.30,
    "alpha2max_low_sp": 0.20,
    "alpha2max_low_phaeo": 0.17,
    "beta2max_diatoms": 0.001,
    "beta2max_sp": 0.001,
    "beta2max_phaeo": 0.04,
    "mu_max_diatoms": 1.44,
    "mu_max_sp": 0.41,
    "mu_max_phaeo": 0.63,
    "grow_Tdep_diatoms": 0.063,
    "grow_Tdep_sp": 0.063,
    "grow_Tdep_phaeo": 0.063,
    "fr_graze_diatoms": 0.19,
    "fr_graze_sp": 0.19,
    "fr_graze_phaeo": 0.19,
    "mort_pre_diatoms": 0.007,
    "mort_pre_sp": 0.007,
    "mort_pre_phaeo": 0.007,
    "mort_Tdep_diatoms": 0.03,
    "mort_Tdep_sp": 0.03,
    "mort_Tdep_phaeo": 0.03,
    "k_exude_diatoms": 0.0,
    "k_exude_sp": 0.0,
    "k_exude_phaeo": 0.0,
    "K_Nit_diatoms": 1.0,
    "K_Nit_sp": 1.0,
    "K_Nit_phaeo": 1.0,
    "K_Am_diatoms": 0.3,
    "K_Am_sp": 0.3,
    "K_Am_phaeo": 0.3,
    "K_Sil_diatoms": 4.0,
    "K_Sil_sp": 0.0,
    "K_Sil_phaeo": 0.0,
    "K_Fe_diatoms": 1.0,
    "K_Fe_sp": 0.2,
    "K_Fe_phaeo": 0.1,
    "f_don_protein": 0.6,
    "kn_bac_protein": 0.2,
    "f_don_Am_protein": 1.0,
    "f_doc_s": 0.5,
    "f_doc_l": 0.5,
    "f_exude_s": 1.0,
    "f_exude_l": 1.0,
    "k_bac_s": 0.03,
    "k_bac_l": 0.03,
    "T_max": 0.0,
    "fsal": 1.0,
    "op_dep_min": 0.1,
    "fr_graze_s": 0.5,
    "fr_graze_e": 0.5,
    "fr_mort2min": 0.9,
    "fr_dFe": 1.0,
    "k_nitrif": 0.046,
    "t_iron_conv": 3065.0,
    "max_loss": 0.9,
    "max_dfe_doc1": 0.2,
    "fr_resp_s": 0.9,
    "y_sk_DMS": 0.7,
    "t_sk_conv": 5.0,
    "t_sk_ox": 12.0,
    "algaltype_diatoms": 0.0,
    "algaltype_sp": 0.0,
    "algaltype_phaeo": 0.0,
    "nitratetype": -1.0,
    "ammoniumtype": 0.0,
    "silicatetype": -1.0,
    "dmspptype": 0.5,
    "dmspdtype": 0.0,
    "humtype": 0.0,
    "dictype_1": -1.0,
    "doctype_s": 0.0,
    "doctype_l": 0.0,
    "dontype_protein": 0.0,
    "fedtype_1": 0.0,
    "feptype_1": 0.5,
    "zaerotype_bc1": -1.0,
    "zaerotype_bc2": -1.0,
    "zaerotype_dust1": -1.0,
    "zaerotype_dust2": -1.0,
    "zaerotype_dust3": -1.0,
    "zaerotype_dust4": -1.0,
    "ratio_C2N_diatoms": 7.0,
    "ratio_C2N_sp": 7.0,
    "ratio_C2N_phaeo": 7.0,
    "ratio_chl2N_diatoms": 2.1,
    "ratio_chl2N_sp": 1.1,
    "ratio_chl2N_phaeo": 0.84,
    "F_abs_chl_diatoms": 2.0,
    "F_abs_chl_sp": 4.0,
    "F_abs_chl_phaeo": 5.0,
    "ratio_C2N_proteins": 5.0,
}

# Copied by icepack_init_zbgc. Zero until init_zbgc; Fortran leaves them undefined.
zbgc_frac_init = None
bgc_tracer_type = None
zbgc_init_frac = None
tau_ret = None
tau_rel = None
R_S2N = None


def _parse_zbgc_value(token, prototype):
    token = token.strip()
    if isinstance(prototype, bool):
        lowered = token.lower()
        if lowered in (".true.", "true"):
            return True
        if lowered in (".false.", "false"):
            return False
        raise ValueError(f"expected a Fortran logical, got {token}")
    if isinstance(prototype, str):
        if len(token) >= 2 and token[0] == token[-1] and token[0] in "'\"":
            return token[1:-1]
        return token
    text = re.sub(r"([0-9.])d([+-]?\d+)", r"\1e\2", token.strip().lower())
    return float(text)


def _read_zbgc_nml(nml_path):
    """Read ``zbgc_nml`` from the same ice_in file as ``input_data``."""
    subname = "(input_zbgc)"
    with open(nml_path, encoding="utf-8") as handle:
        lines = handle.readlines()
    chunks = []
    current = None
    for line in lines:
        out = []
        quote = None
        for char in line:
            if quote:
                out.append(char)
                if char == quote:
                    quote = None
                continue
            if char in "'\"":
                quote = char
                out.append(char)
                continue
            if char == "!":
                break
            out.append(char)
        text = "".join(out).strip()
        if not text:
            continue
        if text.startswith("&"):
            current = text[1:].split()[0]
            continue
        if text == "/":
            current = None
            continue
        if current == "zbgc_nml":
            chunks.append(text)
    if not chunks:
        raise RuntimeError(subname + "ERROR: zbgc_nml reading ")
    assignments = []
    pending = None
    for text in chunks:
        if "=" in text:
            if pending is not None:
                assignments.append(pending)
            pending = text
        else:
            if pending is None:
                raise ValueError(f"zbgc_nml: cannot parse {text}")
            pending = pending + " " + text
    if pending is not None:
        assignments.append(pending)
    state = dict(zbgc_state)
    for item in assignments:
        key, raw = item.split("=", 1)
        match = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)(?:\((\d+)\))?", key.strip())
        if match is None:
            raise ValueError(f"zbgc_nml: cannot parse {key}")
        field = match.group(1)
        if field not in state and field not in ("grid_oS", "l_skS"):
            raise ValueError(f"zbgc_nml: unknown {field}")
        prototype = state[field] if field in state else 0.0
        state[field] = _parse_zbgc_value(raw.strip().rstrip(","), prototype)
    return state


def input_zbgc(nml_path="configuration/ice_in"):
    """Namelist variables, set to default values; may be altered at run time

    author Elizabeth C. Hunke, LANL
           Nicole Jeffery, LANL
    """
    global zbgc_state
    subname = "(input_zbgc)"
    nu = ice_fileunits.nu_diag
    abort_flag = 0
    tr_aero = bool(icepack_tracers.tr_aero)
    ktherm = int(icepack_parameters.ktherm)
    shortwave = str(icepack_parameters.shortwave).strip()

    nu.write(f" {subname}  Reading zbgc_nml\n")
    try:
        state = _read_zbgc_nml(nml_path)
    except OSError as exc:
        raise RuntimeError(subname + "ERROR: zbgc_nml open file " + nml_path) from exc
    zbgc_state = state

    #-----------------------------------------------------------------
    # zsalinity and brine
    #-----------------------------------------------------------------
    if not bool(ice_init.restart):
        nu.write(f" {subname} WARNING: restart = false, setting bgc restart flags to false\n")
        state["restart_bgc"] = False
        state["restart_hbrine"] = False

    if state["solve_zsal"] or state["restart_zsal"]:
        nu.write(f" {subname} ERROR: solve_zsal=T, restart_zsal=T deprecated\n")
        abort_flag = 101
    nblyr = int(icepack_tracers.nblyr)
    if state["tr_brine"] and nblyr < 1:
        nu.write(f" {subname} ERROR: tr_brine=T but no biology layers compiled\n")
        abort_flag = 103

    #-----------------------------------------------------------------
    # biogeochemistry
    #-----------------------------------------------------------------
    # deprecate skl bgc (Aug 2024)
    # no skl code removed yet
    if state["skl_bgc"]:
        nu.write(" ERROR: skl_bgc is not validate and temporarily DEPRECATED\n")
        nu.write(" ERROR: if you would like to use skl_bgc, please contact the Consortium\n")
        abort_flag = 102
    if not state["tr_brine"]:
        if state["solve_zbgc"]:
            nu.write(f" {subname} ERROR: tr_brine = F and solve_zbgc = T\n")
            abort_flag = 104
        if state["tr_zaero"]:
            nu.write(f" {subname} ERROR: tr_brine = F and tr_zaero = T\n")
            abort_flag = 105
    if (state["skl_bgc"] and state["solve_zbgc"]) or (
        state["skl_bgc"] and state["z_tracers"]
    ):
        nu.write(f" {subname} ERROR: skl_bgc and solve_zbgc or z_tracers are both true\n")
        abort_flag = 106
    if state["skl_bgc"] and state["tr_zaero"]:
        nu.write(f" {subname} ERROR: skl_bgc does not use vertical tracers\n")
        abort_flag = 107
    if state["dEdd_algae"] and shortwave[:4] != "dEdd":
        nu.write(f" {subname} ERROR: dEdd_algae = T but shortwave /= dEdd or dEdd_snicar_ad\n")
        abort_flag = 108
    if state["dEdd_algae"] and (not state["tr_bgc_N"]) and (not state["tr_zaero"]):
        nu.write(f" {subname} ERROR: need tr_bgc_N or tr_zaero for dEdd_algae\n")
        abort_flag = 109
    if state["modal_aero"] and (not state["tr_zaero"]) and (not tr_aero):
        nu.write(f" {subname} ERROR: modal_aero T with tr_zaero and tr_aero\n")
        abort_flag = 110
    if state["modal_aero"] and shortwave[:4] != "dEdd":
        nu.write(f" {subname} ERROR: modal_aero = T but shortwave /= dEdd or dEdd_snicar_ad\n")
        abort_flag = 111

    def _too_many(count, limit, text, flag):
        nonlocal abort_flag
        if int(count) > int(limit):
            nu.write(f" {subname}{text}\n")
            abort_flag = flag

    _too_many(icepack_tracers.n_algae, icepack_tracers.max_algae,
              "ERROR: number of algal types exceeds icepack_max_algae", 112)
    _too_many(icepack_tracers.n_doc, icepack_tracers.max_doc,
              "ERROR: number of doc types exceeds icepack_max_doc", 113)
    # Fortran compares n_dic with icepack_max_doc and reports max_dic.
    _too_many(icepack_tracers.n_dic, icepack_tracers.max_doc,
              "ERROR: number of dic types exceeds icepack_max_dic", 114)
    _too_many(icepack_tracers.n_don, icepack_tracers.max_don,
              "ERROR: number of don types exceeds icepack_max_don", 115)
    _too_many(icepack_tracers.n_fed, icepack_tracers.max_fe,
              "ERROR: number of dissolved fe types exceeds icepack_max_fe ", 116)
    _too_many(icepack_tracers.n_fep, icepack_tracers.max_fe,
              "ERROR: number of particulate fe types exceeds icepack_max_fe ", 117)
    if int(icepack_tracers.n_algae) == 0 and state["skl_bgc"]:
        nu.write(f" {subname}ERROR: skl_bgc=T but 0 bgc or algal tracers compiled\n")
        abort_flag = 118
    if int(icepack_tracers.n_algae) == 0 and state["solve_zbgc"]:
        nu.write(f" {subname}ERROR: solve_zbgc=T but 0 zbgc or algal tracers compiled\n")
        abort_flag = 119
    if state["solve_zbgc"] and not state["z_tracers"]:
        nu.write(f" {subname}ERROR: solve_zbgc=T but not z_tracers\n")
        abort_flag = 120
    if state["skl_bgc"] or state["solve_zbgc"]:
        if not state["tr_bgc_N"]:
            nu.write(f" {subname}ERROR: tr_bgc_N must be on for bgc\n")
            abort_flag = 121
        if not state["tr_bgc_Nit"]:
            nu.write(f" {subname}ERROR: tr_bgc_Nit must be on for bgc\n")
            abort_flag = 122
    else:
        # tcraig, allow bgc to be turned off in this case?
        for name in (
            "tr_bgc_N", "tr_bgc_C", "tr_bgc_chl", "tr_bgc_Nit", "tr_bgc_Am",
            "tr_bgc_Sil", "tr_bgc_hum", "tr_bgc_DMS", "tr_bgc_PON",
            "tr_bgc_DON", "tr_bgc_Fe",
        ):
            state[name] = False

    #-----------------------------------------------------------------
    # z layer aerosols
    #-----------------------------------------------------------------
    if state["tr_zaero"] and not state["z_tracers"]:
        nu.write(f" {subname}ERROR: tr_zaero and not z_tracers\n")
        abort_flag = 123
    if int(icepack_tracers.n_zaero) > int(icepack_tracers.max_aero):
        nu.write(f" {subname}ERROR: number of z aerosols exceeds icepack_max_aero\n")
        abort_flag = 124

    #-----------------------------------------------------------------
    # output
    #-----------------------------------------------------------------
    def _log(label, value):
        nu.write(f" {label:<30}  {'T' if value else 'F':>6}\n")

    def _flt(label, value, width="9.6"):
        nu.write(f" {label:<30}  {float(value):{width}}\n")

    def _int(label, value):
        nu.write(f" {label:<30}  {int(value):6d}\n")

    _log("tr_brine                  =", state["tr_brine"])
    if state["tr_brine"]:
        _log("restart_hbrine            =", state["restart_hbrine"])
        _flt("phi_snow                  =", state["phi_snow"])
    _log("skl_bgc                   =", state["skl_bgc"])
    _log("restart_bgc               =", state["restart_bgc"])
    _log("tr_bgc_N                  =", state["tr_bgc_N"])
    _log("tr_bgc_C                  =", state["tr_bgc_C"])
    _log("tr_bgc_chl                =", state["tr_bgc_chl"])
    _log("tr_bgc_Nit                =", state["tr_bgc_Nit"])
    _log("tr_bgc_Am                 =", state["tr_bgc_Am"])
    _log("tr_bgc_Sil                =", state["tr_bgc_Sil"])
    _log("tr_bgc_hum                =", state["tr_bgc_hum"])
    _log("tr_bgc_DMS                =", state["tr_bgc_DMS"])
    _log("tr_bgc_PON                =", state["tr_bgc_PON"])
    _log("tr_bgc_DON                =", state["tr_bgc_DON"])
    _log("tr_bgc_Fe                 =", state["tr_bgc_Fe"])
    _int("n_aero                    =", icepack_tracers.n_aero)
    _int("n_zaero                   =", icepack_tracers.n_zaero)
    _int("n_algae                   =", icepack_tracers.n_algae)
    _int("n_doc                     =", icepack_tracers.n_doc)
    _int("n_dic                     =", icepack_tracers.n_dic)
    _int("n_don                     =", icepack_tracers.n_don)
    _int("n_fed                     =", icepack_tracers.n_fed)
    _int("n_fep                     =", icepack_tracers.n_fep)
    if state["skl_bgc"]:
        nu.write(f" {'bgc_flux_type             =':<30}{state['bgc_flux_type']}\n")
        _log("restore_bgc               =", state["restore_bgc"])
    elif state["z_tracers"]:
        _log("dEdd_algae                =", state["dEdd_algae"])
        _log("modal_aero                =", state["modal_aero"])
        _log("scale_bgc                 =", state["scale_bgc"])
        _log("solve_zbgc                =", state["solve_zbgc"])
        _log("tr_zaero                  =", state["tr_zaero"])
        _int("number of aerosols        =", icepack_tracers.n_zaero)
        _flt("grid_o                    =", state["grid_o"], "9.2")
        _flt("grid_o_t                  =", state["grid_o_t"], "9.2")
        _flt("l_sk                      =", state["l_sk"])
        _flt("initbio_frac              =", state["initbio_frac"], "9.2")
        _flt("frazil_scav               =", state["frazil_scav"], "9.2")

    #-----------------------------------------------------------------
    # abort if abort flag is set
    #-----------------------------------------------------------------
    if abort_flag != 0:
        nu.write(f" {subname} ERROR: abort_flag= {abort_flag}\n")
        raise RuntimeError(subname + " ABORTING on input ERRORS")

    #-----------------------------------------------------------------
    # set values in icepack
    #-----------------------------------------------------------------
    icepack_parameters.icepack_init_parameters(
        ktherm=ktherm, shortwave=shortwave, scale_bgc=state["scale_bgc"],
        skl_bgc=state["skl_bgc"], z_tracers=state["z_tracers"],
        dEdd_algae=state["dEdd_algae"], solve_zbgc=state["solve_zbgc"],
        bgc_flux_type=state["bgc_flux_type"], grid_o=state["grid_o"],
        l_sk=state["l_sk"], initbio_frac=state["initbio_frac"],
        frazil_scav=state["frazil_scav"], phi_snow=state["phi_snow"],
        algal_vel=state["algal_vel"], R_dFe2dust=state["R_dFe2dust"],
        dustFe_sol=state["dustFe_sol"], T_max=state["T_max"], fsal=state["fsal"],
        op_dep_min=state["op_dep_min"], fr_graze_s=state["fr_graze_s"],
        fr_graze_e=state["fr_graze_e"], fr_mort2min=state["fr_mort2min"],
        fr_dFe=state["fr_dFe"], k_nitrif=state["k_nitrif"],
        t_iron_conv=state["t_iron_conv"], max_loss=state["max_loss"],
        max_dfe_doc1=state["max_dfe_doc1"], fr_resp=state["fr_resp"],
        fr_resp_s=state["fr_resp_s"], y_sk_DMS=state["y_sk_DMS"],
        t_sk_conv=state["t_sk_conv"], t_sk_ox=state["t_sk_ox"],
        modal_aero=state["modal_aero"],
    )
    passed = (
        "ratio_Si2N_diatoms", "ratio_Si2N_sp", "ratio_Si2N_phaeo",
        "ratio_S2N_diatoms", "ratio_S2N_sp", "ratio_S2N_phaeo",
        "ratio_Fe2C_diatoms", "ratio_Fe2C_sp", "ratio_Fe2C_phaeo",
        "ratio_Fe2N_diatoms", "ratio_Fe2N_sp", "ratio_Fe2N_phaeo",
        "ratio_C2N_diatoms", "ratio_C2N_sp", "ratio_C2N_phaeo",
        "ratio_chl2N_diatoms", "ratio_chl2N_sp", "ratio_chl2N_phaeo",
        "F_abs_chl_diatoms", "F_abs_chl_sp", "F_abs_chl_phaeo",
        "ratio_Fe2DON", "ratio_C2N_proteins", "ratio_Fe2DOC_s", "ratio_Fe2DOC_l",
        "chlabs_diatoms", "chlabs_sp", "chlabs_phaeo",
        "alpha2max_low_diatoms", "alpha2max_low_sp", "alpha2max_low_phaeo",
        "beta2max_diatoms", "beta2max_sp", "beta2max_phaeo",
        "mu_max_diatoms", "mu_max_sp", "mu_max_phaeo",
        "grow_Tdep_diatoms", "grow_Tdep_sp", "grow_Tdep_phaeo",
        "fr_graze_diatoms", "fr_graze_sp", "fr_graze_phaeo",
        "mort_pre_diatoms", "mort_pre_sp", "mort_pre_phaeo",
        "mort_Tdep_diatoms", "mort_Tdep_sp", "mort_Tdep_phaeo",
        "k_exude_diatoms", "k_exude_sp", "k_exude_phaeo",
        "K_Nit_diatoms", "K_Nit_sp", "K_Nit_phaeo",
        "K_Am_diatoms", "K_Am_sp", "K_Am_phaeo",
        "K_Sil_diatoms", "K_Sil_sp", "K_Sil_phaeo",
        "K_Fe_diatoms", "K_Fe_sp", "K_Fe_phaeo",
        "f_doc_s", "f_doc_l", "f_don_protein", "kn_bac_protein", "f_don_Am_protein",
        "f_exude_s", "f_exude_l", "k_bac_s", "k_bac_l",
        "algaltype_diatoms", "algaltype_sp", "algaltype_phaeo",
        "dictype_1", "doctype_s", "doctype_l", "dontype_protein",
        "fedtype_1", "feptype_1",
        "zaerotype_bc1", "zaerotype_bc2",
        "zaerotype_dust1", "zaerotype_dust2", "zaerotype_dust3", "zaerotype_dust4",
    )
    icepack_parameters.icepack_init_parameters(**{name: state[name] for name in passed})
    # tr_brine goes through the flag setter. BGC flags are not accepted there.
    icepack_tracers.icepack_init_tracer_flags(tr_brine=state["tr_brine"])
    for name in (
        "tr_zaero", "tr_bgc_Nit", "tr_bgc_N", "tr_bgc_DON", "tr_bgc_C", "tr_bgc_chl",
        "tr_bgc_Am", "tr_bgc_Sil", "tr_bgc_DMS", "tr_bgc_Fe", "tr_bgc_PON", "tr_bgc_hum",
    ):
        setattr(icepack_tracers, name, log_kind(state[name]))

    for name, value in state.items():
        setattr(ice_init, name, value)
    ice_restart_column.restart_bgc = bool(state["restart_bgc"])
    ice_restart_column.restart_hbrine = bool(state["restart_hbrine"])
    ice_init.restore_bgc = bool(state["restore_bgc"])


def init_hbrine():
    """Initialize brine height tracer."""
    subname = "(init_hbrine)"
    phi_snow = dbl_kind(icepack_parameters.phi_snow)
    # if (phi_snow .le. c0) phi_snow = c1-rhos/rhoi
    if phi_snow <= c0:
        phi_snow = c1 - rhos / rhoi
    nblyr = int(icepack_tracers.nblyr)
    nilyr = int(icepack_tracers.nilyr)
    bgrid = ice_arrays_column.bgrid
    igrid = ice_arrays_column.igrid
    cgrid = ice_arrays_column.cgrid
    icgrid = ice_arrays_column.icgrid
    swgrid = ice_arrays_column.swgrid
    if bgrid is None or len(bgrid) != nblyr + 2:
        raise RuntimeError(subname + " bgrid is not allocated")

    #-----------------------------------------------------------------
    # Calculate bio gridn: 0 to 1 corresponds to ice top to bottom
    #-----------------------------------------------------------------
    bgrid[...] = c0  # biology nondimensional vertical grid points
    bgrid[nblyr + 1] = c1  # bottom value, Fortran bgrid(nblyr+2)
    igrid[...] = c0  # bgc interface grid points
    igrid[0] = c0    # ice top
    igrid[nblyr] = c1  # ice bottom
    zspace = c1 / max(c1, dbl_kind(nblyr))
    for k in range(2, nblyr + 2):
        bgrid[k - 1] = zspace * (dbl_kind(k) - c1p5)
    for k in range(2, nblyr + 1):
        igrid[k - 1] = p5 * (bgrid[k] + bgrid[k - 1])

    #-----------------------------------------------------------------
    # Calculate CICE cgrid for interpolation ice top (0) to bottom (1)
    #-----------------------------------------------------------------
    cgrid[0] = c0                           # CICE vertical grid top point
    zspace = c1 / dbl_kind(nilyr)           # CICE grid spacing
    for k in range(2, nilyr + 2):
        cgrid[k - 1] = zspace * (dbl_kind(k) - c1p5)

    #-----------------------------------------------------------------
    # Calculate CICE icgrid for ishortwave interpolation top(0) , bottom (1)
    #-----------------------------------------------------------------
    icgrid[0] = c0
    for k in range(2, nilyr + 2):
        icgrid[k - 1] = zspace * (dbl_kind(k) - c1)

    #------------------------------------------------------------------------
    # Calculate CICE swgrid for dEdd ice: top of ice (0) , bottom of ice (1)
    # Does not include snow
    # see icepack_shortwave.F90
    # swgrid represents the layer index of the delta-eddington ice layer index
    #------------------------------------------------------------------------
    swgrid[0] = min(c1 / dbl_kind(60.0), zspace * p25)  # p5 to p25. NJ: allows thinner surface layers
    swgrid[1] = zspace / c2                   # + swgrid(1)
    for k in range(3, nilyr + 2):
        swgrid[k - 1] = zspace * (dbl_kind(k) - c1p5)

    icepack_parameters.icepack_init_parameters(phi_snow=phi_snow)
    ice_arrays_column.first_ice[...] = True
    if bool(icepack_tracers.tr_brine):
        nt_fbri = int(icepack_tracers.nt_fbri)
        ice_state.trcrn[:, :, nt_fbri, :] = c1


def init_bgc_trcr(
    nk, nt_fbri, nt_bgc, nlt_bgc, bgctype, nt_depend,
    bgc_tracer_type, trcr_depend, trcr_base, n_trcr_strata, nt_strata, bio_index,
):
    """Assign dependencies for one biogeochemistry tracer.

    ``nt_bgc`` is a 0-based tracer index. ``nlt_bgc`` stays 1-based.
    A dependency above 2 is ``3 +`` the parent's 0-based index.
    Unused ``nt_strata`` entries are -1.
    """
    subname = "(init_bgc_trcr)"
    bgc_tracer_type[nlt_bgc - 1] = bgctype
    ice_fileunits.nu_diag.write(
        f" {subname} bgc_tracer_type {nlt_bgc} {float(bgc_tracer_type[nlt_bgc - 1])}\n"
    )

    if nk > 1:  # include vertical bgc in snow
        for k in range(nk, nk + 2):
            # Fortran nt_bgc + k
            slot = nt_bgc + k
            trcr_depend[slot] = 2  # snow volume
            trcr_base[slot, 0] = c0
            trcr_base[slot, 1] = c0
            trcr_base[slot, 2] = c1
            n_trcr_strata[slot] = 0
            nt_strata[slot, 0] = -1
            nt_strata[slot, 1] = -1
        trcr_base1 = c0
        trcr_base2 = c1
        trcr_base3 = c0
        n_strata = 1
        nt_strata1 = nt_fbri
        nt_strata2 = -1
    else:  # nk = 1
        trcr_base1 = c1
        trcr_base2 = c0
        trcr_base3 = c0
        n_strata = 0
        nt_strata1 = -1
        nt_strata2 = -1

    for k in range(1, nk + 1):  # in ice. Fortran nt_bgc + k - 1
        slot = nt_bgc + k - 1
        trcr_depend[slot] = nt_depend
        trcr_base[slot, 0] = trcr_base1
        trcr_base[slot, 1] = trcr_base2
        trcr_base[slot, 2] = trcr_base3
        n_trcr_strata[slot] = n_strata
        nt_strata[slot, 0] = nt_strata1
        nt_strata[slot, 1] = nt_strata2

    bio_index[nlt_bgc - 1] = nt_bgc


def init_zbgc():
    """Initialize vertical biogeochemistry

    author Elizabeth C. Hunke, LANL
           Nicole Jeffery, LANL
    """
    global zbgc_frac_init, bgc_tracer_type, zbgc_init_frac, tau_ret, tau_rel, R_S2N

    subname = "(init_zbgc)"
    #------------------------------------------------------------
    #        Tracers have mobile and stationary phases.
    # ice growth allows for retention, ice melt facilitates mobility
    # bgc_tracer_type defines the exchange timescales between these phases
    # -1 : entirely in the mobile phase, no exchange  (this is the default)
    #  0 : retention time scale is tau_min, release time scale is tau_max
    #  1 : retention time scale is tau_max, release time scale is tau_min
    # 0.5: retention time scale is tau_min, release time scale is tau_min
    #  2 : retention time scale is tau_max, release time scale is tau_max
    # tau_min and tau_max are defined in icepack_intfc.f90
    #------------------------------------------------------------
    state = zbgc_state
    skl_bgc = bool(icepack_parameters.skl_bgc)
    z_tracers = bool(icepack_parameters.z_tracers)
    dEdd_algae = bool(icepack_parameters.dEdd_algae)
    initbio_frac = dbl_kind(icepack_parameters.initbio_frac)
    frazil_scav = dbl_kind(icepack_parameters.frazil_scav)
    nbtrcr = int(icepack_tracers.nbtrcr)
    nbtrcr_sw = int(icepack_tracers.nbtrcr_sw)
    nt_fbri = int(icepack_tracers.nt_fbri)
    max_nbtrcr = int(icepack_tracers.max_nbtrcr)
    max_algae = int(icepack_tracers.max_algae)
    max_doc = int(icepack_tracers.max_doc)
    max_dic = int(icepack_tracers.max_dic)
    max_don = int(icepack_tracers.max_don)
    max_fe = int(icepack_tracers.max_fe)
    max_aero = int(icepack_tracers.max_aero)
    nilyr = int(icepack_tracers.nilyr)
    nslyr = int(icepack_tracers.nslyr)
    nblyr = int(icepack_tracers.nblyr)
    ncat = int(icepack_tracers.ncat)

    # Zeroed here. Fortran allocate leaves these undefined when bgc is off.
    count = max_nbtrcr
    zbgc_frac_init = np.zeros(count, dtype=dbl_kind)
    bgc_tracer_type = np.zeros(count, dtype=dbl_kind)
    zbgc_init_frac = np.zeros(count, dtype=dbl_kind)
    tau_ret = np.zeros(count, dtype=dbl_kind)
    tau_rel = np.zeros(count, dtype=dbl_kind)
    R_C2N = np.zeros(max_algae, dtype=dbl_kind)
    R_chl2N = np.zeros(max_algae, dtype=dbl_kind)
    ice_arrays_column.R_C2N = R_C2N
    ice_arrays_column.R_chl2N = R_chl2N
    R_C2N[0] = dbl_kind(state["ratio_C2N_diatoms"])
    R_C2N[1] = dbl_kind(state["ratio_C2N_sp"])
    R_C2N[2] = dbl_kind(state["ratio_C2N_phaeo"])
    R_chl2N[0] = dbl_kind(state["ratio_chl2N_diatoms"])
    R_chl2N[1] = dbl_kind(state["ratio_chl2N_sp"])
    R_chl2N[2] = dbl_kind(state["ratio_chl2N_phaeo"])
    R_S2N = np.array([
        state["ratio_S2N_diatoms"],
        state["ratio_S2N_sp"],
        state["ratio_S2N_phaeo"],
    ], dtype=dbl_kind)

    algaltype = np.zeros(max_algae, dtype=dbl_kind)
    algaltype[0] = dbl_kind(state["algaltype_diatoms"])
    algaltype[1] = dbl_kind(state["algaltype_sp"])
    algaltype[2] = dbl_kind(state["algaltype_phaeo"])
    dictype = np.full(max_dic, -c1, dtype=dbl_kind)  # not copied from dictype_1
    doctype = np.zeros(max_doc, dtype=dbl_kind)
    doctype[0] = dbl_kind(state["doctype_s"])
    if max_doc > 1:
        doctype[1] = dbl_kind(state["doctype_l"])
    dontype = np.zeros(max_don, dtype=dbl_kind)
    dontype[0] = dbl_kind(state["dontype_protein"])
    fedtype = np.zeros(max_fe, dtype=dbl_kind)
    feptype = np.zeros(max_fe, dtype=dbl_kind)
    fedtype[0] = dbl_kind(state["fedtype_1"])
    feptype[0] = dbl_kind(state["feptype_1"])
    zaerotype = np.zeros(max_aero, dtype=dbl_kind)
    for index, name in enumerate((
        "zaerotype_bc1", "zaerotype_bc2", "zaerotype_dust1",
        "zaerotype_dust2", "zaerotype_dust3", "zaerotype_dust4",
    )):
        zaerotype[index] = dbl_kind(state[name])

    #-----------------------------------------------------------------
    # assign tracer dependencies
    # bgc_tracer_type: < 0  purely mobile , >= 0 stationary
    #------------------------------------------------------------------
    trcr_depend = ice_state.trcr_depend
    trcr_base = ice_state.trcr_base
    n_trcr_strata = ice_state.n_trcr_strata
    nt_strata = ice_state.nt_strata
    if bool(icepack_tracers.tr_brine):
        slot = nt_fbri
        trcr_depend[slot] = 1   # volume-weighted
        trcr_base[slot, 0] = c0
        trcr_base[slot, 1] = c1
        trcr_base[slot, 2] = c0
        n_trcr_strata[slot] = 0
        nt_strata[slot, 0] = -1
        nt_strata[slot, 1] = -1

    ntd = 0                    # if nt_fbri /= 0 then use fbri dependency
    # Fortran tests nt_fbri == 0. After numbering from 0, that value is Tsfc.
    # -1 is the unset marker. count_tracers points a closed brine tracer at
    # the unused slot, so this stays 0 once counting has finished.
    if nt_fbri < 0:
        ntd = -1               # otherwise make tracers depend on ice volume
    bio_index = np.zeros(max_nbtrcr, dtype=np.int32)
    bio_index_o = np.zeros(max_nbtrcr, dtype=np.int32)

    def _one(nt_bgc, nlt_bgc, bgctype):
        init_bgc_trcr(
            nk, nt_fbri, int(nt_bgc), int(nlt_bgc), dbl_kind(bgctype), nt_depend,
            bgc_tracer_type, trcr_depend, trcr_base, n_trcr_strata, nt_strata, bio_index,
        )

    if skl_bgc or z_tracers:
        if skl_bgc:
            nk = 1
            nt_depend = 0
        elif z_tracers:  # defined on nblyr+1 in ice
            # and 2 snow layers (snow surface + interior)
            nk = nblyr + 1
            nt_depend = 3 + nt_fbri + ntd
        if bool(icepack_tracers.tr_bgc_N):
            for mm in range(int(icepack_tracers.n_algae)):
                _one(icepack_tracers.nt_bgc_N[mm], icepack_tracers.nlt_bgc_N[mm], algaltype[mm])
                bio_index_o[int(icepack_tracers.nlt_bgc_N[mm]) - 1] = mm + 1
        if bool(icepack_tracers.tr_bgc_Nit):
            _one(icepack_tracers.nt_bgc_Nit, icepack_tracers.nlt_bgc_Nit, state["nitratetype"])
            bio_index_o[int(icepack_tracers.nlt_bgc_Nit) - 1] = max_algae + 1
        if bool(icepack_tracers.tr_bgc_C):
            # Algal C is not yet distinct from algal N
            # * Reqires exudation and/or changing C:N ratios
            # for implementation
            for mm in range(int(icepack_tracers.n_doc)):
                _one(icepack_tracers.nt_bgc_DOC[mm], icepack_tracers.nlt_bgc_DOC[mm], doctype[mm])
                bio_index_o[int(icepack_tracers.nlt_bgc_DOC[mm]) - 1] = max_algae + 1 + mm + 1
            for mm in range(int(icepack_tracers.n_dic)):
                _one(icepack_tracers.nt_bgc_DIC[mm], icepack_tracers.nlt_bgc_DIC[mm], dictype[mm])
                bio_index_o[int(icepack_tracers.nlt_bgc_DIC[mm]) - 1] = (
                    max_algae + max_doc + 1 + mm + 1
                )
        if bool(icepack_tracers.tr_bgc_chl):
            for mm in range(int(icepack_tracers.n_algae)):
                _one(icepack_tracers.nt_bgc_chl[mm], icepack_tracers.nlt_bgc_chl[mm], algaltype[mm])
                bio_index_o[int(icepack_tracers.nlt_bgc_chl[mm]) - 1] = (
                    max_algae + 1 + max_doc + max_dic + mm + 1
                )
        if bool(icepack_tracers.tr_bgc_Am):
            _one(icepack_tracers.nt_bgc_Am, icepack_tracers.nlt_bgc_Am, state["ammoniumtype"])
            bio_index_o[int(icepack_tracers.nlt_bgc_Am) - 1] = (
                2 * max_algae + max_doc + max_dic + 2
            )
        if bool(icepack_tracers.tr_bgc_Sil):
            _one(icepack_tracers.nt_bgc_Sil, icepack_tracers.nlt_bgc_Sil, state["silicatetype"])
            bio_index_o[int(icepack_tracers.nlt_bgc_Sil) - 1] = (
                2 * max_algae + max_doc + max_dic + 3
            )
        if bool(icepack_tracers.tr_bgc_DMS):  # all together
            _one(icepack_tracers.nt_bgc_DMSPp, icepack_tracers.nlt_bgc_DMSPp, state["dmspptype"])
            bio_index_o[int(icepack_tracers.nlt_bgc_DMSPp) - 1] = (
                2 * max_algae + max_doc + max_dic + 4
            )
            _one(icepack_tracers.nt_bgc_DMSPd, icepack_tracers.nlt_bgc_DMSPd, state["dmspdtype"])
            bio_index_o[int(icepack_tracers.nlt_bgc_DMSPd) - 1] = (
                2 * max_algae + max_doc + max_dic + 5
            )
            _one(icepack_tracers.nt_bgc_DMS, icepack_tracers.nlt_bgc_DMS, state["dmspdtype"])
            bio_index_o[int(icepack_tracers.nlt_bgc_DMS) - 1] = (
                2 * max_algae + max_doc + max_dic + 6
            )
        if bool(icepack_tracers.tr_bgc_PON):
            _one(icepack_tracers.nt_bgc_PON, icepack_tracers.nlt_bgc_PON, state["nitratetype"])
            bio_index_o[int(icepack_tracers.nlt_bgc_PON) - 1] = (
                2 * max_algae + max_doc + max_dic + 7
            )
        if bool(icepack_tracers.tr_bgc_DON):
            for mm in range(int(icepack_tracers.n_don)):
                _one(icepack_tracers.nt_bgc_DON[mm], icepack_tracers.nlt_bgc_DON[mm], dontype[mm])
                bio_index_o[int(icepack_tracers.nlt_bgc_DON[mm]) - 1] = (
                    2 * max_algae + max_doc + max_dic + 7 + mm + 1
                )
        if bool(icepack_tracers.tr_bgc_Fe):
            for mm in range(int(icepack_tracers.n_fed)):
                _one(icepack_tracers.nt_bgc_Fed[mm], icepack_tracers.nlt_bgc_Fed[mm], fedtype[mm])
                bio_index_o[int(icepack_tracers.nlt_bgc_Fed[mm]) - 1] = (
                    2 * max_algae + max_doc + max_dic + max_don + 7 + mm + 1
                )
            for mm in range(int(icepack_tracers.n_fep)):
                _one(icepack_tracers.nt_bgc_Fep[mm], icepack_tracers.nlt_bgc_Fep[mm], feptype[mm])
                bio_index_o[int(icepack_tracers.nlt_bgc_Fep[mm]) - 1] = (
                    2 * max_algae + max_doc + max_dic + max_don + max_fe + 7 + mm + 1
                )
        if bool(icepack_tracers.tr_bgc_hum):
            _one(icepack_tracers.nt_bgc_hum, icepack_tracers.nlt_bgc_hum, state["humtype"])
            bio_index_o[int(icepack_tracers.nlt_bgc_hum) - 1] = (
                2 * max_algae + max_doc + 8 + max_dic + max_don + 2 * max_fe + max_aero
            )

    if skl_bgc:
        if dEdd_algae:
            icepack_tracers.nlt_chl_sw = int_kind(1)
            nbtrcr_sw = nilyr + nslyr + 2  # only the bottom layer will be nonzero
    elif z_tracers:  # defined on nblyr+1 in ice
        # and 2 snow layers (snow surface + interior)
        if bool(icepack_tracers.tr_bgc_N) and dEdd_algae:
            icepack_tracers.nlt_chl_sw = int_kind(1)
            nbtrcr_sw = nilyr + nslyr + 2

    if z_tracers:  # defined on nblyr+1 in ice
        # and 2 snow layers (snow surface + interior)
        nk = nblyr + 1
        nt_depend = 3 + nt_fbri + ntd
        if bool(icepack_tracers.tr_zaero):
            for mm in range(int(icepack_tracers.n_zaero)):
                if dEdd_algae:
                    icepack_tracers.nlt_zaero_sw[mm] = int_kind(nbtrcr_sw + 1)
                    nbtrcr_sw = nbtrcr_sw + nilyr + nslyr + 2
                _one(icepack_tracers.nt_zaero[mm], icepack_tracers.nlt_zaero[mm], zaerotype[mm])
                bio_index_o[int(icepack_tracers.nlt_zaero[mm]) - 1] = (
                    2 * max_algae + max_doc + max_dic + max_don + 2 * max_fe + 7 + mm + 1
                )
        if nbtrcr > 0:
            nt_zbgc_frac = int(icepack_tracers.nt_zbgc_frac)
            tau_min = dbl_kind(state["tau_min"])
            tau_max = dbl_kind(state["tau_max"])
            for k in range(1, nbtrcr + 1):
                zbgc_frac_init[k - 1] = c1
                slot = nt_zbgc_frac + k - 1
                trcr_depend[slot] = 3 + nt_fbri
                trcr_base[slot, 0] = c0
                trcr_base[slot, 1] = c1
                trcr_base[slot, 2] = c0
                n_trcr_strata[slot] = 1
                nt_strata[slot, 0] = nt_fbri
                nt_strata[slot, 1] = -1
                tau_ret[k - 1] = c1
                tau_rel[k - 1] = c1
                kind = bgc_tracer_type[k - 1]
                if kind >= c0 and kind < p5:
                    tau_ret[k - 1] = tau_min
                    tau_rel[k - 1] = tau_max
                    zbgc_frac_init[k - 1] = c1
                elif kind >= p5 and kind < c1:
                    tau_ret[k - 1] = tau_min
                    tau_rel[k - 1] = tau_min
                    zbgc_frac_init[k - 1] = c1
                elif kind >= c1 and kind < c2:
                    tau_ret[k - 1] = tau_max
                    tau_rel[k - 1] = tau_min
                    zbgc_frac_init[k - 1] = c1
                elif kind >= c2:
                    tau_ret[k - 1] = tau_max
                    tau_rel[k - 1] = tau_max
                    zbgc_frac_init[k - 1] = c1

    for k in range(nbtrcr):
        zbgc_init_frac[k] = frazil_scav
        if bgc_tracer_type[k] < c0:
            zbgc_init_frac[k] = initbio_frac

    # icepack_init_zbgc copies these arrays. The ratio tables it also fills
    # live with the column package; R_C2N, R_chl2N, and R_S2N are the ones
    # initialization reads back.
    icepack_tracers.bio_index[...] = bio_index
    icepack_tracers.bio_index_o[...] = bio_index_o

    #-----------------------------------------------------------------
    # final consistency checks
    #-----------------------------------------------------------------
    if nbtrcr > max_nbtrcr:
        nu = ice_fileunits.nu_diag
        nu.write(f" {subname} \n")
        nu.write(f" {subname} nbtrcr > icepack_max_nbtrcr\n")
        nu.write(f" {subname} nbtrcr, icepack_max_nbtrcr: {nbtrcr} {max_nbtrcr}\n")
        raise RuntimeError(subname + "ERROR: nbtrcr > icepack_max_nbtrcr")
    if not dEdd_algae:
        nbtrcr_sw = 1
    # tcraig, added 6/1/21, why is nbtrcr_sw set here?
    icepack_tracers.icepack_init_tracer_sizes(nbtrcr_sw=nbtrcr_sw)
    horiz = ice_domain_size.horizontal_shape()
    ice_arrays_column.trcrn_sw = np.zeros(horiz + (nbtrcr_sw, ncat), dtype=dbl_kind)

    nu = ice_fileunits.nu_diag
    if skl_bgc:
        nu.write(f" {'number of bio tracers     =':<30}  {nbtrcr:6d}\n")
        nu.write(f" {'number of Isw tracers     =':<30}  {nbtrcr_sw:6d}\n")
    elif z_tracers:
        nu.write(f" {'number of ztracers        =':<30}  {nbtrcr:6d}\n")
        nu.write(f" {'number of Isw tracers     =':<30}  {nbtrcr_sw:6d}\n")
        nu.write(f" {'initbio_frac              =':<30}  {float(initbio_frac):9.2f}\n")
        nu.write(f" {'frazil_scav               =':<30}  {float(frazil_scav):9.2f}\n")


def _init_ocean_point(i, j):
    """icepack_init_ocean_bio for one grid point. Optional arguments that
    are not passed stay unused.
    """
    ice_flux_bgc.amm[i, j] = c1
    ice_flux_bgc.dmsp[i, j] = p1
    ice_flux_bgc.dms[i, j] = p1
    ice_flux_bgc.algalN[i, j, :] = c0
    ice_flux_bgc.algalN[i, j, 0] = c1  # 0.0026_dbl_kind ! ISPOL, Lannuzel 2013(pennate)
    ice_flux_bgc.algalN[i, j, 1] = dbl_kind(0.0057)  # ISPOL, Lannuzel 2013(small plankton)
    ice_flux_bgc.algalN[i, j, 2] = dbl_kind(0.0027)  # ISPOL, Lannuzel 2013(Phaeocystis)
    ice_flux_bgc.doc[i, j, :] = c0
    ice_flux_bgc.doc[i, j, 0] = dbl_kind(16.2)  # 18% saccharides
    ice_flux_bgc.doc[i, j, 1] = dbl_kind(9.0)   # lipids
    ice_flux_bgc.doc[i, j, 2] = c1
    ice_flux_bgc.dic[i, j, :] = c0
    ice_flux_bgc.dic[i, j, 0] = dbl_kind(1950.0)  # 1950-2260 mmol C/m3 (Tynan et al. 2015)
    ice_flux_bgc.don[i, j, :] = c0
    ice_flux_bgc.don[i, j, 0] = dbl_kind(12.9)  # 64.3 ! 72% Total DOC~90 mmolC/m^3
    ice_flux_bgc.fed[i, j, :] = c0
    ice_flux_bgc.fed[i, j, 0] = dbl_kind(0.4)  # c1 (nM) Lannuzel2007 DFe
    ice_flux_bgc.fep[i, j, :] = c0
    ice_flux_bgc.fep[i, j, 0] = c2  # (nM) van der Merwe 2011
    ice_flux_bgc.hum[i, j] = c1     # mmol C/m^3
    ice_flux_bgc.nit[i, j] = dbl_kind(12.0)
    ice_flux_bgc.sil[i, j] = dbl_kind(25.0)
    ice_flux_bgc.zaeros[i, j, :] = c0


def _load_ocean_bio(i, j):
    """icepack_load_ocean_bio_array for one grid point."""
    max_algae = int(icepack_tracers.max_algae)
    max_doc = int(icepack_tracers.max_doc)
    max_dic = int(icepack_tracers.max_dic)
    max_don = int(icepack_tracers.max_don)
    max_fe = int(icepack_tracers.max_fe)
    max_aero = int(icepack_tracers.max_aero)
    ocean = ice_arrays_column.ocean_bio_all[i, j, :]
    ocean[...] = c0
    R_chl2N = ice_arrays_column.R_chl2N
    algalN = ice_flux_bgc.algalN[i, j, :]
    for k in range(max_algae):
        ocean[k] = algalN[k]
        ks = max_algae + max_doc + max_dic + 1
        ocean[ks + k] = R_chl2N[k] * algalN[k]
    ks = max_algae + 1
    for k in range(max_doc):
        ocean[ks + k] = ice_flux_bgc.doc[i, j, k]
    ks = ks + max_doc
    for k in range(max_dic):
        ocean[ks + k] = ice_flux_bgc.dic[i, j, k]
    ks = 2 * max_algae + max_doc + max_dic + 7
    for k in range(max_don):
        ocean[ks + k] = ice_flux_bgc.don[i, j, k]
    ks = max_algae + 1
    ocean[ks - 1] = ice_flux_bgc.nit[i, j]
    # The loops above use 0-based ks = Fortran ks, then ocean(ks+k) with k starting at 1.
    # Nitrate is a scalar at Fortran ks = max_algae+1, Python index ks_f-1.
    ks = 2 * max_algae + max_doc + 2 + max_dic
    ocean[ks - 1] = ice_flux_bgc.amm[i, j]
    ks = ks + 1
    ocean[ks - 1] = ice_flux_bgc.sil[i, j]
    ks = ks + 1
    ocean[ks - 1] = (
        R_S2N[0] * algalN[0] + R_S2N[1] * algalN[1] + R_S2N[2] * algalN[2]
    )
    ks = ks + 1
    ocean[ks - 1] = ice_flux_bgc.dmsp[i, j]
    ks = ks + 1
    ocean[ks - 1] = ice_flux_bgc.dms[i, j]
    ks = ks + 1
    ocean[ks - 1] = ice_flux_bgc.nit[i, j]
    ks = 2 * max_algae + max_doc + 7 + max_dic + max_don
    for k in range(1, max_fe + 1):
        ocean[ks + k - 1] = ice_flux_bgc.fed[i, j, k - 1]
    ks = ks + max_fe
    for k in range(1, max_fe + 1):
        ocean[ks + k - 1] = ice_flux_bgc.fep[i, j, k - 1]
    ks = ks + max_fe
    for k in range(1, max_aero + 1):
        ocean[ks + k - 1] = ice_flux_bgc.zaeros[i, j, k - 1]
    ks = ks + max_aero + 1
    ocean[ks - 1] = ice_flux_bgc.hum[i, j]


def _init_bgc_column(i, j):
    """icepack_init_bgc. The Fortran driver keeps the result in a local
    ``trcrn_bgc`` that is never copied back onto ``trcrn``.
    """
    if bool(icepack_parameters.restartbgc):
        return
    ncat = int(icepack_tracers.ncat)
    nilyr = int(icepack_tracers.nilyr)
    nbtrcr = int(icepack_tracers.nbtrcr)
    ntrcr = int(icepack_tracers.ntrcr)
    ntrcr_o = int(icepack_tracers.ntrcr_o)
    nt_sice = int(icepack_tracers.nt_sice)
    trcrn = ice_state.trcrn
    trcrn_bgc = np.zeros((ntrcr, ncat), dtype=dbl_kind)
    sicen = np.zeros((nilyr, ncat), dtype=dbl_kind)
    for n in range(ncat):
        for k in range(nilyr):
            sicen[k, n] = trcrn[i, j, nt_sice + k, n]
        for k in range(ntrcr_o, ntrcr):
            trcrn_bgc[k - ntrcr_o, n] = trcrn[i, j, k, n]
    ocean = ice_arrays_column.ocean_bio_all[i, j, :]
    if bool(icepack_parameters.skl_bgc):
        for n in range(ncat):
            for mm in range(nbtrcr):
                trcrn_bgc[int(icepack_tracers.bio_index[mm]) - ntrcr_o, n] = ocean[
                    int(icepack_tracers.bio_index_o[mm]) - 1
                ]
    elif bool(icepack_parameters.scale_bgc) and int(icepack_parameters.ktherm) == 2:
        trtmp = np.zeros(ntrcr + 2, dtype=dbl_kind)
        sss = ice_flux.sss[i, j]
        for n in range(ncat):
            icepack_shortwave.remap_zbgc(
                nilyr, 0, sicen[:, n], trtmp, 0, int(icepack_tracers.nblyr) + 1,
                c1, c1, ice_arrays_column.cgrid[1:nilyr + 1],
                ice_arrays_column.igrid[:int(icepack_tracers.nblyr) + 1], sicen[0, n],
            )
            for mm in range(nbtrcr):
                base = int(icepack_tracers.bio_index[mm]) - ntrcr_o
                trcrn_bgc[base:base + int(icepack_tracers.nblyr) + 1, n] = (
                    trtmp[:int(icepack_tracers.nblyr) + 1]
                    / sss * ocean[int(icepack_tracers.bio_index_o[mm]) - 1]
                )
                trcrn_bgc[
                    base + int(icepack_tracers.nblyr) + 1:
                    base + int(icepack_tracers.nblyr) + 3, n
                ] = c0
    elif nbtrcr > 0 and int(icepack_tracers.nt_fbri) >= 0:
        nblyr = int(icepack_tracers.nblyr)
        nt_zbgc_frac = int(icepack_tracers.nt_zbgc_frac)
        for n in range(ncat):
            for mm in range(nbtrcr):
                base = int(icepack_tracers.bio_index[mm]) - ntrcr_o
                for k in range(nblyr + 1):
                    trcrn_bgc[base + k, n] = (
                        ocean[int(icepack_tracers.bio_index_o[mm]) - 1] * zbgc_init_frac[mm]
                    )
                trcrn_bgc[base + nblyr + 1: base + nblyr + 3, n] = c0  # snow
                trcrn_bgc[nt_zbgc_frac + mm - ntrcr_o, n] = zbgc_frac_init[mm]
    _ = (sicen, trcrn_bgc, ice_flux)


def init_bgc():
    """Initialize biogeochemistry tracers and the ocean concentrations."""
    subname = "(init_bgc)"
    col = ice_arrays_column
    col.bphi[...] = c0
    col.iDi[...] = c0
    col.bTiz[...] = c0
    col.iki[...] = c0
    col.ocean_bio_all[...] = c0
    col.ice_bio_net[...] = c0
    col.snow_bio_net[...] = c0
    col.zfswin[...] = c0
    if col.trcrn_sw is None:
        raise RuntimeError(subname + " trcrn_sw is not allocated")
    col.trcrn_sw[...] = c0

    ilo, ihi, jlo, jhi = ice_domain_size.physical_bounds()
    if not bool(ice_restart_column.restart_bgc):
        for j in range(jlo, jhi + 1):
            for i in range(ilo, ihi + 1):
                _init_ocean_point(i, j)
        ice_forcing_bgc.init_bgc_data(
            ice_flux_bgc.fed[:, :, 0],
            ice_flux_bgc.fep[:, :, 0],
        )
        ice_forcing_bgc.get_forcing_bgc()

    for j in range(jlo, jhi + 1):
        for i in range(ilo, ihi + 1):
            _load_ocean_bio(i, j)

    if not bool(ice_restart_column.restart_bgc):
        for j in range(jlo, jhi + 1):
            for i in range(ilo, ihi + 1):
                _init_bgc_column(i, j)
    else:
        ice_restart_column.read_restart_bgc()

