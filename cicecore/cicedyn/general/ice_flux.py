"""Flux variable declarations; these include fields sent from the coupler
("in"), sent to the coupler ("out"), written to diagnostic history files
("diagnostic"), and used internally ("internal").

author for JAX version: Xinyuan Zhu
XZ: converted from F90 to Python

2006: Converted to free source form (F90) by Elizabeth Hunke
2004: Block structure added by William Lipscomb
      Swappped, revised, and added some subroutines
author Elizabeth C. Hunke, LANL

``alloc_flux``, ``init_coupler_flux``, ``init_flux_atm``,
``init_flux_ocn``, ``init_history_therm``, and ``init_history_dyn``
are ported.
``scale_fluxes`` has a JAX state path for the standalone driver.
Biogeochemistry fluxes are allocated by ``ice_flux_bgc.alloc_flux_bgc``
and zeroed again here.
C and CD velocity points are allocated when ``grid_ice`` is C or CD.
"""

import math

import numpy as np
import jax.numpy as jnp

from cicecore.cicedyn.general import ice_flux_bgc
from cicecore.cicedyn.general import ice_init
from cicecore.cicedyn.general import ice_state
from cicecore.cicedyn.infrastructure import ice_grid
from cicecore.shared import ice_arrays_column
from cicecore.shared import ice_domain_size
from cicecore.shared.ice_constants import c0, c1, c5, c10, c20, c180
from cicecore.shared.ice_kinds_mod import dbl_kind
from icepack.columnphysics import icepack_parameters
from icepack.columnphysics import icepack_tracers
from icepack.columnphysics.icepack_therm_shared import icepack_liquidus_temperature

# if true, pass per-category shortwave to the coupler
send_i2x_per_cat = False
l_mpond_fresh = False  # if true, include freshwater feedback from meltponds

#-----------------------------------------------------------------
# Dynamics component
# All variables are assumed to be on the atm or ocn thermodynamic
# grid except as noted
#
# scale_fluxes divides several of these by aice "in place", so
# the state of some of these variables is not well defined.  In the
# future, we need to refactor and add "_iavg" versions of the
# fields to clearly differentiate fields that have been divided
# by aice and others that are not.  The challenge is that we need
# to go thru each field carefully to see which version is used.
# For instance, in diagnostics, there are places where these
# fields are multiplied by aice to compute things properly.
# strocn[x,y]T_iavg is the first field defined using _iavg.
#-----------------------------------------------------------------
_HORIZ = (
    # in from atmos (if .not.calc_strair)
    "strax",    # wind stress components (N/m^2), on grid_atm_dynu
    "stray",    #                                 on grid_atm_dynv
    # in from ocean
    "uocn",     # ocean current, x-direction (m/s),     on grid_ocn_dynu
    "vocn",     # ocean current, y-direction (m/s),     on grid_ocn_dynv
    "ss_tltx",  # sea surface slope, x-direction (m/m), on grid_ocn_dynu
    "ss_tlty",  # sea surface slope, y-direction,       on grid_ocn_dynv
    "hwater",   # water depth for seabed stress calc (landfast ice)
    # out to atmosphere
    "strairxT",  # stress on ice by air, x-direction at T points, computed in icepack
    "strairyT",  # stress on ice by air, y-direction at T points, computed in icepack
    # out to ocean          T-cell (kg/m s^2)
    # Note, CICE_IN_NEMO uses strocnx and strocny for coupling
    "strocnxT_iavg",  # ice-ocean stress, x-direction at T points, per ice fraction (scaled flux)
    "strocnyT_iavg",  # ice-ocean stress, y-direction at T points, per ice fraction (scaled flux)
    # diagnostic
    "sig1",      # normalized principal stress component
    "sig2",      # normalized principal stress component
    "sigP",      # internal ice pressure (N/m)
    "taubxU",    # seabed stress (x) (N/m^2)
    "taubyU",    # seabed stress (y) (N/m^2)
    "strairxU",  # stress on ice by air, x-direction at U points
    "strairyU",  # stress on ice by air, y-direction at U points
    "strocnxU",  # ice-ocean stress, x-direction at U points, computed in dyn_finish
    "strocnyU",  # ice-ocean stress, y-direction at U points, computed in dyn_finish
    "strtltxU",  # stress due to sea surface slope, x-direction
    "strtltyU",  # stress due to sea surface slope, y-direction
    "strintxU",  # divergence of internal ice stress, x (N/m^2)
    "strintyU",  # divergence of internal ice stress, y (N/m^2)
    "daidtd",    # ice area tendency due to transport   (1/s)
    "dvidtd",    # ice volume tendency due to transport (m/s)
    "dvsdtd",    # snow volume tendency due to transport (m/s)
    "dagedtd",   # ice age tendency due to transport (s/s)
    "dardg1dt",  # rate of area loss by ridging ice (1/s)
    "dardg2dt",  # rate of area gain by new ridges (1/s)
    "dvirdgdt",  # rate of ice volume ridged (m/s)
    "opening",   # rate of opening due to divergence/shear (1/s)
    # restart
    # ice stress tensor in each corner of T cell (kg/s^2)
    "stressp_1", "stressp_2", "stressp_3", "stressp_4",    # sigma11+sigma22
    "stressm_1", "stressm_2", "stressm_3", "stressm_4",    # sigma11-sigma22
    "stress12_1", "stress12_2", "stress12_3", "stress12_4",  # sigma12
    # internal
    "fmU",  # Coriolis param. * mass in U-cell (kg/s)
    "TbU",  # factor for seabed stress (N/m^2)
    #-----------------------------------------------------------------
    # Thermodynamic component
    #-----------------------------------------------------------------
    # in from atmosphere (if calc_Tsfc)
    "zlvl",   # atm level height (momentum) (m)
    "zlvs",   # atm level height (scalar quantities) (m)
    "uatm",   # wind velocity components (m/s), on grid_atm_dynu
    "vatm",   #                                 on grid_atm_dynv
    "wind",   # wind speed (m/s)              , on grid_atm_dynu
    "potT",   # air potential temperature  (K)
    "Tair",   # air temperature  (K)
    "Qa",     # specific humidity (kg/kg)
    "rhoa",   # air density (kg/m^3)
    "swvdr",  # sw down, visible, direct  (W/m^2)
    "swvdf",  # sw down, visible, diffuse (W/m^2)
    "swidr",  # sw down, near IR, direct  (W/m^2)
    "swidf",  # sw down, near IR, diffuse (W/m^2)
    # in from atmosphere
    "swuvrdr",  #  vis uvr flux, direct (W m-2)
    "swuvrdf",  #  vis uvr flux, diffuse (W m-2)
    "swpardr",  #  vis par flux, direct (W m-2)
    "swpardf",  #  vis par flux, diffuse (W m-2)
    "flw",      # incoming longwave radiation (W/m^2)
    # in from atmosphere
    "frain",  # rainfall rate (kg/m^2 s)
    "fsnow",  # snowfall rate (kg/m^2 s)
    # in from ocean
    "sss",           # sea surface salinity (ppt)
    "sst",           # sea surface temperature (C)
    "frzmlt",        # freezing/melting potential (W/m^2)
    "frzmlt_init",   # frzmlt used in current time step (W/m^2)
    "Tf",            # freezing temperature (C)
    "qdp",           # deep ocean heat flux (W/m^2), negative upward
    "hmix",          # mixed layer depth (m)
    "daice_da",      # data assimilation concentration increment rate
                     # (concentration s-1)(only used in hadgem drivers)
    # out to atmosphere (if calc_Tsfc)
    # note Tsfc is in ice_state.F
    "fsens",      # sensible heat flux (W/m^2)
    "flat",       # latent heat flux   (W/m^2)
    "fswabs",     # shortwave flux absorbed in ice and ocean (W/m^2)
    "fswint_ai",  # SW absorbed in ice interior below surface (W/m^2)
    "flwout",     # outgoing longwave radiation (W/m^2)
    "Tref",       # 2m atm reference temperature (K)
    "Qref",       # 2m atm reference spec humidity (kg/kg)
    "Uref",       # 10m atm reference wind speed (m/s)
    "evap",       # evaporative water flux (kg/m^2/s)
    "evaps",      # evaporative water flux over snow (kg/m^2/s)
    "evapi",      # evaporative water flux over ice (kg/m^2/s)
    # albedos aggregated over categories (if calc_Tsfc)
    "alvdr",  # visible, direct   (fraction)
    "alidr",  # near-ir, direct   (fraction)
    "alvdf",  # visible, diffuse  (fraction)
    "alidf",  # near-ir, diffuse  (fraction)
    # grid-box-mean versions
    "alvdr_ai",  # visible, direct   (fraction)
    "alidr_ai",  # near-ir, direct   (fraction)
    "alvdf_ai",  # visible, diffuse  (fraction)
    "alidf_ai",  # near-ir, diffuse  (fraction)
    # components for history
    "albice",    # bare ice albedo
    "albsno",    # snow albedo
    "albpnd",    # melt pond albedo
    "apeff_ai",  # effective pond area used for radiation calculation
    "snowfrac",  # snow fraction used in radiation
    # components for diagnostic
    "alvdr_init",  # visible, direct   (fraction)
    "alidr_init",  # near-ir, direct   (fraction)
    "alvdf_init",  # visible, diffuse  (fraction)
    "alidf_init",  # near-ir, diffuse  (fraction)
    # out to ocean
    # (Note CICE_IN_NEMO does not use these for coupling.
    #  It uses fresh_ai,fsalt_ai,fhocn_ai and fswthru_ai)
    "fpond",          # fresh water flux to ponds (kg/m^2/s)
    "fresh",          # fresh water flux to ocean (kg/m^2/s)
    "fsalt",          # salt flux to ocean (kg/m^2/s)
    "fhocn",          # net heat flux to ocean (W/m^2)
    "fsloss",         # rate of snow loss to leads (kg/m^2/s)
    "fswthru",        # shortwave penetrating to ocean (W/m^2)
    "fswthru_vdr",    # vis dir shortwave penetrating to ocean (W/m^2)
    "fswthru_vdf",    # vis dif shortwave penetrating to ocean (W/m^2)
    "fswthru_idr",    # nir dir shortwave penetrating to ocean (W/m^2)
    "fswthru_idf",    # nir dif shortwave penetrating to ocean (W/m^2)
    "fswthru_uvrdr",  # vis dir uvr SW penetrating to ocean (W/m^2)
    "fswthru_uvrdf",  # vis dif uvr SW penetrating to ocean (W/m^2)
    "fswthru_pardr",  # nir dir par SW penetrating to ocean (W/m^2)
    "fswthru_pardf",  # nir dif par SW penetrating to ocean (W/m^2)
    # internal
    "scale_factor",  # scaling factor for shortwave components
    # quantities passed from ocean mixed layer to atmosphere
    # (for running with CAM)
    "strairx_ocn",  # stress on ocean by air, x-direction
    "strairy_ocn",  # stress on ocean by air, y-direction
    "fsens_ocn",    # sensible heat flux (W/m^2)
    "flat_ocn",     # latent heat flux   (W/m^2)
    "flwout_ocn",   # outgoing longwave radiation (W/m^2)
    "evap_ocn",     # evaporative water flux (kg/m^2/s)
    "alvdr_ocn",    # visible, direct   (fraction)
    "alidr_ocn",    # near-ir, direct   (fraction)
    "alvdf_ocn",    # visible, diffuse  (fraction)
    "alidf_ocn",    # near-ir, diffuse  (fraction)
    "Tref_ocn",     # 2m atm reference temperature (K)
    "Qref_ocn",     # 2m atm reference spec humidity (kg/kg)
    # diagnostic
    "fsurf",     # net surface heat flux (excluding fcondtop)(W/m^2)
    "fcondtop",  # top surface conductive flux        (W/m^2)
    "fcondbot",  # bottom surface conductive flux     (W/m^2)
    "fbot",      # heat flux at bottom surface of ice (excluding excess) (W/m^2)
    "Tbot",      # temperature at bottom surface of ice (deg C)
    "Tsnice",    # temperature at snow ice interface (deg C)
    "congel",    # basal ice growth         (m/step-->cm/day)
    "frazil",    # frazil ice growth        (m/step-->cm/day)
    "snoice",    # snow-ice formation       (m/step-->cm/day)
    "meltt",     # top ice melt             (m/step-->cm/day)
    "melts",     # snow melt                (m/step-->cm/day)
    "meltb",     # basal ice melt           (m/step-->cm/day)
    "meltl",     # lateral ice melt         (m/step-->cm/day)
    "dsnow",     # change in snow thickness (m/step-->cm/day)
    "daidtt",    # ice area tendency thermo.   (s^-1)
    "dvidtt",    # ice volume tendency thermo. (m/s)
    "dvsdtt",    # snow volume tendency thermo. (m/s)
    "dagedtt",   # ice age tendency thermo.    (s/s)
    "mlt_onset",    # day of year that sfc melting begins
    "frz_onset",    # day of year that freezing begins (congel or frazil)
    "frazil_diag",  # frazil ice growth diagnostic (m/step-->cm/day)
    "dpnd_flush",    # pond flushing rate due to ice permeability (m/step)
    "dpnd_expon",    # exponential pond drainage rate (m/step)
    "dpnd_freebd",   # pond drainage rate due to freeboard constraint (m/step)
    "dpnd_initial",  # runoff rate due to rfrac (m/step)
    "dpnd_dlid",     # pond loss/gain (+/-) to ice lid freezing/melting (m/step)
    "dpnd_melt",     # pond 'drainage' due to ice melting (m / step)
    "dpnd_ridge",    # pond 'drainage' due to ridging (m)
    # As above but these remain grid box mean values i.e. they are not
    # divided by aice at end of ice_dynamics.  These are used in
    # CICE_IN_NEMO for coupling and also for generating
    # ice diagnostics and history files as these are more accurate.
    # (The others suffer from problem of incorrect values at grid boxes
    #  that change from an ice free state to an icy state.)
    "fresh_ai",    # fresh water flux to ocean (kg/m^2/s)
    "fsalt_ai",    # salt flux to ocean (kg/m^2/s)
    "fhocn_ai",    # net heat flux to ocean (W/m^2)
    "fswthru_ai",  # shortwave penetrating to ocean (W/m^2)
    # Used with data assimilation in hadgem drivers
    "fresh_da",  # fresh water flux to ocean due to data assim (kg/m^2/s)
    "fsalt_da",  # salt flux to ocean due to data assimilation(kg/m^2/s)
    #-----------------------------------------------------------------
    # internal
    #-----------------------------------------------------------------
    "uatmT",     # uatm on T grid (m/s)
    "vatmT",     # vatm on T grid (m/s)
    "wlat",      # lateral heat rate (m/s)
    "fsw",       # incoming shortwave radiation (W/m^2)
    "fswup",     # outgoing shortwave radiation (W/m^2)
    "coszen",    # cosine solar zenith angle, < 0 for sun below horizon
    "rdg_conv",  # convergence term for ridging (1/s)
    "rdg_shear", # shear term for ridging (1/s)
)

# Second allocate, when grid_ice is C or CD.
# ice stress tensor at U and T locations (grid_ice = 'C|CD') (kg/s^2)
_CD = (
    "taubxN",    # seabed stress (x) at N points (N/m^2)
    "taubyN",    # seabed stress (y) at N points (N/m^2)
    "strairxN",  # stress on ice by air, x-direction at N points
    "strairyN",  # stress on ice by air, y-direction at N points
    "strocnxN",  # ice-ocean stress, x-direction at N points, computed in dyn_finish
    "strocnyN",  # ice-ocean stress, y-direction at N points, computed in dyn_finish
    "strtltxN",  # stress due to sea surface slope, x-direction at N points
    "strtltyN",  # stress due to sea surface slope, y-direction at N points
    "strintxN",  # divergence of internal ice stress, x at N points (N/m^2)
    "strintyN",  # divergence of internal ice stress, y at N points (N/m^2)
    "fmN",       # Coriolis param. * mass in N-cell (kg/s)
    "TbN",       # factor for seabed stress (N/m^2)
    "taubxE",    # seabed stress (x) at E points (N/m^2)
    "taubyE",    # seabed stress (y) at E points (N/m^2)
    "strairxE",  # stress on ice by air, x-direction at E points
    "strairyE",  # stress on ice by air, y-direction at E points
    "strocnxE",  # ice-ocean stress, x-direction at E points, computed in dyn_finish
    "strocnyE",  # ice-ocean stress, y-direction at E points, computed in dyn_finish
    "strtltxE",  # stress due to sea surface slope, x-direction at E points
    "strtltyE",  # stress due to sea surface slope, y-direction at E points
    "strintxE",  # divergence of internal ice stress, x at E points (N/m^2)
    "strintyE",  # divergence of internal ice stress, y at E points (N/m^2)
    "fmE",       # Coriolis param. * mass in E-cell (kg/s)
    "TbE",       # factor for seabed stress (N/m^2)
    "stresspT", "stressmT", "stress12T",  # sigma11+sigma22, sigma11-sigma22, sigma12
    "stresspU", "stressmU", "stress12U",  # "
)

_CAT = (
    # ridging diagnostics in categories
    "dardg1ndt",  # rate of area loss by ridging ice (1/s)
    "dardg2ndt",  # rate of area gain by new ridges (1/s)
    "dvirdgndt",  # rate of ice volume ridged (m/s)
    "aparticn",   # participation function
    "krdgn",      # mean ridge thickness/thickness of ridging ice
    "ardgn",      # fractional area of ridged ice
    "vrdgn",      # volume of ridged ice
    "araftn",     # rafting ice area
    "vraftn",     # rafting ice volume
    "aredistn",   # redistribution function: fraction of new ridge area
    "vredistn",   # redistribution function: fraction of new ridge volume
    "rsiden",     # fraction of ice that melts laterally
    # in from atmosphere (if .not. Tsfc_calc)
    # required for coupling to HadGEM3
    # NOTE: when in CICE_IN_NEMO mode, these are gridbox mean fields,
    # not per ice area. When in standalone mode, these are per ice area.
    "fsurfn_f",     # net flux to top surface, excluding fcondtop
    "fcondtopn_f",  # downward cond flux at top surface (W m-2)
    "fsensn_f",     # sensible heat flux (W m-2)
    "flatn_f",      # latent heat flux (W m-2)
    # in from atmosphere
    # required for coupling in GEOS
    "evapn_f",          # evaporation/sublimation (kg m-2 s-1)
    "dflatndTsfc_f",    # derivative of latent flux w.r.t. Tsfc
    "dfsurfndTsfc_f",   # derivative of surface flux w.r.t. Tsfc
    "meltsn",     # snow melt in category n (m)
    "melttn",     # top melt in category n (m)
    "meltbn",     # bottom melt in category n (m)
    "congeln",    # congelation ice formation in category n (m)
    "snoicen",    # snow-ice formation in category n (m)
    "keffn_top",  # effective thermal conductivity of the top ice layer
                  # on categories (W/m^2/K)
    "fsurfn",     # category fsurf
    "fcondtopn",  # category fcondtop
    "fcondbotn",  # category fcondbot
    "fsensn",     # category sensible heat flux
    "flatn",      # category latent heat flux
    # Pond diagnostics
    "dpnd_flushn",    # category pond flushing rate due to ice permeability (m/step)
    "dpnd_exponn",    # category exponential pond drainage rate (m/step)
    "dpnd_freebdn",   # category pond drainage rate due to freeboard constraint (m/step)
    "dpnd_initialn",  # category runoff rate due to rfrac (m/step)
    "dpnd_dlidn",     # category pond loss/gain (+/-) to ice lid freezing/melting (m/step)
)

for _name in _HORIZ + _CAT + _CD + ("albcnt", "snwcnt", "salinz", "Tmltz"):
    globals()[_name] = None

# shortwave through ice, per category, allocated in init_coupler_flux
fswthrun_ai = None


def alloc_flux():
    """Allocate space for all variables. Values start at zero."""
    ncat = int(ice_domain_size.ncat)
    nilyr = int(ice_domain_size.nilyr)
    if ncat < 1 or nilyr < 1:
        raise ValueError("(alloc_flux) ncat and nilyr must be set before allocation")

    horiz = ice_domain_size.horizontal_shape()
    cat = horiz + (ncat,)
    layers = horiz + (nilyr + 1,)
    streams = horiz + (int(ice_domain_size.max_nstrm),)
    for name in _HORIZ:
        globals()[name] = np.zeros(horiz, dtype=dbl_kind)
    for name in _CAT:
        globals()[name] = np.zeros(cat, dtype=dbl_kind)
    if str(ice_init.grid_ice).strip() in ("C", "CD"):
        for name in _CD:
            globals()[name] = np.zeros(horiz, dtype=dbl_kind)
    globals()["albcnt"] = np.zeros(streams, dtype=dbl_kind)  # counter for zenith angle
    globals()["snwcnt"] = np.zeros(streams, dtype=dbl_kind)  # counter for presence of snow
    globals()["salinz"] = np.zeros(layers, dtype=dbl_kind)  # initial salinity  profile (ppt)
    globals()["Tmltz"] = np.zeros(layers, dtype=dbl_kind)   # initial melting temperature (^oC)


def init_coupler_flux():
    """Initialize all fluxes exchanged with flux coupler
    and some data-derived fields.

    author Elizabeth C. Hunke, LANL
    """
    global fswthrun_ai

    # data fcondtopn_d / -50, -17, -12, -9, -7, -3 /
    # data fsurfn_d    / 0.20, 0.15, 0.10, 0.05, 0.01, 0.01 /
    max_d = 6
    fcondtopn_d = np.array([-50.0, -17.0, -12.0, -9.0, -7.0, -3.0], dtype=dbl_kind)
    fsurfn_d = np.array([0.20, 0.15, 0.10, 0.05, 0.01, 0.01], dtype=dbl_kind)
    # stefan_boltzmann, Tffresh, vonkar, zref, iceruf are read from the
    # parameter module. icepack_query_parameters only copies them out.
    # call icepack_warnings_flush / abort_ice are not ported.
    vonkar = icepack_parameters.vonkar
    zref = icepack_parameters.zref
    iceruf = icepack_parameters.iceruf
    ncat = int(ice_domain_size.ncat)
    horiz = ice_domain_size.horizontal_shape()
    season = str(ice_init.default_season).strip()

    #-----------------------------------------------------------------
    # fluxes received from atmosphere
    #-----------------------------------------------------------------
    zlvl[...] = c10             # atm level height (momentum) (m)
    zlvs[...] = c10             # atm level height (scalar quantities) (m)
    rhoa[...] = dbl_kind(1.3)   # air density (kg/m^3)
    uatm[...] = c5              # wind velocity    (m/s)
    vatm[...] = c5
    strax[...] = dbl_kind(0.05)
    stray[...] = dbl_kind(0.05)
    fsnow[...] = c0             # snowfall rate (kg/m2/s)
                                # fsnow must be 0 for exact restarts
    if season == "winter":
        # typical winter values
        potT[...] = dbl_kind(253.0)    # air potential temp (K)
        Tair[...] = dbl_kind(253.0)    # air temperature  (K)
        Qa[...] = dbl_kind(0.0006)     # specific humidity (kg/kg)
        swvdr[...] = c0                # shortwave radiation (W/m^2)
        swvdf[...] = c0                # shortwave radiation (W/m^2)
        swidr[...] = c0                # shortwave radiation (W/m^2)
        swidf[...] = c0                # shortwave radiation (W/m^2)
        flw[...] = c180                # incoming longwave rad (W/m^2)
        frain[...] = c0                # rainfall rate (kg/m2/s)
        for n in range(1, ncat + 1):   # conductive heat flux (W/m^2)
            fcondtopn_f[:, :, n - 1] = fcondtopn_d[min(n, max_d) - 1]
        fsurfn_f[...] = fcondtopn_f    # surface heat flux (W/m^2)
        flatn_f[...] = c0              # latent heat flux (kg/m2/s)
        fsensn_f[...] = c0             # sensible heat flux (W/m^2)
    elif season == "summer":
        # typical summer values
        potT[...] = dbl_kind(273.0)    # air potential temp (K)
        Tair[...] = dbl_kind(273.0)    # air temperature  (K)
        Qa[...] = dbl_kind(0.0035)     # specific humidity (kg/kg)
        swvdr[...] = dbl_kind(50.0)    # shortwave radiation (W/m^2)
        swvdf[...] = dbl_kind(50.0)    # shortwave radiation (W/m^2)
        swidr[...] = dbl_kind(50.0)    # shortwave radiation (W/m^2)
        swidf[...] = dbl_kind(50.0)    # shortwave radiation (W/m^2)
        flw[...] = dbl_kind(280.0)     # incoming longwave rad (W/m^2)
        frain[...] = c0                # rainfall rate (kg/m2/s)
        for n in range(1, ncat + 1):   # surface heat flux (W/m^2)
            fsurfn_f[:, :, n - 1] = fsurfn_d[min(n, max_d) - 1]
        fcondtopn_f[...] = dbl_kind(0.0)  # conductive heat flux (W/m^2)
        flatn_f[...] = dbl_kind(-2.0)     # latent heat flux (W/m^2)
        fsensn_f[...] = c0                # sensible heat flux (W/m^2)
    else:
        # typical spring values
        potT[...] = dbl_kind(263.15)   # air potential temp (K)
        Tair[...] = dbl_kind(263.15)   # air temperature  (K)
        Qa[...] = dbl_kind(0.001)      # specific humidity (kg/kg)
        swvdr[...] = dbl_kind(25.0)    # shortwave radiation (W/m^2)
        swvdf[...] = dbl_kind(25.0)    # shortwave radiation (W/m^2)
        swidr[...] = dbl_kind(25.0)    # shortwave radiation (W/m^2)
        swidf[...] = dbl_kind(25.0)    # shortwave radiation (W/m^2)
        flw[...] = dbl_kind(230.0)     # incoming longwave rad (W/m^2)
        frain[...] = c0                # rainfall rate (kg/m2/s)
        for n in range(1, ncat + 1):   # surface heat flux (W/m^2)
            fsurfn_f[:, :, n - 1] = fsurfn_d[min(n, max_d) - 1]
        fcondtopn_f[...] = c0             # conductive heat flux (W/m^2)
        flatn_f[...] = dbl_kind(-1.0)     # latent heat flux (W/m^2)
        fsensn_f[...] = c0                # sensible heat flux (W/m^2)

    swuvrdr[...] = c0  # visible uvr flux, direct (W/m^2)
    swuvrdf[...] = c0  # visible uvr flux, diffuse (W/m^2)
    swpardr[...] = c0  # visible par flux, direct (W/m^2)
    swpardf[...] = c0  # visible par flux, diffuse (W/m^2)

    # isotope, aerosol, and biogeochemistry deposition. Allocated by alloc_flux_bgc.
    ice_flux_bgc.fiso_atm[...] = c0       # isotope deposition rate (kg/m2/s)
    ice_flux_bgc.faero_atm[...] = c0      # aerosol deposition rate (kg/m2/s)
    ice_flux_bgc.flux_bio_atm[...] = c0   # zaero and bio deposition rate (kg/m2/s)

    #-----------------------------------------------------------------
    # fluxes received from ocean
    #-----------------------------------------------------------------

    ss_tltx[...] = c0              # sea surface tilt (m/m)
    ss_tlty[...] = c0
    uocn[...] = c0                 # surface ocean currents (m/s)
    vocn[...] = c0
    frzmlt[...] = c0               # freezing/melting potential (W/m^2)
    frzmlt_init[...] = c0          # freezing/melting potential (W/m^2)
    sss[...] = dbl_kind(34.0)      # sea surface salinity (ppt)
    Tf[...] = icepack_liquidus_temperature(sss)  # freezing temp (C)

    sst[...] = Tf                  # sea surface temp (C)
    qdp[...] = c0                  # deep ocean heat flux (W/m^2)
    hmix[...] = c20                # ocean mixed layer depth (m)
    hwater[...] = ice_grid.bathymetry  # ocean water depth (m)
    daice_da[...] = c0             # data assimilation increment rate

    #-----------------------------------------------------------------
    # fluxes sent to atmosphere
    #-----------------------------------------------------------------

    strairxT[...] = c0             # wind stress, T grid
    strairyT[...] = c0

    fsens[...] = c0
    flat[...] = c0
    fswabs[...] = c0
    fswint_ai[...] = c0
    evap[...] = c0
    evaps[...] = c0
    evapi[...] = c0
    Tref[...] = c0
    Qref[...] = c0
    Uref[...] = c0
    alvdr[...] = c0
    alidr[...] = c0
    alvdf[...] = c0
    alidf[...] = c0

    #-----------------------------------------------------------------
    # fluxes sent to ocean
    #-----------------------------------------------------------------

    strocnxT_iavg[...] = c0  # ice-ocean stress, x-direction (T-cell)
    strocnyT_iavg[...] = c0  # ice-ocean stress, y-direction (T-cell)
    fresh[...] = c0
    fsalt[...] = c0
    fpond[...] = c0
    fhocn[...] = c0
    fswthru[...] = c0
    fswthru_vdr[...] = c0
    fswthru_vdf[...] = c0
    fswthru_idr[...] = c0
    fswthru_idf[...] = c0
    fswthru_uvrdr[...] = c0
    fswthru_uvrdf[...] = c0
    fswthru_pardr[...] = c0
    fswthru_pardf[...] = c0
    fresh_da[...] = c0    # data assimilation
    fsalt_da[...] = c0
    ice_flux_bgc.flux_bio[...] = c0  # bgc
    ice_flux_bgc.fnit[...] = c0
    ice_flux_bgc.fsil[...] = c0
    ice_flux_bgc.famm[...] = c0
    ice_flux_bgc.fdmsp[...] = c0
    ice_flux_bgc.fdms[...] = c0
    ice_flux_bgc.fhum[...] = c0
    ice_flux_bgc.fdust[...] = c0
    ice_flux_bgc.falgalN[...] = c0
    ice_flux_bgc.fdoc[...] = c0
    ice_flux_bgc.fdic[...] = c0
    ice_flux_bgc.fdon[...] = c0
    ice_flux_bgc.ffep[...] = c0
    ice_flux_bgc.ffed[...] = c0

    fswthrun_ai = np.zeros(horiz + (ncat,), dtype=dbl_kind)
    fswthrun_ai[...] = c0

    #-----------------------------------------------------------------
    # derived or computed fields
    #-----------------------------------------------------------------

    coszen[...] = c0            # Cosine of the zenith angle
    fsw[...] = c0               # shortwave radiation (W/m^2)
    fswup[...] = c0             # shortwave radiation (W/m^2)
    scale_factor[...] = c1      # shortwave scaling factor
    wind[...] = np.sqrt(uatm**2 + vatm**2)  # wind speed, (m/s)
    # atmo drag for RASM
    cdn = (vonkar / np.log(zref / iceruf)) * (vonkar / np.log(zref / iceruf))
    ice_arrays_column.Cdn_atm[...] = cdn
    alvdr_init[...] = c0
    alidr_init[...] = c0
    alvdf_init[...] = c0
    alidf_init[...] = c0


def init_flux_atm():
    """Initialize some fluxes sent to coupler for use by the atm model.

    author: Elizabeth C. Hunke, LANL
    """
    #-----------------------------------------------------------------
    # initialize albedo and fluxes
    #-----------------------------------------------------------------
    strairxT[...] = c0  # wind stress, T grid
    strairyT[...] = c0
    # for rectangular grid tests without thermo
    # strairxT(:,:,:) = 0.15_dbl_kind
    # strairyT(:,:,:) = 0.15_dbl_kind

    fsurf[...] = c0
    fcondtop[...] = c0
    fsens[...] = c0
    flat[...] = c0
    fswabs[...] = c0
    flwout[...] = c0
    evap[...] = c0
    evaps[...] = c0
    evapi[...] = c0
    Tref[...] = c0
    Qref[...] = c0
    Uref[...] = c0

    ice_flux_bgc.fiso_evap[...] = c0
    ice_flux_bgc.Qref_iso[...] = c0
    ice_flux_bgc.Qa_iso[...] = c0


def init_flux_ocn():
    """Initialize some fluxes sent to coupler for use by the ocean model.

    NOTE: These fluxes should be initialized immediately after the
          call to the coupler.  The atmospheric fluxes can be initialized
          at the beginning of the following time step because they are
          not modified by any subroutines between the call to_coupler
          and the end of the time step.

    author: Elizabeth C. Hunke, LANL
    """
    #-----------------------------------------------------------------
    # fluxes sent
    #-----------------------------------------------------------------
    fresh[...] = c0
    fsalt[...] = c0
    fpond[...] = c0
    fhocn[...] = c0
    fswthru[...] = c0
    fswthru_vdr[...] = c0
    fswthru_vdf[...] = c0
    fswthru_idr[...] = c0
    fswthru_idf[...] = c0
    fswthru_uvrdr[...] = c0
    fswthru_uvrdf[...] = c0
    fswthru_pardr[...] = c0
    fswthru_pardf[...] = c0

    ice_flux_bgc.faero_ocn[...] = c0
    ice_flux_bgc.fiso_ocn[...] = c0
    ice_flux_bgc.HDO_ocn[...] = c0
    ice_flux_bgc.H2_16O_ocn[...] = c0
    ice_flux_bgc.H2_18O_ocn[...] = c0

    if send_i2x_per_cat:
        fswthrun_ai[...] = c0


def init_history_therm(pytree=None, static=None):
    """Initialize thermodynamic fields written to history files.

    authors: William H. Lipscomb, LANL
             Elizabeth C. Hunke, LANL
    """
    if pytree is not None:
        state = dict(pytree["state"])
        flux = dict(state["ice_flux"])
        column = dict(state["ice_arrays_column"])
        ice = state["ice_state"]
        params = pytree["parameters"]["icepack_parameters"]
        flags = static["icepack_tracers"]

        for name in (
            "fsurf", "fcondtop", "fcondbot", "congel", "fbot", "Tbot", "Tsnice",
            "frazil", "snoice", "dsnow", "meltt", "melts", "meltb", "meltl",
            "fsurfn", "fcondtopn", "fcondbotn", "flatn", "fsensn",
            "fresh_ai", "fsalt_ai", "fhocn_ai", "fswthru_ai", "albice",
            "albsno", "albpnd", "apeff_ai", "snowfrac", "frazil_diag",
            "dpnd_flush", "dpnd_expon", "dpnd_freebd", "dpnd_initial",
            "dpnd_dlid", "dpnd_melt", "dpnd_ridge", "dpnd_flushn",
            "dpnd_exponn", "dpnd_freebdn", "dpnd_initialn", "dpnd_dlidn",
        ):
            flux[name] = jnp.zeros_like(flux[name])
        # Temporary initial values used by the end-of-step tendency fields.
        flux["daidtt"] = ice["aice"]
        flux["dvidtt"] = ice["vice"]
        flux["dvsdtt"] = ice["vsno"]
        flux["dagedtt"] = (
            ice["trcr"][:, :, int(flags["nt_iage"])]
            if bool(flags["tr_iage"]) else jnp.zeros_like(flux["dagedtt"])
        )

        # Drag coefficients are set before the atmospheric boundary call.
        column["Cdn_ocn"] = jnp.full_like(column["Cdn_ocn"], params["dragio"])
        atm_drag = (params["vonkar"] / jnp.log(params["zref"] / params["iceruf"])) ** 2
        column["Cdn_atm"] = jnp.full_like(column["Cdn_atm"], atm_drag)
        column["Cdn_atm_ratio"] = jnp.zeros_like(column["Cdn_atm_ratio"])
        if bool(static["icepack_parameters"]["formdrag"]):
            for name in (
                "Cdn_atm_rdg", "Cdn_atm_floe", "Cdn_atm_pond", "Cdn_atm_skin",
                "Cdn_ocn_skin", "Cdn_ocn_keel", "Cdn_ocn_floe", "hfreebd",
                "hdraft", "hridge", "distrdg", "hkeel", "dkeel", "lfloe", "dfloe",
            ):
                column[name] = jnp.zeros_like(column[name])

        state["ice_flux"] = flux
        state["ice_arrays_column"] = column
        result = dict(pytree)
        result["state"] = state
        return result

    # call icepack_query_parameters(formdrag_out=formdrag)
    formdrag = bool(icepack_parameters.formdrag)
    tr_iage = bool(icepack_tracers.tr_iage)
    nt_iage = int(icepack_tracers.nt_iage)
    dragio = icepack_parameters.dragio
    vonkar = icepack_parameters.vonkar
    zref = icepack_parameters.zref
    iceruf = icepack_parameters.iceruf

    fsurf[...] = c0
    fcondtop[...] = c0
    fcondbot[...] = c0
    congel[...] = c0
    fbot[...] = c0
    Tbot[...] = c0
    Tsnice[...] = c0
    frazil[...] = c0
    snoice[...] = c0
    dsnow[...] = c0
    meltt[...] = c0
    melts[...] = c0
    meltb[...] = c0
    meltl[...] = c0
    daidtt[...] = ice_state.aice  # temporary initial area
    dvidtt[...] = ice_state.vice  # temporary initial volume
    dvsdtt[...] = ice_state.vsno  # temporary initial volume
    if tr_iage:
        dagedtt[...] = ice_state.trcr[:, :, nt_iage]  # temporary initial age
    else:
        dagedtt[...] = c0
    fsurfn[...] = c0
    fcondtopn[...] = c0
    fcondbotn[...] = c0
    flatn[...] = c0
    fsensn[...] = c0
    fresh_ai[...] = c0
    fsalt_ai[...] = c0
    fhocn_ai[...] = c0
    fswthru_ai[...] = c0
    albice[...] = c0
    albsno[...] = c0
    albpnd[...] = c0
    apeff_ai[...] = c0
    snowfrac[...] = c0
    frazil_diag[...] = c0

    # Extra pond diagnostics
    dpnd_flush[...] = c0
    dpnd_expon[...] = c0
    dpnd_freebd[...] = c0
    dpnd_initial[...] = c0
    dpnd_dlid[...] = c0
    dpnd_melt[...] = c0
    dpnd_ridge[...] = c0
    dpnd_flushn[...] = c0
    dpnd_exponn[...] = c0
    dpnd_freebdn[...] = c0
    dpnd_initialn[...] = c0
    dpnd_dlidn[...] = c0

    # drag coefficients are computed prior to the atmo_boundary call,
    # during the thermodynamics section
    ice_arrays_column.Cdn_ocn[...] = dragio
    atm_drag = (vonkar / math.log(zref / iceruf)) * (vonkar / math.log(zref / iceruf))
    ice_arrays_column.Cdn_atm[...] = atm_drag  # atmo drag for RASM
    ice_arrays_column.Cdn_atm_ratio[...] = c0

    if formdrag:
        ice_arrays_column.Cdn_atm_rdg[...] = c0
        ice_arrays_column.Cdn_atm_floe[...] = c0
        ice_arrays_column.Cdn_atm_pond[...] = c0
        ice_arrays_column.Cdn_atm_skin[...] = c0
        ice_arrays_column.Cdn_ocn_skin[...] = c0
        ice_arrays_column.Cdn_ocn_keel[...] = c0
        ice_arrays_column.Cdn_ocn_floe[...] = c0
        ice_arrays_column.hfreebd[...] = c0
        ice_arrays_column.hdraft[...] = c0
        ice_arrays_column.hridge[...] = c0
        ice_arrays_column.distrdg[...] = c0
        ice_arrays_column.hkeel[...] = c0
        ice_arrays_column.dkeel[...] = c0
        ice_arrays_column.lfloe[...] = c0
        ice_arrays_column.dfloe[...] = c0


def init_history_dyn(pytree=None, static=None):
    """Initialize dynamic fields written to history files.

    authors: William H. Lipscomb, LANL
             Elizabeth C. Hunke, LANL
    """
    if pytree is not None:
        state = dict(pytree["state"])
        flux = dict(state["ice_flux"])
        ice = dict(state["ice_state"])
        flags = static["icepack_tracers"]

        for name in (
            "sig1", "sig2", "taubxU", "taubyU", "strocnxU", "strocnyU",
            "strairxU", "strairyU", "strtltxU", "strtltyU", "strintxU",
            "strintyU", "dardg1dt", "dardg2dt", "dvirdgdt", "opening",
            "fmU", "ardgn", "vrdgn", "aparticn", "aredistn", "vredistn",
            "dardg1ndt", "dardg2ndt", "dvirdgndt", "araftn", "vraftn",
        ):
            flux[name] = jnp.zeros_like(flux[name])
        for name in ("divu", "shear", "vort", "strength"):
            ice[name] = jnp.zeros_like(ice[name])
        flux["daidtd"] = ice["aice"]
        flux["dvidtd"] = ice["vice"]
        flux["dvsdtd"] = ice["vsno"]
        if bool(flags["tr_iage"]):
            flux["dagedtd"] = ice["trcr"][:, :, int(flags["nt_iage"])]
        flux["krdgn"] = jnp.ones_like(flux["krdgn"])

        if static["ice_dyn_shared"]["grid_ice"] in ("C", "CD"):
            for name in (
                "taubxE", "taubyE", "strocnxE", "strocnyE", "strairxE",
                "strairyE", "strtltxE", "strtltyE", "strintxE", "strintyE",
                "fmE", "TbE", "taubxN", "taubyN", "strocnxN", "strocnyN",
                "strairxN", "strairyN", "strtltxN", "strtltyN", "strintxN",
                "strintyN", "fmN", "TbN",
            ):
                flux[name] = jnp.zeros_like(flux[name])

        state["ice_flux"] = flux
        state["ice_state"] = ice
        result = dict(pytree)
        result["state"] = state
        return result

    tr_iage = bool(icepack_tracers.tr_iage)
    nt_iage = int(icepack_tracers.nt_iage)

    sig1[...] = c0
    sig2[...] = c0
    ice_state.divu[...] = c0
    ice_state.shear[...] = c0
    ice_state.vort[...] = c0
    taubxU[...] = c0
    taubyU[...] = c0
    ice_state.strength[...] = c0
    strocnxU[...] = c0
    strocnyU[...] = c0
    strairxU[...] = c0
    strairyU[...] = c0
    strtltxU[...] = c0
    strtltyU[...] = c0
    strintxU[...] = c0
    strintyU[...] = c0
    dardg1dt[...] = c0
    dardg2dt[...] = c0
    dvirdgdt[...] = c0
    opening[...] = c0
    daidtd[...] = ice_state.aice  # temporary initial area
    dvidtd[...] = ice_state.vice  # temporary initial volume
    dvsdtd[...] = ice_state.vsno  # temporary initial volume
    if tr_iage:
        dagedtd[...] = ice_state.trcr[:, :, nt_iage]  # temporary initial age
    fmU[...] = c0
    ardgn[...] = c0
    vrdgn[...] = c0
    krdgn[...] = c1
    aparticn[...] = c0
    aredistn[...] = c0
    vredistn[...] = c0
    dardg1ndt[...] = c0
    dardg2ndt[...] = c0
    dvirdgndt[...] = c0
    araftn[...] = c0
    vraftn[...] = c0

    grid_ice = str(ice_init.grid_ice).strip()
    if grid_ice == "CD" or grid_ice == "C":
        taubxE[...] = c0
        taubyE[...] = c0
        strocnxE[...] = c0
        strocnyE[...] = c0
        strairxE[...] = c0
        strairyE[...] = c0
        strtltxE[...] = c0
        strtltyE[...] = c0
        strintxE[...] = c0
        strintyE[...] = c0
        fmE[...] = c0
        taubxN[...] = c0
        taubyN[...] = c0
        strocnxN[...] = c0
        strocnyN[...] = c0
        strairxN[...] = c0
        strairyN[...] = c0
        strtltxN[...] = c0
        strtltyN[...] = c0
        strintxN[...] = c0
        strintyN[...] = c0
        fmN[...] = c0


def scale_fluxes(pytree, static):
    """Convert grid-box mean coupled fluxes to values per unit ice area.

    This is the standalone driver's call to ``scale_fluxes``. The optional
    NEMO surface fields and optional ``Uref`` argument are not passed there.
    """
    state = dict(pytree["state"])
    flux = dict(state["ice_flux"])
    bgc = dict(state["ice_flux_bgc"])
    ice = state["ice_state"]
    grid = pytree["constants"]["ice_grid"]
    params = pytree["parameters"]["icepack_parameters"]

    has_ice = grid["tmask"] & (ice["aice"] > 0.0)
    area_inverse = 1.0 / jnp.where(has_ice, ice["aice"], 1.0)
    for name in (
        "strairxT", "strairyT", "fsens", "flat", "fswabs", "evap",
        "fresh", "fsalt", "fhocn", "fswthru", "fswthru_vdr",
        "fswthru_vdf", "fswthru_idr", "fswthru_idf", "alvdr",
        "alidr", "alvdf", "alidf",
    ):
        flux[name] = jnp.where(has_ice, flux[name] * area_inverse, 0.0)
    flux["flwout"] = jnp.where(
        has_ice, flux["flwout"] * area_inverse,
        -params["stefan_boltzmann"] * (flux["Tf"] + params["Tffresh"])**4,
    )
    flux["Tref"] = jnp.where(has_ice, flux["Tref"] * area_inverse, flux["Tair"])
    flux["Qref"] = jnp.where(has_ice, flux["Qref"] * area_inverse, flux["Qa"])
    for name in ("flux_bio", "faero_ocn", "Qref_iso", "fiso_evap", "fiso_ocn"):
        bgc[name] = jnp.where(has_ice[:, :, None],
                              bgc[name] * area_inverse[:, :, None], 0.0)

    state["ice_flux"] = flux
    state["ice_flux_bgc"] = bgc
    result = dict(pytree)
    result["state"] = state
    return result
