"""The albedo and absorbed/transmitted flux parameterizations for
snow over ice, bare ice and ponded ice.

Presently, two methods are included:
  (1) CCSM3
  (2) Delta-Eddington
as two distinct routines.
Either can be called from the ice driver.

The Delta-Eddington method is described here:

Briegleb, B. P., and B. Light (2007): A Delta-Eddington Multiple
   Scattering Parameterization for Solar Radiation in the Sea Ice
   Component of the Community Climate System Model, NCAR Technical
   Note  NCAR/TN-472+STR  February 2007

name: originally ice_albedo

author for JAX version: Xinyuan Zhu
XZ: converted from F90 to Python

authors:  Bruce P. Briegleb, NCAR
          Elizabeth C. Hunke and William H. Lipscomb, LANL
2005, WHL: Moved absorbed_solar from icepack_therm_vertical to this
           module and changed name from ice_albedo
2006, WHL: Added Delta Eddington routines from Bruce Briegleb
2006, ECH: Changed data statements in Delta Eddington routines (no
           longer hardwired)
           Converted to free source form (F90)
2007, BPB: Completely updated Delta-Eddington code, so that:
           (1) multiple snow layers enabled (i.e. nslyr > 1)
           (2) included SSL for snow surface absorption
           (3) added Sswabs for internal snow layer absorption
           (4) variable sea ice layers allowed (i.e. not hardwired)
           (5) updated all inherent optical properties
           (6) included algae absorption for sea ice lowest layer
           (7) very complete internal documentation included
2007, ECH: Improved efficiency
2008, BPB: Added aerosols to Delta Eddington code
2013, ECH: merged with NCAR version, cleaned up

CESMCOUPLED is not defined. One column, no block index. Tracer indices
are 0-based, so a slice is ``array[..., nt]``. Category loops
``n = 1..ncat`` are ``0..ncat-1``. Radiation levels that Fortran stores
with a lower bound of 0 keep that index. Table searches still return a
1-based ``nr``, and ``remap_zbgc`` keeps its local donor arrays 1-based.
An abort raises.
``icepack_prep_radiation`` scales the prior step's radiation on the JAX path.
``remap_zbgc`` is included here because shortwave calls it and the
zbgc module is not in this repository yet.
"""

import math

import numpy as np
import jax.numpy as jnp

from cicecore.shared import ice_arrays_column
from cicecore.shared import ice_fileunits
from icepack.columnphysics import icepack_parameters
from icepack.columnphysics import icepack_shortwave_data as swdata
from icepack.columnphysics import icepack_tracers
from icepack.columnphysics.icepack_kinds import dbl_kind
from icepack.columnphysics.icepack_orbital import compute_coszen


from icepack.columnphysics.icepack_parameters import (
    R_ice,
    R_pnd,
    R_snw,
    Timelt,
    ahmax,
    albicei,
    albicev,
    albocn,
    albsnowi,
    albsnowv,
    albedo_type,
    argcheck,
    awtidf,
    awtidr,
    awtvdf,
    awtvdr,
    c0,
    c1,
    c1p5,
    c2,
    c3,
    c4,
    c10,
    dEdd_algae,
    dT_mlt,
    hi_ssl,
    hp0,
    hp1,
    hpmin,
    hs1,
    hs_min,
    hs_ssl,
    kalg,
    kappav,
    min_bgc,
    modal_aero,
    p01,
    p1,
    p15,
    p25,
    p5,
    p75,
    puny,
    rhofresh,
    rhoi,
    rhos,
    rsnw_mlt,
    rsnw_tmax,
    semi_implicit_Tsfc,
    shortwave,
    sk_l,
    snw_ssp_table,
    snwgrain,
    snwlvlfac,
    snwredist,
    snowpatch,
    z_tracers,
)
from icepack.columnphysics.icepack_parameters import skl_bgc


def icepack_prep_radiation(
    aice, aicen, swvdr, swvdf, swidr, swidf,
    alvdr_ai, alvdf_ai, alidr_ai, alidf_ai, scale_factor,
    fswsfcn, fswintn, fswthrun, fswthrun_vdr, fswthrun_vdf,
    fswthrun_idr, fswthrun_idf, fswpenln, Sswabsn, Iswabsn, puny,
):
    """Scale previous-step absorbed solar fields for the current forcing.

    Arguments include the physical horizontal grid dimensions. Each
    category is scaled only where its ice concentration exceeds puny.
    """
    # Compute netsw scaling factor (new netsw / old netsw).
    netsw = (swvdr * (1 - alvdr_ai) + swvdf * (1 - alvdf_ai)
             + swidr * (1 - alidr_ai) + swidf * (1 - alidf_ai))
    active = (aice > 0) & (scale_factor > puny)
    scale_factor = jnp.where(active, netsw / jnp.where(active, scale_factor, 1), 1)

    # Scale absorbed radiation only in categories containing ice.
    factor = jnp.where(aicen > puny, scale_factor[:, :, None], 1)
    layer_factor = factor[:, :, None, :]
    return (
        scale_factor,
        fswsfcn * factor, fswintn * factor, fswthrun * factor,
        fswthrun_vdr * factor, fswthrun_vdf * factor,
        fswthrun_idr * factor, fswthrun_idf * factor,
        fswpenln * layer_factor, Sswabsn * layer_factor,
        Iswabsn * layer_factor,
    )

# maximum argument of exponential
exp_argmax = c10

_step_first_call = True


def _diag(text):
    ice_fileunits.nu_diag.write(str(text) + "\n")


def _nint(value):
    """Fortran NINT: nearest integer, halfway cases away from zero."""
    value = float(value)
    if value >= 0.0:
        return int(math.floor(value + 0.5))
    return int(math.ceil(value - 0.5))


def _tr(name, default=None):
    """Tracer flag or index. Missing names keep the Fortran default of false."""
    if hasattr(icepack_tracers, name):
        return getattr(icepack_tracers, name)
    if default is None:
        raise AttributeError(name)
    return default


def _chl_scales():
    """R_chl2N and F_abs_chl, built from the column parameters.

    icepack_zbgc fills these from ratio_chl2N_* and F_abs_chl_*.
    Index 0 is Fortran algae type 1.
    """
    R_chl2N = np.array(
        [
            icepack_parameters.ratio_chl2N_diatoms,
            icepack_parameters.ratio_chl2N_sp,
            icepack_parameters.ratio_chl2N_phaeo,
        ],
        dtype=dbl_kind,
    )
    F_abs_chl = np.array(
        [
            icepack_parameters.F_abs_chl_diatoms,
            icepack_parameters.F_abs_chl_sp,
            icepack_parameters.F_abs_chl_phaeo,
        ],
        dtype=dbl_kind,
    )
    return R_chl2N, F_abs_chl


def _sum_aerosol(
    ns, k, aero_mp, offset, scale, modal,
    kaer_bc, waer_bc, gaer_bc, bcenh, kaer, waer, gaer,
    k_bcini, k_bcins, k_bcexs,
):
    """Aerosol optical depth for one layer.

    ``ns`` and ``na`` are Fortran 1-based. ``k`` is a radiation level.
    ``offset`` selects aero_mp(na), (na+1), (na+2), or (na+3).
    """
    taer = c0
    waer_sum = c0
    gaer_sum = c0
    n_aero = int(icepack_tracers.n_aero)
    na = 1
    while na <= 4 * n_aero:
        amp = aero_mp[na - 1 + offset] * scale
        if modal:
            if na == 1:  # interstitial BC
                ie = k_bcexs[k] - 1
                ext = kaer_bc[ns - 1, ie]
                ssa = waer_bc[ns - 1, ie]
                asm = gaer_bc[ns - 1, ie]
                ta = amp * ext
                wa = ta * ssa
                ga = wa * asm
            elif na == 5:  # within-ice BC
                ii = k_bcins[k] - 1
                ig = k_bcini[k] - 1
                ext = kaer_bc[ns - 1, ii]
                ssa = waer_bc[ns - 1, ii]
                asm = gaer_bc[ns - 1, ii]
                ta = amp * ext * bcenh[ns - 1, ii, ig]
                wa = amp * ext * ssa
                ga = wa * asm
            else:  # other species (dust)
                sp = (na - 1) // 4
                ext = kaer[ns - 1, sp]
                ssa = waer[ns - 1, sp]
                asm = gaer[ns - 1, sp]
                ta = amp * ext
                wa = ta * ssa
                ga = wa * asm
        else:
            sp = (na - 1) // 4  # (1+(na-1)/4)
            ext = kaer[ns - 1, sp]
            ssa = waer[ns - 1, sp]
            asm = gaer[ns - 1, sp]
            ta = amp * ext
            wa = ta * ssa
            ga = wa * asm
        taer = taer + ta
        waer_sum = waer_sum + wa
        gaer_sum = gaer_sum + ga
        na = na + 4
    return taer, waer_sum, gaer_sum


def _mix_aero(g, w0, tau, k, taer, waer, gaer):
    g[k] = (g[k] * w0[k] * tau[k] + gaer) / (w0[k] * tau[k] + waer)
    w0[k] = (w0[k] * tau[k] + waer) / (tau[k] + taer)
    tau[k] = tau[k] + taer


def _mix_zaero(g, w0, tau, k, ns, gzaer, wzaer, tzaer):
    g[k] = (g[k] * w0[k] * tau[k] + gzaer[ns - 1, k]) / (
        w0[k] * tau[k] + wzaer[ns - 1, k]
    )
    w0[k] = (w0[k] * tau[k] + wzaer[ns - 1, k]) / (tau[k] + tzaer[ns - 1, k])
    tau[k] = tau[k] + tzaer[ns - 1, k]


def _interface_flux(klevp, trndir, trntdr, trndif, rupdir, rupdif, rdndif):
    """dfdir = fdirdn - fdirup and dfdif = fdifdn - fdifup."""
    dfdir = np.zeros(klevp + 1, dtype=dbl_kind)
    dfdif = np.zeros(klevp + 1, dtype=dbl_kind)
    for k in range(0, klevp + 1):
        # interface scattering
        refk = c1 / (c1 - rdndif[k] * rupdif[k])
        # dfdir = fdirdn - fdirup
        dfdir[k] = (
            trndir[k]
            + (trntdr[k] - trndir[k]) * (c1 - rupdif[k]) * refk
            - trndir[k] * rupdir[k] * (c1 - rdndif[k]) * refk
        )
        if dfdir[k] < puny:
            dfdir[k] = c0  # echmod necessary?
        # dfdif = fdifdn - fdifup
        dfdif[k] = trndif[k] * (c1 - rupdif[k]) * refk
        if dfdif[k] < puny:
            dfdif[k] = c0  # echmod necessary?
    return dfdir, dfdif


def icepack_init_radiation():
    """Initialize data needed for shortwave radiation calculations.

    This should be called after values are set via icepack_init_parameters.
    """
    subname = "(icepack_init_radiation)"

    #-----------------------------------------------------------------
    # Set dEdd parameter tables
    #-----------------------------------------------------------------

    if shortwave[:4] == "dEdd":
        swdata.icepack_shortwave_init_dEdd3band()

    if shortwave.strip() == "dEdd_snicar_ad":
        swdata.icepack_shortwave_init_dEdd5band()

        if snw_ssp_table.strip() == "test":  # 5x5 test table
            swdata.icepack_shortwave_init_snicartest()
        elif snw_ssp_table.strip() == "snicar":  # 5 x 1471 table
            swdata.icepack_shortwave_init_snicar()
        else:
            raise RuntimeError(
                subname
                + "ERROR: snw_ssp_table = "
                + snw_ssp_table.strip()
                + " not supported"
            )

        #------------------------------
        # Check SNICAR SSP data
        #------------------------------
        # Fortran indices in the print are 1-based.
        _diag(subname + " nmbrad_snicar  = " + format(int(swdata.nmbrad_snicar), "8d"))
        _diag(subname + " nspint         = " + format(int(swdata.nspint_5bd), "8d"))
        _diag(subname + " nmodal1        = " + format(int(swdata.nmodal1), "8d"))
        _diag(subname + " nmodal2        = " + format(int(swdata.nmodal2), "8d"))
        _diag(subname + " max_aero       = " + format(int(swdata.max_aero), "8d"))
        _diag(
            subname
            + " ssp_snwextdr("
            + format(1, "5d")
            + ","
            + format(1, "5d")
            + ") = "
            + format(float(swdata.ssp_snwextdr[0, 0]), "22.15g")
        )
        _diag(
            subname
            + " ssp_snwextdr("
            + format(int(swdata.nspint_5bd), "5d")
            + ","
            + format(1, "5d")
            + ") = "
            + format(float(swdata.ssp_snwextdr[swdata.nspint_5bd - 1, 0]), "22.15g")
        )
        _diag(
            subname
            + " ssp_snwextdr("
            + format(1, "5d")
            + ","
            + format(int(swdata.nmbrad_snicar), "5d")
            + ") = "
            + format(float(swdata.ssp_snwextdr[0, swdata.nmbrad_snicar - 1]), "22.15g")
        )
        _diag(
            subname
            + " ssp_snwextdr("
            + format(int(swdata.nspint_5bd), "5d")
            + ","
            + format(int(swdata.nmbrad_snicar), "5d")
            + ") = "
            + format(
                float(swdata.ssp_snwextdr[swdata.nspint_5bd - 1, swdata.nmbrad_snicar - 1]),
                "22.15g",
            )
        )


def shortwave_ccsm3(
    aicen, vicen, vsnon, Tsfcn,
    swvdr, swvdf, swidr, swidf,
    albedo_type_in, albicev_in, albicei_in, albsnowv_in, albsnowi_in, ahmax_in,
    alvdrn, alidrn, alvdfn, alidfn,
    fswsfc, fswint,
    swuvrdr=None, swuvrdf=None, swpardr=None, swpardf=None,
    fswthrun=None,
    fswthrun_vdr=None, fswthrun_vdf=None, fswthrun_idr=None, fswthrun_idf=None,
    fswthrun_uvrdr=None, fswthrun_uvrdf=None, fswthrun_pardr=None, fswthrun_pardf=None,
    fswpenl=None, Iswabs=None, Sswabs=None,
    albin=None, albsn=None, coszen=None,
):
    """Solar radiation: albedo and absorbed shortwave."""
    subname = "(shortwave_ccsm3)"
    ncat = int(icepack_tracers.ncat)

    # For basic shortwave, set coszen to a constant between 0 and 1.
    coszen_out = p5  # sun above the horizon
    if isinstance(coszen, np.ndarray):
        coszen[...] = coszen_out

    for n in range(ncat):  # Fortran n = 1, ncat
        Sswabs[:, n] = c0

        alvdrni = albocn
        alidrni = albocn
        alvdfni = albocn
        alidfni = albocn

        alvdrns = albocn
        alidrns = albocn
        alvdfns = albocn
        alidfns = albocn

        alvdrn[n] = albocn
        alidrn[n] = albocn
        alvdfn[n] = albocn
        alidfn[n] = albocn

        albin[n] = c0
        albsn[n] = c0

        fswsfc[n] = c0
        fswint[n] = c0
        fswthrun[n] = c0
        fswpenl[:, n] = c0
        Iswabs[:, n] = c0

        if aicen[n] > puny:

            #-----------------------------------------------------------------
            # Compute albedos for ice and snow.
            #-----------------------------------------------------------------

            if albedo_type_in.strip() == "constant":
                (
                    alvdrni, alidrni, alvdfni, alidfni,
                    alvdrns, alidrns, alvdfns, alidfns,
                    alvdrn[n], alidrn[n], alvdfn[n], alidfn[n],
                    albin[n], albsn[n],
                ) = constant_albedos(
                    aicen[n], vsnon[n], Tsfcn[n],
                )
            elif albedo_type_in.strip() == "ccsm3":
                (
                    alvdrni, alidrni, alvdfni, alidfni,
                    alvdrns, alidrns, alvdfns, alidfns,
                    alvdrn[n], alidrn[n], alvdfn[n], alidfn[n],
                    albin[n], albsn[n],
                ) = compute_albedos(
                    aicen[n], vicen[n], vsnon[n], Tsfcn[n],
                    albicev_in, albicei_in, albsnowv_in, albsnowi_in, ahmax_in,
                )
            else:
                raise RuntimeError(
                    subname + " ERROR: albedo_type " + albedo_type_in.strip() + " unknown"
                )

            #-----------------------------------------------------------------
            # Compute solar radiation absorbed in ice and penetrating to ocean.
            #-----------------------------------------------------------------

            (
                fswsfc[n], fswint[n], fswthrun[n],
                l_fswthru_vdr, l_fswthru_vdf, l_fswthru_idr, l_fswthru_idf,
                l_fswthru_uvrdr, l_fswthru_uvrdf, l_fswthru_pardr, l_fswthru_pardf,
            ) = absorbed_solar(
                aicen[n], vicen[n], vsnon[n],
                swvdr, swvdf, swidr, swidf,
                alvdrni, alvdfni, alidrni, alidfni,
                alvdrns, alvdfns, alidrns, alidfns,
                swuvrdr, swuvrdf, swpardr, swpardf,
                fswpenl[:, n], Iswabs[:, n],
            )

            if fswthrun_vdr is not None:
                fswthrun_vdr[n] = l_fswthru_vdr
            if fswthrun_vdf is not None:
                fswthrun_vdf[n] = l_fswthru_vdf
            if fswthrun_idr is not None:
                fswthrun_idr[n] = l_fswthru_idr
            if fswthrun_idf is not None:
                fswthrun_idf[n] = l_fswthru_idf
            if fswthrun_uvrdr is not None:
                fswthrun_uvrdr[n] = l_fswthru_uvrdr
            if fswthrun_uvrdf is not None:
                fswthrun_uvrdf[n] = l_fswthru_uvrdf
            if fswthrun_pardr is not None:
                fswthrun_pardr[n] = l_fswthru_pardr
            if fswthrun_pardf is not None:
                fswthrun_pardf[n] = l_fswthru_pardf

    return coszen_out


def shortwave_ccsm3_jax(aicen, vicen, vsnon, Tsfcn,
                        swvdr, swvdf, swidr, swidf,
                        swuvrdr, swuvrdf, swpardr, swpardf,
                        options, params, nilyr):
    """JAX array form of the CCSM3 category albedo and absorption calculation.

    The final axis is category; preceding axes are horizontal cells. Ice
    layers occupy the penultimate axis in the returned absorption arrays.
    """
    present = aicen > params["puny"]
    area = jnp.where(present, aicen, 1.0)
    hs = vsnon / area
    hi = vicen / area
    albocn = params["albocn"]
    albedo_type_in = str(options["albedo_type"]).strip()

    if albedo_type_in == "constant":
        warm = Tsfcn >= -2.0 * params["puny"]
        snow = hs > params["puny"]
        alvdfni = jnp.where(snow, jnp.where(warm, 0.77, 0.81),
                            jnp.where(warm, 0.68, 0.70))
        alidfni = alvdfni
        alvdfns = alvdfni
        alidfns = alidfni
        alvdfn = alvdfni
        alidfn = alidfni
    elif albedo_type_in == "ccsm3":
        fhtan = jnp.arctan(params["ahmax"] * 4.0)
        fh = jnp.minimum(jnp.arctan(hi * 4.0) / fhtan, 1.0)
        albo = albocn * (1.0 - fh)
        fT = jnp.minimum((params["Timelt"] - Tsfcn) - 1.0, 0.0)
        alvdfni = jnp.maximum(params["albicev"] * fh + albo + 0.075 * fT,
                              albocn)
        alidfni = jnp.maximum(params["albicei"] * fh + albo + 0.075 * fT,
                              albocn)
        alvdfns = jnp.where(hs > params["puny"],
                            params["albsnowv"] + 0.1 * fT, 0.0)
        alidfns = jnp.where(hs > params["puny"],
                            params["albsnowi"] + 0.15 * fT, 0.0)
        asnow = jnp.where(hs > params["puny"],
                           hs / (hs + params["snowpatch"]), 0.0)
        alvdfn = alvdfni * (1.0 - asnow) + alvdfns * asnow
        alidfn = alidfni * (1.0 - asnow) + alidfns * asnow
    else:
        raise ValueError("(shortwave_ccsm3) unknown albedo_type: " + albedo_type_in)

    # Direct and diffuse albedos coincide in the CCSM3 parameterization.
    alvdrn = jnp.where(present, alvdfn, albocn)
    alvdfn = jnp.where(present, alvdfn, albocn)
    alidrn = jnp.where(present, alidfn, albocn)
    alidfn = jnp.where(present, alidfn, albocn)
    albicen = jnp.where(present,
                        (params["awtvdr"] + params["awtvdf"]) * alvdfni
                        + (params["awtidr"] + params["awtidf"]) * alidfni, 0.0)
    albsnon = jnp.where(present,
                        (params["awtvdr"] + params["awtvdf"]) * alvdfns
                        + (params["awtidr"] + params["awtidf"]) * alidfns, 0.0)

    # absorbed_solar uses the geometric snow fraction even when the
    # constant-albedo option assigns the same albedo to snow and bare ice.
    asnow = jnp.where(hs > params["puny"],
                       hs / (hs + params["snowpatch"]), 0.0)

    # All near-infrared absorption is at the surface. Visible light can
    # enter the ice interior; 70 percent of absorbed visible light enters
    # bare ice at its upper interface.
    swabsv = (swvdr[..., None] * ((1.0 - alvdfni) * (1.0 - asnow)
                                  + (1.0 - alvdfns) * asnow)
               + swvdf[..., None] * ((1.0 - alvdfni) * (1.0 - asnow)
                                      + (1.0 - alvdfns) * asnow))
    swabsi = (swidr[..., None] * ((1.0 - alidfni) * (1.0 - asnow)
                                  + (1.0 - alidfns) * asnow)
               + swidf[..., None] * ((1.0 - alidfni) * (1.0 - asnow)
                                      + (1.0 - alidfns) * asnow))
    fswpenvdr = swvdr[..., None] * (1.0 - alvdfni) * (1.0 - asnow) * 0.70
    fswpenvdf = swvdf[..., None] * (1.0 - alvdfni) * (1.0 - asnow) * 0.70
    fswpen = fswpenvdr + fswpenvdf
    fswsfcn = jnp.where(present, swabsv + swabsi - fswpen, 0.0)

    k = jnp.arange(nilyr + 1, dtype=hi.dtype)
    transmission = jnp.exp(-params["kappav"] *
                           hi[..., None, :] * k[None, None, :, None] / nilyr)
    fswpenln = fswpen[..., None, :] * transmission
    Iswabsn = fswpenln[..., :-1, :] - fswpenln[..., 1:, :]
    bottom = transmission[..., -1, :]
    fswthrun = fswpen * bottom
    fswintn = fswpen - fswthrun
    spectral = (
        fswpenvdr * bottom, fswpenvdf * bottom,
        jnp.zeros_like(fswthrun), jnp.zeros_like(fswthrun),
        swuvrdr[..., None] * (1.0 - alvdfni) * (1.0 - asnow) * 0.70 * bottom,
        swuvrdf[..., None] * (1.0 - alvdfni) * (1.0 - asnow) * 0.70 * bottom,
        swpardr[..., None] * (1.0 - alvdfni) * (1.0 - asnow) * 0.70 * bottom,
        swpardf[..., None] * (1.0 - alvdfni) * (1.0 - asnow) * 0.70 * bottom,
    )
    fswpenln = jnp.where(present[..., None, :], fswpenln, 0.0)
    Iswabsn = jnp.where(present[..., None, :], Iswabsn, 0.0)
    return (alvdrn, alvdfn, alidrn, alidfn, albicen, albsnon,
            fswsfcn, jnp.where(present, fswintn, 0.0),
            jnp.where(present, fswthrun, 0.0), fswpenln, Iswabsn,
            tuple(jnp.where(present, value, 0.0) for value in spectral))


def compute_albedos(
    aicen, vicen, vsnon, Tsfcn,
    albicev_in, albicei_in, albsnowv_in, albsnowi_in, ahmax_in,
):
    """Compute albedo for each thickness category."""
    # change in temp to give dalb_mlt albedo change
    dT_melt = c1
    dalb_mlt = dbl_kind(-0.075)  # albedo change per dT_melt change in temp for ice
    dalb_mltv = -p1  # albedo vis change per dT_melt change in temp for snow
    dalb_mlti = -p15  # albedo nir change per dT_melt change in temp for snow

    hi = vicen / aicen  # ice thickness (m)
    hs = vsnon / aicen  # snow thickness (m)

    # bare ice, thickness dependence
    fhtan = math.atan(ahmax_in * c4)
    fh = min(math.atan(hi * c4) / fhtan, c1)
    albo = albocn * (c1 - fh)  # effective ocean albedo, function of ice thickness
    alvdfni = albicev_in * fh + albo
    alidfni = albicei_in * fh + albo

    # bare ice, temperature dependence
    dTs = Timelt - Tsfcn  # difference of Tsfc and Timelt
    fT = min(dTs / dT_melt - c1, c0)
    alvdfni = alvdfni - dalb_mlt * fT
    alidfni = alidfni - dalb_mlt * fT

    # avoid negative albedos for thin, bare, melting ice
    alvdfni = max(alvdfni, albocn)
    alidfni = max(alidfni, albocn)

    # Snow albedos are unused when hs <= puny (asnow is then 0).
    alvdfns = c0
    alidfns = c0
    if hs > puny:
        alvdfns = albsnowv_in
        alidfns = albsnowi_in

        # snow on ice, temperature dependence
        alvdfns = alvdfns - dalb_mltv * fT
        alidfns = alidfns - dalb_mlti * fT

    # direct albedos (same as diffuse for now)
    alvdrni = alvdfni
    alidrni = alidfni
    alvdrns = alvdfns
    alidrns = alidfns

    # fractional area of snow cover
    if hs > puny:
        asnow = hs / (hs + snowpatch)
    else:
        asnow = c0

    # combine ice and snow albedos (for coupler)
    alvdfn = alvdfni * (c1 - asnow) + alvdfns * asnow
    alidfn = alidfni * (c1 - asnow) + alidfns * asnow
    alvdrn = alvdrni * (c1 - asnow) + alvdrns * asnow
    alidrn = alidrni * (c1 - asnow) + alidrns * asnow

    # save ice and snow albedos (for history)
    albin = (
        awtvdr * alvdrni + awtidr * alidrni
        + awtvdf * alvdfni + awtidf * alidfni
    )
    albsn = (
        awtvdr * alvdrns + awtidr * alidrns
        + awtvdf * alvdfns + awtidf * alidfns
    )
    return (
        alvdrni, alidrni, alvdfni, alidfni,
        alvdrns, alidrns, alvdfns, alidfns,
        alvdrn, alidrn, alvdfn, alidfn,
        albin, albsn,
    )


def constant_albedos(aicen, vsnon, Tsfcn):
    """Compute albedo for each thickness category."""
    warmice = dbl_kind(0.68)
    coldice = dbl_kind(0.70)
    warmsnow = dbl_kind(0.77)
    coldsnow = dbl_kind(0.81)

    hs = vsnon / aicen  # snow thickness (m)

    if hs > puny:
        # snow, temperature dependence
        if Tsfcn >= -c2 * puny:
            alvdfn = warmsnow
            alidfn = warmsnow
        else:
            alvdfn = coldsnow
            alidfn = coldsnow
    else:  # hs < puny
        # bare ice, temperature dependence
        if Tsfcn >= -c2 * puny:
            alvdfn = warmice
            alidfn = warmice
        else:
            alvdfn = coldice
            alidfn = coldice

    # direct albedos (same as diffuse for now)
    alvdrn = alvdfn
    alidrn = alidfn

    alvdrni = alvdrn
    alidrni = alidrn
    alvdrns = alvdrn
    alidrns = alidrn
    alvdfni = alvdfn
    alidfni = alidfn
    alvdfns = alvdfn
    alidfns = alidfn

    # save ice and snow albedos (for history)
    albin = (
        awtvdr * alvdrni + awtidr * alidrni
        + awtvdf * alvdfni + awtidf * alidfni
    )
    albsn = (
        awtvdr * alvdrns + awtidr * alidrns
        + awtvdf * alvdfns + awtidf * alidfns
    )
    return (
        alvdrni, alidrni, alvdfni, alidfni,
        alvdrns, alidrns, alvdfns, alidfns,
        alvdrn, alidrn, alvdfn, alidfn,
        albin, albsn,
    )


def absorbed_solar(
    aicen, vicen, vsnon,
    swvdr, swvdf, swidr, swidf,
    alvdrni, alvdfni, alidrni, alidfni,
    alvdrns, alvdfns, alidrns, alidfns,
    swuvrdr, swuvrdf, swpardr, swpardf,
    fswpenl, Iswabs,
):
    """Compute solar radiation absorbed in ice and penetrating to ocean.

    authors William H. Lipscomb, LANL
            C. M. Bitz, UW
    """
    i0vis = dbl_kind(0.70)  # fraction of penetrating solar rad (visible)
    nilyr = int(icepack_tracers.nilyr)

    trantop = c0
    tranbot = c0
    swuvrdrpen = c0
    swuvrdfpen = c0
    swpardrpen = c0
    swpardfpen = c0

    hs = vsnon / aicen

    #-----------------------------------------------------------------
    # Fractional snow cover
    #-----------------------------------------------------------------
    if hs > puny:
        asnow = hs / (hs + snowpatch)
    else:
        asnow = c0

    #-----------------------------------------------------------------
    # Shortwave flux absorbed at surface, absorbed internally,
    #  and penetrating to mixed layer.
    # This parameterization assumes that all IR is absorbed at the
    #  surface; only visible is absorbed in the ice interior or
    #  transmitted to the ocean.
    #-----------------------------------------------------------------

    swabsv = (
        swvdr * ((c1 - alvdrni) * (c1 - asnow) + (c1 - alvdrns) * asnow)
        + swvdf * ((c1 - alvdfni) * (c1 - asnow) + (c1 - alvdfns) * asnow)
    )
    swabsi = (
        swidr * ((c1 - alidrni) * (c1 - asnow) + (c1 - alidrns) * asnow)
        + swidf * ((c1 - alidfni) * (c1 - asnow) + (c1 - alidfns) * asnow)
    )
    swabs = swabsv + swabsi

    fswpenvdr = swvdr * (c1 - alvdrni) * (c1 - asnow) * i0vis
    fswpenvdf = swvdf * (c1 - alvdfni) * (c1 - asnow) * i0vis

    # no penetrating radiation in near IR
    # fswpenidr = swidr * (c1-alidrni) * (c1-asnow) * i0nir
    # fswpenidf = swidf * (c1-alidfni) * (c1-asnow) * i0nir

    if swuvrdr is not None:
        swuvrdrpen = swuvrdr * (c1 - alvdrni) * (c1 - asnow) * i0vis
    if swuvrdf is not None:
        swuvrdfpen = swuvrdf * (c1 - alvdfni) * (c1 - asnow) * i0vis
    if swpardr is not None:
        swpardrpen = swpardr * (c1 - alvdrni) * (c1 - asnow) * i0vis
    if swpardf is not None:
        swpardfpen = swpardf * (c1 - alvdfni) * (c1 - asnow) * i0vis

    fswpen = fswpenvdr + fswpenvdf
    fswsfc = swabs - fswpen
    trantop = c1  # transmittance at top of ice

    #-----------------------------------------------------------------
    # penetrating SW absorbed in each ice layer
    #-----------------------------------------------------------------

    for k in range(1, nilyr + 1):  # Fortran k = 1, nilyr
        hi = vicen / aicen
        hilyr = hi / dbl_kind(nilyr)
        tranbot = math.exp(-kappav * hilyr * dbl_kind(k))
        Iswabs[k - 1] = fswpen * (trantop - tranbot)

        # bottom of layer k = top of layer k+1
        trantop = tranbot

        # bgc layer model
        if k == 1:  # surface flux
            fswpenl[k - 1] = fswpen
            fswpenl[k] = fswpen * tranbot
        else:
            fswpenl[k] = fswpen * tranbot

    # SW penetrating thru ice into ocean
    fswthru = fswpen * tranbot
    fswthru_vdr = fswpenvdr * tranbot
    fswthru_vdf = fswpenvdf * tranbot
    fswthru_idr = c0
    fswthru_idf = c0
    fswthru_uvrdr = swuvrdrpen * tranbot
    fswthru_uvrdf = swuvrdfpen * tranbot
    fswthru_pardr = swpardrpen * tranbot
    fswthru_pardf = swpardfpen * tranbot

    # SW absorbed in ice interior
    fswint = fswpen - fswthru
    return (
        fswsfc, fswint, fswthru,
        fswthru_vdr, fswthru_vdf, fswthru_idr, fswthru_idf,
        fswthru_uvrdr, fswthru_uvrdf, fswthru_pardr, fswthru_pardf,
    )


# End ccsm3 shortwave method
# Begin Delta-Eddington shortwave method


def shortwave_dEdd_set_snow(
    R_snw_in, dT_mlt_in, rsnw_mlt_in,
    aice, vsno, Tsfc, fs, hs0_in, hs, rhosnw, rsnw, rsnow,
):
    """Set snow horizontal coverage, density and grain radius diagnostically
    for the Delta-Eddington solar radiation method.

    author:  Bruce P. Briegleb, NCAR
      2013:  E Hunke merged with NCAR version

    Returns ``(fs, hs)``. ``rhosnw`` and ``rsnw`` are written in place.
    """
    nslyr = int(icepack_tracers.nslyr)
    # units for the following are 1.e-6 m (micro-meters)
    rsnw_nonmelt = dbl_kind(500.0)  # nonmelt snow grain radius
    rsnw_sig = dbl_kind(250.0)  # assumed sigma for snow grain radius

    # set snow horizontal fraction
    hs = vsno / aice

    if hs >= hs_min:
        fs = c1
        if hs0_in > puny:
            fs = min(hs / hs0_in, c1)

    if snwgrain:  # use snow grain tracer
        for ks in range(nslyr):  # Fortran ks = 1, nslyr
            rsnw[ks] = max(icepack_parameters.rsnw_fall, rsnow[ks])
            rsnw[ks] = min(rsnw_tmax, rsnw[ks])
            rhosnw[ks] = rhos
    else:
        # bare ice, temperature dependence
        dTs = Timelt - Tsfc
        fT = -min(dTs / dT_mlt_in - c1, c0)
        # tune nonmelt snow grain radius if desired: note that
        # the sign is negative so that if R_snw is 1, then the
        # snow grain radius is reduced and thus albedo increased.
        rsnw_nm = rsnw_nonmelt - R_snw_in * rsnw_sig
        rsnw_nm = max(rsnw_nm, icepack_parameters.rsnw_fall)
        rsnw_nm = min(rsnw_nm, rsnw_mlt_in)
        for ks in range(nslyr):
            # snow density ccsm3 constant value
            rhosnw[ks] = rhos
            # snow grain radius between rsnw_nonmelt and rsnw_mlt
            rsnw[ks] = rsnw_nm + (rsnw_mlt_in - rsnw_nm) * fT
            rsnw[ks] = max(rsnw[ks], icepack_parameters.rsnw_fall)
            rsnw[ks] = min(rsnw[ks], rsnw_mlt_in)

    return fs, hs


def shortwave_dEdd_set_eff(
    aicen, vsnon, alvln, apondn, hpndn, ipndn, ffracn, fsnow, dt, Tsfcn,
    fsn, hsn, dhsn, l_rsnows, rhosnwn, rsnwn, l_initonly,
):
    """Set the 'effective' snow and pond fractions and depths for dEdd.

    author:  Bruce P. Briegleb, NCAR
      2013:  E Hunke merged with NCAR version
      2024:  DCS refactored for sealvl ponds

    Returns ``(fsn, hsn, dhsn, fpn, hpn, apeffn)``.
    """
    subname = "(shortwave_dEdd_set_eff)"
    hsnlvl = hsn  # initialize
    if snwredist.strip() == "bulk":
        if not icepack_tracers.tr_lvl:
            raise RuntimeError(subname + " ERROR: need lvl trcr")
        hsnlvl = hsn / (c1 + snwlvlfac * (c1 - alvln))
        # snow volume over level ice
        alvl = aicen * alvln
        if alvl > puny:
            vsn = hsnlvl * alvl
        else:
            vsn = vsnon
            alvl = aicen
        # set snow properties over level ice
        fsn, hsnlvl = shortwave_dEdd_set_snow(
            R_snw, dT_mlt, rsnw_mlt,
            alvl, vsn, Tsfcn, fsn, icepack_parameters.hs0, hsnlvl,
            rhosnwn, rsnwn, l_rsnows,
        )

    fpn = c0  # fraction of ice covered in pond
    hpn = c0  # pond depth over fpn
    # refrozen pond lid thickness avg over ice
    # allow snow to cover pond ice
    ipn = apondn * ipndn
    dhs = dhsn  # snow depth difference, sea ice - pond
    if (not l_initonly) and ipn > puny and dhs < puny and fsnow * dt > hs_min:
        dhs = hsnlvl - fsnow * dt  # initialize dhs>0
    spn = hsnlvl - dhs  # snow depth on pond ice
    if (not l_initonly) and ipn * spn < puny:
        dhs = c0
    dhsn = dhs  # save: constant until reset to 0

    # not using ipn assumes that lid ice is perfectly clear
    # if (ipn <= 0.3_dbl_kind) then

    # fraction of ice area
    fpn = apondn
    # pond depth over fraction fpn
    hpn = hpndn

    # reduce effective pond area absorbing surface heat flux
    # due to flux already having been used to melt pond ice
    fpn = (c1 - ffracn) * fpn

    # taper pond area with snow on pond ice
    if dhs > puny and spn >= puny and hs1 > puny:
        asnow = min(spn / hs1, c1)
        fpn = (c1 - asnow) * fpn

    # infiltrate snow
    hp = hpn
    if hp > puny:
        hs = hsnlvl
        rp = rhofresh * hp / (rhofresh * hp + rhos * hs)
        if rp < p15:
            fpn = c0
            hpn = c0
        else:
            hmx = hs * (rhofresh - rhos) / rhofresh
            # Fortran sign(c1, hp-hmx): +1 if hp >= hmx, else -1; then max(c0, .)
            tmp = c1 if (hp - hmx) >= c0 else c0
            hp = (rhofresh * hp + rhos * hs * tmp) / (rhofresh - rhos * (c1 - tmp))
            hsn = hsn - hp * fpn * (c1 - tmp)
            hpn = hp * tmp
            fpn = fpn * tmp

    # Zero out fraction of thin ponds for radiation only
    if hpn < hpmin:
        fpn = c0
    fsn = min(fsn, c1 - fpn)

    # endif    ! masking by lid ice
    apeffn = fpn  # for history
    return fsn, hsn, dhsn, fpn, hpn, apeffn


def shortwave_dEdd_set_pond(Tsfc, fs):
    """Set pond fraction and depth diagnostically for
    the Delta-Eddington solar radiation method.

    author:  Bruce P. Briegleb, NCAR
      2013:  E Hunke merged with NCAR version
    """
    dT_pnd = c1  # change in temp for pond fraction and depth

    # bare ice, temperature dependence
    dTs = Timelt - Tsfc
    fT = -min(dTs / dT_pnd - c1, c0)
    # pond
    fp = dbl_kind(0.3) * fT * (c1 - fs)
    hp = dbl_kind(0.3) * fT * (c1 - fs)
    return fp, hp


def remap_zbgc(nlyrn, it, trcrn, trtmp, nr0, nbyrn, hice, hinS, ice_grid, bio_grid, S_min):
    """Remaps tracer fields in a given category from one set of layers to another.
    Grids can be very different and  so can  vertical spaces.

    Local donor/receiver arrays keep Fortran 1-based indexes (slot 0 unused).
    ``ice_grid``, ``bio_grid``, ``trcrn``, and ``trtmp`` are 0-based.
    """
    subname = "(remap_zbgc)"
    if (hinS < c0) or (hice < c0):
        raise RuntimeError(subname + " ice: remap_layers_bgc error")

    ntot = nbyrn + nlyrn + 3
    tracer = np.zeros(ntot + 1, dtype=dbl_kind)
    dgrid = np.zeros(ntot + 1, dtype=dbl_kind)
    rgrid = np.zeros(ntot + 1, dtype=dbl_kind)
    trdr = np.zeros(ntot + 1, dtype=dbl_kind)
    trgrid = np.zeros(ntot + 1, dtype=dbl_kind)

    if nr0 == 0:  # cice to bio
        n_nd = nlyrn
        n_nr = nbyrn
        n_plus = 2
        dgrid[1] = min(-hice + hinS, -hinS + hice, c0)
        dgrid[nlyrn + 2] = min(hinS, hice)
        tracer[1] = trcrn[it]
        tracer[nlyrn + 2] = trcrn[it + nlyrn - 1]
        rgrid[nbyrn + 2] = min(hinS, hice)
        if hice > hinS:
            rgrid[1] = c0
            for kr in range(1, n_nr + 1):
                rgrid[kr + 1] = bio_grid[kr - 1] * hinS
            for kd in range(1, n_nd + 1):
                dgrid[kd + 1] = (ice_grid[kd - 1] - c1) * hice + hinS
                tracer[kd + 1] = trcrn[it + kd - 1]
        else:
            rgrid[1] = -hinS + hice
            for kr in range(1, n_nr + 1):
                rgrid[kr + 1] = (bio_grid[kr - 1] - c1) * hinS + hice
            for kd in range(1, n_nd + 1):
                dgrid[kd + 1] = ice_grid[kd - 1] * hice
                tracer[kd + 1] = trcrn[it + kd - 1]
    else:  # bio to cice
        n_nd = nbyrn
        n_nr = nlyrn
        if hice > hinS:  # add S_min to top layer
            n_plus = 3
            tracer[1] = S_min
            tracer[2] = S_min
            rgrid[1] = -hice + hinS
            rgrid[nlyrn + n_plus - 1] = hinS
            for kr in range(1, n_nr + 1):
                rgrid[kr + 1] = (ice_grid[kr - 1] - c1) * hice + hinS
            dgrid[1] = -hice + hinS
            dgrid[2] = (hinS - hice) * p5
            dgrid[nbyrn + n_plus] = hinS
            tracer[nbyrn + n_plus] = trcrn[it + nbyrn - 1]
            for kd in range(1, n_nd + 1):
                dgrid[kd + 2] = bio_grid[kd - 1] * hinS
                tracer[kd + 2] = trcrn[it + kd - 1]
            tracer[n_plus] = (
                S_min * (hice - hinS)
                + tracer[n_plus] * p5 * (dgrid[n_plus + 1] - dgrid[n_plus])
            ) / (hice - hinS + p5 * (dgrid[n_plus + 1] - dgrid[n_plus]))
            tracer[1] = tracer[n_plus]
            tracer[2] = tracer[n_plus]
        else:
            n_plus = 2
            tracer[1] = trcrn[it]
            tracer[nbyrn + 2] = trcrn[it + nbyrn - 1]
            dgrid[1] = hice - hinS
            dgrid[nbyrn + 2] = hice
            rgrid[nlyrn + 2] = hice
            rgrid[1] = c0
            for kd in range(1, n_nd + 1):
                dgrid[kd + 1] = (bio_grid[kd - 1] - c1) * hinS + hice
                tracer[kd + 1] = trcrn[it + kd - 1]
            for kr in range(1, n_nr + 1):
                rgrid[kr + 1] = ice_grid[kr - 1] * hice

    kdr = 0  # combined indices
    kdi = 1

    for kr in range(1, n_nr + 1):
        kd = kdi
        while kd <= n_nd + n_plus:
            if dgrid[kd] < rgrid[kr + 1]:
                kdr = kdr + 1
                trgrid[kdr] = dgrid[kd]
                trdr[kdr] = tracer[kd]
                kd = kd + 1
            elif dgrid[kd] > rgrid[kr + 1]:
                kdr = kdr + 1
                kdi = kd
                trgrid[kdr] = rgrid[kr + 1]
                trtmp[it + kr - 1] = (
                    trdr[kdr - 1]
                    + (rgrid[kr + 1] - trgrid[kdr - 1])
                    * (tracer[kd] - trdr[kdr - 1])
                    / (dgrid[kd] - trgrid[kdr - 1])
                )
                trdr[kdr] = trtmp[it + kr - 1]
                break
            else:
                kdr = kdr + 1
                kdi = kd + 1
                trgrid[kdr] = rgrid[kr + 1]
                trtmp[it + kr - 1] = tracer[kd]
                trdr[kdr] = tracer[kd]
                break


def compute_shortwave_trcr(bgcN, zaero, trcrn_bgcsw, sw_grid, hin, hbri, i_grid, skl_bgc_in, z_tracers_in):
    """Compute aerosols and algal chlorophyll on shortwave grid.

    authors     Nicole Jeffery, LANL
    """
    subname = "(compute_shortwave_trcr)"
    nilyr = int(icepack_tracers.nilyr)
    nslyr = int(icepack_tracers.nslyr)
    nblyr = int(icepack_tracers.nblyr)
    ntrcr = int(icepack_tracers.ntrcr)
    n_algae = int(icepack_tracers.n_algae)
    n_zaero = int(icepack_tracers.n_zaero)
    nbtrcr_sw = int(icepack_tracers.nbtrcr_sw)
    R_chl2N, F_abs_chl = _chl_scales()

    trtmp0 = np.zeros(ntrcr + 2, dtype=dbl_kind)
    trtmp = np.zeros(ntrcr + 2, dtype=dbl_kind)
    trcrn_bgcsw[:] = c0

    icegrid = np.zeros(nilyr + 1, dtype=dbl_kind)
    for k in range(1, nilyr + 2):  # Fortran k = 1, nilyr+1
        icegrid[k - 1] = sw_grid[k - 1]
    if sw_grid[0] * hin * c2 > hi_ssl and hin > puny:
        icegrid[0] = hi_ssl / c2 / hin
    icegrid[1] = c2 * sw_grid[0] + (sw_grid[1] - sw_grid[0])

    if z_tracers_in:
        tr_bgc_N = _tr("tr_bgc_N", False)
        if tr_bgc_N:
            nt_bgc_N = _tr("nt_bgc_N")
            nlt_chl_sw = int(_tr("nlt_chl_sw"))
            if np.size(bgcN) < n_algae * (nblyr + 3):
                raise RuntimeError(subname + " ERROR: size(bgcN) too small")

            for k in range(1, nblyr + 2):
                for n in range(n_algae):  # Fortran n = 1, n_algae
                    # trtmp0(nt_bgc_N(1) + k-1)
                    it0 = int(nt_bgc_N[0]) + k - 1
                    trtmp0[it0] = trtmp0[it0] + (
                        R_chl2N[n] * F_abs_chl[n]
                        * bgcN[int(nt_bgc_N[n]) - int(nt_bgc_N[0]) + k - 1]
                    )

            top_conc = trtmp0[int(nt_bgc_N[0])] * min_bgc
            remap_zbgc(
                nilyr + 1, int(nt_bgc_N[0]),
                trtmp0[:ntrcr], trtmp[: ntrcr + 2],
                1, nblyr + 1, hin, hbri,
                icegrid[: nilyr + 1], i_grid[: nblyr + 1], top_conc,
            )

            for k in range(1, nilyr + 2):
                # trcrn_bgcsw(nlt_chl_sw+nslyr+k) = trtmp(nt_bgc_N(1) + k-1)
                trcrn_bgcsw[nlt_chl_sw + nslyr + k - 1] = trtmp[int(nt_bgc_N[0]) + k - 1]

            for n in range(n_algae):  # snow contribution
                trcrn_bgcsw[nlt_chl_sw - 1] = trcrn_bgcsw[nlt_chl_sw - 1] + (
                    R_chl2N[n] * F_abs_chl[n]
                    * bgcN[int(nt_bgc_N[n]) - int(nt_bgc_N[0]) + 1 + nblyr + 1 - 1]
                )
                # snow surface layer
                # trcrn_bgcsw(nlt_chl_sw+1:nlt_chl_sw+nslyr)
                trcrn_bgcsw[nlt_chl_sw: nlt_chl_sw + nslyr] = (
                    trcrn_bgcsw[nlt_chl_sw: nlt_chl_sw + nslyr]
                    + R_chl2N[n] * F_abs_chl[n]
                    * bgcN[int(nt_bgc_N[n]) - int(nt_bgc_N[0]) + 1 + nblyr + 2 - 1]
                )
                # only 1 snow layer in zaero

        tr_zaero = _tr("tr_zaero", False)
        if tr_zaero:
            nt_zaero = _tr("nt_zaero")
            nlt_zaero_sw = _tr("nlt_zaero_sw")
            if np.size(zaero) < n_zaero * (nblyr + 3):
                raise RuntimeError(subname + " ERROR: size(zaero) too small")

            for n in range(n_zaero):  # Fortran n = 1, n_zaero
                trtmp0[:] = c0
                trtmp[:] = c0
                for k in range(1, nblyr + 2):
                    trtmp0[int(nt_zaero[n]) + k - 1] = zaero[
                        int(nt_zaero[n]) - int(nt_zaero[0]) + 1 + k - 2
                    ]

                top_conc = trtmp0[int(nt_zaero[n])] * min_bgc
                remap_zbgc(
                    nilyr + 1, int(nt_zaero[n]),
                    trtmp0[:ntrcr], trtmp[: ntrcr + 2],
                    1, nblyr + 1, hin, hbri,
                    icegrid[: nilyr + 1], i_grid[: nblyr + 1], top_conc,
                )

                for k in range(1, nilyr + 2):
                    trcrn_bgcsw[int(nlt_zaero_sw[n]) + nslyr + k - 1] = trtmp[
                        int(nt_zaero[n]) + k - 1
                    ]
                # snow ssl
                trcrn_bgcsw[int(nlt_zaero_sw[n]) - 1] = zaero[
                    int(nt_zaero[n]) - int(nt_zaero[0]) + 1 + nblyr + 1 - 1
                ]
                trcrn_bgcsw[int(nlt_zaero_sw[n]): int(nlt_zaero_sw[n]) + nslyr] = zaero[
                    int(nt_zaero[n]) - int(nt_zaero[0]) + 1 + nblyr + 2 - 1
                ]
    elif skl_bgc_in:
        nt_bgc_N = _tr("nt_bgc_N")
        for nn in range(n_algae):
            trcrn_bgcsw[nbtrcr_sw - 1] = trcrn_bgcsw[nbtrcr_sw - 1] + (
                F_abs_chl[nn] * R_chl2N[nn]
                * bgcN[int(nt_bgc_N[nn]) - int(nt_bgc_N[0]) + 1 - 1]
                * sk_l / hin * dbl_kind(nilyr)
            )


def icepack_step_radiation(
    dt, fbri, aicen, vicen, vsnon, Tsfcn,
    alvln, apndn, hpndn, ipndn,
    aeron, bgcNn, zaeron, trcrn_bgcsw,
    TLAT, TLON,
    yday, sec,
    swvdr, swvdf, swidr, swidf,
    coszen, fsnow,
    alvdrn, alvdfn, alidrn, alidfn,
    fswsfcn, fswintn, fswthrun,
    fswpenln, Sswabsn, Iswabsn,
    albicen, albsnon, albpndn, apeffn, snowfracn,
    dhsn, ffracn,
    calendar_type=None, days_per_year=None, nextsw_cday=None,
    swuvrdr=None, swuvrdf=None, swpardr=None, swpardf=None,
    fswthrun_vdr=None, fswthrun_vdf=None, fswthrun_idr=None, fswthrun_idf=None,
    fswthrun_uvrdr=None, fswthrun_uvrdf=None, fswthrun_pardr=None, fswthrun_pardf=None,
    rsnow=None, l_print_point=False, initonly=False,
):
    """Computes radiation fields.

    authors: William H. Lipscomb, LANL
             David Bailey, NCAR
             Elizabeth C. Hunke, LANL

    Category arrays are one column: ``aicen`` has shape ``(ncat,)``.
    Layer arrays are ``(layer, ncat)`` or ``(tracer, ncat)``, matching
    Fortran ``(:,n)``. Array arguments are updated in place.
    ``coszen`` is intent(inout). A NumPy array is updated in place.
    A Python float cannot be updated, so the returned value is the
    updated cosine. ``calendar_type``, ``days_per_year``, and
    ``nextsw_cday`` belong to the CESMCOUPLED branch, which is not compiled.
    """
    global _step_first_call
    subname = "(icepack_step_radiation)"
    ncat = int(icepack_tracers.ncat)
    # calc_Tsfc is not yet on the Python parameters module; Fortran default is .true.
    calc_Tsfc = getattr(icepack_parameters, "calc_Tsfc", True)

    if (_step_first_call and argcheck == "first") or (argcheck == "always"):
        if snwgrain and rsnow is None:
            raise RuntimeError(subname + " ERROR: snwgrain on, rsnow not passed")
        if semi_implicit_Tsfc:
            if not (
                swuvrdr is not None and swuvrdf is not None
                and swpardr is not None and swpardf is not None
                and fswthrun_uvrdr is not None and fswthrun_uvrdf is not None
                and fswthrun_pardr is not None and fswthrun_pardf is not None
            ):
                raise RuntimeError(
                    subname + " ERROR: semi_implicit_Tsfc=T, missing arguments"
                )
        # CESMCOUPLED is not defined, so the calendar-argument check is skipped.

    hin = c0
    hbri = c0

    # Initialize
    for n in range(ncat):
        alvdrn[n] = c0
        alidrn[n] = c0
        alvdfn[n] = c0
        alidfn[n] = c0
        fswsfcn[n] = c0
        fswintn[n] = c0
        fswthrun[n] = c0
    fswpenln[:, :] = c0
    Iswabsn[:, :] = c0
    Sswabsn[:, :] = c0
    trcrn_bgcsw[:, :] = c0

    # Interpolate z-shortwave tracers to shortwave grid
    if dEdd_algae:
        for n in range(ncat):
            if aicen[n] > puny:
                hin = vicen[n] / aicen[n]
                hbri = fbri[n] * hin
                compute_shortwave_trcr(
                    bgcNn[:, n], zaeron[:, n], trcrn_bgcsw[:, n],
                    ice_arrays_column.swgrid, hin, hbri, ice_arrays_column.igrid,
                    skl_bgc, z_tracers,
                )

    coszen_out = coszen
    if calc_Tsfc:
        if shortwave.strip()[:4] == "dEdd":  # delta Eddington
            coszen_out = run_dEdd(
                dt, aicen, vicen, vsnon, Tsfcn,
                alvln, apndn, hpndn, ipndn, aeron, trcrn_bgcsw,
                TLAT, TLON, calendar_type, days_per_year, nextsw_cday, yday, sec,
                swvdr, swvdf, swidr, swidf, coszen, fsnow,
                alvdrn, alvdfn, alidrn, alidfn,
                fswsfcn, fswintn,
                swuvrdr, swuvrdf, swpardr, swpardf,
                fswthrun,
                fswthrun_vdr, fswthrun_vdf, fswthrun_idr, fswthrun_idf,
                fswthrun_uvrdr, fswthrun_uvrdf, fswthrun_pardr, fswthrun_pardf,
                fswpenln, Sswabsn, Iswabsn,
                albicen, albsnon, albpndn, apeffn, snowfracn,
                dhsn, ffracn, rsnow, l_print_point, initonly,
            )
        elif shortwave.strip()[:4] == "ccsm":
            coszen_out = shortwave_ccsm3(
                aicen, vicen, vsnon, Tsfcn,
                swvdr, swvdf, swidr, swidf,
                albedo_type, albicev, albicei, albsnowv, albsnowi, ahmax,
                alvdrn, alidrn, alvdfn, alidfn,
                fswsfcn, fswintn,
                swuvrdr, swuvrdf, swpardr, swpardf,
                fswthrun,
                fswthrun_vdr, fswthrun_vdf, fswthrun_idr, fswthrun_idf,
                fswthrun_uvrdr, fswthrun_uvrdf, fswthrun_pardr, fswthrun_pardf,
                fswpenln, Iswabsn, Sswabsn,
                albicen, albsnon, coszen,
            )
        else:
            raise RuntimeError(
                subname + " ERROR: shortwave " + shortwave.strip() + " unknown"
            )
    else:  # .not. calc_Tsfc
        # Calculate effective pond area for HadGEM
        if icepack_tracers.tr_pond_topo:
            for n in range(ncat):
                apeffn[n] = c0
                if aicen[n] > puny:
                    # Lid effective if thicker than hp1
                    if apndn[n] * aicen[n] > puny and ipndn[n] < hp1:
                        apeffn[n] = apndn[n]
                    else:
                        apeffn[n] = c0
                    if apndn[n] < puny:
                        apeffn[n] = c0

        # Initialize for safety
        for n in range(ncat):
            alvdrn[n] = c0
            alidrn[n] = c0
            alvdfn[n] = c0
            alidfn[n] = c0
            fswsfcn[n] = c0
            fswintn[n] = c0
            fswthrun[n] = c0
        Iswabsn[:, :] = c0
        Sswabsn[:, :] = c0

    _step_first_call = False
    return coszen_out


def run_dEdd(
    dt, aicen, vicen, vsnon, Tsfcn,
    alvln, apndn, hpndn, ipndn, aeron, trcrn_bgcsw,
    TLAT, TLON, calendar_type, days_per_year, nextsw_cday, yday, sec,
    swvdr, swvdf, swidr, swidf, coszen, fsnow,
    alvdrn, alvdfn, alidrn, alidfn,
    fswsfcn, fswintn,
    swuvrdr, swuvrdf, swpardr, swpardf,
    fswthrun,
    fswthrun_vdr, fswthrun_vdf, fswthrun_idr, fswthrun_idf,
    fswthrun_uvrdr, fswthrun_uvrdf, fswthrun_pardr, fswthrun_pardf,
    fswpenln, Sswabsn, Iswabsn,
    albicen, albsnon, albpndn, apeffn, snowfracn,
    dhsn, ffracn, rsnow, l_print_point, initonly,
):
    """Compute initial data for Delta-Eddington method, specifically,
    the approximate exponential look-up table.

    author:  Bruce P. Briegleb, NCAR
    2011 ECH modified for melt pond tracers
    2013 ECH merged with NCAR version
    2024 DCS refactored for sealvl ponds
    """
    ncat = int(icepack_tracers.ncat)
    nslyr = int(icepack_tracers.nslyr)
    l_initonly = False
    if initonly is not None:
        l_initonly = bool(initonly)

    l_rsnows = np.zeros(nslyr, dtype=dbl_kind)

    # cosine of the zenith angle
    # CESMCOUPLED is not defined.
    # GEOS sets solar angles in driver level when semi_implicit_Tsfc is true.
    coszen_out = coszen
    if not semi_implicit_Tsfc:
        coszen_box = np.array(coszen, dtype=dbl_kind)
        coszen_out = compute_coszen(
            TLAT, TLON, yday, sec, coszen_box,
            days_per_year, nextsw_cday, calendar_type,
        )

    for n in range(ncat):
        # note that rhosnwn, rsnw, fp, hp and Sswabs ARE NOT dimensioned with ncat
        # BPB 19 Dec 2006

        # set snow properties
        fsn = c0
        hsn = c0
        rhosnwn = np.zeros(nslyr, dtype=dbl_kind)
        rsnwn = np.zeros(nslyr, dtype=dbl_kind)
        apeffn[n] = c0  # for history
        snowfracn[n] = c0  # for history

        if aicen[n] > puny:
            if snwgrain:
                l_rsnows[:] = rsnow[:, n]
            fsn, hsn = shortwave_dEdd_set_snow(
                R_snw, dT_mlt, rsnw_mlt,
                aicen[n], vsnon[n], Tsfcn[n], fsn, icepack_parameters.hs0, hsn,
                rhosnwn, rsnwn, l_rsnows,
            )

            # set pond properties
            if icepack_tracers.tr_pond_lvl:
                apond = alvln[n] * apndn[n]
                fsn, hsn, dhsn[n], fpn, hpn, apeffn[n] = shortwave_dEdd_set_eff(
                    aicen[n], vsnon[n], alvln[n], apond, hpndn[n], ipndn[n],
                    ffracn[n], fsnow, dt, Tsfcn[n],
                    fsn, hsn, dhsn[n], l_rsnows, rhosnwn, rsnwn, l_initonly,
                )
            elif icepack_tracers.tr_pond_sealvl:
                apond = apndn[n]
                fsn, hsn, dhsn[n], fpn, hpn, apeffn[n] = shortwave_dEdd_set_eff(
                    aicen[n], vsnon[n], alvln[n], apond, hpndn[n], ipndn[n],
                    ffracn[n], fsnow, dt, Tsfcn[n],
                    fsn, hsn, dhsn[n], l_rsnows, rhosnwn, rsnwn, l_initonly,
                )
            elif icepack_tracers.tr_pond_topo:
                # Lid effective if thicker than hp1
                if apndn[n] * aicen[n] > puny and ipndn[n] < hp1:
                    fpn = apndn[n]
                else:
                    fpn = c0
                if apndn[n] > puny:
                    hpn = hpndn[n]
                else:
                    fpn = c0
                    hpn = c0

                # Zero out fraction of thin ponds for radiation only
                if hpn < hpmin:
                    fpn = c0

                # If ponds are present snow fraction reduced to
                # non-ponded part dEdd scheme
                fsn = min(fsn, c1 - fpn)
                apeffn[n] = fpn
            else:
                fpn = c0
                hpn = c0
                fpn, hpn = shortwave_dEdd_set_pond(Tsfcn[n], fsn)
                apeffn[n] = fpn  # for history
                fpn = c0
                hpn = c0

            snowfracn[n] = fsn  # for history

            (
                alvdrn[n], alvdfn[n], alidrn[n], alidfn[n],
                fswsfcn[n], fswintn[n], fswthrun[n],
                l_fswthru_vdr, l_fswthru_vdf, l_fswthru_idr, l_fswthru_idf,
                l_fswthru_uvrdr, l_fswthru_uvrdf, l_fswthru_pardr, l_fswthru_pardf,
                albicen[n], albsnon[n], albpndn[n],
                coszen_out,
            ) = shortwave_dEdd(
                coszen_out, aicen[n], vicen[n], hsn, fsn,
                rhosnwn, rsnwn, fpn, hpn, aeron[:, n],
                swvdr, swvdf, swidr, swidf,
                swuvrdr, swuvrdf, swpardr, swpardf,
                Sswabsn[:, n], Iswabsn[:, n], fswpenln[:, n],
                trcrn_bgcsw[:, n], l_print_point,
            )

            if fswthrun_vdr is not None:
                fswthrun_vdr[n] = l_fswthru_vdr
            if fswthrun_vdf is not None:
                fswthrun_vdf[n] = l_fswthru_vdf
            if fswthrun_idr is not None:
                fswthrun_idr[n] = l_fswthru_idr
            if fswthrun_idf is not None:
                fswthrun_idf[n] = l_fswthru_idf
            if fswthrun_uvrdr is not None:
                fswthrun_uvrdr[n] = l_fswthru_uvrdr
            if fswthrun_uvrdf is not None:
                fswthrun_uvrdf[n] = l_fswthru_uvrdf
            if fswthrun_pardr is not None:
                fswthrun_pardr[n] = l_fswthru_pardr
            if fswthrun_pardf is not None:
                fswthrun_pardf[n] = l_fswthru_pardf

            if rsnow is not None and (not snwgrain):
                for k in range(nslyr):
                    rsnow[k, n] = rsnwn[k]  # for history

    if isinstance(coszen, np.ndarray):
        coszen[...] = coszen_out
    return coszen_out


def shortwave_dEdd(
    coszen, aice, vice, hs, fs, rhosnw, rsnw, fp, hp, aero,
    swvdr, swvdf, swidr, swidf,
    swuvrdr, swuvrdf, swpardr, swpardf,
    Sswabs, Iswabs, fswpenl, zbio, l_print_point,
):
    """Compute snow/bare ice/ponded ice shortwave albedos, absorbed and transmitted
    flux using the Delta-Eddington solar radiation method.

    NOTE regarding albedo diagnostics:  This method yields zero albedo values
    if there is no incoming solar and thus the albedo diagnostics are masked
    out when the sun is below the horizon.  To estimate albedo from the history
    output (post-processing), compute ice albedo using
    (1 - albedo)*swdn = swabs. -ECH

    author:  Bruce P. Briegleb, NCAR
      2013:  E Hunke merged with NCAR version
    """
    subname = "(shortwave_dEdd)"
    nslyr = int(icepack_tracers.nslyr)
    nilyr = int(icepack_tracers.nilyr)
    n_aero = int(icepack_tracers.n_aero)

    klev = nslyr + nilyr + 1  # number of radiation layers - 1
    klevp = klev + 1  # number of radiation interfaces - 1
                      # (0 layer is included also)

    # set storage albedos and fluxes to zero for accumulation over surface types
    hstmp = c0
    hi = c0
    fi = c0
    alvdr = c0
    alvdf = c0
    alidr = c0
    alidf = c0
    fswsfc = c0
    fswint = c0
    fswthru = c0
    fswthru_vdr = c0
    fswthru_vdf = c0
    fswthru_idr = c0
    fswthru_idf = c0
    fswthru_uvrdr = c0
    fswthru_uvrdf = c0
    fswthru_pardr = c0
    fswthru_pardf = c0
    # compute fraction of nir down direct to total over all points:
    fnidr = c0
    if swidr + swidf > puny:
        fnidr = swidr / (swidr + swidf)
    albice = c0
    albsno = c0
    albpnd = c0
    fswpenl[:] = c0
    Sswabs[:] = c0
    Iswabs[:] = c0

    # compute aerosol mass path
    aero_mp = np.zeros(4 * n_aero, dtype=dbl_kind)
    if icepack_tracers.tr_aero:
        # check 4 layers for each aerosol, a snow SSL, snow below SSL,
        # sea ice SSL, and sea ice below SSL, in that order.
        if np.size(aero) < 4 * n_aero:
            raise RuntimeError(subname + " ERROR: size(aero) too small")
        na = 1
        while na <= 4 * n_aero:
            vsno = hs * aice
            netsw = swvdr + swidr + swvdf + swidf
            if netsw > puny:  # sun above horizon
                aero_mp[na - 1] = aero[na - 1] * vsno
                aero_mp[na] = aero[na] * vsno
                aero_mp[na + 1] = aero[na + 1] * vice
                aero_mp[na + 2] = aero[na + 2] * vice
            na = na + 4

    # compute shortwave radiation accounting for snow/ice (both snow over
    # ice and bare ice) and ponded ice (if any):

    # sea ice points with sun above horizon
    netsw = swvdr + swidr + swvdf + swidf
    if netsw > puny:  # sun above horizon
        coszen = max(puny, coszen)
        # evaluate sea ice thickness and fraction
        hi = vice / aice
        fi = c1 - fs - fp
        # bare sea ice points
        if fi > c0:
            # calculate bare sea ice
            srftyp = 0
            (
                avdrl, avdfl, aidrl, aidfl,
                fswsfc, fswint, fswthru,
                fswthru_vdr, fswthru_vdf, fswthru_idr, fswthru_idf,
                fswthru_uvrdr, fswthru_uvrdf, fswthru_pardr, fswthru_pardf,
            ) = compute_dEdd_3bd(
                klev, klevp, zbio, fnidr, coszen,
                swvdr, swvdf, swidr, swidf, srftyp,
                hstmp, rhosnw, rsnw, hi, hp, fi, aero_mp,
                fswsfc, fswint, fswthru,
                fswthru_vdr, fswthru_vdf, fswthru_idr, fswthru_idf,
                fswthru_uvrdr, fswthru_uvrdf, fswthru_pardr, fswthru_pardf,
                Sswabs, Iswabs, fswpenl,
                swuvrdr, swuvrdf, swpardr, swpardf,
            )
            alvdr = alvdr + avdrl * fi
            alvdf = alvdf + avdfl * fi
            alidr = alidr + aidrl * fi
            alidf = alidf + aidfl * fi
            # for history
            albice = albice + awtvdr * avdrl + awtidr * aidrl + awtvdf * avdfl + awtidf * aidfl

    # sea ice points with sun above horizon
    netsw = swvdr + swidr + swvdf + swidf
    if netsw > puny:  # sun above horizon
        coszen = max(puny, coszen)
        # snow-covered sea ice points
        if fs > c0:
            # calculate snow covered sea ice
            srftyp = 1
            if shortwave.strip() == "dEdd_snicar_ad":
                (
                    avdrl, avdfl, aidrl, aidfl,
                    fswsfc, fswint, fswthru,
                    fswthru_vdr, fswthru_vdf, fswthru_idr, fswthru_idf,
                ) = compute_dEdd_5bd(
                    klev, klevp, zbio, fnidr, coszen,
                    swvdr, swvdf, swidr, swidf, srftyp,
                    hs, rhosnw, rsnw, hi, hp, fs, aero_mp,
                    fswsfc, fswint, fswthru,
                    fswthru_vdr, fswthru_vdf, fswthru_idr, fswthru_idf,
                    Sswabs, Iswabs, fswpenl,
                )
            else:
                # echmod - this can be combined with the 5bd call above, if we use module data
                (
                    avdrl, avdfl, aidrl, aidfl,
                    fswsfc, fswint, fswthru,
                    fswthru_vdr, fswthru_vdf, fswthru_idr, fswthru_idf,
                    fswthru_uvrdr, fswthru_uvrdf, fswthru_pardr, fswthru_pardf,
                ) = compute_dEdd_3bd(
                    klev, klevp, zbio, fnidr, coszen,
                    swvdr, swvdf, swidr, swidf, srftyp,
                    hs, rhosnw, rsnw, hi, hp, fs, aero_mp,
                    fswsfc, fswint, fswthru,
                    fswthru_vdr, fswthru_vdf, fswthru_idr, fswthru_idf,
                    fswthru_uvrdr, fswthru_uvrdf, fswthru_pardr, fswthru_pardf,
                    Sswabs, Iswabs, fswpenl,
                    swuvrdr, swuvrdf, swpardr, swpardf,
                )
            alvdr = alvdr + avdrl * fs
            alvdf = alvdf + avdfl * fs
            alidr = alidr + aidrl * fs
            alidf = alidf + aidfl * fs
            # for history
            albsno = albsno + awtvdr * avdrl + awtidr * aidrl + awtvdf * avdfl + awtidf * aidfl

    hi = c0

    # sea ice points with sun above horizon
    netsw = swvdr + swidr + swvdf + swidf
    if netsw > puny:  # sun above horizon
        coszen = max(puny, coszen)
        hi = vice / aice
        # if nonzero pond fraction and sufficient pond depth
        # if( fp > puny .and. hp > hpmin ) then
        if fp > puny:
            # calculate ponded ice
            srftyp = 2
            (
                avdrl, avdfl, aidrl, aidfl,
                fswsfc, fswint, fswthru,
                fswthru_vdr, fswthru_vdf, fswthru_idr, fswthru_idf,
                fswthru_uvrdr, fswthru_uvrdf, fswthru_pardr, fswthru_pardf,
            ) = compute_dEdd_3bd(
                klev, klevp, zbio, fnidr, coszen,
                swvdr, swvdf, swidr, swidf, srftyp,
                hs, rhosnw, rsnw, hi, hp, fp, aero_mp,
                fswsfc, fswint, fswthru,
                fswthru_vdr, fswthru_vdf, fswthru_idr, fswthru_idf,
                fswthru_uvrdr, fswthru_uvrdf, fswthru_pardr, fswthru_pardf,
                Sswabs, Iswabs, fswpenl,
                swuvrdr, swuvrdf, swpardr, swpardf,
            )
            alvdr = alvdr + avdrl * fp
            alvdf = alvdf + avdfl * fp
            alidr = alidr + aidrl * fp
            alidf = alidf + aidfl * fp
            # for history
            albpnd = albpnd + awtvdr * avdrl + awtidr * aidrl + awtvdf * avdfl + awtidf * aidfl

    # if no incoming shortwave, set albedos to 1
    netsw = swvdr + swidr + swvdf + swidf
    if netsw <= puny:  # sun above horizon
        alvdr = c1
        alvdf = c1
        alidr = c1
        alidf = c1

    if l_print_point and netsw > puny:
        _diag(subname + " printing point")
        _diag(subname + " coszen = " + str(float(coszen)))
        _diag(subname + " swvdr  swvdf = " + str(float(swvdr)) + " " + str(float(swvdf)))
        _diag(subname + " swidr  swidf = " + str(float(swidr)) + " " + str(float(swidf)))
        _diag(subname + " aice = " + str(float(aice)))
        _diag(subname + " hs = " + str(float(hs)))
        _diag(subname + " hp = " + str(float(hp)))
        _diag(subname + " fs = " + str(float(fs)))
        _diag(subname + " fi = " + str(float(fi)))
        _diag(subname + " fp = " + str(float(fp)))
        _diag(subname + " hi = " + str(float(hi)))
        _diag(subname + " alvdr  alvdf = " + str(float(alvdr)) + " " + str(float(alvdf)))
        _diag(subname + " alidr  alidf = " + str(float(alidr)) + " " + str(float(alidf)))
        _diag(
            subname
            + " fswsfc fswint fswthru = "
            + str(float(fswsfc)) + " " + str(float(fswint)) + " " + str(float(fswthru))
        )
        swdn = swvdr + swvdf + swidr + swidf
        swab = fswsfc + fswint + fswthru
        swalb = (1.0 - swab / (swdn + 0.0001))
        _diag(subname + " swdn swab swalb = " + str(float(swdn)) + " " + str(float(swab)) + " " + str(float(swalb)))
        for k in range(1, nslyr + 1):
            _diag(
                subname + " snow layer k    = " + str(k)
                + " rhosnw = " + str(float(rhosnw[k - 1]))
                + " rsnw = " + str(float(rsnw[k - 1]))
            )
        for k in range(1, nslyr + 1):
            _diag(
                subname + " snow layer k    = " + str(k)
                + " Sswabs(k)       = " + str(float(Sswabs[k - 1]))
            )
        for k in range(1, nilyr + 1):
            _diag(
                subname + " sea ice layer k = " + str(k)
                + " Iswabs(k)       = " + str(float(Iswabs[k - 1]))
            )

    return (
        alvdr, alvdf, alidr, alidf,
        fswsfc, fswint, fswthru,
        fswthru_vdr, fswthru_vdf, fswthru_idr, fswthru_idf,
        fswthru_uvrdr, fswthru_uvrdf, fswthru_pardr, fswthru_pardf,
        albice, albsno, albpnd,
        coszen,
    )


# Delta-Eddington solution expressions


def alpha(w, uu, gg, e):
    return p75 * w * uu * ((c1 + gg * (c1 - w)) / (c1 - e * e * uu * uu))


def agamm(w, uu, gg, e):
    return p5 * w * ((c1 + c3 * gg * (c1 - w) * uu * uu) / (c1 - e * e * uu * uu))


def n(uu, et):
    return ((uu + c1) * (uu + c1) / et) - ((uu - c1) * (uu - c1) * et)


def u(w, gg, e):
    return c1p5 * (c1 - w * gg) / e


def el(w, gg):
    return math.sqrt(c3 * (c1 - w) * (c1 - w * gg))


def taus(w, f, t):
    return (c1 - w * f) * t


def omgs(w, f):
    return (c1 - f) * w / (c1 - w * f)


def asys(gg, f):
    return (gg - f) / (c1 - f)


def solution_dEdd(coszen, srftyp, klev, klevp, tau, w0, g, albodr, albodf):
    """Given input vertical profiles of optical properties, evaluate the
    monochromatic Delta-Eddington solution.

    author:  Bruce P. Briegleb, NCAR
      2013:  E Hunke merged with NCAR version

    Returns ``(trndir, trntdr, trndif, rupdir, rupdif, rdndif)``.
    Layer indexes keep the Fortran lower bound of 0.
    """
    # minimum total transmission allowed
    # total transmission is that due to the direct beam; i.e. it includes
    # both the directly transmitted solar beam and the diffuse downwards
    # transmitted radiation resulting from scattering out of the direct beam
    trmin = dbl_kind(0.001)

    # refractive index for sea ice, water; pre-computed, band-independent,
    # diffuse fresnel reflectivities
    refindx = dbl_kind(1.310)  # refractive index of sea ice (water also)
    cp063 = dbl_kind(0.063)  # diffuse fresnel reflectivity from above
    cp455 = dbl_kind(0.455)  # diffuse fresnel reflectivity from below

    ngmax = 8  # number of gaussian angles in hemisphere
    gauspt = np.array(  # gaussian angles (radians)
        [
            0.9894009, 0.9445750, 0.8656312, 0.7554044,
            0.6178762, 0.4580168, 0.2816036, 0.0950125,
        ],
        dtype=dbl_kind,
    )
    gauswt = np.array(  # gaussian weights
        [
            0.0271525, 0.0622535, 0.0951585, 0.1246290,
            0.1495960, 0.1691565, 0.1826034, 0.1894506,
        ],
        dtype=dbl_kind,
    )

    trndir = np.zeros(klevp + 1, dtype=dbl_kind)
    trntdr = np.zeros(klevp + 1, dtype=dbl_kind)
    trndif = np.zeros(klevp + 1, dtype=dbl_kind)
    rupdir = np.zeros(klevp + 1, dtype=dbl_kind)
    rupdif = np.zeros(klevp + 1, dtype=dbl_kind)
    rdndif = np.zeros(klevp + 1, dtype=dbl_kind)

    rdir = np.zeros(klev + 1, dtype=dbl_kind)
    rdif_a = np.zeros(klev + 1, dtype=dbl_kind)
    rdif_b = np.zeros(klev + 1, dtype=dbl_kind)
    tdir = np.zeros(klev + 1, dtype=dbl_kind)
    tdif_a = np.zeros(klev + 1, dtype=dbl_kind)
    tdif_b = np.zeros(klev + 1, dtype=dbl_kind)
    trnlay = np.zeros(klev + 1, dtype=dbl_kind)

    # initialize top interface of top layer
    trndir[0] = c1
    trntdr[0] = c1
    trndif[0] = c1
    rdndif[0] = c0

    # mu0 is cosine solar zenith angle above the fresnel level; make
    # sure mu0 is large enough for stable and meaningful radiation
    # solution: .01 is like sun just touching horizon with its lower edge
    mu0 = max(coszen, p01)

    # mu0n is cosine solar zenith angle used to compute the layer
    # Delta-Eddington solution; it is initially computed to be the
    # value below the fresnel level, i.e. the cosine solar zenith
    # angle below the fresnel level for the refracted solar beam:
    mu0nij = math.sqrt(c1 - ((c1 - mu0 ** 2) / (refindx * refindx)))

    # compute level of fresnel refraction
    # if ponded sea ice, fresnel level is the top of the pond.
    kfrsnl = 0
    # if snow over sea ice or bare sea ice, fresnel level is
    # at base of sea ice SSL (and top of the sea ice DL); the
    # snow SSL counts for one, then the number of snow layers,
    # then the sea ice SSL which also counts for one:
    nslyr = int(icepack_tracers.nslyr)
    if srftyp < 2:
        kfrsnl = nslyr + 2

    # proceed down one layer at a time; if the total transmission to
    # the interface just above a given layer is less than trmin, then no
    # Delta-Eddington computation for that layer is done.

    # begin main level loop
    for k in range(0, klev + 1):
        # initialize all layer apparent optical properties to 0
        rdir[k] = c0
        rdif_a[k] = c0
        rdif_b[k] = c0
        tdir[k] = c0
        tdif_a[k] = c0
        tdif_b[k] = c0
        trnlay[k] = c0

        # compute next layer Delta-eddington solution only if total transmission
        # of radiation to the interface just above the layer exceeds trmin.
        if trntdr[k] > trmin:
            # calculation over layers with penetrating radiation
            tautot = tau[k]
            wtot = w0[k]
            gtot = g[k]
            ftot = gtot * gtot

            ts = taus(wtot, ftot, tautot)
            ws = omgs(wtot, ftot)
            gs = asys(gtot, ftot)
            lm = el(ws, gs)
            ue = u(ws, gs, lm)

            mu0n = mu0nij
            # if level k is above fresnel level and the cell is non-pond, use the
            # non-refracted beam instead
            if srftyp < 2 and k < kfrsnl:
                mu0n = mu0

            exp_min = min(exp_argmax, lm * ts)
            extins = math.exp(-exp_min)
            ne = n(ue, extins)

            # first calculation of rdif, tdif using Delta-Eddington formulas
            # rdif_a(k) = (ue+c1)*(ue-c1)*(c1/extins - extins)/ne
            rdif_a[k] = (ue ** 2 - c1) * (c1 / extins - extins) / ne
            tdif_a[k] = c4 * ue / ne

            # evaluate rdir,tdir for direct beam
            exp_min = min(exp_argmax, ts / mu0n)
            trnlay[k] = math.exp(-exp_min)
            alp = alpha(ws, mu0n, gs, lm)
            gam = agamm(ws, mu0n, gs, lm)
            apg = alp + gam
            amg = alp - gam
            rdir[k] = apg * rdif_a[k] + amg * (tdif_a[k] * trnlay[k] - c1)
            tdir[k] = apg * tdif_a[k] + (amg * rdif_a[k] - apg + c1) * trnlay[k]

            # recalculate rdif,tdif using direct angular integration over rdir,tdir,
            # since Delta-Eddington rdif formula is not well-behaved (it is usually
            # biased low and can even be negative); use ngmax angles and gaussian
            # integration for most accuracy:
            R1 = rdif_a[k]  # use R1 as temporary
            T1 = tdif_a[k]  # use T1 as temporary
            swt = c0
            smr = c0
            smt = c0
            for ng in range(ngmax):  # Fortran ng = 1, ngmax
                mu = gauspt[ng]
                gwt = gauswt[ng]
                swt = swt + mu * gwt
                exp_min = min(exp_argmax, ts / mu)
                trn = math.exp(-exp_min)
                alp = alpha(ws, mu, gs, lm)
                gam = agamm(ws, mu, gs, lm)
                apg = alp + gam
                amg = alp - gam
                rdr = apg * R1 + amg * T1 * trn - amg
                tdr = apg * T1 + amg * R1 * trn - apg * trn + trn
                smr = smr + mu * rdr * gwt
                smt = smt + mu * tdr * gwt
            rdif_a[k] = smr / swt
            tdif_a[k] = smt / swt

            # homogeneous layer
            rdif_b[k] = rdif_a[k]
            tdif_b[k] = tdif_a[k]

            # add fresnel layer to top of desired layer if either
            # air or snow overlies ice; we ignore refraction in ice
            # if a melt pond overlies it:
            if k == kfrsnl:
                # compute fresnel reflection and transmission amplitudes
                # for two polarizations: 1=perpendicular and 2=parallel to
                # the plane containing incident, reflected and refracted rays.
                R1 = (mu0 - refindx * mu0n) / (mu0 + refindx * mu0n)
                R2 = (refindx * mu0 - mu0n) / (refindx * mu0 + mu0n)
                T1 = c2 * mu0 / (mu0 + refindx * mu0n)
                T2 = c2 * mu0 / (refindx * mu0 + mu0n)

                # unpolarized light for direct beam
                Rf_dir_a = p5 * (R1 * R1 + R2 * R2)
                Tf_dir_a = p5 * (T1 * T1 + T2 * T2) * refindx * mu0n / mu0

                # precalculated diffuse reflectivities and transmissivities
                # for incident radiation above and below fresnel layer, using
                # the direct albedos and accounting for complete internal
                # reflection from below; precalculated because high order
                # number of gaussian points (~256) is required for convergence:

                # above
                Rf_dif_a = cp063
                Tf_dif_a = c1 - Rf_dif_a
                # below
                Rf_dif_b = cp455
                Tf_dif_b = c1 - Rf_dif_b

                # the k = kfrsnl layer properties are updated to combined
                # the fresnel (refractive) layer, always taken to be above
                # the present layer k (i.e. be the top interface):

                rintfc = c1 / (c1 - Rf_dif_b * rdif_a[k])
                tdir[k] = (
                    Tf_dir_a * tdir[k]
                    + Tf_dir_a * rdir[k] * Rf_dif_b * rintfc * tdif_a[k]
                )
                rdir[k] = Rf_dir_a + Tf_dir_a * rdir[k] * rintfc * Tf_dif_b
                rdif_a[k] = Rf_dif_a + Tf_dif_a * rdif_a[k] * rintfc * Tf_dif_b
                rdif_b[k] = rdif_b[k] + tdif_b[k] * Rf_dif_b * rintfc * tdif_a[k]
                tdif_a[k] = tdif_a[k] * rintfc * Tf_dif_a
                tdif_b[k] = tdif_b[k] * rintfc * Tf_dif_b

                # update trnlay to include fresnel transmission
                trnlay[k] = Tf_dir_a * trnlay[k]

        # initialize current layer properties to zero; only if total
        # transmission to the top interface of the current layer exceeds the
        # minimum, will these values be computed below.
        # note that we ignore refraction between sea ice and underlying ocean:
        trndir[k + 1] = trndir[k] * trnlay[k]
        refkm1 = c1 / (c1 - rdndif[k] * rdif_a[k])
        tdrrdir = trndir[k] * rdir[k]
        tdndif = trntdr[k] - trndir[k]
        trntdr[k + 1] = trndir[k] * tdir[k] + (tdndif + tdrrdir * rdndif[k]) * refkm1 * tdif_a[k]
        rdndif[k + 1] = rdif_b[k] + (tdif_b[k] * rdndif[k] * refkm1 * tdif_a[k])
        trndif[k + 1] = trndif[k] * refkm1 * tdif_a[k]

    # compute reflectivity to direct and diffuse radiation for layers
    # below by adding succesive layers starting from the underlying
    # ocean and working upwards:
    rupdir[klevp] = albodr
    rupdif[klevp] = albodf

    for k in range(klev, -1, -1):
        # interface scattering
        refkp1 = c1 / (c1 - rdif_b[k] * rupdif[k + 1])
        # dir from top layer plus exp tran ref from lower layer, interface
        # scattered and tran thru top layer from below, plus diff tran ref
        # from lower layer with interface scattering tran thru top from below
        rupdir[k] = rdir[k] + (
            trnlay[k] * rupdir[k + 1]
            + (tdir[k] - trnlay[k]) * rupdif[k + 1]
        ) * refkp1 * tdif_b[k]
        # dif from top layer from above, plus dif tran upwards reflected and
        # interface scattered which tran top from below
        rupdif[k] = rdif_a[k] + tdif_a[k] * rupdif[k + 1] * refkp1 * tdif_b[k]

    return trndir, trntdr, trndif, rupdir, rupdif, rdndif


def shortwave_search(val, array):
    """Search ``array`` for ``val`` and return Fortran 1-based ``nr`` such that
    ``array(nr-1) < val <= array(nr)``.

    If nr cannot be found, an error is thrown.
    This does NOT check that array is sorted because it would be too expensive,
    but it must be sorted to work properly.
    """
    subname = "(shortwave_search)"
    if str(swdata.rsnw_datatype)[:6] != "sorted":
        raise RuntimeError(
            subname + " rsnw_datatype not valid: " + str(swdata.rsnw_datatype).strip()
        )

    nrsize = len(array)
    if nrsize > 10:
        # binary search
        nrl = 1
        nru = nrsize
        nr = (nrl + nru) // 2
        found = False
        nrcnt = 0
        while (not found) and nrcnt < nrsize:
            nrcnt = nrcnt + 1
            if val > array[nr - 1]:
                # Fortran array(nr+1) is past a 1-based table when nr = nrsize.
                if nr >= nrsize:
                    raise RuntimeError(subname + " ERROR: binary search failed")
                if val < array[nr]:  # Fortran array(nr+1)
                    found = True
                    nr = nr + 1
                else:
                    nrl = nr + 1
                    nr = (nrl + nru) // 2
            else:
                # Fortran array(nr-1) is below a 1-based table when nr = 1.
                if nr <= 1:
                    raise RuntimeError(subname + " ERROR: binary search failed")
                if val > array[nr - 2]:  # Fortran array(nr-1)
                    found = True
                else:
                    nru = nr - 1
                    nr = (nrl + nru) // 2
        if not found:
            raise RuntimeError(subname + " ERROR: binary search failed")
    else:
        # linear search
        nr = -1
        for nrcnt in range(2, nrsize + 1):
            if val > array[nrcnt - 2] and val < array[nrcnt - 1]:
                nr = nrcnt
                break
        if nr < 1:
            raise RuntimeError(subname + " ERROR: linear search failed")
    return nr


def _radiation_dzk(klev, hs, hi):
    """Snow and ice radiation-layer thicknesses. Indexes keep the lower bound 0."""
    nslyr = int(icepack_tracers.nslyr)
    nilyr = int(icepack_tracers.nilyr)
    rnilyr = c1 / dbl_kind(nilyr)
    rnslyr = c1 / dbl_kind(nslyr)
    kii = nslyr + 1  # level starting index for sea ice (nslyr+1)
    dzk = np.zeros(klev + 1, dtype=dbl_kind)

    # layer thicknesses
    # snow
    dz = hs * rnslyr
    # for small enough snow thickness, ssl thickness half of top snow layer
    # ech: note this is highly resolution dependent!
    dzk[0] = min(hs_ssl, dz / c2)
    dzk[1] = dz - dzk[0]
    if nslyr > 1:
        for k in range(2, nslyr + 1):
            dzk[k] = dz

    # ice
    dz = hi * rnilyr
    # empirical reduction in sea ice ssl thickness for ice thinner than 1.5m;
    # factor of 30 gives best albedo comparison with limited observations
    dz_ssl = hi_ssl
    # ech: note hardwired parameters
    # if hi < 1.5_dbl_kind: dz_ssl = hi/30._dbl_kind
    dz_ssl = min(hi_ssl, hi / dbl_kind(30.0))
    # set sea ice ssl thickness to half top layer if sea ice thin enough
    # ech: note this is highly resolution dependent!
    dz_ssl = min(dz_ssl, dz / c2)

    dzk[kii] = dz_ssl
    dzk[kii + 1] = dz - dz_ssl
    if kii + 2 <= klev:
        for k in range(kii + 2, klev + 1):
            dzk[k] = dz
    return dzk, kii, rnilyr, rnslyr


def _tune_scat(ki_mn, wi_mn, gi_mn, fp, fm, radius_factor):
    """Tune the scattering coefficient. +R and -R use different fractions.

    R values of +1 correspond approximately to +1 sigma changes in albedo, and
    R values of -1 correspond approximately to -1 sigma changes in albedo.
    Note: the albedo change becomes non-linear for R values > +1 or < -1.
    """
    nband = len(ki_mn)
    ki = np.empty(nband, dtype=dbl_kind)
    wi = np.empty(nband, dtype=dbl_kind)
    gi = np.empty(nband, dtype=dbl_kind)
    if radius_factor >= c0:
        factor = fp * radius_factor
        clamp = False
    else:  # R < 0
        factor = fm * radius_factor
        clamp = True
    for ns in range(nband):
        sigp = ki_mn[ns] * wi_mn[ns] * (c1 + factor)
        if clamp:
            sigp = max(sigp, c0)
        ki[ns] = sigp + ki_mn[ns] * (c1 - wi_mn[ns])
        wi[ns] = sigp / ki[ns]
        gi[ns] = gi_mn[ns]
    return ki, wi, gi


def _grain_index(tmp_gs):
    """Snow grain-size index for modal black carbon. Returns a 1-based index."""
    # grain size index
    # works for 25 < snw_rds < 1625 um (5-band note; the same cuts are used in 3-band)
    if tmp_gs < dbl_kind(125.0):
        tmp1 = tmp_gs / dbl_kind(50.0)
        idx = _nint(tmp1)
    elif tmp_gs < dbl_kind(175.0):
        idx = 2
    else:
        tmp1 = (tmp_gs / dbl_kind(250.0)) + c2
        idx = _nint(tmp1)
    # check bounds
    if idx < 1:
        idx = 1
    if idx > 8:
        idx = 8
    # if k_bcins < 1: k_bcins = 1   ! hardwired
    # if k_bcins > 10: k_bcins = 10
    # if k_bcexs < 1: k_bcexs = 1
    # if k_bcexs > 10: k_bcexs = 10
    return idx


def _modal_bc_indices(klev, snow_radius):
    """Modal BC indices on radiation levels 0..klev.

    ``snow_radius`` is the snow-layer radius (frsnw in 3-band, rsnw in 5-band).
    """
    nslyr = int(icepack_tracers.nslyr)
    k_bcini = np.zeros(klev + 1, dtype=np.int32)
    k_bcins = np.zeros(klev + 1, dtype=np.int32)
    k_bcexs = np.zeros(klev + 1, dtype=np.int32)
    if modal_aero:
        for k in range(0, klev + 1):
            if k < nslyr + 1:  # define indices for snow layer
                # use top rsnw, rhosnw for snow ssl and rest of top layer
                ksnow = max(k, 1)
                tmp_gs = snow_radius[ksnow - 1]
                k_bcini[k] = _grain_index(tmp_gs)
            else:  # use the largest snow grain size for ice
                k_bcini[k] = 8
            # Set index corresponding to BC effective radius.  Here,
            # asssume constant BC effective radius of 100nm
            # (corresponding to index 2)
            k_bcins[k] = 2  # hardwired
            k_bcexs[k] = 2  # hardwired
            # check bounds
            if k_bcini[k] < 1:
                k_bcini[k] = 1
            if k_bcini[k] > 8:
                k_bcini[k] = 8
    return k_bcini, k_bcins, k_bcexs


def _chlorophyll(nspint, klev, dzk, zbio):
    """kabs_chl(ns, k). Only visible band 1 is nonzero."""
    # chla-specific absorption coefficient
    # 0.0023-0.0029 Perovich 1993, also 0.0067 m^2 (mg Chl)^-1
    # found values of 0.006 to 0.023 m^2/ mg  (676 nm)  Neukermans 2014
    # and averages over the 300-700nm of 0.0075 m^2/mg in ice Fritsen (2011)
    # at 440nm values as high as 0.2 m^2/mg in under ice bloom (Balch 2014)
    # Grenfell 1991 uses 0.004 (m^2/mg) which is (0.0078 * spectral weighting)
    # chlorophyll mass extinction cross section (m^2/mg chla)
    kchl_tab = p01
    kabs_chl = np.zeros((nspint, klev + 1), dtype=dbl_kind)
    if _tr("tr_bgc_N", False) and dEdd_algae:  # compute kabs_chl for chlorophyll
        nlt_chl_sw = int(_tr("nlt_chl_sw"))
        for k in range(0, klev + 1):
            # zbio(nlt_chl_sw+k), k is a radiation level starting at 0
            kabs_chl[0, k] = kchl_tab * zbio[nlt_chl_sw + k - 1]
    else:
        k = klev
        kabs_chl[0, k] = kalg * (dbl_kind(0.50) / dzk[k])
    return kabs_chl


def _fill_zaero(
    nspint, klev, dzk, zbio, k_bcini, k_bcins, k_bcexs,
    kaer_bc, waer_bc, gaer_bc, bcenh, kaer, waer, gaer,
):
    """Column aerosol optical depth from z-tracers. Spectral index is 0-based in the arrays."""
    tzaer = np.zeros((nspint, klev + 1), dtype=dbl_kind)
    wzaer = np.zeros((nspint, klev + 1), dtype=dbl_kind)
    gzaer = np.zeros((nspint, klev + 1), dtype=dbl_kind)
    if not (_tr("tr_zaero", False) and dEdd_algae):
        return tzaer, wzaer, gzaer
    nlt_zaero_sw = _tr("nlt_zaero_sw")
    n_zaero = int(icepack_tracers.n_zaero)
    for n in range(1, n_zaero + 1):  # Fortran n = 1, n_zaero
        nlt = int(nlt_zaero_sw[n - 1])
        for k in range(0, klev + 1):
            # zbio(nlt_zaero_sw(n)+k); not weighted by aice
            mass = zbio[nlt + k - 1] * dzk[k]
            for ns in range(1, nspint + 1):
                if modal_aero and n == 1:  # interstitial BC
                    ie = int(k_bcexs[k]) - 1
                    ext = kaer_bc[ns - 1, ie]
                    ssa = waer_bc[ns - 1, ie]
                    asm = gaer_bc[ns - 1, ie]
                    t_add = ext * mass
                    w_add = ext * ssa * mass
                    g_add = ext * ssa * asm * mass
                elif modal_aero and n == 2:  # within-ice BC
                    ii = int(k_bcins[k]) - 1
                    ig = int(k_bcini[k]) - 1
                    ext = kaer_bc[ns - 1, ii]
                    ssa = waer_bc[ns - 1, ii]
                    asm = gaer_bc[ns - 1, ii]
                    t_add = ext * bcenh[ns - 1, ii, ig] * mass
                    w_add = ext * ssa * mass
                    g_add = ext * ssa * asm * mass
                else:  # dust, or every species in the bulk treatment
                    ext = kaer[ns - 1, n - 1]
                    ssa = waer[ns - 1, n - 1]
                    asm = gaer[ns - 1, n - 1]
                    t_add = ext * mass
                    w_add = ext * ssa * mass
                    g_add = ext * ssa * asm * mass
                tzaer[ns - 1, k] = tzaer[ns - 1, k] + t_add
                wzaer[ns - 1, k] = wzaer[ns - 1, k] + w_add
                gzaer[ns - 1, k] = gzaer[ns - 1, k] + g_add
    return tzaer, wzaer, gzaer


def _apply_aero_range(
    ns, k0, k1, aero_mp, offset, scale,
    kaer_bc, waer_bc, gaer_bc, bcenh, kaer, waer, gaer,
    k_bcini, k_bcins, k_bcexs, tau, w0, g,
):
    for k in range(k0, k1 + 1):
        taer, waer_sum, gaer_sum = _sum_aerosol(
            ns, k, aero_mp, offset, scale, modal_aero,
            kaer_bc, waer_bc, gaer_bc, bcenh, kaer, waer, gaer,
            k_bcini, k_bcins, k_bcexs,
        )
        _mix_aero(g, w0, tau, k, taer, waer_sum, gaer_sum)


def _apply_snow_ice_aero(
    ns, nslyr, kii, klev, rnslyr, rnilyr, aero_mp,
    kaer_bc, waer_bc, gaer_bc, bcenh, kaer, waer, gaer,
    k_bcini, k_bcins, k_bcexs,
    tzaer, wzaer, gzaer, tau, w0, g, snow,
):
    """Mix zaero or bulk/modal aerosols into snow (snow=True) or sea ice."""
    if _tr("tr_zaero", False) and dEdd_algae:
        if snow:
            _mix_zaero_range = range(0, nslyr + 1)
        else:
            _mix_zaero_range = range(kii, klev + 1)
        for k in _mix_zaero_range:
            _mix_zaero(g, w0, tau, k, ns, gzaer, wzaer, tzaer)
    elif icepack_tracers.tr_aero:
        if snow:
            # snow SSL
            _apply_aero_range(
                ns, 0, 0, aero_mp, 0, c1,
                kaer_bc, waer_bc, gaer_bc, bcenh, kaer, waer, gaer,
                k_bcini, k_bcins, k_bcexs, tau, w0, g,
            )
            # rest of the snow layers: aero_mp(na+1)*rnslyr
            if nslyr >= 1:
                _apply_aero_range(
                    ns, 1, nslyr, aero_mp, 1, rnslyr,
                    kaer_bc, waer_bc, gaer_bc, bcenh, kaer, waer, gaer,
                    k_bcini, k_bcins, k_bcexs, tau, w0, g,
                )
        else:
            # sea ice SSL: aero_mp(na+2)
            _apply_aero_range(
                ns, kii, kii, aero_mp, 2, c1,
                kaer_bc, waer_bc, gaer_bc, bcenh, kaer, waer, gaer,
                k_bcini, k_bcins, k_bcexs, tau, w0, g,
            )
            # sea ice below the SSL: aero_mp(na+3)*rnilyr
            if kii + 1 <= klev:
                _apply_aero_range(
                    ns, kii + 1, klev, aero_mp, 3, rnilyr,
                    kaer_bc, waer_bc, gaer_bc, bcenh, kaer, waer, gaer,
                    k_bcini, k_bcins, k_bcexs, tau, w0, g,
                )


def _ice_column_optics(
    ns, kii, klev, dzk, kabs_chl,
    ki_ssl, wi_ssl, gi_ssl,
    ki_dl, wi_dl, gi_dl,
    ki_int, wi_int, gi_int,
    tau, w0, g,
):
    """Bare or snow-covered sea ice layers. ``ns`` is Fortran 1-based."""
    nilyr = int(icepack_tracers.nilyr)
    # ssl
    k = kii
    tau[k] = (ki_ssl[ns - 1] + kabs_chl[ns - 1, k]) * dzk[k]
    w0[k] = ki_ssl[ns - 1] / (ki_ssl[ns - 1] + kabs_chl[ns - 1, k]) * wi_ssl[ns - 1]
    g[k] = gi_ssl[ns - 1]
    # dl
    k = kii + 1
    # scale dz for dl relative to 4 even-layer-thickness 1.5m case
    fs = p25 * dbl_kind(nilyr)
    tau[k] = (ki_dl[ns - 1] + kabs_chl[ns - 1, k]) * dzk[k] * fs
    w0[k] = ki_dl[ns - 1] / (ki_dl[ns - 1] + kabs_chl[ns - 1, k]) * wi_dl[ns - 1]
    g[k] = gi_dl[ns - 1]
    # int above lowest layer
    if kii + 2 <= klev - 1:
        for k in range(kii + 2, klev):
            tau[k] = (ki_int[ns - 1] + kabs_chl[ns - 1, k]) * dzk[k]
            w0[k] = ki_int[ns - 1] / (ki_int[ns - 1] + kabs_chl[ns - 1, k]) * wi_int[ns - 1]
            g[k] = gi_int[ns - 1]
    # lowest layer
    k = klev
    # add algae to lowest sea ice layer, visible only:
    kabs = ki_int[ns - 1] * (c1 - wi_int[ns - 1])
    if ns == 1:
        # total layer absorption optical depth fixed at value
        # of kalg*0.50m, independent of actual layer thickness
        kabs = kabs + kabs_chl[ns - 1, k]
    sig = ki_int[ns - 1] * wi_int[ns - 1]
    tau[k] = (kabs + sig) * dzk[k]
    w0[k] = sig / (sig + kabs)
    g[k] = gi_int[ns - 1]


def _snow_table_3bd(ns, radius):
    """Qs, ws, gs for one 3-band snow radius. ``ns`` is Fortran 1-based."""
    tab = swdata.rsnw_tab
    if radius < tab[0]:
        return swdata.Qs_tab[ns - 1, 0], swdata.ws_tab[ns - 1, 0], swdata.gs_tab[ns - 1, 0]
    if radius >= tab[swdata.nmbrad_snw - 1]:
        j = swdata.nmbrad_snw - 1
        return swdata.Qs_tab[ns - 1, j], swdata.ws_tab[ns - 1, j], swdata.gs_tab[ns - 1, j]
    nr = shortwave_search(radius, tab)
    delr = (radius - tab[nr - 2]) / (tab[nr - 1] - tab[nr - 2])
    Qs = swdata.Qs_tab[ns - 1, nr - 2] * (c1 - delr) + swdata.Qs_tab[ns - 1, nr - 1] * delr
    ws = swdata.ws_tab[ns - 1, nr - 2] * (c1 - delr) + swdata.ws_tab[ns - 1, nr - 1] * delr
    gs = swdata.gs_tab[ns - 1, nr - 2] * (c1 - delr) + swdata.gs_tab[ns - 1, nr - 1] * delr
    return Qs, ws, gs


def _snow_layers_3bd(ns, frsnw, rhosnw, dzk, kabs_chl, tau, w0, g):
    """Interpolate snow iops using input snow grain radius, snow density and tabular data."""
    nslyr = int(icepack_tracers.nslyr)
    for k in range(0, nslyr + 1):
        # use top rsnw, rhosnw for snow ssl and rest of top layer
        ksnow = max(k, 1)
        radius = frsnw[ksnow - 1]
        # find snow iops using input snow density and snow grain radius:
        Qs, ws, gs = _snow_table_3bd(ns, radius)
        ks = Qs * (
            (rhosnw[ksnow - 1] / rhoi) * dbl_kind(3.0)
            / (dbl_kind(4.0) * radius * dbl_kind(1.0e-6))
        )
        tau[k] = (ks + kabs_chl[ns - 1, k]) * dzk[k]
        w0[k] = ks / (ks + kabs_chl[ns - 1, k]) * ws
        g[k] = gs


def _snicar_iop(ns, radius, direct):
    """SNICAR snow mass extinction, single-scatter albedo, and asymmetry.

    Direct uses ``<= min`` and ``>= max``. Diffuse uses ``< min`` and ``> max``.
    ``ns`` and the returned table column are Fortran 1-based inside the search.
    """
    if direct:
        ext = swdata.ssp_snwextdr
        alb = swdata.ssp_snwalbdr
        asm = swdata.ssp_sasymmdr
        below = radius <= swdata.rsnw_snicar_min
        above = radius >= swdata.rsnw_snicar_max
    else:
        ext = swdata.ssp_snwextdf
        alb = swdata.ssp_snwalbdf
        asm = swdata.ssp_sasymmdf
        below = radius < swdata.rsnw_snicar_min
        above = radius > swdata.rsnw_snicar_max
    if below:
        return ext[ns - 1, 0], alb[ns - 1, 0], asm[ns - 1, 0]
    if above:
        j = swdata.nmbrad_snicar - 1
        return ext[ns - 1, j], alb[ns - 1, j], asm[ns - 1, j]
    # linear interpolation
    if str(swdata.rsnw_datatype).strip() == "sorted_idelta1":
        # NOTE:  Assumes delta rsnw_snicar_tab is 1 and rsnw_snicar_tab are integers
        # This is just for performance, could call shortwave_search
        nr = math.ceil(float(radius)) - _nint(swdata.rsnw_snicar_min) + 1
    else:
        nr = shortwave_search(radius, swdata.rsnw_snicar_tab)
    # Fortran column nr-1 is below a 1-based table when nr = 1.
    if nr < 2 or nr > int(swdata.nmbrad_snicar):
        raise RuntimeError("(compute_dEdd_5bd) snow radius interpolation index out of table")
    delr = (radius - swdata.rsnw_snicar_tab[nr - 2]) / (
        swdata.rsnw_snicar_tab[nr - 1] - swdata.rsnw_snicar_tab[nr - 2]
    )
    ks = ext[ns - 1, nr - 2] * (c1 - delr) + ext[ns - 1, nr - 1] * delr
    ws = alb[ns - 1, nr - 2] * (c1 - delr) + alb[ns - 1, nr - 1] * delr
    gs = asm[ns - 1, nr - 2] * (c1 - delr) + asm[ns - 1, nr - 1] * delr
    return ks, ws, gs


def _snicar_snow_layers(ns, rsnw, rhosnw, dzk, kabs_chl, direct, tau, w0, g):
    nslyr = int(icepack_tracers.nslyr)
    for k in range(0, nslyr + 1):
        # use top rsnw, rhosnw for snow ssl and rest of top layer
        ksnow = max(k, 1)
        ks, ws, gs = _snicar_iop(ns, rsnw[ksnow - 1], direct)
        tau[k] = (ks * rhosnw[ksnow - 1] + kabs_chl[ns - 1, k]) * dzk[k]
        w0[k] = (
            ks * rhosnw[ksnow - 1]
            / (ks * rhosnw[ksnow - 1] + kabs_chl[ns - 1, k])
            * ws
        )
        g[k] = gs


def _internal_absorption(nslyr, nilyr, srftyp, dfdir, dfdif, swdr, swdf, wdir, wdif, Sabs, Iabs):
    """Snow and sea-ice layer absorption for one spectral band.

    ``wdir`` and ``wdif`` are the direct and diffuse spectral weights.
    """
    # if snow covered ice, set snow internal absorption; else, Sabs=0
    if srftyp == 1:
        ki = 0
        for k in range(1, nslyr + 1):
            # skip snow SSL, since SSL absorption included in the surface
            # absorption fsfc above
            km = k
            kp = km + 1
            ki = ki + 1
            Sabs[ki - 1] = Sabs[ki - 1] + (
                dfdir[km] * swdr * wdir + dfdif[km] * swdf * wdif
                - dfdir[kp] * swdr * wdir - dfdif[kp] * swdf * wdif
            )
    # complex indexing to insure proper absorptions for sea ice
    ki = 0
    for k in range(nslyr + 2, nslyr + 1 + nilyr + 1):
        # for bare ice, DL absorption for sea ice layer 1
        km = k
        kp = km + 1
        # modify for top sea ice layer for snow over sea ice
        if srftyp == 1:
            # must add SSL and DL absorption for sea ice layer 1
            if k == nslyr + 2:
                km = k - 1
                kp = km + 2
        ki = ki + 1
        Iabs[ki - 1] = Iabs[ki - 1] + (
            dfdir[km] * swdr * wdir + dfdif[km] * swdf * wdif
            - dfdir[kp] * swdr * wdir - dfdif[kp] * swdf * wdif
        )


def compute_dEdd_3bd(
    klev, klevp, zbio, fnidr, coszen,
    swvdr, swvdf, swidr, swidf, srftyp,
    hs, rhosnw, rsnw, hi, hp, fi, aero_mp,
    fswsfc, fswint, fswthru,
    fswthru_vdr, fswthru_vdf, fswthru_idr, fswthru_idf,
    fswthru_uvrdr, fswthru_uvrdf, fswthru_pardr, fswthru_pardf,
    Sswabs, Iswabs, fswpenl,
    swuvrdr=None, swuvrdf=None, swpardr=None, swpardf=None,
):
    """Column Delta-Eddington solution for snow, bare ice, or ponded ice.

    Set up optical property profiles, based on snow, sea ice and ponded
    ice IOPs from:

    Briegleb, B. P., and B. Light (2007): A Delta-Eddington Multiple
       Scattering Parameterization for Solar Radiation in the Sea Ice
       Component of the Community Climate System Model, NCAR Technical
       Note  NCAR/TN-472+STR  February 2007

    Divides solar spectrum into 3 intervals: 0.2-0.7, 0.7-1.19, and
    1.19-5.0 micro-meters. The latter two are added (using an assumed
    partition of incident shortwave in the 0.7-5.0 micro-meter band between
    the 0.7-1.19 and 1.19-5.0 micro-meter band) to give the final output
    of 0.2-0.7 visible and 0.7-5.0 near-infrared albedos and fluxes.

    Description of the level and layer index conventions. This is
    for the standard case of one snow layer and four sea ice layers.

    Please read the following; otherwise, there is 99.9% chance you
    will be confused about indices at some point in time........ :)

    CICE4.0 snow treatment has one snow layer above the sea ice. This
    snow layer has finite heat capacity, so that surface absorption must
    be distinguished from internal. The Delta-Eddington solar radiation
    thus adds extra surface scattering layers to both snow and sea ice.
    Note that in the following, we assume a fixed vertical layer structure
    for the radiation calculation. In other words, we always have the
    structure shown below for one snow and four sea ice layers, but for
    ponded ice the pond fills "snow" layer 1 over the sea ice, and for
    bare sea ice the top layers over sea ice are treated as transparent air.

    SSL = surface scattering layer for either snow or sea ice
    DL  = drained layer for sea ice immediately under sea ice SSL
    INT = interior layers for sea ice below the drained layer.

    Notice that the radiation level starts with 0 at the top. Thus,
    the total number radiation layers is klev+1, where klev is the
    sum of nslyr, the number of CCSM snow layers, and nilyr, the
    number of CCSM sea ice layers, plus the sea ice SSL:
    klev = 1 + nslyr + nilyr

    For the standard case illustrated below, nslyr=1, nilyr=4,
    and klev=6, with the number of layer interfaces klevp=klev+1.
    Layer interfaces are the surfaces on which reflectivities,
    transmissivities and fluxes are evaluated.

    CCSM3 Sea Ice Model            Delta-Eddington Solar Radiation
                                       Layers and Interfaces
                               Layer Index             Interface Index
      ---------------------            ---------------------  0
                                    0  \\\\\\   snow SSL    \\\\\\
         snow layer 1                  ---------------------  1
                                    1    rest of snow layer
      +++++++++++++++++++++            +++++++++++++++++++++  2
                                    2  \\\\\\ sea ice SSL   \\\\\\
        sea ice layer 1                ---------------------  3
                                    3      sea ice  DL
      ---------------------            ---------------------  4

        sea ice layer 2             4      sea ice INT

      ---------------------            ---------------------  5

        sea ice layer 3             5      sea ice INT

      ---------------------            ---------------------  6

        sea ice layer 4             6      sea ice INT

      ---------------------            ---------------------  7

    When snow lies over sea ice, the radiation absorbed in the
    snow SSL is used for surface heating, and that in the rest
    of the snow layer for its internal heating. For sea ice in
    this case, all of the radiant heat absorbed in both the
    sea ice SSL and the DL are used for sea ice layer 1 heating.

    When pond lies over sea ice, and for bare sea ice, all of the
    radiant heat absorbed within and above the sea ice SSL is used
    for surface heating, and that absorbed in the sea ice DL is
    used for sea ice layer 1 heating.

    Radiation levels keep the Fortran lower bound of 0. Snow and ice
    layer arrays (Sabs, Iabs, frsnw, rsnw) are 0-based.

    Returns the surface albedos (set, not scaled by ``fi``) and the
    accumulated fluxes. ``Sswabs``, ``Iswabs``, and ``fswpenl`` are
    updated in place.
    """
    nslyr = int(icepack_tracers.nslyr)
    nilyr = int(icepack_tracers.nilyr)
    nspint = int(swdata.nspint_3bd)

    # tuning parameters
    # ice and pond scat coeff fractional change for +- one-sigma in albedo
    fp_ice = dbl_kind(0.15)  # ice fraction of scat coeff for + stn dev in alb
    fm_ice = dbl_kind(0.15)  # ice fraction of scat coeff for - stn dev in alb
    fp_pnd = dbl_kind(2.00)  # ponded ice fraction of scat coeff for + stn dev in alb
    fm_pnd = dbl_kind(0.50)  # ponded ice fraction of scat coeff for - stn dev in alb
    fr_max = dbl_kind(1.00)  # snow grain adjustment factor max
    fr_min = dbl_kind(0.80)  # snow grain adjustment factor min
    cp67 = dbl_kind(0.67)  # nir band weight parameter
    cp78 = dbl_kind(0.78)  # nir band weight parameter
    cp01 = dbl_kind(0.01)  # for ocean visible albedo

    # initialize albedos and fluxes to 0
    Sabs = np.zeros(nslyr, dtype=dbl_kind)
    Iabs = np.zeros(nilyr, dtype=dbl_kind)
    fthrul = np.zeros(nilyr + 1, dtype=dbl_kind)
    avdr = c0
    avdf = c0
    aidr = c0
    aidf = c0
    fsfc = c0
    fint = c0
    fthru = c0
    fthruvdr = c0
    fthruvdf = c0
    fthruidr = c0
    fthruidf = c0
    fthruuvrdr = c0
    fthruuvrdf = c0
    fthrupardr = c0
    fthrupardf = c0

    # spectral weights 2 (0.7-1.19 micro-meters) and 3 (1.19-5.0 micro-meters)
    # are chosen based on 1D calculations using ratio of direct to total
    # near-infrared solar (0.7-5.0 micro-meter) which indicates clear/cloudy
    # conditions: more cloud, the less 1.19-5.0 relative to the
    # 0.7-1.19 micro-meter due to cloud absorption.
    wghtns = np.zeros(nspint, dtype=dbl_kind)
    wghtns[0] = c1
    wghtns[1] = cp67 + (cp78 - cp67) * (c1 - fnidr)
    wghtns[2] = c1 - wghtns[1]

    # find snow grain adjustment factor, dependent upon clear/overcast sky
    # estimate. comparisons with SNICAR show better agreement with DE when
    # this factor is included (clear sky near 1 and overcast near 0.8 give
    # best agreement).  Multiply by rnsw here for efficiency.
    frsnw = np.zeros(nslyr, dtype=dbl_kind)
    for k in range(1, nslyr + 1):
        frsnw[k - 1] = (fr_max * fnidr + fr_min * (c1 - fnidr)) * rsnw[k - 1]
        Sabs[k - 1] = c0

    dzk, kii, rnilyr, rnslyr = _radiation_dzk(klev, hs, hi)

    # adjust sea ice iops with tuning parameters; tune only the
    # scattering coefficient by factors of R_ice, R_pnd
    ki_ssl, wi_ssl, gi_ssl = _tune_scat(
        swdata.ki_ssl_mn_3bd, swdata.wi_ssl_mn_3bd, swdata.gi_ssl_mn_3bd,
        fp_ice, fm_ice, R_ice,
    )
    ki_dl, wi_dl, gi_dl = _tune_scat(
        swdata.ki_dl_mn_3bd, swdata.wi_dl_mn_3bd, swdata.gi_dl_mn_3bd,
        fp_ice, fm_ice, R_ice,
    )
    ki_int, wi_int, gi_int = _tune_scat(
        swdata.ki_int_mn_3bd, swdata.wi_int_mn_3bd, swdata.gi_int_mn_3bd,
        fp_ice, fm_ice, R_ice,
    )
    # adjust ponded ice iops with tuning parameters
    ki_p_ssl, wi_p_ssl, gi_p_ssl = _tune_scat(
        swdata.ki_p_ssl_mn, swdata.wi_p_ssl_mn, swdata.gi_p_ssl_mn,
        fp_pnd, fm_pnd, R_pnd,
    )
    ki_p_int, wi_p_int, gi_p_int = _tune_scat(
        swdata.ki_p_int_mn, swdata.wi_p_int_mn, swdata.gi_p_int_mn,
        fp_pnd, fm_pnd, R_pnd,
    )

    # use srftyp to determine interface index of surface absorption
    if srftyp == 1:
        # snow covered sea ice
        ksrf = 1
    else:
        # bare sea ice or ponded ice
        ksrf = nslyr + 2

    kabs_chl = _chlorophyll(nspint, klev, dzk, zbio)

    # aerosols
    # Cheng: note that aerosol IOPs are related to snow grain radius.
    # CICE adjusted snow grain radius rsnw to frsnw in the original 3-band
    # scheme, while for SNICAR the snow grain radius is used directly.
    k_bcini, k_bcins, k_bcexs = _modal_bc_indices(klev, frsnw)
    tzaer, wzaer, gzaer = _fill_zaero(
        nspint, klev, dzk, zbio, k_bcini, k_bcins, k_bcexs,
        swdata.kaer_bc_3bd, swdata.waer_bc_3bd, swdata.gaer_bc_3bd, swdata.bcenh_3bd,
        swdata.kaer_3bd, swdata.waer_3bd, swdata.gaer_3bd,
    )

    tau = np.zeros(klev + 1, dtype=dbl_kind)
    w0 = np.zeros(klev + 1, dtype=dbl_kind)
    g = np.zeros(klev + 1, dtype=dbl_kind)

    # begin spectral loop
    for ns in range(1, nspint + 1):
        # set optical properties of air/snow/pond overlying sea ice
        if srftyp == 0:
            # air
            for k in range(0, nslyr + 1):
                tau[k] = c0
                w0[k] = c0
                g[k] = c0
        elif srftyp == 1:
            # snow
            _snow_layers_3bd(ns, frsnw, rhosnw, dzk, kabs_chl, tau, w0, g)
            # aerosol in snow
            _apply_snow_ice_aero(
                ns, nslyr, kii, klev, rnslyr, rnilyr, aero_mp,
                swdata.kaer_bc_3bd, swdata.waer_bc_3bd, swdata.gaer_bc_3bd,
                swdata.bcenh_3bd, swdata.kaer_3bd, swdata.waer_3bd, swdata.gaer_3bd,
                k_bcini, k_bcins, k_bcexs, tzaer, wzaer, gzaer, tau, w0, g, True,
            )
        else:  # srftyp == 2
            # pond water layers evenly spaced
            dz = hp / (dbl_kind(nslyr) + c1)
            for k in range(0, nslyr + 1):
                tau[k] = swdata.kw[ns - 1] * dz
                w0[k] = swdata.ww[ns - 1]
                g[k] = swdata.gw[ns - 1]
                # no aerosol in pond

        # set optical properties of sea ice
        if srftyp <= 1:
            # bare or snow-covered sea ice layers
            _ice_column_optics(
                ns, kii, klev, dzk, kabs_chl,
                ki_ssl, wi_ssl, gi_ssl, ki_dl, wi_dl, gi_dl, ki_int, wi_int, gi_int,
                tau, w0, g,
            )
            # aerosol in sea ice
            _apply_snow_ice_aero(
                ns, nslyr, kii, klev, rnslyr, rnilyr, aero_mp,
                swdata.kaer_bc_3bd, swdata.waer_bc_3bd, swdata.gaer_bc_3bd,
                swdata.bcenh_3bd, swdata.kaer_3bd, swdata.waer_3bd, swdata.gaer_3bd,
                k_bcini, k_bcins, k_bcexs, tzaer, wzaer, gzaer, tau, w0, g, False,
            )
        else:  # srftyp == 2
            # sea ice layers under ponds
            k = kii
            tau[k] = ki_p_ssl[ns - 1] * dzk[k]
            w0[k] = wi_p_ssl[ns - 1]
            g[k] = gi_p_ssl[ns - 1]
            k = kii + 1
            tau[k] = ki_p_int[ns - 1] * dzk[k]
            w0[k] = wi_p_int[ns - 1]
            g[k] = gi_p_int[ns - 1]
            if kii + 2 <= klev:
                for k in range(kii + 2, klev + 1):
                    tau[k] = ki_p_int[ns - 1] * dzk[k]
                    w0[k] = wi_p_int[ns - 1]
                    g[k] = gi_p_int[ns - 1]
            # adjust pond iops if pond depth within specified range
            # turn off for sea level ponds
            if (not icepack_tracers.tr_pond_sealvl) and hpmin <= hp < hp0:
                k = kii
                sig_i = ki_ssl[ns - 1] * wi_ssl[ns - 1]
                sig_p = ki_p_ssl[ns - 1] * wi_p_ssl[ns - 1]
                sig = sig_i + (sig_p - sig_i) * (hp / hp0)
                kext = sig + ki_p_ssl[ns - 1] * (c1 - wi_p_ssl[ns - 1])
                tau[k] = kext * dzk[k]
                w0[k] = sig / kext
                g[k] = gi_p_int[ns - 1]
                k = kii + 1
                # scale dz for dl relative to 4 even-layer-thickness 1.5m case
                fs = p25 * dbl_kind(nilyr)
                sig_i = ki_dl[ns - 1] * wi_dl[ns - 1] * fs
                sig_p = ki_p_int[ns - 1] * wi_p_int[ns - 1]
                sig = sig_i + (sig_p - sig_i) * (hp / hp0)
                kext = sig + ki_p_int[ns - 1] * (c1 - wi_p_int[ns - 1])
                tau[k] = kext * dzk[k]
                w0[k] = sig / kext
                g[k] = gi_p_int[ns - 1]
                if kii + 2 <= klev:
                    for k in range(kii + 2, klev + 1):
                        sig_i = ki_int[ns - 1] * wi_int[ns - 1]
                        sig_p = ki_p_int[ns - 1] * wi_p_int[ns - 1]
                        sig = sig_i + (sig_p - sig_i) * (hp / hp0)
                        kext = sig + ki_p_int[ns - 1] * (c1 - wi_p_int[ns - 1])
                        tau[k] = kext * dzk[k]
                        w0[k] = sig / kext
                        g[k] = gi_p_int[ns - 1]
                    # small pond depth transition to bare sea ice

        # set reflectivities for ocean underlying sea ice
        rns = dbl_kind(ns - 1)
        albodr = cp01 * (c1 - min(rns, c1))
        albodf = cp01 * (c1 - min(rns, c1))

        # layer input properties now completely specified: tau, w0, g,
        # albodr, albodf; now compute the Delta-Eddington solution
        # reflectivities and transmissivities for each layer; then,
        # combine the layers going downwards accounting for multiple
        # scattering between layers, and finally start from the
        # underlying ocean and combine successive layers upwards to
        # the surface; see comments in solution_dEdd for more details.
        trndir, trntdr, trndif, rupdir, rupdif, rdndif = solution_dEdd(
            coszen, srftyp, klev, klevp, tau, w0, g, albodr, albodf,
        )
        dfdir, dfdif = _interface_flux(
            klevp, trndir, trntdr, trndif, rupdir, rupdif, rdndif,
        )

        # calculate final surface albedos and fluxes-
        # all absorbed flux above ksrf is included in surface absorption
        if ns == 1:  # visible
            swdr = swvdr
            swdf = swvdf
            avdr = rupdir[0]
            avdf = rupdif[0]
            tmp_0 = dfdir[0] * swdr + dfdif[0] * swdf
            tmp_ks = dfdir[ksrf] * swdr + dfdif[ksrf] * swdf
            tmp_kl = dfdir[klevp] * swdr + dfdif[klevp] * swdf

            # for layer biology: save visible only
            for k in range(nslyr + 2, klevp + 1):  # Start at DL layer of ice after SSL scattering
                # fthrul(k-nslyr-1); Fortran index is 1-based
                fthrul[k - nslyr - 2] = dfdir[k] * swdr + dfdif[k] * swdf

            fsfc = fsfc + tmp_0 - tmp_ks
            fint = fint + tmp_ks - tmp_kl
            fthru = fthru + tmp_kl
            fthruvdr = fthruvdr + dfdir[klevp] * swdr
            fthruvdf = fthruvdf + dfdif[klevp] * swdf
            if swuvrdr is not None:
                fthruuvrdr = dfdir[klevp] * swuvrdr
            if swuvrdf is not None:
                fthruuvrdf = dfdif[klevp] * swuvrdf
            if swpardr is not None:
                fthrupardr = dfdir[klevp] * swpardr
            if swpardf is not None:
                fthrupardf = dfdif[klevp] * swpardf

        else:  # ns > 1, near IR
            swdr = swidr
            swdf = swidf

            # let fr1 = alb_1*swd*wght1 and fr2 = alb_2*swd*wght2 be the ns=2,3
            # reflected fluxes respectively, where alb_1, alb_2 are the band
            # albedos, swd = nir incident shortwave flux, and wght1, wght2 are
            # the 2,3 band weights. thus, the total reflected flux is:
            # fr = fr1 + fr2 = alb_1*swd*wght1 + alb_2*swd*wght2  hence, the
            # 2,3 nir band albedo is alb = fr/swd = alb_1*wght1 + alb_2*wght2
            aidr = aidr + rupdir[0] * wghtns[ns - 1]
            aidf = aidf + rupdif[0] * wghtns[ns - 1]

            tmp_0 = (dfdir[0] * swdr + dfdif[0] * swdf) * wghtns[ns - 1]
            tmp_ks = (dfdir[ksrf] * swdr + dfdif[ksrf] * swdf) * wghtns[ns - 1]
            tmp_kl = (dfdir[klevp] * swdr + dfdif[klevp] * swdf) * wghtns[ns - 1]

            fsfc = fsfc + tmp_0 - tmp_ks
            fint = fint + tmp_ks - tmp_kl
            fthru = fthru + tmp_kl
            fthruidr = fthruidr + dfdir[klevp] * swdr * wghtns[ns - 1]
            fthruidf = fthruidf + dfdif[klevp] * swdf * wghtns[ns - 1]

        # Keep the three-band absorption grouping distinct from SNICAR.
        if srftyp == 1:
            for k in range(1, nslyr + 1):
                km = k
                kp = km + 1
                if ns == 1:
                    Sabs[k - 1] = (
                        Sabs[k - 1] + dfdir[km] * swdr + dfdif[km] * swdf
                        - (dfdir[kp] * swdr + dfdif[kp] * swdf)
                    )
                else:
                    Sabs[k - 1] = Sabs[k - 1] + (
                        dfdir[km] * swdr + dfdif[km] * swdf
                        - (dfdir[kp] * swdr + dfdif[kp] * swdf)
                    ) * wghtns[ns - 1]
        for ki in range(1, nilyr + 1):
            k = nslyr + 1 + ki
            km = k
            kp = km + 1
            if srftyp == 1 and ki == 1:
                km = k - 1
                kp = km + 2
            if ns == 1:
                Iabs[ki - 1] = (
                    Iabs[ki - 1] + dfdir[km] * swdr + dfdif[km] * swdf
                    - (dfdir[kp] * swdr + dfdif[kp] * swdf)
                )
            else:
                Iabs[ki - 1] = Iabs[ki - 1] + (
                    dfdir[km] * swdr + dfdif[km] * swdf
                    - (dfdir[kp] * swdr + dfdif[kp] * swdf)
                ) * wghtns[ns - 1]

    # accumulate fluxes over bare sea ice
    fswsfc = fswsfc + fsfc * fi
    fswint = fswint + fint * fi
    fswthru = fswthru + fthru * fi
    fswthru_vdr = fswthru_vdr + fthruvdr * fi
    fswthru_vdf = fswthru_vdf + fthruvdf * fi
    fswthru_idr = fswthru_idr + fthruidr * fi
    fswthru_idf = fswthru_idf + fthruidf * fi
    fswthru_uvrdr = fswthru_uvrdr + fthruuvrdr * fi
    fswthru_uvrdf = fswthru_uvrdf + fthruuvrdf * fi
    fswthru_pardr = fswthru_pardr + fthrupardr * fi
    fswthru_pardf = fswthru_pardf + fthrupardf * fi

    for k in range(1, nslyr + 1):
        Sswabs[k - 1] = Sswabs[k - 1] + Sabs[k - 1] * fi
    for k in range(1, nilyr + 1):
        Iswabs[k - 1] = Iswabs[k - 1] + Iabs[k - 1] * fi
        # bgc layer
        fswpenl[k - 1] = fswpenl[k - 1] + fthrul[k - 1] * fi
    fswpenl[nilyr] = fswpenl[nilyr] + fthrul[nilyr] * fi

    return (
        avdr, avdf, aidr, aidf,
        fswsfc, fswint, fswthru,
        fswthru_vdr, fswthru_vdf, fswthru_idr, fswthru_idf,
        fswthru_uvrdr, fswthru_uvrdf, fswthru_pardr, fswthru_pardf,
    )


def compute_dEdd_5bd(
    klev, klevp, zbio, fnidr, coszen,
    swvdr, swvdf, swidr, swidf, srftyp,
    hs, rhosnw, rsnw, hi, hp, fi, aero_mp,
    fswsfc, fswint, fswthru,
    fswthru_vdr, fswthru_vdf, fswthru_idr, fswthru_idf,
    Sswabs, Iswabs, fswpenl,
):
    """Five-band SNICAR-AD delta-Eddington column for snow-covered ice.

    The level and layer index convention is the same description as
    ``compute_dEdd_3bd``. The Fortran source repeats that description here.

    SNICAR-AD major changes:
    1. loop through 5 bands based on nsky
    2. use snow grain size rsnow, not scaled frsnw
    3. replace IOPs_tab with IOPs_snicar
    4. replace wghtns with wghtns_5bd

    ``if (srftyp == 1)`` around the snow optics and ``if (srftyp <= 1)``
    around the sea-ice optics are commented out in the Fortran, so the
    active path always sets snow and then ice. ``hp`` and ``fnidr`` stay
    on the interface; the 3-band nir weight that used ``fnidr`` is commented
    out, and this routine does not apply pond-water optics.
    ``k_bcini(:) = c0`` assigns zero into the integer index arrays.

    Returns surface albedos and accumulated fluxes. Layer absorption
    arrays are updated in place. There is no uv/par through-flux.
    """
    # fnidr is unused: the 3-band weight block below is commented out.
    # hp is unused: pond optics are not in the active 5-band path.
    nslyr = int(icepack_tracers.nslyr)
    nilyr = int(icepack_tracers.nilyr)
    nspint = int(swdata.nspint_5bd)

    # tuning parameters
    fp_ice = dbl_kind(0.15)
    fm_ice = dbl_kind(0.15)
    cp01 = dbl_kind(0.01)  # for ocean visible albedo
    # solar zenith angle parameters
    sza_a0 = dbl_kind(0.085730)
    sza_a1 = dbl_kind(-0.630883)
    sza_a2 = dbl_kind(1.303723)
    sza_b0 = dbl_kind(1.467291)
    sza_b1 = dbl_kind(-3.338043)
    sza_b2 = dbl_kind(6.807489)
    mu_75 = dbl_kind(0.2588)  # cos(75 degrees)

    # copy/point to table data for local names
    # asm_prm_ice_drc => ssp_sasymmdr, and the other five ssp arrays,
    # are used through swdata inside _snicar_iop.

    Sabs = np.zeros(nslyr, dtype=dbl_kind)
    Iabs = np.zeros(nilyr, dtype=dbl_kind)
    fthrul = np.zeros(nilyr + 1, dtype=dbl_kind)
    avdr = c0
    avdf = c0
    aidr = c0
    aidf = c0
    fsfc = c0
    fint = c0
    fthru = c0
    fthruvdr = c0
    fthruvdf = c0
    fthruidr = c0
    fthruidf = c0

    # spectral weights - 3 bands
    # this section of code is kept for future mearge between 5band and 3 band
    # subroutines
    # weights 2 (0.7-1.19 micro-meters) and 3 (1.19-5.0 micro-meters)
    # are chosen based on 1D calculations using ratio of direct to total
    # near-infrared solar (0.7-5.0 micro-meter) which indicates clear/cloudy
    # conditions: more cloud, the less 1.19-5.0 relative to the
    # 0.7-1.19 micro-meter due to cloud absorption.
    # wghtns(1) = c1
    # wghtns(2) = cp67 + (cp78-cp67)*(c1-fnidr)
    # wghtns(3) = cp33 + (cp22-cp33)*(c1-fnidr)
    # wghtns(3) = c1 - wghtns(2)

    # spectral weights - 5 bands
    # direct beam incident
    wghtns_5bd_drc = np.zeros(nspint, dtype=dbl_kind)
    wghtns_5bd_drc[0] = c1
    wghtns_5bd_drc[1] = dbl_kind(0.49352158521175)
    wghtns_5bd_drc[2] = dbl_kind(0.18099494230665)
    wghtns_5bd_drc[3] = dbl_kind(0.12094898498813)
    wghtns_5bd_drc[4] = c1 - (
        wghtns_5bd_drc[1] + wghtns_5bd_drc[2] + wghtns_5bd_drc[3]
    )
    # diffuse incident
    wghtns_5bd_dfs = np.zeros(nspint, dtype=dbl_kind)
    wghtns_5bd_dfs[0] = c1
    wghtns_5bd_dfs[1] = dbl_kind(0.58581507618433)
    wghtns_5bd_dfs[2] = dbl_kind(0.20156903770812)
    wghtns_5bd_dfs[3] = dbl_kind(0.10917889346386)
    wghtns_5bd_dfs[4] = c1 - (
        wghtns_5bd_dfs[1] + wghtns_5bd_dfs[2] + wghtns_5bd_dfs[3]
    )

    for k in range(1, nslyr + 1):
        # frsnw(k) = (fr_max*fnidr + fr_min*(c1-fnidr))*rsnw(k)
        Sabs[k - 1] = c0

    dzk, kii, rnilyr, rnslyr = _radiation_dzk(klev, hs, hi)

    # adjust sea ice iops with tuning parameters; tune only the
    # scattering coefficient by factors of R_ice, R_pnd
    # FUTURE-WORK: update 5-band sea ice iops when avalible
    ki_ssl, wi_ssl, gi_ssl = _tune_scat(
        swdata.ki_ssl_mn_5bd, swdata.wi_ssl_mn_5bd, swdata.gi_ssl_mn_5bd,
        fp_ice, fm_ice, R_ice,
    )
    ki_dl, wi_dl, gi_dl = _tune_scat(
        swdata.ki_dl_mn_5bd, swdata.wi_dl_mn_5bd, swdata.gi_dl_mn_5bd,
        fp_ice, fm_ice, R_ice,
    )
    ki_int, wi_int, gi_int = _tune_scat(
        swdata.ki_int_mn_5bd, swdata.wi_int_mn_5bd, swdata.gi_int_mn_5bd,
        fp_ice, fm_ice, R_ice,
    )

    # use srftyp to determine interface index of surface absorption
    ksrf = 1  # snow covered sea ice

    kabs_chl = _chlorophyll(nspint, klev, dzk, zbio)

    # Cheng: CICE adjusted snow grain radius rsnw to frsnw, while for
    # SNICAR there is no need, the tmp_gs is therefore calculated
    # differently from code in subroutine compute_dEdd
    k_bcini, k_bcins, k_bcexs = _modal_bc_indices(klev, rsnw)
    tzaer, wzaer, gzaer = _fill_zaero(
        nspint, klev, dzk, zbio, k_bcini, k_bcins, k_bcexs,
        swdata.kaer_bc_5bd, swdata.waer_bc_5bd, swdata.gaer_bc_5bd, swdata.bcenh_5bd,
        swdata.kaer_5bd, swdata.waer_5bd, swdata.gaer_5bd,
    )

    tau = np.zeros(klev + 1, dtype=dbl_kind)
    w0 = np.zeros(klev + 1, dtype=dbl_kind)
    g = np.zeros(klev + 1, dtype=dbl_kind)

    # begin spectral loop
    for ns in range(1, nspint + 1):
        # for snow-covered sea ice, compute 5 bands
        # if srftyp == 1:   (commented out; snow optics always run)
        dfdir_snicar = np.zeros(klevp + 1, dtype=dbl_kind)
        dfdif_snicar = np.zeros(klevp + 1, dtype=dbl_kind)
        rupdir_snicar = np.zeros(klevp + 1, dtype=dbl_kind)
        rupdif_snicar = np.zeros(klevp + 1, dtype=dbl_kind)

        for nsky in range(1, 3):  # loop for both direct beam and diffuse beam
            if nsky == 1:  # direct incident
                _snicar_snow_layers(ns, rsnw, rhosnw, dzk, kabs_chl, True, tau, w0, g)
            elif nsky == 2:  # diffuse incident
                _snicar_snow_layers(ns, rsnw, rhosnw, dzk, kabs_chl, False, tau, w0, g)

            # aerosol in snow
            _apply_snow_ice_aero(
                ns, nslyr, kii, klev, rnslyr, rnilyr, aero_mp,
                swdata.kaer_bc_5bd, swdata.waer_bc_5bd, swdata.gaer_bc_5bd,
                swdata.bcenh_5bd, swdata.kaer_5bd, swdata.waer_5bd, swdata.gaer_5bd,
                k_bcini, k_bcins, k_bcexs, tzaer, wzaer, gzaer, tau, w0, g, True,
            )

            # set optical properties of sea ice
            # bare or snow-covered sea ice layers
            # if srftyp <= 1:   (commented out; ice optics always run)
            _ice_column_optics(
                ns, kii, klev, dzk, kabs_chl,
                ki_ssl, wi_ssl, gi_ssl, ki_dl, wi_dl, gi_dl, ki_int, wi_int, gi_int,
                tau, w0, g,
            )
            # aerosol in sea ice
            _apply_snow_ice_aero(
                ns, nslyr, kii, klev, rnslyr, rnilyr, aero_mp,
                swdata.kaer_bc_5bd, swdata.waer_bc_5bd, swdata.gaer_bc_5bd,
                swdata.bcenh_5bd, swdata.kaer_5bd, swdata.waer_5bd, swdata.gaer_5bd,
                k_bcini, k_bcins, k_bcexs, tzaer, wzaer, gzaer, tau, w0, g, False,
            )

            # set reflectivities for ocean underlying sea ice
            # if ns == 1 (visible), albedo is 0.1, else, albedo is zero
            rns = dbl_kind(ns - 1)
            albodr = cp01 * (c1 - min(rns, c1))
            albodf = cp01 * (c1 - min(rns, c1))

            trndir, trntdr, trndif, rupdir, rupdif, rdndif = solution_dEdd(
                coszen, srftyp, klev, klevp, tau, w0, g, albodr, albodf,
            )
            dfdir, dfdif = _interface_flux(
                klevp, trndir, trntdr, trndif, rupdir, rupdif, rdndif,
            )

            # note that because the snow IOPs for diffuse and direct incidents
            # are different, the snow albedo needs to be calculated twice for
            # direct incident and diffuse incident respectively
            if nsky == 1:  # direct beam (keep the direct beam results)
                for k in range(0, klevp + 1):
                    dfdir_snicar[k] = dfdir[k]
                    rupdir_snicar[k] = rupdir[k]
            elif nsky == 2:  # diffuse (keep the diffuse incident results)
                for k in range(0, klevp + 1):
                    dfdif_snicar[k] = dfdif[k]
                    rupdif_snicar[k] = rupdif[k]
            # end direct/diffuse nsky loop

        # calculate final surface albedos and fluxes
        # all absorbed flux above ksrf is included in surface absorption
        if ns == 1:  # visible
            swdr = swvdr
            swdf = swvdf
            avdr = rupdir_snicar[0]
            avdf = rupdif_snicar[0]
            tmp_0 = dfdir_snicar[0] * swdr + dfdif_snicar[0] * swdf
            tmp_ks = dfdir_snicar[ksrf] * swdr + dfdif_snicar[ksrf] * swdf
            tmp_kl = dfdir_snicar[klevp] * swdr + dfdif_snicar[klevp] * swdf

            # for layer biology: save visible only
            for k in range(nslyr + 2, klevp + 1):  # Start at DL layer of ice after SSL scattering
                fthrul[k - nslyr - 2] = dfdir_snicar[k] * swdr + dfdif_snicar[k] * swdf

            fsfc = fsfc + tmp_0 - tmp_ks
            fint = fint + tmp_ks - tmp_kl
            fthru = fthru + tmp_kl
            fthruvdr = fthruvdr + dfdir_snicar[klevp] * swdr
            fthruvdf = fthruvdf + dfdif_snicar[klevp] * swdf

            _internal_absorption(
                nslyr, nilyr, srftyp,
                dfdir_snicar, dfdif_snicar, swdr, swdf, c1, c1, Sabs, Iabs,
            )
        else:  # ns > 1, near IR
            swdr = swidr
            swdf = swidf
            aidr = aidr + rupdir_snicar[0] * wghtns_5bd_drc[ns - 1]
            aidf = aidf + rupdif_snicar[0] * wghtns_5bd_dfs[ns - 1]

            tmp_0 = (
                dfdir_snicar[0] * swdr * wghtns_5bd_drc[ns - 1]
                + dfdif_snicar[0] * swdf * wghtns_5bd_dfs[ns - 1]
            )
            tmp_ks = (
                dfdir_snicar[ksrf] * swdr * wghtns_5bd_drc[ns - 1]
                + dfdif_snicar[ksrf] * swdf * wghtns_5bd_dfs[ns - 1]
            )
            tmp_kl = (
                dfdir_snicar[klevp] * swdr * wghtns_5bd_drc[ns - 1]
                + dfdif_snicar[klevp] * swdf * wghtns_5bd_dfs[ns - 1]
            )

            fsfc = fsfc + tmp_0 - tmp_ks
            fint = fint + tmp_ks - tmp_kl
            fthru = fthru + tmp_kl
            fthruidr = fthruidr + dfdir_snicar[klevp] * swdr * wghtns_5bd_drc[ns - 1]
            fthruidf = fthruidf + dfdif_snicar[klevp] * swdf * wghtns_5bd_dfs[ns - 1]

            _internal_absorption(
                nslyr, nilyr, srftyp,
                dfdir_snicar, dfdif_snicar, swdr, swdf,
                wghtns_5bd_drc[ns - 1], wghtns_5bd_dfs[ns - 1], Sabs, Iabs,
            )

    # solar zenith angle parameterization
    # calculate the scaling factor for NIR direct albedo if SZA>75 degrees
    sza_factor = c1
    if srftyp == 1:
        mu0 = max(coszen, p01)
        if mu0 < mu_75:
            sza_c1 = sza_a0 + sza_a1 * mu0 + sza_a2 * mu0 ** 2
            sza_c0 = sza_b0 + sza_b1 * mu0 + sza_b2 * mu0 ** 2
            sza_factor = sza_c1 * (math.log10(float(rsnw[0])) - dbl_kind(6.0)) + sza_c0

    alvdr = avdr
    alvdf = avdf
    alidr = aidr * sza_factor  # sza factor is always larger than or equal to 1
    alidf = aidf

    # accumulate fluxes over bare sea ice
    # note that we assume the reduced NIR energy absorption by snow
    # due to corrected snow albedo is absorbed by the snow single
    # scattering layer only - this is generally true if snow SSL >= 2 cm
    # by the default model set up:
    #      if snow_depth >= 8 cm, SSL = 4 cm, satisfy
    # else if snow_depth >= 4 cm, SSL = snow_depth/2 >= 2 cm, satisfy
    # else    snow_depth < 4 cm, SSL = snow_depth/2, may overcool SSL layer
    fswsfc = fswsfc + (fsfc - (sza_factor - c1) * aidr * swidr) * fi
    fswint = fswint + fint * fi
    fswthru = fswthru + fthru * fi
    fswthru_vdr = fswthru_vdr + fthruvdr * fi
    fswthru_vdf = fswthru_vdf + fthruvdf * fi
    fswthru_idr = fswthru_idr + fthruidr * fi
    fswthru_idf = fswthru_idf + fthruidf * fi

    for k in range(1, nslyr + 1):
        Sswabs[k - 1] = Sswabs[k - 1] + Sabs[k - 1] * fi
    for k in range(1, nilyr + 1):
        Iswabs[k - 1] = Iswabs[k - 1] + Iabs[k - 1] * fi
        # bgc layer
        fswpenl[k - 1] = fswpenl[k - 1] + fthrul[k - 1] * fi
    fswpenl[nilyr] = fswpenl[nilyr] + fthrul[nilyr] * fi

    return (
        alvdr, alvdf, alidr, alidf,
        fswsfc, fswint, fswthru,
        fswthru_vdr, fswthru_vdf, fswthru_idr, fswthru_idf,
    )

