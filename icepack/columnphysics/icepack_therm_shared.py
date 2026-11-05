"""Shared thermo variables, subroutines

author for JAX version: Xinyuan Zhu
XZ: converted from F90 to Python

authors: Elizabeth C. Hunke, LANL

``icepack_init_thermo`` is the October 2024 name ``icepack_init_salinity``.
``icepack_init_trcr`` is ``icepack_init_enthalpy``.

Ported: ``icepack_init_salinity``, ``icepack_salinity_profile``,
``icepack_init_enthalpy``, ``icepack_liquidus_temperature``,
``icepack_sea_freezing_temperature``, ``calculate_Tin_from_qin``,
``icepack_ice_temperature``, ``icepack_snow_temperature``.

The JAX array versions of ``surface_heat_flux``,
``dsurface_heat_flux_dTsf`` and ``adjust_enthalpy`` are ported for the time
loop; their physical callers are not yet connected.
"""

import numpy as np
import jax.numpy as jnp

from icepack.columnphysics import icepack_mushy_physics
from icepack.columnphysics import icepack_parameters
from icepack.columnphysics import icepack_tracers
from icepack.columnphysics.icepack_kinds import dbl_kind, log_kind
from icepack.columnphysics.icepack_parameters import c0, c1, c2, c4, p5, pi

ferrmax = dbl_kind(1.0e-3)  # max allowed energy flux error (W m-2)
                            # recommend ferrmax < 0.01 W m-2

Tmin = dbl_kind(-100.0)  # min allowed internal temperature (deg C)

l_brine = log_kind(False)  # if true, treat brine pocket effects

dfsurfdTs_cpl = None
dflatdTs_cpl = None
fsurf_cpl0 = None
flat_cpl0 = None
fsurf_cpl = None
flat_cpl = None


def adjust_enthalpy(z1, z2, hlyr, hn, qn, puny):
    """Remap a conserved layer quantity to equal-thickness new layers.

    The last axis is vertical; ``z1`` and ``z2`` contain layer interfaces
    and have one more element than ``qn``. Leading axes batch columns.
    """
    nlyr = qn.shape[-1]
    rhlyr = jnp.where(hn > puny,
                      1.0 / jnp.where(hn > puny, hlyr, 1.0), 0.0)
    remapped = jnp.zeros_like(qn)
    for k2 in range(nlyr):
        hq = jnp.zeros_like(hn)
        for k1 in range(nlyr):
            overlap = jnp.maximum(
                jnp.minimum(z1[..., k1 + 1], z2[..., k2 + 1])
                - jnp.maximum(z1[..., k1], z2[..., k2]), 0.0)
            hq = hq + overlap * qn[..., k1]
        remapped = remapped.at[..., k2].set(hq * rhlyr)

    qtot = jnp.zeros_like(hn)
    for k in range(nlyr):
        qtot = qtot + qn[..., k] * (z1[..., k + 1] - z1[..., k])
    thin = jnp.where(hn > 0.0, qtot / jnp.where(hn > 0.0, hn, 1.0), 0.0)
    return jnp.where((hn > puny)[..., None], remapped, thin[..., None])


def surface_heat_flux(Tsf, fswsfc, rhoa, flw, potT, Qa, shcoef, lhcoef,
                      parameters):
    """Compute the downward net heat flux at the snow or ice surface."""
    TsfK = Tsf + parameters["Tffresh"]
    # Retain the reciprocal and operation grouping used by Fortran.
    tmpvar = c1 / TsfK
    qsat = parameters["qqqice"] * jnp.exp(-parameters["TTTice"] * tmpvar)
    Qsfc = qsat / rhoa
    flwdabs = parameters["emissivity"] * flw
    flwoutn = (-parameters["emissivity"] * parameters["stefan_boltzmann"]
               * TsfK**4)
    fsensn = shcoef * (potT - TsfK)
    flatn = lhcoef * (Qa - Qsfc)
    fsurfn = fswsfc + flwdabs + flwoutn + fsensn + flatn
    return flwoutn, fsensn, flatn, fsurfn


def dsurface_heat_flux_dTsf(Tsf, rhoa, shcoef, lhcoef, parameters):
    """Compute CICE's analytic surface-flux derivatives with temperature."""
    TsfK = Tsf + parameters["Tffresh"]
    tmpvar = c1 / TsfK
    qsat = parameters["qqqice"] * jnp.exp(-parameters["TTTice"] * tmpvar)
    dQsfc_dTsf = parameters["TTTice"] * tmpvar * tmpvar * (qsat / rhoa)
    dflwoutn_dTsf = (-parameters["emissivity"]
                      * parameters["stefan_boltzmann"] * c4 * TsfK**3)
    dfsensn_dTsf = -shcoef
    dflatn_dTsf = -lhcoef * dQsfc_dTsf
    dfsurfn_dTsf = dflwoutn_dTsf + dfsensn_dTsf + dflatn_dTsf
    return dfsurfn_dTsf, dflwoutn_dTsf, dfsensn_dTsf, dflatn_dTsf


def calculate_Tin_from_qin_jax(qin, Tmltk, l_brine, parameters):
    """JAX BL99 ice temperature from layer enthalpy and liquidus."""
    if l_brine:
        aa1 = parameters["cp_ice"]
        bb1 = ((parameters["cp_ocn"] - aa1) * Tmltk
               - qin / parameters["rhoi"] - parameters["Lfresh"])
        cc1 = parameters["Lfresh"] * Tmltk
        discriminant = bb1 * bb1 - 4.0 * aa1 * cc1
        Tin = (-bb1 - jnp.sqrt(jnp.maximum(discriminant, 0.0))) / (2.0 * aa1)
        return jnp.minimum(Tin, Tmltk), discriminant < 0.0
    Tin = ((parameters["Lfresh"] + qin / parameters["rhoi"])
           / parameters["cp_ice"])
    return Tin, jnp.zeros_like(Tin, dtype=bool)


def icepack_ice_temperature_jax(qin, Sin, ktherm, l_brine, parameters):
    """Ice temperature for the selected BL99 or mushy scheme."""
    if ktherm == 2:
        temperature = icepack_mushy_physics.icepack_mushy_temperature_mush_jax(
            qin, Sin, parameters)
        return temperature, jnp.zeros_like(temperature, dtype=bool)
    Tmlts = -parameters["depressT"] * Sin
    return calculate_Tin_from_qin_jax(qin, Tmlts, l_brine, parameters)


def icepack_snow_temperature_jax(qin, parameters):
    """Snow temperature for both supported thermodynamic schemes."""
    return ((parameters["Lfresh"] + qin / parameters["rhos"])
            / parameters["cp_ice"])


def icepack_sea_freezing_temperature_jax(sss, tfrz_option, parameters):
    """Ocean freezing point for the configured salinity relation."""
    if tfrz_option == "mushy":
        return icepack_mushy_physics.liquidus_temperature_mush_jax(sss)
    if tfrz_option == "linear_salt":
        return -parameters["depressT"] * sss
    if tfrz_option == "constant":
        return jnp.full_like(sss, parameters["Tocnfrz"])
    if tfrz_option == "minus1p8":
        return jnp.full_like(sss, -1.8)
    raise ValueError("tfrz_option unsupported: " + tfrz_option)


def icepack_init_salinity(sprofile):
    """Initialize the vertical profile of ice salinity.
    This subroutine was renamed from icepack_init_thermo in Oct 2024.

    authors: C. M. Bitz, UW
             William H. Lipscomb, LANL

    sprofile: vertical salinity profile, length nilyr+1, written in place.
    """
    global l_brine
    nilyr = int(icepack_tracers.nilyr)
    if len(sprofile) < nilyr + 1:
        raise ValueError(
            f"(icepack_init_salinity) sprofile length {len(sprofile)}; need {nilyr + 1}"
        )

    #-----------------------------------------------------------------
    # Determine l_brine based on saltmax.
    # Thermodynamic solver will not converge if l_brine is true and
    #  saltmax is close to zero.
    # Set l_brine to false for zero layer thermodynamics
    #-----------------------------------------------------------------

    l_brine = log_kind(False)
    # saltmax and min_salin are read at the call, after icepack_init_parameters.
    if icepack_parameters.saltmax > icepack_parameters.min_salin:
        l_brine = log_kind(True)

    #-----------------------------------------------------------------
    # Prescibe vertical profile of salinity.
    # Note this profile is only used for BL99 thermodynamics.
    #-----------------------------------------------------------------

    if l_brine:
        for k in range(1, nilyr + 1):
            zn = (dbl_kind(k) - p5) / dbl_kind(nilyr)
            # sprofile(k)=(saltmax/c2)*(c1-cos(pi*zn**(nsal/(msal+zn))))
            sprofile[k - 1] = icepack_salinity_profile(zn)
            sprofile[k - 1] = max(sprofile[k - 1], icepack_parameters.min_salin)
        sprofile[nilyr] = icepack_parameters.saltmax
    else:  # .not. l_brine
        for k in range(nilyr + 1):
            sprofile[k] = c0
    return sprofile


def icepack_init_thermo(sprofile):
    """Former name of ``icepack_init_salinity``."""
    return icepack_init_salinity(sprofile)


def icepack_salinity_profile(zn):
    """Initial salinity profile.

    authors: C. M. Bitz, UW
             William H. Lipscomb, LANL

    zn: normalized depth
    returns salinity
    """
    nsal = dbl_kind(0.407)
    msal = dbl_kind(0.573)
    zn = np.asarray(zn, dtype=dbl_kind)
    return (icepack_parameters.saltmax / c2) * (
        c1 - np.cos(pi * zn ** (nsal / (msal + zn)))
    )


def icepack_init_enthalpy(Tair, Tf, Sprofile, Tprofile):
    """Initialize surface temperature and ice and snow enthalpy.
    This subroutine was renamed from icepack_init_trcr in Oct 2024.

    Tair: air temperature (K)
    Tf: freezing temperature (C)
    Sprofile: vertical salinity profile (ppt)
    Tprofile: vertical temperature profile (C)
    returns Tsfc (C), qin (J/m3), qsn (J/m3)
    """
    nilyr = int(icepack_tracers.nilyr)
    nslyr = int(icepack_tracers.nslyr)
    # surface temperature
    Tsfc = Tf  # default
    if bool(icepack_parameters.calc_Tsfc):
        Tsfc = min(icepack_parameters.Tsmelt, Tair - icepack_parameters.Tffresh)  # deg C

    # ice enthalpy
    qin = np.zeros(nilyr, dtype=dbl_kind)
    rhoi = icepack_parameters.rhoi
    cp_ice = icepack_parameters.cp_ice
    cp_ocn = icepack_parameters.cp_ocn
    Lfresh = icepack_parameters.Lfresh
    for k in range(1, nilyr + 1):
        # assume linear temp profile and compute enthalpy
        slope = Tf - Tsfc
        Ti = Tsfc + slope * (dbl_kind(k) - p5) / dbl_kind(nilyr)
        if int(icepack_parameters.ktherm) == 2:
            qin[k - 1] = icepack_mushy_physics.icepack_enthalpy_mush(Ti, Sprofile[k - 1])
        else:
            qin[k - 1] = -(
                rhoi
                * (
                    cp_ice * (Tprofile[k - 1] - Ti)
                    + Lfresh * (c1 - Tprofile[k - 1] / Ti)
                    - cp_ocn * Tprofile[k - 1]
                )
            )

    # snow enthalpy
    qsn = np.zeros(nslyr, dtype=dbl_kind)
    rhos = icepack_parameters.rhos
    for k in range(nslyr):
        Ti = min(c0, Tsfc)
        qsn[k] = -rhos * (Lfresh - cp_ice * Ti)
    return dbl_kind(Tsfc), qin, qsn


def icepack_init_trcr(Tair, Tf, Sprofile, Tprofile):
    """Former name of ``icepack_init_enthalpy``."""
    return icepack_init_enthalpy(Tair, Tf, Sprofile, Tprofile)


def icepack_liquidus_temperature(Sin):
    """Compute liquidus temperature.

    Sin: salinity (ppt)
    returns melting temperature (C)
    """
    Sin = np.asarray(Sin, dtype=dbl_kind)
    if int(icepack_parameters.ktherm) == 2:
        Tmlt = icepack_mushy_physics.liquidus_temperature_mush(Sin)
    else:
        Tmlt = -icepack_parameters.depressT * Sin
    return Tmlt


def icepack_sea_freezing_temperature(sss):
    """Compute ocean freezing temperature.

    sss: sea surface salinity (ppt)
    returns Tf (C)
    """
    subname = "(icepack_sea_freezing_temperature)"
    option = str(icepack_parameters.tfrz_option).strip()
    sss_a = np.asarray(sss, dtype=dbl_kind)
    if option == "mushy":
        Tf = icepack_liquidus_temperature(sss_a)  # deg C
    elif option == "linear_salt":
        Tf = -icepack_parameters.depressT * sss_a  # deg C
    elif option == "constant":
        Tf = np.full(np.shape(sss_a), icepack_parameters.Tocnfrz, dtype=dbl_kind)
    elif option == "minus1p8":
        Tf = np.full(np.shape(sss_a), dbl_kind(-1.8), dtype=dbl_kind)
    else:
        raise ValueError(subname + " tfrz_option unsupported: " + option)
    if np.ndim(sss) == 0:
        return dbl_kind(np.reshape(Tf, ()))
    return Tf


def calculate_Tin_from_qin(qin, Tmltk):
    """Compute the internal ice temperatures from enthalpy using quadratic formula.

    qin: enthalpy
    Tmltk: melting temperature at one level
    returns internal temperature
    """
    qin = np.asarray(qin, dtype=dbl_kind)
    Tmltk = np.asarray(Tmltk, dtype=dbl_kind)
    if bool(l_brine):
        aa1 = icepack_parameters.cp_ice
        bb1 = (icepack_parameters.cp_ocn - icepack_parameters.cp_ice) * Tmltk - qin / icepack_parameters.rhoi - icepack_parameters.Lfresh
        cc1 = icepack_parameters.Lfresh * Tmltk
        csqrt = bb1 * bb1 - c4 * aa1 * cc1
        if np.any(csqrt < c0):
            raise ValueError("(calculate_Tin_from_qin) sqrt error")
        Tin = np.minimum((-bb1 - np.sqrt(csqrt)) / (c2 * aa1), Tmltk)
    else:  # fresh ice
        Tin = (icepack_parameters.Lfresh + qin / icepack_parameters.rhoi) / icepack_parameters.cp_ice
    return Tin


def icepack_ice_temperature(qin, Sin):
    """Compute ice temperature.

    qin: ice enthalpy
    Sin: ice salinity
    returns ice temperature (C)
    """
    if int(icepack_parameters.ktherm) == 2:
        Tin = icepack_mushy_physics.icepack_mushy_temperature_mush(qin, Sin)
    else:
        Tmlts = -icepack_parameters.depressT * np.asarray(Sin, dtype=dbl_kind)
        Tin = calculate_Tin_from_qin(qin, Tmlts)
    return Tin


def icepack_snow_temperature(qin):
    """Compute snow temperature.

    qin: snow enthalpy
    returns snow temperature (C)
    """
    if int(icepack_parameters.ktherm) == 2:
        Tsn = icepack_mushy_physics.temperature_snow(qin)
    else:
        Tsn = (icepack_parameters.Lfresh + np.asarray(qin, dtype=dbl_kind) / icepack_parameters.rhos) / icepack_parameters.cp_ice
    return Tsn
