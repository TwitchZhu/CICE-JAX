"""Mushy-layer liquidus and enthalpy formulas.

author for JAX version: Xinyuan Zhu
XZ: converted from F90 to Python

The liquidus is Assur (1958). These functions are the ones initialization
calls: ``liquidus_temperature_mush`` from the freezing temperature, and
``icepack_enthalpy_mush`` from ``icepack_init_enthalpy``.

Ported: ``icepack_enthalpy_snow``, ``temperature_snow``, ``enthalpy_brine``,
``icepack_enthalpy_mush``, ``enthalpy_mush_liquid_fraction``,
``enthalpy_of_melting``, ``liquidus_brine_salinity_mush``,
``liquidus_temperature_mush``, ``icepack_mushy_liquid_fraction``.

Ported for history temperatures: ``icepack_mushy_temperature_mush``.

The time loop has JAX array formulas for ``conductivity_mush_array``,
``conductivity_snow_array``, ``temperature_mush_liquid_fraction`` and
``icepack_mushy_density_brine``. Mushy time integration is not connected yet.
"""

import numpy as np
import jax.numpy as jnp

from icepack.columnphysics import icepack_parameters
from icepack.columnphysics.icepack_kinds import dbl_kind
from icepack.columnphysics.icepack_parameters import c0, c1, c2, c4, c1000, puny

#-----------------------------------------------------------------
# Constants for Liquidus relation from Assur (1958)
#-----------------------------------------------------------------

# liquidus relation - higher temperature region
az1_liq = dbl_kind(-18.48)
bz1_liq = dbl_kind(0.0)

# liquidus relation - lower temperature region
az2_liq = dbl_kind(-10.3085)
bz2_liq = dbl_kind(62.4)

# liquidus break
Tb_liq = dbl_kind(-7.6362968855167352)  # temperature of liquidus break
Sb_liq = dbl_kind(123.66702800276086)   # salinity of liquidus break

# basic liquidus relation constants
az1p_liq = az1_liq / c1000
bz1p_liq = bz1_liq / c1000
az2p_liq = az2_liq / c1000
bz2p_liq = bz2_liq / c1000


def icepack_enthalpy_snow(zTsn):
    """Enthalpy of snow from snow temperature.

    zTsn: snow layer temperature (C)
    returns snow layer enthalpy (J m-3)
    """
    zTsn = np.asarray(zTsn, dtype=dbl_kind)
    return -icepack_parameters.rhos * (-icepack_parameters.cp_ice * zTsn + icepack_parameters.Lfresh)


def temperature_snow(zqsn):
    """Temperature of snow from the snow enthalpy.

    zqsn: snow layer enthalpy (J m-3)
    returns snow layer temperature (C)
    """
    zqsn = np.asarray(zqsn, dtype=dbl_kind)
    A = c1 / (icepack_parameters.rhos * icepack_parameters.cp_ice)
    B = icepack_parameters.Lfresh / icepack_parameters.cp_ice
    return A * zqsn + B


def enthalpy_brine(zTin):
    """Enthalpy of brine (fully liquid) from temperature.

    zTin: ice layer temperature (C)
    returns brine enthalpy (J m-3)
    """
    return icepack_parameters.cp_ocn * icepack_parameters.rhow * np.asarray(zTin, dtype=dbl_kind)


def liquidus_brine_salinity_mush(zTin):
    """Liquidus relation: equilibrium brine salinity as function of temperature
    based on empirical data from Assur (1958).

    zTin: ice layer temperature (C)
    returns ice brine salinity (ppt)
    """
    zTin = np.asarray(zTin, dtype=dbl_kind)
    # temperature to brine salinity
    J1_liq = bz1_liq / az1_liq
    K1_liq = c1 / c1000
    L1_liq = (c1 + bz1p_liq) / az1_liq
    J2_liq = bz2_liq / az2_liq
    K2_liq = c1 / c1000
    L2_liq = (c1 + bz2p_liq) / az2_liq

    t_high = np.where(zTin > Tb_liq, c1, c0)
    lsubzero = np.where(zTin <= c0, c1, c0)

    Sbr = ((zTin + J1_liq) / (K1_liq * zTin + L1_liq)) * t_high + (
        (zTin + J2_liq) / (K2_liq * zTin + L2_liq)
    ) * (c1 - t_high)
    return Sbr * lsubzero


def liquidus_temperature_mush(Sbr):
    """Liquidus temperature from brine salinity.

    Sbr: ice brine salinity (ppt)
    returns ice layer temperature (C)
    """
    Sbr = np.asarray(Sbr, dtype=dbl_kind)
    # brine salinity to temperature
    M1_liq = az1_liq
    N1_liq = -az1p_liq
    O1_liq = -bz1_liq / az1_liq
    M2_liq = az2_liq
    N2_liq = -az2p_liq
    O2_liq = -bz2_liq / az2_liq

    t_high = np.where(Sbr <= Sb_liq, c1, c0)
    return ((Sbr / (M1_liq + N1_liq * Sbr)) + O1_liq) * t_high + (
        (Sbr / (M2_liq + N2_liq * Sbr)) + O2_liq
    ) * (c1 - t_high)


def icepack_mushy_liquid_fraction(zTin, zSin):
    """Liquid fraction of mush from mush temperature and bulk salinity.

    zTin: ice layer temperature (C)
    zSin: ice layer bulk salinity (ppt)
    returns liquid fraction
    """
    zTin = np.asarray(zTin, dtype=dbl_kind)
    zSin = np.asarray(zSin, dtype=dbl_kind)
    Sbr = np.maximum(liquidus_brine_salinity_mush(zTin), icepack_parameters.puny)
    return zSin / np.maximum(Sbr, zSin)


def icepack_enthalpy_mush(zTin, zSin):
    """Enthalpy of mush from mush temperature and bulk salinity.

    zTin: ice layer temperature (C)
    zSin: ice layer bulk salinity (ppt)
    returns ice layer enthalpy (J m-3)
    """
    zTin = np.asarray(zTin, dtype=dbl_kind)
    phi = icepack_mushy_liquid_fraction(zTin, zSin)
    cp_ocn = icepack_parameters.cp_ocn
    cp_ice = icepack_parameters.cp_ice
    rhoi = icepack_parameters.rhoi
    rhow = icepack_parameters.rhow
    return (
        phi * (cp_ocn * rhow - cp_ice * rhoi) * zTin
        + rhoi * cp_ice * zTin
        - (c1 - phi) * rhoi * icepack_parameters.Lfresh
    )


def enthalpy_mush_liquid_fraction(zTin, phi):
    """Enthalpy of mush from mush temperature and liquid fraction.

    zTin: ice layer temperature (C)
    phi: liquid fraction
    returns ice layer enthalpy (J m-3)
    """
    zTin = np.asarray(zTin, dtype=dbl_kind)
    phi = np.asarray(phi, dtype=dbl_kind)
    cp_ocn = icepack_parameters.cp_ocn
    cp_ice = icepack_parameters.cp_ice
    rhoi = icepack_parameters.rhoi
    rhow = icepack_parameters.rhow
    return (
        phi * (cp_ocn * rhow - cp_ice * rhoi) * zTin
        + rhoi * cp_ice * zTin
        - (c1 - phi) * rhoi * icepack_parameters.Lfresh
    )


def enthalpy_of_melting(zSin):
    """Enthalpy of melting of mush from bulk salinity.
    Energy needed to fully melt mush (T < 0).

    zSin: ice layer bulk salinity (ppt)
    returns melting ice enthalpy (J m-3)
    """
    return icepack_parameters.cp_ocn * icepack_parameters.rhow * liquidus_temperature_mush(zSin)


def icepack_mushy_temperature_mush(zqin, zSin):
    """Temperature of mush from mush enthalpy and bulk salinity.

    zqin: ice enthalpy (J m-3)
    zSin: ice layer bulk salinity (ppt)
    returns ice layer temperature (C)
    """
    zqin = np.asarray(zqin, dtype=dbl_kind)
    zSin = np.asarray(zSin, dtype=dbl_kind)
    rhow = icepack_parameters.rhow
    rhoi = icepack_parameters.rhoi
    cp_ocn = icepack_parameters.cp_ocn
    cp_ice = icepack_parameters.cp_ice
    Lfresh = icepack_parameters.Lfresh

    # quadratic constants - higher temperature region
    AS1_liq = az1p_liq * (rhow * cp_ocn - rhoi * cp_ice)
    AC1_liq = rhoi * cp_ice * az1_liq
    BS1_liq = (c1 + bz1p_liq) * (rhow * cp_ocn - rhoi * cp_ice) + rhoi * Lfresh * az1p_liq
    BQ1_liq = -az1_liq
    BC1_liq = rhoi * cp_ice * bz1_liq - rhoi * Lfresh * az1_liq
    CS1_liq = rhoi * Lfresh * (c1 + bz1p_liq)
    CQ1_liq = -bz1_liq
    CC1_liq = -rhoi * Lfresh * bz1_liq

    # quadratic constants - lower temperature region
    AS2_liq = az2p_liq * (rhow * cp_ocn - rhoi * cp_ice)
    AC2_liq = rhoi * cp_ice * az2_liq
    BS2_liq = (c1 + bz2p_liq) * (rhow * cp_ocn - rhoi * cp_ice) + rhoi * Lfresh * az2p_liq
    BQ2_liq = -az2_liq
    BC2_liq = rhoi * cp_ice * bz2_liq - rhoi * Lfresh * az2_liq
    CS2_liq = rhoi * Lfresh * (c1 + bz2p_liq)
    CQ2_liq = -bz2_liq
    CC2_liq = -rhoi * Lfresh * bz2_liq

    # break enthalpy constants
    D_liq = (
        ((c1 + az1p_liq * Tb_liq + bz1p_liq) / (az1_liq * Tb_liq + bz1_liq))
        * ((cp_ocn * rhow - cp_ice * rhoi) * Tb_liq + Lfresh * rhoi)
    )
    E_liq = cp_ice * rhoi * Tb_liq - Lfresh * rhoi

    # just fully melted enthapy constants
    F1_liq = (-c1000 * cp_ocn * rhow) / az1_liq
    G1_liq = -c1000
    H1_liq = (-bz1_liq * cp_ocn * rhow) / az1_liq
    F2_liq = (-c1000 * cp_ocn * rhow) / az2_liq
    G2_liq = -c1000
    H2_liq = (-bz2_liq * cp_ocn * rhow) / az2_liq

    # warmer than fully melted constants
    I_liq = c1 / (cp_ocn * rhow)

    # just melted enthalpy
    S_low = np.where(zSin < Sb_liq, c1, c0)
    q0 = ((F1_liq * zSin) / (G1_liq + zSin) + H1_liq) * S_low + (
        (F2_liq * zSin) / (G2_liq + zSin) + H2_liq
    ) * (c1 - S_low)
    q_melt = np.where(zqin > q0, c1, c0)

    # break enthalpy
    qb = D_liq * zSin + E_liq
    t_high = np.where(zqin > qb, c1, c0)
    t_low = c1 - t_high

    # quadratic values
    A = (AS1_liq * zSin + AC1_liq) * t_high + (AS2_liq * zSin + AC2_liq) * t_low
    B = (BS1_liq * zSin + BQ1_liq * zqin + BC1_liq) * t_high + (
        BS2_liq * zSin + BQ2_liq * zqin + BC2_liq
    ) * t_low
    C = (CS1_liq * zSin + CQ1_liq * zqin + CC1_liq) * t_high + (
        CS2_liq * zSin + CQ2_liq * zqin + CC2_liq
    ) * t_low

    zTin = (-B + np.sqrt(np.maximum(B * B - c4 * A * C, puny))) / (c2 * A)

    # change T if all melted
    return q_melt * zqin * I_liq + (c1 - q_melt) * zTin


def liquidus_brine_salinity_mush_jax(zTin):
    """JAX equilibrium brine salinity for the two Assur liquidus ranges."""
    J1_liq = bz1_liq / az1_liq
    K1_liq = c1 / c1000
    L1_liq = (c1 + bz1p_liq) / az1_liq
    J2_liq = bz2_liq / az2_liq
    K2_liq = c1 / c1000
    L2_liq = (c1 + bz2p_liq) / az2_liq
    # Fortran multiplies by the precomputed reciprocal, not zTin / c1000.
    upper = (zTin + J1_liq) / (K1_liq * zTin + L1_liq)
    lower = (zTin + J2_liq) / (K2_liq * zTin + L2_liq)
    t_high = zTin > Tb_liq
    lsubzero = zTin <= c0
    Sbr = jnp.where(t_high, upper, lower)
    return jnp.where(lsubzero, Sbr, c0)


def liquidus_temperature_mush_jax(Sbr):
    """JAX inverse liquidus temperature from brine salinity."""
    upper = Sbr / (az1_liq - az1p_liq * Sbr) - bz1_liq / az1_liq
    lower = Sbr / (az2_liq - az2p_liq * Sbr) - bz2_liq / az2_liq
    return jnp.where(Sbr <= Sb_liq, upper, lower)


def icepack_mushy_liquid_fraction_jax(zTin, zSin, parameters):
    """JAX liquid volume fraction of saline sea ice."""
    Sbr = jnp.maximum(liquidus_brine_salinity_mush_jax(zTin),
                      parameters["puny"])
    return zSin / jnp.maximum(Sbr, zSin)


def enthalpy_brine_jax(zTin, parameters):
    """Enthalpy of completely liquid brine at the given temperature."""
    return parameters["cp_ocn"] * parameters["rhow"] * zTin


def icepack_enthalpy_snow_jax(zTsn, parameters):
    """Snow enthalpy from temperature, with differentiable parameters."""
    return -parameters["rhos"] * (
        -parameters["cp_ice"] * zTsn + parameters["Lfresh"])


def temperature_snow_jax(zqsn, parameters):
    """Snow temperature from enthalpy, with differentiable parameters."""
    # Preserve Fortran's reciprocal multiplication near the melting point.
    A = c1 / (parameters["rhos"] * parameters["cp_ice"])
    B = parameters["Lfresh"] / parameters["cp_ice"]
    return A * zqsn + B


def enthalpy_mush_liquid_fraction_jax(zTin, phi, parameters):
    """Mushy sea-ice enthalpy when the liquid volume fraction is known."""
    rhoi = parameters["rhoi"]
    return (phi * (parameters["cp_ocn"] * parameters["rhow"]
                   - parameters["cp_ice"] * rhoi) * zTin
            + rhoi * parameters["cp_ice"] * zTin
            - (1.0 - phi) * rhoi * parameters["Lfresh"])


def icepack_enthalpy_mush_jax(zTin, zSin, parameters):
    """Mushy sea-ice enthalpy from temperature and bulk salinity."""
    phi = icepack_mushy_liquid_fraction_jax(zTin, zSin, parameters)
    return enthalpy_mush_liquid_fraction_jax(zTin, phi, parameters)


def enthalpy_of_melting_jax(zSin, parameters):
    """Enthalpy of fully melted mush at its liquidus temperature."""
    return (parameters["cp_ocn"] * parameters["rhow"]
            * liquidus_temperature_mush_jax(zSin))


def icepack_mushy_temperature_mush_jax(zqin, zSin, parameters):
    """JAX inverse of mushy-layer enthalpy for bulk salinity."""
    rhow = parameters["rhow"]
    rhoi = parameters["rhoi"]
    cp_ocn = parameters["cp_ocn"]
    cp_ice = parameters["cp_ice"]
    Lfresh = parameters["Lfresh"]
    heat_difference = rhow * cp_ocn - rhoi * cp_ice

    AS1 = az1p_liq * heat_difference
    AC1 = rhoi * cp_ice * az1_liq
    BS1 = (1.0 + bz1p_liq) * heat_difference + rhoi * Lfresh * az1p_liq
    BQ1 = -az1_liq
    BC1 = rhoi * cp_ice * bz1_liq - rhoi * Lfresh * az1_liq
    CS1 = rhoi * Lfresh * (1.0 + bz1p_liq)
    CQ1 = -bz1_liq
    CC1 = -rhoi * Lfresh * bz1_liq

    AS2 = az2p_liq * heat_difference
    AC2 = rhoi * cp_ice * az2_liq
    BS2 = (1.0 + bz2p_liq) * heat_difference + rhoi * Lfresh * az2p_liq
    BQ2 = -az2_liq
    BC2 = rhoi * cp_ice * bz2_liq - rhoi * Lfresh * az2_liq
    CS2 = rhoi * Lfresh * (1.0 + bz2p_liq)
    CQ2 = -bz2_liq
    CC2 = -rhoi * Lfresh * bz2_liq

    D = (((1.0 + az1p_liq * Tb_liq + bz1p_liq)
          / (az1_liq * Tb_liq + bz1_liq))
         * (heat_difference * Tb_liq + Lfresh * rhoi))
    E = cp_ice * rhoi * Tb_liq - Lfresh * rhoi
    F1 = -1000.0 * cp_ocn * rhow / az1_liq
    H1 = -bz1_liq * cp_ocn * rhow / az1_liq
    F2 = -1000.0 * cp_ocn * rhow / az2_liq
    H2 = -bz2_liq * cp_ocn * rhow / az2_liq

    q0_upper = F1 * zSin / (zSin - 1000.0) + H1
    q0_lower = F2 * zSin / (zSin - 1000.0) + H2
    q0 = jnp.where(zSin < Sb_liq, q0_upper, q0_lower)
    q_melt = zqin > q0
    upper = zqin > D * zSin + E
    A = jnp.where(upper, AS1 * zSin + AC1, AS2 * zSin + AC2)
    B = jnp.where(upper, BS1 * zSin + BQ1 * zqin + BC1,
                  BS2 * zSin + BQ2 * zqin + BC2)
    C = jnp.where(upper, CS1 * zSin + CQ1 * zqin + CC1,
                  CS2 * zSin + CQ2 * zqin + CC2)
    Tin = (-B + jnp.sqrt(jnp.maximum(B * B - 4.0 * A * C,
                                    parameters["puny"]))) / (2.0 * A)
    return jnp.where(q_melt, zqin / (cp_ocn * rhow), Tin)


def temperature_mush_liquid_fraction_jax(zqin, phi, parameters):
    """Temperature from mush enthalpy and liquid fraction."""
    rhoi = parameters["rhoi"]
    return ((zqin + (1.0 - phi) * rhoi * parameters["Lfresh"])
            / (phi * (parameters["cp_ocn"] * parameters["rhow"]
                      - parameters["cp_ice"] * rhoi)
               + rhoi * parameters["cp_ice"]))


def icepack_mushy_density_brine_jax(Sbr):
    """Empirical brine density (kg/m3) from salinity (ppt)."""
    return 1000.3 + 0.78237 * Sbr + 2.8008e-4 * Sbr**2


def conductivity_mush_array_jax(zqin, zSin, parameters):
    """Mushy ice layer conductivity from enthalpy and salinity."""
    Tmush = icepack_mushy_temperature_mush_jax(zqin, zSin, parameters)
    phi = icepack_mushy_liquid_fraction_jax(Tmush, zSin, parameters)
    return phi * (0.5375 - 2.3) + 2.3


def conductivity_snow_array_jax(shape, parameters):
    """Constant conductivity for each snow layer."""
    return jnp.full(shape, parameters["ksno"])
