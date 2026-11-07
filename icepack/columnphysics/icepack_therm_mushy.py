"""Mushy sea-ice temperature and salinity routines from Icepack.

These routines operate on one column. Ice and snow layer axes follow the
Fortran order, from the upper layer to the lower layer.
"""

from functools import partial

import jax
import jax.numpy as jnp
from jax import lax

from icepack.columnphysics.icepack_parameters import c0, c1

from icepack.columnphysics.icepack_mushy_physics import (
    enthalpy_brine_jax,
    enthalpy_mush_liquid_fraction_jax,
    conductivity_mush_array_jax,
    conductivity_snow_array_jax,
    icepack_mushy_density_brine_jax,
    icepack_mushy_liquid_fraction_jax,
    icepack_enthalpy_snow_jax,
    icepack_mushy_temperature_mush_jax,
    liquidus_brine_salinity_mush_jax,
    liquidus_temperature_mush_jax,
    temperature_mush_liquid_fraction_jax,
    temperature_snow_jax,
)
from icepack.columnphysics.icepack_meltpond_sealvl import (
    pond_height,
    pond_hypsometry,
)
from icepack.columnphysics.icepack_therm_shared import (
    dsurface_heat_flux_dTsf,
    surface_heat_flux,
)


# Solver-exit values, before temperature reconstruction and snow/ice flooding.
# Fluxes: W/m2; temperatures: deg C except potT (K); layer thicknesses: m.
SURFACE_DIAGNOSTIC_FIELDS = (
    "surface_Tsf", "surface_zTsn_top", "surface_zTin_top",
    "surface_hslyr", "surface_hilyr", "surface_ks_top", "surface_km_top",
    "surface_lsnow", "surface_fswsfc", "surface_flwdabs", "surface_flwoutn",
    "surface_fsensn", "surface_flatn", "surface_fsurfn", "surface_fcondtopn",
    "surface_flux_residual", "surface_energy_residual",
    "surface_potT", "surface_Qa", "surface_rhoa", "surface_shcoef",
    "surface_lhcoef", "surface_converged",
)


PICARD_DIAGNOSTIC_FIELDS = (
    "nit",
    "lsnow",
    "lcold",
    "Tsf",
    "zTsn_top",
    "zTin_top",
    "fsurfn",
    "dfsurfn_dTsf",
    "fcondtop",
    "fcondbot",
    "fadvheat",
    "converged",
    "shcoef",
    "lhcoef",
    "zqin_top",
    "zqsn_top",
    "zSin_top",
    "phi_top",
)


def permeability(phi):
    """Permeability of mushy ice from its liquid volume fraction."""
    return 3.0e-8 * jnp.maximum(phi - 0.05, 0.0)**3


def explicit_flow_velocities(zSin, zTin, Tsf, Tbot, dt, sss,
                             qocn, hilyr, hin, parameters):
    """Compute rapid brine drainage and slow desalination in one column."""
    nilyr = zSin.shape[-1]
    Sbr = jnp.zeros((nilyr + 1,), dtype=zSin.dtype)
    qbr = jnp.zeros((nilyr + 1,), dtype=zTin.dtype)
    rho = jnp.zeros((nilyr + 1,), dtype=zTin.dtype)
    phi = jnp.zeros_like(zSin)
    q = jnp.zeros((nilyr + 1,), dtype=zTin.dtype)
    dSdt = jnp.zeros_like(zSin)
    for k in range(nilyr):
        Sbr = Sbr.at[k].set(liquidus_brine_salinity_mush_jax(zTin[k]))
        phi = phi.at[k].set(icepack_mushy_liquid_fraction_jax(
            zTin[k], zSin[k], parameters))
        qbr = qbr.at[k].set(enthalpy_brine_jax(zTin[k], parameters))
        rho = rho.at[k].set(icepack_mushy_density_brine_jax(Sbr[k]))

    # The final slot holds ocean brine; q[0] is the impermeable top surface.
    Sbr = Sbr.at[nilyr].set(sss)
    qbr = qbr.at[nilyr].set(qocn)
    rho_ocn = icepack_mushy_density_brine_jax(sss)
    perm_min = parameters["bignum"]
    perm_harm = jnp.zeros_like(Tsf)
    rho_sum = jnp.zeros_like(Tsf)
    qlimit = 0.2 * hilyr / dt
    ra_constants = (parameters["gravit"] /
                    (parameters["viscosity_dyn"] * 8.824e-8))

    for k in range(nilyr - 1, -1, -1):
        z = ((k + 0.5) / nilyr) * hin
        perm = permeability(phi[k])
        perm_min = jnp.minimum(perm_min, perm)
        perm_harm = perm_harm + 1.0 / jnp.maximum(perm, 1.0e-30)
        rho_sum = rho_sum + rho[k]
        rho_pipe = 0.5 * (rho[k] + rho[max(k - 1, 0)])
        drho = jnp.maximum(rho[k] - rho_ocn, 0.0)
        Ra = drho * (hin - z) * perm_min * ra_constants
        rn = nilyr - k
        L = rn * hilyr
        dx = L * 2.0 * parameters["aspect_rapid_mode"]
        dx2 = dx**2
        Am = dx2 * rn / (parameters["viscosity_dyn"] * perm_harm)
        Bm = -parameters["gravit"] * rho_sum / rn
        Ap = (parameters["pi"] * parameters["a_rapid_mode"]**4
              / (8.0 * parameters["viscosity_dyn"]))
        Bp = -rho_pipe * parameters["gravit"]
        qk = jnp.maximum(
            (Am / jnp.where(dx2 > 0.0, dx2, 1.0))
            * ((-Ap * Bp - Am * Bm) / (Am + Ap) + Bm), 1.0e-30)
        qk = jnp.minimum(
            qk * (jnp.maximum(Ra - parameters["Rac_rapid_mode"], 0.0)
                  / (Ra + parameters["puny"])), qlimit)
        dSdtk = (parameters["dSdt_slow_mode"]
                 * (jnp.maximum(zSin[k] - parameters["phi_c_slow_mode"]
                                * Sbr[k], 0.0)
                    * jnp.maximum(Tbot - Tsf, 0.0)) / (hin + 0.001))
        dSdtk = jnp.maximum(dSdtk, -0.5 * zSin[k] / dt)
        dS_guess = ((qk * (Sbr[k + 1] - Sbr[k]) /
                    jnp.where(hilyr > 0.0, hilyr, 1.0) + dSdtk)
                   * dt * 10.0)
        alpha = jnp.where(jnp.abs(dS_guess) < parameters["puny"],
                          1.0, (0.1 - zSin[k]) /
                          jnp.where(jnp.abs(dS_guess) < parameters["puny"],
                                    1.0, dS_guess))
        alpha = jnp.where((alpha < 0.0) | (alpha > 1.0), 1.0, alpha)
        q = q.at[k + 1].set(qk * alpha)
        dSdt = dSdt.at[k].set(dSdtk * alpha)
    return q, dSdt, Sbr, qbr


def flushing_velocity(zTin, phi, hin, hsn, hilyr, hpond, apond,
                      dt, options, parameters):
    """Darcy flow of pond water downward through permeable mushy ice."""
    if not options["tr_pond"]:
        return jnp.zeros_like(hin)

    ice_mass = jnp.zeros_like(hin)
    perm_harm = jnp.zeros_like(hin)
    phi_min = jnp.ones_like(hin)
    for k in range(zTin.shape[-1]):
        phi_min = jnp.minimum(phi_min, phi[k])
        brine_salinity = liquidus_brine_salinity_mush_jax(zTin[k])
        ice_mass = (ice_mass
                    + phi[k] * icepack_mushy_density_brine_jax(brine_salinity)
                    + (1.0 - phi[k]) * parameters["rhoi"])
        perm_harm = perm_harm + 1.0 / (permeability(phi[k]) + 1.0e-30)
    ice_mass = ice_mass * hilyr
    perm_harm = zTin.shape[-1] / perm_harm
    hocn = (ice_mass + hpond * apond * parameters["rhofresh"]
            + hsn * parameters["rhos"]) / parameters["rhow"]
    if options["tr_pond_sealvl"]:
        hbrine = pond_height(apond, hpond, hin, options, parameters)
    else:
        hbrine = hin + hpond
    dhhead = jnp.maximum(hbrine - hocn, 0.0)
    w = (perm_harm * parameters["rhow"] * parameters["gravit"]
         * (dhhead / jnp.where(hin > 0.0, hin, 1.0))
         / parameters["viscosity_dyn"])
    w = jnp.minimum(w, hpond * apond / dt)
    wlimit = 0.005 * phi_min * hilyr / dt
    w = jnp.where(jnp.abs(w) > parameters["puny"],
                  w * jnp.clip(wlimit /
                               jnp.where(jnp.abs(w) > parameters["puny"],
                                         jnp.abs(w), 1.0), 0.0, 1.0), 0.0)
    return jnp.maximum(w, 0.0)


def flush_pond(w, hpond, apond, dt, dpnd_flush, dpnd_expon,
               zTin, phi, hilyr, hin, hsn, options, parameters):
    """Drain melt ponds by percolation and macroscopic flaws."""
    if not options["tr_pond"]:
        return hpond, apond, dpnd_flush, dpnd_expon

    active = (apond > 0.0) & (hpond > 0.0)
    dhpond = jnp.maximum(-w * dt /
                         jnp.where(apond > 0.0, apond, 1.0), -hpond)
    dpnd_flush = jnp.where(active, -dhpond * apond, dpnd_flush)
    if options["tr_pond_sealvl"]:
        drained_depth, drained_area = pond_hypsometry(
            hpond, apond, options, parameters, dhpond=dhpond, hin=hin)
    else:
        drained_depth = (hpond - w * dt /
                         jnp.where(apond > 0.0, apond, 1.0))
        drained_area = apond
    drained_depth = jnp.maximum(drained_depth, 0.0)

    lambda_pond = 1.0 / (parameters["tscale_pnd_drain"] * 24.0 * 3600.0)
    if options["pndmacr"] == "lambda":
        macro_change = jnp.maximum(
            -lambda_pond * dt * (drained_depth + 0.01), -drained_depth)
    elif options["pndmacr"] == "head":
        ice_mass = jnp.zeros_like(hin)
        for k in range(zTin.shape[-1]):
            ice_mass = ice_mass + (
                phi[k] * icepack_mushy_density_brine_jax(
                    liquidus_brine_salinity_mush_jax(zTin[k]))
                + (1.0 - phi[k]) * parameters["rhoi"])
        ice_mass = ice_mass * hilyr
        hocn = (ice_mass + drained_depth * drained_area
                * parameters["rhofresh"] + hsn * parameters["rhos"])
        hocn = hocn / parameters["rhow"]
        hpsurf = pond_height(drained_area, drained_depth, hin,
                             options, parameters)
        head = hpsurf - hocn
        macro_change = jnp.maximum(
            jnp.minimum(0.0, -lambda_pond * dt * head), -drained_depth)
    else:
        raise ValueError("unsupported pndmacr option")
    dpnd_expon = jnp.where(active, -macro_change * drained_area,
                            dpnd_expon)
    if options["tr_pond_sealvl"]:
        final_depth, final_area = pond_hypsometry(
            drained_depth, drained_area, options, parameters,
            dhpond=macro_change, hin=hin)
    else:
        if options["pndmacr"] != "lambda":
            raise ValueError("head drainage requires sea-level ponds")
        final_depth = (drained_depth - lambda_pond * dt
                       * (drained_depth + 0.01))
        final_area = drained_area
    hpond = jnp.where(active, jnp.maximum(final_depth, 0.0), hpond)
    apond = jnp.where(active, final_area, apond)
    return hpond, apond, dpnd_flush, dpnd_expon


def enthalpy_snow_snowice(dh, hsn, zqsn, parameters):
    """Average the enthalpy of snow converted from the bottom upward."""
    nslyr = zqsn.shape[-1]
    rnlyr = dh / jnp.where(hsn > parameters["puny"], hsn, 1.0) * nslyr
    nlyr = jnp.minimum(jnp.floor(rnlyr).astype(jnp.int32), nslyr - 1)
    # Every positive conversion retains its snow enthalpy, even below puny.
    active = (hsn > parameters["puny"]) & (rnlyr > 0.0)
    safe_rnlyr = jnp.where(active, rnlyr, 1.0)
    snowice_enthalpy = jnp.zeros_like(hsn)
    for k in range(nslyr - 1, -1, -1):
        full_layer = k >= nslyr - nlyr
        partial_layer = k == nslyr - nlyr - 1
        contribution = jnp.where(
            full_layer, zqsn[k] / safe_rnlyr,
            jnp.where(partial_layer,
                      ((rnlyr - nlyr) / safe_rnlyr) * zqsn[k], 0.0))
        snowice_enthalpy = snowice_enthalpy + contribution
    return jnp.where(active, snowice_enthalpy, 0.0)


def update_vertical_tracers_snow(trc, hlyr1, hlyr2):
    """Remap snow tracer values after snow-ice formation."""
    nslyr = trc.shape[-1]
    trc2 = jnp.zeros_like(trc)
    for k2 in range(nslyr):
        z2a = k2 * hlyr2
        z2b = (k2 + 1) * hlyr2
        value = jnp.zeros_like(hlyr2)
        for k1 in range(nslyr):
            z1a = k1 * hlyr1
            z1b = (k1 + 1) * hlyr1
            overlap = jnp.maximum(jnp.minimum(z1b, z2b)
                                  - jnp.maximum(z1a, z2a), 0.0)
            value = value + overlap * trc[k1]
        trc2 = trc2.at[k2].set(value / jnp.where(hlyr2 > 0.0, hlyr2, 1.0))
    return trc2


def update_vertical_tracers_ice(trc, hlyr1, hlyr2, h1, h2, trc0):
    """Remap ice tracer values with new snow ice placed above old ice."""
    nilyr = trc.shape[-1]
    trc2 = jnp.zeros_like(trc)
    for k2 in range(nilyr):
        z2a = k2 * hlyr2
        z2b = (k2 + 1) * hlyr2
        overlap = jnp.maximum(jnp.minimum(h2 - h1, z2b) - z2a, 0.0)
        value = overlap * trc0
        for k1 in range(nilyr):
            z1a = k1 * hlyr1 + h2 - h1
            z1b = (k1 + 1) * hlyr1 + h2 - h1
            overlap = jnp.maximum(jnp.minimum(z1b, z2b)
                                  - jnp.maximum(z1a, z2a), 0.0)
            value = value + overlap * trc[k1]
        trc2 = trc2.at[k2].set(value / jnp.where(hlyr2 > 0.0, hlyr2, 1.0))
    return trc2


def flood_ice(hsn, hin, hslyr, hilyr, zqsn, zqin, phi, dt,
              zSin, Sbr, sss, qocn, smice, smliq, fadvheat,
              snwgrain, parameters):
    """Convert flooded snow to saline snow ice and remap the layers."""
    rho_ocn = icepack_mushy_density_brine_jax(sss)
    ice_mass = jnp.zeros_like(hin)
    for k in range(zqin.shape[-1]):
        ice_density = jnp.minimum(
            phi[k] * icepack_mushy_density_brine_jax(Sbr[k])
            + (1.0 - phi[k]) * parameters["rhoi"], rho_ocn)
        ice_mass = ice_mass + ice_density
    ice_mass = ice_mass * hilyr
    freeboard_density = jnp.maximum(
        ice_mass + hsn * parameters["rhos"] - hin * rho_ocn, 0.0)
    flooded = (hsn > parameters["puny"]) & (freeboard_density > 0.0)
    phi_snowice = 1.0 - parameters["rhos"] / parameters["rhoi"]
    rho_snowice = (phi_snowice * rho_ocn
                   + (1.0 - phi_snowice) * parameters["rhoi"])
    dh = jnp.where(
        flooded,
        jnp.clip(freeboard_density /
                 (rho_ocn - rho_snowice + parameters["rhos"]), 0.0, hsn),
        0.0)
    zqsn_snowice = enthalpy_snow_snowice(dh, hsn, zqsn, parameters)
    hin2 = hin + dh
    hsn2 = hsn - dh
    hilyr2 = hin2 / zqin.shape[-1]
    hslyr2 = hsn2 / zqsn.shape[-1]
    zSin_snowice = phi_snowice * sss
    zqin_snowice = phi_snowice * qocn + zqsn_snowice
    snow_enthalpy = update_vertical_tracers_snow(zqsn, hslyr, hslyr2)
    ice_enthalpy = update_vertical_tracers_ice(
        zqin, hilyr, hilyr2, hin, hin2, zqin_snowice)
    ice_salinity = update_vertical_tracers_ice(
        zSin, hilyr, hilyr2, hin, hin2, zSin_snowice)
    ice_liquid = update_vertical_tracers_ice(
        phi, hilyr, hilyr2, hin, hin2, phi_snowice)
    zqsn = jnp.where(flooded, snow_enthalpy, zqsn)
    zqin = jnp.where(flooded, ice_enthalpy, zqin)
    zSin = jnp.where(flooded, ice_salinity, zSin)
    phi = jnp.where(flooded, ice_liquid, phi)
    if snwgrain:
        snow_remaining = flooded & (hslyr2 > parameters["puny"])
        smice = jnp.where(snow_remaining,
                          update_vertical_tracers_snow(smice, hslyr, hslyr2),
                          smice)
        smliq = jnp.where(snow_remaining,
                          update_vertical_tracers_snow(smliq, hslyr, hslyr2),
                          smliq)
    hilyr = jnp.where(flooded, hilyr2, hilyr)
    hslyr = jnp.where(flooded, hslyr2, hslyr)
    snoice = dh
    hadded = (dh * phi_snowice) / dt
    fadvheat = fadvheat - hadded * qocn
    return (hslyr, hilyr, zqsn, zqin, phi, zSin,
            smice, smliq, snoice, fadvheat)


def solve_salinity(zSin, Sbr, Spond, sss, q, dSdt, w,
                   hilyr, dt):
    """Update bulk salinity with rapid drainage and downward flushing."""
    nilyr = zSin.shape[-1]
    hlyr = jnp.where(hilyr > 0.0, hilyr, 1.0)
    k = 0
    change = ((q[k + 1] * (Sbr[k + 1] - Sbr[k])) / hlyr
              + dSdt[k] + w * (Spond - Sbr[k]) / hlyr) * dt
    zSin = zSin.at[k].set(zSin[k] + jnp.maximum(0.01 - zSin[k], change))
    for k in range(1, nilyr - 1):
        change = ((q[k + 1] * (Sbr[k + 1] - Sbr[k])) / hlyr
                  + dSdt[k] + w * (Sbr[k - 1] - Sbr[k]) / hlyr) * dt
        zSin = zSin.at[k].set(zSin[k] + jnp.maximum(0.01 - zSin[k], change))
    k = nilyr - 1
    change = ((q[k + 1] * (sss - Sbr[k])) / hlyr
              + dSdt[k] + w * (Sbr[k - 1] - Sbr[k]) / hlyr) * dt
    zSin = zSin.at[k].set(zSin[k] + jnp.maximum(0.01 - zSin[k], change))
    return zSin, jnp.min(zSin) < 0.0


def tdma_solve_sparse(a, b, c, d):
    """Solve the sparse tridiagonal system with Fortran's sweep order."""
    n = d.shape[-1]
    # Static scalar lists avoid repeatedly scattering and slicing layer arrays.
    cp = [(c[0] / b[0]).astype(d.dtype)]
    for i in range(1, n - 1):
        cp.append((c[i] / (b[i] - cp[i - 1] * a[i])).astype(d.dtype))
    dp = [(d[0] / b[0]).astype(d.dtype)]
    for i in range(1, n):
        dp.append(((d[i] - dp[i - 1] * a[i]) /
                   (b[i] - cp[i - 1] * a[i])).astype(d.dtype))
    x = [dp[-1]]
    for i in range(n - 2, -1, -1):
        x.append((dp[i] - cp[i] * x[-1]).astype(d.dtype))
    return jnp.stack(x[::-1])


def picard_drainage_fluxes(q, qbr, qocn):
    """Advective heat flux from upward gravity drainage."""
    nilyr = q.shape[-1] - 1
    fadvheat = jnp.zeros_like(qocn)
    for k in range(nilyr - 1):
        fadvheat = fadvheat - q[k + 1] * (qbr[k + 1] - qbr[k])
    fadvheat = fadvheat - q[nilyr] * (qocn - qbr[nilyr - 1])
    return fadvheat


def picard_flushing_fluxes(fadvheat, w, qbr, qpond):
    """Add advective heat from melt-pond flushing."""
    return fadvheat + w * (qbr[-1] - qpond)


def maximum_variables_changes(lsnow, Tsf, Tsf_prev,
                              zTsn, zTsn_prev, zTin, zTin_prev):
    """Largest temperature changes between consecutive Picard iterations."""
    dTsf = jnp.abs(Tsf - Tsf_prev)
    dzTsn = jnp.where(lsnow, jnp.max(jnp.abs(zTsn - zTsn_prev)), 0.0)
    dzTin = jnp.max(jnp.abs(zTin - zTin_prev))
    return dTsf, dzTsn, dzTin


def total_energy_content(lsnow, zqin, zqsn, hilyr, hslyr):
    """Ice and optional snow enthalpy integrated over one column."""
    energy = jnp.zeros_like(hilyr)
    for k in range(zqsn.shape[-1]):
        energy = energy + jnp.where(lsnow, hslyr * zqsn[k], 0.0)
    for k in range(zqin.shape[-1]):
        energy = energy + hilyr * zqin[k]
    return energy


def picard_updates(zTin, Sbr, qbr, parameters):
    """Refresh brine salinity and enthalpy after a temperature solve."""
    for k in range(zTin.shape[-1]):
        Sbr = Sbr.at[k].set(liquidus_brine_salinity_mush_jax(zTin[k]))
        qbr = qbr.at[k].set(enthalpy_brine_jax(zTin[k], parameters))
    return Sbr, qbr


def picard_updates_enthalpy(zTin, qbr, parameters):
    """Refresh only the brine enthalpy after a temperature solve."""
    for k in range(zTin.shape[-1]):
        qbr = qbr.at[k].set(enthalpy_brine_jax(zTin[k], parameters))
    return qbr


def picard_final(lsnow, zqin, zqsn, zTin, zTsn, phi, parameters):
    """Return converged temperatures to ice and snow enthalpy."""
    for k in range(zTin.shape[-1]):
        zqin = zqin.at[k].set(
            enthalpy_mush_liquid_fraction_jax(zTin[k], phi[k], parameters))
    for k in range(zTsn.shape[-1]):
        zqsn = zqsn.at[k].set(jnp.where(
            lsnow, icepack_enthalpy_snow_jax(zTsn[k], parameters), zqsn[k]))
    return zqin, zqsn


def calc_intercell_thickness(lsnow, hilyr, hslyr, nilyr, nslyr):
    """Distances between centers and surfaces of the layered column."""
    snow_dxp = jnp.zeros((nilyr + nslyr + 1,), dtype=hilyr.dtype)
    snow_dxp = snow_dxp.at[0].set(hslyr / 2.0)
    for l in range(1, nslyr):
        snow_dxp = snow_dxp.at[l].set(hslyr)
    snow_dxp = snow_dxp.at[nslyr].set((hilyr + hslyr) / 2.0)
    for l in range(nslyr + 1, nilyr + nslyr):
        snow_dxp = snow_dxp.at[l].set(hilyr)
    snow_dxp = snow_dxp.at[nilyr + nslyr].set(hilyr / 2.0)
    ice_dxp = jnp.zeros_like(snow_dxp)
    ice_dxp = ice_dxp.at[0].set(hilyr / 2.0)
    for l in range(1, nilyr):
        ice_dxp = ice_dxp.at[l].set(hilyr)
    ice_dxp = ice_dxp.at[nilyr].set(hilyr / 2.0)
    return jnp.where(lsnow, snow_dxp, ice_dxp)


def calc_intercell_conductivity(lsnow, km, ks, hilyr, hslyr):
    """Harmonic conductivity at snow and ice layer interfaces."""
    nilyr = km.shape[-1]
    nslyr = ks.shape[-1]
    snow_kcstar = jnp.zeros((nilyr + nslyr + 1,), dtype=km.dtype)
    snow_kcstar = snow_kcstar.at[0].set(ks[0])
    for l in range(1, nslyr):
        snow_kcstar = snow_kcstar.at[l].set(
            2.0 * ks[l] * ks[l - 1] / (ks[l] + ks[l - 1]))
    fe = hilyr / (hilyr + hslyr)
    snow_kcstar = snow_kcstar.at[nslyr].set(
        1.0 / ((1.0 - fe) / ks[-1] + fe / km[0]))
    for k in range(1, nilyr):
        snow_kcstar = snow_kcstar.at[nslyr + k].set(
            2.0 * km[k] * km[k - 1] / (km[k] + km[k - 1]))
    snow_kcstar = snow_kcstar.at[nilyr + nslyr].set(km[-1])
    ice_kcstar = jnp.zeros_like(snow_kcstar)
    ice_kcstar = ice_kcstar.at[0].set(km[0])
    for k in range(1, nilyr):
        ice_kcstar = ice_kcstar.at[k].set(
            2.0 * km[k] * km[k - 1] / (km[k] + km[k - 1]))
    ice_kcstar = ice_kcstar.at[nilyr].set(km[-1])
    return jnp.where(lsnow, snow_kcstar, ice_kcstar)


def prep_picard(lsnow, zqsn, zqin, zSin, hilyr, hslyr,
                km, ks, zTin, zTsn, parameters):
    """Set initial Picard temperatures, interface geometry and energy."""
    Sbr = jnp.zeros_like(zSin)
    phi = jnp.zeros_like(zSin)
    for k in range(zqin.shape[-1]):
        zTin = zTin.at[k].set(
            icepack_mushy_temperature_mush_jax(zqin[k], zSin[k], parameters))
        Sbr = Sbr.at[k].set(liquidus_brine_salinity_mush_jax(zTin[k]))
        phi = phi.at[k].set(
            icepack_mushy_liquid_fraction_jax(zTin[k], zSin[k], parameters))
    for k in range(zqsn.shape[-1]):
        zTsn = zTsn.at[k].set(jnp.where(
            lsnow, temperature_snow_jax(zqsn[k], parameters), zTsn[k]))
    dxp = calc_intercell_thickness(
        lsnow, hilyr, hslyr, zqin.shape[-1], zqsn.shape[-1])
    kcstar = calc_intercell_conductivity(lsnow, km, ks, hilyr, hslyr)
    einit = total_energy_content(lsnow, zqin, zqsn, hilyr, hslyr)
    return zTin, zTsn, Sbr, phi, dxp, kcstar, einit


def check_picard_convergence(lsnow, Tsf, Tsf_prev, zTin, zTin_prev,
                             zTsn, zTsn_prev, phi, Tbot, zqin, zqsn,
                             km, ks, hilyr, hslyr, fswint, einit, dt,
                             fadvheat, parameters):
    """Check temperature iteration and column energy conservation."""
    zqin, zqsn = picard_final(
        lsnow, zqin, zqsn, zTin, zTsn, phi, parameters)
    efinal = total_energy_content(lsnow, zqin, zqsn, hilyr, hslyr)
    dTsf, dzTsn, dzTin = maximum_variables_changes(
        lsnow, Tsf, Tsf_prev, zTsn, zTsn_prev, zTin, zTin_prev)
    # Match Fortran's grouping: divide the temperature difference first.
    # Roundoff in these fluxes can change the sign of surface melt energy.
    fcondbot = 2.0 * km[-1] * ((zTin[-1] - Tbot) / hilyr)
    snow_fcondtop = (2.0 * ks[0] * ((Tsf - zTsn[0]) /
                     jnp.where(hslyr > 0.0, hslyr, 1.0)))
    ice_fcondtop = 2.0 * km[0] * ((Tsf - zTin[0]) / hilyr)
    fcondtop = jnp.where(lsnow, snow_fcondtop, ice_fcondtop)
    ferr = ((efinal - einit) / dt
            - (fcondtop - fcondbot + fswint - fadvheat))
    converged = ((dTsf < 5.0e-4) & (dzTsn < 5.0e-4)
                 & (dzTin < 5.0e-4) & (jnp.abs(ferr) < 0.9e-3))
    return zqin, zqsn, fcondtop, fcondbot, ferr, converged


def matrix_elements_nosnow_melt(Tsf, Tbot, zqin0, qpond, qocn,
                                phi, q, w, hilyr, dxp, kcstar,
                                Iswabs, dt, parameters):
    """Mushy heat matrix for ice without snow and a melting surface."""
    nilyr = zqin0.shape[-1]
    Ap = jnp.zeros_like(zqin0)
    As = jnp.zeros_like(zqin0)
    An = jnp.zeros_like(zqin0)
    b = jnp.zeros_like(zqin0)
    cp_ocn = parameters["cp_ocn"]
    rhow = parameters["rhow"]
    rhoi = parameters["rhoi"]
    cp_ice = parameters["cp_ice"]
    Lfresh = parameters["Lfresh"]
    for k in range(nilyr):
        # Preserve Fortran's division before layer scaling and separate
        # drainage/flushing terms; reassociation changes roundoff.
        storage = ((phi[k] * (cp_ocn * rhow - cp_ice * rhoi)
                    + rhoi * cp_ice) / dt) * hilyr
        upper = kcstar[k] / dxp[k]
        lower = kcstar[k + 1] / dxp[k + 1]
        Ap = Ap.at[k].set(storage + lower + upper
                          + q[k + 1] * cp_ocn * rhow + w * cp_ocn * rhow)
        As = As.at[k].set(-lower - q[k + 1] * cp_ocn * rhow)
        An = An.at[k].set(-upper - w * cp_ocn * rhow)
        b = b.at[k].set(
            (((c1 - phi[k]) * rhoi * Lfresh + zqin0[k]) / dt) * hilyr
            + Iswabs[k])
    b = b.at[0].set(b[0] + (kcstar[0] / dxp[0]) * Tsf + w * qpond)
    An = An.at[0].set(c0)
    b = b.at[-1].set(b[-1] + (kcstar[nilyr] * Tbot) / dxp[nilyr]
                     + q[nilyr] * qocn)
    As = As.at[-1].set(c0)
    return An, Ap, As, b


def matrix_elements_nosnow_cold(Tsf, Tbot, zqin0, qpond, qocn,
                                phi, q, w, hilyr, dxp, kcstar,
                                Iswabs, fsurfn, dfsurfn_dTsf, dt,
                                parameters):
    """Mushy heat matrix for ice without snow and a cold surface."""
    nilyr = zqin0.shape[-1]
    An = jnp.zeros((nilyr + 1,), dtype=zqin0.dtype)
    Ap = jnp.zeros_like(An)
    As = jnp.zeros_like(An)
    b = jnp.zeros_like(An)
    surface_conduct = kcstar[0] / dxp[0]
    Ap = Ap.at[0].set(dfsurfn_dTsf - surface_conduct)
    As = As.at[0].set(surface_conduct)
    b = b.at[0].set(dfsurfn_dTsf * Tsf - fsurfn)
    cp_ocn = parameters["cp_ocn"]
    rhow = parameters["rhow"]
    rhoi = parameters["rhoi"]
    cp_ice = parameters["cp_ice"]
    Lfresh = parameters["Lfresh"]
    for k in range(nilyr):
        l = k + 1
        storage = ((phi[k] * (cp_ocn * rhow - cp_ice * rhoi)
                    + rhoi * cp_ice) / dt) * hilyr
        upper = kcstar[k] / dxp[k]
        lower = kcstar[k + 1] / dxp[k + 1]
        Ap = Ap.at[l].set(storage + lower + upper
                          + q[k + 1] * cp_ocn * rhow + w * cp_ocn * rhow)
        As = As.at[l].set(-lower - q[k + 1] * cp_ocn * rhow)
        An = An.at[l].set(-upper - w * cp_ocn * rhow)
        b = b.at[l].set(
            (((c1 - phi[k]) * rhoi * Lfresh + zqin0[k]) / dt) * hilyr
            + Iswabs[k])
    An = An.at[1].set(-surface_conduct)
    b = b.at[1].add(w * qpond)
    As = As.at[-1].set(c0)
    b = b.at[-1].set(b[-1] + (kcstar[nilyr] * Tbot) / dxp[nilyr]
                     + q[nilyr] * qocn)
    return An, Ap, As, b


def matrix_elements_snow_melt(Tsf, Tbot, zqin0, zqsn0,
                              qpond, qocn, phi, q, w, hilyr, hslyr,
                              dxp, kcstar, Iswabs, Sswabs, dt,
                              parameters):
    """Mushy heat matrix for snow-covered ice with a melting surface."""
    nilyr = zqin0.shape[-1]
    nslyr = zqsn0.shape[-1]
    An = jnp.zeros((nilyr + nslyr,), dtype=zqin0.dtype)
    Ap = jnp.zeros_like(An)
    As = jnp.zeros_like(An)
    b = jnp.zeros_like(An)
    rhos = parameters["rhos"]
    rhoi = parameters["rhoi"]
    cp_ice = parameters["cp_ice"]
    Lfresh = parameters["Lfresh"]
    cp_ocn = parameters["cp_ocn"]
    rhow = parameters["rhow"]
    for k in range(nslyr):
        upper = kcstar[k] / dxp[k]
        lower = kcstar[k + 1] / dxp[k + 1]
        Ap = Ap.at[k].set(((rhos * cp_ice) / dt) * hslyr + lower + upper)
        As = As.at[k].set(-lower)
        An = An.at[k].set(-upper)
        b = b.at[k].set(((rhos * Lfresh + zqsn0[k]) / dt) * hslyr
                         + Sswabs[k])
    An = An.at[0].set(c0)
    b = b.at[0].add(kcstar[0] * Tsf / dxp[0])
    for k in range(nilyr):
        l = nslyr + k
        upper = kcstar[l] / dxp[l]
        lower = kcstar[l + 1] / dxp[l + 1]
        storage = ((phi[k] * (cp_ocn * rhow - cp_ice * rhoi)
                    + rhoi * cp_ice) / dt) * hilyr
        Ap = Ap.at[l].set(storage + lower + upper
                          + q[k + 1] * cp_ocn * rhow + w * cp_ocn * rhow)
        As = As.at[l].set(-lower - q[k + 1] * cp_ocn * rhow)
        An = An.at[l].set(-upper - w * cp_ocn * rhow)
        b = b.at[l].set(
            (((c1 - phi[k]) * rhoi * Lfresh + zqin0[k]) / dt) * hilyr
            + Iswabs[k])
    An = An.at[nslyr].set(-kcstar[nslyr] / dxp[nslyr])
    b = b.at[nslyr].add(w * qpond)
    As = As.at[-1].set(c0)
    b = b.at[-1].set(b[-1] + (kcstar[nilyr + nslyr] * Tbot) /
                     dxp[nilyr + nslyr] + q[nilyr] * qocn)
    return An, Ap, As, b


def matrix_elements_snow_cold(Tsf, Tbot, zqin0, zqsn0,
                              qpond, qocn, phi, q, w, hilyr, hslyr,
                              dxp, kcstar, Iswabs, Sswabs,
                              fsurfn, dfsurfn_dTsf, dt, parameters):
    """Mushy heat matrix for snow-covered ice with a cold surface."""
    nilyr = zqin0.shape[-1]
    nslyr = zqsn0.shape[-1]
    An = jnp.zeros((nilyr + nslyr + 1,), dtype=zqin0.dtype)
    Ap = jnp.zeros_like(An)
    As = jnp.zeros_like(An)
    b = jnp.zeros_like(An)
    surface_conduct = kcstar[0] / dxp[0]
    Ap = Ap.at[0].set(dfsurfn_dTsf - surface_conduct)
    As = As.at[0].set(surface_conduct)
    b = b.at[0].set(dfsurfn_dTsf * Tsf - fsurfn)
    rhos = parameters["rhos"]
    rhoi = parameters["rhoi"]
    cp_ice = parameters["cp_ice"]
    Lfresh = parameters["Lfresh"]
    cp_ocn = parameters["cp_ocn"]
    rhow = parameters["rhow"]
    for k in range(nslyr):
        l = k + 1
        upper = kcstar[k] / dxp[k]
        lower = kcstar[k + 1] / dxp[k + 1]
        Ap = Ap.at[l].set(((rhos * cp_ice) / dt) * hslyr + lower + upper)
        As = As.at[l].set(-lower)
        An = An.at[l].set(-upper)
        b = b.at[l].set(((rhos * Lfresh + zqsn0[k]) / dt) * hslyr
                         + Sswabs[k])
    for k in range(nilyr):
        l = nslyr + k + 1
        m = nslyr + k
        upper = kcstar[m] / dxp[m]
        lower = kcstar[m + 1] / dxp[m + 1]
        storage = ((phi[k] * (cp_ocn * rhow - cp_ice * rhoi)
                    + rhoi * cp_ice) / dt) * hilyr
        Ap = Ap.at[l].set(storage + lower + upper
                          + q[k + 1] * cp_ocn * rhow + w * cp_ocn * rhow)
        As = As.at[l].set(-lower - q[k + 1] * cp_ocn * rhow)
        An = An.at[l].set(-upper - w * cp_ocn * rhow)
        b = b.at[l].set(
            (((c1 - phi[k]) * rhoi * Lfresh + zqin0[k]) / dt) * hilyr
            + Iswabs[k])
    An = An.at[nslyr + 1].set(-kcstar[nslyr] / dxp[nslyr])
    b = b.at[nslyr + 1].add(w * qpond)
    As = As.at[-1].set(c0)
    b = b.at[-1].set(b[-1] + (kcstar[nilyr + nslyr] * Tbot) /
                     dxp[nilyr + nslyr] + q[nilyr] * qocn)
    return An, Ap, As, b


def update_temperatures(lsnow, lcold, T, Tsf, zTin, zTsn):
    """Unpack the tridiagonal solution into surface, snow and ice layers."""
    nilyr = zTin.shape[-1]
    nslyr = zTsn.shape[-1]
    Tsf = jnp.where(lcold, T[0], Tsf)
    snow_temperatures = jnp.where(lcold, T[1:nslyr + 1], T[:nslyr])
    zTsn = jnp.where(lsnow, snow_temperatures, zTsn)
    ice_with_snow = jnp.where(
        lcold, T[nslyr + 1:nslyr + nilyr + 1],
        T[nslyr:nslyr + nilyr])
    ice_without_snow = jnp.where(lcold, T[1:nilyr + 1], T[:nilyr])
    zTin = jnp.where(lsnow, ice_with_snow, ice_without_snow)
    return Tsf, zTin, zTsn


def solve_heat_conduction(lsnow, lcold, Tsf, Tbot, zqin0, zqsn0,
                          phi, dt, qpond, qocn, q, w, hilyr, hslyr,
                          dxp, kcstar, Iswabs, Sswabs, fsurfn,
                          dfsurfn_dTsf, zTin, zTsn, parameters):
    """Solve the cold/melting, snow/no-snow mushy heat matrix."""
    nilyr = zqin0.shape[-1]
    nslyr = zqsn0.shape[-1]
    nmax = nilyr + nslyr + 1

    def no_snow_melt(_):
        diagonals = matrix_elements_nosnow_melt(
            Tsf, Tbot, zqin0, qpond, qocn, phi, q, w,
            hilyr, dxp, kcstar, Iswabs, dt, parameters)
        An, Ap, As, b = (jnp.pad(value, (0, nmax - nilyr))
                         for value in diagonals)
        return An, Ap.at[nilyr:].set(1.0), As, b

    def no_snow_cold(_):
        diagonals = matrix_elements_nosnow_cold(
            Tsf, Tbot, zqin0, qpond, qocn, phi, q, w,
            hilyr, dxp, kcstar, Iswabs, fsurfn,
            dfsurfn_dTsf, dt, parameters)
        An, Ap, As, b = (jnp.pad(value, (0, nmax - nilyr - 1))
                         for value in diagonals)
        return An, Ap.at[nilyr + 1:].set(1.0), As, b

    def snow_melt(_):
        diagonals = matrix_elements_snow_melt(
            Tsf, Tbot, zqin0, zqsn0, qpond, qocn, phi, q, w,
            hilyr, hslyr, dxp, kcstar, Iswabs, Sswabs, dt, parameters)
        An, Ap, As, b = (jnp.pad(value, (0, 1)) for value in diagonals)
        return An, Ap.at[-1].set(1.0), As, b

    def snow_cold(_):
        return matrix_elements_snow_cold(
            Tsf, Tbot, zqin0, zqsn0, qpond, qocn, phi, q, w,
            hilyr, hslyr, dxp, kcstar, Iswabs, Sswabs,
            fsurfn, dfsurfn_dTsf, dt, parameters)

    branch = jnp.where(lsnow, jnp.where(lcold, 3, 2),
                       jnp.where(lcold, 1, 0))
    An, Ap, As, b = lax.switch(
        branch, (no_snow_melt, no_snow_cold, snow_melt, snow_cold),
        operand=None)
    T = tdma_solve_sparse(An, Ap, As, b)
    return update_temperatures(lsnow, lcold, T, Tsf, zTin, zTsn)


# Keep the Picard scan in one cacheable kernel across categories and time steps.
@partial(jax.jit, static_argnames=("semi_implicit", "debug_diagnostics", "profile_picard",
                                 "picard_axis_name"))
def picard_solver(lsnow, lcold, Tsf, zqsn, zqin, zSin, zTin, zTsn,
                  phi, dt, hilyr, hslyr, km, ks, Iswabs, Sswabs,
                  Tbot, fswint, fswsfc, rhoa, flw, potT, Qa,
                  shcoef, lhcoef, qpond, qocn, Spond, sss, q,
                  dSdt, w, semi_implicit, parameters, coupled=None,
                  debug_diagnostics=False, profile_picard=False,
                  picard_axis_name=None, picard_active=True):
    """Iterate the mushy heat matrix, then update ice bulk salinity."""
    if debug_diagnostics:
        nilyr, nslyr = zqin.shape[-1], zqsn.shape[-1]
        diagnostics = jnp.full(
            (8 + 4 * nilyr + 3 * nslyr, len(PICARD_DIAGNOSTIC_FIELDS)),
            jnp.nan, dtype=Tsf.dtype)
        diagnostics = diagnostics.at[3].set(jnp.stack((
            lsnow.astype(Tsf.dtype), lcold.astype(Tsf.dtype), Tsf,
            dt, hilyr, hslyr, Tbot, fswint, fswsfc, rhoa, flw, potT,
            Qa, shcoef, lhcoef, qpond, qocn, Spond)))
        diagnostics = diagnostics.at[4, :3].set(jnp.stack((
            sss, w, jnp.asarray(semi_implicit, dtype=Tsf.dtype))))
        if semi_implicit and coupled is not None:
            diagnostics = diagnostics.at[4, 3:7].set(jnp.stack((
                coupled["dfsurfdTs_cpl"], coupled["dflatdTs_cpl"],
                coupled["fsurf_cpl"], coupled["flat_cpl"])))
        diagnostics = diagnostics.at[5:5 + nilyr, :7].set(jnp.stack((
            zqin, zSin, zTin, phi, km, Iswabs, dSdt), axis=-1))
        diagnostics = diagnostics.at[5 + nilyr:5 + nilyr + nslyr, :4].set(
            jnp.stack((zqsn, zTsn, ks, Sswabs), axis=-1))
        diagnostics = diagnostics.at[
            5 + nilyr + nslyr:6 + 2 * nilyr + nslyr, 0].set(q)
    else:
        diagnostics = None
    zTin, zTsn, Sbr, phi, dxp, kcstar, einit = prep_picard(
        lsnow, zqsn, zqin, zSin, hilyr, hslyr, km, ks,
        zTin, zTsn, parameters)
    if debug_diagnostics:
        diagnostics = diagnostics.at[
            6 + 2 * nilyr + nslyr:6 + 3 * nilyr + nslyr, :3].set(
                jnp.stack((zTin, phi, Sbr), axis=-1))
        diagnostics = diagnostics.at[
            6 + 3 * nilyr + nslyr:6 + 3 * nilyr + 2 * nslyr, 0].set(zTsn)
        diagnostics = diagnostics.at[
            6 + 3 * nilyr + 2 * nslyr:7 + 4 * nilyr + 3 * nslyr, :2].set(
                jnp.stack((dxp, kcstar), axis=-1))
        diagnostics = diagnostics.at[-1, 0].set(einit)
    zqin0 = zqin
    zqsn0 = zqsn
    Tsf_prev = Tsf
    zTin_prev = zTin
    zTsn_prev = zTsn
    qbr = jnp.zeros_like(zTin)
    zero = jnp.zeros_like(Tsf)
    if semi_implicit:
        if coupled is None:
            raise ValueError("semi_implicit_Tsfc requires coupled flux values")
        dfsurfn_dTsf = coupled["dfsurfdTs_cpl"]
        dflatn_dTsf = coupled["dflatdTs_cpl"]
        fsurfn = coupled["fsurf_cpl"] + fswsfc
        flatn = coupled["flat_cpl"]
        flwoutn = zero
        fsensn = zero
    else:
        dfsurfn_dTsf = zero
        dflatn_dTsf = zero
        fsurfn = zero
        flatn = zero
        flwoutn = zero
        fsensn = zero
    state = (Tsf, zqin, zqsn, zTin, zTsn, zSin, phi,
             Tsf_prev, zTin_prev, zTsn_prev, qbr,
             fsurfn, flatn, flwoutn, fsensn,
             zero, zero, zero, jnp.asarray(False),
             dfsurfn_dTsf, dflatn_dTsf, diagnostics,
             jnp.asarray(0, dtype=jnp.int32) if profile_picard else None)

    def iteration(nit, state):
        def advance(state):
            (Tsf, zqin, zqsn, zTin, zTsn, zSin, phi,
             Tsf_prev, zTin_prev, zTsn_prev, qbr,
             fsurfn, flatn, flwoutn, fsensn,
             fcondtop, fcondbot, fadvheat, _,
             dfsurfn_dTsf, dflatn_dTsf, diagnostics, iterations) = state
            if not semi_implicit:
                flwoutn, fsensn, flatn, fsurfn = surface_heat_flux(
                    Tsf, fswsfc, rhoa, flw, potT, Qa,
                    shcoef, lhcoef, parameters)
                dfsurfn_dTsf, _, _, dflatn_dTsf = (
                    dsurface_heat_flux_dTsf(
                        Tsf, rhoa, shcoef, lhcoef, parameters))
            if debug_diagnostics:
                entry = jnp.stack((
                    jnp.asarray(nit + 1, dtype=Tsf.dtype),
                    lsnow.astype(Tsf.dtype), lcold.astype(Tsf.dtype),
                    Tsf, zTsn[0], zTin[0], fsurfn, dfsurfn_dTsf,
                    zero, zero, zero, zero, shcoef, lhcoef,
                    zqin[0], zqsn[0], zSin[0], phi[0],
                ))
                diagnostics = diagnostics.at[0].set(
                    jnp.where(nit == 0, entry, diagnostics[0]))
            Tsf, zTin, zTsn = solve_heat_conduction(
                lsnow, lcold, Tsf, Tbot, zqin0, zqsn0, phi, dt,
                qpond, qocn, q, w, hilyr, hslyr, dxp, kcstar,
                Iswabs, Sswabs, fsurfn, dfsurfn_dTsf, zTin,
                zTsn, parameters)
            qbr = picard_updates_enthalpy(zTin, qbr, parameters)
            fadvheat = picard_drainage_fluxes(q, qbr, qocn)
            fadvheat = picard_flushing_fluxes(
                fadvheat, w, qbr, qpond)
            (zqin, zqsn, fcondtop, fcondbot, _, converged) = (
                check_picard_convergence(
                    lsnow, Tsf, Tsf_prev, zTin, zTin_prev,
                    zTsn, zTsn_prev, phi, Tbot, zqin, zqsn,
                    km, ks, hilyr, hslyr, fswint, einit,
                    dt, fadvheat, parameters))
            if debug_diagnostics:
                snapshot = jnp.stack((
                    jnp.asarray(nit + 1, dtype=Tsf.dtype),
                    lsnow.astype(Tsf.dtype), lcold.astype(Tsf.dtype),
                    Tsf, zTsn[0], zTin[0], fsurfn, dfsurfn_dTsf,
                    fcondtop, fcondbot, fadvheat, converged.astype(Tsf.dtype),
                    shcoef, lhcoef, zqin[0], zqsn[0], zSin[0], phi[0],
                ))
                diagnostics = diagnostics.at[1].set(
                    jnp.where(nit == 0, snapshot, diagnostics[1]))
                diagnostics = diagnostics.at[2].set(snapshot)
            if semi_implicit:
                fsurfn = fsurfn + (Tsf - Tsf_prev) * dfsurfn_dTsf
                flatn = flatn + (Tsf - Tsf_prev) * dflatn_dTsf
            return (Tsf, zqin, zqsn, zTin, zTsn, zSin, phi,
                    Tsf, zTin, zTsn, qbr,
                    fsurfn, flatn, flwoutn, fsensn,
                    fcondtop, fcondbot, fadvheat, converged,
                    dfsurfn_dTsf, dflatn_dTsf, diagnostics,
                    jnp.asarray(nit + 1, dtype=jnp.int32) if profile_picard else iterations)

        return lax.cond(state[18], lambda values: values, advance, state)

    if picard_axis_name is None:
        state = lax.fori_loop(0, 100, iteration, state)
    else:
        def remaining_blocks(values):
            def block(block_index, values):
                any_pending = lax.pmax(
                    (picard_active & ~values[18]).astype(jnp.int32), picard_axis_name) > 0

                def advance_block(values):
                    # Keep loop bounds static for reverse-mode AD; only nit is dynamic.
                    return lax.fori_loop(
                        0, 8,
                        lambda offset, carry: iteration(4 + 8 * block_index + offset, carry),
                        values, unroll=4)

                return lax.cond(any_pending, advance_block, lambda values: values, values)

            return lax.fori_loop(0, 12, block, values)

        def solve_active(values):
            values = lax.fori_loop(0, 4, iteration, values, unroll=4)
            any_pending = lax.pmax(
                (picard_active & ~values[18]).astype(jnp.int32), picard_axis_name) > 0
            return lax.cond(any_pending, remaining_blocks, lambda values: values, values)

        # XLA may read a conditional's scalar predicate back to the host. Check
        # per block, not per iteration; converged columns still freeze every iteration.
        any_active = lax.pmax(jnp.asarray(picard_active, dtype=jnp.int32), picard_axis_name) > 0
        state = lax.cond(any_active, solve_active, lambda values: values, state)
    (Tsf, zqin, zqsn, zTin, zTsn, zSin, phi,
     _, _, _, qbr, fsurfn, flatn, flwoutn, fsensn,
     fcondtop, fcondbot, fadvheat, converged, _, _, diagnostics, iterations) = state
    Sbr, qbr = picard_updates(zTin, Sbr, qbr, parameters)
    zSin, salinity_error = solve_salinity(
        zSin, Sbr, Spond, sss, q, dSdt, w, hilyr, dt)
    if not semi_implicit:
        flwoutn, fsensn, flatn, fsurfn = surface_heat_flux(
            Tsf, fswsfc, rhoa, flw, potT, Qa,
            shcoef, lhcoef, parameters)
    result = (Tsf, zqsn, zqin, zSin, zTin, zTsn, phi,
              fcondtop, fcondbot, fadvheat, flwoutn, fsensn,
              flatn, fsurfn, converged & ~salinity_error)
    if debug_diagnostics:
        result = (*result[:-1], diagnostics, result[-1])
    if profile_picard:
        # Two rows reserve separate counts for the first solve and its retry.
        stats = jnp.stack((jnp.stack((iterations, converged.astype(jnp.int32))),
                           jnp.zeros((2,), dtype=jnp.int32)))
        result = (*result[:-1], stats, result[-1])
    return result


def two_stage_solver_snow(Tsf, zqsn, zqin, zSin, zTin, zTsn,
                          phi, dt, hilyr, hslyr, km, ks, Iswabs,
                          Sswabs, Tbot, fswint, fswsfc, rhoa, flw,
                          potT, Qa, shcoef, lhcoef, qpond, qocn,
                          Spond, sss, q, dSdt, w, options,
                          parameters, coupled=None, picard_active=True):
    """Try the previous snow surface condition, then its alternative."""
    Tsf0, zqsn0, zqin0, zSin0 = Tsf, zqsn, zqin, zSin
    melt_temperature = jnp.zeros_like(Tsf)
    tolerance = 5.0e-4
    ferrmax = 1.0e-3
    semi_implicit = bool(options["semi_implicit_Tsfc"])
    debug_diagnostics = bool(options.get("debug_model_ponds_inputs", False))
    profile_picard = bool(options.get("profile_picard", False))
    picard_axis_name = options.get("picard_axis_name")
    cold_start = Tsf0 < melt_temperature

    def cold_first(_):
        first = picard_solver(
            jnp.asarray(True), jnp.asarray(True), Tsf0,
            zqsn0, zqin0, zSin0, zTin, zTsn, phi,
            dt, hilyr, hslyr, km, ks, Iswabs, Sswabs,
            Tbot, fswint, fswsfc, rhoa, flw, potT, Qa,
            shcoef, lhcoef, qpond, qocn, Spond, sss, q,
            dSdt, w, semi_implicit, parameters, coupled,
            debug_diagnostics=debug_diagnostics, profile_picard=profile_picard,
            picard_axis_name=picard_axis_name, picard_active=picard_active & cold_start)
        keep_first = first[0] < melt_temperature + tolerance

        def try_melt(_):
            second = picard_solver(
                jnp.asarray(True), jnp.asarray(False),
                melt_temperature, zqsn0, zqin0, zSin0,
                zTin, zTsn, phi, dt, hilyr, hslyr, km,
                ks, Iswabs, Sswabs, Tbot, fswint, fswsfc,
                rhoa, flw, potT, Qa, shcoef, lhcoef, qpond,
                qocn, Spond, sss, q, dSdt, w, semi_implicit,
                parameters, coupled,
                debug_diagnostics=debug_diagnostics, profile_picard=profile_picard,
                picard_axis_name=picard_axis_name,
                picard_active=picard_active & cold_start & ~keep_first)
            if profile_picard:
                second = (*second[:-2], jnp.stack((first[-2][0], second[-2][0])), second[-1])
            consistent = first[-1] & (second[7] - second[13] < ferrmax)
            return second, consistent

        return lax.cond(keep_first,
                        lambda _: (first, jnp.asarray(True)),
                        try_melt, operand=None)

    def melt_first(_):
        first = picard_solver(
            jnp.asarray(True), jnp.asarray(False), melt_temperature,
            zqsn0, zqin0, zSin0, zTin, zTsn, phi,
            dt, hilyr, hslyr, km, ks, Iswabs, Sswabs,
            Tbot, fswint, fswsfc, rhoa, flw, potT, Qa,
            shcoef, lhcoef, qpond, qocn, Spond, sss, q,
            dSdt, w, semi_implicit, parameters, coupled,
            debug_diagnostics=debug_diagnostics, profile_picard=profile_picard,
            picard_axis_name=picard_axis_name, picard_active=picard_active & ~cold_start)
        keep_first = first[7] - first[13] < ferrmax

        def try_cold(_):
            second = picard_solver(
                jnp.asarray(True), jnp.asarray(True), Tsf0,
                zqsn0, zqin0, zSin0, zTin, zTsn, phi,
                dt, hilyr, hslyr, km, ks, Iswabs,
                Sswabs, Tbot, fswint, fswsfc, rhoa, flw,
                potT, Qa, shcoef, lhcoef, qpond, qocn,
                Spond, sss, q, dSdt, w, semi_implicit, parameters,
                coupled, debug_diagnostics=debug_diagnostics, profile_picard=profile_picard,
                picard_axis_name=picard_axis_name,
                picard_active=picard_active & ~cold_start & ~keep_first)
            if profile_picard:
                second = (*second[:-2], jnp.stack((first[-2][0], second[-2][0])), second[-1])
            return second, first[-1] & (second[0] < melt_temperature + tolerance)

        return lax.cond(keep_first,
                        lambda _: (first, jnp.asarray(True)),
                        try_cold, operand=None)

    result, consistent = lax.cond(cold_start,
                                  cold_first, melt_first, operand=None)
    return (*result[:-1], result[-1] & consistent)


def two_stage_solver_nosnow(Tsf, zqsn, zqin, zSin, zTin, zTsn,
                            phi, dt, hilyr, hslyr, km, ks, Iswabs,
                            Sswabs, Tbot, fswint, fswsfc, rhoa, flw,
                            potT, Qa, shcoef, lhcoef, qpond, qocn,
                            Spond, sss, q, dSdt, w, options,
                            parameters, coupled=None, picard_active=True):
    """Try the previous bare-ice surface condition, then its alternative."""
    Tsf0, zqsn0, zqin0, zSin0 = Tsf, zqsn, zqin, zSin
    melt_temperature = liquidus_temperature_mush_jax(zSin0[0])
    tolerance = 5.0e-4
    ferrmax = 1.0e-3
    semi_implicit = bool(options["semi_implicit_Tsfc"])
    debug_diagnostics = bool(options.get("debug_model_ponds_inputs", False))
    profile_picard = bool(options.get("profile_picard", False))
    picard_axis_name = options.get("picard_axis_name")
    cold_start = Tsf0 < melt_temperature

    def cold_first(_):
        first = picard_solver(
            jnp.asarray(False), jnp.asarray(True), Tsf0,
            zqsn0, zqin0, zSin0, zTin, zTsn, phi,
            dt, hilyr, hslyr, km, ks, Iswabs, Sswabs,
            Tbot, fswint, fswsfc, rhoa, flw, potT, Qa,
            shcoef, lhcoef, qpond, qocn, Spond, sss, q,
            dSdt, w, semi_implicit, parameters, coupled,
            debug_diagnostics=debug_diagnostics, profile_picard=profile_picard,
            picard_axis_name=picard_axis_name, picard_active=picard_active & cold_start)
        keep_first = first[0] < melt_temperature + tolerance

        def try_melt(_):
            second = picard_solver(
                jnp.asarray(False), jnp.asarray(False),
                melt_temperature, zqsn0, zqin0, zSin0,
                zTin, zTsn, phi, dt, hilyr, hslyr, km,
                ks, Iswabs, Sswabs, Tbot, fswint, fswsfc,
                rhoa, flw, potT, Qa, shcoef, lhcoef, qpond,
                qocn, Spond, sss, q, dSdt, w, semi_implicit,
                parameters, coupled,
                debug_diagnostics=debug_diagnostics, profile_picard=profile_picard,
                picard_axis_name=picard_axis_name,
                picard_active=picard_active & cold_start & ~keep_first)
            if profile_picard:
                second = (*second[:-2], jnp.stack((first[-2][0], second[-2][0])), second[-1])
            return second, first[-1] & (second[7] - second[13] < ferrmax)

        return lax.cond(keep_first,
                        lambda _: (first, jnp.asarray(True)),
                        try_melt, operand=None)

    def melt_first(_):
        if semi_implicit:
            melt_coupled = dict(coupled)
            delta = melt_temperature - Tsf0
            melt_coupled["fsurf_cpl"] = (
                coupled["fsurf_cpl"] + coupled["dfsurfdTs_cpl"] * delta)
            melt_coupled["flat_cpl"] = (
                coupled["flat_cpl"] + coupled["dflatdTs_cpl"] * delta)
        else:
            melt_coupled = coupled
        first = picard_solver(
            jnp.asarray(False), jnp.asarray(False), melt_temperature,
            zqsn0, zqin0, zSin0, zTin, zTsn, phi,
            dt, hilyr, hslyr, km, ks, Iswabs, Sswabs,
            Tbot, fswint, fswsfc, rhoa, flw, potT, Qa,
            shcoef, lhcoef, qpond, qocn, Spond, sss, q,
            dSdt, w, semi_implicit, parameters, melt_coupled,
            debug_diagnostics=debug_diagnostics, profile_picard=profile_picard,
            picard_axis_name=picard_axis_name, picard_active=picard_active & ~cold_start)
        keep_first = first[7] - first[13] < ferrmax

        def try_cold(_):
            second = picard_solver(
                jnp.asarray(False), jnp.asarray(True), Tsf0,
                zqsn0, zqin0, zSin0, zTin, zTsn, phi,
                dt, hilyr, hslyr, km, ks, Iswabs,
                Sswabs, Tbot, fswint, fswsfc, rhoa, flw,
                potT, Qa, shcoef, lhcoef, qpond, qocn,
                Spond, sss, q, dSdt, w, semi_implicit, parameters,
                coupled, debug_diagnostics=debug_diagnostics, profile_picard=profile_picard,
                picard_axis_name=picard_axis_name,
                picard_active=picard_active & ~cold_start & ~keep_first)
            if profile_picard:
                second = (*second[:-2], jnp.stack((first[-2][0], second[-2][0])), second[-1])
            return second, first[-1] & (second[0] < melt_temperature + tolerance)

        return lax.cond(keep_first,
                        lambda _: (first, jnp.asarray(True)),
                        try_cold, operand=None)

    result, consistent = lax.cond(cold_start,
                                  cold_first, melt_first, operand=None)
    return (*result[:-1], result[-1] & consistent)


def temperature_changes_salinity(dt, rhoa, flw, potT, Qa,
                                 shcoef, lhcoef, fswsfc, fswint,
                                 Sswabs, Iswabs, hilyr, hslyr,
                                 apond, hpond, zqin, zTin, zqsn,
                                 zTsn, zSin, Tsf, Tbot, sss,
                                 smice, smliq, dpnd_flush,
                                 dpnd_expon, options, parameters,
                                 coupled=None, picard_active=True):
    """Advance ice enthalpy and salinity for one mushy sea-ice column."""
    nilyr = zqin.shape[-1]
    nslyr = zqsn.shape[-1]
    lsnow = hslyr > parameters["hs_min"] / nslyr
    hin = hilyr * nilyr
    hsn = jnp.where(lsnow, hslyr * nslyr, 0.0)
    qpond = enthalpy_brine_jax(jnp.zeros_like(Tsf), parameters)
    qocn = enthalpy_brine_jax(Tbot, parameters)
    Spond = jnp.zeros_like(Tsf)
    phi = icepack_mushy_liquid_fraction_jax(
        icepack_mushy_temperature_mush_jax(zqin, zSin, parameters),
        zSin, parameters)
    w = flushing_velocity(zTin, phi, hin, hsn, hilyr,
                          hpond, apond, dt, options, parameters)
    q, dSdt, _, _ = explicit_flow_velocities(
        zSin, zTin, Tsf, Tbot, dt, sss, qocn,
        hilyr, hin, parameters)
    km = conductivity_mush_array_jax(zqin, zSin, parameters)
    ks = conductivity_snow_array_jax(zqsn.shape, parameters)

    # Move excessive layer-absorbed sunlight to the surface, as in CICE.
    dswabs = jnp.zeros_like(Tsf)
    if bool(options["sw_redist"]):
        dt_rhoi_hlyr = dt / (parameters["rhoi"] * hilyr)
        for k in range(nilyr):
            Tmlt = liquidus_temperature_mush_jax(zSin[k])
            safe_temperature = jnp.where(
                zTin[k] <= Tmlt - parameters["sw_dtemp"],
                zTin[k], -1.0)
            ci = (parameters["cp_ice"]
                  - parameters["Lfresh"] * Tmlt /
                  safe_temperature**2)
            retained = jnp.where(
                zTin[k] <= Tmlt - parameters["sw_dtemp"],
                jnp.minimum(Iswabs[k], parameters["sw_frac"]
                            * (Tmlt - zTin[k]) * ci / dt_rhoi_hlyr),
                0.0)
            retained = jnp.where(retained < parameters["puny"],
                                 0.0, retained)
            dswabs = dswabs + jnp.minimum(
                Iswabs[k] - retained, fswint)
            Iswabs = Iswabs.at[k].set(retained)
    dswabs = dswabs + jnp.where(lsnow, 0.0, jnp.sum(Sswabs))
    fswsfc = fswsfc + dswabs
    fswint = fswint - dswabs

    def snow_column(_):
        return two_stage_solver_snow(
            Tsf, zqsn, zqin, zSin, zTin, zTsn,
            phi, dt, hilyr, hslyr, km, ks, Iswabs,
            Sswabs, Tbot, fswint, fswsfc, rhoa, flw,
            potT, Qa, shcoef, lhcoef, qpond, qocn,
            Spond, sss, q, dSdt, w, options, parameters,
            coupled, picard_active=picard_active & lsnow)

    def bare_column(_):
        return two_stage_solver_nosnow(
            Tsf, zqsn, zqin, zSin, zTin, zTsn,
            phi, dt, hilyr, hslyr, km, ks, Iswabs,
            Sswabs, Tbot, fswint, fswsfc, rhoa, flw,
            potT, Qa, shcoef, lhcoef, qpond, qocn,
            Spond, sss, q, dSdt, w, options, parameters,
            coupled, picard_active=picard_active & ~lsnow)

    solver_result = lax.cond(lsnow, snow_column, bare_column, operand=None)
    if bool(options.get("profile_picard", False)):
        picard_iterations = solver_result[-2]
        solver_result = (*solver_result[:-2], solver_result[-1])
    if bool(options.get("debug_model_ponds_inputs", False)):
        picard_diagnostics = solver_result[-2]
        solver_result = (*solver_result[:-2], solver_result[-1])
    (Tsf, zqsn, zqin, zSin, zTin, zTsn, phi,
     fcondtop, fcondbot, fadvheat, flwoutn, fsensn,
     flatn, fsurfn, converged) = solver_result
    if bool(options.get("debug_model_ponds_inputs", False)):
        # Preserve the signed imbalance; max(..., 0) in thickness_changes
        # turns only its positive part into surface melt energy (J/m2).
        surface_diagnostics = jnp.stack((
            Tsf, zTsn[0], zTin[0], hslyr, hilyr, ks[0], km[0],
            lsnow.astype(Tsf.dtype), fswsfc, parameters["emissivity"] * flw,
            flwoutn, fsensn, flatn, fsurfn, fcondtop,
            fsurfn - fcondtop, (fsurfn - fcondtop) * dt,
            potT, Qa, rhoa, shcoef, lhcoef, converged.astype(Tsf.dtype),
        ))
    zTsn = jnp.where(lsnow, temperature_snow_jax(zqsn, parameters),
                     zTsn)
    zTin = temperature_mush_liquid_fraction_jax(
        zqin, phi, parameters)
    Sbr = liquidus_brine_salinity_mush_jax(zTin)
    hpond, apond, dpnd_flush, dpnd_expon = flush_pond(
        w, hpond, apond, dt, dpnd_flush, dpnd_expon,
        zTin, phi, hilyr, hin, hsn, options, parameters)
    (hslyr, hilyr, zqsn, zqin, phi, zSin,
     smice, smliq, snoice, fadvheat) = flood_ice(
         hsn, hin, hslyr, hilyr, zqsn, zqin, phi,
         dt, zSin, Sbr, sss, qocn, smice, smliq,
         fadvheat, bool(options["snwgrain"]), parameters)
    result = (fswsfc, fswint, Sswabs, Iswabs, hilyr, hslyr,
            apond, hpond, zqin, zTin, zqsn, zTsn, zSin,
            Tsf, fsensn, flatn, flwoutn, fsurfn,
            fcondtop, fcondbot, fadvheat, snoice,
            smice, smliq, dpnd_flush, dpnd_expon,
            ~converged)
    if bool(options.get("debug_model_ponds_inputs", False)):
        result = (*result, surface_diagnostics, picard_diagnostics)
    if bool(options.get("profile_picard", False)):
        result = (*result, picard_iterations)
    return result
