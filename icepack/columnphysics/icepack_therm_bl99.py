"""BL99 vertical-temperature calculations from icepack_therm_bl99.F90.

The last array axis is the vertical layer or matrix row. Leading axes may
batch independent sea-ice columns for one JAX device.
"""

import jax.numpy as jnp
from jax import lax

from icepack.columnphysics.icepack_therm_shared import (
    dsurface_heat_flux_dTsf,
    surface_heat_flux,
)


def surface_fluxes(Tsf, fswsfc, rhoa, flw, potT, Qa, shcoef, lhcoef,
                   parameters):
    """Compute BL99 surface fluxes and their temperature derivatives."""
    flwoutn, fsensn, flatn, fsurfn = surface_heat_flux(
        Tsf, fswsfc, rhoa, flw, potT, Qa, shcoef, lhcoef, parameters,
    )
    dfsurf_dT, dflwout_dT, dfsens_dT, dflat_dT = dsurface_heat_flux_dTsf(
        Tsf, rhoa, shcoef, lhcoef, parameters,
    )
    return (flwoutn, fsensn, flatn, fsurfn,
            dflwout_dT, dfsens_dT, dflat_dT, dfsurf_dT)


def conductivity(l_snow, hilyr, hslyr, zTin, zSin, nslyr, conduct, parameters):
    """Compute snow and ice interface conductances in Fortran layer order."""
    puny = parameters["puny"]
    kilyr = jnp.zeros_like(zTin)
    for k in range(zTin.shape[-1]):
        denominator = jnp.minimum(-puny, zTin[..., k])
        if conduct == "MU71":
            value = parameters["kice"] + 0.13 * zSin[..., k] / denominator
        else:
            value = ((2.11 - 0.011 * zTin[..., k]
                      + 0.09 * zSin[..., k] / denominator)
                     * parameters["rhoi"] / 917.0)
        kilyr = kilyr.at[..., k].set(jnp.maximum(value, 0.10))

    nilyr = zTin.shape[-1]
    kh = jnp.zeros(zTin.shape[:-1] + (nslyr + nilyr + 1,), dtype=zTin.dtype)
    ksno = parameters["ksno"]
    safe_hslyr = jnp.where(l_snow, hslyr, 1.0)
    kh = kh.at[..., 0].set(jnp.where(l_snow, 2.0 * ksno / safe_hslyr, 0.0))
    top_ice = jnp.where(
        l_snow,
        2.0 * ksno * kilyr[..., 0] /
        (ksno * hilyr + kilyr[..., 0] * safe_hslyr),
        2.0 * kilyr[..., 0] / hilyr,
    )
    kh = kh.at[..., nslyr].set(top_ice)
    kh = kh.at[..., nslyr + nilyr].set(2.0 * kilyr[..., -1] / hilyr)

    for k in range(1, nslyr):
        kh = kh.at[..., k].set(jnp.where(l_snow, ksno / safe_hslyr, 0.0))
    for k in range(1, nilyr):
        value = (2.0 * kilyr[..., k - 1] * kilyr[..., k]
                 / ((kilyr[..., k - 1] + kilyr[..., k]) * hilyr))
        kh = kh.at[..., nslyr + k].set(value)
    return kh


def get_matrix_elements_know_Tsfc(l_snow, Tbot, Tin_init, Tsn_init,
                                  kh, Sswabs, Iswabs, etai, etas,
                                  fcondtopn):
    """Assemble the BL99 system when surface temperature is prescribed."""
    nslyr = Tsn_init.shape[-1]
    nilyr = Tin_init.shape[-1]
    sbdiag = jnp.zeros_like(kh)
    diag = jnp.ones_like(kh)
    spdiag = jnp.zeros_like(kh)
    rhs = jnp.zeros_like(kh)

    if nslyr > 0:
        row = 1
        spdiag = spdiag.at[..., row].set(jnp.where(
            l_snow, -etas[..., 0] * kh[..., row], 0.0))
        diag = diag.at[..., row].set(jnp.where(
            l_snow, 1.0 + etas[..., 0] * kh[..., row], 1.0))
        rhs = rhs.at[..., row].set(jnp.where(
            l_snow,
            Tsn_init[..., 0] + etas[..., 0] * (Sswabs[..., 0] + fcondtopn),
            0.0))
        for k in range(1, nslyr):
            row = k + 1
            sbdiag = sbdiag.at[..., row].set(jnp.where(
                l_snow, -etas[..., k] * kh[..., row - 1], 0.0))
            spdiag = spdiag.at[..., row].set(jnp.where(
                l_snow, -etas[..., k] * kh[..., row], 0.0))
            diag = diag.at[..., row].set(jnp.where(
                l_snow, 1.0 + etas[..., k] *
                (kh[..., row - 1] + kh[..., row]), 1.0))
            rhs = rhs.at[..., row].set(jnp.where(
                l_snow, Tsn_init[..., k] + etas[..., k] * Sswabs[..., k],
                0.0))

    row = nslyr + 1
    if nilyr > 1:
        sbdiag = sbdiag.at[..., row].set(jnp.where(
            l_snow, -etai[..., 0] * kh[..., row - 1], 0.0))
        spdiag = spdiag.at[..., row].set(-etai[..., 0] * kh[..., row])
        diag = diag.at[..., row].set(
            1.0 + etai[..., 0] *
            (kh[..., row] + jnp.where(l_snow, kh[..., row - 1], 0.0)))
        rhs = rhs.at[..., row].set(
            Tin_init[..., 0] + etai[..., 0] * Iswabs[..., 0]
            + jnp.where(l_snow, 0.0, etai[..., 0] * fcondtopn))

        for ki in range(1, nilyr - 1):
            row = nslyr + ki + 1
            sbdiag = sbdiag.at[..., row].set(-etai[..., ki] * kh[..., row - 1])
            spdiag = spdiag.at[..., row].set(-etai[..., ki] * kh[..., row])
            diag = diag.at[..., row].set(
                1.0 + etai[..., ki] * (kh[..., row - 1] + kh[..., row]))
            rhs = rhs.at[..., row].set(
                Tin_init[..., ki] + etai[..., ki] * Iswabs[..., ki])

        row = nslyr + nilyr
        sbdiag = sbdiag.at[..., row].set(-etai[..., -1] * kh[..., row - 1])
        diag = diag.at[..., row].set(
            1.0 + etai[..., -1] * (kh[..., row - 1] + kh[..., row]))
        rhs = rhs.at[..., row].set(
            Tin_init[..., -1] + etai[..., -1] *
            (Iswabs[..., -1] + kh[..., row] * Tbot))
    else:
        sbdiag = sbdiag.at[..., row].set(jnp.where(
            l_snow, -etai[..., 0] * kh[..., row - 1], 0.0))
        diag = diag.at[..., row].set(
            1.0 + etai[..., 0] *
            (kh[..., row] + jnp.where(l_snow, kh[..., row - 1], 0.0)))
        rhs = rhs.at[..., row].set(
            Tin_init[..., 0] + etai[..., 0] *
            (Iswabs[..., 0] + kh[..., row] * Tbot
             + jnp.where(l_snow, 0.0, fcondtopn)))

    return sbdiag, diag, spdiag, rhs


def get_matrix_elements_calc_Tsfc(l_snow, l_cold, Tsf, Tbot, fsurfn,
                                  dfsurf_dT, Tin_init, Tsn_init, kh,
                                  Sswabs, Iswabs, etai, etas):
    """Assemble the BL99 system when CICE solves surface temperature."""
    nslyr = Tsn_init.shape[-1]
    nilyr = Tin_init.shape[-1]
    sbdiag = jnp.zeros_like(kh)
    diag = jnp.ones_like(kh)
    spdiag = jnp.zeros_like(kh)
    rhs = jnp.zeros_like(kh)

    # Cold-surface equation is row 0 with snow, row nslyr without snow.
    snow_surface = l_snow & l_cold
    diag = diag.at[..., 0].set(jnp.where(
        snow_surface, dfsurf_dT - kh[..., 0], 1.0))
    spdiag = spdiag.at[..., 0].set(jnp.where(
        snow_surface, kh[..., 0], 0.0))
    rhs = rhs.at[..., 0].set(jnp.where(
        snow_surface, dfsurf_dT * Tsf - fsurfn, 0.0))

    if nslyr > 0:
        row = 1
        sbdiag = sbdiag.at[..., row].set(jnp.where(
            l_snow & l_cold, -etas[..., 0] * kh[..., 0], 0.0))
        spdiag = spdiag.at[..., row].set(jnp.where(
            l_snow, -etas[..., 0] * kh[..., row], 0.0))
        diag = diag.at[..., row].set(jnp.where(
            l_snow, 1.0 + etas[..., 0] * (kh[..., 0] + kh[..., row]), 1.0))
        rhs = rhs.at[..., row].set(jnp.where(
            l_snow, Tsn_init[..., 0]
            + etas[..., 0] * (Sswabs[..., 0]
                             + jnp.where(l_cold, 0.0, kh[..., 0] * Tsf)),
            0.0))

        for k in range(1, nslyr):
            row = k + 1
            sbdiag = sbdiag.at[..., row].set(jnp.where(
                l_snow, -etas[..., k] * kh[..., row - 1], 0.0))
            spdiag = spdiag.at[..., row].set(jnp.where(
                l_snow, -etas[..., k] * kh[..., row], 0.0))
            diag = diag.at[..., row].set(jnp.where(
                l_snow, 1.0 + etas[..., k] *
                (kh[..., row - 1] + kh[..., row]), 1.0))
            rhs = rhs.at[..., row].set(jnp.where(
                l_snow, Tsn_init[..., k] + etas[..., k] * Sswabs[..., k],
                0.0))

        no_snow_surface = jnp.logical_not(l_snow) & l_cold
        row = nslyr
        sbdiag = sbdiag.at[..., row].set(jnp.where(
            no_snow_surface, 0.0, sbdiag[..., row]))
        diag = diag.at[..., row].set(jnp.where(
            no_snow_surface, dfsurf_dT - kh[..., row], diag[..., row]))
        spdiag = spdiag.at[..., row].set(jnp.where(
            no_snow_surface, kh[..., row], spdiag[..., row]))
        rhs = rhs.at[..., row].set(jnp.where(
            no_snow_surface, dfsurf_dT * Tsf - fsurfn, rhs[..., row]))
    else:
        no_snow_surface = jnp.logical_not(l_snow) & l_cold
        diag = diag.at[..., 0].set(jnp.where(
            no_snow_surface, dfsurf_dT - kh[..., 0], diag[..., 0]))
        spdiag = spdiag.at[..., 0].set(jnp.where(
            no_snow_surface, kh[..., 0], spdiag[..., 0]))
        rhs = rhs.at[..., 0].set(jnp.where(
            no_snow_surface, dfsurf_dT * Tsf - fsurfn, rhs[..., 0]))

    top_connected = l_snow | l_cold
    row = nslyr + 1
    if nilyr > 1:
        sbdiag = sbdiag.at[..., row].set(jnp.where(
            top_connected, -etai[..., 0] * kh[..., row - 1], 0.0))
        spdiag = spdiag.at[..., row].set(-etai[..., 0] * kh[..., row])
        diag = diag.at[..., row].set(
            1.0 + etai[..., 0] * (kh[..., row - 1] + kh[..., row]))
        rhs = rhs.at[..., row].set(
            Tin_init[..., 0] + etai[..., 0] * Iswabs[..., 0]
            + jnp.where(top_connected, 0.0,
                        etai[..., 0] * kh[..., row - 1] * Tsf))

        for ki in range(1, nilyr - 1):
            row = nslyr + ki + 1
            sbdiag = sbdiag.at[..., row].set(-etai[..., ki] * kh[..., row - 1])
            spdiag = spdiag.at[..., row].set(-etai[..., ki] * kh[..., row])
            diag = diag.at[..., row].set(
                1.0 + etai[..., ki] * (kh[..., row - 1] + kh[..., row]))
            rhs = rhs.at[..., row].set(
                Tin_init[..., ki] + etai[..., ki] * Iswabs[..., ki])

        row = nslyr + nilyr
        sbdiag = sbdiag.at[..., row].set(-etai[..., -1] * kh[..., row - 1])
        diag = diag.at[..., row].set(
            1.0 + etai[..., -1] * (kh[..., row - 1] + kh[..., row]))
        rhs = rhs.at[..., row].set(
            Tin_init[..., -1] + etai[..., -1] *
            (Iswabs[..., -1] + kh[..., row] * Tbot))
    else:
        sbdiag = sbdiag.at[..., row].set(jnp.where(
            top_connected, -etai[..., 0] * kh[..., row - 1], 0.0))
        diag = diag.at[..., row].set(
            1.0 + etai[..., 0] * (kh[..., row - 1] + kh[..., row]))
        rhs = rhs.at[..., row].set(
            Tin_init[..., 0] + etai[..., 0] *
            (Iswabs[..., 0] + kh[..., row] * Tbot
             + jnp.where(top_connected, 0.0, kh[..., row - 1] * Tsf)))

    return sbdiag, diag, spdiag, rhs


def tridiag_solver(nmat, sbdiag, diag, spdiag, rhs):
    """Solve CICE's tridiagonal temperature system by forward/back sweep."""
    xout = jnp.zeros_like(rhs)
    wgamma = jnp.zeros_like(rhs)
    wbeta = diag[..., 0]
    xout = xout.at[..., 0].set(rhs[..., 0] / wbeta)

    for k in range(1, nmat):
        wgamma = wgamma.at[..., k].set(spdiag[..., k - 1] / wbeta)
        wbeta = diag[..., k] - sbdiag[..., k] * wgamma[..., k]
        value = (rhs[..., k] - sbdiag[..., k] * xout[..., k - 1]) / wbeta
        xout = xout.at[..., k].set(value)

    for k in range(nmat - 2, -1, -1):
        xout = xout.at[..., k].set(
            xout[..., k] - wgamma[..., k + 1] * xout[..., k + 1]
        )
    return xout


def temperature_changes(dt, rhoa, flw, potT, Qa, shcoef, lhcoef,
                        fswsfc, fswint, Sswabs, Iswabs, hilyr, hslyr,
                        zqin, zTin, zqsn, zTsn, zSin, Tsf, Tbot,
                        fsensn, flatn, flwoutn, fsurfn, fcondtopn,
                        einit, options, parameters, coupled=None):
    """Advance one BL99 ice column's snow, ice and surface temperatures.

    CICE exits the temperature iteration on convergence. A fixed 100-step
    ``lax.scan`` with a stopped branch after convergence keeps reverse-mode
    differentiation available while preserving the Fortran iteration order.
    The final Boolean reports a failed convergence check to the host.
    """
    Tsf = jnp.asarray(Tsf)
    nilyr = zTin.shape[-1]
    nslyr = zTsn.shape[-1]
    puny = parameters["puny"]
    rhoi = parameters["rhoi"]
    rhos = parameters["rhos"]
    cp_ice = parameters["cp_ice"]
    Lfresh = parameters["Lfresh"]
    l_brine = bool(options["l_brine"])
    calc_Tsfc = bool(options["calc_Tsfc"])
    semi_implicit = bool(options["semi_implicit_Tsfc"])
    if nslyr > 0:
        l_snow = hslyr > parameters["hs_min"] / nslyr
    else:
        l_snow = jnp.asarray(False)
    etas = jnp.where(l_snow,
                     dt / (rhos * cp_ice * jnp.where(l_snow, hslyr, 1.0)),
                     0.0) * jnp.ones_like(zTsn)
    Tin_init = zTin
    Tsn_init = zTsn
    Tmlts = -zSin * parameters["depressT"]
    kh = conductivity(l_snow, hilyr, hslyr, zTin, zSin, nslyr,
                      options["conduct"], parameters)
    dt_rhoi_hlyr = dt / (rhoi * hilyr)

    if bool(options["sw_redist"]):
        for k in range(nilyr):
            if l_brine:
                ci = cp_ice - Lfresh * Tmlts[k] / Tin_init[k]**2
                potential = (parameters["sw_frac"] *
                             (Tmlts[k] - Tin_init[k]) * ci / dt_rhoi_hlyr)
            else:
                potential = (-parameters["sw_frac"] * Tin_init[k]
                             * cp_ice / dt_rhoi_hlyr)
            available = jnp.where(
                Tin_init[k] <= Tmlts[k] - parameters["sw_dtemp"],
                jnp.minimum(Iswabs[k], potential), 0.0)
            kept = jnp.where(available < puny, 0.0, available)
            dswabs = jnp.minimum(Iswabs[k] - kept, fswint)
            fswsfc = fswsfc + dswabs
            fswint = fswint - dswabs
            Iswabs = Iswabs.at[k].set(kept)
        for k in range(nslyr):
            potential = jnp.where(
                Tsn_init[k] <= -parameters["sw_dtemp"],
                jnp.minimum(Sswabs[k], -parameters["sw_frac"] *
                            Tsn_init[k] / jnp.where(l_snow, etas[k], 1.0)),
                0.0)
            kept = jnp.where(potential < puny, 0.0, potential)
            dswabs = jnp.where(l_snow,
                               jnp.minimum(Sswabs[k] - kept, fswint), 0.0)
            fswsfc = fswsfc + dswabs
            fswint = fswint - dswabs
            Sswabs = Sswabs.at[k].set(jnp.where(l_snow, kept, Sswabs[k]))

    dfsurf_dT = jnp.zeros_like(Tsf)
    dflat_dT = jnp.zeros_like(Tsf)
    if semi_implicit:
        if coupled is None:
            raise ValueError("semi_implicit_Tsfc requires coupled flux values")
        dfsurf_dT = coupled["dfsurfdTs_cpl"]
        dflat_dT = coupled["dflatdTs_cpl"]
        fsurfn = coupled["fsurf_cpl"] + fswsfc
        flatn = coupled["flat_cpl"]

    carry = {
        "Tsf": Tsf, "zTin": zTin, "zTsn": zTsn,
        "zqin": zqin, "zqsn": zqsn, "kh": kh,
        "fsurfn": fsurfn, "fcondtopn": fcondtopn,
        "fsensn": fsensn, "flatn": flatn, "flwoutn": flwoutn,
        "fcondbot": jnp.zeros_like(Tsf),
        "dfsurf_dT": dfsurf_dT, "dflat_dT": dflat_dT,
        "dflwout_dT": jnp.zeros_like(Tsf),
        "dfsens_dT": jnp.zeros_like(Tsf),
        "dTsf_prev": jnp.zeros_like(Tsf),
        "dTi1_prev": jnp.zeros_like(Tsf),
        "ferr": jnp.zeros_like(Tsf),
        "converged": jnp.asarray(False),
    }

    def scan_body(current, iteration_index):
        def advance(values):
            Tsf_now = values["Tsf"]
            zTin_now = values["zTin"]
            zTsn_now = values["zTsn"]
            kh_now = values["kh"]
            fsurf_now = values["fsurfn"]
            flat_now = values["flatn"]
            flwout_now = values["flwoutn"]
            fsens_now = values["fsensn"]
            dfsurf_now = values["dfsurf_dT"]
            dflat_now = values["dflat_dT"]
            dflwout_now = values["dflwout_dT"]
            dfsens_now = values["dfsens_dT"]
            fcondtop_now = values["fcondtopn"]
            converged = jnp.asarray(True)
            avg_Tsi = jnp.asarray(0.0, dtype=Tsf.dtype)
            enew = jnp.asarray(0.0, dtype=Tsf.dtype)
            einex = jnp.asarray(0.0, dtype=Tsf.dtype)

            if l_brine:
                ci = cp_ice - Lfresh * Tmlts / (zTin_now * Tin_init)
            else:
                ci = jnp.full_like(zTin_now, cp_ice)
            etai = dt_rhoi_hlyr / ci

            if calc_Tsfc:
                if not semi_implicit:
                    (flwout_now, fsens_now, flat_now, fsurf_now,
                     dflwout_now, dfsens_now, dflat_now,
                     dfsurf_now) = surface_fluxes(
                        Tsf_now, fswsfc, rhoa, flw, potT, Qa, shcoef,
                        lhcoef, parameters)
                if nslyr > 0:
                    top_temperature = jnp.where(l_snow, zTsn_now[0], zTin_now[0])
                else:
                    top_temperature = zTin_now[0]
                top_interface = jnp.where(l_snow, kh_now[0], kh_now[nslyr])
                fcondtop_now = top_interface * (Tsf_now - top_temperature)
                Tsf_now = jnp.where(
                    (Tsf_now >= 0.0) & (fsurf_now < fcondtop_now),
                    -puny, Tsf_now)
                Tsf_start = Tsf_now
                l_cold = Tsf_now < 0.0
                sbdiag, diag, spdiag, rhs = get_matrix_elements_calc_Tsfc(
                    l_snow, l_cold, Tsf_now, Tbot, fsurf_now, dfsurf_now,
                    Tin_init, Tsn_init, kh_now, Sswabs, Iswabs, etai, etas)
            else:
                sbdiag, diag, spdiag, rhs = get_matrix_elements_know_Tsfc(
                    l_snow, Tbot, Tin_init, Tsn_init, kh_now, Sswabs,
                    Iswabs, etai, etas, fcondtop_now)

            Tmat = tridiag_solver(nslyr + nilyr + 1,
                                  sbdiag, diag, spdiag, rhs)
            if calc_Tsfc:
                surface_row = jnp.where(l_snow, Tmat[0], Tmat[nslyr])
                Tsf_new = jnp.where(l_cold, surface_row, 0.0)
                dTsf = Tsf_new - Tsf_start
                over_melt = Tsf_new > puny
                oscillating = ((iteration_index > 0) & (Tsf_start <= -puny)
                               & (jnp.abs(dTsf) > puny)
                               & (jnp.abs(values["dTsf_prev"]) > puny)
                               & (-dTsf / (values["dTsf_prev"] + puny**2) > 0.5))
                oscillating = oscillating & ~over_melt
                avg_Tsf = jnp.where(oscillating & l_brine, 1.0, 0.0)
                avg_Tsi = jnp.where((over_melt | oscillating) & l_brine, 1.0, 0.0)
                dTsf = jnp.where(over_melt, -Tsf_start,
                                 jnp.where(oscillating, 0.5 * dTsf, dTsf))
                Tsf_new = jnp.where(over_melt, 0.0, Tsf_new)
                Tsf_now = Tsf_new + avg_Tsf * 0.5 * (Tsf_start - Tsf_new)
                converged = converged & ~over_melt & ~oscillating
            else:
                dTsf = jnp.zeros_like(Tsf_now)

            snow_energy = jnp.zeros_like(enew)
            for k in range(nslyr):
                snow_temp = jnp.where(l_snow, Tmat[k + 1], 0.0)
                if l_brine:
                    snow_temp = jnp.minimum(snow_temp, 0.0)
                snow_temp = snow_temp + avg_Tsi * 0.5 * (zTsn_now[k] - snow_temp)
                zTsn_now = zTsn_now.at[k].set(snow_temp)
                snow_enthalpy = -rhos * (Lfresh - cp_ice * snow_temp)
                snow_energy = snow_energy + hslyr * snow_enthalpy
                values = dict(values)
                values["zqsn"] = values["zqsn"].at[k].set(snow_enthalpy)
            enew = enew + snow_energy

            dTi1_prev = values["dTi1_prev"]
            reduce_kh = jnp.zeros((nilyr,), dtype=bool)
            dqmat = jnp.zeros_like(zTin_now)
            for k in range(nilyr):
                ice_temp = Tmat[k + nslyr + 1]
                over_ice_melt = l_brine & (ice_temp > Tmlts[k] - puny)
                dTmat = jnp.where(over_ice_melt, ice_temp - Tmlts[k], 0.0)
                dq = jnp.where(over_ice_melt,
                               rhoi * dTmat *
                               (cp_ice - Lfresh * Tmlts[k] / ice_temp**2), 0.0)
                ice_temp = jnp.where(over_ice_melt, Tmlts[k], ice_temp)
                reduce_kh = reduce_kh.at[k].set(over_ice_melt)
                dqmat = dqmat.at[k].set(dq)
                if k == 0 and not calc_Tsfc:
                    dTi1 = ice_temp - zTin_now[k]
                    oscillating_ice = ((iteration_index > 0)
                                       & (jnp.abs(dTi1) > puny)
                                       & (jnp.abs(dTi1_prev) > puny)
                                       & (-dTi1 / (dTi1_prev + puny**2) > 0.5))
                    avg_Tsi = jnp.where(oscillating_ice & l_brine, 1.0, avg_Tsi)
                    dTi1_prev = jnp.where(oscillating_ice, 0.5 * dTi1, dTi1)
                    converged = converged & ~oscillating_ice
                ice_temp = ice_temp + avg_Tsi * 0.5 * (zTin_now[k] - ice_temp)
                zTin_now = zTin_now.at[k].set(ice_temp)
                if l_brine:
                    qice = (-rhoi * (cp_ice * (Tmlts[k] - ice_temp)
                                    + Lfresh * (1.0 - Tmlts[k] / ice_temp)
                                    - parameters["cp_ocn"] * Tmlts[k]))
                else:
                    qice = -rhoi * (-cp_ice * ice_temp + Lfresh)
                values = dict(values)
                values["zqin"] = values["zqin"].at[k].set(qice)
                enew = enew + hilyr * qice
                einex = einex + hilyr * dq

            if calc_Tsfc:
                converged = converged & (jnp.abs(dTsf) <= 5.0e-4)
                fsurf_now = fsurf_now + dTsf * dfsurf_now
                if semi_implicit:
                    flat_now = flat_now + dTsf * dflat_now
                if nslyr > 0:
                    top_temperature = jnp.where(l_snow, zTsn_now[0], zTin_now[0])
                else:
                    top_temperature = zTin_now[0]
                top_interface = jnp.where(l_snow, kh_now[0], kh_now[nslyr])
                fcondtop_now = top_interface * (Tsf_now - top_temperature)
                converged = converged & ~((Tsf_now >= 0.0)
                                          & (fsurf_now < fcondtop_now))

            fcondbot = kh_now[nslyr + nilyr] * (zTin_now[-1] - Tbot) + einex / dt
            ferr = jnp.abs((enew - einit) / dt
                           - (fcondtop_now - fcondbot + fswint))
            energy_error = ferr > 0.9e-3
            converged = converged & ~energy_error
            for k in range(nilyr):
                reduce = energy_error & reduce_kh[k] & (dqmat[k] > 0.0)
                denominator = jnp.abs(fcondtop_now - fcondbot)
                frac = jnp.maximum(0.5 * (1.0 - ferr /
                                             jnp.where(denominator > 0.0,
                                                       denominator, 1.0)), 0.1)
                lower_value = kh_now[k + nslyr + 1] * frac
                kh_now = kh_now.at[k + nslyr + 1].set(jnp.where(
                    reduce, lower_value, kh_now[k + nslyr + 1]))
                kh_now = kh_now.at[k + nslyr].set(jnp.where(
                    reduce, lower_value, kh_now[k + nslyr]))

            result = dict(values)
            result.update({
                "Tsf": Tsf_now, "zTin": zTin_now, "zTsn": zTsn_now,
                "kh": kh_now, "fsurfn": fsurf_now,
                "fcondtopn": fcondtop_now, "fcondbot": fcondbot,
                "fsensn": fsens_now, "flatn": flat_now,
                "flwoutn": flwout_now, "dfsurf_dT": dfsurf_now,
                "dflat_dT": dflat_now, "dflwout_dT": dflwout_now,
                "dfsens_dT": dfsens_now, "dTsf_prev": dTsf,
                "dTi1_prev": dTi1_prev, "ferr": ferr,
                "converged": converged,
            })
            return result

        updated = lax.cond(jnp.logical_not(current["converged"]),
                           advance, lambda values: values, current)
        return updated, None

    carry, _ = lax.scan(scan_body, carry, jnp.arange(100))
    if calc_Tsfc and not semi_implicit:
        carry["flwoutn"] = carry["flwoutn"] + carry["dTsf_prev"] * carry["dflwout_dT"]
        carry["fsensn"] = carry["fsensn"] + carry["dTsf_prev"] * carry["dfsens_dT"]
        carry["flatn"] = carry["flatn"] + carry["dTsf_prev"] * carry["dflat_dT"]
    return (fswsfc, fswint, Sswabs, Iswabs,
            carry["zqin"], carry["zTin"], carry["zqsn"], carry["zTsn"],
            carry["Tsf"], carry["fsensn"], carry["flatn"],
            carry["flwoutn"], carry["fsurfn"], carry["fcondtopn"],
            carry["fcondbot"], carry["ferr"],
            jnp.logical_not(carry["converged"]))
