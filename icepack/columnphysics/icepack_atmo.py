"""Atmospheric surface boundary calculations from icepack_atmo.F90."""

import jax.numpy as jnp


STABILITY_MATH_DIAGNOSTIC_FIELDS = (
    "exp_requested", "exp_input_match", "exp_arg", "exp_before", "exp_used", "exp_applied",
    "atan_requested", "atan_input_match", "atan_arg", "atan_before", "atan_used", "atan_applied",
)

ATMO_DIAGNOSTIC_FIELDS = (
    "Tsf",
    "potT",
    "wind",
    "zlvl",
    "zlvs",
    "Qa",
    "rhoa",
    "vmag",
    "rdn",
    "TsfK",
    "delt",
    "ssq",
    "thva",
    "delq",
    "alzm",
    "alzs",
    "cp",
    "ustar",
    "tstar",
    "qstar",
    "holm",
    "hols",
    "psimh",
    "psixh",
    "rd",
    "rh",
    "re",
    "shcoef",
    "lhcoef",
    "psimh_before",
    "psixh_before",
    "psi_replay_applied",
) + tuple(f"{qty}_{name}" for qty in ("momentum", "scalar")
          for name in STABILITY_MATH_DIAGNOSTIC_FIELDS)


def neutral_drag_coeffs(apondn, alvl, vlvl, aice, vice, vsno,
                        aicen, vicen, tr_pond, parameters):
    """Neutral ice-atmosphere and ice-ocean form drag from CICE.

    Category is the last axis; leading axes may hold grid cells. The
    geometric quantities and each drag contribution retain Fortran names.
    """
    puny = parameters["puny"]
    p001 = parameters["p001"]
    rhoi = parameters["rhoi"]
    rhos = parameters["rhos"]
    rhow = parameters["rhow"]
    iceruf = parameters["iceruf"]
    zref = parameters["zref"]
    active = aice > p001
    ai = jnp.where(active, aice, 1.0)
    aii = 1.0 / ai
    astar = 1.0 / (1.0 - (8.0 / 300.0)**2)

    if tr_pond:
        apond = jnp.sum(apondn * aicen, axis=-1)
    else:
        apond = jnp.zeros_like(aice)
    hdraft = (rhoi * vice + rhos * vsno) * aii / rhow
    hfreebd = (vice + vsno) * aii - hdraft
    snow_loaded = hdraft >= vice * aii
    stoessel_freeboard = (
        hdraft * ai * (1.0 - rhoi / rhow)
        + (vsno - (vice - hdraft * ai) * rhoi / rhos)
        * (1.0 - rhos / rhow)) * aii
    hfreebd = jnp.where(snow_loaded, stoessel_freeboard, hfreebd)
    lfloe = 8.0 * (astar / (astar - ai))**0.5
    dfloe = lfloe * (1.0 / jnp.sqrt(ai) - 1.0)

    ardg = jnp.sum((1.0 - alvl) * aicen, axis=-1)
    vrdg = jnp.sum((1.0 - vlvl) * vicen, axis=-1)
    ridged = ardg > p001
    ardg_safe = jnp.where(ridged, ardg, 1.0)
    hridge = (vrdg / ardg_safe * 2.0
              * (0.75 * 4.0 / 1.0 * 0.4 / 0.4)
              / (0.8 + 0.8 * 0.4 / 0.4 * 4.0**2 / 1.0))
    distrdg = (2.0 * hridge * ai / ardg_safe
               * (0.75 / 0.4 * 4.0 / 1.0))
    hkeel = 4.0 * hridge
    dkeel = distrdg
    distrdg_safe = jnp.where(ridged & (distrdg > puny), distrdg, 1.0)
    dkeel_safe = jnp.where(ridged & (dkeel > puny), dkeel, 1.0)
    ridge_above = jnp.maximum(0.0, hridge - hfreebd)
    keel_below = jnp.maximum(0.0, hkeel - hdraft)
    Cdn_atm_skin = jnp.where(
        ridged, jnp.clip(0.0005 * (1.0 - 20.0 * ridge_above /
                                   distrdg_safe), 0.0, 0.02), 0.0005)
    Cdn_ocn_skin = jnp.where(
        ridged, jnp.clip(0.002 * (1.0 - 10.0 * keel_below /
                                  dkeel_safe), 0.0, 0.06), 0.002)
    log_air = jnp.log(zref / iceruf)
    ridge_height = jnp.where(ridge_above > puny, ridge_above, iceruf)
    keel_depth = jnp.where(keel_below > puny, keel_below, iceruf)
    Cdn_atm_rdg = jnp.where(
        ridged & (ridge_above > puny),
        jnp.minimum(0.1 * ridge_above / distrdg_safe
                    * (1.0 - jnp.exp(-0.18 * distrdg_safe / ridge_height))
                    * (jnp.log(ridge_height / iceruf) / log_air)**2,
                    0.02), 0.0)
    Cdn_ocn_keel = jnp.where(
        ridged & (keel_below > puny),
        jnp.clip(0.1 * keel_below / dkeel_safe
                 * (1.0 - jnp.exp(-0.18 * dkeel_safe / keel_depth))
                 * (jnp.log(keel_depth / iceruf) / log_air)**2,
                 0.0, 0.06), 0.0)

    floe_height = jnp.where(hfreebd > puny, hfreebd, 0.000327)
    draft_depth = jnp.where(hdraft > puny, hdraft, 0.000327)
    sca = 1.0 - jnp.exp(-22.0 * 0.5 * (1.0 - ai))
    Cdn_atm_floe = jnp.where(
        hfreebd > puny,
        jnp.clip(0.1 * (jnp.log(floe_height / 0.000327)
                        / jnp.log(zref / 0.000327))**2
                 * sca * hfreebd / lfloe, 0.0, 0.02), 0.0)
    pond_length = 2.26 * (1.0 - apond) + 24.63 * apond
    Cdn_atm_pond = jnp.where(
        hfreebd > puny,
        jnp.minimum(0.1 * apond**(1.0 / (zref * 0.5)) * apond
                    * hfreebd / pond_length
                    * (jnp.log(floe_height / 0.000327)
                       / jnp.log(zref / 0.000327))**2,
                    0.02), 0.0)
    Cdn_ocn_floe = jnp.where(
        hdraft > puny,
        jnp.clip(0.1 * (jnp.log(draft_depth / 0.000327)
                        / jnp.log(zref / 0.000327))**2
                 * sca * hdraft / lfloe, 0.0, 0.06), 0.0)
    Cdn_atm = jnp.minimum(
        Cdn_atm_skin + Cdn_atm_floe + Cdn_atm_pond + Cdn_atm_rdg, 0.02)
    Cdn_ocn = jnp.minimum(
        Cdn_ocn_skin + Cdn_ocn_floe + Cdn_ocn_keel, 0.06)
    zero = jnp.zeros_like(aice)
    return {
        "hfreebd": jnp.where(active, hfreebd, zero),
        "hdraft": jnp.where(active, hdraft, zero),
        "hridge": jnp.where(active & ridged, hridge, zero),
        "distrdg": jnp.where(active & ridged, distrdg, zero),
        "hkeel": jnp.where(active & ridged, hkeel, zero),
        "dkeel": jnp.where(active & ridged, dkeel, zero),
        "lfloe": jnp.where(active, lfloe, zero),
        "dfloe": jnp.where(active, dfloe, zero),
        "Cdn_ocn": jnp.where(active, Cdn_ocn, parameters["dragio"]),
        "Cdn_ocn_skin": jnp.where(active, Cdn_ocn_skin, zero),
        "Cdn_ocn_floe": jnp.where(active, Cdn_ocn_floe, zero),
        "Cdn_ocn_keel": jnp.where(active, Cdn_ocn_keel, zero),
        "Cdn_atm": jnp.where(active, Cdn_atm,
                              (parameters["vonkar"] / log_air)**2),
        "Cdn_atm_skin": jnp.where(active, Cdn_atm_skin, zero),
        "Cdn_atm_floe": jnp.where(active, Cdn_atm_floe, zero),
        "Cdn_atm_pond": jnp.where(active, Cdn_atm_pond, zero),
        "Cdn_atm_rdg": jnp.where(active, Cdn_atm_rdg, zero),
    }


def icepack_atm_boundary(sfctype, Tsf, potT, uatm, vatm, wind, zlvl,
                         Qa, rhoa, strx, stry, Tref, Qref, Cdn_atm,
                         options, parameters, uvel=None, vvel=None,
                         zlvs=None, Qa_iso=None, Qref_iso=None, n_iso=None,
                         debug_diagnostics=False, debug_psi_replay=None,
                         debug_psi_values=None, debug_math_replay=None):
    """CICE constant, similarity, and mixed atmosphere boundary branches."""
    method = options["atmbndy"].strip()
    if debug_diagnostics:
        atmo_diagnostics = jnp.zeros((7, len(ATMO_DIAGNOSTIC_FIELDS)),
                                     dtype=Tsf.dtype)
    if sfctype == "ice":
        Lheat = parameters["Lsub"]
        qqq = parameters["qqqice"]
        TTT = parameters["TTTice"]
    elif sfctype == "ocn":
        Lheat = parameters["Lvap"]
        qqq = parameters["qqqocn"]
        TTT = parameters["TTTocn"]
    else:
        raise ValueError("sfctype must be 'ice' or 'ocn'")

    Cdn_atm_ratio_n = jnp.ones_like(Cdn_atm)
    Uref = jnp.zeros_like(Tsf)
    if method == "constant":
        # atmo_boundary_const: reference T/Q and neutral drag stay unchanged.
        if bool(options["calc_strair"]):
            tau = rhoa * 0.0012 * wind
            strx = tau * uatm
            stry = tau * vatm
        TsfK = Tsf + parameters["Tffresh"]
        # The Fortran constant branch uses ocean saturation coefficients
        # even for an ice surface.
        qsat = parameters["qqqocn"] * jnp.exp(-parameters["TTTocn"] / TsfK)
        delt = potT - TsfK
        delq = Qa - qsat / rhoa
        shcoef = parameters["senscoef"] * parameters["cp_air"] * rhoa * wind
        lhcoef = parameters["latncoef"] * Lheat * rhoa * wind
        result = (strx, stry, Tref, Qref, delt, delq, lhcoef, shcoef,
                  Cdn_atm, Cdn_atm_ratio_n, Uref, Qref_iso)
        return (*result, atmo_diagnostics) if debug_diagnostics else result

    if method not in ("similarity", "mixed"):
        raise ValueError("atmbndy must be 'constant', 'similarity', or 'mixed'")

    vonkar = parameters["vonkar"]
    zref = parameters["zref"]
    cpvir = parameters["cp_wv"] / parameters["cp_air"] - 1.0
    umin = 0.5 if bool(options["highfreq"]) else 1.0
    if uvel is None:
        uvel = jnp.zeros_like(Tsf)
    if vvel is None:
        vvel = jnp.zeros_like(Tsf)

    if sfctype == "ice":
        if bool(options["highfreq"]):
            vmag = jnp.maximum(umin, jnp.sqrt((uatm - uvel)**2 + (vatm - vvel)**2))
        else:
            vmag = jnp.maximum(umin, wind)
        neutral = vonkar / jnp.log(zref / parameters["iceruf"])
        if bool(options["formdrag"]):
            rdn = jnp.where(Cdn_atm > parameters["puny"],
                            jnp.sqrt(jnp.maximum(Cdn_atm, 0.0)), neutral)
            Cdn_atm = jnp.where(Cdn_atm > parameters["puny"], Cdn_atm, neutral**2)
        else:
            rdn = neutral
            Cdn_atm = jnp.full_like(Tsf, neutral**2)
    else:
        vmag = jnp.maximum(umin, wind)
        rdn = jnp.sqrt(0.0027 / vmag + 0.000142 + 0.0000764 * vmag)

    TsfK = Tsf + parameters["Tffresh"]
    delt = potT - TsfK
    qsat = qqq * jnp.exp(-TTT / TsfK)
    ssq = qsat / rhoa
    thva = potT * (1.0 + parameters["zvir"] * Qa)
    delq = Qa - ssq
    alzm = jnp.log(zlvl / zref)
    alzs = jnp.log(zlvs / zref) if zlvs is not None else alzm
    cp = parameters["cp_air"] * (1.0 + cpvir * ssq)

    # Neutral first estimate; fixed upper iteration count preserves JAX
    # reverse-mode differentiation while matching each cell's convergence.
    ustar = rdn * vmag
    tstar = rdn * delt
    qstar = rdn * delq
    ustar_prev = 2.0 * ustar
    hols = jnp.zeros_like(Tsf)
    psixh = jnp.zeros_like(Tsf)
    rd = rdn
    rh = rdn
    re = rdn
    stable = jnp.ones_like(Tsf)
    if debug_diagnostics:
        atmo_diagnostics = atmo_diagnostics.at[0, :32].set(jnp.stack((
            Tsf, potT, wind, zlvl, zlvl if zlvs is None else zlvs,
            Qa, rhoa, vmag, rdn, TsfK, delt, ssq, thva, delq, alzm, alzs,
            cp, ustar, tstar, qstar, 0.0 * Tsf, hols, 0.0 * Tsf, psixh,
            rd, rh, re, 0.0 * Tsf, 0.0 * Tsf,
            0.0 * Tsf, 0.0 * Tsf, 0.0 * Tsf,
        )))
    for iteration in range(int(options["natmiter"])):
        active = jnp.abs(ustar - ustar_prev) / ustar > parameters["atmiter_conv"]
        holm_new = compute_stability_parameter(
            zlvl, thva, ustar, tstar, qstar, Qa, parameters,
        )
        hols_new = (compute_stability_parameter(
            zlvs, thva, ustar, tstar, qstar, Qa, parameters,
        ) if zlvs is not None else holm_new)
        if debug_math_replay is None:
            _, psimh_new = compute_stability_function("momentum", holm_new, parameters)
            stable_new, psixh_new = compute_stability_function("scalar", hols_new, parameters)
        else:
            exp_mask, atan_mask, reference = debug_math_replay
            _, psimh_new, momentum_math = compute_stability_function(
                "momentum", holm_new, parameters,
                (exp_mask[iteration] & active, atan_mask[iteration] & active,
                 reference[iteration]))
            stable_new, psixh_new, scalar_math = compute_stability_function(
                "scalar", hols_new, parameters,
                (exp_mask[iteration] & active, jnp.asarray(False), reference[iteration]))
        psimh_before, psixh_before = psimh_new, psixh_new
        psi_replay_applied = jnp.zeros_like(Tsf)
        if debug_psi_values is not None:
            replay = debug_psi_replay & active
            psimh_new = jnp.where(replay, debug_psi_values[iteration, 0], psimh_new)
            psixh_new = jnp.where(replay, debug_psi_values[iteration, 1], psixh_new)
            psi_replay_applied = replay.astype(Tsf.dtype)
        rd_new = rdn / (1.0 + rdn / vonkar * (alzm - psimh_new))
        rh_new = rdn / (1.0 + rdn / vonkar * (alzs - psixh_new))
        re_new = rdn / (1.0 + rdn / vonkar * (alzs - psixh_new))
        ustar_prev = jnp.where(active, ustar, ustar_prev)
        ustar = jnp.where(active, rd_new * vmag, ustar)
        tstar = jnp.where(active, rh_new * delt, tstar)
        qstar = jnp.where(active, re_new * delq, qstar)
        hols = jnp.where(active, hols_new, hols)
        psixh = jnp.where(active, psixh_new, psixh)
        stable = jnp.where(active, stable_new, stable)
        rd = jnp.where(active, rd_new, rd)
        rh = jnp.where(active, rh_new, rh)
        re = jnp.where(active, re_new, re)
        if debug_diagnostics and iteration < 5:
            values = jnp.stack((
                Tsf, potT, wind, zlvl, zlvl if zlvs is None else zlvs,
                Qa, rhoa, vmag, rdn, TsfK, delt, ssq, thva, delq, alzm, alzs,
                cp, ustar, tstar, qstar, holm_new, hols_new, psimh_new,
                psixh_new, rd, rh, re, 0.0 * Tsf, 0.0 * Tsf,
                psimh_before, psixh_before, psi_replay_applied,
            ))
            atmo_diagnostics = atmo_diagnostics.at[iteration + 1, :32].set(
                jnp.where(active, values, 0.0))
            if debug_math_replay is not None:
                atmo_diagnostics = atmo_diagnostics.at[iteration + 1, 32:].set(
                    jnp.concatenate((momentum_math, scalar_math)))

    if bool(options["calc_strair"]):
        if bool(options["highfreq"]) and sfctype == "ice":
            tau = rhoa * rd * rd
            relative = jnp.sqrt((uatm - uvel)**2 + (vatm - vvel)**2)
            strx = tau * relative * (uatm - uvel)
            stry = tau * relative * (vatm - vvel)
        else:
            tau = rhoa * ustar * rd
            strx = tau * uatm
            stry = tau * vatm
        Cdn_atm_ratio_n = rd * rd / rdn / rdn

    if method == "mixed":
        shcoef = parameters["senscoef"] * parameters["cp_air"] * rhoa * vmag
        lhcoef = parameters["latncoef"] * Lheat * rhoa * vmag
    else:
        shcoef = rhoa * ustar * cp * rh + 1.0
        lhcoef = rhoa * ustar * Lheat * re

    if debug_diagnostics:
        # Record 6 contains only the final transfer coefficients.
        atmo_diagnostics = atmo_diagnostics.at[6, 27:29].set(
            jnp.stack((shcoef, lhcoef)))

    # Two-metre reference temperature, humidity, and wind diagnostics.
    hols = hols * 2.0 / zlvl
    xd = jnp.sqrt(jnp.maximum(jnp.sqrt(jnp.abs(1.0 - 16.0 * hols)), 1.0))
    psi_scalar_unstable = 2.0 * jnp.log((1.0 + xd * xd) / 2.0)
    psix2 = -5.0 * hols * stable + (1.0 - stable) * psi_scalar_unstable
    al2 = jnp.log(zref / 2.0)
    fac = rh / vonkar * (alzs + al2 - psixh + psix2)
    Tref = potT - delt * fac - 0.01 * 2.0
    fac = re / vonkar * (alzs + al2 - psixh + psix2)
    Qref = Qa - delq * fac
    if bool(options["highfreq"]) and sfctype == "ice":
        Uref = jnp.sqrt((uatm - uvel)**2 + (vatm - vvel)**2) * rd / rdn
    else:
        Uref = vmag * rd / rdn

    if sfctype == "ice" and Qa_iso is not None and Qref_iso is not None:
        if n_iso is None:
            n_iso = Qa_iso.shape[-1]
        reference = Qa_iso[..., 1]
        ratio = jnp.where(reference[..., None] > parameters["puny"],
                          Qa_iso / jnp.where(reference[..., None] > parameters["puny"],
                                             reference[..., None], 1.0), 0.0)
        values = Qa_iso - ratio * delq[..., None] * fac[..., None]
        Qref_iso = jnp.zeros_like(Qref_iso).at[..., :n_iso].set(values[..., :n_iso])

    result = (strx, stry, Tref, Qref, delt, delq, lhcoef, shcoef,
              Cdn_atm, Cdn_atm_ratio_n, Uref, Qref_iso)
    return (*result, atmo_diagnostics) if debug_diagnostics else result


def compute_stability_parameter(zlvl, thva, ustar, tstar, qstar, Qa,
                                parameters):
    """Bound the Monin–Obukhov height ratio to [-10, 10]."""
    hol = (parameters["vonkar"] * parameters["gravit"] * zlvl
           * (tstar / thva + qstar / (1.0 / parameters["zvir"] + Qa))
           / ustar**2)
    return jnp.copysign(jnp.minimum(jnp.abs(hol), 10.0), hol)


def compute_stability_function(qty, hol, parameters, debug_math_replay=None):
    """Stable and unstable atmosphere corrections used by CICE."""
    stable = jnp.where(hol >= 0.0, 1.0, 0.0)
    exp_arg = -0.35 * hol
    exp_value = jnp.exp(exp_arg)
    if debug_math_replay is not None:
        exp_requested, atan_requested, reference = debug_math_replay
        exp_before = exp_value
        exp_input_match = exp_arg == reference[0]
        exp_applied = exp_requested & exp_input_match
        exp_value = jnp.where(exp_applied, reference[1], exp_value)
        atan_input_match = jnp.asarray(False)
        atan_applied = jnp.asarray(False)
        atan_before = jnp.zeros_like(hol)
        atan_value = jnp.zeros_like(hol)
    psi_stable = -(0.7 * hol + 0.75 * (hol - 14.3) * exp_value + 10.7)
    xd = jnp.sqrt(jnp.maximum(jnp.sqrt(jnp.abs(1.0 - 16.0 * hol)), 1.0))
    if qty == "momentum":
        atan_value = jnp.arctan(xd)
        if debug_math_replay is not None:
            atan_before = atan_value
            atan_input_match = xd == reference[2]
            atan_applied = atan_requested & atan_input_match
            atan_value = jnp.where(atan_applied, reference[3], atan_value)
        psi_unstable = (jnp.log((1.0 + xd * (2.0 + xd)) * (1.0 + xd * xd) / 8.0)
                        - 2.0 * atan_value + parameters["pih"])
    elif qty == "scalar":
        psi_unstable = 2.0 * jnp.log((1.0 + xd * xd) / 2.0)
    else:
        raise ValueError("qty must be 'momentum' or 'scalar'")
    psi = psi_stable * stable + (1.0 - stable) * psi_unstable
    if debug_math_replay is not None:
        # Reject a reference from another input; report the mismatch instead of forcing it.
        diagnostics = jnp.stack((
            exp_requested.astype(hol.dtype), exp_input_match.astype(hol.dtype),
            exp_arg, exp_before, exp_value, exp_applied.astype(hol.dtype),
            atan_requested.astype(hol.dtype), atan_input_match.astype(hol.dtype),
            xd, atan_before, atan_value, atan_applied.astype(hol.dtype),
        ))
        return stable, psi, diagnostics
    return stable, psi
