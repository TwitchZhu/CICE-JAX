"""Vertical sea-ice thermodynamics from icepack_therm_vertical.F90."""

from functools import partial

import jax
import jax.numpy as jnp
from jax import lax

from icepack.columnphysics.icepack_mushy_physics import (
    enthalpy_brine_jax,
    enthalpy_mush_liquid_fraction_jax,
    enthalpy_of_melting_jax,
    icepack_enthalpy_mush_jax,
    icepack_mushy_temperature_mush_jax,
    liquidus_temperature_mush_jax,
)
from icepack.columnphysics.icepack_therm_shared import (
    adjust_enthalpy,
    icepack_ice_temperature_jax,
)
from icepack.columnphysics.icepack_therm_bl99 import temperature_changes
from icepack.columnphysics.icepack_therm_mushy import (
    SURFACE_DIAGNOSTIC_FIELDS,
    PICARD_DIAGNOSTIC_FIELDS,
    temperature_changes_salinity,
)
from icepack.columnphysics.icepack_atmo import (
    ATMO_DIAGNOSTIC_FIELDS,
    icepack_atm_boundary,
    neutral_drag_coeffs,
)
from icepack.columnphysics.icepack_flux import merge_fluxes, set_sfcflux
from icepack.columnphysics.icepack_age import increment_age
from icepack.columnphysics.icepack_firstyear import update_FYarea
from icepack.columnphysics.icepack_snow import drain_snow
from icepack.columnphysics.icepack_meltpond_lvl import compute_ponds_lvl
from icepack.columnphysics.icepack_meltpond_sealvl import compute_ponds_sealvl
from icepack.columnphysics.icepack_meltpond_topo import compute_ponds_topo
from icepack.columnphysics.icepack_aerosol import update_aerosol
from icepack.columnphysics.icepack_isotope import update_isotope


# Optional diagnostics returned to the driver, never stored as model tracers.
SNOW_DIAGNOSTIC_FIELDS = (
    "snow_melt_entry", "snow_melt_warm", "snow_melt_internal", "snow_melt_top",
    "snow_melt_bottom", "snow_melt_thin", "snow_depth_entry",
    "snow_enthalpy_max_entry", "snow_melting_enthalpy", "etop_mlt_initial",
    "ebot_mlt_initial", "etop_mlt_after_snow", "ebot_mlt_after_ice",
)
POND_DIAGNOSTIC_FIELDS = (
    "aicen", "alvl", "apnd_initial", "hpnd_initial", "apond_after_thermo",
    "apnd_before_ponds", "hpnd_before_ponds", "meltt", "melts", "meltsliq",
    "frain", "rfrac", "Tair", "fsurfn", "dpnd_freebdn", "dpnd_dlidn",
    "apnd_after_ponds", "hpnd_after_ponds", "ipnd_after_ponds",
    "ipnd_before_ponds", "vicen", "vsnon", "dhs", "Tsf", "fcondtopn",
    "pond_flux_residual", "pond_energy_residual", "dt",
) + SNOW_DIAGNOSTIC_FIELDS + SURFACE_DIAGNOSTIC_FIELDS


def init_vertical_profile(aicen, vicen, vsnon, zqin, zqsn, zSin,
                          ktherm, l_brine, parameters):
    """Build one column's layer thickness, temperature and initial energy.

    Fortran warning/abort checks are returned as a Boolean so the JAX step
    can report them on the host after the pure numeric calculation.
    """
    nilyr = zqin.shape[-1]
    nslyr = zqsn.shape[-1]
    puny = parameters["puny"]
    rhoi = parameters["rhoi"]
    rhos = parameters["rhos"]
    cp_ice = parameters["cp_ice"]
    Lfresh = parameters["Lfresh"]
    hin = vicen / aicen
    hsn = vsnon / aicen
    hilyr = hin / nilyr
    hslyr = hsn / nslyr if nslyr > 0 else jnp.zeros_like(hsn)
    einit = jnp.zeros_like(hin)
    failed = jnp.asarray(False)

    for k in range(nslyr):
        snow_present = hslyr > parameters["hs_min"] / nslyr
        enthalpy = jnp.where(snow_present, zqsn[k], -rhos * Lfresh)
        Tmax = jnp.where(
            snow_present,
            -enthalpy * puny * nslyr /
            (rhos * cp_ice * jnp.where(snow_present, vsnon, 1.0)),
            puny)
        temperature = (Lfresh + enthalpy / rhos) / cp_ice
        failed = failed | (temperature > Tmax) | (temperature < -100.0)
        roundoff = temperature > 0.0
        temperature = jnp.where(roundoff, 0.0, temperature)
        enthalpy = jnp.where(roundoff, -rhos * Lfresh, enthalpy)
        zqsn = zqsn.at[k].set(enthalpy)
        einit = einit + hslyr * enthalpy
        if k == 0:
            zTsn = jnp.zeros_like(zqsn)
        zTsn = zTsn.at[k].set(temperature)
    if nslyr == 0:
        zTsn = jnp.zeros_like(zqsn)

    zTin = jnp.zeros_like(zqin)
    tice_high = jnp.asarray(False)
    for k in range(nilyr):
        if ktherm == 1:
            failed = failed | (zSin[k] < parameters["min_salin"] - puny)
        if ktherm == 2:
            Tmlts = liquidus_temperature_mush_jax(zSin[k])
        else:
            Tmlts = -zSin[k] * parameters["depressT"]
        temperature, bad_sqrt = icepack_ice_temperature_jax(
            zqin[k], zSin[k], ktherm, l_brine, parameters)
        failed = failed | bad_sqrt
        if l_brine:
            Tmax = Tmlts
        else:
            Tmax = (-zqin[k] * puny /
                    (rhos * cp_ice * jnp.where(vicen > 0.0, vicen, 1.0)))
        tice_high = tice_high | (temperature > Tmax)
        failed = failed | (temperature < -100.0)
        if ktherm == 2:
            q_melt = enthalpy_of_melting_jax(zSin[k], parameters)
            corrected_q = q_melt - 1.0
            zqin = zqin.at[k].set(jnp.where(tice_high, corrected_q, zqin[k]))
            corrected_T = icepack_mushy_temperature_mush_jax(
                zqin[k], zSin[k], parameters)
            temperature = jnp.where(tice_high, corrected_T, temperature)
        else:
            failed = failed | tice_high
            roundoff = temperature > 0.0
            temperature = jnp.where(roundoff, 0.0, temperature)
            zqin = zqin.at[k].set(jnp.where(
                roundoff, -rhoi * Lfresh, zqin[k]))
        if ktherm == 1:
            at_liquidus = temperature >= -zSin[k] * parameters["depressT"]
            temperature = jnp.where(
                at_liquidus, -zSin[k] * parameters["depressT"] - puny,
                temperature)
            zqin = zqin.at[k].set(jnp.where(
                at_liquidus,
                -zSin[k] * parameters["depressT"] * rhoi * parameters["cp_ocn"],
                zqin[k]))
        zTin = zTin.at[k].set(temperature)
        einit = einit + hilyr * zqin[k]
    return hin, hilyr, hsn, hslyr, zqin, zTin, zqsn, zTsn, einit, failed


def conservation_check_vthermo(dt, fsurfn, flatn, fhocnn, fswint,
                               fsnow, einit, efinal, fadvocn, parameters):
    """Return the energy-flux error and CICE's failure mask.

    Fortran writes detailed warnings and aborts on failure. The JAX time
    step keeps the numeric result for host-side reporting after execution.
    """
    einp = (fsurfn - flatn + fswint - fhocnn
            - fsnow * parameters["Lfresh"] - fadvocn) * dt
    ferr = jnp.abs(efinal - einit - einp) / dt
    return ferr, ferr > 1.1e-3


def update_state_vthermo(Tf, Tsf, hin, hsn, zqin, zSin, zqsn,
                         aicen, vicen, vsnon, ktherm):
    """Update category volumes or clear a completely melted column."""
    melted = hin <= 0.0
    aicen = jnp.where(melted, 0.0, aicen)
    vicen = jnp.where(melted, 0.0, aicen * hin)
    vsnon = jnp.where(melted, 0.0, aicen * hsn)
    Tsf = jnp.where(melted, Tf, Tsf)
    zqin = jnp.where(melted[..., None], 0.0, zqin)
    if ktherm == 2:
        zSin = jnp.where(melted[..., None], 0.0, zSin)
    zqsn = jnp.where(melted[..., None], 0.0, zqsn)
    return Tsf, zqin, zSin, zqsn, aicen, vicen, vsnon


def freeboard(snoice, hin, hsn, zqin, zqsn, dzi, dzs, dsnow,
              massice, massliq, parameters):
    """Turn submerged snow into snow ice, preserving layer energy.

    The last axis of each layer field runs from top to bottom. Leading
    axes may contain any batch of grid cells or ice categories.
    """
    puny = parameters["puny"]
    rhoi = parameters["rhoi"]
    rhos = parameters["rhos"]
    rhow = parameters["rhow"]

    wk1 = hsn - hin * (rhow - rhoi) / rhos
    active = (wk1 > puny) & (hsn > puny)
    dhsn = jnp.where(active, jnp.minimum(wk1 * rhoi / rhow, hsn), 0.0)
    dhin = dhsn * rhos / rhoi
    hqs = jnp.zeros_like(hin)

    # CICE removes snow from the bottom layer first.
    for k in range(dzs.shape[-1] - 1, -1, -1):
        dhs = jnp.where(dhin > puny, jnp.minimum(dhsn, dzs[..., k]), 0.0)
        hsn = hsn - dhs
        dsnow = dsnow - dhs
        massi = jnp.where(dzs[..., k] > puny,
                          jnp.maximum(0.0, 1.0 - dhs / jnp.where(
                              dzs[..., k] > puny, dzs[..., k], 1.0)), 0.0)
        massice = massice.at[..., k].set(jnp.where(
            dhin > puny, massice[..., k] * massi, massice[..., k]))
        massliq = massliq.at[..., k].set(jnp.where(
            dhin > puny, massliq[..., k] * massi, massliq[..., k]))
        dzs = dzs.at[..., k].set(dzs[..., k] - dhs)
        dhsn = jnp.maximum(dhsn - dhs, 0.0)
        hqs = hqs + dhs * zqsn[..., k]

    first_layer = dzi[..., 0] + dhin
    zqin = zqin.at[..., 0].set(jnp.where(
        dhin > puny,
        (dzi[..., 0] * zqin[..., 0] + hqs) /
        jnp.where(dhin > puny, first_layer, 1.0),
        zqin[..., 0]))
    dzi = dzi.at[..., 0].set(jnp.where(dhin > puny, first_layer, dzi[..., 0]))
    hin = hin + jnp.where(dhin > puny, dhin, 0.0)
    snoice = snoice + jnp.where(dhin > puny, dhin, 0.0)
    return snoice, hin, hsn, zqin, dzi, dzs, dsnow, massice, massliq


def frzmlt_bottom_lateral(dt, aice, frzmlt, vicen, vsnon, qicen, qsnon,
                           sst, Tf, strocnxT, strocnyT, Cdn_ocn, aicen,
                           fbot_xfer_type, tr_fsd, parameters,
                           afsdn=None, floe_rad_c=None, floe_binwidth=None):
    """Split available ocean heat between bottom and side melting.

    Category is the last axis; ``qicen`` and ``qsnon`` put the layer axis
    immediately before category. The optional floe-size distribution (FSD)
    modifies side melt through each category's floe radii.
    """
    puny = parameters["puny"]
    active = (aice > puny) & (frzmlt < 0.0)
    Tbot = Tf
    deltaT = jnp.maximum(sst - Tbot, 0.0)
    ustar = jnp.sqrt(jnp.sqrt(strocnxT**2 + strocnyT**2)
                      / parameters["rhow"])
    ustar = jnp.maximum(ustar, parameters["ustar_min"])
    if fbot_xfer_type == "Cdn_ocn":
        cpchr = (-parameters["cp_ocn"] * parameters["rhow"] * Cdn_ocn)
    else:
        cpchr = -parameters["cp_ocn"] * parameters["rhow"] * 0.006
    fbot = jnp.where(active,
                     jnp.maximum(cpchr * deltaT * ustar, frzmlt), 0.0)

    wlat_loc = 1.6e-6 * deltaT**1.36
    rside = (wlat_loc * dt * parameters["pi"]
             / (parameters["floeshape"] * parameters["floediam"]))
    rside = jnp.clip(rside, 0.0, 1.0)
    rsiden = jnp.broadcast_to(rside[..., None], aicen.shape)

    if tr_fsd:
        if afsdn is None or floe_rad_c is None or floe_binwidth is None:
            raise ValueError("FSD side melt requires afsdn and floe radius arrays")
        delta_an = jnp.zeros_like(aicen)
        for k in range(afsdn.shape[-2]):
            delta_an = delta_an + 2.0 * afsdn[..., k, :] / floe_rad_c[k]
        bin1_arealoss = -afsdn[..., 0, :] / floe_binwidth[0]
        delta_an = (delta_an - bin1_arealoss) * (-wlat_loc[..., None]) * dt
        rsiden = jnp.where(aicen > 0.0,
                           jnp.minimum(-delta_an, 1.0), rsiden)

    fside = jnp.zeros_like(aice)
    for n in range(aicen.shape[-1]):
        etot = jnp.zeros_like(aice)
        for k in range(qsnon.shape[-2]):
            etot = etot + qsnon[..., k, n] * vsnon[..., n] / qsnon.shape[-2]
        for k in range(qicen.shape[-2]):
            etot = etot + qicen[..., k, n] * vicen[..., n] / qicen.shape[-2]
        fside = fside + rsiden[..., n] * etot / dt

    xtmp = jnp.clip(frzmlt / (fbot + fside - puny), 0.0, 1.0)
    fbot = jnp.where(active, fbot * xtmp, 0.0)
    rsiden = jnp.where(active[..., None], rsiden * xtmp[..., None], 0.0)
    wlat = jnp.where(active, wlat_loc, 0.0)
    return Tbot, fbot, rsiden, wlat


def thickness_changes(dt, yday, fields, options, parameters):
    """Grow and melt one ice column, then remap its ice and snow layers.

    ``fields`` carries the Fortran inout arguments by their original names.
    Snow and ice layers run from top to bottom on their last axis. The
    returned ``failed`` mask represents Fortran's snow-mass abort checks.
    """
    result = dict(fields)
    ktherm = int(options["ktherm"])
    l_brine = bool(options["l_brine"])
    snwgrain = bool(options["snwgrain"])
    vapor_correction = bool(options["vapor_flux_correction"])
    puny = parameters["puny"]
    rhoi = parameters["rhoi"]
    rhos = parameters["rhos"]
    cp_ice = parameters["cp_ice"]
    Lfresh = parameters["Lfresh"]
    Lvap = parameters["Lvap"]
    nilyr = fields["zqin"].shape[-1]
    nslyr = fields["zqsn"].shape[-1]
    hin = fields["hin"]
    hsn = fields["hsn"]
    hilyr = fields["hilyr"]
    hslyr = fields["hslyr"]
    zqin = fields["zqin"]
    zqsn = fields["zqsn"]
    zSin = fields["zSin"]
    massice = fields["massice"]
    massliq = fields["massliq"]
    smice = fields["smice"]
    smliq = fields["smliq"]
    rsnw = fields["rsnw"]
    meltt = fields["meltt"]
    melts = fields["melts"]
    meltsliq = fields["meltsliq"]
    meltb = fields["meltb"]
    congel = fields["congel"]
    snoice = fields["snoice"]
    dsnow = fields["dsnow"]
    mlt_onset = fields["mlt_onset"]
    frz_onset = fields["frz_onset"]
    fcondbotn = fields["fcondbotn"]
    debug_ponds_inputs = bool(options.get("debug_model_ponds_inputs", False))
    if debug_ponds_inputs:
        snow_melt_entry = melts
        snow_depth_entry = hsn
        snow_enthalpy_max_entry = jnp.max(zqsn)
        snow_melt_warm = jnp.zeros_like(melts)
        snow_melt_internal = jnp.zeros_like(melts)
        snow_melt_top = jnp.zeros_like(melts)
        snow_melt_bottom = jnp.zeros_like(melts)
        snow_melt_thin = jnp.zeros_like(melts)

    # Initially every layer has the same thickness. Mushy ice measures melt
    # energy relative to the fully melted, saline layer enthalpy.
    dzi = jnp.full_like(zqin, hilyr)
    dzs = jnp.full_like(zqsn, hslyr)
    qmlt = (enthalpy_of_melting_jax(zSin, parameters)
            if ktherm == 2 else jnp.zeros_like(zqin))
    qm = zqin - qmlt
    emlt_atm = jnp.zeros_like(hin)
    emlt_ocn = jnp.zeros_like(hin)
    de_vapor = jnp.zeros_like(hin)

    # Fresh ice and snow cannot remain warmer than their melting point.
    if not l_brine:
        for k in range(nslyr):
            Ts = (Lfresh + zqsn[k] / rhos) / cp_ice
            warm = Ts > 0.0
            dhs = jnp.where(warm, cp_ice * Ts * dzs[k] / Lfresh, 0.0)
            mass = massice[k] + massliq[k]
            massi = jnp.where(dzs[k] > puny,
                              jnp.maximum(0.0, 1.0 - dhs /
                                          jnp.where(dzs[k] > puny, dzs[k], 1.0)), 0.0)
            new_ice = massice[k] * massi
            massice = massice.at[k].set(jnp.where(warm, new_ice, massice[k]))
            massliq = massliq.at[k].set(jnp.where(warm, mass - new_ice, massliq[k]))
            dzs = dzs.at[k].set(dzs[k] - dhs)
            melts = melts + dhs
            if debug_ponds_inputs:
                snow_melt_warm = snow_melt_warm + dhs
            zqsn = zqsn.at[k].set(jnp.where(warm, -rhos * Lfresh, zqsn[k]))
        for k in range(nilyr):
            Ti = (Lfresh + zqin[k] / rhoi) / cp_ice
            warm = Ti > 0.0
            dhi = jnp.where(warm, cp_ice * Ti * dzi[k] / Lfresh, 0.0)
            dzi = dzi.at[k].set(dzi[k] - dhi)
            zqin = zqin.at[k].set(jnp.where(warm, -rhoi * Lfresh, zqin[k]))

    # Energy for sublimation/condensation, top melt and bottom growth/melt.
    esub = jnp.maximum(-fields["flatn"] * dt, 0.0)
    econ = jnp.minimum(-fields["flatn"] * dt, 0.0)
    etop_mlt = jnp.maximum((fields["fsurfn"] - fields["fcondtopn"]) * dt, 0.0)
    ebot_mlt = jnp.maximum((fcondbotn - fields["fbot"]) * dt, 0.0)
    ebot_gro = jnp.minimum((fcondbotn - fields["fbot"]) * dt, 0.0)
    if debug_ponds_inputs:
        etop_mlt_initial = etop_mlt
        ebot_mlt_initial = ebot_mlt
    evapn = jnp.zeros_like(hin)
    evapsn = jnp.zeros_like(hin)
    evapin = jnp.zeros_like(hin)
    snow_surface = hsn > puny

    # Condensation joins the upper snow layer if snow is present, otherwise
    # it joins the upper ice layer. With vapor correction it arrives at 0 C.
    if vapor_correction:
        dhs_cond = econ / (-rhos * Lfresh - rhos * Lvap)
        dhi_cond = econ / (-rhoi * Lfresh - rhoi * Lvap)
        hstot = dzs[0] + dhs_cond
        new_qsn = ((dzs[0] * zqsn[0] - dhs_cond * rhos * Lfresh)
                   / jnp.where(hstot > puny, hstot, 1.0))
        if ktherm < 2:
            new_qsn = jnp.minimum(new_qsn, -rhos * Lfresh)
        zqsn = zqsn.at[0].set(jnp.where(
            snow_surface & (hstot > puny), new_qsn, zqsn[0]))
        de_vapor = jnp.where(
            snow_surface, de_vapor,
            de_vapor - dhi_cond * (qm[0] + rhoi * Lfresh))
    else:
        dhs_cond = econ / (zqsn[0] - rhos * Lvap)
        dhi_cond = econ / (qm[0] - rhoi * Lvap)
    dzs = dzs.at[0].set(dzs[0] + jnp.where(snow_surface, dhs_cond, 0.0))
    dzi = dzi.at[0].set(dzi[0] + jnp.where(snow_surface, 0.0, dhi_cond))
    massice = massice.at[0].set(massice[0] + jnp.where(
        snow_surface, dhs_cond * rhos, 0.0))
    evapn = jnp.where(snow_surface, dhs_cond * rhos, dhi_cond * rhoi)
    evapsn = jnp.where(snow_surface, dhs_cond * rhos, 0.0)
    evapin = jnp.where(snow_surface, 0.0, dhi_cond * rhoi)
    emlt_atm = emlt_atm - jnp.where(snow_surface, 0.0, qmlt[0] * dhi_cond)

    # Grow bottom ice using the selected BL99 or mushy freezing scheme.
    if ktherm == 2:
        if options["congel_freeze"] == "one-step":
            qbotm = enthalpy_mush_liquid_fraction_jax(
                fields["Tbot"], parameters["phi_i_mushy"], parameters)
            qbotw = enthalpy_brine_jax(fields["sst"], parameters)
            qbotp = qbotm - qbotw
            dhi = ebot_gro / qbotp
            hstot = dzi[-1] * zSin[-1] + dhi * fields["sss"] * parameters["phi_i_mushy"]
        else:
            qbotm = icepack_enthalpy_mush_jax(
                fields["Tbot"], fields["sss"], parameters)
            qbotp = -Lfresh * rhoi * (1.0 - parameters["phi_i_mushy"])
            qbotw = qbotm - qbotp
            dhi = ebot_gro / qbotp
            hstot = dzi[-1] * zSin[-1] + dhi * fields["sss"]
        hqtot = dzi[-1] * zqin[-1] + dhi * qbotm
        emlt_ocn = emlt_ocn - qbotw * dhi
    else:
        Tmlts = -zSin[-1] * parameters["depressT"]
        if l_brine:
            qbot = -rhoi * (
                cp_ice * (Tmlts - fields["Tbot"])
                + Lfresh * (1.0 - Tmlts / fields["Tbot"])
                - parameters["cp_ocn"] * Tmlts)
            qbot = jnp.minimum(qbot, -0.5 * rhoi * Lfresh)
        else:
            qbot = -rhoi * (-cp_ice * fields["Tbot"] + Lfresh)
        dhi = ebot_gro / qbot
        hqtot = dzi[-1] * zqin[-1] + dhi * qbot
        hstot = jnp.zeros_like(hin)
    dzi = dzi.at[-1].add(dhi)
    zqin = zqin.at[-1].set(jnp.where(
        dzi[-1] > puny, hqtot / jnp.where(dzi[-1] > puny, dzi[-1], 1.0),
        zqin[-1]))
    if ktherm == 2:
        zSin = zSin.at[-1].set(jnp.where(
            dzi[-1] > puny, hstot / jnp.where(dzi[-1] > puny, dzi[-1], 1.0),
            zSin[-1]))
        qmlt = qmlt.at[-1].set(enthalpy_of_melting_jax(zSin[-1], parameters))
    qm = qm.at[-1].set(jnp.where(dzi[-1] > puny,
                                 zqin[-1] - qmlt[-1], qm[-1]))
    congel = congel + dhi
    frz_onset = jnp.where((dhi > puny) & (frz_onset < puny), yday, frz_onset)

    # Remove internal mushy snow melt, then sublimate and melt snow from the top.
    for k in range(nslyr):
        if ktherm == 2:
            internal = zqsn[k] > -rhos * Lfresh
            dhs = jnp.where(internal,
                            jnp.maximum(-dzs[k],
                                        -((zqsn[k] + rhos * Lfresh)
                                          / (rhos * Lfresh)) * dzs[k]), 0.0)
            mass = massice[k] + massliq[k]
            massi = jnp.where(dzs[k] > puny,
                              jnp.maximum(0.0, 1.0 + dhs /
                                          jnp.where(dzs[k] > puny, dzs[k], 1.0)), 0.0)
            new_ice = massice[k] * massi
            massice = massice.at[k].set(jnp.where(internal, new_ice, massice[k]))
            massliq = massliq.at[k].set(jnp.where(internal, mass - new_ice,
                                                  massliq[k]))
            dzs = dzs.at[k].add(dhs)
            zqsn = zqsn.at[k].set(jnp.where(internal, -rhos * Lfresh, zqsn[k]))
            melts = melts - dhs
            if debug_ponds_inputs:
                snow_melt_internal = snow_melt_internal - dhs

        if vapor_correction:
            qsub = -rhos * Lfresh - rhos * Lvap
        else:
            qsub = zqsn[k] - rhos * Lvap
        dhs = jnp.maximum(-dzs[k], esub / qsub)
        if vapor_correction:
            de_vapor = (de_vapor - dhs
                        * jnp.minimum(zqsn[k] + rhos * Lfresh, 0.0))
        mass = massice[k] + massliq[k]
        massi = jnp.where(dzs[k] > puny,
                          1.0 + dhs / jnp.where(dzs[k] > puny, dzs[k], 1.0), 0.0)
        new_ice = massice[k] * massi
        massice = massice.at[k].set(new_ice)
        massliq = massliq.at[k].set(jnp.maximum(0.0, mass + rhos * dhs - new_ice))
        dzs = dzs.at[k].add(dhs)
        esub = jnp.maximum(esub - dhs * qsub, 0.0)
        evapn = evapn + dhs * rhos
        evapsn = evapsn + dhs * rhos

        dhs = jnp.maximum(-dzs[k], etop_mlt / zqsn[k])
        mass = massice[k] + massliq[k]
        massi = jnp.where(dzs[k] > puny,
                          jnp.maximum(0.0, 1.0 + dhs /
                                      jnp.where(dzs[k] > puny, dzs[k], 1.0)), 0.0)
        new_ice = massice[k] * massi
        massice = massice.at[k].set(new_ice)
        massliq = massliq.at[k].set(mass - new_ice)
        dzs = dzs.at[k].add(dhs)
        etop_mlt = jnp.maximum(etop_mlt - dhs * zqsn[k], 0.0)
        mlt_onset = jnp.where((dhs < -puny) & (mlt_onset < puny),
                              yday, mlt_onset)
        melts = melts - dhs

        if debug_ponds_inputs:
            snow_melt_top = snow_melt_top - dhs
    if debug_ponds_inputs:
        etop_mlt_after_snow = etop_mlt

    # Sublimate upper ice, melt upper ice, then melt lower ice from the bottom.
    for k in range(nilyr):
        qsub = qm[k] - rhoi * Lvap
        dhi = jnp.maximum(-dzi[k], esub / qsub)
        dzi = dzi.at[k].add(dhi)
        esub = jnp.maximum(esub - dhi * qsub, 0.0)
        evapn = evapn + dhi * rhoi
        evapin = evapin + dhi * rhoi
        emlt_ocn = emlt_ocn - qmlt[k] * dhi

        melting = qm[k] < 0.0
        dhi = jnp.where(melting,
                        jnp.maximum(-dzi[k], etop_mlt /
                                    jnp.where(melting, qm[k], -1.0)), -dzi[k])
        qm = qm.at[k].set(jnp.where(melting, qm[k], 0.0))
        emlt_ocn = emlt_ocn - jnp.maximum(zqin[k], qmlt[k]) * dhi
        dzi = dzi.at[k].add(dhi)
        etop_mlt = jnp.maximum(etop_mlt - dhi * qm[k], 0.0)
        mlt_onset = jnp.where((dhi < -puny) & (mlt_onset < puny),
                              yday, mlt_onset)
        meltt = meltt - dhi

    for k in range(nilyr - 1, -1, -1):
        melting = qm[k] < 0.0
        dhi = jnp.where(melting,
                        jnp.maximum(-dzi[k], ebot_mlt /
                                    jnp.where(melting, qm[k], -1.0)), -dzi[k])
        qm = qm.at[k].set(jnp.where(melting, qm[k], 0.0))
        emlt_ocn = emlt_ocn - jnp.maximum(zqin[k], qmlt[k]) * dhi
        dzi = dzi.at[k].add(dhi)
        ebot_mlt = jnp.maximum(ebot_mlt - dhi * qm[k], 0.0)
        meltb = meltb - dhi

    # Only after all ice has melted can bottom heat melt remaining snow.
    if debug_ponds_inputs:
        ebot_mlt_after_ice = ebot_mlt
    for k in range(nslyr - 1, -1, -1):
        dhs = jnp.maximum(-dzs[k], ebot_mlt / zqsn[k])
        mass = massice[k] + massliq[k]
        massi = jnp.where(dzs[k] > puny,
                          jnp.maximum(0.0, 1.0 + dhs /
                                      jnp.where(dzs[k] > puny, dzs[k], 1.0)), 0.0)
        new_ice = massice[k] * massi
        massice = massice.at[k].set(new_ice)
        massliq = massliq.at[k].set(mass - new_ice)
        dzs = dzs.at[k].add(dhs)
        ebot_mlt = jnp.maximum(ebot_mlt - dhs * zqsn[k], 0.0)
        melts = melts - dhs
        if debug_ponds_inputs:
            snow_melt_bottom = snow_melt_bottom - dhs

    fhocnn = fields["fbot"] + (esub + etop_mlt + ebot_mlt) / dt
    hsn_new = jnp.where(fields["fsnow"] > 0.0,
                        fields["fsnow"] / rhos * dt, 0.0)
    zqsnew = -rhos * Lfresh
    hstot = dzs[0] + hsn_new
    snowfall = (fields["fsnow"] > 0.0) & (hstot > 0.0)
    qnew = ((dzs[0] * zqsn[0] + hsn_new * zqsnew)
            / jnp.where(snowfall, hstot, 1.0))
    zqsn = zqsn.at[0].set(jnp.where(
        snowfall, jnp.minimum(qnew, zqsnew), zqsn[0]))
    dzs = dzs.at[0].set(jnp.where(snowfall, hstot, dzs[0]))
    massice = massice.at[0].add(jnp.where(snowfall, fields["fsnow"] * dt, 0.0))
    massliq = massliq.at[0].add(fields["frain"] * dt)

    hin = jnp.zeros_like(hin)
    hsn = jnp.zeros_like(hsn)
    for k in range(nilyr):
        hin = hin + dzi[k]
    for k in range(nslyr):
        hsn = hsn + dzs[k]
        dsnow = dsnow + dzs[k] - hslyr

    if snwgrain:
        old_top = jnp.maximum(0.0, dzs[0] - hsn_new)
        new_radius = ((parameters["rsnw_fall"] * hsn_new + rsnw[0] * old_top)
                      / jnp.maximum(hsn_new + old_top, puny))
        new_radius = jnp.maximum(parameters["rsnw_fall"],
                                 jnp.minimum(parameters["rsnw_tmax"], new_radius))
        rsnw = rsnw.at[0].set(jnp.where(hsn_new > 0.0, new_radius, rsnw[0]))

    # Snow submerged below freeboard becomes ice in the BL99 formulation.
    if ktherm != 2:
        (snoice, hin, hsn, zqin, dzi, dzs, dsnow,
         massice, massliq) = freeboard(
            snoice, hin, hsn, zqin, zqsn, dzi, dzs, dsnow,
            massice, massliq, parameters)

    failed = jnp.asarray(False)
    if snwgrain:
        for k in range(nslyr):
            meltsliq = meltsliq + massliq[k]
            layer_present = dzs[k] > puny
            smice = smice.at[k].set(jnp.where(
                layer_present, massice[k] /
                jnp.where(layer_present, dzs[k], 1.0), 0.0))
            smliq = smliq.at[k].set(jnp.where(
                layer_present, massliq[k] /
                jnp.where(layer_present, dzs[k], 1.0), 0.0))
            massice = massice.at[k].set(jnp.where(layer_present, massice[k], 0.0))
            massliq = massliq.at[k].set(jnp.where(layer_present, massliq[k], 0.0))
        for k in range(nslyr):
            failed = failed | (massice[k] < -puny) | (massliq[k] < -puny)
            failed = failed | (smice[k] > parameters["rhofresh"])
            massice = massice.at[k].set(jnp.where(
                (massice[k] < 0.0) & (massice[k] > -puny), 0.0, massice[k]))
            massliq = massliq.at[k].set(jnp.where(
                (massliq[k] < 0.0) & (massliq[k] > -puny), 0.0, massliq[k]))

    # Repartition unequal ice/snow layers to equal thickness, conserving
    # energy and (for mushy ice) salinity.
    hin = jnp.maximum(hin, 0.0)
    hsn = jnp.maximum(hsn, 0.0)
    hilyr = hin / nilyr
    hslyr = hsn / nslyr
    zi1 = [jnp.zeros_like(hin)]
    zi2 = [jnp.zeros_like(hin)]
    for k in range(nilyr - 1):
        zi1.append(zi1[-1] + dzi[k])
        zi2.append(zi2[-1] + hilyr)
    zi1.append(hin)
    zi2.append(hin)
    zi1 = jnp.stack(zi1)
    zi2 = jnp.stack(zi2)
    zqin = adjust_enthalpy(zi1, zi2, hilyr, hin, zqin, puny)
    if ktherm == 2:
        zSin = adjust_enthalpy(zi1, zi2, hilyr, hin, zSin, puny)
    if nslyr > 1:
        zs1 = [jnp.zeros_like(hsn)]
        zs2 = [jnp.zeros_like(hsn)]
        for k in range(nslyr - 1):
            zs1.append(zs1[-1] + dzs[k])
            zs2.append(zs2[-1] + hslyr)
        zs1.append(hsn)
        zs2.append(hsn)
        zs1 = jnp.stack(zs1)
        zs2 = jnp.stack(zs2)
        zqsn = adjust_enthalpy(zs1, zs2, hslyr, hsn, zqsn, puny)
        if snwgrain:
            rsnw = adjust_enthalpy(zs1, zs2, hslyr, hsn, rsnw, puny)
            smice = adjust_enthalpy(zs1, zs2, hslyr, hsn, smice, puny)
            smliq = adjust_enthalpy(zs1, zs2, hslyr, hsn, smliq, puny)
            massice = smice * hslyr
            massliq = smliq * hslyr

    if ktherm == 2:
        thin = (hsn <= puny) | (hin <= 0.0)
        for k in range(nslyr):
            fhocnn = fhocnn + jnp.where(
                thin, zqsn[k] * hsn / (nslyr * dt), 0.0)
            zqsn = zqsn.at[k].set(jnp.where(thin, -rhos * Lfresh, zqsn[k]))
            if snwgrain:
                meltsliq = meltsliq + jnp.where(thin, massice[k], 0.0)
                smice = smice.at[k].set(jnp.where(thin, rhos, smice[k]))
                smliq = smliq.at[k].set(jnp.where(thin, 0.0, smliq[k]))
                rsnw = rsnw.at[k].set(jnp.where(
                    thin, parameters["rsnw_fall"], rsnw[k]))
        melts = melts + jnp.where(thin, hsn, 0.0)
        if debug_ponds_inputs:
            snow_melt_thin = jnp.where(thin, hsn, 0.0)
        hsn = jnp.where(thin, 0.0, hsn)
        hslyr = jnp.where(thin, 0.0, hslyr)

    if vapor_correction:
        fhocnn = fhocnn + de_vapor / dt
    efinal = -evapn * Lvap
    evapn = evapn / dt
    evapsn = evapsn / dt
    evapin = evapin / dt
    for k in range(nslyr):
        efinal = efinal + hslyr * zqsn[k]
    for k in range(nilyr):
        efinal = efinal + hilyr * zqin[k]
    if ktherm == 2:
        fhocnn = fhocnn + emlt_ocn / dt
        efinal = efinal + emlt_atm

    if debug_ponds_inputs:
        result["snow_diagnostics"] = jnp.stack((
            snow_melt_entry, snow_melt_warm, snow_melt_internal, snow_melt_top,
            snow_melt_bottom, snow_melt_thin, snow_depth_entry,
            snow_enthalpy_max_entry, -rhos * Lfresh, etop_mlt_initial,
            ebot_mlt_initial, etop_mlt_after_snow, ebot_mlt_after_ice,
        ))
    result.update({
        "efinal": efinal, "hin": hin, "hilyr": hilyr,
        "hsn": hsn, "hslyr": hslyr, "hsn_new": hsn_new,
        "zqin": zqin, "zqsn": zqsn, "zSin": zSin,
        "smice": smice, "massice": massice,
        "smliq": smliq, "massliq": massliq, "rsnw": rsnw,
        "fcondbotn": fcondbotn, "fhocnn": fhocnn,
        "evapn": evapn, "evapsn": evapsn, "evapin": evapin,
        "meltt": meltt, "melts": melts, "meltsliq": meltsliq,
        "meltb": meltb, "congel": congel, "snoice": snoice,
        "mlt_onset": mlt_onset, "frz_onset": frz_onset,
        "dsnow": dsnow,
    })
    return result, failed


def thermo_vertical(dt, yday, fields, options, parameters, coupled=None,
                    picard_active=True):
    """Advance one vertical ice column using the Fortran argument names.

    ``failed`` combines the original profile, temperature, thickness and
    energy-conservation abort conditions for reporting outside the JAX step.
    """
    ktherm = int(options["ktherm"])

    result = dict(fields)
    zero = jnp.zeros_like(fields["aicen"])
    result.update({
        "flwoutn": zero, "evapn": zero, "evapsn": zero, "evapin": zero,
        "freshn": zero, "fsaltn": zero, "fhocnn": zero, "fadvocn": zero,
        "meltt": zero, "meltb": zero, "melts": zero, "meltsliq": zero,
        "congel": zero, "snoice": zero, "dsnow": zero,
        "massice": jnp.zeros_like(fields["smice"]),
        "massliq": jnp.zeros_like(fields["smliq"]),
    })
    if options["tr_pond"]:
        result["dpnd_flush"] = zero
        result["dpnd_expon"] = zero
    if options["calc_Tsfc"]:
        result.update({
            "fsensn": zero, "flatn": zero,
            "fsurfn": zero, "fcondtopn": zero,
        })

    (hin, hilyr, hsn, hslyr, zqin, zTin, zqsn, zTsn,
     einit, profile_failed) = init_vertical_profile(
         fields["aicen"], fields["vicen"], fields["vsnon"],
         fields["zqin"], fields["zqsn"], fields["zSin"],
         ktherm, options["l_brine"], parameters)
    worki = hin
    works = hsn
    if options["saltflux_option"] == "prognostic":
        saltvol = (parameters["rhoi"] * hin * parameters["p001"]
                   * jnp.sum(fields["zSin"]) / zqin.shape[-1])

    if ktherm == 2:
        temperature_result = temperature_changes_salinity(
            dt, fields["rhoa"], fields["flw"], fields["potT"],
            fields["Qa"], fields["shcoef"], fields["lhcoef"],
            fields["fswsfc"], fields["fswint"], fields["Sswabs"],
            fields["Iswabs"], hilyr, hslyr, fields["apond"],
            fields["hpond"], zqin, zTin, zqsn, zTsn,
            fields["zSin"], fields["Tsf"], fields["Tbot"],
            fields["sss"], fields["smice"], fields["smliq"],
            result.get("dpnd_flush", zero),
            result.get("dpnd_expon", zero), options, parameters,
            coupled=coupled, picard_active=picard_active)
        if bool(options.get("profile_picard", False)):
            result["picard_iterations"] = temperature_result[-1]
            temperature_result = temperature_result[:-1]
        if bool(options.get("debug_model_ponds_inputs", False)):
            result["surface_diagnostics"] = temperature_result[-2]
            result["picard_diagnostics"] = temperature_result[-1]
            temperature_result = temperature_result[:-2]
        (fswsfc, fswint, Sswabs, Iswabs, hilyr, hslyr,
         apond, hpond, zqin, zTin, zqsn, zTsn, zSin,
         Tsf, fsensn, flatn, flwoutn, fsurfn, fcondtopn,
         fcondbotn, fadvocn, snoice, smice, smliq,
         dpnd_flush, dpnd_expon, temperature_failed) = temperature_result
        temperature_error = zero
        result.update({
            "zSin": zSin, "apond": apond, "hpond": hpond,
            "smice": smice, "smliq": smliq, "fadvocn": fadvocn,
            "snoice": snoice, "dpnd_flush": dpnd_flush,
            "dpnd_expon": dpnd_expon,
        })
    else:
        (fswsfc, fswint, Sswabs, Iswabs, zqin, zTin, zqsn, zTsn,
         Tsf, fsensn, flatn, flwoutn, fsurfn, fcondtopn,
         fcondbotn, temperature_error, temperature_failed) = temperature_changes(
             dt, fields["rhoa"], fields["flw"], fields["potT"],
             fields["Qa"], fields["shcoef"], fields["lhcoef"],
             fields["fswsfc"], fields["fswint"], fields["Sswabs"],
             fields["Iswabs"], hilyr, hslyr, zqin, zTin, zqsn, zTsn,
             fields["zSin"], fields["Tsf"], fields["Tbot"],
             result["fsensn"], result["flatn"], result["flwoutn"],
             result["fsurfn"], result["fcondtopn"], einit,
             options, parameters, coupled=coupled)

    if options["snwgrain"]:
        result["massice"] = result["smice"] * hslyr
        result["massliq"] = result["smliq"] * hslyr

    einter = hslyr * jnp.sum(zqsn) + hilyr * jnp.sum(zqin)
    if zTsn.shape[-1] > 0:
        interface_temperature = jnp.where(
            hslyr > parameters["puny"],
            (hilyr * zTsn[-1] + hslyr * zTin[0]) /
            jnp.where(hslyr + hilyr > parameters["puny"],
                      hslyr + hilyr, 1.0),
            Tsf)
    else:
        interface_temperature = Tsf
    result["Tsnice"] = fields["Tsnice"] + jnp.where(
        hilyr > parameters["puny"],
        fields["aicen"] * interface_temperature, 0.0)
    result.update({
        "hin": hin, "hilyr": hilyr, "hsn": hsn, "hslyr": hslyr,
        "zqin": zqin, "zqsn": zqsn,
        "Tsf": Tsf, "fswsfc": fswsfc, "fswint": fswint,
        "Sswabs": Sswabs, "Iswabs": Iswabs,
        "fsensn": fsensn, "flatn": flatn, "flwoutn": flwoutn,
        "fsurfn": fsurfn, "fcondtopn": fcondtopn,
        "fcondbotn": fcondbotn, "einit": einit, "einter": einter,
    })
    result, thickness_failed = thickness_changes(
        dt, yday, result, options, parameters)
    energy_error, conservation_failed = conservation_check_vthermo(
        dt, result["fsurfn"], result["flatn"], result["fhocnn"],
        result["fswint"], result["fsnow"], einit, result["efinal"],
        result["fadvocn"], parameters)

    # Prescribed ice keeps the old thickness after the conservation check.
    if options.get("prescribed_ice", False):
        result["hin"] = worki
        result["fhocnn"] = zero
    dhi = result["hin"] - worki
    dhs = result["hsn"] - works - result["hsn_new"]
    result["freshn"] = (result["evapn"] -
                        (parameters["rhoi"] * dhi + parameters["rhos"] * dhs) / dt)
    if options["saltflux_option"] == "prognostic":
        dfsalt = (parameters["rhoi"] * result["hin"] * parameters["p001"]
                  * jnp.sum(result["zSin"]) / zqin.shape[-1])
        result["fsaltn"] = -(dfsalt - saltvol) / dt
    else:
        result["fsaltn"] = (-parameters["rhoi"] * dhi
                             * parameters["ice_ref_salinity"]
                             * parameters["p001"] / dt)
    result["fhocnn"] = result["fhocnn"] + result["fadvocn"]
    if options["tr_pond_topo"]:
        result["fpond"] = result["fpond"] - jnp.where(
            result["hin"] == 0.0,
            fields["aicen"] * result["apond"] * result["hpond"], 0.0)

    (result["Tsf"], result["zqin"], result["zSin"], result["zqsn"],
     result["aicen"], result["vicen"], result["vsnon"]) = update_state_vthermo(
         fields["Tbot"], result["Tsf"], result["hin"], result["hsn"],
         result["zqin"], result["zSin"], result["zqsn"],
         fields["aicen"], fields["vicen"], fields["vsnon"], ktherm)
    failed = (profile_failed | temperature_failed | thickness_failed
              | conservation_failed)
    return result, failed, {"temperature_error": temperature_error,
                            "energy_error": energy_error}


def icepack_step_therm1(dt, yday, fields, options, parameters,
                       floe_rad_c=None, floe_binwidth=None):
    """Advance vertical thermodynamics for every ice category in one cell.

    ``fields`` uses Fortran argument names. Category is the last axis of
    scalar category arrays; snow/ice layers precede it.
    """
    return _icepack_step_therm1(
        dt, yday, fields, tuple(sorted(options.items())), parameters,
        floe_rad_c, floe_binwidth)


@jax.profiler.annotate_function
def icepack_step_therm1_batch(dt, yday, fields, options, parameters,
                             floe_rad_c=None, floe_binwidth=None):
    """Advance flattened physical cells with a cached batched executable."""
    return _icepack_step_therm1_batch(
        dt, yday, fields, tuple(sorted(options.items())), parameters,
        floe_rad_c, floe_binwidth)


@partial(jax.jit, static_argnames=("options",))
def _icepack_step_therm1_batch(dt, yday, fields, options, parameters,
                              floe_rad_c=None, floe_binwidth=None):
    # Cache vmap's transformation as well as the single-cell computation.
    options = dict(options)
    axis_name = "therm1_cells" if options.get("picard_batch_stop", False) else None
    if axis_name is not None:
        options["picard_axis_name"] = axis_name
    options = tuple(sorted(options.items()))
    return jax.vmap(lambda cell: _icepack_step_therm1(
        dt, yday, cell, options, parameters, floe_rad_c, floe_binwidth),
        axis_name=axis_name)(fields)


@partial(jax.jit, static_argnames=("options",))
def _icepack_step_therm1(dt, yday, fields, options, parameters,
                        floe_rad_c=None, floe_binwidth=None):
    # Keep calendar values dynamic so advancing time reuses the compiled step.
    options = dict(options)
    result = dict(fields)
    ncat = fields["aicen"].shape[-1]
    debug_ponds_inputs = bool(options.get("debug_model_ponds_inputs", False))
    profile_picard = bool(options.get("profile_picard", False))
    if profile_picard:
        result["picard_iterations"] = jnp.zeros((ncat, 2, 2), dtype=jnp.int32)
    if debug_ponds_inputs:
        result["pond_diagnostics"] = jnp.full(
            (ncat, len(POND_DIAGNOSTIC_FIELDS)), jnp.nan, dtype=fields["aicen"].dtype)
        result["atmo_diagnostics"] = jnp.full(
            (ncat, 7, len(ATMO_DIAGNOSTIC_FIELDS)), jnp.nan, dtype=fields["aicen"].dtype)
        result["picard_diagnostics"] = jnp.full(
            (ncat, 8 + 4 * fields["zqin"].shape[0] + 3 * fields["zqsn"].shape[0],
             len(PICARD_DIAGNOSTIC_FIELDS)), jnp.nan, dtype=fields["aicen"].dtype)
    nslyr = fields["zqsn"].shape[0]
    zero = jnp.zeros_like(fields["aice"])
    puny = parameters["puny"]
    result["massicen"] = jnp.zeros_like(fields["smicen"])
    result["massliqn"] = jnp.zeros_like(fields["smliqn"])
    result["fsloss"] = (fields["fsnow"] * (1.0 - fields["aice"]))
    apond = (fields["apnd"] * fields["alvl"]
             if options["tr_pond_lvl"] else fields["apnd"])
    if options["snwredist"] == "bulk":
        worka = (jnp.sum(fields["alvl"] * fields["aicen"])
                 * parameters["snwlvlfac"] /
                 (1.0 + parameters["snwlvlfac"])
                 / jnp.where(fields["aice"] > puny,
                             fields["aice"], 1.0))
        worka = jnp.where(fields["aice"] > puny, worka, 0.0)
        result["fsloss"] = result["fsloss"] + result["fsnow"] * worka
        result["fsnow"] = result["fsnow"] * (1.0 - worka)
    if bool(options["formdrag"]):
        drag = neutral_drag_coeffs(
            apond, result["alvl"], result["vlvl"],
            result["aice"], result["vice"], result["vsno"],
            result["aicen"], result["vicen"],
            bool(options["tr_pond"]), parameters)
        result.update(drag)
    Tbot, fbot, rsiden, wlat = frzmlt_bottom_lateral(
        dt, result["aice"], result["frzmlt"],
        result["vicen"], result["vsnon"],
        result["zqin"], result["zqsn"], result["sst"],
        result["Tf"], result["strocnxT"], result["strocnyT"],
        result["Cdn_ocn"], result["aicen"],
        options["fbot_xfer_type"], bool(options["tr_fsd"]),
        parameters, afsdn=result["afsdn"],
        floe_rad_c=floe_rad_c, floe_binwidth=floe_binwidth)
    result.update({"Tbot": Tbot, "fbot": fbot,
                   "rsiden": rsiden, "wlat": wlat})
    failed = jnp.asarray(False)
    aerosol_warning = jnp.asarray(False)

    def advance_category(n, carry):
        result, failed, aerosol_warning = carry
        result = dict(result)
        active = result["aicen_init"][n] > puny
        cat_zero = jnp.zeros_like(result["aicen"][n])
        cat = {
            "aicen": result["aicen"][n],
            "vicen": result["vicen"][n],
            "vsnon": result["vsnon"][n],
            "Tsf": result["Tsfc"][n],
            "zqsn": result["zqsn"][:, n],
            "zqin": result["zqin"][:, n],
            "zSin": result["zSin"][:, n],
            "apond": apond[n],
            "apnd": result["apnd"][n],
            "hpond": result["hpnd"][n],
            "rsnw": result["rsnwn"][:, n],
            "smice": result["smicen"][:, n],
            "smliq": result["smliqn"][:, n],
            "massice": result["massicen"][:, n],
            "massliq": result["massliqn"][:, n],
            "rhoa": result["rhoa"], "flw": result["flw"],
            "potT": result["potT"], "Qa": result["Qa"],
            "fswsfc": result["fswsfcn"][n],
            "fswint": result["fswintn"][n],
            "Sswabs": result["Sswabsn"][:, n],
            "Iswabs": result["Iswabsn"][:, n],
            "Tbot": result["Tbot"], "Tsnice": result["Tsnice"],
            "sss": result["sss"], "sst": result["sst"],
            "fsnow": result["fsnow"], "fpond": result["fpond"],
            "fbot": result["fbot"], "frain": result["frain"],
            "mlt_onset": result["mlt_onset"],
            "frz_onset": result["frz_onset"],
            "fsensn": result["fsensn"][n],
            "flatn": result["flatn"][n],
            "fsurfn": result["fsurfn"][n],
            "fcondtopn": result["fcondtopn"][n],
            "fcondbotn": result["fcondbotn"][n],
            "flwoutn": cat_zero,
            "dpnd_flush": result["dpnd_flushn"][n],
            "dpnd_expon": result["dpnd_exponn"][n],
            "iage": result["iage"][n], "FY": result["FY"][n],
            "strairxn": cat_zero, "strairyn": cat_zero,
            "Cdn_atm_ratio_n": cat_zero,
            "Trefn": cat_zero, "Qrefn": cat_zero,
            "Urefn": cat_zero, "shcoef": cat_zero,
            "lhcoef": cat_zero,
            "evapn": cat_zero, "evapsn": cat_zero,
            "evapin": cat_zero, "freshn": cat_zero,
            "fsaltn": cat_zero, "fhocnn": cat_zero,
            "fadvocn": cat_zero, "meltt": cat_zero,
            "melts": cat_zero, "meltsliq": cat_zero,
            "meltb": cat_zero, "congel": cat_zero,
            "snoice": cat_zero, "dsnow": cat_zero,
            "hin": cat_zero, "hilyr": cat_zero,
            "hsn": cat_zero, "hslyr": cat_zero,
            "hsn_new": cat_zero, "einit": cat_zero,
            "einter": cat_zero, "efinal": cat_zero,
            "aerosno": result["aerosno"][:, :, n],
            "aeroice": result["aeroice"][:, :, n],
            "isosno": result["isosno"][:, n],
            "isoice": result["isoice"][:, n],
            "faero_ocn": result["faero_ocn"],
            "fiso_ocnn": jnp.zeros_like(result["fiso_ocn"]),
            "fiso_evapn": jnp.zeros_like(result["fiso_evap"]),
            "Qrefn_iso": jnp.zeros_like(result["Qref_iso"]),
            "aerosol_warning": jnp.asarray(False),
        }
        if profile_picard:
            cat["picard_iterations"] = jnp.zeros((2, 2), dtype=jnp.int32)
        if debug_ponds_inputs:
            cat["snow_diagnostics"] = jnp.full(
                (len(SNOW_DIAGNOSTIC_FIELDS),), jnp.nan, dtype=cat_zero.dtype)
            cat["surface_diagnostics"] = jnp.full(
                (len(SURFACE_DIAGNOSTIC_FIELDS),), jnp.nan, dtype=cat_zero.dtype)
            cat["atmo_diagnostics"] = jnp.full(
                (7, len(ATMO_DIAGNOSTIC_FIELDS)), jnp.nan, dtype=cat_zero.dtype)
            cat["picard_diagnostics"] = jnp.full(
                (8 + 4 * cat["zqin"].shape[0] + 3 * cat["zqsn"].shape[0],
                 len(PICARD_DIAGNOSTIC_FIELDS)), jnp.nan, dtype=cat_zero.dtype)

        def active_column(cat):
            if bool(options["calc_Tsfc"]) or bool(options["calc_strair"]):
                boundary_result = icepack_atm_boundary(
                     "ice", cat["Tsf"], cat["potT"],
                     result["uatm"], result["vatm"], result["wind"],
                     result["zlvl"], cat["Qa"], cat["rhoa"],
                     cat["strairxn"], cat["strairyn"],
                     cat["Trefn"], cat["Qrefn"],
                     result["Cdn_atm"], options, parameters,
                     uvel=result["uvel"], vvel=result["vvel"],
                     zlvs=result["zlvs"], Qa_iso=result["Qa_iso"],
                     Qref_iso=cat["Qrefn_iso"],
                     debug_diagnostics=debug_ponds_inputs,
                     debug_psi_replay=(result["debug_psi_replay"][n]
                                       if options.get("debug_replay_atmo_psi", False) else None),
                     debug_psi_values=(result["debug_psi_values"][n]
                                       if options.get("debug_replay_atmo_psi", False) else None),
                     debug_math_replay=((result["debug_exp_replay"][n],
                                         result["debug_atan_replay"][n],
                                         result["debug_math_values"][n])
                                        if options.get("debug_replay_atmo_math", False) else None))
                if debug_ponds_inputs:
                    cat["atmo_diagnostics"] = boundary_result[-1]
                    boundary_result = boundary_result[:-1]
                (strairxn, strairyn, Trefn, Qrefn, _, _,
                 lhcoef, shcoef, _, Cdn_atm_ratio_n,
                 Urefn, Qrefn_iso) = boundary_result
                cat.update({"strairxn": strairxn,
                            "strairyn": strairyn, "Trefn": Trefn,
                            "Qrefn": Qrefn, "lhcoef": lhcoef,
                            "shcoef": shcoef,
                            "Cdn_atm_ratio_n": Cdn_atm_ratio_n,
                            "Urefn": Urefn, "Qrefn_iso": Qrefn_iso})
            if not bool(options["calc_strair"]):
                cat["strairxn"] = result["strax"]
                cat["strairyn"] = result["stray"]
            if bool(options["tr_iage"]):
                cat["iage"] = increment_age(dt, cat["iage"])
            if bool(options["tr_FY"]):
                cat["FY"] = update_FYarea(
                    dt, result["lmask_n"], result["lmask_s"],
                    yday, cat["FY"], parameters["secday"])
            if not bool(options["calc_Tsfc"]):
                (cat["flatn"], cat["fsensn"], cat["fsurfn"],
                 cat["fcondtopn"]) = set_sfcflux(
                     cat["aicen"], result["flatn_f"][n],
                     result["fsensn_f"][n],
                     result["fsurfn_f"][n],
                     result["fcondtopn_f"][n])
            if bool(options["semi_implicit_Tsfc"]):
                coupled = {
                    "dfsurfdTs_cpl": result["dfsurfdT"][n],
                    "dflatdTs_cpl": result["dflatdT"][n],
                    "fsurf_cpl": result["fsurfn_f"][n],
                    "flat_cpl": result["flatn_f"][n],
                }
            else:
                coupled = None
            if bool(options.get("debug_replay_atmo_coefficients", False)):
                # Replace only the two heat-transfer inputs; retain original atmo diagnostics.
                cat["shcoef"] = jnp.where(
                    result["debug_atmo_replay"][n], result["debug_shcoef"][n], cat["shcoef"])
                cat["lhcoef"] = jnp.where(
                    result["debug_atmo_replay"][n], result["debug_lhcoef"][n], cat["lhcoef"])
            cat, therm_failed, _ = thermo_vertical(
                dt, yday, cat, options, parameters, coupled=coupled,
                picard_active=active & result.get("picard_active", jnp.asarray(True)))
            if options["tr_pond_lvl"]:
                cat["apnd"] = jnp.where(
                    result["alvl"][n] > puny,
                    jnp.maximum(cat["apond"] /
                                jnp.where(result["alvl"][n] > puny,
                                          result["alvl"][n], 1.0), 1.0),
                    0.0)
            else:
                cat["apnd"] = cat["apond"]
            if bool(options["tr_aero"]):
                (cat["aerosno"], cat["aeroice"], cat["faero_ocn"],
                 conservation_error, negative_first) = update_aerosol(
                     dt, cat["meltt"], cat["melts"], cat["meltb"],
                     cat["congel"], cat["snoice"], cat["fsnow"],
                     cat["aerosno"], cat["aeroice"],
                     result["aicen_init"][n],
                     result["vicen_init"][n],
                     result["vsnon_init"][n], cat["vicen"],
                     cat["vsnon"], cat["aicen"], result["faero_atm"],
                     cat["faero_ocn"], options, parameters)
                cat["aerosol_warning"] = (jnp.any(conservation_error)
                                          | negative_first)
            if bool(options["tr_iso"]):
                (cat["isosno"], cat["isoice"], cat["fiso_ocnn"],
                 cat["fiso_evapn"]) = update_isotope(
                     dt, cat["meltt"], cat["melts"], cat["meltb"],
                     cat["congel"], cat["snoice"], cat["evapn"],
                     cat["fsnow"], cat["Tsf"], cat["Qrefn_iso"],
                     cat["isosno"], cat["isoice"],
                     result["aicen_init"][n],
                     result["vicen_init"][n],
                     result["vsnon_init"][n], cat["vicen"],
                     cat["vsnon"], cat["aicen"],
                     result["fiso_atm"], cat["fiso_evapn"],
                     cat["fiso_ocnn"], result["HDO_ocn"],
                     result["H2_16O_ocn"], result["H2_18O_ocn"],
                     options, parameters)
            return cat, therm_failed

        cat, cat_failed = lax.cond(
            active, active_column,
            lambda cat: (cat, jnp.asarray(False)), cat)
        if profile_picard:
            result["picard_iterations"] = result["picard_iterations"].at[n].set(
                cat["picard_iterations"])
        if debug_ponds_inputs:
            result["atmo_diagnostics"] = result["atmo_diagnostics"].at[n].set(cat["atmo_diagnostics"])
            result["picard_diagnostics"] = result["picard_diagnostics"].at[n].set(cat["picard_diagnostics"])
        failed = failed | cat_failed
        aerosol_warning = aerosol_warning | cat["aerosol_warning"]
        result["fpond"] = cat["fpond"]

        if bool(options["snwgrain"]):
            massliq, meltsliq = drain_snow(
                cat["vsnon"], cat["aicen"],
                cat["massice"], cat["massliq"], cat["meltsliq"],
                bool(options["use_smliq_pnd"]), parameters)
            cat["massliq"] = massliq
            cat["meltsliq"] = meltsliq

        if bool(options["tr_pond_lvl"]):
            rfrac = (parameters["rfracmin"]
                     + (parameters["rfracmax"] - parameters["rfracmin"])
                     * cat["aicen"])
            if debug_ponds_inputs:
                pond_inputs = (
                    cat["aicen"], result["alvl"][n], fields["apnd"][n], fields["hpnd"][n],
                    cat["apond"], cat["apnd"], cat["hpond"], cat["meltt"], cat["melts"],
                    cat["meltsliq"], cat["frain"], rfrac, result["Tair"], cat["fsurfn"],
                )
                pond_entry = (
                    result["ipnd"][n], cat["vicen"], cat["vsnon"], result["dhsn"][n],
                    cat["Tsf"], cat["fcondtopn"], cat["fsurfn"] - cat["fcondtopn"],
                    (cat["fsurfn"] - cat["fcondtopn"]) * dt, dt,
                )
            (cat["apnd"], cat["hpond"], ipnd, ffrac,
             freebd, initial, dlid, flush) = compute_ponds_lvl(
                 dt, rfrac, cat["meltt"], cat["melts"],
                 cat["frain"], result["Tair"], cat["fsurfn"],
                 result["dhsn"][n], cat["aicen"], cat["vicen"],
                 cat["vsnon"], cat["zqin"], cat["zSin"],
                 cat["Tsf"], result["alvl"][n], cat["apnd"],
                 cat["hpond"], result["ipnd"][n], cat["meltsliq"],
                 result["dpnd_freebdn"][n],
                 result["dpnd_initialn"][n],
                 result["dpnd_dlidn"][n], cat["dpnd_flush"],
                 options, parameters)
            result["ipnd"] = result["ipnd"].at[n].set(ipnd)
            result["ffracn"] = result["ffracn"].at[n].set(ffrac)
            result["dpnd_freebdn"] = result["dpnd_freebdn"].at[n].set(freebd)
            result["dpnd_initialn"] = result["dpnd_initialn"].at[n].set(initial)
            result["dpnd_dlidn"] = result["dpnd_dlidn"].at[n].set(dlid)
            cat["dpnd_flush"] = flush
            if debug_ponds_inputs:
                result["pond_diagnostics"] = result["pond_diagnostics"].at[n].set(jnp.concatenate((
                    jnp.stack(pond_inputs + (freebd, dlid, cat["apnd"], cat["hpond"], ipnd)
                              + pond_entry),
                    cat["snow_diagnostics"], cat["surface_diagnostics"])))
        elif bool(options["tr_pond_sealvl"]):
            (cat["apnd"], cat["hpond"], ipnd, ffrac,
             freebd, dlid, flush) = compute_ponds_sealvl(
                 dt, cat["meltt"], cat["melts"], cat["frain"],
                 result["Tair"], cat["fsurfn"], cat["Tsf"],
                 result["dhsn"][n], cat["aicen"], cat["vicen"],
                 cat["vsnon"], cat["zqin"], cat["zSin"],
                 cat["apnd"], cat["hpond"], result["ipnd"][n],
                 cat["meltsliq"], result["dpnd_freebdn"][n],
                 result["dpnd_dlidn"][n], cat["dpnd_flush"],
                 options, parameters)
            result["ipnd"] = result["ipnd"].at[n].set(ipnd)
            result["ffracn"] = result["ffracn"].at[n].set(ffrac)
            result["dpnd_freebdn"] = result["dpnd_freebdn"].at[n].set(freebd)
            result["dpnd_dlidn"] = result["dpnd_dlidn"].at[n].set(dlid)
            cat["dpnd_flush"] = flush
        elif bool(options["tr_pond_topo"]):
            rfrac = (parameters["rfracmin"]
                     + (parameters["rfracmax"] - parameters["rfracmin"])
                     * cat["aicen"])
            if bool(options["snwgrain"]) and bool(options["use_smliq_pnd"]):
                pond = (rfrac / parameters["rhofresh"]
                        * (cat["meltt"] * parameters["rhoi"]
                           + cat["meltsliq"]))
            else:
                pond = (rfrac / parameters["rhofresh"]
                        * (cat["meltt"] * parameters["rhoi"]
                           + cat["melts"] * parameters["rhos"]
                           + cat["frain"] * dt))
            topopond = active
            old_apnd = cat["apnd"]
            new_apnd = jnp.where(old_apnd < puny, 1.0, old_apnd)
            new_hpnd = jnp.where(old_apnd < puny, 0.0, cat["hpond"])
            cat["hpond"] = jnp.where(
                topopond, (pond + new_hpnd * new_apnd) / new_apnd,
                cat["hpond"])
            cat["apnd"] = jnp.where(topopond, new_apnd, cat["apnd"])
            result["fpond"] = result["fpond"] + jnp.where(
                topopond, pond * cat["aicen"], 0.0)

        for name, source in (
            ("aicen", "aicen"), ("vicen", "vicen"),
            ("vsnon", "vsnon"), ("Tsfc", "Tsf"),
            ("apnd", "apnd"), ("hpnd", "hpond"),
            ("iage", "iage"), ("FY", "FY"),
            ("fswsfcn", "fswsfc"), ("fswintn", "fswint"),
            ("fsensn", "fsensn"), ("flatn", "flatn"),
            ("fsurfn", "fsurfn"), ("fcondtopn", "fcondtopn"),
            ("fcondbotn", "fcondbotn"),
            ("melttn", "meltt"), ("meltsn", "melts"),
            ("meltbn", "meltb"), ("congeln", "congel"),
            ("snoicen", "snoice"), ("dsnown", "dsnow"),
            ("meltsliqn", "meltsliq"),
            ("dpnd_flushn", "dpnd_flush"),
            ("dpnd_exponn", "dpnd_expon"),
        ):
            result[name] = result[name].at[n].set(cat[source])
        for name, source in (
            ("zqsn", "zqsn"), ("zqin", "zqin"),
            ("zSin", "zSin"), ("Sswabsn", "Sswabs"),
            ("Iswabsn", "Iswabs"), ("rsnwn", "rsnw"),
            ("smicen", "smice"), ("smliqn", "smliq"),
            ("massicen", "massice"), ("massliqn", "massliq"),
            ("isosno", "isosno"), ("isoice", "isoice"),
        ):
            result[name] = result[name].at[:, n].set(cat[source])
        result["aerosno"] = result["aerosno"].at[:, :, n].set(cat["aerosno"])
        result["aeroice"] = result["aeroice"].at[:, :, n].set(cat["aeroice"])
        for name in ("Tsnice", "mlt_onset", "frz_onset"):
            result[name] = cat[name]
        result["faero_ocn"] = cat["faero_ocn"]

        if bool(options["snwgrain"]):
            snow_present = cat["vsnon"] > puny
            mass_factor = (nslyr * cat["aicen"] /
                           jnp.where(snow_present, cat["vsnon"], 1.0))
            result["smicen"] = result["smicen"].at[:, n].set(
                jnp.where(snow_present, cat["massice"] * mass_factor,
                          parameters["rhos"]))
            result["smliqn"] = result["smliqn"].at[:, n].set(
                jnp.where(snow_present, cat["massliq"] * mass_factor, 0.0))

        def merge_category(total):
            category = {
                "strairxn": cat["strairxn"],
                "strairyn": cat["strairyn"],
                "Cdn_atm_ratio_n": cat["Cdn_atm_ratio_n"],
                "fsurfn": cat["fsurfn"],
                "fcondtopn": cat["fcondtopn"],
                "fcondbotn": cat["fcondbotn"],
                "fsensn": cat["fsensn"], "flatn": cat["flatn"],
                "fswabsn": cat["fswsfc"] + cat["fswint"]
                             + result["fswthrun"][n],
                "flwoutn": cat["flwoutn"],
                "evapn": cat["evapn"], "evapsn": cat["evapsn"],
                "evapin": cat["evapin"],
                "Trefn": cat["Trefn"], "Qrefn": cat["Qrefn"],
                "Urefn": cat["Urefn"],
                "freshn": cat["freshn"],
                "fsaltn": cat["fsaltn"],
                "fhocnn": cat["fhocnn"],
                "fswthrun": result["fswthrun"][n],
                "melttn": cat["meltt"], "meltsn": cat["melts"],
                "meltbn": cat["meltb"],
                "congeln": cat["congel"],
                "snoicen": cat["snoice"],
                "dsnown": cat["dsnow"],
                "meltsliqn": cat["meltsliq"],
                "dpnd_flushn": cat["dpnd_flush"],
                "dpnd_exponn": cat["dpnd_expon"],
                "dpnd_freebdn": total["dpnd_freebdn"][n],
                "dpnd_initialn": total["dpnd_initialn"][n],
                "dpnd_dlidn": total["dpnd_dlidn"][n],
                "Qrefn_iso": cat["Qrefn_iso"],
                "fiso_ocnn": cat["fiso_ocnn"],
                "fiso_evapn": cat["fiso_evapn"],
            }
            for band in ("vdr", "vdf", "idr", "idf",
                         "uvrdr", "uvrdf", "pardr", "pardf"):
                key = "fswthrun_" + band
                category[key] = total[key][n]
            return merge_fluxes(
                result["aicen_init"][n], category, total,
                options, bool(options["snwgrain"]),
                parameters["emissivity"], result["flw"])

        result = lax.cond(active, merge_category,
                          lambda total: total, result)
        if bool(options["send_i2x_per_cat"]):
            result["fswthrun_ai"] = result["fswthrun_ai"].at[n].set(
                result["fswthrun"][n] * result["aicen_init"][n])
        return result, failed, aerosol_warning

    # Categories share pond/onset state and accumulated fluxes. Preserve their
    # update order while keeping the entire sweep on the device.
    result, failed, aerosol_warning = lax.fori_loop(
        0, ncat, advance_category, (result, failed, aerosol_warning))

    if bool(options["tr_pond_topo"]):
        (result["apnd"], result["hpnd"], result["ipnd"],
         result["fpond"], topo_failed) = compute_ponds_topo(
             dt, int(options["ktherm"]), result["aice"],
             result["aicen"], result["vice"], result["vicen"],
             result["vsno"], result["vsnon"], result["meltt"],
             result["fsurf"], result["fpond"], result["Tsfc"],
             result["Tf"], result["zqin"], result["zSin"],
             result["apnd"], result["hpnd"], result["ipnd"],
             options, parameters)
        failed = failed | topo_failed
    return result, failed, aerosol_warning
