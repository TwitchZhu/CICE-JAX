"""Topographic melt ponds from icepack_meltpond_topo.F90."""

import jax.numpy as jnp
from jax import lax

from icepack.columnphysics.icepack_therm_shared import (
    calculate_Tin_from_qin_jax,
)


def permeability_phi(qicen, sicen, options, parameters):
    """Permeability of the least permeable ice layer below a pond."""
    Tmlt = -sicen * parameters["depressT"]
    Tin, _ = calculate_Tin_from_qin_jax(
        qicen, Tmlt, bool(options["l_brine"]), parameters)
    cold = jnp.max(Tin) <= -2.0
    Sbr_cold = (-1.2 - 21.8 * Tin - 0.919 * Tin**2
                - 0.01878 * Tin**3)
    Sbr_warm = -17.6 * Tin - 0.389 * Tin**2 - 0.00362 * Tin**3
    Sbr = jnp.where(cold, Sbr_cold, Sbr_warm)
    zero_salinity = jnp.any(Sbr == 0.0)
    phi = sicen / jnp.where(Sbr != 0.0, Sbr, 1.0)
    return 3.0e-8 * jnp.min(phi)**3, zero_salinity


def calc_hpond(aicen, asnon, hsnon, alfan, volp,
               cum_max_vol, parameters):
    """Find the highest partly filled topographic layer and pond surface."""
    ncat = aicen.shape[0]

    def filled(_):
        m_index = jnp.asarray(0)
        chosen = jnp.asarray(False)
        rem_vol = volp
        for n in range(ncat):
            select = ~chosen & (volp <= cum_max_vol[n])
            m_index = jnp.where(select, n + 1, m_index)
            prior = cum_max_vol[n - 1] if n > 0 else 0.0
            rem_vol = jnp.where(select, volp - prior, rem_vol)
            chosen = chosen | select
        m_index = jnp.minimum(ncat - 1, m_index)
        hitl = jnp.zeros((ncat + 2,), dtype=aicen.dtype)
        aicetl = jnp.zeros_like(hitl)
        layer_top = alfan[jnp.minimum(m_index, ncat - 1)]
        layer_base = alfan[jnp.maximum(m_index - 1, 0)]
        for n in range(ncat):
            in_layer = n < m_index
            snow_height = jnp.maximum(
                jnp.minimum(hsnon[n] + alfan[n] - layer_base,
                            layer_top - layer_base), 0.0)
            hitl = hitl.at[n + 1].set(jnp.where(in_layer,
                                                   snow_height, 0.0))
            aicetl = aicetl.at[n + 1].set(jnp.where(in_layer,
                                                      asnon[n], 0.0))
            aicetl = aicetl.at[0].add(jnp.where(
                in_layer, aicen[n] - asnon[n], 0.0))
        hitl = hitl.at[m_index + 1].set(layer_top - layer_base)
        # Fortran uses a short bubble sort; the order of equal snow heights
        # matters for its subsequent accumulation, so keep the same swaps.
        for ns in range(1, ncat + 1):
            for n in range(ncat + 1 - ns):
                swap = (n <= m_index - ns + 1) & (hitl[n] > hitl[n + 1])
                old_h, next_h = hitl[n], hitl[n + 1]
                old_a, next_a = aicetl[n], aicetl[n + 1]
                hitl = hitl.at[n].set(jnp.where(swap, next_h, old_h))
                hitl = hitl.at[n + 1].set(jnp.where(swap, old_h, next_h))
                aicetl = aicetl.at[n].set(jnp.where(swap, next_a, old_a))
                aicetl = aicetl.at[n + 1].set(jnp.where(swap, old_a, next_a))
        hpond = jnp.zeros_like(volp)
        found = jnp.asarray(False)
        for n in range(1, ncat + 2):
            snow_area = jnp.sum(aicetl[n:])
            area = jnp.sum(aicetl) - (parameters["rhos"] /
                                           parameters["rhow"]) * snow_area
            volume = (hitl[n] - hitl[n - 1]) * area
            choose = (n <= m_index + 1) & ~found & (volume >= rem_vol)
            candidate = (rem_vol / jnp.where(area != 0.0, area, 1.0)
                         + hitl[n - 1] + layer_base - alfan[0])
            hpond = jnp.where(choose, candidate, hpond)
            rem_vol = jnp.where((n <= m_index + 1) & ~choose,
                                rem_vol - volume, rem_vol)
            found = found | choose
        return hpond, m_index

    return lax.cond(volp < parameters["puny"],
                    lambda _: (jnp.zeros_like(volp), jnp.asarray(0)),
                    filled, operand=None)


def pond_area(dt, ktherm, aice, vice, vsno, aicen, vicen,
              vsnon, qicen, sicen, volpn, volp, options,
              parameters):
    """Fill topographic depressions and drain them through permeable ice."""
    ncat = aicen.shape[0]
    puny = parameters["puny"]
    active = aicen >= puny
    safe_aicen = jnp.where(active, aicen, 1.0)
    hicen = jnp.where(active, vicen / safe_aicen, 0.0)
    hsnon = jnp.where(active, vsnon / safe_aicen, 0.0)
    reduced_aicen = jnp.where(
        active, aicen * jnp.maximum(0.2, -0.024 * hicen + 0.832), 0.0)
    reduced_aicen = reduced_aicen.at[-1].set(jnp.where(active[-1], 1.0, 0.0))
    asnon = reduced_aicen
    alfan = 0.6 * hicen
    betan = 0.4 * hicen
    cum_max = jnp.zeros((ncat,), dtype=aicen.dtype)
    negative_volume = jnp.asarray(False)
    bad_salinity = jnp.asarray(False)
    for n in range(ncat - 1):
        new_level = ((alfan[n + 1] >= alfan[n])
                     & (alfan[n + 1] > 0.0))
        level_depth = alfan[n + 1] - alfan[n]
        volume = (cum_max[n - 1] if n > 0 else 0.0)
        volume = volume + level_depth * jnp.sum(reduced_aicen[:n + 1])
        for ns in range(n + 1):
            snow_depth = jnp.maximum(
                jnp.minimum(hsnon[ns] + alfan[ns] - alfan[n],
                            level_depth), 0.0)
            volume = volume - (parameters["rhos"] /
                               parameters["rhow"]) * asnon[ns] * snow_depth
        old_volume = cum_max[n - 1] if n > 0 else 0.0
        volume = jnp.where(new_level, volume, old_volume)
        negative_volume = negative_volume | (volume < 0.0)
        cum_max = cum_max.at[n].set(volume)
    cum_max = cum_max.at[-1].set(cum_max[-2] if ncat > 1 else 0.0)
    excess = volp >= cum_max[-1]
    drain = jnp.where(excess, volp - cum_max[-1] + puny, 0.0)
    volp = volp - drain
    dvolp = drain
    tiny = excess & (volp < puny)
    dvolp = dvolp + jnp.where(tiny, volp, 0.0)
    volp = jnp.where(tiny, 0.0, volp)
    hpond, m_index = calc_hpond(
        reduced_aicen, asnon, hsnon, alfan, volp, cum_max, parameters)
    apondn = jnp.zeros_like(aicen)
    hpondn = jnp.zeros_like(aicen)
    for n in range(ncat):
        covered = n < m_index
        hpondn = hpondn.at[n].set(jnp.where(
            covered, jnp.maximum(hpond - alfan[n] + alfan[0], 0.0), 0.0))
        apondn = apondn.at[n].set(jnp.where(
            covered, reduced_aicen[n], 0.0))

    floe_weight = ((vsno * parameters["rhos"]
                    + parameters["rhoi"] * vice
                    + parameters["rhow"] * volp) / aice)
    hsl_rel = (floe_weight / parameters["rhow"]
               - (jnp.sum(betan * aicen) / aice + alfan[0]))
    pressure_head = (parameters["gravit"] * parameters["rhow"]
                     * jnp.maximum(hpond - hsl_rel, 0.0))

    if ktherm != 2:
        permflag = jnp.asarray(False)
        for n in range(ncat - 1):
            should_drain = (pressure_head > 0.0) & (hicen[n] > 0.0)
            def drain_category(state):
                volp, dvolp, permflag, zero_salinity = state
                perm, bad = permeability_phi(
                    qicen[:, n], sicen[:, n], options, parameters)
                flow = (perm * apondn[n] * pressure_head * dt /
                        (parameters["viscosity_dyn"] * hicen[n]))
                dvolp = dvolp + jnp.minimum(flow, volp)
                volp = jnp.maximum(volp - flow, 0.0)
                tiny = volp < puny
                dvolp = dvolp + jnp.where(tiny, volp, 0.0)
                volp = jnp.where(tiny, 0.0, volp)
                return volp, dvolp, permflag | (perm > 0.0), (
                    zero_salinity | bad)
            volp, dvolp, permflag, bad_salinity = lax.cond(
                should_drain, drain_category,
                lambda state: state,
                (volp, dvolp, permflag, bad_salinity))
        def recompute(_):
            new_height, new_index = calc_hpond(
                reduced_aicen, asnon, hsnon, alfan, volp,
                cum_max, parameters)
            new_area = apondn
            new_depth = hpondn
            for n in range(ncat):
                covered = n < new_index
                new_depth = new_depth.at[n].set(jnp.where(
                    covered, new_height - alfan[n] + alfan[0],
                    new_depth[n]))
                new_area = new_area.at[n].set(jnp.where(
                    covered, reduced_aicen[n], new_area[n]))
            return new_area, new_depth, new_index
        apondn, hpondn, m_index = lax.cond(
            permflag, recompute,
            lambda _: (apondn, hpondn, m_index), operand=None)
    else:
        bad_salinity = jnp.asarray(False)

    volpn = jnp.zeros_like(volpn)
    for n in range(ncat - 1):
        covered_lower = n < m_index - 1
        category_volume = (apondn[n] * hpondn[n]
                           - parameters["rhos"] / parameters["rhow"]
                           * asnon[n] * jnp.minimum(hsnon[n], hpondn[n]))
        volpn = volpn.at[n].set(jnp.where(
            covered_lower, category_volume, volpn[n]))
    lower_volume = jnp.zeros_like(volp)
    for n in range(ncat - 1):
        lower_volume = lower_volume + jnp.where(
            n < m_index - 1, volpn[n], 0.0)
    highest_volume = jnp.where(m_index == 1, volp,
                               jnp.maximum(volp - lower_volume, 0.0))
    for n in range(ncat):
        highest = n == m_index - 1
        volpn = volpn.at[n].set(jnp.where(
            highest, highest_volume, volpn[n]))
        empty_highest = highest & (m_index > 1) & (volp <= lower_volume)
        hpondn = hpondn.at[n].set(jnp.where(
            empty_highest, 0.0, hpondn[n]))
        apondn = apondn.at[n].set(jnp.where(
            empty_highest, 0.0, apondn[n]))
    for n in range(ncat):
        reduce_previous = (n == m_index - 2) & (volp + puny < lower_volume)
        volpn = volpn.at[n].set(jnp.where(
            reduce_previous, volpn[n] - lower_volume + volp, volpn[n]))
    for n in range(ncat):
        in_pond = n < m_index
        present = in_pond & (apondn[n] > puny)
        dvolp = dvolp + jnp.where(in_pond & ~present, volpn[n], 0.0)
        hpondn = hpondn.at[n].set(jnp.where(
            present, volpn[n] / jnp.where(present, apondn[n], 1.0), 0.0))
        volpn = volpn.at[n].set(jnp.where(present, volpn[n], 0.0))
        apondn = apondn.at[n].set(jnp.where(present, apondn[n], 0.0))
    return volpn, volp, apondn, hpondn, dvolp, (
        negative_volume | bad_salinity)


def compute_ponds_topo(dt, ktherm, aice, aicen, vice, vicen,
                       vsno, vsnon, meltt, fsurf, fpond, Tsfcn,
                       Tf, qicen, sicen, apnd, hpnd, ipnd,
                       options, parameters):
    """Redistribute topographic ponds and freeze or melt their ice lids."""
    _ = Tf  # The Fortran routine also receives, but does not use, Tf.
    puny = parameters["puny"]
    ncat = aicen.shape[0]
    volp = jnp.sum(hpnd * apnd * aicen)
    vuin = ipnd * apnd * aicen
    volpn = jnp.zeros_like(aicen)
    apondn = jnp.zeros_like(aicen)
    hpondn = jnp.zeros_like(aicen)
    Tp = parameters["Timelt"] - 0.15
    hi = jnp.where(aice > puny, vice /
                   jnp.where(aice > puny, aice, 1.0), 0.0)
    ponds_present = ((aice > 0.01) & (hi > 0.1)
                     & (volp > 1.0e-4 * aice))

    def advance(_):
        lid_volume = vuin
        volpn, volp, apondn, hpondn, drain, failed = pond_area(
            dt, ktherm, aice, vice, vsno, aicen, vicen,
            vsnon, qicen, sicen, jnp.zeros_like(aicen),
            volp, options, parameters)
        fpond_new = fpond - drain
        for n in range(ncat - 1):
            has_lid = lid_volume[n] > puny
            warm = Tsfcn[n] > Tp
            melt_volume = jnp.minimum(meltt * apondn[n], lid_volume[n])
            melting = has_lid & warm & (melt_volume > puny)
            lid = lid_volume[n] - jnp.where(melting, melt_volume, 0.0)
            water = volpn[n] + jnp.where(melting, melt_volume, 0.0)
            fpond_new = fpond_new + jnp.where(melting, melt_volume, 0.0)
            last_lid = melting & (lid < puny) & (water > puny)
            water = water + jnp.where(last_lid, lid, 0.0)
            fpond_new = fpond_new + jnp.where(last_lid, lid, 0.0)
            lid = jnp.where(last_lid, 0.0, lid)
            has_water = volpn[n] > puny
            dTice = jnp.maximum(-Tsfcn[n] - 0.15, 0.0)
            omega = (parameters["kice"] * dTice /
                     (parameters["Lfresh"] * parameters["rhoi"]))
            safe_area = jnp.where(aicen[n] > 0.0, aicen[n], 1.0)
            dHui = (jnp.sqrt(2.0 * omega * dt
                             + (lid_volume[n] / safe_area)**2)
                    - lid_volume[n] / safe_area)
            freeze_volume = jnp.minimum(dHui * apondn[n], volpn[n])
            freezing_lid = has_lid & ~warm & has_water & (freeze_volume > puny)
            lid = lid + jnp.where(freezing_lid, freeze_volume, 0.0)
            water = water - jnp.where(freezing_lid, freeze_volume, 0.0)
            fpond_new = fpond_new - jnp.where(freezing_lid, freeze_volume, 0.0)
            first_lid = ~has_lid
            dHui = jnp.maximum(
                -fsurf * dt / (parameters["Lfresh"] * parameters["rhoi"]),
                0.0)
            new_volume = jnp.minimum(dHui * apondn[n], volpn[n])
            new_freeze = first_lid & (new_volume > puny)
            lid = jnp.where(new_freeze, new_volume, lid)
            water = water - jnp.where(new_freeze, new_volume, 0.0)
            fpond_new = fpond_new - jnp.where(new_freeze, new_volume, 0.0)
            lid_volume = lid_volume.at[n].set(lid)
            volpn = volpn.at[n].set(water)
            depth = water / jnp.where(apondn[n] > 0.0,
                                      apondn[n], 1.0)
            hpondn = hpondn.at[n].set(jnp.where(
                melting | freezing_lid | new_freeze, depth,
                hpondn[n]))
        return volpn, lid_volume, apondn, hpondn, fpond_new, failed

    def remove(_):
        return (jnp.zeros_like(volpn), jnp.zeros_like(vuin),
                apondn, hpondn, fpond - volp, jnp.asarray(False))

    volpn, vuin, apondn, hpondn, fpond, failed = lax.cond(
        ponds_present, advance, remove, operand=None)
    for n in range(ncat):
        lid_without_water = ((aicen[n] > puny) & (volpn[n] < puny)
                             & (vuin[n] > puny))
        vuin = vuin.at[n].set(jnp.where(lid_without_water,
                                        0.0, vuin[n]))
        lid_present = apondn[n] > puny
        ipnd = ipnd.at[n].set(jnp.where(
            lid_present, vuin[n] /
            jnp.where(lid_present, apondn[n], 1.0), 0.0))
        ice_present = aicen[n] > puny
        apnd = apnd.at[n].set(jnp.where(
            ice_present, apondn[n] /
            jnp.where(ice_present, aicen[n], 1.0), 0.0))
        hpnd = hpnd.at[n].set(jnp.where(ice_present, hpondn[n], 0.0))
        ipnd = ipnd.at[n].set(jnp.where(ice_present, ipnd[n], 0.0))
    return apnd, hpnd, ipnd, fpond, failed
