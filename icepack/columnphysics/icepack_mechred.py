"""Mechanical redistribution of sea ice, from icepack_mechred.F90.

The functions below operate on whole grids, with the ice category as the
last axis. They contain the ridging distribution and ice strength used by
both the momentum equation and the later category redistribution.
"""

from functools import partial

import jax
import jax.numpy as jnp

from icepack.columnphysics.icepack_itd import cleanup_itd_jax, column_sum


def asum_ridging(aicen, aice0):
    """Total area of ice and open water, which may exceed one after transport."""
    asum = aice0
    for n in range(aicen.shape[-1]):
        asum = asum + aicen[..., n]
    return asum


def ridge_prep(dt, hin_max, rdg_conv, rdg_shear, asum):
    """Compute net closing and opening rates as in Flato and Hibler (1995)."""
    # The final category must contain every thickness produced by ridging.
    hin_max = hin_max.at[-1].set(1.0e8)
    closing_net = 0.25 * rdg_shear + rdg_conv
    divu_adv = (1.0 - asum) / dt
    closing_net = jnp.where(divu_adv < 0.0,
                            jnp.maximum(closing_net, -divu_adv), closing_net)
    opning = closing_net + divu_adv
    return hin_max, closing_net, divu_adv, opning


def ridge_itd(aice0, aicen, vicen, krdg_partic, krdg_redist, mu_rdg, puny,
              mraft_init=None):
    """Ridging participation and redistribution for both CICE options.

    The leading category is open water. ``apartic[..., 0]`` therefore has
    the same meaning as the Fortran element apartic(0).
    """
    area = jnp.concatenate((jnp.where(aice0 > puny, aice0, 0.0)[..., None],
                            jnp.where(aicen > puny, aicen, 0.0)), axis=-1)
    cumulative = [area[..., 0]]
    for n in range(1, area.shape[-1]):
        cumulative.append(cumulative[-1] + area[..., n])
    gsum = jnp.stack(cumulative, axis=-1)
    work = 1.0 / jnp.where(gsum[..., -1:] > 0.0, gsum[..., -1:], 1.0)
    gsum = gsum * work
    gprev = jnp.concatenate((jnp.zeros_like(gsum[..., :1]), gsum[..., :-1]), axis=-1)

    if krdg_partic == 0:
        # Fortran Gstar = p15 = 0.15 (icepack_parameters.F90).
        gstar = 0.15
        apartic = jnp.where(
            gsum < gstar,
            (gsum - gprev) / gstar * (2.0 - (gprev + gsum) / gstar),
            jnp.where(gprev < gstar,
                      (gstar - gprev) / gstar * (2.0 - (gstar + gprev) / gstar),
                      0.0),
        )
    elif krdg_partic == 1:
        # Fortran astar = p05 = 0.05, not p5 = 0.5.
        astar = 0.05
        astari = 1.0 / astar
        scale = 1.0 / (1.0 - jnp.exp(-astari))
        apartic = (jnp.exp(-gprev * astari) * scale
                   - jnp.exp(-gsum * astari) * scale)
    else:
        raise ValueError("krdg_partic must be 0 or 1")

    active = aicen > puny
    if mraft_init is None:
        mraft_init = jnp.zeros_like(aicen)
    hi = jnp.where(active, vicen / jnp.where(active, aicen, 1.0), 0.0)
    hrmin = jnp.where(active, jnp.minimum(2.0 * hi, hi + 1.0), 0.0)
    if krdg_redist == 0:
        # Uniform ridged thickness distribution; Hstar = 25 m.
        hrmax = jnp.where(active, jnp.maximum(2.0 * jnp.sqrt(25.0 * hi),
                                               hrmin + puny), 0.0)
        hrexp = jnp.zeros_like(hi)
        krdg = jnp.where(active, 0.5 * (hrmin + hrmax) /
                          jnp.where(active, hi, 1.0), 1.0)
        mraft = mraft_init
    elif krdg_redist == 1:
        # Negative exponential ridged thickness distribution.
        hi = jnp.where(active, jnp.maximum(hi, puny), 0.0)
        hrmin = jnp.where(active, jnp.minimum(2.0 * hi, hi + 1.0), 0.0)
        hrexp = jnp.where(active, mu_rdg * jnp.sqrt(hi), 0.0)
        hrmax = jnp.zeros_like(hi)
        krdg = jnp.where(active, (hrmin + hrexp) /
                          jnp.where(active, hi, 1.0), 1.0)
        # Fortran uses SIGN(1,x), which is positive when x is zero.
        raft = (hi + 1.0 - hrmin) >= 0.0
        xtmp = jnp.where(raft, (2.0 * hi + hrexp) /
                         jnp.where(active, hi, 1.0) - krdg, 0.0)
        mraft = jnp.where(active,
                          jnp.where(raft & ((puny - jnp.abs(xtmp)) >= 0.0),
                                    1.0, 0.0), mraft_init)
    else:
        raise ValueError("krdg_redist must be 0 or 1")

    aksum = apartic[..., 0]
    for n in range(aicen.shape[-1]):
        aksum = aksum + apartic[..., n + 1] * (1.0 - 1.0 / krdg[..., n])
    return aksum, apartic, hrmin, hrmax, hrexp, krdg, mraft


def icepack_ice_strength(aice, vice, aice0, aicen, vicen, options, parameters):
    """Ice strength from Rothrock (1975) or Hibler (1979)."""
    if int(options["kstrength"]) != 1:
        return parameters["Pstar"] * vice * jnp.exp(-parameters["Cstar"] * (1.0 - aice))

    puny = parameters["puny"]
    aksum, apartic, hrmin, hrmax, hrexp, krdg, _ = ridge_itd(
        aice0, aicen, vicen, int(options["krdg_partic"]),
        int(options["krdg_redist"]), parameters["mu_rdg"], puny,
    )
    active = (aicen > puny) & (apartic[..., 1:] > 0.0)
    hi = jnp.where(active, vicen / jnp.where(active, aicen, 1.0), 0.0)
    if int(options["krdg_redist"]) == 0:
        h2rdg = (hrmax**3 - hrmin**3) / (
            3.0 * jnp.where(active, hrmax - hrmin, 1.0)
        )
    else:
        h2rdg = hrmin * hrmin + 2.0 * hrmin * hrexp + 2.0 * hrexp * hrexp
    dh2rdg = -hi * hi + h2rdg / krdg
    strength = column_sum(aicen.shape[-1],
                          jnp.where(active, apartic[..., 1:] * dh2rdg, 0.0))
    # Expanded EVP masks include neighboring open-water cells. Avoid a
    # 0/0 branch there so reverse differentiation stays finite.
    usable = jnp.abs(aksum) > puny
    return jnp.where(usable,
                     parameters["Cf"] * parameters["Cp"] * strength
                     / jnp.where(usable, aksum, 1.0), 0.0)


def ridge_shift(dt, hin_max, fields, layout, flags, options, parameters,
                aksum, apartic, hrmin, hrmax, hrexp, krdg,
                closing_net, opning, totals, bio_index=()):
    """Move ridging ice, snow and extensive tracers between categories."""
    aicen = fields["aicen"]
    vicen = fields["vicen"]
    vsnon = fields["vsnon"]
    trcrn = fields["trcrn"]
    aice0 = fields["aice0"]
    ncat = aicen.shape[0]
    ntrcr = trcrn.shape[0]
    puny = parameters["puny"]
    fsnowrdg = 0.5
    base = layout["trcr_base"]
    strata = layout["nt_strata"]
    strata_count = layout["n_trcr_strata"]
    atrcrn = []
    for it in range(ntrcr):
        amount = trcrn[it] * (base[it][0] * aicen
                               + base[it][1] * vicen
                               + base[it][2] * vsnon)
        for layer in range(strata_count[it]):
            amount = amount * trcrn[strata[it][layer]]
        atrcrn.append(amount)
    atrcrn = jnp.stack(atrcrn, axis=0)
    aicen_init, vicen_init, vsnon_init = aicen, vicen, vsnon

    # Reduce gross closing and opening by the same fraction whenever the
    # requested area loss exceeds open water or a source category.
    safe_aksum = jnp.where(aksum != 0.0, aksum, 1.0)
    closing_gross = closing_net / safe_aksum
    wk1 = apartic[0] * closing_gross * dt
    limit = (apartic[0] > 0.0) & (wk1 > aice0)
    tmpfac = jnp.where(limit, aice0 / jnp.where(limit, wk1, 1.0), 1.0)
    closing_gross = closing_gross * tmpfac
    opning = opning * tmpfac
    for n in range(ncat):
        wk1 = apartic[n + 1] * closing_gross * dt
        limit = ((aicen[n] > puny) & (apartic[n + 1] > 0.0)
                 & (wk1 > aicen[n]))
        tmpfac = jnp.where(limit,
                           aicen[n] / jnp.where(limit, wk1, 1.0), 1.0)
        closing_gross = closing_gross * tmpfac
        opning = opning * tmpfac
    aice0 = aice0 - apartic[0] * closing_gross * dt + opning * dt
    failed = aice0 < -puny
    aice0 = jnp.maximum(aice0, 0.0)
    aopen = opning * dt
    ardg1 = totals["ardg1"]
    ardg2 = totals["ardg2"]
    virdg = totals["virdg"]
    # Fortran initializes these once in ridge_ice, not on each iteration.
    ardg1n = totals["ardg1n"]
    ardg2n = totals["ardg2n"]
    virdgn = totals["virdgn"]
    msnow_mlt = totals["msnow_mlt"]
    esnow_mlt = totals["esnow_mlt"]
    mpond = totals["mpond"]
    maero = totals["maero"]
    miso = totals["miso"]
    mbio = totals["mbio"]
    aredistn = fields["aredistn"]
    vredistn = fields["vredistn"]

    for n in range(ncat):
        active = ((aicen_init[n] > puny) & (apartic[n + 1] > 0.0)
                  & (closing_gross > 0.0))
        requested = apartic[n + 1] * closing_gross * dt
        failed = failed | (active & (requested > aicen_init[n] + puny))
        loss = jnp.where(active, jnp.minimum(aicen_init[n], requested), 0.0)
        gain = loss / jnp.where(krdg[n] != 0.0, krdg[n], 1.0)
        afrac = loss / jnp.where(aicen_init[n] > 0.0, aicen_init[n], 1.0)
        ice_loss = vicen_init[n] * afrac
        snow_loss = vsnon_init[n] * afrac
        aicen = aicen.at[n].add(-loss)
        vicen = vicen.at[n].add(-ice_loss)
        vsnon = vsnon.at[n].add(-snow_loss)
        ardg1 = ardg1 + loss
        ardg2 = ardg2 + gain
        virdg = virdg + ice_loss
        ardg1n = ardg1n.at[n].set(jnp.where(active, loss, ardg1n[n]))
        ardg2n = ardg2n.at[n].set(jnp.where(active, gain, ardg2n[n]))
        virdgn = virdgn.at[n].set(jnp.where(active, ice_loss, virdgn[n]))
        msnow_mlt = msnow_mlt + parameters["rhos"] * snow_loss * (1.0 - fsnowrdg)
        for k in range(int(options["nslyr"])):
            esnow_mlt = (esnow_mlt + snow_loss
                          * trcrn[int(flags["nt_qsno"]) + k, n]
                          / int(options["nslyr"]) * (1.0 - fsnowrdg))
        if bool(flags["tr_aero"]):
            for it in range(int(flags["n_aero"])):
                slot = int(flags["nt_aero"]) + 4 * it
                maero = maero.at[it].add(
                    snow_loss * (1.0 - fsnowrdg)
                    * (trcrn[slot, n] + trcrn[slot + 1, n]))
        if bool(flags["tr_iso"]):
            for it in range(int(flags["n_iso"])):
                miso = miso.at[it].add(
                    snow_loss * (1.0 - fsnowrdg)
                    * (trcrn[int(flags["nt_isosno"]) + it, n]
                       + trcrn[int(flags["nt_isoice"]) + it, n]))
        if bool(options["z_tracers"]):
            dzssl = 0.5 / int(options["nslyr"])
            for it in range(int(flags["nbtrcr"])):
                slot = int(bio_index[it]) + int(options["nblyr"]) + 1
                mbio = mbio.at[it].add(
                    snow_loss * (1.0 - fsnowrdg)
                    * (trcrn[slot, n] * dzssl
                       + trcrn[slot + 1, n] * (1.0 - dzssl)))
        if bool(flags["tr_pond_lvl"]):
            mpond = (mpond + loss * trcrn[int(flags["nt_apnd"]), n]
                     * trcrn[int(flags["nt_hpnd"]), n]
                     * trcrn[int(flags["nt_alvl"]), n])
        elif bool(flags["tr_pond_topo"]) or bool(flags["tr_pond_sealvl"]):
            mpond = (mpond + loss * trcrn[int(flags["nt_apnd"]), n]
                     * trcrn[int(flags["nt_hpnd"]), n])

        for it in range(ntrcr):
            work = trcrn[it, n] * (
                base[it][0] * loss + base[it][1] * ice_loss
                + base[it][2] * snow_loss)
            for layer in range(strata_count[it]):
                work = work * trcrn[strata[it][layer], n]
            atrcrn = atrcrn.at[it, n].add(-work)

        fractions_area = []
        fractions_volume = []
        for nr in range(ncat):
            if int(options["krdg_redist"]) == 0:
                overlap = ((hrmin[n] < hin_max[nr + 1])
                           & (hrmax[n] > hin_max[nr]))
                hL = jnp.maximum(hrmin[n], hin_max[nr])
                hR = jnp.minimum(hrmax[n], hin_max[nr + 1])
                dhr = hrmax[n] - hrmin[n]
                dhr2 = hrmax[n] ** 2 - hrmin[n] ** 2
                farea = jnp.where(active & overlap,
                                  (hR - hL) / jnp.where(active & overlap, dhr, 1.0), 0.0)
                fvol = jnp.where(active & overlap,
                                 (hR ** 2 - hL ** 2)
                                 / jnp.where(active & overlap, dhr2, 1.0), 0.0)
            else:
                hi1 = hrmin[n]
                hexp = hrexp[n]
                hL = jnp.maximum(hi1, hin_max[nr])
                expL_arg = jnp.minimum((hL - hi1) /
                                       jnp.where(active, hexp, 1.0), 100.0)
                expL = jnp.exp(-expL_arg)
                if nr < ncat - 1:
                    hR = hin_max[nr + 1]
                    expR_arg = jnp.minimum((hR - hi1) /
                                           jnp.where(active, hexp, 1.0), 100.0)
                    expR = jnp.exp(-expR_arg)
                    farea = jnp.where(active & (hi1 < hR), expL - expR, 0.0)
                    fvol = jnp.where(active & (hi1 < hR),
                                     ((hL + hexp) * expL - (hR + hexp) * expR)
                                     / jnp.where(active, hi1 + hexp, 1.0), 0.0)
                else:
                    farea = jnp.where(active, expL, 0.0)
                    fvol = jnp.where(active,
                                     (hL + hexp) * expL
                                     / jnp.where(active, hi1 + hexp, 1.0), 0.0)
            fractions_area.append(farea)
            fractions_volume.append(fvol)
        farea = jnp.stack(fractions_area)
        fvol = jnp.stack(fractions_volume)
        if int(options["krdg_redist"]) == 1 and n == 0:
            aredistn = jnp.where(active, farea * gain, aredistn)
            vredistn = jnp.where(active, fvol * ice_loss, vredistn)
        aicen = aicen + farea * gain
        vicen = vicen + fvol * ice_loss
        vsnon = vsnon + fvol * snow_loss * fsnowrdg
        for it in range(ntrcr):
            if (bool(flags["tr_pond_lvl"])
                    and it in (int(flags["nt_alvl"]), int(flags["nt_vlvl"]))):
                continue
            work = (base[it][0] * gain * farea
                    + base[it][1] * ice_loss * fvol
                    + base[it][2] * snow_loss * fvol * fsnowrdg) * trcrn[it, n]
            for layer in range(strata_count[it]):
                if (bool(flags["tr_brine"])
                        and strata[it][layer] == int(flags["nt_fbri"])):
                    work = work * trcrn[strata[it][layer], n]
                else:
                    work = jnp.zeros_like(work)
            atrcrn = atrcrn.at[it].add(work)

    # Recover intensive tracers after all source categories have shifted.
    for nr in range(ncat):
        for it in range(ntrcr):
            amount = atrcrn[it, nr]
            if layout["trcr_depend"][it] == 0:
                value = jnp.where(aicen[nr] > puny,
                                  amount / jnp.where(aicen[nr] > puny,
                                                     aicen[nr], 1.0), 0.0)
                if it == int(flags["nt_Tsfc"]):
                    value = jnp.where(aicen[nr] > puny, value, fields["Tf"])
            else:
                value = jnp.zeros_like(amount)
                for carrier in (base[it][0] * aicen[nr],
                                base[it][1] * vicen[nr],
                                base[it][2] * vsnon[nr]):
                    value = jnp.where(carrier > 0.0,
                                      amount / jnp.where(carrier > 0.0,
                                                         carrier, 1.0), value)
                for layer in range(strata_count[it]):
                    parent = trcrn[strata[it][layer], nr]
                    value = jnp.where(parent > 0.0,
                                      value / jnp.where(parent > 0.0,
                                                        parent, 1.0), 0.0)
            trcrn = trcrn.at[it, nr].set(value)
        if bool(flags["tr_brine"]):
            slot = int(flags["nt_fbri"])
            trcrn = trcrn.at[slot, nr].set(jnp.where(
                vicen[nr] <= 0.0, 1.0, trcrn[slot, nr]))
        if str(options["snwredist"]).startswith("ITD"):
            for k in range(int(options["nslyr"])):
                slot = int(flags["nt_rhos"]) + k
                trcrn = trcrn.at[slot, nr].set(jnp.where(
                    vsnon[nr] <= 0.0, parameters["rhosnew"],
                    trcrn[slot, nr]))
        if bool(options["snwgrain"]):
            for k in range(int(options["nslyr"])):
                for slot, default in (
                    (int(flags["nt_rsnw"]) + k, parameters["rsnw_fall"]),
                    (int(flags["nt_smice"]) + k, parameters["rhos"]),
                ):
                    trcrn = trcrn.at[slot, nr].set(jnp.where(
                        vsnon[nr] <= 0.0, default, trcrn[slot, nr]))

    result = dict(fields)
    result.update({"aicen": aicen, "vicen": vicen, "vsnon": vsnon,
                   "trcrn": trcrn, "aice0": aice0,
                   "aredistn": aredistn, "vredistn": vredistn})
    totals = dict(totals)
    totals["ardg1"] = ardg1
    totals["ardg2"] = ardg2
    totals["virdg"] = virdg
    totals["aopen"] = aopen
    totals["ardg1n"] = ardg1n
    totals["ardg2n"] = ardg2n
    totals["virdgn"] = virdgn
    totals["msnow_mlt"] = msnow_mlt
    totals["esnow_mlt"] = esnow_mlt
    totals["maero"] = maero
    totals["miso"] = miso
    totals["mbio"] = mbio
    totals["mpond"] = mpond
    return result, totals, opning, failed


@partial(jax.jit, static_argnames=("layout", "flags", "options", "bio_index"))
def ridge_ice(dt, ndtd, hin_max, fields, layout, flags, options,
              parameters, bio_index=()):
    """Iterate ridging up to CICE's 20-iteration limit on one ocean cell."""
    layout = dict(layout)
    flags = dict(flags)
    options = dict(options)
    puny = parameters["puny"]
    ncat = fields["aicen"].shape[0]
    initial = fields
    asum = asum_ridging(fields["aicen"], fields["aice0"])
    hin_max, closing_net, _, opning = ridge_prep(
        dt, hin_max, fields["rdg_conv"], fields["rdg_shear"], asum)
    totals = {
        "ardg1": jnp.zeros_like(asum), "ardg2": jnp.zeros_like(asum),
        "virdg": jnp.zeros_like(asum), "aopen": jnp.zeros_like(asum),
        "ardg1n": jnp.zeros_like(fields["aicen"]),
        "ardg2n": jnp.zeros_like(fields["aicen"]),
        "virdgn": jnp.zeros_like(fields["aicen"]),
        "msnow_mlt": jnp.zeros_like(asum),
        "esnow_mlt": jnp.zeros_like(asum),
        "mpond": jnp.zeros_like(asum),
        "maero": jnp.zeros_like(fields["faero_ocn"]),
        "miso": jnp.zeros_like(fields["fiso_ocn"]),
        "mbio": jnp.zeros_like(fields["flux_bio"]),
    }

    # A fixed-count loop supports reverse-mode differentiation; a dynamic
    # while loop would not. Once the area closes, remaining iterations skip.
    def iteration(_, carry):
        current, accumulated, closing, opening, done, failed = carry

        def advance(state):
            current, accumulated, closing, opening, _, failed = state
            aksum, apartic, hrmin, hrmax, hrexp, krdg, mraft = ridge_itd(
                current["aice0"], current["aicen"], current["vicen"],
                int(options["krdg_partic"]), int(options["krdg_redist"]),
                parameters["mu_rdg"], puny, current["mraftn"])
            current = dict(current)
            current["aparticn"] = apartic[1:]
            current["krdgn"] = krdg
            current["mraftn"] = mraft
            current, accumulated, opening, shift_failed = ridge_shift(
                dt, hin_max, current, layout, flags, options, parameters,
                aksum, apartic, hrmin, hrmax, hrexp, krdg,
                closing, opening, accumulated, bio_index)
            area = asum_ridging(current["aicen"], current["aice0"])
            done = jnp.abs(area - 1.0) < puny
            divu_adv = (1.0 - area) / dt
            closing = jnp.where(done, 0.0, jnp.maximum(0.0, -divu_adv))
            opening = jnp.where(done, 0.0, jnp.maximum(0.0, divu_adv))
            return (current, accumulated, closing, opening,
                    done, failed | shift_failed)

        def advance_unfinished(state):
            return jax.lax.cond(state[-2], lambda value: value, advance, state)

        axis_name = options.get("ridge_axis_name")
        if axis_name is not None:
            # A batch-wide predicate keeps this branch scalar under vmap.
            # The per-column predicate still masks already converged columns.
            any_unfinished = jax.lax.pmax((~done).astype(jnp.int32), axis_name) > 0
            return jax.lax.cond(
                any_unfinished, advance_unfinished, lambda state: state, carry)
        return advance_unfinished(carry)

    fields = dict(fields)
    fields["mraftn"] = jnp.zeros_like(fields["aicen"])
    fields, totals, _, _, done, failed = jax.lax.fori_loop(
        0, 20, iteration,
        (fields, totals, closing_net, opning,
         jnp.asarray(False), jnp.asarray(False)))
    failed = failed | ~done
    dti = 1.0 / dt
    fields.update({
        "dardg1dt": totals["ardg1"] * dti,
        "dardg2dt": totals["ardg2"] * dti,
        "dvirdgdt": totals["virdg"] * dti,
        "opening": totals["aopen"] * dti,
        "dardg1ndt": totals["ardg1n"] * dti,
        "dardg2ndt": totals["ardg2n"] * dti,
        "dvirdgndt": totals["virdgn"] * dti,
        "araftn": fields["mraftn"] * totals["ardg2n"],
        "vraftn": fields["mraftn"] * totals["virdgn"],
        "fresh": fields["fresh"] + totals["msnow_mlt"] / (ndtd * dt),
        "fhocn": fields["fhocn"] + totals["esnow_mlt"] / (ndtd * dt),
        "faero_ocn": fields["faero_ocn"] + totals["maero"] / (ndtd * dt),
        "fiso_ocn": fields["fiso_ocn"] + totals["miso"] / (ndtd * dt),
        "flux_bio": fields["flux_bio"] + totals["mbio"] / (ndtd * dt),
        "fpond": fields["fpond"] - totals["mpond"],
        "dpnd_ridge": totals["mpond"],
    })
    area = asum_ridging(fields["aicen"], fields["aice0"])
    failed = failed | (jnp.abs(area - 1.0) > puny)

    if bool(options["conserv_check"]):
        nilyr = int(options["nilyr"])
        nslyr = int(options["nslyr"])
        initial_eice = jnp.sum(
            initial["trcrn"][int(flags["nt_qice"]):int(flags["nt_qice"]) + nilyr]
            * initial["vicen"][None, :] / nilyr)
        final_eice = jnp.sum(
            fields["trcrn"][int(flags["nt_qice"]):int(flags["nt_qice"]) + nilyr]
            * fields["vicen"][None, :] / nilyr)
        initial_esno = jnp.sum(
            initial["trcrn"][int(flags["nt_qsno"]):int(flags["nt_qsno"]) + nslyr]
            * initial["vsnon"][None, :] / nslyr)
        final_esno = jnp.sum(
            fields["trcrn"][int(flags["nt_qsno"]):int(flags["nt_qsno"]) + nslyr]
            * fields["vsnon"][None, :] / nslyr)
        initial_sice = jnp.sum(
            initial["trcrn"][int(flags["nt_sice"]):int(flags["nt_sice"]) + nilyr]
            * initial["vicen"][None, :] / nilyr)
        final_sice = jnp.sum(
            fields["trcrn"][int(flags["nt_sice"]):int(flags["nt_sice"]) + nilyr]
            * fields["vicen"][None, :] / nilyr)
        initial_vbri = initial["vicen"]
        final_vbri = fields["vicen"]
        if bool(flags["tr_brine"]):
            initial_vbri = (initial_vbri
                            * initial["trcrn"][int(flags["nt_fbri"])])
            final_vbri = (final_vbri
                          * fields["trcrn"][int(flags["nt_fbri"])])
        failed = (failed
                  | (jnp.abs(jnp.sum(initial["vicen"])
                             - jnp.sum(fields["vicen"])) > puny)
                  | (jnp.abs(jnp.sum(initial["vsnon"])
                             - jnp.sum(fields["vsnon"])
                             - totals["msnow_mlt"] / parameters["rhos"]) > puny)
                  | (jnp.abs(initial_eice - final_eice)
                     > puny * parameters["Lfresh"] * parameters["rhoi"])
                  | (jnp.abs(initial_esno - final_esno - totals["esnow_mlt"])
                     > puny * parameters["Lfresh"] * parameters["rhos"])
                  | (jnp.abs(initial_sice - final_sice) > puny)
                  | (jnp.abs(jnp.sum(initial_vbri) - jnp.sum(final_vbri))
                     > puny * 10.0))
    return fields, failed


def icepack_step_ridge(dt, ndtd, hin_max, fields, layout, flags,
                       options, parameters, bio_index=(), column_call=None):
    """Run ridging and cleanup, optionally batching at the stage boundaries."""
    if column_call is None:
        def column_call(name, function, *arrays):
            return function(*arrays)

    # Fixed tracer indices and scheme switches select the compiled ridge branch.
    # State fields and physical parameters remain dynamic JAX inputs.
    def redistribute(cell):
        return ridge_ice(
            dt, ndtd, hin_max, cell, tuple(sorted(layout.items())),
            tuple(sorted(flags.items())), tuple(sorted(options.items())),
            parameters, tuple(bio_index))

    fields, failed = column_call("iteration", redistribute, fields)

    def cleanup(cell):
        return cleanup_itd_jax(
            dt * ndtd, hin_max.at[-1].set(1.0e8),
            cell["aicen"], cell["trcrn"], cell["vicen"], cell["vsnon"],
            cell["aice0"], cell["aice"], cell["first_ice"],
            cell["fpond"], cell["fresh"], cell["fsalt"], cell["fhocn"],
            cell["faero_ocn"], cell["fiso_ocn"], cell["flux_bio"],
            cell["Tf"], layout, flags, options, parameters, bio_index)

    (aicen, trcrn, vicen, vsnon, aice0, aice,
     first_ice, fpond, fresh, fsalt, fhocn,
     faero_ocn, fiso_ocn, flux_bio, cleanup_failed) = column_call("cleanup", cleanup, fields)
    fields = dict(fields)
    fields.update({
        "aicen": aicen, "trcrn": trcrn, "vicen": vicen, "vsnon": vsnon,
        "aice0": aice0, "aice": aice, "first_ice": first_ice,
        "fpond": fpond, "fresh": fresh, "fsalt": fsalt, "fhocn": fhocn,
        "faero_ocn": faero_ocn, "fiso_ocn": fiso_ocn, "flux_bio": flux_bio,
    })
    return fields, failed | cleanup_failed
