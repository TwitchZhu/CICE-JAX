"""Sea-level melt-pond geometry from icepack_meltpond_sealvl.F90."""

import jax.numpy as jnp
from jax import lax

from icepack.columnphysics.icepack_therm_shared import (
    calculate_Tin_from_qin_jax,
)


def calc_pndasp(hin, parameters):
    """Aspect ratio of pond depth to area for the sea-level scheme."""
    return (hin * (parameters["rhow"] - parameters["rhosi"])
            / (parameters["rhofresh"] * parameters["apnd_sl"]**2
               - 2.0 * parameters["rhow"] * parameters["apnd_sl"]
               + parameters["rhow"]))


def pond_hypsometry(hpnd, apnd, options, parameters,
                    dhpond=None, dvpond=None, hin=None):
    """Update pond area and depth while conserving their product."""
    if (dhpond is None) == (dvpond is None):
        raise ValueError("supply exactly one of dhpond and dvpond")
    if options["pndhyps"] == "sealevel":
        if hin is None:
            raise ValueError("sealevel ponds require hin")
        pndasp = calc_pndasp(hin, parameters)
    elif options["pndhyps"] == "fixed":
        pndasp = parameters["pndaspect"]
    else:
        raise ValueError("unsupported pndhyps option")
    dv = dvpond if dvpond is not None else dhpond * apnd
    vp = apnd * hpnd + dv
    present = vp > parameters["puny"]
    candidate_area = jnp.sqrt(jnp.maximum(vp, 0.0) /
                              jnp.where(pndasp > 0.0, pndasp, 1.0))
    apnd = jnp.where(present, jnp.minimum(candidate_area, 1.0), 0.0)
    hpnd = jnp.where(present,
                     jnp.where(candidate_area < 1.0,
                               candidate_area * pndasp, vp), 0.0)
    return hpnd, apnd


def pond_height(apond, hpnd, hin, options, parameters):
    """Height of the pond surface above the mean ice base."""
    if options["pndhead"] == "perched":
        return hin + hpnd
    if options["pndhead"] != "hyps":
        raise ValueError("unsupported pndhead option")
    if options["pndhyps"] == "sealevel":
        pndasp = calc_pndasp(hin, parameters)
    elif options["pndhyps"] == "fixed":
        pndasp = parameters["pndaspect"]
    else:
        raise ValueError("unsupported pndhyps option")
    return jnp.where(apond < 1.0,
                     hin - pndasp + 2.0 * pndasp * apond,
                     hin + hpnd)


def brine_permeability(qicen, salin, Tmlt, options, parameters):
    """BL99 brine permeability from the least permeable ice layer."""
    Tin, _ = calculate_Tin_from_qin_jax(
        qicen, Tmlt, bool(options["l_brine"]), parameters)
    safe_Tin = jnp.where(Tin != 0.0, Tin, -1.0)
    Sbr = 1.0 / (0.001 - parameters["depressT"] / safe_Tin)
    phi = salin / Sbr
    phi = jnp.where(phi < 0.05, 0.0, phi)
    return 3.0e-8 * jnp.min(phi)**3


def compute_ponds_sealvl(dt, meltt, melts, frain, Tair, fsurfn,
                         Tsfcn, dhs, aicen, vicen, vsnon,
                         qicen, sicen, apnd, hpnd, ipnd,
                         meltsliqn, dpnd_freebdn, dpnd_dlidn,
                         dpnd_flushn, options, parameters):
    """Update sea-level pond volume, refrozen lid, freeboard and drainage."""
    zero = jnp.zeros_like(apnd)

    def update_pond(_):
        hi = vicen / aicen
        hs = vsnon / aicen
        vpondn = hpnd * apnd

        def thin_ice(_):
            new_ipnd = zero if options["frzpnd"] == "hlid" else ipnd
            return (zero, zero, new_ipnd, zero, vpondn,
                    dpnd_dlidn, dpnd_flushn)

        def thick_ice(_):
            if bool(options["use_smliq_pnd"]):
                dvpondn = ((meltt * parameters["rhoi"] + meltsliqn)
                           / parameters["rhofresh"])
            else:
                dvpondn = ((meltt * parameters["rhoi"]
                            + melts * parameters["rhos"] + frain * dt)
                           / parameters["rhofresh"])
            dvn_temp = dvpondn
            ffrac = zero
            hlid = ipnd
            if options["frzpnd"] == "cesm":
                Tp = parameters["Timelt"] - 2.0
                dTs = jnp.maximum(Tp - Tsfcn, 0.0)
                dvpondn = (dvpondn - vpondn *
                           (1.0 - jnp.exp(0.01 * dTs / Tp)))
            elif options["frzpnd"] == "hlid":
                Ts = Tair - parameters["Tffresh"]
                bdt = (-2.0 * Ts * parameters["kice"] * dt /
                       (parameters["rhoi"] * parameters["Lfresh"]))
                new_lid = 0.5 * jnp.sqrt(jnp.maximum(bdt, 1.0e-30))
                new_lid = jnp.where(
                    hlid > new_lid,
                    0.5 * bdt / jnp.where(hlid > 0.0, hlid, 1.0),
                    new_lid)
                new_lid = jnp.minimum(
                    new_lid,
                    hpnd * parameters["rhofresh"] / parameters["rhoi"])
                melt_lid = -jnp.minimum(
                    jnp.maximum(fsurfn * dt /
                                (parameters["rhoi"] * parameters["Lfresh"]),
                                0.0), hlid)
                dhlid = jnp.where(
                    dvpondn == 0.0, jnp.where(Ts < 0.0, new_lid, 0.0),
                    melt_lid)
                hlid = jnp.maximum(hlid + dhlid, 0.0)
                ffrac = jnp.where(
                    (dvpondn != 0.0) & (hs - dhs < parameters["puny"]),
                    jnp.where(fsurfn > parameters["puny"],
                              jnp.minimum(
                                  -dhlid * parameters["rhoi"]
                                  * parameters["Lfresh"] /
                                  (dt * jnp.where(fsurfn > 0.0,
                                                  fsurfn, 1.0)), 1.0),
                              1.0), 0.0)
                dvpondn = (dvpondn - dhlid * apnd
                           * parameters["rhoi"] / parameters["rhofresh"])
            else:
                raise ValueError("unsupported frzpnd option")
            dpnd_dlidn = dvn_temp - dvpondn
            hpnd_new, apnd_new = pond_hypsometry(
                hpnd, apnd, options, parameters, dvpond=dvpondn, hin=hi)

            if options["pndfrbd"] == "floor":
                dhpond = (((parameters["rhow"] - parameters["rhoi"])
                           * hi - parameters["rhos"] * hs)
                          / parameters["rhofresh"] - hpnd_new)
            elif options["pndfrbd"] == "category":
                dhpond = jnp.where(
                    apnd_new > parameters["puny"],
                    (((parameters["rhow"] - parameters["rhoi"])
                      * hi - parameters["rhos"] * hs)
                     / (parameters["rhofresh"]
                        * jnp.where(apnd_new > 0.0, apnd_new, 1.0))
                     - hpnd_new), 0.0)
            else:
                raise ValueError("unsupported pndfrbd option")
            dhpond = jnp.minimum(dhpond, 0.0)
            dpnd_freebdn = -dhpond * apnd_new
            hpnd_new, apnd_new = pond_hypsometry(
                hpnd_new, apnd_new, options, parameters,
                dhpond=dhpond, hin=hi)
            empty = ((hpnd_new <= parameters["puny"])
                     | (apnd_new <= parameters["puny"]))
            hpnd_new = jnp.where(empty, 0.0, hpnd_new)
            apnd_new = jnp.where(empty, 0.0, apnd_new)
            hlid = jnp.where(empty, 0.0, hlid)

            if int(options["ktherm"]) != 2:
                active = ((hpnd_new > 0.0)
                          & (parameters["dpscale"] > parameters["puny"]))

                def drain_pond(_):
                    draft = ((parameters["rhos"] * hs
                              + parameters["rhoi"] * hi
                              + parameters["rhofresh"]
                              * hpnd_new * apnd_new)
                             / parameters["rhow"])
                    hpsurf = pond_height(apnd_new, hpnd_new, hi,
                                         options, parameters)
                    pressure_head = (
                        parameters["gravit"] * parameters["rhow"]
                        * jnp.maximum(hpsurf - draft, 0.0))
                    Tmlt = -sicen * parameters["depressT"]
                    perm = brine_permeability(
                        qicen, sicen, Tmlt, options, parameters)
                    drain = (perm * pressure_head * dt
                             / hi / parameters["viscosity_dyn"]
                             * parameters["dpscale"])
                    dhpond = -jnp.minimum(drain, hpnd_new)
                    drained_hpnd, drained_apnd = pond_hypsometry(
                        hpnd_new, apnd_new, options, parameters,
                        dhpond=dhpond, hin=hi)
                    return (drained_hpnd, drained_apnd,
                            -dhpond * apnd_new)

                hpnd_new, apnd_new, dpnd_flushn = lax.cond(
                    active, drain_pond,
                    lambda _: (hpnd_new, apnd_new, dpnd_flushn),
                    operand=None)
            new_ipnd = hlid if options["frzpnd"] == "hlid" else ipnd
            return (apnd_new, hpnd_new, new_ipnd, ffrac,
                    dpnd_freebdn, dpnd_dlidn, dpnd_flushn)

        return lax.cond(hi < parameters["hi_min"],
                        thin_ice, thick_ice, operand=None)

    return lax.cond(aicen > parameters["puny"], update_pond,
                    lambda _: (apnd, hpnd, ipnd, zero,
                               dpnd_freebdn, dpnd_dlidn,
                               dpnd_flushn), operand=None)
