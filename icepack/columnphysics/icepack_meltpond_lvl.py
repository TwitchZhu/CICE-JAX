"""Level-ice melt ponds from icepack_meltpond_lvl.F90."""

import jax.numpy as jnp
from jax import lax

from icepack.columnphysics.icepack_parameters import c0, c1, c2, c10, p01, p5
from icepack.columnphysics.icepack_meltpond_sealvl import brine_permeability


def compute_ponds_lvl(dt, rfrac, meltt, melts, frain, Tair, fsurfn,
                      dhs, aicen, vicen, vsnon, qicen, sicen,
                      Tsfcn, alvl, apnd, hpnd, ipnd, meltsliqn,
                      dpnd_freebdn, dpnd_initialn, dpnd_dlidn,
                      dpnd_flushn, options, parameters):
    """Update level-ice ponds in the order of Fortran compute_ponds_lvl.

    The masks pond_cell and active replace the outer cell/thin-ice branches.
    Safe denominators keep unselected where expressions finite. Permeability
    drainage retains a lax.cond because it calls another physics routine.
    """
    puny = parameters["puny"]
    hi_min = parameters["hi_min"]
    rhofresh = parameters["rhofresh"]
    rhoi = parameters["rhoi"]
    rhos = parameters["rhos"]
    rhow = parameters["rhow"]
    Lfresh = parameters["Lfresh"]
    kice = parameters["kice"]
    Timelt = parameters["Timelt"]
    Tffresh = parameters["Tffresh"]
    pndaspect = parameters["pndaspect"]
    dpscale = parameters["dpscale"]
    gravit = parameters["gravit"]
    depressT = parameters["depressT"]
    viscosity_dyn = parameters["viscosity_dyn"]
    Td = c2
    rexp = p01

    # Initialize.
    zero = jnp.zeros_like(apnd)
    volpn = hpnd * aicen * alvl * apnd
    ffrac = zero

    # Identify grid cells where ponds can be; remove ponds on thin ice.
    pond_cell = aicen * alvl > puny**2
    aicen_safe = jnp.where(pond_cell, aicen, c1)
    hi = vicen / aicen_safe
    hs = vsnon / aicen_safe
    alvl_tmp = alvl
    thin_ice = hi < hi_min
    active = pond_cell & ~thin_ice

    # Initialize pond area as fraction of ice.
    apondn = apnd * alvl_tmp
    hlid = ipnd

    # Update pond volume: add melt water.
    if bool(options["use_smliq_pnd"]):
        dvn = rfrac / rhofresh * (meltt * rhoi + meltsliqn) * aicen
    else:
        dvn = (rfrac / rhofresh
               * (meltt * rhoi + melts * rhos + frain * dt) * aicen)

    # dvn is per grid area; runoff diagnostics are per category area.
    dpnd_initialn = jnp.where(
        active, dvn * (c1 - rfrac) /
        jnp.where(active, rfrac * aicen, c1), dpnd_initialn)
    dvn_temp = dvn

    # Shrink pond volume under freezing conditions.
    if options["frzpnd"] == "cesm":
        Tp = Timelt - Td
        dTs = jnp.maximum(Tp - Tsfcn, c0)
        dvn = dvn - volpn * (c1 - jnp.exp(rexp * dTs / Tp))
    elif options["frzpnd"] == "hlid":
        # Stefan approximation: fresh pond water, ice grows from existing lid.
        # Fortran: if (dvn == c0), freeze pond when Ts < c0.
        Ts = Tair - Tffresh
        freezing = active & (dvn == c0) & (Ts < c0)
        bdt = -c2 * Ts * kice * dt / (rhoi * Lfresh)
        # Mask the sqrt input where Fortran skips freezing; keep active bdt unchanged.
        bdt = jnp.where(freezing, bdt, c0)
        dhlid_freeze = p5 * jnp.sqrt(bdt)
        dhlid_freeze = jnp.where(
            hlid > dhlid_freeze,
            p5 * bdt / jnp.where(hlid > c0, hlid, c1), dhlid_freeze)
        dhlid_freeze = jnp.minimum(dhlid_freeze, hpnd * rhofresh / rhoi)
        dhlid_freeze = jnp.where(freezing, dhlid_freeze, c0)

        # Fortran: else, convert refrozen pond ice back to water.
        dhlid_melt = jnp.maximum(fsurfn * dt / (rhoi * Lfresh), c0)
        dhlid_melt = -jnp.minimum(dhlid_melt, hlid)
        dhlid = jnp.where(dvn == c0, dhlid_freeze, dhlid_melt)
        hlid = jnp.where(
            dvn == c0, hlid + dhlid, jnp.maximum(hlid + dhlid, c0))

        # Fraction of fsurfn used to melt snow-free pond ice.
        ffrac = jnp.where(
            (dvn != c0) & (hs - dhs < puny),
            jnp.where(fsurfn > puny,
                      jnp.minimum(-dhlid * rhoi * Lfresh /
                                  (dt * jnp.where(fsurfn > c0, fsurfn, c1)), c1),
                      c1), c0)
        alid = apondn * aicen
        dvn = dvn - dhlid * alid * rhoi / rhofresh
    else:
        raise ValueError("unsupported frzpnd option")

    volpn = volpn + dvn
    # Sign reversed from dvn: lost/gained water due to lid freezing/melting.
    dpnd_dlidn = jnp.where(active, (dvn_temp - dvn) / aicen_safe, dpnd_dlidn)

    # Update pond area and depth.
    # Fortran: if (volpn <= c0), remove the pond depression.
    apondn = jnp.where(volpn <= c0, c0, apondn)
    volpn = jnp.maximum(volpn, c0)

    # Fortran: existing ponds / new ponds / runoff from deformed ice.
    existing = apondn * aicen > puny
    can_create = alvl_tmp * aicen > c10 * puny
    apondn_existing = jnp.clip(
        apondn + p5 * dvn /
        (pndaspect * jnp.where(existing, apondn, c1) * aicen_safe),
        c0, alvl_tmp)
    hpondn_existing = jnp.where(
        apondn_existing > puny,
        volpn / jnp.where(apondn_existing > c0,
                          apondn_existing * aicen_safe, c1), c0)
    apondn_new = jnp.minimum(
        jnp.sqrt(jnp.maximum(volpn / (pndaspect * aicen_safe), c0)), alvl_tmp)
    hpondn_new = pndaspect * apondn_new
    apondn = jnp.where(existing, apondn_existing,
                       jnp.where(can_create, apondn_new, c0))
    hpondn = jnp.where(existing, hpondn_existing,
                       jnp.where(can_create, hpondn_new, c0))
    apondn = jnp.maximum(apondn, c0)

    # Limit pond depth to maintain nonnegative freeboard.
    hpond_tmp = hpondn
    hpondn = jnp.minimum(hpondn, ((rhow - rhoi) * hi - rhos * hs) / rhofresh)
    dpnd_freebdn = jnp.where(active, (hpond_tmp - hpondn) * apondn, dpnd_freebdn)

    # Convert from fraction of category ice to fraction of grid cell.
    apondn = apondn * aicen
    volpn = hpondn * apondn
    # Fully drained/frozen ponds lose both their depression and their ice lid.
    empty = volpn <= c0
    volpn = jnp.where(empty, c0, volpn)
    apondn = jnp.where(empty, c0, apondn)
    hpondn = jnp.where(empty, c0, hpondn)
    hlid = jnp.where(empty, c0, hlid)

    # Drainage due to permeability (flushing); dpscale=0 disables it.
    # ktherm=2 handles flushing in the mushy thermodynamics instead.
    if int(options["ktherm"]) != 2:
        def drain_pond(state):
            apondn, hpondn, volpn, dpnd_flushn = state
            draft = (rhos * hs + rhoi * hi) / rhow + hpondn
            deltah = hpondn + hi - draft
            pressure_head = gravit * rhow * jnp.maximum(deltah, c0)
            # Use the initial salinity and melting-temperature profiles.
            Tmlt = -sicen * depressT
            perm = brine_permeability(qicen, sicen, Tmlt, options, parameters)
            drain = perm * pressure_head * dt / (viscosity_dyn * hi) * dpscale
            deltah = jnp.minimum(drain, hpondn)
            dvn = -deltah * apondn
            volpn = volpn + dvn
            apondn = jnp.clip(
                apondn + p5 * dvn / (pndaspect * apondn), c0, alvl_tmp * aicen)
            hpondn = jnp.where(
                apondn > puny, volpn / jnp.where(apondn > c0, apondn, c1), c0)
            dpnd_flushn = -dvn / aicen
            return apondn, hpondn, volpn, dpnd_flushn

        apondn, hpondn, volpn, dpnd_flushn = lax.cond(
            active & (hpondn > c0) & (dpscale > puny),
            drain_pond, lambda state: state,
            (apondn, hpondn, volpn, dpnd_flushn))

    # Reload tracer array. Thin ice loses its pond; ineligible cells retain
    # their input tracers. Diagnostic increments above update only active ice.
    hpnd = jnp.where(pond_cell, jnp.where(thin_ice, c0, hpondn), hpnd)
    apnd = jnp.where(
        pond_cell, jnp.where(thin_ice, c0,
                             apondn / jnp.where(pond_cell, aicen * alvl_tmp, c1)),
        apnd)
    if options["frzpnd"] == "hlid":
        ipnd = jnp.where(pond_cell, jnp.where(thin_ice, c0, hlid), ipnd)
    ffrac = jnp.where(active, ffrac, zero)
    return (apnd, hpnd, ipnd, ffrac, dpnd_freebdn,
            dpnd_initialn, dpnd_dlidn, dpnd_flushn)
