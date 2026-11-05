"""Water-isotope tracers in the vertical sea-ice thermodynamics step."""

import jax.numpy as jnp


def isoice_alpha(growth_rate, species, fractionation):
    """Fractionation when ocean water freezes onto the ice base."""
    if fractionation == "nfrac" or species == "H2_16O":
        return jnp.ones_like(growth_rate)
    if fractionation == "cfrac":
        if species == "HDO":
            return jnp.asarray(1.02120)
        if species == "H2_18O":
            return jnp.asarray(1.00291)
    if fractionation == "gfrac":
        isotope_factor = 7.2852 if species == "HDO" else 1.0
        return (1.0 + isotope_factor * 1.2280e-3
                + 0.7311e-3 * jnp.exp(-growth_rate / 8.0100e8)
                + 0.8441e-3 * jnp.exp(-growth_rate / 0.7800e6))
    raise ValueError("unsupported isotope fractionation option")


def wiso_alpi(species_index, temperature):
    """Ice/vapor fractionation for water isotope species 2, 3 or 4."""
    if species_index == 2:
        return jnp.ones_like(temperature)
    if species_index == 3:
        return jnp.exp(16289.0 / temperature**2 - 9.45e-2)
    if species_index == 4:
        return jnp.exp(11.839 / temperature - 28.224e-3)
    raise ValueError("unsupported water isotope species")


def update_isotope(dt, meltt, melts, meltb, congel, snoice, evap,
                   fsnow, Tsfc, Qref_iso, isosno, isoice,
                   aice_old, vice_old, vsno_old, vicen, vsnon,
                   aicen, fiso_atm, fiso_evapn, fiso_ocnn,
                   HDO_ocn, H2_16O_ocn, H2_18O_ocn,
                   options, parameters):
    """Move HDO, ordinary water and H2-18O among snow, ice and ocean."""
    _ = evap  # Fortran derives evaporation from the volume budget below.
    if isosno.shape[0] > 3:
        raise ValueError("Fortran update_isotope supports at most three species")
    puny = parameters["puny"]
    rhoi = parameters["rhoi"]
    rhos = parameters["rhos"]
    hs_old = vsno_old / aice_old
    hi_old = vice_old / aice_old
    dzsno = hs_old
    dzice = hi_old
    old_area = jnp.where(aicen > puny, aicen, aice_old)
    hs = vsnon / old_area
    hi = vicen / old_area
    if int(options["ktherm"]) == 2:
        dhs_snoice = snoice
    else:
        dhs_snoice = snoice * rhoi / rhos
    evaps = hs - (hs_old - melts - dhs_snoice + fsnow / rhos * dt)
    evapi = hi - (hi_old - meltt - meltb + congel + snoice)
    TsfK = Tsfc + parameters["Tffresh"]
    fractionation = options["isotope_frac_method"]

    ratio = jnp.ones_like(Qref_iso)
    alphai = jnp.ones_like(Qref_iso)
    if fractionation != "nfrac":
        ratio = jnp.where(
            Qref_iso[1] > puny,
            Qref_iso / jnp.where(Qref_iso[1] > puny,
                                 Qref_iso[1], 1.0), 0.0)
        alphai = alphai.at[0].set(wiso_alpi(3, TsfK))
        alphai = alphai.at[1].set(wiso_alpi(2, TsfK))
        if alphai.shape[0] > 2:
            alphai = alphai.at[2].set(wiso_alpi(4, TsfK))
    else:
        ratio = jnp.where(Qref_iso[1] <= puny, 0.0, ratio)

    # Condensation to snow and ice uses humidity isotope ratios.
    snow_condense = evaps > 0.0
    work = jnp.where(snow_condense,
                     alphai * ratio * rhos * evaps * aicen, 0.0)
    fiso_evapn = fiso_evapn + work / dt
    isosno = isosno + work
    dzsno = dzsno + jnp.where(snow_condense, evaps, 0.0)
    ice_condense = evapi > 0.0
    work = jnp.where(ice_condense,
                     alphai * ratio * rhoi * evapi * aicen, 0.0)
    fiso_evapn = fiso_evapn + work / dt
    isoice = isoice + work
    dzice = dzice + jnp.where(ice_condense, evapi, 0.0)

    ocean = jnp.stack((HDO_ocn, H2_16O_ocn, H2_18O_ocn))[:isoice.shape[0]]
    alpha = jnp.stack(tuple(
        isoice_alpha(congel / dt, species, fractionation)
        for species in ("HDO", "H2_16O", "H2_18O")
    ))[:isoice.shape[0]]
    work = jnp.where(congel > 0.0,
                     alpha * ocean * rhoi * congel * aicen, 0.0)
    isoice = isoice + work
    fiso_ocnn = fiso_ocnn - work / dt
    dzice = dzice + jnp.where(congel > 0.0, congel, 0.0)

    # Sublimation removes isotope in proportion to the remaining layer mass.
    snow_sublime = evaps < 0.0
    sloss = jnp.where(
        snow_sublime & (dzsno > puny),
        isosno * jnp.minimum(-evaps, dzsno) /
        jnp.where(dzsno > puny, dzsno, 1.0), 0.0)
    sloss = jnp.minimum(sloss, isosno)
    isosno = isosno - sloss
    fiso_evapn = fiso_evapn - sloss / dt
    dzsno = dzsno + jnp.where(snow_sublime, evaps, 0.0)
    snow_gone = snow_sublime & (dzsno <= 0.0)
    fiso_evapn = fiso_evapn - jnp.where(snow_gone, isosno / dt, 0.0)
    isosno = jnp.where(snow_gone, 0.0, isosno)
    dzsno = jnp.where(snow_gone, 0.0, dzsno)

    ice_sublime = evapi < 0.0
    sloss = jnp.where(
        ice_sublime & (dzice > puny),
        isoice * jnp.minimum(-evapi, dzice) /
        jnp.where(dzice > puny, dzice, 1.0), 0.0)
    sloss = jnp.minimum(sloss, isoice)
    isoice = isoice - sloss
    fiso_evapn = fiso_evapn - sloss / dt
    dzice = dzice + jnp.where(ice_sublime, evapi, 0.0)
    ice_gone = ice_sublime & (dzice <= 0.0)
    fiso_evapn = fiso_evapn - jnp.where(ice_gone, isoice / dt, 0.0)
    isoice = jnp.where(ice_gone, 0.0, isoice)
    dzice = jnp.where(ice_gone, 0.0, dzice)

    # Top snow melt, top ice melt, then basal ice melt.
    snow_melt = melts > 0.0
    sloss = jnp.where(
        snow_melt & (dzsno > puny),
        isosno * jnp.minimum(melts, dzsno) /
        jnp.where(dzsno > puny, dzsno, 1.0), 0.0)
    sloss = jnp.minimum(sloss, isosno)
    isosno = isosno - sloss
    fiso_ocnn = fiso_ocnn + sloss / dt
    dzsno = dzsno - jnp.where(snow_melt, melts, 0.0)
    snow_gone = snow_melt & (dzsno <= 0.0)
    fiso_ocnn = fiso_ocnn + jnp.where(snow_gone, isosno / dt, 0.0)
    isosno = jnp.where(snow_gone, 0.0, isosno)
    dzsno = jnp.where(snow_gone, 0.0, dzsno)

    top_melt = meltt > 0.0
    sloss = jnp.where(
        top_melt & (dzice > puny),
        isoice * jnp.minimum(meltt, dzice) /
        jnp.where(dzice > puny, dzice, 1.0), 0.0)
    sloss = jnp.minimum(sloss, isoice)
    isoice = isoice - sloss
    fiso_ocnn = fiso_ocnn + sloss / dt
    dzice = dzice - jnp.where(top_melt, meltt, 0.0)
    ice_gone = top_melt & (dzice <= 0.0)
    # The original adds the residual mass directly, without dividing by dt.
    fiso_ocnn = fiso_ocnn + jnp.where(ice_gone, isoice, 0.0)
    isoice = jnp.where(ice_gone, 0.0, isoice)
    dzice = jnp.where(ice_gone, 0.0, dzice)

    bottom_melt = meltb > 0.0
    sloss = jnp.where(
        bottom_melt & (dzice > puny),
        jnp.maximum(meltb - dzice, 0.0) * isoice /
        jnp.where(dzice > puny, dzice, 1.0), 0.0)
    sloss = jnp.minimum(sloss, isoice)
    isoice = isoice - sloss
    fiso_ocnn = fiso_ocnn + sloss / dt
    dzice = dzice - jnp.where(bottom_melt, meltb, 0.0)
    ice_gone = bottom_melt & (dzice <= 0.0)
    # Keep the Fortran residual-mass assignment here as well.
    fiso_ocnn = fiso_ocnn + jnp.where(ice_gone, isoice, 0.0)
    isoice = jnp.where(ice_gone, 0.0, isoice)
    dzice = jnp.where(ice_gone, 0.0, dzice)

    snowfall = fsnow > 0.0
    isosno = isosno + jnp.where(
        snowfall, fiso_atm * aicen * dt, 0.0)
    dzsno = dzsno + jnp.where(snowfall, fsnow / rhos * dt, 0.0)

    snow_ice = dhs_snoice > 0.0
    sloss = jnp.where(
        snow_ice & (dzsno > puny),
        jnp.minimum(dhs_snoice, dzsno) * isosno /
        jnp.where(dzsno > puny, dzsno, 1.0), 0.0)
    sloss = jnp.minimum(sloss, isosno)
    isosno = isosno - sloss
    isoice = isoice + sloss
    dzsno = dzsno - jnp.where(snow_ice, dhs_snoice, 0.0)
    dzice = dzice + jnp.where(snow_ice, snoice, 0.0)
    snow_gone = snow_ice & (dzsno <= 0.0)
    fiso_ocnn = fiso_ocnn + jnp.where(snow_gone, isosno / dt, 0.0)
    isosno = jnp.where(snow_gone, 0.0, isosno)

    fiso_ocnn = jnp.where(aicen > puny, fiso_ocnn / aicen, fiso_ocnn)
    fiso_evapn = jnp.where(aicen > puny, fiso_evapn / aicen,
                           fiso_evapn)
    return isosno, isoice, fiso_ocnn, fiso_evapn
