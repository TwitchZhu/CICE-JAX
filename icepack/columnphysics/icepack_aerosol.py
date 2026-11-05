"""Icepack aerosol changes during vertical thermodynamics."""

import jax
import jax.numpy as jnp


def update_snow_bgc(dt, meltt, melts, meltb, congel, snoice, fsnow,
                    trcrn, bio_index, aice_old, vice_old, vsno_old,
                    vicen, vsnon, aicen, flux_bio_atm, flux_bio,
                    bio_index_o, kscavz, nslyr, nblyr, parameters):
    """Move vertical BGC tracers through the two snow layers."""
    del meltt, meltb, congel, vice_old  # Passed by the Fortran caller but unused.
    nbtrcr = flux_bio_atm.shape[0]
    puny = parameters["puny"]
    hs_old = jnp.where(aice_old > puny,
                        vsno_old / jnp.where(aice_old > puny, aice_old, 1.0),
                        0.0)
    dzssl = hs_old / nslyr / 2.0
    dzint = hs_old - dzssl
    ar = jnp.where(aicen > 0.0, 1.0 / jnp.where(aicen > 0.0, aicen, 1.0), 1.0)
    hs = jnp.where(aicen > 0.0, vsnon * ar,
                   jnp.where(aice_old > 0.0,
                             vsnon / jnp.where(aice_old > 0.0, aice_old, 1.0),
                             0.0))
    hi = jnp.where(aicen > 0.0, vicen * ar, 0.0)
    hilyr = hi / nblyr
    dzssl_new = hs / nslyr / 2.0
    dhs_melts = -melts
    dhs_snoice = snoice * parameters["rhoi"] / parameters["rhos"]
    dhs_evap = (hs - hs_old - dhs_melts + dhs_snoice
                - fsnow / parameters["rhos"] * dt)
    ssl_slots = jnp.asarray(bio_index[:nbtrcr]) + nblyr + 1
    int_slots = ssl_slots + 1
    surface = trcrn[ssl_slots]
    interior = trcrn[int_slots]
    zbgc_snow = jnp.zeros_like(flux_bio_atm)
    zbgc_atm = jnp.zeros_like(flux_bio_atm)

    def thin_snow(_):
        inventory = surface * dzssl + interior * dzint
        into_ocean = hilyr < parameters["hs_ssl_min"]
        new_flux = flux_bio + jnp.where(
            into_ocean, inventory / dt + flux_bio_atm, 0.0)
        snow = jnp.where(into_ocean, 0.0, inventory)
        atm = jnp.where(into_ocean, 0.0, flux_bio_atm * dt)
        updated = trcrn.at[ssl_slots].set(0.0)
        updated = updated.at[int_slots].set(0.0)
        return updated, new_flux, snow, atm, jnp.asarray(False)

    def snow_layers(_):
        aerosno_ssl = surface * dzssl
        aerosno_int = interior * dzint
        aerotot0 = aerosno_ssl + aerosno_int
        old_flux = flux_bio
        snow = zbgc_snow
        atm = zbgc_atm
        new_flux = flux_bio
        ssl = dzssl
        snow_int = dzint

        # Evaporation can eliminate either snow layer.
        snow_int = snow_int + jnp.minimum(ssl + dhs_evap, 0.0)
        ssl = jnp.maximum(ssl + dhs_evap, 0.0)
        aerosno_int = aerosno_int + jnp.where(ssl <= puny, aerosno_ssl, 0.0)
        aerosno_ssl = jnp.where(ssl <= puny, 0.0, aerosno_ssl)
        snow = snow + jnp.where(snow_int <= puny,
                                aerosno_ssl + aerosno_int, 0.0)
        aerosno_ssl = jnp.where(snow_int <= puny, 0.0, aerosno_ssl)
        aerosno_int = jnp.where(snow_int <= puny, 0.0, aerosno_int)

        # New snowfall buries part of the surface scattering layer.
        snowfall = jnp.maximum(fsnow, 0.0) / parameters["rhos"] * dt
        dz = jnp.minimum(snowfall, ssl)
        sloss1 = jnp.where(ssl > puny,
                           aerosno_ssl * dz / jnp.where(ssl > puny, ssl, 1.0),
                           0.0)
        aerosno_ssl = jnp.maximum(0.0, aerosno_ssl - sloss1)
        aerosno_int = aerosno_int + sloss1
        ssl = ssl - dz + snowfall
        snow_int = snow_int + dz
        aerosno_int = aerosno_int + jnp.where(ssl <= puny, aerosno_ssl, 0.0)
        aerosno_ssl = jnp.where(ssl <= puny, 0.0, aerosno_ssl)
        snow = snow + jnp.where(snow_int <= puny,
                                aerosno_ssl + aerosno_int, 0.0)
        aerosno_ssl = jnp.where(snow_int <= puny, 0.0, aerosno_ssl)
        aerosno_int = jnp.where(snow_int <= puny, 0.0, aerosno_int)

        # Melt scavenges both snow reservoirs into the ice.
        scavenging = kscavz[jnp.asarray(bio_index_o[:nbtrcr]) - 1]
        melting = -dhs_melts > puny
        sloss1 = jnp.where(ssl > puny,
                           scavenging * aerosno_ssl
                           * jnp.minimum(-dhs_melts, ssl)
                           / jnp.where(ssl > puny, ssl, 1.0), 0.0)
        sloss2 = jnp.where(snow_int > puny,
                           scavenging * aerosno_int
                           * jnp.maximum(-dhs_melts - ssl, 0.0)
                           / jnp.where(snow_int > puny, snow_int, 1.0), 0.0)
        aerosno_ssl = jnp.where(melting,
                                jnp.maximum(0.0, aerosno_ssl - sloss1), aerosno_ssl)
        aerosno_int = jnp.where(melting,
                                jnp.maximum(0.0, aerosno_int - sloss2), aerosno_int)
        snow = snow + jnp.where(melting, sloss1 + sloss2, 0.0)
        melted_int = snow_int + jnp.minimum(ssl + dhs_melts, 0.0)
        melted_ssl = jnp.maximum(ssl + dhs_melts, 0.0)
        ssl = jnp.where(melting, melted_ssl, ssl)
        snow_int = jnp.where(melting, melted_int, snow_int)
        aerosno_int = aerosno_int + jnp.where(melting & (ssl <= puny),
                                               aerosno_ssl, 0.0)
        aerosno_ssl = jnp.where(melting & (ssl <= puny), 0.0, aerosno_ssl)
        snow = snow + jnp.where(melting & (snow_int <= puny),
                                aerosno_ssl + aerosno_int, 0.0)
        aerosno_ssl = jnp.where(melting & (snow_int <= puny), 0.0, aerosno_ssl)
        aerosno_int = jnp.where(melting & (snow_int <= puny), 0.0, aerosno_int)
        ssl = jnp.where(melting, jnp.maximum(ssl, 0.0), ssl)
        snow_int = jnp.where(melting, jnp.maximum(snow_int, 0.0), snow_int)

        # Snow-ice formation transfers scavenged material to the ocean.
        making_ice = dhs_snoice > puny
        sloss2 = jnp.where((snow_int > puny) & (aerosno_int > 0.0),
                           jnp.minimum(dhs_snoice, snow_int) * aerosno_int
                           / jnp.where(snow_int > puny, snow_int, 1.0), 0.0)
        sloss1 = jnp.where((ssl > puny) & (aerosno_ssl > 0.0),
                           jnp.maximum(dhs_snoice - snow_int, 0.0) * aerosno_ssl
                           / jnp.where(ssl > puny, ssl, 1.0), 0.0)
        aerosno_int = jnp.where(making_ice,
                                jnp.maximum(0.0, aerosno_int - sloss2), aerosno_int)
        aerosno_ssl = jnp.where(making_ice,
                                jnp.maximum(0.0, aerosno_ssl - sloss1), aerosno_ssl)
        new_flux = new_flux + jnp.where(
            making_ice, scavenging * (sloss1 + sloss2) / dt, 0.0)
        snow = snow + jnp.where(
            making_ice, (1.0 - scavenging) * (sloss1 + sloss2), 0.0)
        ssl = jnp.where(making_ice,
                        jnp.maximum(0.0, ssl - jnp.maximum(dhs_snoice - snow_int, 0.0)),
                        ssl)
        snow_int = jnp.where(making_ice,
                             jnp.maximum(snow_int - dhs_snoice, 0.0), snow_int)

        # Deposit atmospheric material into the surface or interior layer.
        split = jnp.where(ssl >= parameters["hs_ssl"] * 0.5, 0.0,
                          (parameters["hs_ssl"] * 0.5 - ssl)
                          / (parameters["hs_ssl"] * 0.5))
        aerosno_ssl = aerosno_ssl + flux_bio_atm * dt * (1.0 - split)
        aerosno_int = aerosno_int + flux_bio_atm * dt * split
        aerosno_int = aerosno_int + jnp.where(ssl <= puny, aerosno_ssl, 0.0)
        aerosno_ssl = jnp.where(ssl <= puny, 0.0, aerosno_ssl)
        snow = snow + jnp.where(snow_int <= puny,
                                jnp.maximum(0.0, aerosno_ssl + aerosno_int), 0.0)
        aerosno_ssl = jnp.where(snow_int <= puny, 0.0, aerosno_ssl)
        aerosno_int = jnp.where(snow_int <= puny, 0.0, aerosno_int)

        final_hs = jnp.where(aicen > 0.0, vsnon * ar, 0.0)
        dzssl_final = final_hs / nslyr / 2.0
        dzint_final = jnp.maximum(0.0, final_hs - dzssl_final)
        dznew_down = jnp.minimum(dzssl_final - ssl, 0.0)
        move_down = jnp.where((ssl > puny) & (aerosno_ssl > 0.0),
                              dznew_down * aerosno_ssl
                              / jnp.where(ssl > puny, ssl, 1.0), 0.0)
        dznew_up = jnp.maximum(dzssl_final - ssl, 0.0)
        move_up = jnp.where((snow_int > puny) & (aerosno_int > 0.0),
                            aerosno_int * dznew_up
                            / jnp.where(snow_int > puny, snow_int, 1.0), 0.0)
        moved = jnp.where((snow_int > puny) & (aerosno_int > 0.0),
                           move_up, move_down)
        residual_snow = aerosno_ssl + aerosno_int
        aerosno_ssl = jnp.where(final_hs > parameters["hs_min"],
                                jnp.maximum(0.0, aerosno_ssl + moved), 0.0)
        aerosno_int = jnp.where(final_hs > parameters["hs_min"],
                                jnp.maximum(0.0, aerosno_int - moved), 0.0)
        snow = snow + jnp.where(final_hs > parameters["hs_min"], 0.0,
                                residual_snow)
        valid_layers = ((dzssl_final > puny) & (dzint_final > puny)
                        & (vsnon > puny))
        ssl_conc = jnp.where(valid_layers,
                             aerosno_ssl / jnp.where(valid_layers, dzssl_final, 1.0), 0.0)
        int_conc = jnp.where(valid_layers,
                             aerosno_int / jnp.where(valid_layers, dzint_final, 1.0), 0.0)
        updated = trcrn.at[ssl_slots].set(ssl_conc)
        updated = updated.at[int_slots].set(int_conc)
        aerotot = aerosno_ssl + aerosno_int + snow + atm
        conservation = aerotot - aerotot0 - (flux_bio_atm - (new_flux - old_flux)) * dt
        scale = jnp.where((aerotot0 > aerotot) & (aerotot0 > 0.0),
                          aerotot0, jnp.where(aerotot > 0.0, aerotot, 1.0))
        failed = jnp.any((conservation / scale > puny)
                          | (snow + atm < 0.0)
                          | (aerosno_ssl < -puny) | (aerosno_int < -puny))
        return updated, new_flux, snow, atm, failed

    return jax.lax.cond(dzssl_new < parameters["hs_ssl_min"],
                        thin_snow, snow_layers, None)


def update_aerosol(dt, meltt, melts, meltb, congel, snoice, fsnow,
                   aerosno, aeroice, aice_old, vice_old, vsno_old,
                   vicen, vsnon, aicen, faero_atm, faero_ocn,
                   options, parameters):
    """Move deposited aerosol between snow, ice, and the ocean.

    The second array axis is the surface scattering layer followed by the
    interior, as in ``icepack_aerosol.F90``.
    """
    puny = parameters["puny"]
    rhoi = parameters["rhoi"]
    rhos = parameters["rhos"]
    # Fortran defines these six scavenging fractions in DATA statements.
    kscav = jnp.asarray((0.03, 0.20, 0.02, 0.02, 0.01, 0.01),
                        dtype=aerosno.dtype)[:aerosno.shape[0]]
    kscavsi = kscav
    focn_old = faero_ocn
    aerotot0 = jnp.sum(aerosno + aeroice, axis=1)

    hs_old = vsno_old / aice_old
    hi_old = vice_old / aice_old
    dzssl = jnp.minimum(hs_old / (2.0 * options["nslyr"]),
                        parameters["hs_ssl"])
    dzssli = jnp.minimum(hi_old / (2.0 * options["nilyr"]),
                         parameters["hi_ssl"])
    dzint = hs_old - dzssl
    dzinti = hi_old - dzssli
    ar = 1.0 / jnp.where(aicen > 0.0, aicen, aice_old)
    hs = vsnon * ar
    hi = vicen * ar
    dhs_melts = -melts
    dhi_snoice = snoice
    dhs_snoice = dhi_snoice * rhoi / rhos
    dhi_meltt = -meltt
    dhi_meltb = -meltb
    dhi_congel = congel
    dhs_evap = hs - (hs_old + dhs_melts - dhs_snoice
                     + fsnow / rhos * dt)
    dhi_evap = hi - (hi_old + dhi_meltt + dhi_meltb
                     + dhi_congel + dhi_snoice)

    # Evaporation and basal ice growth change layer thickness without
    # removing aerosol mass.
    dzint = dzint + jnp.minimum(dzssl + dhs_evap, 0.0)
    dzinti = dzinti + jnp.minimum(dzssli + dhi_evap, 0.0)
    dzssl = jnp.maximum(dzssl + dhs_evap, 0.0)
    dzssli = jnp.maximum(dzssli + dhi_evap, 0.0)
    dzinti = dzinti + dhi_congel

    # Surface snow melt scavenges the two snow reservoirs.
    snow_melt = -dhs_melts > puny
    sloss1 = jnp.where(
        snow_melt & (dzssl > puny),
        kscav * aerosno[:, 0] * jnp.minimum(-dhs_melts, dzssl)
        / jnp.where(dzssl > puny, dzssl, 1.0), 0.0)
    aerosno = aerosno.at[:, 0].add(-sloss1)
    sloss2 = jnp.where(
        snow_melt & (dzint > puny),
        kscav * aerosno[:, 1] * jnp.maximum(-dhs_melts - dzssl, 0.0)
        / jnp.where(dzint > puny, dzint, 1.0), 0.0)
    aerosno = aerosno.at[:, 1].add(-sloss2)
    faero_ocn = faero_ocn + (sloss1 + sloss2) / dt
    dzint_new = dzint + jnp.minimum(dzssl + dhs_melts, 0.0)
    dzssl_new = jnp.maximum(dzssl + dhs_melts, 0.0)
    dzint = jnp.where(snow_melt, dzint_new, dzint)
    dzssl = jnp.where(snow_melt, dzssl_new, dzssl)
    snow_ssl_gone = snow_melt & (dzssl <= puny)
    aerosno = aerosno.at[:, 1].set(jnp.where(
        snow_ssl_gone, aerosno[:, 0] + aerosno[:, 1], aerosno[:, 1]))
    aerosno = aerosno.at[:, 0].set(jnp.where(
        snow_ssl_gone, 0.0, aerosno[:, 0]))
    snow_gone = snow_melt & (dzint <= puny)
    aeroice = aeroice.at[:, 0].add(jnp.where(
        snow_gone, aerosno[:, 0] + aerosno[:, 1], 0.0))
    aerosno = jnp.where(snow_gone, 0.0, aerosno)
    dzint = jnp.where(snow_gone, jnp.maximum(dzint, 0.0), dzint)

    # Surface ice melt scavenges ice aerosol.
    ice_top_melt = -dhi_meltt > puny
    sloss1 = jnp.where(
        ice_top_melt & (dzssli > puny),
        kscav * aeroice[:, 0] * jnp.minimum(-dhi_meltt, dzssli)
        / jnp.where(dzssli > puny, dzssli, 1.0), 0.0)
    aeroice = aeroice.at[:, 0].add(-sloss1)
    sloss2 = jnp.where(
        ice_top_melt & (dzinti > puny),
        kscav * aeroice[:, 1] * jnp.maximum(-dhi_meltt - dzssli, 0.0)
        / jnp.where(dzinti > puny, dzinti, 1.0), 0.0)
    aeroice = aeroice.at[:, 1].add(-sloss2)
    faero_ocn = faero_ocn + (sloss1 + sloss2) / dt
    dzinti_new = dzinti + jnp.minimum(dzssli + dhi_meltt, 0.0)
    dzssli_new = jnp.maximum(dzssli + dhi_meltt, 0.0)
    dzinti = jnp.where(ice_top_melt, dzinti_new, dzinti)
    dzssli = jnp.where(ice_top_melt, dzssli_new, dzssli)
    ice_ssl_gone = ice_top_melt & (dzssli <= puny)
    aeroice = aeroice.at[:, 1].set(jnp.where(
        ice_ssl_gone, aeroice[:, 0] + aeroice[:, 1], aeroice[:, 1]))
    aeroice = aeroice.at[:, 0].set(jnp.where(
        ice_ssl_gone, 0.0, aeroice[:, 0]))
    ice_gone = ice_top_melt & (dzinti <= puny)
    faero_ocn = faero_ocn + jnp.where(
        ice_gone, (aeroice[:, 0] + aeroice[:, 1]) / dt, 0.0)
    aeroice = jnp.where(ice_gone, 0.0, aeroice)
    dzinti = jnp.where(ice_gone, jnp.maximum(dzinti, 0.0), dzinti)

    # Basal melt releases all aerosol in the melted portion to the ocean.
    bottom_melt = -dhi_meltb > puny
    sloss1 = jnp.where(
        bottom_melt & (dzssli > puny),
        jnp.maximum(-dhi_meltb - dzinti, 0.0) * aeroice[:, 0]
        / jnp.where(dzssli > puny, dzssli, 1.0), 0.0)
    aeroice = aeroice.at[:, 0].add(-sloss1)
    sloss2 = jnp.where(
        bottom_melt & (dzinti > puny),
        jnp.minimum(-dhi_meltb, dzinti) * aeroice[:, 1]
        / jnp.where(dzinti > puny, dzinti, 1.0), 0.0)
    aeroice = aeroice.at[:, 1].add(-sloss2)
    faero_ocn = faero_ocn + (sloss1 + sloss2) / dt
    dzssli = jnp.where(
        bottom_melt, dzssli + jnp.minimum(dzinti + dhi_meltb, 0.0), dzssli)
    dzinti = jnp.where(
        bottom_melt, jnp.maximum(dzinti + dhi_meltb, 0.0), dzinti)

    dzssl = dzssl + jnp.where(fsnow > 0.0, fsnow / rhos * dt, 0.0)

    # Snow-ice formation transfers snow aerosol into the new ice surface.
    snow_ice = dhs_snoice > puny
    sloss2 = jnp.where(
        snow_ice & (dzint > puny),
        jnp.minimum(dhs_snoice, dzint) * aerosno[:, 1]
        / jnp.where(dzint > puny, dzint, 1.0), 0.0)
    aerosno = aerosno.at[:, 1].add(-sloss2)
    sloss1 = jnp.where(
        snow_ice & (dzssl > puny),
        jnp.maximum(dhs_snoice - dzint, 0.0) * aerosno[:, 0]
        / jnp.where(dzssl > puny, dzssl, 1.0), 0.0)
    aerosno = aerosno.at[:, 0].add(-sloss1)
    transfer = sloss1 + sloss2
    aeroice = aeroice.at[:, 0].add((1.0 - kscavsi) * transfer)
    faero_ocn = faero_ocn + kscavsi * transfer / dt
    dzssl = dzssl - jnp.where(
        snow_ice, jnp.maximum(dhs_snoice - dzint, 0.0), 0.0)
    dzint = jnp.where(snow_ice, jnp.maximum(dzint - dhs_snoice, 0.0),
                      dzint)
    dzssli = dzssli + jnp.where(snow_ice, dhi_snoice, 0.0)

    # Atmospheric deposition goes to snow when it exceeds hs_min.
    hs = jnp.where(aicen > 0.0, vsnon * ar, 0.0)
    to_snow = hs > parameters["hs_min"]
    aerosno = aerosno.at[:, 0].add(jnp.where(
        to_snow, faero_atm * dt * aicen, 0.0))
    aeroice = aeroice.at[:, 0].add(jnp.where(
        to_snow, 0.0, faero_atm * dt * aicen))

    # Empty vertical reservoirs pass their mass down to the next reservoir.
    hs = jnp.where(aicen > 0.0, vsnon * ar, 0.0)
    hi = jnp.where(aicen > 0.0, vicen * ar, 0.0)
    snow_ssl_empty = dzssl <= puny
    aerosno = aerosno.at[:, 1].set(jnp.where(
        snow_ssl_empty, aerosno[:, 1] + aerosno[:, 0], aerosno[:, 1]))
    aerosno = aerosno.at[:, 0].set(jnp.where(
        snow_ssl_empty, 0.0, aerosno[:, 0]))
    snow_int_empty = dzint <= puny
    aeroice = aeroice.at[:, 0].add(jnp.where(
        snow_int_empty, aerosno[:, 1], 0.0))
    aerosno = aerosno.at[:, 1].set(jnp.where(
        snow_int_empty, 0.0, aerosno[:, 1]))
    ice_ssl_empty = dzssli <= puny
    aeroice = aeroice.at[:, 1].set(jnp.where(
        ice_ssl_empty, aeroice[:, 1] + aeroice[:, 0], aeroice[:, 1]))
    aeroice = aeroice.at[:, 0].set(jnp.where(
        ice_ssl_empty, 0.0, aeroice[:, 0]))
    ice_int_empty = dzinti <= puny
    faero_ocn = faero_ocn + jnp.where(
        ice_int_empty, (aeroice[:, 0] + aeroice[:, 1]) / dt, 0.0)
    aeroice = jnp.where(ice_int_empty, 0.0, aeroice)

    hslyr = hs / options["nslyr"]
    hilyr = hi / options["nilyr"]
    dzssl_new = jnp.minimum(hslyr / 2.0, parameters["hs_ssl"])
    dzssli_new = jnp.minimum(hilyr / 2.0, parameters["hi_ssl"])
    snow_present = hs > parameters["hs_min"]
    dznew = jnp.minimum(dzssl_new - dzssl, 0.0)
    sloss1 = jnp.where(
        snow_present & (dzssl > puny),
        dznew * aerosno[:, 0] / jnp.where(dzssl > puny, dzssl, 1.0), 0.0)
    dznew = jnp.maximum(dzssl_new - dzssl, 0.0)
    sloss1 = sloss1 + jnp.where(
        snow_present & (dzint > puny),
        aerosno[:, 1] * dznew / jnp.where(dzint > puny, dzint, 1.0), 0.0)
    aerosno = aerosno.at[:, 0].add(sloss1)
    aerosno = aerosno.at[:, 1].add(-sloss1)
    aeroice = aeroice.at[:, 0].add(jnp.where(
        snow_present, 0.0, aerosno[:, 0] + aerosno[:, 1]))
    aerosno = jnp.where(snow_present, aerosno, 0.0)

    ice_present = vicen > puny
    dznew = jnp.minimum(dzssli_new - dzssli, 0.0)
    sloss2 = jnp.where(
        ice_present & (dzssli > puny),
        dznew * aeroice[:, 0] / jnp.where(dzssli > puny, dzssli, 1.0), 0.0)
    dznew = jnp.maximum(dzssli_new - dzssli, 0.0)
    sloss2 = sloss2 + jnp.where(
        ice_present & (dzinti > puny),
        aeroice[:, 1] * dznew / jnp.where(dzinti > puny, dzinti, 1.0), 0.0)
    aeroice = aeroice.at[:, 0].add(sloss2)
    aeroice = aeroice.at[:, 1].add(-sloss2)
    faero_ocn = faero_ocn + jnp.where(
        ice_present, 0.0, (aeroice[:, 0] + aeroice[:, 1]) / dt)
    aeroice = jnp.where(ice_present, aeroice, 0.0)

    aerotot = jnp.sum(aerosno + aeroice, axis=1)
    conservation_error = ((aerotot - aerotot0)
                          - (faero_atm * aicen
                             - (faero_ocn - focn_old)) * dt > puny)
    # CICE reports and clips negative values for the first aerosol only.
    negative_first = jnp.any(aerosno[0] < -puny) | jnp.any(aeroice[0] < -puny)
    aerosno = aerosno.at[0].set(jnp.where(
        negative_first, jnp.maximum(aerosno[0], 0.0), aerosno[0]))
    aeroice = aeroice.at[0].set(jnp.where(
        negative_first, jnp.maximum(aeroice[0], 0.0), aeroice[0]))
    return aerosno, aeroice, faero_ocn, conservation_error, negative_first
