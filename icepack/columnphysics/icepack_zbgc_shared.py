"""JAX vertical biology-grid operations from icepack_zbgc_shared.F90."""

import jax
import jax.numpy as jnp


def remap_zbgc_jax(tracer, ice_grid, bio_grid, hice, hinS,
                   S_min, cice_to_bio=True):
    """Linearly remap a profile between ice and biology layer centers.

    ``tracer`` contains only the donor layers. The two end values are
    extended exactly as in the Fortran routine before interpolation.
    """
    top = jnp.minimum(jnp.minimum(-hice + hinS, -hinS + hice), 0.0)
    if cice_to_bio:
        donor_values = jnp.concatenate((tracer[:1], tracer, tracer[-1:]))

        def brine_within_ice(_):
            donor_grid = jnp.concatenate((
                jnp.asarray([top]), (ice_grid - 1.0) * hice + hinS,
                jnp.asarray([jnp.minimum(hinS, hice)])))
            receiver_grid = bio_grid * hinS
            return jnp.interp(receiver_grid, donor_grid, donor_values)

        def brine_above_ice(_):
            donor_grid = jnp.concatenate((
                jnp.asarray([top]), ice_grid * hice,
                jnp.asarray([jnp.minimum(hinS, hice)])))
            receiver_grid = (bio_grid - 1.0) * hinS + hice
            return jnp.interp(receiver_grid, donor_grid, donor_values)

        return jax.lax.cond(hice > hinS, brine_within_ice,
                            brine_above_ice, operand=None)

    def brine_within_ice(_):
        donor_grid = jnp.concatenate((
            jnp.asarray([-hice + hinS, 0.5 * (hinS - hice)]),
            bio_grid * hinS, jnp.asarray([hinS])))
        donor_values = jnp.concatenate((
            jnp.asarray([S_min, S_min]), tracer, tracer[-1:]))
        transition_width = 0.5 * (donor_grid[3] - donor_grid[2])
        transition = ((S_min * (hice - hinS)
                       + donor_values[2] * transition_width)
                      / (hice - hinS + transition_width))
        donor_values = donor_values.at[:3].set(transition)
        receiver_grid = (ice_grid - 1.0) * hice + hinS
        return jnp.interp(receiver_grid, donor_grid, donor_values)

    def brine_above_ice(_):
        donor_grid = jnp.concatenate((
            jnp.asarray([hice - hinS]),
            (bio_grid - 1.0) * hinS + hice,
            jnp.asarray([hice])))
        donor_values = jnp.concatenate((tracer[:1], tracer, tracer[-1:]))
        return jnp.interp(ice_grid * hice, donor_grid, donor_values)

    return jax.lax.cond(hice > hinS, brine_within_ice,
                        brine_above_ice, operand=None)


def merge_bgc_fluxes_skl(aicen, trcrn, flux_bion, flux_bio,
                         PP_net, upNOn, upNHn, upNO, upNH,
                         grow_net, grow_alg, nt_bgc_N, R_C2N,
                         phi_sk, sk_l, secday, R_gC2molC,
                         fr_resp, puny):
    """Accumulate skeletal category biology into the grid cell budgets."""
    flux_bio = flux_bio + flux_bion * aicen
    for mm in range(len(nt_bgc_N)):
        tmp = phi_sk * sk_l * aicen * secday
        PP_net = (PP_net + grow_alg[mm] * tmp * R_C2N[mm]
                  * R_gC2molC * (1.0 - fr_resp))
        grow_net = (grow_net + grow_alg[mm] * tmp
                    / (trcrn[nt_bgc_N[mm]] + puny))
        upNO = upNO + upNOn[mm] * tmp
        upNH = upNH + upNHn[mm] * tmp
    return flux_bio, PP_net, upNO, upNH, grow_net


def regrid_stationary(C_stationary, hbri_old, hbri, dt,
                      top_conc, igrid, flux_bio, puny,
                      melt_b=0.0, con_gel=0.0):
    """Move a stationary biological tracer as the brine column changes."""
    nblyr = C_stationary.shape[0] - 1
    zspace = jnp.full((nblyr + 1,), 1.0 / nblyr,
                      dtype=C_stationary.dtype)
    zspace = zspace.at[0].set(0.5 / nblyr)
    zspace = zspace.at[-1].set(0.5 / nblyr)
    sum_i = jnp.sum(C_stationary * zspace)
    donor = jnp.where(hbri_old > 0.0,
                      C_stationary[::-1]
                      / jnp.where(hbri_old > 0.0, hbri_old, 1.0),
                      0.0)
    melting = melt_b > 0.0
    freezing = ~melting & (con_gel > 0.0)
    rising = ~melting & ~freezing & (hbri > hbri_old)
    htemp = jnp.where(melting, hbri_old - melt_b,
                      jnp.where(freezing, hbri_old + con_gel,
                                jnp.where(rising, hbri, 0.0)))

    def move_profile(_):
        def melt_profile(_):
            return remap_zbgc_jax(donor, igrid, igrid,
                                  hbri_old, htemp, top_conc,
                                  cice_to_bio=True)

        def grow_profile(_):
            return remap_zbgc_jax(donor, igrid, igrid,
                                  htemp, hbri_old, top_conc,
                                  cice_to_bio=False)

        remapped = jax.lax.cond(melting, melt_profile, grow_profile, None)
        updated = remapped[::-1] * htemp
        sum_f = jnp.sum(updated * zspace)
        correct = ((con_gel > 0.0) & (top_conc <= 0.0)
                   & (jnp.abs(sum_i - sum_f) > puny))
        corrected = jnp.maximum(0.0, updated + (sum_i - sum_f))
        updated = jnp.where(correct, corrected, updated)
        sum_f = jnp.where(correct, jnp.sum(corrected * zspace), sum_f)
        return updated, flux_bio + (sum_i - sum_f) / dt

    return jax.lax.cond(htemp > 0.0, move_profile,
                        lambda _: (C_stationary, flux_bio), None)


def merge_bgc_fluxes(dt, aicen, vicen, vsnon, iphin, trcrn,
                     flux_bion, flux_bio, upNOn, upNHn,
                     upNO, upNH, zbgc_snown, zbgc_atmn,
                     zbgc_snow, zbgc_atm, PP_net,
                     ice_bio_net, snow_bio_net, grow_alg,
                     grow_net, totalChla, iTin, iSin,
                     bioPorosityIceCell, bioSalinityIceCell,
                     bioTemperatureIceCell, bio_index,
                     nt_fbri, nt_bgc_N, nlt_bgc_N,
                     R_chl2N, R_C2N, nslyr, solve_zbgc,
                     secday, fr_resp, R_gC2molC, puny):
    """Sum category biology inventories, fluxes and production diagnostics.

    ``bio_index``, ``nt_bgc_N`` and ``nlt_bgc_N`` are zero-based here.
    """
    nblyr = iphin.shape[0] - 1
    zspace = jnp.full((nblyr + 1,), 1.0 / nblyr, dtype=trcrn.dtype)
    zspace = zspace.at[0].set(0.5 / nblyr)
    zspace = zspace.at[-1].set(0.5 / nblyr)
    brine_volume = trcrn[nt_fbri] * vicen
    for mm in range(len(bio_index)):
        slot = bio_index[mm]
        ice_bio_net = ice_bio_net.at[mm].add(
            jnp.sum(trcrn[slot:slot + nblyr + 1] * zspace)
            * brine_volume)
        dvssl = 0.5 * vsnon / nslyr
        dvint = vsnon - dvssl
        snow_bio_net = snow_bio_net.at[mm].add(
            trcrn[slot + nblyr + 1] * dvssl
            + trcrn[slot + nblyr + 2] * dvint)
        flux_bio = flux_bio.at[mm].add(flux_bion[mm] * aicen)
        zbgc_snow = zbgc_snow.at[mm].add(zbgc_snown[mm] * aicen / dt)
        zbgc_atm = zbgc_atm.at[mm].add(zbgc_atmn[mm] * aicen / dt)

    bioPorosityIceCell = bioPorosityIceCell + iphin * vicen
    bioSalinityIceCell = bioSalinityIceCell + iSin * vicen
    bioTemperatureIceCell = bioTemperatureIceCell + iTin * vicen
    if solve_zbgc:
        for mm in range(len(nt_bgc_N)):
            totalChla = totalChla + ice_bio_net[nlt_bgc_N[mm]] * R_chl2N[mm]
            for k in range(nblyr + 1):
                tmp = iphin[k] * brine_volume * zspace[k] * secday
                PP_net = (PP_net + grow_alg[k, mm] * tmp
                          * (1.0 - fr_resp) * R_C2N[mm] * R_gC2molC)
                grow_net = (grow_net + grow_alg[k, mm] * tmp
                            / (trcrn[nt_bgc_N[mm] + k] + puny))
                upNO = upNO + upNOn[k, mm] * tmp
                upNH = upNH + upNHn[k, mm] * tmp
    return (flux_bio, zbgc_snow, zbgc_atm, PP_net,
            ice_bio_net, snow_bio_net, grow_net, totalChla,
            upNO, upNH, bioPorosityIceCell,
            bioSalinityIceCell, bioTemperatureIceCell)
