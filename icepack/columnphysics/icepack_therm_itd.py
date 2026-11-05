"""Thermodynamic ice-thickness distribution routines from CICE."""

from functools import partial

import jax
import jax.numpy as jnp
from jax import lax

from icepack.columnphysics.icepack_itd import (
    cleanup_itd_jax, column_sum, reduce_area, shift_ice_jax,
)
from icepack.columnphysics.icepack_fsd import (
    fsd_add_new_ice_jax, fsd_lateral_growth_jax, fsd_weld_thermo_jax,
    get_subdt_fsd_jax, icepack_cleanup_fsd_jax,
)
from icepack.columnphysics.icepack_mushy_physics import (
    icepack_enthalpy_mush_jax,
    liquidus_temperature_mush_jax,
)
from icepack.columnphysics.icepack_isotope import isoice_alpha
from icepack.columnphysics.icepack_zbgc import (
    add_new_ice_bgc, lateral_melt_bgc,
)


def fit_line(aicen, hice, hbL, hbR, puny):
    """Fit linear category ice-thickness distribution g(eta)."""
    active = (aicen > puny) & (hbR - hbL > puny)
    hL = jnp.where(active, hbL, 0.0)
    hR = jnp.where(active, hbR, 0.0)
    h13 = (1.0 / 3.0) * (2.0 * hL + hR)
    h23 = (1.0 / 3.0) * (hL + 2.0 * hR)
    hR = jnp.where(active & (hice < h13), 3.0 * hice - 2.0 * hL, hR)
    hL = jnp.where(active & (hice > h23), 3.0 * hice - 2.0 * hR, hL)

    dhr = 1.0 / jnp.where(active, hR - hL, 1.0)
    wk1 = 6.0 * aicen * dhr
    wk2 = (hice - hL) * dhr
    g0 = jnp.where(active, wk1 * (2.0 / 3.0 - wk2), 0.0)
    g1 = jnp.where(active, 2.0 * dhr * wk1 * (wk2 - 0.5), 0.0)
    return g0, g1, hL, hR


def update_vertical_tracers(trc, h1, h2, trc0):
    """Conservatively remap old tracer layers after ice grows at the base.

    The last axis contains ``nilyr`` equal-thickness ice layers. The new
    bottom segment has tracer value ``trc0``. Each new layer receives the
    overlap-weighted old tracer and new-bottom contributions.
    """
    nilyr = trc.shape[-1]
    output = jnp.zeros_like(trc)
    for k2 in range(nilyr):
        z2a = k2 * h2 / nilyr
        z2b = (k2 + 1) * h2 / nilyr
        updated = jnp.zeros_like(h2)
        for k1 in range(nilyr):
            z1a = k1 * h1 / nilyr
            z1b = (k1 + 1) * h1 / nilyr
            overlap = jnp.maximum(jnp.minimum(z1b, z2b) - jnp.maximum(z1a, z2a), 0.0)
            updated = updated + overlap * trc[..., k1]
        overlap = jnp.maximum(jnp.minimum(h2, z2b) - jnp.maximum(h1, z2a), 0.0)
        updated = updated + overlap * trc0
        output = output.at[..., k2].set(nilyr * updated / h2)
    return output


def linear_itd(hin_max, aicen_init, vicen_init, aicen, trcrn,
               vicen, vsnon, aice, aice0, fpond, Tf, dpnd_melt,
               layout, flags, options, parameters):
    """Remap the linear ice-thickness distribution after vertical growth."""
    return _linear_itd(
        hin_max, aicen_init, vicen_init, aicen, trcrn, vicen, vsnon,
        aice, aice0, fpond, Tf, dpnd_melt,
        tuple(sorted(layout.items())), tuple(sorted(flags.items())),
        tuple(sorted(options.items())), parameters)


@partial(jax.jit, static_argnames=("layout", "flags", "options"))
def _linear_itd(hin_max, aicen_init, vicen_init, aicen, trcrn,
                vicen, vsnon, aice, aice0, fpond, Tf, dpnd_melt,
                layout, flags, options, parameters):
    layout = dict(layout)
    flags = dict(flags)
    options = dict(options)
    ncat = aicen.shape[0]
    puny = parameters["puny"]
    nilyr = int(options["nilyr"])
    nslyr = int(options["nslyr"])
    nt_qice = int(flags["nt_qice"])
    nt_qsno = int(flags["nt_qsno"])
    nt_sice = int(flags["nt_sice"])

    if bool(options["conserv_check"]):
        energy_ice = jnp.zeros_like(aicen)
        energy_snow = jnp.zeros_like(aicen)
        salt_ice = jnp.zeros_like(aicen)
        for k in range(nilyr):
            energy_ice = energy_ice + trcrn[nt_qice + k] * vicen / nilyr
            salt_ice = salt_ice + trcrn[nt_sice + k] * vicen / nilyr
        for k in range(nslyr):
            energy_snow = energy_snow + trcrn[nt_qsno + k] * vsnon / nslyr
        brine = (trcrn[int(flags["nt_fbri"])] * vicen
                 if bool(flags["tr_brine"]) else jnp.zeros_like(vicen))
        initial_budget = tuple(column_sum(ncat, quantity) for quantity in (
            vicen, vsnon, energy_ice, energy_snow, salt_ice, brine))
    initial_thickness = jnp.where(
        aicen_init > puny,
        vicen_init / jnp.where(aicen_init > puny, aicen_init, 1.0), 0.0)
    thickness = jnp.where(
        aicen > puny, vicen / jnp.where(aicen > puny, aicen, 1.0), 0.0)
    change = jnp.where(aicen > puny, thickness - initial_thickness, 0.0)
    hbnew = jnp.zeros((ncat + 1,), dtype=aicen.dtype).at[0].set(hin_max[0])
    remap = jnp.asarray(True)
    failed = jnp.asarray(False)
    for n in range(ncat - 1):
        both = (initial_thickness[n] > puny) & (initial_thickness[n + 1] > puny)
        ordering = initial_thickness[n + 1] > initial_thickness[n]
        slope = ((change[n + 1] - change[n]) /
                 jnp.where(ordering,
                           initial_thickness[n + 1] - initial_thickness[n],
                           1.0))
        new_boundary = jnp.where(
            both,
            hin_max[n + 1] + change[n] + slope * (
                hin_max[n + 1] - initial_thickness[n]),
            jnp.where(initial_thickness[n] > puny,
                      hin_max[n + 1] + change[n],
                      jnp.where(initial_thickness[n + 1] > puny,
                                hin_max[n + 1] + change[n + 1],
                                hin_max[n + 1])))
        hbnew = hbnew.at[n + 1].set(new_boundary)
        failed = failed | (both & ~ordering)
        remap = remap & ~((aicen[n] > puny) &
                          (thickness[n] >= new_boundary))
        remap = remap & ~((aicen[n + 1] > puny) &
                          (thickness[n + 1] <= new_boundary))
        remap = remap & (new_boundary <= hin_max[n + 2])
        remap = remap & (new_boundary >= hin_max[n])
    highest = jnp.where(
        aicen[-1] > puny,
        3.0 * thickness[-1] - 2.0 * hbnew[-2],
        hin_max[-1])
    hbnew = hbnew.at[-1].set(jnp.maximum(highest, hin_max[-2]))

    def remap_categories(state):
        (aicen, trcrn, vicen, vsnon, aice, aice0,
         fpond, dpnd_melt, thickness, failed) = state
        g0, g1, hL, hR = fit_line(
            aicen[0], initial_thickness[0], hbnew[0], hin_max[1], puny)
        melting = (aicen[0] > puny) & (change[0] < 0.0)
        melt_width = jnp.minimum(-change[0], hin_max[1])
        etamax = jnp.minimum(melt_width, hR) - hL
        x2 = 0.5 * etamax * etamax
        integrated_loss = g1 * x2 + g0 * etamax
        maximum_loss = aicen[0] * (
            1.0 - thickness[0] /
            jnp.where(initial_thickness[0] > 0.0,
                      initial_thickness[0], 1.0))
        # Fortran applies damax only inside the melting/etamax>0 branch.
        area_loss = jnp.where(
            melting & (etamax > 0.0),
            jnp.minimum(integrated_loss, maximum_loss), 0.0)
        previous_area = aicen[0]
        aicen = aicen.at[0].add(-area_loss)
        thickness = thickness.at[0].set(jnp.where(
            area_loss > 0.0,
            thickness[0] * previous_area /
            jnp.where(aicen[0] > 0.0, aicen[0], 1.0), thickness[0]))
        if bool(flags["tr_pond_topo"]):
            fpond = fpond - area_loss * trcrn[
                int(flags["nt_apnd"]), 0] * trcrn[int(flags["nt_hpnd"]), 0]
        if bool(flags["tr_pond"]):
            lost_pond = (area_loss * trcrn[int(flags["nt_apnd"]), 0]
                         * trcrn[int(flags["nt_hpnd"]), 0])
            if bool(flags["tr_pond_lvl"]):
                lost_pond = lost_pond * trcrn[int(flags["nt_alvl"]), 0]
            dpnd_melt = dpnd_melt + lost_pond
        left_boundary = jnp.where(
            (aicen[0] > puny) & (change[0] >= 0.0),
            jnp.minimum(change[0], hin_max[1]), hbnew[0])
        hb = hbnew.at[0].set(left_boundary)
        g0n, g1n, hLn, hRn = fit_line(
            aicen, thickness, hb[:-1], hb[1:], puny)
        donor = jnp.full((ncat,), -1, dtype=jnp.int32)
        daice = jnp.zeros_like(aicen)
        dvice = jnp.zeros_like(vicen)
        for n in range(ncat - 1):
            to_right = hb[n + 1] > hin_max[n + 1]
            nd = jnp.where(to_right, n, n + 1)
            lower = jnp.where(
                to_right,
                jnp.maximum(hin_max[n + 1], hLn[n]) - hLn[n], 0.0)
            upper = jnp.where(
                to_right,
                jnp.minimum(hb[n + 1], hRn[n]) - hLn[n],
                jnp.minimum(hin_max[n + 1], hRn[n + 1]) - hLn[n + 1])
            x1 = upper - lower
            wk1 = lower * lower
            wk2 = upper * upper
            x2 = 0.5 * (wk2 - wk1)
            wk1 = wk1 * lower
            wk2 = wk2 * upper
            x3 = (1.0 / 3.0) * (wk2 - wk1)
            area_shift = g1n[nd] * x2 + g0n[nd] * x1
            volume_shift = (g1n[nd] * x3 + g0n[nd] * x2
                            + area_shift * hLn[nd])
            valid = upper > lower
            area_shift = jnp.where(valid, area_shift, 0.0)
            volume_shift = jnp.where(valid, volume_shift, 0.0)
            valid = valid & (area_shift >= aicen[nd] * puny)
            valid = valid & (volume_shift >= vicen[nd] * puny)
            whole = ((area_shift > aicen[nd] * (1.0 - puny))
                     | (volume_shift > vicen[nd] * (1.0 - puny)))
            area_shift = jnp.where(whole, aicen[nd], area_shift)
            volume_shift = jnp.where(whole, vicen[nd], volume_shift)
            donor = donor.at[n].set(jnp.where(valid, nd, -1))
            daice = daice.at[n].set(jnp.where(valid, area_shift, 0.0))
            dvice = dvice.at[n].set(jnp.where(valid, volume_shift, 0.0))

        # Snow enthalpy and comparative snow density remain positive while
        # weighted tracers are transferred across thickness boundaries.
        trcrn = trcrn.at[nt_qsno:nt_qsno + nslyr].add(
            parameters["rhos"] * parameters["Lfresh"])
        if str(options["snwredist"])[:3] == "ITD":
            start = int(flags["nt_rhos"])
            trcrn = trcrn.at[start:start + nslyr].set(jnp.maximum(
                trcrn[start:start + nslyr] - parameters["rhosmin"], 0.0))
        (aicen, trcrn, vicen, vsnon, thickness,
         shift_failed) = shift_ice_jax(
             aicen, trcrn, vicen, vsnon, thickness,
             donor, daice, dvice, Tf, layout, flags, options, parameters)
        trcrn = trcrn.at[nt_qsno:nt_qsno + nslyr].add(
            -parameters["rhos"] * parameters["Lfresh"])
        if str(options["snwredist"])[:3] == "ITD":
            start = int(flags["nt_rhos"])
            trcrn = trcrn.at[start:start + nslyr].add(
                parameters["rhosmin"])
        thin = ((parameters["hi_min"] > 0.0) & (aicen[0] > puny)
                & (thickness[0] < parameters["hi_min"]))
        extra_loss = jnp.where(
            thin, aicen[0] * (1.0 - thickness[0] /
                              jnp.where(parameters["hi_min"] > 0.0,
                                        parameters["hi_min"], 1.0)), 0.0)
        aicen = aicen.at[0].add(-extra_loss)
        if bool(flags["tr_pond_topo"]):
            fpond = fpond - extra_loss * trcrn[
                int(flags["nt_apnd"]), 0] * trcrn[int(flags["nt_hpnd"]), 0]
        aice = column_sum(ncat, aicen)
        aice0 = jnp.maximum(1.0 - aice, 0.0)
        return (aicen, trcrn, vicen, vsnon, aice, aice0,
                fpond, dpnd_melt, thickness, failed | shift_failed)

    state = (aicen, trcrn, vicen, vsnon, aice, aice0,
             fpond, dpnd_melt, thickness, failed)
    state = lax.cond(remap, remap_categories, lambda state: state, state)
    (aicen, trcrn, vicen, vsnon, aice, aice0,
     fpond, dpnd_melt, _, failed) = state
    aice = column_sum(ncat, aicen)
    aice0 = jnp.maximum(1.0 - aice, 0.0)
    if bool(options["conserv_check"]):
        energy_ice = jnp.zeros_like(aicen)
        energy_snow = jnp.zeros_like(aicen)
        salt_ice = jnp.zeros_like(aicen)
        for k in range(nilyr):
            energy_ice = energy_ice + trcrn[nt_qice + k] * vicen / nilyr
            salt_ice = salt_ice + trcrn[nt_sice + k] * vicen / nilyr
        for k in range(nslyr):
            energy_snow = energy_snow + trcrn[nt_qsno + k] * vsnon / nslyr
        brine = (trcrn[int(flags["nt_fbri"])] * vicen
                 if bool(flags["tr_brine"]) else jnp.zeros_like(vicen))
        final_budget = tuple(column_sum(ncat, quantity) for quantity in (
            vicen, vsnon, energy_ice, energy_snow, salt_ice, brine))
        for before, after, tolerance in zip(
            initial_budget, final_budget,
            (puny, puny, puny * parameters["Lfresh"] * parameters["rhoi"],
             puny * parameters["Lfresh"] * parameters["rhos"],
             puny, puny * 10.0),
        ):
            failed = failed | (jnp.abs(after - before) > tolerance)
    return (aicen, trcrn, vicen, vsnon, aice, aice0,
            fpond, dpnd_melt, failed)


def lateral_melt(dt, fpond, fresh, fsalt, fhocn, faero_ocn,
                 fiso_ocn, rsiden, meltl, wlat, aicen, vicen,
                 vsnon, trcrn, flux_bio, d_afsd_latm, dpnd_melt,
                 flags, options, parameters, bio_index=(),
                 floe_rad_c=None, floe_binwidth=None):
    """Melt each category from the side and send its contents to the ocean."""
    return _lateral_melt(
        dt, fpond, fresh, fsalt, fhocn, faero_ocn,
        fiso_ocn, rsiden, meltl, wlat, aicen, vicen,
        vsnon, trcrn, flux_bio, d_afsd_latm, dpnd_melt,
        tuple(sorted(flags.items())), tuple(sorted(options.items())), parameters,
        tuple(bio_index), floe_rad_c, floe_binwidth)


@partial(jax.jit, static_argnames=("flags", "options", "bio_index"))
def _lateral_melt(dt, fpond, fresh, fsalt, fhocn, faero_ocn,
                  fiso_ocn, rsiden, meltl, wlat, aicen, vicen,
                  vsnon, trcrn, flux_bio, d_afsd_latm, dpnd_melt,
                  flags, options, parameters, bio_index=(),
                  floe_rad_c=None, floe_binwidth=None):
    flags = dict(flags)
    options = dict(options)
    ncat = aicen.shape[0]
    puny = parameters["puny"]
    nilyr = int(options["nilyr"])
    nslyr = int(options["nslyr"])
    old_ice = vicen
    old_snow = vsnon
    old_area = aicen
    initial_trcrn = trcrn
    any_lateral_melt = jnp.any(rsiden > 0.0)
    if bool(flags["tr_fsd"]):
        d_afsd_latm = jnp.zeros_like(d_afsd_latm)
        fsd_start = int(flags["nt_fsd"])
        nfsd = int(options["nfsd"])
        afsdn = icepack_cleanup_fsd_jax(
            trcrn[fsd_start:fsd_start + nfsd], puny)
        original_fsd = afsdn
    for n in range(ncat):
        fraction = rsiden[n]
        fresh = fresh + (parameters["rhoi"] * vicen[n]
                         + parameters["rhos"] * vsnon[n]) * fraction / dt
        if options["saltflux_option"] == "prognostic":
            salinity = jnp.zeros_like(fsalt)
            for k in range(nilyr):
                salinity = salinity + trcrn[int(flags["nt_sice"]) + k, n] / nilyr
        else:
            salinity = parameters["ice_ref_salinity"]
        fsalt = (fsalt + parameters["rhoi"] * vicen[n]
                 * salinity * parameters["p001"] * fraction / dt)
        if bool(flags["tr_pond"]):
            pond_volume = (aicen[n] * trcrn[int(flags["nt_apnd"]), n]
                           * trcrn[int(flags["nt_hpnd"]), n] * fraction)
            if bool(flags["tr_pond_topo"]):
                fpond = fpond - pond_volume
            if bool(flags["tr_pond_lvl"]):
                pond_volume = pond_volume * trcrn[int(flags["nt_alvl"]), n]
            dpnd_melt = dpnd_melt + pond_volume
        meltl = meltl + old_ice[n] * fraction
        aicen = aicen.at[n].set(old_area[n] * (1.0 - fraction))
        vicen = vicen.at[n].set(old_ice[n] * (1.0 - fraction))
        vsnon = vsnon.at[n].set(old_snow[n] * (1.0 - fraction))
        if bool(flags["tr_fsd"]):
            active = (fraction > puny) & (aicen[n] > puny)
            elapsed = jnp.zeros_like(dt)
            distribution = afsdn[:, n]

            def advance_fsd(_, carry):
                distribution, elapsed = carry
                flux = jnp.zeros((nfsd + 1,), dtype=trcrn.dtype)
                for k in range(1, nfsd):
                    flux = flux.at[k].set(
                        -wlat * distribution[k] / floe_binwidth[k])
                tendency = jnp.zeros_like(distribution)
                radial_weight = jnp.sum(distribution / floe_rad_c)
                for k in range(nfsd):
                    tendency = tendency.at[k].set(
                        flux[k] - flux[k + 1]
                        - 2.0 * wlat * distribution[k]
                        * (1.0 / floe_rad_c[k] - radial_weight))
                subdt = jnp.minimum(
                    get_subdt_fsd_jax(distribution, tendency, puny,
                                      parameters["bignum"]),
                    dt - elapsed)
                advancing = active & (elapsed < dt) & (subdt > 0.0)
                distribution = jnp.where(
                    advancing, distribution + subdt * tendency, distribution)
                elapsed = jnp.where(advancing, elapsed + subdt, elapsed)
                return distribution, elapsed

            distribution, elapsed = lax.fori_loop(
                0, 100, advance_fsd, (distribution, elapsed))
            afsdn = afsdn.at[:, n].set(distribution)
        for k in range(nilyr):
            fhocn = (fhocn + trcrn[int(flags["nt_qice"]) + k, n]
                     * fraction / dt * old_ice[n] / nilyr)
        for k in range(nslyr):
            fhocn = (fhocn + trcrn[int(flags["nt_qsno"]) + k, n]
                     * fraction / dt * old_snow[n] / nslyr)
        if bool(flags["tr_aero"]):
            for k in range(int(flags["n_aero"])):
                start = int(flags["nt_aero"]) + 4 * k
                amount = (old_snow[n] * (trcrn[start, n]
                                         + trcrn[start + 1, n])
                          + old_ice[n] * (trcrn[start + 2, n]
                                          + trcrn[start + 3, n]))
                faero_ocn = faero_ocn.at[k].add(amount * fraction / dt)
        if bool(flags["tr_iso"]):
            for k in range(int(flags["n_iso"])):
                amount = (old_snow[n] * trcrn[int(flags["nt_isosno"]) + k, n]
                          + old_ice[n] * trcrn[int(flags["nt_isoice"]) + k, n])
                fiso_ocn = fiso_ocn.at[k].add(amount * fraction / dt)
        if bool(options["z_tracers"]):
            snow_surface = 0.5 * old_snow[n] / nslyr
            snow_interior = old_snow[n] - snow_surface
            for k in range(int(flags["nbtrcr"])):
                start = int(bio_index[k])
                amount = (trcrn[start + int(options["nblyr"]) + 1, n]
                          * snow_surface
                          + trcrn[start + int(options["nblyr"]) + 2, n]
                          * snow_interior)
                flux_bio = flux_bio.at[k].add(amount * fraction / dt)
    if bool(flags["tr_fsd"]):
        afsdn = icepack_cleanup_fsd_jax(afsdn, puny)
        trcrn = trcrn.at[fsd_start:fsd_start + nfsd].set(afsdn)
        d_afsd_latm = jnp.sum(
            afsdn * aicen[None, :] - original_fsd * old_area[None, :],
            axis=1)
        trcrn = jnp.where(any_lateral_melt, trcrn, initial_trcrn)
        d_afsd_latm = jnp.where(any_lateral_melt, d_afsd_latm, 0.0)
    if bool(options["z_tracers"]):
        flux_bio = lateral_melt_bgc(
            dt, rsiden, old_ice, trcrn, flux_bio,
            bio_index, flags, options)
    return (fpond, fresh, fsalt, fhocn, faero_ocn, fiso_ocn,
            meltl, aicen, vicen, vsnon, trcrn, flux_bio,
            d_afsd_latm, dpnd_melt)


def add_new_ice(dt, hin_max, aicen, trcrn, vicen, aice0, aice,
                frzmlt, frazil, frz_onset, yday, fresh, fsalt,
                Tf, sss, salinz, flux_bio, ocean_bio, frazil_diag,
                fiso_ocn, HDO_ocn, H2_16O_ocn, H2_18O_ocn,
                flags, options, parameters, wave_sig_ht=None,
                wave_spectrum=None, wavefreq=None,
                d_afsd_latg=None, d_afsd_newi=None,
                floe_rad_c=None, floe_rad_h=None,
                floe_binwidth=None, bio_index=(),
                zbgc_init_frac=None):
    """Form frazil ice from the ocean freezing potential and place it in ITD."""
    return _add_new_ice(
        dt, hin_max, aicen, trcrn, vicen, aice0, aice,
        frzmlt, frazil, frz_onset, yday, fresh, fsalt,
        Tf, sss, salinz, flux_bio, ocean_bio, frazil_diag,
        fiso_ocn, HDO_ocn, H2_16O_ocn, H2_18O_ocn,
        tuple(sorted(flags.items())), tuple(sorted(options.items())), parameters,
        wave_sig_ht, wave_spectrum, wavefreq, d_afsd_latg, d_afsd_newi,
        floe_rad_c, floe_rad_h, floe_binwidth, tuple(bio_index), zbgc_init_frac)


@partial(jax.jit, static_argnames=("flags", "options", "bio_index"))
def _add_new_ice(dt, hin_max, aicen, trcrn, vicen, aice0, aice,
                 frzmlt, frazil, frz_onset, yday, fresh, fsalt,
                 Tf, sss, salinz, flux_bio, ocean_bio, frazil_diag,
                 fiso_ocn, HDO_ocn, H2_16O_ocn, H2_18O_ocn,
                 flags, options, parameters, wave_sig_ht=None,
                 wave_spectrum=None, wavefreq=None,
                 d_afsd_latg=None, d_afsd_newi=None,
                 floe_rad_c=None, floe_rad_h=None,
                 floe_binwidth=None, bio_index=(),
                 zbgc_init_frac=None):
    # Calendar values and physical parameters stay dynamic across time steps.
    flags = dict(flags)
    options = dict(options)
    ncat = aicen.shape[0]
    nilyr = int(options["nilyr"])
    puny = parameters["puny"]
    ktherm = int(options["ktherm"])
    nt_qice = int(flags["nt_qice"])
    nt_sice = int(flags["nt_sice"])
    initial_area = aicen
    initial_volume = vicen
    if ktherm == 2:
        reduction = parameters["dSin0_frazil"]
        Si0new = jnp.where(
            sss > 2.0 * reduction,
            sss - reduction,
            sss**2 / (4.0 * reduction))
        Sprofile = jnp.full((nilyr,), Si0new)
        Ti = jnp.minimum(
            liquidus_temperature_mush_jax(
                Si0new / parameters["phi_init"]),
            parameters["Tliquidus_max"])
        qi0new = icepack_enthalpy_mush_jax(Ti, Si0new, parameters)
    else:
        Si0new = jnp.zeros_like(sss)
        Sprofile = salinz
        qi0new = -parameters["rhoi"] * parameters["Lfresh"]
    fnew = jnp.maximum(frzmlt, 0.0)
    new_volume = -fnew * dt / qi0new
    initial_new_volume = new_volume
    frazil = new_volume
    frz_onset = jnp.where(
        (frazil > puny) & (frz_onset < puny), yday, frz_onset)
    if options["cpl_frazil"] != "external":
        if bool(options["update_ocn_f"]):
            dfresh = -parameters["rhoi"] * new_volume / dt
        elif options["cpl_frazil"] == "fresh_ice_correction" and ktherm == 2:
            pure_volume = fnew * dt / (
                parameters["rhoi"] * parameters["Lfresh"])
            dfresh = -parameters["rhoi"] * (new_volume - pure_volume) / dt
            frazil_diag = frazil - pure_volume
        else:
            dfresh = jnp.zeros_like(fresh)
        if options["saltflux_option"] == "prognostic":
            fsalt = fsalt + Si0new * parameters["p001"] * dfresh
        else:
            fsalt = (fsalt + parameters["ice_ref_salinity"]
                     * parameters["p001"] * dfresh)
        fresh = fresh + dfresh

    d_an_latg = jnp.zeros_like(aicen)
    lateral_volume = jnp.zeros_like(vicen)
    radial_growth = jnp.zeros_like(new_volume)
    tot_latg = jnp.zeros_like(new_volume)
    failed = jnp.asarray(False)
    if bool(flags["tr_fsd"]):
        nfsd = int(options["nfsd"])
        fsd_start = int(flags["nt_fsd"])
        d_afsd_latg = jnp.zeros_like(d_afsd_latg)
        d_afsd_newi = jnp.zeros_like(d_afsd_newi)
        afsdn = icepack_cleanup_fsd_jax(
            trcrn[fsd_start:fsd_start + nfsd], puny)
        trcrn = trcrn.at[fsd_start:fsd_start + nfsd].set(afsdn)
        (d_an_latg, radial_growth, _, tot_latg) = fsd_lateral_growth_jax(
            dt, aice, aicen, vicen, new_volume, afsdn,
            floe_rad_c, puny)
        lateral_volume = d_an_latg * vicen / jnp.where(
            aicen > puny, aicen, 1.0)
        total_lateral = jnp.sum(lateral_volume)
        lateral_volume = jnp.where(
            total_lateral > new_volume,
            lateral_volume * new_volume /
            jnp.where(total_lateral > 0.0, total_lateral, 1.0),
            lateral_volume)
        new_volume = new_volume - jnp.sum(lateral_volume)
        frazil = frazil - jnp.sum(lateral_volume)

    highest_new_ice = (0.9 * hin_max[1] if ncat > 1
                       else parameters["bignum"])
    open_area = aice0 - tot_latg
    open_water = open_area > puny
    new_thickness = jnp.maximum(
        new_volume / jnp.where(open_water, open_area, 1.0),
        parameters["hfrazilmin"])
    surplus = (open_water & (new_thickness > highest_new_ice)
               & (open_area + puny < 1.0))
    new_thickness = jnp.where(surplus, highest_new_ice, new_thickness)
    new_area = jnp.where(
        open_water,
        jnp.where(surplus, open_area,
                  new_volume / jnp.where(new_thickness > 0.0,
                                         new_thickness, 1.0)), 0.0)
    surplus_volume = jnp.where(
        surplus, new_volume - new_area * new_thickness,
        jnp.where(open_water, 0.0, new_volume))
    surplus_height = surplus_volume / jnp.where(aice > puny, aice, 1.0)
    surplus_height = jnp.where(new_volume > 0.0, surplus_height, 0.0)
    new_area = jnp.where(new_volume > 0.0, new_area, 0.0)
    # Without FSD, Fortran retains vi0new unless excess ice is redistributed.
    if bool(flags["tr_fsd"]):
        new_ice_volume = new_area * new_thickness
    else:
        new_ice_volume = jnp.where(surplus, new_area * new_thickness, new_volume)
    new_ice_volume = jnp.where(
        (new_volume > 0.0) & open_water, new_ice_volume, 0.0)
    d_an_newi = jnp.zeros_like(aicen).at[0].set(new_area)
    d_an_tot = d_an_latg + d_an_newi
    vin0new = lateral_volume.at[0].add(new_ice_volume)

    def distribute_surplus(n, carry):
        trcrn, vicen, fiso_ocn = carry
        added_volume = surplus_height * aicen[n]
        old_volume = vicen[n]
        new_total = old_volume + added_volume
        adding = surplus_height > 0.0
        if bool(flags["tr_iage"]):
            age = int(flags["nt_iage"])
            updated = ((trcrn[age, n] * old_volume + dt * added_volume)
                       / jnp.where(new_total > puny, new_total, 1.0))
            trcrn = trcrn.at[age, n].set(jnp.where(
                adding & (new_total > puny), updated, trcrn[age, n]))
        if bool(flags["tr_lvl"]):
            level_volume = int(flags["nt_vlvl"])
            level_area = int(flags["nt_alvl"])
            updated = ((trcrn[level_volume, n] * old_volume
                        + trcrn[level_area, n] * added_volume)
                       / jnp.where(new_total > 0.0, new_total, 1.0))
            trcrn = trcrn.at[level_volume, n].set(jnp.where(
                adding & (old_volume > puny), updated,
                trcrn[level_volume, n]))
        if bool(flags["tr_aero"]):
            for it in range(int(flags["n_aero"])):
                start = int(flags["nt_aero"]) + 4 * it + 2
                for layer in range(2):
                    old = trcrn[start + layer, n]
                    trcrn = trcrn.at[start + layer, n].set(jnp.where(
                        adding & (new_total > puny),
                        old * old_volume /
                        jnp.where(new_total > puny, new_total, 1.0), old))
        if bool(flags["tr_iso"]):
            for it, species, ocean in (
                (0, "HDO", HDO_ocn), (1, "H2_16O", H2_16O_ocn),
                (2, "H2_18O", H2_18O_ocn),
            ):
                if it >= int(flags["n_iso"]):
                    continue
                fraction = isoice_alpha(
                    jnp.zeros_like(frazil), species,
                    options["isotope_frac_method"]) * ocean
                index = int(flags["nt_isoice"]) + it
                old = trcrn[index, n]
                updated = ((old * old_volume
                            + fraction * parameters["rhoi"] * added_volume)
                           / jnp.where(new_total > puny, new_total, 1.0))
                trcrn = trcrn.at[index, n].set(jnp.where(
                    adding & (new_total > puny), updated, old))
                fiso_ocn = fiso_ocn.at[it].add(jnp.where(
                    adding, -fraction * parameters["rhoi"]
                    * added_volume / dt, 0.0))
        vicen = vicen.at[n].set(jnp.where(adding, new_total, old_volume))
        remap_old_volume = new_total - added_volume
        if ktherm == 2:
            safe_new_total = jnp.where(
                adding & (new_total > 0.0), new_total, 1.0)
            ice_energy = update_vertical_tracers(
                trcrn[nt_qice:nt_qice + nilyr, n], remap_old_volume,
                safe_new_total, qi0new)
            ice_salt = update_vertical_tracers(
                trcrn[nt_sice:nt_sice + nilyr, n], remap_old_volume,
                safe_new_total, Si0new)
        else:
            denominator = jnp.where(new_total > 0.0, new_total, 1.0)
            ice_energy = ((trcrn[nt_qice:nt_qice + nilyr, n] * remap_old_volume
                           + qi0new * added_volume) / denominator)
            ice_salt = ((trcrn[nt_sice:nt_sice + nilyr, n] * remap_old_volume
                         + Sprofile * added_volume) / denominator)
        trcrn = trcrn.at[nt_qice:nt_qice + nilyr, n].set(jnp.where(
            adding & (new_total > 0.0), ice_energy,
            trcrn[nt_qice:nt_qice + nilyr, n]))
        trcrn = trcrn.at[nt_sice:nt_sice + nilyr, n].set(jnp.where(
            adding & (new_total > 0.0), ice_salt,
            trcrn[nt_sice:nt_sice + nilyr, n]))
        return trcrn, vicen, fiso_ocn

    # Carry isotope fluxes in category order along with the updated ice state.
    trcrn, vicen, fiso_ocn = lax.fori_loop(
        0, ncat, distribute_surplus, (trcrn, vicen, fiso_ocn))

    if bool(flags["tr_fsd"]):
        area2 = initial_area + d_an_latg
        for n in range(ncat):
            old_area = aicen[n]
            old_volume = vicen[n]
            added = (d_an_tot[n] > 0.0) & (vin0new[n] > 0.0)
            aicen = aicen.at[n].set(jnp.where(
                added, old_area + d_an_tot[n], old_area))
            vicen = vicen.at[n].set(jnp.where(
                added, old_volume + vin0new[n], old_volume))
            volume_active = added & (vicen[n] > puny)
            aice0 = aice0 - jnp.where(added, d_an_tot[n], 0.0)
            area_denominator = jnp.where(aicen[n] > 0.0, aicen[n], 1.0)
            volume_denominator = jnp.where(vicen[n] > 0.0, vicen[n], 1.0)
            surface = int(flags["nt_Tsfc"])
            surface_new = jnp.minimum(
                (trcrn[surface, n] * old_area + Tf * d_an_tot[n])
                / area_denominator, 0.0)
            trcrn = trcrn.at[surface, n].set(jnp.where(
                added, surface_new, trcrn[surface, n]))
            if bool(flags["tr_FY"]):
                index = int(flags["nt_FY"])
                value = jnp.minimum(
                    (trcrn[index, n] * old_area + d_an_tot[n])
                    / area_denominator, 1.0)
                trcrn = trcrn.at[index, n].set(jnp.where(
                    added, value, trcrn[index, n]))

            (fsd_trcrn, updated_latg, updated_newi,
             fsd_failed) = fsd_add_new_ice_jax(
                 n, dt, new_area, d_an_latg, d_an_newi,
                 radial_growth, area2, wave_sig_ht, wave_spectrum,
                 wavefreq, d_afsd_latg, d_afsd_newi,
                 afsdn, initial_area, aicen, trcrn,
                 fsd_start, bool(options["wave_spec"]),
                 floe_rad_c, floe_rad_h, floe_binwidth, parameters)
            trcrn = jnp.where(added, fsd_trcrn, trcrn)
            d_afsd_latg = jnp.where(added, updated_latg, d_afsd_latg)
            d_afsd_newi = jnp.where(added, updated_newi, d_afsd_newi)
            failed = failed | (added & fsd_failed)
            if bool(flags["tr_iage"]):
                index = int(flags["nt_iage"])
                age = ((trcrn[index, n] * old_volume + dt * vin0new[n])
                       / volume_denominator)
                trcrn = trcrn.at[index, n].set(jnp.where(
                    volume_active, age, trcrn[index, n]))
            if bool(flags["tr_aero"]):
                for it in range(int(flags["n_aero"])):
                    for layer in range(2):
                        index = int(flags["nt_aero"]) + 4 * it + 2 + layer
                        trcrn = trcrn.at[index, n].set(jnp.where(
                            volume_active, trcrn[index, n] * old_volume
                            / volume_denominator, trcrn[index, n]))
            if bool(flags["tr_iso"]):
                for it, species, ocean in (
                    (0, "HDO", HDO_ocn), (1, "H2_16O", H2_16O_ocn),
                    (2, "H2_18O", H2_18O_ocn),
                ):
                    if it >= int(flags["n_iso"]):
                        continue
                    concentration = isoice_alpha(
                        jnp.zeros_like(frazil), species,
                        options["isotope_frac_method"]) * ocean
                    # The original expression updates category one even when
                    # lateral growth is added to another thickness category.
                    index = int(flags["nt_isoice"]) + it
                    old = trcrn[index, 0]
                    value = (old * old_volume + concentration
                             * parameters["rhoi"] * new_volume
                             / jnp.where(vicen[0] > 0.0, vicen[0], 1.0))
                    trcrn = trcrn.at[index, 0].set(jnp.where(volume_active, value, old))
                    fiso_ocn = fiso_ocn.at[it].add(jnp.where(
                        volume_active, -concentration * parameters["rhoi"]
                        * new_volume / dt, 0.0))
            if bool(flags["tr_lvl"]):
                area_index = int(flags["nt_alvl"])
                volume_index = int(flags["nt_vlvl"])
                old_level_area = trcrn[area_index, n]
                trcrn = trcrn.at[area_index, n].set(jnp.where(
                    volume_active, (old_level_area * old_area + d_an_tot[n])
                    / area_denominator, old_level_area))
                trcrn = trcrn.at[volume_index, n].set(jnp.where(
                    volume_active, (trcrn[volume_index, n] * old_volume
                            + vin0new[n]) / volume_denominator,
                    trcrn[volume_index, n]))
                if bool(flags["tr_pond_lvl"]):
                    index = int(flags["nt_apnd"])
                    new_level_area = trcrn[area_index, n]
                    pond_active = volume_active & (new_level_area > puny)
                    apnd = (trcrn[index, n] * old_level_area * old_area
                            / jnp.where(pond_active,
                                        new_level_area * aicen[n], 1.0))
                    trcrn = trcrn.at[index, n].set(jnp.where(
                        pond_active, apnd, trcrn[index, n]))
            if bool(flags["tr_pond_topo"]):
                index = int(flags["nt_apnd"])
                trcrn = trcrn.at[index, n].set(jnp.where(
                    volume_active, trcrn[index, n] * old_area / area_denominator,
                    trcrn[index, n]))
            for k in range(nilyr):
                for tracer, new_value in (
                    (nt_qice + k, qi0new), (nt_sice + k, Sprofile[k]),
                ):
                    value = ((trcrn[tracer, n] * old_volume
                              + new_value * vin0new[n]) / volume_denominator)
                    trcrn = trcrn.at[tracer, n].set(jnp.where(
                        added, value, trcrn[tracer, n]))
        if bool(flags["tr_brine"]) or int(flags["nbtrcr"]) > 0:
            trcrn, flux_bio, bgc_failed = add_new_ice_bgc(
                dt, ncat, initial_area, initial_volume,
                initial_new_volume, aicen, vicen, vin0new,
                trcrn, ocean_bio, flux_bio, surplus_height,
                d_an_tot, bio_index, zbgc_init_frac,
                flags, options, parameters)
            failed = failed | bgc_failed
        return (aicen, trcrn, vicen, aice0, frazil, frz_onset,
                fresh, fsalt, frazil_diag, fiso_ocn, flux_bio,
                initial_area, initial_volume, initial_new_volume,
                new_ice_volume, new_area, surplus_height,
                d_afsd_latg, d_afsd_newi, failed)

    old_area = aicen[0]
    old_volume = vicen[0]
    added = (new_area > 0.0) & (new_ice_volume > 0.0)
    aicen = aicen.at[0].set(jnp.where(added, old_area + new_area, old_area))
    aice0 = aice0 - jnp.where(added, new_area, 0.0)
    vicen = vicen.at[0].set(jnp.where(added, old_volume + new_ice_volume, old_volume))
    volume_active = added & (vicen[0] > puny)
    area_denominator = jnp.where(aicen[0] > 0.0, aicen[0], 1.0)
    volume_denominator = jnp.where(vicen[0] > 0.0, vicen[0], 1.0)
    surface = int(flags["nt_Tsfc"])
    updated = jnp.minimum(
        (trcrn[surface, 0] * old_area + Tf * new_area)
        / area_denominator, 0.0)
    trcrn = trcrn.at[surface, 0].set(jnp.where(
        added, updated, trcrn[surface, 0]))
    if bool(flags["tr_FY"]):
        index = int(flags["nt_FY"])
        updated = jnp.minimum(
            (trcrn[index, 0] * old_area + new_area)
            / area_denominator, 1.0)
        trcrn = trcrn.at[index, 0].set(jnp.where(
            added, updated, trcrn[index, 0]))
    if bool(flags["tr_iage"]):
        index = int(flags["nt_iage"])
        updated = ((trcrn[index, 0] * old_volume
                    + dt * new_ice_volume) / volume_denominator)
        trcrn = trcrn.at[index, 0].set(jnp.where(
            volume_active, updated, trcrn[index, 0]))
    if bool(flags["tr_aero"]):
        for it in range(int(flags["n_aero"])):
            for layer in range(2):
                index = int(flags["nt_aero"]) + 4 * it + 2 + layer
                trcrn = trcrn.at[index, 0].set(jnp.where(
                    volume_active, trcrn[index, 0] * old_volume /
                    volume_denominator, trcrn[index, 0]))
    if bool(flags["tr_iso"]):
        for it, species, ocean in (
            (0, "HDO", HDO_ocn), (1, "H2_16O", H2_16O_ocn),
            (2, "H2_18O", H2_18O_ocn),
        ):
            if it >= int(flags["n_iso"]):
                continue
            fraction = isoice_alpha(
                jnp.zeros_like(frazil), species,
                options["isotope_frac_method"]) * ocean
            index = int(flags["nt_isoice"]) + it
            # Preserve the original Fortran grouping of this expression.
            updated = (trcrn[index, 0] * old_volume
                       + fraction * parameters["rhoi"]
                       * new_ice_volume / volume_denominator)
            trcrn = trcrn.at[index, 0].set(jnp.where(
                volume_active, updated, trcrn[index, 0]))
            fiso_ocn = fiso_ocn.at[it].add(jnp.where(
                volume_active, -fraction * parameters["rhoi"]
                * new_ice_volume / dt, 0.0))
    if bool(flags["tr_lvl"]):
        level_area = int(flags["nt_alvl"])
        level_volume = int(flags["nt_vlvl"])
        old_level_area = trcrn[level_area, 0]
        trcrn = trcrn.at[level_area, 0].set(jnp.where(
            volume_active, (old_level_area * old_area + new_area)
            / area_denominator, old_level_area))
        trcrn = trcrn.at[level_volume, 0].set(jnp.where(
            volume_active, (trcrn[level_volume, 0] * old_volume
                    + new_ice_volume) / volume_denominator,
            trcrn[level_volume, 0]))
        if bool(flags["tr_pond_lvl"]):
            index = int(flags["nt_apnd"])
            new_level_area = trcrn[level_area, 0]
            pond_active = volume_active & (new_level_area > puny)
            apnd = (trcrn[index, 0] * old_level_area * old_area
                    / jnp.where(pond_active,
                                new_level_area * aicen[0], 1.0))
            trcrn = trcrn.at[index, 0].set(jnp.where(
                pond_active, apnd, trcrn[index, 0]))
    if bool(flags["tr_pond_topo"]):
        index = int(flags["nt_apnd"])
        trcrn = trcrn.at[index, 0].set(jnp.where(
            volume_active, trcrn[index, 0] * old_area / area_denominator,
            trcrn[index, 0]))
    for k in range(nilyr):
        for tracer, new_value in (
            (nt_qice + k, qi0new), (nt_sice + k, Sprofile[k]),
        ):
            updated = ((trcrn[tracer, 0] * old_volume
                        + new_value * new_ice_volume) / volume_denominator)
            trcrn = trcrn.at[tracer, 0].set(jnp.where(
                added, updated, trcrn[tracer, 0]))
    if bool(flags["tr_brine"]) or int(flags["nbtrcr"]) > 0:
        trcrn, flux_bio, bgc_failed = add_new_ice_bgc(
            dt, 1, initial_area, initial_volume,
            initial_new_volume, aicen, vicen,
            jnp.zeros_like(vicen).at[0].set(new_ice_volume),
            trcrn, ocean_bio, flux_bio, surplus_height,
            d_an_tot, bio_index, zbgc_init_frac,
            flags, options, parameters)
        failed = failed | bgc_failed
    return (aicen, trcrn, vicen, aice0, frazil, frz_onset,
            fresh, fsalt, frazil_diag, fiso_ocn, flux_bio,
            initial_area, initial_volume, initial_new_volume,
            new_ice_volume, new_area, surplus_height,
            d_afsd_latg, d_afsd_newi, failed)


@partial(jax.jit, static_argnames=("layout", "flags", "options"))
def _transport_itd(hin_max, aicen_init, vicen_init, Tf, work,
                   layout, flags, options, parameters):
    # Include the ice-presence condition in the compiled boundary so vmap's
    # batched branch selection is compiled with the category remap.
    def advance(state):
        (aicen, trcrn, vicen, vsnon, aice, aice0,
         fpond, dpnd_melt, failed) = state
        (aicen, trcrn, vicen, vsnon, aice, aice0,
         fpond, dpnd_melt, error) = _linear_itd(
            hin_max, aicen_init, vicen_init,
            aicen, trcrn, vicen, vsnon, aice, aice0, fpond,
            Tf, dpnd_melt, layout, flags, options, parameters)
        return (aicen, trcrn, vicen, vsnon, aice, aice0,
                fpond, dpnd_melt, failed | error)

    return lax.cond(work[4] > parameters["puny"], advance, lambda state: state, work)


def icepack_step_therm2(dt, hin_max, fields, layout, flags, options,
                        parameters, bio_index=(), floe_rad_c=None,
                        floe_binwidth=None, floe_area_c=None, iweld=None,
                        zbgc_init_frac=None, column_call=None):
    """Advance a column, or batch each substage through column_call.

    The optional host dispatcher times concrete batched arrays between
    transformations. Every physical subroutine still receives one column.
    """
    if column_call is None:
        def column_call(name, function, *arrays):
            return function(*arrays)

    fields = dict(fields)
    aicen = fields["aicen"]
    trcrn = fields["trcrn"]
    vicen = fields["vicen"]
    vsnon = fields["vsnon"]
    aice = fields["aice"]
    aice0 = fields["aice0"]
    fpond = fields["fpond"]
    fresh = fields["fresh"] + fields["frain"] * aice
    fsalt = fields["fsalt"]
    fhocn = fields["fhocn"]
    faero_ocn = fields["faero_ocn"]
    fiso_ocn = fields["fiso_ocn"]
    flux_bio = jnp.zeros_like(fields["flux_bio"])
    dpnd_melt = fields["dpnd_melt"]
    failed = jnp.zeros_like(aice, dtype=bool)
    itd_failed = jnp.zeros_like(aice, dtype=bool)
    weld_failed = jnp.zeros_like(aice, dtype=bool)
    area_entry = column_sum(aicen.shape[-1], aicen)

    aice = area_entry
    aice0 = jnp.maximum(1.0 - aice, 0.0)
    if int(options["kitd"]) == 1:
        itd_layout = tuple(sorted(layout.items()))
        itd_flags = tuple(sorted(flags.items()))
        itd_options = tuple(sorted(options.items()))

        def transport(work, cell):
            return _transport_itd(
                hin_max, cell["aicen_init"], cell["vicen_init"], cell["Tf"], work,
                itd_layout, itd_flags, itd_options, parameters)

        (aicen, trcrn, vicen, vsnon, aice, aice0,
         fpond, dpnd_melt, failed) = column_call(
             "itd", transport,
             (aicen, trcrn, vicen, vsnon, aice, aice0,
              fpond, dpnd_melt, failed), fields)
        itd_failed = failed
    area_after_itd = column_sum(aicen.shape[-1], aicen)

    def new_ice(work, cell):
        aicen, trcrn, vicen, aice0, aice, fresh, fsalt, fiso_ocn, flux_bio = work
        return add_new_ice(
            dt, hin_max, aicen, trcrn, vicen, aice0, aice,
            cell["frzmlt"], cell["frazil"], cell["frz_onset"],
            cell["yday"], fresh, fsalt, cell["Tf"], cell["sss"],
            cell["salinz"], flux_bio, cell["ocean_bio"],
            cell["frazil_diag"], fiso_ocn, cell["HDO_ocn"],
            cell["H2_16O_ocn"], cell["H2_18O_ocn"],
            flags, options, parameters,
            cell["wave_sig_ht"], cell["wave_spectrum"],
            cell["wavefreq"], cell["d_afsd_latg"],
            cell["d_afsd_newi"], floe_rad_c,
            cell["floe_rad_h"], floe_binwidth,
            bio_index, zbgc_init_frac)

    (aicen, trcrn, vicen, aice0, frazil, frz_onset,
     fresh, fsalt, frazil_diag, fiso_ocn, flux_bio,
     _, _, _, _, _, _, d_afsd_latg, d_afsd_newi,
     new_ice_failed) = column_call(
         "new_ice", new_ice,
         (aicen, trcrn, vicen, aice0, aice, fresh, fsalt, fiso_ocn, flux_bio), fields)
    failed = failed | new_ice_failed
    area_after_new_ice = column_sum(aicen.shape[-1], aicen)

    def melt_sides(work, cell):
        (fpond, fresh, fsalt, fhocn, faero_ocn, fiso_ocn,
         aicen, vicen, vsnon, trcrn, flux_bio, dpnd_melt) = work
        return lateral_melt(
            dt, fpond, fresh, fsalt, fhocn, faero_ocn,
            fiso_ocn, cell["rsiden"], cell["meltl"],
            cell["wlat"], aicen, vicen, vsnon, trcrn, flux_bio,
            cell["d_afsd_latm"], dpnd_melt,
            flags, options, parameters, bio_index,
            floe_rad_c, floe_binwidth)

    (fpond, fresh, fsalt, fhocn, faero_ocn, fiso_ocn,
     meltl, aicen, vicen, vsnon, trcrn, flux_bio,
     d_afsd_latm, dpnd_melt) = column_call(
         "lateral_melt", melt_sides,
         (fpond, fresh, fsalt, fhocn, faero_ocn, fiso_ocn,
          aicen, vicen, vsnon, trcrn, flux_bio, dpnd_melt), fields)
    area_after_lateral_melt = column_sum(aicen.shape[-1], aicen)

    d_afsd_weld = fields["d_afsd_weld"]
    if bool(flags["tr_fsd"]):
        def weld(work, cell):
            aicen, trcrn, d_afsd_weld = work
            return fsd_weld_thermo_jax(
                dt, cell["frzmlt"], aicen, trcrn, d_afsd_weld,
                int(flags["nt_fsd"]), floe_area_c, iweld, parameters)

        trcrn, d_afsd_weld, weld_failed = column_call(
            "fsd_weld", weld, (aicen, trcrn, d_afsd_weld), fields)
        failed = failed | weld_failed

    if aicen.shape[-1] == 1:
        def reduce_single_category(work, cell):
            aicen, vicen, dpnd_melt, trcrn = work
            return reduce_area(
                hin_max[0], aicen[0], vicen[0],
                cell["aicen_init"][0], cell["vicen_init"][0],
                dpnd_melt, trcrn, flags)

        first_area, first_volume, dpnd_melt = column_call(
            "reduce_area", reduce_single_category, (aicen, vicen, dpnd_melt, trcrn), fields)
        aicen = aicen.at[..., 0].set(first_area)
        vicen = vicen.at[..., 0].set(first_volume)

    def cleanup(work, cell):
        (aicen, trcrn, vicen, vsnon, aice0, aice, fpond, fresh, fsalt,
         fhocn, faero_ocn, fiso_ocn, flux_bio) = work
        return cleanup_itd_jax(
            dt, hin_max, aicen, trcrn, vicen, vsnon,
            aice0, aice, cell["first_ice"], fpond, fresh, fsalt,
            fhocn, faero_ocn, fiso_ocn, flux_bio, cell["Tf"],
            layout, flags, options, parameters, bio_index, diagnose=True)

    (aicen, trcrn, vicen, vsnon, aice0, aice,
     first_ice, fpond, fresh, fsalt, fhocn,
     faero_ocn, fiso_ocn, flux_bio, cleanup_failed,
     cleanup_reasons) = column_call(
         "cleanup", cleanup,
         (aicen, trcrn, vicen, vsnon, aice0, aice, fpond, fresh, fsalt,
          fhocn, faero_ocn, fiso_ocn, flux_bio), fields)
    failed = failed | cleanup_failed
    fields.update({
        "aicen": aicen, "trcrn": trcrn, "vicen": vicen,
        "vsnon": vsnon, "aice": aice, "aice0": aice0,
        "first_ice": first_ice, "fpond": fpond, "fresh": fresh,
        "fsalt": fsalt, "fhocn": fhocn, "faero_ocn": faero_ocn,
        "fiso_ocn": fiso_ocn, "flux_bio": flux_bio,
        "frazil": frazil, "frazil_diag": frazil_diag,
        "frz_onset": frz_onset, "meltl": meltl,
        "dpnd_melt": dpnd_melt, "d_afsd_latm": d_afsd_latm,
        "d_afsd_latg": d_afsd_latg, "d_afsd_newi": d_afsd_newi,
        "d_afsd_weld": d_afsd_weld,
    })
    return (fields, failed,
            (itd_failed, new_ice_failed, weld_failed,
             cleanup_failed, *cleanup_reasons),
            (area_entry, area_after_itd, area_after_new_ice,
             area_after_lateral_melt, column_sum(aicen.shape[-1], aicen)))
