"""Differentiable Delta-Eddington shortwave column calculations.

Radiation interfaces keep the Fortran lower bound of zero. Each function
handles one ice category in one physical grid cell; the driver applies vmap.
"""

from functools import partial

import jax
import jax.numpy as jnp


# Reuse the two optical sweeps across bands, categories and time steps.
@partial(jax.jit, static_argnames=("srftyp", "nslyr"))
def solution_dEdd_jax(coszen, srftyp, tau, w0, g, albodr, albodf, nslyr):
    """Reflectance and transmittance of a vertical optical column."""
    klev = tau.shape[0] - 1
    klevp = klev + 1
    trndir = jnp.zeros(klevp + 1, dtype=tau.dtype).at[0].set(1.0)
    trntdr = jnp.zeros(klevp + 1, dtype=tau.dtype).at[0].set(1.0)
    trndif = jnp.zeros(klevp + 1, dtype=tau.dtype).at[0].set(1.0)
    rdndif = jnp.zeros(klevp + 1, dtype=tau.dtype)
    rdir = jnp.zeros(klev + 1, dtype=tau.dtype)
    rdif_a = jnp.zeros(klev + 1, dtype=tau.dtype)
    rdif_b = jnp.zeros(klev + 1, dtype=tau.dtype)
    tdir = jnp.zeros(klev + 1, dtype=tau.dtype)
    tdif_a = jnp.zeros(klev + 1, dtype=tau.dtype)
    tdif_b = jnp.zeros(klev + 1, dtype=tau.dtype)
    trnlay = jnp.zeros(klev + 1, dtype=tau.dtype)
    gauspt = jnp.asarray((0.9894009, 0.9445750, 0.8656312, 0.7554044,
                          0.6178762, 0.4580168, 0.2816036, 0.0950125),
                         dtype=tau.dtype)
    gauswt = jnp.asarray((0.0271525, 0.0622535, 0.0951585, 0.1246290,
                          0.1495960, 0.1691565, 0.1826034, 0.1894506),
                         dtype=tau.dtype)
    mu0 = jnp.maximum(coszen, 0.01)
    refindx = 1.310
    mu0nij = jnp.sqrt(1.0 - (1.0 - mu0 ** 2) / refindx ** 2)
    kfrsnl = nslyr + 2 if srftyp < 2 else 0

    def descend(k, carry):
        (trndir, trntdr, trndif, rdndif, rdir, rdif_a, rdif_b,
         tdir, tdif_a, tdif_b, trnlay) = carry

        def penetrating(_):
            wtot = w0[k]
            gtot = g[k]
            ftot = gtot ** 2
            ts = (1.0 - wtot * ftot) * tau[k]
            ws = (1.0 - ftot) * wtot / (1.0 - wtot * ftot)
            gs = (gtot - ftot) / (1.0 - ftot)
            lm = jnp.sqrt(3.0 * (1.0 - ws) * (1.0 - ws * gs))
            ue = 1.5 * (1.0 - ws * gs) / lm
            mu0n = mu0 if srftyp < 2 and kfrsnl > 0 else mu0nij
            mu0n = jnp.where(k < kfrsnl, mu0n, mu0nij)
            extins = jnp.exp(-jnp.minimum(10.0, lm * ts))
            ne = (ue + 1.0) ** 2 / extins - (ue - 1.0) ** 2 * extins
            rdifa = (ue ** 2 - 1.0) * (1.0 / extins - extins) / ne
            tdifa = 4.0 * ue / ne
            trn = jnp.exp(-jnp.minimum(10.0, ts / mu0n))
            alp = 0.75 * ws * mu0n * (1.0 + gs * (1.0 - ws)) / (1.0 - lm ** 2 * mu0n ** 2)
            gam = 0.5 * ws * (1.0 + 3.0 * gs * (1.0 - ws) * mu0n ** 2) / (1.0 - lm ** 2 * mu0n ** 2)
            apg = alp + gam
            amg = alp - gam
            rdirk = apg * rdifa + amg * (tdifa * trn - 1.0)
            tdirk = apg * tdifa + (amg * rdifa - apg + 1.0) * trn

            # Integrate the direct-beam solution over eight Gaussian angles
            # to obtain the diffuse reflectance and transmission.
            ext = jnp.exp(-jnp.minimum(10.0, ts / gauspt))
            alps = 0.75 * ws * gauspt * (1.0 + gs * (1.0 - ws)) / (1.0 - lm ** 2 * gauspt ** 2)
            gams = 0.5 * ws * (1.0 + 3.0 * gs * (1.0 - ws) * gauspt ** 2) / (1.0 - lm ** 2 * gauspt ** 2)
            apgs = alps + gams
            amgs = alps - gams
            rdr = apgs * rdifa + amgs * tdifa * ext - amgs
            tdr = apgs * tdifa + amgs * rdifa * ext - apgs * ext + ext
            # Preserve Fortran's angular sum and product order.
            swt = jnp.zeros_like(ts)
            smr = jnp.zeros_like(ts)
            smt = jnp.zeros_like(ts)
            for ng in range(8):
                swt = swt + gauspt[ng] * gauswt[ng]
                smr = smr + gauspt[ng] * rdr[ng] * gauswt[ng]
                smt = smt + gauspt[ng] * tdr[ng] * gauswt[ng]
            rdifa = smr / swt
            tdifa = smt / swt
            rdifb = rdifa
            tdifb = tdifa

            def fresnel(values):
                rdirk, rdifa, rdifb, tdirk, tdifa, tdifb, trn = values
                R1 = (mu0 - refindx * mu0n) / (mu0 + refindx * mu0n)
                R2 = (refindx * mu0 - mu0n) / (refindx * mu0 + mu0n)
                T1 = 2.0 * mu0 / (mu0 + refindx * mu0n)
                T2 = 2.0 * mu0 / (refindx * mu0 + mu0n)
                Rf_dir_a = 0.5 * (R1 ** 2 + R2 ** 2)
                Tf_dir_a = 0.5 * (T1 ** 2 + T2 ** 2) * refindx * mu0n / mu0
                Rf_dif_a, Rf_dif_b = 0.063, 0.455
                Tf_dif_a, Tf_dif_b = 1.0 - Rf_dif_a, 1.0 - Rf_dif_b
                rintfc = 1.0 / (1.0 - Rf_dif_b * rdifa)
                tdir_new = (Tf_dir_a * tdirk
                            + Tf_dir_a * rdirk * Rf_dif_b * rintfc * tdifa)
                rdir_new = Rf_dir_a + Tf_dir_a * rdirk * rintfc * Tf_dif_b
                rdifa_new = Rf_dif_a + Tf_dif_a * rdifa * rintfc * Tf_dif_b
                rdifb_new = rdifb + tdifb * Rf_dif_b * rintfc * tdifa
                return (rdir_new, rdifa_new, rdifb_new, tdir_new,
                        tdifa * rintfc * Tf_dif_a,
                        tdifb * rintfc * Tf_dif_b, Tf_dir_a * trn)

            return jax.lax.cond(k == kfrsnl, fresnel, lambda x: x,
                                (rdirk, rdifa, rdifb, tdirk, tdifa, tdifb, trn))

        values = jax.lax.cond(trntdr[k] > 0.001, penetrating,
                              lambda _: (tau[k] * 0.0,) * 7, None)
        rdirk, rdifa, rdifb, tdirk, tdifa, tdifb, trnk = values
        rdir = rdir.at[k].set(rdirk)
        rdif_a = rdif_a.at[k].set(rdifa)
        rdif_b = rdif_b.at[k].set(rdifb)
        tdir = tdir.at[k].set(tdirk)
        tdif_a = tdif_a.at[k].set(tdifa)
        tdif_b = tdif_b.at[k].set(tdifb)
        trnlay = trnlay.at[k].set(trnk)
        trndir = trndir.at[k + 1].set(trndir[k] * trnk)
        refkm1 = 1.0 / (1.0 - rdndif[k] * rdifa)
        tdrrdir = trndir[k] * rdirk
        tdndif = trntdr[k] - trndir[k]
        trntdr = trntdr.at[k + 1].set(
            trndir[k] * tdirk + (tdndif + tdrrdir * rdndif[k]) * refkm1 * tdifa)
        rdndif = rdndif.at[k + 1].set(rdifb + tdifb * rdndif[k] * refkm1 * tdifa)
        trndif = trndif.at[k + 1].set(trndif[k] * refkm1 * tdifa)
        return (trndir, trntdr, trndif, rdndif, rdir, rdif_a, rdif_b,
                tdir, tdif_a, tdif_b, trnlay)

    (trndir, trntdr, trndif, rdndif, rdir, rdif_a, rdif_b,
     tdir, tdif_a, tdif_b, trnlay) = jax.lax.fori_loop(
         0, klev + 1, descend,
         (trndir, trntdr, trndif, rdndif, rdir, rdif_a, rdif_b,
          tdir, tdif_a, tdif_b, trnlay))

    rupdir = jnp.zeros(klevp + 1, dtype=tau.dtype).at[klevp].set(albodr)
    rupdif = jnp.zeros(klevp + 1, dtype=tau.dtype).at[klevp].set(albodf)

    def ascend(index, carry):
        rupdir, rupdif = carry
        k = klev - index
        refkp1 = 1.0 / (1.0 - rdif_b[k] * rupdif[k + 1])
        direct = rdir[k] + (trnlay[k] * rupdir[k + 1]
                           + (tdir[k] - trnlay[k]) * rupdif[k + 1]) * refkp1 * tdif_b[k]
        diffuse = rdif_a[k] + tdif_a[k] * rupdif[k + 1] * refkp1 * tdif_b[k]
        return rupdir.at[k].set(direct), rupdif.at[k].set(diffuse)

    rupdir, rupdif = jax.lax.fori_loop(0, klev + 1, ascend, (rupdir, rupdif))
    return trndir, trntdr, trndif, rupdir, rupdif, rdndif


def interface_flux_jax(trndir, trntdr, trndif, rupdir, rupdif, rdndif, puny):
    """Net downward direct and diffuse flux at every optical interface."""
    refk = 1.0 / (1.0 - rdndif * rupdif)
    dfdir = (trndir + (trntdr - trndir) * (1.0 - rupdif) * refk
             - trndir * rupdir * (1.0 - rdndif) * refk)
    dfdif = trndif * (1.0 - rupdif) * refk
    return jnp.where(dfdir < puny, 0.0, dfdir), jnp.where(dfdif < puny, 0.0, dfdif)


def shortwave_dEdd_set_snow_jax(aice, vsno, Tsfc, rsnow, options, params):
    """Snow coverage, density and grain radii for one category."""
    hs = vsno / aice
    snow_cover = jnp.where(
        params["hs0"] > params["puny"],
        jnp.minimum(hs / jnp.where(params["hs0"] > params["puny"],
                                   params["hs0"], 1.0), 1.0), 1.0)
    fs = jnp.where(hs >= params["hs_min"], snow_cover, 0.0)
    rhosnw = jnp.full_like(rsnow, params["rhos"])
    if bool(options["snwgrain"]):
        rsnw = jnp.clip(rsnow, params["rsnw_fall"], params["rsnw_tmax"])
    else:
        fT = -jnp.minimum((params["Timelt"] - Tsfc) / params["dT_mlt"] - 1.0, 0.0)
        rsnw_nm = jnp.clip(500.0 - params["R_snw"] * 250.0,
                           params["rsnw_fall"], params["rsnw_mlt"])
        radius = jnp.clip(rsnw_nm + (params["rsnw_mlt"] - rsnw_nm) * fT,
                          params["rsnw_fall"], params["rsnw_mlt"])
        rsnw = jnp.full_like(rsnow, radius)
    return fs, hs, rhosnw, rsnw


def shortwave_dEdd_set_eff_jax(aicen, vsnon, alvln, apondn, hpndn,
                                ipndn, ffracn, fsnow, dt, Tsfcn,
                                fsn, hsn, dhsn, rsnow,
                                rhosnwn, rsnwn, options, params):
    """Snow and pond fractions when pond tracers are active."""
    hsnlvl = hsn
    if str(options["snwredist"]).strip() == "bulk":
        if not bool(options["tr_lvl"]):
            raise ValueError("(shortwave_dEdd_set_eff) bulk snow needs level-ice tracer")
        hsnlvl = hsn / (1.0 + params["snwlvlfac"] * (1.0 - alvln))
        alvl = aicen * alvln
        vsn = jnp.where(alvl > params["puny"], hsnlvl * alvl, vsnon)
        alvl = jnp.where(alvl > params["puny"], alvl, aicen)
        fsn, hsnlvl, rhosnwn, rsnwn = shortwave_dEdd_set_snow_jax(
            alvl, vsn, Tsfcn, rsnow, options, params)

    ipn = apondn * ipndn
    dhs = jnp.where((ipn > params["puny"]) & (dhsn < params["puny"])
                    & (fsnow * dt > params["hs_min"]),
                    hsnlvl - fsnow * dt, dhsn)
    spn = hsnlvl - dhs
    dhs = jnp.where(ipn * spn < params["puny"], 0.0, dhs)
    fpn = (1.0 - ffracn) * apondn
    hpn = hpndn
    asnow = jnp.minimum(spn / jnp.where(params["hs1"] > params["puny"],
                                         params["hs1"], 1.0), 1.0)
    fpn = jnp.where((dhs > params["puny"]) & (spn >= params["puny"])
                    & (params["hs1"] > params["puny"]),
                    (1.0 - asnow) * fpn, fpn)
    hp = hpn
    has_pond = hp > params["puny"]
    rp = params["rhofresh"] * hp / jnp.where(
        has_pond, params["rhofresh"] * hp + params["rhos"] * hsnlvl, 1.0)
    infiltrates = has_pond & (rp >= 0.15)
    hmx = hsnlvl * (params["rhofresh"] - params["rhos"]) / params["rhofresh"]
    tmp = jnp.where(hp >= hmx, 1.0, 0.0)
    hp_new = (params["rhofresh"] * hp + params["rhos"] * hsnlvl * tmp) / (
        params["rhofresh"] - params["rhos"] * (1.0 - tmp))
    hsn = jnp.where(infiltrates, hsn - hp_new * fpn * (1.0 - tmp), hsn)
    hpn = jnp.where(infiltrates, hp_new * tmp, hpn)
    fpn = jnp.where(infiltrates, fpn * tmp, fpn)
    fpn = jnp.where(has_pond & (rp < 0.15), 0.0, fpn)
    hpn = jnp.where(has_pond & (rp < 0.15), 0.0, hpn)
    fpn = jnp.where(hpn < params["hpmin"], 0.0, fpn)
    fsn = jnp.minimum(fsn, 1.0 - fpn)
    return fsn, hsn, dhs, fpn, hpn, fpn, rhosnwn, rsnwn


def radiation_dzk_jax(hs, hi, nslyr, nilyr, params):
    """Thickness of each snow, scattering and sea-ice optical layer."""
    klev = nslyr + nilyr + 1
    kii = nslyr + 1
    dzk = jnp.zeros(klev + 1, dtype=jnp.result_type(hs, hi))
    dz = hs * (1.0 / nslyr)
    dzk = dzk.at[0].set(jnp.minimum(params["hs_ssl"], dz / 2.0))
    dzk = dzk.at[1].set(dz - dzk[0])
    for k in range(2, nslyr + 1):
        dzk = dzk.at[k].set(dz)
    dz = hi * (1.0 / nilyr)
    dz_ssl = jnp.minimum(params["hi_ssl"], hi / 30.0)
    dz_ssl = jnp.minimum(dz_ssl, dz / 2.0)
    dzk = dzk.at[kii].set(dz_ssl)
    dzk = dzk.at[kii + 1].set(dz - dz_ssl)
    for k in range(kii + 2, klev + 1):
        dzk = dzk.at[k].set(dz)
    return dzk


def tune_scat_jax(ki_mn, wi_mn, gi_mn, fp, fm, radius_factor):
    """Adjust scattering cross section for the requested albedo tuning."""
    factor = jnp.where(radius_factor >= 0.0, fp, fm) * radius_factor
    sigp = ki_mn * wi_mn * (1.0 + factor)
    sigp = jnp.where(radius_factor < 0.0, jnp.maximum(sigp, 0.0), sigp)
    ki = sigp + ki_mn * (1.0 - wi_mn)
    return ki, sigp / ki, gi_mn


def grain_index_jax(radius):
    """One-based modal black-carbon snow grain-size table index."""
    index = jnp.where(radius < 125.0, jnp.floor(radius / 50.0 + 0.5),
                      jnp.where(radius < 175.0, 2.0,
                                jnp.floor(radius / 250.0 + 2.0 + 0.5)))
    return jnp.clip(index.astype(jnp.int32), 1, 8)


def snow_table_3bd_jax(ns, radius, data):
    """Linear interpolation of three-band snow optical properties."""
    table = data["rsnw_tab"]
    upper = jnp.clip(jnp.searchsorted(table, radius, side="left"),
                     1, table.shape[0] - 1)
    lower = upper - 1
    frac = (radius - table[lower]) / (table[upper] - table[lower])
    frac = jnp.where(radius <= table[0], 0.0,
                     jnp.where(radius >= table[-1], 1.0, frac))
    for_name = ("Qs_tab", "ws_tab", "gs_tab")
    return tuple(data[name][ns, lower] * (1.0 - frac)
                 + data[name][ns, upper] * frac for name in for_name)


def snicar_iop_jax(ns, radius, direct, data):
    """Linear interpolation of SNICAR direct or diffuse snow optics."""
    table = data["rsnw_snicar_tab"]
    upper = jnp.clip(jnp.searchsorted(table, radius, side="left"),
                     1, table.shape[0] - 1)
    lower = upper - 1
    frac = (radius - table[lower]) / (table[upper] - table[lower])
    frac = jnp.where(radius <= table[0], 0.0,
                     jnp.where(radius >= table[-1], 1.0, frac))
    suffix = "dr" if direct else "df"
    names = ("ssp_snwext" + suffix, "ssp_snwalb" + suffix,
             "ssp_sasymm" + suffix)
    return tuple(data[name][ns, lower] * (1.0 - frac)
                 + data[name][ns, upper] * frac for name in names)


def chlorophyll_jax(nspint, dzk, zbio, options, params):
    """Visible-band algae absorption at optical radiation levels."""
    klev = dzk.shape[0] - 1
    absorption = jnp.zeros((nspint, klev + 1), dtype=dzk.dtype)
    if bool(options["tr_bgc_N"]) and bool(options["dEdd_algae"]):
        nlt = int(options["nlt_chl_sw"])
        absorption = absorption.at[0, :].set(0.01 * zbio[nlt - 1:nlt + klev])
    else:
        absorption = absorption.at[0, klev].set(params["kalg"] * (0.5 / dzk[klev]))
    return absorption


def ice_column_optics_jax(ns, dzk, kabs_chl, ki_ssl, wi_ssl, gi_ssl,
                           ki_dl, wi_dl, gi_dl, ki_int, wi_int, gi_int,
                           nslyr, nilyr):
    """Sea-ice SSL, drained layer and interior optical properties."""
    klev = dzk.shape[0] - 1
    kii = nslyr + 1
    tau = jnp.zeros_like(dzk)
    w0 = jnp.zeros_like(dzk)
    g = jnp.zeros_like(dzk)
    k = kii
    kext = ki_ssl[ns] + kabs_chl[ns, k]
    tau = tau.at[k].set(kext * dzk[k])
    w0 = w0.at[k].set(ki_ssl[ns] / kext * wi_ssl[ns])
    g = g.at[k].set(gi_ssl[ns])
    k = kii + 1
    kext = ki_dl[ns] + kabs_chl[ns, k]
    tau = tau.at[k].set(kext * dzk[k] * (0.25 * nilyr))
    w0 = w0.at[k].set(ki_dl[ns] / kext * wi_dl[ns])
    g = g.at[k].set(gi_dl[ns])
    for k in range(kii + 2, klev):
        kext = ki_int[ns] + kabs_chl[ns, k]
        tau = tau.at[k].set(kext * dzk[k])
        w0 = w0.at[k].set(ki_int[ns] / kext * wi_int[ns])
        g = g.at[k].set(gi_int[ns])
    # Lowest layer has a separately accumulated chlorophyll absorber.
    kabs = ki_int[ns] * (1.0 - wi_int[ns])
    if ns == 0:
        kabs = kabs + kabs_chl[ns, klev]
    sig = ki_int[ns] * wi_int[ns]
    tau = tau.at[klev].set((kabs + sig) * dzk[klev])
    w0 = w0.at[klev].set(sig / (sig + kabs))
    g = g.at[klev].set(gi_int[ns])
    return tau, w0, g


def aerosol_optics_jax(ns, tau, w0, g, dzk, zbio, aero_mp,
                       snow_radius, snow, data, options):
    """Mix dissolved or bulk/modal aerosols into snow or ice optics."""
    if not (bool(options["tr_aero"]) or
            (bool(options["tr_zaero"]) and bool(options["dEdd_algae"]))):
        return tau, w0, g
    klev = dzk.shape[0] - 1
    nslyr = int(options["nslyr"])
    nilyr = int(options["nilyr"])
    n_aero = int(options["n_aero"])
    five_band = bool(options["five_band"])
    suffix = "_5bd" if five_band else "_3bd"
    kaer_bc = data["kaer_bc" + suffix]
    waer_bc = data["waer_bc" + suffix]
    gaer_bc = data["gaer_bc" + suffix]
    bcenh = data["bcenh" + suffix]
    kaer = data["kaer" + suffix]
    waer = data["waer" + suffix]
    gaer = data["gaer" + suffix]
    first = 0 if snow else nslyr + 1
    last = nslyr if snow else klev

    if bool(options["tr_zaero"]) and bool(options["dEdd_algae"]):
        n_zaero = int(options["n_zaero"])
        nlt_zaero_sw = options["nlt_zaero_sw"]
        for k in range(first, last + 1):
            ksnow = max(k, 1)
            bcini = grain_index_jax(snow_radius[ksnow - 1]) if k < nslyr + 1 else 8
            ie = 1
            ii = 1
            taer = jnp.asarray(0.0, dtype=tau.dtype)
            waer_sum = taer
            gaer_sum = taer
            for species in range(n_zaero):
                mass = zbio[int(nlt_zaero_sw[species]) + k - 1] * dzk[k]
                if bool(options["modal_aero"]) and species == 0:
                    ext = kaer_bc[ns, ie]
                    ssa = waer_bc[ns, ie]
                    asm = gaer_bc[ns, ie]
                    ta = ext * mass
                    wa = ext * ssa * mass
                    ga = ext * ssa * asm * mass
                elif bool(options["modal_aero"]) and species == 1:
                    ext = kaer_bc[ns, ii]
                    ssa = waer_bc[ns, ii]
                    asm = gaer_bc[ns, ii]
                    ta = ext * bcenh[ns, ii, bcini - 1] * mass
                    wa = ext * ssa * mass
                    ga = ext * ssa * asm * mass
                else:
                    ext = kaer[ns, species]
                    ssa = waer[ns, species]
                    asm = gaer[ns, species]
                    ta = ext * mass
                    wa = ta * ssa
                    ga = wa * asm
                taer = taer + ta
                waer_sum = waer_sum + wa
                gaer_sum = gaer_sum + ga
            denom_w = w0[k] * tau[k] + waer_sum
            denom_t = tau[k] + taer
            g = g.at[k].set((g[k] * w0[k] * tau[k] + gaer_sum) / denom_w)
            w0 = w0.at[k].set(denom_w / denom_t)
            tau = tau.at[k].set(denom_t)
    elif bool(options["tr_aero"]):
        for k in range(first, last + 1):
            ksnow = max(k, 1)
            bcini = grain_index_jax(snow_radius[ksnow - 1]) if k < nslyr + 1 else 8
            if snow:
                offset = 0 if k == 0 else 1
                scale = 1.0 if k == 0 else 1.0 / nslyr
            else:
                offset = 2 if k == nslyr + 1 else 3
                scale = 1.0 if k == nslyr + 1 else 1.0 / nilyr
            taer = jnp.asarray(0.0, dtype=tau.dtype)
            waer_sum = taer
            gaer_sum = taer
            for species in range(n_aero):
                amp = aero_mp[4 * species + offset] * scale
                if bool(options["modal_aero"]) and species < 2:
                    ext = kaer_bc[ns, 1]
                    ssa = waer_bc[ns, 1]
                    asm = gaer_bc[ns, 1]
                    ta = amp * ext
                    if species == 1:
                        ta = ta * bcenh[ns, 1, bcini - 1]
                    wa = amp * ext * ssa
                    ga = wa * asm
                else:
                    ext = kaer[ns, species]
                    ssa = waer[ns, species]
                    asm = gaer[ns, species]
                    ta = amp * ext
                    wa = ta * ssa
                    ga = wa * asm
                taer = taer + ta
                waer_sum = waer_sum + wa
                gaer_sum = gaer_sum + ga
            denom_w = w0[k] * tau[k] + waer_sum
            denom_t = tau[k] + taer
            g = g.at[k].set((g[k] * w0[k] * tau[k] + gaer_sum) / denom_w)
            w0 = w0.at[k].set(denom_w / denom_t)
            tau = tau.at[k].set(denom_t)
    return tau, w0, g


def optical_layers_jax(ns, srftyp, direct, dzk, kabs_chl,
                       snow_radius, rhosnw, hp, zbio, aero_mp,
                       data, options, params):
    """Construct optical depth, scattering albedo and asymmetry by layer."""
    nslyr = int(options["nslyr"])
    nilyr = int(options["nilyr"])
    klev = dzk.shape[0] - 1
    kii = nslyr + 1
    five_band = bool(options["five_band"])
    suffix = "_5bd" if five_band else "_3bd"
    fp_ice, fm_ice = 0.15, 0.15
    ki_ssl, wi_ssl, gi_ssl = tune_scat_jax(
        data["ki_ssl_mn" + suffix], data["wi_ssl_mn" + suffix],
        data["gi_ssl_mn" + suffix], fp_ice, fm_ice, params["R_ice"])
    ki_dl, wi_dl, gi_dl = tune_scat_jax(
        data["ki_dl_mn" + suffix], data["wi_dl_mn" + suffix],
        data["gi_dl_mn" + suffix], fp_ice, fm_ice, params["R_ice"])
    ki_int, wi_int, gi_int = tune_scat_jax(
        data["ki_int_mn" + suffix], data["wi_int_mn" + suffix],
        data["gi_int_mn" + suffix], fp_ice, fm_ice, params["R_ice"])
    tau = jnp.zeros_like(dzk)
    w0 = jnp.zeros_like(dzk)
    g = jnp.zeros_like(dzk)

    if srftyp == 1:
        for k in range(nslyr + 1):
            ksnow = max(k, 1) - 1
            radius = snow_radius[ksnow]
            if five_band:
                ks, ws, gs = snicar_iop_jax(ns, radius, direct, data)
                ks = ks * rhosnw[ksnow]
            else:
                Qs, ws, gs = snow_table_3bd_jax(ns, radius, data)
                ks = Qs * ((rhosnw[ksnow] / params["rhoi"]) * 3.0
                           / (4.0 * radius * 1.0e-6))
            kext = ks + kabs_chl[ns, k]
            tau = tau.at[k].set(kext * dzk[k])
            w0 = w0.at[k].set(ks / kext * ws)
            g = g.at[k].set(gs)
        tau, w0, g = aerosol_optics_jax(
            ns, tau, w0, g, dzk, zbio, aero_mp, snow_radius,
            True, data, options)
    elif srftyp == 2:
        dz = hp / (nslyr + 1)
        for k in range(nslyr + 1):
            tau = tau.at[k].set(data["kw"][ns] * dz)
            w0 = w0.at[k].set(data["ww"][ns])
            g = g.at[k].set(data["gw"][ns])

    if srftyp <= 1:
        tau_ice, w0_ice, g_ice = ice_column_optics_jax(
            ns, dzk, kabs_chl, ki_ssl, wi_ssl, gi_ssl,
            ki_dl, wi_dl, gi_dl, ki_int, wi_int, gi_int, nslyr, nilyr)
        tau = tau.at[kii:].set(tau_ice[kii:])
        w0 = w0.at[kii:].set(w0_ice[kii:])
        g = g.at[kii:].set(g_ice[kii:])
        tau, w0, g = aerosol_optics_jax(
            ns, tau, w0, g, dzk, zbio, aero_mp, snow_radius,
            False, data, options)
    else:
        ki_p_ssl, wi_p_ssl, gi_p_ssl = tune_scat_jax(
            data["ki_p_ssl_mn"], data["wi_p_ssl_mn"],
            data["gi_p_ssl_mn"], 2.0, 0.50, params["R_pnd"])
        ki_p_int, wi_p_int, gi_p_int = tune_scat_jax(
            data["ki_p_int_mn"], data["wi_p_int_mn"],
            data["gi_p_int_mn"], 2.0, 0.50, params["R_pnd"])
        for k in range(kii, klev + 1):
            ssl = k == kii
            ki = ki_p_ssl[ns] if ssl else ki_p_int[ns]
            wi = wi_p_ssl[ns] if ssl else wi_p_int[ns]
            gi = gi_p_ssl[ns] if ssl else gi_p_int[ns]
            tau = tau.at[k].set(ki * dzk[k])
            w0 = w0.at[k].set(wi)
            g = g.at[k].set(gi)
        if not bool(options["tr_pond_sealvl"]):
            transition = (hp >= params["hpmin"]) & (hp < params["hp0"])
            fraction = hp / params["hp0"]
            for k in range(kii, klev + 1):
                if k == kii:
                    sig_i = ki_ssl[ns] * wi_ssl[ns]
                    sig_p = ki_p_ssl[ns] * wi_p_ssl[ns]
                    absorption = ki_p_ssl[ns] * (1.0 - wi_p_ssl[ns])
                elif k == kii + 1:
                    sig_i = ki_dl[ns] * wi_dl[ns] * (0.25 * nilyr)
                    sig_p = ki_p_int[ns] * wi_p_int[ns]
                    absorption = ki_p_int[ns] * (1.0 - wi_p_int[ns])
                else:
                    sig_i = ki_int[ns] * wi_int[ns]
                    sig_p = ki_p_int[ns] * wi_p_int[ns]
                    absorption = ki_p_int[ns] * (1.0 - wi_p_int[ns])
                sig = sig_i + (sig_p - sig_i) * fraction
                kext = sig + absorption
                tau = tau.at[k].set(jnp.where(transition, kext * dzk[k], tau[k]))
                w0 = w0.at[k].set(jnp.where(transition, sig / kext, w0[k]))
                g = g.at[k].set(jnp.where(transition, gi_p_int[ns], g[k]))
    return tau, w0, g


def compute_dEdd_3bd_jax(srftyp, coszen, hi, hs, hp, fi, rhosnw,
                          rsnw, aero_mp, zbio, swvdr, swvdf, swidr, swidf,
                          swuvrdr, swuvrdf, swpardr, swpardf,
                          data, options, params):
    """Three-band shortwave over bare ice, snow or a melt pond."""
    nslyr = int(options["nslyr"])
    nilyr = int(options["nilyr"])
    klev = nslyr + nilyr + 1
    klevp = klev + 1
    dzk = radiation_dzk_jax(hs, hi, nslyr, nilyr, params)
    kabs_chl = chlorophyll_jax(3, dzk, zbio, options, params)
    fnidr = jnp.where(swidr + swidf > params["puny"],
                       swidr / jnp.where(swidr + swidf > params["puny"],
                                          swidr + swidf, 1.0), 0.0)
    nir_weight = 0.67 + (0.78 - 0.67) * (1.0 - fnidr)
    weights = (1.0, nir_weight, 1.0 - nir_weight)
    snow_radius = (fnidr + 0.80 * (1.0 - fnidr)) * rsnw
    ksrf = 1 if srftyp == 1 else nslyr + 2
    ice_top = nslyr + 2 + jnp.arange(nilyr)
    ice_bottom = ice_top + 1
    # Snow-covered ice includes the drained layer in the first ice layer.
    if srftyp == 1:
        ice_top = jnp.where(jnp.arange(nilyr) == 0, ice_top - 1, ice_top)
    alvdr = jnp.asarray(0.0, dtype=hi.dtype)
    alvdf = alvdr
    alidr = alvdr
    alidf = alvdr
    fswsfc = alvdr
    fswint = alvdr
    fswthru = alvdr
    spectral = jnp.zeros(8, dtype=hi.dtype)
    Sswabs = jnp.zeros(nslyr, dtype=hi.dtype)
    Iswabs = jnp.zeros(nilyr, dtype=hi.dtype)
    fswpenl = jnp.zeros(nilyr + 1, dtype=hi.dtype)

    for ns in range(3):
        tau, w0, g = optical_layers_jax(
            ns, srftyp, True, dzk, kabs_chl, snow_radius,
            rhosnw, hp, zbio, aero_mp, data, options, params)
        (trndir, trntdr, trndif, rupdir, rupdif,
         rdndif) = solution_dEdd_jax(
            coszen, srftyp, tau, w0, g,
            0.01 if ns == 0 else 0.0, 0.01 if ns == 0 else 0.0, nslyr)
        dfdir, dfdif = interface_flux_jax(
            trndir, trntdr, trndif, rupdir, rupdif, rdndif, params["puny"])
        swdr = swvdr if ns == 0 else swidr
        swdf = swvdf if ns == 0 else swidf
        weight = weights[ns]
        top = (dfdir[0] * swdr + dfdif[0] * swdf) * weight
        surface = (dfdir[ksrf] * swdr + dfdif[ksrf] * swdf) * weight
        bottom = (dfdir[klevp] * swdr + dfdif[klevp] * swdf) * weight
        fswsfc = fswsfc + top - surface
        fswint = fswint + surface - bottom
        fswthru = fswthru + bottom

        if ns == 0:
            alvdr = rupdir[0]
            alvdf = rupdif[0]
            fswpenl = (dfdir[nslyr + 2:klevp + 1] * swdr
                       + dfdif[nslyr + 2:klevp + 1] * swdf)
            spectral = spectral.at[0].set(dfdir[klevp] * swdr)
            spectral = spectral.at[1].set(dfdif[klevp] * swdf)
            spectral = spectral.at[4].set(dfdir[klevp] * swuvrdr)
            spectral = spectral.at[5].set(dfdif[klevp] * swuvrdf)
            spectral = spectral.at[6].set(dfdir[klevp] * swpardr)
            spectral = spectral.at[7].set(dfdif[klevp] * swpardf)
        else:
            alidr = alidr + rupdir[0] * weight
            alidf = alidf + rupdif[0] * weight
            spectral = spectral.at[2].add(dfdir[klevp] * swdr * weight)
            spectral = spectral.at[3].add(dfdif[klevp] * swdf * weight)

        if srftyp == 1:
            if ns == 0:
                Sswabs = (Sswabs + dfdir[1:nslyr + 1] * swdr
                          + dfdif[1:nslyr + 1] * swdf
                          - (dfdir[2:nslyr + 2] * swdr
                             + dfdif[2:nslyr + 2] * swdf))
            else:
                Sswabs = Sswabs + (
                    dfdir[1:nslyr + 1] * swdr + dfdif[1:nslyr + 1] * swdf
                    - (dfdir[2:nslyr + 2] * swdr
                       + dfdif[2:nslyr + 2] * swdf)) * weight
        if ns == 0:
            Iswabs = (Iswabs + dfdir[ice_top] * swdr + dfdif[ice_top] * swdf
                      - (dfdir[ice_bottom] * swdr + dfdif[ice_bottom] * swdf))
        else:
            Iswabs = Iswabs + (
                dfdir[ice_top] * swdr + dfdif[ice_top] * swdf
                - (dfdir[ice_bottom] * swdr + dfdif[ice_bottom] * swdf)) * weight

    return (alvdr, alvdf, alidr, alidf, fi * fswsfc, fi * fswint,
            fi * fswthru, fi * spectral, fi * Sswabs, fi * Iswabs,
            fi * fswpenl)


def compute_dEdd_5bd_jax(coszen, hi, hs, fi, rhosnw, rsnw,
                          aero_mp, zbio, swvdr, swvdf, swidr, swidf,
                          data, options, params):
    """Five-band SNICAR snow with distinct direct and diffuse optics."""
    nslyr = int(options["nslyr"])
    nilyr = int(options["nilyr"])
    klev = nslyr + nilyr + 1
    klevp = klev + 1
    dzk = radiation_dzk_jax(hs, hi, nslyr, nilyr, params)
    kabs_chl = chlorophyll_jax(5, dzk, zbio, options, params)
    direct_weights = (1.0, 0.49352158521175, 0.18099494230665,
                      0.12094898498813,
                      1.0 - 0.49352158521175 - 0.18099494230665 - 0.12094898498813)
    diffuse_weights = (1.0, 0.58581507618433, 0.20156903770812,
                       0.10917889346386,
                       1.0 - 0.58581507618433 - 0.20156903770812 - 0.10917889346386)
    ice_top = nslyr + 2 + jnp.arange(nilyr)
    ice_bottom = ice_top + 1
    ice_top = jnp.where(jnp.arange(nilyr) == 0, ice_top - 1, ice_top)
    alvdr = jnp.asarray(0.0, dtype=hi.dtype)
    alvdf = alvdr
    alidr = alvdr
    alidf = alvdr
    fswsfc = alvdr
    fswint = alvdr
    fswthru = alvdr
    spectral = jnp.zeros(8, dtype=hi.dtype)
    Sswabs = jnp.zeros(nslyr, dtype=hi.dtype)
    Iswabs = jnp.zeros(nilyr, dtype=hi.dtype)
    fswpenl = jnp.zeros(nilyr + 1, dtype=hi.dtype)

    for ns in range(5):
        # The snow optical tables differ between direct and diffuse beams.
        flux_by_sky = []
        for direct in (True, False):
            tau, w0, g = optical_layers_jax(
                ns, 1, direct, dzk, kabs_chl, rsnw,
                rhosnw, 0.0, zbio, aero_mp, data, options, params)
            (trndir, trntdr, trndif, rupdir, rupdif,
             rdndif) = solution_dEdd_jax(
                coszen, 1, tau, w0, g,
                0.01 if ns == 0 else 0.0, 0.01 if ns == 0 else 0.0, nslyr)
            dfdir, dfdif = interface_flux_jax(
                trndir, trntdr, trndif, rupdir, rupdif, rdndif, params["puny"])
            flux_by_sky.append((dfdir, dfdif, rupdir, rupdif))
        dfdir, _, rupdir, _ = flux_by_sky[0]
        _, dfdif, _, rupdif = flux_by_sky[1]
        swdr = swvdr if ns == 0 else swidr
        swdf = swvdf if ns == 0 else swidf
        wdir = direct_weights[ns]
        wdif = diffuse_weights[ns]
        top = dfdir[0] * swdr * wdir + dfdif[0] * swdf * wdif
        surface = dfdir[1] * swdr * wdir + dfdif[1] * swdf * wdif
        bottom = dfdir[klevp] * swdr * wdir + dfdif[klevp] * swdf * wdif
        fswsfc = fswsfc + top - surface
        fswint = fswint + surface - bottom
        fswthru = fswthru + bottom

        if ns == 0:
            alvdr = rupdir[0]
            alvdf = rupdif[0]
            fswpenl = (dfdir[nslyr + 2:klevp + 1] * swdr
                       + dfdif[nslyr + 2:klevp + 1] * swdf)
            spectral = spectral.at[0].set(dfdir[klevp] * swdr)
            spectral = spectral.at[1].set(dfdif[klevp] * swdf)
        else:
            alidr = alidr + rupdir[0] * wdir
            alidf = alidf + rupdif[0] * wdif
            spectral = spectral.at[2].add(dfdir[klevp] * swdr * wdir)
            spectral = spectral.at[3].add(dfdif[klevp] * swdf * wdif)

        above = (dfdir[1:nslyr + 1] * swdr * wdir
                 + dfdif[1:nslyr + 1] * swdf * wdif)
        below = (dfdir[2:nslyr + 2] * swdr * wdir
                 + dfdif[2:nslyr + 2] * swdf * wdif)
        Sswabs = Sswabs + (above - below)
        above = dfdir[ice_top] * swdr * wdir + dfdif[ice_top] * swdf * wdif
        below = dfdir[ice_bottom] * swdr * wdir + dfdif[ice_bottom] * swdf * wdif
        Iswabs = Iswabs + (above - below)

    # SNICAR adjusts NIR direct albedo above a 75-degree solar zenith angle.
    mu0 = jnp.maximum(coszen, 0.01)
    sza_c1 = 0.085730 - 0.630883 * mu0 + 1.303723 * mu0 ** 2
    sza_c0 = 1.467291 - 3.338043 * mu0 + 6.807489 * mu0 ** 2
    sza_factor = jnp.where(mu0 < 0.2588,
                           sza_c1 * (jnp.log10(rsnw[0]) - 6.0) + sza_c0, 1.0)
    fswsfc = fswsfc - (sza_factor - 1.0) * alidr * swidr
    return (alvdr, alvdf, alidr * sza_factor, alidf,
            fi * fswsfc, fi * fswint, fi * fswthru,
            fi * spectral, fi * Sswabs, fi * Iswabs, fi * fswpenl)


def shortwave_dEdd_jax(coszen, aice, vice, hs, fs, rhosnw, rsnw,
                        fp, hp, aero, zbio,
                        swvdr, swvdf, swidr, swidf,
                        swuvrdr, swuvrdf, swpardr, swpardf,
                        data, options, params):
    """Combine bare, snow and pond shortwave over one ice category."""
    nslyr = int(options["nslyr"])
    nilyr = int(options["nilyr"])
    zero = jnp.asarray(0.0, dtype=aice.dtype)
    empty = (zero, zero, zero, zero, zero, zero, zero,
             jnp.zeros(8, dtype=aice.dtype),
             jnp.zeros(nslyr, dtype=aice.dtype),
             jnp.zeros(nilyr, dtype=aice.dtype),
             jnp.zeros(nilyr + 1, dtype=aice.dtype))
    netsw = swvdr + swidr + swvdf + swidf
    vsno = hs * aice
    aero_mp = jnp.zeros_like(aero)
    if bool(options["tr_aero"]):
        for species in range(int(options["n_aero"])):
            start = 4 * species
            aero_mp = aero_mp.at[start:start + 2].set(aero[start:start + 2] * vsno)
            aero_mp = aero_mp.at[start + 2:start + 4].set(aero[start + 2:start + 4] * vice)
    options_3bd = dict(options)
    options_3bd["five_band"] = False

    def sunlit(_):
        hi = vice / aice
        fi = 1.0 - fs - fp
        bare = jax.lax.cond(
            fi > 0.0,
            lambda _: compute_dEdd_3bd_jax(
                0, coszen, hi, 0.0, hp, fi, rhosnw, rsnw,
                aero_mp, zbio, swvdr, swvdf, swidr, swidf,
                swuvrdr, swuvrdf, swpardr, swpardf, data, options_3bd, params),
            lambda _: empty, None)
        if bool(options["five_band"]):
            snow = jax.lax.cond(
                fs > 0.0,
                lambda _: compute_dEdd_5bd_jax(
                    coszen, hi, hs, fs, rhosnw, rsnw, aero_mp, zbio,
                    swvdr, swvdf, swidr, swidf, data, options, params),
                lambda _: empty, None)
        else:
            snow = jax.lax.cond(
                fs > 0.0,
                lambda _: compute_dEdd_3bd_jax(
                    1, coszen, hi, hs, hp, fs, rhosnw, rsnw,
                    aero_mp, zbio, swvdr, swvdf, swidr, swidf,
                    swuvrdr, swuvrdf, swpardr, swpardf, data, options_3bd, params),
                lambda _: empty, None)
        pond = jax.lax.cond(
            fp > params["puny"],
            lambda _: compute_dEdd_3bd_jax(
                2, coszen, hi, hs, hp, fp, rhosnw, rsnw,
                aero_mp, zbio, swvdr, swvdf, swidr, swidf,
                swuvrdr, swuvrdf, swpardr, swpardf, data, options_3bd, params),
            lambda _: empty, None)
        albedos = tuple(bare[k] * fi + snow[k] * fs + pond[k] * fp
                        for k in range(4))
        fluxes = tuple(bare[k] + snow[k] + pond[k] for k in range(4, 11))
        awvdr, awvdf = params["awtvdr"], params["awtvdf"]
        awidr, awidf = params["awtidr"], params["awtidf"]
        albice = awvdr * bare[0] + awidr * bare[2] + awvdf * bare[1] + awidf * bare[3]
        albsno = awvdr * snow[0] + awidr * snow[2] + awvdf * snow[1] + awidf * snow[3]
        albpnd = awvdr * pond[0] + awidr * pond[2] + awvdf * pond[1] + awidf * pond[3]
        return albedos + fluxes + (albice, albsno, albpnd)

    def sunless(_):
        return ((jnp.asarray(1.0, dtype=aice.dtype),) * 4
                + empty[4:] + (zero, zero, zero))

    return jax.lax.cond(netsw > params["puny"], sunlit, sunless, None)


def compute_coszen_jax(tlat, tlon, yday, sec, orbital, params):
    """Standalone CICE solar zenith cosine for the next radiation step."""
    calday = yday + sec / params["secday"]
    lambm = orbital["lambm0"] + (calday - 80.5) * 2.0 * jnp.pi / 365.0
    lmm = lambm - orbital["mvelpp"]
    eccen = orbital["eccen"]
    sinl = jnp.sin(lmm)
    lamb = lambm + eccen * (
        2.0 * sinl + eccen * (
            1.25 * jnp.sin(2.0 * lmm)
            + eccen * ((13.0 / 12.0) * jnp.sin(3.0 * lmm) - 0.25 * sinl)))
    decln = jnp.arcsin(jnp.sin(orbital["obliqr"]) * jnp.sin(lamb))
    return (jnp.sin(tlat) * jnp.sin(decln)
            + jnp.cos(tlat) * jnp.cos(decln)
            * jnp.cos((sec / params["secday"] - 0.5) * 2.0 * jnp.pi + tlon))


def remap_zbgc_jax(tracer, nlyrn, nbyrn, hice, hinS,
                    ice_grid, bio_grid, S_min):
    """Bio-grid to shortwave-grid tracer remap used by radiation."""
    def with_surface_extension(_):
        donor_grid = jnp.concatenate((
            jnp.asarray([-hice + hinS, (hinS - hice) * 0.5]),
            bio_grid[:nbyrn] * hinS,
            jnp.asarray([hinS])))
        top = (S_min * (hice - hinS)
               + tracer[0] * 0.5 * (donor_grid[3] - donor_grid[2])) / (
                   hice - hinS + 0.5 * (donor_grid[3] - donor_grid[2]))
        donor_tracer = jnp.concatenate((
            jnp.asarray([top, top]), tracer[:nbyrn], tracer[nbyrn - 1:nbyrn]))
        receiver_grid = (ice_grid[:nlyrn] - 1.0) * hice + hinS
        return jnp.interp(receiver_grid, donor_grid, donor_tracer)

    def without_surface_extension(_):
        donor_grid = jnp.concatenate((
            jnp.asarray([hice - hinS]),
            (bio_grid[:nbyrn] - 1.0) * hinS + hice,
            jnp.asarray([hice])))
        donor_tracer = jnp.concatenate((
            tracer[:1], tracer[:nbyrn], tracer[nbyrn - 1:nbyrn]))
        receiver_grid = ice_grid[:nlyrn] * hice
        return jnp.interp(receiver_grid, donor_grid, donor_tracer)

    return jax.lax.cond(hice > hinS, with_surface_extension,
                        without_surface_extension, None)


def compute_shortwave_trcr_jax(bgcN, zaero, sw_grid, i_grid,
                                hin, hbri, options, params):
    """Map algal chlorophyll and vertical aerosols to optical levels."""
    nslyr = int(options["nslyr"])
    nilyr = int(options["nilyr"])
    nblyr = int(options["nblyr"])
    zbio = jnp.zeros(int(options["nbtrcr_sw"]), dtype=bgcN.dtype)
    icegrid = sw_grid[:nilyr + 1]
    icegrid = icegrid.at[0].set(jnp.where(
        (sw_grid[0] * hin * 2.0 > params["hi_ssl"]) & (hin > params["puny"]),
        params["hi_ssl"] / (2.0 * jnp.where(hin > params["puny"], hin, 1.0)),
        icegrid[0]))
    icegrid = icegrid.at[1].set(2.0 * sw_grid[0] + (sw_grid[1] - sw_grid[0]))

    if bool(options["z_tracers"]):
        if bool(options["tr_bgc_N"]):
            first = int(options["nt_bgc_N"][0])
            chl = jnp.zeros(nblyr + 1, dtype=bgcN.dtype)
            scales = (
                params["ratio_chl2N_diatoms"] * params["F_abs_chl_diatoms"],
                params["ratio_chl2N_sp"] * params["F_abs_chl_sp"],
                params["ratio_chl2N_phaeo"] * params["F_abs_chl_phaeo"],
            )
            for species in range(int(options["n_algae"])):
                base = int(options["nt_bgc_N"][species]) - first
                chl = chl + scales[species] * bgcN[base:base + nblyr + 1]
            mapped = remap_zbgc_jax(
                chl, nilyr + 1, nblyr + 1, hin, hbri,
                icegrid, i_grid, chl[0] * params["min_bgc"])
            nlt = int(options["nlt_chl_sw"])
            zbio = zbio.at[nlt + nslyr:nlt + nslyr + nilyr + 1].set(mapped)
            for species in range(int(options["n_algae"])):
                base = int(options["nt_bgc_N"][species]) - first
                zbio = zbio.at[nlt - 1].add(scales[species] * bgcN[base + nblyr + 1])
                zbio = zbio.at[nlt:nlt + nslyr].add(
                    scales[species] * bgcN[base + nblyr + 2])

        if bool(options["tr_zaero"]):
            first = int(options["nt_zaero"][0])
            for species in range(int(options["n_zaero"])):
                base = int(options["nt_zaero"][species]) - first
                mapped = remap_zbgc_jax(
                    zaero[base:base + nblyr + 1],
                    nilyr + 1, nblyr + 1, hin, hbri,
                    icegrid, i_grid, zaero[base] * params["min_bgc"])
                nlt = int(options["nlt_zaero_sw"][species])
                zbio = zbio.at[nlt + nslyr:nlt + nslyr + nilyr + 1].set(mapped)
                zbio = zbio.at[nlt - 1].set(zaero[base + nblyr + 1])
                zbio = zbio.at[nlt:nlt + nslyr].set(zaero[base + nblyr + 2])
    elif bool(options["skl_bgc"]):
        first = int(options["nt_bgc_N"][0])
        for species in range(int(options["n_algae"])):
            base = int(options["nt_bgc_N"][species]) - first
            scale = (params[("F_abs_chl_diatoms", "F_abs_chl_sp", "F_abs_chl_phaeo")[species]]
                     * params[("ratio_chl2N_diatoms", "ratio_chl2N_sp", "ratio_chl2N_phaeo")[species]])
            zbio = zbio.at[-1].add(
                scale * bgcN[base] * params["sk_l"] / hin * nilyr)
    return zbio


def icepack_step_radiation_jax(aice, vice, vsno, Tsfc, alvl,
                               apnd, hpnd, ipnd, aero, bgcN, zaero,
                               fbri, rsnow, dhsn, ffracn,
                               coszen, fsnow, dt,
                               swvdr, swvdf, swidr, swidf,
                               swuvrdr, swuvrdf, swpardr, swpardf,
                               sw_grid, i_grid, data, options, params):
    """One-category Delta-Eddington radiation and optical tracer fields."""
    return _icepack_step_radiation_jax(
        aice, vice, vsno, Tsfc, alvl, apnd, hpnd, ipnd, aero, bgcN, zaero,
        fbri, rsnow, dhsn, ffracn, coszen, fsnow, dt,
        swvdr, swvdf, swidr, swidf, swuvrdr, swuvrdf, swpardr, swpardf,
        sw_grid, i_grid, data, tuple(sorted(options.items())), params)


@partial(jax.jit, static_argnames=("options",))
def _icepack_step_radiation_jax(aice, vice, vsno, Tsfc, alvl,
                                apnd, hpnd, ipnd, aero, bgcN, zaero,
                                fbri, rsnow, dhsn, ffracn,
                                coszen, fsnow, dt,
                                swvdr, swvdf, swidr, swidf,
                                swuvrdr, swuvrdf, swpardr, swpardf,
                                sw_grid, i_grid, data, options, params):
    # Freeze only scheme/layout selectors. Forcing and optical tables stay
    # dynamic so every time step can reuse the same compiled radiation kernel.
    options = dict(options)
    nslyr = int(options["nslyr"])
    nilyr = int(options["nilyr"])
    zero = jnp.asarray(0.0, dtype=aice.dtype)
    zbio_empty = jnp.zeros(int(options["nbtrcr_sw"]), dtype=aice.dtype)
    outputs_empty = ((zero,) * 10
                     + (jnp.zeros(8, dtype=aice.dtype),
                        jnp.zeros(nslyr, dtype=aice.dtype),
                        jnp.zeros(nilyr, dtype=aice.dtype),
                        jnp.zeros(nilyr + 1, dtype=aice.dtype),
                        zero, zero, dhsn, zbio_empty))

    def with_ice(_):
        hin = vice / aice
        zbio = zbio_empty
        if bool(options["dEdd_algae"]):
            zbio = compute_shortwave_trcr_jax(
                bgcN, zaero, sw_grid, i_grid, hin, fbri * hin,
                options, params)
        fsn, hsn, rhosnwn, rsnwn = shortwave_dEdd_set_snow_jax(
            aice, vsno, Tsfc, rsnow, options, params)
        apeff = zero
        dhs = dhsn
        if bool(options["tr_pond_lvl"]) or bool(options["tr_pond_sealvl"]):
            apond = alvl * apnd if bool(options["tr_pond_lvl"]) else apnd
            (fsn, hsn, dhs, fpn, hpn, apeff,
             rhosnwn, rsnwn) = shortwave_dEdd_set_eff_jax(
                aice, vsno, alvl, apond, hpnd, ipnd, ffracn,
                fsnow, dt, Tsfc, fsn, hsn, dhsn, rsnow,
                rhosnwn, rsnwn, options, params)
        elif bool(options["tr_pond_topo"]):
            fpn = jnp.where((apnd * aice > params["puny"])
                            & (ipnd < params["hp1"]), apnd, 0.0)
            hpn = jnp.where(apnd > params["puny"], hpnd, 0.0)
            fpn = jnp.where(hpn < params["hpmin"], 0.0, fpn)
            fsn = jnp.minimum(fsn, 1.0 - fpn)
            apeff = fpn
        else:
            fT = -jnp.minimum((params["Timelt"] - Tsfc) - 1.0, 0.0)
            apeff = 0.3 * fT * (1.0 - fsn)
            fpn = zero
            hpn = zero

        (alvdr, alvdf, alidr, alidf, fswsfc, fswint, fswthru,
         spectral, Sswabs, Iswabs, fswpenl,
         albice, albsno, albpnd) = shortwave_dEdd_jax(
            coszen, aice, vice, hsn, fsn, rhosnwn, rsnwn,
            fpn, hpn, aero, zbio,
            swvdr, swvdf, swidr, swidf,
            swuvrdr, swuvrdf, swpardr, swpardf,
            data, options, params)
        return (alvdr, alvdf, alidr, alidf, fswsfc, fswint, fswthru,
                albice, albsno, albpnd, spectral, Sswabs, Iswabs,
                fswpenl, apeff, fsn, dhs, zbio)

    return jax.lax.cond(aice > params["puny"], with_ice,
                        lambda _: outputs_empty, None)
