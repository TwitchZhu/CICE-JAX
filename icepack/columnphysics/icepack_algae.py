"""JAX sea-ice biological column routines from icepack_algae.F90."""

import jax
import jax.numpy as jnp

from icepack.columnphysics.icepack_zbgc_shared import (
    merge_bgc_fluxes, merge_bgc_fluxes_skl, regrid_stationary,
    remap_zbgc_jax,
)
from icepack.columnphysics.icepack_aerosol import update_snow_bgc


def algal_dyn(dt, dEdd_algae, fswthru, reactb, ltrcrn, T_bot,
              flags, tracer_indices, parameters, bio_parameters):
    """Algal growth and nutrient reactions from icepack_algae.F90::algal_dyn.

    Tracer indices retain Fortran's one-based numbering in the static tree.
    The returned reaction vector is an amount over ``dt``, not a rate.
    """
    n_algae = int(flags["n_algae"])
    n_doc = int(flags["n_doc"])
    n_dic = int(flags["n_dic"])
    n_don = int(flags["n_don"])
    n_fed = int(flags["n_fed"])
    n_fep = int(flags["n_fep"])
    dtype = ltrcrn.dtype
    zero = jnp.asarray(0.0, dtype=dtype)
    puny = parameters["puny"]
    secday = parameters["secday"]

    # Initial brine concentrations and chlorophyll optical depth.
    Nitin = ltrcrn[int(tracer_indices["nlt_bgc_Nit"]) - 1]
    Nin = jnp.stack([ltrcrn[k - 1] for k in tracer_indices["nlt_bgc_N"][:n_algae]])
    op_dep = jnp.sum(bio_parameters["chlabs"][:n_algae]
                     * bio_parameters["R_chl2N"][:n_algae] * Nin)
    Amin = (ltrcrn[int(tracer_indices["nlt_bgc_Am"]) - 1]
            if flags["tr_bgc_Am"] else zero)
    Silin = (ltrcrn[int(tracer_indices["nlt_bgc_Sil"]) - 1]
             if flags["tr_bgc_Sil"] else zero)
    DOCin = (jnp.stack([ltrcrn[k - 1] for k in tracer_indices["nlt_bgc_DOC"][:n_doc]])
             if flags["tr_bgc_C"] and n_doc else jnp.zeros((n_doc,), dtype=dtype))
    DONin = (jnp.stack([ltrcrn[k - 1] for k in tracer_indices["nlt_bgc_DON"][:n_don]])
             if flags["tr_bgc_DON"] and n_don else jnp.zeros((n_don,), dtype=dtype))
    Fedin = (jnp.stack([ltrcrn[k - 1] for k in tracer_indices["nlt_bgc_Fed"][:n_fed]])
             if flags["tr_bgc_Fe"] and n_fed else jnp.zeros((n_fed,), dtype=dtype))
    Fepin = (jnp.stack([ltrcrn[k - 1] for k in tracer_indices["nlt_bgc_Fep"][:n_fep]])
             if flags["tr_bgc_Fe"] and n_fep else jnp.zeros((n_fep,), dtype=dtype))
    DMSPdin = (ltrcrn[int(tracer_indices["nlt_bgc_DMSPd"]) - 1]
               if flags["tr_bgc_DMS"] else zero)
    DMSin = (ltrcrn[int(tracer_indices["nlt_bgc_DMS"]) - 1]
             if flags["tr_bgc_DMS"] else zero)
    Fed_tot = jnp.sum(Fedin)
    Fep_tot = jnp.sum(Fepin)
    rFed = jnp.where(Fed_tot > puny, Fedin / jnp.maximum(Fed_tot, puny), 0.0)
    rFep = jnp.where(Fep_tot > puny, Fepin / jnp.maximum(Fep_tot, puny), 0.0)

    # Light, nitrate, ammonium, silicate and iron limitation.
    Iavg_loc = jnp.where((op_dep > parameters["op_dep_min"]) & jnp.logical_not(dEdd_algae),
                         fswthru * (-jnp.expm1(-op_dep))
                         / jnp.where(op_dep > 0.0, op_dep, 1.0), fswthru)
    L_lim = ((1.0 - jnp.exp(-bio_parameters["alpha2max_low"][:n_algae] * Iavg_loc))
             * jnp.exp(-bio_parameters["beta2max"][:n_algae] * Iavg_loc))
    Nit_lim = Nitin / (Nitin + bio_parameters["K_Nit"][:n_algae])
    Am_lim = (Amin / (Amin + bio_parameters["K_Am"][:n_algae])
              if flags["tr_bgc_Am"] else jnp.zeros_like(Nit_lim))
    N_lim = jnp.minimum(1.0, Nit_lim + Am_lim) if flags["tr_bgc_Am"] else Nit_lim
    Sil_lim = jnp.ones_like(Nit_lim)
    if flags["tr_bgc_Sil"]:
        K_Sil = bio_parameters["K_Sil"][:n_algae]
        Sil_lim = jnp.where(K_Sil > 0.0,
                            Silin / jnp.where(K_Sil > 0.0, Silin + K_Sil, 1.0), 1.0)
    Fe_lim = jnp.ones_like(Nit_lim)
    if flags["tr_bgc_Fe"]:
        K_Fe = bio_parameters["K_Fe"][:n_algae]
        Fe_lim = jnp.where(K_Fe > 0.0,
                           Fed_tot / jnp.where(K_Fe > 0.0, Fed_tot + K_Fe, 1.0), 1.0)

    dTemp = jnp.minimum(T_bot - parameters["T_max"], 0.0)
    growmax_N = (bio_parameters["mu_max"][:n_algae] / secday
                 * jnp.exp(bio_parameters["grow_Tdep"][:n_algae] * dTemp)
                 * Nin * parameters["fsal"])
    grow_N = jnp.minimum(jnp.minimum(L_lim, N_lim), Sil_lim)
    if n_fed:
        grow_N = jnp.minimum(grow_N, Fe_lim)
    grow_N = grow_N * growmax_N
    U_Am = jnp.minimum(grow_N, Am_lim * growmax_N)
    U_Nit = grow_N - U_Am
    U_Sil = bio_parameters["R_Si2N"][:n_algae] * grow_N
    U_Fe = bio_parameters["R_Fe2N"][:n_algae] * grow_N

    U_Am_tot = jnp.sum(U_Am)
    U_Nit_tot = jnp.sum(U_Nit)
    U_Sil_tot = jnp.sum(U_Sil)
    U_Fe_tot = jnp.sum(U_Fe)
    U_Am_f = jnp.where(U_Am_tot > 0.0, U_Am / jnp.where(U_Am_tot > 0.0, U_Am_tot, 1.0), 0.0)
    U_Nit_f = jnp.where(U_Nit_tot > 0.0, U_Nit / jnp.where(U_Nit_tot > 0.0, U_Nit_tot, 1.0), 0.0)
    U_Sil_f = jnp.where(U_Sil_tot > 0.0, U_Sil / jnp.where(U_Sil_tot > 0.0, U_Sil_tot, 1.0), 0.0)
    U_Fe_f = jnp.where(U_Fe_tot > 0.0, U_Fe / jnp.where(U_Fe_tot > 0.0, U_Fe_tot, 1.0), 0.0)
    if flags["tr_bgc_Sil"]:
        U_Sil_tot = jnp.minimum(U_Sil_tot, parameters["max_loss"] * Silin / dt)
    if flags["tr_bgc_Fe"]:
        U_Fe_tot = jnp.minimum(U_Fe_tot, parameters["max_loss"] * Fed_tot / dt)
    U_Nit_tot = jnp.minimum(U_Nit_tot, parameters["max_loss"] * Nitin / dt)
    U_Am_tot = jnp.minimum(U_Am_tot, parameters["max_loss"] * Amin / dt)
    U_Am = U_Am_f * U_Am_tot
    U_Nit = U_Nit_f * U_Nit_tot
    U_Sil = U_Sil_f * U_Sil_tot
    U_Fe = U_Fe_f * U_Fe_tot
    R_Si2N = bio_parameters["R_Si2N"][:n_algae]
    R_Fe2N = bio_parameters["R_Fe2N"][:n_algae]
    grow_N = jnp.minimum(U_Nit + U_Am, U_Fe / R_Fe2N)
    grow_N = jnp.where(R_Si2N > 0.0,
                       jnp.minimum(grow_N, U_Sil / jnp.where(R_Si2N > 0.0, R_Si2N, 1.0)),
                       grow_N)
    fr_Am = jnp.zeros_like(grow_N)
    if flags["tr_bgc_Am"]:
        fr_Am = jnp.where(grow_N > 0.0,
                          jnp.minimum(U_Am / jnp.where(grow_N > 0.0, grow_N, 1.0), 1.0),
                          0.5)
    U_Nit = (1.0 - fr_Am) * grow_N
    U_Am = fr_Am * grow_N
    U_Sil = R_Si2N * grow_N
    U_Fe = R_Fe2N * grow_N

    # Algal growth, grazing, respiration and mortality.
    resp = parameters["fr_resp"] * grow_N
    graze = jnp.minimum(parameters["max_loss"] * Nin / dt,
                        grow_N * bio_parameters["fr_graze"][:n_algae]
                        * (Nin / bio_parameters["graze_conc"])
                        ** bio_parameters["graze_exponent"][:n_algae])
    mort = jnp.minimum(parameters["max_loss"] * Nin / dt,
                       bio_parameters["mort_pre"][:n_algae]
                       * jnp.exp(bio_parameters["mort_Tdep"][:n_algae] * dTemp)
                       * Nin / secday)
    N_s = grow_N * dt
    N_r = (graze + mort + resp) * dt
    R_C2N = bio_parameters["R_C2N"][:n_algae]
    graze_N = jnp.sum(graze)
    graze_C = jnp.sum(R_C2N * graze)
    mort_N = jnp.sum(mort)
    mort_C = jnp.sum(R_C2N * mort)
    resp_N = jnp.sum(resp)

    Am_s_e = graze_N * (1.0 - parameters["fr_graze_s"]) * parameters["fr_graze_e"] * dt
    Am_s_mo = mort_N * parameters["fr_mort2min"] * dt
    Am_s = resp_N * dt + Am_s_e + Am_s_mo
    Am_r = jnp.sum(U_Am) * dt
    Nit_r = jnp.sum(U_Nit) * dt
    Sil_r = jnp.sum(U_Sil) * dt
    Fed_tot_r = jnp.sum(U_Fe) * dt
    DIC_r = jnp.zeros((n_dic,), dtype=dtype)
    DIC_s = jnp.zeros((n_dic,), dtype=dtype)
    if n_dic:
        DIC_r = DIC_r.at[0].set(jnp.sum((1.0 - parameters["fr_resp"])
                                           * grow_N * R_C2N) * dt)

    # Nitrification turns ammonium into nitrate. Zoo tracks nitrogen leaving
    # the explicitly represented biological pools.
    nitrification = parameters["k_nitrif"] / secday * Amin * dt
    Am_r = Am_r + nitrification
    Nit_s = nitrification
    if flags["tr_bgc_Am"]:
        Zoo_s_a = graze_N * (1.0 - parameters["fr_graze_e"]) * (1.0 - parameters["fr_graze_s"]) * dt
        Zoo_s_m = mort_N * dt - Am_s_mo
    else:
        Zoo_s_a = graze_N * dt * (1.0 - parameters["fr_graze_s"])
        Zoo_s_m = mort_N * dt
    Zoo_s_s = graze_N * parameters["fr_graze_s"] * dt
    Zoo_s_b = zero
    DON_r = jnp.zeros((n_don,), dtype=dtype)
    DON_s = jnp.zeros((n_don,), dtype=dtype)
    if flags["tr_bgc_DON"]:
        for n in range(n_don):
            DON_r = DON_r.at[n].set(bio_parameters["kn_bac"][n] / secday * DONin[n] * dt)
            DON_s = DON_s.at[n].set(graze_N * dt - Am_s_e + mort_N * dt - Am_s_mo)
            Zoo_s_s = Zoo_s_s - DON_s[n]
            Zoo_s_b = Zoo_s_b + DON_r[n] * (1.0 - bio_parameters["f_don_Am"][n])
            Am_s = Am_s + DON_r[n] * bio_parameters["f_don_Am"][n]
            if n_dic:
                DIC_s = DIC_s.at[0].add(DON_r[n] * bio_parameters["R_C2N_DON"][n])
    Zoo = Zoo_s_a + Zoo_s_s + Zoo_s_m + Zoo_s_b

    DOC_r = jnp.zeros((n_doc,), dtype=dtype)
    DOC_s = jnp.zeros((n_doc,), dtype=dtype)
    for n in range(n_doc):
        DOC_r = DOC_r.at[n].set(bio_parameters["k_bac"][n] / secday * DOCin[n] * dt)
        DON_carbon = (DON_s[0] * bio_parameters["R_C2N_DON"][0] if n_don else zero)
        DOC_s = DOC_s.at[n].set(bio_parameters["f_doc"][n]
                                 * (graze_C * dt + mort_C * dt - DON_carbon))
        if n_dic:
            DIC_s = DIC_s.at[0].add(DOC_r[n])

    # Iron recycling and particulate-to-dissolved conversion.
    Fed_s = jnp.zeros((n_fed,), dtype=dtype)
    Fed_r = jnp.zeros((n_fed,), dtype=dtype)
    Fep_s = jnp.zeros((n_fep,), dtype=dtype)
    Fep_r = jnp.zeros((n_fep,), dtype=dtype)
    if n_fed:
        Fed_s = Fed_s.at[0].add(Am_s * R_Fe2N[0] * parameters["fr_dFe"])
    if flags["tr_bgc_C"] and flags["tr_bgc_Fe"] and n_doc:
        convert = (DOCin[0] > 0.0) & (Fed_tot / jnp.where(DOCin[0] > 0.0, DOCin[0], 1.0)
                                        < parameters["max_dfe_doc1"])
        Fep_r = jnp.where(convert, Fepin / parameters["t_iron_conv"] * dt / secday, Fep_r)
        Fed_s = Fed_s + rFed * jnp.sum(Fep_r)
    if flags["tr_bgc_Fe"]:
        Fed_r = Fed_r + rFed * Fed_tot_r
        Fep_s = Fep_s + rFep * (Am_s * R_Fe2N[0] * (1.0 - parameters["fr_dFe"]))

    # Dissolved DMSP from respiration and mortality, then conversion to DMS.
    DMSPd_s = jnp.sum((parameters["fr_resp_s"] * resp
                       + parameters["fr_mort2min"] * mort)
                      * bio_parameters["R_S2N"][:n_algae]) * dt
    DMSPd_r = DMSPdin * dt / (parameters["t_sk_conv"] * secday)
    DMS_s = parameters["y_sk_DMS"] * DMSPd_r
    DMS_r = DMSin * dt / (parameters["t_sk_ox"] * secday)

    # Write the reaction amounts back in the original Fortran tracer order.
    dN = zero
    dC = zero
    for k, one_based in enumerate(tracer_indices["nlt_bgc_N"][:n_algae]):
        value = N_s[k] - N_r[k]
        reactb = reactb.at[one_based - 1].set(value)
        dN = dN + value
        dC = dC + value * R_C2N[k]
    if flags["tr_bgc_C"]:
        for k, one_based in enumerate(tracer_indices["nlt_bgc_DOC"][:n_doc]):
            value = DOC_s[k] - DOC_r[k]
            reactb = reactb.at[one_based - 1].set(value)
            dC = dC + value
        for k, one_based in enumerate(tracer_indices["nlt_bgc_DIC"][:n_dic]):
            value = DIC_s[k] - DIC_r[k]
            reactb = reactb.at[one_based - 1].set(value)
            dC = dC + value
    reactb = reactb.at[int(tracer_indices["nlt_bgc_Nit"]) - 1].set(Nit_s - Nit_r)
    dN = dN + Nit_s - Nit_r
    if flags["tr_bgc_Am"]:
        reactb = reactb.at[int(tracer_indices["nlt_bgc_Am"]) - 1].set(Am_s - Am_r)
        dN = dN + Am_s - Am_r
    if flags["tr_bgc_Sil"]:
        reactb = reactb.at[int(tracer_indices["nlt_bgc_Sil"]) - 1].set(-Sil_r)
    if flags["tr_bgc_DON"]:
        for k, one_based in enumerate(tracer_indices["nlt_bgc_DON"][:n_don]):
            value = DON_s[k] - DON_r[k]
            reactb = reactb.at[one_based - 1].set(value)
            dN = dN + value
            dC = dC + value * bio_parameters["R_C2N_DON"][k]
    Cerror = dC
    if flags["tr_bgc_Fe"]:
        for k, one_based in enumerate(tracer_indices["nlt_bgc_Fed"][:n_fed]):
            reactb = reactb.at[one_based - 1].set(Fed_s[k] - Fed_r[k])
        for k, one_based in enumerate(tracer_indices["nlt_bgc_Fep"][:n_fep]):
            reactb = reactb.at[one_based - 1].set(Fep_s[k] - Fep_r[k])
    if flags["tr_bgc_DMS"]:
        reactb = reactb.at[int(tracer_indices["nlt_bgc_DMSPd"]) - 1].set(DMSPd_s - DMSPd_r)
        reactb = reactb.at[int(tracer_indices["nlt_bgc_DMS"]) - 1].set(DMS_s - DMS_r)
    tolerance = jnp.maximum(puny, jnp.max(jnp.abs(reactb)) * 1.0e-13)
    conserve_C = ((jnp.abs(dC) <= tolerance) & (jnp.abs(dN) <= tolerance)
                  if flags["tr_bgc_C"] else jnp.asarray(True))
    return reactb, grow_N, U_Nit, U_Am, Zoo, Cerror, conserve_C, nitrification


def skl_biogeochemistry(dt, flux_bio, ocean_bio, meltb, congel,
                        fswthru, first_ice, trcrn, Tf, flags, options,
                        tracer_indices, tracer_type, parameters,
                        bio_parameters):
    """Skeletal-layer reactions and ice-ocean exchange from Fortran."""
    del first_ice  # The Fortran argument is passed through but not used here.
    nbtrcr = int(flags["nbtrcr"])
    sk_l = parameters["sk_l"]
    phi_sk = bio_parameters["phi_sk"]
    puny = parameters["puny"]
    initial_flux_bio = flux_bio
    initial_trcrn = trcrn
    # bio_index and nt_bgc_N are zero based in CICE-JAX initialization;
    # nlt_bgc_* remain one based biological-vector positions.
    bio_index = jnp.asarray(tracer_indices["bio_index"][:nbtrcr])
    cinit = trcrn[bio_index] * sk_l / phi_sk
    cinit_v = cinit / sk_l
    invalid_initial = jnp.any(cinit < 0.0)
    stationary = tracer_type[:nbtrcr] >= 0.0
    PVflag = jnp.where(stationary, 0.0, 1.0)
    cling = jnp.where(stationary, 1.0, 0.0)
    ice_growth = (congel - meltb) / dt

    # The two Fortran flux schemes set the interface piston velocity and
    # seed stationary tracers differently during growth and melting.
    if options["bgc_flux_type"].strip() == "Jin2006":
        grow_val = jnp.minimum(ice_growth, 1.85e-5)
        PV_growth = -jnp.minimum(
            jnp.abs(0.5 * (9.667e-9 + 38.8 * grow_val
                            - 1.04e7 * grow_val ** 2)),
            0.9 * sk_l / dt)
        PV_melt = jnp.minimum(
            jnp.abs(0.05 * (38.8 * ice_growth - 1.04e7 * ice_growth ** 2)),
            0.9 * sk_l / dt)
        PVt = jnp.where(ice_growth > 0.0, PV_growth, PV_melt)
        f_meltn = jnp.where(stationary & (ice_growth < 0.0),
                            PVt * cinit_v, 0.0)
        congel_alg = jnp.where(
            stationary & (ice_growth > 0.0)
            & (cinit < ocean_bio[:nbtrcr] * sk_l / phi_sk),
            (ocean_bio[:nbtrcr] * sk_l / phi_sk - cinit) / dt, 0.0)
    else:
        PVt = jnp.where(ice_growth > 0.0, -1.0e-6, 0.0)
        congel_alg = jnp.where(
            stationary & (ice_growth >= 0.0)
            & (cinit_v < ocean_bio[:nbtrcr] / phi_sk),
            (ocean_bio[:nbtrcr] * sk_l / phi_sk - cinit) / dt, 0.0)
        f_meltn = jnp.where(stationary & (ice_growth < 0.0),
                            jnp.minimum(1.0, meltb / sk_l) * cinit / dt, 0.0)

    react = jnp.zeros((nbtrcr,), dtype=trcrn.dtype)
    (react, grow_alg_skl, upNOn, upNHn, Zoo_skl, Cerror,
     conserve_C, nitrification) = algal_dyn(
        dt, options["dEdd_algae"], fswthru, react,
        jnp.where(invalid_initial, 0.0, cinit_v), Tf,
        flags, tracer_indices, parameters, bio_parameters)

    # SIGN(PVflag, PVt) gives mobile tracers a signed transport coefficient.
    PVflag = jnp.copysign(PVflag, PVt)
    cinit_tmp = jnp.maximum(0.0, cinit_v + react)
    flux_bio_temp = (PVflag * PVt * cinit_tmp
                     - PVflag * jnp.minimum(0.0, PVt) * ocean_bio[:nbtrcr]
                     + f_meltn * cling - congel_alg)
    flux_bio_temp = jnp.where(cinit_tmp * sk_l < flux_bio_temp * dt,
                              cinit_tmp * sk_l / dt * (1.0 - puny),
                              flux_bio_temp)
    cinit = cinit_tmp * sk_l - flux_bio_temp * dt
    flux_bio = flux_bio.at[:nbtrcr].add(flux_bio_temp * phi_sk)
    trcrn = trcrn.at[bio_index].set(cinit * phi_sk / sk_l)
    invalid_final = jnp.any(cinit < 0.0)
    # Fortran returns before applying any category change for a negative
    # initial skeletal inventory; retain a flag for the caller's diagnostics.
    flux_bio = jnp.where(invalid_initial, initial_flux_bio, flux_bio)
    trcrn = jnp.where(invalid_initial, initial_trcrn, trcrn)
    grow_alg_skl = jnp.where(invalid_initial, 0.0, grow_alg_skl)
    upNOn = jnp.where(invalid_initial, 0.0, upNOn)
    upNHn = jnp.where(invalid_initial, 0.0, upNHn)
    return (flux_bio, trcrn, grow_alg_skl, upNOn, upNHn,
            Zoo_skl, Cerror, conserve_C, nitrification,
            invalid_initial | invalid_final)


def sklbio(dt, Tf, flux_bio, ocean_bio, aicen, meltb, congel,
           fswthru, first_ice, trcrn, PP_net, upNO, upNH,
           grow_net, flags, options, tracer_indices, tracer_type,
           parameters, bio_parameters):
    """Run skeletal biology and add category fluxes to cell diagnostics."""
    flux_bion = jnp.zeros_like(ocean_bio)
    (flux_bion, trcrn, grow_alg, upNOn, upNHn,
     Zoo_skl, Cerror, conserve_C, nitrification,
     invalid) = skl_biogeochemistry(
        dt, flux_bion, ocean_bio, meltb, congel, fswthru,
        first_ice, trcrn, Tf, flags, options, tracer_indices,
        tracer_type, parameters, bio_parameters)
    nt_bgc_N = tracer_indices["nt_bgc_N"][:int(flags["n_algae"])]
    flux_bio, PP_net, upNO, upNH, grow_net = merge_bgc_fluxes_skl(
        aicen, trcrn, flux_bion, flux_bio, PP_net, upNOn,
        upNHn, upNO, upNH, grow_net, grow_alg, nt_bgc_N,
        bio_parameters["R_C2N"], bio_parameters["phi_sk"],
        parameters["sk_l"], parameters["secday"],
        bio_parameters["R_gC2molC"], parameters["fr_resp"],
        parameters["puny"])
    return (trcrn, flux_bio, PP_net, upNO, upNH, grow_net,
            Zoo_skl, Cerror, conserve_C, nitrification, invalid)


def thin_ice_flux(hin, hin_old, Cin, source, dt, ocean_bio, puny):
    """Transfer tracer mass when the brine layer is too thin for transport."""
    nblyr = Cin.shape[0] - 1
    dh = hin - hin_old

    def shrinking(_):
        zspace = jnp.full((nblyr + 1,), 1.0 / nblyr, dtype=Cin.dtype)
        zspace = zspace.at[0].set(0.5 / nblyr)
        zspace = zspace.at[-1].set(0.5 / nblyr)
        sum_bio = jnp.sum(Cin / hin_old * zspace)
        updated = Cin / hin_old * hin
        flux_o_tot = -dh * sum_bio / dt + source / dt
        return updated, flux_o_tot

    def growing(_):
        dC = dh * ocean_bio
        updated = Cin + dC
        flux_o_tot = -dC / dt + source / dt
        return updated, flux_o_tot

    return jax.lax.cond((dh <= 0.0) & (hin_old > puny),
                        shrinking, growing, None)


def bgc_column_sum(hsnow, hbrine, xin, nslyr):
    """Integrate one tracer over the ice brine and two snow layers."""
    nblyr = xin.shape[0] - 3
    dzssl = 0.5 * hsnow / nslyr
    dzint = jnp.maximum(0.0, hsnow - dzssl)
    ice_total = (xin[0] + xin[nblyr]) * hbrine * 0.5 / nblyr
    ice_total = ice_total + jnp.sum(xin[1:nblyr]) * hbrine / nblyr
    return ice_total + dzssl * xin[nblyr + 1] + dzint * xin[nblyr + 2]


def bgc_carbon_sum(hbrine, xin, nblyr, nt_bgc_N, nt_bgc_DOC,
                   nt_bgc_DIC, nt_bgc_DON, nt_bgc_hum,
                   R_C2N, R_C2N_DON, flags):
    """Integrate carbon in the biological tracers over brine height."""
    zspace = jnp.full((nblyr + 1,), 1.0 / nblyr, dtype=xin.dtype)
    zspace = zspace.at[0].set(0.5 / nblyr)
    zspace = zspace.at[-1].set(0.5 / nblyr)
    total = jnp.asarray(0.0, dtype=xin.dtype)
    if flags["tr_bgc_N"]:
        for m in range(int(flags["n_algae"])):
            slot = nt_bgc_N[m]
            total = total + jnp.sum(
                xin[slot:slot + nblyr + 1] * zspace) * hbrine * R_C2N[m]
    if flags["tr_bgc_C"]:
        for slots in (nt_bgc_DOC[:int(flags["n_doc"])],
                      nt_bgc_DIC[:int(flags["n_dic"])]):
            for slot in slots:
                total = total + jnp.sum(
                    xin[slot:slot + nblyr + 1] * zspace) * hbrine
    if flags["tr_bgc_DON"]:
        for m in range(int(flags["n_don"])):
            slot = nt_bgc_DON[m]
            total = total + jnp.sum(
                xin[slot:slot + nblyr + 1] * zspace) * hbrine * R_C2N_DON[m]
    if flags["tr_bgc_hum"]:
        total = total + jnp.sum(
            xin[nt_bgc_hum:nt_bgc_hum + nblyr + 1] * zspace) * hbrine
    return total


def bgc_carbon_flux(flux_bio_atm, flux_bion, nlt_bgc_N,
                    nlt_bgc_DOC, nlt_bgc_DIC, nlt_bgc_DON,
                    nlt_bgc_hum, R_C2N, R_C2N_DON, flags):
    """Sum the atmospheric and ocean carbon fluxes for one category."""
    total = jnp.asarray(0.0, dtype=flux_bion.dtype)
    if flags["tr_bgc_N"]:
        for m in range(int(flags["n_algae"])):
            slot = nlt_bgc_N[m] - 1
            total = total - (flux_bio_atm[slot] - flux_bion[slot]) * R_C2N[m]
    if flags["tr_bgc_C"]:
        for slots in (nlt_bgc_DOC[:int(flags["n_doc"])],
                      nlt_bgc_DIC[:int(flags["n_dic"])]):
            for one_based in slots:
                slot = one_based - 1
                total = total - flux_bio_atm[slot] + flux_bion[slot]
    if flags["tr_bgc_DON"]:
        for m in range(int(flags["n_don"])):
            slot = nlt_bgc_DON[m] - 1
            total = total - (flux_bio_atm[slot] - flux_bion[slot]) * R_C2N_DON[m]
    if flags["tr_bgc_hum"]:
        slot = nlt_bgc_hum - 1
        total = total - flux_bio_atm[slot] + flux_bion[slot]
    return total


def tridiag_solverz(sbdiag, diag, spdiag, rhs):
    """Solve the tridiagonal vertical BGC system in Fortran row order."""
    nmat = diag.shape[0]
    xout = jnp.zeros_like(rhs)
    wgamma = jnp.zeros_like(rhs)
    wbeta = diag[0]
    xout = xout.at[0].set(rhs[0] / wbeta)
    for k in range(1, nmat):
        gamma = spdiag[k - 1] / wbeta
        wgamma = wgamma.at[k].set(gamma)
        wbeta = diag[k] - sbdiag[k] * gamma
        xout = xout.at[k].set((rhs[k] - sbdiag[k] * xout[k - 1]) / wbeta)
    for k in range(nmat - 2, -1, -1):
        xout = xout.at[k].set(xout[k] - wgamma[k + 1] * xout[k + 1])
    return xout


def check_conservation_FCT(C_init, C_new, S_top, S_bot,
                           L_bot, L_top, dt, fluxbio, source):
    """Account for the low order vertical solution and its ocean flux."""
    nblyr = C_init.shape[0] - 1
    zspace = jnp.full((nblyr + 1,), 1.0 / nblyr, dtype=C_init.dtype)
    zspace = zspace.at[0].set(0.5 / nblyr)
    zspace = zspace.at[-1].set(0.5 / nblyr)
    C_init_tot = jnp.sum(C_init * zspace)
    C_new_tot = jnp.sum(C_new * zspace)
    C_low = C_new
    accuracyC = 1.0e-11 * jnp.maximum(
        1.0, jnp.maximum(C_init_tot, C_new_tot))
    fluxbio = fluxbio + (C_init_tot - C_new_tot + source) / dt
    diff_dt = (C_new_tot - C_init_tot
               - (S_top + S_bot + L_bot * C_new[-1]
                  + L_top * C_new[0]) * dt)
    negative = jnp.min(C_low) < 0.0
    conservation_warning = jnp.abs(diff_dt) > accuracyC
    return C_low, fluxbio, negative, conservation_warning


def compute_FCT_matrix(C_in, dt, darcyV, dhtop, dhbot,
                       iphin_N, iDin, hbri_old, atm_add,
                       bphin_N, C_top, C_bot, bgrid, grid_o):
    """Build the low order advection and diffusion matrix for brine tracers."""
    nblyr = C_in.shape[0] - 1
    zspace = 1.0 / nblyr
    ML = jnp.full_like(C_in, zspace)
    ML = ML.at[0].set(0.5 * zspace)
    ML = ML.at[-1].set(0.5 * zspace)
    K_diag = jnp.zeros_like(C_in)
    K_spdiag = jnp.zeros_like(C_in)
    K_sbdiag = jnp.zeros_like(C_in)
    S_diag = jnp.zeros_like(C_in)
    S_spdiag = jnp.zeros_like(C_in)
    S_sbdiag = jnp.zeros_like(C_in)
    D_spdiag = jnp.zeros_like(C_in)
    D_sbdiag = jnp.zeros_like(C_in)
    iDin_phi = jnp.zeros_like(C_in)
    iDin_phi = iDin_phi.at[-1].set(iDin[-1] / iphin_N[-1])
    iDin_phi = iDin_phi.at[-2].set(
        0.5 * (iDin[-1] / iphin_N[-1] + iDin[-2] / iphin_N[-2]))

    vel = (bgrid[1] * dhbot - (bgrid[1] - 1.0) * dhtop) / dt
    vel = vel + darcyV / bphin_N[1]
    K_diag = K_diag.at[0].set(0.5 * vel / hbri_old)
    dphi_dx = (iphin_N[-1] - iphin_N[-2]) / zspace
    vel = (bgrid[nblyr] * dhbot
           - (bgrid[nblyr] - 1.0) * dhtop) / dt
    vel = (vel + darcyV / bphin_N[nblyr]) / hbri_old
    vel2 = dhbot / hbri_old / dt + darcyV / hbri_old
    K_diag = K_diag.at[-1].set(
        jnp.minimum(0.0, vel2)
        - iDin_phi[-1] / (zspace + grid_o / hbri_old)
        + 0.5 * (-vel + iDin_phi[-2] / bphin_N[nblyr] * dphi_dx))

    for k in range(nblyr - 1):
        vel = (bgrid[k + 1] * dhbot
               - (bgrid[k + 1] - 1.0) * dhtop) / dt
        vel = vel + darcyV / bphin_N[k + 1]
        iDin_phi = iDin_phi.at[k + 1].set(
            0.5 * (iDin[k + 2] / iphin_N[k + 2]
                   + iDin[k + 1] / iphin_N[k + 1]))
        dphi_dx = (iphin_N[k + 1] - iphin_N[k]) / zspace
        K_spdiag = K_spdiag.at[k].set(
            0.5 * (vel / hbri_old
                   - iDin_phi[k] / bphin_N[k + 1] * dphi_dx))
        dphi_dx = (0.0 if k == 0 else
                    (iphin_N[k + 1] - iphin_N[k]) / zspace)
        K_sbdiag = K_sbdiag.at[k + 1].set(
            -0.5 * (vel / hbri_old
                    - iDin_phi[k] / bphin_N[k + 1] * dphi_dx))
        if k == 0:
            K_diag = K_diag.at[k].set(K_diag[0])
        else:
            K_diag = K_diag.at[k].set(K_spdiag[k] + K_sbdiag[k])
        S_diag = S_diag.at[k + 1].set(
            -(iDin_phi[k] + iDin_phi[k + 1]) / zspace)
        S_spdiag = S_spdiag.at[k].set(iDin_phi[k] / zspace)
        S_sbdiag = S_sbdiag.at[k + 1].set(iDin_phi[k] / zspace)

    k = nblyr - 1
    vel = (bgrid[nblyr] * dhbot
           - (bgrid[nblyr] - 1.0) * dhtop) / dt
    vel = vel + darcyV / bphin_N[nblyr]
    dphi_dx = (iphin_N[nblyr] - iphin_N[nblyr - 1]) / zspace
    K_spdiag = K_spdiag.at[k].set(
        0.5 * (vel / hbri_old
               - iDin_phi[k] / bphin_N[nblyr] * dphi_dx))
    K_sbdiag = K_sbdiag.at[nblyr].set(
        -0.5 * (vel / hbri_old
                - iDin_phi[k] / bphin_N[nblyr] * dphi_dx))
    K_diag = K_diag.at[k].set(K_spdiag[k] + K_sbdiag[k])
    S_diag = S_diag.at[nblyr].set(-iDin_phi[k] / zspace)
    S_spdiag = S_spdiag.at[k].set(iDin_phi[k] / zspace)
    S_sbdiag = S_sbdiag.at[nblyr].set(iDin_phi[k] / zspace)

    for k in range(nblyr):
        diffusion = jnp.maximum(
            -K_spdiag[k], jnp.maximum(0.0, -K_sbdiag[k + 1]))
        D_spdiag = D_spdiag.at[k].set(diffusion)
        D_sbdiag = D_sbdiag.at[k + 1].set(diffusion)
    D_diag = -D_spdiag - D_sbdiag

    vel2 = -(dhtop / hbri_old / dt
             + darcyV / bphin_N[0] / hbri_old)
    Qtop = jnp.maximum(0.0, vel2 * C_top + atm_add / dt)
    vel = dhbot / hbri_old / dt + darcyV / hbri_old
    Qbot = (jnp.maximum(0.0, vel * C_bot)
            + iDin_phi[-1] * C_bot / (zspace + grid_o / hbri_old))
    Sink_bot = K_diag[-1] + K_spdiag[-2]
    Sink_top = K_diag[0] + K_sbdiag[1]

    spdiag = -dt * (D_spdiag + K_spdiag + S_spdiag)
    sbdiag = -dt * (D_sbdiag + K_sbdiag + S_sbdiag)
    diag = ML - dt * (D_diag + K_diag + S_diag)
    rhs = ML * C_in
    rhs = rhs.at[0].add(dt * Qtop)
    rhs = rhs.at[-1].add(dt * Qbot)
    return (sbdiag, diag, spdiag, rhs, Qbot, Qtop,
            Sink_bot, Sink_top, D_sbdiag, D_spdiag, ML)


def compute_FCT_corr(C_in, C_low, dt, D_sbdiag, D_spdiag, ML, puny):
    """Apply the flux-corrected transport limiter to the low order profile."""
    nblyr = C_in.shape[0] - 1
    zspace = 1.0 / nblyr
    M_spdiag = jnp.full_like(C_in, zspace / 6.0).at[-1].set(0.0)
    M_sbdiag = jnp.full_like(C_in, zspace / 6.0).at[0].set(0.0)
    F_spdiag = jnp.zeros_like(C_in)
    F_sbdiag = jnp.zeros_like(C_in)
    for k in range(nblyr):
        fsp = (M_spdiag[k] * (C_low[k] - C_in[k]
                             - C_low[k + 1] + C_in[k + 1]) / dt
               + D_spdiag[k] * (C_low[k] - C_low[k + 1]))
        fsb = (M_sbdiag[k + 1] * (C_low[k + 1] - C_in[k + 1]
                                 - C_low[k] + C_in[k]) / dt
               + D_sbdiag[k + 1] * (C_low[k + 1] - C_low[k]))
        F_spdiag = F_spdiag.at[k].set(jnp.where(
            fsp * (C_low[k] - C_low[k + 1]) > 0.0, 0.0, fsp))
        F_sbdiag = F_sbdiag.at[k + 1].set(jnp.where(
            fsb * (C_low[k + 1] - C_low[k]) > 0.0, 0.0, fsb))

    Pplus = jnp.zeros_like(C_in)
    Pminus = jnp.zeros_like(C_in)
    Qplus = jnp.zeros_like(C_in)
    Qminus = jnp.zeros_like(C_in)
    Pplus = Pplus.at[0].set(jnp.maximum(0.0, F_spdiag[0]))
    Pminus = Pminus.at[0].set(jnp.minimum(0.0, F_spdiag[0]))
    Pplus = Pplus.at[-1].set(jnp.maximum(0.0, F_sbdiag[-1]))
    Pminus = Pminus.at[-1].set(jnp.minimum(0.0, F_sbdiag[-1]))
    Qplus = Qplus.at[0].set(jnp.maximum(0.0, C_low[1] - C_low[0]))
    Qminus = Qminus.at[0].set(jnp.minimum(0.0, C_low[1] - C_low[0]))
    Qplus = Qplus.at[-1].set(jnp.maximum(0.0, C_low[-2] - C_low[-1]))
    Qminus = Qminus.at[-1].set(jnp.minimum(0.0, C_low[-2] - C_low[-1]))
    for k in range(1, nblyr):
        Pplus = Pplus.at[k].set(
            jnp.maximum(0.0, F_spdiag[k])
            + jnp.maximum(0.0, F_sbdiag[k]))
        Pminus = Pminus.at[k].set(
            jnp.minimum(0.0, F_spdiag[k])
            + jnp.minimum(0.0, F_sbdiag[k]))
        Qplus = Qplus.at[k].set(jnp.maximum(
            0.0, jnp.maximum(C_low[k + 1] - C_low[k],
                             C_low[k - 1] - C_low[k])))
        Qminus = Qminus.at[k].set(jnp.minimum(
            0.0, jnp.minimum(C_low[k + 1] - C_low[k],
                             C_low[k - 1] - C_low[k])))
    Rplus = jnp.minimum(1.0, ML * Qplus / dt / (Pplus + puny))
    Rminus = jnp.minimum(1.0, ML * Qminus / dt / (Pminus - puny))
    a_spdiag = jnp.zeros_like(C_in)
    a_sbdiag = jnp.zeros_like(C_in)
    for k in range(nblyr):
        a_spdiag = a_spdiag.at[k].set(jnp.where(
            F_spdiag[k] > 0.0,
            jnp.minimum(Rplus[k], Rminus[k + 1]),
            jnp.minimum(Rminus[k], Rplus[k + 1])))
        a_sbdiag = a_sbdiag.at[k + 1].set(jnp.where(
            F_sbdiag[k + 1] > 0.0,
            jnp.minimum(Rplus[k + 1], Rminus[k]),
            jnp.minimum(Rminus[k + 1], Rplus[k])))
    F_diag = a_spdiag * F_spdiag + a_sbdiag * F_sbdiag
    corrected = C_low + dt * F_diag / ML
    return jnp.where(jnp.max(jnp.abs(F_spdiag)) > 0.0,
                     corrected, C_low)


def z_biogeochemistry(dt, fields, flags, options, parameters,
                      bio_parameters, tracer_type, tau_ret, tau_rel,
                      zbgc_frac_init):
    """Advance one vertical biological column as in z_biogeochemistry."""
    trcrn = fields["trcrn"]
    flux_bio = fields["flux_bio"]
    ocean_bio = fields["ocean_bio"]
    nbtrcr = int(flags["nbtrcr"])
    nblyr = int(options["nblyr"])
    n_algae = int(flags["n_algae"])
    puny = parameters["puny"]
    accuracy = 1.0e-13
    dz = jnp.full((nblyr + 1,), 1.0 / nblyr, dtype=trcrn.dtype)
    dz = dz.at[0].set(0.5 / nblyr).at[-1].set(0.5 / nblyr)
    bio_index = flags["bio_index"][:nbtrcr]  # zero based in CICE-JAX
    frac_start = int(flags["nt_zbgc_frac"])
    hbri = fields["hbri"]
    hbri_old = fields["hbri_old"]
    iphin_N = fields["iphin"]
    bphin_N = fields["bphin"].at[0].set(fields["bphi_min"])
    bphin_N = bphin_N.at[-1].set(1.0)
    in_init_cons = jnp.zeros((nblyr + 1, nbtrcr), dtype=trcrn.dtype)
    flux_bio_tmp = jnp.zeros((nbtrcr,), dtype=trcrn.dtype)
    failed = jnp.asarray(False)
    for m in range(nbtrcr):
        for k in range(nblyr + 1):
            slot = int(bio_index[m]) + k
            value = trcrn[slot]
            tiny = jnp.abs(value) < accuracy
            flux_bio_tmp = flux_bio_tmp.at[m].set(jnp.where(
                tiny, value * hbri_old * dz[k] / dt, flux_bio_tmp[m]))
            trcrn = trcrn.at[slot].set(jnp.where(tiny, 0.0, value))
            in_init_cons = in_init_cons.at[k, m].set(
                jnp.where(tiny, 0.0, value * hbri_old))
            failed = failed | (trcrn[slot] < 0.0)

    hin = fields["vicen"] / fields["aicen"]
    hin_old = fields["hice_old"]
    phi_max = jnp.max(bphin_N[1:nblyr + 1])
    S_col = 4.0 * jnp.pi * 3.0e3 ** 2
    P_b = jnp.pi * 4.7 ** 2
    V_c = 4.0 * jnp.pi * 3.0e3 ** 3 / 3.0
    V_alg = jnp.pi / 6.0 * 4.7 * 10.0 ** 2
    Sat_conc = (0.1 * 0.3 * 0.7854 * (1.0 - phi_max) / V_c
                * S_col / P_b * V_alg ** 0.88 * 0.0408 * 1.0e9)

    dust_Fe = jnp.zeros((nbtrcr,), dtype=trcrn.dtype)
    if (flags["tr_zaero"] and int(flags["n_zaero"]) > 2
            and flags["tr_bgc_Fe"] and options["use_atm_dust_iron"]):
        for m in range(2, int(flags["n_zaero"])):
            aerosol = flags["nlt_zaero"][m] - 1
            fed = flags["nlt_bgc_Fed"][0] - 1
            dust_Fe = dust_Fe.at[fed].add(
                (fields["zbgc_snow"][aerosol] + fields["zbgc_atm"][aerosol])
                * parameters["R_dFe2dust"] * parameters["dustFe_sol"])

    atm_add_cons = jnp.zeros((nbtrcr,), dtype=trcrn.dtype)
    C_top = jnp.zeros((nbtrcr,), dtype=trcrn.dtype)
    C_bot = jnp.zeros((nbtrcr,), dtype=trcrn.dtype)
    ocean_bio_dh = jnp.zeros((nbtrcr,), dtype=trcrn.dtype)
    exp_ret = jnp.zeros((nbtrcr,), dtype=trcrn.dtype)
    exp_rel = jnp.zeros((nbtrcr,), dtype=trcrn.dtype)
    source = jnp.zeros((nbtrcr,), dtype=trcrn.dtype)
    top_motion = (fields["dh_top"]
                  + fields["darcy_V"] / bphin_N[0] * dt)
    dhflood = jnp.maximum(0.0, -fields["dh_direct"])
    for m in range(nbtrcr):
        ret = jnp.where(tau_ret[m] > 0.0,
                         jnp.exp(-jnp.minimum(
                             dt / jnp.where(tau_ret[m] > 0.0, tau_ret[m], 1.0),
                             10.0)), 0.0)
        rel = jnp.where(tau_rel[m] > 0.0,
                         jnp.exp(-jnp.minimum(
                             dt / jnp.where(tau_rel[m] > 0.0, tau_rel[m], 1.0),
                             10.0)), 0.0)
        if m + 1 != flags["nlt_bgc_N"][0]:
            ret = jnp.where(hin_old > hin, 1.0, ret)
            rel = jnp.where(hin_old <= hin, 1.0, rel)
        elif flags["tr_bgc_N"]:
            ret = jnp.where(hin_old > hin + parameters["algal_vel"] * dt,
                             1.0, ret)
            rel = jnp.where(hin_old <= hin + parameters["algal_vel"] * dt,
                             1.0, rel)
        exp_ret = exp_ret.at[m].set(ret)
        exp_rel = exp_rel.at[m].set(rel)
        top = in_init_cons[0, m] * trcrn[frac_start + m]
        addition = fields["zbgc_snow"][m] + fields["zbgc_atm"][m] + dust_Fe[m]
        source = source.at[m].set(jnp.abs(addition))
        top = jnp.where(top_motion < -puny,
                         addition / jnp.abs(top_motion + puny) * hbri_old, top)
        C_top = C_top.at[m].set(top)
        atm_add_cons = atm_add_cons.at[m].set(jnp.where(
            (top_motion >= -puny)
            & (jnp.abs(addition + ocean_bio[m] * bphin_N[0] * dhflood) > puny),
            jnp.abs(addition) + ocean_bio[m] * bphin_N[0] * dhflood,
            jnp.where(top_motion >= -puny, jnp.abs(addition), 0.0)))
        C_bot = C_bot.at[m].set(ocean_bio[m] * hbri_old * iphin_N[-1])
        ocean_bio_dh = ocean_bio_dh.at[m].set(C_bot[m])

    zfswin = remap_zbgc_jax(
        fields["fswthrul"], fields["icgrid"], fields["igrid"],
        hin, hbri, 0.0)
    biomat_cons = in_init_cons
    mobile = tracer_type[:nbtrcr] >= 0.0
    for m in range(nbtrcr):
        def thick_column(_):
            fraction = trcrn[frac_start + m]
            init_mobile = in_init_cons[:, m] * fraction
            stationary = jnp.maximum(0.0, in_init_cons[:, m] - init_mobile)
            dmobile = (jnp.where(mobile[m],
                                 init_mobile * (exp_ret[m] - 1.0), 0.0)
                       + stationary * (1.0 - exp_rel[m]))
            init_mobile = jnp.maximum(0.0, init_mobile + dmobile)
            stationary = jnp.maximum(0.0, stationary - dmobile)
            excess = jnp.maximum(0.0, stationary - Sat_conc * hbri_old)
            init_mobile = init_mobile + excess
            stationary = stationary - excess
            (sbdiag, diag, spdiag, rhs, S_bot, S_top,
             L_bot, L_top, D_sbdiag, D_spdiag, ML) = compute_FCT_matrix(
                init_mobile, dt, fields["darcy_V"], fields["dh_top"],
                fields["dh_bot"], iphin_N, fields["iDin"], hbri_old,
                atm_add_cons[m], bphin_N, C_top[m], C_bot[m],
                fields["bgrid"], parameters["grid_o"])
            biocons = tridiag_solverz(sbdiag, diag, spdiag, rhs)
            biomat_low, new_flux, negative, warning = check_conservation_FCT(
                init_mobile, biocons, S_top, S_bot,
                L_bot, L_top, dt, flux_bio[m], source[m])
            biocons = compute_FCT_corr(init_mobile, biocons, dt,
                                        D_sbdiag, D_spdiag, ML, puny)
            stationary_move = (m + 1 != flags["nlt_bgc_N"][0])
            if not stationary_move and flags["tr_bgc_N"]:
                stationary_move = ((fields["meltb"] > parameters["algal_vel"] * dt)
                                   | (fields["aicen"] < 0.001))
            if isinstance(stationary_move, bool):
                if stationary_move:
                    stationary, new_flux = regrid_stationary(
                        stationary, hbri_old, hbri, dt, 0.0,
                        fields["igrid"], new_flux, puny,
                        fields["meltb"], fields["congel"])
            else:
                stationary, new_flux = jax.lax.cond(
                    stationary_move,
                    lambda _: regrid_stationary(
                        stationary, hbri_old, hbri, dt, 0.0,
                        fields["igrid"], new_flux, puny,
                        fields["meltb"], fields["congel"]),
                    lambda _: (stationary, new_flux), None)
            combined = biocons + stationary
            sum_initial = jnp.sum(in_init_cons[:, m] * dz)
            sum_total = jnp.sum(combined * dz)
            sum_new = jnp.sum(biocons * dz)
            fraction = jnp.where(sum_total > 0.0, sum_new / jnp.where(
                sum_total > 0.0, sum_total, 1.0), zbgc_frac_init[m])
            failed_FCT = ((jnp.abs((sum_initial - sum_total + source[m]) / dt
                                     - new_flux)
                           > jnp.maximum(puny, accuracy * jnp.abs(new_flux)))
                          | (jnp.min(biocons) < 0.0)
                          | (jnp.min(stationary) < 0.0)
                          | negative | warning)
            return combined, new_flux, fraction, failed_FCT

        def thin_column(_):
            combined, new_flux = thin_ice_flux(
                hbri, hbri_old, in_init_cons[:, m], source[m], dt,
                ocean_bio[m], puny)
            return combined, new_flux, trcrn[frac_start + m], jnp.asarray(False)

        combined, new_flux, fraction, failed_FCT = jax.lax.cond(
            (hbri_old > 0.05) & (hbri > 0.05),
            thick_column, thin_column, None)
        biomat_cons = biomat_cons.at[:, m].set(combined)
        flux_bio = flux_bio.at[m].set(new_flux)
        trcrn = trcrn.at[frac_start + m].set(fraction)
        failed = failed | failed_FCT

    biomat_brine = biomat_cons / jnp.where(hbri > 0.0, hbri, 1.0)
    biomat_brine = biomat_brine / jnp.where(
        iphin_N[:, None] > 0.0, iphin_N[:, None], 1.0)
    react = jnp.zeros_like(biomat_brine)
    grow_alg = jnp.zeros((nblyr + 1, n_algae), dtype=trcrn.dtype)
    upNOn = jnp.zeros_like(grow_alg)
    upNHn = jnp.zeros_like(grow_alg)
    Zoo = fields["Zoo"]
    Cerror = jnp.zeros((nblyr + 1,), dtype=trcrn.dtype)
    conserve_C = jnp.ones((nblyr + 1,), dtype=jnp.bool_)
    nitrification = jnp.zeros((nblyr + 1,), dtype=trcrn.dtype)
    if options["solve_zbgc"] and n_algae:
        for k in range(nblyr + 1):
            (reaction, growth, nitrate, ammonium, zoo,
             carbon_error, conserved, nitrified) = algal_dyn(
                 dt, options["dEdd_algae"], zfswin[k], react[k],
                 biomat_brine[k], fields["iTin"][k], flags, flags,
                 parameters, bio_parameters)
            react = react.at[k].set(reaction)
            grow_alg = grow_alg.at[k].set(growth)
            upNOn = upNOn.at[k].set(nitrate)
            upNHn = upNHn.at[k].set(ammonium)
            Zoo = Zoo.at[k].set(zoo)
            Cerror = Cerror.at[k].set(carbon_error)
            conserve_C = conserve_C.at[k].set(conserved)
            nitrification = nitrification.at[k].set(nitrified)

    nitrate_mobile = jnp.asarray(0.0, dtype=trcrn.dtype)
    nitrate_total = jnp.asarray(0.0, dtype=trcrn.dtype)
    for m in range(nbtrcr):
        for k in range(nblyr + 1):
            bulk = (biomat_brine[k, m] + react[k, m]) * iphin_N[k]
            if flags["tr_bgc_C"] and m + 1 == flags["nlt_bgc_DIC"][0]:
                dic_demand = bulk <= -accuracy
                flux_bio = flux_bio.at[m].add(jnp.where(
                    dic_demand, bulk * dz[k] * hbri / dt, 0.0))
                bulk = jnp.where(dic_demand, 0.0, bulk)
            if m + 1 == flags["nlt_bgc_Nit"]:
                fraction = trcrn[frac_start + m]
                nit_after = biomat_brine[k, m] - nitrification[k] + react[k, m]
                mobile_nit = jnp.maximum(0.0, nit_after * iphin_N[k] * fraction)
                stationary_nit = jnp.maximum(
                    0.0, ((1.0 - fraction) * nit_after + nitrification[k])
                    * iphin_N[k])
                nitrate_mobile = nitrate_mobile + mobile_nit * dz[k]
                nitrate_total = nitrate_total + (mobile_nit + stationary_nit) * dz[k]
            tiny = jnp.abs(bulk) < accuracy
            flux_bio = flux_bio.at[m].add(jnp.where(
                conserve_C[k] & tiny, bulk * dz[k] * hbri / dt, 0.0))
            failed = (failed | (~conserve_C[k] & flags["tr_bgc_C"])
                      | (bulk > 1.0e8) | (bulk < 0.0))
            trcrn = trcrn.at[int(bio_index[m]) + k].set(
                jnp.maximum(0.0, jnp.where(tiny, 0.0, bulk)))
        if m + 1 == flags["nlt_bgc_Nit"]:
            new_fraction = jnp.where(
                (jnp.max(nitrification) > 0.0) & (nitrate_total > 0.0),
                nitrate_mobile / jnp.where(nitrate_total > 0.0, nitrate_total, 1.0),
                jnp.where(jnp.max(nitrification) > 0.0,
                          zbgc_frac_init[m], trcrn[frac_start + m]))
            trcrn = trcrn.at[frac_start + m].set(new_fraction)
        flux_bio = flux_bio.at[m].add(flux_bio_tmp[m])
    result = dict(fields)
    result.update({"trcrn": trcrn, "flux_bio": flux_bio,
                   "zfswin": zfswin, "Zoo": Zoo,
                   "ocean_bio_dh": ocean_bio_dh,
                   "grow_alg": grow_alg, "upNOn": upNOn,
                   "upNHn": upNHn})
    return result, failed


def zbio(dt, fields, flags, options, parameters, bio_parameters,
         tracer_type, tau_ret, tau_rel, zbgc_frac_init, kscavz):
    """Run snow transport, vertical biology and category flux accounting."""
    nbtrcr = int(flags["nbtrcr"])
    nblyr = int(options["nblyr"])
    nslyr = int(options["nslyr"])
    bio_index = flags["bio_index"][:nbtrcr]
    nt_bgc_N = flags["nt_bgc_N"][:int(flags["n_algae"])]
    nlt_bgc_N = tuple(k - 1 for k in flags["nlt_bgc_N"][:int(flags["n_algae"])])
    nt_bgc_DOC = flags["nt_bgc_DOC"][:int(flags["n_doc"])]
    nt_bgc_DIC = flags["nt_bgc_DIC"][:int(flags["n_dic"])]
    nt_bgc_DON = flags["nt_bgc_DON"][:int(flags["n_don"])]
    nt_bgc_hum = int(flags["nt_bgc_hum"])
    initial_carbon = bgc_carbon_sum(
        fields["hbri_old"], fields["trcrn"], nblyr,
        nt_bgc_N, nt_bgc_DOC, nt_bgc_DIC, nt_bgc_DON,
        nt_bgc_hum, bio_parameters["R_C2N"],
        bio_parameters["R_C2N_DON"], flags)

    (trcrn, snow_flux, zbgc_snown, zbgc_atmn,
     snow_failed) = update_snow_bgc(
        dt, fields["meltt"], fields["melts"], fields["meltb"],
        fields["congel"], fields["snoice"], fields["fsnow"],
        fields["trcrn"], bio_index, fields["aice_old"],
        fields["vice_old"], fields["vsno_old"],
        fields["vicen"], fields["vsnon"], fields["aicen"],
        fields["flux_bio_atm"], jnp.zeros((nbtrcr,), dtype=fields["trcrn"].dtype),
        flags["bio_index_o"][:nbtrcr], kscavz, nslyr, nblyr, parameters)

    vertical_fields = dict(fields)
    vertical_fields.update({
        "trcrn": trcrn,
        "flux_bio": jnp.zeros((nbtrcr,), dtype=trcrn.dtype),
        "zbgc_snow": zbgc_snown,
        "zbgc_atm": zbgc_atmn,
        "Zoo": jnp.zeros((nblyr + 1,), dtype=trcrn.dtype),
    })
    vertical, vertical_failed = z_biogeochemistry(
        dt, vertical_fields, flags, options, parameters,
        bio_parameters, tracer_type, tau_ret, tau_rel, zbgc_frac_init)
    flux_bion = vertical["flux_bio"] + snow_flux
    final_carbon = bgc_carbon_sum(
        fields["hbri"], vertical["trcrn"], nblyr,
        nt_bgc_N, nt_bgc_DOC, nt_bgc_DIC, nt_bgc_DON,
        nt_bgc_hum, bio_parameters["R_C2N"],
        bio_parameters["R_C2N_DON"], flags)
    carbon_flux = bgc_carbon_flux(
        fields["flux_bio_atm"], flux_bion,
        flags["nlt_bgc_N"], flags["nlt_bgc_DOC"],
        flags["nlt_bgc_DIC"], flags["nlt_bgc_DON"],
        flags["nlt_bgc_hum"], bio_parameters["R_C2N"],
        bio_parameters["R_C2N_DON"], flags)
    carbon_error = initial_carbon - carbon_flux * dt - final_carbon
    carbon_failed = (flags["tr_bgc_C"]
                     & (jnp.abs(carbon_error) > jnp.maximum(
                         parameters["puny"],
                         1.0e-13 * jnp.maximum(initial_carbon, final_carbon))))
    (flux_bio, zbgc_snow, zbgc_atm, PP_net, ice_bio_net,
     snow_bio_net, grow_net, totalChla, upNO, upNH,
     bioPorosityIceCell, bioSalinityIceCell,
     bioTemperatureIceCell) = merge_bgc_fluxes(
         dt, fields["aicen"], fields["vicen"], fields["vsnon"],
         fields["iphin"], vertical["trcrn"], flux_bion,
         fields["flux_bio"], vertical["upNOn"], vertical["upNHn"],
         fields["upNO"], fields["upNH"], zbgc_snown, zbgc_atmn,
         fields["fbio_snoice"], fields["fbio_atmice"], fields["PP_net"],
         fields["ice_bio_net"], fields["snow_bio_net"],
         vertical["grow_alg"], fields["grow_net"], fields["totalChla"],
         fields["iTin"], fields["iSin"],
         fields["bioPorosityIceCell"], fields["bioSalinityIceCell"],
         fields["bioTemperatureIceCell"], bio_index,
         int(flags["nt_fbri"]), nt_bgc_N, nlt_bgc_N,
         bio_parameters["R_chl2N"], bio_parameters["R_C2N"],
         nslyr, options["solve_zbgc"], parameters["secday"],
         parameters["fr_resp"], bio_parameters["R_gC2molC"],
         parameters["puny"])
    result = dict(fields)
    result.update({
        "trcrn": vertical["trcrn"], "zfswin": vertical["zfswin"],
        "Zoo": vertical["Zoo"], "ocean_bio_dh": vertical["ocean_bio_dh"],
        "flux_bion": flux_bion, "flux_bio": flux_bio,
        "fbio_snoice": zbgc_snow, "fbio_atmice": zbgc_atm,
        "PP_net": PP_net, "ice_bio_net": ice_bio_net,
        "snow_bio_net": snow_bio_net, "grow_net": grow_net,
        "totalChla": totalChla, "upNO": upNO, "upNH": upNH,
        "bioPorosityIceCell": bioPorosityIceCell,
        "bioSalinityIceCell": bioSalinityIceCell,
        "bioTemperatureIceCell": bioTemperatureIceCell,
    })
    return result, snow_failed | vertical_failed | carbon_failed
