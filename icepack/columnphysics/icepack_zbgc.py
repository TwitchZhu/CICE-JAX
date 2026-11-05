"""JAX translations of category biogeochemistry in icepack_zbgc.F90."""

import jax.numpy as jnp
import jax

from icepack.columnphysics.icepack_itd import column_sum
from icepack.columnphysics.icepack_brine import (
    compute_microS_mushy, preflushing_changes, update_hbrine,
)
from icepack.columnphysics.icepack_algae import sklbio, zbio


def icepack_load_ocean_bio_array(fields, ocean_bio_all,
                                 R_chl2N, R_S2N, flags):
    """Pack ocean tracers in Icepack's fixed biological tracer order."""
    max_algae = int(flags["max_algae"])
    max_doc = int(flags["max_doc"])
    max_dic = int(flags["max_dic"])
    max_don = int(flags["max_don"])
    max_fe = int(flags["max_fe"])
    max_aero = int(flags["max_aero"])
    ocean = jnp.zeros_like(ocean_bio_all)
    algalN = fields["algalN"]
    for k in range(max_algae):
        ocean = ocean.at[k].set(algalN[k])
        slot = max_algae + max_doc + max_dic + 1 + k
        ocean = ocean.at[slot].set(R_chl2N[k] * algalN[k])
    slot = max_algae + 1
    for k in range(max_doc):
        ocean = ocean.at[slot + k].set(fields["doc"][k])
    slot = slot + max_doc
    for k in range(max_dic):
        ocean = ocean.at[slot + k].set(fields["dic"][k])
    slot = 2 * max_algae + max_doc + max_dic + 7
    for k in range(max_don):
        ocean = ocean.at[slot + k].set(fields["don"][k])
    slot = max_algae + 1
    ocean = ocean.at[slot - 1].set(fields["nit"])
    slot = 2 * max_algae + max_doc + max_dic + 2
    ocean = ocean.at[slot - 1].set(fields["amm"])
    ocean = ocean.at[slot].set(fields["sil"])
    ocean = ocean.at[slot + 1].set(jnp.sum(R_S2N[:3] * algalN[:3]))
    ocean = ocean.at[slot + 2].set(fields["dmsp"])
    ocean = ocean.at[slot + 3].set(fields["dms"])
    ocean = ocean.at[slot + 4].set(fields["nit"])
    slot = 2 * max_algae + max_doc + max_dic + max_don + 7
    for k in range(max_fe):
        ocean = ocean.at[slot + k].set(fields["fed"][k])
    slot = slot + max_fe
    for k in range(max_fe):
        ocean = ocean.at[slot + k].set(fields["fep"][k])
    slot = slot + max_fe
    for k in range(max_aero):
        ocean = ocean.at[slot + k].set(fields["zaeros"][k])
    ocean = ocean.at[slot + max_aero].set(fields["hum"])
    return ocean


def update_vertical_bio_tracers(trc, h1, h2, trc0, zspace, puny):
    """Remap interface biology after adding new brine at the bottom."""
    nbiolyr = trc.shape[0]
    trc2 = jnp.zeros_like(trc)
    z2b = jnp.zeros_like(h2)
    for k2 in range(nbiolyr):
        z2a = z2b
        z2b = z2b + h2 * zspace[k2]
        z1b = jnp.zeros_like(h1)
        amount = jnp.zeros_like(h2)
        for k1 in range(nbiolyr):
            z1a = z1b
            z1b = z1b + h1 * zspace[k1]
            overlap = jnp.maximum(
                jnp.minimum(z1b, z2b) - jnp.maximum(z1a, z2a), 0.0)
            amount = amount + overlap * trc[k1]
        overlap = jnp.maximum(
            jnp.minimum(h2, z2b) - jnp.maximum(h1, z2a), 0.0)
        amount = amount + overlap * trc0
        trc2 = trc2.at[k2].set(amount / (
            zspace[k2] * jnp.where(h2 > puny, h2, 1.0)))
    return jnp.where(h2 > puny, trc2, trc)


def add_new_ice_bgc(dt, ncats, aicen_init, vicen_init, vi0_init,
                    aicen, vicen, vin0new, trcrn, ocean_bio,
                    flux_bio, hsurp, d_an_tot, bio_index,
                    zbgc_init_frac, flags, options, parameters):
    """Add ocean tracers to new brine in every thickness category."""
    ncat = aicen.shape[0]
    nblyr = int(options["nblyr"])
    nbtrcr = int(flags["nbtrcr"])
    puny = parameters["puny"]
    zspace = jnp.full((nblyr + 1,), 1.0 / nblyr)
    zspace = zspace.at[0].set(0.5 / nblyr)
    zspace = zspace.at[-1].set(0.5 / nblyr)
    if bool(flags["tr_brine"]):
        vbrin = trcrn[int(flags["nt_fbri"])] * vicen_init
    else:
        vbrin = vicen_init
    initial_brine = column_sum(ncat, vbrin) + vi0_init
    for m in range(nbtrcr):
        flux_bio = flux_bio.at[m].add(
            -vi0_init / dt * ocean_bio[m] * zbgc_init_frac[m])

    for n in range(ncat):
        added = hsurp > 0.0
        old_brine = vbrin[n]
        vsurp = hsurp * aicen_init[n]
        new_brine = old_brine + vsurp
        vbrin = vbrin.at[n].set(jnp.where(added, new_brine, old_brine))
        vice_new = vicen_init[n] + vsurp
        if bool(flags["tr_brine"]):
            slot = int(flags["nt_fbri"])
            value = jnp.where(
                vice_new > puny,
                new_brine / jnp.where(vice_new > puny, vice_new, 1.0),
                jnp.where(vicen[n] <= 0.0, 1.0, trcrn[slot, n]))
            trcrn = trcrn.at[slot, n].set(jnp.where(
                added, value, trcrn[slot, n]))
        for m in range(nbtrcr):
            slot = int(bio_index[m])
            concentration = ocean_bio[m] * zbgc_init_frac[m]
            remapped = update_vertical_bio_tracers(
                trcrn[slot:slot + nblyr + 1, n], old_brine,
                jnp.where(added, new_brine, 0.0),
                concentration, zspace, puny)
            trcrn = trcrn.at[slot:slot + nblyr + 1, n].set(jnp.where(
                added, remapped, trcrn[slot:slot + nblyr + 1, n]))

    for n in range(ncats):
        added = (vin0new[n] > 0.0) & (d_an_tot[n] > 0.0)
        old_brine = vbrin[n]
        new_brine = old_brine + vin0new[n]
        vbrin = vbrin.at[n].set(jnp.where(added, new_brine, old_brine))
        if bool(flags["tr_brine"]):
            slot = int(flags["nt_fbri"])
            value = jnp.where(
                vicen[n] > puny,
                new_brine / jnp.where(vicen[n] > puny, vicen[n], 1.0),
                1.0)
            trcrn = trcrn.at[slot, n].set(jnp.where(
                added, value, trcrn[slot, n]))
        for m in range(nbtrcr):
            slot = int(bio_index[m])
            concentration = ocean_bio[m] * zbgc_init_frac[m]
            current = trcrn[slot:slot + nblyr + 1, n]
            mixed = (current * old_brine + concentration * vin0new[n]) / (
                jnp.where(new_brine > puny, new_brine, 1.0))
            trcrn = trcrn.at[slot:slot + nblyr + 1, n].set(jnp.where(
                added & (new_brine > puny), mixed, current))
    failed = (bool(flags["tr_brine"]) and bool(options["conserv_check"]))
    failed = jnp.asarray(failed) & (
        jnp.abs(column_sum(ncat, vbrin) - initial_brine) > puny)
    return trcrn, flux_bio, failed


def lateral_melt_bgc(dt, rsiden, vicen_init, trcrn, flux_bio,
                     bio_index, flags, options):
    """Send laterally melted brine tracers to the ocean."""
    nblyr = int(options["nblyr"])
    zspace = jnp.full((nblyr + 1,), 1.0 / nblyr)
    zspace = zspace.at[0].set(0.5 / nblyr)
    zspace = zspace.at[-1].set(0.5 / nblyr)
    for m in range(int(flags["nbtrcr"])):
        for n in range(vicen_init.shape[0]):
            for k in range(nblyr + 1):
                flux_bio = flux_bio.at[m].add(
                    trcrn[int(flags["nt_fbri"]), n] * vicen_init[n]
                    * zspace[k] * trcrn[int(bio_index[m]) + k, n]
                    * rsiden[n] / dt)
    return flux_bio


def icepack_biogeochemistry(dt, fields, flags, options, parameters,
                            bio_parameters=None, tracer_type=None,
                            bio_constants=None):
    """Advance brine microstructure and category biogeochemistry."""
    trcrn = fields["trcrn"]
    first_ice = fields["first_ice"]
    hin_old = fields["hin_old"]
    iDi = fields["iDi"]
    bphi = fields["bphi"]
    bTiz = fields["bTiz"]
    dhbr_top = fields["dhbr_top"]
    dhbr_bot = fields["dhbr_bot"]
    darcy_V = fields["darcy_V"]
    hbri = fields["hbri"]
    biological = bool(options["skl_bgc"] or options["z_tracers"])
    flux_bio = fields["flux_bio"] if biological else None
    PP_net = fields["PP_net"] if biological else None
    upNO = fields["upNO"] if biological else None
    upNH = fields["upNH"] if biological else None
    grow_net = fields["grow_net"] if biological else None
    bio_state = ({name: fields[name] for name in (
        "flux_bio", "flux_bio_atm", "fbio_snoice", "fbio_atmice",
        "ice_bio_net", "snow_bio_net", "PP_net", "upNO", "upNH",
        "grow_net", "zfswin", "Zoo")}
        if options["z_tracers"] else {})
    bgc_failed = jnp.asarray(False)
    puny = parameters["puny"]
    fbri_slot = int(flags["nt_fbri"])
    nilyr = int(options["nilyr"])
    for n in range(trcrn.shape[-1]):
        previous_ice = fields["aicen_init"][n] > puny
        hin_before = jnp.where(
            previous_ice,
            fields["vicen_init"][n]
            / jnp.where(previous_ice, fields["aicen_init"][n], 1.0),
            0.0)
        hin_old = hin_old.at[n].set(hin_before)
        first_ice = first_ice.at[n].set(jnp.where(
            previous_ice, first_ice[n], True))
        if flags["tr_brine"]:
            trcrn = trcrn.at[fbri_slot, n].set(jnp.where(
                previous_ice, trcrn[fbri_slot, n], 1.0))
            if options["z_tracers"]:
                # Fortran initializes mobile fractions only when a category
                # acquires ice for the first time.
                frac_start = int(flags["nt_zbgc_frac"])
                for m in range(int(flags["nbtrcr"])):
                    slot = frac_start + m
                    trcrn = trcrn.at[slot, n].set(jnp.where(
                        previous_ice, trcrn[slot, n],
                        bio_constants["zbgc_frac_init"][m]))

            def advance_category(carry):
                (trcrn, first_ice, iDi, bphi, bTiz, dhbr_top,
                 dhbr_bot, darcy_V, hbri, bio_state, bgc_failed) = carry
                fbri = jnp.where(trcrn[fbri_slot, n] <= 0.0,
                                 1.0, trcrn[fbri_slot, n])
                trcrn = trcrn.at[fbri_slot, n].set(fbri)
                (hin, hsn, hbr_old, top_change, bottom_change,
                 _) = preflushing_changes(
                     fields["aicen"][n], fields["vicen"][n],
                     fields["vsnon"][n], fields["meltbn"][n],
                     fields["melttn"][n], fields["congeln"][n],
                     fields["snoicen"][n], hin_before, fbri)
                (bTin, bphin, _, iphin, iTin, iSin, diffusivity,
                 kperm, bphi_min) = compute_microS_mushy(
                     trcrn[:, n], hin_before, hbr_old,
                     fields["sss"], fields["sst"],
                     fields["cgrid"], fields["bgrid"], fields["igrid"],
                     int(flags["nt_sice"]), int(flags["nt_qice"]),
                     nilyr, parameters)
                (hbr, fbri, top_change, bottom_change, _, _,
                 darcy, _, dh_direct, bphi_o) = update_hbrine(
                     fields["melttn"][n], fields["meltsn"][n], dt,
                     hin, hsn, hin_before, hbr_old, fbri,
                     top_change, bottom_change, kperm, bphi_min,
                     bphin[1], fields["aice0"], parameters)
                trcrn = trcrn.at[fbri_slot, n].set(fbri)
                iDi = iDi.at[:, n].set(diffusivity)
                bphi = bphi.at[:, n].set(bphin)
                bTiz = bTiz.at[:, n].set(bTin)
                dhbr_top = dhbr_top.at[n].set(top_change)
                dhbr_bot = dhbr_bot.at[n].set(bottom_change)
                darcy_V = darcy_V.at[n].set(darcy)
                hbri = hbri + hbr * fields["aicen"][n]
                if options["z_tracers"]:
                    zfields = {
                        "trcrn": trcrn[:, n], "meltt": fields["melttn"][n],
                        "melts": fields["meltsn"][n], "meltb": fields["meltbn"][n],
                        "congel": fields["congeln"][n], "snoice": fields["snoicen"][n],
                        "fsnow": fields["fsnow"], "aice_old": fields["aicen_init"][n],
                        "vice_old": fields["vicen_init"][n],
                        "vsno_old": fields["vsnon_init"][n],
                        "aicen": fields["aicen"][n], "vicen": fields["vicen"][n],
                        "vsnon": fields["vsnon"][n],
                        "first_ice": first_ice[n], "hice_old": hin_before,
                        "hbri_old": hbr_old, "hbri": hbr,
                        "ocean_bio": fields["ocean_bio"],
                        "bphin": bphin, "iphin": iphin,
                        "iDin": diffusivity, "fswthrul": fields["fswpenln"][:, n],
                        "dh_top": top_change, "dh_bot": bottom_change,
                        "darcy_V": darcy, "bphi_min": bphi_o,
                        "iTin": iTin, "iSin": iSin, "dh_direct": dh_direct,
                        "bgrid": fields["bgrid"], "igrid": fields["igrid"],
                        "icgrid": fields["icgrid"],
                        "flux_bio": bio_state["flux_bio"],
                        "flux_bio_atm": bio_state["flux_bio_atm"],
                        "fbio_snoice": bio_state["fbio_snoice"],
                        "fbio_atmice": bio_state["fbio_atmice"],
                        "ice_bio_net": bio_state["ice_bio_net"],
                        "snow_bio_net": bio_state["snow_bio_net"],
                        "PP_net": bio_state["PP_net"],
                        "upNO": bio_state["upNO"], "upNH": bio_state["upNH"],
                        "grow_net": bio_state["grow_net"],
                        "zfswin": bio_state["zfswin"][:, n],
                        "Zoo": bio_state["Zoo"][:, n],
                        "totalChla": jnp.zeros((), dtype=trcrn.dtype),
                        "bioPorosityIceCell": jnp.zeros((options["nblyr"] + 1,), dtype=trcrn.dtype),
                        "bioSalinityIceCell": jnp.zeros((options["nblyr"] + 1,), dtype=trcrn.dtype),
                        "bioTemperatureIceCell": jnp.zeros((options["nblyr"] + 1,), dtype=trcrn.dtype),
                    }
                    advanced, failed = zbio(
                        dt, zfields, flags, options, parameters, bio_parameters,
                        tracer_type, bio_constants["tau_ret"],
                        bio_constants["tau_rel"],
                        bio_constants["zbgc_frac_init"], bio_constants["kscavz"])
                    trcrn = trcrn.at[:, n].set(advanced["trcrn"])
                    for name in ("flux_bio", "fbio_snoice", "fbio_atmice",
                                 "ice_bio_net", "snow_bio_net", "PP_net",
                                 "upNO", "upNH", "grow_net"):
                        bio_state[name] = advanced[name]
                    bio_state["zfswin"] = bio_state["zfswin"].at[:, n].set(advanced["zfswin"])
                    bio_state["Zoo"] = bio_state["Zoo"].at[:, n].set(advanced["Zoo"])
                    bgc_failed = bgc_failed | failed
                return (trcrn, first_ice, iDi, bphi, bTiz,
                        dhbr_top, dhbr_bot, darcy_V, hbri,
                        bio_state, bgc_failed)

            (trcrn, first_ice, iDi, bphi, bTiz, dhbr_top,
             dhbr_bot, darcy_V, hbri, bio_state, bgc_failed) = jax.lax.cond(
                 fields["aicen"][n] > puny, advance_category,
                 lambda carry: carry,
                 (trcrn, first_ice, iDi, bphi, bTiz, dhbr_top,
                  dhbr_bot, darcy_V, hbri, bio_state, bgc_failed))

        if options["z_tracers"]:
            # An extinct category returns its remaining brine inventory to
            # the ocean, using the old area and thickness as in Fortran.
            vanished = fields["aicen"][n] <= puny
            zspace = jnp.full((options["nblyr"] + 1,),
                              1.0 / options["nblyr"], dtype=trcrn.dtype)
            zspace = zspace.at[0].set(0.5 / options["nblyr"])
            zspace = zspace.at[-1].set(0.5 / options["nblyr"])
            for m in range(int(flags["nbtrcr"])):
                slot = int(flags["bio_index"][m])
                inventory = jnp.sum(trcrn[slot:slot + options["nblyr"] + 1, n]
                                    * zspace) * trcrn[fbri_slot, n]
                bio_state["flux_bio"] = bio_state["flux_bio"].at[m].add(
                    jnp.where(vanished, inventory * fields["vicen_init"][n] / dt, 0.0))
                trcrn = trcrn.at[slot:slot + options["nblyr"] + 1, n].set(
                    jnp.where(vanished, 0.0,
                              trcrn[slot:slot + options["nblyr"] + 1, n]))

        if options["skl_bgc"] and not options["z_tracers"]:
            def advance_skeletal(carry):
                (trcrn, flux_bio, PP_net, upNO, upNH,
                 grow_net, bgc_failed) = carry
                (tracer_column, flux_bio, PP_net, upNO, upNH,
                 grow_net, _, _, conserve_C, _, invalid) = sklbio(
                    dt, fields["Tf"], flux_bio, fields["ocean_bio"],
                    fields["aicen"][n], fields["meltbn"][n],
                    fields["congeln"][n], fields["fswthrun"][n],
                    first_ice[n], trcrn[:, n], PP_net, upNO, upNH,
                    grow_net, flags, options, flags, tracer_type,
                    parameters, bio_parameters)
                trcrn = trcrn.at[:, n].set(tracer_column)
                bgc_failed = bgc_failed | invalid | ~conserve_C
                return (trcrn, flux_bio, PP_net, upNO, upNH,
                        grow_net, bgc_failed)

            (trcrn, flux_bio, PP_net, upNO, upNH,
             grow_net, bgc_failed) = jax.lax.cond(
                 fields["aicen"][n] > puny, advance_skeletal,
                 lambda carry: carry,
                 (trcrn, flux_bio, PP_net, upNO, upNH,
                  grow_net, bgc_failed))
        first_ice = first_ice.at[n].set(jnp.where(
            fields["aicen"][n] > puny, False, first_ice[n]))
    result = dict(fields)
    result.update({
        "trcrn": trcrn, "first_ice": first_ice, "hin_old": hin_old,
        "iDi": iDi, "bphi": bphi, "bTiz": bTiz,
        "dhbr_top": dhbr_top, "dhbr_bot": dhbr_bot,
        "darcy_V": darcy_V, "hbri": hbri,
    })
    if options["z_tracers"]:
        result.update(bio_state)
        result["bgc_failed"] = bgc_failed
    if options["skl_bgc"] and not options["z_tracers"]:
        result.update({"flux_bio": flux_bio, "PP_net": PP_net,
                       "upNO": upNO, "upNH": upNH,
                       "grow_net": grow_net, "bgc_failed": bgc_failed})
    return result
