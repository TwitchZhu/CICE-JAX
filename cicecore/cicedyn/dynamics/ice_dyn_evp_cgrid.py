"""C/CD-grid elastic-viscous-plastic sea ice dynamics.

author for JAX version: Xinyuan Zhu
XZ: converted from F90 to Python

author: JF Lemieux, ECCC
Nov 2021

Bouillon, S., T. Fichefet, V. Legat and G. Madec (2013). The
elastic-viscous-plastic method revisited. Ocean Model., 71, 2-12.
Kimmritz, M., S. Danilov and M. Losch (2016). The adaptive EVP method
for solving the sea ice momentum equation. Ocean Model., 101, 59-67.
"""

import jax
import jax.numpy as jnp

from cicecore.cicedyn.dynamics import ice_dyn_shared
from cicecore.cicedyn.dynamics.ice_dyn_shared import shift
from cicecore.cicedyn.infrastructure.ice_grid import grid_average_X2Y_jax
from cicecore.cicedyn.infrastructure.comm.serial.ice_boundary import ice_halo_update_jax
from icepack.columnphysics.icepack_mechred import icepack_ice_strength


def evp(pytree, static, dt):
    """Fortran evp C and CD branches, with fixed-shape masked JAX subcycles."""
    state = dict(pytree["state"])
    ice, flux, shared, work = (dict(state[n]) for n in (
        "ice_state", "ice_flux", "ice_dyn_shared", "ice_dyn_evp"))
    grid = pytree["constants"]["ice_grid"]
    params = pytree["parameters"]["icepack_parameters"]
    dyn = pytree["parameters"]["ice_dyn_shared"]
    options, names = static["ice_dyn_shared"], static["ice_init"]
    bounds = static["ice_domain_size"]["physical_bounds"]
    iglob = pytree["constants"]["ice_boundary"]["iglob"]
    is_c = names["grid_ice"].strip() == "C"
    ilo, ihi, jlo, jhi = bounds

    def halo(value, location="T", kind="scalar"):
        return ice_halo_update_jax(value, location, kind, static, iglob)

    def average(value, source, destination, old, mode="S"):
        return grid_average_X2Y_jax(mode, value, source, old, destination, grid, bounds)

    # preparation for dynamics
    work["tmass"], shared["iceTmask"] = ice_dyn_shared.dyn_prep1(
        ice["aice"], ice["vice"], ice["vsno"], grid["tmask"], bounds, params, dyn)
    shared["iceTmask"] = halo(shared["iceTmask"])
    work["tmass"], ice["aice_init"] = halo(work["tmass"]), halo(ice["aice_init"])
    column = dict(state["ice_arrays_column"])
    column["Cdn_ocn"] = halo(column["Cdn_ocn"])
    for name in ("uocn", "vocn", "ss_tltx", "ss_tlty"):
        flux[name] = halo(flux[name], "T", "vector")
    # Set wind stress to values supplied via NEMO or other forcing
    # Map T to U, N, E as needed
    # This wind stress is rotated on u grid and multiplied by aice in coupler
    if static["icepack_parameters"]["calc_strair"]:
        windx, windy = halo(flux["strairxT"], "T", "vector"), halo(flux["strairyT"], "T", "vector")
        wind_gridx = wind_gridy = "T"
    else:
        windx, windy = flux["strax"], flux["stray"]
        wind_gridx, wind_gridy = names["grid_atm_dynu"], names["grid_atm_dynv"]
    fields = {}
    for loc in ("U", "N", "E"):
        suffix = "" if loc == "U" else loc
        mass = loc.lower() + "mass"
        # convert fields from T to U grid
        for value, source, target in (
            (work["tmass"], "T", mass),
            (column["Cdn_ocn"], "T", "cdn_ocn" + loc),
            (flux["uocn"], names["grid_ocn_dynu"], "uocn" + loc),
            (flux["vocn"], names["grid_ocn_dynv"], "vocn" + loc),
            (flux["ss_tltx"], names["grid_ocn_dynu"], "ss_tltx" + loc),
            (flux["ss_tlty"], names["grid_ocn_dynv"], "ss_tlty" + loc),
        ):
            work[target] = average(value, source, loc, work[target])
        ai = average(ice["aice_init"], "T", loc, ice["aiU"] if loc == "U" else work["ai" + loc])
        if loc == "U":
            ice["aiU"] = ai
        else:
            work["ai" + loc] = ai
        flux["strairx" + loc] = average(windx, wind_gridx, loc, flux["strairx" + loc], "F")
        flux["strairy" + loc] = average(windy, wind_gridy, loc, flux["strairy" + loc], "F")
        f = {"aiX": ai, "Xmass": work[mass], "Xmassdti": work[mass + "dti"],
             "fcor": shared["fcor" + suffix + "_blk"],
             "iceXmask": shared["ice" + loc + "mask"],
             "uvel": ice["uvel" + suffix], "vvel": ice["vvel" + suffix],
             "uvel_init": shared["uvel" + suffix + "_init"],
             "vvel_init": shared["vvel" + suffix + "_init"], "TbU": flux["Tb" + loc]}
        for name in ("uocn", "vocn", "ss_tltx", "ss_tlty", "waterx", "watery", "forcex", "forcey"):
            f[name] = work[name + loc]
        for name in ("strairx", "strairy", "fm", "strtltx", "strtlty", "strocnx", "strocny",
                     "strintx", "strinty", "taubx", "tauby"):
            f[name] = flux[name + loc]
        for stem in ("stressp", "stressm", "stress12"):
            for k in range(1, 5):
                f[f"{stem}_{k}"] = flux[f"{stem}_{k}"]
        # more preparation for dynamics on N grid / E grid
        fields[loc] = ice_dyn_shared.dyn_prep2(
            f, shared["iceTmask"], grid["umaskCD" if loc == "U" else loc.lower() + "mask"],
            bounds, dt, params, dyn, options)
    maskT = jnp.zeros_like(shared["iceTmask"]).at[ilo:ihi+2, jlo:jhi+2].set(
        shared["iceTmask"][ilo:ihi+2, jlo:jhi+2])
    maskU = fields["U"]["iceXmask"]
    # ice strength
    ice["strength"] = halo(jnp.where(maskT, icepack_ice_strength(
        ice["aice"], ice["vice"], ice["aice0"], ice["aicen"], ice["vicen"],
        static["icepack_parameters"], params), 0.0))
    for loc, mask in (("T", shared["iceTmask"]), ("U", maskU)):
        for stem in ("stressp", "stressm", "stress12"):
            flux[stem + loc] = jnp.where(mask, flux[stem + loc], 0.0)
    for name in ("strengthU", "divergU", "tensionU", "shearU", "deltaU",
                 "zetax2T", "zetax2U", "etax2T", "etax2U"):
        work[name] = jnp.zeros_like(work[name])
    # seabed stress factor TbU (TbU is part of Cb coefficient)
    if options["seabed_stress"]:
        for loc in ("E", "N"):
            f = fields[loc]
            if options["seabed_stress_method"].strip() == "LKD":
                f["TbU"] = ice_dyn_shared.seabed_stress_factor_LKD(
                    ice["vice"], ice["aice"], flux["hwater"], f["TbU"], f["iceXmask"], loc, dyn)
            else:
                f["TbU"] = ice_dyn_shared.seabed_stress_factor_prob(
                    ice["aicen"], ice["vicen"], flux["hwater"], column["hin_max"],
                    shared["iceTmask"], f["iceXmask"], loc, params, dyn)
    # velocities may have changed in dyn_prep2
    for loc in ("E", "N"):
        for name in ("uvel", "vvel"):
            fields[loc][name] = halo(fields[loc][name], loc, "vector")
    if is_c:
        fields["N"]["uvel"] = halo(average(fields["E"]["uvel"], "E", "N", fields["N"]["uvel"], "A") * grid["npm"], "N", "vector")
        fields["E"]["vvel"] = halo(average(fields["N"]["vvel"], "N", "E", fields["E"]["vvel"], "A") * grid["epm"], "E", "vector")
    for name, loc in (("uvel", "E"), ("vvel", "N")):
        fields["U"][name] = halo(average(fields[loc][name], loc, "U", fields["U"][name], "A") * grid["uvm"], "U", "vector")

    def subcycle(_, carry):
        fields, stress, work = carry
        fields = {loc: dict(f) for loc, f in fields.items()}
        stress, work = dict(stress), dict(work)
        E, N, U = fields["E"], fields["N"], fields["U"]
        # strain rates at U point
        # NOTE these are actually strain rates * area  (m^2/s)
        du, tu, su, deltaU = ice_dyn_shared.strain_rates_U(
            E["uvel"], E["vvel"], N["uvel"], N["vvel"], U["uvel"], U["vvel"],
            grid, work, maskU, dyn)
        if is_c:
            su = halo(su, "U")
        work.update(divergU=du, tensionU=tu, shearU=su, deltaU=deltaU)
        divT, tensionT, shearT, deltaT = ice_dyn_shared.strain_rates_T(
            E["uvel"], E["vvel"], N["uvel"], N["vvel"], grid, maskT, dyn)
        if is_c:
            # Square of shear strain rate at T obtained from interpolation of
            # U point values (Bouillon et al., 2013, Kimmritz et al., 2016
            area = grid["uarea"]
            denom = area + shift(area, 0, -1) + shift(area, -1, -1) + shift(area, -1)
            sq = su**2 * area
            sh = su * area
            shearTsqr = (sq + shift(sq, 0, -1) + shift(sq, -1, -1) + shift(sq, -1)) / denom
            shearT = (sh + shift(sh, 0, -1) + shift(sh, -1, -1) + shift(sh, -1)) / denom
            deltaT = jnp.sqrt(divT**2 + dyn["e_factor"] * (tensionT**2 + shearTsqr))
        # viscosities and replacement pressure at T point
        zeta, eta, pressure = ice_dyn_shared.visc_replpress(
            ice["strength"], shared["DminTarea"], deltaT, dyn)
        work["zetax2T"] = halo(jnp.where(maskT, zeta, work["zetax2T"]))
        work["etax2T"] = halo(jnp.where(maskT, eta, work["etax2T"]))
        relax, retain = dyn["arlx1i"], 1.0 - dyn["arlx1i"] * dyn["revp"]
        # the stresses                            ! kg/s^2
        # NOTE: for comp. efficiency 2 x zeta and 2 x eta are used in the code
        for stem, target in (("stressp", zeta * divT - pressure),
                             ("stressm", eta * tensionT), ("stress12", 0.5 * eta * shearT)):
            name = stem + "T"
            stress[name] = jnp.where(maskT, (stress[name] * retain + relax * target) * dyn["denom1"], stress[name])
            if not is_c or stem != "stress12":
                stress[name] = halo(stress[name])
        # viscosities and replacement pressure at U point
        # avg_zeta: Bouillon et al. 2013, C1 method of Kimmritz et al. 2016
        # avg_strength: C2 method of Kimmritz et al. 2016
        if options["visc_method"].strip() == "avg_strength":
            work["strengthU"] = average(ice["strength"], "T", "U", work["strengthU"])
            # only need etax2U here, but other terms are calculated with etax2U
            # minimal extra calculations here even though it seems like there is
            zu, eu, pu = ice_dyn_shared.visc_replpress(
                work["strengthU"], dyn["deltaminEVP"] * grid["uarea"], deltaU, dyn)
        else:
            work["etax2U"] = average(work["etax2T"], "T", "U", work["etax2U"])
            if not is_c:
                work["zetax2U"] = average(work["zetax2T"], "T", "U", work["zetax2U"])
            zu, eu = work["zetax2U"], work["etax2U"]
            pu = (1.0 - dyn["Ktens"]) / (1.0 + dyn["Ktens"]) * zu * deltaU
        for stem, target in (("stressp", zu * du - pu), ("stressm", eu * tu), ("stress12", 0.5 * eu * su)):
            if is_c and stem != "stress12":
                continue
            name = stem + "U"
            stress[name] = halo(jnp.where(maskU, (stress[name] * retain + relax * target) * dyn["denom1"], stress[name]), "U")
        # Computes divergence of stress tensor at the E or N point for the mom equation
        pT, mT, sT = (stress[n + "T"] for n in ("stressp", "stressm", "stress12"))
        pU, mU, sU = (stress[n + "U"] for n in ("stressp", "stressm", "stress12"))
        dxE, dyE, dxN, dyN, dxT, dyT, dxU, dyU = (grid[n] for n in ("dxE", "dyE", "dxN", "dyN", "dxT", "dyT", "dxU", "dyU"))
        ex = grid["earear"] * (0.5 * dyE * (shift(pT, 1) - pT)
            + 0.5 / dyE * (shift(dyT**2 * mT, 1) - dyT**2 * mT)
            + 1.0 / dxE * (dxU**2 * sU - shift(dxU**2 * sU, 0, -1)))
        ny = grid["narear"] * (0.5 * dxN * (shift(pT, 0, 1) - pT)
            - 0.5 / dxN * (shift(dxT**2 * mT, 0, 1) - dxT**2 * mT)
            + 1.0 / dyN * (dyU**2 * sU - shift(dyU**2 * sU, -1)))
        E["strintx"] = jnp.where(E["iceXmask"], ex, E["strintx"])
        N["strinty"] = jnp.where(N["iceXmask"], ny, N["strinty"])
        if not is_c:
            ey = grid["earear"] * (0.5 * dxE * (pU - shift(pU, 0, -1))
                - 0.5 / dxE * (dxU**2 * mU - shift(dxU**2 * mU, 0, -1))
                + 1.0 / dyE * (shift(dyT**2 * sT, 1) - dyT**2 * sT))
            nx = grid["narear"] * (0.5 * dyN * (pU - shift(pU, -1))
                + 0.5 / dyN * (dyU**2 * mU - shift(dyU**2 * mU, -1))
                + 1.0 / dxN * (shift(dxT**2 * sT, 0, 1) - dxT**2 * sT))
            E["strinty"] = jnp.where(E["iceXmask"], ey, E["strinty"])
            N["strintx"] = jnp.where(N["iceXmask"], nx, N["strintx"])
        # momentum equation
        fields["E"] = ice_dyn_shared.stepuv_CD(E, work["cdn_ocnE"], params, dyn, "u" if is_c else "both")
        fields["N"] = ice_dyn_shared.stepuv_CD(N, work["cdn_ocnN"], params, dyn, "v" if is_c else "both")
        E, N = fields["E"], fields["N"]
        E["uvel"], N["vvel"] = halo(E["uvel"], "E", "vector"), halo(N["vvel"], "N", "vector")
        if is_c:
            N["uvel"] = average(E["uvel"], "E", "N", N["uvel"], "A") * grid["npm"]
            E["vvel"] = average(N["vvel"], "N", "E", E["vvel"], "A") * grid["epm"]
        N["uvel"], E["vvel"] = halo(N["uvel"], "N", "vector"), halo(E["vvel"], "E", "vector")
        U["uvel"] = halo(average(E["uvel"], "E", "U", U["uvel"], "A") * grid["uvm"], "U", "vector")
        U["vvel"] = halo(average(N["vvel"], "N", "U", U["vvel"], "A") * grid["uvm"], "U", "vector")
        return fields, stress, work

    stress = {stem + loc: flux[stem + loc] for stem in ("stressp", "stressm", "stress12") for loc in ("T", "U")}
    fields, stress, work = jax.lax.fori_loop(0, int(options["ndte"]), subcycle, (fields, stress, work))
    flux.update(stress)
    for stem in ("stressp", "stressm", "stress12"):
        for k in range(1, 5):
            flux[f"{stem}_{k}"] = fields["U"][f"{stem}_{k}"]
    # Force symmetry across the tripole seam. The reference applies this to
    # the corner fields only; special C/CD T/U stress treatment is its TODO.
    if static["ice_domain"]["ns_boundary_type"].strip() in ("tripole", "tripoleT"):
        for stem in ("stressp", "stressm", "stress12"):
            for destination, source in ((1, 3), (3, 1), (2, 4), (4, 2)):
                name = f"{stem}_{destination}"
                flux[name] = ice_halo_update_jax(
                    flux[name], "T", "scalar", static, iglob, tripole_only=True,
                    source=flux[f"{stem}_{source}"], seam_average=False)
    E, N = fields["E"], fields["N"]
    # save quantities for mechanical redistribution
    divT, tensionT, shearT, deltaT = ice_dyn_shared.strain_rates_T(
        E["uvel"], E["vvel"], N["uvel"], N["vvel"], grid, maskT, dyn)
    if is_c:
        # DeltaT is calc by strain_rates_T but replaced by calculation below.
        area, su = grid["uarea"], work["shearU"]
        sq = su**2 * area
        shearTsqr = (sq + shift(sq, 0, -1) + shift(sq, -1, -1) + shift(sq, -1)) / (
            area + shift(area, 0, -1) + shift(area, -1, -1) + shift(area, -1))
        deltaT = jnp.sqrt(divT**2 + dyn["e_factor"] * (tensionT**2 + shearTsqr))
    ice["divu"] = jnp.where(maskT, divT * grid["tarear"], 0.0)
    flux["rdg_conv"] = -jnp.minimum(ice["divu"], 0.0)
    flux["rdg_shear"] = jnp.where(maskT, 0.5 * (deltaT * grid["tarear"] - jnp.abs(ice["divu"])), 0.0)
    # diagnostic only...maybe we dont want to use shearTsqr here????
    # shear = sqrt(tension**2 + shearing**2)
    ice["shear"] = jnp.where(maskT, grid["tarear"] * jnp.sqrt(tensionT**2 + shearT**2), 0.0)
    # vorticity
    ice["vort"] = jnp.where(maskT, grid["tarear"] * (
        grid["dyE"] * E["vvel"] - shift(grid["dyE"] * E["vvel"], -1)
        - (grid["dxN"] * N["uvel"] - shift(grid["dxN"] * N["uvel"], 0, -1))), 0.0)
    # ice-ocean stress
    for loc in ("U", "N", "E"):
        f = fields[loc]
        suffix = "" if loc == "U" else loc
        f["strocnx"], f["strocny"] = ice_dyn_shared.dyn_finish(
            f["uvel"], f["vvel"], f["uocn"], f["vocn"], f["aiX"], f["fm"],
            work["cdn_ocn" + loc], f["strocnx"], f["strocny"], f["iceXmask"], params)
        shared["ice" + loc + "mask"] = f["iceXmask"]
        for name in ("uvel", "vvel"):
            ice[name + suffix] = f[name]
            shared[name + suffix + "_init"] = f[name + "_init"]
        work[loc.lower() + "massdti"] = f["Xmassdti"]
        for name in ("waterx", "watery", "forcex", "forcey"):
            work[name + loc] = f[name]
        for name in ("fm", "strtltx", "strtlty", "strocnx", "strocny", "strintx", "strinty", "taubx", "tauby"):
            flux[name + loc] = f[name]
        flux["Tb" + loc] = f["TbU"]
    for name, loc in (("strintx", "E"), ("strinty", "N")):
        flux[name + loc] = halo(flux[name + loc], loc, "vector")
        flux[name + "U"] = average(flux[name + loc], loc, "U", flux[name + "U"])  # diagnostic
    state.update(ice_state=ice, ice_flux=flux, ice_dyn_shared=shared, ice_dyn_evp=work, ice_arrays_column=column)
    return {**pytree, "state": state}
