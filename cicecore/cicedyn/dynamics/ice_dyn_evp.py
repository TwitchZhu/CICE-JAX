"""Elastic-viscous-plastic sea ice dynamics model
Computes ice velocity and deformation

See:

Hunke, E. C., and J. K. Dukowicz (1997). An elastic-viscous-plastic model
for sea ice dynamics. J. Phys. Oceanogr., 27, 1849-1867.

Hunke, E. C. (2001).  Viscous-Plastic Sea Ice Dynamics with the EVP Model:
Linearization Issues. J. Comput. Phys., 170, 18-38.

Hunke, E. C., and J. K. Dukowicz (2002).  The Elastic-Viscous-Plastic
Sea Ice Dynamics Model in General Orthogonal Curvilinear Coordinates
on a Sphere - Incorporation of Metric Terms. Mon. Weather Rev.,
130, 1848-1865.

Hunke, E. C., and J. K. Dukowicz (2003).  The sea ice momentum
equation in the free drift regime.  Los Alamos Tech. Rep. LA-UR-03-2219.

Hibler, W. D. (1979). A dynamic thermodynamic sea ice model. J. Phys.
Oceanogr., 9, 817-846.

Bouillon, S., T. Fichefet, V. Legat and G. Madec (2013).  The
elastic-viscous-plastic method revisited.  Ocean Model., 71, 2-12.

author for JAX version: Xinyuan Zhu
XZ: converted from F90 to Python

author: Elizabeth C. Hunke, LANL

2003: Vectorized by Clifford Chen (Fujitsu) and William Lipscomb (LANL)
2004: Block structure added by William Lipscomb
2005: Removed boundary calls for stress arrays (WHL)
2006: Streamlined for efficiency by Elizabeth Hunke
      Converted to free source form (F90)

One grid, no block index. The horizontal driver calls ``evp`` as in Fortran.
The shared-memory algorithm uses flattened JAX kernels and fixed-size masks
instead of the Fortran MPI gather/scatter and OpenMP storage machinery.
"""

from functools import partial

import numpy as np
import jax
import jax.numpy as jnp

from cicecore.cicedyn.dynamics import ice_dyn_shared
from cicecore.cicedyn.infrastructure.ice_grid import grid_average_X2Y_jax
from cicecore.cicedyn.infrastructure.comm.serial.ice_boundary import (
    ice_halo_update_center_jax, ice_halo_update_jax,
)
from icepack.columnphysics.icepack_mechred import icepack_ice_strength
from cicecore.cicedyn.infrastructure import ice_grid
from cicecore.shared import ice_calendar
from cicecore.shared import ice_domain_size
from cicecore.shared.ice_constants import c1, p027, p055, p111, p166, p222, p333
from cicecore.shared.ice_kinds_mod import dbl_kind


def stress(uvel, vvel, strength, stress_fields, grid, shared,
           iceTmask, dyn_parameters):
    """B-grid EVP stress and the eight momentum-force combinations."""
    # Fortran stress: strain rates at each T-cell corner are multiplied by
    # cell area; the corner order is NE, NW, SW, SE (indices 1 through 4).
    div, tension, shear, delta = ice_dyn_shared.strain_rates(
        uvel, vvel, grid["dxT"], grid["dyT"],
        shared["cxp"], shared["cyp"], shared["cxm"], shared["cym"],
        iceTmask, dyn_parameters)
    zetax2, etax2, rep_prs = ice_dyn_shared.visc_replpress(
        strength[..., None], shared["DminTarea"][..., None],
        delta, dyn_parameters)
    relax = dyn_parameters["arlx1i"]
    retain = 1.0 - relax * dyn_parameters["revp"]
    denom = dyn_parameters["denom1"]
    result = dict(stress_fields)
    # As in Fortran, zetax2 and etax2 are twice the bulk and shear viscosity.
    for corner in range(4):
        k = corner + 1
        name = f"stressp_{k}"
        result[name] = jnp.where(iceTmask, (
            stress_fields[name] * retain
            + relax * (zetax2[..., corner] * div[..., corner]
                       - rep_prs[..., corner])) * denom, stress_fields[name])
        name = f"stressm_{k}"
        result[name] = jnp.where(iceTmask, (
            stress_fields[name] * retain
            + relax * etax2[..., corner] * tension[..., corner]) * denom,
            stress_fields[name])
        name = f"stress12_{k}"
        result[name] = jnp.where(iceTmask, (
            stress_fields[name] * retain
            + relax * 0.5 * etax2[..., corner] * shear[..., corner]) * denom,
            stress_fields[name])

    return result, stress_combinations(result, grid, shared, iceTmask)


def stress_combinations(result, grid, shared, iceTmask):
    """Combinations of the stresses for the momentum equation (kg/s^2)."""
    pn, pw, psw, pse = (result[f"stressp_{k}"] for k in range(1, 5))
    mn, mw, msw, mse = (result[f"stressm_{k}"] for k in range(1, 5))
    tn, tw, tsw, tse = (result[f"stress12_{k}"] for k in range(1, 5))
    ssigpn, ssigps = pn + pw, psw + pse
    ssigpe, ssigpw = pn + pse, pw + psw
    ssigmn, ssigms = mn + mw, msw + mse
    ssigme, ssigmw = mn + mse, mw + msw
    ssig12n, ssig12s = tn + tw, tsw + tse
    ssig12e, ssig12w = tn + tse, tw + tsw
    ssigp1, ssigp2 = (pn + psw) * p055, (pw + pse) * p055
    ssigm1, ssigm2 = (mn + msw) * p055, (mw + mse) * p055
    ssig121, ssig122 = (tn + tsw) * p111, (tw + tse) * p111
    csigpne = p111 * pn + ssigp2 + p027 * psw
    csigpnw = p111 * pw + ssigp1 + p027 * pse
    csigpsw = p111 * psw + ssigp2 + p027 * pn
    csigpse = p111 * pse + ssigp1 + p027 * pw
    csigmne = p111 * mn + ssigm2 + p027 * msw
    csigmnw = p111 * mw + ssigm1 + p027 * mse
    csigmsw = p111 * msw + ssigm2 + p027 * mn
    csigmse = p111 * mse + ssigm1 + p027 * mw
    csig12ne = p222 * tn + ssig122 + p055 * tsw
    csig12nw = p222 * tw + ssig121 + p055 * tse
    csig12sw = p222 * tsw + ssig122 + p055 * tn
    csig12se = p222 * tse + ssig121 + p055 * tw
    str12ew = 0.5 * grid["dxT"] * (p333 * ssig12e + p166 * ssig12w)
    str12we = 0.5 * grid["dxT"] * (p333 * ssig12w + p166 * ssig12e)
    str12ns = 0.5 * grid["dyT"] * (p333 * ssig12n + p166 * ssig12s)
    str12sn = 0.5 * grid["dyT"] * (p333 * ssig12s + p166 * ssig12n)
    dxhy, dyhx = shared["dxhy"], shared["dyhx"]
    xp = 0.25 * grid["dyT"] * (p333 * ssigpn + p166 * ssigps)
    xm = 0.25 * grid["dyT"] * (p333 * ssigmn + p166 * ssigms)
    str1 = -xp - xm - str12ew + dxhy * (-csigpne + csigmne) + dyhx * csig12ne
    str2 = xp + xm - str12we + dxhy * (-csigpnw + csigmnw) + dyhx * csig12nw
    xp = 0.25 * grid["dyT"] * (p333 * ssigps + p166 * ssigpn)
    xm = 0.25 * grid["dyT"] * (p333 * ssigms + p166 * ssigmn)
    str3 = -xp - xm + str12ew + dxhy * (-csigpse + csigmse) + dyhx * csig12se
    str4 = xp + xm + str12we + dxhy * (-csigpsw + csigmsw) + dyhx * csig12sw
    xp = 0.25 * grid["dxT"] * (p333 * ssigpe + p166 * ssigpw)
    xm = 0.25 * grid["dxT"] * (p333 * ssigme + p166 * ssigmw)
    str5 = -xp + xm - str12ns - dyhx * (csigpne + csigmne) + dxhy * csig12ne
    str6 = xp - xm - str12sn - dyhx * (csigpse + csigmse) + dxhy * csig12se
    xp = 0.25 * grid["dxT"] * (p333 * ssigpw + p166 * ssigpe)
    xm = 0.25 * grid["dxT"] * (p333 * ssigmw + p166 * ssigme)
    str7 = -xp + xm + str12ns - dyhx * (csigpnw + csigmnw) + dxhy * csig12nw
    str8 = xp - xm + str12sn - dyhx * (csigpsw + csigmsw) + dxhy * csig12sw
    # These eight combinations are the str(:,:,1:8) inputs to Fortran stepu.
    stress_terms = jnp.stack((str1, str2, str3, str4,
                              str5, str6, str7, str8), axis=-1)
    return jnp.where(iceTmask[..., None], stress_terms, 0.0)


@partial(jax.jit, static_argnames=("ndte", "halo_config"))
def evp_subcycles(fields, strength, cdn_ocnU, grid, shared, params,
                  dyn_params, iglob, ndte, halo_config):
    """Run EVP substeps with a stable compiled function across time steps."""
    bounds, ghost, nx, ew, ns = halo_config
    halo_static = {
        "ice_domain_size": {
            "physical_bounds": bounds, "nghost": ghost, "nx_global": nx},
        "ice_domain": {"ew_boundary_type": ew, "ns_boundary_type": ns},
    }

    def momentum_substep(_, current):
        # Fortran standard_2d loop: stress -> stepu -> velocity halo update.
        stress_fields, stress_terms = stress(
            current["uvel"], current["vvel"], strength, current,
            grid, shared, shared["iceTmask"], dyn_params)
        updated = ice_dyn_shared.stepu(
            stress_fields, stress_terms, cdn_ocnU,
            grid["uarear"], current["iceXmask"], params, dyn_params)
        for name in ("uvel", "vvel"):
            updated[name] = ice_halo_update_jax(
                updated[name], "U", "vector", halo_static, iglob)
        return updated

    return jax.lax.fori_loop(0, ndte, momentum_substep, fields, unroll=4)


def evp(pytree, static, dt, solver="evp"):
    """EVP dynamics, with grid dispatch inside the Fortran-style entry point."""
    grid_ice = static["ice_init"]["grid_ice"].strip()
    if grid_ice in ("C", "CD"):
        if solver != "evp" or static["ice_dyn_shared"]["evp_algorithm"].strip() != "standard_2d":
            raise ValueError("C/CD grids require standard_2d EVP")
        from cicecore.cicedyn.dynamics import ice_dyn_evp_cgrid
        return ice_dyn_evp_cgrid.evp(pytree, static, dt)
    if grid_ice != "B":
        raise ValueError("unknown grid_ice: " + grid_ice)
    algorithm = static["ice_dyn_shared"]["evp_algorithm"].strip()
    if algorithm not in ("standard_2d", "shared_mem_1d"):
        raise ValueError("unknown evp_algorithm: " + algorithm)
    if solver == "evp":
        config = tuple((name, tuple(sorted(static[name].items()))) for name in (
            "ice_init", "ice_domain_size", "ice_domain",
            "ice_dyn_shared", "icepack_parameters"))
        updates = _evp_bgrid_compiled(pytree, config, dt)
        state = dict(pytree["state"])
        for name, changes in updates.items():
            state[name] = {**state[name], **changes}
        result = dict(pytree)
        result["state"] = state
        return result
    return _evp_bgrid(pytree, static, dt, solver)


@partial(jax.jit, static_argnames=("config",))
def _evp_bgrid_compiled(pytree, config, dt):
    # Compile preparation, subcycling and writeback together. Calendar values
    # are excluded from the cache key and dt stays a dynamic input.
    static = {name: dict(values) for name, values in config}
    result = _evp_bgrid(pytree, static, dt)
    # Compare JAX tracer identities while tracing to return reassigned fields.
    updates = {}
    for group, values in result["state"].items():
        changes = {name: value for name, value in values.items()
                   if value is not pytree["state"][group].get(name) and value.size}
        if changes:
            updates[group] = changes
    return updates


def _evp_bgrid(pytree, static, dt, solver="evp"):
    algorithm = static["ice_dyn_shared"]["evp_algorithm"].strip()
    state = dict(pytree["state"])
    ice = dict(state["ice_state"])
    flux = dict(state["ice_flux"])
    shared = dict(state["ice_dyn_shared"])
    evp_state = dict(state["ice_dyn_evp"])
    # EAP and VP use local work arrays in Fortran, with the same preparation.
    if solver != "evp":
        for name in ("tmass", "umass", "umassdti", "cdn_ocnU", "uocnU", "vocnU",
                     "ss_tltxU", "ss_tltyU", "waterxU", "wateryU", "forcexU", "forceyU"):
            evp_state[name] = jnp.zeros_like(ice["uvel"])
    grid = pytree["constants"]["ice_grid"]
    params = pytree["parameters"]["icepack_parameters"]
    dyn_params = pytree["parameters"]["ice_dyn_shared"]
    bounds = static["ice_domain_size"]["physical_bounds"]
    iglob = pytree["constants"]["ice_boundary"]["iglob"]
    grid_names = static["ice_init"]
    options = static["icepack_parameters"]
    flux["rdg_conv"] = jnp.zeros_like(flux["rdg_conv"])
    flux["rdg_shear"] = jnp.zeros_like(flux["rdg_shear"])
    for name in ("divu", "shear", "vort"):
        ice[name] = jnp.zeros_like(ice[name])
    # Fortran evp: dyn_prep1 forms T-cell mass and the active-ice mask.
    evp_state["tmass"], shared["iceTmask"] = ice_dyn_shared.dyn_prep1(
        ice["aice"], ice["vice"], ice["vsno"], grid["tmask"],
        bounds, params, dyn_params)
    shared["iceTmask"] = ice_halo_update_center_jax(
        shared["iceTmask"], static, iglob)
    for name in ("tmass",):
        evp_state[name] = ice_halo_update_center_jax(
            evp_state[name], static, iglob)
    for name in ("aice_init",):
        ice[name] = ice_halo_update_center_jax(ice[name], static, iglob)
    column = dict(state["ice_arrays_column"])
    cdn_ocn = ice_halo_update_center_jax(column["Cdn_ocn"], static, iglob)
    column["Cdn_ocn"] = cdn_ocn
    for name in ("uocn", "vocn", "ss_tltx", "ss_tlty"):
        flux[name] = ice_halo_update_jax(
            flux[name], "T", "vector", static, iglob)
    # Move mass, concentration, ocean velocity, and sea-surface tilt to U
    # points before the B-grid momentum solve.
    for source, target, location in (
        (evp_state["tmass"], "umass", "T"),
        (ice["aice_init"], "aiU", "T"),
        (cdn_ocn, "cdn_ocnU", "T"),
        (flux["uocn"], "uocnU", grid_names["grid_ocn_dynu"]),
        (flux["vocn"], "vocnU", grid_names["grid_ocn_dynv"]),
        (flux["ss_tltx"], "ss_tltxU", grid_names["grid_ocn_dynu"]),
        (flux["ss_tlty"], "ss_tltyU", grid_names["grid_ocn_dynv"]),
    ):
        if target == "aiU":
            ice[target] = grid_average_X2Y_jax(
                "S", source, location, ice[target], "U", grid, bounds)
        else:
            evp_state[target] = grid_average_X2Y_jax(
                "S", source, location, evp_state[target], "U", grid, bounds)
    # Fortran evp: map supplied wind stress or computed T-grid stress to U.
    if not options["calc_strair"]:
        windx, windy = flux["strax"], flux["stray"]
        wind_gridx = grid_names["grid_atm_dynu"]
        wind_gridy = grid_names["grid_atm_dynv"]
    else:
        windx = ice_halo_update_jax(
            flux["strairxT"], "T", "vector", static, iglob)
        windy = ice_halo_update_jax(
            flux["strairyT"], "T", "vector", static, iglob)
        wind_gridx = wind_gridy = "T"
    flux["strairxU"] = grid_average_X2Y_jax(
        "F", windx, wind_gridx, flux["strairxU"], "U", grid, bounds)
    flux["strairyU"] = grid_average_X2Y_jax(
        "F", windy, wind_gridy, flux["strairyU"], "U", grid, bounds)

    fields = {
        "aiX": ice["aiU"], "Xmass": evp_state["umass"],
        "Xmassdti": evp_state["umassdti"], "fcor": shared["fcor_blk"],
        "uocn": evp_state["uocnU"], "vocn": evp_state["vocnU"],
        "strairx": flux["strairxU"], "strairy": flux["strairyU"],
        "ss_tltx": evp_state["ss_tltxU"],
        "ss_tlty": evp_state["ss_tltyU"],
        "iceXmask": shared["iceUmask"], "fm": flux["fmU"],
        "uvel": ice["uvel"], "vvel": ice["vvel"],
        "uvel_init": shared["uvel_init"],
        "vvel_init": shared["vvel_init"],
        "waterx": evp_state["waterxU"],
        "watery": evp_state["wateryU"],
        "forcex": evp_state["forcexU"],
        "forcey": evp_state["forceyU"],
        "strtltx": flux["strtltxU"], "strtlty": flux["strtltyU"],
        "strocnx": flux["strocnxU"], "strocny": flux["strocnyU"],
        "strintx": flux["strintxU"], "strinty": flux["strintyU"],
        "taubx": flux["taubxU"], "tauby": flux["taubyU"],
        "TbU": flux["TbU"],
    }
    for stem in ("stressp", "stressm", "stress12"):
        for corner in range(1, 5):
            name = f"{stem}_{corner}"
            fields[name] = flux[name]
    # dyn_prep2 selects active U points and can reset their velocities.
    fields = ice_dyn_shared.dyn_prep2(
        fields, shared["iceTmask"], grid["umask"], bounds, dt,
        params, dyn_params, static["ice_dyn_shared"])
    # Fortran's indxTi/indxTj include only physical, north and east cells.
    ilo, ihi, jlo, jhi = bounds
    iceTmask = jnp.zeros_like(shared["iceTmask"]).at[
        ilo:ihi + 2, jlo:jhi + 2].set(
            shared["iceTmask"][ilo:ihi + 2, jlo:jhi + 2])
    # Ice strength is computed at active T points before stress subcycling.
    ice["strength"] = jnp.where(
        iceTmask,
        icepack_ice_strength(
            ice["aice"], ice["vice"], ice["aice0"],
            ice["aicen"], ice["vicen"], options, params), 0.0)
    ice["strength"] = ice_halo_update_center_jax(
        ice["strength"], static, iglob)
    # Fortran refreshes velocity ghosts here because dyn_prep2 can change them.
    for name in ("uvel", "vvel"):
        fields[name] = ice_halo_update_jax(
            fields[name], "U", "vector", static, iglob)
    if static["ice_dyn_shared"]["seabed_stress"]:
        method = static["ice_dyn_shared"]["seabed_stress_method"].strip()
        if method == "LKD":
            fields["TbU"] = ice_dyn_shared.seabed_stress_factor_LKD(
                ice["vice"], ice["aice"], flux["hwater"], fields["TbU"],
                fields["iceXmask"], "U", dyn_params)
        elif method == "probabilistic":
            fields["TbU"] = ice_dyn_shared.seabed_stress_factor_prob(
                ice["aicen"], ice["vicen"], flux["hwater"],
                state["ice_arrays_column"]["hin_max"],
                shared["iceTmask"], fields["iceXmask"], "U",
                params, dyn_params)
        else:
            raise ValueError("unknown seabed stress method: " + method)
    fields["Umassdti"] = fields["Xmassdti"]

    halo_config = (
        bounds, int(static["ice_domain_size"]["nghost"]),
        int(static["ice_domain_size"]["nx_global"]),
        static["ice_domain"]["ew_boundary_type"].strip(),
        static["ice_domain"]["ns_boundary_type"].strip(),
    )
    solve_shared = {**shared, "iceTmask": iceTmask, "iceTmask_full": shared["iceTmask"]}
    if solver == "eap":
        from cicecore.cicedyn.dynamics import ice_dyn_eap
        fields, state["ice_dyn_eap"], diagnostics = ice_dyn_eap.eap_subcycles(
            fields, state["ice_dyn_eap"], ice["strength"], evp_state["cdn_ocnU"],
            grid, solve_shared, params, dyn_params, pytree["parameters"]["ice_dyn_eap"],
            static, iglob)
    elif solver == "vp":
        from cicecore.cicedyn.dynamics import ice_dyn_vp
        fields, state["ice_dyn_vp"] = ice_dyn_vp.vp_solve(
            fields, state["ice_dyn_vp"], ice["strength"], evp_state["cdn_ocnU"],
            grid, solve_shared, params, dyn_params, pytree["parameters"]["ice_dyn_vp"],
            static, iglob)
    elif algorithm == "shared_mem_1d":
        from cicecore.cicedyn.dynamics.ice_dyn_evp1d import dyn_evp1d_run
        fields = dyn_evp1d_run(fields, ice["strength"], evp_state["cdn_ocnU"],
            {n: grid[n] for n in ("dxT", "dyT", "HTN", "HTE", "uarear")},
            {"iceTmask": iceTmask}, {"rhow": params["rhow"]}, dyn_params,
            iglob, int(static["ice_dyn_shared"]["ndte"]), halo_config)
    else:
        fields = evp_subcycles(
            fields, ice["strength"], evp_state["cdn_ocnU"],
            {name: grid[name] for name in ("dxT", "dyT", "uarear")},
            {name: solve_shared[name] for name in (
                "iceTmask", "cxp", "cyp", "cxm", "cym", "DminTarea", "dxhy", "dyhx")},
            {"rhow": params["rhow"]}, dyn_params, iglob,
            int(static["ice_dyn_shared"]["ndte"]), halo_config)
    # Deformations feed ridging; dyn_finish computes ocean stress from the
    # final ice velocity.
    if solver == "eap":
        for name in ("vort", "shear", "divu"):
            ice[name] = diagnostics[name]
        flux["rdg_conv"] = diagnostics["rdg_conv"]
    else:
        (ice["vort"], ice["shear"], ice["divu"],
         flux["rdg_conv"], flux["rdg_shear"]) = ice_dyn_shared.deformations(
             fields["uvel"], fields["vvel"], grid, shared,
             iceTmask, dyn_params)
    if solver != "eap" and static["ice_domain"]["ns_boundary_type"].strip() in ("tripole", "tripoleT"):
        # Keep Fortran's sequential paired copies, including the T-fold row.
        for stem in ("stressp", "stressm", "stress12"):
            for destination, source in ((1, 3), (3, 1), (2, 4), (4, 2)):
                name = f"{stem}_{destination}"
                fields[name] = ice_halo_update_jax(
                    fields[name], "T", "scalar", static, iglob,
                    tripole_only=True, source=fields[f"{stem}_{source}"],
                    seam_average=False)
    fields["strocnx"], fields["strocny"] = ice_dyn_shared.dyn_finish(
        fields["uvel"], fields["vvel"], fields["uocn"], fields["vocn"],
        fields["aiX"], fields["fm"], evp_state["cdn_ocnU"],
        fields["strocnx"], fields["strocny"], fields["iceXmask"], params)
    ice["uvel"], ice["vvel"] = fields["uvel"], fields["vvel"]
    shared["iceUmask"] = fields["iceXmask"]
    shared["uvel_init"] = fields["uvel_init"]
    shared["vvel_init"] = fields["vvel_init"]
    evp_state["umassdti"] = fields["Xmassdti"]
    for name, target in (("waterx", "waterxU"), ("watery", "wateryU"),
                         ("forcex", "forcexU"), ("forcey", "forceyU")):
        evp_state[target] = fields[name]
    for name, target in (("fm", "fmU"), ("TbU", "TbU"),
                         ("strtltx", "strtltxU"), ("strtlty", "strtltyU"),
                         ("strocnx", "strocnxU"), ("strocny", "strocnyU"),
                         ("strintx", "strintxU"), ("strinty", "strintyU"),
                         ("taubx", "taubxU"), ("tauby", "taubyU")):
        flux[target] = fields[name]
    for stem in ("stressp", "stressm", "stress12"):
        for corner in range(1, 5):
            name = f"{stem}_{corner}"
            flux[name] = fields[name]
    state.update({"ice_state": ice, "ice_flux": flux,
                  "ice_arrays_column": column,
                  "ice_dyn_shared": shared})
    if solver == "evp":
        state["ice_dyn_evp"] = evp_state
    result = dict(pytree)
    result["state"] = state
    return result

# all c or cd
uocnN = None     # i ocean current (m/s)
vocnN = None     # j ocean current (m/s)
ss_tltxN = None  # sea surface slope, x-direction (m/m)
ss_tltyN = None  # sea surface slope, y-direction (m/m)
cdn_ocnN = None  # ocn drag coefficient
waterxN = None   # for ocean stress calculation, x (m/s)
wateryN = None   # for ocean stress calculation, y (m/s)
forcexN = None   # work array: combined atm stress and ocn tilt, x
forceyN = None   # work array: combined atm stress and ocn tilt, y
aiN = None       # ice fraction on N-grid
nmass = None     # total mass of ice and snow (N grid)
nmassdti = None  # mass of N-cell/dte (kg/m^2 s)

# all c or d
uocnE = None     # i ocean current (m/s)
vocnE = None     # j ocean current (m/s)
ss_tltxE = None  # sea surface slope, x-direction (m/m)
ss_tltyE = None  # sea surface slope, y-direction (m/m)
cdn_ocnE = None  # ocn drag coefficient
waterxE = None   # for ocean stress calculation, x (m/s)
wateryE = None   # for ocean stress calculation, y (m/s)
forcexE = None   # work array: combined atm stress and ocn tilt, x
forceyE = None   # work array: combined atm stress and ocn tilt, y
aiE = None       # ice fraction on E-grid
emass = None     # total mass of ice and snow (E grid)
emassdti = None  # mass of E-cell/dte (kg/m^2 s)

ratiodxN = None   # - dxN(i+1,j)   / dxN(i,j)
ratiodyE = None   # - dyE(i  ,j+1) / dyE(i,j)
ratiodxNr = None  #   1 / ratiodxN
ratiodyEr = None  #   1 / ratiodyE

strengthU = None  # strength averaged to U points
divergU = None    # div array on U points, differentiate from divu
tensionU = None   # tension array on U points
shearU = None     # shear array on U points
deltaU = None     # delta array on U points
zetax2T = None    # zetax2 = 2*zeta (bulk viscosity)
zetax2U = None    # zetax2T averaged to U points
etax2T = None     # etax2  = 2*eta  (shear viscosity)
etax2U = None     # etax2T averaged to U points

uocnU = None     # i ocean current (m/s)
vocnU = None     # j ocean current (m/s)
ss_tltxU = None  # sea surface slope, x-direction (m/m)
ss_tltyU = None  # sea surface slope, y-direction (m/m)
cdn_ocnU = None  # ocn drag coefficient
tmass = None     # total mass of ice and snow (kg/m^2)
waterxU = None   # for ocean stress calculation, x (m/s)
wateryU = None   # for ocean stress calculation, y (m/s)
forcexU = None   # work array: combined atm stress and ocn tilt, x
forceyU = None   # work array: combined atm stress and ocn tilt, y
umass = None     # total mass of ice and snow (u grid)
umassdti = None  # mass of U-cell/dte (kg/m^2 s)


def _zero(shape):
    return np.zeros(shape, dtype=dbl_kind)


def _bundle(names):
    horiz = ice_domain_size.horizontal_shape()
    allocated = {}
    for name in names:
        allocated[name] = _zero(horiz)
    return allocated


def init_evp():
    """Elastic-viscous-plastic dynamics driver."""
    global uocnU, vocnU, ss_tltxU, ss_tltyU, cdn_ocnU, tmass
    global waterxU, wateryU, forcexU, forceyU, umass, umassdti
    global strengthU, divergU, tensionU, shearU, deltaU
    global zetax2T, zetax2U, etax2T, etax2U
    global uocnN, vocnN, ss_tltxN, ss_tltyN, cdn_ocnN
    global waterxN, wateryN, forcexN, forceyN, aiN, nmass, nmassdti
    global uocnE, vocnE, ss_tltxE, ss_tltyE, cdn_ocnE
    global waterxE, wateryE, forcexE, forceyE, aiE, emass, emassdti
    global ratiodxN, ratiodyE, ratiodxNr, ratiodyEr

    subname = "(init_evp)"

    ice_dyn_shared.init_dyn_shared(ice_calendar.dt_dyn)

    # allocate c and cd grid var. Follow structucre of eap
    allocated = _bundle((
        "uocnU",     # i ocean current (m/s)
        "vocnU",     # j ocean current (m/s)
        "ss_tltxU",  # sea surface slope, x-direction (m/m)
        "ss_tltyU",  # sea surface slope, y-direction (m/m)
        "cdn_ocnU",  # ocn drag coefficient
        "tmass",     # total mass of ice and snow (kg/m^2)
        "waterxU",   # for ocean stress calculation, x (m/s)
        "wateryU",   # for ocean stress calculation, y (m/s)
        "forcexU",   # work array: combined atm stress and ocn tilt, x
        "forceyU",   # work array: combined atm stress and ocn tilt, y
        "umass",     # total mass of ice and snow (u grid)
        "umassdti",  # mass of U-cell/dte (kg/m^2 s)
    ))
    uocnU = allocated["uocnU"]
    vocnU = allocated["vocnU"]
    ss_tltxU = allocated["ss_tltxU"]
    ss_tltyU = allocated["ss_tltyU"]
    cdn_ocnU = allocated["cdn_ocnU"]
    tmass = allocated["tmass"]
    waterxU = allocated["waterxU"]
    wateryU = allocated["wateryU"]
    forcexU = allocated["forcexU"]
    forceyU = allocated["forceyU"]
    umass = allocated["umass"]
    umassdti = allocated["umassdti"]

    grid = str(ice_dyn_shared.grid_ice).strip()
    if grid in ("CD", "C"):
        allocated = _bundle((
            "strengthU", "divergU", "tensionU", "shearU", "deltaU",
            "zetax2T", "zetax2U", "etax2T", "etax2U",
        ))
        strengthU = allocated["strengthU"]
        divergU = allocated["divergU"]
        tensionU = allocated["tensionU"]
        shearU = allocated["shearU"]
        deltaU = allocated["deltaU"]
        zetax2T = allocated["zetax2T"]
        zetax2U = allocated["zetax2U"]
        etax2T = allocated["etax2T"]
        etax2U = allocated["etax2U"]

        allocated = _bundle((
            "uocnN", "vocnN", "ss_tltxN", "ss_tltyN", "cdn_ocnN",
            "waterxN", "wateryN", "forcexN", "forceyN", "aiN", "nmass", "nmassdti",
        ))
        uocnN = allocated["uocnN"]
        vocnN = allocated["vocnN"]
        ss_tltxN = allocated["ss_tltxN"]
        ss_tltyN = allocated["ss_tltyN"]
        cdn_ocnN = allocated["cdn_ocnN"]
        waterxN = allocated["waterxN"]
        wateryN = allocated["wateryN"]
        forcexN = allocated["forcexN"]
        forceyN = allocated["forceyN"]
        aiN = allocated["aiN"]
        nmass = allocated["nmass"]
        nmassdti = allocated["nmassdti"]

        allocated = _bundle((
            "uocnE", "vocnE", "ss_tltxE", "ss_tltyE", "cdn_ocnE",
            "waterxE", "wateryE", "forcexE", "forceyE", "aiE", "emass", "emassdti",
        ))
        uocnE = allocated["uocnE"]
        vocnE = allocated["vocnE"]
        ss_tltxE = allocated["ss_tltxE"]
        ss_tltyE = allocated["ss_tltyE"]
        cdn_ocnE = allocated["cdn_ocnE"]
        waterxE = allocated["waterxE"]
        wateryE = allocated["wateryE"]
        forcexE = allocated["forcexE"]
        forceyE = allocated["forceyE"]
        aiE = allocated["aiE"]
        emass = allocated["emass"]
        emassdti = allocated["emassdti"]

        allocated = _bundle(("ratiodxN", "ratiodyE", "ratiodxNr", "ratiodyEr"))
        ratiodxN = allocated["ratiodxN"]
        ratiodyE = allocated["ratiodyE"]
        ratiodxNr = allocated["ratiodxNr"]
        ratiodyEr = allocated["ratiodyEr"]

        ilo, ihi, jlo, jhi = ice_domain_size.physical_bounds()
        ip = slice(ilo, ihi + 1)
        jp = slice(jlo, jhi + 1)
        ratiodxN[ip, jp] = -ice_grid.dxN[ilo + 1:ihi + 2, jp] / ice_grid.dxN[ip, jp]
        ratiodyE[ip, jp] = -ice_grid.dyE[ip, jlo + 1:jhi + 2] / ice_grid.dyE[ip, jp]
        ratiodxNr[ip, jp] = c1 / ratiodxN[ip, jp]
        ratiodyEr[ip, jp] = c1 / ratiodyE[ip, jp]
