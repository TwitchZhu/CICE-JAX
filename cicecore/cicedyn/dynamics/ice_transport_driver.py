"""Drivers for remapping and upwind ice transport.

authors: Elizabeth C. Hunke and William H. Lipscomb, LANL

author for JAX version: Xinyuan Zhu
XZ: converted from F90 to Python

2004: Revised by William Lipscomb from ice_transport_mpdata.
      Stripped out mpdata, retained upwind, and added block structure.
2006: Incorporated remap transport driver and renamed from
      ice_transport_upwind.
2011: ECH moved edgearea arrays into ice_transport_remap.F90

``init_transport`` is the initialization step. ``transport_remap`` and
``transport_upwind`` are the time loop. One grid, no timers.

NOTE: For remapping, hice and hsno are considered tracers.
      ntrace is not equal to ntrcr!
"""

from functools import partial

import numpy as np
import jax
import jax.numpy as jnp

from cicecore.cicedyn.dynamics import ice_transport_remap
from cicecore.cicedyn.general import ice_init
from cicecore.cicedyn.general import ice_state
from cicecore.shared import ice_fileunits
from cicecore.shared.ice_kinds_mod import int_kind
from icepack.columnphysics import icepack_tracers
from cicecore.cicedyn.infrastructure.comm.serial.ice_boundary import (
    ice_halo_update_center_jax, ice_halo_update_jax,
)

# type of advection scheme used
# 'upwind' => 1st order donor cell scheme
# 'remap' => remapping scheme
advection = "remap"

# number of tracers in use
ntrace = int_kind(0)

# = 1, 2, or 3 (depends on 0, 1 or 2 other tracers)
tracer_type = None
# tracer dependencies (see below)
depend = None

# true if a tracer has dependent tracers
has_dependents = None

# if true, check conservation
conserv_check = False

# polynomial order of quadrature integrals
# linear=1, quadratic=2, cubic=3
integral_order = 3

# if true, find departure points using
# corrected midpoint velocity
l_dp_midpt = True

_NAMES = (
    (1, "nt_Tsfc", "nt_Tsfc     "),
    (1, "nt_qice", "nt_qice     "),
    (1, "nt_qsno", "nt_qsno     "),
    (1, "nt_sice", "nt_sice     "),
    (1, "nt_fbri", "nt_fbri     "),
    (1, "nt_iage", "nt_iage     "),
    (1, "nt_FY", "nt_FY       "),
    (1, "nt_alvl", "nt_alvl     "),
    (1, "nt_vlvl", "nt_vlvl     "),
    (1, "nt_apnd", "nt_apnd     "),
    (1, "nt_hpnd", "nt_hpnd     "),
    (1, "nt_ipnd", "nt_ipnd     "),
    (1, "nt_smice", "nt_smice    "),
    (1, "nt_smliq", "nt_smliq    "),
    (1, "nt_rhos", "nt_rhos     "),
    (1, "nt_rsnw", "nt_rsnw     "),
    (1, "nt_fsd", "nt_fsd      "),
    (1, "nt_isosno", "nt_isosno   "),
    (1, "nt_isoice", "nt_isoice   "),
)


def init_transport():
    """Wrapper for init_remap. Upwind transport needs no grid init.

    authors William H. Lipscomb, LANL

    ``depend`` and ``tracer_type`` use Fortran 1-based transport positions.
    hi is 1 and hs is 2 in the diagnostic. A dependency value of 0, 1, or 2
    is the remap-tracer index of hi/hs (or none). A value above 2 is
    ``3 +`` the parent tracer's 0-based index, equal to Fortran's
    ``2 +`` the 1-based index. The printed transport index stays 1-based.
    """
    global ntrace, tracer_type, depend, has_dependents, conserv_check, advection
    subname = "(init_transport)"
    # call ice_timer_start(timer_advect)  ! advection
    # Timers are not ported.

    ntrcr = int(icepack_tracers.ntrcr)
    if ntrcr < 1:
        raise RuntimeError(subname + " ntrcr is not set")

    advection = str(ice_init.advection).strip()
    conserv_check = bool(ice_init.conserv_check)

    ntrace = int_kind(2 + ntrcr)  # hice,hsno,trcr

    count = int(ntrace)
    tracer_type = np.zeros(count, dtype=int_kind)
    depend = np.zeros(count, dtype=int_kind)
    has_dependents = np.zeros(count, dtype=bool)

    # define tracer dependency arrays
    # see comments in remapping routine

    depend[0:2] = 0  # hice, hsno
    tracer_type[0:2] = 1  # no dependency

    k = 2
    trcr_depend = ice_state.trcr_depend
    for nt in range(1, ntrcr + 1):
        parent = int(trcr_depend[nt - 1])
        depend[k + nt - 1] = parent  # 0 for ice area tracers
        # 1 for ice volume tracers
        # 2 for snow volume tracers
        tracer_type[k + nt - 1] = 2  # depends on 1 other tracer
        if parent == 0:
            tracer_type[k + nt - 1] = 1  # depends on no other tracers
        elif parent > 2:
            # trcr_depend(trcr_depend(nt)-2): parent index in the tracer array
            # parent - 3 is that parent's 0-based slot. The stored code is
            # 3 + the slot, equal to Fortran's 2 + the 1-based index.
            if int(trcr_depend[parent - 2 - 1]) > 0:
                tracer_type[k + nt - 1] = 3  # depends on 2 other tracers

    has_dependents[:] = False
    for nt in range(1, count + 1):
        if int(depend[nt - 1]) > 0:
            nt1 = int(depend[nt - 1])
            has_dependents[nt1 - 1] = True
            if nt1 > nt:
                ice_fileunits.nu_diag.write(
                    f"Tracer nt2 = {nt} depends on tracer nt1 = {nt1}\n"
                )
                raise RuntimeError(
                    subname + "ERROR: remap transport: Must have nt2 > nt1"
                )

    # diagnostic output
    # One process, so the master-task write always runs.
    nu = ice_fileunits.nu_diag
    nu.write(" tracer          index  depend  type has_dependents\n")
    _write_row(nu, "hi          ", 1)
    _write_row(nu, "hs          ", 2)
    for nt in range(k + 1, k + ntrcr + 1):
        trcr_index = nt - k - 1
        for _stride, attr, label in _NAMES:
            if trcr_index == int(getattr(icepack_tracers, attr)):
                _write_row(nu, label, nt)
        if hasattr(icepack_tracers, "nt_bgc_Nit"):
            if trcr_index == int(icepack_tracers.nt_bgc_Nit):
                _write_row(nu, "nt_bgc_Nit  ", nt)
    nu.write(" \n")
    nu.flush()

    if advection.strip() == "remap":
        ice_transport_remap.init_remap()  # grid quantities

    # call ice_timer_stop(timer_advect)  ! advection


def _write_row(nu, name, nt):
    flag = "T" if bool(has_dependents[nt - 1]) else "F"
    nu.write(
        f" {name}  {nt:6d}  {int(depend[nt - 1]):6d}  "
        f"{int(tracer_type[nt - 1]):4d}    {flag:>4}\n"
    )


def upwind_field_jax(phi, uee, vnn, HTE, HTN, tarea, dt, bounds):
    """Transport independent scalar fields with Fortran's donor-cell flux."""
    ilo, ihi, jlo, jhi = bounds
    east = jnp.concatenate((phi[1:, :, :], phi[-1:, :, :]), axis=0)
    north = jnp.concatenate((phi[:, 1:, :], phi[:, -1:, :]), axis=1)
    u = uee[..., None]
    v = vnn[..., None]
    worka = 0.5 * dt * HTE[..., None] * (
        (u + jnp.abs(u)) * phi + (u - jnp.abs(u)) * east)
    workb = 0.5 * dt * HTN[..., None] * (
        (v + jnp.abs(v)) * phi + (v - jnp.abs(v)) * north)
    west = jnp.pad(worka[:-1, :, :], ((1, 0), (0, 0), (0, 0)))
    south = jnp.pad(workb[:, :-1, :], ((0, 0), (1, 0), (0, 0)))
    physical = jnp.zeros(phi.shape[:2], dtype=bool).at[
        ilo:ihi + 1, jlo:jhi + 1].set(True)
    safe_area = jnp.where(physical, tarea, 1.0)
    updated = phi - (worka - west + workb - south) / safe_area[..., None]
    return jnp.where(physical[..., None], updated, phi)


def state_to_work_jax(aice0, aicen, vicen, vsnon, trcrn, layout):
    """Pack category volume-weighted tracers for horizontal advection."""
    work = [aice0]
    base = layout["trcr_base"]
    strata_count = layout["n_trcr_strata"]
    strata = layout["nt_strata"]
    for n in range(aicen.shape[-1]):
        area = aicen[..., n]
        ice_volume = vicen[..., n]
        snow_volume = vsnon[..., n]
        work.extend((area, ice_volume, snow_volume))
        for it in range(trcrn.shape[-2]):
            amount = trcrn[..., it, n] * (
                base[it][0] * area + base[it][1] * ice_volume
                + base[it][2] * snow_volume)
            for layer in range(strata_count[it]):
                amount = amount * trcrn[..., strata[it][layer], n]
            work.append(amount)
    return jnp.stack(work, axis=-1)


def work_to_state_jax(work, tmask, Tf, layout, flags,
                      options, parameters):
    """Recover category tracers from the advected extensive quantities."""
    ntrcr = len(layout["trcr_depend"])
    ncat = (work.shape[-1] - 1) // (ntrcr + 3)
    base = layout["trcr_base"]
    strata_count = layout["n_trcr_strata"]
    strata = layout["nt_strata"]
    puny = parameters["puny"]
    aice0 = work[..., 0]
    area_columns = []
    ice_columns = []
    snow_columns = []
    tracer_columns = []
    for n in range(ncat):
        start = 1 + n * (ntrcr + 3)
        area = work[..., start]
        ice_volume = work[..., start + 1]
        snow_volume = work[..., start + 2]
        trcrn = jnp.zeros(work.shape[:2] + (ntrcr,), dtype=work.dtype)
        for it in range(ntrcr):
            amount = work[..., start + 3 + it]
            if layout["trcr_depend"][it] == 0:
                value = jnp.where(area > puny,
                                  amount / jnp.where(area > puny, area, 1.0),
                                  0.0)
                if it == int(flags["nt_Tsfc"]):
                    value = jnp.where(area > puny, value, Tf)
            else:
                # CICE tests the three possible carriers in order. Each
                # positive carrier replaces the previous quotient.
                value = jnp.zeros_like(amount)
                for carrier in (base[it][0] * area,
                                base[it][1] * ice_volume,
                                base[it][2] * snow_volume):
                    value = jnp.where(carrier > 0.0,
                                      amount / jnp.where(carrier > 0.0,
                                                         carrier, 1.0), value)
                for layer in range(strata_count[it]):
                    parent = trcrn[..., strata[it][layer]]
                    value = jnp.where(parent > 0.0, value / jnp.where(
                        parent > 0.0, parent, 1.0), 0.0)
            trcrn = trcrn.at[..., it].set(value)
        if flags["tr_brine"]:
            slot = int(flags["nt_fbri"])
            trcrn = trcrn.at[..., slot].set(jnp.where(
                ice_volume <= 0.0, 1.0, trcrn[..., slot]))
        if str(options["snwredist"]).startswith("ITD"):
            for k in range(int(options["nslyr"])):
                slot = int(flags["nt_rhos"]) + k
                trcrn = trcrn.at[..., slot].set(jnp.where(
                    snow_volume <= 0.0, parameters["rhosnew"],
                    trcrn[..., slot]))
        if options["snwgrain"]:
            for k in range(int(options["nslyr"])):
                for slot, default in (
                    (int(flags["nt_rsnw"]) + k, parameters["rsnw_fall"]),
                    (int(flags["nt_smice"]) + k, parameters["rhos"]),
                ):
                    trcrn = trcrn.at[..., slot].set(jnp.where(
                        snow_volume <= 0.0, default, trcrn[..., slot]))
        surface_slot = int(flags["nt_Tsfc"])
        trcrn = trcrn.at[..., surface_slot].set(jnp.where(
            tmask, trcrn[..., surface_slot], 0.0))
        area_columns.append(area)
        ice_columns.append(ice_volume)
        snow_columns.append(snow_volume)
        tracer_columns.append(trcrn)
    return (aice0, jnp.stack(area_columns, axis=-1),
            jnp.stack(ice_columns, axis=-1),
            jnp.stack(snow_columns, axis=-1),
            jnp.stack(tracer_columns, axis=-1))


def transport_upwind(pytree, static, dt):
    """Advect open water, category ice/snow and all tracers on one grid."""
    state = dict(pytree["state"])
    ice = dict(state["ice_state"])
    grid = pytree["constants"]["ice_grid"]
    flux = state["ice_flux"]
    bounds = static["ice_domain_size"]["physical_bounds"]
    iglob = pytree["constants"]["ice_boundary"]["iglob"]
    if static["ice_init"]["grid_ice"].strip() in ("C", "CD"):
        uee = ice["uvelE"]
        vnn = ice["vvelN"]
    else:
        # B-grid velocities live at NE corners; Fortran averages them to
        # east and north cell faces before forming the donor-cell flux.
        uee = jnp.zeros_like(ice["uvel"])
        vnn = jnp.zeros_like(ice["vvel"])
        ilo, ihi, jlo, jhi = bounds
        uee = uee.at[ilo:ihi + 1, jlo:jhi + 1].set(0.5 * (
            ice["uvel"][ilo:ihi + 1, jlo:jhi + 1]
            + ice["uvel"][ilo:ihi + 1, jlo - 1:jhi]))
        vnn = vnn.at[ilo:ihi + 1, jlo:jhi + 1].set(0.5 * (
            ice["vvel"][ilo:ihi + 1, jlo:jhi + 1]
            + ice["vvel"][ilo - 1:ihi, jlo:jhi + 1]))
        uee = ice_halo_update_jax(uee, "E", "vector", static, iglob)
        vnn = ice_halo_update_jax(vnn, "N", "vector", static, iglob)

    layout = static["ice_state_tracers"]
    flags = static["icepack_tracers"]
    options = {
        "snwredist": static["icepack_parameters"]["snwredist"],
        "snwgrain": static["icepack_parameters"]["snwgrain"],
        "nslyr": static["ice_domain_size"]["nslyr"],
    }
    parameters = pytree["parameters"]["icepack_parameters"]
    work = state_to_work_jax(
        ice["aice0"], ice["aicen"], ice["vicen"], ice["vsnon"],
        ice["trcrn"], layout)
    work = upwind_field_jax(
        work, uee, vnn, grid["HTE"], grid["HTN"], grid["tarea"],
        dt, bounds)
    (ice["aice0"], ice["aicen"], ice["vicen"], ice["vsnon"],
     ice["trcrn"]) = work_to_state_jax(
         work, grid["tmask"], flux["Tf"], layout, flags, options, parameters)
    # bound_state updates these four arrays after the transport, while
    # the open-water fraction keeps the values returned by work_to_state.
    for name in ("aicen", "trcrn", "vicen", "vsnon"):
        ice[name] = ice_halo_update_center_jax(ice[name], static, iglob)
    state["ice_state"] = ice
    result = dict(pytree)
    result["state"] = state
    return result


def transport_remap(pytree, static, dt):
    """Incremental remapping on B/C/CD grids, including dependent tracers."""
    state = dict(pytree["state"])
    ice = dict(state["ice_state"])
    status = dict(state["ice_step_mod"])
    sizes = static["ice_domain_size"]
    domain = static["ice_domain"]
    grid_ice = static["ice_init"]["grid_ice"].strip()
    names = ("aice0", "aicen", "vicen", "vsnon", "trcrn", "uvel", "vvel")
    if grid_ice in ("C", "CD"):
        names += ("uvelE", "vvelN")
    grid = pytree["constants"]["ice_grid"]
    parameters = pytree["parameters"]["icepack_parameters"]
    advanced, failed, conserv_failed = _transport_remap(
        {name: ice[name] for name in names},
        {name: grid[name] for name in (
            "hm", "dxU", "dyU", "HTN", "HTE", "earea", "narea", "tarear", "tarea")},
        {name: parameters[name] for name in ("puny", "rhos", "Lfresh")},
        pytree["constants"]["ice_boundary"]["iglob"], dt,
        (tuple(sizes["physical_bounds"]), int(sizes["nghost"]), int(sizes["nx_global"]),
         domain["ew_boundary_type"], domain["ns_boundary_type"]),
        grid_ice, bool(static["ice_init"]["conserv_check"]),
        int(static["icepack_tracers"]["nt_qsno"]), int(sizes["nslyr"]),
        tuple(int(value) for value in tracer_type),
        tuple(int(value) for value in depend),
        tuple(bool(value) for value in has_dependents))
    ice.update(advanced)
    status["remap_failed"] = status["remap_failed"] | failed
    status["remap_conservation_failed"] = status["remap_conservation_failed"] | conserv_failed
    state["ice_state"] = ice
    state["ice_step_mod"] = status
    result = dict(pytree)
    result["state"] = state
    return result


@partial(jax.jit, static_argnames=(
    "halo_config", "grid_ice", "conserv_check", "nt_qsno", "nslyr",
    "types", "parents", "dependent"))
def _transport_remap(ice, grid, parameters, iglob, dt, halo_config,
                     grid_ice, conserv_check, nt_qsno, nslyr,
                     types, parents, dependent):
    """Compile the complete remap with fixed layout and dynamic model arrays."""
    ice = dict(ice)
    puny = parameters["puny"]
    bounds, nghost, nx_global, ew, ns = halo_config
    static = {
        "ice_domain_size": {"physical_bounds": bounds, "nghost": nghost,
                            "nx_global": nx_global},
        "ice_domain": {"ew_boundary_type": ew, "ns_boundary_type": ns},
    }
    ncat = ice["aicen"].shape[-1]
    ntrcr = ice["trcrn"].shape[-2]

    # state_to_tracers: hi and hs are transport tracers, distinct from trcrn.
    mm = jnp.concatenate((ice["aice0"][..., None], ice["aicen"]), axis=-1)
    snow_offset = jnp.zeros((ntrcr,), dtype=ice["trcrn"].dtype)
    snow_offset = snow_offset.at[nt_qsno:nt_qsno + nslyr].set(
        parameters["rhos"] * parameters["Lfresh"])
    present = ice["aicen"] > puny
    safe_area = jnp.where(present, ice["aicen"], 1.0)
    hi = jnp.where(present, ice["vicen"] / safe_area, 0.0)
    hs = jnp.where(present, ice["vsnon"] / safe_area, 0.0)
    tracers = jnp.where(present[:, :, None, :],
                        ice["trcrn"] + snow_offset[None, None, :, None], 0.0)
    trm = jnp.concatenate((hi[:, :, None, :], hs[:, :, None, :], tracers), axis=2)

    dpx, dpy, departure_failed = ice_transport_remap.departure_points(
        ice["uvel"], ice["vvel"], grid["dxU"], grid["dyU"],
        grid["HTN"], grid["HTE"], dt, bounds, nghost)
    if nghost == 1:
        dpx = ice_halo_update_jax(dpx, "U", "vector", static, iglob)
        dpy = ice_halo_update_jax(dpy, "U", "vector", static, iglob)

    # Categories are independent. Each device tracer loop processes all ice
    # categories together, while open water has no tracer state.
    water_center, water_x, water_y, _, _, _ = ice_transport_remap.construct_fields(
        mm[..., 0], None, grid["hm"], types, parents, dependent, puny, bounds, nghost)
    mc, mx, my, tracer_center, tracer_x, tracer_y = jax.vmap(
        lambda mass, tracers: ice_transport_remap.construct_fields(
            mass, tracers, grid["hm"], types, parents, dependent, puny, bounds, nghost),
        in_axes=(-1, -1), out_axes=-1)(mm[..., 1:], trm)
    mass_center = jnp.concatenate((water_center[..., None], mc), axis=-1)
    mass_x = jnp.concatenate((water_x[..., None], mx), axis=-1)
    mass_y = jnp.concatenate((water_y[..., None], my), axis=-1)
    if nghost == 1:
        mass_center = ice_halo_update_center_jax(mass_center, static, iglob)
        mass_x = ice_halo_update_jax(mass_x, "T", "vector", static, iglob)
        mass_y = ice_halo_update_jax(mass_y, "T", "vector", static, iglob)
        tracer_center = ice_halo_update_center_jax(tracer_center, static, iglob)
        tracer_x = ice_halo_update_jax(tracer_x, "T", "vector", static, iglob)
        tracer_y = ice_halo_update_jax(tracer_y, "T", "vector", static, iglob)

    edgearea_e = edgearea_n = None
    if grid_ice in ("C", "CD"):
        # If l_edge_flux_adj is true, compute edgearea by taking the
        # divergence of the velocity field. Otherwise, initialize edgearea.
        edgearea_e = ice["uvelE"] * grid["HTE"] * dt
        edgearea_n = ice["vvelN"] * grid["HTN"] * dt
    east = ice_transport_remap.locate_triangles(
        dpx, dpy, grid["dxU"], grid["dyU"], grid["earea"],
        grid["narea"], bounds, "east", puny, 1.0e-16, edgearea_e)
    north = ice_transport_remap.locate_triangles(
        dpx, dpy, grid["dxU"], grid["dyU"], grid["earea"],
        grid["narea"], bounds, "north", puny, 1.0e-16, edgearea_n)

    failures = [departure_failed]
    if edgearea_e is not None:
        # If l_edge_flux_adj = T, this area is compared to edgearea as a bug check.
        ilo, ihi, jlo, jhi = bounds
        for geometry, edgearea, factor, cells in (
            (east, edgearea_e, grid["earea"], (slice(ilo-1, ihi+1), slice(jlo, jhi+1))),
            (north, edgearea_n, grid["narea"], (slice(ilo, ihi+1), slice(jlo-1, jhi+1))),
        ):
            area_sum = jnp.sum(geometry[-1], axis=-1)
            mismatch = ((jnp.abs(area_sum - edgearea) > 1.0e-13 * factor) & (edgearea != 0.0)) | ~jnp.isfinite(area_sum)
            failures.append(jnp.zeros_like(mismatch).at[cells].set(mismatch[cells]))
    water_east, _ = ice_transport_remap.transport_integrals(
        mass_center[..., 0], mass_x[..., 0], mass_y[..., 0],
        None, None, None, east, types, parents)
    water_north, _ = ice_transport_remap.transport_integrals(
        mass_center[..., 0], mass_x[..., 0], mass_y[..., 0],
        None, None, None, north, types, parents)
    water_mass, _, water_failed = ice_transport_remap.update_fields(
        mm[..., 0], None, water_east, water_north, None, None,
        grid["tarear"], types, parents, bounds, puny)
    integrate = jax.vmap(
        lambda mc, mx, my, tc, tx, ty, geometry: ice_transport_remap.transport_integrals(
            mc, mx, my, tc, tx, ty, geometry, types, parents),
        in_axes=(-1, -1, -1, -1, -1, -1, None), out_axes=-1)
    me, te = integrate(mass_center[..., 1:], mass_x[..., 1:], mass_y[..., 1:],
                       tracer_center, tracer_x, tracer_y, east)
    mn, tn = integrate(mass_center[..., 1:], mass_x[..., 1:], mass_y[..., 1:],
                       tracer_center, tracer_x, tracer_y, north)
    new_mass, new_trm, category_failed = jax.vmap(
        lambda mass, tracers, me, mn, te, tn: ice_transport_remap.update_fields(
            mass, tracers, me, mn, te, tn, grid["tarear"], types, parents, bounds, puny),
        in_axes=(-1, -1, -1, -1, -1, -1), out_axes=-1)(mm[..., 1:], trm, me, mn, te, tn)
    new_mm = jnp.concatenate((water_mass[..., None], new_mass), axis=-1)
    failures.extend((water_failed, jnp.any(category_failed, axis=-1)))

    conserv_failed = jnp.asarray(False)
    if conserv_check:
        ilo, ihi, jlo, jhi = bounds
        physical = jnp.zeros_like(mm[..., 0], dtype=bool).at[
            ilo:ihi + 1, jlo:jhi + 1].set(True)
        weighted_area = jnp.where(physical, grid["tarea"], 0.0)
        for n in range(ncat + 1):
            updated_mass = new_mm[..., n]
            initial_sum = jnp.sum(weighted_area * mm[..., n])
            final_sum = jnp.sum(weighted_area * updated_mass)
            conserv_failed |= (initial_sum > puny) & (
                jnp.abs(final_sum - initial_sum)
                / jnp.where(initial_sum > puny, initial_sum, 1.0) > puny)
            if n != 0:
                old_tracers = trm[..., n - 1]
                updated_tracers = new_trm[..., n - 1]
                for nt in range(len(types)):
                    old_amount = old_tracers[..., nt]
                    new_amount = updated_tracers[..., nt]
                    if types[nt] == 2:
                        old_amount *= old_tracers[..., parents[nt] - 1]
                        new_amount *= updated_tracers[..., parents[nt] - 1]
                    elif types[nt] == 3:
                        parent = parents[nt] - 1
                        grandparent = parents[parent] - 1
                        old_amount *= (old_tracers[..., parent]
                                       * old_tracers[..., grandparent])
                        new_amount *= (updated_tracers[..., parent]
                                       * updated_tracers[..., grandparent])
                    old_sum = jnp.sum(weighted_area * mm[..., n] * old_amount)
                    new_sum = jnp.sum(weighted_area * updated_mass * new_amount)
                    conserv_failed |= (jnp.abs(old_sum) > puny) & (
                        jnp.abs(new_sum - old_sum)
                        / jnp.where(jnp.abs(old_sum) > puny, jnp.abs(old_sum), 1.0)
                        > puny)
    ice["aice0"] = new_mm[..., 0]
    area = new_mm[..., 1:]
    fields = new_trm
    present = area > 0.0
    ice["aicen"] = jnp.where(present, area, ice["aicen"])
    ice["vicen"] = jnp.where(present, area * fields[:, :, 0, :], ice["vicen"])
    ice["vsnon"] = jnp.where(present, area * fields[:, :, 1, :], ice["vsnon"])
    restored = fields[:, :, 2:, :] - snow_offset[None, None, :, None]
    ice["trcrn"] = jnp.where(present[:, :, None, :], restored, ice["trcrn"])
    for name in ("aicen", "trcrn", "vicen", "vsnon"):
        ice[name] = ice_halo_update_center_jax(ice[name], static, iglob)
    failed = jnp.any(jnp.stack(failures, axis=0), axis=0)
    advanced = {name: ice[name] for name in ("aice0", "aicen", "vicen", "vsnon", "trcrn")}
    return advanced, failed, conserv_failed
