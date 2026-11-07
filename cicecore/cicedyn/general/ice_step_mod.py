"""Common CICE time-step routines, translated from ice_step_mod.F90.

The routines here consume and return a JAX pytree. Initialization remains
in the existing NumPy modules; no time-step routine reads their globals.
"""

import time
from functools import partial

import jax
import jax.numpy as jnp

from cicecore.cicedyn.infrastructure.ice_grid import grid_average_X2Y_jax
from cicecore.cicedyn.infrastructure.comm.serial.ice_boundary import (
    ice_halo_update_center_jax, ice_halo_update_jax,
)
from cicecore.cicedyn.dynamics import ice_dyn_eap, ice_dyn_evp, ice_dyn_vp, ice_transport_driver
from cicecore.cicedyn.general import ice_flux
from cicecore.cicedyn.analysis import ice_diagnostics
from cicecore.cicedyn.infrastructure.comm.serial import ice_timers
from cicecore.shared import ice_fileunits
from icepack.columnphysics.icepack_itd import icepack_aggregate_jax
from icepack.columnphysics import icepack_itd
from icepack.columnphysics.icepack_mechred import icepack_step_ridge
from icepack.columnphysics.icepack_shortwave import (
    icepack_prep_radiation, shortwave_ccsm3_jax,
)
from icepack.columnphysics.icepack_shortwave_jax import (
    compute_coszen_jax, compute_shortwave_trcr_jax,
    icepack_step_radiation_jax,
)
from icepack.columnphysics.icepack_atmo import icepack_atm_boundary
from icepack.columnphysics.icepack_ocean import icepack_ocn_mixed_layer
from icepack.columnphysics.icepack_snow import snow_redist, update_snow_radius
from icepack.columnphysics.icepack_therm_shared import icepack_ice_temperature_jax
from icepack.columnphysics.icepack_therm_vertical import icepack_step_therm1_batch
from icepack.columnphysics.icepack_therm_itd import icepack_step_therm2
from icepack.columnphysics.icepack_zbgc import (
    icepack_biogeochemistry, icepack_load_ocean_bio_array,
)
from icepack.columnphysics.icepack_wavefracspec import icepack_step_wavefracture


# Temporary gx3 causal experiment; enabled explicitly by that case's runner.
DEBUG_REPLAY_ATMO_COEFFICIENTS = False
DEBUG_REPLAY_ATMO_PSI = False
DEBUG_REPLAY_ATMO_MATH = False
PROFILE_PICARD_STEPS = None
PICARD_BATCH_STOP = False
THERM2_BATCH_JIT = False
RIDGE_BATCH_JIT = False
RIDGE_BATCH_STOP = False


def save_init(pytree, static):
    """Save initial area, category volumes, and surface temperature."""
    state = dict(pytree["state"])
    ice = dict(state["ice_state"])
    nt_Tsfc = int(static["icepack_tracers"]["nt_Tsfc"])
    # Save ice area passed to the coupler, and each category's initial state.
    ice["aice_init"] = ice["aice"]
    ice["aicen_init"] = ice["aicen"]
    ice["vicen_init"] = ice["vicen"]
    ice["vsnon_init"] = ice["vsnon"]
    ice["Tsfcn_init"] = ice["trcrn"][:, :, nt_Tsfc, :]
    state["ice_state"] = ice
    result = dict(pytree)
    result["state"] = state
    return result


def step_prep(pytree, static):
    """Save starting ice state and average wind and optional ice velocity."""
    pytree = save_init(pytree, static)
    state = dict(pytree["state"])
    flux = dict(state["ice_flux"])
    icep = dict(state["ice_step_mod"])
    ice = state["ice_state"]
    grid = pytree["constants"]["ice_grid"]
    bounds = static["ice_domain_size"]["physical_bounds"]
    init = static["ice_init"]

    # Wind on atmospheric dynamics points -> thermodynamics T points.
    flux["uatmT"] = grid_average_X2Y_jax(
        "S", flux["uatm"], init["grid_atm_dynu"], flux["uatmT"], "T", grid, bounds
    )
    flux["vatmT"] = grid_average_X2Y_jax(
        "S", flux["vatm"], init["grid_atm_dynv"], flux["vatmT"], "T", grid, bounds
    )

    # Fortran allocates these once and retains them when highfreq is off.
    if bool(static["icepack_parameters"]["highfreq"]):
        icep["uvelT_icep"] = grid_average_X2Y_jax(
            "A", ice["uvel"], "U", icep["uvelT_icep"], "T", grid, bounds
        )
        icep["vvelT_icep"] = grid_average_X2Y_jax(
            "A", ice["vvel"], "U", icep["vvelT_icep"], "T", grid, bounds
        )

    state["ice_flux"] = flux
    state["ice_step_mod"] = icep
    result = dict(pytree)
    result["state"] = state
    return result


@jax.profiler.annotate_function
def prep_radiation(pytree, static):
    """Scale radiation saved from the prior step using current shortwave."""
    state = dict(pytree["state"])
    flux = dict(state["ice_flux"])
    column = dict(state["ice_arrays_column"])
    ice = state["ice_state"]
    flux_names = ("swvdr", "swvdf", "swidr", "swidf", "scale_factor") + tuple(
        name + suffix for name in ("alvdr", "alvdf", "alidr", "alidf")
        for suffix in ("_ai", "_init"))
    column_names = (
        "fswsfcn", "fswintn", "fswthrun", "fswthrun_vdr", "fswthrun_vdf",
        "fswthrun_idr", "fswthrun_idf", "fswpenln", "Sswabsn", "Iswabsn",
    )
    flux_updates, column_updates = _prep_radiation_compiled(
        ice["aice"], ice["aicen"],
        {name: flux[name] for name in flux_names},
        {name: column[name] for name in column_names},
        pytree["parameters"]["icepack_parameters"]["puny"],
        static["ice_domain_size"]["physical_bounds"], column_names)
    flux.update(flux_updates)
    column.update(column_updates)
    state["ice_flux"] = flux
    state["ice_arrays_column"] = column
    result = dict(pytree)
    result["state"] = state
    return result


@partial(jax.jit, static_argnames=("bounds", "column_names"))
def _prep_radiation_compiled(aice, aicen, flux, column, puny, bounds, column_names):
    """Compile radiation updates with only their required inputs and outputs."""
    column = dict(column)
    flux_updates = {}
    ilo, ihi, jlo, jhi = bounds
    cells = (slice(ilo, ihi + 1), slice(jlo, jhi + 1))

    # Fortran first zeros the entire albedo snapshots, then copies only
    # physical cells from the previous radiation call.
    for name in ("alvdr", "alvdf", "alidr", "alidf"):
        flux_updates[name + "_init"] = jnp.zeros_like(flux[name + "_init"]).at[cells].set(
            flux[name + "_ai"][cells]
        )

    scaled = icepack_prep_radiation(
        aice[cells], aicen[cells],
        flux["swvdr"][cells], flux["swvdf"][cells],
        flux["swidr"][cells], flux["swidf"][cells],
        flux["alvdr_ai"][cells], flux["alvdf_ai"][cells],
        flux["alidr_ai"][cells], flux["alidf_ai"][cells],
        flux["scale_factor"][cells],
        *(column[name][cells] for name in column_names),
        puny,
    )
    flux_updates["scale_factor"] = flux["scale_factor"].at[cells].set(scaled[0])
    for name, value in zip(column_names, scaled[1:]):
        column[name] = column[name].at[cells].set(value)

    return flux_updates, column


def _timed_column_call(timing, prefix, name, function, *arrays):
    """Time one batched stage on the host, outside JAX transformations."""
    stage = prefix + "." + name
    if timing.get("detailed_physics", True):
        ice_fileunits.nu_diag.write(f"TIMING_BEGIN step={timing['step']} stage={stage}\n")
        ice_fileunits.nu_diag.flush()
        jax.block_until_ready(arrays)
    started = time.perf_counter()
    with jax.profiler.TraceAnnotation(stage):
        result = jax.vmap(function)(*arrays)
        ice_timers.record_stage(timing, stage, started, result)
    return result


@jax.profiler.annotate_function
def step_therm1(pytree, static, timing=None):
    """Apply Icepack vertical thermodynamics over the physical T cells."""
    stage_started = time.perf_counter()
    state = dict(pytree["state"])
    ice = dict(state["ice_state"])
    flux = dict(state["ice_flux"])
    column = dict(state["ice_arrays_column"])
    bgc = dict(state["ice_flux_bgc"])
    icep = dict(state["ice_step_mod"])
    grid = pytree["constants"]["ice_grid"]
    tracers = static["icepack_tracers"]
    dimensions = static["ice_domain_size"]
    ncat = int(dimensions["ncat"])
    nslyr = int(dimensions["nslyr"])
    nilyr = int(dimensions["nilyr"])
    nfsd = int(dimensions["nfsd"]) if bool(tracers["tr_fsd"]) else 0
    ilo, ihi, jlo, jhi = dimensions["physical_bounds"]
    cell_shape = (ihi - ilo + 1, jhi - jlo + 1)
    number_of_cells = cell_shape[0] * cell_shape[1]

    options = dict(static["icepack_parameters"])
    options.update(tracers)
    options.update(static["icepack_therm_shared"])
    options["send_i2x_per_cat"] = static["ice_flux"]["send_i2x_per_cat"]
    options["nslyr"] = nslyr
    options["nilyr"] = nilyr
    options["prescribed_ice"] = False
    # Keep this static option fixed throughout the run to avoid recompiling at step 5.
    options["profile_picard"] = (
        PROFILE_PICARD_STEPS is not None and int(options["ktherm"]) == 2)
    options["picard_batch_stop"] = PICARD_BATCH_STOP and int(options["ktherm"]) == 2
    init = static["ice_init"]
    options["debug_model_ponds_inputs"] = (
        bool(init["debug_model"]) and bool(init["debug_model_ponds_inputs"])
        and int(static["ice_calendar"]["istep1"]) >= int(init["debug_model_step"]))
    options["debug_replay_atmo_coefficients"] = (
        DEBUG_REPLAY_ATMO_COEFFICIENTS and options["debug_model_ponds_inputs"]
        and int(static["ice_calendar"]["istep1"]) == 1)
    options["debug_replay_atmo_psi"] = (
        DEBUG_REPLAY_ATMO_PSI and options["debug_model_ponds_inputs"]
        and int(static["ice_calendar"]["istep1"]) == 1)
    options["debug_replay_atmo_math"] = (
        DEBUG_REPLAY_ATMO_MATH and options["debug_model_ponds_inputs"]
        and int(static["ice_calendar"]["istep1"]) == 1)
    if sum((DEBUG_REPLAY_ATMO_COEFFICIENTS, DEBUG_REPLAY_ATMO_PSI, DEBUG_REPLAY_ATMO_MATH)) > 1:
        raise ValueError("Choose only one atmospheric replay experiment")
    # This setting is a fixed parameter in the original isotope module.
    options["isotope_frac_method"] = "cfrac"
    parameters = pytree["parameters"]["icepack_parameters"]
    ice_names = (
        "aice", "aicen", "aicen_init", "vicen_init", "vsnon_init",
        "vice", "vicen", "vsno", "vsnon",
    )
    flux_names = (
        "frzmlt", "sst", "Tf", "rsiden", "fbot", "Tbot", "Tsnice",
        "meltsn", "melttn", "meltbn", "congeln", "snoicen",
        "wind", "rhoa", "potT", "Qa", "zlvl", "zlvs", "strax", "stray",
        "flatn", "fsensn", "fsurfn", "fcondtopn", "flw", "fsnow",
        "fpond", "sss", "mlt_onset", "frz_onset", "fcondbotn",
        "fcondbot", "fsloss", "frain", "Tair", "strairxT", "strairyT",
        "fsurf", "fcondtop", "fsens", "flat", "fswabs", "flwout",
        "evap", "evaps", "evapi", "Tref", "Qref", "Uref", "fresh",
        "fsalt", "fhocn", "fswthru", "fswthru_vdr", "fswthru_vdf",
        "fswthru_idr", "fswthru_idf", "fswthru_uvrdr", "fswthru_uvrdf",
        "fswthru_pardr", "fswthru_pardf", "meltt", "melts", "meltb",
        "congel", "snoice", "flatn_f", "fsensn_f", "fsurfn_f",
        "fcondtopn_f", "dsnow", "dpnd_flush", "dpnd_expon",
        "dpnd_freebd", "dpnd_initial", "dpnd_dlid", "dpnd_flushn",
        "dpnd_exponn", "dpnd_freebdn", "dpnd_initialn", "dpnd_dlidn",
        "fswthrun_ai", "wlat",
    )
    column_names = (
        "Cdn_ocn", "Cdn_ocn_skin", "Cdn_ocn_floe", "Cdn_ocn_keel",
        "Cdn_atm_ratio", "Cdn_atm", "Cdn_atm_skin", "Cdn_atm_floe",
        "Cdn_atm_rdg", "Cdn_atm_pond", "hfreebd", "hdraft", "hridge",
        "distrdg", "hkeel", "dkeel", "lfloe", "dfloe", "fswsfcn",
        "fswintn", "Sswabsn", "Iswabsn", "meltsliqn", "meltsliq",
        "fswthrun", "fswthrun_vdr", "fswthrun_vdf",
        "fswthrun_idr", "fswthrun_idf", "fswthrun_uvrdr",
        "fswthrun_uvrdf", "fswthrun_pardr", "fswthrun_pardf",
        "dhsn", "ffracn",
    )
    bgc_names = (
        "Qa_iso", "Qref_iso", "fiso_atm", "fiso_ocn", "fiso_evap",
        "HDO_ocn", "H2_16O_ocn", "H2_18O_ocn", "dsnown",
    )
    if not options["profile_picard"] and not options["debug_model_ponds_inputs"]:
        updates = _therm1_compiled(
            ice, flux, column, bgc, icep, grid, dimensions["physical_bounds"],
            tuple(sorted(tracers.items())), nilyr, nslyr, nfsd,
            tuple(sorted(options.items())),
            (ice_names, flux_names, column_names, bgc_names), parameters,
            static["ice_calendar"]["dt"], static["ice_calendar"]["yday"])
        for name, original, changes in zip(
                ("ice_state", "ice_flux", "ice_arrays_column", "ice_flux_bgc", "ice_step_mod"),
                (ice, flux, column, bgc, icep), updates):
            state[name] = {**original, **changes}
        result = {**pytree, "state": state}
        ice_timers.record_stage(timing, "physics.therm1.compiled", stage_started, result)
        return result
    flattened, valid = _therm1_prepare(
        ice, flux, column, bgc, icep, grid, dimensions["physical_bounds"],
        tuple(sorted(tracers.items())), nilyr, nslyr, nfsd,
        bool(options["snwgrain"]), bool(options["picard_batch_stop"]),
        (ice_names, flux_names, column_names, bgc_names))
    replay_atmo = (options["debug_replay_atmo_coefficients"] or options["debug_replay_atmo_psi"]
                   or options["debug_replay_atmo_math"])
    if replay_atmo:
        fields = {
            name: value.reshape(cell_shape + value.shape[1:])
            for name, value in flattened.items()
        }
        if (cell_shape != (100, 116) or ncat != 5
                or int(options["ktherm"]) != 2
                or not bool(options["calc_Tsfc"])
                or bool(options["semi_implicit_Tsfc"])
                or float(static["ice_calendar"]["dt"]) != 3600.0):
            raise ValueError("Atmospheric replay requires the gx3 one-step debug case")
        points = set(zip(init["debug_model_ponds_i"], init["debug_model_ponds_j"]))
        if not {(59, 110), (95, 104)}.issubset(points):
            raise ValueError("Atmospheric replay requires both diagnosed pond points")
    if options["debug_replay_atmo_coefficients"]:
        fields["debug_atmo_replay"] = jnp.zeros_like(fields["aicen"], dtype=bool)
        fields["debug_shcoef"] = jnp.zeros_like(fields["aicen"])
        fields["debug_lhcoef"] = jnp.zeros_like(fields["aicen"])
        # Values from fortran_pond_debug/cice.runlog.261030-201956.
        for i, j, n, shcoef, lhcoef in (
            (59, 110, 2, 1.52822884906101137e+01, 4.02819885464967520e+04),
            (95, 104, 1, 2.56510868339752953e+01, 6.93791816194914863e+04),
        ):
            if (i, j) not in points:
                raise ValueError("Atmospheric coefficient replay requires both diagnosed pond points")
            index = (i - 1, j - 1, n - 1)
            fields["debug_atmo_replay"] = fields["debug_atmo_replay"].at[index].set(True)
            fields["debug_shcoef"] = fields["debug_shcoef"].at[index].set(shcoef)
            fields["debug_lhcoef"] = fields["debug_lhcoef"].at[index].set(lhcoef)

    if options["debug_replay_atmo_psi"] or options["debug_replay_atmo_math"]:
        if int(options["natmiter"]) != 5 or options["atmbndy"].strip() != "similarity":
            raise ValueError("Stability replay requires similarity with natmiter=5")
    if options["debug_replay_atmo_psi"]:
        fields["debug_psi_replay"] = jnp.zeros_like(fields["aicen"], dtype=bool)
        fields["debug_psi_values"] = jnp.zeros(
            cell_shape + (ncat, 5, 2), dtype=fields["aicen"].dtype)
        # Fortran records 1..5, columns psimh/psixh; same reference as coefficient replay.
        for i, j, n, values in (
            (59, 110, 2, (
                (-6.74720936580218478e-01, -6.74720936580218478e-01),
                (-7.21170212080210149e-01, -7.21170212080210149e-01),
                (-7.24362257678711430e-01, -7.24362257678711430e-01),
                (-7.24581592011695363e-01, -7.24581592011695363e-01),
                (-7.24596662958418847e-01, -7.24596662958418847e-01))),
            (95, 104, 1, (
                (5.26005531957764561e-01, 9.52660811379775430e-01),
                (5.23375307910811660e-01, 9.48264011586484878e-01),
                (5.23398186996037573e-01, 9.48302269440609469e-01),
                (5.23397988305249751e-01, 9.48301937195588307e-01),
                (5.23397990030781468e-01, 9.48301940080973016e-01))),
        ):
            index = (i - 1, j - 1, n - 1)
            fields["debug_psi_replay"] = fields["debug_psi_replay"].at[index].set(True)
            fields["debug_psi_values"] = fields["debug_psi_values"].at[index].set(
                jnp.asarray(values, dtype=fields["aicen"].dtype))

    if options["debug_replay_atmo_math"]:
        fields["debug_exp_replay"] = jnp.zeros(cell_shape + (ncat, 5), dtype=bool)
        fields["debug_atan_replay"] = jnp.zeros(cell_shape + (ncat, 5), dtype=bool)
        fields["debug_math_values"] = jnp.zeros(
            cell_shape + (ncat, 5, 4), dtype=fields["aicen"].dtype)
        # stability_probe_fortran.log: exp input/output, then atan input/output.
        # Record is one-based, as in the atmospheric iteration logs.
        for i, j, n, record, replay_exp, replay_atan, values in (
            (59, 110, 2, 5, True, False,
             (-5.17425146091501573e-02, 9.49573336629562226e-01, 0.0, 0.0)),
            (95, 104, 1, 1, False, True,
             (0.0, 0.0, 1.49007042278595625e+00, 9.79724412023601610e-01)),
            (95, 104, 1, 5, False, True,
             (0.0, 0.0, 1.48771605131007933e+00, 9.78992513919328511e-01)),
        ):
            index = (i - 1, j - 1, n - 1, record - 1)
            fields["debug_exp_replay"] = fields["debug_exp_replay"].at[index].set(replay_exp)
            fields["debug_atan_replay"] = fields["debug_atan_replay"].at[index].set(replay_atan)
            fields["debug_math_values"] = fields["debug_math_values"].at[index].set(
                jnp.asarray(values, dtype=fields["aicen"].dtype))

    if replay_atmo:
        flattened = {
            name: value.reshape((number_of_cells,) + value.shape[2:])
            for name, value in fields.items()
        }
    ice_timers.record_stage(timing, "physics.therm1.prepare", stage_started, (flattened, valid))
    stage_started = time.perf_counter()
    advanced, failed, aerosol_warning = icepack_step_therm1_batch(
        static["ice_calendar"]["dt"], static["ice_calendar"]["yday"],
        flattened, options, parameters,
        column["floe_rad_c"] if nfsd else None,
        column["floe_binwidth"] if nfsd else None)
    ice_timers.record_stage(timing, "physics.therm1.columns", stage_started,
                            (advanced, failed, aerosol_warning))
    if options["profile_picard"]:
        stage_started = time.perf_counter()
        device_stats = _picard_iteration_summary(advanced["picard_iterations"], valid)
        ice_timers.record_stage(timing, "physics.therm1.picard_stats_gpu", stage_started, device_stats)
        step = int(static["ice_calendar"]["istep1"])
        if PROFILE_PICARD_STEPS[0] <= step <= PROFILE_PICARD_STEPS[1]:
            stage_started = time.perf_counter()
            histograms, unconverged = jax.device_get(device_stats)
            ice_timers.record_stage(timing, "physics.therm1.picard_stats_d2h", stage_started)
            for category, (hist_pair, failed_pair) in enumerate(
                zip(histograms.tolist(), unconverged.tolist()), start=1
            ):
                for attempt, hist, failed_count in zip(("first", "retry"), hist_pair, failed_pair):
                    count = sum(hist)
                    if count == 0:
                        continue
                    cumulative = 0
                    p50 = p95 = 0
                    for nit, number in enumerate(hist):
                        cumulative += number
                        if p50 == 0 and cumulative >= 0.50 * count:
                            p50 = nit
                        if p95 == 0 and cumulative >= 0.95 * count:
                            p95 = nit
                    mean = sum(nit * number for nit, number in enumerate(hist)) / count
                    maximum = max(nit for nit, number in enumerate(hist) if number)
                    bins = ",".join(f"{nit}:{number}" for nit, number in enumerate(hist) if number)
                    ice_fileunits.nu_diag.write(
                        f"PICARD_STATS step={step} category={category} attempt={attempt} "
                        f"columns={count} mean_iterations={mean:.3f} p50={p50} p95={p95} "
                        f"max={maximum} reached_100={hist[100]} unconverged={failed_count} "
                        f"histogram={bins}\n")
            ice_fileunits.nu_diag.flush()
    stage_started = time.perf_counter()
    if options["debug_model_ponds_inputs"]:
        # Log outside vmap so only the requested cells reach the host callback.
        points = tuple((i, j) for i, j in zip(init["debug_model_ponds_i"], init["debug_model_ponds_j"])
                       if i > 0 and j > 0)
        samples = []
        for global_i, global_j in points:
            if not (1 <= global_i <= cell_shape[0] and 1 <= global_j <= cell_shape[1]):
                raise ValueError("debug_model_ponds_i/j is outside the physical domain")
            cell_index = (global_i - 1) * cell_shape[1] + global_j - 1
            samples.append(advanced["pond_diagnostics"][cell_index])
        step = int(static["ice_calendar"]["istep1"])
        jax.debug.callback(lambda values: ice_diagnostics.print_pond_inputs(step, points, values),
                           jnp.stack(samples), ordered=True)
        solver_points = tuple((i, j, n) for i, j in points for n in range(1, ncat + 1))
        if solver_points:
            atmo_samples = []
            picard_samples = []
            for global_i, global_j, category in solver_points:
                cell_index = (global_i - 1) * cell_shape[1] + global_j - 1
                atmo_samples.append(advanced["atmo_diagnostics"][cell_index, category - 1])
                picard_samples.append(advanced["picard_diagnostics"][cell_index, category - 1])
            jax.debug.callback(
                lambda atmo, picard: ice_diagnostics.print_pond_solver(
                    step, solver_points, atmo, picard, nilyr, nslyr,
                    options["debug_replay_atmo_coefficients"],
                    options["debug_replay_atmo_psi"], options["debug_replay_atmo_math"]),
                jnp.stack(atmo_samples), jnp.stack(picard_samples), ordered=True)
    updates = _therm1_writeback(
        ice, flux, column, bgc, icep, flattened, advanced,
        failed, aerosol_warning, valid, dimensions["physical_bounds"],
        tuple(sorted(tracers.items())), bool(options["snwgrain"]),
        (ice_names, flux_names, column_names, bgc_names), parameters["puny"])
    for name, original, changes in zip(
            ("ice_state", "ice_flux", "ice_arrays_column", "ice_flux_bgc", "ice_step_mod"),
            (ice, flux, column, bgc, icep), updates):
        state[name] = {**original, **changes}
    result = dict(pytree)
    result["state"] = state
    ice_timers.record_stage(timing, "physics.therm1.writeback", stage_started, result)
    return result


@partial(jax.jit, static_argnames=("bounds", "tracers", "nilyr", "nslyr", "nfsd",
                                 "options", "field_names"))
def _therm1_compiled(ice, flux, column, bgc, icep, grid, bounds, tracers,
                     nilyr, nslyr, nfsd, options, field_names, parameters, dt, yday):
    # Keep extracted fields and solver intermediates inside one executable.
    options = dict(options)
    flattened, valid = _therm1_prepare(
        ice, flux, column, bgc, icep, grid, bounds, tracers, nilyr, nslyr, nfsd,
        bool(options["snwgrain"]), bool(options["picard_batch_stop"]), field_names)
    advanced, failed, aerosol_warning = icepack_step_therm1_batch(
        dt, yday, flattened, options, parameters,
        column["floe_rad_c"] if nfsd else None,
        column["floe_binwidth"] if nfsd else None)
    return _therm1_writeback(
        ice, flux, column, bgc, icep, flattened, advanced,
        failed, aerosol_warning, valid, bounds, tracers,
        bool(options["snwgrain"]), field_names, parameters["puny"])


@jax.profiler.annotate_function
@partial(jax.jit, static_argnames=("bounds", "tracers", "nilyr", "nslyr", "nfsd",
                                 "snwgrain", "picard_batch_stop", "field_names"))
def _therm1_prepare(ice, flux, column, bgc, icep, grid, bounds, tracers,
                    nilyr, nslyr, nfsd, snwgrain, picard_batch_stop, field_names):
    """Compile column extraction, tracer conversions and flattening together."""
    tracers = dict(tracers)
    trcrn = ice["trcrn"]
    ncat = ice["aicen"].shape[-1]
    n_iso = int(tracers["n_iso"]) if bool(tracers["tr_iso"]) else 0
    n_aero = int(tracers["n_aero"]) if bool(tracers["tr_aero"]) else 0
    ilo, ihi, jlo, jhi = bounds
    cells = (slice(ilo, ihi + 1), slice(jlo, jhi + 1))
    cell_shape = (ihi - ilo + 1, jhi - jlo + 1)
    number_of_cells = cell_shape[0] * cell_shape[1]
    valid = grid["tmask"][cells] | grid["opmask"][cells]
    fields = {}
    if picard_batch_stop:
        fields["picard_active"] = valid
    for source, names in zip((ice, flux, column, bgc), field_names):
        for name in names:
            fields[name] = source[name][cells]
    for name in ("lmask_n", "lmask_s"):
        fields[name] = grid[name][cells]
    fields["uatm"] = flux["uatmT"][cells]
    fields["vatm"] = flux["vatmT"][cells]
    fields["strocnxT"] = flux["strocnxT_iavg"][cells]
    fields["strocnyT"] = flux["strocnyT_iavg"][cells]
    fields["dfsurfdT"] = flux["dfsurfndTsfc_f"][cells]
    fields["dflatdT"] = flux["dflatndTsfc_f"][cells]
    fields["uvel"] = icep["uvelT_icep"][cells]
    fields["vvel"] = icep["vvelT_icep"][cells]

    nt_qsno = int(tracers["nt_qsno"])
    nt_qice = int(tracers["nt_qice"])
    nt_sice = int(tracers["nt_sice"])
    for name, start, length in (
        ("zqsn", nt_qsno, nslyr),
        ("zqin", nt_qice, nilyr),
        ("zSin", nt_sice, nilyr),
    ):
        fields[name] = trcrn[cells + (slice(start, start + length), slice(None))]
    for name, tracer in (
        ("Tsfc", "nt_Tsfc"), ("alvl", "nt_alvl"),
        ("vlvl", "nt_vlvl"), ("apnd", "nt_apnd"),
        ("hpnd", "nt_hpnd"), ("ipnd", "nt_ipnd"),
        ("iage", "nt_iage"), ("FY", "nt_FY"),
    ):
        fields[name] = trcrn[cells + (int(tracers[tracer]), slice(None))]
    if snwgrain:
        for name, tracer in (
            ("rsnwn", "nt_rsnw"), ("smicen", "nt_smice"),
            ("smliqn", "nt_smliq"),
        ):
            start = int(tracers[tracer])
            fields[name] = trcrn[cells + (slice(start, start + nslyr), slice(None))]
    else:
        for name in ("rsnwn", "smicen", "smliqn"):
            fields[name] = jnp.zeros(cell_shape + (nslyr, ncat), dtype=trcrn.dtype)
    if n_iso:
        for name, tracer, volume in (
            ("isosno", "nt_isosno", "vsnon_init"),
            ("isoice", "nt_isoice", "vicen_init"),
        ):
            start = int(tracers[tracer])
            concentration = trcrn[cells + (slice(start, start + n_iso), slice(None))]
            fields[name] = concentration * fields[volume][..., None, :]
    else:
        for name in ("isosno", "isoice"):
            fields[name] = jnp.zeros(cell_shape + (0, ncat), dtype=trcrn.dtype)
    if n_aero:
        snow = []
        sea_ice = []
        start = int(tracers["nt_aero"])
        for k in range(n_aero):
            snow.append(trcrn[cells + (slice(start + 4*k, start + 4*k + 2), slice(None))])
            sea_ice.append(trcrn[cells + (slice(start + 4*k + 2, start + 4*k + 4), slice(None))])
        fields["aerosno"] = jnp.stack(snow, axis=2) * fields["vsnon_init"][..., None, None, :]
        fields["aeroice"] = jnp.stack(sea_ice, axis=2) * fields["vicen_init"][..., None, None, :]
    else:
        for name in ("aerosno", "aeroice"):
            fields[name] = jnp.zeros(cell_shape + (0, 2, ncat), dtype=trcrn.dtype)
    fields["faero_atm"] = bgc["faero_atm"][cells + (slice(0, n_aero),)]
    fields["faero_ocn"] = bgc["faero_ocn"][cells + (slice(0, n_aero),)]
    # Atmospheric isotope humidity retains the allocated species width.
    if nfsd:
        start = int(tracers["nt_fsd"])
        fields["afsdn"] = trcrn[cells + (slice(start, start + nfsd), slice(None))]
    else:
        fields["afsdn"] = jnp.zeros(cell_shape + (0, ncat), dtype=trcrn.dtype)
    flattened = {
        name: value.reshape((number_of_cells,) + value.shape[2:])
        for name, value in fields.items()
    }
    return flattened, valid.reshape(-1)


@jax.jit
def _picard_iteration_summary(iterations, valid):
    # Input axes are cell, category, attempt, (iteration count, heat convergence).
    nit = iterations[..., 0].transpose(1, 2, 0)
    used = valid[None, None, :] & (nit > 0)
    histogram = jax.vmap(jax.vmap(
        lambda values, weights: jnp.bincount(values, weights=weights, length=101)
    ))(nit, used.astype(jnp.int32))
    unconverged = jnp.sum(
        used & (iterations[..., 1].transpose(1, 2, 0) == 0), axis=-1, dtype=jnp.int32)
    return histogram, unconverged


@jax.profiler.annotate_function
@partial(jax.jit, static_argnames=("bounds", "tracers", "snwgrain", "field_names"))
def _therm1_writeback(ice, flux, column, bgc, icep, flattened, advanced,
                      failed, aerosol_warning, valid, bounds, tracers,
                      snwgrain, field_names, puny):
    """Mask and write therm1 outputs in one compiled device operation."""
    original_groups = (ice, flux, column, bgc, icep)
    ice, flux, column, bgc, icep = (
        dict(ice), dict(flux), dict(column), dict(bgc), dict(icep))
    tracers = dict(tracers)
    ice_names, flux_names, column_names, bgc_names = field_names
    ilo, ihi, jlo, jhi = bounds
    cells = (slice(ilo, ihi + 1), slice(jlo, jhi + 1))
    cell_shape = (ihi - ilo + 1, jhi - jlo + 1)
    number_of_cells = cell_shape[0] * cell_shape[1]
    nslyr = flattened["zqsn"].shape[1]
    nilyr = flattened["zqin"].shape[1]
    n_iso = flattened["isoice"].shape[1]
    n_aero = flattened["aeroice"].shape[1]
    nt_qsno = int(tracers["nt_qsno"])
    nt_qice = int(tracers["nt_qice"])
    nt_sice = int(tracers["nt_sice"])
    trcrn = ice["trcrn"]
    fields = {}
    for name, original in flattened.items():
        selected = original
        if name in advanced:
            shape = (number_of_cells,) + (1,) * (original.ndim - 1)
            selected = jnp.where(valid.reshape(shape), advanced[name], original)
        fields[name] = selected.reshape(cell_shape + original.shape[1:])

    for name in ice_names:
        if name in advanced:
            ice[name] = ice[name].at[cells].set(fields[name])
    for name in flux_names:
        if name in advanced:
            flux[name] = flux[name].at[cells].set(fields[name])
    for name in column_names:
        if name in advanced:
            column[name] = column[name].at[cells].set(fields[name])
    for name in bgc_names:
        if name in advanced and name not in (
            "Qa_iso", "Qref_iso", "fiso_atm", "fiso_ocn", "fiso_evap"
        ):
            bgc[name] = bgc[name].at[cells].set(fields[name])
    for name in ("Qref_iso", "fiso_ocn", "fiso_evap"):
        bgc[name] = bgc[name].at[cells + (slice(0, n_iso),)].set(
            fields[name][..., :n_iso])
    bgc["faero_ocn"] = bgc["faero_ocn"].at[cells + (slice(0, n_aero),)].set(fields["faero_ocn"])

    for name, start, length in (
        ("zqsn", nt_qsno, nslyr),
        ("zqin", nt_qice, nilyr),
        ("zSin", nt_sice, nilyr),
    ):
        trcrn = trcrn.at[cells + (slice(start, start + length), slice(None))].set(fields[name])
    for name, tracer in (
        ("Tsfc", "nt_Tsfc"), ("alvl", "nt_alvl"),
        ("vlvl", "nt_vlvl"), ("apnd", "nt_apnd"),
        ("hpnd", "nt_hpnd"), ("ipnd", "nt_ipnd"),
        ("iage", "nt_iage"), ("FY", "nt_FY"),
    ):
        trcrn = trcrn.at[cells + (int(tracers[tracer]), slice(None))].set(fields[name])
    if snwgrain:
        for name, tracer in (
            ("rsnwn", "nt_rsnw"), ("smicen", "nt_smice"),
            ("smliqn", "nt_smliq"),
        ):
            start = int(tracers[tracer])
            trcrn = trcrn.at[cells + (slice(start, start + nslyr), slice(None))].set(fields[name])
    if n_iso:
        for name, tracer, volume in (
            ("isosno", "nt_isosno", "vsnon"),
            ("isoice", "nt_isoice", "vicen"),
        ):
            start = int(tracers[tracer])
            present = fields[volume] > puny
            concentration = jnp.where(
                present[..., None, :],
                fields[name] / jnp.where(present[..., None, :],
                                          fields[volume][..., None, :], 1.0),
                fields[name])
            trcrn = trcrn.at[cells + (slice(start, start + n_iso), slice(None))].set(concentration)
    if n_aero:
        start = int(tracers["nt_aero"])
        for k in range(n_aero):
            for name, offset, volume in (
                ("aerosno", 0, "vsnon"), ("aeroice", 2, "vicen")
            ):
                present = fields[volume] > puny
                concentration = jnp.where(
                    present[..., None, :],
                    fields[name][..., k, :, :] /
                    jnp.where(present[..., None, :],
                              fields[volume][..., None, :], 1.0),
                    fields[name][..., k, :, :])
                trcrn = trcrn.at[
                    cells + (slice(start + 4*k + offset, start + 4*k + offset + 2), slice(None))
                ].set(concentration)
    ice["trcrn"] = trcrn
    icep["therm1_failed"] = icep["therm1_failed"].at[cells].set(
        failed.reshape(cell_shape) & valid.reshape(cell_shape))
    icep["therm1_aerosol_warning"] = icep["therm1_aerosol_warning"].at[cells].set(
        aerosol_warning.reshape(cell_shape) & valid.reshape(cell_shape))
    # Tracer identity selects reassigned fields at trace time, not by value.
    return tuple({name: value for name, value in after.items()
                  if value is not before[name]}
                 for before, after in zip(original_groups, (ice, flux, column, bgc, icep)))


def biogeochemistry(pytree, static):
    """Advance active brine cells before the second thermodynamic stage."""
    flags = static["icepack_tracers"]
    options = static["icepack_parameters"]
    if bool(options["z_tracers"]) and not bool(flags["tr_brine"]):
        raise ValueError("z_tracers requires the brine tracer")
    if not bool(flags["tr_brine"]) and not bool(options["skl_bgc"]):
        return pytree
    state = dict(pytree["state"])
    ice = dict(state["ice_state"])
    flux = state["ice_flux"]
    bgc = dict(state["ice_flux_bgc"])
    column = dict(state["ice_arrays_column"])
    icep = dict(state["ice_step_mod"])
    grid = pytree["constants"]["ice_grid"]
    ilo, ihi, jlo, jhi = static["ice_domain_size"]["physical_bounds"]
    cells = (slice(ilo, ihi + 1), slice(jlo, jhi + 1))
    cell_shape = ice["aice"][cells].shape
    number_of_cells = cell_shape[0] * cell_shape[1]
    nilyr = int(static["ice_domain_size"]["nilyr"])
    nblyr = int(static["ice_domain_size"]["nblyr"])
    stage_options = dict(options)
    stage_options.update({"nilyr": nilyr, "nblyr": nblyr,
                          "nslyr": int(static["ice_domain_size"]["nslyr"])})
    parameters = pytree["parameters"]["icepack_parameters"]
    bio_parameters = dict(pytree["parameters"]["icepack_zbgc_shared"])
    # icepack_init_zbgc derives the species tables from namelist scalars.
    # Build them from the scalar pytree leaves at each step so a perturbation
    # of any original physical parameter follows the JAX gradient path.
    for name, prefix in (
        ("R_Si2N", "ratio_Si2N"), ("R_S2N", "ratio_S2N"),
        ("R_Fe2C", "ratio_Fe2C"), ("R_Fe2N", "ratio_Fe2N"),
        ("R_C2N", "ratio_C2N"), ("R_chl2N", "ratio_chl2N"),
        ("F_abs_chl", "F_abs_chl"), ("chlabs", "chlabs"),
        ("alpha2max_low", "alpha2max_low"), ("beta2max", "beta2max"),
        ("mu_max", "mu_max"), ("grow_Tdep", "grow_Tdep"),
        ("fr_graze", "fr_graze"), ("mort_pre", "mort_pre"),
        ("mort_Tdep", "mort_Tdep"), ("k_exude", "k_exude"),
        ("K_Nit", "K_Nit"), ("K_Am", "K_Am"),
        ("K_Sil", "K_Sil"), ("K_Fe", "K_Fe"),
    ):
        bio_parameters[name] = jnp.stack([
            parameters[prefix + "_" + species]
            for species in ("diatoms", "sp", "phaeo")
        ])
    bio_parameters.update({
        "R_C2N_DON": jnp.stack([parameters["ratio_C2N_proteins"]]),
        "R_Fe2DON": jnp.stack([parameters["ratio_Fe2DON"]]),
        "R_Fe2DOC": jnp.stack([parameters["ratio_Fe2DOC_s"],
                                parameters["ratio_Fe2DOC_l"], 0.0]),
        "f_doc": jnp.stack([parameters["f_doc_s"], parameters["f_doc_l"], 0.0]),
        "f_exude": jnp.stack([parameters["f_exude_s"], parameters["f_exude_l"], 0.0]),
        "k_bac": jnp.stack([parameters["k_bac_s"], parameters["k_bac_l"], 0.0]),
        "f_don": jnp.stack([parameters["f_don_protein"]]),
        "kn_bac": jnp.stack([parameters["kn_bac_protein"]]),
        "f_don_Am": jnp.stack([parameters["f_don_Am_protein"]]),
    })
    fields = {}
    sources = (
        (ice, ("aicen_init", "vicen_init", "aicen", "vicen", "vsnon",
               "aice0", "trcrn")),
        (flux, ("meltbn", "melttn", "congeln", "snoicen", "meltsn",
                "sss", "sst")),
        (bgc, ("hin_old",)),
        (column, ("first_ice", "iDi", "bphi", "bTiz", "dhbr_top",
                  "dhbr_bot", "darcy_V", "hbri")),
    )
    if options["skl_bgc"]:
        sources += (
            (flux, ("Tf",)),
            (column, ("fswthrun", "PP_net", "upNO", "upNH", "grow_net")),
        )
    if options["z_tracers"]:
        sources += (
            (ice, ("vsnon_init",)),
            (flux, ("fsnow",)),
            (column, ("fswpenln", "zfswin", "Zoo", "fbio_snoice",
                      "fbio_atmice", "ice_bio_net", "snow_bio_net",
                      "PP_net", "upNO", "upNH", "grow_net")),
        )
    for source, names in sources:
        for name in names:
            fields[name] = source[name][cells]
    for name in (("bgrid", "igrid", "cgrid", "icgrid")
                 if options["z_tracers"] else ("bgrid", "igrid", "cgrid")):
        fields[name] = jnp.broadcast_to(
            column[name], cell_shape + column[name].shape)
    ocean_fields = {
        name: bgc[name][cells]
        for name in ("nit", "amm", "sil", "dmsp", "dms", "algalN",
                     "doc", "don", "dic", "fed", "fep", "zaeros", "hum")
    }
    ocean_flat = {
        name: value.reshape((number_of_cells,) + value.shape[2:])
        for name, value in ocean_fields.items()
    }
    ocean_bio_all = column["ocean_bio_all"][cells]
    ocean_all_flat = ocean_bio_all.reshape(
        (number_of_cells,) + ocean_bio_all.shape[2:])
    packed_ocean = jax.vmap(lambda source, target:
        icepack_load_ocean_bio_array(
            source, target,
            bio_parameters["R_chl2N"], bio_parameters["R_S2N"], flags)
    )(ocean_flat, ocean_all_flat)
    valid = (grid["tmask"][cells] | grid["opmask"][cells]).reshape(-1)
    selected_ocean = jnp.where(valid[:, None], packed_ocean,
                               ocean_all_flat)
    column["ocean_bio_all"] = column["ocean_bio_all"].at[cells].set(
        selected_ocean.reshape(ocean_bio_all.shape))
    nbtrcr = int(flags["nbtrcr"])
    # bio_index_o retains Fortran's one-based ocean slots after initialization.
    ocean_indices = jnp.asarray(flags["bio_index_o"][:nbtrcr]) - 1
    selected_bio = selected_ocean[:, ocean_indices]
    old_bio = column["ocean_bio"][cells + (slice(0, nbtrcr),)]
    selected_bio = jnp.where(valid[:, None], selected_bio,
                             old_bio.reshape(number_of_cells, nbtrcr))
    column["ocean_bio"] = column["ocean_bio"].at[
        cells + (slice(0, nbtrcr),)
    ].set(selected_bio.reshape(cell_shape + (nbtrcr,)))
    if options["skl_bgc"] or options["z_tracers"]:
        fields["ocean_bio"] = column["ocean_bio"][cells + (slice(0, nbtrcr),)]
        fields["flux_bio"] = bgc["flux_bio"][cells + (slice(0, nbtrcr),)]
    if options["z_tracers"]:
        fields["flux_bio_atm"] = bgc["flux_bio_atm"][cells + (slice(0, nbtrcr),)]
    flattened = {
        name: value.reshape((number_of_cells,) + value.shape[2:])
        for name, value in fields.items()
    }
    advanced = jax.vmap(lambda cell: icepack_biogeochemistry(
        static["ice_calendar"]["dt"], cell, flags,
        stage_options, parameters,
        bio_parameters if options["skl_bgc"] or options["z_tracers"] else None,
        pytree["constants"]["ice_init_column"]["bgc_tracer_type"]
        if options["skl_bgc"] or options["z_tracers"] else None,
        pytree["constants"]["ice_init_column"]
        if options["z_tracers"] else None))(flattened)
    for source, names in (
        (ice, ("trcrn",)),
        (bgc, ("hin_old",)),
        (column, ("first_ice", "iDi", "bphi", "bTiz", "dhbr_top",
                  "dhbr_bot", "darcy_V", "hbri")),
    ):
        for name in names:
            original = fields[name]
            shape = (number_of_cells,) + (1,) * (original.ndim - 2)
            selected = jnp.where(valid.reshape(shape), advanced[name],
                                 flattened[name])
            source[name] = source[name].at[cells].set(
                selected.reshape(original.shape))
    if options["skl_bgc"] or options["z_tracers"]:
        for source, names in (
            (bgc, ("flux_bio",)),
            (column, ("PP_net", "upNO", "upNH", "grow_net")),
        ) + (((column, ("zfswin", "Zoo", "fbio_snoice", "fbio_atmice",
                       "ice_bio_net", "snow_bio_net")),)
             if options["z_tracers"] else ()):
            for name in names:
                original = fields[name]
                shape = (number_of_cells,) + (1,) * (original.ndim - 2)
                selected = jnp.where(valid.reshape(shape), advanced[name],
                                     flattened[name])
                source[name] = source[name].at[cells].set(
                    selected.reshape(original.shape))
        icep["bgc_failed"] = icep["bgc_failed"].at[cells].set(
            advanced["bgc_failed"].reshape(cell_shape)
            & valid.reshape(cell_shape))
    state.update({"ice_state": ice, "ice_flux_bgc": bgc,
                  "ice_arrays_column": column, "ice_step_mod": icep})
    result = dict(pytree)
    result["state"] = state
    return result


@jax.profiler.annotate_function
def step_therm2(pytree, static, timing=None):
    """Advance the horizontal cells through the second thermodynamic stage."""
    stage_started = time.perf_counter()
    flags = static["icepack_tracers"]
    options = dict(static["icepack_parameters"])
    if bool(options["skl_bgc"]) and not bool(options["z_tracers"]):
        raise NotImplementedError(
            "step_therm2 needs a decision on skeletal new-ice/lateral-melt BGC indexing")

    state = dict(pytree["state"])
    ice = dict(state["ice_state"])
    flux = dict(state["ice_flux"])
    bgc = dict(state["ice_flux_bgc"])
    column = dict(state["ice_arrays_column"])
    icep = dict(state["ice_step_mod"])
    grid = pytree["constants"]["ice_grid"]
    n_aero = int(flags["n_aero"]) if bool(flags["tr_aero"]) else 0
    n_iso = int(flags["n_iso"]) if bool(flags["tr_iso"]) else 0
    nbtrcr = int(flags["nbtrcr"])
    nfsd = int(static["ice_domain_size"]["nfsd"]) if bool(flags["tr_fsd"]) else 0
    options.update({
        "nilyr": int(static["ice_domain_size"]["nilyr"]),
        "nslyr": int(static["ice_domain_size"]["nslyr"]),
        "nblyr": int(static["ice_domain_size"]["nblyr"]),
        "nfsd": nfsd,
        "isotope_frac_method": "cfrac",
        "cleanup_category_unroll": icepack_itd.CLEANUP_CATEGORY_UNROLL,
        "rebin_static_tracers": icepack_itd.REBIN_STATIC_TRACERS,
    })
    parameters = dict(pytree["parameters"]["icepack_parameters"])
    parameters["Tmin"] = pytree["parameters"]["icepack_therm_shared"]["Tmin"]
    field_names = (
        ("aice", "aice0", "aicen", "vicen", "vsnon",
         "aicen_init", "vicen_init"),
        ("Tf", "sss", "rsiden", "meltl", "wlat", "frzmlt",
         "frazil", "frain", "fpond", "fresh", "fsalt", "fhocn",
         "frazil_diag", "frz_onset", "dpnd_melt", "salinz"),
        ("HDO_ocn", "H2_16O_ocn", "H2_18O_ocn"),
        ("first_ice", "wave_sig_ht", "wave_spectrum",
         "d_afsd_latg", "d_afsd_newi", "d_afsd_latm", "d_afsd_weld"),
    )
    floe_rad_h = (pytree["constants"]["icepack_fsd"]["floe_rad_h"]
                  if nfsd else jnp.zeros((0,), dtype=ice["aice"].dtype))
    flattened, column = _therm2_prepare(
        ice, flux, bgc, column, grid, floe_rad_h,
        static["ice_calendar"]["yday"], static["ice_domain_size"]["physical_bounds"],
        (n_aero, n_iso, nbtrcr),
        bool(nfsd and str(options["wave_height_type"]).strip() == "internal"),
        field_names)
    ice_timers.record_stage(timing, "physics.therm2.prepare", stage_started, flattened)
    stage_started = time.perf_counter()

    def advance_columns(cell, column_call=None):
        return icepack_step_therm2(
            static["ice_calendar"]["dt"], column["hin_max"], cell,
            static["ice_state_tracers"], flags, options, parameters,
            static["icepack_tracers"]["bio_index"]
            if "bio_index" in static["icepack_tracers"] else (),
            column["floe_rad_c"] if nfsd else None,
            column["floe_binwidth"] if nfsd else None,
            pytree["constants"]["icepack_fsd"]["floe_area_c"]
            if nfsd else None,
            pytree["constants"]["icepack_fsd"]["iweld"]
            if nfsd else None,
            pytree["constants"]["ice_init_column"]["zbgc_init_frac"], column_call=column_call)

    if THERM2_BATCH_JIT:
        config = tuple(tuple(sorted(values.items())) for values in (
            static["ice_state_tracers"], flags, options))
        advanced, failed, failure_stages, area_stages = _therm2_columns_compiled(
            flattened, static["ice_calendar"]["dt"], column["hin_max"], parameters, config,
            (column["floe_rad_c"] if nfsd else None,
             column["floe_binwidth"] if nfsd else None,
             pytree["constants"]["icepack_fsd"]["floe_area_c"] if nfsd else None,
             pytree["constants"]["icepack_fsd"]["iweld"] if nfsd else None,
             pytree["constants"]["ice_init_column"]["zbgc_init_frac"]))
    elif timing is None:
        advanced, failed, failure_stages, area_stages = jax.vmap(advance_columns)(flattened)
    else:
        advanced, failed, failure_stages, area_stages = advance_columns(
            flattened, partial(_timed_column_call, timing, "physics.therm2.columns"))
    ice_timers.record_stage(timing, "physics.therm2.columns", stage_started,
                            (advanced, failed, failure_stages, area_stages))
    stage_started = time.perf_counter()
    ice, flux, bgc, column, icep = _therm2_writeback(
        ice, flux, bgc, column, icep, grid, flattened, advanced,
        failed, failure_stages, area_stages,
        static["ice_domain_size"]["physical_bounds"], (n_aero, n_iso, nbtrcr), field_names)
    state.update({"ice_state": ice, "ice_flux": flux,
                  "ice_flux_bgc": bgc, "ice_arrays_column": column,
                  "ice_step_mod": icep})
    result = dict(pytree)
    result["state"] = state
    ice_timers.record_stage(timing, "physics.therm2.writeback", stage_started, result)
    return result


@jax.profiler.annotate_function
@partial(jax.jit, static_argnames=("config",))
def _therm2_columns_compiled(fields, dt, hin_max, parameters, config, fsd_inputs):
    """Compile the full batched therm2 sequence, including interstage work."""
    layout, flags, options = (dict(values) for values in config)
    floe_rad_c, floe_binwidth, floe_area_c, iweld, zbgc_init_frac = fsd_inputs
    return jax.vmap(lambda cell: icepack_step_therm2(
        dt, hin_max, cell, layout, flags, options, parameters,
        flags.get("bio_index", ()), floe_rad_c, floe_binwidth,
        floe_area_c, iweld, zbgc_init_frac))(fields)


@partial(jax.jit, static_argnames=("bounds", "counts", "internal_waves", "field_names"))
def _therm2_prepare(ice, flux, bgc, column, grid, floe_rad_h, yday,
                    bounds, counts, internal_waves, field_names):
    """Compile the slices and broadcasts feeding the column solvers together."""
    column = dict(column)
    ilo, ihi, jlo, jhi = bounds
    cells = (slice(ilo, ihi + 1), slice(jlo, jhi + 1))
    cell_shape = (ihi - ilo + 1, jhi - jlo + 1)
    number_of_cells = cell_shape[0] * cell_shape[1]
    n_aero, n_iso, nbtrcr = counts
    if internal_waves:
        wave_height = 4.0 * jnp.sqrt(jnp.sum(
            column["wave_spectrum"][cells]
            * column["dwavefreq"][None, None, :], axis=-1))
        valid_cells = grid["tmask"][cells] | grid["opmask"][cells]
        column["wave_sig_ht"] = column["wave_sig_ht"].at[cells].set(
            jnp.where(valid_cells, wave_height, column["wave_sig_ht"][cells]))
    fields = {}
    for source, names in zip((ice, flux, bgc, column), field_names):
        for name in names:
            fields[name] = source[name][cells]
    fields["trcrn"] = ice["trcrn"][cells]
    fields["faero_ocn"] = bgc["faero_ocn"][cells + (slice(0, n_aero),)]
    fields["fiso_ocn"] = bgc["fiso_ocn"][cells + (slice(0, n_iso),)]
    fields["flux_bio"] = bgc["flux_bio"][cells + (slice(0, nbtrcr),)]
    fields["ocean_bio"] = column["ocean_bio"][cells + (slice(0, nbtrcr),)]
    fields["yday"] = jnp.full(cell_shape, yday)
    fields["wavefreq"] = jnp.broadcast_to(
        column["wavefreq"], cell_shape + column["wavefreq"].shape)
    fields["floe_rad_h"] = jnp.broadcast_to(
        floe_rad_h, cell_shape + floe_rad_h.shape)
    return {
        name: value.reshape((number_of_cells,) + value.shape[2:])
        for name, value in fields.items()
    }, column


def _therm2_writeback(ice, flux, bgc, column, icep, grid, flattened, advanced,
                      failed, failure_stages, area_stages, bounds, counts, field_names):
    updates = _therm2_writeback_compiled(
        ice, flux, bgc, column, icep, grid, flattened, advanced,
        failed, failure_stages, area_stages, bounds, counts, field_names)
    return tuple({**source, **changes} for source, changes in zip(
        (ice, flux, bgc, column, icep), updates))


@partial(jax.jit, static_argnames=("bounds", "counts", "field_names"))
def _therm2_writeback_compiled(ice, flux, bgc, column, icep, grid, flattened, advanced,
                               failed, failure_stages, area_stages, bounds, counts,
                               field_names):
    """Fuse masking and state updates while retaining all failure diagnostics."""
    ice_updates, flux_updates, bgc_updates, column_updates, icep_updates = {}, {}, {}, {}, {}
    ilo, ihi, jlo, jhi = bounds
    cells = (slice(ilo, ihi + 1), slice(jlo, jhi + 1))
    cell_shape = (ihi - ilo + 1, jhi - jlo + 1)
    number_of_cells = cell_shape[0] * cell_shape[1]
    n_aero, n_iso, nbtrcr = counts
    valid = (grid["tmask"][cells] | grid["opmask"][cells]).reshape(-1)
    for source, changes, names in zip(
        (ice, flux, bgc, column),
        (ice_updates, flux_updates, bgc_updates, column_updates), field_names
    ):
        for name in names:
            if name in advanced and source[name].size:
                original = flattened[name]
                shape = (number_of_cells,) + (1,) * (original.ndim - 1)
                selected = jnp.where(
                    valid.reshape(shape), advanced[name], original)
                changes[name] = source[name].at[cells].set(
                    selected.reshape(cell_shape + original.shape[1:]))
    original = flattened["trcrn"]
    selected = jnp.where(valid[:, None, None], advanced["trcrn"],
                         original)
    ice_updates["trcrn"] = ice["trcrn"].at[cells].set(
        selected.reshape(cell_shape + original.shape[1:]))
    for name in ("faero_ocn", "fiso_ocn", "flux_bio"):
        count = {"faero_ocn": n_aero, "fiso_ocn": n_iso,
                 "flux_bio": nbtrcr}[name]
        if count:
            selected = jnp.where(valid[:, None], advanced[name], flattened[name])
            bgc_updates[name] = bgc[name].at[cells + (slice(0, count),)].set(
                selected.reshape(cell_shape + (count,)))
    icep_updates["therm2_failed"] = icep["therm2_failed"].at[cells].set(
        failed.reshape(cell_shape) & valid.reshape(cell_shape))
    for name, stage_failed in zip(
        ("therm2_itd_failed", "therm2_new_ice_failed",
         "therm2_weld_failed", "therm2_cleanup_failed",
         "therm2_cleanup_initial_area_failed",
         "therm2_cleanup_rebin_failed",
         "therm2_cleanup_negative_area_failed",
         "therm2_cleanup_excess_area_failed"), failure_stages
    ):
        icep_updates[name] = icep[name].at[cells].set(
            stage_failed.reshape(cell_shape) & valid.reshape(cell_shape))
    for name, area in zip(
        ("therm2_area_entry", "therm2_area_after_itd",
         "therm2_area_after_new_ice", "therm2_area_after_lateral_melt",
         "therm2_area_after_cleanup"), area_stages
    ):
        icep_updates[name] = icep[name].at[cells].set(area.reshape(cell_shape))
    return ice_updates, flux_updates, bgc_updates, column_updates, icep_updates


def step_dyn_wave(pytree, static):
    """Fracture the floe-size distribution in every physical grid cell."""
    state = dict(pytree["state"])
    ice = dict(state["ice_state"])
    column = dict(state["ice_arrays_column"])
    icep = dict(state["ice_step_mod"])
    flags = static["icepack_tracers"]
    options = static["icepack_parameters"]
    parameters = pytree["parameters"]["icepack_parameters"]
    ilo, ihi, jlo, jhi = static["ice_domain_size"]["physical_bounds"]
    cells = (slice(ilo, ihi + 1), slice(jlo, jhi + 1))
    cell_shape = ice["aice"][cells].shape
    number_of_cells = cell_shape[0] * cell_shape[1]
    trcrn = ice["trcrn"][cells]
    aicen = ice["aicen"][cells]
    aice = ice["aice"][cells]
    vice = ice["vice"][cells]
    spectrum = column["wave_spectrum"][cells]
    height = column["wave_sig_ht"][cells]
    random_spectrum = str(options["wave_spec_type"]).strip() == "random"
    if random_spectrum:
        split_keys = jax.random.split(
            state["wave_rng"]["key"], number_of_cells + 1)
        wave_rng = dict(state["wave_rng"])
        wave_rng["key"] = split_keys[0]
        state["wave_rng"] = wave_rng
        cell_keys = split_keys[1:]
    else:
        cell_keys = jax.random.split(jax.random.key(0), number_of_cells)

    advanced, d_afsd_wave, _, warning = jax.vmap(
        lambda ai, vi, an, spec, tracers, wave_height, cell_key:
        icepack_step_wavefracture(
            static["ice_calendar"]["dt"], ai, vi, an, spec,
            column["wavefreq"], column["dwavefreq"], tracers,
            wave_height, column["floe_rad_l"], column["floe_rad_c"],
            int(flags["nt_fsd"]), str(options["wave_spec_type"]).strip(),
            parameters, cell_key)
    )(aice.reshape(number_of_cells), vice.reshape(number_of_cells),
      aicen.reshape((number_of_cells,) + aicen.shape[2:]),
      spectrum.reshape((number_of_cells,) + spectrum.shape[2:]),
      trcrn.reshape((number_of_cells,) + trcrn.shape[2:]),
      height.reshape(number_of_cells), cell_keys)
    ice["trcrn"] = ice["trcrn"].at[cells].set(advanced.reshape(trcrn.shape))
    column["d_afsd_wave"] = column["d_afsd_wave"].at[cells].set(
        d_afsd_wave.reshape(column["d_afsd_wave"][cells].shape))
    icep["wave_failed"] = icep["wave_failed"].at[cells].set(
        warning.reshape(cell_shape))
    state["ice_state"] = ice
    state["ice_arrays_column"] = column
    state["ice_step_mod"] = icep
    result = dict(pytree)
    result["state"] = state
    return result


@jax.profiler.annotate_function
def step_dyn_horiz(pytree, static, timing=None):
    """Momentum solve, ice-ocean stress average, and horizontal transport."""
    dt = static["ice_calendar"]["dt_dyn"]
    stage_started = time.perf_counter()
    with jax.profiler.TraceAnnotation("physics.horizontal.momentum"):
        pytree = ice_flux.init_history_dyn(pytree, static)
        kdyn = int(static["ice_dyn_shared"]["kdyn"])
        if kdyn == 1:
            pytree = ice_dyn_evp.evp(pytree, static, dt)
        elif kdyn == 2:
            pytree = ice_dyn_eap.eap(pytree, static, dt)
        elif kdyn == 3:
            pytree = ice_dyn_vp.implicit_solver(pytree, static, dt)
        elif kdyn > 3:
            raise ValueError("unknown kdyn: " + str(kdyn))
        ice_timers.record_stage(timing, "physics.horizontal.momentum", stage_started, pytree)
    stage_started = time.perf_counter()

    stress_config = tuple((name, tuple(sorted(static[name].items()))) for name in (
        "ice_domain_size", "ice_domain"))
    stress_updates = _horizontal_stress_compiled(pytree, stress_config)
    state = dict(pytree["state"])
    state["ice_flux"] = {**state["ice_flux"], **stress_updates}
    result = dict(pytree)
    result["state"] = state
    ice_timers.record_stage(timing, "physics.horizontal.stress", stage_started, result)
    # With kdyn <= 0, transport uses the existing velocity, as in Fortran.
    advection = static["ice_init"]["advection"].strip()
    stage_started = time.perf_counter()
    if advection == "upwind":
        result = ice_transport_driver.transport_upwind(result, static, dt)
    elif advection == "remap":
        result = ice_transport_driver.transport_remap(result, static, dt)
    elif advection != "none":
        raise ValueError("unknown advection: " + advection)
    ice_timers.record_stage(timing, "physics.horizontal.transport", stage_started, result)
    return result


@jax.profiler.annotate_function
@partial(jax.jit, static_argnames=("config",))
def _horizontal_stress_compiled(pytree, config):
    """Fuse ice-area normalization, U-grid halos and T-grid interpolation."""
    static = {name: dict(values) for name, values in config}
    # Fortran step_dyn_horiz: divide U-grid ocean stress by ice area before
    # averaging it to T points for thermodynamics and coupling.
    state = dict(pytree["state"])
    flux = dict(state["ice_flux"])
    ice = state["ice_state"]
    grid = pytree["constants"]["ice_grid"]
    bounds = static["ice_domain_size"]["physical_bounds"]
    iglob = pytree["constants"]["ice_boundary"]["iglob"]
    aiU = ice["aiU"]
    ilo, ihi, jlo, jhi = bounds
    physical = jnp.zeros_like(aiU, dtype=bool).at[
        ilo:ihi + 1, jlo:jhi + 1].set(True)
    nonzero = physical & (aiU != 0.0)
    work1 = jnp.where(nonzero,
                      flux["strocnxU"] / jnp.where(nonzero, aiU, 1.0), 0.0)
    work2 = jnp.where(nonzero,
                      flux["strocnyU"] / jnp.where(nonzero, aiU, 1.0), 0.0)
    work1 = ice_halo_update_jax(work1, "U", "vector", static, iglob)
    work2 = ice_halo_update_jax(work2, "U", "vector", static, iglob)
    flux["strocnxT_iavg"] = grid_average_X2Y_jax(
        "F", work1, "U", flux["strocnxT_iavg"], "T", grid, bounds)
    flux["strocnyT_iavg"] = grid_average_X2Y_jax(
        "F", work2, "U", flux["strocnyT_iavg"], "T", grid, bounds)
    return {name: flux[name] for name in ("strocnxT_iavg", "strocnyT_iavg")}


def step_dyn_ridge(pytree, static, timing=None):
    """Apply mechanical redistribution and category cleanup on ocean cells."""
    stage_started = time.perf_counter()
    state = dict(pytree["state"])
    ice = dict(state["ice_state"])
    flux = dict(state["ice_flux"])
    bgc = dict(state["ice_flux_bgc"])
    column = dict(state["ice_arrays_column"])
    icep = dict(state["ice_step_mod"])
    flags = static["icepack_tracers"]
    options = dict(static["icepack_parameters"])
    options.update({
        "nilyr": int(static["ice_domain_size"]["nilyr"]),
        "nslyr": int(static["ice_domain_size"]["nslyr"]),
        "nblyr": int(static["ice_domain_size"]["nblyr"]),
        "cleanup_category_unroll": icepack_itd.CLEANUP_CATEGORY_UNROLL,
        "rebin_static_tracers": icepack_itd.REBIN_STATIC_TRACERS,
        "ridge_batch_stop": RIDGE_BATCH_STOP,
    })
    parameters = dict(pytree["parameters"]["icepack_parameters"])
    parameters["Tmin"] = pytree["parameters"]["icepack_therm_shared"]["Tmin"]
    n_aero = int(flags["n_aero"]) if bool(flags["tr_aero"]) else 0
    n_iso = int(flags["n_iso"]) if bool(flags["tr_iso"]) else 0
    nbtrcr = int(flags["nbtrcr"])
    field_names = (
        ("aice", "aice0", "aicen", "vicen", "vsnon", "trcrn"),
        ("rdg_conv", "rdg_shear", "dardg1dt", "dardg2dt",
         "dvirdgdt", "opening", "fpond", "fresh", "fsalt", "fhocn",
         "dpnd_ridge", "aparticn", "krdgn", "aredistn", "vredistn",
         "dardg1ndt", "dardg2ndt", "dvirdgndt", "araftn", "vraftn", "Tf"),
        ("first_ice",),
    )
    flattened = _ridge_prepare(
        ice, flux, bgc, column, static["ice_domain_size"]["physical_bounds"],
        (n_aero, n_iso, nbtrcr), field_names)
    ice_timers.record_stage(timing, "physics.ridge.prepare", stage_started, flattened)
    stage_started = time.perf_counter()

    def advance_columns(cell, column_call=None):
        return icepack_step_ridge(
            static["ice_calendar"]["dt_dyn"],
            int(static["ice_calendar"]["ndtd"]), column["hin_max"], cell,
            static["ice_state_tracers"], flags, options, parameters,
            flags["bio_index"] if "bio_index" in flags else (), column_call=column_call)

    if RIDGE_BATCH_JIT:
        config = tuple(tuple(sorted(values.items())) for values in (
            static["ice_state_tracers"], flags, options))
        advanced, failed = _ridge_columns_compiled(
            flattened, static["ice_calendar"]["dt_dyn"],
            int(static["ice_calendar"]["ndtd"]), column["hin_max"], parameters, config)
    elif timing is None:
        advanced, failed = jax.vmap(advance_columns)(flattened)
    else:
        advanced, failed = advance_columns(
            flattened, partial(_timed_column_call, timing, "physics.ridge.columns"))
    ice_timers.record_stage(timing, "physics.ridge.columns", stage_started, (advanced, failed))
    stage_started = time.perf_counter()
    updates = _ridge_writeback(
        ice, flux, bgc, column, icep, pytree["constants"]["ice_grid"]["tmask"],
        flattened, advanced, failed, static["ice_domain_size"]["physical_bounds"],
        (n_aero, n_iso, nbtrcr), field_names)
    for name, original, changes in zip(
            ("ice_state", "ice_flux", "ice_flux_bgc", "ice_arrays_column", "ice_step_mod"),
            (ice, flux, bgc, column, icep), updates):
        state[name] = {**original, **changes}
    result = dict(pytree)
    result["state"] = state
    ice_timers.record_stage(timing, "physics.ridge.writeback", stage_started, result)
    return result


@jax.profiler.annotate_function
@partial(jax.jit, static_argnames=("config",))
def _ridge_columns_compiled(fields, dt, ndtd, hin_max, parameters, config):
    """Compile ridging and cleanup together across all physical columns."""
    layout, flags, options = (dict(values) for values in config)
    axis_name = "ridge_cells" if options.get("ridge_batch_stop", False) else None
    if axis_name is not None:
        options["ridge_axis_name"] = axis_name
    return jax.vmap(lambda cell: icepack_step_ridge(
        dt, ndtd, hin_max, cell, layout, flags, options, parameters,
        flags.get("bio_index", ())), axis_name=axis_name)(fields)


@jax.profiler.annotate_function
@partial(jax.jit, static_argnames=("bounds", "counts", "field_names"))
def _ridge_prepare(ice, flux, bgc, column, bounds, counts, field_names):
    """Pack physical cells and optional tracers in one compiled operation."""
    ilo, ihi, jlo, jhi = bounds
    cells = (slice(ilo, ihi + 1), slice(jlo, jhi + 1))
    number_of_cells = (ihi - ilo + 1) * (jhi - jlo + 1)
    n_aero, n_iso, nbtrcr = counts
    fields = {}
    for source, names in zip((ice, flux, column), field_names):
        for name in names:
            fields[name] = source[name][cells]
    for name, count in (("faero_ocn", n_aero),
                        ("fiso_ocn", n_iso), ("flux_bio", nbtrcr)):
        fields[name] = bgc[name][cells + (slice(0, count),)]
    return {
        name: value.reshape((number_of_cells,) + value.shape[2:])
        for name, value in fields.items()
    }


@jax.profiler.annotate_function
@partial(jax.jit, static_argnames=("bounds", "counts", "field_names"))
def _ridge_writeback(ice, flux, bgc, column, icep, tmask, flattened, advanced,
                      failed, bounds, counts, field_names):
    """Fuse ocean masking and ridge output updates in one compiled operation."""
    original_groups = (ice, flux, bgc, column, icep)
    ice, flux, bgc, column, icep = (
        dict(ice), dict(flux), dict(bgc), dict(column), dict(icep))
    ilo, ihi, jlo, jhi = bounds
    cells = (slice(ilo, ihi + 1), slice(jlo, jhi + 1))
    cell_shape = (ihi - ilo + 1, jhi - jlo + 1)
    number_of_cells = cell_shape[0] * cell_shape[1]
    n_aero, n_iso, nbtrcr = counts
    valid = tmask[cells].reshape(-1)
    for source, names in zip((ice, flux, column), field_names):
        for name in names:
            if name in advanced:
                original = flattened[name]
                shape = (number_of_cells,) + (1,) * (original.ndim - 1)
                selected = jnp.where(valid.reshape(shape),
                                     advanced[name], original)
                source[name] = source[name].at[cells].set(
                    selected.reshape(cell_shape + original.shape[1:]))
    for name, count in (("faero_ocn", n_aero),
                        ("fiso_ocn", n_iso), ("flux_bio", nbtrcr)):
        original = flattened[name]
        selected = jnp.where(valid[:, None], advanced[name], original)
        bgc[name] = bgc[name].at[cells + (slice(0, count),)].set(
            selected.reshape(cell_shape + original.shape[1:]))
    column["hin_max"] = column["hin_max"].at[-1].set(1.0e8)
    icep["ridge_failed"] = icep["ridge_failed"].at[cells].set(
        failed.reshape(cell_shape) & valid.reshape(cell_shape))
    return tuple({name: value for name, value in after.items()
                  if value is not before[name]}
                 for before, after in zip(original_groups, (ice, flux, bgc, column, icep)))


@jax.profiler.annotate_function
def update_state(pytree, static, dt, offset=None, tendency=None):
    """Update category ghosts and aggregate the ice state over categories.

    Fortran calls this after thermodynamics, each dynamics substep and
    optionally after advanced snow physics. The tendency arrays contain
    the beginning-of-stage values before this call.
    """
    config = tuple((name, tuple(sorted(static[name].items()))) for name in (
        "ice_domain_size", "ice_domain", "ice_state_tracers",
        "icepack_tracers", "icepack_parameters"))
    state = dict(pytree["state"])
    ice = {name: state["ice_state"][name]
           for name in ("aicen", "trcrn", "vicen", "vsnon")}
    flux_names = ["Tf"]
    if offset is not None:
        suffix = "t" if tendency == "thermo" else "d"
        flux_names.extend(name + suffix for name in ("daidt", "dvidt", "dvsdt"))
        if bool(static["icepack_tracers"]["tr_iage"]):
            flux_names.append("dagedt" + suffix)
    flux = {name: state["ice_flux"][name] for name in flux_names}
    parameter_names = ["puny"]
    if str(static["icepack_parameters"]["snwredist"])[:3] == "ITD":
        parameter_names.append("rhosnew")
    if bool(static["icepack_parameters"]["snwgrain"]):
        parameter_names.extend(("rsnw_fall", "rhos"))
    parameters = {name: pytree["parameters"]["icepack_parameters"][name]
                  for name in parameter_names}
    ice_updates, flux_updates = _update_state_compiled(
        ice, flux, parameters, pytree["constants"]["ice_boundary"]["iglob"],
        config, dt, offset, tendency, offset is not None and offset > 0)
    state["ice_state"] = {**state["ice_state"], **ice_updates}
    state["ice_flux"] = {**state["ice_flux"], **flux_updates}
    result = dict(pytree)
    result["state"] = state
    return result


@partial(jax.jit, static_argnames=("config", "tendency", "positive_offset"))
def _update_state_compiled(ice, flux, parameters, iglob,
                           config, dt, offset, tendency, positive_offset):
    # Select the age-tendency branch on the host; dt and offset stay dynamic.
    static = {name: dict(values) for name, values in config}
    ice = dict(ice)
    flux = dict(flux)
    for name in ("aicen", "trcrn", "vicen", "vsnon"):
        ice[name] = ice_halo_update_center_jax(ice[name], static, iglob)

    aice, trcr, vice, vsno, aice0 = icepack_aggregate_jax(
        ice["aicen"], ice["trcrn"], ice["vicen"], ice["vsnon"],
        flux["Tf"], static["ice_state_tracers"], static["icepack_tracers"],
        static["icepack_parameters"], parameters,
    )
    ice["aice"] = aice
    ice["trcr"] = trcr
    ice["vice"] = vice
    ice["vsno"] = vsno
    ice["aice0"] = aice0

    if offset is not None:
        suffix = "t" if tendency == "thermo" else "d"
        for name, current in (("daidt", aice), ("dvidt", vice), ("dvsdt", vsno)):
            key = name + suffix
            flux[key] = (current - flux[key]) / dt
        if bool(static["icepack_tracers"]["tr_iage"]):
            age = trcr[:, :, int(static["icepack_tracers"]["nt_iage"])]
            key = "dagedt" + suffix
            if positive_offset:
                flux[key] = jnp.where(age > 0, (age - flux[key] - offset) / dt, flux[key])
            else:
                flux[key] = (age - flux[key]) / dt

    del flux["Tf"]
    return ice, flux


def step_snow(pytree, static):
    """Redistribute snow, age grains and count snow-covered cells."""
    options = static["icepack_parameters"]
    snwredist = str(options["snwredist"]).strip()
    if snwredist.startswith("ITD") and snwredist not in ("ITDsd", "ITDrdg"):
        raise ValueError("(step_snow) unsupported snwredist: " + snwredist)

    state = dict(pytree["state"])
    flux = dict(state["ice_flux"])
    ice = dict(state["ice_state"])
    icep = dict(state["ice_step_mod"])
    params = pytree["parameters"]["icepack_parameters"]
    puny = params["puny"]
    ilo, ihi, jlo, jhi = static["ice_domain_size"]["physical_bounds"]
    nstreams = int(static["ice_calendar"]["nstreams"])
    cells = (slice(ilo, ihi + 1), slice(jlo, jhi + 1))
    tracers = static["icepack_tracers"]
    nslyr = int(static["ice_domain_size"]["nslyr"])
    dt = static["ice_calendar"]["dt"]
    initial_snow_mass = (params["rhos"] * jnp.sum(ice["vsnon"][cells], axis=-1)
                         + flux["fresh"][cells] * dt)
    mass_error = jnp.zeros_like(initial_snow_mass, dtype=bool)
    energy_error = jnp.zeros_like(initial_snow_mass, dtype=bool)

    if snwredist.startswith("ITD"):
        nt_qsno = int(tracers["nt_qsno"])
        nt_rhos = int(tracers["nt_rhos"])
        qsn = ice["trcrn"][cells + (slice(nt_qsno, nt_qsno + nslyr), slice(None))]
        rhos_cmpn = ice["trcrn"][cells + (slice(nt_rhos, nt_rhos + nslyr), slice(None))]
        rhos_cmpn = jnp.where(rhos_cmpn < params["rhosmin"],
                               params["rhosnew"], rhos_cmpn)
        if snwredist == "ITDrdg":
            alvl = ice["trcrn"][cells + (int(tracers["nt_alvl"]), slice(None))]
            vlvl = ice["trcrn"][cells + (int(tracers["nt_vlvl"]), slice(None))]
        else:
            alvl = jnp.zeros_like(ice["aicen"][cells])
            vlvl = jnp.zeros_like(ice["aicen"][cells])
        (vsnon, qsn, fresh, fhocn, fsloss, rhos_cmpn,
         mass_error, energy_error) = snow_redist(
            dt, flux["wind"][cells], ice["aice"][cells],
            ice["aicen"][cells], ice["vicen"][cells], ice["vsnon"][cells],
            qsn, alvl, vlvl, flux["fresh"][cells], flux["fhocn"][cells],
            flux["fsloss"][cells], rhos_cmpn, flux["fsnow"][cells],
            snwredist, params)
        ice["vsnon"] = ice["vsnon"].at[cells].set(vsnon)
        ice["trcrn"] = ice["trcrn"].at[
            cells + (slice(nt_qsno, nt_qsno + nslyr), slice(None))
        ].set(qsn)
        ice["trcrn"] = ice["trcrn"].at[
            cells + (slice(nt_rhos, nt_rhos + nslyr), slice(None))
        ].set(rhos_cmpn)
        for name, value in (("fresh", fresh), ("fhocn", fhocn),
                            ("fsloss", fsloss)):
            flux[name] = flux[name].at[cells].set(value)
    final_snow_mass = (params["rhos"] * jnp.sum(ice["vsnon"][cells], axis=-1)
                       + flux["fresh"][cells] * dt)
    mass_error = mass_error | (jnp.abs(initial_snow_mass - final_snow_mass) > puny)
    icep["snow_mass_error"] = icep["snow_mass_error"].at[cells].set(mass_error)
    icep["snow_energy_error"] = icep["snow_energy_error"].at[cells].set(energy_error)

    if bool(options["snwgrain"]):
        nilyr = int(static["ice_domain_size"]["nilyr"])
        ktherm = int(options["ktherm"])
        l_brine = bool(static["icepack_therm_shared"]["l_brine"])
        nt_qsno = int(tracers["nt_qsno"])
        nt_rsnw = int(tracers["nt_rsnw"])
        nt_smice = int(tracers["nt_smice"])
        nt_smliq = int(tracers["nt_smliq"])
        for n in range(ice["aicen"].shape[-1]):
            area = ice["aicen"][cells + (n,)]
            present = area > puny
            hin = jnp.where(present,
                            ice["vicen"][cells + (n,)] /
                            jnp.where(present, area, 1.0), 0.0)
            hsn = jnp.where(present,
                            ice["vsnon"][cells + (n,)] /
                            jnp.where(present, area, 1.0), 0.0)
            qice = ice["trcrn"][cells + (int(tracers["nt_qice"]), n)]
            sice = ice["trcrn"][cells + (int(tracers["nt_sice"]), n)]
            qice = jnp.where(present, qice, -params["rhoi"] * params["Lfresh"])
            sice = jnp.where(present, sice, params["min_salin"])
            zTin1, _ = icepack_ice_temperature_jax(
                qice, sice, ktherm, l_brine, params)
            zTin1 = jnp.where(present, zTin1, 0.0)
            Tsfc = ice["trcrn"][cells + (int(tracers["nt_Tsfc"]), n)]
            qsn = ice["trcrn"][cells + (slice(nt_qsno, nt_qsno + nslyr), n)]
            rsnw = ice["trcrn"][cells + (slice(nt_rsnw, nt_rsnw + nslyr), n)]
            smice = ice["trcrn"][cells + (slice(nt_smice, nt_smice + nslyr), n)]
            smliq = ice["trcrn"][cells + (slice(nt_smliq, nt_smliq + nslyr), n)]
            rsnw = jnp.maximum(params["rsnw_fall"], rsnw)
            # Snow layers are the third axis in trcrn and the last axis in
            # the one-category Icepack grain-growth routine.
            rsnw, smice, smliq = update_snow_radius(
                dt, rsnw, hin, Tsfc, zTin1, hsn, qsn,
                smice, smliq, nilyr, params)
            ice["trcrn"] = ice["trcrn"].at[
                cells + (slice(nt_rsnw, nt_rsnw + nslyr), n)
            ].set(rsnw)
            ice["trcrn"] = ice["trcrn"].at[
                cells + (slice(nt_smice, nt_smice + nslyr), n)
            ].set(smice)
            ice["trcrn"] = ice["trcrn"].at[
                cells + (slice(nt_smliq, nt_smliq + nslyr), n)
            ].set(smliq)

    # Fortran counts the aggregate vsno from the preceding update_state;
    # the update after step_snow has not happened yet.
    snow = jnp.where(ice["vsno"] > puny, 1.0, 0.0)
    flux["snwcnt"] = flux["snwcnt"].at[
        ilo:ihi + 1, jlo:jhi + 1, :nstreams
    ].add(snow[ilo:ihi + 1, jlo:jhi + 1, None])
    state["ice_flux"] = flux
    state["ice_state"] = ice
    state["ice_step_mod"] = icep
    result = dict(pytree)
    result["state"] = state
    return result


@jax.profiler.annotate_function
def step_radiation(pytree, static):
    """Update shortwave albedo and absorption in the physical T cells."""
    config = tuple((name, tuple(sorted(static[name].items()))) for name in (
        "ice_domain_size", "icepack_parameters", "icepack_tracers"))
    calendar = static["ice_calendar"]
    column_updates, flux_updates = _step_radiation_compiled(
        pytree, config, calendar["dt"], calendar["yday"], calendar["msec"])
    state = dict(pytree["state"])
    state["ice_arrays_column"] = {**state["ice_arrays_column"], **column_updates}
    state["ice_flux"] = {**state["ice_flux"], **flux_updates}
    result = dict(pytree)
    result["state"] = state
    return result


# This executable still triggers CUDA graph replay faults on the WSL driver.
@partial(jax.jit, static_argnames=("config",),
         compiler_options={"xla_gpu_enable_command_buffer": ""})
def _step_radiation_compiled(pytree, config, dt, yday, msec):
    # Include input packing and masked writeback in the cached batched executable.
    static = {name: dict(values) for name, values in config}
    static["ice_calendar"] = {"dt": dt, "yday": yday, "msec": msec}
    state = dict(pytree["state"])
    column = dict(state["ice_arrays_column"])
    flux = dict(state["ice_flux"])
    ice = state["ice_state"]
    grid = pytree["constants"]["ice_grid"]
    options = static["icepack_parameters"]
    tracers = static["icepack_tracers"]
    params = pytree["parameters"]["icepack_parameters"]
    ilo, ihi, jlo, jhi = static["ice_domain_size"]["physical_bounds"]
    cells = (slice(ilo, ihi + 1), slice(jlo, jhi + 1))
    active = grid["tmask"][cells] | grid["opmask"][cells]
    category_active = active[..., None]
    layer_active = active[..., None, None]
    present = ice["aicen"][cells] > params["puny"]

    if bool(options.get("calc_Tsfc", True)):
        shortwave = str(options["shortwave"]).strip()
        if shortwave.startswith("ccsm"):
            nt_Tsfc = int(tracers["nt_Tsfc"])
            Tsfcn = ice["trcrn"][cells + (nt_Tsfc, slice(None))]
            (alvdrn, alvdfn, alidrn, alidfn, albicen, albsnon,
             fswsfcn, fswintn, fswthrun, fswpenln, Iswabsn,
             spectral) = shortwave_ccsm3_jax(
                ice["aicen"][cells], ice["vicen"][cells],
                ice["vsnon"][cells], Tsfcn,
                flux["swvdr"][cells], flux["swvdf"][cells],
                flux["swidr"][cells], flux["swidf"][cells],
                flux["swuvrdr"][cells], flux["swuvrdf"][cells],
                flux["swpardr"][cells], flux["swpardf"][cells],
                options, params, int(static["ice_domain_size"]["nilyr"]),
            )
            for name, value in (
                ("alvdrn", alvdrn), ("alvdfn", alvdfn),
                ("alidrn", alidrn), ("alidfn", alidfn),
                ("albicen", albicen), ("albsnon", albsnon),
                ("fswsfcn", fswsfcn), ("fswintn", fswintn),
                ("fswthrun", fswthrun),
            ):
                old = column[name][cells]
                column[name] = column[name].at[cells].set(
                    jnp.where(category_active, value, old))
            for name, value in (("fswpenln", fswpenln),
                                ("Iswabsn", Iswabsn)):
                old = column[name][cells]
                column[name] = column[name].at[cells].set(
                    jnp.where(layer_active, value, old))
            column["Sswabsn"] = column["Sswabsn"].at[cells].set(
                jnp.where(layer_active, 0.0, column["Sswabsn"][cells]))
            for name, value in zip(
                ("fswthrun_vdr", "fswthrun_vdf", "fswthrun_idr",
                 "fswthrun_idf", "fswthrun_uvrdr", "fswthrun_uvrdf",
                 "fswthrun_pardr", "fswthrun_pardf"), spectral,
            ):
                old = column[name][cells]
                # Icepack only assigns these optional fluxes for categories
                # with ice; the other category entries retain their values.
                column[name] = column[name].at[cells].set(
                    jnp.where(category_active & present, value, old))
            flux["coszen"] = flux["coszen"].at[cells].set(
                jnp.where(active, 0.5, flux["coszen"][cells]))
        elif shortwave.startswith("dEdd"):
            nx, ny, ncat = ice["aicen"][cells].shape
            nslyr = int(static["ice_domain_size"]["nslyr"])
            nilyr = int(static["ice_domain_size"]["nilyr"])
            nblyr = int(static["ice_domain_size"]["nblyr"])
            n_aero = int(tracers["n_aero"]) if bool(tracers["tr_aero"]) else 0
            n_algae = int(tracers["n_algae"])
            n_zaero = int(tracers["n_zaero"]) if bool(tracers["tr_zaero"]) else 0
            options_dEdd = dict(options)
            options_dEdd.update(tracers)
            options_dEdd.update({"nslyr": nslyr, "nilyr": nilyr,
                                 "nblyr": nblyr, "n_aero": n_aero,
                                 "n_algae": n_algae, "n_zaero": n_zaero,
                                 "five_band": shortwave == "dEdd_snicar_ad"})
            data = pytree["constants"]["icepack_shortwave_data"]
            coszen = flux["coszen"][cells]
            if not bool(options["semi_implicit_Tsfc"]):
                coszen = compute_coszen_jax(
                    grid["TLAT"][cells], grid["TLON"][cells],
                    static["ice_calendar"]["yday"],
                    static["ice_calendar"]["msec"],
                    pytree["constants"]["icepack_orbital"], params)
            netsw = (flux["swvdr"][cells] + flux["swidr"][cells]
                     + flux["swvdf"][cells] + flux["swidf"][cells])
            # Fortran updates coszen in place when any ice category is sunlit.
            coszen = jnp.where(
                jnp.any(present, axis=-1) & (netsw > params["puny"]),
                jnp.maximum(params["puny"], coszen), coszen)
            flux["coszen"] = flux["coszen"].at[cells].set(
                jnp.where(active, coszen, flux["coszen"][cells]))

            trcrn = ice["trcrn"][cells]
            def tracer_field(index):
                return trcrn[:, :, int(index), :]

            def tracer_layers(first, count):
                return jnp.moveaxis(
                    trcrn[:, :, first:first + count, :], -1, -2)

            if n_aero:
                aeron = tracer_layers(int(tracers["nt_aero"]), 4 * n_aero)
            else:
                aeron = jnp.zeros((nx, ny, ncat, 0), dtype=trcrn.dtype)
            if n_algae:
                bgcNn = tracer_layers(int(tracers["nt_bgc_N"][0]),
                                      n_algae * (nblyr + 3))
            else:
                bgcNn = jnp.zeros((nx, ny, ncat, 0), dtype=trcrn.dtype)
            if n_zaero:
                zaeron = tracer_layers(int(tracers["nt_zaero"][0]),
                                       n_zaero * (nblyr + 3))
            else:
                zaeron = jnp.zeros((nx, ny, ncat, 0), dtype=trcrn.dtype)
            fbri = (tracer_field(tracers["nt_fbri"])
                    if bool(tracers["tr_brine"]) else jnp.zeros_like(ice["aicen"][cells]))
            rsnow = (tracer_layers(int(tracers["nt_rsnw"]), nslyr)
                     if bool(options["snwgrain"])
                     else jnp.zeros((nx, ny, ncat, nslyr), dtype=trcrn.dtype))
            def category_flux(name):
                return jnp.broadcast_to(flux[name][cells][..., None], (nx, ny, ncat))

            inputs = (
                ice["aicen"][cells], ice["vicen"][cells], ice["vsnon"][cells],
                tracer_field(tracers["nt_Tsfc"]), tracer_field(tracers["nt_alvl"]),
                tracer_field(tracers["nt_apnd"]), tracer_field(tracers["nt_hpnd"]),
                tracer_field(tracers["nt_ipnd"]), aeron, bgcNn, zaeron,
                fbri, rsnow, column["dhsn"][cells], column["ffracn"][cells],
                jnp.broadcast_to(coszen[..., None], (nx, ny, ncat)),
                category_flux("fsnow"),
                jnp.full((nx, ny, ncat), static["ice_calendar"]["dt"]),
                *(category_flux(name) for name in (
                    "swvdr", "swvdf", "swidr", "swidf", "swuvrdr",
                    "swuvrdf", "swpardr", "swpardf")),
            )
            flattened = tuple(value.reshape((nx * ny * ncat,) + value.shape[3:])
                              for value in inputs)
            outputs = jax.vmap(lambda *args: icepack_step_radiation_jax(
                *args, column["swgrid"], column["igrid"],
                data, options_dEdd, params))(*flattened)
            shaped = tuple(value.reshape((nx, ny, ncat) + value.shape[1:])
                           for value in outputs)
            for name, value in zip(
                ("alvdrn", "alvdfn", "alidrn", "alidfn", "fswsfcn",
                 "fswintn", "fswthrun"), shaped[:7],
            ):
                column[name] = column[name].at[cells].set(
                    jnp.where(category_active, value, column[name][cells]))
            for name, value in zip(("albicen", "albsnon", "albpndn"), shaped[7:10]):
                column[name] = column[name].at[cells].set(
                    jnp.where(category_active & present, value, column[name][cells]))
            for index, name in enumerate((
                "fswthrun_vdr", "fswthrun_vdf", "fswthrun_idr",
                "fswthrun_idf", "fswthrun_uvrdr", "fswthrun_uvrdf",
                "fswthrun_pardr", "fswthrun_pardf",
            )):
                column[name] = column[name].at[cells].set(
                    jnp.where(category_active & present, shaped[10][..., index],
                              column[name][cells]))
            for name, value in (("Sswabsn", shaped[11]),
                                ("Iswabsn", shaped[12]),
                                ("fswpenln", shaped[13])):
                converted = jnp.moveaxis(value, -1, -2)
                column[name] = column[name].at[cells].set(
                    jnp.where(layer_active, converted, column[name][cells]))
            for name, value in (("apeffn", shaped[14]),
                                ("snowfracn", shaped[15])):
                column[name] = column[name].at[cells].set(
                    jnp.where(category_active, value, column[name][cells]))
            column["dhsn"] = column["dhsn"].at[cells].set(
                jnp.where(category_active & present, shaped[16], column["dhsn"][cells]))
            if bool(options["dEdd_algae"]) and (
                    bool(tracers["tr_zaero"]) or bool(tracers["tr_bgc_N"])):
                zbio = jnp.moveaxis(shaped[17], -1, -2)
                column["trcrn_sw"] = column["trcrn_sw"].at[cells].set(
                    jnp.where(layer_active, zbio, 0.0))
        else:
            raise ValueError("(step_radiation) unknown shortwave: " + shortwave)
    else:
        for name in ("alvdrn", "alvdfn", "alidrn", "alidfn",
                     "fswsfcn", "fswintn", "fswthrun"):
            column[name] = column[name].at[cells].set(
                jnp.where(category_active, 0.0, column[name][cells]))
        for name in ("Iswabsn", "Sswabsn", "fswpenln"):
            column[name] = column[name].at[cells].set(
                jnp.where(layer_active, 0.0, column[name][cells]))
        if bool(tracers["tr_pond_topo"]):
            apndn = ice["trcrn"][cells + (int(tracers["nt_apnd"]), slice(None))]
            ipndn = ice["trcrn"][cells + (int(tracers["nt_ipnd"]), slice(None))]
            apeffn = jnp.where(present & (apndn * ice["aicen"][cells] > params["puny"])
                                & (ipndn < params["hp1"]), apndn, 0.0)
            column["apeffn"] = column["apeffn"].at[cells].set(
                jnp.where(category_active, apeffn, column["apeffn"][cells]))

    # Icepack maps shortwave-active tracers before selecting its radiation
    # method, including when surface temperature calculation is disabled.
    if (bool(options["dEdd_algae"])
            and (bool(tracers["tr_zaero"]) or bool(tracers["tr_bgc_N"]))
            and (not bool(options["calc_Tsfc"])
                 or str(options["shortwave"]).strip().startswith("ccsm"))):
        nx, ny, ncat = ice["aicen"][cells].shape
        nslyr = int(static["ice_domain_size"]["nslyr"])
        nilyr = int(static["ice_domain_size"]["nilyr"])
        nblyr = int(static["ice_domain_size"]["nblyr"])
        n_algae = int(tracers["n_algae"])
        n_zaero = int(tracers["n_zaero"])
        remap_options = dict(options)
        remap_options.update(tracers)
        remap_options.update({"nslyr": nslyr, "nilyr": nilyr,
                              "nblyr": nblyr, "n_algae": n_algae,
                              "n_zaero": n_zaero})
        trcrn = ice["trcrn"][cells]
        if n_algae:
            first = int(tracers["nt_bgc_N"][0])
            bgcN = jnp.moveaxis(
                trcrn[:, :, first:first + n_algae * (nblyr + 3), :], -1, -2)
        else:
            bgcN = jnp.zeros((nx, ny, ncat, 0), dtype=trcrn.dtype)
        if n_zaero:
            first = int(tracers["nt_zaero"][0])
            zaero = jnp.moveaxis(
                trcrn[:, :, first:first + n_zaero * (nblyr + 3), :], -1, -2)
        else:
            zaero = jnp.zeros((nx, ny, ncat, 0), dtype=trcrn.dtype)
        fbri = (trcrn[:, :, int(tracers["nt_fbri"]), :]
                if bool(tracers["tr_brine"])
                else jnp.zeros_like(ice["aicen"][cells]))
        size = nx * ny * ncat
        area = ice["aicen"][cells].reshape(size)
        volume = ice["vicen"][cells].reshape(size)
        brine = fbri.reshape(size)
        bgcN = bgcN.reshape((size, bgcN.shape[-1]))
        zaero = zaero.reshape((size, zaero.shape[-1]))
        zbio = jax.vmap(lambda aice, vice, fbri_col, bgc_col, zaero_col:
                        jax.lax.cond(
                            aice > params["puny"],
                            lambda _: compute_shortwave_trcr_jax(
                                bgc_col, zaero_col, column["swgrid"],
                                column["igrid"], vice / aice,
                                fbri_col * vice / aice, remap_options, params),
                            lambda _: jnp.zeros(int(tracers["nbtrcr_sw"]),
                                                 dtype=trcrn.dtype), None))(
                                area, volume, brine, bgcN, zaero)
        zbio = jnp.moveaxis(
            zbio.reshape((nx, ny, ncat, int(tracers["nbtrcr_sw"]))), -1, -2)
        column["trcrn_sw"] = column["trcrn_sw"].at[cells].set(
            jnp.where(layer_active, zbio, 0.0))

    # At trace time, dictionary copies share the original tracer objects.
    # Return only reassigned, nonempty arrays across the compiled boundary.
    return (
        {name: value for name, value in column.items()
         if value is not state["ice_arrays_column"][name] and value.size},
        {name: value for name, value in flux.items()
         if value is not state["ice_flux"][name] and value.size},
    )


def ocean_mixed_layer(pytree, static):
    """Standalone ocean heat storage and freezing potential calculation."""
    state = dict(pytree["state"])
    flux = dict(state["ice_flux"])
    column = dict(state["ice_arrays_column"])
    tmask = pytree["constants"]["ice_grid"]["tmask"]
    aice = state["ice_state"]["aice"]
    options = static["icepack_parameters"]
    params = pytree["parameters"]["icepack_parameters"]

    # Fortran zeros these six fields on land before processing ocean cells.
    for name in ("sst", "frzmlt", "flwout_ocn", "fsens_ocn", "flat_ocn", "evap_ocn"):
        flux[name] = jnp.where(tmask, flux[name], 0.0)

    (strairx, strairy, Tref, Qref, delt, delq, lhcoef, shcoef,
     Cdn_atm, ratio, _, _) = icepack_atm_boundary(
        "ocn", flux["sst"], jnp.where(tmask, flux["potT"], params["Tffresh"]),
        jnp.where(tmask, flux["uatmT"], 0.0),
        jnp.where(tmask, flux["vatmT"], 0.0),
        jnp.where(tmask, flux["wind"], 1.0),
        jnp.where(tmask, flux["zlvl"], params["zref"]),
        jnp.where(tmask, flux["Qa"], 0.0),
        jnp.where(tmask, flux["rhoa"], 1.0),
        flux["strairx_ocn"], flux["strairy_ocn"],
        flux["Tref_ocn"], flux["Qref_ocn"], column["Cdn_atm"], options, params,
    )
    for name, value in (("strairx_ocn", strairx), ("strairy_ocn", strairy),
                        ("Tref_ocn", Tref), ("Qref_ocn", Qref)):
        flux[name] = jnp.where(tmask, value, flux[name])
    column["Cdn_atm"] = jnp.where(tmask, Cdn_atm, column["Cdn_atm"])
    column["Cdn_atm_ratio"] = jnp.where(tmask, ratio, column["Cdn_atm_ratio"])

    # CICE assigns the same ocean albedo to every shortwave band.
    for name in ("alvdr_ocn", "alidr_ocn", "alvdf_ocn", "alidf_ocn"):
        flux[name] = jnp.full_like(flux[name], params["albocn"])

    (sst, flwout, fsens, flat, evap, qdp, frzmlt) = icepack_ocn_mixed_layer(
        flux["alvdr_ocn"], flux["swvdr"], flux["alidr_ocn"], flux["swidr"],
        flux["alvdf_ocn"], flux["swvdf"], flux["alidf_ocn"], flux["swidf"],
        flux["sst"], shcoef, lhcoef, flux["flw"], delt, delq,
        aice, flux["fhocn"], flux["fswthru"],
        jnp.where(tmask, flux["hmix"], 1.0),
        flux["Tf"], flux["qdp"], static["ice_calendar"]["dt"], params,
    )
    for name, value in (("sst", sst), ("flwout_ocn", flwout),
                        ("fsens_ocn", fsens), ("flat_ocn", flat),
                        ("evap_ocn", evap), ("qdp", qdp), ("frzmlt", frzmlt)):
        flux[name] = jnp.where(tmask, value, flux[name])

    state["ice_flux"] = flux
    state["ice_arrays_column"] = column
    result = dict(pytree)
    result["state"] = state
    return result
