"""Main driver for time stepping of CICE, from CICE_RunMod.F90.

Fortran authors: Elizabeth C. Hunke, Philip W. Jones and
William H. Lipscomb, LANL. E. Hunke moved the ESMF code in 2008.

The first step receives the forcing already loaded by CICE_Initialize.
Later forcing records are read on the host; the state-dependent SST,
idealized wind stress, and BGC restoring updates are carried on JAX leaves.
"""

import time
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from cicecore.cicedyn.analysis import ice_diagnostics
from cicecore.cicedyn.analysis import ice_diagnostics_bgc
from cicecore.cicedyn.analysis import ice_history_bgc
from cicecore.cicedyn.analysis import ice_history
from cicecore.cicedyn.analysis import ice_history_shared
from cicecore.cicedyn.analysis.ice_step_snapshot import write_step_entry
from cicecore.cicedyn.dynamics import ice_dyn_eap, ice_dyn_evp, ice_dyn_shared
from cicecore.cicedyn.dynamics import ice_dyn_vp, ice_transport_driver, ice_transport_remap
from cicecore.cicedyn.general import ice_flux
from cicecore.cicedyn.general import ice_flux_bgc, ice_state
from cicecore.cicedyn.general import ice_forcing, ice_forcing_bgc
from cicecore.cicedyn.general import ice_init
from cicecore.cicedyn.general import ice_step_mod
from cicecore.cicedyn.infrastructure import ice_restart_driver
from cicecore.cicedyn.infrastructure import ice_restoring
from cicecore.cicedyn.infrastructure.comm.serial.ice_boundary import ice_halo_update_center_jax
from cicecore.cicedyn.infrastructure.comm.serial import ice_timers
from cicecore.cicedyn.infrastructure.io.io_netcdf import ice_restart
from cicecore.shared import ice_arrays_column
from cicecore.shared import ice_calendar
from cicecore.shared import ice_domain_size
from cicecore.shared import ice_fileunits
from cicecore.shared import ice_restart_column
from icepack.columnphysics import icepack_parameters, icepack_therm_shared
from icepack.columnphysics import icepack_tracers
from icepack.columnphysics.icepack_itd import column_sum


def _diagnostic_snapshot(pytree, progress, step_number, stage, step_start):
    """Synchronize JAX work before recording a diagnostic stage boundary."""
    sync_start = time.perf_counter()
    ice_timers.block_until_ready(pytree)
    sync_seconds = time.perf_counter() - sync_start
    fields = [f"step={step_number}", f"stage={stage}",
              f"elapsed={time.perf_counter() - step_start:.2f}s",
              f"sync={sync_seconds:.2f}s"]

    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            name, _, value = line.partition(":")
            if name in ("VmRSS", "VmHWM", "VmSwap"):
                fields.append(f"{name}={int(value.split()[0]) / 1024:.1f}MiB")
    except (OSError, ValueError, IndexError):
        fields.append("process_memory=unavailable")

    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                fields.append(f"wsl_available={int(line.split()[1]) / 1024:.1f}MiB")
                break
    except (OSError, ValueError, IndexError):
        fields.append("wsl_available=unavailable")

    try:
        current = Path("/sys/fs/cgroup/memory.current")
        if current.exists():
            fields.append(f"cgroup_current={int(current.read_text()) / 2**20:.1f}MiB")
            limit = Path("/sys/fs/cgroup/memory.max").read_text().strip()
            fields.append("cgroup_max=" +
                          (f"{int(limit) / 2**20:.1f}MiB" if limit != "max" else "unlimited"))
            events = dict(line.split() for line in
                          Path("/sys/fs/cgroup/memory.events").read_text().splitlines())
            fields.append(f"cgroup_oom={events.get('oom', 'unavailable')}")
            fields.append(f"cgroup_oom_kill={events.get('oom_kill', 'unavailable')}")
    except (OSError, ValueError):
        fields.append("cgroup_memory=unavailable")

    try:
        device = jax.devices()[0]
        stats = device.memory_stats() if device.platform == "gpu" else None
        if stats:
            for key, label in (("bytes_in_use", "gpu_in_use"),
                               ("peak_bytes_in_use", "gpu_peak"),
                               ("bytes_limit", "gpu_limit")):
                if key in stats:
                    fields.append(f"{label}={stats[key] / 2**20:.1f}MiB")
    except (RuntimeError, NotImplementedError):
        fields.append("gpu_memory=unavailable")

    progress.write("DIAG " + " ".join(fields) + "\n")
    progress.flush()


_ATMOSPHERE_FIELDS = (
    "Tair", "fsw", "flw", "frain", "fsnow", "Qa",
    "rhoa", "uatm", "vatm", "strax", "stray", "zlvl",
    "wind", "swvdr", "swvdf", "swidr", "swidf", "potT",
)


def CICE_Run(pytree, static, diagnose=False, profile_dir=None, profile_steps=(5, 7),
             detailed_physics_steps=None, warmup_steps=7):
    """Advance CICE, keeping the file and calendar work on the host."""
    if not isinstance(warmup_steps, int) or warmup_steps < 1:
        raise ValueError("warmup_steps must be a positive integer")
    profile_start, profile_end = profile_steps
    if profile_dir is not None and not (1 <= profile_start <= profile_end):
        raise ValueError("profile_steps must be an inclusive positive step range")
    profile_active = False
    diagnostic_values, unpack_diagnostics = ice_diagnostics.make_runtime_diags(
        static, pytree, ice_calendar.dt)
    if ice_history_shared.avail_hist_fields:
        ice_history_shared.start_device_history()
    status_names = tuple(name for name in pytree["state"]["ice_step_mod"]
                         if name.endswith(("_failed", "_error", "_warning")))
    status_counts = jax.jit(lambda status: jnp.stack([
        jnp.count_nonzero(status[name]) for name in status_names]))
    pending_forcing = {}
    completed_steps = 0
    measured_steps = 0
    loop_started = None
    loop_seconds = None
    loop_completed = False
    ice_fileunits.nu_diag.write(
        "TIMING enabled: synchronized wall time includes dispatch and compilation.\n"
        "TIMING forcing is charged to the step consuming it; step 1 forcing is in initialization.\n"
        "TIMING dotted stages are included in their parent; sync_seconds is included in seconds.\n")
    ice_fileunits.nu_diag.write(
        f"TIMING detailed physics steps={detailed_physics_steps}; None means all steps.\n"
        "TIMING physics completion stays synchronized when physics substage timing is disabled.\n")
    ice_fileunits.nu_diag.write(
        f"TIMING_LOOP_CONFIG warmup_steps={warmup_steps} "
        f"profiling_enabled={profile_dir is not None} diagnose={diagnose}\n"
        "TIMING_LOOP includes forcing, physics, diagnostics, history/restart, logging "
        "and existing timing instrumentation; later compilation is not subtracted.\n")
    #--------------------------------------------------------------------
    # initialize error code and step timer
    #--------------------------------------------------------------------
    ice_timers.configure_timer_scopes(static)
    ice_timers.ice_timer_start(ice_timers.timer_step)
    try:
        #--------------------------------------------------------------------
        # timestep loop
        #--------------------------------------------------------------------
        while True:
            step_number = int(ice_calendar.istep1)
            if profile_dir is not None and step_number == profile_start:
                ice_timers.block_until_ready(pytree)
                jax.effects_barrier()
                Path(profile_dir).mkdir(parents=True, exist_ok=True)
                with ice_timers.timing_phase("profiling"):
                    jax.profiler.start_trace(str(profile_dir), create_perfetto_trace=True)
                profile_active = True
                ice_fileunits.nu_diag.write(
                    f"PROFILE started steps={profile_start}-{profile_end} directory={profile_dir}\n")
                ice_fileunits.nu_diag.flush()
            step_start = time.perf_counter()
            detailed_physics = (detailed_physics_steps is None
                                or step_number in detailed_physics_steps)
            timing = {"step": step_number, "seconds": pending_forcing,
                      "detailed_physics": detailed_physics}
            pending_forcing = {}
            ice_fileunits.nu_diag.write(f"Time step {step_number}: started\n")
            ice_fileunits.nu_diag.flush()
            stage_started = time.perf_counter()
            history_arrays = [getattr(ice_history_shared, name)
                              for _, name, _ in ice_history_shared._RANKS.values()]
            ice_timers.record_stage(timing, "input_sync", stage_started,
                                    (pytree, history_arrays))
            if diagnose:
                _diagnostic_snapshot(pytree, ice_fileunits.nu_diag,
                                     step_number, "start", step_start)
            static["ice_calendar"]["yday"] = float(ice_calendar.yday)
            static["ice_calendar"]["msec"] = int(ice_calendar.msec)
            static["ice_calendar"]["istep1"] = int(ice_calendar.istep1)
            stage_started = time.perf_counter()
            with jax.profiler.StepTraceAnnotation("physics_step", step_num=step_number):
                pytree = ice_step(pytree, static,
                                  progress=ice_fileunits.nu_diag if detailed_physics or diagnose else None,
                                  diagnose=diagnose, step_start=step_start, timing=timing)
                ice_timers.record_stage(timing, "physics", stage_started, pytree)

            # Icepack reports fatal column errors during the step. Stop
            # before CICE_Finalize can report a successful run.
            status_started = time.perf_counter()
            step_status = pytree["state"]["ice_step_mod"]
            device_counts = status_counts(step_status)
            ice_timers.record_stage(timing, "status.compute_gpu", status_started, device_counts)
            stage_started = time.perf_counter()
            counts = dict(zip(status_names, jax.device_get(device_counts)))
            ice_timers.record_stage(timing, "status.d2h", stage_started)
            failures = []
            for name in (
                "therm1_failed", "bgc_failed", "therm2_failed",
                "remap_failed", "remap_conservation_failed",
                "ridge_failed", "snow_mass_error",
                "snow_energy_error",
            ):
                count = int(counts[name])
                if count:
                    failures.append(f"{name}={count}")
            if failures:
                for name in (
                    "therm2_itd_failed", "therm2_new_ice_failed",
                    "therm2_weld_failed", "therm2_cleanup_failed",
                    "therm2_cleanup_initial_area_failed",
                    "therm2_cleanup_rebin_failed",
                    "therm2_cleanup_negative_area_failed",
                    "therm2_cleanup_excess_area_failed",
                ):
                    count = int(counts[name])
                    if count:
                        failures.append(f"{name}={count}")
                area_mask = np.asarray(
                    step_status["therm2_cleanup_initial_area_failed"]
                )
                if np.any(area_mask):
                    first = tuple(int(i) for i in np.argwhere(area_mask)[0])
                    failures.append(f"first_bad_cell={first}")
                    for name in (
                        "therm2_area_entry", "therm2_area_after_itd",
                        "therm2_area_after_new_ice",
                        "therm2_area_after_lateral_melt",
                        "therm2_area_after_cleanup",
                    ):
                        values = np.asarray(step_status[name])[area_mask]
                        failures.append(
                            f"{name}=[{values.min():.12g},{values.max():.12g}]"
                        )
                raise RuntimeError(
                    f"(ice_step) istep1={ice_calendar.istep1}: "
                    + ", ".join(failures)
                )
            for name in ("therm1_aerosol_warning", "wave_failed"):
                count = int(counts[name])
                if count:
                    ice_fileunits.nu_diag.write(
                        f"(ice_step) istep1={ice_calendar.istep1}: "
                        f"{name}={count}\n"
                    )

            ice_timers.record_stage(timing, "status", status_started)
            diagnostics_started = time.perf_counter()
            report = None
            if ice_calendar.diagfreq != 0 and ice_calendar.istep % ice_calendar.diagfreq == 0:
                stage_started = time.perf_counter()
                device_report = diagnostic_values(pytree, ice_calendar.dt)
                ice_timers.record_stage(timing, "diagnostics.compute_gpu", stage_started, device_report)
                stage_started = time.perf_counter()
                report = unpack_diagnostics(jax.device_get(device_report))
                ice_timers.record_stage(timing, "diagnostics.d2h", stage_started)

            # Catalog callbacks see device arrays; the scope restores CPU
            # forcing buffers afterwards. It does not download the state.
            with ice_history.device_fields(pytree):
                if report is not None:
                    stage_started = time.perf_counter()
                    ice_diagnostics.runtime_diags(report)
                    if bool(icepack_parameters.skl_bgc) or bool(icepack_parameters.z_tracers):
                        ice_diagnostics_bgc.bgc_diags()
                    if bool(icepack_tracers.tr_brine):
                        ice_diagnostics_bgc.hbrine_diags()
                    ice_fileunits.nu_diag.flush()
                    ice_timers.record_stage(timing, "diagnostics.log_points", stage_started)
                ice_timers.record_stage(timing, "diagnostics", diagnostics_started)

                stage_started = time.perf_counter()
                ice_timers.ice_timer_start(ice_timers.timer_hist)
                try:
                    pytree = ice_history.accum_hist(ice_calendar.dt, pytree=pytree, timing=timing)
                    history_arrays = [getattr(ice_history_shared, name)
                                      for _, name, _ in ice_history_shared._RANKS.values()]
                    ice_timers.record_stage(timing, "history", stage_started, (pytree, history_arrays))
                finally:
                    ice_timers.ice_timer_stop(ice_timers.timer_hist)

            if ice_calendar.write_restart == 1:
                restart_started = time.perf_counter()
                ice_timers.ice_timer_start(ice_timers.timer_readwrite)
                try:
                    # Restart writers need writable arrays. Synchronize only
                    # on output steps, after the history/onset resets.
                    stage_started = time.perf_counter()
                    for name, module in (
                        ("ice_state", ice_state), ("ice_flux", ice_flux),
                        ("ice_flux_bgc", ice_flux_bgc),
                        ("ice_arrays_column", ice_arrays_column),
                    ):
                        for field, value in jax.device_get(pytree["state"][name]).items():
                            setattr(module, field, np.array(value, copy=True))
                    if int(ice_init.kdyn) == 2:
                        for field, value in jax.device_get(pytree["state"]["ice_dyn_eap"]).items():
                            setattr(ice_dyn_eap, field, np.array(value, copy=True))
                    for field in ("iceUmask", "iceNmask", "iceEmask"):
                        if field in pytree["state"]["ice_dyn_shared"]:
                            setattr(ice_dyn_shared, field, np.array(
                                pytree["state"]["ice_dyn_shared"][field], copy=True))
                    for field, value in jax.device_get(pytree["parameters"]["icepack_parameters"]).items():
                        setattr(icepack_parameters, field, np.asarray(value).item()
                                if np.asarray(value).ndim == 0 else np.asarray(value))
                    ice_timers.record_stage(timing, "restart.d2h_host_copy", stage_started)
                    stage_started = time.perf_counter()
                    ice_restart_driver.dumpfile()
                    if bool(icepack_tracers.tr_iage):
                        ice_restart_column.write_restart_age()
                    if bool(icepack_tracers.tr_FY):
                        ice_restart_column.write_restart_FY()
                    if bool(icepack_tracers.tr_lvl):
                        ice_restart_column.write_restart_lvl()
                    if bool(icepack_tracers.tr_pond_lvl):
                        ice_restart_column.write_restart_pond_lvl()
                    if bool(icepack_tracers.tr_pond_sealvl):
                        ice_restart_column.write_restart_pond_sealvl()
                    if bool(icepack_tracers.tr_pond_topo):
                        ice_restart_column.write_restart_pond_topo()
                    if bool(icepack_tracers.tr_snow):
                        ice_restart_column.write_restart_snow()
                    if bool(icepack_tracers.tr_fsd):
                        ice_restart_column.write_restart_fsd()
                    if bool(icepack_tracers.tr_iso):
                        ice_restart_column.write_restart_iso()
                    if bool(icepack_tracers.tr_aero):
                        ice_restart_column.write_restart_aero()
                    if bool(icepack_tracers.tr_brine):
                        ice_restart_column.write_restart_hbrine()
                    if bool(icepack_parameters.skl_bgc) or bool(icepack_parameters.z_tracers):
                        ice_restart_column.write_restart_bgc()
                    if int(ice_init.kdyn) == 2:
                        ice_dyn_eap.write_restart_eap()
                    ice_restart.final_restart_write()
                    ice_timers.record_stage(timing, "restart.io", stage_started)
                finally:
                    ice_timers.ice_timer_stop(ice_timers.timer_readwrite)
                ice_timers.record_stage(timing, "restart", restart_started)

            # tcraig, use advance_timestep now
            # istep/istep1 and date are updated here, after ice_step.
            ice_calendar.advance_timestep()
            if diagnose:
                _diagnostic_snapshot(pytree, ice_fileunits.nu_diag,
                                     step_number, "host_output_done", step_start)
            ice_fileunits.nu_diag.write(
                f"Time step {step_number}: {time.perf_counter() - step_start:.2f} seconds "
                "(physics, diagnostics, history, restart)\n"
            )
            try:
                device = jax.devices()[0]
                stats = device.memory_stats() if device.platform == "gpu" else None
                if stats is None:
                    ice_fileunits.nu_diag.write(
                        f"Time step {step_number}: GPU memory statistics unavailable\n"
                    )
                else:
                    fields = []
                    for key, label in (
                        ("bytes_in_use", "in_use"),
                        ("peak_bytes_in_use", "peak_in_use"),
                        ("bytes_limit", "allocator_limit"),
                    ):
                        if key in stats:
                            fields.append(f"{label}={stats[key] / 2**20:.1f} MiB")
                    ice_fileunits.nu_diag.write(
                        f"Time step {step_number}: GPU memory "
                        + (", ".join(fields) if fields else "statistics unavailable")
                        + "\n"
                    )
            except (RuntimeError, NotImplementedError) as exc:
                ice_fileunits.nu_diag.write(
                    f"Time step {step_number}: GPU memory statistics unavailable "
                    f"({type(exc).__name__}: {exc})\n"
                )
            ice_fileunits.nu_diag.flush()
            timing["wall_seconds"] = (time.perf_counter() - step_start
                                      + timing["seconds"].get("forcing", 0.0))
            ice_timers.print_step_timing(timing)
            if profile_active and (step_number >= profile_end or ice_calendar.stop_now >= 1):
                ice_timers.block_until_ready(pytree)
                jax.effects_barrier()
                profile_active = False
                with ice_timers.timing_phase("profiling"):
                    jax.profiler.stop_trace()
                ice_fileunits.nu_diag.write(f"PROFILE saved directory={profile_dir}\n")
                ice_fileunits.nu_diag.flush()
            completed_steps += 1
            if loop_started is not None:
                measured_steps += 1
            if ice_calendar.stop_now >= 1:
                break

            # Forcing for the first measured step is prepared below, before
            # the next iteration. Count executed steps, including on restart.
            if completed_steps == warmup_steps:
                history_arrays = [getattr(ice_history_shared, name)
                                  for _, name, _ in ice_history_shared._RANKS.values()]
                ice_timers.block_until_ready((pytree, history_arrays))
                jax.effects_barrier()
                ice_fileunits.nu_diag.write(
                    f"TIMING_LOOP_START first_step={int(ice_calendar.istep1)}\n")
                ice_fileunits.nu_diag.flush()
                loop_started = time.perf_counter()

            ice_fileunits.nu_diag.write(
                f"Time step {int(ice_calendar.istep1)}: entering forcing\n"
            )
            ice_fileunits.nu_diag.flush()
            next_timing = {"step": int(ice_calendar.istep1), "seconds": {}}
            forcing_started = time.perf_counter()
            ice_timers.ice_timer_start(ice_timers.timer_couple)
            try:
                # For now the wave spectrum is constant in time. Fortran
                # leaves get_wave_spec commented out. The first step used
                # forcing loaded during CICE_Initialize.
                old_flux = pytree["state"]["ice_flux"]
                old_bgc = pytree["state"]["ice_flux_bgc"]
                stage_started = time.perf_counter()
                ice_forcing.get_forcing_atmo(defer_ice_correction=True)
                ice_forcing.get_forcing_ocn(ice_calendar.dt)
                if bool(icepack_tracers.tr_iso):
                    ice_forcing_bgc.fiso_default()
                if bool(icepack_tracers.tr_aero) or bool(icepack_tracers.tr_zaero):
                    ice_forcing_bgc.faero_default()
                if bool(icepack_parameters.skl_bgc) or bool(icepack_parameters.z_tracers):
                    ice_forcing_bgc.get_forcing_bgc()
                if bool(icepack_parameters.z_tracers):
                    ice_forcing_bgc.get_atm_bgc()
                ice_timers.record_stage(next_timing, "forcing.cpu_read", stage_started)
                stage_started = time.perf_counter()

                # Keep unchanged leaves from the previous JAX step. Copying
                # the entire host module here would detach, for example, a
                # prognostic SST when no ocean restoring is requested.
                state = dict(pytree["state"])
                flux = dict(old_flux)
                bgc = dict(old_bgc)
                column = dict(state["ice_arrays_column"])
                atmosphere = None
                if "JRA55" in str(ice_init.atm_data_type):
                    atmosphere = jnp.asarray(np.stack([
                        getattr(ice_flux, name) for name in _ATMOSPHERE_FIELDS]))
                elif str(ice_init.atm_data_type).strip() != "default":
                    for name in _ATMOSPHERE_FIELDS:
                        flux[name] = jnp.asarray(getattr(ice_flux, name))
                    if str(ice_init.atm_data_type).strip() == "hadgem":
                        for name in ("fsurfn_f", "fcondtopn_f", "flatn_f"):
                            flux[name] = jnp.asarray(getattr(ice_flux, name))
                ocn_type = str(ice_init.ocn_data_type).strip()
                if ocn_type == "clim":
                    ocean_fields = ("sss", "Tf")
                elif ocn_type in ("ncar", "ISPOL"):
                    ocean_fields = ("sss", "Tf", "uocn", "vocn",
                                    "ss_tltx", "ss_tlty", "qdp", "hmix")
                elif ocn_type == "hadgem_sst_uvocn":
                    ocean_fields = ("uocn", "vocn")
                elif ocn_type == "oned":
                    ocean_fields = ("sss", "Tf", "uocn", "vocn",
                                    "ss_tltx", "ss_tlty", "qdp", "hmix",
                                    "frzmlt")
                elif ocn_type == "box2001" or ocn_type.startswith("uniform_") or ocn_type == "calm":
                    ocean_fields = ("uocn", "vocn")
                else:
                    ocean_fields = ()
                for name in ocean_fields:
                    flux[name] = jnp.asarray(getattr(ice_flux, name))
                clear_flux = {name: old_flux[name] for name in (
                    "strairxT", "strairyT", "fsurf", "fcondtop", "fsens",
                    "flat", "fswabs", "flwout", "evap", "evaps", "evapi",
                    "Tref", "Qref", "Uref", "fresh", "fsalt", "fpond",
                    "fhocn", "fswthru", "fswthru_vdr", "fswthru_vdf",
                    "fswthru_idr", "fswthru_idf", "fswthru_uvrdr",
                    "fswthru_uvrdf", "fswthru_pardr", "fswthru_pardf",
                )}
                if bool(ice_flux.send_i2x_per_cat):
                    clear_flux["fswthrun_ai"] = old_flux["fswthrun_ai"]
                clear_bgc = {name: old_bgc[name] for name in (
                    "fiso_evap", "Qref_iso", "Qa_iso", "faero_ocn",
                    "fiso_ocn", "HDO_ocn", "H2_16O_ocn", "H2_18O_ocn",
                )}
                temperature = None
                if "JRA55" in str(ice_init.atm_data_type):
                    temperature = (
                        flux["Tair"], flux["potT"], state["ice_state"]["aice"],
                        pytree["constants"]["ice_grid"]["hm"],
                        pytree["parameters"]["icepack_parameters"]["Tffresh"])
                flux_updates, bgc_updates = _forcing_prepare_compiled(
                    clear_flux, clear_bgc, temperature, ice_domain_size.physical_bounds(),
                    atmosphere=atmosphere)
                flux.update(flux_updates)
                bgc.update(bgc_updates)
                if bool(icepack_tracers.tr_iso):
                    bgc["fiso_atm"] = jnp.asarray(ice_flux_bgc.fiso_atm)
                if bool(icepack_tracers.tr_aero) or bool(icepack_tracers.tr_zaero):
                    bgc["faero_atm"] = jnp.asarray(ice_flux_bgc.faero_atm)
                if bool(icepack_parameters.z_tracers):
                    bgc["flux_bio_atm"] = jnp.asarray(ice_flux_bgc.flux_bio_atm)
                if bool(icepack_parameters.skl_bgc) or bool(icepack_parameters.z_tracers):
                    ilo, ihi, jlo, jhi = ice_domain_size.physical_bounds()
                    cell = (slice(ilo, ihi + 1), slice(jlo, jhi + 1))
                    ispol = str(ice_init.bgc_data_type).strip() == "ISPOL"
                    for active, name, slots in (
                        (icepack_tracers.tr_bgc_Sil, "sil", (
                            2 * icepack_tracers.max_algae + icepack_tracers.max_doc
                            + 3 + icepack_tracers.max_dic - 1,)),
                        (icepack_tracers.tr_bgc_Nit, "nit", (
                            icepack_tracers.max_algae,
                            2 * icepack_tracers.max_algae + icepack_tracers.max_doc
                            + 7 + icepack_tracers.max_dic - 1)),
                    ):
                        if active:
                            values = jnp.asarray(getattr(ice_flux_bgc, name))
                            bgc[name] = values if ispol else bgc[name].at[cell].set(values[cell])
                        if active or ispol:
                            for slot in slots:
                                index = cell + (int(slot),)
                                column["ocean_bio_all"] = column["ocean_bio_all"].at[index].set(
                                    jnp.asarray(ice_arrays_column.ocean_bio_all[index]))

                # Idealized box/uniform winds multiply stress by the current
                # ice area. Restore that multiplication on the device.
                if str(ice_init.atm_data_type).strip() in (
                    "box2001", "uniform_northeast", "uniform_north",
                    "uniform_east", "uniform_south", "uniform_west", "calm",
                ):
                    # These readers replace wind only. prepare_forcing must
                    # split the current snowfall, which bulk snow redistribution
                    # may have changed during the previous device step.
                    ilo, ihi, jlo, jhi = ice_domain_size.physical_bounds()
                    cell = (slice(ilo, ihi + 1), slice(jlo, jhi + 1))
                    units = str(ice_init.precip_units).strip()
                    secday = pytree["parameters"]["icepack_parameters"]["secday"]
                    if units == "mm_per_month":
                        factor = 12.0 / (secday * ice_calendar.days_per_year)
                    elif units == "mm_per_day":
                        factor = 1.0 / secday
                    elif units == "m_per_sec":
                        factor = 1000.0
                    else:  # Other units were validated by prepare_forcing.
                        factor = 1.0
                    precip = jnp.maximum(old_flux["fsnow"][cell], 0.0) * factor
                    rain = flux["Tair"][cell] >= pytree["parameters"]["icepack_parameters"]["Tffresh"]
                    flux["frain"] = old_flux["frain"].at[cell].set(jnp.where(rain, precip, 0.0))
                    flux["fsnow"] = old_flux["fsnow"].at[cell].set(jnp.where(rain, 0.0, precip))
                    tau = flux["rhoa"] * .0012 * flux["wind"]
                    stressx = state["ice_state"]["aice"] * tau * flux["uatm"]
                    stressy = state["ice_state"]["aice"] * tau * flux["vatm"]
                    if bool(ice_init.rotate_wind) and not bool(icepack_parameters.calc_strair):
                        angle = pytree["constants"]["ice_grid"]["ANGLET"]
                        rotated_x = stressx * jnp.cos(angle) + stressy * jnp.sin(angle)
                        rotated_y = stressy * jnp.cos(angle) - stressx * jnp.sin(angle)
                        ilo, ihi, jlo, jhi = ice_domain_size.physical_bounds()
                        cell = (slice(ilo, ihi + 1), slice(jlo, jhi + 1))
                        stressx = stressx.at[cell].set(rotated_x[cell])
                        stressy = stressy.at[cell].set(rotated_y[cell])
                    flux["strax"] = stressx
                    flux["stray"] = stressy

                # Ocean restoring retains the derivative of the previous
                # prognostic SST. The host reader supplies only the target.
                if bool(ice_init.restore_ocn) and ocn_type in (
                    "clim", "ncar", "ISPOL", "hadgem_sst", "hadgem_sst_uvocn",
                ):
                    target = (ice_forcing.c1intp * ice_forcing.sst_data[:, :, 0]
                              + ice_forcing.c2intp * ice_forcing.sst_data[:, :, 1])
                    weight = ice_calendar.dt / ice_forcing.trest
                    flux["sst"] = old_flux["sst"] + (jnp.asarray(target) - old_flux["sst"]) * weight
                elif ocn_type == "oned":
                    flux["sst"] = flux["Tf"]

                # The BGC climatology restores nitrate and silicate toward
                # observed fields. Keep the old device state in this update.
                if (str(ice_init.bgc_data_type).strip() == "clim" and
                        bool(ice_init.restore_bgc) and
                        (bool(icepack_parameters.skl_bgc) or bool(icepack_parameters.z_tracers))):
                    ilo, ihi, jlo, jhi = ice_domain_size.physical_bounds()
                    cell = (slice(ilo, ihi + 1), slice(jlo, jhi + 1))
                    weight = ice_calendar.dt / ice_forcing.trest
                    for flag, name, target, slot in (
                        (icepack_tracers.tr_bgc_Sil, "sil", ice_forcing_bgc.sildat,
                         2 * icepack_tracers.max_algae + icepack_tracers.max_doc +
                         3 + icepack_tracers.max_dic - 1),
                        (icepack_tracers.tr_bgc_Nit, "nit", ice_forcing_bgc.nitdat,
                         icepack_tracers.max_algae),
                    ):
                        if flag:
                            restored = old_bgc[name][cell] + (
                                jnp.asarray(target)[cell] - old_bgc[name][cell]) * weight
                            bgc[name] = bgc[name].at[cell].set(restored)
                            column["ocean_bio_all"] = column["ocean_bio_all"].at[
                                cell + (int(slot),)].set(restored)
                    if bool(icepack_tracers.tr_bgc_Nit):
                        pon_slot = (2 * icepack_tracers.max_algae +
                                    icepack_tracers.max_doc + 7 +
                                    icepack_tracers.max_dic - 1)
                        column["ocean_bio_all"] = column["ocean_bio_all"].at[
                            cell + (int(pon_slot),)].set(bgc["nit"][cell])

                state["ice_flux"] = flux
                state["ice_flux_bgc"] = bgc
                state["ice_arrays_column"] = column
                pytree = dict(pytree)
                pytree["state"] = state
                ice_timers.record_stage(next_timing, "forcing.upload_prepare_gpu", stage_started, pytree)
            finally:
                ice_timers.ice_timer_stop(ice_timers.timer_couple)
            ice_timers.record_stage(next_timing, "forcing", forcing_started)
            pending_forcing = next_timing["seconds"]
        if loop_started is not None:
            history_arrays = [getattr(ice_history_shared, name)
                              for _, name, _ in ice_history_shared._RANKS.values()]
            ice_timers.block_until_ready((pytree, history_arrays))
            jax.effects_barrier()
            ice_fileunits.nu_diag.flush()
            loop_seconds = time.perf_counter() - loop_started
        loop_completed = True
    finally:
        ice_timers.ice_timer_stop(ice_timers.timer_step)
        if profile_active:
            with ice_timers.timing_phase("profiling"):
                jax.profiler.stop_trace()
        seconds = f"{loop_seconds:.6f}" if loop_seconds is not None else "unavailable"
        average = (f"{loop_seconds / measured_steps:.6f}"
                   if loop_seconds is not None and measured_steps else "unavailable")
        reason = ("aborted" if not loop_completed else
                  "no_steps_after_warmup" if not measured_steps else "ok")
        ice_fileunits.nu_diag.write(
            f"TIMING_LOOP completed={loop_completed} reason={reason} "
            f"warmup_steps={min(completed_steps, warmup_steps)} "
            f"measured_steps={measured_steps} wall_seconds={seconds} "
            f"seconds_per_step={average} profiling_enabled={profile_dir is not None}\n")
        ice_fileunits.nu_diag.flush()

    return pytree


@jax.profiler.annotate_function
@partial(jax.jit, static_argnames=("bounds",))
def _forcing_prepare_compiled(clear_flux, clear_bgc, temperature, bounds, atmosphere=None):
    """Unpack JRA55 inputs with flux resets and the state-dependent cap."""
    flux = ({name: atmosphere[index] for index, name in enumerate(_ATMOSPHERE_FIELDS)}
            if atmosphere is not None else {})
    if temperature is not None:
        tair, potT, aice, hm, Tffresh = temperature
        if atmosphere is not None:
            tair, potT = flux["Tair"], flux["potT"]
        # The cap is above freezing, so the host rain/snow split stays valid.
        tair = jnp.where(aice > 0.1, jnp.minimum(tair, (Tffresh + 0.1) * hm), tair)
        ilo, ihi, jlo, jhi = bounds
        cell = (slice(ilo, ihi + 1), slice(jlo, jhi + 1))
        flux["Tair"] = tair
        flux["potT"] = potT.at[cell].set(tair[cell])
    flux.update({name: jnp.zeros_like(value) for name, value in clear_flux.items()})
    bgc = {name: jnp.zeros_like(value) for name, value in clear_bgc.items()}
    return flux, bgc


@jax.profiler.annotate_function
@partial(jax.jit, static_argnames=("config",))
def _physics_prepare_compiled(pytree, config):
    """Fuse diagnostic resets, initial snapshots and wind interpolation."""
    original = pytree["state"]
    static = {name: dict(values) for name, values in config}
    pytree = ice_diagnostics.init_mass_diags(pytree, static)
    pytree = ice_flux.init_history_therm(pytree, static)
    pytree = ice_history_bgc.init_history_bgc(pytree, static)
    pytree = ice_step_mod.step_prep(pytree, static)
    return {"state": {
        group: {name: value for name, value in fields.items()
                if value is not original[group][name]}
        for group, fields in pytree["state"].items()
    }}


def ice_step(pytree, static, progress=None, diagnose=False, step_start=None, timing=None):
    """One model step; physical subroutines will consume and return a pytree.

    Fortran order: boundary restoration; diagnostic reset; step_prep;
    radiation preparation; thermodynamics and BGC; state update; wave;
    ndtd dynamics/ridging/state updates; snow/state update; radiation;
    coupling_prep; scale_factor halo; diagnostics/history/restart output.
    History and restart I/O run on the host; optional debug checkpoints
    use JAX callbacks to print the state at the same places as Fortran.
    """
    flags = static["icepack_tracers"]
    parameters = static["icepack_parameters"]
    calendar = static["ice_calendar"]
    dt = calendar["dt"]
    dt_dyn = calendar["dt_dyn"]
    stage_started = time.perf_counter()
    write_step_entry(pytree, static)
    pytree = ice_diagnostics.debug_ice(pytree, static, "beginning time step")

    # Restoring on grid boundaries.
    if bool(static["ice_init"]["restore_ice"]):
        pytree = ice_restoring.ice_HaloRestore(pytree, static)

    # Initialize diagnostics and save initial state values.
    prepare_config = tuple((name, tuple(sorted(static[name].items()))) for name in (
        "ice_domain_size", "ice_domain", "ice_init", "icepack_parameters", "icepack_tracers"))
    updates = _physics_prepare_compiled(pytree, prepare_config)
    state = dict(pytree["state"])
    for name, changes in updates["state"].items():
        state[name] = {**state[name], **changes}
    pytree = {**pytree, "state": state}
    ice_timers.record_stage(timing, "physics.prepare", stage_started, pytree)
    step_number = int(calendar["istep1"])
    if int(parameters["ktherm"]) >= 0:
        stage_started = time.perf_counter()
        if bool(parameters["calc_Tsfc"]):
            pytree = ice_step_mod.prep_radiation(pytree, static)
        ice_timers.record_stage(timing, "physics.prepare_radiation", stage_started, pytree)
        pytree = ice_diagnostics.debug_ice(pytree, static, "post prep_radiation")
        if diagnose:
            _diagnostic_snapshot(pytree, progress, step_number,
                                 "prep_done", step_start)
        if progress is not None:
            progress.write("  entering step_therm1\n")
            progress.flush()
        stage_started = time.perf_counter()
        pytree = ice_step_mod.step_therm1(pytree, static, timing=timing)
        ice_timers.record_stage(timing, "physics.therm1", stage_started, pytree)
        if diagnose:
            _diagnostic_snapshot(pytree, progress, step_number,
                                 "therm1_done", step_start)
        pytree = ice_diagnostics.debug_ice(pytree, static, "post step_therm1")
        stage_started = time.perf_counter()
        pytree = ice_step_mod.biogeochemistry(pytree, static)
        ice_timers.record_stage(timing, "physics.bgc", stage_started, pytree)
        if diagnose:
            _diagnostic_snapshot(pytree, progress, step_number,
                                 "bgc_done", step_start)
        pytree = ice_diagnostics.debug_ice(pytree, static, "post biogeochemistry")
        if progress is not None:
            progress.write("  entering step_therm2\n")
            progress.flush()
        stage_started = time.perf_counter()
        pytree = ice_step_mod.step_therm2(pytree, static, timing=timing)
        ice_timers.record_stage(timing, "physics.therm2", stage_started, pytree)
        if diagnose:
            _diagnostic_snapshot(pytree, progress, step_number,
                                 "therm2_done", step_start)
        pytree = ice_diagnostics.debug_ice(pytree, static, "post step_therm2")

    # Clean up and update thermodynamic tendency diagnostics.
    stage_started = time.perf_counter()
    pytree = ice_step_mod.update_state(
        pytree, static, dt=dt, offset=dt, tendency="thermo"
    )
    ice_timers.record_stage(timing, "physics.update_thermo", stage_started, pytree)
    if diagnose:
        _diagnostic_snapshot(pytree, progress, step_number,
                             "thermo_update_done", step_start)

    # Wave fracture is outside dynamics subcycling.
    if bool(flags["tr_fsd"]) and bool(parameters["wave_spec"]):
        stage_started = time.perf_counter()
        pytree = ice_step_mod.step_dyn_wave(pytree, static)
        ice_timers.record_stage(timing, "physics.wave", stage_started, pytree)

    for _ in range(int(calendar["ndtd"])):
        if progress is not None:
            progress.write("  entering step_dyn_horiz\n")
            progress.flush()
        stage_started = time.perf_counter()
        pytree = ice_step_mod.step_dyn_horiz(pytree, static, timing=timing)
        ice_timers.record_stage(timing, "physics.horizontal", stage_started, pytree)
        pytree = ice_diagnostics.debug_ice(pytree, static, "post step_dyn_horiz")
        if int(static["ice_dyn_shared"]["kridge"]) > 0:
            stage_started = time.perf_counter()
            pytree = ice_step_mod.step_dyn_ridge(pytree, static, timing=timing)
            ice_timers.record_stage(timing, "physics.ridge", stage_started, pytree)
        pytree = ice_diagnostics.debug_ice(pytree, static, "post step_dyn_ridge")
        stage_started = time.perf_counter()
        pytree = ice_step_mod.update_state(
            pytree, static, dt=dt_dyn, offset=0.0, tendency="dynamics"
        )
        ice_timers.record_stage(timing, "physics.update_dynamics", stage_started, pytree)
    pytree = ice_diagnostics.debug_ice(pytree, static, "post dynamics")
    if diagnose:
        _diagnostic_snapshot(pytree, progress, step_number,
                             "dynamics_done", step_start)

    # Advanced snow physics may change area/volume, so update once more.
    if bool(flags["tr_snow"]):
        stage_started = time.perf_counter()
        pytree = ice_step_mod.step_snow(pytree, static)
        pytree = ice_step_mod.update_state(pytree, static, dt=dt)
        ice_timers.record_stage(timing, "physics.snow", stage_started, pytree)

    if int(parameters["ktherm"]) >= 0:
        if progress is not None:
            progress.write("  entering step_radiation and coupling\n")
            progress.flush()
        stage_started = time.perf_counter()
        pytree = ice_step_mod.step_radiation(pytree, static)
        ice_timers.record_stage(timing, "physics.radiation", stage_started, pytree)
    pytree = ice_diagnostics.debug_ice(pytree, static, "post step_radiation")
    stage_started = time.perf_counter()
    pytree = coupling_prep(pytree, static)
    ice_timers.record_stage(timing, "physics.coupling", stage_started, pytree)
    pytree = ice_diagnostics.debug_ice(pytree, static, "post coupling_prep")
    if diagnose:
        _diagnostic_snapshot(pytree, progress, step_number,
                             "radiation_coupling_done", step_start)

    # The next step's shortwave scaling needs the T-cell scalar ghosts.
    stage_started = time.perf_counter()
    state = dict(pytree["state"])
    flux = dict(state["ice_flux"])
    flux["scale_factor"] = ice_halo_update_center_jax(
        flux["scale_factor"], static, pytree["constants"]["ice_boundary"]["iglob"]
    )
    state["ice_flux"] = flux
    pytree = dict(pytree)
    pytree["state"] = state
    ice_timers.record_stage(timing, "physics.final_halo", stage_started, pytree)
    return pytree


@jax.profiler.annotate_function
def coupling_prep(pytree, static):
    """Prepare area-weighted albedo and fluxes for atmosphere/ocean coupling.

    This translates the array operations of CICE_RunMod.F90 for one full
    horizontal grid. The internal mixed layer uses the configured
    atmospheric boundary branch.
    """
    config = tuple((name, tuple(sorted(static[name].items()))) for name in (
        "ice_domain_size", "ice_init", "ice_flux", "icepack_parameters", "icepack_tracers"))
    updates = _coupling_prep_compiled(
        pytree, config, static["ice_calendar"]["dt"], int(static["ice_calendar"]["nstreams"]))
    state = dict(pytree["state"])
    for name, changes in updates.items():
        state[name] = {**state[name], **changes}
    result = dict(pytree)
    result["state"] = state
    return result


@partial(jax.jit, static_argnames=("config", "nstreams"))
def _coupling_prep_compiled(pytree, config, dt, nstreams):
    # dt is dynamic; only layout and scheme selectors belong in the cache key.
    original_state = pytree["state"]
    static = {name: dict(values) for name, values in config}
    static["ice_calendar"] = {"dt": dt, "nstreams": nstreams}
    state = dict(pytree["state"])
    flux = dict(state["ice_flux"])
    column = state["ice_arrays_column"]
    ice = state["ice_state"]
    bgc = dict(state["ice_flux_bgc"])
    params = pytree["parameters"]["icepack_parameters"]
    puny = params["puny"]
    rhofresh = params["rhofresh"]
    dt = static["ice_calendar"]["dt"]
    nstreams = int(static["ice_calendar"]["nstreams"])
    nbtrcr = int(static["icepack_tracers"]["nbtrcr"])

    # Save current value of frzmlt for diagnostics before the mixed layer.
    flux["frzmlt_init"] = flux["frzmlt"]
    if bool(static["ice_init"]["oceanmixed_ice"]):
        state["ice_flux"] = flux
        mixed = dict(pytree)
        mixed["state"] = state
        pytree = ice_step_mod.ocean_mixed_layer(mixed, static)
        state = dict(pytree["state"])
        flux = dict(state["ice_flux"])
        column = state["ice_arrays_column"]

    # Aggregate category albedos over ice-covered cells. The sun-above-
    # horizon test applies to albice, albsno and albpnd in the Fortran.
    aicen = ice["aicen"]
    ncat = aicen.shape[-1]
    ilo, ihi, jlo, jhi = static["ice_domain_size"]["physical_bounds"]
    physical = jnp.zeros(aicen.shape[:2], dtype=bool)
    physical = physical.at[ilo:ihi + 1, jlo:jhi + 1].set(True)
    area = jnp.where((aicen > puny) & physical[:, :, None], aicen, 0.0)
    netsw = flux["swvdr"] + flux["swidr"] + flux["swvdf"] + flux["swidf"]
    cszn = jnp.where(netsw > puny, 1.0, 0.0)
    flux["albcnt"] = flux["albcnt"].at[:, :, :nstreams].add(cszn[:, :, None])

    for name, category_name in (
        ("alvdf", "alvdfn"), ("alidf", "alidfn"),
        ("alvdr", "alvdrn"), ("alidr", "alidrn"),
    ):
        flux[name] = column_sum(ncat, jnp.where(
            area > 0.0, column[category_name] * area, 0.0))
    for name, category_name in (
        ("albice", "albicen"), ("albsno", "albsnon"),
        ("albpnd", "albpndn"),
    ):
        flux[name] = column_sum(ncat, jnp.where(
            (area > 0.0) & (cszn[:, :, None] > 0.0),
            column[category_name] * area, 0.0))
    flux["apeff_ai"] = column_sum(ncat, jnp.where(
        area > 0.0, column["apeffn"] * area, 0.0))
    flux["snowfrac"] = column_sum(ncat, jnp.where(
        area > 0.0, column["snowfracn"] * area, 0.0))

    # Reduce fresh by fpond for coupling.
    if bool(static["ice_flux"]["l_mpond_fresh"]):
        flux["fpond"] = flux["fpond"] * rhofresh / dt
        flux["fresh"] = flux["fresh"] - flux["fpond"]

    # Store grid-box mean albedos and fluxes before scale_fluxes divides
    # selected fields by ice area.
    for name in ("alvdf", "alidf", "alvdr", "alidr"):
        flux[name + "_ai"] = flux[name]
    for name in ("fresh", "fsalt", "fhocn", "fswthru"):
        flux[name + "_ai"] = flux[name]
    flux["fswup"] = ice["aice_init"] * flux["fsw"] - flux["fswabs"]
    if nbtrcr > 0:
        bgc["flux_bio_ai"] = bgc["flux_bio_ai"].at[:, :, :nbtrcr].set(
            bgc["flux_bio"][:, :, :nbtrcr]
        )

    # Save net shortwave for the scale_factor halo update.
    flux["scale_factor"] = (
        flux["swvdr"] * (1.0 - flux["alvdr_ai"])
        + flux["swvdf"] * (1.0 - flux["alvdf_ai"])
        + flux["swidr"] * (1.0 - flux["alidr_ai"])
        + flux["swidf"] * (1.0 - flux["alidf_ai"])
    )

    state["ice_flux"] = flux
    state["ice_flux_bgc"] = bgc
    result = dict(pytree)
    result["state"] = state

    result = ice_flux.scale_fluxes(result, static)
    # Select reassigned tracer objects at trace time, including mixed-layer updates.
    updates = {}
    for group, values in result["state"].items():
        changes = {name: value for name, value in values.items()
                   if value is not original_state[group].get(name) and value.size}
        if changes:
            updates[group] = changes
    return updates
