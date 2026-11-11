"""This module contains routines for supporting multiple CPU timers
and accumulates time for each individual block and node (task).

author for JAX version: Xinyuan Zhu
XZ: converted from F90 to Python

2005: Adapted from POP by William Lipscomb
      Replaced 'stdout' by 'nu_diag'

Wall time comes from ``time.perf_counter``. One grid, so the Fortran
per-block and per-task clocks are not accumulated. A running timer is
stopped for the print and then restarted. When a result is a JAX
array, the caller waits until that result is ready before
``ice_timer_stop``.
"""

import time
from contextlib import contextmanager
from threading import get_ident

import jax
from jax import monitoring

from cicecore.shared import ice_fileunits

# max number of timers
max_timers = 50

timer_total = None       # total time
timer_step = None        # time stepping
timer_dynamics = None    # dynamics
timer_advect = None      # horizontal advection
timer_column = None      # column
timer_thermo = None      # thermodynamics
timer_sw = None          # radiative transfer
timer_ponds = None       # melt ponds
timer_ridge = None       # ridging
timer_catconv = None     # category conversions
timer_fsd = None         # floe size distribution
timer_couple = None      # coupling
timer_readwrite = None   # read/write
timer_diags = None       # diagnostics/history
timer_hist = None        # diagnostics/history
timer_bound = None       # boundary updates
timer_bundbound = None   # boundary updates
timer_bgc = None         # biogeochemistry
timer_forcing = None     # forcing
timer_evp1dcore = None   # timer only loop
timer_evp = None         # timer including conversion 1d/2d
timer_updstate = None    # update state

_timers = None
_run_timing = None
_compile_listeners_registered = False
_COMPILE_EVENTS = {
    "/jax/core/compile/jaxpr_trace_duration": "trace",
    "/jax/core/compile/jaxpr_to_mlir_module_duration": "lowering",
    "/jax/core/compile/backend_compile_duration": "backend",
}

# Reuse synchronized stage boundaries; never add a GPU barrier for this table.
# IDs follow init_ice_timers and the standalone Fortran timer registration order.
_STAGE_TIMERS = {
    "physics.prepare": (5, 6),
    "physics.prepare_radiation": (5, 6, 7),
    "physics.therm1": (5, 6),
    "physics.bgc": (5, 6, 16),
    "physics.therm2": (5, 6),
    "physics.update_thermo": (5, 6, 20),
    "physics.wave": (5, 9),
    "physics.horizontal.momentum": (3,),
    "physics.horizontal.transport": (4,),
    "physics.ridge": (5, 8),
    "physics.update_dynamics": (20,),
    "physics.snow": (5, 6),
    "physics.radiation": (5, 6, 7),
    "physics.coupling": (5, 6, 10),
    "physics.final_halo": (5, 6, 14),
    "diagnostics": (12,),
    "forcing.cpu_read": (17,),
    "forcing.upload_prepare_gpu": (17,),
}


def _compile_started(event, value, **metadata):
    if _run_timing is not None and event in _COMPILE_EVENTS:
        _run_timing.compile_starts[(get_ident(), event, value)] = time.perf_counter()


def _compile_finished(event, start_time, end_time, **metadata):
    if _run_timing is not None and event in _COMPILE_EVENTS:
        started = _run_timing.compile_starts.pop((get_ident(), event, start_time), None)
        if started is not None:
            _run_timing.intervals.append((started, time.perf_counter(), "compile"))
            _run_timing.compile_events[_COMPILE_EVENTS[event]] += 1


@contextmanager
def timing_phase(name):
    started = time.perf_counter()
    try:
        yield
    finally:
        if _run_timing is not None:
            _run_timing.intervals.append((started, time.perf_counter(), name))


class RunTiming:
    """Partition model wall time; nested compilation spans count only once."""

    def __init__(self):
        global _run_timing, _compile_listeners_registered
        if _run_timing is not None:
            raise RuntimeError("A model timing session is already active")
        self.available = all(hasattr(monitoring, name) for name in (
            "register_scalar_listener", "register_event_time_span_listener"))
        if self.available and not _compile_listeners_registered:
            monitoring.register_scalar_listener(_compile_started)
            monitoring.register_event_time_span_listener(_compile_finished)
            _compile_listeners_registered = True
        self.intervals = []
        self.compile_starts = {}
        self.compile_events = dict.fromkeys(_COMPILE_EVENTS.values(), 0)
        self.started = time.perf_counter()
        self.unix_started = time.time()
        _run_timing = self

    def totals(self, finished):
        # Higher-priority intervals are subtracted from their enclosing phases.
        priority = ("compile", "history.write", "restart", "profiling",
                    "initialization", "run", "finalize", "other")
        events = [(self.started, "other", 1), (finished, "other", -1)]
        for start, end, name in self.intervals:
            start, end = max(start, self.started), min(end, finished)
            if end > start:
                events.extend(((start, name, 1), (end, name, -1)))
        active = dict.fromkeys(priority, 0)
        totals = dict.fromkeys(priority, 0.0)
        self.segments = []
        previous = self.started
        for stamp, name, change in sorted(events):
            category = next((key for key in priority if active[key]), "other")
            totals[category] += stamp - previous
            if stamp > previous:
                if self.segments and self.segments[-1][2] == category:
                    self.segments[-1] = (self.segments[-1][0], stamp, category)
                else:
                    self.segments.append((previous, stamp, category))
            active[name] += change
            previous = stamp
        return totals

    def finish(self, completed):
        global _run_timing
        finished = time.perf_counter()
        _run_timing = None
        totals = self.totals(finished)
        nu = ice_fileunits.nu_diag
        measured = self.available and not self.compile_starts
        compile_time = f"{totals['compile']:.6f}" if measured else "unavailable"
        run_time = f"{totals['run']:.6f}" if measured else "unavailable"
        nu.write("\nTIMING_TOTAL definition: compile covers JAX tracing, lowering and "
                 "backend compilation/cache loading; overlapping intervals count once.\n")
        nu.write("TIMING_TOTAL definition: run excludes measured compilation, history/restart "
                 "output and profiler start/stop; includes forcing, diagnostics, dispatch and GPU waits.\n")
        nu.write(f"TIMING_TOTAL completed={str(completed).lower()} "
                 f"compilation_measured={str(measured).lower()} "
                 f"wall_seconds={finished - self.started:.6f} "
                 f"compile_seconds={compile_time} run_seconds={run_time} "
                 f"output_seconds={totals['history.write'] + totals['restart']:.6f} "
                 f"initialization_seconds={totals['initialization']:.6f} "
                 f"profiler_control_seconds={totals['profiling']:.6f} "
                 f"finalize_seconds={totals['finalize']:.6f} other_seconds={totals['other']:.6f}\n")
        nu.write(f"TIMING_OUTPUT history_seconds={totals['history.write']:.6f} "
                 f"restart_seconds={totals['restart']:.6f} excludes_compilation={str(measured).lower()}\n")
        nu.write("TIMING_COMPILE " + " ".join(
            f"{name}_events={count}" for name, count in self.compile_events.items()) + "\n")
        if not measured:
            nu.write("TIMING_TOTAL WARNING: compilation accounting unavailable/incomplete; "
                     "phase residuals may contain compilation and are not compile-free timings.\n")
        for start, end, phase in self.segments:
            nu.write(f"TIMING_INTERVAL phase={phase} start_monotonic={start:.9f} "
                     f"end_monotonic={end:.9f} "
                     f"start_unix={self.unix_started + start - self.started:.6f} "
                     f"end_unix={self.unix_started + end - self.started:.6f}\n")
        nu.flush()


def block_until_ready(value):
    """Wait leaf by leaf to avoid older JAX/WSL CUDA stream-creation hangs."""
    for leaf in jax.tree.leaves(value):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()
    return value


def record_stage(timing, name, started, ready=None, emit=True):
    """Record synchronized wall time, including dispatch and first compilation.

    Dotted names are nested intervals. Their times are already included in
    their parent and must not be added to the parent when computing totals.
    """
    if _run_timing is not None and name in ("history.write", "restart"):
        _run_timing.intervals.append((started, time.perf_counter(), name))
    if timing is None:
        return
    if name.startswith("physics.") and not timing.get("detailed_physics", True):
        return
    sync_started = time.perf_counter()
    if ready is not None:
        block_until_ready(ready)
        jax.effects_barrier()
    finished = time.perf_counter()
    sync_seconds = finished - sync_started if ready is not None else 0.0
    elapsed = finished - started
    if _timers is not None:
        for timer_id in _STAGE_TIMERS.get(name, ()):
            timer = _timers[timer_id - 1]
            if timer.disabled:
                continue
            timer.accum += elapsed
            timer.intervals.append((started, finished))
            timer.steps.add(timing["step"])
            timer.used = True
    seconds = timing["seconds"]
    seconds[name] = seconds.get(name, 0.0) + elapsed
    if emit:
        nu = ice_fileunits.nu_diag
        nu.write(f"TIMING step={timing['step']} stage={name} "
                 f"seconds={elapsed:.6f} sync_seconds={sync_seconds:.6f} "
                 f"step_stage_seconds={seconds[name]:.6f}\n")
        nu.flush()


def print_step_timing(timing):
    """Print inclusive top-level costs and the largest measured substage."""
    seconds = timing["seconds"]
    top = sorted(((name, value) for name, value in seconds.items() if "." not in name),
                 key=lambda item: item[1], reverse=True)
    total = sum(value for _, value in top)
    nu = ice_fileunits.nu_diag
    fields = [f"{name}={value:.3f}s({100.0 * value / total if total else 0.0:.1f}%)"
              for name, value in top]
    nu.write(f"TIMING_SUMMARY step={timing['step']} measured_seconds={total:.6f} "
             + " ".join(fields) + "\n")
    if "wall_seconds" in timing:
        nu.write(f"TIMING_WALL step={timing['step']} seconds={timing['wall_seconds']:.6f} "
                 f"detailed_physics={timing.get('detailed_physics', True)} "
                 "includes_consumed_forcing=true\n")
    details = sorted(((name, value) for name, value in seconds.items() if "." in name),
                     key=lambda item: item[1], reverse=True)
    nu.write(f"TIMING_DETAIL step={timing['step']} "
             + " ".join(f"{name}={value:.3f}s" for name, value in details) + "\n")
    transfers = ("status.d2h", "diagnostics.d2h", "history.write.d2h", "restart.d2h_host_copy",
                 "physics.therm1.picard_stats_d2h")
    transfer_seconds = sum(seconds.get(name, 0.0) for name in transfers)
    nu.write(f"TIMING_TRANSFER step={timing['step']} d2h_host_seconds={transfer_seconds:.6f} "
             "included_in_parent_totals=true\n")
    leaves = []
    for name, value in seconds.items():
        children = [cost for other, cost in seconds.items()
                    if "." in other and other.rsplit(".", 1)[0] == name]
        if children:
            leaves.append((name + ".other", max(0.0, value - sum(children))))
        else:
            leaves.append((name, value))
    if leaves:
        name, elapsed = max(leaves, key=lambda item: item[1])
        nu.write(f"TIMING_HOTSPOT step={timing['step']} stage={name} seconds={elapsed:.6f}\n")
    nu.flush()


class _Timer:
    def __init__(self):
        self.name = "unknown_timer_name"
        self.in_use = False
        self.started = False
        self.used = False
        self.t0 = 0.0
        self.accum = 0.0
        self.intervals = []
        self.steps = set()
        self.disabled = False
        self.note = ""


def init_ice_timers():
    """This routine initializes machine parameters and timer structures
    for computing cpu time from F90 intrinsic timer functions.
    """
    global _timers
    global timer_total, timer_step, timer_dynamics, timer_advect
    global timer_column, timer_thermo, timer_sw, timer_ridge, timer_fsd
    global timer_couple, timer_readwrite, timer_diags, timer_hist
    global timer_bound, timer_bundbound, timer_bgc, timer_forcing
    global timer_evp1dcore, timer_evp, timer_updstate

    # Call F90 intrinsic system_clock to determine clock rate
    # and maximum cycles for single-processor runs.  If no clock
    # available, print message.
    # time.perf_counter is the wall clock and does not report a cycle rate.

    # initialize timer structures
    _timers = [_Timer() for _ in range(max_timers)]

    timer_total = _get_ice_timer("Total")
    timer_step = _get_ice_timer("TimeLoop")
    timer_dynamics = _get_ice_timer("Dynamics")
    timer_advect = _get_ice_timer("Advection")
    timer_column = _get_ice_timer("Column")
    timer_thermo = _get_ice_timer("Thermo")
    timer_sw = _get_ice_timer("Shortwave")
    # timer_ponds = _get_ice_timer("Meltponds")
    timer_ridge = _get_ice_timer("Ridging")
    # timer_catconv = _get_ice_timer("Cat Conv")
    timer_fsd = _get_ice_timer("FloeSize")
    timer_couple = _get_ice_timer("Coupling")
    timer_readwrite = _get_ice_timer("ReadWrite")
    timer_diags = _get_ice_timer("Diags    ")
    timer_hist = _get_ice_timer("History  ")
    timer_bound = _get_ice_timer("Bound")
    timer_bundbound = _get_ice_timer("Bundbound")
    timer_bgc = _get_ice_timer("BGC")
    timer_forcing = _get_ice_timer("Forcing")
    timer_evp1dcore = _get_ice_timer("evp1dcore")
    timer_evp = _get_ice_timer("evp")
    timer_updstate = _get_ice_timer("UpdState")


def configure_timer_scopes(static):
    """Explain fused/partial scopes instead of reporting unmeasured work as zero."""
    notes = {
        3: "sampled momentum solve incl. history initialization; broader than solver alone",
        4: "sampled horizontal transport incl. fused halos",
        5: "sampled Thermo + Ridging + wave fracture; overlaps child timers",
        6: "sampled thermodynamic stages incl. preparation, update, radiation and coupling",
        7: "sampled radiation preparation + shortwave solve",
        8: "sampled ridging incl. preparation/writeback",
        9: "sampled wave fracture only; FSD work fused into other column stages is not separated",
        10: "full forcing coupling + sampled physics.coupling; mixed coverage",
        12: "runtime diagnostics only; initial mass/history diagnostics fused into physics.prepare",
        14: "sampled final scale_factor halo only; other halos fused into physics JITs",
        15: "not separately measured: bundled halos fused into physics JITs; no MPI exchange",
        16: "sampled biogeochemistry stage",
        17: "timestep forcing read/upload/preparation; excludes initialization forcing",
        18: "not separately measured: no standalone 1D EVP core timer",
        19: "not separately measured: EVP inner loop fused with setup/writeback in Dynamics",
        20: "sampled thermo/dynamics update_state; snow-stage update not separated",
    }
    for number, note in notes.items():
        _timers[number - 1].note = note
    parameters = static["icepack_parameters"]
    flags = static["icepack_tracers"]
    if not (bool(flags["tr_fsd"]) and bool(parameters["wave_spec"])):
        _timers[timer_fsd - 1].note = "wave fracture disabled; other FSD work not separately measured"
        _timers[timer_fsd - 1].disabled = True
    if not (bool(parameters["skl_bgc"]) or bool(flags["tr_brine"])):
        _timers[timer_bgc - 1].note = "biogeochemistry disabled"
        _timers[timer_bgc - 1].disabled = True


def _get_ice_timer(name_choice):
    """This routine initializes a timer with a given name and returns a
    timer id.
    """
    subname = "(get_ice_timer)"
    # search for next free timer
    for n, timer in enumerate(_timers):
        if not timer.in_use:
            timer.name = name_choice
            timer.in_use = True
            timer.started = False
            timer.used = False
            timer.t0 = 0.0
            timer.accum = 0.0
            return n + 1
    raise RuntimeError(subname + "ERROR: Exceeded maximum number of timers")


def ice_timer_start(timer_id):
    """This routine starts a given node timer if it has not already
    been started by another thread.  If block information is available,
    the appropriate block timer is also started.
    """
    subname = "(ice_timer_start)"
    timer = _timers[timer_id - 1]
    if not timer.in_use:
        raise RuntimeError(subname + "ERROR: attempt to start undefined timer")

    # if called from outside a block loop, start node timer
    # stop timer if already started
    if timer.started:
        ice_timer_stop(timer_id)

    # start node timer
    timer.started = True
    timer.used = True
    timer.t0 = time.perf_counter()


def ice_timer_stop(timer_id):
    """This routine stops a given node timer if appropriate.  If block
    information is available the appropriate block timer is also stopped.
    """
    subname = "(ice_timer_stop)"
    timer = _timers[timer_id - 1]
    if not timer.in_use:
        raise RuntimeError(subname + "ERROR: attempt to stop undefined timer")

    # if called from outside a block loop, stop node timer
    # correct for wraparound and accumulate time
    if timer.started:
        finished = time.perf_counter()
        timer.accum = timer.accum + (finished - timer.t0)
        timer.intervals.append((timer.t0, finished))
        timer.started = False


def ice_timer_print(timer_id, stats=False):
    """Prints the accumulated time for a given timer and optional
    statistics for that timer. It is assumed that this routine
    is called outside of a block loop.
    """
    subname = "(ice_timer_print)"
    timer = _timers[timer_id - 1]
    if not timer.in_use:
        raise RuntimeError(subname + "ERROR: attempt to print undefined timer")

    # if timer has been defined, check to see whether it is currently
    # running.  If it is, stop the timer and print the info.
    restart = timer.started
    if restart:
        ice_timer_stop(timer_id)

    # Find max node time and print that time as default timer result.
    # One grid, so the accumulated seconds are that time.
    nu = ice_fileunits.nu_diag
    value = f"{timer.accum:11.2f} seconds" if timer.used or timer.disabled else "        N/A seconds"
    nu.write(f"Timer {timer_id:3d}: {timer.name:<9.9}{value}\n")
    # Compilation events can nest; merge before intersecting measured spans.
    compilation = []
    if _run_timing is not None and _run_timing.available and not _run_timing.compile_starts:
        for start, end, phase in sorted(_run_timing.intervals):
            if phase != "compile":
                continue
            if compilation and start <= compilation[-1][1]:
                compilation[-1] = (compilation[-1][0], max(end, compilation[-1][1]))
            else:
                compilation.append((start, end))
        compile_seconds = sum(max(0., min(end, right) - max(start, left))
                              for start, end in timer.intervals for left, right in compilation)
        compile_text = f"{compile_seconds:.6f}"
        residual_text = f"{max(0., timer.accum - compile_seconds):.6f}"
    else:
        compile_text = residual_text = "unavailable"
    if not timer.used and not timer.disabled:
        compile_text = residual_text = "unavailable"
    steps = ",".join(str(step) for step in sorted(timer.steps)) or "none"
    nu.write(f"  TIMER_SCOPE id={timer_id} measured={str(timer.used).lower()} "
             f"disabled={str(timer.disabled).lower()} stage_steps={steps} "
             f"compile_seconds={compile_text} excluding_compile_seconds={residual_text} "
             f"note={timer.note or 'host wall-clock scope; may include output/profiling'}\n")
    if stats and (timer.used or timer.disabled):
        # One grid: node min, max, and mean are this same value.
        # Block statistics are not accumulated.
        nu.write(f"  Timer stats (node): min = {timer.accum:11.2f} seconds\n")
        nu.write(f"                      max = {timer.accum:11.2f} seconds\n")
        nu.write(f"                      mean= {timer.accum:11.2f} seconds\n")
    nu.flush()

    if restart:
        ice_timer_start(timer_id)


def ice_timer_print_all(stats=False):
    """Prints the accumulated time for all timers and optional
    statistics for that timer. It is assumed that this routine
    is called outside of a block loop.
    """
    nu = ice_fileunits.nu_diag
    nu.write("\nTiming information:\n\n")
    nu.write("Timer IDs/names match standalone Fortran. Times overlap; do not sum them.\n"
             "Physics totals cover only synchronized detailed steps, without extrapolation.\n"
             "TIMER_SCOPE lists coverage and compile-subtracted measured time, not necessarily pure run.\n"
             "N/A means not separately measured, never zero cost. TIMING_TOTAL is the full run/output partition.\n\n")
    for n, timer in enumerate(_timers, start=1):
        if timer.in_use:
            ice_timer_print(n, stats)
    nu.flush()
