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

import jax

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


def record_stage(timing, name, started, ready=None, emit=True):
    """Record synchronized wall time, including dispatch and first compilation.

    Dotted names are nested intervals. Their times are already included in
    their parent and must not be added to the parent when computing totals.
    """
    if timing is None:
        return
    if name.startswith("physics.") and not timing.get("detailed_physics", True):
        return
    sync_started = time.perf_counter()
    if ready is not None:
        jax.block_until_ready(ready)
        jax.effects_barrier()
    finished = time.perf_counter()
    sync_seconds = finished - sync_started if ready is not None else 0.0
    elapsed = finished - started
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
        timer.accum = timer.accum + (time.perf_counter() - timer.t0)
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
    nu.write(
        f"Timer {timer_id:3d}: {timer.name:<9.9}{timer.accum:11.2f} seconds\n"
    )
    if stats:
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
    for n, timer in enumerate(_timers, start=1):
        # Timers that were never started stay out of the log.
        if timer.in_use and timer.used:
            ice_timer_print(n, stats)
    nu.flush()
