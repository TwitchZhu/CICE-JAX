"""Diagnostic information output during run.

authors: Elizabeth C. Hunke, LANL
         Bruce P. Briegleb, NCAR

author for JAX version: Xinyuan Zhu
XZ: converted from F90 to Python

2004: Block structure added by William Lipscomb
2006: Converted to free source form (F90) by Elizabeth Hunke

``init_diags`` locates the two diagnostic points. ``runtime_diags`` is the
time loop. One grid, so the global min/max of a scalar is the local value
and the task number is 0. Stored ``piloc`` and ``pjloc`` are 0-based.
"""

from collections import OrderedDict
from types import SimpleNamespace

import numpy as np
import jax
import jax.numpy as jnp

from cicecore.cicedyn.general import ice_init
from cicecore.cicedyn.general import ice_flux, ice_flux_bgc, ice_state
from cicecore.cicedyn.dynamics import ice_dyn_shared
from cicecore.cicedyn.infrastructure import ice_grid
from cicecore.shared import ice_domain_size
from cicecore.shared import ice_fileunits
from cicecore.shared.ice_constants import c180, c360, m2_to_km2, p5
from cicecore.shared.ice_kinds_mod import dbl_kind, int_kind
from icepack.columnphysics import icepack_parameters, icepack_tracers
from icepack.columnphysics.icepack_therm_vertical import POND_DIAGNOSTIC_FIELDS
from icepack.columnphysics.icepack_atmo import (
    ATMO_DIAGNOSTIC_FIELDS, STABILITY_MATH_DIAGNOSTIC_FIELDS,
)
from icepack.columnphysics.icepack_therm_mushy import PICARD_DIAGNOSTIC_FIELDS

# diagnostic output file
diag_file = "ice_diag.d"

# if true, debug model at high level
debug_model = False
# if true, print point data
print_points = False
# if true, print global data
print_global = True

# begin printing at istep1=debug_model_step
debug_model_step = 0

# total number of points to be printed
npnt = 2

# Set to true to identify unstable fast-moving ice.
# if true, check for speed > umax_stab
check_umax = False

# ice speed threshold for instability (m/s)
umax_stab = dbl_kind(1.0)
# min aice value for ice extent calc
aice_extmin = dbl_kind(0.15)

# latitude of diagnostic points
latpnt = None
# longitude of diagnostic points
lonpnt = None

# i index for points
iindx = 0
# j index for points
jindx = 0
# block index for points
bindx = 0

# change in mean ice thickness (m)
pdhi = None
# change in mean snow thickness (m)
pdhs = None
# change in ice and snow energy (W m-2)
pde = None

# latitude, longitude of points
plat = None
plon = None

# location of diagnostic points
piloc = None
pjloc = None
pbloc = None
pmloc = None

# location of debug_model point, local i index
debug_model_i = -1
# location of debug_model point, local j index
debug_model_j = -1
# location of debug_model point, local block number
debug_model_iblk = -1
# location of debug_model point, local task number
debug_model_task = -1


def init_mass_diags(pytree, static):
    """Save hemispheric ice/snow mass, energy, and salt at step start.

    One process owns the full grid, so global_sum is a sum over the
    physical cells weighted by the northern/southern area masks.
    """
    state = dict(pytree["state"])
    diagnostics = dict(state["ice_diagnostics"])
    ice = state["ice_state"]
    grid = pytree["constants"]["ice_grid"]
    tracers = static["icepack_tracers"]
    sizes = static["ice_domain_size"]
    params = pytree["parameters"]["icepack_parameters"]
    ilo, ihi, jlo, jhi = sizes["physical_bounds"]
    cells = (slice(ilo, ihi + 1), slice(jlo, jhi + 1))
    north = grid["tarean"][cells]
    south = grid["tareas"][cells]
    if static["ice_domain"]["ns_boundary_type"].strip() == "tripoleT":
        # Serial global_sum discards the redundant half of the folded T row.
        iglob = pytree["constants"]["ice_boundary"]["iglob"][ilo:ihi + 1]
        keep = iglob <= int(sizes["nx_global"]) // 2
        north = north.at[:, -1].set(jnp.where(keep, north[:, -1], 0))
        south = south.at[:, -1].set(jnp.where(keep, south[:, -1], 0))

    # Total ice/snow volume and mass in each hemisphere.
    shmaxn = jnp.sum(ice["vice"][cells] * north)
    shmaxs = jnp.sum(ice["vice"][cells] * south)
    snwmxn = jnp.sum(ice["vsno"][cells] * north)
    snwmxs = jnp.sum(ice["vsno"][cells] * south)
    diagnostics["totmin"] = params["rhoi"] * shmaxn
    diagnostics["totmis"] = params["rhoi"] * shmaxs
    diagnostics["totmn"] = diagnostics["totmin"] + params["rhos"] * snwmxn
    diagnostics["totms"] = diagnostics["totmis"] + params["rhos"] * snwmxs

    # total_energy and total_salt: sum category/layer contributions only
    # over ocean cells; the Fortran work arrays start at zero elsewhere.
    trcrn = ice["trcrn"]
    vicen = ice["vicen"]
    vsnon = ice["vsnon"]
    nilyr = int(sizes["nilyr"])
    nslyr = int(sizes["nslyr"])
    qice = trcrn[:, :, int(tracers["nt_qice"]):int(tracers["nt_qice"]) + nilyr, :]
    qsno = trcrn[:, :, int(tracers["nt_qsno"]):int(tracers["nt_qsno"]) + nslyr, :]
    sice = trcrn[:, :, int(tracers["nt_sice"]):int(tracers["nt_sice"]) + nilyr, :]
    energy = jnp.sum(qice, axis=2) * vicen / nilyr + jnp.sum(qsno, axis=2) * vsnon / nslyr
    salt = jnp.sum(sice, axis=2) * vicen / nilyr
    energy = jnp.where(grid["tmask"], jnp.sum(energy, axis=2), 0)
    salt = jnp.where(grid["tmask"], jnp.sum(salt, axis=2), 0)
    diagnostics["toten"] = jnp.sum(energy[cells] * north)
    diagnostics["totes"] = jnp.sum(energy[cells] * south)
    diagnostics["totsn"] = jnp.sum(salt[cells] * north)
    diagnostics["totss"] = jnp.sum(salt[cells] * south)

    # Save values at the printed diagnostic points for water/heat budgets.
    if bool(static["ice_init"]["print_points"]):
        points_i = diagnostics["piloc"]
        points_j = diagnostics["pjloc"]
        active = diagnostics["pmloc"] == 0
        safe_i = jnp.clip(points_i, 0, ice["vice"].shape[0] - 1)
        safe_j = jnp.clip(points_j, 0, ice["vice"].shape[1] - 1)
        diagnostics["pdhi"] = jnp.where(active, ice["vice"][safe_i, safe_j], diagnostics["pdhi"])
        diagnostics["pdhs"] = jnp.where(active, ice["vsno"][safe_i, safe_j], diagnostics["pdhs"])
        diagnostics["pde"] = jnp.where(active, energy[safe_i, safe_j], diagnostics["pde"])

    trcr = ice["trcr"]
    if bool(tracers["tr_iso"]):
        # Preserve the Fortran inner k loop: every n receives the same
        # sum over all isotope species.
        count = int(tracers["n_iso"])
        snow_iso = trcr[:, :, int(tracers["nt_isosno"]):int(tracers["nt_isosno"]) + count]
        ice_iso = trcr[:, :, int(tracers["nt_isoice"]):int(tracers["nt_isoice"]) + count]
        work = ice["vsno"] * jnp.sum(snow_iso, axis=2) + ice["vice"] * jnp.sum(ice_iso, axis=2)
        total_n = jnp.sum(work[cells] * north)
        total_s = jnp.sum(work[cells] * south)
        diagnostics["totison"] = diagnostics["totison"].at[:count].set(total_n)
        diagnostics["totisos"] = diagnostics["totisos"].at[:count].set(total_s)

    if bool(tracers["tr_aero"]):
        for n in range(int(tracers["n_aero"])):
            nt = int(tracers["nt_aero"]) + 4 * n
            work = ((trcr[:, :, nt] + trcr[:, :, nt + 1]) * ice["vsno"]
                    + (trcr[:, :, nt + 2] + trcr[:, :, nt + 3]) * ice["vice"])
            diagnostics["totaeron"] = diagnostics["totaeron"].at[n].set(jnp.sum(work[cells] * north))
            diagnostics["totaeros"] = diagnostics["totaeros"].at[n].set(jnp.sum(work[cells] * south))

    if bool(tracers["tr_pond_topo"]):
        nt_apnd = int(tracers["nt_apnd"])
        nt_hpnd = int(tracers["nt_hpnd"])
        work = jnp.sum(ice["aicen"] * trcrn[:, :, nt_apnd, :] * trcrn[:, :, nt_hpnd, :], axis=2)
        diagnostics["totmn"] += params["rhofresh"] * jnp.sum(work[cells] * north)
        diagnostics["totms"] += params["rhofresh"] * jnp.sum(work[cells] * south)

    state["ice_diagnostics"] = diagnostics
    result = dict(pytree)
    result["state"] = state
    return result


def init_diags():
    """Find tasks for diagnostic points.

    authors: Elizabeth C. Hunke and William H. Lipscomb, LANL

    One grid. ``global_minval`` and ``global_maxval`` are the local value.
    ``piloc`` and ``pjloc`` are 0-based. The log prints those same indices.
    """
    global latpnt, lonpnt, plat, plon, piloc, pjloc, pbloc, pmloc
    global iindx, jindx, bindx
    global debug_model, print_points, print_global, diag_file
    global debug_model_step, debug_model_i, debug_model_j
    global debug_model_iblk, debug_model_task
    subname = "(init_diags)"
    _ = subname

    diag_file = ice_init.diag_file
    debug_model = bool(ice_init.debug_model)
    print_points = bool(ice_init.print_points)
    print_global = bool(ice_init.print_global)
    debug_model_step = int(ice_init.debug_model_step)
    debug_model_i = int(ice_init.debug_model_i)
    debug_model_j = int(ice_init.debug_model_j)
    debug_model_iblk = int(ice_init.debug_model_iblk)
    debug_model_task = int(ice_init.debug_model_task)

    # tcraig, do this all the time now for print_points_state usage
    # if (print_points) then

    puny = icepack_parameters.puny
    rad_to_deg = icepack_parameters.rad_to_deg
    nu = ice_fileunits.nu_diag
    nu.write(" \n")
    nu.write(" Find indices of diagnostic points \n")

    latpnt = np.array(ice_init.latpnt, dtype=dbl_kind).copy()
    lonpnt = np.array(ice_init.lonpnt, dtype=dbl_kind).copy()
    if latpnt.shape[0] != npnt or lonpnt.shape[0] != npnt:
        raise ValueError(
            f"(init_diags) latpnt/lonpnt length {latpnt.shape[0]}, {lonpnt.shape[0]}; need {npnt}"
        )

    piloc = np.full(npnt, -1, dtype=int_kind)
    pjloc = np.full(npnt, -1, dtype=int_kind)
    pbloc = np.full(npnt, -1, dtype=int_kind)
    pmloc = np.full(npnt, -999, dtype=int_kind)
    plat = np.full(npnt, dbl_kind(-999.0), dtype=dbl_kind)
    plon = np.full(npnt, dbl_kind(-999.0), dtype=dbl_kind)

    # find minimum distance to diagnostic points on this processor
    ilo, ihi, jlo, jhi = ice_domain_size.physical_bounds()
    hm = ice_grid.hm
    tlat = ice_grid.TLAT
    tlon = ice_grid.TLON
    # my_task == master_task. One task, number 0.
    my_task = 0
    for n in range(npnt):
        if lonpnt[n] > c180:
            lonpnt[n] = lonpnt[n] - c360
            ice_init.lonpnt[n] = float(lonpnt[n])

        iindx = -1
        jindx = -1
        bindx = -1
        mindis = dbl_kind(540.0)  # 360. + 180.

        if abs(latpnt[n]) < c360 and abs(lonpnt[n]) < c360:
            # MDT, 09/2017: Comment out OpenMP directives since loop is not thread-safe
            # This is computing closest point, Could add a CRITICAL but it's just initialization
            # Strict < keeps the first cell when distances tie, matching the Fortran scan.
            for j in range(jlo, jhi + 1):
                for i in range(ilo, ihi + 1):
                    if hm[i, j] > p5:
                        latdis = abs(latpnt[n] - tlat[i, j] * rad_to_deg)
                        londis = abs(lonpnt[n] - tlon[i, j] * rad_to_deg) * np.cos(tlat[i, j])
                        totdis = np.sqrt(latdis**2 + londis**2)
                        if totdis < mindis:
                            mindis = totdis
                            jindx = j
                            iindx = i
                            bindx = 0

        # find global minimum distance to diagnostic points
        # One task: the global minimum is the local minimum.
        mindis_g = mindis

        # save indices of minimum-distance grid cell
        if mindis <= dbl_kind(180.0) and abs(mindis_g - mindis) < puny:
            piloc[n] = iindx
            pjloc[n] = jindx
            pbloc[n] = bindx
            pmloc[n] = my_task
            plat[n] = tlat[iindx, jindx] * rad_to_deg
            plon[n] = tlon[iindx, jindx] * rad_to_deg

        # communicate to all processors
        # One task: global_maxval returns the local value.

        # write to log file
        nu.write(" \n")
        nu.write(f" found point{n + 1:4d}\n")
        nu.write("   lat    lon   TLAT   TLON     i     j   block  task\n")
        nu.write(
            f"{float(latpnt[n]):6.1f} {float(lonpnt[n]):6.1f} "
            f"{float(plat[n]):6.1f} {float(plon[n]):6.1f} "
            f" {int(piloc[n]):4d}  {int(pjloc[n]):4d}  "
            f"{int(pbloc[n]):4d}  {int(pmloc[n]):4d}\n"
        )
    nu.flush()


def debug_ice(pytree, static, label):
    """Print one cell at a Fortran debug checkpoint without changing state."""
    if not bool(static["ice_init"]["debug_model"]):
        return pytree
    if int(static["ice_calendar"]["istep1"]) < int(static["ice_init"]["debug_model_step"]):
        return pytree
    if bool(static["ice_init"]["debug_model_ponds"]):
        ice = pytree["state"]["ice_state"]
        column = pytree["state"]["ice_arrays_column"]
        tracers = static["icepack_tracers"]
        nilyr = int(static["ice_domain_size"]["nilyr"])
        nslyr = int(static["ice_domain_size"]["nslyr"])
        names = ("aicen", "vicen", "alvl", "apnd", "hpnd", "ipnd", "vsnon", "Tsf",
                 "fswsfcn", "fswintn") + tuple(
            f"{field}{k + 1}" for field, count in (("qice", nilyr), ("sice", nilyr), ("qsno", nslyr))
            for k in range(count))
        ilo, ihi, jlo, jhi = static["ice_domain_size"]["physical_bounds"]
        points = tuple((i, j) for i, j in zip(static["ice_init"]["debug_model_ponds_i"],
                                              static["ice_init"]["debug_model_ponds_j"])
                       if i > 0 and j > 0)
        samples = []
        for global_i, global_j in points:
            # Pond point lists use global 1-based indices, independent of halo width.
            i, j = ilo + global_i - 1, jlo + global_j - 1
            if i > ihi or j > jhi:
                raise ValueError("debug_model_ponds_i/j is outside the physical domain")
            samples.append(jnp.stack((ice["aicen"][i, j], ice["vicen"][i, j],
                                      *(ice["trcrn"][i, j, int(tracers[index]), :]
                                        for index in ("nt_alvl", "nt_apnd", "nt_hpnd", "nt_ipnd")),
                                      ice["vsnon"][i, j], ice["trcrn"][i, j, int(tracers["nt_Tsfc"]), :],
                                      column["fswsfcn"][i, j], column["fswintn"][i, j],
                                      *(ice["trcrn"][i, j, int(tracers[index]) + k, :]
                                        for index, count in (("nt_qice", nilyr), ("nt_sice", nilyr),
                                                             ("nt_qsno", nslyr))
                                        for k in range(count))), axis=-1))
        step = int(static["ice_calendar"]["istep1"])
        jax.debug.callback(lambda values: print_pond_states(label, step, points, values, names),
                           jnp.stack(samples), ordered=True)
    i = int(static["ice_init"]["debug_model_i"])
    j = int(static["ice_init"]["debug_model_j"])
    task = int(static["ice_init"]["debug_model_task"])
    block = int(static["ice_init"]["debug_model_iblk"])
    if i < 0 and j < 0 and task < 0 and block < 0:
        i, j, task, block = int(piloc[0]), int(pjloc[0]), int(pmloc[0]), int(pbloc[0])
    if i < 0 or j < 0 or task < 0 or block < 0:
        raise ValueError("(debug_ice) ERROR: debug_model_[i,j,iblk,mytask] not set correctly")
    if task == 0 and block == 0:
        jax.debug.callback(lambda state: print_state(state, label, i, j,
                                                    int(static["ice_calendar"]["istep1"])),
                           pytree["state"], ordered=True)
    return pytree


def print_pond_states(label, step, points, samples, names):
    """Write category pond states through the standard diagnostic stream."""
    nu = ice_fileunits.nu_diag
    for (i, j), categories in zip(points, samples):
        for n, sample in enumerate(categories, start=1):
            values = dict(zip(names, map(float, sample)))
            area = values["aicen"] * values["alvl"] * values["apnd"]
            nu.write(f"(pond_stage) step={step} checkpoint={label} i={i} j={j} cat={n} "
                     + " ".join(f"{key}={value:.16e}" for key, value in values.items())
                     + f" pond_area={area:.16e} pond_volume={area * values['hpnd']:.16e}\n")
    nu.flush()


def print_pond_inputs(step, points, samples):
    """Write therm1 pond intermediates outside the vectorized physics routine."""
    nu = ice_fileunits.nu_diag
    for (i, j), categories in zip(points, samples):
        for n, values in enumerate(categories, start=1):
            nu.write(f"(pond_therm1) step={step} i={i} j={j} cat={n} "
                     + " ".join(f"{name}={float(value):.16e}"
                                for name, value in zip(POND_DIAGNOSTIC_FIELDS, values)) + "\n")
    nu.flush()


def print_pond_solver(step, points, atmo_samples, picard_samples, nilyr, nslyr,
                      replay_atmo_coefficients=False, replay_atmo_psi=False,
                      replay_atmo_math=False):
    """Write bounded solver diagnostics after selecting cells outside vmap."""
    nu = ice_fileunits.nu_diag
    for point, atmo, picard in zip(points, atmo_samples, picard_samples):
        i, j, n = point
        for prefix, rows, names, start in (
            ("atmo_jax", atmo, ATMO_DIAGNOSTIC_FIELDS[:29], 0),
            ("picard_jax", picard[:3], PICARD_DIAGNOSTIC_FIELDS, 1),
        ):
            if prefix == "picard_jax" and not np.isfinite(picard[3, 3]):
                continue
            for record, values in enumerate(rows, start=start):
                if prefix == "atmo_jax" and (not np.isfinite(atmo[0, 8]) or atmo[0, 8] == 0.0):
                    break
                if (prefix == "atmo_jax" and record < 6
                        and (not np.isfinite(values[8]) or values[8] == 0.0)):
                    continue
                nu.write(f"({prefix}) step={step} i={i} j={j} cat={n} record={record} "
                         + " ".join(f"{name}={float(value):.17e}"
                                    for name, value in zip(names, values)) + "\n")
        if replay_atmo_psi:
            for record in range(1, 6):
                values = atmo[record]
                nu.write(f"(atmo_psi_replay) step={step} i={i} j={j} cat={n} record={record} "
                         f"applied={float(values[31]):.17e} "
                         f"holm={float(values[20]):.17e} hols={float(values[21]):.17e} "
                         f"psimh_before={float(values[29]):.17e} "
                         f"psimh_used={float(values[22]):.17e} "
                         f"psixh_before={float(values[30]):.17e} "
                         f"psixh_used={float(values[23]):.17e}\n")
        if replay_atmo_math:
            width = len(STABILITY_MATH_DIAGNOSTIC_FIELDS)
            for record in range(1, 6):
                for kind in (1, 2):
                    start = 32 + (kind - 1) * width
                    values = atmo[record, start:start + width]
                    nu.write(f"(atmo_math_replay) step={step} i={i} j={j} cat={n} "
                             f"record={record} kind={kind} "
                             f"hol={float(atmo[record, 19 + kind]):.17e} "
                             f"psi={float(atmo[record, 21 + kind]):.17e} "
                             + " ".join(f"{name}={float(value):.17e}"
                                        for name, value in zip(STABILITY_MATH_DIAGNOSTIC_FIELDS, values))
                             + "\n")
        if not np.isfinite(picard[3, 2]):
            continue
        if replay_atmo_coefficients:
            nu.write(f"(atmo_coeff_replay) step={step} i={i} j={j} cat={n} "
                     f"shcoef_before={float(atmo[6, 27]):.17e} "
                     f"shcoef_used={float(picard[3, 13]):.17e} "
                     f"lhcoef_before={float(atmo[6, 28]):.17e} "
                     f"lhcoef_used={float(picard[3, 14]):.17e}\n")
        scalar_names = (
            "lsnow", "lcold", "Tsf", "dt", "hilyr", "hslyr", "Tbot",
            "fswint", "fswsfc", "rhoa", "flw", "potT", "Qa", "shcoef",
            "lhcoef", "qpond", "qocn", "Spond")
        extra_names = ("sss", "w", "semi_implicit_Tsfc")
        if picard[4, 2] != 0.0:
            extra_names += ("dfsurfdTs_cpl", "dflatdTs_cpl", "fsurf_cpl", "flat_cpl")
        ninterfaces = nilyr + (nslyr if picard[3, 0] != 0.0 else 0) + 1
        groups = (
            ("entry", "scalar", 3, 1, 0, scalar_names),
            ("entry", "scalar", 4, 1, 1, extra_names),
            ("entry", "ice", 5, nilyr, 1,
             ("zqin", "zSin", "zTin", "phi", "km", "Iswabs", "dSdt")),
            ("entry", "snow", 5 + nilyr, nslyr, 1, ("zqsn", "zTsn", "ks", "Sswabs")),
            ("entry", "darcy", 5 + nilyr + nslyr, nilyr + 1, 0, ("q",)),
            ("prepared", "ice", 6 + 2 * nilyr + nslyr, nilyr, 1, ("zTin", "phi", "Sbr")),
            ("prepared", "snow", 6 + 3 * nilyr + nslyr, nslyr, 1, ("zTsn",)),
            ("prepared", "interface", 6 + 3 * nilyr + 2 * nslyr,
             ninterfaces, 1, ("dxp", "kcstar")),
            ("prepared", "scalar", 7 + 4 * nilyr + 3 * nslyr, 1, 0, ("einit",)),
        )
        for phase, kind, first, count, lower, names in groups:
            for k in range(count):
                values = picard[first + k]
                nu.write(f"(picard_entry_jax) step={step} i={i} j={j} cat={n} "
                         f"phase={phase} kind={kind} k={lower + k} "
                         + " ".join(f"{name}={float(value):.17e}"
                                    for name, value in zip(names, values)) + "\n")
    nu.flush()


def print_state(state, label, i, j, istep1):
    """Write the state/forcing values of one debug cell to the diagnostic log."""
    s = state["ice_state"]
    f = state["ice_flux"]
    tr = icepack_tracers
    p = icepack_parameters
    nu = ice_fileunits.nu_diag
    nu.write(f"(print_state) {label}\n")
    nu.write(f"(print_state) istep1, my_task, i, j, iblk: {istep1} 0 {i} {j} 0\n")
    nu.write(f"(print_state) Global i and j: {i} {j}\n")
    nu.write(f"(print_state) Lat, Lon (degrees): "
             f"{ice_grid.TLAT[i, j] * p.rad_to_deg} "
             f"{ice_grid.TLON[i, j] * p.rad_to_deg}\n")
    nu.write(f"aice {s['aice'][i, j]}\naice0 {s['aice0'][i, j]}\n")
    for n in range(int(ice_domain_size.ncat)):
        nu.write(f"n = {n + 1}\n")
        for name in ("aicen", "vicen", "vsnon"):
            nu.write(f"{name} {s[name][i, j, n]}\n")
        if s["aicen"][i, j, n] > p.puny:
            nu.write(f"hin {s['vicen'][i, j, n] / s['aicen'][i, j, n]}\n")
            nu.write(f"hsn {s['vsnon'][i, j, n] / s['aicen'][i, j, n]}\n")
        nu.write(f"Tsfcn {s['trcrn'][i, j, tr.nt_Tsfc, n]}\n")
        if tr.tr_fsd:
            nu.write(f"afsdn {s['trcrn'][i, j, tr.nt_fsd, n]}\n")
    eidebug = esdebug = 0.0
    for n in range(int(ice_domain_size.ncat)):
        for k in range(int(ice_domain_size.nilyr)):
            qi = s["trcrn"][i, j, tr.nt_qice + k, n]
            nu.write(f"qice, cat {n + 1} layer {k + 1} {qi}\n")
            eidebug += qi
            if s["aicen"][i, j, n] > p.puny:
                nu.write(f"qi/rhoi {qi / p.rhoi}\n")
        if s["vsnon"][i, j, n] > p.puny:
            for k in range(int(ice_domain_size.nslyr)):
                qs = s["trcrn"][i, j, tr.nt_qsno + k, n]
                esdebug += qs
                nu.write(f"qsnow, cat {n + 1} layer {k + 1} {qs}\n")
                nu.write(f"qs/rhos {qs / p.rhos}\n")
                nu.write(f"Tsnow {(p.Lfresh + qs / p.rhos) / p.cp_ice}\n")
        for k in range(int(ice_domain_size.nilyr)):
            nu.write(f"sice, cat {n + 1} layer {k + 1} "
                     f"{s['trcrn'][i, j, tr.nt_sice + k, n]}\n")
    nu.write(f"qice(i,j) {eidebug}\nqsnow(i,j) {esdebug}\n")
    for name in ("uvel", "vvel"):
        nu.write(f"{name}(i,j) {s[name][i, j]}\n")
    if ice_dyn_shared.grid_ice in ("C", "CD"):
        for name in (("uvelE", "uvelN") if ice_dyn_shared.grid_ice == "C"
                     else ("uvelE", "vvelE", "uvelN", "vvelN")):
            nu.write(f"{name}(i,j) {s[name][i, j]}\n")
    for title, names in (
        ("atm states and fluxes", ("uatm", "vatm", "potT", "Tair", "Qa", "rhoa",
                                  "swvdr", "swvdf", "swidr", "swidf", "flw", "frain", "fsnow")),
        ("ocn states and fluxes", ("frzmlt", "sst", "sss", "Tf", "uocn", "vocn",
                                  "strtltxU", "strtltyU")),
        ("srf states and fluxes", ("Tref", "Qref", "Uref", "fsens", "flat",
                                  "evap", "flwout")),
    ):
        nu.write(title + "\n")
        for name in names:
            nu.write(f"            {name:8s}= {f[name][i, j]}\n")
    nu.flush()
    # endif                     ! print_points


def make_runtime_diags(static):
    """Compile once per run; capture only fixed switches, layout and point indices."""
    config = {name: dict(static[name]) for name in (
        "ice_domain_size", "ice_domain", "icepack_tracers",
        "icepack_parameters", "ice_dyn_shared",
    )}
    config["points"] = tuple(
        (int(piloc[n]), int(pjloc[n]), int(pmloc[n])) for n in range(npnt)
    )
    config["print_global"] = bool(print_global)
    config["print_points"] = bool(print_points)
    config["aice_extmin"] = float(aice_extmin)
    return jax.jit(lambda pytree, dt: _runtime_diags_values(pytree, dt, config))


def _runtime_diags_values(pytree, dt, config):
    """Reduce full-grid diagnostics on device; return only totals and point values."""
    p = SimpleNamespace(**{**config["icepack_parameters"],
                           **pytree["parameters"]["icepack_parameters"]})
    tr = SimpleNamespace(**config["icepack_tracers"])
    sizes = SimpleNamespace(**config["ice_domain_size"])
    s = SimpleNamespace(**pytree["state"]["ice_state"])
    f = SimpleNamespace(**pytree["state"]["ice_flux"])
    b = SimpleNamespace(**pytree["state"]["ice_flux_bgc"])
    col = SimpleNamespace(**pytree["state"]["ice_arrays_column"])
    grid = pytree["constants"]["ice_grid"]
    diagnostics = pytree["state"]["ice_diagnostics"]
    report = {"summary": OrderedDict(), "budget": OrderedDict(),
              "isotopes": [], "aerosols": [], "points": []}
    ilo, ihi, jlo, jhi = sizes.physical_bounds
    cells = (slice(ilo, ihi + 1), slice(jlo, jhi + 1))
    north = grid["tarean"][cells]
    south = grid["tareas"][cells]
    if config["ice_domain"]["ns_boundary_type"].strip() == "tripoleT":
        # Serial global_sum excludes the duplicate half of the folded row.
        iglob = pytree["constants"]["ice_boundary"]["iglob"][ilo:ihi + 1]
        keep = iglob <= sizes.nx_global // 2
        north = north.at[:, -1].set(jnp.where(keep, north[:, -1], 0))
        south = south.at[:, -1].set(jnp.where(keep, south[:, -1], 0))

    def sum_pair(field):
        values = field[cells]
        return jnp.sum(values * north), jnp.sum(values * south)

    def max_pair(field):
        values = field[cells]
        return tuple(jnp.where(jnp.any(mask), jnp.max(jnp.where(mask, values, -jnp.inf)), 0)
                     for mask in (maskn, masks))

    aice = s.aice
    aice_init = s.aice_init
    vice = s.vice
    vsno = s.vsno
    arean, areas = sum_pair(aice)
    arean *= m2_to_km2
    areas *= m2_to_km2
    extentn, extents = sum_pair(aice >= config["aice_extmin"])
    extentn *= m2_to_km2
    extents *= m2_to_km2
    shmaxn, shmaxs = sum_pair(vice)
    snwmxn, snwmxs = sum_pair(vsno)
    ptotn = ptots = 0.0
    if tr.tr_pond_topo:
        pond = jnp.sum(s.aicen * s.trcrn[:, :, tr.nt_apnd, :] *
                      s.trcrn[:, :, tr.nt_hpnd, :], axis=2)
        ptotn, ptots = sum_pair(pond)

    # Interpolate momentum-grid velocities before reducing kinetic energy.
    grid_ice = config["ice_dyn_shared"]["grid_ice"]
    zero = jnp.zeros_like(vice)
    if grid_ice == "B":
        uvelT = ice_grid.grid_average_X2Y_jax("A", s.uvel, "U", zero, "T", grid, sizes.physical_bounds)
        vvelT = ice_grid.grid_average_X2Y_jax("A", s.vvel, "U", zero, "T", grid, sizes.physical_bounds)
        speed = jnp.sqrt(s.uvel**2 + s.vvel**2)
    elif grid_ice == "C":
        uvelT = ice_grid.grid_average_X2Y_jax("A", s.uvelE, "E", zero, "T", grid, sizes.physical_bounds)
        vvelT = ice_grid.grid_average_X2Y_jax("A", s.vvelN, "N", zero, "T", grid, sizes.physical_bounds)
        uvelN = ice_grid.grid_average_X2Y_jax("A", s.uvelE, "E", zero, "N", grid, sizes.physical_bounds)
        vvelE = ice_grid.grid_average_X2Y_jax("A", s.vvelN, "N", zero, "E", grid, sizes.physical_bounds)
        speed = jnp.maximum(jnp.hypot(s.uvelE, vvelE), jnp.hypot(uvelN, s.vvelN))
    else:
        # Match the four weighted terms of grid_average_X2Y_NE (EN -> T).
        velocities = []
        for east, northward in ((s.uvelE, s.uvelN), (s.vvelE, s.vvelN)):
            acc = jnp.zeros_like(vice[cells])
            weight = jnp.zeros_like(acc)
            for field, area, di, dj in (
                (northward, grid["narea"], 0, -1), (northward, grid["narea"], 0, 0),
                (east, grid["earea"], -1, 0), (east, grid["earea"], 0, 0),
            ):
                sl = (slice(ilo + di, ihi + 1 + di), slice(jlo + dj, jhi + 1 + dj))
                acc = acc + field[sl] * area[sl]
                weight = weight + area[sl]
            velocities.append(zero.at[cells].set(
                jnp.where(weight != 0, acc / jnp.where(weight != 0, weight, 1), 0)))
        uvelT, vvelT = velocities
        speed = jnp.maximum(jnp.hypot(s.uvelE, s.vvelE), jnp.hypot(s.uvelN, s.vvelN))
    ketotn, ketots = sum_pair(0.5 * (p.rhos * vsno + p.rhoi * vice) *
                              (uvelT**2 + vvelT**2))
    urmsn = 2.0 * ketotn / (p.rhoi * shmaxn + p.rhos * snwmxn + p.puny)
    urmss = 2.0 * ketots / (p.rhoi * shmaxs + p.rhos * snwmxs + p.puny)
    urmsn = jnp.where(urmsn > p.puny, jnp.sqrt(jnp.maximum(urmsn, 0)), 0)
    urmss = jnp.where(urmss > p.puny, jnp.sqrt(jnp.maximum(urmss, 0)), 0)

    albedo = (f.alvdr * p.awtvdr + f.alidr * p.awtidr +
              f.alvdf * p.awtvdf + f.alidf * p.awtidf)
    lit = albedo[cells] > p.puny
    an = jnp.sum(aice[cells] * north * lit)
    a_s = jnp.sum(aice[cells] * south * lit)
    albtotn = jnp.where(an > 0, jnp.sum(aice[cells] * albedo[cells] * north * lit) / jnp.where(an > 0, an, 1), 0)
    albtots = jnp.where(a_s > 0, jnp.sum(aice[cells] * albedo[cells] * south * lit) / jnp.where(a_s > 0, a_s, 1), 0)
    maskn = grid["lmask_n"][cells].astype(bool)
    masks = grid["lmask_s"][cells].astype(bool)
    hmaxn, hmaxs = max_pair(vice)
    umaxn, umaxs = max_pair(speed)
    pmaxn, pmaxs = (value / 1000.0 for value in max_pair(s.strength))
    report["max_speed"] = jnp.stack((umaxn, umaxs))

    for label, x, y in (
        ("total ice area  (km^2) = ", arean, areas),
        ("total ice extent(km^2) = ", extentn, extents),
        ("total ice volume (m^3) = ", shmaxn, shmaxs),
        ("total snw volume (m^3) = ", snwmxn, snwmxs),
        ("tot kinetic energy (J) = ", ketotn, ketots),
        ("rms ice speed    (m/s) = ", urmsn, urmss),
        ("average albedo         = ", albtotn, albtots),
        ("max ice volume     (m) = ", hmaxn, hmaxs),
        ("max ice speed    (m/s) = ", umaxn, umaxs),
        ("max strength    (kN/m) = ", pmaxn, pmaxs),
    ):
        report["summary"][label] = jnp.stack((x, y))

    if config["print_global"]:
        # total_energy and total_salt aggregate the same category/layer
        # tracers as init_mass_diags, with land cells excluded.
        qice = s.trcrn[:, :, tr.nt_qice:tr.nt_qice + sizes.nilyr, :]
        qsno = s.trcrn[:, :, tr.nt_qsno:tr.nt_qsno + sizes.nslyr, :]
        sice = s.trcrn[:, :, tr.nt_sice:tr.nt_sice + sizes.nilyr, :]
        energy = (jnp.sum(qice, axis=2) * s.vicen / sizes.nilyr +
                  jnp.sum(qsno, axis=2) * s.vsnon / sizes.nslyr)
        energy = jnp.where(grid["tmask"], jnp.sum(energy, axis=2), 0.0)
        salt = jnp.where(grid["tmask"],
                        jnp.sum(jnp.sum(sice, axis=2) * s.vicen / sizes.nilyr, axis=2), 0.0)
        etotn, etots = sum_pair(energy)
        stotn, stots = sum_pair(salt)
        evpn, evps = (x * dt for x in sum_pair(f.evap * aice))
        shmaxnt = shmaxst = 0.0
        if tr.tr_brine:
            shmaxnt, shmaxst = sum_pair(vice * s.trcr[:, :, tr.nt_fbri])
        sfsaltn, sfsalts = (x * dt for x in sum_pair(f.fsalt_ai))
        sfreshn, sfreshs = (x * dt for x in sum_pair(f.fresh_ai))
        spondn = sponds = 0.0
        if tr.tr_pond_topo:
            spondn, sponds = (x * dt for x in sum_pair(f.fpond))
        fhocnn, fhocns = sum_pair(f.fhocn_ai)
        if p.calc_Tsfc:
            atmo_heat = ((f.fswabs - f.fswthru + f.fsens + f.flwout) * aice +
                         f.flw * aice_init)
        else:
            atmo_heat = (f.fsurf - f.flat) * aice
        fhatmn, fhatms = sum_pair(atmo_heat)
        fswnetn, fswnets = sum_pair(f.fswabs * aice)
        swdown = ((aice_init - f.alvdr_init) * f.swvdr +
                  (aice_init - f.alidr_init) * f.swidr +
                  (aice_init - f.alvdf_init) * f.swvdf +
                  (aice_init - f.alidf_init) * f.swidf)
        fswdnn, fswdns = sum_pair(swdown)
        fhfrzn, fhfrzs = sum_pair(jnp.maximum(0.0, f.frzmlt_init))
        rnn, rns = (x * dt for x in sum_pair(f.frain * aice_init))
        snn, sns = (x * dt for x in sum_pair(f.fsnow * aice_init))
        frazil = f.frazil
        if not p.update_ocn_f and p.ktherm == 2 and p.cpl_frazil == "fresh_ice_correction":
            frazil = frazil - f.frazil_diag
        frzn, frzs = (x * p.rhoi for x in sum_pair(frazil))

        micen, mices = p.rhoi * shmaxn, p.rhoi * shmaxs
        msnwn, msnws = p.rhos * snwmxn, p.rhos * snwmxs
        mpndn, mpnds = p.rhofresh * ptotn, p.rhofresh * ptots
        mtotn, mtots = micen + msnwn + mpndn, mices + msnws + mpnds
        delmin, delmis = mtotn - diagnostics["totmn"], mtots - diagnostics["totms"]
        delmxn, delmxs = micen - diagnostics["totmin"], mices - diagnostics["totmis"]
        if not p.update_ocn_f:
            delmxn -= frzn
            delmxs -= frzs
        fluxn = jnp.where(arean > 0, rnn + snn + evpn - sfreshn + (0.0 if p.update_ocn_f else frzn), 0)
        fluxs = jnp.where(areas > 0, rns + sns + evps - sfreshs + (0.0 if p.update_ocn_f else frzs), 0)
        werrn, werrs = (fluxn - delmin) / (mtotn + 1), (fluxs - delmis) / (mtots + 1)
        delein, deleis = etotn - diagnostics["toten"], etots - diagnostics["totes"]
        fhatmn += (-snn * p.Lfresh + evpn * p.Lvap) / dt
        fhatms += (-sns * p.Lfresh + evps * p.Lvap) / dt
        hnetn = (fhatmn - fhocnn - fhfrzn) * dt
        hnets = (fhatms - fhocns - fhfrzs) * dt
        herrn, herrs = (hnetn - delein) / (etotn - 1), (hnets - deleis) / (etots - 1)
        swerrn, swerrs = (fswnetn - fswdnn) / (fswnetn - 1), (fswnets - fswdns) / (fswnets - 1)
        if p.saltflux_option == "prognostic":
            msltn, mslts = stotn * p.rhoi * 0.001, stots * p.rhoi * 0.001
            delmsltn = p.rhoi * (stotn - diagnostics["totsn"]) * 0.001
            delmslts = p.rhoi * (stots - diagnostics["totss"]) * 0.001
        else:
            msltn, mslts = micen * p.ice_ref_salinity * 0.001, mices * p.ice_ref_salinity * 0.001
            delmsltn, delmslts = delmxn * p.ice_ref_salinity * 0.001, delmxs * p.ice_ref_salinity * 0.001
        serrn = (sfsaltn + delmsltn) / (msltn + 1)
        serrs = (sfsalts + delmslts) / (mslts + 1)

        fields = [
            ("arwt rain h2o kg in dt = ", rnn, rns),
            ("arwt snow h2o kg in dt = ", snn, sns),
            ("arwt evap h2o kg in dt = ", evpn, evps),
            ("arwt frzl h2o kg in dt = ", frzn, frzs),
        ]
        if tr.tr_pond_topo:
            fields.append(("arwt fpnd h2o kg in dt = ", spondn, sponds))
        fields.extend([
            ("arwt frsh h2o kg in dt = ", sfreshn, sfreshs),
            ("arwt ice mass (kg)     = ", micen, mices),
            ("arwt snw mass (kg)     = ", msnwn, msnws),
        ])
        if tr.tr_pond_topo:
            fields.append(("arwt pnd mass (kg)     = ", mpndn, mpnds))
        fields.extend([
            ("arwt tot mass (kg)     = ", mtotn, mtots),
            ("arwt tot mass chng(kg) = ", delmin, delmis),
            ("arwt water flux        = ", fluxn, fluxs),
            ("water flux error       = ", werrn, werrs),
            ("arwt atm heat flux (W) = ", fhatmn, fhatms),
            ("arwt ocn heat flux (W) = ", fhocnn, fhocns),
            ("arwt frzl heat flux(W) = ", fhfrzn, fhfrzs),
            ("arwt tot energy    (J) = ", etotn, etots),
            ("arwt net heat      (J) = ", hnetn, hnets),
            ("arwt tot energy chng(J)= ", delein, deleis),
            ("arwt heat error        = ", herrn, herrs),
            ("arwt incoming sw (W)   = ", fswdnn, fswdns),
            ("arwt absorbed sw (W)   = ", fswnetn, fswnets),
            ("arwt swdn error        = ", swerrn, swerrs),
            ("total brine tr (m^3)   = ", shmaxnt, shmaxst),
            ("arwt salt mass (kg)    = ", msltn, mslts),
            ("arwt salt mass chng(kg)= ", delmsltn, delmslts),
            ("arwt salt flx in dt(kg)= ", sfsaltn, sfsalts),
            ("arwt salt flx error    = ", serrn, serrs),
        ])
        for label, x, y in fields:
            report["budget"][label] = jnp.stack((x, y))

        # Fortran's isotope loop sums all isotope species for each n.
        if tr.tr_iso:
            iso = (vsno * jnp.sum(s.trcr[:, :, tr.nt_isosno:tr.nt_isosno + tr.n_iso], axis=2) +
                   vice * jnp.sum(s.trcr[:, :, tr.nt_isoice:tr.nt_isoice + tr.n_iso], axis=2))
            isototn, isotots = sum_pair(iso)
            isomaxn, isomaxs = max_pair(iso)
            for n in range(tr.n_iso):
                an, a_s = (x * dt for x in sum_pair(b.fiso_atm[:, :, n] * aice_init))
                on, os = (x * dt for x in sum_pair(b.fiso_ocn[:, :, n] * aice))
                errn = (diagnostics["totison"][n] - isototn + an - on) / (isototn + 1)
                errs = (diagnostics["totisos"][n] - isotots + a_s - os) / (isotots + 1)
                report["isotopes"].append(OrderedDict())
                for label, x, y in (("fiso_atm (kg/m2)       = ", an, a_s),
                                    ("fiso_ocn (kg/m2)       = ", on, os),
                                    ("total iso (kg/m2)      = ", isototn, isotots),
                                    ("iso error              = ", errn, errs),
                                    ("maximum iso (kg/m2)    = ", isomaxn, isomaxs)):
                    report["isotopes"][-1][label] = jnp.stack((x, y))
        if tr.tr_aero:
            for n in range(tr.n_aero):
                nt = tr.nt_aero + 4 * n
                aero = ((s.trcr[:, :, nt] + s.trcr[:, :, nt + 1]) * vsno +
                        (s.trcr[:, :, nt + 2] + s.trcr[:, :, nt + 3]) * vice)
                totaln, totals = sum_pair(aero)
                maxn, maxs = max_pair(aero)
                an, a_s = (x * dt for x in sum_pair(b.faero_atm[:, :, n] * aice_init))
                on, os = (x * dt for x in sum_pair(b.faero_ocn[:, :, n] * aice))
                errn = (diagnostics["totaeron"][n] - totaln + an - on) / (totaln + 1)
                errs = (diagnostics["totaeros"][n] - totals + a_s - os) / (totals + 1)
                report["aerosols"].append(OrderedDict())
                for label, x, y in (("faero_atm (kg/m2)      = ", an, a_s),
                                    ("faero_ocn (kg/m2)      = ", on, os),
                                    ("total aero (kg/m2)     = ", totaln, totals),
                                    ("aero error             = ", errn, errs),
                                    ("maximum aero (kg/m2)   = ", maxn, maxs)):
                    report["aerosols"][-1][label] = jnp.stack((x, y))
    if config["print_points"]:
        # These are the last-step values at the two selected cells, not
        # history averages. init_mass_diags saved their starting values.
        point_values = []
        for n, (i, j, task) in enumerate(config["points"]):
            if task != 0:
                point_values.append({})
                continue
            a = aice[i, j]
            hi = jnp.where(a != 0, vice[i, j] / jnp.where(a != 0, a, 1), 0)
            hs = jnp.where(a != 0, vsno[i, j] / jnp.where(a != 0, a, 1), 0)
            fsd = 0.0
            if tr.tr_fsd:
                for nc in range(sizes.ncat):
                    for k in range(sizes.nfsd):
                        fsd += (s.trcrn[i, j, tr.nt_fsd + k, nc] *
                                col.floe_rad_c[k] * s.aicen[i, j, nc] / jnp.where(a != 0, a, 1))
                fsd = jnp.where(a != 0, fsd, 0)
            hbr = s.trcr[i, j, tr.nt_fbri] * hi if tr.tr_brine else 0.0
            snow_grain = snow_density = snow_ice = snow_liq = 0.0
            if tr.tr_snow:
                for k in range(sizes.nslyr):
                    snow_grain += s.trcr[i, j, tr.nt_rsnw + k]
                    snow_density += s.trcr[i, j, tr.nt_rhos + k]
                    snow_ice += s.trcr[i, j, tr.nt_smice + k] * vsno[i, j]
                    snow_liq += s.trcr[i, j, tr.nt_smliq + k] * vsno[i, j]
                snow_grain /= sizes.nslyr
                snow_density /= sizes.nslyr
                snow_ice /= sizes.nslyr
                snow_liq /= sizes.nslyr
                snow_grain = jnp.where(vsno[i, j] > 0, snow_grain, 0)
                snow_density = jnp.where(vsno[i, j] > 0, snow_density, 0)
                snow_ice = jnp.where(vsno[i, j] > 0, snow_ice, 0)
                snow_liq = jnp.where(vsno[i, j] > 0, snow_liq, 0)
            salt_cell = 0.0
            energy_cell = 0.0
            for nc in range(sizes.ncat):
                for k in range(sizes.nilyr):
                    salt_cell += (s.trcrn[i, j, tr.nt_sice + k, nc] *
                                  s.vicen[i, j, nc] / sizes.nilyr)
                    energy_cell += (s.trcrn[i, j, tr.nt_qice + k, nc] *
                                    s.vicen[i, j, nc] / sizes.nilyr)
                for k in range(sizes.nslyr):
                    energy_cell += (s.trcrn[i, j, tr.nt_qsno + k, nc] *
                                    s.vsnon[i, j, nc] / sizes.nslyr)
            salt_cell = jnp.where(grid["tmask"][i, j], salt_cell, 0)
            energy_cell = jnp.where(grid["tmask"][i, j], energy_cell, 0)
            point_values.append({
                "Tair": f.Tair[i, j] - p.Tffresh, "Qa": f.Qa[i, j],
                "fsnow": f.fsnow[i, j] * dt / p.rhos,
                "frain": f.frain[i, j] * dt / p.rhow,
                "fsw": f.fsw[i, j], "flw": f.flw[i, j], "aice": a,
                "hi": hi, "hs": hs, "hbr": hbr, "fsd": fsd,
                "salt": jnp.where(vice[i, j] != 0, salt_cell / jnp.where(vice[i, j] != 0, vice[i, j], 1), 0),
                "Tsfc": s.trcr[i, j, tr.nt_Tsfc],
                "evap": f.evap[i, j] * dt / p.rhoi,
                "fswabs": f.fswabs[i, j], "flwout": f.flwout[i, j],
                "flat": f.flat[i, j], "fsens": f.fsens[i, j],
                "fsurf": f.fsurf[i, j], "fcondtop": f.fcondtop[i, j],
                "meltt": f.meltt[i, j], "meltb": f.meltb[i, j],
                "meltl": f.meltl[i, j], "snoice": f.snoice[i, j],
                "dsnow": f.dsnow[i, j], "frazil": f.frazil[i, j],
                "congel": f.congel[i, j],
                "dhi": vice[i, j] - diagnostics["pdhi"][n], "dhs": vsno[i, j] - diagnostics["pdhs"][n],
                "de": -(energy_cell - diagnostics["pde"][n]) / dt,
                "sst": f.sst[i, j], "sss": f.sss[i, j],
                "Tf": f.Tf[i, j], "fhocn": -f.fhocn[i, j],
                "snow_grain": snow_grain, "snow_density": snow_density,
                "snow_ice": snow_ice, "snow_liq": snow_liq,
                "snow_mass": p.rhos * vsno[i, j],
            })

        report["points"] = point_values
    return report


def runtime_diags(report):
    """Write small diagnostic results already transferred from the device."""
    nu = ice_fileunits.nu_diag
    p = icepack_parameters
    tr = icepack_tracers
    umaxn, umaxs = report["max_speed"]
    if check_umax and (umaxn > umax_stab or umaxs > umax_stab):
        # Only the exceptional warning path inspects the host velocity fields.
        s = ice_state
        if ice_dyn_shared.grid_ice == "B":
            speed = np.sqrt(s.uvel**2 + s.vvel**2)
        elif ice_dyn_shared.grid_ice == "C":
            uvelN = np.zeros_like(s.vice)
            vvelE = np.zeros_like(s.vice)
            ice_grid.grid_average_X2Y("A", s.uvelE, "E", uvelN, "N")
            ice_grid.grid_average_X2Y("A", s.vvelN, "N", vvelE, "E")
            speed = np.maximum(np.hypot(s.uvelE, vvelE), np.hypot(uvelN, s.vvelN))
        else:
            speed = np.maximum(np.hypot(s.uvelE, s.vvelE), np.hypot(s.uvelN, s.vvelN))
        target = umaxn if umaxn > umax_stab else umaxs
        for i, j in zip(*np.where(np.abs(speed - target) < p.puny)):
            nu.write(" \nWarning, large ice speed\n")
            nu.write(f"my_task, iblk, i, j, umax: 0 0 {i} {j} {target}\n")

    nu.write(f"{'Arctic':>51}  {'Antarctic':>24}\n")
    for label, (x, y) in report["summary"].items():
        nu.write(f"{label:25s}  {x:24.17e}  {y:24.17e}\n")
    if print_global:
        nu.write(" ----------------------------\n")
        for label, (x, y) in report["budget"].items():
            nu.write(f"{label:25s}  {x:24.17e}  {y:24.17e}\n")
        for title, groups in (("isotope", report["isotopes"]), ("aerosol", report["aerosols"])):
            for n, fields in enumerate(groups):
                nu.write(f"  {title} {n + 1}\n")
                for label, (x, y) in fields.items():
                    nu.write(f"{label:25s}  {x:24.17e}  {y:24.17e}\n")
    if print_points:
        point_values = report["points"]

        nu.write("                          \n")
        nu.write(f"{'Lat, Long':25s}  {plat[0]:6.1f} {plon[0]:6.1f}  "
                 f"{plat[1]:6.1f} {plon[1]:6.1f}\n")
        nu.write(f"{'my_task, iblk, i, j':25s}  "
                 f"{pmloc[0]} {pbloc[0]} {piloc[0]} {pjloc[0]}  "
                 f"{pmloc[1]} {pbloc[1]} {piloc[1]} {pjloc[1]}\n")
        nu.write("----------atm----------\n")
        rows = [("air temperature (C)    = ", "Tair"),
                ("specific humidity      = ", "Qa"),
                ("snowfall (m)           = ", "fsnow"),
                ("rainfall (m)           = ", "frain")]
        if p.calc_Tsfc:
            rows.extend([("shortwave radiation sum= ", "fsw"),
                         ("longwave radiation     = ", "flw")])
        else:
            rows.extend([("total surface heat flux= ", "fsurf"),
                         ("top sfc conductive flux= ", "fcondtop"),
                         ("latent heat flx        = ", "flat")])
        rows.extend([("----------ice----------", None),
                     ("area fraction          = ", "aice"),
                     ("avg ice thickness (m)  = ", "hi"),
                     ("avg snow depth (m)     = ", "hs"),
                     ("avg salinity (ppt)     = ", "salt"),
                     ("avg brine thickness (m)= ", "hbr")])
        if tr.tr_fsd:
            rows.append(("avg fsd rep radius (m) = ", "fsd"))
        if p.calc_Tsfc:
            rows.extend([("surface temperature(C) = ", "Tsfc"),
                         ("absorbed shortwave flx = ", "fswabs"),
                         ("outward longwave flx   = ", "flwout"),
                         ("sensible heat flx      = ", "fsens"),
                         ("latent heat flx        = ", "flat")])
        rows.extend([("subl/cond (m ice)      = ", "evap"),
                     ("top melt (m)           = ", "meltt"),
                     ("bottom melt (m)        = ", "meltb"),
                     ("lateral melt (m)       = ", "meltl"),
                     ("new ice (m)            = ", "frazil"),
                     ("congelation (m)        = ", "congel"),
                     ("snow-ice (m)           = ", "snoice"),
                     ("snow change (m)        = ", "dsnow"),
                     ("effective dhi (m)      = ", "dhi"),
                     ("effective dhs (m)      = ", "dhs"),
                     ("intnl enrgy chng(W/m^2)= ", "de")])
        if tr.tr_snow:
            if p.snwredist.strip() != "none":
                rows.append(("avg snow density(kg/m3)= ", "snow_density"))
            if p.snwgrain:
                rows.extend([("avg snow grain radius  = ", "snow_grain"),
                             ("mass ice in snow(kg/m2)= ", "snow_ice"),
                             ("mass liq in snow(kg/m2)= ", "snow_liq"),
                             ("mass std snow   (kg/m2)= ", "snow_mass")])
        rows.extend([("----------ocn----------", None),
                     ("sst (C)                = ", "sst"),
                     ("sss (ppt)              = ", "sss"),
                     ("freezing temp (C)      = ", "Tf"),
                     ("heat used (W/m^2)      = ", "fhocn")])
        for label, key in rows:
            if key is None:
                nu.write(label + "\n")
            else:
                nu.write(f"{label:25s}  {point_values[0][key]:24.17f}  "
                         f"{point_values[1][key]:24.17f}\n")
    nu.flush()
