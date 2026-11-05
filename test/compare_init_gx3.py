"""Compare gx3 initialization logs and iceh_ic files.

Defaults are relative to the repository root. Fortran text before
istep1: 24 is the initialization section. Integer indexes must match.
Log reals use relative tolerance 1e-12. iceh_ic is float32: relative
tolerance 1e-6, or the absolute tolerance for that field when the
values are near zero.

A row is PASS, FAIL, or LIST. LIST is an expected or one-sided item
and does not change the exit code. Any FAIL exits with status 1.
"""

import os
import re
import sys
from pathlib import Path

import numpy as np

REL_LOG = 1e-12
REL_HIST = 1e-6
# Absolute tolerance is SCALE * 1e-6, about one single-precision unit
# of the field's typical magnitude. Names not listed use DEFAULT_SCALE.
DEFAULT_SCALE = 1.0
SCALE = {}
for scale, names in (
    (1.0, "aice alvl ardg apond apond_ai apeff apeff_ai scale_factor sig1 sig2 ice_present uvel vvel uatm vatm atmspd uocn vocn ocnspd strairx strairy dagedtt dagedtd"),
    (10.0, "hi hs vlvl vrdg hpond hpond_ai ipond ipond_ai snow_ai rain_ai congel frazil snoice meltt melts meltb meltl fresh_ai evap_ai divu shear daidtt daidtd dvidtt dvidtd iage"),
    (40.0, "sice sss"),
    (100.0, "Tsfc sst Tair albsni"),
    (360.0, "atmdir ocndir TLON TLAT ULON ULAT NLON NLAT ELON ELAT lont_bounds latt_bounds lon lat"),
    (365.0, "mlt_onset frz_onset"),
    (1.0e3, "fswdn flwdn fswint_ai fswabs_ai flat_ai fsens_ai flwup_ai frzmlt fbot fhocn_ai fswthru_ai"),
    (1.0e5, "strength sigP trsig"),
    (1.0e-4, "fsalt_ai"),
    (1.0e11, "tarea uarea narea earea"),
):
    for name in names.split():
        SCALE[name] = scale

NUM = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?"

ROOT = Path(__file__).resolve().parents[1]
FORTRAN_LOG = Path("../CICE/cice-dirs/cases/gx3_test/logs/cice.runlog.261008-141559")
FORTRAN_IC = Path("../CICE/cice-dirs/runs/gx3_test/history/iceh_ic.2005-01-01-00000.nc")
PYTHON_LOG = Path("cases/gx3_test/cice.runlog")
PYTHON_IC = Path("cases/gx3_test/history/iceh_ic.2005-01-01-00000.nc")

rows = []
fail_count = 0


def add(name, status, abs_diff, rel_diff, where):
    global fail_count
    if status == "FAIL":
        fail_count += 1
    rows.append((name, status, abs_diff, rel_diff, where))


def close_pair(left, right, rel_tol, abs_tol):
    abs_diff = abs(left - right)
    scale = max(abs(left), abs(right))
    rel_diff = abs_diff / scale if scale != 0.0 else 0.0
    ok = abs_diff <= abs_tol or rel_diff <= rel_tol
    return ok, abs_diff, rel_diff


def init_text(text):
    match = re.search(r"^istep1:\s+24\b", text, re.M)
    if match is None:
        return text
    return text[: match.start()]


def triples(text, pattern):
    return [(float(a), float(b), float(c), rest) for a, b, c, rest in re.findall(pattern, text)]


def compare_series(title, left, right, rel_tol):
    count = min(len(left), len(right))
    for index in range(count):
        a0, a1, a2, a_name = left[index]
        b0, b1, b2, b_name = right[index]
        name = f"{title} {a_name}#{index + 1}"
        if str(a_name) != str(b_name):
            add(name, "FAIL", None, None, f"name {a_name} vs {b_name}")
            continue
        worst_abs = 0.0
        worst_rel = 0.0
        ok = True
        for left_value, right_value in ((a0, b0), (a1, b1), (a2, b2)):
            good, abs_diff, rel_diff = close_pair(left_value, right_value, rel_tol, 0.0)
            ok = ok and good
            if abs_diff >= worst_abs:
                worst_abs = abs_diff
                worst_rel = rel_diff
        add(name, "PASS" if ok else "FAIL", worst_abs, worst_rel, "min,max,sum")
    if len(left) != len(right):
        add(
            title,
            "FAIL",
            None,
            None,
            f"count fortran={len(left)} python={len(right)}",
        )


def namelist_map(text):
    start = text.find("Overview of model configuration")
    if start < 0:
        return None
    marker = text.find("n_fep", start)
    if marker < 0:
        return None
    end = text.find("\n", marker)
    found = {}
    for line in text[start:end].splitlines():
        if "=" not in line:
            continue
        left, right = line.split("=", 1)
        name = left.strip()
        if not name:
            continue
        value = right.split(" :")[0].strip()
        found.setdefault(name, []).append(value)
    return found


def same_tokens(left, right):
    # The two runs use different machine roots for the same CICE/input files.
    left_n = re.sub(r"^.*?/CICE/input/", "INPUT/", left)
    right_n = re.sub(r"^.*?/CICE/input/", "INPUT/", right)
    path_only = left_n == right_n and (left_n != left or right_n != right)
    left_tok = left.split()
    right_tok = right.split()
    if left_tok and len(left_tok) == len(right_tok):
        try:
            numbers = [(float(a), float(b)) for a, b in zip(left_tok, right_tok)]
        except ValueError:
            numbers = None
        if numbers is not None:
            worst_abs = 0.0
            worst_rel = 0.0
            ok = True
            for a, b in numbers:
                good, abs_diff, rel_diff = close_pair(a, b, REL_LOG, 0.0)
                ok = ok and good
                if abs_diff >= worst_abs:
                    worst_abs = abs_diff
                    worst_rel = rel_diff
            return ok, worst_abs, worst_rel, False
    if left_n == right_n:
        return True, 0.0, 0.0, path_only
    return False, None, None, False


def tracer_block(text):
    header = "(icepack_write_tracer_indices)"
    start = text.find(header)
    if start < 0:
        return None
    values = {}
    for line in text[start:].splitlines()[1:]:
        if not line.strip():
            break
        match = re.match(r"\s*([A-Za-z0-9_]+)\s*=\s*(-?\d+)", line)
        if match is None:
            break
        values[match.group(1)] = int(match.group(2))
    return values


def expected_index(name, fortran_value):
    # The log prints Fortran's 1-based tracer numbers. Storage stays 0-based.
    return fortran_value


def transport_rows(text):
    header = "tracer          index  depend  type has_dependents"
    start = text.find(header)
    if start < 0:
        return None
    parsed = []
    for line in text[start:].splitlines()[1:]:
        if not line.strip():
            break
        parts = line.split()
        if len(parts) < 5:
            continue
        parsed.append((parts[0], int(parts[1]), int(parts[2]), int(parts[3]), parts[4]))
    return parsed


def history_mask(variable, data):
    if not np.issubdtype(data.dtype, np.floating):
        return np.zeros(data.shape, dtype=bool)
    mask = ~np.isfinite(data)
    for key in ("_FillValue", "missing_value"):
        if key in variable.ncattrs():
            fill = np.asarray(getattr(variable, key), dtype=data.dtype)
            mask = mask | (data == fill)
    return mask | (np.abs(data) >= 1.0e20)


def compare_history(fortran_path, python_path):
    try:
        from netCDF4 import Dataset
    except ImportError:
        add("iceh_ic", "FAIL", None, None, "netCDF4 is not installed")
        return
    fortran = Dataset(fortran_path)
    python = Dataset(python_path)
    try:
        fortran_names = set(fortran.variables)
        python_names = set(python.variables)
        for name in sorted(python_names - fortran_names):
            add(f"iceh {name}", "LIST", None, None, "python only")
        for name in sorted(fortran_names):
            if name not in python_names:
                add(f"iceh {name}", "FAIL", None, None, "missing in python")
                continue
            left = fortran.variables[name]
            right = python.variables[name]
            left_data = np.array(left[:])
            right_data = np.array(right[:])
            if left_data.shape != right_data.shape:
                add(
                    f"iceh {name}",
                    "FAIL",
                    None,
                    None,
                    f"shape {left_data.shape} vs {right_data.shape}",
                )
                continue
            if left_data.dtype.kind in "SU":
                ok = np.array_equal(left_data, right_data)
                add(f"iceh {name}", "PASS" if ok else "FAIL", None, None, "text")
                continue
            if not np.issubdtype(left_data.dtype, np.number):
                add(f"iceh {name}", "LIST", None, None, f"dtype {left_data.dtype}")
                continue
            left_miss = history_mask(left, left_data)
            right_miss = history_mask(right, right_data)
            if left_miss.shape != right_miss.shape or np.any(left_miss != right_miss):
                mismatch = np.argwhere(left_miss != right_miss)
                where = tuple(int(item) for item in mismatch[0]) if len(mismatch) else "mask"
                add(f"iceh {name}", "FAIL", None, None, f"missing-mask {where} count={len(mismatch)}")
                continue
            valid = ~left_miss
            if not np.any(valid):
                add(f"iceh {name}", "PASS", 0.0, 0.0, "all missing")
                continue
            if not np.issubdtype(left_data.dtype, np.floating):
                diff = np.abs(left_data.astype(np.float64) - right_data.astype(np.float64))
                diff = np.where(valid, diff, 0.0)
                flat = int(np.argmax(diff))
                worst = float(diff.ravel()[flat])
                loc = np.unravel_index(flat, diff.shape)
                add(f"iceh {name}", "PASS" if worst == 0.0 else "FAIL", worst, None, loc)
                continue
            left_f = left_data.astype(np.float64)
            right_f = right_data.astype(np.float64)
            abs_diff = np.abs(left_f - right_f)
            scale = np.maximum(np.abs(left_f), np.abs(right_f))
            rel_diff = np.where(scale == 0.0, 0.0, abs_diff / scale)
            atol = SCALE.get(name, DEFAULT_SCALE) * REL_HIST
            bad = valid & (abs_diff > atol) & (rel_diff > REL_HIST)
            if np.any(bad):
                score = np.where(bad, rel_diff, -1.0)
                flat = int(np.argmax(score))
                status = "FAIL"
            else:
                shown = np.where(valid, abs_diff, -1.0)
                flat = int(np.argmax(shown))
                status = "PASS"
            loc = np.unravel_index(flat, abs_diff.shape)
            where = loc
            if name == "albsni" and status == "FAIL":
                where = (f"{loc} fortran={left_f.ravel()[flat]:.9g} "
                         f"python={right_f.ravel()[flat]:.9g} bad_cells={np.count_nonzero(bad)}")
            if name == "blkmask" and status == "FAIL":
                # This identifies Fortran's task and block decomposition.
                status = "LIST"
            add(f"iceh {name}", status, float(abs_diff.ravel()[flat]), float(rel_diff.ravel()[flat]), where)
    finally:
        fortran.close()
        python.close()


os.chdir(ROOT)
missing = [str(path) for path in (FORTRAN_LOG, FORTRAN_IC, PYTHON_LOG, PYTHON_IC) if not path.is_file()]
if missing:
    for path in missing:
        add(path, "FAIL", None, None, "file not found")
else:
    fortran_init = init_text(FORTRAN_LOG.read_text(encoding="utf-8", errors="replace"))
    python_init = init_text(PYTHON_LOG.read_text(encoding="utf-8", errors="replace"))

    restart_re = re.compile(rf"min,\s*max,\s*sum\s*=\s*({NUM})\s+({NUM})\s+({NUM})\s+(\S+)")
    compare_series(
        "restart",
        triples(fortran_init, restart_re),
        triples(python_init, restart_re),
        REL_LOG,
    )

    grid_re = re.compile(
        rf"\((?:ice_read_global|ice_read_xyt)\)\s+read_global\s+\S+\s+(\d+)\s+({NUM})\s+({NUM})\s+({NUM})"
    )
    # Fortran reads grid record 1 and kmt once more for the block
    # distribution, before (init_grid1). Only the reads after it match Python.
    grid_start = fortran_init.rfind("(init_grid1)")
    fortran_grid_text = fortran_init[grid_start:] if grid_start >= 0 else fortran_init
    # The record number is the name so a mismatched record fails the row.
    fortran_grid = [
        (float(b), float(c), float(d), f"nrec{a}")
        for a, b, c, d in re.findall(grid_re, fortran_grid_text)
    ]
    python_grid = [
        (float(b), float(c), float(d), f"nrec{a}")
        for a, b, c, d in re.findall(grid_re, python_init)
    ]
    compare_series("grid", fortran_grid, python_grid, REL_LOG)

    hin_re = re.compile(rf"({NUM})\s*<\s*Cat\s+(\d+)\s*<\s*({NUM})")
    fortran_hin = hin_re.findall(fortran_init[fortran_init.find("hin_max(n-1)"):])
    python_hin = hin_re.findall(python_init[python_init.find("hin_max(n-1)"):])
    if len(fortran_hin) < 5 or len(python_hin) < 5:
        add("hin_max", "FAIL", None, None, f"count fortran={len(fortran_hin)} python={len(python_hin)}")
    else:
        for index in range(5):
            left0, cat, left1 = fortran_hin[index]
            right0, _cat, right1 = python_hin[index]
            ok0, abs0, rel0 = close_pair(float(left0), float(right0), REL_LOG, 0.0)
            ok1, abs1, rel1 = close_pair(float(left1), float(right1), REL_LOG, 0.0)
            worst = abs0 if abs0 >= abs1 else abs1
            rel = rel0 if abs0 >= abs1 else rel1
            add(f"hin_max cat{cat}", "PASS" if ok0 and ok1 else "FAIL", worst, rel, "bounds")

    fortran_tracers = tracer_block(fortran_init)
    python_tracers = tracer_block(python_init)
    if fortran_tracers is None or python_tracers is None:
        add(
            "tracer_indices",
            "FAIL",
            None,
            None,
            f"section present fortran={fortran_tracers is not None} python={python_tracers is not None}",
        )
    else:
        for name in sorted(set(fortran_tracers) | set(python_tracers)):
            if name not in fortran_tracers or name not in python_tracers:
                add(f"tracer {name}", "LIST", None, None, "one side only")
                continue
            expect = expected_index(name, fortran_tracers[name])
            got = python_tracers[name]
            add(
                f"tracer {name}",
                "PASS" if got == expect else "FAIL",
                abs(got - expect),
                None,
                f"fortran {fortran_tracers[name]} python {got} expect {expect}",
            )

    def first_int(pattern, text):
        match = re.search(pattern, text)
        if match is None:
            return None
        return int(match.group(1))

    fortran_ntrcr = first_int(r"(?<![A-Za-z0-9_])ntrcr\s*=\s*(-?\d+)", fortran_init)
    python_ntrcr = first_int(r"(?<![A-Za-z0-9_])ntrcr\s*=\s*(-?\d+)", python_init)
    if fortran_ntrcr is None or python_ntrcr is None:
        add("ntrcr", "FAIL", None, None, f"fortran={fortran_ntrcr} python={python_ntrcr}")
    else:
        add("ntrcr", "PASS" if fortran_ntrcr == python_ntrcr else "FAIL", abs(fortran_ntrcr - python_ntrcr), None, f"{fortran_ntrcr} vs {python_ntrcr}")

    fortran_rows = transport_rows(fortran_init)
    python_rows = transport_rows(python_init)
    if fortran_rows is None or python_rows is None:
        add("transport", "FAIL", None, None, "table missing")
    else:
        count = min(len(fortran_rows), len(python_rows))
        for index in range(count):
            same = fortran_rows[index] == python_rows[index]
            add(
                f"transport {fortran_rows[index][0]}",
                "PASS" if same else "FAIL",
                None,
                None,
                f"{fortran_rows[index]} vs {python_rows[index]}",
            )
        if len(fortran_rows) != len(python_rows):
            add("transport", "FAIL", None, None, f"count {len(fortran_rows)} vs {len(python_rows)}")

    point_re = re.compile(
        rf"found point\s+(\d+)\s*\n[^\n]*\n\s*({NUM})\s+({NUM})\s+({NUM})\s+({NUM})\s+([+-]?\d+)\s+([+-]?\d+)\s+(\d+)"
    )
    fortran_points = point_re.findall(fortran_init)
    python_points = point_re.findall(python_init)
    block_match = re.search(r"block_size_x,_y\s*=\s*(\d+)\s+(\d+)", fortran_init)
    if not fortran_points or not python_points or block_match is None:
        add("diagnostic_points", "FAIL", None, None, "points or block size missing")
    else:
        block_nx = int(block_match.group(1))
        block_ny = int(block_match.group(2))
        for index, fortran_point in enumerate(fortran_points):
            if index >= len(python_points):
                add(f"point {fortran_point[0]}", "FAIL", None, None, "missing in python")
                continue
            python_point = python_points[index]
            # Fortran prints the task-local block number (bindx = iblk in
            # init_diags), not the global block, so only the position inside
            # a block is checked: i_glob = (iblock-1)*block_size_x + i - 1.
            di = int(python_point[5]) - (int(fortran_point[5]) - 1)
            dj = int(python_point[6]) - (int(fortran_point[6]) - 1)
            i_ok = di >= 0 and di % block_nx == 0
            j_ok = dj >= 0 and dj % block_ny == 0
            worst_abs = 0.0
            worst_rel = 0.0
            values_ok = True
            for slot in range(1, 5):
                good, abs_diff, rel_diff = close_pair(
                    float(fortran_point[slot]), float(python_point[slot]), REL_LOG, 1e-6
                )
                values_ok = values_ok and good
                if abs_diff >= worst_abs:
                    worst_abs = abs_diff
                    worst_rel = rel_diff
            add(
                f"point {fortran_point[0]}",
                "PASS" if i_ok and j_ok and values_ok else "FAIL",
                worst_abs,
                worst_rel,
                f"python i,j {python_point[5]},{python_point[6]}; fortran block-local i,j "
                f"{fortran_point[5]},{fortran_point[6]} in {block_nx}x{block_ny} blocks",
            )

    def four_floats(pattern, text):
        match = re.search(pattern, text)
        if match is None:
            return None
        return [float(item) for item in match.groups()]

    fortran_evp = four_floats(rf"arlx, arlxi, brlx, denom1\s+({NUM})\s+({NUM})\s+({NUM})\s+({NUM})", fortran_init)
    python_evp = four_floats(rf"arlx, arlxi, brlx, denom1\s+({NUM})\s+({NUM})\s+({NUM})\s+({NUM})", python_init)
    labels = ("arlx", "arlxi", "brlx", "denom1")
    if fortran_evp is None or python_evp is None:
        add("evp", "FAIL", None, None, "arlx line missing")
    else:
        for label, left, right in zip(labels, fortran_evp, python_evp):
            good, abs_diff, rel_diff = close_pair(left, right, REL_LOG, 0.0)
            add(label, "PASS" if good else "FAIL", abs_diff, rel_diff, "evp")
    for label, pattern in (
        ("dt_subcyle", rf"dt_subcyle\s*=\s*({NUM})"),
        ("tdamp", rf"tdamp\s*=\s*({NUM})"),
    ):
        left = four_floats(pattern, fortran_init)
        right = four_floats(pattern, python_init)
        if left is None or right is None:
            add(label, "FAIL", None, None, "line missing")
        else:
            good, abs_diff, rel_diff = close_pair(left[0], right[0], REL_LOG, 0.0)
            add(label, "PASS" if good else "FAIL", abs_diff, rel_diff, "evp")

    lat_re = re.compile(rf"\(NElatlon\) min/max\s+(\S+):\s+({NUM})\s+({NUM})")
    fortran_lat = {name: (float(a), float(b)) for name, a, b in lat_re.findall(fortran_init)}
    python_lat = {name: (float(a), float(b)) for name, a, b in lat_re.findall(python_init)}
    if not fortran_lat or not python_lat:
        add("NElatlon", "FAIL", None, None, f"lines fortran={len(fortran_lat)} python={len(python_lat)}")
    else:
        for name in ("ULON", "ULAT", "TLON", "TLAT", "NLON", "NLAT", "ELON", "ELAT"):
            key = name
            if key not in fortran_lat or key not in python_lat:
                add(f"NElatlon {name}", "FAIL", None, None, "line missing")
                continue
            ok0, abs0, rel0 = close_pair(fortran_lat[key][0], python_lat[key][0], REL_LOG, 0.0)
            ok1, abs1, rel1 = close_pair(fortran_lat[key][1], python_lat[key][1], REL_LOG, 0.0)
            worst = abs0 if abs0 >= abs1 else abs1
            rel = rel0 if abs0 >= abs1 else rel1
            add(f"NElatlon {name}", "PASS" if ok0 and ok1 else "FAIL", worst, rel, "min,max")

    time_res = (
        ("calc_timesteps from", re.compile(rf"modified npt from\s+(\d+)\s+(\S+)\s+with dt=\s*({NUM})")),
        ("calc_timesteps to", re.compile(rf"\(calc_timesteps\)\s+to\s+(\d+)\s+(\S+)\s+with dt=\s*({NUM})")),
        ("calc_timesteps start", re.compile(r"start time is\s+(\S+)")),
        ("calc_timesteps end", re.compile(r"end time is\s+(\S+)")),
    )
    for label, pattern in time_res:
        left = pattern.search(fortran_init)
        right = pattern.search(python_init)
        if left is None or right is None:
            add(label, "FAIL", None, None, "line missing")
            continue
        if left.groups()[-1:] and len(left.groups()) == 3:
            good, abs_diff, rel_diff = close_pair(float(left.group(3)), float(right.group(3)), REL_LOG, 0.0)
            text_ok = left.group(1) == right.group(1) and left.group(2) == right.group(2)
            add(label, "PASS" if good and text_ok else "FAIL", abs_diff, rel_diff, f"{left.groups()} vs {right.groups()}")
        else:
            add(label, "PASS" if left.group(1) == right.group(1) else "FAIL", None, None, f"{left.group(1)} vs {right.group(1)}")

    fortran_nml = namelist_map(fortran_init)
    python_nml = namelist_map(python_init)
    if fortran_nml is None or python_nml is None:
        add("namelist", "FAIL", None, None, "overview missing")
    else:
        for name in sorted(set(fortran_nml) | set(python_nml)):
            if name not in fortran_nml or name not in python_nml:
                side = "fortran only" if name in fortran_nml else "python only"
                add(f"nml {name}", "LIST", None, None, side)
                continue
            pairs = min(len(fortran_nml[name]), len(python_nml[name]))
            for index in range(pairs):
                ok, abs_diff, rel_diff, path_only = same_tokens(
                    fortran_nml[name][index], python_nml[name][index]
                )
                status = "LIST" if path_only else ("PASS" if ok else "FAIL")
                where = "input path prefix" if path_only else "value"
                add(f"nml {name}#{index + 1}", status, abs_diff, rel_diff, where)
            if len(fortran_nml[name]) != len(python_nml[name]):
                add(f"nml {name}", "FAIL", None, None, "occurrence count differs")

    compare_history(FORTRAN_IC, PYTHON_IC)

    if "Timing information:" in python_init:
        add("timer table", "LIST", None, None, "Python init_restart prints timers; Fortran waits until finalize")
    for label in ("Processors:", "block_size_x,_y", "(init_domain_distribution)", "(init_grid1)"):
        if label in fortran_init and label not in python_init:
            add(label, "LIST", None, None, "Fortran block or processor line, not in Python")

for name, status, abs_diff, rel_diff, where in rows:
    abs_text = "" if abs_diff is None else f"  max_abs={abs_diff:.6e}"
    rel_text = "" if rel_diff is None else f"  max_rel={rel_diff:.6e}"
    print(f"{name}  {status}{abs_text}{rel_text}  {where}")

passed = sum(1 for _name, status, _a, _r, _w in rows if status == "PASS")
listed = sum(1 for _name, status, _a, _r, _w in rows if status == "LIST")
print(f"SUMMARY  pass={passed}  fail={fail_count}  list={listed}")
sys.exit(1 if fail_count else 0)
