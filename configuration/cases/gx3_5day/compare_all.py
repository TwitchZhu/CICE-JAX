"""Report all restart variables and core history variables across three runs."""

import argparse
import csv
from datetime import datetime
from itertools import combinations
from pathlib import Path

import netCDF4
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--history-only", action="store_true",
                    help="Compare existing history files without requiring a completed final restart.")
args = parser.parse_args()

CASE = Path(__file__).resolve().parent
runs = {"GNU": CASE / "gnu", "Intel": CASE / "intel", "JAX": CASE}
filename = "iced.2005-01-06-00000.nc"
for label, directory in runs.items():
    if args.history_only:
        break
    logs = sorted(
        (path for folder in (directory, directory / "logs")
         for path in folder.glob("cice.runlog.*") if path.is_file()),
        key=lambda path: path.name,
    )
    run_log = logs[-1] if logs else directory / "cice.runlog"
    if not run_log.is_file():
        raise FileNotFoundError(f"Missing {label} run log: {directory}")
    print(f"{label} run log: {run_log}")
    if "CICE COMPLETED SUCCESSFULLY" not in run_log.read_text(encoding="utf-8", errors="replace"):
        raise RuntimeError(f"Missing completion marker in latest run: {run_log}")
    with netCDF4.Dataset(directory / "restart" / filename) as nc:
        for name, expected in {"istep1": 120, "myear": 2005, "mmonth": 1, "mday": 6, "msec": 0}.items():
            if nc.getncattr(name) != expected:
                raise ValueError(f"Wrong final time: {label} {name} != {expected}")

output = CASE / "comparison_logs" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
output.mkdir(parents=True)
columns = ["variable", "status", "dimensions", "dtype_left", "dtype_right", "size",
           "mask_mismatch", "nonfinite_left", "nonfinite_right", "finite_pairs",
           "unequal_finite", "max_abs", "rmse", "relative_l2_to_left"]
primary_variables = ("aice", "aicen", "hi", "vicen", "hs", "vsnon", "uvel", "vvel")
history_files = sorted({path.name for directory in runs.values()
                        for path in (directory / "history").glob("*.nc")})
history_report = output / "history.log"
with history_report.open("x", encoding="utf-8") as stream:
    stream.write("Core history variables: " + ", ".join(primary_variables) + "\n")
    stream.write("Files are paired by filename; iceh_ic files describe initialization.\n")
    stream.write("Metrics use jointly unmasked finite values, without area weighting.\n")
    stream.write("relative_l2_to_left = norm(right - left) / norm(left).\n")
    stream.write("No numerical acceptance threshold applied.\n")
    if not history_files:
        stream.write("NO_HISTORY_FILES\n")
comparisons = []
history_summaries = []
restart_results = []
for left_label, right_label in combinations(runs, 2):
    if not args.history_only:
        comparisons.append(("restart", filename, left_label, right_label))
    comparisons.extend(("history", name, left_label, right_label) for name in history_files)
for kind, name, left_label, right_label in comparisons:
    left_path = runs[left_label] / kind / name
    right_path = runs[right_label] / kind / name
    if kind == "history":
        summary_label = f"{left_label}–{right_label} | {name}"
        history_summaries.append(f"{summary_label}，比较未完成。")
        with history_report.open("a", encoding="utf-8") as stream:
            stream.write(f"\n{left_label} vs {right_label} | {name}\n")
            stream.write(f"left={left_path.relative_to(CASE)}\nright={right_path.relative_to(CASE)}\n")
            if not left_path.is_file() or not right_path.is_file():
                history_summaries[-1] = f"{summary_label}，缺少 history 文件，无法比较。"
                stream.write(f"MISSING_FILE left={not left_path.is_file()} right={not right_path.is_file()}\n")
                continue
    rows = []
    with netCDF4.Dataset(left_path) as left, netCDF4.Dataset(right_path) as right:
        if kind == "history":
            with history_report.open("a", encoding="utf-8") as stream:
                if "time" not in left.variables or "time" not in right.variables:
                    history_summaries[-1] = f"{summary_label}，缺少时间坐标，无法比较。"
                    stream.write("MISSING_TIME_COORDINATE; comparison skipped\n")
                    continue
                lt, rt = left["time"], right["time"]
                stream.write(f"time_left={lt[:].tolist()} time_right={rt[:].tolist()}\n")
                stream.write(f"time_units_left={getattr(lt, 'units', '')} "
                             f"time_units_right={getattr(rt, 'units', '')}\n")
                if (getattr(lt, "units", "") != getattr(rt, "units", "")
                        or getattr(lt, "calendar", "standard") != getattr(rt, "calendar", "standard")
                        or np.ma.getmaskarray(lt[:]).any() or np.ma.getmaskarray(rt[:]).any()
                        or not np.array_equal(lt[:], rt[:])):
                    stream.write("TIME_MISMATCH; comparison skipped\n")
                    history_summaries[-1] = f"{summary_label}，时间坐标不一致，未比较变量数值。"
                    continue
        variables = left.variables.keys() | right.variables.keys()
        ordered_variables = [name for name in primary_variables if name in variables]
        if kind == "restart":
            ordered_variables.extend(sorted(variables.difference(primary_variables)))
        else:
            with history_report.open("a", encoding="utf-8") as stream:
                absent = [name for name in primary_variables if name not in variables]
                stream.write("Absent in both files: " + (", ".join(absent) or "none") + "\n")
        for name in ordered_variables:
            row = {"variable": name}
            rows.append(row)
            if name not in left.variables or name not in right.variables:
                row["status"] = "MISSING_LEFT" if name not in left.variables else "MISSING_RIGHT"
                continue
            lv, rv = left[name], right[name]
            row.update(dimensions=str(lv.dimensions), dtype_left=str(lv.dtype),
                       dtype_right=str(rv.dtype), size=lv.size)
            if kind == "history":
                row.update(units_left=getattr(lv, "units", ""), units_right=getattr(rv, "units", ""))
                if row["units_left"] != row["units_right"]:
                    row["status"] = "UNITS_MISMATCH"
                    continue
            if lv.dimensions != rv.dimensions or lv.shape != rv.shape:
                row["status"] = "SHAPE_OR_DIMENSION_MISMATCH"
                continue
            a, b = np.ma.asarray(lv[:]), np.ma.asarray(rv[:])
            ma, mb = np.ma.getmaskarray(a), np.ma.getmaskarray(b)
            row["mask_mismatch"] = int(np.count_nonzero(ma != mb))
            av, bv = np.asarray(a.data), np.asarray(b.data)
            if av.dtype.kind not in "biuf" or bv.dtype.kind not in "biuf":
                row["status"] = "UNSUPPORTED_DTYPE"
                continue
            row["nonfinite_left"] = int(np.count_nonzero(~ma & ~np.isfinite(av)))
            row["nonfinite_right"] = int(np.count_nonzero(~mb & ~np.isfinite(bv)))
            valid = ~ma & ~mb & np.isfinite(av) & np.isfinite(bv)
            row["finite_pairs"] = int(np.count_nonzero(valid))
            row["unequal_finite"] = int(np.count_nonzero(av[valid] != bv[valid]))
            exact = not any(row[key] for key in (
                "mask_mismatch", "nonfinite_left", "nonfinite_right", "unequal_finite"))
            row["status"] = "EXACT" if exact else "DIFFERENCES"
            if not row["finite_pairs"]:
                row["status"] = "NO_FINITE_PAIRS"
            if row["finite_pairs"]:
                aa, bb = av[valid].astype(np.float64), bv[valid].astype(np.float64)
                diff = bb - aa
                row["max_abs"] = float(np.max(np.abs(diff)))
                row["rmse"] = float(np.sqrt(np.mean(diff * diff)))
                denominator = float(np.linalg.norm(aa))
                row["relative_l2_to_left"] = (float(np.linalg.norm(diff)) / denominator
                    if denominator else (0.0 if not np.any(diff) else "undefined_zero_reference"))
    if kind == "restart":
        restart_results.append((left_label, right_label, rows))
        report = output / f"{left_label}-{right_label}.csv"
        with report.open("x", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        print(f"{left_label} vs {right_label}: {report}")
    else:
        exact_names = [row["variable"] for row in rows if row["status"] == "EXACT"]
        if rows and len(exact_names) == len(rows):
            history_summaries[-1] = (
                f"{summary_label}，已比较的 {len(rows)} 个核心变量 "
                f"{', '.join(exact_names)} 均完全一致，最大绝对差、RMSE、相对 L2 误差均为 0，"
                "缺测掩膜一致，未掩膜位置无 NaN/Inf。"
            )
        elif rows:
            differences = ", ".join(f"{row['variable']} ({row['status']})"
                                    for row in rows if row["status"] != "EXACT")
            history_summaries[-1] = (
                f"{summary_label}，已检查 {len(rows)} 个核心变量，"
                f"{len(exact_names)} 个完全一致。其余变量需检查，{differences}。"
            )
        else:
            history_summaries[-1] = f"{summary_label}，两端均未输出指定的核心变量，无法比较。"
        with history_report.open("a", encoding="utf-8") as stream:
            for row in rows:
                stream.write(f"{row['variable']:<8} {row['status']}\n")
                stream.write("  " + "  ".join(f"{key}={value:.8e}" if isinstance(value, float)
                             else f"{key}={value}" for key, value in row.items()
                             if key not in ("variable", "status")) + "\n")
with history_report.open("a", encoding="utf-8") as stream:
    stream.write("\n汇总结论\n")
    stream.write("\n".join(history_summaries) + "\n" if history_summaries else "没有 history 文件可供比较。\n")
if restart_results:
    restart_report = output / "restart.log"
    with restart_report.open("x", encoding="utf-8") as stream:
        stream.write(f"最终 restart 对比 | {filename} | 第 120 步\n")
        stream.write("核心变量优先，三组对比汇总于本文件。\n")
        stream.write("最大绝对差和 RMSE 使用变量原始单位，relative_l2_pct 为相对 L2 误差百分比。\n")
        stream.write("统计使用共同未掩膜的有限值，不做面积加权。未设置数值验收阈值。\n")
        for title, primary in (("核心状态变量", True), ("次要变量", False)):
            stream.write(f"\n{title}\n")
            for left_label, right_label, rows in restart_results:
                selected = [row for row in rows if (row["variable"] in primary_variables) == primary]
                exact_count = sum(row["status"] == "EXACT" for row in selected)
                difference_count = sum(row["status"] == "DIFFERENCES" for row in selected)
                other_count = len(selected) - exact_count - difference_count
                stream.write(f"\n{left_label}–{right_label}\n")
                stream.write(f"结论，共检查 {len(selected)} 个变量，{exact_count} 个完全一致，"
                             f"{difference_count} 个存在差异，{other_count} 个无法完成数值比较。\n")
                if primary:
                    absent = [name for name in primary_variables if name not in {row["variable"] for row in rows}]
                    stream.write("两端均未输出的核心变量名，" + (", ".join(absent) or "无") + "。\n")
                stream.write(f"{'variable':<14} {'status':<30} {'max_abs':>16} {'rmse':>16} "
                             f"{'relative_l2_pct':>18} {'mask_mismatch':>14} "
                             f"{'nonfinite_L/R':>17} {'finite_pairs':>14} {'unequal_finite':>16}\n")
                for row in selected:
                    metrics = []
                    for key in ("max_abs", "rmse", "relative_l2_to_left"):
                        value = row.get(key, "N/A")
                        if isinstance(value, float):
                            if key == "relative_l2_to_left":
                                value *= 100
                            value = f"{value:.8e}"
                        metrics.append(value)
                    nonfinite = f"{row.get('nonfinite_left', 'N/A')}/{row.get('nonfinite_right', 'N/A')}"
                    stream.write(f"{row['variable']:<14} {row['status']:<30} "
                                 f"{metrics[0]:>16} {metrics[1]:>16} {metrics[2]:>18} "
                                 f"{row.get('mask_mismatch', 'N/A'):>14} {nonfinite:>17} "
                                 f"{row.get('finite_pairs', 'N/A'):>14} {row.get('unequal_finite', 'N/A'):>16}\n")
    print(f"Final restart comparisons: {restart_report}")
print(f"Core history comparisons: {history_report}")
print("Reports written. No numerical acceptance threshold applied; inspect coverage, masks and nonfinite counts first.")
