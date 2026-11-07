"""Create a standalone, single-GPU CICE-JAX case from a bundled preset."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--case", required=True, type=Path,
                        help="New case directory, relative to the current directory.")
    parser.add_argument("--preset", choices=("gx3_5day",), default="gx3_5day",
                        help="Bundled experiment (default: gx3_5day).")
    parser.add_argument("--inputdata", required=True, type=Path,
                        help="Input root containing CICE_data/, relative to the current directory.")
    args = parser.parse_args()

    case = args.case.expanduser().absolute()
    if case.exists() or case.is_symlink():
        parser.error(f"case already exists: {case}; choose a new name")
    case = case.resolve()
    inputdata = args.inputdata.expanduser().resolve()
    for name in ("CICE_data/ic/gx3/iced_gx3_v5.nc",
                 "CICE_data/grid/gx3/grid_gx3.bin",
                 "CICE_data/grid/gx3/kmt_gx3.bin"):
        if not (inputdata / name).is_file():
            parser.error(f"missing input file: {inputdata / name}")
    if not (inputdata / "CICE_data/forcing/gx3").is_dir():
        parser.error(f"missing forcing directory: {inputdata / 'CICE_data/forcing/gx3'}")
    try:
        relative_input = Path(os.path.relpath(inputdata, case)).as_posix()
        relative_source = Path(os.path.relpath(ROOT, case)).as_posix()
    except ValueError as error:
        parser.error(f"case, source and input data must support relative paths: {error}")
    if any(char in relative_input for char in "'\r\n"):
        parser.error("input path cannot contain a single quote or newline in ice_in")

    preset = ROOT / "configuration" / "cases" / args.preset
    reference = preset / "reference"
    ice_in = (reference / "ice_in").read_text(encoding="utf-8")
    mods = "\n".join((reference / f"set_nml.{name}").read_text(encoding="utf-8")
                     for name in ("gx3", "diag1", "run5day"))
    version = (reference / "version.txt").read_text(encoding="utf-8").strip()
    # Keep the official physics; use one full-grid JAX process.
    mods += ("\nnprocs = 1\nnx_global = 100\nny_global = 116\n"
             "block_size_x = 100\nblock_size_y = 116\nmax_blocks = -1\n"
             "distribution_type = 'roundrobin'\nprocessor_shape = 'slenderX2'\n"
             f"version_name = '{version.replace(' ', '_')}'\n")
    for line in mods.splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "!")):
            continue
        name, value = line.split("=", 1)
        pattern = re.compile(r"(?m)^([ \t]*" + re.escape(name.strip()) + r"[ \t]*=[ \t]*).*$")
        ice_in, count = pattern.subn(lambda match: match.group(1) + value.strip(), ice_in)
        if count != 1:
            parser.error(f"expected one namelist entry for {name.strip()}, found {count}")
    ice_in = ice_in.replace("ICE_MACHINE_INPUTDATA", relative_input)
    if "ICE_MACHINE_INPUTDATA" in ice_in or "ICE_SANDBOX" in ice_in:
        parser.error("unresolved input/source placeholder in preset")

    metadata = {
        "preset": args.preset,
        "configuration": "quick_suite / smoke gx3 8x2 diag1,run5day; JAX full grid",
        "source": relative_source,
        "inputdata": relative_input,
        "cice_version": version,
        "initial_namelist_sha256": hashlib.sha256(ice_in.encode("utf-8")).hexdigest(),
        "reference_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(reference.iterdir()) if path.is_file()
        },
    }
    case.mkdir(parents=True)
    (case / "ice_in").write_text(ice_in, encoding="utf-8", newline="\n")
    (case / "case.json").write_text(json.dumps(metadata, indent=2) + "\n",
                                    encoding="utf-8", newline="\n")
    shutil.copyfile(ROOT / "configuration/scripts/run_case.py", case / "run.py")
    shutil.copyfile(preset / "compare_all.py", case / "compare_all.py")
    for name in ("history", "restart"):
        (case / name).mkdir()
    print(f"Created CICE-JAX case: {case}")
    print(f"Preset: {args.preset}; 100x116, dt=3600 s, 5 days / 120 steps")
    print(f"Source relative to case: {relative_source}")
    print(f"Input relative to case: {relative_input}")
    print("Run the generated run.py with your CUDA-enabled Python environment.")
    print("Instructions: docs/SERVER_RUN.md in the source repository")


if __name__ == "__main__":
    main()
