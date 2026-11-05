"""Windows port of the CICE ``cice.setup --case`` command.

The Linux script is ``../CICE/cice.setup`` (csh). This file runs the case
path used by

    python cice_setup.py -c CASE -g gx3

Templates and ``set_nml.*`` are read from ``../CICE/configuration/scripts``.
Nothing under ``../CICE`` is written. ``-m`` and ``-e`` are ignored: the
Python run does not load a machine env or ``Macros`` file.

``--test`` and ``--suite`` are not ported. Those paths compile Fortran and
submit ``yhbatch`` jobs. ``ICE_MACHINE_INPUTDATA`` is the local
``../CICE/input``, written into ``ice_in`` as an absolute path.

Block sizes follow ``configuration/scripts/cice_decomp.csh`` for one task.
``-p`` is ignored. ``max_blocks`` stays -1 because that script does not
export the count it computes.
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CICE = ROOT.parent / "CICE"
SCRIPTS = CICE / "configuration" / "scripts"


def die(message):
    print(f"cice.setup: ERROR, {message}", file=sys.stderr)
    raise SystemExit(1)


case = None
grid = "gx3"
pes = "1"
sets = ""
ignore_user_set = False
index = 0
argv = sys.argv[1:]
while index < len(argv):
    option = argv[index]
    if option in ("-h", "--help"):
        print(
            "cice_setup.py --case CASE [-g GRID] [-s SETS]\n"
            "Windows port of cice.setup --case. -m/-e/-p are ignored. --test and --suite are not ported."
        )
        raise SystemExit(0)
    if option == "--ignore-user-set":
        ignore_user_set = True
        index += 1
        continue
    if index + 1 >= len(argv) or argv[index + 1].startswith("-"):
        die(f"{option} is missing an argument")
    value = argv[index + 1]
    if option in ("--case", "-c"):
        case = value
    elif option in ("--mach", "-m", "--env", "-e", "--pes", "-p") or option.startswith("--mach") or option.startswith("--env"):
        pass
    elif option in ("--grid", "-g"):
        grid = value
    elif option in ("--set", "-s") or option.startswith("--set"):
        sets = value
    elif option in ("--test", "--suite"):
        die(f"{option} is not ported; this Windows setup only creates a case")
    else:
        die(f"unknown option {option}")
    index += 2
if case is None:
    die("--case is required")

if not ignore_user_set:
    user_set = Path.home() / ".cice_set"
    if user_set.is_file():
        extra = user_set.read_text(encoding="utf-8").replace(" ", "").strip()
        if extra:
            sets = extra if not sets else extra + "," + sets
            print(f"adding sets from {user_set}: {extra}")

parts = pes.split("x")
if not parts or any(not part.isdigit() for part in parts):
    die(
        f"-p {pes} must be [m], [m]x[n], [m]x[n]x[bx]x[by], "
        "or [m]x[n]x[bx]x[by]x[mb]"
    )
numbers = [int(part) for part in parts]
if any(number == 0 for number in numbers):
    die("-p cannot contain zeros")
if len(numbers) == 5:
    task, thrd, blckx, blcky, max_blocks = numbers
elif len(numbers) == 4:
    task, thrd, blckx, blcky = numbers
    max_blocks = -1
elif len(numbers) == 2:
    task, thrd = numbers
    blckx = 0
    blcky = 0
    max_blocks = -1
elif len(numbers) == 1:
    task = numbers[0]
    thrd = 1
    blckx = 0
    blcky = 0
    max_blocks = -1
else:
    die(
        f"-p {pes} must be [m], [m]x[n], [m]x[n]x[bx]x[by], "
        "or [m]x[n]x[bx]x[by]x[mb]"
    )

# cice_decomp.csh: default block size from the grid and the PE count.
cicepes = task * thrd
if grid == "col":
    nx, ny = 5, 5
    default_x, default_y = (5, 5) if cicepes <= 1 else (1, 1)
elif grid == "gbox128":
    nx, ny = 128, 128
    if cicepes <= 1:
        default_x, default_y = 128, 128
    elif cicepes <= 8:
        default_x, default_y = 32, 32
    elif cicepes <= 32:
        default_x, default_y = 16, 16
    else:
        default_x, default_y = 8, 8
elif grid == "gbox180":
    nx, ny = 180, 180
    if cicepes <= 1:
        default_x, default_y = 180, 180
    elif cicepes <= 36:
        default_x, default_y = 30, 30
    else:
        default_x, default_y = 9, 9
elif grid == "gbox80":
    nx, ny = 80, 80
    if cicepes <= 1:
        default_x, default_y = 80, 80
    elif cicepes <= 8:
        default_x, default_y = 20, 20
    else:
        default_x, default_y = 8, 8
elif grid == "gbox12":
    nx, ny = 12, 12
    if cicepes <= 1:
        default_x, default_y = 12, 12
    elif cicepes <= 8:
        default_x, default_y = 4, 4
    else:
        default_x, default_y = 2, 2
elif grid == "gx3":
    nx, ny = 100, 116
    if cicepes <= 1:
        default_x, default_y = 100, 116
    elif cicepes <= 8:
        default_x, default_y = 25, 29
    elif cicepes <= 32:
        default_x, default_y = 5, 29
    else:
        default_x, default_y = 5, 4
elif grid == "gx1":
    nx, ny = 320, 384
    if cicepes <= 16:
        default_x, default_y = 40, 48
    elif cicepes <= 20:
        default_x, default_y = 32, 48
    elif cicepes <= 32:
        default_x, default_y = 20, 24
    elif cicepes <= 40:
        default_x, default_y = 16, 24
    elif cicepes < 80:
        default_x, default_y = 10, 16
    elif cicepes == 80:
        default_x, default_y = 8, 16
    else:
        default_x, default_y = 10, 12
elif grid == "tx1":
    nx, ny = 360, 240
    if cicepes <= 16:
        default_x, default_y = 90, 60
    elif cicepes <= 64:
        default_x, default_y = 20, 20
    else:
        default_x, default_y = 10, 10
elif grid == "none":
    nx, ny = 1, 1
    default_x, default_y = 1, 1
else:
    die(f"unknown grid {grid}")

# User block size overrides the default. One side alone is illegal.
if blckx > 0 and blcky > 0:
    default_x, default_y = blckx, blcky
elif blckx < 1 and blcky < 1:
    pass
else:
    die("user defined blocksize illegal")
blckx, blcky = default_x, default_y

# Odd PE count, a block that does not divide the grid, or a shape that
# is not two blocks wide per task, uses roundrobin instead of cartesian.
decomp = "cartesian"
if cicepes % 2 != 0:
    decomp = "roundrobin"
if ny % (blcky * 2) != 0:
    decomp = "roundrobin"
if nx % blckx != 0:
    decomp = "roundrobin"
if (nx * 2) % (cicepes * blckx) != 0:
    decomp = "roundrobin"
shape = "slenderX2"

version_path = CICE / "cicecore" / "version.txt"
if version_path.is_file():
    version = version_path.read_text(encoding="utf-8").splitlines()[0].replace(" ", "_")
else:
    version = "unknown"

case_path = Path(case)
if not case_path.is_absolute():
    case_path = Path.cwd() / case_path
if case_path.exists():
    die(f"case {case_path} already exists")
input_root = (CICE / "input").resolve()
sandbox = CICE.resolve()

template = SCRIPTS / "ice_in"
if not template.is_file():
    die(f"{template} not found")

mods = (
    "nprocs            = {task}\n"
    "nx_global         = {nx}\n"
    "ny_global         = {ny}\n"
    "block_size_x      = {blckx}\n"
    "block_size_y      = {blcky}\n"
    "max_blocks        = {max_blocks}\n"
    "distribution_type = '{decomp}'\n"
    "processor_shape   = '{shape}'\n"
    "version_name      = '{version}'\n"
).format(
    task=task,
    nx=nx,
    ny=ny,
    blckx=blckx,
    blcky=blcky,
    max_blocks=max_blocks,
    decomp=decomp,
    shape=shape,
    version=version,
)

set_names = [grid]
for name in sets.split(","):
    name = name.strip()
    if not name or name == "none":
        continue
    bundle = SCRIPTS / "options" / f"set_files.{name}"
    if bundle.is_file():
        print(f"adding options files from set_files.{name}")
        for line in bundle.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                set_names.append(line)
    else:
        set_names.append(name)

for name in set_names:
    nml = SCRIPTS / "options" / f"set_nml.{name}"
    envfile = SCRIPTS / "options" / f"set_env.{name}"
    if not nml.is_file() and not envfile.is_file():
        die(f"options/set_nml.{name} or set_env.{name} not found")
    if nml.is_file():
        print(f"adding namelist mods set_nml.{name}")
        mods = mods + "\n" + nml.read_text(encoding="utf-8", errors="replace")
    if envfile.is_file():
        print(f"adding env mods set_env.{name} (build settings, not written on Windows)")

# parse_namelist.sh: first token is the name, the next token is the value.
ice_in = template.read_text(encoding="utf-8", errors="replace")
seen = {}
for line in mods.splitlines():
    stripped = line.strip()
    if not stripped or stripped.startswith("#") or stripped.startswith("!"):
        continue
    match = re.match(r"(\S+)\s*=\s*(\S+)", stripped)
    if match is None:
        die(f"cannot parse namelist mod {stripped}")
    name, value = match.group(1), match.group(2)
    if name in seen and seen[name] != value:
        print(f"parse_namelist WARNING: re-overriding {name} from {seen[name]} to {value}")
    seen[name] = value
    pattern = re.compile(r"(?m)^([ \t]*" + re.escape(name) + r"[ \t]*=[ \t]*).*$")
    if pattern.search(ice_in) is None:
        die(f"parsing error for {name}")
    ice_in = pattern.sub(lambda found, new=value: found.group(1) + new, ice_in)

ice_in = ice_in.replace("ICE_MACHINE_INPUTDATA", input_root.as_posix())
ice_in = ice_in.replace("ICE_SANDBOX", sandbox.as_posix())

case_path.mkdir(parents=True)
(case_path / "ice_in").write_text(ice_in, encoding="utf-8", newline="\n")
(case_path / "README.case").write_text(
    " ".join(sys.argv) + "\n"
    + f"ICE_GRID     = {grid} ({nx}x{ny}) blocksize={blckx}x{blcky} max_blocks={max_blocks}\n"
    + f"ICE_DECOMP   = {decomp} {shape}\n"
    + f"ICE_NTASKS   = {task}\n"
    + f"ICE_NTHRDS   = {thrd}\n"
    + f"input        = {relative_input}\n",
    encoding="utf-8",
    newline="\n",
)
print(f"ICE_CASEDIR  = {case_path}")
print(f"ICE_GRID     = {grid} ({nx}x{ny}) blocksize={blckx}x{blcky}")
print(f"ICE_DECOMP   = {decomp} {shape}")
print(f"wrote {case_path / 'ice_in'}")
