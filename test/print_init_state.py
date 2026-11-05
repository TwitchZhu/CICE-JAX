"""Print module-level state after CICE_Initialize.

Writes cases/gx3_test/init_state.txt. Initialization is run again
so the dump does not depend on another process. The diagnostic log is
rewritten to cases/gx3_test/cice.runlog.
"""

import os
import sys
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cicecore.cicedyn.general import ice_init
from cicecore.cicedyn.infrastructure import ice_domain
from cicecore.drivers.standalone.cice.CICE_InitMod import CICE_Initialize
from cicecore.shared import ice_domain_size
from cicecore.shared import ice_fileunits
from cicecore.shared import ice_init_column

MODULES = (
    "cicecore.shared.ice_calendar",
    "cicecore.shared.ice_domain_size",
    "cicecore.cicedyn.infrastructure.ice_domain",
    "cicecore.cicedyn.infrastructure.comm.serial.ice_boundary",
    "cicecore.shared.ice_constants",
    "cicecore.shared.ice_kinds_mod",
    "cicecore.shared.ice_fileunits",
    "cicecore.cicedyn.infrastructure.comm.serial.ice_timers",
    "cicecore.cicedyn.infrastructure.ice_grid",
    "cicecore.cicedyn.infrastructure.ice_read_write",
    "cicecore.cicedyn.general.ice_state",
    "cicecore.shared.ice_arrays_column",
    "cicecore.shared.ice_init_column",
    "cicecore.shared.ice_restart_column",
    "cicecore.cicedyn.general.ice_flux",
    "cicecore.cicedyn.general.ice_flux_bgc",
    "cicecore.cicedyn.general.ice_forcing",
    "cicecore.cicedyn.general.ice_forcing_bgc",
    "cicecore.cicedyn.infrastructure.ice_restoring",
    "cicecore.cicedyn.dynamics.ice_dyn_shared",
    "cicecore.cicedyn.dynamics.ice_dyn_evp",
    "cicecore.cicedyn.dynamics.ice_transport_driver",
    "cicecore.cicedyn.dynamics.ice_transport_remap",
    "cicecore.cicedyn.dynamics.ice_dyn_eap",
    "cicecore.cicedyn.dynamics.ice_dyn_vp",
    "icepack.columnphysics.icepack_parameters",
    "icepack.columnphysics.icepack_tracers",
    "icepack.columnphysics.icepack_shortwave_data",
    "icepack.columnphysics.icepack_orbital",
    "icepack.columnphysics.icepack_shortwave",
    "icepack.columnphysics.icepack_itd",
    "icepack.columnphysics.icepack_therm_shared",
    "icepack.columnphysics.icepack_mushy_physics",
    "icepack.columnphysics.icepack_fsd",
    "icepack.columnphysics.icepack_snow",
    "icepack.columnphysics.icepack_kinds",
    "cicecore.cicedyn.general.ice_init",
    "cicecore.cicedyn.analysis.ice_history_shared",
    "cicecore.cicedyn.analysis.ice_history",
    "cicecore.cicedyn.analysis.ice_history_mechred",
    "cicecore.cicedyn.analysis.ice_history_pond",
    "cicecore.cicedyn.analysis.ice_history_snow",
    "cicecore.cicedyn.analysis.ice_history_fsd",
    "cicecore.cicedyn.analysis.ice_history_drag",
    "cicecore.cicedyn.analysis.ice_history_bgc",
    "cicecore.cicedyn.analysis.ice_diagnostics",
    "cicecore.cicedyn.infrastructure.io.io_netcdf.ice_history_write",
    "cicecore.cicedyn.infrastructure.io.io_netcdf.ice_restart",
    "cicecore.cicedyn.infrastructure.ice_restart_driver",
)

NOT_PORTED = (
    "ice_blocks",
    "ice_restart_shared",
)


def _range_text(array):
    data = np.asarray(array)
    if data.size == 0:
        return "empty"
    if data.dtype == np.bool_ or np.issubdtype(data.dtype, np.bool_):
        return f"true={int(np.count_nonzero(data))} size={data.size}"
    if not np.issubdtype(data.dtype, np.number):
        return f"dtype={data.dtype}"
    finite = np.isfinite(data)
    if not np.any(finite):
        return "no finite values"
    kept = data[finite]
    return (
        f"min={np.min(kept)!r} max={np.max(kept)!r} "
        f"sum={np.sum(kept)!r} nonzero={int(np.count_nonzero(kept))}"
    )


def _write_value(out, name, value, nx, ny):
    if value is None:
        out.write(f"  {name}  None\n")
        return
    if isinstance(value, (bool, int, float, str)):
        out.write(f"  {name}  {type(value).__name__}  {value!r}\n")
        return
    if isinstance(value, np.generic):
        out.write(f"  {name}  {type(value).__name__}  {value.item()!r}\n")
        return
    if isinstance(value, (list, tuple)):
        out.write(f"  {name}  {type(value).__name__}  len={len(value)}  {value!r}\n")
        return
    if isinstance(value, np.ndarray):
        out.write(
            f"  {name}  ndarray {value.dtype} shape={value.shape}  {_range_text(value)}\n"
        )
        if (
            value.ndim >= 2
            and value.shape[0] == nx + 2
            and value.shape[1] == ny + 2
        ):
            interior = value[1:-1, 1:-1]
            out.write(f"    physical  {_range_text(interior)}\n")
        if value.size <= 12 and np.issubdtype(value.dtype, np.number):
            out.write(f"    values {value.tolist()!r}\n")
        return
    out.write(f"  {name}  {type(value).__name__}\n")


def _write_module(out, module_name, nx, ny):
    module = sys.modules[module_name]
    names = []
    for name, value in vars(module).items():
        if name.startswith("_"):
            continue
        if isinstance(value, (types.ModuleType, types.FunctionType, type)):
            continue
        if callable(value):
            continue
        names.append(name)
    out.write(f"\nmodule {module_name}  count={len(names)}\n")
    for name in names:
        _write_value(out, name, getattr(module, name), nx, ny)


os.chdir(ROOT / "cases" / "gx3_test")
Path("history").mkdir(exist_ok=True)
Path("restart").mkdir(exist_ok=True)

class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


log = open("cice.runlog", "w", encoding="utf-8")
console = sys.stdout
sys.stdout = _Tee(console, log)
ice_fileunits.nu_diag = sys.stdout
ice_init.input_data.__defaults__ = ("ice_in",)
ice_init_column.input_zbgc.__defaults__ = ("ice_in",)
ice_domain.init_domain_blocks.__defaults__ = ("ice_in",)

try:
    CICE_Initialize()
    nx = int(ice_domain_size.nx_global)
    ny = int(ice_domain_size.ny_global)
    with open("init_state.txt", "w", encoding="utf-8") as out:
        out.write("state after CICE_Initialize returns\n")
        out.write(f"nx_global={nx} ny_global={ny}\n")
        out.write("physical region is [1:-1, 1:-1] when the array is (nx+2, ny+2, ...)\n")
        for name in NOT_PORTED:
            out.write(f"\nmodule {name}  not ported\n")
        for module_name in MODULES:
            __import__(module_name)
            _write_module(out, module_name, nx, ny)
finally:
    log.flush()
    log.close()
    sys.stdout = console
