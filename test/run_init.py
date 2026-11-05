"""Run CICE_Initialize in cases/gx3_test and keep the diagnostic log."""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cicecore.cicedyn.general import ice_init
from cicecore.cicedyn.infrastructure import ice_domain
from cicecore.drivers.standalone.cice.CICE_InitMod import CICE_Initialize
from cicecore.shared import ice_fileunits
from cicecore.shared import ice_init_column

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
# input_data sets nu_diag back to sys.stdout when diag_type is stdout.
ice_fileunits.nu_diag = sys.stdout
# cice_init calls these three with no path. Keep the case ice_in.
ice_init.input_data.__defaults__ = ("ice_in",)
ice_init_column.input_zbgc.__defaults__ = ("ice_in",)
ice_domain.init_domain_blocks.__defaults__ = ("ice_in",)

try:
    CICE_Initialize()
finally:
    log.flush()
    log.close()
    sys.stdout = console
