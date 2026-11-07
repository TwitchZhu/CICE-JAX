"""Check standalone case creation without JAX or real forcing datasets."""

import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class CaseSetupTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="cice_case_")
        self.addCleanup(self.temporary.cleanup)
        self.work = Path(self.temporary.name)
        self.inputdata = self.work / "input data"
        for name in ("ic/gx3/iced_gx3_v5.nc", "grid/gx3/grid_gx3.bin",
                     "grid/gx3/kmt_gx3.bin"):
            path = self.inputdata / "CICE_data" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        (self.inputdata / "CICE_data/forcing/gx3").mkdir(parents=True)

    def create(self, *extra, entry="cice.setup"):
        return subprocess.run(
            [sys.executable, "-S", str(ROOT / entry), "--case", "runs/new case",
             "--inputdata", "input data", *extra],
            cwd=self.work, text=True, capture_output=True)

    def test_external_case_paths_and_five_day_configuration(self):
        result = self.create()
        self.assertEqual(result.returncode, 0, result.stderr)
        case = self.work / "runs/new case"
        metadata = json.loads((case / "case.json").read_text())
        self.assertEqual((case / metadata["source"]).resolve(), ROOT)
        self.assertEqual((case / metadata["inputdata"]).resolve(), self.inputdata)
        self.assertFalse(Path(metadata["source"]).is_absolute())
        self.assertFalse(Path(metadata["inputdata"]).is_absolute())
        nml = (case / "ice_in").read_text()
        entries = dict(re.findall(r"(?m)^\s*(\w+)\s*=\s*(.*?)\s*$", nml))
        expected = {
            "nprocs": "1", "nx_global": "100", "ny_global": "116",
            "block_size_x": "100", "block_size_y": "116",
            "year_init": "2005", "dt": "3600.0", "npt_unit": "'d'", "npt": "5",
            "diagfreq": "1", "dumpfreq": "'d'", "dumpfreq_n": "5",
            "ktherm": "2", "ndte": "240", "advection": "'remap'",
        }
        for key, value in expected.items():
            self.assertEqual(entries[key], value, key)
        self.assertNotIn("ICE_MACHINE_INPUTDATA", nml)
        self.assertEqual((case / entries["ice_ic"].strip("'")).resolve(),
                         self.inputdata / "CICE_data/ic/gx3/iced_gx3_v5.nc")
        for name in ("run.py", "compare_all.py"):
            compile((case / name).read_text(encoding="utf-8"), name, "exec")
        # Help must work before importing GPU packages or starting a simulation.
        help_result = subprocess.run(
            [sys.executable, "-S", str(case / "run.py"), "--help"],
            cwd=self.work, text=True, capture_output=True)
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        self.assertIn("--profile", help_result.stdout)
        self.assertFalse(list(case.glob("cice.runlog.*")))

    def test_existing_case_is_not_modified(self):
        self.assertEqual(self.create().returncode, 0)
        case = self.work / "runs/new case"
        (case / "ice_in").write_text("user configuration")
        result = self.create()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("case already exists", result.stderr)
        self.assertEqual((case / "ice_in").read_text(), "user configuration")

    def test_missing_input_leaves_no_case(self):
        (self.inputdata / "CICE_data/grid/gx3/kmt_gx3.bin").unlink()
        result = self.create()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("missing input file", result.stderr)
        self.assertFalse((self.work / "runs").exists())

    def test_python_entry_and_unknown_preset(self):
        result = self.create("--preset", "not_a_preset")
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.work / "runs").exists())
        result = self.create(entry="cice_setup.py")
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
