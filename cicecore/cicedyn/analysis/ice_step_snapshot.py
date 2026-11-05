"""Lossless, read-only snapshots at two consecutive debug step entrances."""

import json
from datetime import datetime
from pathlib import Path

import numpy as np

from cicecore.shared import ice_constants, ice_fileunits
from icepack.columnphysics import icepack_parameters, icepack_therm_shared


def write_step_entry(pytree, static):
    init = static["ice_init"]
    step = int(static["ice_calendar"]["istep1"])
    first_step = max(1, int(init["debug_model_step"]))
    if not bool(init["debug_model"]) or not first_step <= step <= first_step + 1:
        return

    # Snapshot device leaves, not their possibly stale initialization mirrors.
    values = {}
    for group, modules in pytree.items():
        for module, fields in modules.items():
            for name, value in fields.items():
                values[f"{group}.{module}.{name}"] = value
    for module, fields in static.items():
        for name, value in fields.items():
            values[f"static.{module}.{name}"] = value
    # Some translated routines still import these scalars directly.
    for module in (ice_constants, icepack_parameters, icepack_therm_shared):
        for name, value in vars(module).items():
            if not name.startswith("_") and isinstance(
                    value, (bool, int, float, str, np.generic)):
                values[f"host.{module.__name__.rsplit('.', 1)[-1]}.{name}"] = value

    arrays = {}
    fields = {}
    for name, value in values.items():
        try:
            array = np.asarray(value)
            if array.dtype.kind not in "biufSU":
                raise TypeError(f"unsupported dtype {array.dtype}")
        except (TypeError, ValueError) as error:
            fields[name] = {"status": "unsupported", "reason": str(error)}
            continue
        arrays[name] = array
        fields[name] = {"status": "saved", "shape": list(array.shape),
                        "dtype": str(array.dtype)}

    output = Path("comparison_logs")
    output.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    stem = output / f"step_entry_jax.{step:08d}.{stamp}"
    # Append suffixes: with_suffix would remove the timestamp.
    data_path = Path(str(stem) + ".npz")
    metadata_path = Path(str(stem) + ".json")
    with data_path.open("xb") as stream:
        np.savez(stream, **arrays)
    metadata = {
        "format": "CICE_STEP_ENTRY_JAX_V1", "step": step,
        "checkpoint": "ice_step entrance, before restoring and diagnostic resets",
        "physical_bounds": list(static["ice_domain_size"]["physical_bounds"]),
        "fields": fields,
    }
    with metadata_path.open("x", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)
        stream.write("\n")
    ice_fileunits.nu_diag.write(f"(step_entry) complete: {metadata_path} {data_path}\n")
    ice_fileunits.nu_diag.flush()
