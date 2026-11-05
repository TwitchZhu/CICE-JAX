"""I/O unit manager for tracking, assigning and reserving I/O unit numbers.

author for JAX version: Xinyuan Zhu
XZ: converted from F90 to Python

2007: ECH added dynamic file units, modified from POP_IOUnitsMod.F90
2006: ECH converted to free source form (F90)
author: Elizabeth C. Hunke, LANL

The Fortran unit-number table is not ported. ``nu_diag`` is the
diagnostic stream shared by the model.
"""

import sys

# diagnostics output file, unit number may be overwritten
# Python holds the stream. The Fortran unit number was ice_stdout (6).
nu_diag = sys.stdout
