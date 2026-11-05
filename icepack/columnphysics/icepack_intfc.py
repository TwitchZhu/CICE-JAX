"""Public parameters and interface routines for the icepack column package.

author for JAX version: Xinyuan Zhu
XZ: converted from F90 to Python

authors: Elizabeth C. Hunke, LANL

Host models import this module instead of the column-physics modules.
Most names are re-exports. icepack_configure is defined here.
Names from modules that are not ported yet are omitted, as is r16_kind.
"""

from icepack.columnphysics.icepack_kinds import char_len as icepack_char_len
from icepack.columnphysics.icepack_kinds import char_len_long as icepack_char_len_long
from icepack.columnphysics.icepack_kinds import dbl_kind as icepack_dbl_kind
from icepack.columnphysics.icepack_kinds import int8_kind as icepack_int8_kind
from icepack.columnphysics.icepack_kinds import int_kind as icepack_int_kind
from icepack.columnphysics.icepack_kinds import log_kind as icepack_log_kind
from icepack.columnphysics.icepack_kinds import real_kind as icepack_real_kind

from icepack.columnphysics.icepack_parameters import icepack_init_parameters
from icepack.columnphysics.icepack_parameters import icepack_query_parameters
from icepack.columnphysics.icepack_parameters import icepack_recompute_constants
from icepack.columnphysics.icepack_parameters import icepack_write_parameters
from icepack.columnphysics.icepack_parameters import secday, spval_const
from icepack.columnphysics.icepack_parameters import c0, c1, c1p5, c2, c3, c4, c5, c6, c8
from icepack.columnphysics.icepack_parameters import c10, c15, c16, c20, c25, c100, c1000
from icepack.columnphysics.icepack_parameters import p001, p01, p1, p2, p4, p5, p6, p05
from icepack.columnphysics.icepack_parameters import p15, p25, p75, p333, p666

from icepack.columnphysics.icepack_tracers import icepack_init_tracer_flags
from icepack.columnphysics.icepack_tracers import icepack_init_tracer_indices
from icepack.columnphysics.icepack_tracers import icepack_init_tracer_sizes
from icepack.columnphysics.icepack_tracers import icepack_query_tracer_flags

from icepack.columnphysics.icepack_therm_shared import icepack_init_enthalpy
from icepack.columnphysics.icepack_therm_shared import icepack_init_salinity
from icepack.columnphysics.icepack_therm_shared import icepack_init_thermo
from icepack.columnphysics.icepack_therm_shared import icepack_init_trcr
from icepack.columnphysics.icepack_therm_shared import icepack_liquidus_temperature
from icepack.columnphysics.icepack_therm_shared import icepack_salinity_profile
from icepack.columnphysics.icepack_therm_shared import icepack_sea_freezing_temperature

from icepack.columnphysics.icepack_itd import icepack_aggregate
from icepack.columnphysics.icepack_itd import icepack_init_itd
from icepack.columnphysics.icepack_itd import icepack_init_itd_hist


def icepack_configure():
    """Call before any other part of the column physics is used."""
    icepack_recompute_constants()
