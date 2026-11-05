"""Form drag history output.

author for JAX version: Xinyuan Zhu
XZ: converted from F90 to Python

2013 module for form drag parameters
authors Michel Tsamados, David Schroeder, CPOM
"""

from cicecore.cicedyn.analysis import ice_history_shared
from cicecore.shared import ice_arrays_column
from cicecore.shared.ice_constants import c0, c1
from icepack.columnphysics import icepack_parameters

# Defaults match ice_history_drag.F90. The detailed fields follow f_drag.
_F = {
    "Cdn_atm": "x",
    "Cdn_ocn": "x",
    "drag": "x",
}


def _add(name, unit, desc, gather, freq_name):
    ice_history_shared.define_hist_field(
        name, unit, desc, "none", c1, c0, _F[freq_name], gather,
    )


def init_hist_drag_2D():
    """Register form-drag history fields when formdrag is on."""
    ice_history_shared.note_nml("(init_hist_drag_2D)", "icefields_drag_nml")
    ice_history_shared.apply_history_flags(_F)
    if not bool(icepack_parameters.formdrag):
        return

    def column(name):
        return lambda: getattr(ice_arrays_column, name)

    _add("Cdn_atm", "none", "atmospheric drag coefficient", column("Cdn_atm"), "Cdn_atm")
    _add("Cdn_ocn", "none", "ocean drag coefficient", column("Cdn_ocn"), "Cdn_ocn")
    if str(_F["drag"]).strip() == "x":
        return
    _add("hfreebd", "m", "hfreebd: freeboard", column("hfreebd"), "drag")
    _add("hdraft", "m", "hdraft: draught", column("hdraft"), "drag")
    _add("hridge", "m", "hridge: ridge height", column("hridge"), "drag")
    _add("distrdg", "m", "distrdg: distance between ridges", column("distrdg"), "drag")
    _add("hkeel", "m", "hkeel: keel depth", column("hkeel"), "drag")
    _add("dkeel", "m", "dkeel: distance between keels", column("dkeel"), "drag")
    _add("lfloe", "m", "lfloe: floe length", column("lfloe"), "drag")
    _add("dfloe", "m", "dfloe: distance between floes", column("dfloe"), "drag")
    _add("Cdn_atm_rdg", "none", "atmospheric drag from ridges", column("Cdn_atm_rdg"), "drag")
    _add("Cdn_atm_floe", "none", "atmospheric drag from floe edges", column("Cdn_atm_floe"), "drag")
    _add("Cdn_atm_pond", "none", "atmospheric drag from ponds", column("Cdn_atm_pond"), "drag")
    _add("Cdn_atm_skin", "none", "atmospheric skin drag", column("Cdn_atm_skin"), "drag")
    _add("Cdn_atm_ratio", "none", "ratio of form drag to total atmospheric drag", column("Cdn_atm_ratio"), "drag")
    _add("Cdn_ocn_keel", "none", "ocean drag from keels", column("Cdn_ocn_keel"), "drag")
    _add("Cdn_ocn_floe", "none", "ocean drag from floe edges", column("Cdn_ocn_floe"), "drag")
    _add("Cdn_ocn_skin", "none", "ocean skin drag", column("Cdn_ocn_skin"), "drag")
