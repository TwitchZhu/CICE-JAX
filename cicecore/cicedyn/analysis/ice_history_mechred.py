"""Mechanical redistribution history output.

author for JAX version: Xinyuan Zhu
XZ: converted from F90 to Python

2012 Elizabeth Hunke split code from ice_history.F90
"""


import jax.numpy as jnp

from cicecore.cicedyn.analysis import ice_history_shared
from cicecore.cicedyn.general import ice_flux
from cicecore.cicedyn.general import ice_state
from cicecore.shared.ice_constants import c0, c1, c100, mps_to_cmpdy
from cicecore.shared.ice_kinds_mod import dbl_kind
from icepack.columnphysics import icepack_parameters
from icepack.columnphysics import icepack_tracers

# Defaults match ice_history_mechred.F90.
_F = {
    "ardg": "m",
    "vrdg": "m",
    "alvl": "m",
    "vlvl": "m",
    "dardg1dt": "m",
    "dardg2dt": "m",
    "dvirdgdt": "m",
    "opening": "m",
    "ardgn": "x",
    "vrdgn": "x",
    "dardg1ndt": "x",
    "dardg2ndt": "x",
    "dvirdgndt": "x",
    "aparticn": "x",
    "krdgn": "x",
    "aredistn": "x",
    "vredistn": "x",
    "araftn": "x",
    "vraftn": "x",
    "sirdgconc": "m",
    "sirdgthick": "m",
}


def _add(name, unit, desc, comment, cona, conb, gather, rank="2D", avg_ice_present="none", mask_ice_free_points=False):
    ice_history_shared.define_hist_field(
        name, unit, desc, comment, cona, conb, _F[name], gather,
        rank=rank, avg_ice_present=avg_ice_present, mask_ice_free_points=mask_ice_free_points,
    )


def _tracer(index_name):
    nt = int(getattr(icepack_tracers, index_name))
    if nt < 0:
        return jnp.zeros(ice_state.aice.shape, dtype=dbl_kind)
    return ice_state.trcr[:, :, nt]


def _tracer_n(index_name):
    ncat = ice_history_shared.ncat_hist
    nt = int(getattr(icepack_tracers, index_name))
    if nt < 0:
        return jnp.zeros(ice_state.aicen.shape[:2] + (ncat,), dtype=dbl_kind)
    return ice_state.trcrn[:, :, nt, :ncat]


def init_hist_mechred_2D():
    """Register 2D ridging and level-ice history fields."""
    ice_history_shared.note_nml("(init_hist_mechred_2D)", "icefields_mechred_nml")
    ice_history_shared.apply_history_flags(_F)
    secday = icepack_parameters.icepack_query_parameters("secday")["secday"]
    strain = secday * c100

    def level_area():
        return ice_state.aice * _tracer("nt_alvl")

    def level_volume():
        return ice_state.vice * _tracer("nt_vlvl")

    def ridge_area():
        return ice_state.aice * (c1 - _tracer("nt_alvl"))

    def ridge_volume():
        return ice_state.vice * (c1 - _tracer("nt_vlvl"))

    _add("alvl", "1", "level ice area fraction", "none", c1, c0, level_area)
    _add("vlvl", "m", "level ice volume", "grid cell mean level ice thickness", c1, c0, level_volume)
    _add("ardg", "1", "ridged ice area fraction", "none", c1, c0, ridge_area)
    _add("vrdg", "m", "ridged ice volume", "grid cell mean level ridged thickness", c1, c0, ridge_volume)
    _add("dardg1dt", "%/day", "ice area ridging rate", "none", strain, c0, lambda: ice_flux.dardg1dt)
    _add("dardg2dt", "%/day", "ridge area formation rate", "none", strain, c0, lambda: ice_flux.dardg2dt)
    _add("dvirdgdt", "cm/day", "ice volume ridging rate", "none", mps_to_cmpdy, c0, lambda: ice_flux.dvirdgdt)
    _add("opening", "%/day", "lead area opening rate", "none", strain, c0, lambda: ice_flux.opening)
    _add("sirdgconc", "%", "percentage of ridged sea ice", "area percentage of sea ice surface that is ridged sea ice", c100, c0, ridge_area)
    _add(
        "sirdgthick", "m", "ridged ice thickness",
        "total volume of ridged sea ice divided by area of ridges",
        c1, c0, ridge_volume, avg_ice_present="ridge", mask_ice_free_points=True,
    )


def init_hist_mechred_3Dc():
    """Register category ridging history fields."""
    secday = icepack_parameters.icepack_query_parameters("secday")["secday"]
    strain = secday * c100
    ncat = ice_history_shared.ncat_hist

    def cat(name):
        return jnp.asarray(getattr(ice_flux, name))[:, :, :ncat]

    def ridge_area_n():
        return ice_state.aicen[:, :, :ncat] * (c1 - _tracer_n("nt_alvl"))

    def ridge_volume_n():
        return ice_state.vicen[:, :, :ncat] * (c1 - _tracer_n("nt_vlvl"))

    _add("ardgn", "1", "ridged ice area, categories", "none", c1, c0, ridge_area_n, rank="3Dc")
    _add("vrdgn", "m", "ridged ice volume, categories", "none", c1, c0, ridge_volume_n, rank="3Dc")
    _add("dardg1ndt", "%/day", "ice area ridging rate, categories", "none", strain, c0, lambda: cat("dardg1ndt"), rank="3Dc")
    _add("dardg2ndt", "%/day", "ridge area formation rate, categories", "none", strain, c0, lambda: cat("dardg2ndt"), rank="3Dc")
    _add("dvirdgndt", "cm/day", "ice volume ridging rate, categories", "none", mps_to_cmpdy, c0, lambda: cat("dvirdgndt"), rank="3Dc")
    _add("krdgn", "1", "ridging participation function, categories", "none", c1, c0, lambda: cat("krdgn"), rank="3Dc")
    _add("aparticn", "1", "area participation function, categories", "none", c1, c0, lambda: cat("aparticn"), rank="3Dc")
    _add("aredistn", "1", "area redistribution function, categories", "none", c1, c0, lambda: cat("aredistn"), rank="3Dc")
    _add("vredistn", "1", "volume redistribution function, categories", "none", c1, c0, lambda: cat("vredistn"), rank="3Dc")
    _add("araftn", "1", "rafting area participation, categories", "none", c1, c0, lambda: cat("araftn"), rank="3Dc")
    _add("vraftn", "1", "rafting volume participation, categories", "none", c1, c0, lambda: cat("vraftn"), rank="3Dc")
