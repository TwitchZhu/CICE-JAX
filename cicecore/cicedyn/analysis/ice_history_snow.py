"""Snow tracer history output.

author for JAX version: Xinyuan Zhu
XZ: converted from F90 to Python
"""


import jax.numpy as jnp

from cicecore.cicedyn.analysis import ice_history_shared
from cicecore.cicedyn.general import ice_flux
from cicecore.cicedyn.general import ice_state
from cicecore.shared import ice_arrays_column
from cicecore.shared.ice_constants import c0
from cicecore.shared.ice_kinds_mod import dbl_kind
from icepack.columnphysics import icepack_tracers
from icepack.columnphysics.icepack_tracers import icepack_query_tracer_flags

# Defaults match ice_history_snow.F90. Nothing is registered unless tr_snow.
_F = {
    "smassice": "m",
    "smassicen": "x",
    "smassliq": "m",
    "smassliqn": "x",
    "rhos_cmp": "m",
    "rhos_cmpn": "x",
    "rhos_cnt": "m",
    "rhos_cntn": "x",
    "rsnw": "m",
    "rsnwn": "x",
    "meltsliq": "m",
    "fsloss": "x",
    "sisndmasswind": "m",
}


def _add(name, unit, desc, comment, gather, rank="2D"):
    ice_history_shared.define_hist_field(
        name, unit, desc, comment, c0 + 1, c0, _F[name], gather, rank=rank,
    )


def _layers(nt, count):
    """Sum snow-layer tracers. nt is 0-based; a closed tracer is -1."""
    work = jnp.zeros(ice_state.aice.shape, dtype=dbl_kind)
    if int(nt) < 0 or count < 1:
        return work
    for k in range(count):
        work = work + ice_state.trcr[:, :, int(nt) + k]
    return work / dbl_kind(count)


def _layers_n(nt, count):
    ncat = ice_history_shared.ncat_hist
    work = jnp.zeros(ice_state.aicen[:, :, :ncat].shape, dtype=dbl_kind)
    if int(nt) < 0 or count < 1:
        return work
    for k in range(count):
        work = work + ice_state.trcrn[:, :, int(nt) + k, :ncat]
    return work / dbl_kind(count)


def init_hist_snow_2D(dt):
    """Register 2D snow-physics history fields."""
    del dt
    if icepack_query_tracer_flags("tr_snow")["tr_snow"]:
        ice_history_shared.note_nml("(init_hist_snow_2D)", "icefields_snow_nml")
    ice_history_shared.apply_history_flags(_F)
    if not icepack_query_tracer_flags("tr_snow")["tr_snow"]:
        return
    nz = ice_history_shared.nzslyr

    def mass(index_name):
        def gather():
            # ice or liquid mass per unit area, scaled by snow volume
            work = _layers(getattr(icepack_tracers, index_name), nz)
            return work * ice_state.vsno
        return gather

    def mean(index_name):
        def gather():
            return _layers(getattr(icepack_tracers, index_name), nz)
        return gather

    def content():
        ice = _layers(icepack_tracers.nt_smice, nz)
        liquid = _layers(icepack_tracers.nt_smliq, nz)
        return ice + liquid

    _add("smassice", "kg/m^2", "ice mass per unit area in snow", "none", mass("nt_smice"))
    _add("smassliq", "kg/m^2", "liquid mass per unit area in snow", "none", mass("nt_smliq"))
    _add("rhos_cmp", "kg/m^3", "snow density: compaction", "none", mean("nt_rhos"))
    _add("rhos_cnt", "kg/m^3", "snow density: content", "none", content)
    _add("rsnw", "10^-6 m", "average snow grain radius", "none", mean("nt_rsnw"))
    _add("meltsliq", "kg/m^2/s", "snow liquid mass lost to melt", "none", lambda: ice_arrays_column.meltsliq)
    _add("fsloss", "kg/m^2/s", "snow mass lost to leads", "none", lambda: ice_flux.fsloss)
    _add("sisndmasswind", "kg/m^2/s", "snow mass change through wind", "none", lambda: ice_flux.fsloss)


def init_hist_snow_3Dc():
    """Register category snow-physics history fields."""
    if not icepack_query_tracer_flags("tr_snow")["tr_snow"]:
        return
    nz = ice_history_shared.nzslyr
    ncat = ice_history_shared.ncat_hist

    def mass_n(index_name):
        def gather():
            work = _layers_n(getattr(icepack_tracers, index_name), nz)
            return work * ice_state.vsnon[:, :, :ncat]
        return gather

    def mean_n(index_name):
        def gather():
            return _layers_n(getattr(icepack_tracers, index_name), nz)
        return gather

    def content_n():
        return _layers_n(icepack_tracers.nt_smice, nz) + _layers_n(icepack_tracers.nt_smliq, nz)

    _add("smassicen", "kg/m^2", "ice mass in snow, categories", "none", mass_n("nt_smice"), rank="3Dc")
    _add("smassliqn", "kg/m^2", "liquid mass in snow, categories", "none", mass_n("nt_smliq"), rank="3Dc")
    _add("rhos_cmpn", "kg/m^3", "snow density: compaction, categories", "none", mean_n("nt_rhos"), rank="3Dc")
    _add("rhos_cntn", "kg/m^3", "snow density: content, categories", "none", content_n, rank="3Dc")
    _add("rsnwn", "10^-6 m", "snow grain radius, categories", "none", mean_n("nt_rsnw"), rank="3Dc")
