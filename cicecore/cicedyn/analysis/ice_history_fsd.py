"""Floe size distribution history output.

author for JAX version: Xinyuan Zhu
XZ: converted from F90 to Python

2019 Elizabeth Hunke
"""


import jax.numpy as jnp

from cicecore.cicedyn.analysis import ice_history_shared
from cicecore.cicedyn.general import ice_state
from cicecore.shared import ice_arrays_column
from cicecore.shared.ice_constants import c0, c1, c2, c4, c8
from cicecore.shared.ice_kinds_mod import dbl_kind
from icepack.columnphysics import icepack_parameters
from icepack.columnphysics import icepack_tracers
from icepack.columnphysics.icepack_tracers import icepack_query_tracer_flags

# Defaults match ice_history_fsd.F90. Nothing is registered unless tr_fsd.
_F = {
    "afsd": "m",
    "afsdn": "m",
    "dafsd_newi": "m",
    "dafsd_latg": "m",
    "dafsd_latm": "m",
    "dafsd_wave": "m",
    "dafsd_weld": "m",
    "wave_sig_ht": "m",
    "aice_ww": "x",
    "diam_ww": "x",
    "hice_ww": "x",
    "fsdrad": "x",
    "fsdperim": "x",
}


def _add(name, unit, desc, comment, cona, gather, rank="2D"):
    ice_history_shared.define_hist_field(
        name, unit, desc, comment, cona, c0, _F[name], gather, rank=rank,
    )


def _area_bins():
    """Floe-size area fraction summed over thickness categories. Shape (i, j, nfsd)."""
    nfsd = ice_history_shared.nfsd_hist
    ncat = ice_history_shared.ncat_hist
    nt = int(icepack_tracers.nt_fsd)
    work = jnp.zeros(ice_state.aice.shape + (nfsd,), dtype=dbl_kind)
    if nt < 0:
        return work
    for k in range(nfsd):
        for n in range(ncat):
            work = work.at[:, :, k].add(ice_state.aicen[:, :, n] * ice_state.trcrn[:, :, nt + k, n])
    return work


def init_hist_fsd_2D():
    """Register 2D floe-size history fields."""
    if icepack_query_tracer_flags("tr_fsd")["tr_fsd"]:
        ice_history_shared.note_nml("(init_hist_fsd_2D)", "icefields_fsd_nml")
    ice_history_shared.apply_history_flags(_F)
    if not icepack_query_tracer_flags("tr_fsd")["tr_fsd"]:
        return
    _add("wave_sig_ht", "m", "significant wave height", "none", c1, lambda: ice_arrays_column.wave_sig_ht)
    _add("aice_ww", "1", "ice area in the wave-affected zone", "none", c1, _area_bins_sum)
    _add("diam_ww", "m", "number-mean floe diameter", "none", c1, _diam_ww)
    _add("hice_ww", "m", "mean ice thickness in the wave-affected zone", "none", c1, _hice_ww)
    _add("fsdrad", "m", "mean floe radius", "none", c1, _fsdrad)
    _add("fsdperim", "1/m", "floe perimeter density", "none", c1, _fsdperim)


def _area_bins_sum():
    return jnp.sum(_area_bins(), axis=-1)


def _diam_ww():
    # number-mean diameter, following Eqn. 5 in HT2017
    queried = icepack_parameters.icepack_query_parameters("floeshape")
    floeshape = queried["floeshape"]
    nfsd = ice_history_shared.nfsd_hist
    ncat = ice_history_shared.ncat_hist
    nt = int(icepack_tracers.nt_fsd)
    radius = jnp.asarray(ice_arrays_column.floe_rad_c[:nfsd], dtype=dbl_kind)
    work = jnp.zeros(ice_state.aice.shape, dtype=dbl_kind)
    norm = jnp.zeros(ice_state.aice.shape, dtype=dbl_kind)
    if nt < 0:
        return work
    for n in range(ncat):
        for k in range(nfsd):
            count = ice_state.aicen[:, :, n] * ice_state.trcrn[:, :, nt + k, n] / (c4 * floeshape * radius[k] ** 2)
            work = work + count * radius[k]
            norm = norm + count
    present = norm > c0
    work = jnp.where(present, c2 * work / jnp.where(present, norm, c1), work)
    # >= smallest resolved diameter
    return jnp.maximum(c2 * radius[0], work)


def _hice_ww():
    puny = icepack_parameters.icepack_query_parameters("puny")["puny"]
    area = _area_bins_sum()
    nfsd = ice_history_shared.nfsd_hist
    ncat = ice_history_shared.ncat_hist
    nt = int(icepack_tracers.nt_fsd)
    volume = jnp.zeros(ice_state.aice.shape, dtype=dbl_kind)
    if nt >= 0:
        for n in range(ncat):
            for k in range(nfsd):
                volume = volume + ice_state.vicen[:, :, n] * ice_state.trcrn[:, :, nt + k, n]
    work = jnp.full(ice_state.aice.shape, c1, dtype=dbl_kind)
    present = area > puny
    return jnp.where(present, volume / jnp.where(present, area, c1), work)


def _fsdrad():
    puny = icepack_parameters.icepack_query_parameters("puny")["puny"]
    nfsd = ice_history_shared.nfsd_hist
    ncat = ice_history_shared.ncat_hist
    nt = int(icepack_tracers.nt_fsd)
    radius = jnp.asarray(ice_arrays_column.floe_rad_c[:nfsd], dtype=dbl_kind)
    work = jnp.zeros(ice_state.aice.shape, dtype=dbl_kind)
    if nt < 0:
        return work
    present = ice_state.aice > puny
    for k in range(nfsd):
        for n in range(ncat):
            piece = ice_state.trcrn[:, :, nt + k, n] * radius[k] * ice_state.aicen[:, :, n]
            work = work + jnp.where(present, piece / jnp.where(present, ice_state.aice, c1), c0)
    return work


def _fsdperim():
    puny = icepack_parameters.icepack_query_parameters("puny")["puny"]
    floeshape = icepack_parameters.icepack_query_parameters("floeshape")["floeshape"]
    nfsd = ice_history_shared.nfsd_hist
    ncat = ice_history_shared.ncat_hist
    nt = int(icepack_tracers.nt_fsd)
    radius = jnp.asarray(ice_arrays_column.floe_rad_c[:nfsd], dtype=dbl_kind)
    work = jnp.zeros(ice_state.aice.shape, dtype=dbl_kind)
    if nt < 0:
        return work
    present = ice_state.aice > puny
    for k in range(nfsd):
        for n in range(ncat):
            piece = (
                c8 * floeshape * ice_state.trcrn[:, :, nt + k, n] * radius[k]
                * ice_state.aicen[:, :, n]
                / (c4 * floeshape * radius[k] ** 2 * jnp.where(present, ice_state.aice, c1))
            )
            work = work + jnp.where(present, piece, c0)
    return work


def init_hist_fsd_3Df(dt):
    """Register floe-size category history fields."""
    if not icepack_query_tracer_flags("tr_fsd")["tr_fsd"]:
        return
    nfsd = ice_history_shared.nfsd_hist
    width = jnp.asarray(ice_arrays_column.floe_binwidth[:nfsd], dtype=dbl_kind)

    def afsd():
        return _area_bins() / width

    def tendency(name):
        def gather():
            return jnp.asarray(getattr(ice_arrays_column, name))[:, :, :nfsd] / dbl_kind(dt)
        return gather

    _add("afsd", "1/m", "floe size distribution", "none", c1, afsd, rank="3Df")
    _add("dafsd_newi", "1/s", "floe size tendency from new ice", "none", c1, tendency("d_afsd_newi"), rank="3Df")
    _add("dafsd_latg", "1/s", "floe size tendency from lateral growth", "none", c1, tendency("d_afsd_latg"), rank="3Df")
    _add("dafsd_latm", "1/s", "floe size tendency from lateral melt", "none", c1, tendency("d_afsd_latm"), rank="3Df")
    _add("dafsd_wave", "1/s", "floe size tendency from waves", "none", c1, tendency("d_afsd_wave"), rank="3Df")
    _add("dafsd_weld", "1/s", "floe size tendency from welding", "none", c1, tendency("d_afsd_weld"), rank="3Df")


def init_hist_fsd_4Df():
    """Register floe size by thickness category."""
    if not icepack_query_tracer_flags("tr_fsd")["tr_fsd"]:
        return
    nfsd = ice_history_shared.nfsd_hist
    ncat = ice_history_shared.ncat_hist

    def afsdn():
        nt = int(icepack_tracers.nt_fsd)
        width = jnp.asarray(ice_arrays_column.floe_binwidth[:nfsd], dtype=dbl_kind)
        work = jnp.zeros(ice_state.aice.shape + (nfsd, ncat), dtype=dbl_kind)
        if nt < 0:
            return work
        return (ice_state.trcrn[:, :, nt:nt + nfsd, :ncat]
                * ice_state.aicen[:, :, None, :ncat] / width[None, None, :, None])

    _add("afsdn", "1/m", "floe size distribution, categories", "none", c1, afsdn, rank="4Df")
