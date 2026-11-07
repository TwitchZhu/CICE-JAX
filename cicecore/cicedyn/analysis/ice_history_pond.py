"""Melt pond history output.

author for JAX version: Xinyuan Zhu
XZ: converted from F90 to Python

2012 Elizabeth Hunke split code from ice_history.F90
"""


import jax.numpy as jnp

from cicecore.cicedyn.analysis import ice_history_shared
from cicecore.cicedyn.general import ice_flux
from cicecore.cicedyn.general import ice_state
from cicecore.shared import ice_arrays_column
from cicecore.shared.ice_constants import c0, c1, c100
from cicecore.shared.ice_kinds_mod import dbl_kind
from icepack.columnphysics import icepack_parameters
from icepack.columnphysics import icepack_tracers
from icepack.columnphysics.icepack_tracers import icepack_query_tracer_flags

# Defaults match ice_history_pond.F90. Fields are dropped when ponds are off.
_DEFAULTS = {
    "apondn": "m",
    "apeffn": "m",
    "hpondn": "m",
    "apond": "x",
    "apond_ai": "x",
    "hpond": "x",
    "hpond_ai": "x",
    "ipond": "x",
    "ipond_ai": "x",
    "apeff": "x",
    "apeff_ai": "x",
    "dpnd_flush": "x",
    "dpnd_expon": "x",
    "dpnd_freebd": "x",
    "dpnd_initial": "x",
    "dpnd_dlid": "x",
    "dpnd_melt": "x",
    "dpnd_ridge": "x",
    "dpnd_flushn": "x",
    "dpnd_exponn": "x",
    "dpnd_freebdn": "x",
    "dpnd_initialn": "x",
    "dpnd_dlidn": "x",
    "simpconc": "m",
    "simpeffconc": "m",
    "simpthick": "m",
    "simprefrozen": "m",
}
_F = dict(_DEFAULTS)


def _flags():
    return icepack_query_tracer_flags(
        "tr_pond", "tr_pond_lvl", "tr_pond_topo", "tr_pond_sealvl",
    )


def _prepare():
    """Turn pond history off when that pond tracer is off.

    Topographic ponds do not yet write the drainage diagnostics.
    tcraig, July 2025.
    """
    global _F
    _F = dict(_DEFAULTS)
    ice_history_shared.note_nml("(init_hist_pond_2D)", "icefields_pond_nml")
    ice_history_shared.apply_history_flags(_F)
    flags = _flags()
    if not flags["tr_pond"]:
        for name in _F:
            _F[name] = "x"
    if flags["tr_pond_topo"]:
        for name in (
            "dpnd_flush", "dpnd_expon", "dpnd_freebd", "dpnd_initial",
            "dpnd_dlid", "dpnd_melt", "dpnd_ridge",
            "dpnd_flushn", "dpnd_exponn", "dpnd_freebdn", "dpnd_initialn", "dpnd_dlidn",
        ):
            _F[name] = "x"
    return flags


def _add(name, unit, desc, comment, cona, conb, gather, rank="2D", avg_ice_present="none", mask_ice_free_points=False):
    ice_history_shared.define_hist_field(
        name, unit, desc, comment, cona, conb, _F[name], gather,
        rank=rank, avg_ice_present=avg_ice_present, mask_ice_free_points=mask_ice_free_points,
    )


def init_hist_pond_2D():
    """Register 2D melt-pond history fields."""
    flags = _prepare()
    if not flags["tr_pond"]:
        return

    def fraction(trcr, aice):
        if flags["tr_pond_lvl"]:
            return trcr[:, :, icepack_tracers.nt_alvl] * trcr[:, :, icepack_tracers.nt_apnd]
        if flags["tr_pond_sealvl"] or flags["tr_pond_topo"]:
            return trcr[:, :, icepack_tracers.nt_apnd]
        return jnp.zeros(aice.shape, dtype=dbl_kind)

    def pond_inputs():
        return ice_state.trcr, ice_state.aice

    def weighted_inputs():
        return ice_state.trcr, ice_state.aice, ice_state.aice_init

    pond_fraction = ice_history_shared.device_gather(fraction, pond_inputs)
    weighted = ice_history_shared.device_gather(
        lambda trcr, aice, area: area * fraction(trcr, aice), weighted_inputs)
    depth = ice_history_shared.device_gather(
        lambda trcr, aice: fraction(trcr, aice) * trcr[:, :, icepack_tracers.nt_hpnd], pond_inputs)
    lid = ice_history_shared.device_gather(
        lambda trcr, aice: fraction(trcr, aice) * trcr[:, :, icepack_tracers.nt_ipnd], pond_inputs)
    depth_ai = ice_history_shared.device_gather(
        lambda trcr, aice, area: (area * fraction(trcr, aice)) * trcr[:, :, icepack_tracers.nt_hpnd],
        weighted_inputs)
    lid_ai = ice_history_shared.device_gather(
        lambda trcr, aice, area: (area * fraction(trcr, aice)) * trcr[:, :, icepack_tracers.nt_ipnd],
        weighted_inputs)

    puny = icepack_parameters.icepack_query_parameters("puny")["puny"]

    def effective_compute(aice, apeff_ai):
        # effective pond fraction of the ice, not of the grid cell
        present = aice > puny
        return jnp.where(present, apeff_ai / jnp.where(present, aice, c1), c0)

    effective = ice_history_shared.device_gather(
        effective_compute, lambda: (ice_state.aice, ice_flux.apeff_ai))

    _add("apond", "1", "melt pond fraction of sea ice", "none", c1, c0, pond_fraction)
    _add("apond_ai", "1", "melt pond fraction of grid cell", "weighted by ice area", c1, c0, weighted)
    _add("hpond", "m", "mean melt pond depth over sea ice", "none", c1, c0, depth)
    _add("hpond_ai", "m", "mean melt pond depth over grid cell", "weighted by ice area", c1, c0, depth_ai)
    _add("ipond", "m", "mean pond ice thickness over sea ice", "none", c1, c0, lid)
    _add("ipond_ai", "m", "mean pond ice thickness over grid cell", "weighted by ice area", c1, c0, lid_ai)
    _add("apeff", "1", "radiation-effective pond area fraction of sea ice", "none", c1, c0, effective)
    _add("apeff_ai", "1", "radiation-effective pond area fraction over grid cell", "weighted by ice area", c1, c0, lambda: ice_flux.apeff_ai)
    _add("dpnd_flush", "m/s", "pond flushing rate due to ice permeability", "none", c1, c0, lambda: ice_flux.dpnd_flush)
    _add("dpnd_expon", "m/s", "exponential pond drainage rate", "none", c1, c0, lambda: ice_flux.dpnd_expon)
    _add("dpnd_freebd", "m/s", "pond drainage rate due to freeboard constraint", "none", c1, c0, lambda: ice_flux.dpnd_freebd)
    _add("dpnd_initial", "m/s", "runoff rate due to rfrac", "none", c1, c0, lambda: ice_flux.dpnd_initial)
    _add("dpnd_dlid", "m/s", "pond loss / gain to ice lid freezing / melting", "none", c1, c0, lambda: ice_flux.dpnd_dlid)
    _add("dpnd_melt", "m/s", "pond drainage due to ice melting", "none", c1, c0, lambda: ice_flux.dpnd_melt)
    _add("dpnd_ridge", "m", "pond drainage due to ridging", "none", c1, c0, lambda: ice_flux.dpnd_ridge)
    _add("simpconc", "%", "percentage of sea ice covered by melt ponds", "area percentage of sea-ice surface that is covered by melt ponds", c100, c0, pond_fraction)
    _add("simpeffconc", "%", "percentage of sea ice covered by effective melt ponds", "area percentage of sea-ice surface that is covered by open melt ponds", c100, c0, lambda: ice_flux.apeff_ai)
    _add("simprefrozen", "m", "thickness of refrozen ice on melt ponds", "volume of refrozen ice on melt ponds divided by melt pond covered area", c1, c0, lid_ai, avg_ice_present="pond", mask_ice_free_points=True)
    _add("simpthick", "m", "melt pond depth", "average depth of melt ponds on sea ice, that is melt pond volume divided by melt pond area", c1, c0, depth_ai, avg_ice_present="pond", mask_ice_free_points=True)


def init_hist_pond_3Dc():
    """Register category melt-pond history fields."""
    if not _flags()["tr_pond"]:
        return
    ncat = ice_history_shared.ncat_hist
    def apondn():
        return ice_state.trcrn[:, :, icepack_tracers.nt_apnd, :ncat]

    def hpondn():
        return apondn() * ice_state.trcrn[:, :, icepack_tracers.nt_hpnd, :ncat]

    def cat(name):
        return jnp.asarray(getattr(ice_flux, name))[:, :, :ncat]

    _add("apondn", "1", "melt pond fraction, category", "none", c1, c0, apondn, rank="3Dc")
    _add("hpondn", "m", "melt pond depth, category", "none", c1, c0, hpondn, rank="3Dc")
    _add("apeffn", "1", "effective melt pond fraction, category", "none", c1, c0, lambda: jnp.asarray(ice_arrays_column.apeffn)[:, :, :ncat], rank="3Dc")
    _add("dpnd_flushn", "m/s", "category pond flushing rate due to ice permeability", "none", c1, c0, lambda: cat("dpnd_flushn"), rank="3Dc")
    _add("dpnd_exponn", "m/s", "category exponential pond drainage rate", "none", c1, c0, lambda: cat("dpnd_exponn"), rank="3Dc")
    _add("dpnd_freebdn", "m/s", "category pond drainage rate due to freeboard constraint", "none", c1, c0, lambda: cat("dpnd_freebdn"), rank="3Dc")
    _add("dpnd_initialn", "m/s", "category runoff rate due to rfrac", "none", c1, c0, lambda: cat("dpnd_initialn"), rank="3Dc")
    _add("dpnd_dlidn", "m/s", "category pond loss / gain to ice lid freezing / melting", "none", c1, c0, lambda: cat("dpnd_dlidn"), rank="3Dc")
