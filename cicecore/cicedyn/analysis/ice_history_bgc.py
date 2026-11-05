"""Biogeochemistry history output.

author for JAX version: Xinyuan Zhu
XZ: converted from F90 to Python

authors Elizabeth C. Hunke and Nicole Jeffery, LANL
2012 Elizabeth Hunke split code from ice_history.F90

2D diagnostics, category brine, and the vertical biology profiles are
registered here. Every frequency starts at 'x'. A tracer that is off
forces its fields back to 'x'. Derived frequencies (fluxes, column
integrals, snow, mobile fraction, mixed layer) copy the parent flag.
"""

import numpy as np
import jax.numpy as jnp

from cicecore.cicedyn.analysis import ice_history_shared
from cicecore.cicedyn.general import ice_flux_bgc
from cicecore.cicedyn.general import ice_state
from cicecore.shared import ice_arrays_column
from cicecore.shared import ice_domain_size
from cicecore.shared.ice_constants import c0, c1, c100
from cicecore.shared.ice_kinds_mod import dbl_kind
from icepack.columnphysics import icepack_parameters
from icepack.columnphysics import icepack_tracers
from icepack.columnphysics.icepack_tracers import icepack_query_tracer_flags


def init_history_bgc(pytree, static):
    """Clear BGC history fields at the start of each time step.

    This follows init_history_bgc in ice_history_bgc.F90. The fields are
    cleared even when the corresponding tracer switch is disabled.
    """
    state = dict(pytree["state"])
    column = dict(state["ice_arrays_column"])
    flux = dict(state["ice_flux_bgc"])
    for name in (
        "PP_net", "grow_net", "upNO", "upNH", "hbri", "ice_bio_net",
        "snow_bio_net", "fbio_snoice", "fbio_atmice", "bTiz", "bphi",
        "zfswin", "iDi", "iki",
    ):
        column[name] = jnp.zeros_like(column[name])
    for name in (
        "flux_bio", "flux_bio_ai", "fnit", "fsil", "famm", "fdmsp",
        "fdms", "fhum", "fdust", "falgalN", "fdoc", "fdic", "fdon",
        "ffep", "ffed",
    ):
        flux[name] = jnp.zeros_like(flux[name])
    state["ice_arrays_column"] = column
    state["ice_flux_bgc"] = flux
    result = dict(pytree)
    result["state"] = state
    return result

# Defaults match ice_history_bgc.F90: every flag starts as 'x'.
# The namelist icefields_bgc_nml is not read; these are the source defaults.
_DEFAULTS = {name: "x" for name in (
    "fiso_atm", "fiso_ocn", "iso", "faero_atm", "faero_ocn", "aero",
    "fbio", "fbio_ai", "zaero", "bgc_S", "bgc_N", "bgc_C", "bgc_DOC", "bgc_DIC",
    "bgc_chl", "bgc_Nit", "bgc_Am", "bgc_Sil", "bgc_DMSPp", "bgc_DMSPd", "bgc_DMS",
    "bgc_DON", "bgc_Fe", "bgc_hum", "bgc_PON", "bgc_ml", "upNO", "upNH",
    "bTin", "bphi", "iDi", "iki", "fbri", "hbri", "zfswin", "grownet",
    "bionet", "biosnow", "PPnet", "algalpeak", "zbgc_frac",
    "bgc_Fed", "bgc_Fep",
)}
_F = dict(_DEFAULTS)


def _add(flag, vname, unit, desc, comment, gather, rank, cona=c1):
    ice_history_shared.define_hist_field(
        vname, unit, desc, comment, cona, c0, _F[flag], gather, rank=rank,
    )


def _tag(number):
    """Fortran species suffix, 1-based and three digits: 1 -> '001'."""
    return f"{int(number):03d}"


def _off(*names):
    for name in names:
        _F[name] = "x"


def _copy(source, *names):
    for name in names:
        _F[name] = _F[source]


def _gate_bgc_frequencies():
    """Apply tracer flags and the derived-frequency copies.

    specified by combinations of the namelist frequencies.
    """
    flags = icepack_query_tracer_flags(
        "tr_iso", "tr_aero", "tr_brine", "tr_zaero",
        "tr_bgc_Nit", "tr_bgc_Am", "tr_bgc_Sil", "tr_bgc_DMS", "tr_bgc_PON",
        "tr_bgc_N", "tr_bgc_C", "tr_bgc_chl", "tr_bgc_DON", "tr_bgc_Fe", "tr_bgc_hum",
    )
    if not flags["tr_iso"]:
        _off("fiso_atm", "fiso_ocn", "iso")
    if not flags["tr_aero"]:
        _off("faero_atm", "faero_ocn", "aero")
    if not flags["tr_brine"]:
        _off("fbri", "hbri")

    _copy("bionet", "zaeronet")
    _copy("biosnow", "zaerosnow")
    _copy("zbgc_frac", "zaerofrac")
    _copy("fbio", "fzaero")
    _copy("fbio_ai", "fzaero_ai")
    if not flags["tr_zaero"]:
        _off("zaero", "fzaero", "fzaero_ai", "zaeronet", "zaerosnow", "zaerofrac")

    # skeletal biology has no vertical column, snow, or mobile fraction
    if bool(icepack_parameters.skl_bgc):
        _off("bionet", "biosnow", "zfswin", "zbgc_frac", "algalpeak")

    _copy("bgc_ml", "bgc_Nit_ml", "bgc_N_ml", "bgc_DOC_ml", "bgc_DIC_ml", "bgc_Sil_ml", "bgc_hum_ml", "bgc_Am_ml", "bgc_DMSP_ml", "bgc_DMS_ml", "bgc_DON_ml", "bgc_Fed_ml", "bgc_Fep_ml")
    _copy("bionet", "Nitnet", "Amnet", "Nnet", "chlnet", "Cnet", "DOCnet", "DICnet", "DONnet", "Fednet", "Fepnet", "Silnet", "humnet", "DMSPpnet", "DMSPdnet", "DMSnet", "PONnet")
    _copy("biosnow", "Nitsnow", "Amsnow", "Nsnow", "chlsnow", "Csnow", "DOCsnow", "DICsnow", "DONsnow", "Fedsnow", "Fepsnow", "Silsnow", "humsnow", "DMSPpsnow", "DMSPdsnow", "DMSsnow", "PONsnow")
    _copy("zbgc_frac", "Nitfrac", "Amfrac", "Nfrac", "chlfrac", "DOCfrac", "DICfrac", "DONfrac", "Fedfrac", "Fepfrac", "Silfrac", "humfrac", "DMSPpfrac", "DMSPdfrac", "DMSfrac", "PONfrac")
    _copy("bgc_DMS", "bgc_DMSPp", "bgc_DMSPd")
    _copy("bgc_Fe", "bgc_Fed", "bgc_Fep")
    _copy("fbio", "fDOC", "fDIC", "fDON", "fFed", "fFep", "fNit", "fAm", "fN", "fSil", "fhum", "fPON", "fDMSPp", "fDMSPd", "fDMS")
    _copy("fbio_ai", "fDOC_ai", "fDIC_ai", "fDON_ai", "fFed_ai", "fFep_ai", "fNit_ai", "fAm_ai", "fN_ai", "fSil_ai", "fhum_ai", "fPON_ai", "fDMSPp_ai", "fDMSPd_ai", "fDMS_ai")

    if not flags["tr_bgc_N"]:
        _off("bgc_N", "bgc_N_ml", "fN", "fN_ai", "Nsnow", "Nfrac", "Nnet", "algalpeak")
    _F["peakval"] = _F["algalpeak"]
    if not flags["tr_bgc_Nit"]:
        _off("upNO", "bgc_Nit", "bgc_Nit_ml", "fNit", "fNit_ai", "Nitsnow", "Nitfrac", "Nitnet")
    if not flags["tr_bgc_C"]:
        _off("bgc_C", "bgc_DOC", "bgc_DIC", "fDOC", "fDOC_ai", "bgc_DOC_ml", "bgc_DIC_ml", "Csnow", "DOCsnow", "DICsnow", "DOCfrac", "DICfrac", "Cnet", "DOCnet", "DICnet")
    if not flags["tr_bgc_chl"]:
        _off("bgc_chl", "chlnet", "chlsnow", "chlfrac")
    if not flags["tr_bgc_Am"]:
        _off("upNH", "bgc_Am", "bgc_Am_ml", "fAm", "fAm_ai", "Amsnow", "Amfrac", "Amnet")
    if not flags["tr_bgc_Sil"]:
        _off("bgc_Sil", "bgc_Sil_ml", "fSil", "fSil_ai", "Silnet", "Silsnow", "Silfrac")
    if not flags["tr_bgc_hum"]:
        _off("bgc_hum", "bgc_hum_ml", "fhum", "fhum_ai", "humnet", "humsnow", "humfrac")
    if not flags["tr_bgc_DMS"]:
        _off("bgc_DMS", "bgc_DMSPp", "bgc_DMSPd", "bgc_DMSP_ml", "bgc_DMS_ml", "fDMS", "fDMSPp", "fDMSPd", "fDMS_ai", "fDMSPp_ai", "fDMSPd_ai", "DMSnet", "DMSPpnet", "DMSPdnet", "DMSsnow", "DMSPpsnow", "DMSPdsnow", "DMSfrac", "DMSPpfrac", "DMSPdfrac")
    if not flags["tr_bgc_DON"]:
        _off("bgc_DON", "bgc_DON_ml", "DONsnow", "DONfrac", "DONnet", "fDON", "fDON_ai")
    if not flags["tr_bgc_Fe"]:
        _off("bgc_Fe", "bgc_Fed", "bgc_Fed_ml", "Fedsnow", "Fedfrac", "Fednet", "fFed", "fFed_ai", "bgc_Fep", "bgc_Fep_ml", "Fepsnow", "Fepfrac", "Fepnet", "fFep", "fFep_ai")
    if not flags["tr_bgc_PON"] or bool(icepack_parameters.skl_bgc):
        _off("bgc_PON", "PONsnow", "PONfrac", "PONnet", "fPON", "fPON_ai")

    _F["bgc_Nit_cat1"] = _F["bgc_Nit"]
    _F["bgc_Am_cat1"] = _F["bgc_Am"]
    _F["bgc_N_cat1"] = _F["bgc_N"]
    _F["bgc_DOC_cat1"] = _F["bgc_DOC"]
    _F["bgc_DIC_cat1"] = _F["bgc_DIC"]
    _F["bgc_DON_cat1"] = _F["bgc_DON"]
    _F["bgc_Fed_cat1"] = _F["bgc_Fe"]
    _F["bgc_Fep_cat1"] = _F["bgc_Fe"]
    _F["bgc_Sil_cat1"] = _F["bgc_Sil"]
    _F["bgc_hum_cat1"] = _F["bgc_hum"]
    _F["bgc_DMSPd_cat1"] = _F["bgc_DMSPd"]
    _F["bgc_DMS_cat1"] = _F["bgc_DMS"]
    _F["bgc_PON_cat1"] = _F["bgc_PON"]
    return flags


def _on(name):
    return str(_F[name]).strip() != "x"


def _zeros():
    return jnp.zeros(ice_state.aice.shape, dtype=dbl_kind)


def _tracer_plane(slot, scale=c1):
    def gather(slot=int(slot), scale=scale):
        if slot < 0:
            return _zeros()
        return scale * ice_state.trcr[:, :, slot]
    return gather


def _species(count, values):
    count = int(count)
    values = np.atleast_1d(np.asarray(values))
    return range(min(count, values.size))


def _add2(flag, vname, unit, desc, comment, gather):
    _add(flag, vname, unit, desc, comment, gather, "2D")


def init_hist_bgc_2D():
    """Register the 2D isotope, aerosol, and biogeochemistry diagnostics."""
    global _F
    _F = dict(_DEFAULTS)
    ice_history_shared.note_nml("(init_hist_bgc_2D)", "icefields_bgc_nml")
    ice_history_shared.apply_history_flags(_F)
    flags = _gate_bgc_frequencies()
    active = flags["tr_iso"] or flags["tr_aero"] or flags["tr_brine"] or bool(icepack_parameters.skl_bgc)
    if not active:
        return

    rhos = icepack_parameters.rhos
    rhoi = icepack_parameters.rhoi
    sk_l = icepack_parameters.sk_l

    if _on("iso"):
        for n in range(int(icepack_tracers.n_iso)):
            tag = _tag(n + 1)
            _add2("iso", "isosno" + tag, "kg/kg", "snow isotope mass concentration", "none", _tracer_plane(int(icepack_tracers.nt_isosno) + n, c1 / rhos))
            _add2("iso", "isoice" + tag, "kg/kg", "ice isotope mass concentration", "none", _tracer_plane(int(icepack_tracers.nt_isoice) + n, c1 / rhos))
    if _on("fiso_atm"):
        for n in range(int(icepack_tracers.n_iso)):
            _add2("fiso_atm", "fiso_atm" + _tag(n + 1), "kg/m^2 s", "isotope deposition rate", "none", lambda n=n: ice_flux_bgc.fiso_atm[:, :, n])
    if _on("fiso_ocn"):
        for n in range(int(icepack_tracers.n_iso)):
            _add2("fiso_ocn", "fiso_ocn" + _tag(n + 1), "kg/m^2 s", "isotope flux to ocean", "none", lambda n=n: ice_flux_bgc.fiso_ocn[:, :, n])

    # Aerosols. Four slots per species: snow ssl, snow int, ice ssl, ice int.
    if _on("aero"):
        base = int(icepack_tracers.nt_aero)
        for n in range(int(icepack_tracers.n_aero)):
            tag = _tag(n + 1)
            _add2("aero", "aerosnossl" + tag, "kg/kg", "snow ssl aerosol mass", "none", _tracer_plane(base + 4 * n, c1 / rhos))
            _add2("aero", "aerosnoint" + tag, "kg/kg", "snow int aerosol mass", "none", _tracer_plane(base + 1 + 4 * n, c1 / rhos))
            _add2("aero", "aeroicessl" + tag, "kg/kg", "ice ssl aerosol mass", "none", _tracer_plane(base + 2 + 4 * n, c1 / rhoi))
            _add2("aero", "aeroiceint" + tag, "kg/kg", "ice int aerosol mass", "none", _tracer_plane(base + 3 + 4 * n, c1 / rhoi))
    if _on("faero_atm"):
        for n in range(int(icepack_tracers.n_aero)):
            _add2("faero_atm", "faero_atm" + _tag(n + 1), "kg/m^2 s", "aerosol deposition rate", "none", lambda n=n: ice_flux_bgc.faero_atm[:, :, n])
    if _on("faero_ocn"):
        for n in range(int(icepack_tracers.n_aero)):
            _add2("faero_ocn", "faero_ocn" + _tag(n + 1), "kg/m^2 s", "aerosol flux to ocean", "none", lambda n=n: ice_flux_bgc.faero_ocn[:, :, n])

    if bool(icepack_parameters.skl_bgc):
        # skeletal layer: bottom 3 cm
        _skl_species("bgc_N", icepack_tracers.n_algae, icepack_tracers.nt_bgc_N, "algal_N", "mmol/m^2", "Bulk ice bottom algae (nitrogen)", sk_l)
        _skl_species("bgc_chl", icepack_tracers.n_algae, icepack_tracers.nt_bgc_chl, "algal_chl", "mg chl/m^2", "Bulk ice bottom algae (chlorophyll)", sk_l)
        _skl_carbon()
        _skl_species("bgc_DOC", icepack_tracers.n_doc, icepack_tracers.nt_bgc_DOC, "DOC", "mmol/m^2", "Bulk DOC (carbon)", sk_l)
        _skl_species("bgc_DIC", icepack_tracers.n_dic, icepack_tracers.nt_bgc_DIC, "DIC", "mmol/m^2", "Bulk DIC (carbon)", sk_l)
        _skl_species("bgc_DON", icepack_tracers.n_don, icepack_tracers.nt_bgc_DON, "DON", "mmol/m^2", "Bulk ice bottom DON (nitrogen)", sk_l)
        _skl_species("bgc_Fed", icepack_tracers.n_fed, icepack_tracers.nt_bgc_Fed, "dFe", "umol/m^2", "Bulk ice bottom dissolved Fe (iron)", sk_l)
        _skl_species("bgc_Fep", icepack_tracers.n_fep, icepack_tracers.nt_bgc_Fep, "pFe", "umol/m^2", "Bulk ice bottom particulate Fe (iron)", sk_l)
        _skl_one("bgc_Nit", "nt_bgc_Nit", "Nit", "mmol/m^2", "Bulk skeletal nutrient (nitrate)", sk_l)
        _skl_one("bgc_Am", "nt_bgc_Am", "Am", "mmol/m^2", "Bulk skeletal nutrient (ammonia/um)", sk_l)
        _skl_one("bgc_Sil", "nt_bgc_Sil", "Sil", "mmol/m^2", "Bulk skeletal nutrient (silicate)", sk_l, "skelelal layer: bottom 3 cm")
        _skl_one("bgc_hum", "nt_bgc_hum", "hum", "mmol/m^2", "Bulk skeletal humic material (carbon)", sk_l, "skelelal layer: bottom 3 cm")
        _skl_one("bgc_PON", "nt_bgc_PON", "PON", "mmol/m^2", "Bulk skeletal nutrient (silicate)", sk_l, "skelelal layer: bottom 3 cm")
        _skl_one("bgc_DMSPp", "nt_bgc_DMSPp", "DMSPp", "mmol/m^2", "Bulk particulate S in algae (DMSPp)", sk_l)
        _skl_one("bgc_DMSPd", "nt_bgc_DMSPd", "DMSPd", "mmol/m^2", "Bulk dissolved skl precursor (DSMPd)", sk_l)
        _skl_one("bgc_DMS", "nt_bgc_DMS", "DMS", "mmol/m^2", "Bulk dissolved skl trace gas (DMS)", sk_l)

    # vertical and skeletal layer biogeochemistry
    _ml_species("bgc_DOC_ml", icepack_tracers.n_doc, icepack_tracers.nlt_bgc_DOC, "ml_DOC", "mmol/m^3", "mixed layer DOC (carbon)")
    _ml_species("bgc_DIC_ml", icepack_tracers.n_dic, icepack_tracers.nlt_bgc_DIC, "ml_DIC", "mmol/m^3", "mixed layer DIC (carbon)")
    _ml_species("bgc_DON_ml", icepack_tracers.n_don, icepack_tracers.nlt_bgc_DON, "ml_DON", "mmol/m^3", "mixed layer DON (nitrogen)")
    _ml_species("bgc_Fed_ml", icepack_tracers.n_fed, icepack_tracers.nlt_bgc_Fed, "ml_dFe", "nM", "mixed layer dissolved Fe (iron)")
    _ml_species("bgc_Fep_ml", icepack_tracers.n_fep, icepack_tracers.nlt_bgc_Fep, "ml_pFe", "nM", "mixed layer particulate Fe (iron)")
    _ml_species("bgc_N_ml", icepack_tracers.n_algae, icepack_tracers.nlt_bgc_N, "ml_N", "mmol/m^3", "mixed layer nitrogen")
    _ml_one("bgc_Nit_ml", "nlt_bgc_Nit", "ml_Nit", "mmol/m^3", "mixed layer nutrient (nitrate)")
    _ml_one("bgc_Am_ml", "nlt_bgc_Am", "ml_Am", "mmol/m^3", "mixed layer nutrient (ammonia/um)")
    _ml_one("bgc_Sil_ml", "nlt_bgc_Sil", "ml_Sil", "mmol/m^3", "mixed layer nutrient (silicate)")
    _ml_one("bgc_hum_ml", "nlt_bgc_hum", "ml_hum", "mmol/m^3", "mixed layer humic material (carbon)")
    _ml_one("bgc_DMSP_ml", "nlt_bgc_DMSPd", "ml_DMSP", "mmol/m^3", "mixed layer precursor (DMSP)")
    _ml_one("bgc_DMS_ml", "nlt_bgc_DMS", "ml_DMS", "mmol/m^3", "mixed layer trace gas (DMS)")

    _flux_one("fNit", "fNit_ai", "nlt_bgc_Nit", "fNit", "nitrate flux ice to ocn (cpl)", "if positive, ocean gains nitrate", "nitrate flux ice to ocean", "weighted by ice area")
    _flux_one("fAm", "fAm_ai", "nlt_bgc_Am", "fAm", "ammonium flux ice to ocn (cpl)", "if positive, ocean gains ammonium", "ammonium flux ice to ocean", "weighted by ice area")
    _flux_species("fN", "fN_ai", icepack_tracers.n_algae, icepack_tracers.nlt_bgc_N, "fN", "mmol/m^2/s", "algal N flux ice to ocn (cpl)", "algal N flux ice to ocean", "if positive, ocean gains algal N")
    _flux_species("fDOC", "fDOC_ai", icepack_tracers.n_doc, icepack_tracers.nlt_bgc_DOC, "fDOC", "mmol/m^2/s", "DOC flux ice to ocn (cpl)", "DOC flux ice to ocn")
    _flux_species("fDIC", "fDIC_ai", icepack_tracers.n_dic, icepack_tracers.nlt_bgc_DIC, "fDIC", "mmol/m^2/s", "DIC flux ice to ocn (cpl)", "DIC flux ice to ocn")
    _flux_species("fDON", "fDON_ai", icepack_tracers.n_don, icepack_tracers.nlt_bgc_DON, "fDON", "mmol/m^2/s", "DON flux ice to ocn (cpl)", "DON flux ice to ocn")
    _flux_species("fFed", "fFed_ai", icepack_tracers.n_fed, icepack_tracers.nlt_bgc_Fed, "fdFe", "umol/m^2/s", "dFe flux ice to ocn (cpl)", "dFe flux ice to ocn")
    _flux_species("fFep", "fFep_ai", icepack_tracers.n_fep, icepack_tracers.nlt_bgc_Fep, "fpFe", "umol/m^2/s", "pFe flux ice to ocn (cpl)", "pFe flux ice to ocn")
    _flux_one("fSil", "fSil_ai", "nlt_bgc_Sil", "fSil", "silicate flux ice to ocn (cpl)", "positive into ocean", "silicate flux ice to ocean", "weighted by ice area")
    _flux_one("fhum", "fhum_ai", "nlt_bgc_hum", "fhum", "humic matter (carbon) flux ice to ocn (cpl)", "positive into ocean", "humic matter (carbon) flux ice to ocean", "weighted by ice area")
    _flux_one("fPON", "fPON_ai", "nlt_bgc_PON", "fPON", "PON flux ice to ocean", "positive into ocean", "PON flux ice to ocean", "weighted by ice area")
    _flux_one("fDMS", "fDMS_ai", "nlt_bgc_DMS", "fDMS", "DMS flux ice to ocean", "positive into ocean", "DMS flux ice to ocean", "weighted by ice area")
    _flux_one("fDMSPd", "fDMSPd_ai", "nlt_bgc_DMSPd", "fDMSPd", "DMSPd flux ice to ocean", "positive into ocean", "DMSPd flux ice to ocean", "weighted by ice area")
    _flux_one("fDMSPp", "fDMSPp_ai", "nlt_bgc_DMSPp", "fDMSPp", "DMSPp flux ice to ocean", "positive into ocean", "DMSPp flux ice to ocean", "weighted by ice area")
    _add2("PPnet", "PP_net", "mg C/m^2/d", "Net Primary Production", "weighted by ice area", lambda: ice_arrays_column.PP_net)
    _add2("grownet", "grow_net", "m/d", "Net specific growth", "weighted by brine or skl volume ", lambda: ice_arrays_column.grow_net)
    _add2("upNO", "upNO", "mmol/m^2/d", "Tot algal Nit uptake rate", "weighted by ice area", lambda: ice_arrays_column.upNO)
    _add2("upNH", "upNH", "mmol/m^2/d", "Tot algal Am uptake rate", "weighted by ice area", lambda: ice_arrays_column.upNH)

    if bool(icepack_parameters.z_tracers):
        _flux_species("fzaero", "fzaero_ai", icepack_tracers.n_zaero, icepack_tracers.nlt_zaero, "fzaero", "kg/m^2/s", "z aerosol flux ice to ocn (cpl)", "z aerosol flux ice to ocn")
        _algal_peak()
        _column_species("zaeronet", icepack_tracers.n_zaero, icepack_tracers.nlt_zaero, "zaero_net", "kg/m^2", "Net z aerosol", "ice")
        _column_algae("chlnet", "chl_net", "mg chl/m^2", "Net chlorophyll", "ice", "chl")
        _column_algae("Nnet", "algalN_net", "mmol/m^2", "Net algal nitrogen", "ice", "N")
        _column_algae("Cnet", "algalC_net", "mmol/m^2", "Net algal carbon", "ice", "C")
        _column_species("DOCnet", icepack_tracers.n_doc, icepack_tracers.nlt_bgc_DOC, "DOC_net", "mmol/m^2", "Net DOC", "ice")
        _column_species("DICnet", icepack_tracers.n_dic, icepack_tracers.nlt_bgc_DIC, "DIC_net", "mmol/m^2", "Net DIC", "ice")
        _column_species("DONnet", icepack_tracers.n_don, icepack_tracers.nlt_bgc_DON, "DON_net", "mmol/m^2", "Net DON", "ice")
        _column_species("Fednet", icepack_tracers.n_fed, icepack_tracers.nlt_bgc_Fed, "dFe_net", "umol/m^2", "Net dFe", "ice")
        _column_species("Fepnet", icepack_tracers.n_fep, icepack_tracers.nlt_bgc_Fep, "pFe_net", "umol/m^2", "Net pFe", "ice")
        _column_one("Nitnet", "nlt_bgc_Nit", "Nit_net", "Net Nitrate", "ice")
        _column_one("Amnet", "nlt_bgc_Am", "Am_net", "Net Ammonium", "ice")
        _column_one("Silnet", "nlt_bgc_Sil", "Sil_net", "Net Silicate", "ice")
        _column_one("humnet", "nlt_bgc_hum", "hum_net", "Net humic material (carbon)", "ice")
        _column_one("DMSPpnet", "nlt_bgc_DMSPp", "DMSPp_net", "Net DMSPp", "ice")
        _column_one("DMSPdnet", "nlt_bgc_DMSPd", "DMSPd_net", "Net DMSPd", "ice")
        _column_one("DMSnet", "nlt_bgc_DMS", "DMS_net", "Net DMS", "ice")
        _column_one("PONnet", "nlt_bgc_PON", "PON_net", "Net Nitrate if no reactions", "ice")
        _column_species("zaerosnow", icepack_tracers.n_zaero, icepack_tracers.nlt_zaero, "zaero_snow", "kg/m^2", "Snow z aerosol", "snow")
        _column_algae("chlsnow", "chl_snow", "mg chl/m^2", "Snow chlorophyll", "snow", "chl")
        _column_algae("Nsnow", "algalN_snow", "mmol/m^2", "Snow algal nitrogen", "snow", "N")
        _column_algae("Csnow", "algalC_snow", "mmol/m^2", "Snow algal carbon", "snow", "C")
        _column_species("DOCsnow", icepack_tracers.n_doc, icepack_tracers.nlt_bgc_DOC, "DOC_snow", "mmol/m^2", "Snow DOC", "snow")
        _column_species("DICsnow", icepack_tracers.n_dic, icepack_tracers.nlt_bgc_DIC, "DIC_snow", "mmol/m^2", "Snow DIC", "snow")
        _column_species("DONsnow", icepack_tracers.n_don, icepack_tracers.nlt_bgc_DON, "DON_snow", "mmol/m^2", "Snow DON", "snow")
        _column_species("Fedsnow", icepack_tracers.n_fed, icepack_tracers.nlt_bgc_Fed, "dFe_snow", "umol/m^2", "Snow dFe", "snow")
        _column_species("Fepsnow", icepack_tracers.n_fep, icepack_tracers.nlt_bgc_Fep, "pFe_snow", "umol/m^2", "Snow pFe", "snow")
        _column_one("Nitsnow", "nlt_bgc_Nit", "Nit_snow", "Snow Nitrate", "snow")
        _column_one("Amsnow", "nlt_bgc_Am", "Am_snow", "Snow Ammonium", "snow")
        _column_one("Silsnow", "nlt_bgc_Sil", "Sil_snow", "Snow Silicate", "snow")
        _column_one("humsnow", "nlt_bgc_hum", "hum_snow", "Snow humic material (carbon)", "snow")
        _column_one("DMSPpsnow", "nlt_bgc_DMSPp", "DMSPp_snow", "Snow DMSPp", "snow")
        _column_one("DMSPdsnow", "nlt_bgc_DMSPd", "DMSPd_snow", "Snow DMSPd", "snow")
        _column_one("DMSsnow", "nlt_bgc_DMS", "DMS_snow", "Snow DMS", "snow")
        _column_one("PONsnow", "nlt_bgc_PON", "PON_snow", "Snow Nitrate if no reactions", "snow")
        _frac_species("zaerofrac", icepack_tracers.n_zaero, icepack_tracers.nlt_zaero, "zaero_frac", "1", "Mobile frac z aerosol")
        _chl_frac()
        _frac_species("Nfrac", icepack_tracers.n_algae, icepack_tracers.nlt_bgc_N, "algalN_frac", "1", "Mobile frac algal nitrogen")
        _frac_species("DOCfrac", icepack_tracers.n_doc, icepack_tracers.nlt_bgc_DOC, "DOC_frac", "1", "Mobile frac DOC")
        _frac_species("DICfrac", icepack_tracers.n_dic, icepack_tracers.nlt_bgc_DIC, "DIC_frac", "1", "Mobile frac DIC")
        _frac_species("DONfrac", icepack_tracers.n_don, icepack_tracers.nlt_bgc_DON, "DON_frac", "1", "Mobile frac DON")
        _frac_species("Fedfrac", icepack_tracers.n_fed, icepack_tracers.nlt_bgc_Fed, "dFe_frac", "1", "Mobile frac dFe")
        _frac_species("Fepfrac", icepack_tracers.n_fep, icepack_tracers.nlt_bgc_Fep, "pFe_frac", "1", "Mobile frac pFe")
        _frac_one("Nitfrac", "nlt_bgc_Nit", "Nit_frac", "Mobile frac Nitrate")
        _frac_one("Amfrac", "nlt_bgc_Am", "Am_frac", "Mobile frac Ammonium")
        _frac_one("Silfrac", "nlt_bgc_Sil", "Sil_frac", "Mobile frac Silicate")
        _frac_one("humfrac", "nlt_bgc_hum", "hum_frac", "Mobile frac humic material")
        _frac_one("DMSPpfrac", "nlt_bgc_DMSPp", "DMSPp_frac", "Mobile frac DMSPp")
        _frac_one("DMSPdfrac", "nlt_bgc_DMSPd", "DMSPd_frac", "Mobile frac DMSPd")
        _frac_one("DMSfrac", "nlt_bgc_DMS", "DMS_frac", "Mobile frac DMS")
        _frac_one("PONfrac", "nlt_bgc_PON", "PON_frac", "Mobile frac Nitrate if no reactions")

    # brine
    _add2("hbri", "hbrine", "m", "Brine height", "distance from ice bottom to brine surface", lambda: ice_arrays_column.hbri)


def _skl_species(flag, count, slots, stem, unit, desc, sk_l):
    if not _on(flag):
        return
    for n in _species(count, slots):
        _add2(flag, stem + _tag(n + 1), unit, desc, "skeletal layer: bottom 3 cm", _tracer_plane(int(slots[n]), sk_l))


def _skl_carbon():
    if not _on("bgc_C"):
        return
    ratios = np.asarray(ice_arrays_column.R_C2N)
    slots = icepack_tracers.nt_bgc_N
    sk_l = icepack_parameters.sk_l
    for n in _species(icepack_tracers.n_algae, slots):
        _add2("bgc_C", "algal_C" + _tag(n + 1), "mmol C/m^2", "Bulk ice bottom diatoms (carbon)", "skeletal layer: bottom 3 cm", _tracer_plane(int(slots[n]), sk_l * ratios[n]))


def _skl_one(flag, nt_name, vname, unit, desc, sk_l, comment="skeletal layer: bottom 3 cm"):
    if not _on(flag):
        return
    _add2(flag, vname, unit, desc, comment, _tracer_plane(int(getattr(icepack_tracers, nt_name)), sk_l))


def _layer(array_name, nlt):
    """nlt is 1-based. 0 means that tracer type is off."""
    def gather(array_name=array_name, nlt=int(nlt)):
        if nlt < 1:
            return _zeros()
        return jnp.asarray(getattr(ice_arrays_column, array_name))[:, :, nlt - 1]
    return gather


def _ml_species(flag, count, slots, stem, unit, desc):
    if not _on(flag):
        return
    for n in _species(count, slots):
        _add2(flag, stem + _tag(n + 1), unit, desc, "upper ocean", _layer("ocean_bio", slots[n]))


def _ml_one(flag, nlt_name, vname, unit, desc):
    if not _on(flag):
        return
    _add2(flag, vname, unit, desc, "upper ocean", _layer("ocean_bio", getattr(icepack_tracers, nlt_name)))


def _bio_flux(nlt, area_weighted):
    name = "flux_bio_ai" if area_weighted else "flux_bio"
    def gather(name=name, nlt=int(nlt)):
        if nlt < 1:
            return _zeros()
        return jnp.asarray(getattr(ice_flux_bgc, name))[:, :, nlt - 1]
    return gather


def _flux_one(flag, flag_ai, nlt_name, stem, desc, comment, desc_ai, comment_ai):
    nlt = getattr(icepack_tracers, nlt_name)
    if _on(flag):
        _add2(flag, stem, "mmol/m^2/s", desc, comment, _bio_flux(nlt, False))
    if _on(flag_ai):
        _add2(flag_ai, stem + "_ai", "mmol/m^2/s", desc_ai, comment_ai, _bio_flux(nlt, True))


def _flux_species(flag, flag_ai, count, slots, stem, unit, desc, desc_ai, comment="positive to ocean"):
    for n in _species(count, slots):
        tag = _tag(n + 1)
        if _on(flag):
            _add2(flag, stem + tag, unit, desc, comment, _bio_flux(slots[n], False))
        if _on(flag_ai):
            _add2(flag_ai, stem + "_ai" + tag, unit, desc_ai, "weighted by ice area", _bio_flux(slots[n], True))


def _column_plane(which, nlt, scale=c1):
    array_name = "snow_bio_net" if which == "snow" else "ice_bio_net"
    def gather(array_name=array_name, nlt=int(nlt), scale=scale):
        if nlt < 1:
            return _zeros()
        return scale * jnp.asarray(getattr(ice_arrays_column, array_name))[:, :, nlt - 1]
    return gather


def _column_species(flag, count, slots, stem, unit, desc, which):
    if not _on(flag):
        return
    for n in _species(count, slots):
        _add2(flag, stem + _tag(n + 1), unit, desc, "weighted by ice area ", _column_plane(which, slots[n]))


def _column_algae(flag, stem, unit, desc, which, kind):
    if not _on(flag):
        return
    slots = icepack_tracers.nlt_bgc_N
    if kind == "chl":
        ratios = np.asarray(ice_arrays_column.R_chl2N)
    elif kind == "C":
        ratios = np.asarray(ice_arrays_column.R_C2N)
    else:
        ratios = None
    for n in _species(icepack_tracers.n_algae, slots):
        scale = c1 if ratios is None else ratios[n]
        _add2(flag, stem + _tag(n + 1), unit, desc, "weighted by ice area ", _column_plane(which, slots[n], scale))


def _column_one(flag, nlt_name, vname, desc, which):
    if not _on(flag):
        return
    _add2(flag, vname, "mmol/m^2", desc, "weighted by ice area", _column_plane(which, getattr(icepack_tracers, nlt_name)))


def _mobile(nlt):
    """Mobile fraction lives at nt_zbgc_frac plus the 1-based type index."""
    def gather(nlt=int(nlt)):
        base = int(icepack_tracers.nt_zbgc_frac)
        if base < 0 or nlt < 1:
            return _zeros()
        return ice_state.trcr[:, :, base + nlt - 1]
    return gather


def _frac_species(flag, count, slots, stem, unit, desc):
    if not _on(flag):
        return
    for n in _species(count, slots):
        _add2(flag, stem + _tag(n + 1), unit, desc, "averaged over depth ", _mobile(slots[n]))


def _frac_one(flag, nlt_name, vname, desc):
    if not _on(flag):
        return
    _add2(flag, vname, "1", desc, "averaged over depth", _mobile(getattr(icepack_tracers, nlt_name)))


def _chl_frac():
    # Fortran stores this one at the 1-based nlt index, not at nt_zbgc_frac.
    if not _on("chlfrac"):
        return
    slots = icepack_tracers.nlt_bgc_N
    for n in _species(icepack_tracers.n_algae, slots):
        _add2("chlfrac", "chl_frac" + _tag(n + 1), "mg chl/m^2", "Mobile frac chlorophyll", "averaged over depth ", _tracer_plane(int(slots[n]) - 1))


def _algal_peak():
    """zgrid level of the unique maximum. 0 if no distinct peak.

    The written level is 1-based, matching Fortran maxloc.
    """
    if not (_on("algalpeak") or _on("peakval")):
        return
    slots = icepack_tracers.nt_bgc_N
    nblyr = int(ice_domain_size.nblyr)
    puny = icepack_parameters.puny
    for n in _species(icepack_tracers.n_algae, slots):
        nt = int(slots[n])

        def levels(nt=nt):
            loc = jnp.zeros(ice_state.aice.shape, dtype=dbl_kind)
            peak = jnp.zeros(ice_state.aice.shape, dtype=dbl_kind)
            if nt < 0:
                return loc, peak
            profile = ice_state.trcr[:, :, nt : nt + nblyr + 1]
            present = ice_state.aice > c0
            if profile.shape[-1] < 1:
                return loc, peak
            imax = jnp.argmax(profile, axis=-1)
            maxv = jnp.max(profile, axis=-1)
            if profile.shape[-1] == 1:
                unique = maxv > puny
            else:
                order = jnp.sort(profile, axis=-1)
                unique = (order[..., -2] - maxv) < -puny
            loc = jnp.where(present & unique, imax + 1, c0)
            peak = jnp.where(present, maxv, c0)
            return loc, peak

        if _on("algalpeak"):
            _add2("algalpeak", "peak_loc" + _tag(n + 1), "1", "zgrid level of peak chla", "0 if no distinct peak", lambda levels=levels: levels()[0])
        if _on("peakval"):
            _add2("peakval", "peak_val" + _tag(n + 1), "mg/m^3", "Bulk concentration of peak chla", "at peak_loc", lambda levels=levels: levels()[1])


def init_hist_bgc_3Dc():
    """Register the category brine fraction."""
    if not icepack_query_tracer_flags("tr_brine")["tr_brine"]:
        return
    ncat = ice_history_shared.ncat_hist

    def fbri():
        nt = int(icepack_tracers.nt_fbri)
        work = jnp.zeros(ice_state.aicen[:, :, :ncat].shape, dtype=dbl_kind)
        if nt < 0:
            return work
        return ice_state.trcrn[:, :, nt, :ncat]

    _add("fbri", "fbrine", "1", "brine tracer fraction of ice volume, cat", "none", fbri, "3Dc")


def _weighted_column(source):
    """Sum a category vertical profile weighted by category ice area.

    source has shape (i, j, level, category). The result is (i, j, level).
    """
    ncat = ice_history_shared.ncat_hist
    levels = source.shape[2]
    work = jnp.zeros(ice_state.aice.shape + (levels,), dtype=dbl_kind)
    area = ice_state.aicen[:, :, :ncat]
    for n in range(ncat):
        present = area[:, :, n] > c0
        work = work + jnp.where(present[..., None], source[..., n] * area[..., n, None], c0)
    return work


def init_hist_bgc_3Db():
    """Register vertical biology profiles on the bio grid."""
    if not bool(icepack_parameters.z_tracers):
        return
    nblyr = int(ice_domain_size.nblyr)
    nzb = ice_history_shared.nzblyr
    secday = icepack_parameters.icepack_query_parameters("secday")["secday"]

    def profile(name):
        def gather():
            # bTiz, bphi are (i, j, level, category)
            return _weighted_column(jnp.asarray(getattr(ice_arrays_column, name))[:, :, :nzb, :])
        return gather

    def zfsw():
        source = jnp.asarray(ice_arrays_column.zfswin)
        work = _weighted_column(source[:, :, : nblyr + 1, :])
        # the lowest bio level repeats the lowest ice level
        extra = _weighted_column(source[:, :, nblyr : nblyr + 1, :])
        out = jnp.zeros(ice_state.aice.shape + (nzb,), dtype=dbl_kind)
        out = out.at[:, :, : nblyr + 1].set(work)
        out = out.at[:, :, nzb - 1].set(extra[:, :, 0])
        return out

    _add("bTin", "bTizn", "C", "ice internal temperatures on bio grid", "interpolated to bio grid", profile("bTiz"), "3Db")
    _add("bphi", "bphizn", "%", "porosity", "brine volume fraction", profile("bphi"), "3Db", cona=c100)
    _add("iDi", "iDin", "m^2/d", "interface diffusivity", "on bio interface grid", _idi, "3Db", cona=secday)
    _add("iki", "ikin", "mm^2", "permeability", "on bio interface grid", _iki, "3Db", cona=dbl_kind(1.0e6))
    _add("zfswin", "zfswin", "W/m^2", "internal ice PAR", "on bio interface grid", zfsw, "3Db")


def _idi():
    ncat = ice_history_shared.ncat_hist
    nzb = ice_history_shared.nzblyr
    source = jnp.asarray(ice_arrays_column.iDi)
    area = ice_state.aicen[:, :, :ncat]
    volume = ice_state.vicen[:, :, :ncat]
    work = jnp.zeros(ice_state.aice.shape + (nzb,), dtype=dbl_kind)
    for n in range(ncat):
        present = area[:, :, n] > c0
        for k in range(nzb - 1):
            work = work.at[:, :, k].add(jnp.where(
                present, source[:, :, k, n] * volume[:, :, n] ** 2
                / jnp.where(present, area[:, :, n], c1), c0))
        work = work.at[:, :, nzb - 1].set(jnp.where(
            present, work[:, :, nzb - 2], work[:, :, nzb - 1]))
    return work


def _iki():
    ncat = ice_history_shared.ncat_hist
    nblyr = int(ice_domain_size.nblyr)
    nzb = ice_history_shared.nzblyr
    source = jnp.asarray(ice_arrays_column.iki)
    area = ice_state.aicen[:, :, :ncat]
    work = jnp.zeros(ice_state.aice.shape + (nzb,), dtype=dbl_kind)
    for n in range(ncat):
        present = area[:, :, n] > c0
        for k in range(nzb - 1):
            work = work.at[:, :, k].add(jnp.where(present, source[:, :, k, n] * area[:, :, n], c0))
        work = work.at[:, :, nzb - 1].add(jnp.where(present, source[:, :, nblyr, n] * area[:, :, n], c0))
    return work


def _column_profile(nt, ocean):
    """Snow, ice, and ocean values on the aerosol/biology vertical grid."""
    nblyr = int(ice_domain_size.nblyr)
    nz = ice_history_shared.nzalyr
    puny = icepack_parameters.icepack_query_parameters("puny")["puny"]
    work = jnp.zeros(ice_state.aice.shape + (nz,), dtype=dbl_kind)
    if int(nt) < 0:
        return work
    present = ice_state.aice > puny
    # snow occupies the first two levels, ice the next nblyr+1, ocean the last
    work = work.at[:, :, 0:2].set(ice_state.trcr[:, :, int(nt) + nblyr + 1:int(nt) + nblyr + 3])
    work = work.at[:, :, 2:nblyr + 3].set(ice_state.trcr[:, :, int(nt):int(nt) + nblyr + 1])
    work = work.at[:, :, nblyr + 3].set(ocean)
    return jnp.where(present[..., None], work, c0)


def init_hist_bgc_3Da():
    """Register snow/ice/ocean biology profiles.

    Vertical tracers are written only when z_tracers is true.

    One field is registered for each active tracer species. Names follow
    the Fortran vname without the species suffix when only one species
    is written; a species index is appended when several share the flag.
    """
    if not bool(icepack_parameters.z_tracers):
        return
    if all(str(_F[name]).strip() == "x" for name in (
        "zaero", "bgc_N", "bgc_C", "bgc_chl", "bgc_DOC", "bgc_DIC", "bgc_DON",
        "bgc_Fed", "bgc_Fep", "bgc_Nit", "bgc_Am", "bgc_Sil", "bgc_hum",
        "bgc_DMSPd", "bgc_DMSPp", "bgc_DMS", "bgc_PON",
    )):
        return
    # Individual flags still decide which names are stored.
    _register_species("zaero", "kg/kg", "vertical aerosol", icepack_tracers.nt_zaero, "nlt_zaero", scale="density")
    _register_species("bgc_N", "mmol/m^3", "vertical algal nitrogen", icepack_tracers.nt_bgc_N, "nlt_bgc_N")
    _register_scalar("bgc_Nit", "mmol/m^3", "vertical nitrate", "nt_bgc_Nit", "nlt_bgc_Nit")
    _register_scalar("bgc_Am", "mmol/m^3", "vertical ammonium", "nt_bgc_Am", "nlt_bgc_Am")
    _register_scalar("bgc_Sil", "mmol/m^3", "vertical silicate", "nt_bgc_Sil", "nlt_bgc_Sil")
    _register_scalar("bgc_hum", "mmol/m^3", "vertical humic material", "nt_bgc_hum", "nlt_bgc_hum")
    _register_scalar("bgc_DMSPd", "mmol/m^3", "vertical dissolved DMSP", "nt_bgc_DMSPd", "nlt_bgc_DMSPd")
    _register_scalar("bgc_DMSPp", "mmol/m^3", "vertical particulate DMSP", "nt_bgc_DMSPp", "nlt_bgc_DMSPp")
    _register_scalar("bgc_DMS", "mmol/m^3", "vertical DMS", "nt_bgc_DMS", "nlt_bgc_DMS")
    _register_scalar("bgc_PON", "mmol/m^3", "vertical PON", "nt_bgc_PON", "nlt_bgc_PON")


def _ocean(index):
    column = jnp.asarray(ice_arrays_column.ocean_bio)
    if np.ndim(index) == 0:
        return column[:, :, int(index)]
    return column[:, :, int(index)]


def _register_scalar(name, unit, desc, nt_name, nlt_name):
    if str(_F[name]).strip() == "x":
        return
    if not hasattr(icepack_tracers, nt_name):
        return

    def gather(nt_name=nt_name, nlt_name=nlt_name):
        nt = int(getattr(icepack_tracers, nt_name))
        nlt = int(getattr(icepack_tracers, nlt_name))
        return _column_profile(nt, _ocean(nlt))

    _add(name, name, unit, desc, "snow, ice, and ocean", gather, "3Da")


def _register_species(name, unit, desc, nt_values, nlt_name, scale=None):
    if str(_F[name]).strip() == "x":
        return
    values = np.atleast_1d(np.asarray(nt_values))
    for species, nt in enumerate(values):
        vname = name if values.size == 1 else f"{name}{species + 1}"

        def gather(nt=int(nt), species=species, scale=scale, nlt_name=nlt_name):
            nlt_values = np.atleast_1d(np.asarray(getattr(icepack_tracers, nlt_name)))
            ocean = _ocean(nlt_values[species])
            profile = _column_profile(nt, ocean)
            if scale == "density":
                rhoi = icepack_parameters.rhoi
                rhos = icepack_parameters.rhos
                rhow = icepack_parameters.rhow
                profile = profile.at[:, :, 0:2].divide(rhos)
                profile = profile.at[:, :, 2:-1].divide(rhoi)
                profile = profile.at[:, :, -1].divide(rhow)
            return profile

        ice_history_shared.define_hist_field(
            vname, unit, desc, "snow, ice, and ocean", c1, c0, _F[name], gather, rank="3Da",
        )
