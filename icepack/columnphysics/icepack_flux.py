"""Flux operations shared by Icepack thermodynamics.

The standalone driver receives surface fluxes per unit ice area. The
NEMO-specific division by ice concentration is not used here.
"""

def merge_fluxes(aicen, category, total, tracer_flags, snwgrain, emissivity, flw):
    """Merge one category's fluxes into grid-cell means.

    ``category`` and ``total`` use the Fortran field names. Missing
    optional inputs retain the previous total, as with ``present`` in
    merge_fluxes from icepack_flux.F90.
    """
    total = dict(total)
    for source, target in (
        ("strairxn", "strairxT"), ("strairyn", "strairyT"),
        ("Cdn_atm_ratio_n", "Cdn_atm_ratio"),
        ("fsurfn", "fsurf"), ("fcondtopn", "fcondtop"),
        ("fcondbotn", "fcondbot"), ("fsensn", "fsens"),
        ("flatn", "flat"), ("fswabsn", "fswabs"),
        ("evapn", "evap"), ("evapsn", "evaps"), ("evapin", "evapi"),
        ("Trefn", "Tref"), ("Qrefn", "Qref"), ("Urefn", "Uref"),
        ("freshn", "fresh"), ("fsaltn", "fsalt"), ("fhocnn", "fhocn"),
        ("fswthrun", "fswthru"),
        ("fswthrun_vdr", "fswthru_vdr"), ("fswthrun_vdf", "fswthru_vdf"),
        ("fswthrun_idr", "fswthru_idr"), ("fswthrun_idf", "fswthru_idf"),
        ("fswthrun_uvrdr", "fswthru_uvrdr"),
        ("fswthrun_uvrdf", "fswthru_uvrdf"),
        ("fswthrun_pardr", "fswthru_pardr"),
        ("fswthrun_pardf", "fswthru_pardf"),
        ("melttn", "meltt"), ("meltbn", "meltb"),
        ("meltsn", "melts"), ("dsnown", "dsnow"),
        ("congeln", "congel"), ("snoicen", "snoice"),
    ):
        if source in category and target in total:
            total[target] = total[target] + category[source] * aicen

    # The upward longwave excludes reflected incoming radiation.
    if "flwoutn" in category and "flwout" in total:
        total["flwout"] = total["flwout"] + (
            category["flwoutn"] - (1 - emissivity) * flw
        ) * aicen

    if bool(tracer_flags["tr_iso"]):
        for source, target in (
            ("Qrefn_iso", "Qref_iso"),
            ("fiso_ocnn", "fiso_ocn"),
            ("fiso_evapn", "fiso_evap"),
        ):
            if source in category and target in total:
                total[target] = total[target] + category[source] * aicen[..., None]

    if snwgrain and "meltsliqn" in category and "meltsliq" in total:
        total["meltsliq"] = total["meltsliq"] + category["meltsliqn"] * aicen

    if bool(tracer_flags["tr_pond"]):
        for source, target in (
            ("dpnd_flushn", "dpnd_flush"),
            ("dpnd_exponn", "dpnd_expon"),
            ("dpnd_freebdn", "dpnd_freebd"),
            ("dpnd_initialn", "dpnd_initial"),
            ("dpnd_dlidn", "dpnd_dlid"),
        ):
            if source in category and target in total:
                total[target] = total[target] + category[source] * aicen
    return total


def set_sfcflux(aicen, flatn_f, fsensn_f, fsurfn_f, fcondtopn_f):
    """Set surface fluxes when Icepack does not calculate surface temperature.

    The standalone branch of set_sfcflux in icepack_flux.F90 has
    raicen=1; the four prescribed fluxes therefore pass through.
    """
    _ = aicen
    fsurfn = fsurfn_f
    fcondtopn = fcondtopn_f
    flatn = flatn_f
    fsensn = fsensn_f
    return flatn, fsensn, fsurfn, fcondtopn
