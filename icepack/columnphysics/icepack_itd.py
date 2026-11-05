"""Routines to initialize the ice thickness distribution and
utilities to redistribute ice among categories. These routines
are not specific to a particular numerical implementation.

See Bitz, C.M., and W.H. Lipscomb, 1999:
An energy-conserving thermodynamic model of sea ice,
J. Geophys. Res., 104, 15,669--15,677.

See Bitz, C.M., M.M. Holland, A.J. Weaver, M. Eby, 2001:
Simulating the ice-thickness distribution in a climate model,
J. Geophys. Res., 106, 2441--2464.

author for JAX version: Xinyuan Zhu
XZ: converted from F90 to Python

authors: C. M. Bitz, UW
         William H. Lipscomb and Elizabeth C. Hunke, LANL

2003: Vectorized by Clifford Chen (Fujitsu) and William Lipscomb

2004 WHL: Added multiple snow layers, block structure, cleanup_itd
2006 ECH: Added WMO standard ice thickness categories as kcatbound=2
          Streamlined for efficiency
          Converted to free source form (F90)
2014 ECH: Converted to column package

``hin_max`` uses the Fortran lower bound of 0, so index 0 is ``hin_max(0)``.

Ported: ``icepack_init_itd``, ``icepack_init_itd_hist``,
``icepack_aggregate`` and the ``cleanup_itd`` path used by restart
concentration adjustment. Its ``shift_ice`` transfers whole categories,
as ``rebin`` does. The time loop also needs partial transfers and must extend
that routine. ``icepack_compute_tracers`` lives in ``icepack_tracers``.
The time loop has JAX ``reduce_area``, ``column_sum`` and
``column_conservation_check`` paths.
"""

from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

from cicecore.shared import ice_fileunits
from icepack.columnphysics import icepack_parameters
from icepack.columnphysics import icepack_tracers
from icepack.columnphysics import icepack_therm_shared
from icepack.columnphysics.icepack_tracers import icepack_compute_tracers
from icepack.columnphysics.icepack_kinds import dbl_kind
from icepack.columnphysics.icepack_parameters import c0, c1, c2, c3, c15, c25, c100, p1, p5


# Set before stepping; passed explicitly in the cleanup JIT cache key.
CLEANUP_CATEGORY_UNROLL = 1
REBIN_STATIC_TRACERS = False


def column_sum(nsum, xin):
    """Sum the first ``nsum`` categories in Fortran's iteration order."""
    xout = jnp.zeros_like(xin[..., 0])
    for n in range(nsum):
        xout = xout + xin[..., n]
    return xout


def column_conservation_check(x1, x2, max_err):
    """Return CICE's per-cell conservation failure mask for host reporting."""
    return jnp.abs(x2 - x1) > max_err


def reduce_area(hin_max, aicen, vicen, aicen_init, vicen_init,
                dpnd_melt, trcrn, flags):
    """Reduce category-one ice area during melt, following CSM 1.0.

    Half of the ice volume loss changes thickness and the remainder
    changes fractional area. ``trcrn`` has tracer and category as its
    last two axes, matching the packed CICE state.
    """
    area_before = aicen
    hi0 = jnp.where(aicen_init > 0.0,
                    vicen_init / jnp.where(aicen_init > 0.0, aicen_init, 1.0), 0.0)
    hi1 = jnp.where(aicen > 0.0, vicen / jnp.where(aicen > 0.0, aicen, 1.0), 0.0)
    thin = (hi1 <= hin_max) & (hin_max > 0.0)
    aicen = jnp.where(thin, vicen / jnp.where(hin_max > 0.0, hin_max, 1.0), aicen)
    hi1 = jnp.where(thin, hin_max, hi1)
    melting = (aicen > 0.0) & (hi1 < hi0)
    hi1 = jnp.where(melting, vicen / jnp.where(aicen > 0.0, aicen, 1.0), hi1)
    aicen = jnp.where(melting,
                       2.0 * vicen / jnp.where(melting, hi1 + hi0, 1.0), aicen)
    da = area_before - aicen
    if bool(flags["tr_pond"]):
        dpnd_melt = (da * trcrn[..., int(flags["nt_apnd"]), 0]
                     * trcrn[..., int(flags["nt_hpnd"]), 0])
        if bool(flags["tr_pond_lvl"]):
            dpnd_melt = dpnd_melt * trcrn[..., int(flags["nt_alvl"]), 0]
    return aicen, vicen, dpnd_melt


def icepack_init_itd(hin_max):
    """Initialize area fraction and thickness boundaries for the itd model.

    authors: William H. Lipscomb and Elizabeth C. Hunke, LANL
             C. M. Bitz, UW

    hin_max: category limits (m), length ncat+1, index 0 is the lower limit.
             Written in place.
    """
    subname = "(icepack_init_itd)"
    ncat = int(icepack_tracers.ncat)
    if len(hin_max) < ncat + 1:
        raise ValueError(f"{subname} hin_max length {len(hin_max)}; need {ncat + 1}")

    # thinnest 3 categories combined
    wmo5 = np.array([0.30, 0.70, 1.20, 2.00, 999.0], dtype=dbl_kind)
    # thinnest 2 categories combined
    wmo6 = np.array([0.15, 0.30, 0.70, 1.20, 2.00, 999.0], dtype=dbl_kind)
    # echmod wmo6a
    # data wmo6 /0.30_dbl_kind, 0.70_dbl_kind,
    #            1.20_dbl_kind, 2.00_dbl_kind,
    #            4.56729_dbl_kind,
    #             999._dbl_kind /
    # all thickness categories
    wmo7 = np.array([0.10, 0.15, 0.30, 0.70, 1.20, 2.00, 999.0], dtype=dbl_kind)

    rncat = dbl_kind(ncat)
    d1 = dbl_kind(3.0) / rncat
    d2 = dbl_kind(0.5) / rncat
    b1 = p1          # asymptotic category width (m)
    b2 = c3          # thickness for which participation function is small (m)
    b3 = max(rncat * (rncat - c1), c2 * b2 / b1)
    kcatbound = int(icepack_parameters.kcatbound)
    kitd = int(icepack_parameters.kitd)
    hi_min = icepack_parameters.hi_min

    #-----------------------------------------------------------------
    # Choose category boundaries based on one of four options.
    #
    # The first formula (kcatbound = 0) was used in Lipscomb (2001)
    #  and in CICE versions 3.0 and 3.1.
    #
    # The second formula is more user-friendly in the sense that it
    #  is easy to obtain round numbers for category boundaries:
    #
    #    H(n) = n * [d1 + d2*(n-1)]
    #
    # Default values are d1 = 300/ncat, d2 = 50/ncat.
    # For ncat = 5, boundaries in cm are 60, 140, 240, 360, which are
    #  close to the standard values given by the first formula.
    # For ncat = 10, boundaries in cm are 30, 70, 120, 180, 250, 330,
    #  420, 520, 630.
    #
    # The third option provides support for World Meteorological
    #  Organization classification based on thickness.  The full
    #  WMO thickness distribution is used if ncat = 7;  if ncat=5
    #  or ncat = 6, some of the thinner categories are combined.
    # For ncat = 5,  boundaries are         30, 70, 120, 200, >200 cm.
    # For ncat = 6,  boundaries are     15, 30, 70, 120, 200, >200 cm.
    # For ncat = 7,  boundaries are 10, 15, 30, 70, 120, 200, >200 cm.
    #
    # The fourth formula asymptotes to a particular category width as
    # the number of categories increases, given by the parameter b1.
    # The parameter b3 is computed so that the category boundaries
    # are even numbers.
    #
    #    H(n) = b1 * [n + b3*n*(n+1)/(2*N*(N-1))] for N=ncat
    #
    # kcatbound=-1 is available only for 1-category runs, with
    # boundaries 0 and 100 m.
    #-----------------------------------------------------------------

    if kcatbound == -1:  # single category
        hin_max[0] = c0
        hin_max[1] = c100

    elif kcatbound == 0:  # original scheme
        if kitd == 1:
            # linear remapping itd category limits
            cc1 = c3 / rncat
            cc2 = c15 * cc1
            cc3 = c3
            hin_max[0] = c0  # minimum ice thickness, m
        else:
            # delta function itd category limits
            cc1 = max(dbl_kind(1.1) / rncat, hi_min)
            cc2 = c25 * cc1
            cc3 = dbl_kind(2.25)
            # hin_max(0) should not be zero
            # use some caution in making it less than 0.10
            hin_max[0] = hi_min  # minimum ice thickness, m

        for n in range(1, ncat + 1):
            x1 = dbl_kind(n - 1) / rncat
            hin_max[n] = hin_max[n - 1] + cc1 + cc2 * (c1 + np.tanh(cc3 * (x1 - c1)))

    elif kcatbound == 1:  # new scheme
        hin_max[0] = c0
        for n in range(1, ncat + 1):
            rn = dbl_kind(n)
            hin_max[n] = rn * (d1 + (rn - c1) * d2)

    elif kcatbound == 2:  # WMO standard
        if ncat == 5:
            hin_max[0] = c0
            hin_max[1:ncat + 1] = wmo5
        elif ncat == 6:
            hin_max[0] = c0
            hin_max[1:ncat + 1] = wmo6
        elif ncat == 7:
            hin_max[0] = c0
            hin_max[1:ncat + 1] = wmo7
        else:
            raise ValueError(
                subname + " kcatbound=2 (WMO) must have ncat=5, 6 or 7"
            )

    elif kcatbound == 3:  # asymptotic scheme
        hin_max[0] = c0
        for n in range(1, ncat + 1):
            rn = dbl_kind(n)
            hin_max[n] = b1 * (rn + b3 * rn * (rn + c1) / (c2 * rncat * (rncat - c1)))

    if kitd == 1:
        hin_max[ncat] = dbl_kind(999.9)  # arbitrary big number
    return hin_max


def icepack_init_itd_hist(hin_max, c_hi_range):
    """Initialize area fraction and thickness boundaries for the itd model.

    authors: William H. Lipscomb and Elizabeth C. Hunke, LANL
             C. M. Bitz, UW

    hin_max: category limits (m), index 0 is the lower limit.
    c_hi_range: string for history output, one entry per category. Written in place.

    Category-bound text goes to ``nu_diag``. The Fortran stored the same
    lines in ``icepack_warnings`` and flushed them there.
    """
    subname = "(icepack_init_itd_hist)"
    ncat = int(icepack_tracers.ncat)
    nu = ice_fileunits.nu_diag
    nu.write(" \n")
    nu.write(f" {subname}\n")
    nu.write(" hin_max(n-1) < Cat n < hin_max(n)\n")
    for n in range(1, ncat + 1):
        nu.write(f" {hin_max[n - 1]}  < Cat  {n}  <  {hin_max[n]}\n")
        # Write integer n to character string
        c_nc = f"{n:2d}"
        # Write hin_max to character string
        c_hinmax1 = f"{float(hin_max[n - 1]):7.3f}"
        c_hinmax2 = f"{float(hin_max[n]):7.3f}"
        # Save character string to write to history file
        c_hi_range[n - 1] = c_hinmax1 + "m < hi Cat " + c_nc + " < " + c_hinmax2 + "m"
    nu.write(" \n")
    nu.flush()
    return c_hi_range


def icepack_aggregate(
    aicen,
    trcrn,
    vicen,
    vsnon,
    trcr_depend,
    trcr_base,
    n_trcr_strata,
    nt_strata,
    Tf,
):
    """Aggregate ice state variables over thickness categories.

    authors: C. M. Bitz, UW
             W. H. Lipscomb, LANL

    One column. ``aicen``, ``vicen``, and ``vsnon`` have shape ``(ncat,)``.
    ``trcrn`` has shape ``(ntrcr, ncat)`` and is not modified.
    Tracer indices inside ``nt_strata`` are 0-based.
    Returns ``(aice, trcr, vice, vsno, aice0)``.
    """
    ncat = int(icepack_tracers.ncat)
    ntrcr = int(icepack_tracers.ntrcr)
    #-----------------------------------------------------------------
    # Initialize
    #-----------------------------------------------------------------
    aice0 = c1
    aice = c0
    vice = c0
    vsno = c0
    atrcr = np.zeros(ntrcr, dtype=dbl_kind)

    #-----------------------------------------------------------------
    # Aggregate
    #-----------------------------------------------------------------
    for n in range(ncat):
        aice = aice + aicen[n]
        vice = vice + vicen[n]
        vsno = vsno + vsnon[n]
        for it in range(ntrcr):
            atrcrn = trcrn[it, n] * (
                trcr_base[it, 0] * aicen[n]
                + trcr_base[it, 1] * vicen[n]
                + trcr_base[it, 2] * vsnon[n]
            )
            if int(n_trcr_strata[it]) > 0:  # additional tracer layers
                for itl in range(int(n_trcr_strata[it])):
                    ntr = int(nt_strata[it, itl])
                    atrcrn = atrcrn * trcrn[ntr, n]
            atrcr[it] = atrcr[it] + atrcrn

    # Open water fraction
    aice0 = max(c1 - aice, c0)

    # Tracers
    trcr = icepack_compute_tracers(
        trcr_depend,
        atrcr,
        aice,
        vice,
        vsno,
        trcr_base,
        n_trcr_strata,
        nt_strata,
        Tf,
    )
    return aice, trcr, vice, vsno, aice0


def icepack_aggregate_jax(aicen, trcrn, vicen, vsnon, Tf, layout, flags, options, parameters):
    """JAX aggregate of all thickness categories over the whole grid.

    Follows icepack_aggregate and icepack_compute_tracers in their Fortran
    loop order. The tracer layout is fixed at initialization; ice, snow,
    tracers and physical parameters remain differentiable device values.
    """
    return _icepack_aggregate_jax(
        aicen, trcrn, vicen, vsnon, Tf,
        tuple(sorted(layout.items())), tuple(sorted(flags.items())),
        tuple(sorted(options.items())), parameters)


@partial(jax.jit, static_argnames=("layout", "flags", "options"))
def _icepack_aggregate_jax(aicen, trcrn, vicen, vsnon, Tf,
                          layout, flags, options, parameters):
    # Both post-thermodynamics and post-dynamics updates reuse this kernel.
    layout = dict(layout)
    flags = dict(flags)
    options = dict(options)
    ncat = aicen.shape[-1]
    ntrcr = trcrn.shape[-2]
    aice = jnp.zeros_like(Tf)
    vice = jnp.zeros_like(Tf)
    vsno = jnp.zeros_like(Tf)
    atrcr = jnp.zeros_like(trcrn[..., 0])
    depend = layout["trcr_depend"]
    strata_n = layout["n_trcr_strata"]
    strata = layout["nt_strata"]
    base = layout["trcr_base"]

    # Initialize and aggregate category quantities and carried tracers.
    def accumulate_category(n, carry):
        aice, vice, vsno, atrcr = carry
        aice = aice + aicen[..., n]
        vice = vice + vicen[..., n]
        vsno = vsno + vsnon[..., n]
        for it in range(ntrcr):
            carried = trcrn[..., it, n] * (
                base[it][0] * aicen[..., n]
                + base[it][1] * vicen[..., n]
                + base[it][2] * vsnon[..., n]
            )
            for itl in range(strata_n[it]):
                carried = carried * trcrn[..., strata[it][itl], n]
            atrcr = atrcr.at[..., it].add(carried)
        return aice, vice, vsno, atrcr

    aice, vice, vsno, atrcr = lax.fori_loop(
        0, ncat, accumulate_category, (aice, vice, vsno, atrcr))

    aice0 = jnp.maximum(1 - aice, 0)
    trcr = jnp.zeros_like(atrcr)
    puny = parameters["puny"]
    for it in range(ntrcr):
        if depend[it] == 0:
            value = jnp.where(aice > puny, atrcr[..., it] / jnp.where(aice > puny, aice, 1), 0)
            if it == int(flags["nt_Tsfc"]):
                value = jnp.where(aice > puny, value, Tf)
        else:
            value = jnp.zeros_like(aice)
            for quantity in (base[it][0] * aice, base[it][1] * vice, base[it][2] * vsno):
                value = jnp.where(quantity > 0, atrcr[..., it] / jnp.where(quantity > 0, quantity, 1), value)
            for itl in range(strata_n[it]):
                carrier = trcr[..., strata[it][itl]]
                value = jnp.where(carrier > 0, value / jnp.where(carrier > 0, carrier, 1), 0)
        trcr = trcr.at[..., it].set(value)

    # Defaults for brine and snow tracers in ice-free or snow-free cells.
    if bool(flags["tr_brine"]):
        slot = int(flags["nt_fbri"])
        trcr = trcr.at[..., slot].set(jnp.where(vice <= 0, 1, trcr[..., slot]))
    if str(options["snwredist"])[:3] == "ITD":
        start = int(flags["nt_rhos"])
        stop = start + int(flags["nslyr"])
        trcr = trcr.at[..., start:stop].set(
            jnp.where(vsno[..., None] <= 0, parameters["rhosnew"], trcr[..., start:stop])
        )
    if bool(options["snwgrain"]):
        start = int(flags["nt_rsnw"])
        stop = start + int(flags["nslyr"])
        trcr = trcr.at[..., start:stop].set(
            jnp.where(vsno[..., None] <= 0, parameters["rsnw_fall"], trcr[..., start:stop])
        )
        start = int(flags["nt_smice"])
        stop = start + int(flags["nslyr"])
        trcr = trcr.at[..., start:stop].set(
            jnp.where(vsno[..., None] <= 0, parameters["rhos"], trcr[..., start:stop])
        )
    return aice, trcr, vice, vsno, aice0


def shift_ice_jax(aicen, trcrn, vicen, vsnon, hicen,
                  donor, daice, dvice, Tf, layout, flags, options, parameters,
                  static_tracers=False):
    """Transfer partial category area, ice volume, snow, and tracers."""
    ncat = aicen.shape[0]
    ntrcr = trcrn.shape[0]
    puny = parameters["puny"]
    initial_area = aicen
    initial_snow = vsnon
    base = jnp.asarray(layout["trcr_base"], dtype=trcrn.dtype)
    strata_count = layout["n_trcr_strata"]
    strata = layout["nt_strata"]
    depend = jnp.asarray(layout["trcr_depend"], dtype=jnp.int32)
    strata_indices = tuple(jnp.asarray([
        strata[it][level] if level < strata_count[it] else 0 for it in range(ntrcr)
    ], dtype=jnp.int32) for level in range(max(strata_count, default=0)))
    strata_active = tuple(jnp.asarray([
        level < count for count in strata_count
    ]) for level in range(len(strata_indices)))
    atrcrn = trcrn * (base[:, 0, None] * aicen[None, :]
                      + base[:, 1, None] * vicen[None, :]
                      + base[:, 2, None] * vsnon[None, :])
    for indices, active in zip(strata_indices, strata_active):
        atrcrn = atrcrn * jnp.where(active[:, None], trcrn[indices, :], 1.0)

    # Check every boundary against the state before any category transfer.
    def check_boundary(n, carry):
        daice, dvice, failed = carry
        donor_valid = donor[n] >= 0
        nd = jnp.maximum(donor[n], 0)
        area = aicen[nd]
        volume = vicen[nd]
        transfer_area = daice[n]
        transfer_volume = dvice[n]
        tiny_negative = ((transfer_area < 0.0)
                         & (transfer_area > -puny * area))
        daice_negative = ((transfer_area < 0.0)
                          & (transfer_area <= -puny * area))
        transfer_area = jnp.where(tiny_negative, 0.0, transfer_area)
        transfer_volume = jnp.where(tiny_negative, 0.0, transfer_volume)
        tiny_negative_volume = ((transfer_volume < 0.0)
                                & (transfer_volume > -puny * volume))
        dvice_negative = ((transfer_volume < 0.0)
                          & (transfer_volume <= -puny * volume))
        transfer_area = jnp.where(
            tiny_negative_volume, 0.0, transfer_area)
        transfer_volume = jnp.where(
            tiny_negative_volume, 0.0, transfer_volume)
        close_area = (transfer_area > area * (1.0 - puny))
        daice_greater_aicen = (close_area
                              & (transfer_area >= area * (1.0 + puny)))
        whole = close_area & (transfer_area < area * (1.0 + puny))
        transfer_area = jnp.where(whole, area, transfer_area)
        transfer_volume = jnp.where(whole, volume, transfer_volume)
        close_volume = (transfer_volume > volume * (1.0 - puny))
        dvice_greater_vicen = (close_volume
                              & (transfer_volume >= volume * (1.0 + puny)))
        whole = close_volume & (transfer_volume < volume * (1.0 + puny))
        transfer_area = jnp.where(whole, area, transfer_area)
        transfer_volume = jnp.where(whole, volume, transfer_volume)
        failed = failed | (donor_valid & (
                           (daice_negative & (transfer_area <= -puny * area))
                           | (dvice_negative & (transfer_volume <= -puny * volume))
                           | (daice_greater_aicen
                              & (transfer_area >= area * (1.0 + puny)))
                           | (dvice_greater_vicen
                              & (transfer_volume >= volume * (1.0 + puny)))))
        daice = daice.at[n].set(jnp.where(donor_valid, transfer_area, daice[n]))
        dvice = dvice.at[n].set(jnp.where(donor_valid, transfer_volume, dvice[n]))
        return daice, dvice, failed

    failed = jnp.asarray(False)
    if ncat > 1:
        daice, dvice, failed = lax.fori_loop(
            0, ncat - 1, check_boundary, (daice, dvice, failed))

    def transfer_boundary(n, carry):
        aicen, vicen, vsnon, atrcrn = carry
        donor_valid = donor[n] >= 0
        nd = jnp.maximum(donor[n], 0)
        nr = jnp.where(nd == n, n + 1, n)
        moving = donor_valid & (daice[n] > 0.0)
        area_change = jnp.where(moving, daice[n], 0.0)
        volume_change = jnp.where(moving, dvice[n], 0.0)
        aicen = aicen.at[nd].add(-area_change).at[nr].add(area_change)
        vicen = vicen.at[nd].add(-volume_change).at[nr].add(volume_change)
        worka = area_change / jnp.where(moving, initial_area[nd], 1.0)
        snow_change = jnp.where(moving, initial_snow[nd] * worka, 0.0)
        vsnon = vsnon.at[nd].add(-snow_change).at[nr].add(snow_change)
        carried = trcrn[:, nd] * (base[:, 0] * area_change
                                  + base[:, 1] * volume_change
                                  + base[:, 2] * snow_change)
        for indices, active in zip(strata_indices, strata_active):
            carried = carried * jnp.where(active, trcrn[indices, nd], 1.0)
        atrcrn = atrcrn.at[:, nd].add(-carried)
        atrcrn = atrcrn.at[:, nr].add(carried)
        return aicen, vicen, vsnon, atrcrn

    if ncat > 1:
        aicen, vicen, vsnon, atrcrn = lax.fori_loop(
            0, ncat - 1, transfer_boundary, (aicen, vicen, vsnon, atrcrn))
    present = aicen > puny
    hicen = jnp.where(present, vicen / jnp.where(present, aicen, 1.0), 0.0)

    def restore_category(area, volume, snow, carried):
        ice_present = area > puny

        def restore_tracer(it, updated):
            def area_tracer(_):
                default = jnp.where(it == int(flags["nt_Tsfc"]), Tf, 0.0)
                return jnp.where(ice_present, carried[it] /
                                 jnp.where(ice_present, area, 1.0), default)

            def weighted_tracer(_):
                value = jnp.zeros_like(Tf)
                for quantity in (
                    base[it, 0] * area,
                    base[it, 1] * volume,
                    base[it, 2] * snow,
                ):
                    value = jnp.where(
                        quantity > 0.0, carried[it] /
                        jnp.where(quantity > 0.0, quantity, 1.0), value)
                if static_tracers:
                    parents = tuple((strata[it][level], True)
                                    for level in range(strata_count[it]))
                else:
                    parents = tuple((indices[it], active[it])
                                    for indices, active in zip(strata_indices, strata_active))
                for parent, active in parents:
                    carrier = updated[parent]
                    divided = jnp.where(
                        carrier > 0.0, value /
                        jnp.where(carrier > 0.0, carrier, 1.0), 0.0)
                    value = jnp.where(active, divided, value)
                return value

            if static_tracers:
                value = (area_tracer(None) if layout["trcr_depend"][it] == 0
                         else weighted_tracer(None))
            else:
                value = lax.cond(depend[it] == 0, area_tracer, weighted_tracer, None)
            return updated.at[it].set(value)

        # Each tracer sees its recovered parents, in the original tracer order.
        updated = jnp.zeros_like(carried)
        if static_tracers:
            # Layout indices become constants at trace time, not device gathers.
            for it in range(ntrcr):
                updated = restore_tracer(it, updated)
        else:
            updated = lax.fori_loop(0, ntrcr, restore_tracer, updated)
        if bool(flags["tr_brine"]):
            updated = updated.at[int(flags["nt_fbri"])].set(jnp.where(
                volume > 0.0, updated[int(flags["nt_fbri"])], 1.0))
        if str(options["snwredist"])[:3] == "ITD":
            start = int(flags["nt_rhos"])
            count = int(options["nslyr"])
            updated = updated.at[start:start + count].set(jnp.where(
                snow > 0.0, updated[start:start + count],
                parameters["rhosnew"]))
        if bool(options["snwgrain"]):
            count = int(options["nslyr"])
            for tracer, default in (
                ("nt_rsnw", parameters["rsnw_fall"]),
                ("nt_smice", parameters["rhos"]),
            ):
                start = int(flags[tracer])
                updated = updated.at[start:start + count].set(jnp.where(
                    snow > 0.0, updated[start:start + count], default))
        return updated

    trcrn = jax.vmap(restore_category, in_axes=(0, 0, 0, 1), out_axes=1)(
        aicen, vicen, vsnon, atrcrn)
    return aicen, trcrn, vicen, vsnon, hicen, failed


def rebin_jax(aicen, trcrn, vicen, vsnon, hin_max, Tf,
               layout, flags, options, parameters, static_tracers=False):
    """Move whole categories whose mean thickness crosses their bounds."""
    ncat = aicen.shape[0]
    puny = parameters["puny"]
    present = aicen > puny
    hicen = jnp.where(
        present, vicen / jnp.where(present, aicen, 1.0), 0.0)
    thin_first = (present[0] & (hicen[0] <= hin_max[0])
                  & (hin_max[0] > 0.0))
    aicen = aicen.at[0].set(jnp.where(
        thin_first, vicen[0] / jnp.where(hin_max[0] > 0.0,
                                        hin_max[0], 1.0), aicen[0]))
    hicen = hicen.at[0].set(jnp.where(thin_first, hin_max[0], hicen[0]))
    if ncat == 1:
        return aicen, trcrn, vicen, vsnon, jnp.asarray(False)

    def move_category(index, state, upward):
        aicen, trcrn, vicen, vsnon, hicen, _ = state
        n = index if upward else ncat - 2 - index
        nd = n if upward else n + 1
        if upward:
            shifting = (aicen[nd] > puny) & (hicen[nd] > hin_max[n + 1])
        else:
            shifting = (aicen[nd] > puny) & (hicen[nd] <= hin_max[n + 1])
        donors = jnp.full((ncat,), -1, dtype=jnp.int32).at[n].set(nd)
        daice = jnp.zeros_like(aicen).at[n].set(aicen[nd])
        dvice = jnp.zeros_like(vicen).at[n].set(vicen[nd])

        def shift(current):
            aicen, trcrn, vicen, vsnon, hicen, failed = current
            aicen, trcrn, vicen, vsnon, hicen, error = shift_ice_jax(
                aicen, trcrn, vicen, vsnon, hicen,
                donors, daice, dvice, Tf,
                layout, flags, options, parameters, static_tracers=static_tracers)
            return aicen, trcrn, vicen, vsnon, hicen, failed | error

        return lax.cond(shifting, shift, lambda current: current, state)

    # Transfers change the next donor. Preserve the forward and reverse sweeps.
    state = (aicen, trcrn, vicen, vsnon, hicen, jnp.asarray(False))
    state = lax.fori_loop(0, ncat - 1, lambda n, current: move_category(n, current, True), state)
    aicen, trcrn, vicen, vsnon, hicen, failed = lax.fori_loop(
        0, ncat - 1, lambda n, current: move_category(n, current, False), state)
    return aicen, trcrn, vicen, vsnon, failed


def cleanup_itd_jax(dt, hin_max, aicen, trcrn, vicen, vsnon,
                    aice0, aice, first_ice, fpond, fresh, fsalt,
                    fhocn, faero_ocn, fiso_ocn, flux_bio, Tf,
                    layout, flags, options, parameters,
                    bio_index=(), limit_aice=True, dorebin=True,
                    diagnose=False):
    """Use a stable compiled cleanup for both thermodynamics and ridging."""
    # Only selectors used by cleanup/shift belong in the cache key. Therm2
    # supplies extra options, and requests error details from the same kernel.
    flag_names = (
        "tr_pond_topo", "nt_apnd", "nt_hpnd", "tr_aero", "n_aero", "nt_aero",
        "tr_iso", "n_iso", "nt_isoice", "nt_isosno", "nbtrcr", "tr_brine",
        "nt_fbri", "nt_qice", "nt_sice", "nt_Tsfc", "nt_qsno", "nt_rhos",
        "nt_smice", "nt_rsnw",
    )
    option_names = (
        "nilyr", "nslyr", "nblyr", "snwredist", "snwgrain",
        "saltflux_option", "skl_bgc", "z_tracers",
    )
    result = _cleanup_itd_jax(
        dt, hin_max, aicen, trcrn, vicen, vsnon, aice0, aice,
        first_ice, fpond, fresh, fsalt, fhocn, faero_ocn, fiso_ocn, flux_bio, Tf,
        tuple(sorted(layout.items())),
        tuple((name, flags[name]) for name in flag_names if name in flags),
        tuple((name, options[name]) for name in option_names if name in options),
        parameters, tuple(bio_index), limit_aice, dorebin,
        min(options.get("cleanup_category_unroll", CLEANUP_CATEGORY_UNROLL), aicen.shape[0]),
        options.get("rebin_static_tracers", REBIN_STATIC_TRACERS))
    return result if diagnose else result[:-1]


@partial(jax.jit, static_argnames=(
    "layout", "flags", "options", "bio_index", "limit_aice", "dorebin",
    "category_unroll", "static_tracers"))
def _cleanup_itd_jax(dt, hin_max, aicen, trcrn, vicen, vsnon,
                     aice0, aice, first_ice, fpond, fresh, fsalt,
                     fhocn, faero_ocn, fiso_ocn, flux_bio, Tf,
                     layout, flags, options, parameters,
                     bio_index=(), limit_aice=True, dorebin=True,
                     category_unroll=1, static_tracers=False):
    """Rebin and remove residual categories, returning conserved fluxes."""
    layout = dict(layout)
    flags = dict(flags)
    options = dict(options)
    ncat = aicen.shape[0]
    nilyr = int(options["nilyr"])
    nslyr = int(options["nslyr"])
    nblyr = int(options["nblyr"])
    puny = parameters["puny"]
    aice = column_sum(ncat, aicen)
    aice0 = jnp.maximum(1.0 - aice, 0.0)
    initial_area_failed = limit_aice & ((aice > 1.0 + puny) | (aice < -puny))
    failed = initial_area_failed
    rebin_failed = jnp.asarray(False)
    if dorebin:
        def do_rebin(state):
            aicen, trcrn, vicen, vsnon, _ = state
            aicen, trcrn, vicen, vsnon, error = rebin_jax(
                aicen, trcrn, vicen, vsnon, hin_max, Tf,
                layout, flags, options, parameters, static_tracers=static_tracers)
            return aicen, trcrn, vicen, vsnon, error
        aicen, trcrn, vicen, vsnon, rebin_failed = lax.cond(
            aice > puny, do_rebin, lambda state: state,
            (aicen, trcrn, vicen, vsnon, rebin_failed))
        failed = failed | rebin_failed
    # Fortran carries the pre-rebin totals into zap_small_areas.
    negative_area_failed = jnp.asarray(False)
    excess_area_failed = jnp.asarray(False)
    if limit_aice:
        residual = ((aice < jnp.maximum(parameters["itd_area_min"], puny))
                    | (aice * parameters["rhoi"]
                       < jnp.maximum(parameters["itd_mass_min"], puny)))

        def remove_small_category(n, carry):
            (aicen, trcrn, vicen, vsnon, aice0, first_ice,
             fpond, fresh, fsalt, fhocn, faero_ocn, fiso_ocn, flux_bio,
             negative_area_failed, failed) = carry
            tiny = ((jnp.abs(aicen[n]) <= puny)
                    & ((aicen[n] != 0.0) | (vicen[n] != 0.0)
                       | (vsnon[n] != 0.0)))
            zap = tiny | residual
            negative_area_failed = negative_area_failed | (aicen[n] < -puny)
            failed = failed | negative_area_failed
            if bool(flags["tr_pond_topo"]):
                fpond = fpond - jnp.where(
                    zap, aicen[n] * trcrn[int(flags["nt_apnd"]), n]
                    * trcrn[int(flags["nt_hpnd"]), n], 0.0)
            if bool(flags["tr_aero"]):
                for it in range(int(flags["n_aero"])):
                    slot = int(flags["nt_aero"]) + 4 * it + 2
                    faero_ocn = faero_ocn.at[it].add(jnp.where(
                        zap, vicen[n] * (trcrn[slot, n]
                                         + trcrn[slot + 1, n]) / dt, 0.0))
            if bool(flags["tr_iso"]):
                for it in range(int(flags["n_iso"])):
                    slot = int(flags["nt_isoice"]) + it
                    fiso_ocn = fiso_ocn.at[it].add(jnp.where(
                        zap, vicen[n] * trcrn[slot, n] / dt, 0.0))
            if bool(options["skl_bgc"]):
                for it in range(int(flags["nbtrcr"])):
                    flux_bio = flux_bio.at[it].add(jnp.where(
                        zap, trcrn[int(bio_index[it]), n] * aicen[n]
                        * parameters["sk_l"] / dt, 0.0))
            elif bool(options["z_tracers"]):
                brine_volume = (vicen[n] / nblyr
                                * trcrn[int(flags["nt_fbri"]), n])
                for it in range(int(flags["nbtrcr"])):
                    for k in range(nblyr + 1):
                        factor = 0.5 if k in (0, nblyr) else 1.0
                        flux_bio = flux_bio.at[it].add(jnp.where(
                            zap, trcrn[int(bio_index[it]) + k, n]
                            * brine_volume * factor / dt, 0.0))
            for k in range(nilyr):
                slot = int(flags["nt_qice"]) + k
                fhocn = fhocn + jnp.where(
                    zap, trcrn[slot, n] * vicen[n] / (nilyr * dt), 0.0)
                trcrn = trcrn.at[slot, n].set(jnp.where(
                    zap, 0.0, trcrn[slot, n]))
            fresh = fresh + jnp.where(
                zap, parameters["rhoi"] * vicen[n] / dt, 0.0)
            if options["saltflux_option"] == "prognostic":
                salinity = jnp.zeros_like(fsalt)
                for k in range(nilyr):
                    salinity = (salinity + trcrn[
                        int(flags["nt_sice"]) + k, n] / nilyr)
            else:
                salinity = parameters["ice_ref_salinity"]
            fsalt = fsalt + jnp.where(
                zap, parameters["rhoi"] * vicen[n]
                * salinity * parameters["p001"] / dt, 0.0)
            aice0 = aice0 + jnp.where(zap, aicen[n], 0.0)
            aicen = aicen.at[n].set(jnp.where(zap, 0.0, aicen[n]))
            vicen = vicen.at[n].set(jnp.where(zap, 0.0, vicen[n]))
            trcrn = trcrn.at[int(flags["nt_Tsfc"]), n].set(jnp.where(
                zap, Tf, trcrn[int(flags["nt_Tsfc"]), n]))
            if bool(flags["tr_aero"]):
                for it in range(int(flags["n_aero"])):
                    slot = int(flags["nt_aero"]) + 4 * it
                    faero_ocn = faero_ocn.at[it].add(jnp.where(
                        zap, vsnon[n] * (trcrn[slot, n]
                                         + trcrn[slot + 1, n]) / dt, 0.0))
            if bool(flags["tr_iso"]):
                for it in range(int(flags["n_iso"])):
                    slot = int(flags["nt_isosno"]) + it
                    fiso_ocn = fiso_ocn.at[it].add(jnp.where(
                        zap, vsnon[n] * trcrn[slot, n] / dt, 0.0))
            if bool(options["z_tracers"]):
                snow_surface = 0.5 * vsnon[n] / nslyr
                snow_interior = vsnon[n] - snow_surface
                for it in range(int(flags["nbtrcr"])):
                    slot = int(bio_index[it]) + nblyr + 1
                    amount = (trcrn[slot, n] * snow_surface
                              + trcrn[slot + 1, n] * snow_interior)
                    flux_bio = flux_bio.at[it].add(jnp.where(
                        zap, amount / dt, 0.0))
            for k in range(nslyr):
                slot = int(flags["nt_qsno"]) + k
                fhocn = fhocn + jnp.where(
                    zap, trcrn[slot, n] * vsnon[n] / (nslyr * dt), 0.0)
                trcrn = trcrn.at[slot, n].set(jnp.where(
                    zap, 0.0, trcrn[slot, n]))
            fresh = fresh + jnp.where(
                zap, parameters["rhos"] * vsnon[n] / dt, 0.0)
            vsnon = vsnon.at[n].set(jnp.where(zap, 0.0, vsnon[n]))
            trcrn = trcrn.at[1:, n].set(jnp.where(zap, 0.0, trcrn[1:, n]))
            if bool(flags["tr_brine"]):
                slot = int(flags["nt_fbri"])
                trcrn = trcrn.at[slot, n].set(jnp.where(
                    zap, 1.0, trcrn[slot, n]))
            if str(options["snwredist"])[:3] == "ITD":
                start = int(flags["nt_rhos"])
                trcrn = trcrn.at[start:start + nslyr, n].set(jnp.where(
                    zap, parameters["rhosnew"],
                    trcrn[start:start + nslyr, n]))
            if bool(options["snwgrain"]):
                for tracer, default in (
                    ("nt_smice", parameters["rhos"]),
                    ("nt_rsnw", parameters["rsnw_fall"]),
                ):
                    start = int(flags[tracer])
                    trcrn = trcrn.at[start:start + nslyr, n].set(jnp.where(
                        zap, default, trcrn[start:start + nslyr, n]))
            first_ice = first_ice.at[n].set(jnp.where(
                zap, True, first_ice[n]))
            return (aicen, trcrn, vicen, vsnon, aice0, first_ice,
                    fpond, fresh, fsalt, fhocn, faero_ocn, fiso_ocn, flux_bio,
                    negative_area_failed, failed)

        (aicen, trcrn, vicen, vsnon, aice0, first_ice,
         fpond, fresh, fsalt, fhocn, faero_ocn, fiso_ocn, flux_bio,
         negative_area_failed, failed) = lax.fori_loop(
            0, ncat, remove_small_category,
            (aicen, trcrn, vicen, vsnon, aice0, first_ice,
             fpond, fresh, fsalt, fhocn, faero_ocn, fiso_ocn, flux_bio,
             negative_area_failed, failed), unroll=category_unroll)

        excess_area_failed = aice > 1.0 + puny
        failed = failed | excess_area_failed
        excess = (aice > 1.0) & (aice < 1.0 + puny)
        fraction = jnp.where(excess, (aice - 1.0) /
                             jnp.where(aice > 0.0, aice, 1.0), 0.0)

        def correct_excess_category(n, carry):
            (aicen, vicen, vsnon, fpond, fresh, fsalt, fhocn,
             faero_ocn, fiso_ocn, flux_bio) = carry
            if bool(flags["tr_pond_topo"]):
                fpond = fpond - (aicen[n] * trcrn[int(flags["nt_apnd"]), n]
                                 * trcrn[int(flags["nt_hpnd"]), n]
                                 * fraction)
            if bool(flags["tr_aero"]):
                for it in range(int(flags["n_aero"])):
                    slot = int(flags["nt_aero"]) + 4 * it
                    amount = (vsnon[n] * (trcrn[slot, n]
                                          + trcrn[slot + 1, n])
                              + vicen[n] * (trcrn[slot + 2, n]
                                            + trcrn[slot + 3, n]))
                    faero_ocn = faero_ocn.at[it].add(amount * fraction / dt)
            if bool(flags["tr_iso"]):
                for it in range(int(flags["n_iso"])):
                    amount = (vsnon[n] * trcrn[int(flags["nt_isosno"]) + it, n]
                              + vicen[n] * trcrn[int(flags["nt_isoice"]) + it, n])
                    fiso_ocn = fiso_ocn.at[it].add(amount * fraction / dt)
            if bool(options["skl_bgc"]):
                for it in range(int(flags["nbtrcr"])):
                    flux_bio = flux_bio.at[it].add(
                        trcrn[int(bio_index[it]), n] * fraction
                        * parameters["sk_l"] / dt)
            elif bool(options["z_tracers"]):
                brine_volume = (fraction * vicen[n] / nblyr
                                * trcrn[int(flags["nt_fbri"]), n])
                snow_surface = 0.5 * vsnon[n] / nslyr
                snow_interior = vsnon[n] - snow_surface
                for it in range(int(flags["nbtrcr"])):
                    for k in range(nblyr + 1):
                        factor = 0.5 if k in (0, nblyr) else 1.0
                        flux_bio = flux_bio.at[it].add(
                            trcrn[int(bio_index[it]) + k, n]
                            * brine_volume * factor / dt)
                    slot = int(bio_index[it]) + nblyr + 1
                    flux_bio = flux_bio.at[it].add(
                        (trcrn[slot, n] * snow_surface
                         + trcrn[slot + 1, n] * snow_interior)
                        * fraction / dt)
            for k in range(nilyr):
                fhocn = (fhocn + trcrn[int(flags["nt_qice"]) + k, n]
                         * vicen[n] * fraction / (nilyr * dt))
            for k in range(nslyr):
                fhocn = (fhocn + trcrn[int(flags["nt_qsno"]) + k, n]
                         * vsnon[n] * fraction / (nslyr * dt))
            fresh = (fresh + (parameters["rhoi"] * vicen[n]
                              + parameters["rhos"] * vsnon[n])
                     * fraction / dt)
            if options["saltflux_option"] == "prognostic":
                salinity = jnp.zeros_like(fsalt)
                for k in range(nilyr):
                    salinity = salinity + trcrn[
                        int(flags["nt_sice"]) + k, n] / nilyr
            else:
                salinity = parameters["ice_ref_salinity"]
            fsalt = (fsalt + parameters["rhoi"] * vicen[n] * salinity
                     * parameters["p001"] * fraction / dt)
            aicen = aicen.at[n].set(aicen[n] * (1.0 - fraction))
            vicen = vicen.at[n].set(vicen[n] * (1.0 - fraction))
            vsnon = vsnon.at[n].set(vsnon[n] * (1.0 - fraction))
            return (aicen, vicen, vsnon, fpond, fresh, fsalt, fhocn,
                    faero_ocn, fiso_ocn, flux_bio)

        (aicen, vicen, vsnon, fpond, fresh, fsalt, fhocn,
         faero_ocn, fiso_ocn, flux_bio) = lax.fori_loop(
            0, ncat, correct_excess_category,
            (aicen, vicen, vsnon, fpond, fresh, fsalt, fhocn,
             faero_ocn, fiso_ocn, flux_bio), unroll=category_unroll)
        # Keep the category-removal increments in aice0, as in Fortran.
        aice = jnp.where(excess, 1.0, aice)
        aice0 = jnp.where(excess, 0.0, aice0)

    # CICE also removes an entire snow column if any snow layer has an
    # impossible temperature. The removed snow enters the ocean budgets.
    def remove_invalid_snow(n, carry):
        (trcrn, vsnon, fresh, fhocn, faero_ocn, fiso_ocn, flux_bio) = carry
        area_present = aicen[n] > puny
        snow_depth = vsnon[n] / jnp.where(area_present, aicen[n], 1.0)
        zap_snow = jnp.asarray(False)
        for k in range(nslyr):
            enthalpy = trcrn[int(flags["nt_qsno"]) + k, n]
            deep_snow = snow_depth > parameters["hs_min"]
            checked_enthalpy = jnp.where(
                deep_snow, enthalpy,
                -parameters["rhos"] * parameters["Lfresh"])
            maximum = jnp.where(
                deep_snow,
                -checked_enthalpy * puny * nslyr /
                jnp.where(vsnon[n] > 0.0,
                          parameters["rhos"] * parameters["cp_ice"]
                          * vsnon[n], 1.0), puny)
            temperature = (parameters["Lfresh"] + checked_enthalpy
                           / parameters["rhos"]) / parameters["cp_ice"]
            zap_snow = zap_snow | (area_present
                                  & ((temperature < parameters["Tmin"])
                                     | (temperature > maximum)))
        if bool(flags["tr_aero"]):
            for it in range(int(flags["n_aero"])):
                slot = int(flags["nt_aero"]) + 4 * it
                faero_ocn = faero_ocn.at[it].add(jnp.where(
                    zap_snow, vsnon[n] * (trcrn[slot, n]
                                           + trcrn[slot + 1, n]) / dt, 0.0))
        if bool(flags["tr_iso"]):
            for it in range(int(flags["n_iso"])):
                slot = int(flags["nt_isosno"]) + it
                fiso_ocn = fiso_ocn.at[it].add(jnp.where(
                    zap_snow, vsnon[n] * trcrn[slot, n] / dt, 0.0))
        if bool(options["z_tracers"]):
            snow_surface = 0.5 * vsnon[n] / nslyr
            snow_interior = vsnon[n] - snow_surface
            for it in range(int(flags["nbtrcr"])):
                slot = int(bio_index[it]) + nblyr + 1
                amount = (trcrn[slot, n] * snow_surface
                          + trcrn[slot + 1, n] * snow_interior)
                flux_bio = flux_bio.at[it].add(jnp.where(
                    zap_snow, amount / dt, 0.0))
        for k in range(nslyr):
            slot = int(flags["nt_qsno"]) + k
            fhocn = fhocn + jnp.where(
                zap_snow, trcrn[slot, n] * vsnon[n] / (nslyr * dt), 0.0)
            trcrn = trcrn.at[slot, n].set(jnp.where(
                zap_snow, 0.0, trcrn[slot, n]))
        fresh = fresh + jnp.where(
            zap_snow, parameters["rhos"] * vsnon[n] / dt, 0.0)
        vsnon = vsnon.at[n].set(jnp.where(zap_snow, 0.0, vsnon[n]))
        return (trcrn, vsnon, fresh, fhocn, faero_ocn, fiso_ocn, flux_bio)

    (trcrn, vsnon, fresh, fhocn, faero_ocn, fiso_ocn, flux_bio) = lax.fori_loop(
        0, ncat, remove_invalid_snow,
        (trcrn, vsnon, fresh, fhocn, faero_ocn, fiso_ocn, flux_bio),
        unroll=category_unroll)
    result = (aicen, trcrn, vicen, vsnon, aice0, aice,
              first_ice, fpond, fresh, fsalt, fhocn,
              faero_ocn, fiso_ocn, flux_bio, failed)
    return result + ((initial_area_failed, rebin_failed,
                      negative_area_failed, excess_area_failed),)


def aggregate_area(aicen):
    """Compute total ice area and open-water fraction."""
    aice = c0
    for n in range(int(icepack_tracers.ncat)):
        aice += aicen[n]
    return aice, max(c1 - aice, c0)


def shift_ice(aicen, trcrn, vicen, vsnon, donor, receiver,
              trcr_depend, trcr_base, n_trcr_strata, nt_strata, Tf):
    """Move one entire category across a thickness boundary.

    ``rebin`` is the only caller and always transfers the donor's full area
    and volume. The tracer sums follow the Fortran ``shift_ice`` routine.
    """
    ntrcr = int(icepack_tracers.ntrcr)
    ncat = int(icepack_tracers.ncat)
    atrcrn = np.zeros((ntrcr, ncat), dtype=dbl_kind)
    for n in range(ncat):
        for it in range(ntrcr):
            atrcrn[it, n] = trcrn[it, n] * (
                trcr_base[it, 0] * aicen[n]
                + trcr_base[it, 1] * vicen[n]
                + trcr_base[it, 2] * vsnon[n]
            )
            for itl in range(int(n_trcr_strata[it])):
                atrcrn[it, n] *= trcrn[int(nt_strata[it, itl]), n]

    atrcrn[:, receiver] += atrcrn[:, donor]
    atrcrn[:, donor] = c0
    aicen[receiver] += aicen[donor]
    vicen[receiver] += vicen[donor]
    vsnon[receiver] += vsnon[donor]
    aicen[donor] = c0
    vicen[donor] = c0
    vsnon[donor] = c0
    for n in range(ncat):
        trcrn[:, n] = icepack_compute_tracers(
            trcr_depend, atrcrn[:, n],
            aicen[n], vicen[n], vsnon[n], trcr_base, n_trcr_strata, nt_strata, Tf,
        )


def rebin(aicen, trcrn, vicen, vsnon, hin_max, Tf,
          trcr_depend, trcr_base, n_trcr_strata, nt_strata):
    """Shift ice whose mean category thickness lies outside its bounds."""
    ncat = int(icepack_tracers.ncat)
    puny = icepack_parameters.puny
    hicen = np.zeros(ncat, dtype=dbl_kind)
    for n in range(ncat):
        if aicen[n] > puny:
            hicen[n] = vicen[n] / aicen[n]
    # Make sure thickness of the first category is at least hin_max(0).
    if aicen[0] > puny and hicen[0] <= hin_max[0] and hin_max[0] > c0:
        aicen[0] = vicen[0] / hin_max[0]
        hicen[0] = hin_max[0]

    # Move categories whose ice is too thick upward.
    for n in range(ncat - 1):
        if aicen[n] > puny and hicen[n] > hin_max[n + 1]:
            shift_ice(aicen, trcrn, vicen, vsnon, n, n + 1,
                      trcr_depend, trcr_base, n_trcr_strata, nt_strata, Tf)
            hicen[n] = c0
            hicen[n + 1] = vicen[n + 1] / aicen[n + 1]

    # Move categories whose ice is too thin downward.
    for n in range(ncat - 2, -1, -1):
        if aicen[n + 1] > puny and hicen[n + 1] <= hin_max[n + 1]:
            shift_ice(aicen, trcrn, vicen, vsnon, n + 1, n,
                      trcr_depend, trcr_base, n_trcr_strata, nt_strata, Tf)
            hicen[n + 1] = c0
            hicen[n] = vicen[n] / aicen[n]


def zap_snow(dt, trcrn, vsnon, dfresh, dfhocn,
             dfaero_ocn, tr_aero, dfiso_ocn, dflux_bio, aicen):
    """Melt one category's snow and transfer conserved quantities to ocean."""
    t = icepack_tracers
    p = icepack_parameters
    nslyr = int(t.nslyr)
    nblyr = int(t.nblyr)
    if tr_aero:
        for it in range(int(t.n_aero)):
            base = int(t.nt_aero) + 4 * it
            dfaero_ocn[it] += vsnon * (trcrn[base] + trcrn[base + 1]) / dt
    if bool(t.tr_iso):
        for it in range(int(t.n_iso)):
            dfiso_ocn[it] += vsnon * trcrn[int(t.nt_isosno) + it] / dt
    if bool(p.z_tracers):
        dvssl = p5 * vsnon / nslyr  # snow surface layer
        dvint = vsnon - dvssl       # snow interior
        for it in range(int(t.nbtrcr)):
            base = int(t.bio_index[it]) + nblyr + 1
            dflux_bio[it] += (trcrn[base] * dvssl + trcrn[base + 1] * dvint) / dt
    # Snow enthalpy and snow volume.
    for k in range(nslyr):
        slot = int(t.nt_qsno) + k
        dfhocn += trcrn[slot] * vsnon / nslyr / dt
        trcrn[slot] = c0
    dfresh += p.rhos * vsnon / dt
    return c0, dfresh, dfhocn


def zap_small_areas(dt, aice, aice0, aicen, trcrn, vicen, vsnon,
                    tr_aero, tr_pond_topo, first_ice, Tf,
                    dfaero_ocn, dfiso_ocn, dflux_bio):
    """Remove residual ice and account for its ocean fluxes."""
    t = icepack_tracers
    p = icepack_parameters
    ncat = int(t.ncat)
    nilyr = int(t.nilyr)
    nslyr = int(t.nslyr)
    nblyr = int(t.nblyr)
    puny = p.puny
    dfpond = dfresh = dfsalt = dfhocn = c0
    zap_category = np.zeros(ncat, dtype=bool)
    for n in range(ncat):
        if abs(aicen[n]) <= puny and (
            abs(aicen[n]) != c0 or abs(vicen[n]) != c0 or abs(vsnon[n]) != c0
        ):
            zap_category[n] = True
    zap_residual = (
        aice < max(p.itd_area_min, puny)
        or aice * p.rhoi < max(p.itd_mass_min, puny)
    )

    # I. Zap categories with very small areas.
    for n in range(ncat):
        if aicen[n] < -puny:
            raise RuntimeError("(zap_small_areas) Zap ice: negative ice area")
        if not (zap_category[n] or zap_residual):
            continue
        if tr_pond_topo:
            dfpond -= (aicen[n] * trcrn[int(t.nt_apnd), n]
                       * trcrn[int(t.nt_hpnd), n])
        if tr_aero:
            for it in range(int(t.n_aero)):
                base = int(t.nt_aero) + 4 * it
                dfaero_ocn[it] += vicen[n] * (
                    trcrn[base + 2, n] + trcrn[base + 3, n]
                ) / dt
        if bool(t.tr_iso):
            for it in range(int(t.n_iso)):
                dfiso_ocn[it] += vicen[n] * trcrn[int(t.nt_isoice) + it, n] / dt
        if bool(p.skl_bgc) and int(t.nbtrcr) > 0:
            for it in range(int(t.nbtrcr)):
                dflux_bio[it] += (trcrn[int(t.bio_index[it]), n]
                                  * aicen[n] * p.sk_l / dt)
        elif bool(p.z_tracers) and int(t.nbtrcr) > 0:
            bvol = vicen[n] / nblyr * trcrn[int(t.nt_fbri), n]
            for it in range(int(t.nbtrcr)):
                base = int(t.bio_index[it])
                for k in range(nblyr + 1):
                    factor = p5 if k == 0 or k == nblyr else c1
                    dflux_bio[it] += trcrn[base + k, n] * bvol * factor / dt

        # Zap ice energy; use ocean heat to melt ice.
        for k in range(nilyr):
            slot = int(t.nt_qice) + k
            dfhocn += trcrn[slot, n] * vicen[n] / nilyr / dt
            trcrn[slot, n] = c0
        # Add ice water and salt to the ocean.
        dfresh += p.rhoi * vicen[n] / dt
        if str(p.saltflux_option).strip() == "prognostic":
            sicen = c0
            for k in range(nilyr):
                sicen += trcrn[int(t.nt_sice) + k, n] / nilyr
        else:
            sicen = p.ice_ref_salinity
        dfsalt += p.rhoi * vicen[n] * sicen * 0.001 / dt
        aice0 += aicen[n]
        aicen[n] = c0
        vicen[n] = c0
        trcrn[int(t.nt_Tsfc), n] = Tf
        vsnon[n], dfresh, dfhocn = zap_snow(
            dt, trcrn[:, n], vsnon[n], dfresh, dfhocn, dfaero_ocn,
            tr_aero, dfiso_ocn, dflux_bio, aicen[n],
        )
        # The first tracer is Tsfc; reset all remaining tracers.
        if int(t.ntrcr) >= 2:
            trcrn[1:int(t.ntrcr), n] = c0
        if bool(t.tr_brine):
            trcrn[int(t.nt_fbri), n] = c1
        if str(p.snwredist)[:3] == "ITD":
            trcrn[int(t.nt_rhos):int(t.nt_rhos) + nslyr, n] = p.rhosnew
        if bool(p.snwgrain):
            trcrn[int(t.nt_smice):int(t.nt_smice) + nslyr, n] = p.rhos
            trcrn[int(t.nt_rsnw):int(t.nt_rsnw) + nslyr, n] = p.rsnw_fall
        first_ice[n] = True

    # II. Roundoff excess area: remove the same fraction from each category.
    if aice > c1 + puny:
        raise RuntimeError("(zap_small_areas) Zap ice: excess ice area")
    if aice > c1:
        fraction = (aice - c1) / aice
        for n in range(ncat):
            if tr_pond_topo:
                dfpond -= (aicen[n] * trcrn[int(t.nt_apnd), n]
                           * trcrn[int(t.nt_hpnd), n] * fraction)
            if tr_aero:
                for it in range(int(t.n_aero)):
                    base = int(t.nt_aero) + 4 * it
                    dfaero_ocn[it] += (
                        vsnon[n] * (trcrn[base, n] + trcrn[base + 1, n])
                        + vicen[n] * (trcrn[base + 2, n] + trcrn[base + 3, n])
                    ) * fraction / dt
            if bool(p.skl_bgc) and int(t.nbtrcr) > 0:
                for it in range(int(t.nbtrcr)):
                    dflux_bio[it] += (trcrn[int(t.bio_index[it]), n]
                                      * p.sk_l * fraction / dt)
            elif bool(p.z_tracers) and int(t.nbtrcr) > 0:
                bvol = fraction * vicen[n] / nblyr * trcrn[int(t.nt_fbri), n]
                dvssl = p5 * vsnon[n] / nslyr
                dvint = vsnon[n] - dvssl
                for it in range(int(t.nbtrcr)):
                    base = int(t.bio_index[it])
                    for k in range(nblyr + 1):
                        factor = p5 if k == 0 or k == nblyr else c1
                        dflux_bio[it] += trcrn[base + k, n] * bvol * factor / dt
                    dflux_bio[it] += (
                        trcrn[base + nblyr + 1, n] * dvssl
                        + trcrn[base + nblyr + 2, n] * dvint
                    ) * fraction / dt
            if bool(t.tr_iso):
                for it in range(int(t.n_iso)):
                    dfiso_ocn[it] += (
                        vsnon[n] * trcrn[int(t.nt_isosno) + it, n]
                        + vicen[n] * trcrn[int(t.nt_isoice) + it, n]
                    ) * fraction / dt
            for k in range(nilyr):
                dfhocn += (trcrn[int(t.nt_qice) + k, n]
                           * vicen[n] / nilyr * fraction / dt)
            for k in range(nslyr):
                dfhocn += (trcrn[int(t.nt_qsno) + k, n]
                           * vsnon[n] / nslyr * fraction / dt)
            dfresh += (p.rhoi * vicen[n] + p.rhos * vsnon[n]) * fraction / dt
            if str(p.saltflux_option).strip() == "prognostic":
                sicen = c0
                for k in range(nilyr):
                    sicen += trcrn[int(t.nt_sice) + k, n] / nilyr
            else:
                sicen = p.ice_ref_salinity
            dfsalt += p.rhoi * vicen[n] * sicen * 0.001 * fraction / dt
            aicen[n] *= c1 / aice
            vicen[n] *= c1 / aice
            vsnon[n] *= c1 / aice
            # Tracers are unchanged.
        aice = c1
        aice0 = c0
    return aice, aice0, dfpond, dfresh, dfsalt, dfhocn


def zap_snow_temperature(dt, aicen, trcrn, vsnon, tr_aero,
                         dfaero_ocn, dfiso_ocn, dflux_bio):
    """Melt all snow in a category if any layer temperature is invalid."""
    t = icepack_tracers
    p = icepack_parameters
    dfresh = dfhocn = c0
    nslyr = int(t.nslyr)
    for n in range(int(t.ncat)):
        l_zap = False
        if aicen[n] > p.puny:
            hsn = vsnon[n] / aicen[n]
            for k in range(nslyr):
                if hsn > p.hs_min:
                    zqsn = trcrn[int(t.nt_qsno) + k, n]
                    Tmax = -zqsn * p.puny * nslyr / (p.rhos * p.cp_ice * vsnon[n])
                else:
                    zqsn = -p.rhos * p.Lfresh
                    Tmax = p.puny
                zTsn = (p.Lfresh + zqsn / p.rhos) / p.cp_ice
                if zTsn < icepack_therm_shared.Tmin or zTsn > Tmax:
                    l_zap = True
                    ice_fileunits.nu_diag.write(
                        f"(zap_snow_temperature) layer {k + 1}: "
                        f"temperature {zTsn} outside [{icepack_therm_shared.Tmin}, {Tmax}]\n"
                    )
        if l_zap:
            vsnon[n], fresh, hocn = zap_snow(
                dt, trcrn[:, n], vsnon[n], c0, c0, dfaero_ocn,
                tr_aero, dfiso_ocn, dflux_bio, aicen[n],
            )
            dfresh += fresh
            dfhocn += hocn
    return dfresh, dfhocn


def cleanup_itd(dt, hin_max, aicen, trcrn, vicen, vsnon,
                tr_aero, tr_pond_topo, first_ice,
                trcr_depend, trcr_base, n_trcr_strata, nt_strata, Tf,
                limit_aice=True, dorebin=True):
    """Rebin and remove invalid residual ice, updating conserved fluxes.

    Returns area, open-water area, four scalar flux increments, and the
    aerosol, isotope, and biology ocean flux increments.
    """
    t = icepack_tracers
    p = icepack_parameters
    dfaero_ocn = np.zeros(int(t.n_aero), dtype=dbl_kind)
    dfiso_ocn = np.zeros(int(t.n_iso), dtype=dbl_kind)
    dflux_bio = np.zeros(int(t.nbtrcr), dtype=dbl_kind)
    aice, aice0 = aggregate_area(aicen)
    if limit_aice and (aice > c1 + p.puny or aice < -p.puny):
        raise RuntimeError("(cleanup_itd) aggregate ice area out of bounds")
    if dorebin and aice > p.puny:
        rebin(aicen, trcrn, vicen, vsnon, hin_max, Tf,
              trcr_depend, trcr_base, n_trcr_strata, nt_strata)
    dfpond = dfresh = dfsalt = dfhocn = c0
    if limit_aice:
        aice, aice0, dfpond, dfresh, dfsalt, dfhocn = zap_small_areas(
            dt, aice, aice0, aicen, trcrn, vicen, vsnon,
            tr_aero, tr_pond_topo, first_ice, Tf,
            dfaero_ocn, dfiso_ocn, dflux_bio,
        )
    snow_fresh, snow_hocn = zap_snow_temperature(
        dt, aicen, trcrn, vsnon, tr_aero,
        dfaero_ocn, dfiso_ocn, dflux_bio,
    )
    dfresh += snow_fresh
    dfhocn += snow_hocn
    return (aice, aice0, dfpond, dfresh, dfsalt, dfhocn,
            dfaero_ocn, dfiso_ocn, dflux_bio)
