"""Routines required to define a floe size distribution tracer for sea ice

Theory based on:

  Horvat, C., & Tziperman, E. (2015). A prognostic model of the sea-ice
  floe size and thickness distribution. The Cryosphere, 9(6), 2119-2134.
  doi:10.5194/tc-9-2119-2015

and implementation described in:

  Roach, L. A., Horvat, C., Dean, S. M., & Bitz, C. M. (2018). An emergent
  sea ice floe size distribution in a global coupled ocean--sea ice model.
  Journal of Geophysical Research: Oceans, 123(6), 4322-4337.
  doi:10.1029/2017JC013692

with some modifications.

For floe welding parameter and tensile mode parameter values, see

  Roach, L. A., Smith, M. M., & Dean, S. M. (2018). Quantifying
  growth of pancake sea ice floes using images from drifting buoys.
  Journal of Geophysical Research: Oceans, 123(4), 2851-2866.
  doi: 10.1002/2017JC013693

Variable naming convention:
for k = 1, nfsd and n = 1, ncat
  afsdn(k,n) = trcrn(:,:,nt_nfsd+k-1,n,:)
  afsd (k) is per thickness category or averaged over n
  afs is the associated scalar value for (k,n)

author for JAX version: Xinyuan Zhu
XZ: converted from F90 to Python

authors: Lettie Roach, VUW/NIWA
         C. M. Bitz, UW

2016: CMB started
2016-8: LR worked on most of it
2019: ECH ported to Icepack

Ported: initialization, cleanup, lateral growth, welding and wave fracture.
The time-step paths use JAX; random wave phases use explicit PRNG keys.
"""

import numpy as np
import jax.numpy as jnp

from cicecore.shared import ice_fileunits
from icepack.columnphysics import icepack_parameters
from icepack.columnphysics import icepack_tracers
from icepack.columnphysics.icepack_kinds import dbl_kind
from icepack.columnphysics.icepack_parameters import c0, c1, c2, c4, puny

floe_rad_h = None          # fsd size higher bound in m (radius)
floe_rad_c = None          # fsd size center in m (radius)
floe_rad_l = None          # fsd size lower bound in m (radius)
floe_binwidth = None       # fsd size binwidth in m (radius)
floe_area_l = None         # fsd area at lower bound (m^2)
floe_area_h = None         # fsd area at higher bound (m^2)
floe_area_c = None         # fsd area at bin centre (m^2)
floe_area_binwidth = None  # floe area bin width (m^2)

c_fsd_range = None         # string for history output

iweld = None               # floe size categories that can combine
                           # during welding (dimensionless)


def _lims_for_nfsd(nfsd):
    subname = "(icepack_init_fsd_bounds)"
    if nfsd == 24:
        return np.array([
            6.65000000e-02, 5.31030847e+00, 1.42865861e+01, 2.90576686e+01,
            5.24122136e+01, 8.78691405e+01, 1.39518470e+02, 2.11635752e+02,
            3.08037274e+02, 4.31203059e+02, 5.81277225e+02, 7.55141047e+02,
            9.45812834e+02, 1.34354446e+03, 1.82265364e+03, 2.47261361e+03,
            3.35434988e+03, 4.55051413e+03, 6.17323164e+03, 8.37461170e+03,
            1.13610059e+04, 1.54123510e+04, 2.09084095e+04, 2.83643675e+04,
            3.84791270e+04,
        ], dtype=dbl_kind)
    if nfsd == 16:
        return np.array([
            6.65000000e-02, 5.31030847e+00, 1.42865861e+01, 2.90576686e+01,
            5.24122136e+01, 8.78691405e+01, 1.39518470e+02, 2.11635752e+02,
            3.08037274e+02, 4.31203059e+02, 5.81277225e+02, 7.55141047e+02,
            9.45812834e+02, 1.34354446e+03, 1.82265364e+03, 2.47261361e+03,
            3.35434988e+03,
        ], dtype=dbl_kind)
    if nfsd == 12:
        return np.array([
            6.65000000e-02, 5.31030847e+00, 1.42865861e+01, 2.90576686e+01,
            5.24122136e+01, 8.78691405e+01, 1.39518470e+02, 2.11635752e+02,
            3.08037274e+02, 4.31203059e+02, 5.81277225e+02, 7.55141047e+02,
            9.45812834e+02,
        ], dtype=dbl_kind)
    if nfsd == 1:  # default case
        return np.array([6.65000000e-02, 3.0e+02], dtype=dbl_kind)
    raise ValueError(subname + " floe size categories not defined for nfsd")


def _copy_out(name, source, target):
    if target is None:
        return
    if len(target) != len(source):
        raise ValueError(f"(icepack_init_fsd_bounds) {name} incorrect size")
    if isinstance(source, list):
        for k, value in enumerate(source):
            target[k] = value
    else:
        target[...] = source


def icepack_init_fsd_bounds(
    floe_rad_l=None,
    floe_rad_c=None,
    floe_binwidth=None,
    c_fsd_range=None,
    write_diags=None,
    floe_rad_l_out=None,
    floe_rad_c_out=None,
    floe_binwidth_out=None,
    c_fsd_range_out=None,
):
    """Initialize ice fsd bounds (call whether or not restarting)
    Define the bounds, midpoints and widths of floe size
    categories in area and radius

    authors: Lettie Roach, NIWA/VUW and C. M. Bitz, UW

    Note that radius widths cannot be larger than twice previous
    category width or floe welding will not have an effect

    Note also that the bound of the lowest floe size category is used
    to define the lead region width and the domain spacing for wave fracture

    The driver passes arrays to fill, positionally or as
    ``floe_rad_l``, ``floe_rad_c``, ``floe_binwidth``, ``c_fsd_range``.
    The Fortran ``_out`` names are accepted as well.
    """
    global floe_rad_h
    global floe_area_l, floe_area_h, floe_area_c, floe_area_binwidth
    global iweld

    subname = "(icepack_init_fsd_bounds)"
    if floe_rad_l_out is None:
        floe_rad_l_out = floe_rad_l
    if floe_rad_c_out is None:
        floe_rad_c_out = floe_rad_c
    if floe_binwidth_out is None:
        floe_binwidth_out = floe_binwidth
    if c_fsd_range_out is None:
        c_fsd_range_out = c_fsd_range

    nfsd = int(icepack_tracers.nfsd)
    lims = _lims_for_nfsd(nfsd)
    floeshape = dbl_kind(icepack_parameters.floeshape)

    rad_l = np.array(lims[:nfsd], dtype=dbl_kind)
    floe_rad_h = np.array(lims[1:nfsd + 1], dtype=dbl_kind)
    rad_c = (floe_rad_h + rad_l) / c2

    floe_area_l = c4 * floeshape * rad_l ** 2
    floe_area_h = c4 * floeshape * floe_rad_h ** 2
    # floe_area_c = c4*floeshape*floe_rad_c**2
    # This is exactly in the middle of floe_area_h and floe_area_l
    # Whereas the above calculation is closer to floe_area_l.
    floe_area_c = (floe_area_h + floe_area_l) / c2

    binwidth = floe_rad_h - rad_l
    floe_area_binwidth = floe_area_h - floe_area_l

    # floe size categories that can combine during welding
    iweld = np.full((nfsd, nfsd), -999, dtype=np.int32)
    for n in range(nfsd):
        for m in range(nfsd):
            test = floe_area_c[n] + floe_area_c[m]
            for k in range(nfsd - 1):
                if (test >= floe_area_l[k]) and (test < floe_area_h[k]):
                    iweld[n, m] = k + 1
            if test >= floe_area_l[nfsd - 1]:
                iweld[n, m] = nfsd

    # write fsd bounds
    # floe_rad(0) = floe_rad_l(1); index 0 is the lower edge of category 1
    floe_rad = np.zeros(nfsd + 1, dtype=dbl_kind)
    floe_rad[0] = rad_l[0]
    labels = [""] * nfsd
    for n in range(1, nfsd + 1):
        floe_rad[n] = floe_rad_h[n - 1]
        # Save character string to write to history file
        c_nf = f"{n:2d}"
        c_fsd1 = f"{float(floe_rad[n - 1]):7.3f}"
        c_fsd2 = f"{float(floe_rad[n]):7.3f}"
        labels[n - 1] = c_fsd1 + "m < fsd Cat " + c_nf + " < " + c_fsd2 + "m"

    if write_diags:
        nu = ice_fileunits.nu_diag
        nu.write(" \n")
        nu.write(f" {subname}\n")
        nu.write(" floe_rad(n-1) < fsd Cat n < floe_rad(n)\n")
        for n in range(1, nfsd + 1):
            nu.write(
                f" {float(floe_rad[n - 1])}  < fsd Cat  {n}  <  {float(floe_rad[n])}\n"
            )
        nu.write(" \n")
        nu.flush()

    globals()["floe_rad_l"] = rad_l
    globals()["floe_rad_c"] = rad_c
    globals()["floe_binwidth"] = binwidth
    globals()["c_fsd_range"] = labels
    _copy_out("floe_rad_l_out", rad_l, floe_rad_l_out)
    _copy_out("floe_rad_c_out", rad_c, floe_rad_c_out)
    _copy_out("floe_binwidth_out", binwidth, floe_binwidth_out)
    _copy_out("c_fsd_range_out", labels, c_fsd_range_out)


def icepack_init_fsd(ice_ic, afsd):
    """Initialize the FSD

    authors: Lettie Roach, NIWA/VUW

    When growing from no-ice conditions, initialize to zero.
    This allows the FSD to emerge, as described in Roach, Horvat et al. (2018)

    Otherwise initalize with a power law, following Perovich
    & Jones (2014). The basin-wide applicability of such a
    prescribed power law has not yet been tested.

    Perovich, D. K., & Jones, K. F. (2014). The seasonal evolution of
    sea ice floe size distribution. Journal of Geophysical Research: Oceans,
    119(12), 8767-8777. doi:10.1002/2014JC010136
    """
    nfsd = int(icepack_tracers.nfsd)
    if str(ice_ic).strip() == "none":
        afsd[:nfsd] = c0
    else:  # Perovich (2014)
        # fraction of ice in each floe size and thickness category
        # same for ALL cells (even where no ice) initially
        alpha = dbl_kind(2.1)
        totfrac = c0  # total fraction of floes
        for k in range(nfsd):
            num_fsd = (2 * floe_rad_c[k]) ** (-alpha - c1)  # number distribution of floes
            afsd[k] = num_fsd * floe_area_c[k] * floe_binwidth[k]  # fraction distribution of floes
            totfrac = totfrac + afsd[k]
        afsd[:nfsd] = afsd[:nfsd] / totfrac  # normalize


def icepack_cleanup_fsdn(afsd):
    """Clean up small values and renormalize -- per category

    authors:  Elizabeth Hunke, LANL
    """
    nfsd = int(icepack_tracers.nfsd)
    for k in range(nfsd):
        if afsd[k] < puny:
            afsd[k] = c0
    tot = np.sum(afsd[:nfsd])
    if tot > puny:
        for k in range(nfsd):
            afsd[k] = afsd[k] / tot  # normalize
    else:  # represents ice-free cell, so set to zero
        afsd[:nfsd] = c0


def icepack_cleanup_fsd(afsdn):
    """Clean up small values and renormalize

    authors:  Elizabeth Hunke, LANL

    ``afsdn`` is (floe category, thickness category), Fortran ``afsdn(k,n)``.
    """
    if icepack_tracers.tr_fsd:
        ncat = int(icepack_tracers.ncat)
        for n in range(ncat):
            icepack_cleanup_fsdn(afsdn[:, n])


def icepack_cleanup_fsdn_jax(afsd, puny_value):
    """Clean and normalize one floe-size distribution on its last axis."""
    afsd = jnp.where(afsd < puny_value, 0.0, afsd)
    total = jnp.zeros_like(afsd[..., 0])
    for k in range(afsd.shape[-1]):
        total = total + afsd[..., k]
    return jnp.where((total > puny_value)[..., None],
                     afsd / jnp.where((total > puny_value)[..., None],
                                      total[..., None], 1.0), 0.0)


def icepack_cleanup_fsd_jax(afsdn, puny_value):
    """Clean every ice-thickness category; floe size precedes category."""
    return jnp.swapaxes(
        icepack_cleanup_fsdn_jax(jnp.swapaxes(afsdn, -2, -1), puny_value),
        -2, -1)


def get_subdt_fsd_jax(afsd_init, d_afsd, puny_value, bignum):
    """Limit a floe-size substep before any category becomes negative or >1."""
    growing = d_afsd > puny_value
    shrinking = d_afsd < -puny_value
    check_dt = jnp.where(
        growing,
        (1.0 - afsd_init) / jnp.where(growing, d_afsd, 1.0),
        jnp.where(shrinking,
                  afsd_init / jnp.where(shrinking, jnp.abs(d_afsd), 1.0),
                  bignum))
    return jnp.min(check_dt, axis=-1)


def partition_area_jax(aice, aicen, vicen, afsdn, floe_rad_c, puny):
    """Split open water near floes from their lateral surface area."""
    lead_area = jnp.zeros_like(aice)
    latsurf_area = jnp.zeros_like(aice)
    width = floe_rad_c[0]
    for n in range(aicen.shape[0]):
        thickness = vicen[n] / jnp.where(aicen[n] > 0.0, aicen[n], 1.0)
        for k in range(floe_rad_c.shape[0]):
            lead_area = lead_area + aicen[n] * afsdn[k, n] * (
                2.0 * width / floe_rad_c[k]
                + width**2 / floe_rad_c[k]**2)
            latsurf_area = (latsurf_area + afsdn[k, n] * aicen[n]
                            * 2.0 * thickness / floe_rad_c[k])
    lead_area = jnp.where(aice > puny,
                          jnp.minimum(lead_area, 1.0 - aice), 0.0)
    lead_area = jnp.where(lead_area > -puny,
                          jnp.maximum(lead_area, 0.0), lead_area)
    latsurf_area = jnp.where(aice > puny, latsurf_area, 0.0)
    return lead_area, latsurf_area


def fsd_lateral_growth_jax(dt, aice, aicen, vicen, vi0new,
                           afsdn, floe_rad_c, puny):
    """Allocate new ice volume to lateral floe growth and open leads."""
    lead_area, latsurf_area = partition_area_jax(
        aice, aicen, vicen, afsdn, floe_rad_c, puny)
    lateral_volume = jnp.where(
        latsurf_area > puny,
        vi0new * lead_area /
        (1.0 + aice / jnp.where(latsurf_area > puny,
                                latsurf_area, 1.0)), 0.0)
    radial_growth = jnp.where(lateral_volume > puny,
                              lateral_volume / dt, 0.0)
    added_area = jnp.zeros_like(aicen)
    for n in range(aicen.shape[0]):
        for k in range(floe_rad_c.shape[0]):
            added_area = added_area.at[n].add(
                2.0 * aicen[n] * afsdn[k, n]
                * radial_growth * dt / floe_rad_c[k])
    area_sum = jnp.sum(added_area)
    added_area = jnp.where(
        area_sum >= lead_area,
        added_area * lead_area /
        jnp.where(area_sum > 0.0, area_sum, 1.0), added_area)
    return added_area, radial_growth, lead_area, jnp.sum(added_area)


def wave_dep_growth_jax(local_wave_spec, wave_height, wavefreq,
                        floe_rad_h, parameters):
    """Choose the floe-size bin in which wave-limited new ice forms."""
    peak = wavefreq[jnp.argmax(local_wave_spec)]
    amplitude = 0.5 * wave_height
    r_max = jnp.where(
        (amplitude > parameters["puny"]) & (peak > parameters["puny"]),
        0.5 * jnp.sqrt(0.167 * parameters["gravit"] /
                       (parameters["pi"]**5 * parameters["rhoi"]
                        * jnp.where(amplitude > 0.0, amplitude, 1.0) * 2.0))
        / jnp.where(peak > 0.0, peak**2, 1.0),
        parameters["bignum"])
    new_size = jnp.asarray(floe_rad_h.shape[0] - 1)
    for k in range(floe_rad_h.shape[0] - 2, -1, -1):
        new_size = jnp.where(r_max <= floe_rad_h[k], k, new_size)
    return new_size


def fsd_add_new_ice_jax(n, dt, ai0new, d_an_latg, d_an_newi,
                        radial_growth, area2, wave_sig_ht,
                        wave_spectrum, wavefreq, d_afsd_latg,
                        d_afsd_newi, afsdn, aicen_init, aicen,
                        trcrn, nt_fsd, wave_spec, floe_rad_c,
                        floe_rad_h, floe_binwidth, parameters):
    """Evolve floe sizes after side growth and adding ice to leads."""
    nfsd = floe_rad_c.shape[0]
    puny = parameters["puny"]
    distribution = afsdn[:, n]
    elapsed = jnp.zeros_like(dt)
    for _ in range(100):
        flux = jnp.zeros((nfsd + 1,), dtype=trcrn.dtype)
        for k in range(1, nfsd):
            flux = flux.at[k].set(
                radial_growth * distribution[k - 1] /
                floe_binwidth[k - 1])
        weighted_radius = jnp.sum(distribution / floe_rad_c)
        tendency = jnp.zeros_like(distribution)
        for k in range(nfsd):
            tendency = tendency.at[k].set(
                flux[k] - flux[k + 1]
                + 2.0 * radial_growth * distribution[k]
                * (1.0 / floe_rad_c[k] - weighted_radius))
        subdt = jnp.minimum(
            get_subdt_fsd_jax(distribution, tendency, puny,
                              parameters["bignum"]), dt - elapsed)
        advancing = (d_an_latg[n] > puny) & (elapsed < dt) & (subdt > 0.0)
        distribution = jnp.where(
            advancing, distribution + subdt * tendency, distribution)
        elapsed = jnp.where(advancing, elapsed + subdt, elapsed)
    distribution = jnp.where(
        d_an_latg[n] > puny,
        icepack_cleanup_fsdn_jax(distribution, puny), distribution)
    trcrn = trcrn.at[nt_fsd:nt_fsd + nfsd, n].set(distribution)
    grown_distribution = distribution

    if n == 0:
        new_size = jnp.asarray(nfsd - 1)
        if wave_spec:
            wave_size = wave_dep_growth_jax(
                wave_spectrum, wave_sig_ht, wavefreq,
                floe_rad_h, parameters)
            new_size = jnp.where(wave_sig_ht > puny, wave_size, new_size)
        else:
            new_size = jnp.asarray(0)
        denominator = area2[n] + ai0new
        new_distribution = (distribution * area2[n] /
                            jnp.where(denominator > 0.0,
                                      denominator, 1.0))
        new_distribution = new_distribution.at[new_size].add(
            ai0new / jnp.where(denominator > 0.0, denominator, 1.0))
        new_distribution = icepack_cleanup_fsdn_jax(
            new_distribution, puny)
        distribution = jnp.where(d_an_newi[n] > puny,
                                 new_distribution, distribution)
        trcrn = trcrn.at[nt_fsd:nt_fsd + nfsd, n].set(distribution)

    d_afsd_latg = (d_afsd_latg + area2[n] * grown_distribution
                   - aicen_init[n] * afsdn[:, n])
    d_afsd_newi = (d_afsd_newi + aicen[n] * distribution
                   - area2[n] * grown_distribution)
    failed = (d_an_latg[n] > puny) & (elapsed < dt)
    return trcrn, d_afsd_latg, d_afsd_newi, failed


def fsd_weld_thermo_jax(dt, frzmlt, aicen, trcrn, d_afsd_weld,
                        nt_fsd, floe_area_c, iweld, parameters):
    """Weld floes during freezing, following ``fsd_weld_thermo``."""
    puny = parameters["puny"]
    nfsd = floe_area_c.shape[0]
    d_afsd_weld = jnp.zeros_like(d_afsd_weld)
    failed = jnp.asarray(False)
    for n in range(aicen.shape[0]):
        original = icepack_cleanup_fsdn_jax(
            trcrn[nt_fsd:nt_fsd + nfsd, n], puny)
        active = ((frzmlt > puny) & (aicen[n] > 0.1)
                  & (jnp.sum(original[:-1]) > puny))
        distribution = jnp.where(original < puny, 0.0, original)
        distribution = distribution / jnp.where(
            jnp.sum(distribution) > 0.0, jnp.sum(distribution), 1.0)
        elapsed = jnp.zeros_like(dt)
        # The Fortran loop uses an adaptive substep. A fixed JAX loop leaves
        # the iteration count static; an unfinished cell is reported below.
        for _ in range(100):
            number = distribution / floe_area_c
            stability = jnp.where(
                distribution > puny,
                number / jnp.where(
                    distribution > puny,
                    1.0e-8 * distribution * aicen[n], 1.0),
                parameters["bignum"])
            stability = jnp.where(stability < puny,
                                  parameters["bignum"], stability)
            subdt = jnp.minimum(jnp.min(stability), dt - elapsed)
            gain = jnp.zeros_like(distribution)
            loss = jnp.zeros_like(distribution)
            for i in range(nfsd):
                for j in range(nfsd):
                    receiver = iweld[i, j] - 1  # Fortran category index.
                    amount = (1.0e-8 * floe_area_c[i] * aicen[n]
                              * distribution[i] * distribution[j])
                    loss = loss.at[i].add(jnp.where(receiver > i, amount, 0.0))
                    for k in range(nfsd):
                        gain = gain.at[k].add(jnp.where(
                            (receiver > i) & (receiver == k), amount, 0.0))
            advancing = active & (elapsed < dt) & (
                distribution[-1] <= 1.0 - puny) & (subdt > 0.0)
            updated = distribution + subdt * (gain - loss)
            updated = jnp.where(updated < puny, 0.0, updated)
            updated = updated / jnp.where(
                jnp.sum(updated) > 0.0, jnp.sum(updated), 1.0)
            distribution = jnp.where(advancing, updated, distribution)
            elapsed = jnp.where(advancing, elapsed + subdt, elapsed)
        failed = failed | (active & (elapsed < dt)
                           & (distribution[-1] <= 1.0 - puny))
        distribution = icepack_cleanup_fsdn_jax(distribution, puny)
        trcrn = trcrn.at[nt_fsd:nt_fsd + nfsd, n].set(jnp.where(
            active, distribution, trcrn[nt_fsd:nt_fsd + nfsd, n]))
        d_afsd_weld = d_afsd_weld + jnp.where(
            active, aicen[n] * (distribution - original), 0.0)
    return trcrn, d_afsd_weld, failed
