"""JAX wave-fracture calculations from icepack_wavefracspec.F90."""

import jax
import jax.numpy as jnp
from jax import lax

from icepack.columnphysics.icepack_fsd import (
    get_subdt_fsd_jax, icepack_cleanup_fsd_jax,
)


def get_dafsd_wave(afsd_init, fracture_hist, frac, puny):
    """Return the floe-category area change and Fortran warning condition.

    Floe categories are the last axis; ``frac`` has source then destination
    axes. The loops preserve the original accumulation order.
    """
    nfsd = afsd_init.shape[-1]
    prefix = jnp.zeros_like(afsd_init[..., 0])
    omega = jnp.zeros_like(afsd_init)
    for k in range(nfsd):
        omega = omega.at[..., k].set(afsd_init[..., k] * prefix)
        prefix = prefix + fracture_hist[..., k]

    d_afsd = jnp.zeros_like(afsd_init)
    total_change = jnp.zeros_like(prefix)
    for k in range(nfsd):
        gain = jnp.zeros_like(prefix)
        for source in range(nfsd):
            gain = gain + omega[..., source] * frac[..., source, k]
        change = gain - omega[..., k]
        d_afsd = d_afsd.at[..., k].set(change)
        total_change = total_change + change
    return jnp.where(jnp.abs(d_afsd) < puny, 0.0, d_afsd), total_change > puny


def get_fraclengths(X, eta, hbar, puny, straincrit=3.0e-5):
    """Find floe fracture spacings from wave-height extrema.

    This matches the Fortran peakfinder's first-tie rule within 10 m on
    either side. Extrema triplets are max-min-max or min-max-min. Only the
    distances between successive points above critical strain are returned;
    all remaining array entries are zero.
    """
    nx = eta.shape[0]
    # The Fortran wave domain has dx=1 m and threshold=10 m.
    spacing = 10
    is_max = jnp.ones((nx,), dtype=bool)
    is_min = jnp.ones((nx,), dtype=bool)
    for distance in range(1, spacing + 1):
        previous_max = jnp.pad(eta, (distance, 0), constant_values=-jnp.inf)[:nx]
        following_max = jnp.pad(eta, (0, distance), constant_values=-jnp.inf)[distance:]
        previous_min = jnp.pad(eta, (distance, 0), constant_values=jnp.inf)[:nx]
        following_min = jnp.pad(eta, (0, distance), constant_values=jnp.inf)[distance:]
        is_max = is_max & (eta > previous_max) & (eta >= following_max)
        is_min = is_min & (eta < previous_min) & (eta <= following_min)

    is_extremum = is_max | is_min
    indices = jnp.arange(nx, dtype=jnp.int32)
    previous_scan = lax.associative_scan(
        jnp.maximum, jnp.where(is_extremum, indices, -1))
    following_scan = lax.associative_scan(
        jnp.minimum, jnp.where(is_extremum, indices, nx), reverse=True)
    j_neg = jnp.concatenate((jnp.array([-1], dtype=indices.dtype),
                             previous_scan[:-1]))
    j_pos = jnp.concatenate((following_scan[1:],
                             jnp.array([nx], dtype=indices.dtype)))
    neg = jnp.clip(j_neg, 0, nx - 1)
    pos = jnp.clip(j_pos, 0, nx - 1)
    interior = (indices > 0) & (indices < nx - 1)
    triplet = (interior & (j_neg >= 0) & (j_pos < nx) &
               ((is_max[neg] & is_min & is_max[pos]) |
                (is_min[neg] & is_max & is_min[pos])))
    delta = X - X[neg]
    delta_pos = X[pos] - X
    denominator = delta * delta_pos * (delta + delta_pos)
    strain = jnp.abs(
        hbar * (eta[neg] * delta_pos - eta * (delta_pos + delta)
                + eta[pos] * delta)
        / jnp.where(triplet & (denominator != 0.0), denominator, 1.0))
    fractured = triplet & (denominator != 0.0) & (strain > straincrit)
    fractured = fractured & (hbar > puny)

    positions = jnp.nonzero(fractured, size=nx, fill_value=0)[0]
    count = jnp.sum(fractured)
    distances = X[positions[1:]] - X[positions[:-1]]
    distances = jnp.where(jnp.arange(nx - 1) < count - 1, distances, 0.0)
    return jnp.concatenate((distances, jnp.zeros((1,), dtype=X.dtype)))


def wave_frac(wavefreq, dwavefreq, hbar, spec_efreq,
              floe_rad_l, floe_rad_c, wave_spec_type,
              puny, gravit, pi, bignum, key=None):
    """Build the fracture histogram from a wave-spectrum realization.

    The Fortran wave domain is 10 km long at 1 m spacing. Random spectra
    draw one phase per frequency and iteration. A fixed-length scan freezes
    the result after convergence so reverse-mode JAX differentiation remains
    available; the updated key is returned to the caller.
    """
    if wave_spec_type == "random":
        if key is None:
            raise ValueError("random wave spectrum requires a JAX PRNG key")
        max_iter = 100
    else:
        max_iter = 1
        if key is None:
            key = jax.random.key(0)

    nx = 10000
    nfreq = wavefreq.shape[0]
    nfsd = floe_rad_l.shape[0]
    X = jnp.arange(1, nx + 1, dtype=wavefreq.dtype)
    wavelength = gravit / (2.0 * pi * wavefreq**2)
    spec_coeff = jnp.sqrt(2.0 * spec_efreq * dwavefreq)
    initial = (
        jnp.zeros((nfsd,), dtype=wavefreq.dtype),
        jnp.zeros((nfsd,), dtype=wavefreq.dtype),
        jnp.asarray(bignum, dtype=wavefreq.dtype),
        key,
        jnp.asarray(0, dtype=jnp.int32),
    )

    def scan_body(carry, _):
        frac_local, histogram, error, current_key, count = carry

        def advance(values):
            previous, counts, _, phase_key, iteration = values
            if wave_spec_type == "random":
                phase_key, sample_key = jax.random.split(phase_key)
                rand_array = jax.random.uniform(
                    sample_key, (nfreq,), dtype=wavefreq.dtype)
            else:
                rand_array = jnp.full_like(wavefreq, 0.5)
            phi = 2.0 * pi * rand_array

            # Surface height is the sum of unattenuated spectral components.
            eta = jnp.zeros_like(X)
            for k in range(nfreq):
                eta = eta + spec_coeff[k] * jnp.cos(
                    2.0 * pi * X / wavelength[k] + phi[k])
            lengths = lax.cond(
                (jnp.sum(jnp.abs(eta)) > puny) & (hbar > puny),
                lambda _: get_fraclengths(X, eta, hbar, puny),
                lambda _: jnp.zeros_like(X),
                operand=None)
            radii = lengths / 2.0

            # Count fracture lengths in floe-size bins. Fortran accumulates
            # the histogram across phase realizations rather than clearing it.
            updated_counts = counts
            for k in range(nfsd - 1):
                in_bin = ((radii > floe_rad_l[0]) &
                          (radii >= floe_rad_l[k]) &
                          (radii < floe_rad_l[k + 1]))
                updated_counts = updated_counts.at[k].add(jnp.sum(in_bin))
            updated_counts = updated_counts.at[-1].add(
                jnp.sum(radii > floe_rad_l[-1]))
            weighted = floe_rad_c * updated_counts
            total = jnp.sum(weighted)
            new_fraction = jnp.where(
                jnp.all(radii < floe_rad_l[0]), 0.0,
                weighted / jnp.where(total != 0.0, total, 1.0))
            if wave_spec_type == "random":
                new_error = jnp.sum(jnp.abs(new_fraction - previous)) / nfsd
            else:
                new_error = jnp.asarray(0.0, dtype=wavefreq.dtype)
            return (new_fraction, updated_counts, new_error,
                    phase_key, iteration + 1)

        current = lax.cond((count < max_iter) & (error > 6.5e-4),
                           advance, lambda values: values, carry)
        return current, None

    (frac_local, _, _, key, count), _ = lax.scan(
        scan_body, initial, None, length=max_iter)
    warning = (count >= 100) if wave_spec_type == "random" else jnp.asarray(False)
    return frac_local, key, warning


def icepack_step_wavefracture(dt, aice, vice, aicen, wave_spectrum,
                              wavefreq, dwavefreq, trcrn, wave_height,
                              floe_rad_l, floe_rad_c, nt_fsd,
                              wave_spec_type, parameters, key=None):
    """Fracture floe-size categories in one horizontal cell."""
    puny = parameters["puny"]
    nfsd = floe_rad_l.shape[0]
    ncat = aicen.shape[0]
    d_afsd_wave = jnp.zeros((nfsd,), dtype=trcrn.dtype)
    afsdn = trcrn[nt_fsd:nt_fsd + nfsd, :]
    active = ((jnp.min(afsdn[0, :]) < 1.0 - puny)
              & (aice > 0.01) & (wave_height > 0.1))
    if key is None:
        if wave_spec_type == "random":
            raise ValueError("random wave spectrum requires a JAX PRNG key")
        key = jax.random.key(0)

    def fracture(carry):
        trcrn, d_afsd_wave, key = carry
        hbar = vice / aice
        fracture_hist, key, spectrum_warning = wave_frac(
            wavefreq, dwavefreq, hbar, wave_spectrum,
            floe_rad_l, floe_rad_c, wave_spec_type,
            puny, parameters["gravit"], parameters["pi"],
            parameters["bignum"], key)
        afsdn = trcrn[nt_fsd:nt_fsd + nfsd, :]

        def apply_histogram(values):
            trcrn, d_afsd_wave = values
            afsdn = icepack_cleanup_fsd_jax(
                trcrn[nt_fsd:nt_fsd + nfsd, :], puny)
            trcrn = trcrn.at[nt_fsd:nt_fsd + nfsd, :].set(afsdn)
            frac = jnp.zeros((nfsd, nfsd), dtype=trcrn.dtype)
            for source in range(1, nfsd):
                frac = frac.at[source, :source].set(fracture_hist[:source])
            row_sum = jnp.sum(frac, axis=1, keepdims=True)
            frac = jnp.where(row_sum > 0.0,
                             frac / jnp.where(row_sum > 0.0, row_sum, 1.0),
                             0.0)
            subcycle_warning = jnp.asarray(False)
            for n in range(ncat):
                initial = trcrn[nt_fsd:nt_fsd + nfsd, n]
                category_active = ((aicen[n] > puny)
                                   & (jnp.sum(initial) > puny)
                                   & (initial[0] < 1.0))

                def advance_category(afsd_init):
                    # A bounded scan permits reverse-mode gradients through
                    # the adaptive FSD substeps. The original warns after 100.
                    def subcycle(state, _):
                        current, elapsed, count, warning = state
                        advancing = ((elapsed < dt)
                                     & (current[0] < 1.0 - puny))
                        tendency, nonconserved = get_dafsd_wave(
                            current, fracture_hist, frac, puny)
                        subdt = jnp.minimum(
                            get_subdt_fsd_jax(
                                current, tendency, puny,
                                parameters["bignum"]), dt)
                        subdt = jnp.where(advancing, subdt, 0.0)
                        updated = current + subdt * tendency
                        warning = (warning | (advancing & nonconserved)
                                   | (advancing & (count >= 100))
                                   | (advancing & (subdt <= 0.0))
                                   | (jnp.min(updated) < -puny)
                                   | (jnp.max(updated) > 1.0 + puny))
                        return (updated, elapsed + subdt,
                                count + advancing.astype(jnp.int32),
                                warning), None

                    initial_state = (afsd_init,
                                     jnp.asarray(0.0, dtype=trcrn.dtype),
                                     jnp.asarray(0, dtype=jnp.int32),
                                     jnp.asarray(False))
                    (current, elapsed, _, warning), _ = lax.scan(
                        subcycle, initial_state, None, length=256)
                    warning = warning | ((elapsed < dt)
                                         & (current[0] < 1.0 - puny))
                    cons_error = jnp.sum(current) - 1.0
                    current = current.at[0].add(
                        jnp.where(cons_error < 0.0, -cons_error, 0.0))
                    # For area gain, subtract from the largest floe category
                    # that can absorb the entire error, as in Fortran.
                    selected = jnp.asarray(False)
                    for k in range(nfsd - 1, -1, -1):
                        can_absorb = ((cons_error >= 0.0) & ~selected
                                      & (current[k] > cons_error))
                        current = current.at[k].add(jnp.where(
                            can_absorb, -cons_error, 0.0))
                        selected = selected | can_absorb
                    warning = warning | ((cons_error >= 0.0) & ~selected)
                    total = jnp.sum(current)
                    current = current / jnp.where(total > 0.0, total, 1.0)
                    return current, warning

                updated, warning = lax.cond(
                    category_active, advance_category,
                    lambda value: (value, jnp.asarray(False)), initial)
                trcrn = trcrn.at[nt_fsd:nt_fsd + nfsd, n].set(updated)
                trcrn = trcrn.at[nt_fsd:nt_fsd + nfsd, :].set(
                    icepack_cleanup_fsd_jax(
                        trcrn[nt_fsd:nt_fsd + nfsd, :], puny))
                d_afsd_wave = d_afsd_wave + jnp.where(
                    category_active, aicen[n] * (updated - initial), 0.0)
                subcycle_warning = subcycle_warning | warning
            return trcrn, d_afsd_wave, subcycle_warning

        trcrn, d_afsd_wave, subcycle_warning = lax.cond(
            jnp.max(fracture_hist) > puny, apply_histogram,
            lambda values: (values[0], values[1], jnp.asarray(False)),
            (trcrn, d_afsd_wave))
        return trcrn, d_afsd_wave, key, spectrum_warning | subcycle_warning

    return lax.cond(
        active, fracture,
        lambda carry: (carry[0], carry[1], carry[2], jnp.asarray(False)),
        (trcrn, d_afsd_wave, key))
