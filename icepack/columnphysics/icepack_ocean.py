"""Ocean surface heat balance from icepack_ocean.F90."""

import jax.numpy as jnp


def icepack_ocn_mixed_layer(alvdr_ocn, swvdr, alidr_ocn, swidr,
                            alvdf_ocn, swvdf, alidf_ocn, swidf,
                            sst, shcoef, lhcoef, flw, delt, delq,
                            aice, fhocn, fswthru, hmix, Tf, qdp, dt,
                            parameters):
    """Update SST and freezing potential with the mixed-layer heat budget.

    SST changes due to fluxes through ice are handled by the vertical
    thermodynamics routine in CICE, before this ocean-surface calculation.
    """
    # Shortwave radiative flux.
    swabs = ((1.0 - alvdr_ocn) * swvdr + (1.0 - alidr_ocn) * swidr
             + (1.0 - alvdf_ocn) * swvdf + (1.0 - alidf_ocn) * swidf)
    # Ocean surface temperature in Kelvin and outgoing longwave radiation.
    TsfK = sst + parameters["Tffresh"]
    flwout_ocn = -parameters["stefan_boltzmann"] * TsfK**4
    # Downward latent and sensible heat fluxes.
    fsens_ocn = shcoef * delt
    flat_ocn = lhcoef * delq
    evap_ocn = -flat_ocn / parameters["Lvap"]

    # Exchange with the atmosphere and ice above the mixed layer.
    heat_capacity = parameters["cprho"] * hmix
    sst = sst + dt * (
        (fsens_ocn + flat_ocn + flwout_ocn + flw + swabs) * (1.0 - aice)
        + fhocn + fswthru
    ) / heat_capacity
    # Suppress cooling by the deep ocean when the mixed layer would freeze.
    qdp = jnp.where((sst <= Tf) & (qdp > 0.0), 0.0, qdp)
    sst = sst - qdp * dt / heat_capacity
    # Freezing/melting potential is capped at 1000 W/m^2.
    frzmlt = jnp.clip((Tf - sst) * parameters["cprho"] * hmix / dt,
                       -1000.0, 1000.0)
    sst = jnp.where(sst <= Tf, Tf, sst)
    return sst, flwout_ocn, fsens_ocn, flat_ocn, evap_ocn, qdp, frzmlt
