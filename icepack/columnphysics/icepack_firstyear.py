"""First-year sea-ice tracer update from icepack_firstyear.F90."""

import jax.numpy as jnp


def update_FYarea(dt, nhmask, shmask, yday, FYarea, secday):
    """Promote first-year ice to multi-year ice on each hemisphere's date.

    CICE zeros the first-year area at day 259 in the north and day 75 in
    the south. New frazil ice formed just before the date is also promoted.
    """
    north = nhmask & (yday >= 259.0) & (yday < 259.0 + dt / secday)
    south = shmask & (yday >= 75.0) & (yday < 75.0 + dt / secday)
    return jnp.where(north | south, 0.0, FYarea)
