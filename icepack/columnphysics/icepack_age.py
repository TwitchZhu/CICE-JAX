"""Ice age tracer update from icepack_age.F90."""


def increment_age(dt, iage):
    """Increase ice age by the thermodynamic time step."""
    return iage + dt
