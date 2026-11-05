"""Primary state variables in various configurations.
Note: other state variables are at the end of this...
The primary state variable names are:
-------------------------------------------------------------------
for each category   aggregated over     units
                      categories
-------------------------------------------------------------------
aicen(i,j,n)         aice(i,j)           ---
vicen(i,j,n)         vice(i,j)           m
vsnon(i,j,n)         vsno(i,j)           m
trcrn(i,j,it,n)      trcr(i,j,it)

Area is dimensionless because aice is the fractional area
(normalized so that the sum over all categories, including open
water, is 1.0).  That is why vice/vsno have units of m instead of m^3.

Variable names follow these rules:

(1) For 3D variables (indices i,j,n), write 'ice' or 'sno' or
    'sfc' and put an 'n' at the end.
(2) For 2D variables (indices i,j) aggregated over all categories,
    write 'ice' or 'sno' or 'sfc' without the 'n'.
(3) For 2D variables (indices i,j) associated with an individual
    category, write 'i' or 's' instead of 'ice' or 'sno' and put an 'n'
    at the end: e.g. hin, hsn.  These are not declared here
    but in individual modules (e.g., icepack_therm_vertical).

author for JAX version: Xinyuan Zhu
XZ: converted from F90 to Python

2006: Converted to free form source (F90) by Elizabeth Hunke
2004: Block structure added by William Lipscomb
authors C. M. Bitz, UW
        Elizabeth C. Hunke and William H. Lipscomb, LANL

This step allocates and provides ``bound_state``.
Arrays include the ghost cells and have no block index.
``maskhalo_bound`` is read from ``domain_nml`` by ``init_domain_blocks``.
The serial ``ice_HaloMask`` stores the mask and does not change local copies.
"""

import numpy as np

from cicecore.shared import ice_domain_size
from cicecore.shared.ice_constants import (
    c0,
    field_loc_center,
    field_type_scalar,
)
from cicecore.shared.ice_kinds_mod import dbl_kind, int_kind
from cicecore.cicedyn.infrastructure import ice_domain
from cicecore.cicedyn.infrastructure.comm.serial.ice_boundary import (
    ice_HaloMask,
    ice_halo_update,
)

#-----------------------------------------------------------------
# state of the ice aggregated over all categories
#-----------------------------------------------------------------
aice = None  # concentration of ice on T grid
aiU = None   # concentration of ice on U grid
vice = None  # volume per unit area of ice          (m)
vsno = None  # volume per unit area of snow         (m)
trcr = None  # ice tracers
             # 1: surface temperature of ice/snow (C)
             # Python stores that tracer at index 0.

#-----------------------------------------------------------------
# state of the ice for each category
#-----------------------------------------------------------------
aice0 = None  # concentration of open water
aicen = None  # concentration of ice
vicen = None  # volume per unit area of ice          (m)
vsnon = None  # volume per unit area of snow         (m)
trcrn = None  # tracers
              # 1: surface temperature of ice/snow (C)
              # Python stores that tracer at index 0.

#-----------------------------------------------------------------
# tracers infrastructure arrays
#-----------------------------------------------------------------
trcr_depend = None    # = 0 for ice area tracers
                      # = 1 for ice volume tracers
                      # = 2 for snow volume tracers
n_trcr_strata = None  # number of underlying tracer layers
nt_strata = None      # indices of underlying tracer layers
trcr_base = None      # = 0 or 1 depending on tracer dependency
                      # argument 2:  (1) aice, (2) vice, (3) vsno

#-----------------------------------------------------------------
# dynamic variables closely related to the state of the ice
#-----------------------------------------------------------------
uvel = None      # x-component of velocity on U grid (m/s)
vvel = None      # y-component of velocity on U grid (m/s)
uvelE = None     # x-component of velocity on E grid (m/s)
vvelE = None     # y-component of velocity on E grid (m/s)
uvelN = None     # x-component of velocity on N grid (m/s)
vvelN = None     # y-component of velocity on N grid (m/s)
divu = None      # strain rate I component, velocity divergence (1/s)
shear = None     # strain rate II component (1/s)
vort = None      # vorticity (1/s)
strength = None  # ice strength (N/m)

#-----------------------------------------------------------------
# ice state at start of time step, saved for later in the step
#-----------------------------------------------------------------
aice_init = None   # initial concentration of ice, for diagnostics
aicen_init = None  # initial ice concentration, for linear ITD
vicen_init = None  # initial ice volume (m), for linear ITD
vsnon_init = None  # initial snow volume (m), for aerosol
Tsfcn_init = None  # initial ice surface temperature (degC)


def _zero(shape):
    return np.zeros(shape, dtype=dbl_kind)


def alloc_state():
    """Allocate space for all state variables. Values start at zero."""
    global aice, aiU, vice, vsno, aice0
    global uvel, vvel, uvelE, vvelE, uvelN, vvelN
    global divu, shear, vort, strength, aice_init
    global aicen, vicen, vsnon, aicen_init, vicen_init, vsnon_init, Tsfcn_init
    global trcr, trcrn
    global trcr_depend, n_trcr_strata, nt_strata, trcr_base

    ncat = int(ice_domain_size.ncat)
    ntrcr = int(ice_domain_size.ntrcr)
    if ntrcr < 1 or ncat < 1:
        raise ValueError("(alloc_state) ntrcr and ncat must be set before allocation")

    horiz = ice_domain_size.horizontal_shape()
    cat = horiz + (ncat,)
    tracers = horiz + (ntrcr,)
    tracers_cat = horiz + (ntrcr, ncat)

    aice = _zero(horiz)
    aiU = _zero(horiz)
    vice = _zero(horiz)
    vsno = _zero(horiz)
    aice0 = _zero(horiz)
    uvel = _zero(horiz)
    vvel = _zero(horiz)
    uvelE = _zero(horiz)
    vvelE = _zero(horiz)
    uvelN = _zero(horiz)
    vvelN = _zero(horiz)
    divu = _zero(horiz)
    shear = _zero(horiz)
    vort = _zero(horiz)
    strength = _zero(horiz)
    aice_init = _zero(horiz)
    aicen = _zero(cat)
    vicen = _zero(cat)
    vsnon = _zero(cat)
    aicen_init = _zero(cat)
    vicen_init = _zero(cat)
    vsnon_init = _zero(cat)
    Tsfcn_init = _zero(cat)
    trcr = _zero(tracers)
    trcrn = _zero(tracers_cat)

    count = int(ntrcr)
    trcr_depend = np.zeros(count, dtype=int_kind)
    n_trcr_strata = np.zeros(count, dtype=int_kind)
    nt_strata = np.full((count, 2), -1, dtype=int_kind)
    trcr_base = np.zeros((count, 3), dtype=dbl_kind)
    trcr_base[...] = c0


def bound_state(aicen, vicen, vsnon, ntrcr, trcrn):
    """Fill ghost cells of the category ice state.

    aicen, vicen, vsnon: category area and volume, shape (i, j, ncat)
    ntrcr: number of tracers in use
    trcrn: category tracers, shape (i, j, ntrcr, ncat)
    """
    # The Fortran argument list is kept. One grid, so there is no block index.
    _ = ntrcr

    ice_halo_update(aicen, field_loc_center, field_type_scalar)

    if ice_domain.maskhalo_bound:
        # halomask = 1 where any category has ice area
        halomask = np.zeros(aicen.shape[:2], dtype=int_kind)
        halomask[np.any(aicen > c0, axis=-1)] = 1
        # call ice_HaloMask(halo_info_aicemask, halo_info, halomask)
        # halos are not masked for local copies
        ice_HaloMask(halomask)
        ice_halo_update(trcrn, field_loc_center, field_type_scalar)
        ice_halo_update(vicen, field_loc_center, field_type_scalar)
        ice_halo_update(vsnon, field_loc_center, field_type_scalar)
        # call ice_HaloDestroy(halo_info_aicemask)
    else:
        ice_halo_update(trcrn, field_loc_center, field_type_scalar)
        ice_halo_update(vicen, field_loc_center, field_type_scalar)
        ice_halo_update(vsnon, field_loc_center, field_type_scalar)
