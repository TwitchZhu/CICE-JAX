"""Transports quantities using the second-order conservative remapping
scheme developed by John Dukowicz and John Baumgardner (DB) and modified
for sea ice by William Lipscomb and Elizabeth Hunke.

References:

Dukowicz, J. K., and J. R. Baumgardner, 2000: Incremental
 remapping as a transport/advection algorithm, J. Comput. Phys.,
 160, 318-335.

Lipscomb, W. H., and E. C. Hunke, 2004: Modeling sea ice
 transport using incremental remapping, Mon. Wea. Rev., 132,
 1341-1354.

Lemieux, J.-F., Lipscomb, W. H., Craig, A., Bailey, D. A.,
Hunke, E. C., Blain, P., Rasmussen, T. A. S., Bentsen, M.,
Dupont, F., Hebert, D., and Allard, R., 2024: CICE on a
C-grid: new momentum, stress, and transport schemes for CICEv6.5,
Geosci. Model Dev., 17, 6703-6724.

authors William H. Lipscomb, LANL
        John Baumgardner, LANL

author for JAX version: Xinyuan Zhu
XZ: converted from F90 to Python

2003: Vectorized by Clifford Chen (Fujitsu) and William Lipscomb
2004-05: Block structure added (WHL)
2006: Moved remap driver to ice_transport_driver
      Geometry changes:
      (1) Reconstruct fields in stretched logically rectangular coordinates
      (2) Modify geometry so that the area flux across each edge
          can be specified (following an idea of Mats Bentsen)
2010: ECH removed unnecessary grid arrays and optional arguments from
      horizontal_remap
2023: TAR, DMI Remove commented code and unnecessary arrays
2024: JFL and WHL improved robustness of Bentsen's approach for area flux.

Initialization ports ``init_remap`` only. ``horizontal_remap`` is the
time loop.

Here is some information about how the incremental remapping scheme
works in CICE and how it can be adapted for use in other models.

The remapping routine is designed to transport a generic mass-like
field (in CICE, the ice fractional area) along with an arbitrary number
of tracers in two dimensions.  The velocity components are assumed
to lie at grid cell corners and the transported scalars at cell centers.
Incremental remapping has the following desirable properties:

(1) Tracer monotonicity is preserved.  That is, no new local
    extrema are produced in fields like ice thickness or internal
    energy.
(2) The reconstucted mass and tracer fields vary linearly in x and y.
    This means that remapping is 2nd-order accurate in space,
    except where horizontal gradients are limited to preserve
    monotonicity.
(3) There are economies of scale.  Transporting a single field
    is rather expensive, but additional fields have a relatively
    low marginal cost.

The following generic conservation equations may be solved:

           dm/dt = del*(u*m)             (0)
      d(m*T1)/dt = del*(u*m*T1)          (1)
   d(m*T1*T2)/dt = del*(u*m*T1*T2)       (2)
d(m*T1*T2*T3)/dt = del*(u*m*T1*T2*T3)    (3)

where d is a partial derivative, del is the 2D divergence operator,
u is the horizontal velocity, m is the mass density field, and
T1, T2, and T3 are tracers.

In CICE, these equations have the form

              da/dt = del*(u*a)          (4)
dv/dt =   d(a*h)/dt = del*(u*a*h)        (5)
de/dt = d(a*h*q)/dt = del*(u*a*h*q)      (6)
           d(aT)/dt = del*(u*a*t)        (7)

where a = fractional ice area, v = ice/snow volume, h = v/a = thickness,
e = ice/snow internal energy (J/m^2), q = e/v = internal energy per
unit volume (J/m^3), and T is a tracer.  These equations express
conservation of ice area, volume, internal energy, and area-weighted
tracer, respectively.

(Note: In CICE, a, v and e are prognostic quantities from which
 h and q are diagnosed.  The remapping routine works with tracers,
 which means that h and q must be derived from a, v, and e before
 calling the remapping routine.)

Earlier versions of CICE assumed fixed ice and snow density.
Beginning with CICE 4.0, the ice and snow density can be variable.
In this case, equations (5) and (6) are replaced by

dv/dt =        d(a*h)/dt = del*(u*a*h)          (8)
dm/dt =    d(a*h*rho)/dt = del*(u*a*h*rho)      (9)
de/dt = d(a*h*rho*qm)/dt = del*(u*a*h*rho*qm)   (10)

where rho = density and qm = internal energy per unit mass (J/kg).
Eq. (9) expresses mass conservation, which in the variable-density
case is no longer equivalent to volume conservation (8).

Tracers satisfying equations of the form (1) are called "type 1."
In CICE the paradigmatic type 1 tracers are hi and hs.

Tracers satisfying equations of the form (2) are called "type 2".
The paradigmatic type 2 tracers are qi and qs (or rhoi and rhos
 in the variable-density case).

Tracers satisfying equations of the form (3) are called "type 3."
The paradigmatic type 3 tracers are qmi and qms in the variable-density
case.  There are no such tracers in the constant-density case.

The fields a, T1, and T2 are reconstructed in each grid cell with
2nd-order accuracy.  T3 is reconstructed with 1st-order accuracy
(i.e., it is transported in upwind fashion) in order to avoid
additional mathematical complexity.

The mass-like field lives in the array "mm" (shorthand for mean
mass) and the tracers fields in the array "tm" (mean tracers).
In order to transport tracers correctly, the remapping routine
needs to know the tracers types and relationships.  This is done
as follows:

Each field in the "tm" array is assigned an index, 1:ntrace.
(Note: ntrace is not the same as ntrcr, the number of tracers
in the trcrn state variable array.  For remapping purposes we
have additional tracers hi and hs.)

The tracer types (1,2,3) are contained in the "tracer_type" array.
For standard CICE:

    tracer_type = (1 1 1 2 2 2 2 2)

Type 2 and type 3 tracers are said to depend on type 1 tracers.
For instance, qi depends on hi, which is to say that
there is a conservation equation of the form (2) or (6).
Thus we define a "depend" array.  For standard CICE:

         depend = (0 0 0 1 1 1 1 2)

which implies that elements 1-3 (hi, hs, Ts) are type 1,
elements 4-7 (qi) depend on element 1 (hi), and element 8 (qs)
depends on element 2 (hs).

We also define a logical array "has_dependents".  In standard CICE:

 has_dependents = (T T F F F F F F),

which means that only elements 1 and 2 (hi and hs) have dependent
tracers.

For the variable-density case, things are a bit more complicated.
Suppose we have 4 variable-density ice layers and one variable-
density snow layer.  Then the indexing is as follows:
1    = hi
2    = hs
3    = Ts
4-7  = rhoi
8    = rhos
9-12 = qmi
13   = qms

The key arrays are:

   tracer_type = (1 1 1 2 2 2 2 2 3 3 3 3 3)

        depend = (0 0 0 1 1 1 1 2 4 5 6 7 8)

has_dependents = (T T F T T T T T F F F F F)

which imply that hi and hs are type 1 with dependents rhoi and rhos,
while rhoi and rhos are type 2 with dependents qmi and qms.

Tracers added to the ntrcr array are handled automatically
by the remapping with little extra coding.  It is necessary
only to provide the correct type and dependency information.

When using this routine in other models, most of the tracer dependency
apparatus may be irrelevant.  In a layered ocean model, for example,
the transported fields are the layer thickness h (the mass density
field) and two or more tracers (T, S, and various trace species).
Suppose there are just two tracers, T and S.  Then the tracer arrays
have the values:

   tracer_type = (1 1)
        depend = (0 0)
has_dependents = (F F)

which is to say that all tracer transport equations are of the form (1).

The tracer dependency arrays are optional input arguments for the
main remapping subroutine.  If these arrays are not passed in, they
take on the default values tracer_type(:) = 1, depend(:) = 0, and
has_dependents(:) = F, which are appropriate for most purposes.

Another optional argument is integral_order.  If integral_order = 1,
then the triangle integrals are exact for linear functions of x and y.
If integral_order = 2, these integrals are exact for both linear and
quadratic functions.  If integral_order = 3, integrals are exact for
cubic functions as well.  If all tracers are of type 1, then the
integrals of mass*tracer are quadratic, and integral_order = 2 is
sufficient.  In CICE, where there are type 2 tracers, we integrate
functions of the form mass*tracer1*tracer2.  Thus integral_order = 3
is required for exactness, though integral_order = 2 may be good enough
in practice.

Finally, a few words about the edgearea fields:

In earlier versions of this scheme, the divergence of the velocity
field implied by the remapping was, in general, different from the
value of del*u computed in the dynamics.  For energetic consistency
(in CICE as well as in layered ocean models such as HYPOP),
these two values should agree.  This can be ensured by using the
edge flux adjustment (EFA) method by setting l_edge_flux_adj = T.
The EFA method specifies the area transported across each grid cell
edge in the arrays edgearea_e and edgearea_n.  The departure regions
are then tweaked, following an idea by Mats Bentsen, such that they
have the desired area.  If l_edge_flux_adj = F, these regions are
not tweaked, and the edgearea arrays are output variables.
"""

from functools import partial
from itertools import groupby

import jax
import jax.numpy as jnp

from cicecore.cicedyn.general import ice_init
from icepack.columnphysics.icepack_itd import column_sum

# number of groups of triangles that
# contribute transports across each edge
ngroups = 6
nvert = 3  # number of vertices in a triangle

# for triangle integral formulas
p5625m = -9.0 / 16.0
p52083 = 25.0 / 48.0

# if true, prescribe area flux across each edge
# if false, area flux is determined internally
# and is passed out
l_edge_flux_adj = False

bugcheck = False


def init_remap():
    """Grid quantities used by the remapping transport scheme.

    Note: Arrays needed for nonuniform grids has been deleted.
          They can be found in version 6.5 and earlier

    author William H. Lipscomb, LANL
    """
    global l_edge_flux_adj
    #-------------------------------------------------------------------
    # Set logical l_edge_flux_adj depending of the grid type.
    #
    # If l_edge_flux_adj is true, the area of each departure region is
    # computed in advance (e.g., by taking the divergence of the
    # velocity field and passed to locate_triangles.  The departure
    # regions are adjusted to obtain the desired area.
    # If false, edgearea is computed in locate_triangles and passed out.
    #
    # l_edge_flux_adj = .false. has been the default approach in CICE.
    # It is used like this for the B-grid. However, idealized tests with
    # the C-grid have shown that l_edge_flux_adj = .false. leads to a
    # checkerboard pattern in prognostic fields such as aice Using
    # l_edge_flux_adj = .true. eliminates the checkerboard pattern in
    # C-grid simulations (Lemieux et al. 2024).
    #-------------------------------------------------------------------
    grid_ice = str(ice_init.grid_ice).strip()
    if grid_ice == "CD" or grid_ice == "C":
        l_edge_flux_adj = True
    else:
        l_edge_flux_adj = False


def limited_gradient(phi, phimask, cnx, cny, bounds, nghost):
    """Fortran limited_gradient in scaled coordinates."""
    ilo, ihi, jlo, jhi = bounds
    home = phi[1:-1, 1:-1]
    neighbors = []
    for di, dj in ((-1, 1), (0, 1), (1, 1), (-1, 0),
                   (1, 0), (-1, -1), (0, -1), (1, -1)):
        neighbor = phi[1 + di:phi.shape[0] - 1 + di,
                       1 + dj:phi.shape[1] - 1 + dj]
        mask = phimask[1 + di:phi.shape[0] - 1 + di,
                       1 + dj:phi.shape[1] - 1 + dj]
        neighbors.append(mask * neighbor + (1.0 - mask) * home)
    nw, north, ne, west, east, sw, south, se = neighbors
    gxtmp = 0.5 * (east - west)
    gytmp = 0.5 * (north - south)
    pmn = jnp.minimum(jnp.minimum(jnp.minimum(nw, north), jnp.minimum(ne, west)),
                      jnp.minimum(jnp.minimum(home, east),
                                  jnp.minimum(jnp.minimum(sw, south), se))) - home
    pmx = jnp.maximum(jnp.maximum(jnp.maximum(nw, north), jnp.maximum(ne, west)),
                      jnp.maximum(jnp.maximum(home, east),
                                  jnp.maximum(jnp.maximum(sw, south), se))) - home
    cx = cnx[1:-1, 1:-1]
    cy = cny[1:-1, 1:-1]
    w1 = (0.5 - cx) * gxtmp + (0.5 - cy) * gytmp
    w2 = (0.5 - cx) * gxtmp - (0.5 + cy) * gytmp
    w3 = -(0.5 + cx) * gxtmp - (0.5 + cy) * gytmp
    w4 = (0.5 - cy) * gytmp - (0.5 + cx) * gxtmp
    qmn = jnp.minimum(jnp.minimum(w1, w2), jnp.minimum(w3, w4))
    qmx = jnp.maximum(jnp.maximum(w1, w2), jnp.maximum(w3, w4))
    lower = jnp.where(jnp.abs(qmn) > jnp.abs(pmn),
                      jnp.maximum(0.0, pmn / jnp.where(qmn != 0.0, qmn, 1.0)), 1.0)
    upper = jnp.where(jnp.abs(qmx) > jnp.abs(pmx),
                      jnp.maximum(0.0, pmx / jnp.where(qmx != 0.0, qmx, 1.0)), 1.0)
    factor = jnp.minimum(lower, upper)
    ii = jnp.arange(1, phi.shape[0] - 1)[:, None]
    jj = jnp.arange(1, phi.shape[1] - 1)[None, :]
    active = (ii >= ilo - nghost + 1) & (ii <= ihi + nghost - 1)
    active &= (jj >= jlo - nghost + 1) & (jj <= jhi + nghost - 1)
    active &= phimask[1:-1, 1:-1] > 0.0
    gx = jnp.zeros_like(phi).at[1:-1, 1:-1].set(jnp.where(active, factor * gxtmp, 0.0))
    gy = jnp.zeros_like(phi).at[1:-1, 1:-1].set(jnp.where(active, factor * gytmp, 0.0))
    return gx, gy


def construct_fields(mm, tm, hm, tracer_type, depend, has_dependents,
                     puny, bounds, nghost):
    """Reconstruct limited linear mass and tracer fields in each cell."""
    zeros = jnp.zeros_like(mm)
    mmask = (mm > puny).astype(mm.dtype)
    mx, my = limited_gradient(mm, hm, zeros, zeros, bounds, nghost)
    mc = jnp.where(mmask > 0.0, mm, 0.0)
    safe_mm = jnp.where(mmask > 0.0, mm, 1.0)
    mxav = jnp.where(mmask > 0.0, mx * (1.0 / 12.0) / safe_mm, 0.0)
    myav = jnp.where(mmask > 0.0, my * (1.0 / 12.0) / safe_mm, 0.0)
    if tm is None:
        return mc, mx, my, None, None, None
    parents = jnp.asarray(depend, dtype=jnp.int32) - 1

    def reconstruct(nt, carry, kind, has_children):
        tc, tx, ty, mtxav, mtyav = carry
        field = tm[..., nt]

        def independent_tracer(_):
            gx, gy = limited_gradient(field, mmask, mxav, myav, bounds, nghost)
            center = jnp.where(mmask > 0.0, field - gx * mxav - gy * myav, 0.0)

            def parent_moments(_):
                tracer_mask = mmask * (jnp.abs(field) > puny)
                safe_product = jnp.where(tracer_mask > 0.0, mm * field, 1.0)
                reciprocal = 1.0 / safe_product
                center_x = jnp.where(tracer_mask > 0.0,
                                     (mc * gx + mx * center) * (1.0 / 12.0) * reciprocal, 0.0)
                center_y = jnp.where(tracer_mask > 0.0,
                                     (mc * gy + my * center) * (1.0 / 12.0) * reciprocal, 0.0)
                return center_x, center_y

            center_x, center_y = parent_moments(None) if has_children else (zeros, zeros)
            return center, gx, gy, center_x, center_y

        def dependent_tracer(_):
            parent = parents[nt]
            tracer_mask = mmask * (jnp.abs(tm[..., parent]) > puny)
            gx, gy = limited_gradient(field, tracer_mask,
                                      mtxav[..., parent], mtyav[..., parent], bounds, nghost)
            center = jnp.where(mmask > 0.0,
                               field - gx * mtxav[..., parent] - gy * mtyav[..., parent], 0.0)
            return center, gx, gy, zeros, zeros

        def twice_dependent_tracer(_):
            center = jnp.where(mmask > 0.0, field, 0.0)
            return center, zeros, zeros, zeros, zeros

        values = (independent_tracer, dependent_tracer, twice_dependent_tracer)[kind - 1](None)
        return tuple(array.at[..., nt].set(value)
                     for array, value in zip(carry, values))

    # Compile one branch per contiguous layout run; parents still precede children.
    carry = tuple(jnp.zeros_like(tm) for _ in range(5))
    layout = tuple(zip(tracer_type, has_dependents))
    for (kind, has_children), group in groupby(range(len(layout)), key=layout.__getitem__):
        indices = tuple(group)
        carry = jax.lax.fori_loop(
            indices[0], indices[-1] + 1,
            partial(reconstruct, kind=kind, has_children=has_children), carry)
    tc, tx, ty, _, _ = carry
    return mc, mx, my, tc, tx, ty


def departure_points(uvel, vvel, dxu, dyu, HTN, HTE, dt, bounds, nghost):
    """Backtrace corner trajectories with Fortran's midpoint velocity."""
    ilo, ihi, jlo, jhi = bounds
    ii = jnp.arange(uvel.shape[0])[:, None]
    jj = jnp.arange(uvel.shape[1])[None, :]
    active = (ii >= ilo - nghost + 1) & (ii <= ihi + nghost - 1)
    active &= (jj >= jlo - nghost + 1) & (jj <= jhi + nghost - 1)
    dx = -dt * uvel
    dy = -dt * vvel
    east = jnp.pad(HTN[1:, :], ((0, 1), (0, 0)))
    north = jnp.pad(HTE[:, 1:], ((0, 0), (0, 1)))
    failed = active & ((dx < -HTN) | (dx > east) | (dy < -HTE) | (dy > north))
    scaled_x = dx / jnp.where(dxu != 0.0, dxu, 1.0)
    scaled_y = dy / jnp.where(dyu != 0.0, dyu, 1.0)
    mpx = 0.5 * scaled_x
    mpy = 0.5 * scaled_y
    ii2 = ii + (mpx >= 0.0).astype(jnp.int32)
    jj2 = jj + (mpy >= 0.0).astype(jnp.int32)
    mpxt = mpx + jnp.where(mpx >= 0.0, -0.5, 0.5)
    mpyt = mpy + jnp.where(mpy >= 0.0, -0.5, 0.5)
    i0 = jnp.clip(ii2 - 1, 0, uvel.shape[0] - 1)
    j0 = jnp.clip(jj2 - 1, 0, uvel.shape[1] - 1)
    i1 = jnp.clip(ii2, 0, uvel.shape[0] - 1)
    j1 = jnp.clip(jj2, 0, uvel.shape[1] - 1)
    ump = (uvel[i0, j0] * (mpxt - 0.5) * (mpyt - 0.5)
           - uvel[i1, j0] * (mpxt + 0.5) * (mpyt - 0.5)
           + uvel[i1, j1] * (mpxt + 0.5) * (mpyt + 0.5)
           - uvel[i0, j1] * (mpxt - 0.5) * (mpyt + 0.5))
    vmp = (vvel[i0, j0] * (mpxt - 0.5) * (mpyt - 0.5)
           - vvel[i1, j0] * (mpxt + 0.5) * (mpyt - 0.5)
           + vvel[i1, j1] * (mpxt + 0.5) * (mpyt + 0.5)
           - vvel[i0, j1] * (mpxt - 0.5) * (mpyt + 0.5))
    nonzero = active & ((uvel != 0.0) | (vvel != 0.0))
    return (jnp.where(active, jnp.where(nonzero, -dt * ump, dx), 0.0),
            jnp.where(active, jnp.where(nonzero, -dt * vmp, dy), 0.0), failed)


def _set_triangle(xp, yp, iflux, jflux, areafact, group, condition,
                  vertices, shift, factor, ii, jj):
    """Record one of the Fortran locate_triangles cases."""
    x = jnp.stack((vertices[0], vertices[2], vertices[4]), axis=-1)
    y = jnp.stack((vertices[1], vertices[3], vertices[5]), axis=-1)
    xp = xp.at[..., group, :].set(jnp.where(condition[..., None], x, xp[..., group, :]))
    yp = yp.at[..., group, :].set(jnp.where(condition[..., None], y, yp[..., group, :]))
    iflux = iflux.at[..., group].set(jnp.where(condition, ii + shift[0], iflux[..., group]))
    jflux = jflux.at[..., group].set(jnp.where(condition, jj + shift[1], jflux[..., group]))
    areafact = areafact.at[..., group].set(jnp.where(condition, factor, areafact[..., group]))
    return xp, yp, iflux, jflux, areafact


def locate_triangles(dpx, dpy, dxu, dyu, earea, narea,
                     bounds, edge, puny, eps16, edgearea=None):
    """Departure polygons, including the C/CD edge flux adjustment."""
    nx, ny = dpx.shape
    ilo, ihi, jlo, jhi = bounds
    ii, jj = jnp.meshgrid(jnp.arange(nx), jnp.arange(ny), indexing="ij")
    dx = dpx / jnp.where(dxu != 0.0, dxu, 1.0)
    dy = dpy / jnp.where(dyu != 0.0, dyu, 1.0)
    if edge == "north":
        il = jnp.clip(ii - 1, 0, nx - 1)
        xdl, ydl = -0.5 + dx[il, jj], dy[il, jj]
        xdr, ydr = 0.5 + dx, dy
        fac = narea
        other = earea
        tl, bl, tr, br, tc, bc = (-1, 1), (-1, 0), (1, 1), (1, 0), (0, 1), (0, 0)
        left, right = (-1, 0), (1, 0)
        etl, ebl, etr, ebr = (-1, 1), (-1, 0), (0, 1), (0, 0)
        active = (ii >= ilo) & (ii <= ihi) & (jj >= jlo - 1) & (jj <= jhi)
    else:
        js = jnp.clip(jj - 1, 0, ny - 1)
        xdl, ydl = -0.5 - dy, dx
        xdr, ydr = 0.5 - dy[ii, js], dx[ii, js]
        fac = earea
        other = narea
        tl, bl, tr, br, tc, bc = (1, 1), (0, 1), (1, -1), (0, -1), (1, 0), (0, 0)
        left, right = (0, 1), (0, -1)
        etl, ebl, etr, ebr = (1, 0), (0, 0), (1, -1), (0, -1)
        active = (ii >= ilo - 1) & (ii <= ihi) & (jj >= jlo) & (jj <= jhi)

    def area_at(field, shift):
        return field[jnp.clip(ii + shift[0], 0, nx - 1),
                     jnp.clip(jj + shift[1], 0, ny - 1)]

    xdm = 0.5 * (xdr + xdl)
    ydm = 0.5 * (ydr + ydl)
    xil = jnp.full_like(xdl, -0.5)
    xir = jnp.full_like(xdl, 0.5)
    dleft = jnp.where(xdm != xdl, xdm - xdl, 1.0)
    dright = jnp.where(xdr != xdm, xdr - xdm, 1.0)
    yil = (-0.5 * (ydm - ydl) + xdm * ydl - xdl * ydm) / dleft
    yir = (0.5 * (ydr - ydm) - xdm * ydr + xdr * ydm) / dright
    md = (ydr - ydl) / jnp.where(xdr != xdl, xdr - xdl, 1.0)
    xic = jnp.where(jnp.abs(md) > puny,
                    xdl - ydl / jnp.where(md != 0.0, md, 1.0), 0.0)
    zero = jnp.zeros_like(xdl)
    half_l, half_r = jnp.full_like(xdl, -0.5), jnp.full_like(xdl, 0.5)
    xp = jnp.zeros((nx, ny, 6, 3), dtype=dpx.dtype)
    yp = jnp.zeros_like(xp)
    iflux = jnp.broadcast_to(ii[..., None], (nx, ny, 6))
    jflux = jnp.broadcast_to(jj[..., None], (nx, ny, 6))
    areafact = jnp.zeros((nx, ny, 6), dtype=dpx.dtype)

    def put(group, condition, vertices, shift, factor):
        nonlocal xp, yp, iflux, jflux, areafact
        xp, yp, iflux, jflux, areafact = _set_triangle(
            xp, yp, iflux, jflux, areafact, group, active & condition,
            vertices, shift, factor, ii, jj)

    # TL/BL and TR/BR triangles, including split triangles when the
    # departure segment crosses a neighboring cell boundary.
    a = (xdl < -0.5) & (yil > 0.0) & (ydl >= 0.0)
    put(0, a, (half_l, zero, xil, yil, xdl, ydl), tl, -area_at(other, etl))
    a = (xdl < -0.5) & (yil < 0.0) & (ydl < 0.0)
    put(0, a, (half_l, zero, xdl, ydl, xil, yil), bl, area_at(other, ebl))
    a = (xdl < -0.5) & (yil < 0.0) & (ydl >= 0.0)
    put(0, a, (half_l, zero, xdl, ydl, xic, zero), tl, area_at(fac, left))
    put(2, a, (half_l, zero, xic, zero, xil, yil), bl, area_at(other, ebl))
    a = (xdl < -0.5) & (yil > 0.0) & (ydl < 0.0)
    put(2, a, (half_l, zero, xil, yil, xic, zero), tl, -area_at(other, etl))
    put(0, a, (half_l, zero, xic, zero, xdl, ydl), bl, -area_at(fac, left))

    a = (xdr >= 0.5) & (yir > 0.0) & (ydr >= 0.0)
    put(1, a, (half_r, zero, xdr, ydr, xir, yir), tr, -area_at(other, etr))
    a = (xdr >= 0.5) & (yir < 0.0) & (ydr < 0.0)
    put(1, a, (half_r, zero, xir, yir, xdr, ydr), br, area_at(other, ebr))
    a = (xdr >= 0.5) & (yir < 0.0) & (ydr >= 0.0)
    put(1, a, (half_r, zero, xic, zero, xdr, ydr), tr, area_at(fac, right))
    put(2, a, (half_r, zero, xir, yir, xic, zero), br, area_at(other, ebr))
    a = (xdr >= 0.5) & (yir > 0.0) & (ydr < 0.0)
    put(2, a, (half_r, zero, xic, zero, xir, yir), tr, -area_at(other, etr))
    put(1, a, (half_r, zero, xdr, ydr, xic, zero), br, -area_at(fac, right))

    left_out = xdl < -0.5
    right_out = xdr > 0.5
    xdl = jnp.where(left_out, xil, xdl)
    ydl = jnp.where(left_out, yil, ydl)
    xdr = jnp.where(right_out, xir, xdr)
    ydr = jnp.where(right_out, yir, ydr)
    xicl, xicr = xic, xic
    if edgearea is not None:
        # For l_edge_flux_adj = T, shift the midpoint so that the departure
        # region has the prescribed area
        # Sum the areas of the left and right triangles.
        # Note that yp(i,j,1,ng) = 0 for all triangles, so we can
        # drop those terms from the area formula.
        areas = 0.5 * ((xp[..., 1] - xp[..., 0]) * yp[..., 2]
                        - yp[..., 1] * (xp[..., 2] - xp[..., 0])) * areafact
        remaining = edgearea - areas[..., 0] - areas[..., 1] - areas[..., 2]
        # Check whether the central triangles lie in one grid cell or two.
        # If all are in one grid cell, then adjust the area of the central
        # region so that the sum of all triangle areas is equal to the
        # prescribed value.
        # If two triangles are in one grid cell and one is in the other,
        # then compute the area of the lone triangle. Then adjust
        # the area of the remaining two-triangle region so that the sum of
        # all triangle areas has the prescribed value.
        same = ydl * ydr >= 0.0  # Both DPs lie on same side of x-axis
        fix_left = ~same & (xic < 0.0) & (xic > -0.5)  # fix ICL = IC
        fix_right = ~same & (xic >= 0.0) & (xic < 0.5)  # fix ICR = IR
        # compute required area of central departure region
        w1 = 2.0 * remaining / fac + (xdr + 0.5) * ydl + (0.5 - xdl) * ydr
        w2 = (xdr - xdl)**2 + (ydr - ydl)**2
        w = w1 / jnp.where(w2 != 0.0, w2, 1.0)
        xnew = xdm + (ydr - ydl) * w
        ynew = ydm - (xdr - xdl) * w
        # compute area of (lone) triangle adjacent to left corner
        area4 = 0.5 * (-0.5 - xic) * ydl * fac
        w1 = 2.0 * (remaining - area4) / fac + (0.5 - xic) * ydr
        w2 = (xdr - xic)**2 + ydr**2
        w = w1 / jnp.where(w2 != 0.0, w2, 1.0)
        xnew = jnp.where(fix_left, 0.5 * (xdr + xic) + ydr * w, xnew)
        ynew = jnp.where(fix_left, 0.5 * ydr - (xdr - xic) * w, ynew)
        # compute area of (lone) triangle adjacent to right corner
        area4 = 0.5 * (xic - 0.5) * ydr * fac
        w1 = 2.0 * (remaining - area4) / fac + (xic + 0.5) * ydl
        w2 = (xic - xdl)**2 + ydl**2
        w = w1 / jnp.where(w2 != 0.0, w2, 1.0)
        xnew = jnp.where(fix_right, 0.5 * (xic + xdl) - ydl * w, xnew)
        ynew = jnp.where(fix_right, 0.5 * ydl - (xic - xdl) * w, ynew)
        changed = same | fix_left | fix_right
        xdm, ydm = jnp.where(changed, xnew, xdm), jnp.where(changed, ynew, ydm)
        # compute left and right intersection points
        dl, dr = xdm - xdl, xdr - xdm
        dyl, dyr = ydm - ydl, ydr - ydm
        # Use inverse slopes so vertical segments intersect at their own x.
        il = jnp.where(jnp.abs(dyl) > puny * jnp.abs(dl),
            xdl - ydl * dl / jnp.where(dyl != 0.0, dyl, 1.0), 0.0)
        ir = jnp.where(jnp.abs(dyr) > puny * jnp.abs(dr),
            xdr - ydr * dr / jnp.where(dyr != 0.0, dyr, 1.0), 0.0)
        xicl = jnp.where(same | fix_right, il, xicl)
        xicr = jnp.where(same | fix_left, ir, xicr)
    same_top = (ydl >= 0.0) & (ydr >= 0.0)
    same_bottom = (ydl <= 0.0) & (ydr <= 0.0)
    a = same_top & (ydm >= 0.0)
    put(3, a, (half_l, zero, half_r, zero, xdl, ydl), tc, -fac)
    put(4, a, (half_r, zero, xdr, ydr, xdl, ydl), tc, -fac)
    put(5, a, (xdl, ydl, xdr, ydr, xdm, ydm), tc, -fac)
    a = same_top & (ydm < 0.0)
    put(3, a, (half_l, zero, xicl, zero, xdl, ydl), tc, -fac)
    put(4, a, (half_r, zero, xdr, ydr, xicr, zero), tc, -fac)
    put(5, a, (xicr, zero, xicl, zero, xdm, ydm), bc, fac)
    a = same_bottom & (ydm <= 0.0)
    put(3, a, (half_l, zero, xdl, ydl, half_r, zero), bc, fac)
    put(4, a, (half_r, zero, xdl, ydl, xdr, ydr), bc, fac)
    put(5, a, (xdl, ydl, xdm, ydm, xdr, ydr), bc, fac)
    a = same_bottom & (ydm > 0.0)
    put(3, a, (half_l, zero, xdl, ydl, xicl, zero), bc, fac)
    put(4, a, (half_r, zero, xicr, zero, xdr, ydr), bc, fac)
    put(5, a, (xicl, zero, xicr, zero, xdm, ydm), tc, -fac)

    # Cross-axis cases are completed below in the same Fortran order.
    a = (ydl > 0.0) & (ydr < 0.0) & (xic >= 0.0) & (ydm >= 0.0)
    put(3, a, (half_l, zero, xicr, zero, xdl, ydl), tc, -fac)
    put(4, a, (half_r, zero, xicr, zero, xdr, ydr), bc, fac)
    put(5, a, (xdl, ydl, xicr, zero, xdm, ydm), tc, -fac)
    a = (ydl > 0.0) & (ydr < 0.0) & (xic >= 0.0) & (ydm < 0.0)
    put(3, a, (half_l, zero, xicl, zero, xdl, ydl), tc, -fac)
    put(4, a, (half_r, zero, xicr, zero, xdr, ydr), bc, fac)
    put(5, a, (xicr, zero, xicl, zero, xdm, ydm), bc, fac)
    a = (ydl > 0.0) & (ydr < 0.0) & (xic < 0.0) & (ydm < 0.0)
    put(3, a, (half_l, zero, xicl, zero, xdl, ydl), tc, -fac)
    put(4, a, (half_r, zero, xicl, zero, xdr, ydr), bc, fac)
    put(5, a, (xdr, ydr, xicl, zero, xdm, ydm), bc, fac)
    a = (ydl > 0.0) & (ydr < 0.0) & (xic < 0.0) & (ydm >= 0.0)
    put(3, a, (half_l, zero, xicl, zero, xdl, ydl), tc, -fac)
    put(4, a, (half_r, zero, xicr, zero, xdr, ydr), bc, fac)
    put(5, a, (xicl, zero, xicr, zero, xdm, ydm), tc, -fac)
    a = (ydl < 0.0) & (ydr > 0.0) & (xic < 0.0) & (ydm >= 0.0)
    put(3, a, (half_l, zero, xdl, ydl, xicl, zero), bc, fac)
    put(4, a, (half_r, zero, xdr, ydr, xicl, zero), tc, -fac)
    put(5, a, (xicl, zero, xdr, ydr, xdm, ydm), tc, -fac)
    a = (ydl < 0.0) & (ydr > 0.0) & (xic < 0.0) & (ydm < 0.0)
    put(3, a, (half_l, zero, xdl, ydl, xicl, zero), bc, fac)
    put(4, a, (half_r, zero, xdr, ydr, xicr, zero), tc, -fac)
    put(5, a, (xicr, zero, xicl, zero, xdm, ydm), bc, fac)
    a = (ydl < 0.0) & (ydr > 0.0) & (xic >= 0.0) & (ydm < 0.0)
    put(3, a, (half_l, zero, xdl, ydl, xicr, zero), bc, fac)
    put(4, a, (half_r, zero, xdr, ydr, xicr, zero), tc, -fac)
    put(5, a, (xicr, zero, xdl, ydl, xdm, ydm), bc, fac)
    a = (ydl < 0.0) & (ydr > 0.0) & (xic >= 0.0) & (ydm >= 0.0)
    put(3, a, (half_l, zero, xdl, ydl, xicl, zero), bc, fac)
    put(4, a, (half_r, zero, xdr, ydr, xicr, zero), tc, -fac)
    put(5, a, (xicl, zero, xicr, zero, xdm, ydm), tc, -fac)

    triarea = 0.5 * ((xp[..., 1] - xp[..., 0]) * (yp[..., 2] - yp[..., 0])
                      - (yp[..., 1] - yp[..., 0]) * (xp[..., 2] - xp[..., 0])) * areafact
    triarea = jnp.where(jnp.abs(triarea) < eps16 * fac[..., None], 0.0, triarea)
    shift_i = iflux - ii[..., None]
    shift_j = jflux - jj[..., None]
    if edge == "north":
        xp = xp - shift_i[..., None]
        yp = yp + 0.5 - shift_j[..., None]
    else:
        old_x = xp
        xp = yp + 0.5 - shift_i[..., None]
        yp = -old_x - shift_j[..., None]
    return xp, yp, iflux, jflux, triarea


def transport_integrals(mc, mx, my, tc, tx, ty, geometry,
                        tracer_type, depend):
    """Cubic triangle integrals of mass and its dependent tracers."""
    xp, yp, iflux, jflux, triarea = geometry
    nx, ny = mc.shape
    source_i = jnp.clip(iflux, 0, nx - 1)
    source_j = jnp.clip(jflux, 0, ny - 1)
    xc = (1.0 / 3.0) * (xp[..., 0] + xp[..., 1] + xp[..., 2])
    yc = (1.0 / 3.0) * (yp[..., 0] + yp[..., 1] + yp[..., 2])
    xq = jnp.concatenate((xc[..., None], 0.4 * xp + 0.6 * xc[..., None]), axis=-1)
    yq = jnp.concatenate((yc[..., None], 0.4 * yp + 0.6 * yc[..., None]), axis=-1)
    weights = jnp.array((-9.0 / 16.0, 25.0 / 48.0,
                         25.0 / 48.0, 25.0 / 48.0), dtype=mc.dtype)
    mass = weights * (mc[source_i, source_j][..., None]
                      + xq * mx[source_i, source_j][..., None]
                      + yq * my[source_i, source_j][..., None])
    # Fortran accumulates the four quadrature points, then six groups in order.
    msum = column_sum(4, mass)
    mxsum = column_sum(4, mass * xq)
    mysum = column_sum(4, mass * yq)
    mxxsum = column_sum(4, mass * xq * xq)
    mxysum = column_sum(4, mass * xq * yq)
    myysum = column_sum(4, mass * yq * yq)
    mflx = column_sum(triarea.shape[-1], triarea * msum)
    if tc is None:
        return mflx, None
    parents = jnp.asarray(depend, dtype=jnp.int32) - 1

    def integrate(nt, carry, kind):
        mtsums, mtxsums, mtysums, tracer_fluxes = carry
        center = tc[..., nt][source_i, source_j]

        def independent_tracer(_):
            gx = tx[..., nt][source_i, source_j]
            gy = ty[..., nt][source_i, source_j]
            mtsum = msum * center + mxsum * gx + mysum * gy
            mtxsum = mxsum * center + mxxsum * gx + mxysum * gy
            mtysum = mysum * center + mxysum * gx + myysum * gy
            return mtsum, mtxsum, mtysum

        def dependent_tracer(_):
            parent = parents[nt]
            gx = tx[..., nt][source_i, source_j]
            gy = ty[..., nt][source_i, source_j]
            mtsum = (mtsums[..., parent] * center
                     + mtxsums[..., parent] * gx + mtysums[..., parent] * gy)
            return mtsum, jnp.zeros_like(mtsum), jnp.zeros_like(mtsum)

        def twice_dependent_tracer(_):
            mtsum = mtsums[..., parents[nt]] * center
            return mtsum, jnp.zeros_like(mtsum), jnp.zeros_like(mtsum)

        mtsum, mtxsum, mtysum = (
            independent_tracer, dependent_tracer, twice_dependent_tracer)[kind - 1](None)
        return (mtsums.at[..., nt].set(mtsum),
                mtxsums.at[..., nt].set(mtxsum),
                mtysums.at[..., nt].set(mtysum),
                tracer_fluxes.at[..., nt].set(
                    column_sum(triarea.shape[-1], triarea * mtsum)))

    shape = msum.shape + (len(tracer_type),)
    carry = (jnp.zeros(shape, dtype=mc.dtype),
             jnp.zeros(shape, dtype=mc.dtype),
             jnp.zeros(shape, dtype=mc.dtype), jnp.zeros_like(tc))
    for kind, group in groupby(range(len(tracer_type)), key=tracer_type.__getitem__):
        indices = tuple(group)
        carry = jax.lax.fori_loop(
            indices[0], indices[-1] + 1, partial(integrate, kind=kind), carry)
    _, _, _, tracer_fluxes = carry
    return mflx, tracer_fluxes


def update_fields(mm, tm, east_mass, north_mass, east_tracer, north_tracer,
                  tarear, tracer_type, depend, bounds, puny):
    """Fortran update_fields: conservative divergence and dependent recovery."""
    ilo, ihi, jlo, jhi = bounds
    physical = jnp.zeros_like(mm, dtype=bool).at[ilo:ihi + 1, jlo:jhi + 1].set(True)
    west_mass = jnp.pad(east_mass[:-1, :], ((1, 0), (0, 0)))
    south_mass = jnp.pad(north_mass[:, :-1], ((0, 0), (1, 0)))
    new_mass = mm - (east_mass - west_mass + north_mass - south_mass) * tarear
    failed = physical & (new_mass < -puny)
    new_mass = jnp.where(physical, jnp.maximum(new_mass, 0.0), mm)
    if tm is None:
        return new_mass, None, failed
    west_tracer = jnp.pad(east_tracer[:-1, :, :], ((1, 0), (0, 0), (0, 0)))
    south_tracer = jnp.pad(north_tracer[:, :-1, :], ((0, 0), (1, 0), (0, 0)))
    parents = jnp.asarray(depend, dtype=jnp.int32) - 1

    def recover(nt, new_tracers, kind):
        def independent_tracer(_):
            return mm * tm[..., nt], new_mass

        def dependent_tracer(_):
            parent = parents[nt]
            return (mm * tm[..., parent] * tm[..., nt],
                    new_mass * new_tracers[..., parent])

        def twice_dependent_tracer(_):
            parent = parents[nt]
            grandparent = parents[parent]
            old = (mm * tm[..., grandparent]
                   * tm[..., parent] * tm[..., nt])
            denominator = (new_mass * new_tracers[..., grandparent]
                           * new_tracers[..., parent])
            return old, denominator

        old, denominator = (
            independent_tracer, dependent_tracer, twice_dependent_tracer)[kind - 1](None)
        change = (east_tracer[..., nt] - west_tracer[..., nt]
                  + north_tracer[..., nt] - south_tracer[..., nt]) * tarear
        valid = physical & (denominator != 0.0)
        value = jnp.where(valid, (old - change) /
                          jnp.where(valid, denominator, 1.0), 0.0)
        return new_tracers.at[..., nt].set(jnp.where(physical, value, tm[..., nt]))

    new_tracers = jnp.zeros_like(tm)
    for kind, group in groupby(range(len(tracer_type)), key=tracer_type.__getitem__):
        indices = tuple(group)
        new_tracers = jax.lax.fori_loop(
            indices[0], indices[-1] + 1, partial(recover, kind=kind), new_tracers)
    return new_mass, new_tracers, failed
