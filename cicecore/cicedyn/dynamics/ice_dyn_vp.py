"""Viscous-plastic sea ice dynamics model
Computes ice velocity and deformation

See:

Lemieux, J.-F., Tremblay, B., Thomas, S., Sedlacek, J., and Mysak, L. A. (2008),
Using the preconditioned Generalized Minimum RESidual (GMRES) method to solve
the sea-ice momentum equation, J. Geophys. Res., 113, C10004, doi:10.1029/2007JC004680.

Hibler, W. D., and Ackley, S. F. (1983), Numerical simulation of the Weddell Sea pack ice,
J. Geophys. Res., 88( C5), 2873-2887, doi:10.1029/JC088iC05p02873.

Y. Saad. A Flexible Inner-Outer Preconditioned GMRES Algorithm. SIAM J. Sci. Comput.,
14(2):461-469, 1993. URL: https://doi.org/10.1137/0914028, doi:10.1137/0914028.

C. T. Kelley, Iterative Methods for Linear and Nonlinear Equations, SIAM, 1995.
(https://www.siam.org/books/textbooks/fr16_book.pdf)

Y. Saad, Iterative Methods for Sparse Linear Systems. SIAM, 2003.
(http://www-users.cs.umn.edu/~saad/IterMethBook_2ndEd.pdf)

Walker, H. F., & Ni, P. (2011). Anderson Acceleration for Fixed-Point Iterations.
SIAM Journal on Numerical Analysis, 49(4), 1715-1735. https://doi.org/10.1137/10078356X

Fang, H., & Saad, Y. (2009). Two classes of multisecant methods for nonlinear acceleration.
Numerical Linear Algebra with Applications, 16(3), 197-221. https://doi.org/10.1002/nla.617

Birken, P. (2015) Termination criteria for inexact fixed-point schemes.
Numer. Linear Algebra Appl., 22: 702-716. doi: 10.1002/nla.1982.

author for JAX version: Xinyuan Zhu
XZ: converted from F90 to Python

authors: JF Lemieux, ECCC, Philppe Blain, ECCC

One grid, no block index. ``init_vp`` copies the namelist solver controls
and allocates the compressed-index arrays. Time stepping uses masked JAX
vectors, restarted right-preconditioned GMRES and Picard/Anderson iteration.
"""

import numpy as np
import jax
import jax.numpy as jnp

from cicecore.cicedyn.dynamics import ice_dyn_shared
from cicecore.cicedyn.general import ice_init
from cicecore.shared import ice_calendar
from cicecore.shared import ice_domain_size
from cicecore.shared.ice_kinds_mod import dbl_kind, int_kind, log_kind


def implicit_solver(pytree, static, dt):
    from cicecore.cicedyn.dynamics import ice_dyn_evp
    return ice_dyn_evp.evp(pytree, static, dt, solver="vp")


def stress_vp(velocity, zeta, eta, pressure, grid, shared, dyn):
    """Compute stresses using the viscosities from the nonlinear iteration."""
    div, tension, shear, _ = ice_dyn_shared.strain_rates(
        velocity[..., 0], velocity[..., 1], grid["dxT"], grid["dyT"],
        shared["cxp"], shared["cyp"], shared["cxm"], shared["cym"], shared["iceTmask"], dyn)
    # the stresses                            ! kg/s^2
    # (1) northeast, (2) northwest, (3) southwest, (4) southeast
    stresses = {}
    for stem, value in (("stressp", zeta * div - pressure),
                        ("stressm", eta * tension), ("stress12", 0.5 * eta * shear)):
        for k in range(1, 5):
            stresses[f"{stem}_{k}"] = jnp.where(shared["iceTmask"], value[..., k-1], 0.0)
    return stresses


def stress_divergence(stresses, grid, shared):
    from cicecore.cicedyn.dynamics.ice_dyn_evp import stress_combinations
    # divergence of the internal stress tensor
    terms = stress_combinations(stresses, grid, shared, shared["iceTmask"])
    terms = jnp.pad(terms, ((0, 1), (0, 1), (0, 0)))
    x = grid["uarear"] * (terms[:-1, :-1, 0] + terms[1:, :-1, 1]
        + terms[:-1, 1:, 2] + terms[1:, 1:, 3])
    y = grid["uarear"] * (terms[:-1, :-1, 4] + terms[:-1, 1:, 5]
        + terms[1:, :-1, 6] + terms[1:, 1:, 7])
    return jnp.stack((x, y), axis=-1)


def fgmres(matvec, precondition, rhs, initial, tolerance, maxinner,
           maxouter, ortho_type, monitor=False, label="fgmres"):
    """Flexible right-preconditioned GMRES, including restarts and happy breakdown.

    Fixed-size Arnoldi storage replaces the compressed Fortran vectors. Entries
    outside the active physical domain are zero and do not enter reductions.
    """
    shape = rhs.shape
    b, x = rhs.reshape(-1), initial.reshape(-1)
    n = b.size
    # solution is zero if RHS is zero
    x = jnp.where(jnp.linalg.norm(b) == 0.0, jnp.zeros_like(x), x)
    r = b - matvec(x.reshape(shape)).reshape(-1)
    norm = jnp.linalg.norm(r)
    tol = tolerance * norm

    def restart(_, outer):
        x, residual, norm, nbiter = outer

        def solve(outer):
            x, residual, norm, nbiter = outer
            # Cells with no ice should be zero-initialized
            V = jnp.zeros((maxinner + 1, n), dtype=b.dtype).at[0].set(residual / norm)
            Z = jnp.zeros((maxinner, n), dtype=b.dtype)
            H = jnp.zeros((maxinner + 1, maxinner), dtype=b.dtype)
            cs, sn = jnp.zeros(maxinner, b.dtype), jnp.zeros(maxinner, b.dtype)
            g = jnp.zeros(maxinner + 1, b.dtype).at[0].set(norm)

            def arnoldi(j, inner):
                V, Z, H, cs, sn, g, count, resnorm, stopped = inner

                def advance(inner):
                    V, Z, H, cs, sn, g, count, resnorm, stopped = inner
                    # precondition the current Arnoldi vector
                    z = precondition(V[j].reshape(shape)).reshape(-1)
                    Z = Z.at[j].set(z)
                    w = matvec(z.reshape(shape)).reshape(-1)
                    # Orthogonalize the new vector
                    if ortho_type == "cgs":
                        # First loop of Gram-Schmidt (compute coefficients)
                        h = jnp.where(jnp.arange(maxinner + 1) <= j, V @ w, 0.0)
                        # Second loop of Gram-Schmidt (orthonormalize)
                        w = jax.lax.fori_loop(0, maxinner + 1,
                            lambda k, v: v - h[k] * V[k], w)
                    else:
                        # Modified Gram-Schmidt orthogonalisation process
                        def orthogonalize(k, carry):
                            w, h = carry
                            value = jnp.where(k <= j, jnp.dot(V[k], w), 0.0)
                            return w - value * V[k], h.at[k].set(value)
                        w, h = jax.lax.fori_loop(0, maxinner + 1, orthogonalize,
                                                (w, jnp.zeros(maxinner + 1, b.dtype)))
                    # Compute norm of new Arnoldi vector and update Hessenberg matrix
                    beta = jnp.linalg.norm(w)
                    h = h.at[j + 1].set(beta)
                    # Watch out for happy breakdown
                    V = V.at[j + 1].set(w / jnp.where(beta != 0.0, beta, 1.0))
                    # Apply previous Givens rotation to the last column of the Hessenberg matrix
                    def rotate(k, h):
                        t = h[k]
                        a, d = cs[k] * t + sn[k] * h[k+1], -sn[k] * t + cs[k] * h[k+1]
                        return h.at[k].set(jnp.where(k < j, a, h[k])).at[k+1].set(jnp.where(k < j, d, h[k+1]))
                    h = jax.lax.fori_loop(0, maxinner, rotate, h)
                    # Compute and apply new Givens rotation
                    nu = jnp.sqrt(h[j]**2 + h[j+1]**2)
                    c = h[j] / jnp.where(nu != 0.0, nu, 1.0)
                    s = h[j+1] / jnp.where(nu != 0.0, nu, 1.0)
                    cs, sn = cs.at[j].set(c), sn.at[j].set(s)
                    g = g.at[j+1].set(-s * g[j]).at[j].set(c * g[j])
                    h = h.at[j].set(c * h[j] + s * h[j+1]).at[j+1].set(0.0)
                    H = H.at[:, j].set(h)
                    resnorm = jnp.abs(g[j+1])
                    if monitor:
                        jax.debug.print("monitor_" + label + ": iter={i} residual={r}", i=nbiter+j+1, r=resnorm)
                    return V, Z, H, cs, sn, g, (j+1).astype(count.dtype), resnorm, (resnorm <= tol) | (nu == 0.0) | (beta == 0.0)

                return jax.lax.cond(stopped, lambda x: x, advance, inner)

            V, Z, H, cs, sn, g, count, resnorm, _ = jax.lax.fori_loop(
                0, maxinner, arnoldi, (V, Z, H, cs, sn, g, jnp.int32(0), norm, jnp.bool_(False)))
            # Solve the (now upper triangular) system "hessenberg * sol_hess = rhs_hess"
            def backsub(i, y):
                k = maxinner - 1 - i
                value = (g[k] - jnp.dot(H[k], y)) / jnp.where(H[k, k] != 0.0, H[k, k], 1.0)
                return y.at[k].set(jnp.where(k < count, value, 0.0))
            y = jax.lax.fori_loop(0, maxinner, backsub, jnp.zeros(maxinner, b.dtype))
            # Form linear combination to get new solution iterate
            x = jax.lax.fori_loop(0, maxinner, lambda k, v: v + y[k] * Z[k], x)
            # Apply the Givens rotation in reverse order to g := gamma_(m+1) * e_(m+1).
            # The residual is V_(m+1) * Q_m^T * (gamma_(m+1) * e_(m+1)) (Saad p. 177).
            tail = jnp.zeros_like(g).at[count].set(g[count])
            def undo(i, tail):
                k = maxinner - 1 - i
                a, d = -sn[k] * tail[k+1], cs[k] * tail[k+1]
                return tail.at[k].set(jnp.where(k < count, a, tail[k])).at[k+1].set(jnp.where(k < count, d, tail[k+1]))
            tail = jax.lax.fori_loop(0, maxinner, undo, tail)
            residual = jax.lax.fori_loop(0, maxinner + 1,
                lambda k, v: v + tail[k] * V[k], jnp.zeros_like(b))
            return x, residual, jnp.linalg.norm(residual), nbiter + count

        return jax.lax.cond((norm > tol) & jnp.isfinite(norm), solve, lambda x: x, outer)

    x, _, residual_norm, nbiter = jax.lax.fori_loop(
        0, maxouter, restart, (x, r, norm, jnp.int32(0)))
    return x.reshape(shape), residual_norm, nbiter


def vp_solve(fields, vp_state, strength, Cw, grid, shared, params, dyn, vp_params, static, iglob):
    from cicecore.cicedyn.infrastructure.comm.serial.ice_boundary import ice_halo_update_jax
    options = static["ice_dyn_vp"]
    if options["fpfunc_andacc"] != 1:
        raise ValueError("Fortran VP fixed point function g_2(x) is not implemented (fpfunc_andacc=2)")
    mask = fields["iceXmask"][..., None]
    maskT = shared["iceTmask"][..., None]
    velocity = jnp.where(mask, jnp.stack((fields["uvel"], fields["vvel"]), -1), 0.0)
    # Compute the constant component of b(u,v) i.e. the part of b(u,v) that
    # does not depend on (u,v) and thus do not change during the nonlinear iteration
    bfix = jnp.stack((fields["Xmassdti"] * fields["uvel_init"] + fields["forcex"],
                     fields["Xmassdti"] * fields["vvel_init"] + fields["forcey"]), -1)
    water = jnp.stack((fields["waterx"], fields["watery"]), -1)
    ocean = jnp.stack((fields["uocn"], fields["vocn"]), -1)

    def halo(v):
        return jnp.stack([ice_halo_update_jax(v[..., k], "U", "vector", static, iglob) for k in (0, 1)], -1)

    def coefficients(v, vprev):
        # Calc zetax2, etax2, dPr/dx, dPr/dy, Cb and vrel = f(uprev_k, vprev_k)
        vh = halo(v)
        _, _, _, delta = ice_dyn_shared.strain_rates(
            vh[..., 0], vh[..., 1], grid["dxT"], grid["dyT"], shared["cxp"],
            shared["cyp"], shared["cxm"], shared["cym"], shared["iceTmask"], dyn)
        zeta, eta, pressure = ice_dyn_shared.visc_replpress(
            strength[..., None], shared["DminTarea"][..., None], delta, dyn)
        zeta, eta, pressure = (jnp.where(maskT, a, 0.0) for a in (zeta, eta, pressure))
        vlin = 0.5 * vprev + 0.5 * v if options["use_mean_vrel"] else v
        drag = fields["aiX"] * params["rhow"] * Cw * jnp.sqrt(jnp.sum((ocean - vlin)**2, axis=-1))
        basal = fields["TbU"] / (jnp.sqrt(jnp.sum(vlin**2, axis=-1)) + 5.0e-5)
        return zeta, eta, pressure, drag, basal

    def matvec(v, zeta, eta, drag, basal, update_halo=True):
        vh = halo(v) if update_halo else v
        stress = stress_vp(vh, zeta, eta, jnp.zeros_like(zeta), grid, shared, dyn)
        force = stress_divergence(stress, grid, shared)
        # Form Au and Av
        cca = fields["Xmassdti"] + drag + basal
        ccb = fields["fm"]  # cosw=1, sinw=0
        value = jnp.stack((cca * v[..., 0] - ccb * v[..., 1],
                           cca * v[..., 1] + ccb * v[..., 0]), -1) - force
        return jnp.where(mask, value, 0.0)

    dim = int(options["dim_andacc"])
    history_shape = (max(dim, 1), velocity.size)
    historyF = jnp.zeros(history_shape, velocity.dtype)
    historyG = jnp.zeros_like(historyF)
    zeta, eta, pressure, drag, basal = coefficients(velocity, velocity)
    initial = (velocity, velocity, historyF, historyG, jnp.zeros_like(velocity),
               jnp.zeros_like(velocity), jnp.asarray(0.0, velocity.dtype), jnp.bool_(False),
               zeta, eta, pressure, drag, basal, jnp.asarray(0.0, velocity.dtype), jnp.int32(0))

    def nonlinear(it, carry):
        def advance(carry):
            v, vprev, F, G, old_res, old_g, tol, done, _, _, _, _, _, _, linear_its = carry
            zeta, eta, pressure, drag, basal = coefficients(v, vprev)
            pressure_stress = stress_vp(jnp.zeros_like(v), jnp.zeros_like(zeta),
                                       jnp.zeros_like(eta), pressure, grid, shared, dyn)
            # prepare b vector (RHS)
            rhs = jnp.where(mask, bfix + drag[..., None] * water
                            + stress_divergence(pressure_stress, grid, shared), 0.0)
            residual = rhs - matvec(v, zeta, eta, drag, basal)
            norm = jnp.linalg.norm(residual)
            # Compute relative tolerance at first iteration
            tol = jnp.where(it == 0, vp_params["reltol_nonlin"] * norm, tol)
            done = (norm <= tol) | ~jnp.isfinite(norm)
            if options["monitor_nonlin"]:
                jax.debug.print("monitor_nonlin: iter_nonlin={i} nonlin_res_L2norm={r}", i=it, r=norm)

            def fixed_point(args):
                v, F, G, old_res, old_g, linear_its = args
                diagonal = jnp.ones_like(v)
                if options["precond"] != "ident":
                    # Form the diagonal of the matrix A(u,v).
                    # The stencil spans 3x3 U points. Nine disjoint colors per
                    # component extract its diagonal without forming a dense matrix.
                    # Zero ghosts reproduce the local formDiag_step1/step2 terms.
                    ii, jj = jnp.indices(v.shape[:2])
                    def diagonal_color(k, diagonal):
                        component, color = k // 9, k % 9
                        selected = mask[..., 0] & (ii % 3 == color // 3) & (jj % 3 == color % 3)
                        basis = jnp.zeros_like(v).at[..., component].set(selected.astype(v.dtype))
                        value = matvec(basis, zeta, eta, drag, basal, update_halo=False)
                        return diagonal.at[..., component].set(jnp.where(selected, value[..., component], diagonal[..., component]))
                    diagonal = jax.lax.fori_loop(0, 18, diagonal_color, diagonal)

                def A(x):
                    return matvec(x, zeta, eta, drag, basal)

                def precondition(x):
                    if options["precond"] == "ident":
                        return x  # identity (no preconditioner)
                    if options["precond"] == "diag":
                        return x / diagonal  # Jacobi preconditioner (diagonal)
                    # PGMRES (Jacobi-preconditioned GMRES)
                    # Initialize preconditioned vector to 0 ! TODO: try with wx = vx or vx/diagx
                    result, _, _ = fgmres(A, lambda y: y / diagonal, x, jnp.zeros_like(x),
                        vp_params["reltol_pgmres"], int(options["dim_pgmres"]), int(options["maxits_pgmres"]),
                        options["ortho_type"], options["monitor_pgmres"], "pgmres")
                    return result

                # g_1(x) = FGMRES(A(x), b(x))
                g, _, nbiter = fgmres(A, precondition, rhs, v, vp_params["reltol_fgmres"],
                    int(options["dim_fgmres"]), int(options["maxits_fgmres"]), options["ortho_type"], options["monitor_fgmres"])
                # Compute fixed point residual f(x) = g(x) - x
                res = g - v
                solution = g
                if dim > 0:
                    # Begin Anderson acceleration
                    use = it > int(options["start_andacc"])
                    count = jnp.minimum(jnp.maximum(it - int(options["start_andacc"]), 0), dim)
                    # Delete first column and add column
                    F = jnp.where(use, jnp.concatenate((F[1:], (res - old_res).reshape(1, -1)), axis=0), F)
                    G = jnp.where(use, jnp.concatenate((G[1:], (g - old_g).reshape(1, -1)), axis=0), G)
                    # Solve least square problem for coefficients.
                    # Rebuild the small QR factorization after deleting a column;
                    # empty/dependent columns have zero weight in the fixed-size history.
                    valid = jnp.arange(dim) >= dim - count
                    matrix = jnp.where(valid[:, None], F, 0.0).T
                    def qr_column(k, qr):
                        Q, R = qr
                        def project(j, carry):
                            column, R = carry
                            value = jnp.where(j < k, jnp.dot(Q[:, j], column), 0.0)
                            return column - value * Q[:, j], R.at[j, k].set(value)
                        column, R = jax.lax.fori_loop(0, dim, project, (matrix[:, k], R))
                        norm = jnp.linalg.norm(column)
                        Q = Q.at[:, k].set(column / jnp.where(norm != 0.0, norm, 1.0))
                        return Q, R.at[k, k].set(norm)
                    Q, R = jax.lax.fori_loop(0, dim, qr_column,
                        (jnp.zeros_like(matrix), jnp.zeros((dim, dim), v.dtype)))
                    rhs_tri = Q.T @ res.reshape(-1)
                    def triangular(i, coeffs):
                        k = dim - 1 - i
                        value = (rhs_tri[k] - jnp.dot(R[k], coeffs)) / jnp.where(R[k, k] != 0.0, R[k, k], 1.0)
                        return coeffs.at[k].set(jnp.where(R[k, k] != 0.0, value, 0.0))
                    coeffs = jax.lax.fori_loop(0, dim, triangular, jnp.zeros(dim, v.dtype))
                    accelerated = g.reshape(-1) - G.T @ coeffs
                    # Apply damping: x = x - (1-beta) (res - Q*R*coeffs)
                    beta = vp_params["damping_andacc"]
                    accelerated -= jnp.where((beta > 0.0) & (beta != 1.0),
                        (1.0 - beta) * (res.reshape(-1) - matrix @ coeffs), 0.0)
                    solution = jnp.where(use, accelerated.reshape(v.shape), g)
                return jnp.where(mask, solution, 0.0), F, G, res, g, linear_its + nbiter

            newv, F, G, old_res, old_g, linear_its = jax.lax.cond(
                done, lambda x: x, fixed_point, (v, F, G, old_res, old_g, linear_its))
            return newv, v, F, G, old_res, old_g, tol, done, zeta, eta, pressure, drag, basal, norm, linear_its

        return jax.lax.cond(carry[7], lambda x: x, advance, carry)

    (velocity, _, _, _, _, _, tolerance, done, zeta, eta, pressure, drag, basal,
     residual_norm, linear_its) = jax.lax.fori_loop(0, int(options["maxits_nonlin"]) + 1, nonlinear, initial)
    # Compute stresses
    velocity = halo(velocity)
    stresses = stress_vp(velocity, zeta, eta, pressure, grid, shared, dyn)
    result = dict(fields)
    for name, value in stresses.items():
        result[name] = jnp.where(shared["iceTmask"], value, fields[name])
    result["uvel"], result["vvel"] = velocity[..., 0], velocity[..., 1]
    # Compute seabed stress (diagnostic)
    result["taubx"] = jnp.where(mask[..., 0], -velocity[..., 0] * basal, fields["taubx"])
    result["tauby"] = jnp.where(mask[..., 0], -velocity[..., 1] * basal, fields["tauby"])
    vp_state = {**vp_state, "residual_norm": residual_norm,
                "converged": jnp.isfinite(residual_norm) & (residual_norm <= tolerance),
                "linear_iterations": linear_its}
    return result, vp_state

# namelist parameters

maxits_nonlin = None   # max nb of iteration for nonlinear solver
dim_fgmres = None      # size of fgmres Krylov subspace
dim_pgmres = None      # size of pgmres Krylov subspace
maxits_fgmres = None   # max nb of iteration for fgmres
maxits_pgmres = None   # max nb of iteration for pgmres
fpfunc_andacc = None   # fixed point function for Anderson acceleration:
                       # 1: g(x) = FMGRES(A(x),b(x)), 2: g(x) = x - A(x)x + b(x)
dim_andacc = None      # size of Anderson minimization matrix (number of saved previous residuals)
start_andacc = None    # acceleration delay factor (acceleration starts at this iteration)

monitor_nonlin = None  # print nonlinear residual norm
monitor_fgmres = None  # print fgmres residual norm
monitor_pgmres = None  # print pgmres residual norm
use_mean_vrel = None   # use mean of previous 2 iterates to compute vrel (see Hibler and Ackley 1983)

reltol_nonlin = None   # nonlinear stopping criterion: reltol_nonlin*res(k=0)
reltol_fgmres = None   # fgmres stopping criterion: reltol_fgmres*res(k)
reltol_pgmres = None   # pgmres stopping criterion: reltol_pgmres*res(k)
damping_andacc = None  # damping factor for Anderson acceleration
reltol_andacc = None   # relative tolerance for Anderson acceleration

precond = None         # preconditioner for fgmres: 'ident' (identity), 'diag' (diagonal),
                       # 'pgmres' (Jacobi-preconditioned GMRES)
algo_nonlin = None     # nonlinear algorithm: 'picard' (Picard iteration), 'anderson' (Anderson acceleration)
ortho_type = None      # type of orthogonalization for FGMRES ('cgs' or 'mgs')

# module variables

icellT = None    # no. of cells where iceTmask = .true.
icellU = None    # no. of cells where iceUmask = .true.

indxTi = None    # compressed index in i-direction
indxTj = None    # compressed index in j-direction
indxUi = None    # compressed index in i-direction
indxUj = None    # compressed index in j-direction
residual_norm = None
converged = None
linear_iterations = None


def init_vp():
    """Initialize parameters and variables needed for the vp dynamics

    author: Philippe Blain, ECCC
    """
    global maxits_nonlin, dim_fgmres, dim_pgmres, maxits_fgmres, maxits_pgmres
    global fpfunc_andacc, dim_andacc, start_andacc
    global monitor_nonlin, monitor_fgmres, monitor_pgmres, use_mean_vrel
    global reltol_nonlin, reltol_fgmres, reltol_pgmres, damping_andacc, reltol_andacc
    global precond, algo_nonlin, ortho_type
    global icellT, icellU, indxTi, indxTj, indxUi, indxUj
    global residual_norm, converged, linear_iterations

    # Namelist targets live on ice_init. Copy them onto this module.
    maxits_nonlin = int_kind(ice_init.maxits_nonlin)
    dim_fgmres = int_kind(ice_init.dim_fgmres)
    dim_pgmres = int_kind(ice_init.dim_pgmres)
    maxits_fgmres = int_kind(ice_init.maxits_fgmres)
    maxits_pgmres = int_kind(ice_init.maxits_pgmres)
    fpfunc_andacc = int_kind(ice_init.fpfunc_andacc)
    dim_andacc = int_kind(ice_init.dim_andacc)
    start_andacc = int_kind(ice_init.start_andacc)
    monitor_nonlin = log_kind(ice_init.monitor_nonlin)
    monitor_fgmres = log_kind(ice_init.monitor_fgmres)
    monitor_pgmres = log_kind(ice_init.monitor_pgmres)
    use_mean_vrel = log_kind(ice_init.use_mean_vrel)
    reltol_nonlin = dbl_kind(ice_init.reltol_nonlin)
    reltol_fgmres = dbl_kind(ice_init.reltol_fgmres)
    reltol_pgmres = dbl_kind(ice_init.reltol_pgmres)
    damping_andacc = dbl_kind(ice_init.damping_andacc)
    reltol_andacc = dbl_kind(ice_init.reltol_andacc)
    precond = str(ice_init.precond).strip()
    algo_nonlin = str(ice_init.algo_nonlin).strip()
    ortho_type = str(ice_init.ortho_type).strip()

    ice_dyn_shared.init_dyn_shared(ice_calendar.dt_dyn)

    # Initialize module variables. One grid: the block dimension is 1.
    nx = int(ice_domain_size.nx_global) + 2 * int(ice_domain_size.nghost)
    ny = int(ice_domain_size.ny_global) + 2 * int(ice_domain_size.nghost)
    icellT = np.zeros(1, dtype=int_kind)
    icellU = np.zeros(1, dtype=int_kind)
    compressed = nx * ny
    indxTi = np.zeros(compressed, dtype=int_kind)
    indxTj = np.zeros(compressed, dtype=int_kind)
    indxUi = np.zeros(compressed, dtype=int_kind)
    indxUj = np.zeros(compressed, dtype=int_kind)
    residual_norm = np.asarray(0.0, dtype=dbl_kind)
    converged = np.asarray(True, dtype=log_kind)
    linear_iterations = np.asarray(0, dtype=int_kind)
