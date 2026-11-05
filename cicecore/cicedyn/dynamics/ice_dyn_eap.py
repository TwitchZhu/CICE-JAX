"""Elastic-anisotropic sea ice dynamics model
Computes ice velocity and deformation

See:

Wilchinsky, A.V. and D.L. Feltham (2006). Modelling the rheology of
sea ice as a collection of diamond-shaped floes.
Journal of Non-Newtonian Fluid Mechanics, 138(1), 22-32.

Tsamados, M., D.L. Feltham, and A.V. Wilchinsky (2013). Impact on new
anisotropic rheology on simulations of Arctic sea ice. JGR, 118, 91-107.

author for JAX version: Xinyuan Zhu
XZ: converted from F90 to Python

authors: Michel Tsamados, CPOM
         David Schroeder, CPOM

One grid, no block index. ``init_eap`` builds the lookup table and
allocates the structure-tensor fields. ``read_restart_eap`` reads the
continue-run structure tensor. The JAX time-step driver carries all evolving
tensors explicitly; lookup tables are transferred with the initialized state.
"""

import numpy as np
import jax
import jax.numpy as jnp

from cicecore.cicedyn.dynamics import ice_dyn_shared
from cicecore.cicedyn.general import ice_flux
from cicecore.cicedyn.general import ice_init
from cicecore.cicedyn.infrastructure import ice_domain
from cicecore.cicedyn.infrastructure.comm.serial.ice_boundary import scatter_global
from cicecore.shared import ice_calendar
from cicecore.shared import ice_domain_size
from cicecore.shared import ice_fileunits
from cicecore.shared.ice_constants import (
    c0,
    c1,
    c2,
    c12,
    field_loc_center,
    field_type_scalar,
    p5,
)
from cicecore.shared.ice_kinds_mod import dbl_kind
from icepack.columnphysics import icepack_parameters


def eap(pytree, static, dt):
    from cicecore.cicedyn.dynamics import ice_dyn_evp
    return ice_dyn_evp.evp(pytree, static, dt, solver="eap")


def update_stress_rdg(divu, tension, shear, a11, a12, strength, tables, parameters):
    """Stress updated depending on strain rate and structure tensor."""
    puny = parameters["puny"]
    # compute eigenvalues, eigenvectors and angles for structure tensor, strain rates
    # 1) structure tensor
    a22 = 1.0 - a11
    # gamma: angle between general coordinates and principal axis of A
    # here Tan2gamma = 2 a12 / (a11 - a22)
    active = (jnp.abs(a11 - a22) > puny) | (jnp.abs(a12) > puny)
    denom = jnp.sqrt(jnp.where(active, (a11 - a22)**2 + 4.0 * a12**2, 1.0))
    q11 = jnp.where(active, 0.5 + (a11 - a22) * 0.5 / denom, 1.0)
    q22 = jnp.where(active, 0.5 - (a11 - a22) * 0.5 / denom, puny)
    q12 = jnp.where(active, a12 / denom, puny)
    # rotation Q*atemp*Q^T
    aprime = q11 * a11 + 2.0 * q12 * a12 + q22 * a22
    # make first principal value the largest
    aprime = jnp.maximum(aprime, 1.0 - aprime)
    # 2) strain rate
    d11, d12, d22 = 0.5 * (divu + tension), shear * 0.5, 0.5 * (divu - tension)
    # here Tan2alpha = 2 dtemp12 / (dtemp11 - dtemp22)
    active = (jnp.abs(d11 - d22) > puny) | (jnp.abs(d12) > puny)
    denom = jnp.sqrt(jnp.where(active, (d11 - d22)**2 + 4.0 * d12**2, 1.0))
    qd11 = jnp.where(active, 0.5 + (d11 - d22) * 0.5 / denom, 1.0)
    qd22 = jnp.where(active, 0.5 - (d11 - d22) * 0.5 / denom, puny)
    qd12 = jnp.where(active, d12 / denom, puny)
    d1 = qd11 * d11 + 2.0 * qd12 * d12 + qd22 * d22
    d2 = qd22 * d11 - 2.0 * qd12 * d12 + qd11 * d22
    # In cos and sin values
    # invleng = c1/sqrt(dtemp1*dtemp1 + dtemp2*dtemp2) ! not sure if this is neccessary
    # dtemp1 = dtemp1*invleng
    # dtemp2 = dtemp2*invleng
    x = jnp.where((jnp.abs(d1) > puny) | (jnp.abs(d2) > puny),
                  jnp.where(d1 == 0.0, parameters["pih"], jnp.arctan2(d2, d1)), 0.0)
    # echmod to ensure the angle lies between pi/4 and 9 pi/4
    x = jnp.where(x < parameters["piq"], x + parameters["pi2"], x)
    # echmod require 0 <= x < (nx_yield-1)*dx = 2 pi
    # x = mod(x+pi2, pi2)
    # y: angle between major principal axis of strain rate and structure tensor
    # y = gamma - alpha
    # Expressesed componently with
    # Tany = (Singamma*Cosgamma - Sinalpha*Cosgamma)/(Cos^2gamma - Sin^alpha)
    t1, t2 = q12 - qd12, q11 - qd22
    # invleng = c1/sqrt(Tany_1*Tany_1 + Tany_2*Tany_2) ! not sure if this is neccessary
    # Tany_1 = Tany_1*invleng
    # Tany_2 = Tany_2*invleng
    y = jnp.where((jnp.abs(t1) > puny) | (jnp.abs(t2) > puny),
                  jnp.where(t2 == 0.0, parameters["pih"], jnp.arctan2(t1, t2)), 0.0)
    # to make sure y is between 0 and pi
    y = jnp.where(y > parameters["pi"], y - parameters["pi"], y)
    y = jnp.where(y < 0.0, y + parameters["pi"], y)
    # tcraig, temporary, should be moved to namelist
    # turns on interpolation in stress_rdg
    # Fortran interpolate_stress_rdg is a compile-time .false. parameter.
    kx = jnp.trunc((x - parameters["piq"] - parameters["pi"]) * parameters["invdx"]).astype(jnp.int32)
    ky = jnp.trunc(y * parameters["invdy"]).astype(jnp.int32)
    ka = jnp.trunc((aprime - 0.5) * parameters["invda"]).astype(jnp.int32)
    # Determine sigma_r(A1,Zeta,y) and sigma_s (see Section A1)
    r11, r12, r22, s11, s12, s22 = (tables[n][kx, ky, ka] for n in (
        "s11r", "s12r", "s22r", "s11s", "s12s", "s22s"))
    # Calculate mean ice stress over a collection of floes (Equation 3)
    kf, invsin = parameters["kfriction"], parameters["invsin"]
    sp = strength * (r11 + kf * s11 + r22 + kf * s22) * invsin
    sm = strength * (r11 + kf * s11 - r22 - kf * s22) * invsin
    ss = strength * (r12 + kf * s12) * invsin
    # Back - rotation of the stress from principal axes into general coordinates
    # Update stress
    sig11, sig22 = 0.5 * (sp + sm), 0.5 * (sp - sm)
    sg11 = q11 * sig11 + q22 * sig22 - 2.0 * q12 * ss
    sg12 = q12 * sig11 - q12 * sig22 + (q11 - q22) * ss
    sg22 = q22 * sig11 + q11 * sig22 + 2.0 * q12 * ss
    # Compute ridging and sliding functions in general coordinates (Equation 11)
    rr11 = q11 * r11 - 2.0 * q12 * r12 + q22 * r22
    rr12 = q11 * r12 + q12 * (r11 - r22) - q22 * r12
    rr22 = q22 * r11 + 2.0 * q12 * r12 + q11 * r22
    rs11 = q11 * s11 - 2.0 * q12 * s12 + q22 * s22
    rs12 = q11 * s12 + q12 * (s11 - s22) - q22 * s12
    rs22 = q22 * s11 + 2.0 * q12 * s12 + q11 * s22
    alphar = rr11 * d11 + 2.0 * rr12 * d12 + rr22 * d22
    alphas = rs11 * d11 + 2.0 * rs12 * d12 + rs22 * d22
    return sg11 + sg22, sg11 - sg22, sg12, alphar, alphas


def calc_ffrac(stressp, stressm, stress12, a1x, a2x):
    """Computes term in evolution equation for structure tensor which determines
    the ice floe re-orientation due to fracture.
    Eq. 7: Ffrac = -kf(A-S) or = 0 depending on sigma_1 and sigma_2
    """
    sigma11, sigma12, sigma22 = 0.5 * (stressp + stressm), stress12, 0.5 * (stressp - stressm)
    # if ((sigma11-sigma22) == c0) then sigma11-sigma22 == 0 => stressn ==0
    gamma = jnp.where(stressm == 0.0, jnp.pi / 4.0, 0.5 * jnp.arctan2(2.0 * sigma12, sigma11 - sigma22))
    # rotate tensor to get into sigma principal axis
    q11, q12 = jnp.cos(gamma), jnp.sin(gamma)
    c2, cs, s2 = q11 * q11, q11 * q12, q12 * q12
    sigma_1 = c2 * sigma11 + 2.0 * cs * sigma12 + s2 * sigma22  # S(1,1)
    sigma_2 = s2 * sigma11 - 2.0 * cs * sigma12 + c2 * sigma22  # S(2,2)
    # Pure divergence
    # Unconfined compression: cracking of blocks not along the axial splitting direction
    # which leads to the loss of their shape, so we again model it through diffusion
    # Shear faulting
    active = ((sigma_1 >= 0.0) & (sigma_2 < 0.0)) | (
        (sigma_1 <= 0.0) & (sigma_2 != 0.0)
        & (sigma_1 / jnp.where(sigma_2 != 0.0, sigma_2, 1.0) <= 0.3)
        & ~((sigma_1 >= 0.0) & (sigma_2 >= 0.0)))
    # Horizontal spalling
    return jnp.where(active, 0.001 * (a1x - s2), 0.0), jnp.where(active, 0.001 * (a2x + cs), 0.0)


def eap_subcycles(fields, eap_state, strength, Cw, grid, shared,
                  params, dyn, eap_params, static, iglob):
    from cicecore.cicedyn.dynamics.ice_dyn_evp import stress_combinations
    from cicecore.cicedyn.infrastructure.comm.serial.ice_boundary import ice_halo_update_jax
    eap_state = dict(eap_state)
    mask = shared["iceTmask"]
    # Initialize structure tensor
    for k in range(1, 5):
        eap_state[f"a11_{k}"] = jnp.where(shared["iceTmask_full"], eap_state[f"a11_{k}"], jnp.where(grid["tmask"], 0.5, 0.0))
        eap_state[f"a12_{k}"] = jnp.where(shared["iceTmask_full"], eap_state[f"a12_{k}"], 0.0)
    for n in ("e11", "e12", "e22", "s11", "s12", "s22", "yieldstress11", "yieldstress12", "yieldstress22"):
        eap_state[n] = jnp.zeros_like(strength)
    diagnostics = {n: jnp.zeros_like(strength) for n in ("vort", "shear", "divu", "rdg_conv")}

    def subcycle(ksub, carry):
        fields, a, diagnostics = carry
        fields, a = dict(fields), dict(a)
        div, tension, shear, _ = ice_dyn_shared.strain_rates(
            fields["uvel"], fields["vvel"], grid["dxT"], grid["dyT"],
            shared["cxp"], shared["cyp"], shared["cxm"], shared["cym"], mask, dyn)
        # Stress updated depending on strain rate and structure tensor
        sp, sm, ss, ar, _ = update_stress_rdg(
            div, tension, shear, jnp.stack([a[f"a11_{k}"] for k in range(1, 5)], -1),
            jnp.stack([a[f"a12_{k}"] for k in range(1, 5)], -1), strength[..., None], a, eap_params)
        # on last subcycle, save quantities for mechanical redistribution
        # The final carry retains diagnostics from the last stress update.
        vort, sheardiag, divdiag, _, _ = ice_dyn_shared.deformations(
            fields["uvel"], fields["vvel"], grid, shared, mask, dyn)
        diagnostics = {"vort": vort, "shear": sheardiag, "divu": divdiag,
            "rdg_conv": jnp.where(mask, -jnp.minimum(0.25 * (ar[..., 0] + ar[..., 1] + ar[..., 2] + ar[..., 3]), 0.0) * grid["tarear"], 0.0)}
        # rdg_shear=0 for computing closing_net in ridge_prep
        # rdg_shear(i,j) = p25*(alphasne + alphasnw + alphassw + alphasse) * tarear(i,j)
        sumdiv = div[..., 0] + div[..., 1] + div[..., 3] + div[..., 2]
        sumt = tension[..., 0] + tension[..., 1] + tension[..., 3] + tension[..., 2]
        sums = shear[..., 0] + shear[..., 1] + shear[..., 3] + shear[..., 2]
        for n, val in (("e11", 0.125 * (sumdiv + sumt)), ("e22", 0.125 * (sumdiv - sumt)), ("e12", 0.125 * sums)):
            a[n] = jnp.where(mask, val * grid["tarear"], a[n])
        # elastic relaxation, see Eq. A12-A14
        for stem, target in (("stressp", sp), ("stressm", sm), ("stress12", ss)):
            for k in range(1, 5):
                name = f"{stem}_{k}"
                fields[name] = jnp.where(mask, (fields[name] + target[..., k-1] * dyn["arlx1i"]) * dyn["denom1"], fields[name])
        p = sum(fields[f"stressp_{k}"] for k in range(1, 5))
        m = sum(fields[f"stressm_{k}"] for k in range(1, 5))
        s = sum(fields[f"stress12_{k}"] for k in range(1, 5))
        yp, ym, ys = (x[..., 0] + x[..., 1] + x[..., 2] + x[..., 3] for x in (sp, sm, ss))
        for n, val in (("s11", 0.125 * (p + m)), ("s22", 0.125 * (p - m)), ("s12", 0.25 * s),
                       ("yieldstress11", 0.125 * (yp + ym)), ("yieldstress22", 0.125 * (yp - ym)), ("yieldstress12", 0.25 * ys)):
            a[n] = jnp.where(mask, val, a[n])
        # combinations of the stresses for the momentum equation ! kg/s^2
        terms = stress_combinations(fields, grid, shared, mask)
        # momentum equation
        fields = ice_dyn_shared.stepu(fields, terms, Cw, grid["uarear"], fields["iceXmask"], params, dyn)
        # evolution of structure tensor A
        # only called every 10th timestep
        update = mask & (ksub % 10 == 0)
        for k in range(1, 5):
            a11, a12 = a[f"a11_{k}"], a[f"a12_{k}"]
            m11, m12 = calc_ffrac(fields[f"stressp_{k}"], fields[f"stressm_{k}"], fields[f"stress12_{k}"], a11, a12)
            a[f"a11_{k}"] = jnp.where(update, (a11 * dyn["dtei"] + 0.0001 - m11) / (dyn["dtei"] + 0.0002), a11)  # implicit
            a[f"a12_{k}"] = jnp.where(update, (a12 * dyn["dtei"] - m12) / (dyn["dtei"] + 0.0002), a12)  # implicit
        # average structure tensor
        for stem in ("a11", "a12"):
            a[stem] = jnp.where(update, 0.25 * sum(a[f"{stem}_{k}"] for k in range(1, 5)), a[stem])
        for n in ("uvel", "vvel"):
            fields[n] = ice_halo_update_jax(fields[n], "U", "vector", static, iglob)
        return fields, a, diagnostics

    return jax.lax.fori_loop(0, int(static["ice_dyn_shared"]["ndte"]), subcycle, (fields, eap_state, diagnostics))

# Look-up table needed for calculating structure tensor
nx_yield = 41
ny_yield = 41
na_yield = 21

s11r = None
s12r = None
s22r = None
s11s = None
s12s = None
s22s = None

a11_1 = None
a11_2 = None
a11_3 = None
a11_4 = None
a12_1 = None
a12_2 = None
a12_3 = None
a12_4 = None

# history
e11 = None            # components of strain rate tensor (1/s)
e12 = None
e22 = None
yieldstress11 = None  # components of yield stress tensor (kg/s^2)
yieldstress12 = None
yieldstress22 = None
s11 = None            # components of stress tensor (kg/s^2)
s12 = None
s22 = None
a11 = None            # components of structure tensor ()
a12 = None

# private for reuse, set in init_eap
puny = None
pi = None
pi2 = None
piq = None
pih = None

kfriction = dbl_kind(0.45)

invdx = None
invdy = None
invda = None
invsin = None


def _zero(shape):
    return np.zeros(shape, dtype=dbl_kind)


def w1(a):
    """Function : w1 (see Gaussian function psi in Tsamados et al 2013)"""
    a = dbl_kind(a)
    return (
        dbl_kind(-223.87569446)
        + dbl_kind(2361.2198663) * a
        - dbl_kind(10606.56079975) * a * a
        + dbl_kind(26315.50025642) * a * a * a
        - dbl_kind(38948.30444297) * a * a * a * a
        + dbl_kind(34397.72407466) * a * a * a * a * a
        - dbl_kind(16789.98003081) * a * a * a * a * a * a
        + dbl_kind(3495.82839237) * a * a * a * a * a * a * a
    )


def w2(a):
    """Function : w2 (see Gaussian function psi in Tsamados et al 2013)"""
    a = dbl_kind(a)
    return (
        dbl_kind(-6670.68911883)
        + dbl_kind(70222.33061536) * a
        - dbl_kind(314871.71525448) * a * a
        + dbl_kind(779570.02793492) * a * a * a
        - dbl_kind(1151098.82436864) * a * a * a * a
        + dbl_kind(1013896.59464498) * a * a * a * a * a
        - dbl_kind(493379.44906738) * a * a * a * a * a * a
        + dbl_kind(102356.551518) * a * a * a * a * a * a * a
    )


def _yield_pieces(x, y, z, phi):
    """Shared tensor pieces of s11kr, s12kr, s22kr, s11ks, s12ks, s22ks.

    Commented-out t1/t2 products in the Fortran are the unused alternatives.
    """
    # p = phi
    p = phi
    n1t2i11 = np.cos(z + pih - p) * np.cos(z + p)
    n1t2i12 = np.cos(z + pih - p) * np.sin(z + p)
    n1t2i21 = np.sin(z + pih - p) * np.cos(z + p)
    n1t2i22 = np.sin(z + pih - p) * np.sin(z + p)
    n2t1i11 = np.cos(z - pih + p) * np.cos(z - p)
    n2t1i12 = np.cos(z - pih + p) * np.sin(z - p)
    n2t1i21 = np.sin(z - pih + p) * np.cos(z - p)
    n2t1i22 = np.sin(z - pih + p) * np.sin(z - p)
    # t1t2i11 = cos(z-p) * cos(z+p)
    # t1t2i12 = cos(z-p) * sin(z+p)
    # t1t2i21 = sin(z-p) * cos(z+p)
    # t1t2i22 = sin(z-p) * sin(z+p)
    # t2t1i11 = cos(z+p) * cos(z-p)
    # t2t1i12 = cos(z+p) * sin(z-p)
    # t2t1i21 = sin(z+p) * cos(z-p)
    # t2t1i22 = sin(z+p) * sin(z-p)
    # In expression of tensor d, with this formulatin d(x)=-d(x+pi)
    # Solution, when diagonalizing always check sgn(a11-a22) if > then keep x else x=x-pi/2
    cy2 = np.cos(y) * np.cos(y)
    tany = np.tan(y)
    d11 = cy2 * (np.cos(x) + np.sin(x) * tany * tany)
    d12 = cy2 * tany * (-np.cos(x) + np.sin(x))
    d22 = cy2 * (np.sin(x) + np.cos(x) * tany * tany)
    IIn1t2 = n1t2i11 * d11 + (n1t2i12 + n1t2i21) * d12 + n1t2i22 * d22
    IIn2t1 = n2t1i11 * d11 + (n2t1i12 + n2t1i21) * d12 + n2t1i22 * d22
    Hen1t2 = np.where(-IIn1t2 >= puny, c1, c0)
    Hen2t1 = np.where(-IIn2t1 >= puny, c1, c0)
    t1t2i11 = np.cos(z - p) * np.cos(z + p)
    t1t2i12 = np.cos(z - p) * np.sin(z + p)
    t1t2i21 = np.sin(z - p) * np.cos(z + p)
    t1t2i22 = np.sin(z - p) * np.sin(z + p)
    t2t1i11 = np.cos(z + p) * np.cos(z - p)
    t2t1i12 = np.cos(z + p) * np.sin(z - p)
    t2t1i21 = np.sin(z + p) * np.cos(z - p)
    t2t1i22 = np.sin(z + p) * np.sin(z - p)
    IIt1t2 = t1t2i11 * d11 + (t1t2i12 + t1t2i21) * d12 + t1t2i22 * d22
    return {
        "n1t2i11": n1t2i11,
        "n1t2i12": n1t2i12,
        "n1t2i21": n1t2i21,
        "n1t2i22": n1t2i22,
        "n2t1i11": n2t1i11,
        "n2t1i12": n2t1i12,
        "n2t1i21": n2t1i21,
        "n2t1i22": n2t1i22,
        "t1t2i11": t1t2i11,
        "t1t2i12": t1t2i12,
        "t1t2i21": t1t2i21,
        "t1t2i22": t1t2i22,
        "t2t1i11": t2t1i11,
        "t2t1i12": t2t1i12,
        "t2t1i21": t2t1i21,
        "t2t1i22": t2t1i22,
        "Hen1t2": Hen1t2,
        "Hen2t1": Hen2t1,
        "IIt1t2": IIt1t2,
    }


def s11kr(x, y, z, phi):
    """Function : s11kr"""
    part = _yield_pieces(x, y, z, phi)
    return -part["Hen1t2"] * part["n1t2i11"] - part["Hen2t1"] * part["n2t1i11"]


def s12kr(x, y, z, phi):
    """Function : s12kr"""
    part = _yield_pieces(x, y, z, phi)
    s12r0 = -part["Hen1t2"] * part["n1t2i12"] - part["Hen2t1"] * part["n2t1i12"]
    s21r0 = -part["Hen1t2"] * part["n1t2i21"] - part["Hen2t1"] * part["n2t1i21"]
    return p5 * (s12r0 + s21r0)


def s22kr(x, y, z, phi):
    """Function : s22r"""
    part = _yield_pieces(x, y, z, phi)
    return -part["Hen1t2"] * part["n1t2i22"] - part["Hen2t1"] * part["n2t1i22"]


def s11ks(x, y, z, phi):
    """Function : s11ks"""
    part = _yield_pieces(x, y, z, phi)
    # t2t1i12, t2t1i21, t2t1i22 are unused in this component
    signed = np.copysign(c1, part["IIt1t2"] + puny)
    return signed * (part["Hen1t2"] * part["t1t2i11"] + part["Hen2t1"] * part["t2t1i11"])


def s12ks(x, y, z, phi):
    """Function : s12ks"""
    part = _yield_pieces(x, y, z, phi)
    signed = np.copysign(c1, part["IIt1t2"] + puny)
    s12s0 = signed * (part["Hen1t2"] * part["t1t2i12"] + part["Hen2t1"] * part["t2t1i12"])
    s21s0 = signed * (part["Hen1t2"] * part["t1t2i21"] + part["Hen2t1"] * part["t2t1i21"])
    return p5 * (s12s0 + s21s0)


def s22ks(x, y, z, phi):
    """Function : s22ks"""
    part = _yield_pieces(x, y, z, phi)
    signed = np.copysign(c1, part["IIt1t2"] + puny)
    return signed * (part["Hen1t2"] * part["t1t2i22"] + part["Hen2t1"] * part["t2t1i22"])


def _clip_eps(values, eps6):
    return np.where(np.abs(values) < eps6, c0, values)


def init_eap():
    """Initialize parameters and variables needed for the eap dynamics
    (based on init_dyn)
    """
    global puny, pi, pi2, piq, pih
    global a11_1, a11_2, a11_3, a11_4, a12_1, a12_2, a12_3, a12_4
    global e11, e12, e22, s11, s12, s22, a11, a12
    global yieldstress11, yieldstress12, yieldstress22
    global s11r, s12r, s22r, s11s, s12s, s22s
    global invdx, invdy, invda, invsin

    subname = "(init_eap)"
    # call icepack_query_parameters. Column-physics failures raise.
    puny = dbl_kind(icepack_parameters.puny)
    pi = dbl_kind(icepack_parameters.pi)
    pi2 = dbl_kind(icepack_parameters.pi2)
    piq = dbl_kind(icepack_parameters.piq)
    pih = dbl_kind(icepack_parameters.pih)

    phi = pi / c12  # diamond shaped floe smaller angle (default phi = 30 deg)

    ice_dyn_shared.init_dyn_shared(ice_calendar.dt_dyn)

    horiz = ice_domain_size.horizontal_shape()
    a11_1 = _zero(horiz)
    a11_2 = _zero(horiz)
    a11_3 = _zero(horiz)
    a11_4 = _zero(horiz)
    a12_1 = _zero(horiz)
    a12_2 = _zero(horiz)
    a12_3 = _zero(horiz)
    a12_4 = _zero(horiz)
    e11 = _zero(horiz)
    e12 = _zero(horiz)
    e22 = _zero(horiz)
    yieldstress11 = _zero(horiz)
    yieldstress12 = _zero(horiz)
    yieldstress22 = _zero(horiz)
    s11 = _zero(horiz)
    s12 = _zero(horiz)
    s22 = _zero(horiz)
    a11 = _zero(horiz)
    a12 = _zero(horiz)

    e11[...] = c0
    e12[...] = c0
    e22[...] = c0
    s11[...] = c0
    s12[...] = c0
    s22[...] = c0
    yieldstress11[...] = c0
    yieldstress12[...] = c0
    yieldstress22[...] = c0
    a11_1[...] = p5
    a11_2[...] = p5
    a11_3[...] = p5
    a11_4[...] = p5
    a12_1[...] = c0
    a12_2[...] = c0
    a12_3[...] = c0
    a12_4[...] = c0
    if ice_flux.rdg_shear is None:
        raise RuntimeError(subname + " rdg_shear is not allocated")
    ice_flux.rdg_shear[...] = c0

    #-----------------------------------------------------------------
    # create lookup table for eap dynamics (see Appendix A1)
    #-----------------------------------------------------------------
    eps6 = dbl_kind(1.0e-6)
    nz = 100
    da = p5 / dbl_kind(na_yield - 1)
    ainit = p5 - da
    dx = pi / dbl_kind(nx_yield - 1)
    xinit = pi + piq - dx
    dz = pi / dbl_kind(nz)
    zinit = -pih
    dy = pi / dbl_kind(ny_yield - 1)
    yinit = -dy
    invdx = c1 / dx
    invdy = c1 / dy
    invda = c1 / da

    s11r = _zero((nx_yield, ny_yield, na_yield))
    s12r = _zero((nx_yield, ny_yield, na_yield))
    s22r = _zero((nx_yield, ny_yield, na_yield))
    s11s = _zero((nx_yield, ny_yield, na_yield))
    s12s = _zero((nx_yield, ny_yield, na_yield))
    s22s = _zero((nx_yield, ny_yield, na_yield))

    ix = np.arange(1, nx_yield + 1, dtype=dbl_kind)
    iy = np.arange(1, ny_yield + 1, dtype=dbl_kind)
    iz = np.arange(1, nz + 1, dtype=dbl_kind)
    x = (xinit + ix * dx)[:, None, None]
    y = (yinit + iy * dy)[None, :, None]
    z = (zinit + iz * dz)[None, None, :]
    sin_two_phi = np.sin(c2 * phi)

    for ia in range(1, na_yield):
        aval = ainit + dbl_kind(ia) * da
        weight = w1(aval) * np.exp(-w2(aval) * z * z) * dz / sin_two_phi
        s11r[:, :, ia - 1] = np.sum(s11kr(x, y, z, phi) * weight, axis=2)
        s12r[:, :, ia - 1] = np.sum(s12kr(x, y, z, phi) * weight, axis=2)
        s22r[:, :, ia - 1] = np.sum(s22kr(x, y, z, phi) * weight, axis=2)
        s11s[:, :, ia - 1] = np.sum(s11ks(x, y, z, phi) * weight, axis=2)
        s12s[:, :, ia - 1] = np.sum(s12ks(x, y, z, phi) * weight, axis=2)
        s22s[:, :, ia - 1] = np.sum(s22ks(x, y, z, phi) * weight, axis=2)

    x_last = (xinit + ix * dx)[:, None]
    y_last = (yinit + iy * dy)[None, :]
    s11r[:, :, na_yield - 1] = p5 * s11kr(x_last, y_last, c0, phi) / sin_two_phi
    s12r[:, :, na_yield - 1] = p5 * s12kr(x_last, y_last, c0, phi) / sin_two_phi
    s22r[:, :, na_yield - 1] = p5 * s22kr(x_last, y_last, c0, phi) / sin_two_phi
    s11s[:, :, na_yield - 1] = p5 * s11ks(x_last, y_last, c0, phi) / sin_two_phi
    s12s[:, :, na_yield - 1] = p5 * s12ks(x_last, y_last, c0, phi) / sin_two_phi
    s22s[:, :, na_yield - 1] = p5 * s22ks(x_last, y_last, c0, phi) / sin_two_phi

    s11r[...] = _clip_eps(s11r, eps6)
    s12r[...] = _clip_eps(s12r, eps6)
    s22r[...] = _clip_eps(s22r, eps6)
    s11s[...] = _clip_eps(s11s, eps6)
    s12s[...] = _clip_eps(s12s, eps6)
    s22s[...] = _clip_eps(s22s, eps6)

    # Factor to maintain the same stress as in EVP (see Section 3)
    # Can be set to 1 otherwise
    invstressconviso = c1 / (c1 + kfriction * kfriction)
    invsin = c1 / np.sin(pi2 / c12) * invstressconviso


def _eap_restart_path():
    """Supplemental restart name from ice_restart: restart_file + '.eap'."""
    subname = "(read_restart_eap)"
    pointer = str(ice_init.pointer_file).strip()
    with open(pointer, encoding="utf-8") as handle:
        filename0 = handle.readline().strip()
    restart_file = str(ice_init.restart_file).strip()
    n = filename0.find(restart_file)
    if n < 0:
        raise RuntimeError(subname + " ERROR: eap restart: filename discrepancy")
    string1 = filename0[:n]
    string2 = filename0[n + len(restart_file) :]
    return string1 + restart_file + ".eap" + string2


def _skip_unformatted_header(stream):
    """Skip the istep/time header record written before the EAP fields."""
    mark = stream.read(4)
    if len(mark) < 4:
        raise ValueError("(read_restart_eap) short EAP header")
    # integer + two double-precision reals, or three 8-byte values
    accepted = (20, 24)
    for trial in (">", "<"):
        size = int(np.frombuffer(mark, dtype=np.dtype(trial + "i4"))[0])
        if size in accepted:
            payload = stream.read(size)
            tail = stream.read(4)
            if len(payload) != size or len(tail) != 4:
                raise ValueError("(read_restart_eap) short EAP header")
            return
    extra = stream.read(4)
    mark = mark + extra
    for trial in (">", "<"):
        size = int(np.frombuffer(mark, dtype=np.dtype(trial + "i8"))[0])
        if size in accepted:
            payload = stream.read(size)
            tail = stream.read(8)
            if len(payload) != size or len(tail) != 8:
                raise ValueError("(read_restart_eap) short EAP header")
            return
    raise ValueError("(read_restart_eap) bad EAP header marker")


def write_restart_eap():
    """Write the eight EAP structure-tensor components to the core dump."""
    from cicecore.cicedyn.infrastructure.io.io_netcdf import ice_restart

    for stem in ("a11", "a12"):
        for k in (1, 3, 2, 4):
            name = f"{stem}_{k}"
            ice_restart.write_restart_field(globals()[name], name)


def read_restart_eap():
    """Reads all values needed for elastic anisotropic plastic dynamics restart"""
    subname = "(read_restart_eap)"
    for name in (
        "a11_1", "a11_2", "a11_3", "a11_4", "a12_1", "a12_2", "a12_3", "a12_4",
    ):
        if globals()[name] is None:
            raise RuntimeError(subname + " " + name + " is not allocated")

    # The current netCDF writer places EAP fields in the core restart.
    # Retain the older supplemental binary read below for older files.
    from cicecore.cicedyn.infrastructure.io.io_netcdf import ice_restart
    if ice_restart.query_field("a11_1"):
        for name in ("a11_1", "a11_3", "a11_2", "a11_4",
                     "a12_1", "a12_3", "a12_2", "a12_4"):
            ice_restart.read_restart_field(
                globals()[name], name, 1, True, field_loc_center, field_type_scalar
            )
        if str(ice_domain.ns_boundary_type).strip() in ("tripole", "tripoleT"):
            from cicecore.cicedyn.infrastructure.comm.serial.ice_boundary import ice_HaloUpdate_stress
            for stem in ("a11", "a12"):
                for first, second in ((1, 3), (3, 1), (2, 4), (4, 2)):
                    name = f"{stem}_{first}"
                    other = f"{stem}_{second}"
                    globals()[name][:] = ice_HaloUpdate_stress(
                        globals()[name], globals()[other], field_loc_center, field_type_scalar
                    )
        ghost = int(ice_domain_size.nghost)
        for name in ("a11_1", "a11_2", "a11_3", "a11_4",
                     "a12_1", "a12_2", "a12_3", "a12_4"):
            globals()[name][:, :ghost] = c0
            globals()[name][:ghost, :] = c0
        return

    if bool(ice_init.restart_ext):
        raise NotImplementedError(subname + " restart_ext for legacy binary")

    diag = True
    #-----------------------------------------------------------------
    # Structure tensor must be read and scattered in pairs in order
    # to properly match corner values across a tripole grid cut.
    #-----------------------------------------------------------------
    ice_fileunits.nu_diag.write(" structure tensor restart data\n")
    ice_fileunits.nu_diag.flush()

    path = _eap_restart_path()
    nx = int(ice_domain_size.nx_global)
    ny = int(ice_domain_size.ny_global)
    nbytes = nx * ny * 8
    names = ("a11_1", "a11_3", "a11_2", "a11_4", "a12_1", "a12_3", "a12_2", "a12_4")
    with open(path, "rb") as stream:
        _skip_unformatted_header(stream)
        for name in names:
            mark = stream.read(4)
            if len(mark) < 4:
                raise ValueError(f"{subname} short sequential record for {name}")
            length = None
            marker = 4
            for trial in (">", "<"):
                size = int(np.frombuffer(mark, dtype=np.dtype(trial + "i4"))[0])
                if size == nbytes:
                    length = size
                    break
            if length is None:
                extra = stream.read(4)
                mark = mark + extra
                marker = 8
                for trial in (">", "<"):
                    size = int(np.frombuffer(mark, dtype=np.dtype(trial + "i8"))[0])
                    if size == nbytes:
                        length = size
                        break
            if length != nbytes:
                raise ValueError(f"{subname} bad sequential marker for {name} in {path}")
            raw = stream.read(nbytes)
            stream.read(marker)
            if len(raw) != nbytes:
                raise ValueError(f"{subname} missing sequential record {name}")
            values = None
            for trial in (">", "<"):
                sample = np.frombuffer(raw, dtype=np.dtype(trial + "f8"))
                if (
                    sample.size > 0
                    and np.isfinite(sample).all()
                    and np.max(np.abs(sample)) <= 1.0e10
                ):
                    values = np.array(sample, dtype=dbl_kind)
                    break
            if values is None:
                raise ValueError(f"{subname} could not decode {name}")
            work = values.reshape((nx, ny), order="F")
            if diag:
                ice_fileunits.nu_diag.write(
                    f" (read_restart_eap) read vname  {name}\n"
                )
            scatter_global(globals()[name], work, field_loc_center, field_type_scalar)

    ns_boundary_type = str(ice_domain.ns_boundary_type).strip()
    if ns_boundary_type in ("tripole", "tripoleT"):
        from cicecore.cicedyn.infrastructure.comm.serial.ice_boundary import ice_HaloUpdate_stress
        for stem in ("a11", "a12"):
            for first, second in ((1, 3), (3, 1), (2, 4), (4, 2)):
                name = f"{stem}_{first}"
                globals()[name][:] = ice_HaloUpdate_stress(
                    globals()[name], globals()[f"{stem}_{second}"], field_loc_center, field_type_scalar)

    #-----------------------------------------------------------------
    # Ensure unused values in west and south ghost cells are 0
    #-----------------------------------------------------------------
    ghost = int(ice_domain_size.nghost)
    for name in (
        "a11_1", "a11_2", "a11_3", "a11_4", "a12_1", "a12_2", "a12_3", "a12_4",
    ):
        field = globals()[name]
        field[:, :ghost] = c0
        field[:ghost, :] = c0
