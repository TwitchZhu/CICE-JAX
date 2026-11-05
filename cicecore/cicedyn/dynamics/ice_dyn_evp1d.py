"""One-dimensional EVP kernels on a single JAX device.

author for JAX version: Xinyuan Zhu
XZ: converted from F90 to Python

Numerical equations from ice_dyn_core1d.F90 and the subcycle ordering from
ice_dyn_evp1d.F90. Fixed-size flattened arrays replace the MPI gather/scatter,
OpenMP allocation and NUMA placement. Halo updates use the single-grid backend.
"""

from functools import partial

import jax
import jax.numpy as jnp

from cicecore.cicedyn.dynamics.ice_dyn_shared import visc_replpress, shift
from cicecore.cicedyn.dynamics.ice_dyn_evp import stress_combinations
from cicecore.cicedyn.infrastructure.comm.serial.ice_boundary import ice_halo_update_jax


@partial(jax.jit, static_argnames=("ndte", "halo_config"))
def dyn_evp1d_run(fields, strength, Cw, grid, shared, params, dyn, iglob, ndte, halo_config):
    shape = fields["uvel"].shape
    index = jnp.arange(fields["uvel"].size).reshape(shape)
    west, south, southwest, east, north, northeast = (
        shift(index, di, dj).reshape(-1)
        for di, dj in ((-1, 0), (0, -1), (-1, -1), (1, 0), (0, 1), (1, 1)))
    fields = {n: value.reshape(-1) for n, value in fields.items()}
    grid = {n: value.reshape(-1) for n, value in grid.items()}
    strength, Cw = strength.reshape(-1), Cw.reshape(-1)
    maskT = shared["iceTmask"].reshape(-1)
    bounds, ghost, nx, ew, ns = halo_config
    static = {"ice_domain_size": {"physical_bounds": bounds, "nghost": ghost, "nx_global": nx},
              "ice_domain": {"ew_boundary_type": ew, "ns_boundary_type": ns}}
    dxT, dyT, htn, hte = (grid[n] for n in ("dxT", "dyT", "HTN", "HTE"))
    htnm1, htem1 = htn[south], hte[west]
    cxp, cyp = 1.5 * htn - 0.5 * htnm1, 1.5 * hte - 0.5 * htem1
    cxm, cym = -(1.5 * htnm1 - 0.5 * htn), -(1.5 * htem1 - 0.5 * hte)
    tmparea = dxT * dyT  # necessary to split calc of DminTarea. Otherwize not binary identical
    DminTarea = dyn["deltaminEVP"] * tmparea
    metrics = {"dxhy": 0.5 * (hte - htem1), "dyhx": 0.5 * (htn - htnm1)}

    def subcycle(_, fields):
        result = dict(fields)
        u, v = fields["uvel"], fields["vvel"]
        uw, us, usw, vw, vs, vsw = u[west], u[south], u[southwest], v[west], v[south], v[southwest]
        # strain rates - NOTE these are actually strain rates * area  (m^2/s)
        # divergence  =  e_11 + e_22
        div = jnp.stack((cyp*u - dyT*uw + cxp*v - dxT*vs,
                         cym*uw + dyT*u + cxp*vw - dxT*vsw,
                         cym*usw + dyT*us + cxm*vsw + dxT*vw,
                         cyp*us - dyT*usw + cxm*vs + dxT*v), -1)
        # tension strain rate  =  e_11 - e_22
        tension = jnp.stack((-cym*u - dyT*uw + cxm*v + dxT*vs,
                             -cyp*uw + dyT*u + cxm*vw + dxT*vsw,
                             -cyp*usw + dyT*us + cxp*vsw - dxT*vw,
                             -cym*us - dyT*usw + cxp*vs - dxT*v), -1)
        # shearing strain rate  =  2*e_12
        shear = jnp.stack((-cym*v - dyT*vw - cxm*u - dxT*us,
                           -cyp*vw + dyT*v - cxm*uw - dxT*usw,
                           -cyp*vsw + dyT*vs - cxp*usw + dxT*uw,
                           -cym*vs - dyT*vsw - cxp*us + dxT*u), -1)
        delta = jnp.sqrt(div**2 + dyn["e_factor"] * (tension**2 + shear**2))
        # viscosities and replacement pressure
        zeta, eta, pressure = visc_replpress(strength[:, None], DminTarea[:, None], delta, dyn)
        # the stresses                            ! kg/s^2
        # (1) northeast, (2) northwest, (3) southwest, (4) southeast
        # NOTE: for comp. efficiency 2 x zeta and 2 x eta are used in the code
        for stem, target in (("stressp", zeta*div - pressure),
                             ("stressm", eta*tension), ("stress12", 0.5*eta*shear)):
            for k in range(1, 5):
                name = f"{stem}_{k}"
                result[name] = jnp.where(maskT,
                    (fields[name] * (1.0 - dyn["arlx1i"] * dyn["revp"])
                     + dyn["arlx1i"] * target[:, k-1]) * dyn["denom1"], fields[name])
        # combinations of the stresses for the momentum equation ! kg/s^2
        terms = stress_combinations(result, grid, metrics, maskT)
        # divergence of the internal stress tensor
        strintx = grid["uarear"] * (terms[:, 0] + terms[east, 1] + terms[north, 2] + terms[northeast, 3])
        strinty = grid["uarear"] * (terms[:, 4] + terms[north, 5] + terms[east, 6] + terms[northeast, 7])
        # integrate the momentum equation
        # (magnitude of relative ocean current)*rhow*drag*aice
        vrel = fields["aiX"] * params["rhow"] * Cw * jnp.sqrt((fields["uocn"] - u)**2 + (fields["vocn"] - v)**2)
        # ice/ocean stress
        taux = vrel * fields["waterx"]  # NOTE this is not the entire
        tauy = vrel * fields["watery"]  # ocn stress term
        Cb = fields["TbU"] / (jnp.sqrt(u**2 + v**2) + 5.0e-5)  # for basal stress
        # revp = 0 for classic evp, 1 for revised evp
        cca = (dyn["brlx"] + dyn["revp"]) * fields["Umassdti"] + vrel + Cb
        ccb = fields["fm"]  # cosw=1, sinw=0
        mask = fields["iceXmask"]
        ab2 = jnp.where(mask, cca**2 + ccb**2, 1.0)
        # finally, the velocity components
        cc1 = strintx + fields["forcex"] + taux + fields["Umassdti"] * (dyn["brlx"]*u + dyn["revp"]*fields["uvel_init"])
        cc2 = strinty + fields["forcey"] + tauy + fields["Umassdti"] * (dyn["brlx"]*v + dyn["revp"]*fields["vvel_init"])
        unew, vnew = (cca*cc1 + ccb*cc2) / ab2, (cca*cc2 - ccb*cc1) / ab2
        # calculate seabed stress component for outputs
        # only needed on last iteration.
        for name, value in (("strintx", strintx), ("strinty", strinty),
                             ("uvel", unew), ("vvel", vnew), ("taubx", -unew*Cb), ("tauby", -vnew*Cb)):
            result[name] = jnp.where(mask, value, fields[name])
        for name in ("uvel", "vvel"):
            result[name] = ice_halo_update_jax(result[name].reshape(shape), "U", "vector", static, iglob).reshape(-1)
        return result

    fields = jax.lax.fori_loop(0, ndte, subcycle, fields)
    return {n: value.reshape(shape) for n, value in fields.items()}
