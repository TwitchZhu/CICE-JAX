"""JAX translations of brine dynamics in icepack_brine.F90."""

import jax.numpy as jnp

from icepack.columnphysics.icepack_mushy_physics import (
    icepack_mushy_liquid_fraction_jax,
    icepack_mushy_temperature_mush_jax,
)
from icepack.columnphysics.icepack_zbgc_shared import remap_zbgc_jax


def preflushing_changes(aicen, vicen, vsnon, meltb, meltt,
                        congel, snoice, hice_old, fbri):
    """Find ice and brine height changes before surface flushing."""
    failed = fbri <= 0.0
    hin = vicen / jnp.where(aicen > 0.0, aicen, 1.0)
    hsn = vsnon / jnp.where(aicen > 0.0, aicen, 1.0)
    hin_old = jnp.maximum(0.0, hin + meltb + meltt - congel - snoice)
    dhice = hin_old - hice_old
    dhbr_top = meltt - snoice - dhice
    dhbr_bot = congel - meltb
    hbr_old = fbri * hice_old
    return hin, hsn, hbr_old, dhbr_top, dhbr_bot, failed


def prepare_hbrine(bSin, bTin, sss, bgrid, igrid, parameters):
    """Compute brine salinity, density, porosity and mean permeability."""
    nblyr = bSin.shape[0] - 2
    puny = parameters["puny"]
    rhosi = parameters["rhosi"]
    rhow = parameters["rhow"]

    # The bottom biology point is ocean water; the other points are ice.
    brine_sal = -21.4 * bTin - 0.886 * bTin**2 - 0.012 * bTin**3
    brine_sal = brine_sal.at[nblyr + 1].set(sss)
    brine_rho = 1000.0 + 0.8 * brine_sal
    brine_rho = brine_rho.at[nblyr + 1].set(rhow)
    bphin = jnp.clip(bSin * rhosi / jnp.where(
        brine_sal * brine_rho != 0.0, brine_sal * brine_rho, 1.0),
        puny, 1.0)
    bphin = bphin.at[nblyr + 1].set(1.0)
    kin = 3.0e-8 * bphin**3
    k_min = jnp.min(kin[1:nblyr + 1])
    kperm = jnp.where(k_min > 0.0,
                      nblyr / jnp.sum(1.0 / kin[1:nblyr + 1]), 0.0)
    bphi_min = bphin[0]

    ibrine_sal = jnp.zeros(nblyr + 1, dtype=bSin.dtype)
    ibrine_rho = jnp.zeros(nblyr + 1, dtype=bSin.dtype)
    iTin = jnp.zeros(nblyr + 1, dtype=bSin.dtype)
    iSin = jnp.zeros(nblyr + 1, dtype=bSin.dtype)
    iphin = jnp.zeros(nblyr + 1, dtype=bSin.dtype)
    ibrine_sal = ibrine_sal.at[0].set(brine_sal[1])
    ibrine_sal = ibrine_sal.at[nblyr].set(brine_sal[nblyr + 1])
    ibrine_rho = ibrine_rho.at[0].set(brine_rho[1])
    ibrine_rho = ibrine_rho.at[nblyr].set(brine_rho[nblyr + 1])
    iTin = iTin.at[0].set(bTin[1])
    iTin = iTin.at[nblyr].set(bTin[nblyr])
    iSin = iSin.at[0].set(bSin[1])
    iSin = iSin.at[nblyr].set(bSin[nblyr])
    iphin = iphin.at[0].set(bphin[1])
    iphin = iphin.at[nblyr].set(bphin[nblyr])
    for k in range(1, nblyr):
        igrp = igrid[k + 1] - igrid[k]
        igrm = igrid[k] - igrid[k - 1]
        rigr = 1.0 / (igrid[k + 1] - igrid[k - 1])
        ibrine_sal = ibrine_sal.at[k].set(
            (brine_sal[k + 1] * igrp + brine_sal[k] * igrm) * rigr)
        ibrine_rho = ibrine_rho.at[k].set(
            (brine_rho[k + 1] * igrp + brine_rho[k] * igrm) * rigr)
        iTin = iTin.at[k].set((bTin[k + 1] * igrp + bTin[k] * igrm) * rigr)
        iSin = iSin.at[k].set((bSin[k + 1] * igrp + bSin[k] * igrm) * rigr)
        iphin = iphin.at[k].set(jnp.clip(
            (bphin[k + 1] * igrp + bphin[k] * igrm) * rigr, puny, 1.0))
    return (brine_sal, brine_rho, ibrine_sal, ibrine_rho,
            bphin, iphin, iTin, iSin, kperm, bphi_min)


def calculate_drho(brine_rho, ibrine_rho, bgrid, igrid, rhow):
    """Find the density difference around each biology interface."""
    nblyr = ibrine_rho.shape[0] - 1
    rho_a = jnp.zeros_like(ibrine_rho)
    rho_b = jnp.zeros_like(ibrine_rho)
    rho_2a = jnp.zeros_like(ibrine_rho)
    rho_2b = jnp.zeros_like(ibrine_rho)
    drho = jnp.zeros_like(ibrine_rho)

    # Preserve the separate top, interior and bottom formulas in Fortran.
    for k in range(nblyr + 1):
        if k == 1:
            rho_a = rho_a.at[k].set(
                (brine_rho[k] * bgrid[k]
                 + 0.5 * (ibrine_rho[k] + brine_rho[k])
                 * (igrid[k] - bgrid[k])) / igrid[k])
            rho_b = rho_b.at[k].set(brine_rho[k])
        elif 1 < k < nblyr:
            rho_a = rho_a.at[k].set(
                (rho_a[k - 1] * igrid[k - 1]
                 + 0.5 * (ibrine_rho[k - 1] + brine_rho[k])
                 * (bgrid[k] - igrid[k - 1])
                 + 0.5 * (ibrine_rho[k] + brine_rho[k])
                 * (igrid[k] - bgrid[k])) / igrid[k])
            rho_b = rho_b.at[k].set(brine_rho[k])
        else:
            rho_a = rho_a.at[nblyr].set(
                (rho_a[nblyr - 1] * igrid[nblyr - 1]
                 + 0.5 * (ibrine_rho[nblyr - 1] + brine_rho[nblyr])
                 * (bgrid[nblyr] - igrid[nblyr - 1])
                 + brine_rho[nblyr]
                 * (igrid[nblyr] - bgrid[nblyr])) / igrid[nblyr])
            rho_a = rho_a.at[0].set(brine_rho[1])
            rho_b = rho_b.at[nblyr].set(brine_rho[nblyr])
            rho_b = rho_b.at[0].set(brine_rho[1])

    for k in range(nblyr + 1):
        if k == 0:
            rho_2a = rho_2a.at[0].set(
                (rho_a[0] * bgrid[1]
                 + 0.5 * (brine_rho[1] + ibrine_rho[1])
                 * (igrid[1] - bgrid[1])) / igrid[1])
            rho_2b = rho_2b.at[0].set(brine_rho[1])
        else:
            if 2 * k + 1 < nblyr + 1:
                rho_2a = rho_2a.at[k].set(rho_a[2 * k])
            else:
                rho_2a = rho_2a.at[k].set(
                    0.5 * (rho_a[nblyr] + rhow * (2.0 * igrid[k] - 1.0))
                    / igrid[k])
            rho_2b = rho_2b.at[k].set(brine_rho[k + 1])
        drho = drho.at[k].set(jnp.maximum(
            rho_b[k] - rho_2b[k],
            jnp.maximum(0.0, jnp.maximum(
                2.0 * (rho_a[k] - rho_2a[k]),
                2.0 * (brine_rho[k] - brine_rho[k + 1]) / nblyr))))
    return drho


def compute_microS_mushy(trcrn, hice_old, hbr_old, sss, sst,
                         cgrid, bgrid, igrid, nt_sice, nt_qice,
                         nilyr, parameters):
    """Remap ice thermodynamics and compute brine diffusion and permeability."""
    nblyr = bgrid.shape[0] - 2
    surface_S = parameters["min_salin"]
    cSin = trcrn[nt_sice:nt_sice + nilyr]
    cqin = trcrn[nt_qice:nt_qice + nilyr]
    # CICE ice layers and biology layers use distinct vertical centers.
    remapped_s = remap_zbgc_jax(
        cSin, cgrid[1:nilyr + 1], bgrid[1:nblyr + 1],
        hice_old, hice_old, surface_S)
    remapped_q = remap_zbgc_jax(
        cqin, cgrid[1:nilyr + 1], bgrid[1:nblyr + 1],
        hice_old, hice_old, surface_S)
    bqin = jnp.minimum(0.0, remapped_q)
    bSin_ice = jnp.maximum(0.01, remapped_s)
    bTin_ice = icepack_mushy_temperature_mush_jax(
        bqin, bSin_ice, parameters)
    bphi_ice = icepack_mushy_liquid_fraction_jax(
        bTin_ice, bSin_ice, parameters)
    bSin = jnp.concatenate((bSin_ice[:1], bSin_ice,
                            jnp.asarray([sss])))
    bTin = jnp.concatenate((bTin_ice[:1], bTin_ice,
                            jnp.asarray([sst])))
    bphin = jnp.concatenate((bphi_ice[:1], bphi_ice,
                             jnp.asarray([1.0], dtype=bphi_ice.dtype)))
    (brine_sal, brine_rho, ibrine_sal, ibrine_rho,
     bphin, iphin, iTin, iSin, kperm, bphi_min) = prepare_hbrine(
        bSin, bTin, sss, bgrid, igrid, parameters)
    drho = calculate_drho(brine_rho, ibrine_rho, bgrid, igrid,
                          parameters["rhow"])
    iDin = jnp.zeros(nblyr + 1, dtype=trcrn.dtype)
    for k in range(1, nblyr + 1):
        permeability = 3.0e-8 * iphin[k]**3
        hbr_squared = jnp.where(hbr_old > parameters["puny"],
                                hbr_old**2, 1.0)
        diffusion = jnp.where(hbr_old > parameters["puny"],
                              iphin[k] * 1.0e-9 / hbr_squared, 0.0)
        drainage = jnp.where(
            hbr_old >= 0.05,
            parameters["l_sk"] * permeability * parameters["gravit"]
            / 2.2 * drho[k] / hbr_squared, 0.0)
        iDin = iDin.at[k].set(diffusion + drainage)
    return bTin, bphin, bSin, iphin, iTin, iSin, iDin, kperm, bphi_min


def update_hbrine(meltt, melts, dt, hin, hsn, hin_old,
                  hbr_old, fbri, dhS_top, dhS_bottom,
                  kperm, bphi_min, bphin, aice0, parameters):
    """Update brine height, Darcy drainage, flooding and runoff."""
    thinS = 0.05  # icepack_zbgc_shared.F90 parameter
    puny = parameters["puny"]
    h_ocn = (parameters["rhosi"] / parameters["rhow"] * hin
             + parameters["rhos"] / parameters["rhow"] * hsn)
    thick = ((hbr_old > thinS) & (hin_old > thinS)
             & (hin > thinS))

    top_thick = (-jnp.maximum(0.0, jnp.minimum(hin_old - hbr_old, meltt))
                 * parameters["rhoi"] / parameters["rhow"]
                 - jnp.maximum(0.0, melts) * parameters["rhos"]
                 / parameters["rhow"])
    hbr_thick = jnp.maximum(puny, hbr_old + dhS_bottom - top_thick)
    hbrocn = hbr_thick - h_ocn
    darcy_coeff = jnp.maximum(
        0.0, kperm * parameters["gravit"] /
        jnp.where(hbr_old > 0.0,
                  2.1e-6 * hbr_old, 1.0))

    above = thick & (hbrocn > 0.0) & (hbr_thick > thinS)
    below = thick & (hbrocn < 0.0) & (hbr_thick > thinS)
    porosity = jnp.where(above, bphin, bphi_min)
    runoff = -top_thick * aice0
    excess = jnp.maximum(0.0, hbrocn - runoff)
    exp_arg = darcy_coeff / jnp.where(porosity > 0.0, porosity, 1.0) * dt
    excess_new = jnp.where(
        exp_arg > 30.0,
        0.0, excess * jnp.exp(-exp_arg))
    deficit_new = jnp.where(
        exp_arg > 30.0,
        0.0, hbrocn * jnp.exp(-exp_arg))
    flooding = jnp.maximum(0.0, deficit_new - hbrocn) * aice0
    hbr_above = jnp.maximum(thinS, h_ocn + excess_new)
    hbr_below = jnp.maximum(thinS, h_ocn + deficit_new)
    drain_above = (-jnp.sign(excess) * (excess - (hbr_above - h_ocn))
                   / dt * porosity)
    drain_below = (-jnp.sign(hbrocn)
                   * (hbrocn - deficit_new + flooding) / dt * porosity)
    darcy_V = jnp.where(above, drain_above,
                         jnp.where(below, drain_below, 0.0))
    dhS_top_thick = jnp.where(
        above, top_thick - darcy_V * dt /
        jnp.where(porosity > 0.0, porosity, 1.0) + runoff,
        jnp.where(below, top_thick - darcy_V * dt /
                  jnp.where(porosity > 0.0, porosity, 1.0)
                  - flooding, top_thick))
    dh_direct = jnp.where(above, runoff,
                          jnp.where(below, -flooding, 0.0))
    hbr_thick = jnp.where(above, hbr_above,
                           jnp.where(below, hbr_below, hbr_thick))

    hbrmin = jnp.minimum(thinS, hin)
    hbr_thin = jnp.maximum(hbrmin, hbr_old + dhS_bottom - dhS_top)
    height_diff = hbr_thin - h_ocn
    hbr_thin = jnp.where(
        jnp.abs(height_diff) > 0.001,
        jnp.maximum(hbrmin, h_ocn + jnp.sign(height_diff) * 0.001),
        hbr_thin)
    dhS_top_thin = hbr_old - hbr_thin + dhS_bottom
    hbr = jnp.where(thick, hbr_thick, hbr_thin)
    dhS_top = jnp.where(thick, dhS_top_thick, dhS_top_thin)
    fbri = hbr / jnp.where(hin > 0.0, hin, 1.0)
    return (hbr, fbri, dhS_top, dhS_bottom,
            dhS_top, dhS_bottom, darcy_V, darcy_V,
            dh_direct, porosity)
