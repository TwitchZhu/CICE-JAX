"""snow redistribution and metamorphism

author for JAX version: Xinyuan Zhu
XZ: converted from F90 to Python

authors Elizabeth Hunke, LANL
        Nicole Jeffery, LANL

``icepack_init_snow`` builds the snow-metamorphism lookup table.
``init_snowtable`` stays in ``ice_forcing``. Snow grain metamorphism has a
JAX time-step path, as does wind redistribution through ``snow_redist``.
"""

import numpy as np
import jax.numpy as jnp

from icepack.columnphysics import icepack_parameters
from icepack.columnphysics.icepack_therm_shared import adjust_enthalpy
from icepack.columnphysics.icepack_kinds import dbl_kind, int_kind, log_kind

# minimum volume growth rate 1.28x10^-8 mm^3/s/4/pi
drsnw_min_o = dbl_kind(1.0186)  # Bun 1989  (um^3/s)

min_rhos = None  # snowtable axis data, assumes linear data
del_rhos = None
min_Tgrd = None
del_Tgrd = None
min_T = None
del_T = None

lin_rhos = log_kind(False)  # flag to specify whether the snowtable dimensions are linear
lin_Tgrd = log_kind(False)  # this will allow quick lookup
lin_T = log_kind(False)


def _axis(count, minimum, delta):
    values = np.zeros(count, dtype=dbl_kind)
    for n in range(1, count + 1):
        values[n - 1] = dbl_kind(minimum) + dbl_kind(n - 1) * dbl_kind(delta)
    return values


def _require_shape(array, shape, message):
    if array is None:
        return False
    if tuple(np.shape(array)) != tuple(shape):
        raise ValueError(message)
    return True


def icepack_init_snow():
    """Updates snow tracers

    authors: Elizabeth C. Hunke, LANL
             Nicole Jeffery, LANL
    """
    global min_rhos, del_rhos, min_Tgrd, del_Tgrd, min_T, del_T
    global lin_rhos, lin_Tgrd, lin_T

    subname = "(icepack_init_snow)"

    #-----------------------------------------------------------------
    # Snow metamorphism lookup table
    #-----------------------------------------------------------------

    # if snw_aging_table = 'snicar'
    # best-fit parameters are read from a table (netcdf)
    # snowage_tau, snowage_kappa, snowage_drdt0
    #  11 temperatures from 223.15 to 273.15 K by 5.0
    #  31 temperature gradients from 0 to 300 K/m by 10
    #   8 snow densities from 50 to 400 kg/m3 by 50

    # if snw_aging_table = 'test'
    # for testing Icepack without netcdf,
    # use a subsampled, hard-coded table
    #   5 temperatures from 243.15 by 5.0 K
    #   5 temperature gradients from 0 to 40 K/m by 10
    #   1 snow density at 300 kg/m3

    # if snw_aging_table = 'file'
    # all data has to be passed into icepack_parameters

    if not bool(icepack_parameters.snwgrain):
        return

    table = str(icepack_parameters.snw_aging_table).strip()
    if table == "snicar":  # read netcdf file
        isnw_rhos = 8    # maxiumum snow density index
        isnw_Tgrd = 31   # maxiumum temperature gradient index
        isnw_T = 11      # maxiumum temperature index
        min_rhos = dbl_kind(50.0)  # snowtable dimension data
        del_rhos = dbl_kind(50.0)
        lin_rhos = log_kind(True)
        min_Tgrd = dbl_kind(0.0)
        del_Tgrd = dbl_kind(10.0)
        lin_Tgrd = log_kind(True)
        min_T = dbl_kind(223.15)
        del_T = dbl_kind(5.0)
        lin_T = log_kind(True)
        shape = (isnw_rhos, isnw_Tgrd, isnw_T)
        # check if these are already allocated/set, if so, make sure size is OK
        # The netcdf values are read by init_snowtable, which stays in ice_forcing.
        if not _require_shape(
            icepack_parameters.snowage_tau, shape,
            subname + "ERROR: snowage_tau size snw_aging_table=snicar",
        ):
            icepack_parameters.snowage_tau = np.zeros(shape, dtype=dbl_kind)
        if not _require_shape(
            icepack_parameters.snowage_kappa, shape,
            subname + "ERROR: snowage_kappa size snw_aging_table=snicar",
        ):
            icepack_parameters.snowage_kappa = np.zeros(shape, dtype=dbl_kind)
        if not _require_shape(
            icepack_parameters.snowage_drdt0, shape,
            subname + "ERROR: snowage_drdt0 size snw_aging_table=snicar",
        ):
            icepack_parameters.snowage_drdt0 = np.zeros(shape, dtype=dbl_kind)
        icepack_parameters.snowage_rhos = _axis(isnw_rhos, min_rhos, del_rhos)
        icepack_parameters.snowage_Tgrd = _axis(isnw_Tgrd, min_Tgrd, del_Tgrd)
        icepack_parameters.snowage_T = _axis(isnw_T, min_T, del_T)
    elif table == "file":
        isnw_rhos = -1  # maxiumum snow density index
        isnw_Tgrd = -1  # maxiumum temperature gradient index
        isnw_T = -1     # maxiumum temperature index
    elif table == "test":
        isnw_rhos = 1   # maxiumum snow density index
        isnw_Tgrd = 5   # maxiumum temperature gradient index
        isnw_T = 5      # maxiumum temperature index
        min_rhos = dbl_kind(300.0)  # snowtable dimension data
        del_rhos = dbl_kind(50.0)
        lin_rhos = log_kind(True)
        min_Tgrd = dbl_kind(0.0)
        del_Tgrd = dbl_kind(10.0)
        lin_Tgrd = log_kind(True)
        min_T = dbl_kind(243.15)
        del_T = dbl_kind(5.0)
        lin_T = log_kind(True)
        shape = (isnw_rhos, isnw_Tgrd, isnw_T)
        icepack_parameters.snowage_rhos = _axis(isnw_rhos, min_rhos, del_rhos)
        icepack_parameters.snowage_Tgrd = _axis(isnw_Tgrd, min_Tgrd, del_Tgrd)
        icepack_parameters.snowage_T = _axis(isnw_T, min_T, del_T)

        # Subset of dry snow aging parameters. Fortran reshape is column-major.
        icepack_parameters.snowage_tau = np.array([
            3.34947394, 4.02124159, 4.03328223,
            3.02686921, 2.14125851, 3.97008737,
            4.72725821, 3.65313459, 2.41198936,
            2.53065623e-1, 4.60286630, 4.99721440,
            3.29168191, 2.66426779e-1, 9.15830596e-5,
            5.33186128, 4.90833452, 1.55269141,
            1.31225526e-3, 9.36078196e-4, 6.25428631,
            5.04394794, 2.92857366e-3, 9.01488751e-3,
            1.19037046e-2,
        ], dtype=dbl_kind).reshape(shape, order="F")

        icepack_parameters.snowage_kappa = np.array([
            0.60824438, 0.56442972, 0.5527807, 0.64299537, 0.77672359,
            0.57105932, 0.52791041, 0.59868076, 0.7487191, 1.57946877,
            0.54236508, 0.52458285, 0.65520877, 1.52356017, 4.37789838,
            0.51449138, 0.54494334, 0.91628508, 3.28847035, 3.64418487,
            0.48538708, 0.55386601, 2.81247103, 2.72445522, 2.8230216,
        ], dtype=dbl_kind).reshape(shape, order="F")

        icepack_parameters.snowage_drdt0 = np.array([
            1.26597871, 1.26602416, 1.26613263, 1.26620414, 1.26629424,
            1.92418877, 1.92430063, 1.92445964, 1.92451557, 1.92469806,
            2.79086547, 2.79147315, 2.79137562, 2.79150846, 2.79216439,
            3.85605846, 3.85668001, 3.85844559, 3.86073682, 3.863199,
            5.0861907, 5.08765668, 5.09200195, 5.09665276, 5.10079895,
        ], dtype=dbl_kind).reshape(shape, order="F")
    else:
        raise ValueError(subname + "ERROR: snw_aging_table value")

    icepack_parameters.isnw_rhos = int_kind(isnw_rhos)
    icepack_parameters.isnw_Tgrd = int_kind(isnw_Tgrd)
    icepack_parameters.isnw_T = int_kind(isnw_T)


def snow_dry_metamorph(dt, rsnw, zqsn, Tsfc, zTin1, hsn, hin,
                       nilyr, parameters):
    """Dry snow grain growth for one category over a batch of grid cells.

    Snow layers occupy the last axis, from the surface downward. The three
    lookup-table axes retain their Fortran order: density, gradient, temperature.
    """
    nslyr = zqsn.shape[-1]
    puny = parameters["puny"]
    dzs = hsn / nslyr
    dzi = hin / nilyr
    dz = dzs + dzi
    zTsn = (parameters["Lfresh"] + zqsn / parameters["rhos"]) / parameters["cp_ice"]
    zdTdz = jnp.zeros_like(zTsn)
    limit = 10.0 * parameters["snowage_Tgrd"].shape[0]

    if nslyr == 1:
        gradient = jnp.abs(zTsn[..., 0] * dzi + zTin1 * dzs
                           - Tsfc * dz) / (dz * hsn + puny)
        zdTdz = zdTdz.at[..., 0].set(jnp.minimum(limit, gradient))
    else:
        for k in range(1, nslyr):
            if k == 1:
                gradient = (jnp.abs((zTsn[..., 0] + zTsn[..., 1]) * 0.5
                                    - Tsfc) / (dzs + puny))
            else:
                gradient = (jnp.abs(zTsn[..., k - 2] - zTsn[..., k])
                            * 0.5 / (dzs + puny))
            zdTdz = zdTdz.at[..., k - 1].set(jnp.minimum(limit, gradient))
        gradient = jnp.abs(
            zTsn[..., -1] * dzi + zTin1 * dzs
            - (zTsn[..., -1] + zTsn[..., -2]) * 0.5 * dz
        ) / (dz * dzs + puny)
        zdTdz = zdTdz.at[..., -1].set(jnp.minimum(limit, gradient))

    # Fortran NINT rounds half steps upward for the physical table range.
    rhos_axis = parameters["snowage_rhos"]
    gradient_axis = parameters["snowage_Tgrd"]
    temperature_axis = parameters["snowage_T"]
    rhos_delta = (rhos_axis[1] - rhos_axis[0]
                  if rhos_axis.shape[0] > 1 else rhos_axis[0])
    gradient_delta = (gradient_axis[1] - gradient_axis[0]
                      if gradient_axis.shape[0] > 1 else gradient_axis[0])
    temperature_delta = (temperature_axis[1] - temperature_axis[0]
                         if temperature_axis.shape[0] > 1 else temperature_axis[0])
    rhos_idx = jnp.clip(
        jnp.floor((parameters["rhos"] - rhos_axis[0]) / rhos_delta + 0.5).astype(int),
        0, rhos_axis.shape[0] - 1)
    gradient_idx = jnp.clip(
        jnp.floor((zdTdz - gradient_axis[0]) / gradient_delta + 0.5).astype(int),
        0, gradient_axis.shape[0] - 1)
    temperature_idx = jnp.clip(
        jnp.floor(jnp.abs(zTsn + parameters["Tffresh"]
                          - temperature_axis[0]) / temperature_delta + 0.5).astype(int),
        0, temperature_axis.shape[0] - 1)
    tau = parameters["snowage_tau"][rhos_idx, gradient_idx, temperature_idx]
    kappa = parameters["snowage_kappa"][rhos_idx, gradient_idx, temperature_idx]
    drdt0 = parameters["snowage_drdt0"][rhos_idx, gradient_idx, temperature_idx]
    dr_fresh = jnp.maximum(0.0, rsnw - parameters["rsnw_fall"])
    return drdt0 * (tau / (dr_fresh + tau)) ** (1.0 / kappa) * (dt / 3600.0)


def snow_wet_metamorph(dt, rsnw, smice, smliq, parameters):
    """Brun wet-snow grain growth, with the original 10% liquid cap."""
    active = (smice + smliq > 0.0) & (rsnw > 0.0)
    fliq = jnp.minimum(smliq / jnp.where(active, smice + smliq, 1.0), 0.1)
    return jnp.where(
        active,
        parameters["snw_growth_wet"] * fliq**3 * dt
        / (4.0 * parameters["pi"] * jnp.where(active, rsnw**2, 1.0)),
        0.0,
    )


def update_snow_radius(dt, rsnw, hin, Tsfc, zTin1, hsn,
                       zqsn, smice, smliq, nilyr, parameters):
    """Age snow grains, keeping snow and ice mass tracers as in Fortran."""
    active = (hsn > parameters["puny"]) & (hin > parameters["puny"])
    dry = snow_dry_metamorph(dt, rsnw, zqsn, Tsfc, zTin1, hsn, hin, nilyr,
                             parameters)
    wet = snow_wet_metamorph(dt, rsnw, smice, smliq, parameters)
    minimum = (parameters["drsnw_min"] * drsnw_min_o * dt
               / jnp.where(rsnw > 0.0, rsnw**2, 1.0))
    grown = jnp.minimum(parameters["rsnw_tmax"],
                        rsnw + jnp.maximum(dry, minimum) + wet)
    no_snow = jnp.maximum(parameters["rsnw_fall"],
                          jnp.minimum(parameters["rsnw_tmax"], rsnw))
    rsnw = jnp.where(active[..., None], grown, no_snow)
    smice = jnp.where(active[..., None], smice, parameters["rhos"])
    smliq = jnp.where(active[..., None], smliq, 0.0)
    return rsnw, smice, smliq


def drain_snow(vsnon, aicen, massice, massliq, meltsliq,
               use_smliq_pnd, parameters):
    """Pass excess liquid downward through snow layers, top to bottom.

    ``massice`` and ``massliq`` are layer masses per ice-covered area; snow
    layers occupy the last axis. ``meltsliq`` is updated only for the pond
    option, exactly as in the Fortran routine.
    """
    nslyr = massliq.shape[-1]
    hsn = jnp.where(aicen > 0.0, vsnon / jnp.where(aicen > 0.0, aicen, 1.0), 0.0)
    active = hsn > parameters["puny"]
    hslyr = jnp.where(active, hsn / nslyr, 1.0)
    inflow = jnp.zeros_like(hsn)
    for k in range(nslyr):
        liquid = massliq[..., k] + inflow
        phi_ice = jnp.minimum(1.0, massice[..., k] / (parameters["rhoi"] * hslyr))
        phi_liq = liquid / (parameters["rhofresh"] * hslyr)
        outflow = jnp.maximum(
            0.0, (phi_liq - parameters["snwliq_max"] * (1.0 - phi_ice))
            * parameters["rhofresh"] * hslyr)
        massliq = massliq.at[..., k].set(jnp.where(active, liquid - outflow,
                                                   massliq[..., k]))
        inflow = jnp.where(active, outflow, 0.0)
    if use_smliq_pnd:
        meltsliq = jnp.where(active, inflow, meltsliq)
    return massliq, meltsliq


def snow_redist(dt, wind, aice, ain, vin, vsn, zqsn, alvl, vlvl,
                fresh, fhocn, fsloss, rhos_cmpn, fsnow,
                snwredist, parameters):
    """Move wind-eroded snow among ice categories and compact new snow.

    The last two axes of snow fields are snow layer and ice category. The
    Fortran ``ITDsd`` variance calculation is commented out, so its variance
    and suspended mass remain zero. ``ITDrdg`` uses level and ridged ice.
    Conservation failures are returned for host-side reporting.
    """
    puny = parameters["puny"]
    rhos = parameters["rhos"]
    Lfresh = parameters["Lfresh"]
    nslyr = zqsn.shape[-2]
    ncat = vsn.shape[-1]
    active = aice > puny
    initial_vsn = vsn
    initial_zqsn = zqsn
    initial_fresh = fresh
    initial_fhocn = fhocn
    initial_fsloss = fsloss
    initial_rhos_cmpn = rhos_cmpn
    # Conservation checks: retain the starting snow volume and energy.
    tmp1 = jnp.zeros_like(aice)
    tmp3 = jnp.zeros_like(aice)
    for n in range(ncat):
        tmp1 = tmp1 + vsn[..., n]
        for k in range(nslyr):
            tmp3 = tmp3 + vsn[..., n] * zqsn[..., k, n] / nslyr

    # Category thickness and sums. Temporarily remove latent heat from snow
    # enthalpy so redistribution cannot create positive snow enthalpy.
    hin = jnp.zeros_like(vsn)
    hsn = jnp.zeros_like(vsn)
    hsn_new = jnp.zeros_like(vsn)
    suma = jnp.zeros_like(aice)
    sumv = jnp.zeros_like(aice)
    for n in range(ncat):
        present = ain[..., n] > puny
        hin = hin.at[..., n].set(jnp.where(
            present, vin[..., n] / jnp.where(present, ain[..., n], 1.0), 0.0))
        hsn = hsn.at[..., n].set(jnp.where(
            present, vsn[..., n] / jnp.where(present, ain[..., n], 1.0), 0.0))
        hsn_new = hsn_new.at[..., n].set(hsn[..., n])
        suma = suma + ain[..., n]
        sumv = sumv + vin[..., n]
        for k in range(nslyr):
            q = jnp.minimum(zqsn[..., k, n] + Lfresh * rhos, 0.0)
            zqsn = zqsn.at[..., k, n].set(q)

    # Standard deviation of ice thickness; ITDrdg includes level/ridged ice.
    work = jnp.zeros_like(aice)
    asnw_lvl = jnp.zeros_like(aice)
    ardg = jnp.zeros_like(vsn)
    sfac = jnp.zeros_like(vsn)
    if snwredist == "ITDrdg":
        for n in range(ncat):
            ardg_n = 1.0 - alvl[..., n]
            al = alvl[..., n] * ain[..., n]
            ar = ardg_n * ain[..., n]
            hlvl = jnp.where(al > puny, vin[..., n] * vlvl[..., n]
                             / jnp.where(al > puny, al, 1.0), 0.0)
            hrdg = jnp.where(ar > puny, vin[..., n] * (1.0 - vlvl[..., n])
                             / jnp.where(ar > puny, ar, 1.0), 0.0)
            work = work + al * (hlvl - sumv)**2 + ar * (hrdg - sumv)**2
            sfac_n = jnp.where(
                ardg_n > 0.0,
                jnp.minimum(parameters["snwlvlfac"], alvl[..., n]
                            / jnp.where(ardg_n > 0.0, ardg_n, 1.0)),
                parameters["snwlvlfac"])
            ardg = ardg.at[..., n].set(ardg_n)
            sfac = sfac.at[..., n].set(sfac_n)
            asnw_lvl = asnw_lvl + al - sfac_n * ar
        asnw_lvl = asnw_lvl / jnp.where(suma > 0.0, suma, 1.0)
    if snwredist == "ITDrdg":
        ITDsd = jnp.where(work > 0.0,
                          jnp.sqrt(jnp.where(work > 0.0, work, 1.0)), 0.0)
    else:
        ITDsd = jnp.zeros_like(aice)
    # Fraction of suspended snow lost in leads, then potential suspension.
    flost = (1.0 - suma) * jnp.exp(-ITDsd)
    alost = 1.0 - suma * (1.0 - flost)

    msusp = jnp.zeros_like(aice)
    for n in range(ncat):
        Vseas = jnp.maximum((rhos_cmpn[..., 0, n] - 44.6) / 174.0, 0.0)
        potential = (ain[..., n] * 1.0e-5 * dt *
                     jnp.maximum(wind - Vseas, 0.0) *
                     (parameters["rhosmax"] - rhos_cmpn[..., 0, n]) /
                     (parameters["rhosmax"] *
                      jnp.where(ITDsd > puny, ITDsd, 1.0)))
        msusp = msusp + jnp.where(ITDsd > puny, potential, 0.0)

    # Erode the top snow layers first and track suspended mass and energy.
    msnw_susp = jnp.zeros_like(aice)
    esnw_susp = jnp.zeros_like(aice)
    klyr = jnp.ones_like(vsn, dtype=jnp.int32)
    e_erosion = jnp.zeros_like(vsn)
    for n in range(ncat):
        present = ain[..., n] > puny
        m_erosion = jnp.where(present, jnp.minimum(msusp, rhos * vsn[..., n]), 0.0)
        eroding = m_erosion > puny
        summ = jnp.zeros_like(aice)
        dmp = m_erosion
        e_erosion_n = jnp.zeros_like(aice)
        hsn_new_n = hsn_new[..., n]
        klyr_n = klyr[..., n]
        for k in range(nslyr):
            taking = eroding & (dmp > 0.0)
            dhsn = jnp.where(
                taking,
                jnp.minimum(hsn[..., n] / nslyr,
                            dmp / (rhos * jnp.where(present, ain[..., n], 1.0))),
                0.0)
            msnw_susp = msnw_susp + dhsn * rhos * ain[..., n]
            hsn_new_n = hsn_new_n - dhsn
            e_erosion_n = e_erosion_n + dhsn * zqsn[..., k, n] * ain[..., n]
            klyr_n = jnp.where(taking, k + 1, klyr_n)
            summ = summ + jnp.where(taking, rhos * vsn[..., n] / nslyr, 0.0)
            dmp = jnp.where(taking, jnp.maximum(m_erosion - summ, 0.0), dmp)
        hsn_new = hsn_new.at[..., n].set(hsn_new_n)
        klyr = klyr.at[..., n].set(klyr_n)
        e_erosion = e_erosion.at[..., n].set(e_erosion_n)
        esnw_susp = esnw_susp + e_erosion_n

    # Redeposit snow by category and remap unequal layers to equal layers.
    e_redep = jnp.zeros_like(vsn)
    for n in range(ncat):
        if snwredist == "ITDrdg":
            work = (ain[..., n] * (1.0 - flost) *
                    (ardg[..., n] * (1.0 + sfac[..., n]) + asnw_lvl))
        else:
            work = ain[..., n] * (1.0 - flost)
        m_redep = msnw_susp * work
        e_redeptmp = esnw_susp * work
        present = ain[..., n] > puny
        dhsn = jnp.where(present, m_redep /
                         (rhos * jnp.where(present, ain[..., n], 1.0)), 0.0)
        redeposit = present & (jnp.abs(dhsn) > 0.0)
        new_vsn = (hsn_new[..., n] + dhsn) * ain[..., n]
        vsn = vsn.at[..., n].set(jnp.where(redeposit, new_vsn, vsn[..., n]))
        e_redep = e_redep.at[..., n].set(jnp.where(redeposit, e_redeptmp, 0.0))
        de = e_redeptmp / klyr[..., n]
        sumq = jnp.zeros_like(aice)
        for k in range(nslyr):
            affected = redeposit & (k < klyr[..., n])
            q = ((ain[..., n] * hsn_new[..., n] * zqsn[..., k, n] + de)
                 / jnp.where(redeposit, vsn[..., n], 1.0))
            sumq = sumq + jnp.where(affected, jnp.maximum(q, 0.0), 0.0)
            zqsn = zqsn.at[..., k, n].set(jnp.where(
                affected, jnp.minimum(q, 0.0), zqsn[..., k, n]))
        for k in range(nslyr):
            last = redeposit & (klyr[..., n] == k + 1)
            zqsn = zqsn.at[..., k, n].set(jnp.where(
                last, jnp.minimum(zqsn[..., k, n] + sumq, 0.0),
                zqsn[..., k, n]))

        hsn_new_n = hsn_new[..., n] + jnp.where(redeposit, dhsn, 0.0)
        if nslyr > 1:
            old_layer = hsn[..., n] / nslyr
            dzi = jnp.stack([
                old_layer + jnp.where(k < klyr[..., n],
                                      dhsn / klyr[..., n], 0.0)
                for k in range(nslyr)
            ], axis=-1)
            z1 = [jnp.zeros_like(aice)]
            z2 = [jnp.zeros_like(aice)]
            for k in range(nslyr - 1):
                z1.append(z1[-1] + dzi[..., k])
                z2.append(z2[-1] + hsn_new_n / nslyr)
            z1.append(hsn_new_n)
            z2.append(hsn_new_n)
            remapped = adjust_enthalpy(
                jnp.stack(z1, axis=-1), jnp.stack(z2, axis=-1),
                hsn_new_n / nslyr, hsn_new_n, zqsn[..., :, n], puny)
            zqsn = zqsn.at[..., :, n].set(jnp.where(
                redeposit[..., None], remapped, zqsn[..., :, n]))
        hsn_new = hsn_new.at[..., n].set(hsn_new_n)
        for k in range(nslyr):
            zqsn = zqsn.at[..., k, n].add(Lfresh * rhos)

    # Account for snow lost into leads and check mass/energy conservation.
    mlost = msnw_susp * alost
    fsloss = fsloss + mlost / dt
    tmp2 = jnp.zeros_like(aice)
    for n in range(ncat):
        tmp2 = tmp2 + vsn[..., n]
    scale = jnp.where(tmp2 > tmp1, tmp1 / jnp.where(tmp2 > tmp1, tmp2, 1.0), 1.0)
    vsn = vsn * scale[..., None]
    tmp2 = jnp.zeros_like(aice)
    for n in range(ncat):
        tmp2 = tmp2 + vsn[..., n]
    fresh = fresh + jnp.where(tmp2 < tmp1, rhos * (tmp1 - tmp2) / dt, 0.0)
    mass_error = jnp.abs(tmp1 - (tmp2 + mlost / rhos)) > puny

    tmp4 = jnp.zeros_like(aice)
    tmp5 = jnp.zeros_like(aice)
    for n in range(ncat):
        for k in range(nslyr):
            tmp4 = tmp4 + vsn[..., n] * zqsn[..., k, n] / nslyr
        tmp5 = tmp5 - e_erosion[..., n] + e_redep[..., n]
    tmp5 = tmp5 + esnw_susp * alost
    fhocn = fhocn + (tmp3 - tmp4) / dt
    energy_error = jnp.abs(tmp5) > nslyr * Lfresh * puny

    # Compact freshly fallen or redistributed snow with the local wind.
    drhos = parameters["drhosdwind"] * jnp.maximum(wind - parameters["windmin"], 0.0)
    for n in range(ncat):
        has_snow = vsn[..., n] > puny
        hslab = jnp.where(
            fsnow > 0.0,
            jnp.maximum(jnp.minimum(
                fsnow * dt / (rhos + drhos), hsn_new[..., n] - hsn[..., n]), 0.0),
            0.0)
        hslyr = jnp.where(has_snow, hsn_new[..., n] / nslyr, 1.0)
        for k in range(nslyr):
            work = jnp.clip(hslab - hslyr * k, 0.0, hslyr)
            density = jnp.minimum(
                rhos_cmpn[..., k, n] + drhos * work / hslyr,
                parameters["rhosmax"])
            rhos_cmpn = rhos_cmpn.at[..., k, n].set(jnp.where(
                has_snow, density, rhos_cmpn[..., k, n]))

    vsn = jnp.where(active[..., None], vsn, initial_vsn)
    zqsn = jnp.where(active[..., None, None], zqsn, initial_zqsn)
    fresh = jnp.where(active, fresh, initial_fresh)
    fhocn = jnp.where(active, fhocn, initial_fhocn)
    fsloss = jnp.where(active, fsloss, initial_fsloss)
    rhos_cmpn = jnp.where(active[..., None, None], rhos_cmpn, initial_rhos_cmpn)
    return (vsn, zqsn, fresh, fhocn, fsloss, rhos_cmpn,
            active & mass_error, active & energy_error)
