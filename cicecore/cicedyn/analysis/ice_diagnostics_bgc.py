"""Point diagnostics for brine and biogeochemistry, from ice_diagnostics_bgc.F90.

The host driver lends device arrays to these writers. Index the diagnostic
point before converting to host values; never download a horizontal field.
Diagnostic output does not take part in differentiation.
"""

import numpy as np

from cicecore.cicedyn.analysis import ice_diagnostics
from cicecore.cicedyn.general import ice_flux_bgc, ice_state
from cicecore.shared import ice_arrays_column, ice_domain_size, ice_fileunits
from cicecore.shared.ice_constants import mps_to_cmpdy
from icepack.columnphysics import icepack_parameters, icepack_tracers


def hbrine_diags():
    """Write dynamic brine height and salinity at the diagnostic points."""
    if not ice_diagnostics.print_points:
        return

    nu = ice_fileunits.nu_diag
    tr = icepack_tracers
    state = ice_state
    salinity = []
    category_salinity = []
    for n in range(ice_diagnostics.npnt):
        if int(ice_diagnostics.pmloc[n]) != 0:
            salinity.append(None)
            category_salinity.append(None)
            continue
        i = int(ice_diagnostics.piloc[n])
        j = int(ice_diagnostics.pjloc[n])
        brine = state.trcr[i, j, tr.nt_fbri] * state.vice[i, j] / state.aice[i, j] if state.aice[i, j] > 0 else 0.0
        brine1 = (state.trcrn[i, j, tr.nt_fbri, 0] * state.vicen[i, j, 0] /
                  state.aicen[i, j, 0] if state.aicen[i, j, 0] > 0 else 0.0)
        salinity.append((brine, state.trcrn[i, j, tr.nt_fbri, 0], brine1,
                         ice_arrays_column.darcy_V[i, j, 0]))
        category_salinity.append((state.trcrn[i, j, tr.nt_sice:tr.nt_sice + ice_domain_size.nilyr, 0],
                                  state.trcr[i, j, tr.nt_sice:tr.nt_sice + ice_domain_size.nilyr]))

    nu.write("------ hbrine ------\n")
    for k, label in enumerate(("hbrine, (m)        = ", "fbri, cat1 (m)     = ",
                               "hbrine cat1, (m)   = ", "darcy_V cat1, (m/s)= ")):
        nu.write(f"{label:25s}  {salinity[0][k]:24.17f}  {salinity[1][k]:24.17f}\n")
    if icepack_parameters.ktherm == 2:
        nu.write("------ Thermosaline Salinity ------\n")
        nu.write(f"{'Sice1(1) cat1 S (ppt)':25s}  {'Sice1(2) cat1 S':25s}\n")
        for k in range(ice_domain_size.nilyr):
            nu.write(f"{category_salinity[0][0][k]:24.17f}  {category_salinity[1][0][k]:24.17f}\n")
        nu.write(f"{'Sice(1) bulk S (ppt)':25s}  {'Sice(2) bulk S':25s}\n")
        for k in range(ice_domain_size.nilyr):
            nu.write(f"{category_salinity[0][1][k]:24.17f}  {category_salinity[1][1][k]:24.17f}\n")
    nu.flush()


def bgc_diags():
    """Write BGC point fields when that optional output is requested."""
    if not ice_diagnostics.print_points:
        return

    nu = ice_fileunits.nu_diag
    tr = icepack_tracers
    p = icepack_parameters
    col = ice_arrays_column
    state = ice_state
    bgc = ice_flux_bgc
    nblyr = int(ice_domain_size.nblyr)
    nilyr = int(ice_domain_size.nilyr)
    nslyr = int(ice_domain_size.nslyr)
    klev = 1 + nilyr + nslyr
    flux_units = mps_to_cmpdy / 100.0
    points = []

    for n in range(ice_diagnostics.npnt):
        if int(ice_diagnostics.pmloc[n]) != 0:
            points.append({})
            continue
        i, j = int(ice_diagnostics.piloc[n]), int(ice_diagnostics.pjloc[n])
        row = {}
        for key, active, nt, nlt in (
            ("NO", tr.tr_bgc_Nit, tr.nt_bgc_Nit, tr.nlt_bgc_Nit),
            ("Am", tr.tr_bgc_Am, tr.nt_bgc_Am, tr.nlt_bgc_Am),
            ("Sil", tr.tr_bgc_Sil, tr.nt_bgc_Sil, tr.nlt_bgc_Sil),
            ("hum", tr.tr_bgc_hum, tr.nt_bgc_hum, tr.nlt_bgc_hum),
            ("PON", tr.tr_bgc_PON, tr.nt_bgc_PON, tr.nlt_bgc_PON),
            ("DMSPp", tr.tr_bgc_DMS, tr.nt_bgc_DMSPp, tr.nlt_bgc_DMSPp),
            ("DMSPd", tr.tr_bgc_DMS, tr.nt_bgc_DMSPd, tr.nlt_bgc_DMSPd),
            ("DMS", tr.tr_bgc_DMS, tr.nt_bgc_DMS, tr.nlt_bgc_DMS),
        ):
            row[key + "_ocn"] = float(col.ocean_bio[i, j, nlt]) if active else 0.0
            row[key + "_sk"] = float(state.trcr[i, j, nt]) if active and p.skl_bgc else 0.0
            row[key + "_flux"] = float(bgc.flux_bio[i, j, nlt] * flux_units) if active else 0.0
            if p.z_tracers:
                row[key + "_ice"] = (state.trcr[i, j, nt:nt + nblyr + 1].copy()
                                     if active else np.zeros(nblyr + 1))
                row[key + "_snow"] = (state.trcr[i, j, nt + nblyr + 1:nt + nblyr + 3].copy()
                                      if active else np.zeros(2))
        if p.z_tracers:
            row["zfswin"] = np.sum(col.zfswin[i, j] * state.vicen[i, j][None, :], axis=1)
            row["Zoo"] = np.sum(col.Zoo[i, j] * state.vicen[i, j][None, :], axis=1)
            if state.vice[i, j] > 0:
                row["zfswin"] /= state.vice[i, j]
                row["Zoo"] /= state.vice[i, j]
            row["NO_atm"] = (col.fbio_atmice[i, j, tr.nlt_bgc_Nit] * flux_units
                             if tr.tr_bgc_Nit else 0.0)
            row["NO_snow_flux"] = (col.fbio_snoice[i, j, tr.nlt_bgc_Nit] * flux_units
                                   if tr.tr_bgc_Nit else 0.0)
            row["Am_atm"] = (col.fbio_atmice[i, j, tr.nlt_bgc_Am] * flux_units
                             if tr.tr_bgc_Am else 0.0)
            row["Am_snow_flux"] = (col.fbio_snoice[i, j, tr.nlt_bgc_Am] * flux_units
                                   if tr.tr_bgc_Am else 0.0)
            row["grow_net"] = float(col.grow_net[i, j]) if tr.tr_bgc_N else 0.0

        for name, active, count, nt_array, nlt_array in (
            ("N", tr.tr_bgc_N, tr.n_algae, tr.nt_bgc_N, tr.nlt_bgc_N),
            ("DOC", tr.tr_bgc_C, tr.n_doc, tr.nt_bgc_DOC, tr.nlt_bgc_DOC),
            ("DON", tr.tr_bgc_DON, tr.n_don, tr.nt_bgc_DON, tr.nlt_bgc_DON),
            ("Fed", tr.tr_bgc_Fe, tr.n_fed, tr.nt_bgc_Fed, tr.nlt_bgc_Fed),
            ("Fep", tr.tr_bgc_Fe, tr.n_fep, tr.nt_bgc_Fep, tr.nlt_bgc_Fep),
            ("zaero", tr.tr_zaero, tr.n_zaero, tr.nt_zaero, tr.nlt_zaero),
        ):
            ocean = np.zeros(count)
            skeletal = np.zeros(count)
            ice_profile = np.zeros((nblyr + 1, count))
            snow_profile = np.zeros((2, count))
            flux = np.zeros(count)
            total = np.zeros(count)
            atm_source = np.zeros(count)
            atm_ice = np.zeros(count)
            snow_ice = np.zeros(count)
            for k in range(count):
                if not active:
                    continue
                nt, nlt = int(nt_array[k]), int(nlt_array[k])
                ocean[k] = col.ocean_bio[i, j, nlt]
                if p.skl_bgc:
                    skeletal[k] = state.trcr[i, j, nt]
                if p.z_tracers:
                    ice_profile[:, k] = state.trcr[i, j, nt:nt + nblyr + 1]
                    snow_profile[:, k] = state.trcr[i, j, nt + nblyr + 1:nt + nblyr + 3]
                flux[k] = bgc.flux_bio[i, j, nlt] * flux_units
                if p.z_tracers and name == "N":
                    total[k] = col.ice_bio_net[i, j, nlt]
                if p.z_tracers and name == "zaero":
                    atm_source[k] = bgc.flux_bio_atm[i, j, nlt] * flux_units
                    atm_ice[k] = col.fbio_atmice[i, j, nlt] * flux_units
                    snow_ice[k] = col.fbio_snoice[i, j, nlt] * flux_units
            row[name] = (ocean, skeletal, ice_profile, snow_profile, flux,
                         total, atm_source, atm_ice, snow_ice)

        # The shortwave grid includes the snow top, internal snow and ice
        # layers, and the bottom interface (k=0..klev in Fortran).
        if p.dEdd_algae:
            row["chl_sw"] = (col.trcrn_sw[i, j, tr.nlt_chl_sw:tr.nlt_chl_sw + klev + 1, 0].copy()
                             if tr.tr_bgc_N else np.zeros(klev + 1))
            row["zaero_sw"] = np.zeros((klev + 1, tr.n_zaero))
            if tr.tr_zaero:
                for k in range(tr.n_zaero):
                    nt = int(tr.nlt_zaero_sw[k])
                    row["zaero_sw"][:, k] = col.trcrn_sw[i, j, nt:nt + klev + 1, 0]
        points.append(row)

    # The Fortran formats 900/802 pair the two geographic points. These
    # local writers are used by every optional tracer group below.
    def write_pair(label, left, right):
        nu.write(f"{label:25s}  {float(left):24.17f}  {float(right):24.17f}\n")

    def write_profile(label, name, column=None, snow=True):
        nu.write(f"{label:25s}  {label:25s}\n")
        if snow:
            for k in range(2):
                left = points[0][name + "_snow"][k] if column is None else points[0][name][3][k, column]
                right = points[1][name + "_snow"][k] if column is None else points[1][name][3][k, column]
                write_pair("", left, right)
        for k in range(nblyr + 1):
            left = points[0][name + "_ice"][k] if column is None else points[0][name][2][k, column]
            right = points[1][name + "_ice"][k] if column is None else points[1][name][2][k, column]
            write_pair("", left, right)

    if p.z_tracers:
        nu.write("zfswin PAR\n")
        for k in range(nblyr + 1):
            write_pair("", points[0]["zfswin"][k], points[1]["zfswin"][k])
        nu.write("Losses: Zoo, Brine Conc.\n")
        for k in range(nblyr + 1):
            write_pair("", points[0]["Zoo"][k], points[1]["Zoo"][k])

    for name, enabled, title in (
        ("NO", tr.tr_bgc_Nit, "nitrate conc. (mmol/m^3) or flux (mmol/m^2/d)"),
        ("PON", tr.tr_bgc_PON and p.z_tracers, "PON snow + ice conc. (mmol/m^3)"),
        ("hum", tr.tr_bgc_hum, "hum snow + ice conc. (mmolC/m^3)"),
        ("Am", tr.tr_bgc_Am, "ammonium conc. (mmol/m^3) or flux (mmol/m^2/d)"),
    ):
        if not enabled:
            continue
        nu.write("---------------------------------------------------\n" + title + "\n")
        if name != "PON":
            write_pair("Ocean conc =", points[0][name + "_ocn"], points[1][name + "_ocn"])
            write_pair("ice-ocean flux =", points[0][name + "_flux"], points[1][name + "_flux"])
        if p.skl_bgc:
            write_pair("Bulk ice conc. =", points[0][name + "_sk"], points[1][name + "_sk"])
        elif p.z_tracers:
            if name in ("NO", "Am"):
                write_pair("atm-ice flux =", points[0][name + "_atm"], points[1][name + "_atm"])
                write_pair("snow-ice flux =", points[0][name + "_snow_flux"], points[1][name + "_snow_flux"])
            write_profile(name, name)

    if tr.tr_bgc_N:
        nu.write("---------------------------------------------------\n")
        if p.z_tracers:
            write_pair("tot algal growth (1/d) =", points[0]["grow_net"], points[1]["grow_net"])
        for k in range(tr.n_algae):
            nu.write(f"algal conc. (mmol N/m^3) or flux (mmol N/m^2/d), type: {k + 1}\n")
            write_pair("Ocean conc =", points[0]["N"][0][k], points[1]["N"][0][k])
            write_pair("ice-ocean flux =", points[0]["N"][4][k], points[1]["N"][4][k])
            if p.skl_bgc:
                write_pair("Bulk ice conc. =", points[0]["N"][1][k], points[1]["N"][1][k])
            elif p.z_tracers:
                write_pair("Tot ice (mmolN/m^2) =", points[0]["N"][5][k], points[1]["N"][5][k])
                write_profile("algal N", "N", k)

    for name, enabled, count, title in (
        ("DOC", tr.tr_bgc_C, min(1, tr.n_doc), "DOC conc. (mmol C/m^3)"),
        ("DON", tr.tr_bgc_DON, tr.n_don, "DON conc. (mmol N/m^3)"),
        ("Fed", tr.tr_bgc_Fe, tr.n_fed, "dFe conc. (nM)"),
        ("Fep", tr.tr_bgc_Fe, tr.n_fep, "pFe conc. (nM)"),
    ):
        if not enabled:
            continue
        for k in range(count):
            nu.write(f"---------------------------------------------------\n{title}, type: {k + 1}\n")
            write_pair("Ocean conc =", points[0][name][0][k], points[1][name][0][k])
            if p.skl_bgc:
                write_pair("Bulk ice conc. =", points[0][name][1][k], points[1][name][1][k])
            elif p.z_tracers:
                write_profile(name, name, k)

    if tr.tr_bgc_DMS:
        nu.write("---------------------------------------------------\nDMS (mmol/m^3)\n")
        for name in ("DMSPp", "DMS"):
            write_pair("Ocean " + name + " =", points[0][name + "_ocn"], points[1][name + "_ocn"])
        if p.skl_bgc:
            for name in ("DMSPp", "DMSPd", "DMS"):
                write_pair("Ice " + name + " =", points[0][name + "_sk"], points[1][name + "_sk"])

    if tr.tr_zaero and p.z_tracers:
        for k in range(tr.n_zaero):
            nu.write(f"---------------------------------------------------\naerosol conc. (kg/m^3) or flux (kg/m^2/d), type: {k + 1}\n")
            for label, index in (("Atm source flux =", 6), ("ice-ocean flux*aice =", 4),
                                 ("atm-ice flux*aice =", 7), ("snow-ice flux*aice =", 8)):
                write_pair(label, points[0]["zaero"][index][k], points[1]["zaero"][index][k])
            write_profile("aerosol", "zaero", k)
    if p.dEdd_algae:
        if tr.tr_zaero:
            for k in range(tr.n_zaero):
                nu.write("Cat 1 aerosol conc. (kg/m^3) on delta-Eddington grid\n")
                for lev in range(klev + 1):
                    write_pair("", points[0]["zaero_sw"][lev, k], points[1]["zaero_sw"][lev, k])
        if tr.tr_bgc_N:
            nu.write("Cat 1 chl (mg/m^3) on delta-Eddington grid\n")
            for lev in range(klev + 1):
                write_pair("", points[0]["chl_sw"][lev], points[1]["chl_sw"][lev])
    nu.flush()
