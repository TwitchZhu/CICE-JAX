"""Flux variable declarations for biogeochemistry

author for JAX version: Xinyuan Zhu
XZ: converted from F90 to Python

author Elizabeth C. Hunke, LANL

``alloc_flux_bgc`` is ported. ``bgcflux_ice_to_ocn`` is not.
``cpl_bgc`` is the namelist value on ``ice_init``.
"""

import numpy as np

from cicecore.cicedyn.general import ice_init
from cicecore.shared import ice_arrays_column
from cicecore.shared import ice_domain_size
from cicecore.shared.ice_constants import c0
from cicecore.shared.ice_kinds_mod import dbl_kind, log_kind

# in from atmosphere
# coupling variable for both tr_aero and tr_zaero
fiso_atm = None   # isotope deposition rate (kg/m^2 s)
faero_atm = None  # aerosol deposition rate (kg/m^2 s)

flux_bio_atm = None  # all bio fluxes to ice from atmosphere

# out to ocean
fiso_ocn = None   # isotope flux to ocean  (kg/m^2/s)
faero_ocn = None  # aerosol flux to ocean  (kg/m^2/s)

flux_bio = None     # all bio fluxes to ocean
flux_bio_ai = None  # all bio fluxes to ocean, averaged over grid cell

# internal
cpl_bgc = None  # switch to couple BGC via drivers

hin_old = None  # old ice thickness
dsnown = None   # change in snow thickness in category n (m)

nit = None    # ocean nitrate (mmol/m^3)
amm = None    # ammonia/um (mmol/m^3)
sil = None    # silicate (mmol/m^3)
dmsp = None   # dmsp (mmol/m^3)
dms = None    # dms (mmol/m^3)
hum = None    # humic material carbon (mmol/m^3)
fnit = None   # ice-ocean nitrate flux (mmol/m^2/s), positive to ocean
famm = None   # ice-ocean ammonia/um flux (mmol/m^2/s), positive to ocean
fsil = None   # ice-ocean silicate flux (mmol/m^2/s), positive to ocean
fdmsp = None  # ice-ocean dmsp (mmol/m^2/s), positive to ocean
fdms = None   # ice-ocean dms (mmol/m^2/s), positive to ocean
fhum = None   # ice-ocean humic material carbon (mmol/m^2/s), positive to ocean
fdust = None  # ice-ocean dust flux (kg/m^2/s), positive to ocean

algalN = None   # ocean algal nitrogen (mmol/m^3) (diatoms, pico, phaeo)
falgalN = None  # ice-ocean algal nitrogen flux (mmol/m^2/s) (diatoms, pico, phaeo)

doc = None   # ocean doc (mmol/m^3)  (saccharids, lipids, tbd )
fdoc = None  # ice-ocean doc flux (mmol/m^2/s)  (saccharids, lipids, tbd)

don = None   # ocean don (mmol/m^3) (proteins and amino acids)
fdon = None  # ice-ocean don flux (mmol/m^2/s) (proteins and amino acids)

dic = None   # ocean dic (mmol/m^3)
fdic = None  # ice-ocean dic flux (mmol/m^2/s)

fed = None   # ocean dissolved fe (nM)
fep = None   # ocean particulate fe (nM)
ffed = None  # ice-ocean dissolved fe flux (umol/m^2/s)
ffep = None  # ice-ocean particulate fe flux (umol/m^2/s)

zaeros = None  # ocean aerosols (mmol/m^3)

# isotopes
# coupling variable for tr_iso
fiso_evap = None  # isotope evaporation rate (kg/m^2 s)
Qa_iso = None     # isotope specific humidity (kg/kg)
Qref_iso = None   # 2m atm reference isotope spec humidity (kg/kg)

# coupling variable for tr_iso
HDO_ocn = None     # seawater concentration of HDO (kg/kg)
H2_16O_ocn = None  # seawater concentration of H2_16O (kg/kg)
H2_18O_ocn = None  # seawater concentration of H2_18O (kg/kg)


def alloc_flux_bgc():
    """Allocate space for all variables."""
    global cpl_bgc
    global nit, amm, sil, dmsp, dms, hum
    global fnit, famm, fsil, fdmsp, fdms, fhum, fdust
    global hin_old, dsnown, HDO_ocn, H2_16O_ocn, H2_18O_ocn
    global Qa_iso, Qref_iso, fiso_atm, fiso_evap, fiso_ocn
    global faero_atm, faero_ocn, zaeros
    global flux_bio_atm, flux_bio, flux_bio_ai
    global algalN, falgalN, doc, fdoc, don, fdon, dic, fdic
    global fed, fep, ffed, ffep

    ncat = int(ice_domain_size.ncat)
    if ncat < 1:
        raise ValueError("(alloc_flux_bgc): ncat must be set before allocation")
    horiz = ice_domain_size.horizontal_shape()
    # icepack_max_iso. The tracers module parameter is 3.
    max_iso = 3
    max_aero = int(ice_arrays_column.max_aero)
    max_nbtrcr = int(ice_arrays_column.max_nbtrcr)
    max_algae = int(ice_arrays_column.max_algae)
    max_doc = int(ice_arrays_column.max_doc)
    max_don = int(ice_arrays_column.max_don)
    max_dic = int(ice_arrays_column.max_dic)
    max_fe = int(ice_arrays_column.max_fe)

    def _zero(shape):
        return np.zeros(shape, dtype=dbl_kind)

    nit = _zero(horiz)          # ocean nitrate (mmol/m^3)
    amm = _zero(horiz)          # ammonia/um (mmol/m^3)
    sil = _zero(horiz)          # silicate (mmol/m^3)
    dmsp = _zero(horiz)         # dmsp (mmol/m^3)
    dms = _zero(horiz)          # dms (mmol/m^3)
    hum = _zero(horiz)          # humic material carbon (mmol/m^3)
    fnit = _zero(horiz)         # ice-ocean nitrate flux (mmol/m^2/s), positive to ocean
    famm = _zero(horiz)         # ice-ocean ammonia/um flux (mmol/m^2/s), positive to ocean
    fsil = _zero(horiz)         # ice-ocean silicate flux (mmol/m^2/s), positive to ocean
    fdmsp = _zero(horiz)        # ice-ocean dmsp (mmol/m^2/s), positive to ocean
    fdms = _zero(horiz)         # ice-ocean dms (mmol/m^2/s), positive to ocean
    fhum = _zero(horiz)         # ice-ocean humic material carbon (mmol/m^2/s), positive to ocean
    fdust = _zero(horiz)        # ice-ocean dust flux (kg/m^2/s), positive to ocean
    hin_old = _zero(horiz + (ncat,))   # old ice thickness
    dsnown = _zero(horiz + (ncat,))    # change in snow thickness in category n (m)
    HDO_ocn = _zero(horiz)      # seawater concentration of HDO (kg/kg)
    H2_16O_ocn = _zero(horiz)   # seawater concentration of H2_16O (kg/kg)
    H2_18O_ocn = _zero(horiz)   # seawater concentration of H2_18O (kg/kg)
    Qa_iso = _zero(horiz + (max_iso,))      # isotope specific humidity (kg/kg)
    Qref_iso = _zero(horiz + (max_iso,))    # 2m atm reference isotope spec humidity (kg/kg)
    fiso_atm = _zero(horiz + (max_iso,))    # isotope deposition rate (kg/m^2 s)
    fiso_evap = _zero(horiz + (max_iso,))   # isotope evaporation rate (kg/m^2 s)
    fiso_ocn = _zero(horiz + (max_iso,))    # isotope flux to ocean  (kg/m^2/s)
    faero_atm = _zero(horiz + (max_aero,))  # aerosol deposition rate (kg/m^2 s)
    faero_ocn = _zero(horiz + (max_aero,))  # aerosol flux to ocean  (kg/m^2/s)
    zaeros = _zero(horiz + (max_aero,))     # ocean aerosols (mmol/m^3)
    flux_bio_atm = _zero(horiz + (max_nbtrcr,))  # all bio fluxes to ice from atmosphere
    flux_bio = _zero(horiz + (max_nbtrcr,))      # all bio fluxes to ocean
    flux_bio_ai = _zero(horiz + (max_nbtrcr,))   # all bio fluxes to ocean, averaged over grid cell
    algalN = _zero(horiz + (max_algae,))    # ocean algal nitrogen (mmol/m^3) (diatoms, pico, phaeo)
    falgalN = _zero(horiz + (max_algae,))   # ice-ocn algalN flux (mmol/m^2/s) (diatoms, pico, phaeo)
    doc = _zero(horiz + (max_doc,))         # ocean doc (mmol/m^3)  (saccharids, lipids, tbd )
    fdoc = _zero(horiz + (max_doc,))        # ice-ocean doc flux (mmol/m^2/s)  (saccharids, lipids, tbd)
    don = _zero(horiz + (max_don,))         # ocean don (mmol/m^3) (proteins and amino acids)
    fdon = _zero(horiz + (max_don,))        # ice-ocean don flux (mmol/m^2/s) (proteins and amino acids)
    dic = _zero(horiz + (max_dic,))         # ocean dic (mmol/m^3)
    fdic = _zero(horiz + (max_dic,))        # ice-ocean dic flux (mmol/m^2/s)
    fed = _zero(horiz + (max_fe,))          # ocean dissolved fe (nM)
    fep = _zero(horiz + (max_fe,))          # ocean particulate fe (nM)
    ffed = _zero(horiz + (max_fe,))         # ice-ocean dissolved fe flux (umol/m^2/s)
    ffep = _zero(horiz + (max_fe,))         # ice-ocean particulate fe flux (umol/m^2/s)

    # The namelist value is stored by input_data on ice_init.
    cpl_bgc = log_kind(ice_init.cpl_bgc)

    nit[...] = c0
    amm[...] = c0
    sil[...] = c0
    dmsp[...] = c0
    dms[...] = c0
    hum[...] = c0
    fnit[...] = c0
    famm[...] = c0
    fsil[...] = c0
    fdmsp[...] = c0
    fdms[...] = c0
    fhum[...] = c0
    fdust[...] = c0
    hin_old[...] = c0
    dsnown[...] = c0
    HDO_ocn[...] = c0
    H2_16O_ocn[...] = c0
    H2_18O_ocn[...] = c0
    Qa_iso[...] = c0
    Qref_iso[...] = c0
    fiso_atm[...] = c0
    fiso_evap[...] = c0
    fiso_ocn[...] = c0
    faero_atm[...] = c0
    faero_ocn[...] = c0
    zaeros[...] = c0
    flux_bio_atm[...] = c0
    flux_bio[...] = c0
    flux_bio_ai[...] = c0
    algalN[...] = c0
    falgalN[...] = c0
    doc[...] = c0
    fdoc[...] = c0
    don[...] = c0
    fdon[...] = c0
    dic[...] = c0
    fdic[...] = c0
    fed[...] = c0
    fep[...] = c0
    ffed[...] = c0
    ffep[...] = c0
