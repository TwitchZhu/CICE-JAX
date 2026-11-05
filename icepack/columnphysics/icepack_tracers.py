"""Indices and flags associated with the tracer infrastructure.
Grid-dependent and max_trcr-dependent arrays are declared in ice_state.F90.

author for JAX version: Xinyuan Zhu
XZ: converted from F90 to Python

author: Elizabeth C. Hunke, LANL

Index assignment follows count_tracers in cicecore/shared/ice_init_column.F90.

Tracer indices are 0-based, so ``trcr[..., nt]`` is that tracer. Fortran
tracer 1 is index 0. Layer tracers (``nt_qice``, ``nt_qsno``, ``nt_sice``)
store the first layer; the next layers follow immediately. A tracer that
is switched off is -1 while counting, then pointed at the unused last slot.

gx3 with ``nilyr=7``, ``nslyr=1`` and the default tracers ends at:

    nt_Tsfc=0, nt_qice=1, nt_qsno=8, nt_sice=9,
    nt_iage=16, nt_FY=17, nt_alvl=18, nt_vlvl=19,
    nt_apnd=20, nt_hpnd=21, nt_ipnd=22,
    ntrcr_o=23, ntrcr=24

``ntrcr`` is the number of slots. The last slot is the unused tracer CICE
appends. ``nlt_*`` stays a 1-based index into the biogeochemistry type
list, and 0 means that type is off.
"""

import numpy as np

from icepack.columnphysics import icepack_parameters
from icepack.columnphysics.icepack_kinds import dbl_kind, int_kind, log_kind
from icepack.columnphysics.icepack_parameters import (
    c0,
    c1,
    puny,
    rhos,
    rhosnew,
    rsnw_fall,
    snwgrain,
    snwredist,
)

#-----------------------------------------------------------------
# dimensions
#-----------------------------------------------------------------

ntrcr = int_kind(0)    # number of tracers in use
ntrcr_o = int_kind(0)  # number of non-bio tracers in use
ncat = int_kind(0)     # number of ice categories in use
nilyr = int_kind(0)    # number of ice layers per category
nslyr = int_kind(0)    # number of snow layers per category
nblyr = int_kind(0)    # number of bio/brine layers per category
nfsd = int_kind(0)     # number of fsd layers
n_iso = int_kind(0)    # number of isotopes in use
n_aero = int_kind(0)   # number of aerosols in use
n_zaero = int_kind(0)  # number of z aerosols in use
n_algae = int_kind(0)  # number of algae in use
n_doc = int_kind(0)    # number of DOC pools in use
n_dic = int_kind(0)    # number of DIC pools in use
n_don = int_kind(0)    # number of DON pools in use
n_fed = int_kind(0)    # number of Fe pools in use dissolved Fe
n_fep = int_kind(0)    # number of Fe pools in use particulate Fe

nt_Tsfc = int_kind(-1)    # ice/snow temperature
nt_qice = int_kind(-1)    # volume-weighted ice enthalpy (in layers)
nt_qsno = int_kind(-1)    # volume-weighted snow enthalpy (in layers)
nt_sice = int_kind(-1)    # volume-weighted ice bulk salinity (CICE grid layers)
nt_fbri = int_kind(-1)    # volume fraction of ice with dynamic salt (hinS/vicen*aicen)
nt_iage = int_kind(-1)    # volume-weighted ice age
nt_FY = int_kind(-1)      # area-weighted first-year ice area
nt_alvl = int_kind(-1)    # level ice area fraction
nt_vlvl = int_kind(-1)    # level ice volume fraction
nt_apnd = int_kind(-1)    # melt pond area fraction
nt_hpnd = int_kind(-1)    # melt pond depth
nt_ipnd = int_kind(-1)    # melt pond refrozen lid thickness
nt_fsd = int_kind(-1)     # floe size distribution
nt_smice = int_kind(-1)   # mass of ice in snow
nt_smliq = int_kind(-1)   # mass of liquid water in snow
nt_rhos = int_kind(-1)    # snow density
nt_rsnw = int_kind(-1)    # snow grain radius
nt_isosno = int_kind(-1)  # starting index for isotopes in snow
nt_isoice = int_kind(-1)  # starting index for isotopes in ice
nt_aero = int_kind(-1)    # starting index for aerosols in ice

tr_iage = log_kind(False)        # if true, use age tracer
tr_FY = log_kind(False)          # if true, use first-year area tracer
tr_lvl = log_kind(False)         # if true, use level ice tracer
tr_pond = log_kind(False)        # if true, use melt pond tracer
tr_pond_lvl = log_kind(False)    # if true, use level-ice pond tracer
tr_pond_topo = log_kind(False)   # if true, use explicit topography-based ponds
tr_pond_sealvl = log_kind(False) # if true, use sealvl pond parameterization
tr_snow = log_kind(False)        # if true, use snow redistribution or metamorphosis tracers
tr_iso = log_kind(False)         # if true, use isotope tracers
tr_aero = log_kind(False)        # if true, use aerosol tracers
tr_brine = log_kind(False)       # if true, brine height differs from ice thickness
tr_fsd = log_kind(False)         # if true, use floe size distribution

#-----------------------------------------------------------------
# biogeochemistry
#-----------------------------------------------------------------

max_algae = 3
max_dic = 1
max_doc = 3
max_don = 1
max_fe = 2
max_iso = 3
max_aero = 6
nmodal1 = 10  # dimension for modal aerosol radiation parameters
nmodal2 = 8   # dimension for modal aerosol radiation parameters
max_nbtrcr = (
    max_algae * 2 + max_dic + max_doc + max_don + 5 + 3 + max_fe * 2 + max_aero
)

tr_zaero = log_kind(False)     # if true, black carbon as tracers (n_zaero)
tr_bgc_Nit = log_kind(False)   # if true Nitrate tracer in ice
tr_bgc_N = log_kind(False)     # if true, algal nitrogen tracers (n_algae)
tr_bgc_DON = log_kind(False)   # if true, DON pools are tracers (n_don)
tr_bgc_C = log_kind(False)     # if true, algal carbon tracers + DOC and DIC
tr_bgc_chl = log_kind(False)   # if true, algal chlorophyll tracers
tr_bgc_Am = log_kind(False)    # if true, ammonia/um as nutrient tracer
tr_bgc_Sil = log_kind(False)   # if true, silicon as nutrient tracer
tr_bgc_DMS = log_kind(False)   # if true, DMS as tracer
tr_bgc_Fe = log_kind(False)    # if true, Fe as tracer
tr_bgc_PON = log_kind(False)   # if true, PON as tracer
tr_bgc_hum = log_kind(False)   # if true, humic material as tracer

nbtrcr = int_kind(0)     # number of bgc tracers in use
nbtrcr_sw = int_kind(0)  # number of bgc tracers which impact shortwave
nlt_chl_sw = int_kind(0) # points to total chla in trcrn_sw

nt_bgc_Nit = int_kind(-1)
nt_bgc_Am = int_kind(-1)
nt_bgc_Sil = int_kind(-1)
nt_bgc_DMSPp = int_kind(-1)
nt_bgc_DMSPd = int_kind(-1)
nt_bgc_DMS = int_kind(-1)
nt_bgc_PON = int_kind(-1)
nt_bgc_hum = int_kind(-1)
nt_zbgc_frac = int_kind(-1)

nlt_bgc_Nit = int_kind(0)
nlt_bgc_Am = int_kind(0)
nlt_bgc_Sil = int_kind(0)
nlt_bgc_DMSPp = int_kind(0)
nlt_bgc_DMSPd = int_kind(0)
nlt_bgc_DMS = int_kind(0)
nlt_bgc_PON = int_kind(0)
nlt_bgc_hum = int_kind(0)

nlt_zaero_sw = np.zeros(max_aero, dtype=int_kind)  # points to aerosol in trcrn_sw
nlt_bgc_N = np.zeros(max_algae, dtype=int_kind)
nlt_bgc_chl = np.zeros(max_algae, dtype=int_kind)
nlt_bgc_DOC = np.zeros(max_doc, dtype=int_kind)
nlt_bgc_DON = np.zeros(max_don, dtype=int_kind)
nlt_bgc_DIC = np.zeros(max_dic, dtype=int_kind)
nlt_bgc_Fed = np.zeros(max_fe, dtype=int_kind)
nlt_bgc_Fep = np.zeros(max_fe, dtype=int_kind)
nlt_zaero = np.zeros(max_aero, dtype=int_kind)
nt_bgc_N = np.full(max_algae, -1, dtype=int_kind)
nt_bgc_chl = np.full(max_algae, -1, dtype=int_kind)
nt_bgc_DOC = np.full(max_doc, -1, dtype=int_kind)
nt_bgc_DON = np.full(max_don, -1, dtype=int_kind)
nt_bgc_DIC = np.full(max_dic, -1, dtype=int_kind)
nt_bgc_Fed = np.full(max_fe, -1, dtype=int_kind)
nt_bgc_Fep = np.full(max_fe, -1, dtype=int_kind)
nt_zaero = np.full(max_aero, -1, dtype=int_kind)
bio_index_o = np.zeros(max_nbtrcr, dtype=int_kind)  # nlt_bgc_NO to ocean concentration
bio_index = np.zeros(max_nbtrcr, dtype=int_kind)    # nlt_bgc_N to nt_bgc_N


def icepack_init_tracer_flags(**kwargs):
    """Set tracer active flags.

    Each argument is optional. ``tr_pond_sealvl`` then hard-wires the pond
    geometry options.
    """
    global tr_iage, tr_FY, tr_lvl, tr_pond, tr_pond_lvl, tr_pond_topo
    global tr_pond_sealvl, tr_snow, tr_fsd, tr_iso, tr_aero, tr_brine

    unknown = [
        name
        for name in kwargs
        if name
        not in (
            "tr_iage",
            "tr_FY",
            "tr_lvl",
            "tr_pond",
            "tr_pond_lvl",
            "tr_pond_topo",
            "tr_pond_sealvl",
            "tr_snow",
            "tr_fsd",
            "tr_iso",
            "tr_aero",
            "tr_brine",
        )
    ]
    if unknown:
        raise KeyError(unknown)

    if "tr_iage" in kwargs:
        tr_iage = log_kind(kwargs["tr_iage"])
    if "tr_FY" in kwargs:
        tr_FY = log_kind(kwargs["tr_FY"])
    if "tr_lvl" in kwargs:
        tr_lvl = log_kind(kwargs["tr_lvl"])
    if "tr_pond" in kwargs:
        tr_pond = log_kind(kwargs["tr_pond"])
    if "tr_pond_lvl" in kwargs:
        tr_pond_lvl = log_kind(kwargs["tr_pond_lvl"])
    if "tr_pond_topo" in kwargs:
        tr_pond_topo = log_kind(kwargs["tr_pond_topo"])
    if "tr_pond_sealvl" in kwargs:
        tr_pond_sealvl = log_kind(kwargs["tr_pond_sealvl"])
    if "tr_snow" in kwargs:
        tr_snow = log_kind(kwargs["tr_snow"])
    if "tr_fsd" in kwargs:
        tr_fsd = log_kind(kwargs["tr_fsd"])
    if "tr_iso" in kwargs:
        tr_iso = log_kind(kwargs["tr_iso"])
    if "tr_aero" in kwargs:
        tr_aero = log_kind(kwargs["tr_aero"])
    if "tr_brine" in kwargs:
        tr_brine = log_kind(kwargs["tr_brine"])

    if tr_pond_sealvl:
        icepack_parameters.set_values(
            pndhyps="sealevel",
            pndfrbd="category",
            pndhead="hyps",
            pndmacr="head",
        )
    else:
        icepack_parameters.set_values(
            pndhyps="sealevel",
            pndfrbd="floor",
            pndhead="perched",
            pndmacr="lambda",
        )


def icepack_query_tracer_flags(*names, **kwargs):
    """Query tracer active flags.

    Names may use the Fortran ``_out`` suffix. Returns a dict of the
    requested flags. Each argument is optional; only the names that are
    passed are copied out.
    """
    allowed = (
        "tr_iage",
        "tr_FY",
        "tr_lvl",
        "tr_pond",
        "tr_pond_lvl",
        "tr_pond_topo",
        "tr_pond_sealvl",
        "tr_snow",
        "tr_fsd",
        "tr_iso",
        "tr_aero",
        "tr_brine",
        "tr_zaero",
        "tr_bgc_Nit",
        "tr_bgc_N",
        "tr_bgc_DON",
        "tr_bgc_C",
        "tr_bgc_chl",
        "tr_bgc_Am",
        "tr_bgc_Sil",
        "tr_bgc_DMS",
        "tr_bgc_Fe",
        "tr_bgc_hum",
        "tr_bgc_PON",
    )
    requested = [*(names), *kwargs]
    found = {}
    module = globals()
    for key in requested:
        name = key[:-4] if key.endswith("_out") else key
        if name not in allowed:
            raise KeyError(name)
        # if (present(tr_*_out)) tr_*_out = tr_*
        found[name] = module[name]
    return found


def icepack_init_tracer_sizes(**kwargs):
    """Set the number of column tracers. Each argument is optional."""
    global ncat, nilyr, nslyr, nblyr, nfsd
    global n_algae, n_doc, n_don, n_dic, n_fed, n_fep, n_zaero, n_iso, n_aero
    global ntrcr, ntrcr_o, nbtrcr, nbtrcr_sw

    unknown = [
        name
        for name in kwargs
        if name
        not in (
            "ncat",
            "nilyr",
            "nslyr",
            "nblyr",
            "nfsd",
            "n_algae",
            "n_doc",
            "n_don",
            "n_dic",
            "n_fed",
            "n_fep",
            "n_zaero",
            "n_iso",
            "n_aero",
            "ntrcr",
            "ntrcr_o",
            "nbtrcr",
            "nbtrcr_sw",
        )
    ]
    if unknown:
        raise KeyError(unknown)

    if "ncat" in kwargs:
        ncat = int_kind(kwargs["ncat"])
    if "nilyr" in kwargs:
        nilyr = int_kind(kwargs["nilyr"])
    if "nslyr" in kwargs:
        nslyr = int_kind(kwargs["nslyr"])
    if "nblyr" in kwargs:
        nblyr = int_kind(kwargs["nblyr"])
    if "nfsd" in kwargs:
        nfsd = int_kind(kwargs["nfsd"])
    if "n_algae" in kwargs:
        n_algae = int_kind(kwargs["n_algae"])
    if "n_doc" in kwargs:
        n_doc = int_kind(kwargs["n_doc"])
    if "n_don" in kwargs:
        n_don = int_kind(kwargs["n_don"])
    if "n_dic" in kwargs:
        n_dic = int_kind(kwargs["n_dic"])
    if "n_fed" in kwargs:
        n_fed = int_kind(kwargs["n_fed"])
    if "n_fep" in kwargs:
        n_fep = int_kind(kwargs["n_fep"])
    if "n_zaero" in kwargs:
        n_zaero = int_kind(kwargs["n_zaero"])
    if "n_iso" in kwargs:
        n_iso = int_kind(kwargs["n_iso"])
    if "n_aero" in kwargs:
        n_aero = int_kind(kwargs["n_aero"])
    if "ntrcr" in kwargs:
        ntrcr = int_kind(kwargs["ntrcr"])
    if "ntrcr_o" in kwargs:
        ntrcr_o = int_kind(kwargs["ntrcr_o"])
    if "nbtrcr" in kwargs:
        nbtrcr = int_kind(kwargs["nbtrcr"])
    if "nbtrcr_sw" in kwargs:
        nbtrcr_sw = int_kind(kwargs["nbtrcr_sw"])


def icepack_init_tracer_indices(**kwargs):
    """Set column tracer indices. Each argument is optional."""
    global nt_Tsfc, nt_qice, nt_qsno, nt_sice, nt_fbri
    global nt_iage, nt_FY, nt_alvl, nt_vlvl, nt_apnd, nt_hpnd, nt_ipnd
    global nt_fsd, nt_smice, nt_smliq, nt_rhos, nt_rsnw
    global nt_isosno, nt_isoice, nt_aero

    unknown = [
        name
        for name in kwargs
        if name
        not in (
            "nt_Tsfc",
            "nt_qice",
            "nt_qsno",
            "nt_sice",
            "nt_fbri",
            "nt_iage",
            "nt_FY",
            "nt_alvl",
            "nt_vlvl",
            "nt_apnd",
            "nt_hpnd",
            "nt_ipnd",
            "nt_fsd",
            "nt_smice",
            "nt_smliq",
            "nt_rhos",
            "nt_rsnw",
            "nt_isosno",
            "nt_isoice",
            "nt_aero",
        )
    ]
    if unknown:
        raise KeyError(unknown)

    if "nt_Tsfc" in kwargs:
        nt_Tsfc = int_kind(kwargs["nt_Tsfc"])
    if "nt_qice" in kwargs:
        nt_qice = int_kind(kwargs["nt_qice"])
    if "nt_qsno" in kwargs:
        nt_qsno = int_kind(kwargs["nt_qsno"])
    if "nt_sice" in kwargs:
        nt_sice = int_kind(kwargs["nt_sice"])
    if "nt_fbri" in kwargs:
        nt_fbri = int_kind(kwargs["nt_fbri"])
    if "nt_iage" in kwargs:
        nt_iage = int_kind(kwargs["nt_iage"])
    if "nt_FY" in kwargs:
        nt_FY = int_kind(kwargs["nt_FY"])
    if "nt_alvl" in kwargs:
        nt_alvl = int_kind(kwargs["nt_alvl"])
    if "nt_vlvl" in kwargs:
        nt_vlvl = int_kind(kwargs["nt_vlvl"])
    if "nt_apnd" in kwargs:
        nt_apnd = int_kind(kwargs["nt_apnd"])
    if "nt_hpnd" in kwargs:
        nt_hpnd = int_kind(kwargs["nt_hpnd"])
    if "nt_ipnd" in kwargs:
        nt_ipnd = int_kind(kwargs["nt_ipnd"])
    if "nt_fsd" in kwargs:
        nt_fsd = int_kind(kwargs["nt_fsd"])
    if "nt_smice" in kwargs:
        nt_smice = int_kind(kwargs["nt_smice"])
    if "nt_smliq" in kwargs:
        nt_smliq = int_kind(kwargs["nt_smliq"])
    if "nt_rhos" in kwargs:
        nt_rhos = int_kind(kwargs["nt_rhos"])
    if "nt_rsnw" in kwargs:
        nt_rsnw = int_kind(kwargs["nt_rsnw"])
    if "nt_isosno" in kwargs:
        nt_isosno = int_kind(kwargs["nt_isosno"])
    if "nt_isoice" in kwargs:
        nt_isoice = int_kind(kwargs["nt_isoice"])
    if "nt_aero" in kwargs:
        nt_aero = int_kind(kwargs["nt_aero"])


def count_tracers():
    """Assign tracer indices for the switches already stored on this module.

    When ``tr_aero`` is false, CICE zeros ``n_aero`` on the driver domain
    module and leaves this module's ``n_aero`` at the namelist value.
    ``ice_init.count_tracers`` performs that driver-side assignment.
    A switched-off tracer is -1 until the unused slot is appended, then
    that index is stored in its ``nt_*``. Biogeochemistry tracers that stay
    off remain -1. ``nlt_*`` is unchanged: 0 means off, and a live type is
    1-based. ``ntrcr`` is still the slot count.
    """
    global ntrcr, ntrcr_o, nbtrcr, nbtrcr_sw, nlt_chl_sw, nt_zbgc_frac
    global nt_Tsfc, nt_qice, nt_qsno, nt_sice, nt_fbri
    global nt_iage, nt_FY, nt_alvl, nt_vlvl, nt_apnd, nt_hpnd, nt_ipnd
    global nt_fsd, nt_smice, nt_smliq, nt_rhos, nt_rsnw
    global nt_isosno, nt_isoice, nt_aero
    global nt_bgc_Nit, nt_bgc_Am, nt_bgc_Sil, nt_bgc_DMSPp, nt_bgc_DMSPd
    global nt_bgc_DMS, nt_bgc_PON, nt_bgc_hum
    global nlt_bgc_Nit, nlt_bgc_Am, nlt_bgc_Sil, nlt_bgc_DMSPp, nlt_bgc_DMSPd
    global nlt_bgc_DMS, nlt_bgc_PON, nlt_bgc_hum

    # ntrcr_local is the next free 0-based slot, and the count so far.
    ntrcr_local = 0
    nt_Tsfc_local = ntrcr_local
    ntrcr_local += 1

    nt_qice_local = ntrcr_local
    ntrcr_local += nilyr

    nt_qsno_local = ntrcr_local
    ntrcr_local += nslyr

    nt_sice_local = ntrcr_local
    ntrcr_local += nilyr

    nt_iage_local = -1
    if tr_iage:
        nt_iage_local = ntrcr_local
        ntrcr_local += 1

    nt_FY_local = -1
    if tr_FY:
        nt_FY_local = ntrcr_local
        ntrcr_local += 1

    nt_alvl_local = -1
    nt_vlvl_local = -1
    if tr_lvl:
        nt_alvl_local = ntrcr_local
        ntrcr_local += 1
        nt_vlvl_local = ntrcr_local
        ntrcr_local += 1

    nt_apnd_local = -1
    nt_hpnd_local = -1
    nt_ipnd_local = -1
    if tr_pond:
        nt_apnd_local = ntrcr_local
        ntrcr_local += 1
        nt_hpnd_local = ntrcr_local
        ntrcr_local += 1
        if tr_pond_lvl or tr_pond_sealvl or tr_pond_topo:
            nt_ipnd_local = ntrcr_local
            ntrcr_local += 1

    nt_smice_local = -1
    nt_smliq_local = -1
    nt_rhos_local = -1
    nt_rsnw_local = -1
    if tr_snow:
        nt_smice_local = ntrcr_local
        ntrcr_local += nslyr
        nt_smliq_local = ntrcr_local
        ntrcr_local += nslyr
        nt_rhos_local = ntrcr_local
        ntrcr_local += nslyr
        nt_rsnw_local = ntrcr_local
        ntrcr_local += nslyr

    nt_fsd_local = -1
    if tr_fsd:
        nt_fsd_local = ntrcr_local
        ntrcr_local += nfsd

    nt_isosno_local = -1
    nt_isoice_local = -1
    if tr_iso:
        nt_isosno_local = ntrcr_local
        ntrcr_local += n_iso
        nt_isoice_local = ntrcr_local
        ntrcr_local += n_iso

    nt_aero_local = -1
    if tr_aero:
        nt_aero_local = ntrcr_local
        ntrcr_local += 4 * n_aero

    nbtrcr_local = 0
    nbtrcr_sw_local = 0
    ntrcr_o_local = ntrcr_local
    nt_fbri_local = -1
    if tr_brine:
        nt_fbri_local = ntrcr_local
        ntrcr_local += 1

    #-----------------------------------------------------------------
    # initialize zbgc tracer indices
    #-----------------------------------------------------------------
    nt_zbgc_frac_local = -1
    nlt_chl_sw_local = 0
    nt_bgc_N_local = np.full(max_algae, -1, dtype=int_kind)
    nt_bgc_chl_local = np.full(max_algae, -1, dtype=int_kind)
    nlt_bgc_N_local = np.zeros(max_algae, dtype=int_kind)
    nlt_bgc_chl_local = np.zeros(max_algae, dtype=int_kind)
    nt_bgc_DIC_local = np.full(max_dic, -1, dtype=int_kind)
    nlt_bgc_DIC_local = np.zeros(max_dic, dtype=int_kind)
    nt_bgc_DOC_local = np.full(max_doc, -1, dtype=int_kind)
    nlt_bgc_DOC_local = np.zeros(max_doc, dtype=int_kind)
    nt_bgc_DON_local = np.full(max_don, -1, dtype=int_kind)
    nlt_bgc_DON_local = np.zeros(max_don, dtype=int_kind)
    nt_bgc_Fed_local = np.full(max_fe, -1, dtype=int_kind)
    nt_bgc_Fep_local = np.full(max_fe, -1, dtype=int_kind)
    nlt_bgc_Fed_local = np.zeros(max_fe, dtype=int_kind)
    nlt_bgc_Fep_local = np.zeros(max_fe, dtype=int_kind)
    nt_zaero_local = np.full(max_aero, -1, dtype=int_kind)
    nlt_zaero_local = np.zeros(max_aero, dtype=int_kind)
    nlt_zaero_sw_local = np.zeros(max_aero, dtype=int_kind)
    nt_bgc_Nit_local = -1
    nt_bgc_Am_local = -1
    nt_bgc_Sil_local = -1
    nt_bgc_DMSPp_local = -1
    nt_bgc_DMSPd_local = -1
    nt_bgc_DMS_local = -1
    nt_bgc_PON_local = -1
    nt_bgc_hum_local = -1
    nlt_bgc_Nit_local = 0
    nlt_bgc_Am_local = 0
    nlt_bgc_Sil_local = 0
    nlt_bgc_DMSPp_local = 0
    nlt_bgc_DMSPd_local = 0
    nlt_bgc_DMS_local = 0
    nlt_bgc_PON_local = 0
    nlt_bgc_hum_local = 0
    # nlt_bgc_C = 0
    # nt_bgc_C = 0

    def _add_bgc(nk_bgc):
        nonlocal ntrcr_local, nbtrcr_local
        start = ntrcr_local
        ntrcr_local += nk_bgc
        nbtrcr_local += 1
        return start, nbtrcr_local

    if icepack_parameters.skl_bgc or icepack_parameters.z_tracers:
        if icepack_parameters.skl_bgc:
            nk = 1
        elif icepack_parameters.z_tracers:  # defined on nblyr+1 in ice
            # and 2 snow layers (snow surface + interior)
            nk = int(nblyr) + 1
        nk_bgc = nk                 # number of bgc layers in ice
        if nk > 1:
            nk_bgc = nk + 2         # number of bgc layers in ice and snow

        #-----------------------------------------------------------------
        # count tracers and assign tracer indices
        #-----------------------------------------------------------------
        if tr_bgc_N:
            for mm in range(int(n_algae)):
                nt_bgc_N_local[mm], nlt_bgc_N_local[mm] = _add_bgc(nk_bgc)
        if tr_bgc_Nit:
            nt_bgc_Nit_local, nlt_bgc_Nit_local = _add_bgc(nk_bgc)
        if tr_bgc_C:
            # Algal C is not yet distinct from algal N
            # * Reqires exudation and/or changing C:N ratios
            # for implementation
            # do mm = 1,n_algae
            #    nt_bgc_C(mm) = ntrcr + 1
            for mm in range(int(n_doc)):
                nt_bgc_DOC_local[mm], nlt_bgc_DOC_local[mm] = _add_bgc(nk_bgc)
            for mm in range(int(n_dic)):
                nt_bgc_DIC_local[mm], nlt_bgc_DIC_local[mm] = _add_bgc(nk_bgc)
        if tr_bgc_chl:
            for mm in range(int(n_algae)):
                nt_bgc_chl_local[mm], nlt_bgc_chl_local[mm] = _add_bgc(nk_bgc)
        if tr_bgc_Am:
            nt_bgc_Am_local, nlt_bgc_Am_local = _add_bgc(nk_bgc)
        if tr_bgc_Sil:
            nt_bgc_Sil_local, nlt_bgc_Sil_local = _add_bgc(nk_bgc)
        if tr_bgc_DMS:  # all together
            nt_bgc_DMSPp_local, nlt_bgc_DMSPp_local = _add_bgc(nk_bgc)
            nt_bgc_DMSPd_local, nlt_bgc_DMSPd_local = _add_bgc(nk_bgc)
            nt_bgc_DMS_local, nlt_bgc_DMS_local = _add_bgc(nk_bgc)
        if tr_bgc_PON:
            nt_bgc_PON_local, nlt_bgc_PON_local = _add_bgc(nk_bgc)
        if tr_bgc_DON:
            for mm in range(int(n_don)):
                nt_bgc_DON_local[mm], nlt_bgc_DON_local[mm] = _add_bgc(nk_bgc)
        if tr_bgc_Fe:
            for mm in range(int(n_fed)):
                nt_bgc_Fed_local[mm], nlt_bgc_Fed_local[mm] = _add_bgc(nk_bgc)
            for mm in range(int(n_fep)):
                nt_bgc_Fep_local[mm], nlt_bgc_Fep_local[mm] = _add_bgc(nk_bgc)
        if tr_bgc_hum:
            nt_bgc_hum_local, nlt_bgc_hum_local = _add_bgc(nk_bgc)

    if icepack_parameters.z_tracers:  # defined on nblyr+1 in ice
        # and 2 snow layers (snow surface + interior)
        # z layer aerosols
        if tr_zaero:
            for mm in range(int(n_zaero)):
                nt_zaero_local[mm], nlt_zaero_local[mm] = _add_bgc(nk_bgc)
        if nbtrcr_local > 0:
            nt_zbgc_frac_local = ntrcr_local
            ntrcr_local += nbtrcr_local

    # Unused tracer. Inactive indices are redirected here.
    # tcx, +1 here is the unused tracer, want to get rid of it
    # -1 means the tracer was not opened. 0 is surface temperature.
    unused = ntrcr_local
    ntrcr_local += 1
    if nt_iage_local < 0:
        nt_iage_local = unused
    if nt_FY_local < 0:
        nt_FY_local = unused
    if nt_alvl_local < 0:
        nt_alvl_local = unused
    if nt_vlvl_local < 0:
        nt_vlvl_local = unused
    if nt_apnd_local < 0:
        nt_apnd_local = unused
    if nt_hpnd_local < 0:
        nt_hpnd_local = unused
    if nt_ipnd_local < 0:
        nt_ipnd_local = unused
    if nt_smice_local < 0:
        nt_smice_local = unused
    if nt_smliq_local < 0:
        nt_smliq_local = unused
    if nt_rhos_local < 0:
        nt_rhos_local = unused
    if nt_rsnw_local < 0:
        nt_rsnw_local = unused
    if nt_fsd_local < 0:
        nt_fsd_local = unused
    if nt_isosno_local < 0:
        nt_isosno_local = unused
    if nt_isoice_local < 0:
        nt_isoice_local = unused
    if nt_aero_local < 0:
        nt_aero_local = unused
    if nt_fbri_local < 0:
        nt_fbri_local = unused
    # if (nt_bgc_S <= 0) nt_bgc_S = ntrcr

    ntrcr = int_kind(ntrcr_local)
    ntrcr_o = int_kind(ntrcr_o_local)
    nbtrcr = int_kind(nbtrcr_local)
    nbtrcr_sw = int_kind(nbtrcr_sw_local)
    nt_Tsfc = int_kind(nt_Tsfc_local)
    nt_qice = int_kind(nt_qice_local)
    nt_qsno = int_kind(nt_qsno_local)
    nt_sice = int_kind(nt_sice_local)
    nt_iage = int_kind(nt_iage_local)
    nt_FY = int_kind(nt_FY_local)
    nt_alvl = int_kind(nt_alvl_local)
    nt_vlvl = int_kind(nt_vlvl_local)
    nt_apnd = int_kind(nt_apnd_local)
    nt_hpnd = int_kind(nt_hpnd_local)
    nt_ipnd = int_kind(nt_ipnd_local)
    nt_smice = int_kind(nt_smice_local)
    nt_smliq = int_kind(nt_smliq_local)
    nt_rhos = int_kind(nt_rhos_local)
    nt_rsnw = int_kind(nt_rsnw_local)
    nt_fsd = int_kind(nt_fsd_local)
    nt_isosno = int_kind(nt_isosno_local)
    nt_isoice = int_kind(nt_isoice_local)
    nt_aero = int_kind(nt_aero_local)
    nt_fbri = int_kind(nt_fbri_local)
    nt_bgc_Nit = int_kind(nt_bgc_Nit_local)
    nt_bgc_Am = int_kind(nt_bgc_Am_local)
    nt_bgc_Sil = int_kind(nt_bgc_Sil_local)
    nt_bgc_DMSPp = int_kind(nt_bgc_DMSPp_local)
    nt_bgc_DMSPd = int_kind(nt_bgc_DMSPd_local)
    nt_bgc_DMS = int_kind(nt_bgc_DMS_local)
    nt_bgc_PON = int_kind(nt_bgc_PON_local)
    nt_bgc_hum = int_kind(nt_bgc_hum_local)
    nt_zbgc_frac = int_kind(nt_zbgc_frac_local)
    nlt_bgc_Nit = int_kind(nlt_bgc_Nit_local)
    nlt_bgc_Am = int_kind(nlt_bgc_Am_local)
    nlt_bgc_Sil = int_kind(nlt_bgc_Sil_local)
    nlt_bgc_DMSPp = int_kind(nlt_bgc_DMSPp_local)
    nlt_bgc_DMSPd = int_kind(nlt_bgc_DMSPd_local)
    nlt_bgc_DMS = int_kind(nlt_bgc_DMS_local)
    nlt_bgc_PON = int_kind(nlt_bgc_PON_local)
    nlt_bgc_hum = int_kind(nlt_bgc_hum_local)
    nlt_chl_sw = int_kind(nlt_chl_sw_local)
    nt_bgc_N[...] = nt_bgc_N_local
    nt_bgc_chl[...] = nt_bgc_chl_local
    nt_bgc_DOC[...] = nt_bgc_DOC_local
    nt_bgc_DON[...] = nt_bgc_DON_local
    nt_bgc_DIC[...] = nt_bgc_DIC_local
    nt_bgc_Fed[...] = nt_bgc_Fed_local
    nt_bgc_Fep[...] = nt_bgc_Fep_local
    nt_zaero[...] = nt_zaero_local
    nlt_bgc_N[...] = nlt_bgc_N_local
    nlt_bgc_chl[...] = nlt_bgc_chl_local
    nlt_bgc_DOC[...] = nlt_bgc_DOC_local
    nlt_bgc_DON[...] = nlt_bgc_DON_local
    nlt_bgc_DIC[...] = nlt_bgc_DIC_local
    nlt_bgc_Fed[...] = nlt_bgc_Fed_local
    nlt_bgc_Fep[...] = nlt_bgc_Fep_local
    nlt_zaero[...] = nlt_zaero_local
    nlt_zaero_sw[...] = nlt_zaero_sw_local


def _flag(value):
    return "T" if value else "F"


def _slot(value):
    """Print a 0-based tracer slot the way Fortran prints it.

    Off stays -1 in this module and is written as 0. A live slot is one
    larger, so surface temperature prints as 1.
    """
    value = int(value)
    if value < 0:
        return 0
    return value + 1


def icepack_write_tracer_flags(iounit):
    """Write the column tracer switches.

    author Elizabeth C. Hunke, LANL
    """
    iounit.write("(icepack_write_tracer_flags):\n")
    iounit.write(f"  tr_iage = {_flag(tr_iage)}\n")
    iounit.write(f"  tr_FY   = {_flag(tr_FY)}\n")
    iounit.write(f"  tr_lvl  = {_flag(tr_lvl)}\n")
    iounit.write(f"  tr_pond = {_flag(tr_pond)}\n")
    iounit.write(f"  tr_pond_lvl  = {_flag(tr_pond_lvl)}\n")
    iounit.write(f"  tr_pond_topo = {_flag(tr_pond_topo)}\n")
    iounit.write(f"  tr_pond_sealvl = {_flag(tr_pond_sealvl)}\n")
    iounit.write(f"  tr_snow    = {_flag(tr_snow)}\n")
    iounit.write(f"  tr_fsd     = {_flag(tr_fsd)}\n")
    iounit.write(f"  tr_iso     = {_flag(tr_iso)}\n")
    iounit.write(f"  tr_aero    = {_flag(tr_aero)}\n")
    iounit.write(f"  tr_brine   = {_flag(tr_brine)}\n")
    iounit.write(f"  tr_zaero   = {_flag(tr_zaero)}\n")
    iounit.write(f"  tr_bgc_Nit = {_flag(tr_bgc_Nit)}\n")
    iounit.write(f"  tr_bgc_N   = {_flag(tr_bgc_N)}\n")
    iounit.write(f"  tr_bgc_DON = {_flag(tr_bgc_DON)}\n")
    iounit.write(f"  tr_bgc_C   = {_flag(tr_bgc_C)}\n")
    iounit.write(f"  tr_bgc_chl = {_flag(tr_bgc_chl)}\n")
    iounit.write(f"  tr_bgc_Am  = {_flag(tr_bgc_Am)}\n")
    iounit.write(f"  tr_bgc_Sil = {_flag(tr_bgc_Sil)}\n")
    iounit.write(f"  tr_bgc_DMS = {_flag(tr_bgc_DMS)}\n")
    iounit.write(f"  tr_bgc_Fe  = {_flag(tr_bgc_Fe)}\n")
    iounit.write(f"  tr_bgc_hum = {_flag(tr_bgc_hum)}\n")
    iounit.write(f"  tr_bgc_PON = {_flag(tr_bgc_PON)}\n")


def icepack_write_tracer_sizes(iounit):
    """Write the number of column tracers.

    author Elizabeth C. Hunke, LANL
    """
    iounit.write("(icepack_write_tracer_sizes):\n")
    iounit.write("  fixed parameters: \n")
    iounit.write(f"  max_algae_out = {max_algae:12d}\n")
    iounit.write(f"  max_dic_out   = {max_dic:12d}\n")
    iounit.write(f"  max_doc_out   = {max_doc:12d}\n")
    iounit.write(f"  max_don_out   = {max_don:12d}\n")
    iounit.write(f"  max_fe_out    = {max_fe:12d}\n")
    iounit.write(f"  nmodal1_out   = {nmodal1:12d}\n")
    iounit.write(f"  nmodal2_out   = {nmodal2:12d}\n")
    iounit.write(f"  max_iso_out   = {max_iso:12d}\n")
    iounit.write(f"  max_aero_out  = {max_aero:12d}\n")
    iounit.write(f"  max_nbtrcr_out= {max_nbtrcr:12d}\n")
    iounit.write("  model defined parameters: \n")
    iounit.write(f"  ncat      = {int(ncat):12d}\n")
    iounit.write(f"  nilyr     = {int(nilyr):12d}\n")
    iounit.write(f"  nslyr     = {int(nslyr):12d}\n")
    iounit.write(f"  nblyr     = {int(nblyr):12d}\n")
    iounit.write(f"  nfsd      = {int(nfsd):12d}\n")
    iounit.write(f"  n_algae   = {int(n_algae):12d}\n")
    iounit.write(f"  n_DOC     = {int(n_doc):12d}\n")
    iounit.write(f"  n_DON     = {int(n_don):12d}\n")
    iounit.write(f"  n_DIC     = {int(n_dic):12d}\n")
    iounit.write(f"  n_fed     = {int(n_fed):12d}\n")
    iounit.write(f"  n_fep     = {int(n_fep):12d}\n")
    iounit.write(f"  n_zaero   = {int(n_zaero):12d}\n")
    iounit.write(f"  n_aero    = {int(n_aero):12d}\n")
    iounit.write(f"  n_iso     = {int(n_iso):12d}\n")
    iounit.write(f"  ntrcr     = {int(ntrcr):12d}\n")
    iounit.write(f"  ntrcr_o   = {int(ntrcr_o):12d}\n")
    iounit.write(f"  nbtrcr    = {int(nbtrcr):12d}\n")
    iounit.write(f"  nbtrcr_sw = {int(nbtrcr_sw):12d}\n")


def icepack_write_tracer_indices(iounit):
    """Write the column tracer indices.

    author Elizabeth C. Hunke, LANL
    """
    iounit.write("(icepack_write_tracer_indices):\n")
    iounit.write(f"  nt_Tsfc = {_slot(nt_Tsfc):12d}\n")
    iounit.write(f"  nt_qice = {_slot(nt_qice):12d}\n")
    iounit.write(f"  nt_qsno = {_slot(nt_qsno):12d}\n")
    iounit.write(f"  nt_sice = {_slot(nt_sice):12d}\n")
    iounit.write(f"  nt_fbri = {_slot(nt_fbri):12d}\n")
    iounit.write(f"  nt_iage = {_slot(nt_iage):12d}\n")
    iounit.write(f"  nt_FY   = {_slot(nt_FY):12d}\n")
    iounit.write(f"  nt_alvl = {_slot(nt_alvl):12d}\n")
    iounit.write(f"  nt_vlvl = {_slot(nt_vlvl):12d}\n")
    iounit.write(f"  nt_apnd = {_slot(nt_apnd):12d}\n")
    iounit.write(f"  nt_hpnd = {_slot(nt_hpnd):12d}\n")
    iounit.write(f"  nt_ipnd = {_slot(nt_ipnd):12d}\n")
    iounit.write(f"  nt_fsd  = {_slot(nt_fsd):12d}\n")
    iounit.write(f"  nt_smice      = {_slot(nt_smice):12d}\n")
    iounit.write(f"  nt_smliq      = {_slot(nt_smliq):12d}\n")
    iounit.write(f"  nt_rhos       = {_slot(nt_rhos):12d}\n")
    iounit.write(f"  nt_rsnw       = {_slot(nt_rsnw):12d}\n")
    iounit.write(f"  nt_isosno     = {_slot(nt_isosno):12d}\n")
    iounit.write(f"  nt_isoice     = {_slot(nt_isoice):12d}\n")
    iounit.write(f"  nt_aero       = {_slot(nt_aero):12d}\n")
    iounit.write(f"  nt_bgc_Nit    = {_slot(nt_bgc_Nit):12d}\n")
    iounit.write(f"  nt_bgc_Am     = {_slot(nt_bgc_Am):12d}\n")
    iounit.write(f"  nt_bgc_Sil    = {_slot(nt_bgc_Sil):12d}\n")
    iounit.write(f"  nt_bgc_DMSPp  = {_slot(nt_bgc_DMSPp):12d}\n")
    iounit.write(f"  nt_bgc_DMSPd  = {_slot(nt_bgc_DMSPd):12d}\n")
    iounit.write(f"  nt_bgc_DMS    = {_slot(nt_bgc_DMS):12d}\n")
    iounit.write(f"  nt_bgc_hum    = {_slot(nt_bgc_hum):12d}\n")
    iounit.write(f"  nt_bgc_PON    = {_slot(nt_bgc_PON):12d}\n")
    iounit.write(f"  nlt_bgc_Nit   = {int(nlt_bgc_Nit):12d}\n")
    iounit.write(f"  nlt_bgc_Am    = {int(nlt_bgc_Am):12d}\n")
    iounit.write(f"  nlt_bgc_Sil   = {int(nlt_bgc_Sil):12d}\n")
    iounit.write(f"  nlt_bgc_DMSPp = {int(nlt_bgc_DMSPp):12d}\n")
    iounit.write(f"  nlt_bgc_DMSPd = {int(nlt_bgc_DMSPd):12d}\n")
    iounit.write(f"  nlt_bgc_DMS   = {int(nlt_bgc_DMS):12d}\n")
    iounit.write(f"  nlt_bgc_hum   = {int(nlt_bgc_hum):12d}\n")
    iounit.write(f"  nlt_bgc_PON   = {int(nlt_bgc_PON):12d}\n")
    iounit.write(f"  nlt_chl_sw    = {int(nlt_chl_sw):12d}\n")
    iounit.write(f"  nt_zbgc_frac  = {_slot(nt_zbgc_frac):12d}\n")
    iounit.write(f"  max_nbtrcr = {max_nbtrcr:12d}\n")
    for k in range(max_nbtrcr):
        bio = int(bio_index[k])
        bio_o = int(bio_index_o[k])
        iounit.write(f"  bio_index_o(k) = {k + 1:12d} {0 if bio_o <= 0 else bio_o + 1:12d}\n")
        iounit.write(f"  bio_index(k)   = {k + 1:12d} {0 if bio <= 0 else bio + 1:12d}\n")
    iounit.write(f"  max_algae = {max_algae:12d}\n")
    for k in range(max_algae):
        iounit.write(f"  nt_bgc_N(k)  = {k + 1:12d} {_slot(nt_bgc_N[k]):12d}\n")
        iounit.write(f"  nlt_bgc_N(k) = {k + 1:12d} {int(nlt_bgc_N[k]):12d}\n")
        # Algal C is not a separate tracer. Fortran keeps nt_bgc_C at 0.
        iounit.write(f"  nt_bgc_C(k)  = {k + 1:12d} {0:12d}\n")
        iounit.write(f"  nlt_bgc_C(k) = {k + 1:12d} {0:12d}\n")
        iounit.write(f"  nt_bgc_chl(k)  = {k + 1:12d} {_slot(nt_bgc_chl[k]):12d}\n")
        iounit.write(f"  nlt_bgc_chl(k) = {k + 1:12d} {int(nlt_bgc_chl[k]):12d}\n")
    iounit.write(f"  max_DOC = {max_doc:12d}\n")
    for k in range(max_doc):
        iounit.write(f"  nt_bgc_DOC(k)  = {k + 1:12d} {_slot(nt_bgc_DOC[k]):12d}\n")
        iounit.write(f"  nlt_bgc_DOC(k) = {k + 1:12d} {int(nlt_bgc_DOC[k]):12d}\n")
    iounit.write(f"  max_DON = {max_don:12d}\n")
    for k in range(max_don):
        iounit.write(f"  nt_bgc_DON(k)  = {k + 1:12d} {_slot(nt_bgc_DON[k]):12d}\n")
        iounit.write(f"  nlt_bgc_DON(k) = {k + 1:12d} {int(nlt_bgc_DON[k]):12d}\n")
    iounit.write(f"  max_DIC = {max_dic:12d}\n")
    for k in range(max_dic):
        iounit.write(f"  nt_bgc_DIC(k)  = {k + 1:12d} {_slot(nt_bgc_DIC[k]):12d}\n")
        iounit.write(f"  nlt_bgc_DIC(k) = {k + 1:12d} {int(nlt_bgc_DIC[k]):12d}\n")
    iounit.write(f"  max_fe = {max_fe:12d}\n")
    for k in range(max_fe):
        iounit.write(f"  nt_bgc_Fed(k)  = {k + 1:12d} {_slot(nt_bgc_Fed[k]):12d}\n")
        iounit.write(f"  nlt_bgc_Fed(k) = {k + 1:12d} {int(nlt_bgc_Fed[k]):12d}\n")
        iounit.write(f"  nt_bgc_Fep(k)  = {k + 1:12d} {_slot(nt_bgc_Fep[k]):12d}\n")
        iounit.write(f"  nlt_bgc_Fep(k) = {k + 1:12d} {int(nlt_bgc_Fep[k]):12d}\n")
    iounit.write(f"  max_aero = {max_aero:12d}\n")
    for k in range(max_aero):
        iounit.write(f"  nt_zaero(k)     = {k + 1:12d} {_slot(nt_zaero[k]):12d}\n")
        iounit.write(f"  nlt_zaero(k)    = {k + 1:12d} {int(nlt_zaero[k]):12d}\n")
        iounit.write(f"  nlt_zaero_sw(k) = {k + 1:12d} {int(nlt_zaero_sw[k]):12d}\n")


def icepack_compute_tracers(
    trcr_depend,
    atrcrn,
    aicen,
    vicen,
    vsnon,
    trcr_base,
    n_trcr_strata,
    nt_strata,
    Tf,
):
    """Compute aggregate tracers from category sums.

    ``atrcrn`` is aicen*trcrn or vicen*trcrn or vsnon*trcrn, one value
    per tracer. ``nt_strata`` holds 0-based tracer indices.
    Returns a new tracer vector of length ``ntrcr``. The loop order matches
    the Fortran so a dependent tracer is divided by a tracer already stored.
    """
    count = int(ntrcr)
    trcr_out = np.zeros(count, dtype=dbl_kind)
    #-----------------------------------------------------------------
    # Compute new tracers
    #-----------------------------------------------------------------
    for it in range(count):
        divisor = (
            trcr_base[it, 0] * aicen,
            trcr_base[it, 1] * vicen,
            trcr_base[it, 2] * vsnon,
        )
        if int(trcr_depend[it]) == 0:  # ice area tracers
            if aicen > puny:
                trcr_out[it] = atrcrn[it] / aicen
            else:
                trcr_out[it] = c0
                if it == int(nt_Tsfc):
                    trcr_out[it] = Tf  # surface temperature
        else:
            work = c0
            for k in range(3):
                if divisor[k] > c0:
                    work = atrcrn[it] / divisor[k]
            trcr_out[it] = work  # save it
            if int(n_trcr_strata[it]) > 0:  # additional tracer layers
                for itl in range(int(n_trcr_strata[it])):
                    ntr = int(nt_strata[it, itl])
                    if ntr >= 0 and trcr_out[ntr] > c0:
                        trcr_out[it] = trcr_out[it] / trcr_out[ntr]
                    else:
                        trcr_out[it] = c0

    if vicen <= c0 and bool(tr_brine):
        trcr_out[int(nt_fbri)] = c1
    if vsnon <= c0:
        if str(snwredist)[:3] == "ITD":
            for k in range(int(nslyr)):
                trcr_out[int(nt_rhos) + k] = rhosnew
        if bool(snwgrain):
            for k in range(int(nslyr)):
                trcr_out[int(nt_rsnw) + k] = rsnw_fall
                trcr_out[int(nt_smice) + k] = rhos
    return trcr_out
