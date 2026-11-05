"""Grid-dependent arrays needed for column package.
These were originally module variables in modules that became part of
the column package.

author for JAX version: Xinyuan Zhu
XZ: converted from F90 to Python

author: Elizabeth C. Hunke, LANL

Allocation follows ``alloc_arrays_column``: drag, shortwave, biogeochemistry,
and the floe-size distribution are all allocated. Tracer maxima match the
icepack parameters, not the tracers switched on for one grid.
"""

import numpy as np

from cicecore.shared import ice_domain_size
from cicecore.shared.ice_constants import c0
from cicecore.shared.ice_kinds_mod import dbl_kind, log_kind
from icepack.columnphysics import icepack_parameters

# icepack_tracers parameter sizes. max_nbtrcr is the sum written there.
max_algae = 3
max_dic = 1
max_doc = 3
max_don = 1
max_fe = 2
max_aero = 6
max_nbtrcr = (
    max_algae * 2 + max_dic + max_doc + max_don + 5 + 3 + max_fe * 2 + max_aero
)

# icepack_atmo.F90
# Cdn variables on the T-grid
Cdn_atm = None        # atm drag coefficient
Cdn_ocn = None        # ocn drag coefficient
                      # form drag
hfreebd = None        # freeboard (m)
hdraft = None         # draft of ice + snow column (Stoessel1993)
hridge = None         # ridge height
distrdg = None        # distance between ridges
hkeel = None          # keel depth
dkeel = None          # distance between keels
lfloe = None          # floe length
dfloe = None          # distance between floes
Cdn_atm_skin = None   # neutral skin drag coefficient
Cdn_atm_floe = None   # neutral floe edge drag coefficient
Cdn_atm_pond = None   # neutral pond edge drag coefficient
Cdn_atm_rdg = None    # neutral ridge drag coefficient
Cdn_ocn_skin = None   # skin drag coefficient
Cdn_ocn_floe = None   # floe edge drag coefficient
Cdn_ocn_keel = None   # keel drag coefficient
Cdn_atm_ratio = None  # ratio drag atm / neutral drag atm

#-------------------------------------------------------------------
# a note regarding hi_min and hin_max(0):
# both represent a minimum ice thickness.  hin_max(0) is
# intended to be used for particular numerical implementations
# of category conversions in the ice thickness distribution.
# hi_min is a more general purpose parameter, but is specifically
# for maintaining stability in the thermodynamics.
# hin_max(0) = 0.1 m for the delta function itd
# hin_max(0) = 0.0 m for linear remapping
#
# Also note that the upper limit on the thickest category
# is only used for the linear remapping scheme
# and it is not a true upper limit on the thickness
#-------------------------------------------------------------------

# icepack_itd.F90
hin_max = None      # category limits (m)
c_hi_range = None   # string for history output

# icepack_snow.F90
meltsliq = None   # snow melt mass (kg/m^2/step-->kg/m^2/day)
meltsliqn = None  # snow melt mass in category n (kg/m^2)

# icepack_meltpond_lvl.F90
dhsn = None    # depth difference for snow on sea ice and pond ice
ffracn = None  # fraction of fsurfn used to melt ipond

# icepack_shortwave.F90
# category albedos
alvdrn = None  # visible direct albedo           (fraction)
alidrn = None  # near-ir direct albedo           (fraction)
alvdfn = None  # visible diffuse albedo          (fraction)
alidfn = None  # near-ir diffuse albedo          (fraction)

# albedo components for history
albicen = None  # bare ice
albsnon = None  # snow
albpndn = None  # pond
apeffn = None   # effective pond area used for radiation calculation

snowfracn = None  # Category snow fraction used in radiation

# shortwave components
Iswabsn = None  # SW radiation absorbed in ice layers (W m-2)
Sswabsn = None  # SW radiation absorbed in snow layers (W m-2)
fswsfcn = None         # SW absorbed at ice/snow surface (W m-2)
fswthrun = None        # SW through ice to ocean            (W/m^2)
fswthrun_vdr = None    # vis dir SW through ice to ocean            (W/m^2)
fswthrun_vdf = None    # vis dif SW through ice to ocean            (W/m^2)
fswthrun_idr = None    # nir dir SW through ice to ocean            (W/m^2)
fswthrun_idf = None    # nir dif SW through ice to ocean            (W/m^2)
fswthrun_uvrdr = None  # vis uvr dir SW through ice to ocean        (W/m^2)
fswthrun_uvrdf = None  # vis uvr dif SW through ice to ocean        (W/m^2)
fswthrun_pardr = None  # vis par dir SW through ice to ocean        (W/m^2)
fswthrun_pardf = None  # vis par dif SW through ice to ocean        (W/m^2)
fswintn = None         # SW absorbed in ice interior, below surface (W m-2)
fswpenln = None        # visible SW entering ice layers (W m-2)
trcrn_sw = None        # tracers active in the shortwave calculation

# biogeochemistry components
bgrid = None   # biology nondimensional vertical grid points
igrid = None   # biology vertical interface points
cgrid = None   # CICE vertical coordinate
icgrid = None  # interface grid for CICE (shortwave variable)
swgrid = None  # grid for ice tracers used in dEdd scheme

first_ice_real = None  # .true. = c1, .false. = c0
first_ice = None       # distinguishes ice that disappears (e.g. melts)
                       # and reappears (e.g. transport) in a grid cell
                       # during a single time step from ice that was
                       # there the entire time step (true until ice forms)

ocean_bio = None  # contains all the ocean bgc tracer concentrations

# diagnostic fluxes
fbio_snoice = None  # fluxes from snow to ice
fbio_atmice = None  # fluxes from atm to ice

ocean_bio_all = None  # fixed order, all values even for tracers false
                      # N(1:icepack_max_algae) = 1:icepack_max_algae
                      # Nit = icepack_max_algae + 1
                      # DOC(1:icepack_max_doc) = icepack_max_algae + 2: icepack_max_algae + icepack_max_doc + 1
                      # DIC(1:icepack_max_dic) = icepack_max_algae + icepack_max_doc + 2: icepack_max_algae + icepack_max_doc + 1 + icepack_max_dic
                      # chl(1:icepack_max_algae) =  icepack_max_algae + icepack_max_doc + 2 + icepack_max_dic:
                      #                     2*icepack_max_algae + icepack_max_doc + 1 + icepack_max_dic
                      # Am =  2*icepack_max_algae + icepack_max_doc + 2 + icepack_max_dic
                      # Sil=  2*icepack_max_algae + icepack_max_doc + 3 + icepack_max_dic
                      # DMSPp=  2*icepack_max_algae + icepack_max_doc + 4 + icepack_max_dic
                      # DMSPd=  2*icepack_max_algae + icepack_max_doc + 5 + icepack_max_dic
                      # DMS  =  2*icepack_max_algae + icepack_max_doc + 6 + icepack_max_dic
                      # PON  =  2*icepack_max_algae + icepack_max_doc + 7 + icepack_max_dic
                      # DON(1:icepack_max_don)  =  2*icepack_max_algae + icepack_max_doc + 8 + icepack_max_dic:
                      #                    2*icepack_max_algae + icepack_max_doc + 7 + icepack_max_dic + icepack_max_don
                      # Fed(1:icepack_max_fe) = 2*icepack_max_algae + icepack_max_doc + 8 + icepack_max_dic + icepack_max_don:
                      #                2*icepack_max_algae + icepack_max_doc + 7 + icepack_max_dic + icepack_max_don + icepack_max_fe
                      # Fep(1:icepack_max_fe) = 2*icepack_max_algae + icepack_max_doc + 8 + icepack_max_dic + icepack_max_don + icepack_max_fe:
                      #                2*icepack_max_algae + icepack_max_doc + 7 + icepack_max_dic + icepack_max_don + 2*icepack_max_fe
                      # zaero(1:icepack_max_aero) = 2*icepack_max_algae + icepack_max_doc + 8 + icepack_max_dic + icepack_max_don + 2*icepack_max_fe:
                      #                     2*icepack_max_algae + icepack_max_doc + 7 + icepack_max_dic + icepack_max_don + 2*icepack_max_fe
                      #                     + icepack_max_aero
                      # humic ==  2*icepack_max_algae + icepack_max_doc + 8 + icepack_max_dic + icepack_max_don + 2*icepack_max_fe
                      #                     + icepack_max_aero

algal_peak = None  # vertical location of algal maximum, 0 if no maximum

Zoo = None  # N losses accumulated in timestep (ie. zooplankton/bacteria)
            # mmol/m^3

dhbr_top = None  # brine top change
dhbr_bot = None  # brine bottom change

grow_net = None  # Specific growth rate (/s) per grid cell
PP_net = None    # Total production (mg C/m^2/s) per grid cell
hbri = None      # brine height, area-averaged for comparison with hi (m)

bphi = None  # porosity of layers
bTiz = None  # layer temperatures interpolated on bio grid (C)

darcy_V = None  # darcy velocity positive up (m/s)

chl_net = None  # Total chla (mg chla/m^2) per grid cell
NO_net = None   # Total nitrate per grid cell

sice_rho = None  # avg sea ice density  (kg/m^3)  ! ech: diagnostic only?

zfswin = None  # Shortwave flux into layers interpolated on bio grid  (W/m^2)

iDi = None  # igrid Diffusivity (m^2/s)
iki = None  # Ice permeability (m^2)

upNO = None  # nitrate uptake rate (mmol/m^2/d) times aice
upNH = None  # ammonium uptake rate (mmol/m^2/d) times aice

ice_bio_net = None   # depth integrated tracer (mmol/m^2)
snow_bio_net = None  # depth integrated snow tracer (mmol/m^2)

# floe size distribution
floe_rad_l = None     # fsd size lower bound in m (radius)
floe_rad_c = None     # fsd size bin centre in m (radius)
floe_binwidth = None  # fsd size bin width in m (radius)
c_fsd_range = None    # fsd floe_rad bounds (m)
wave_sig_ht = None    # significant height of waves (m)
wavefreq = None       # wave frequencies
dwavefreq = None      # wave frequency bin widths
wave_spectrum = None  # wave spectrum
# change in floe size distribution due to processes
d_afsd_newi = None
d_afsd_latg = None
d_afsd_latm = None
d_afsd_wave = None
d_afsd_weld = None


def _zero(shape):
    return np.zeros(shape, dtype=dbl_kind)


def alloc_arrays_column():
    """Allocate column arrays. Values start at zero."""
    global Cdn_atm, Cdn_ocn, hfreebd, hdraft, hridge, distrdg, hkeel, dkeel
    global lfloe, dfloe
    global Cdn_atm_skin, Cdn_atm_floe, Cdn_atm_pond, Cdn_atm_rdg
    global Cdn_ocn_skin, Cdn_ocn_floe, Cdn_ocn_keel, Cdn_atm_ratio
    global meltsliq, meltsliqn, dhsn, ffracn
    global alvdrn, alidrn, alvdfn, alidfn, albicen, albsnon, albpndn
    global apeffn, snowfracn, fswsfcn, fswthrun
    global fswthrun_vdr, fswthrun_vdf, fswthrun_idr, fswthrun_idf
    global fswthrun_uvrdr, fswthrun_uvrdf, fswthrun_pardr, fswthrun_pardf
    global fswintn, first_ice_real, first_ice
    global dhbr_top, dhbr_bot, darcy_V, sice_rho
    global Iswabsn, Sswabsn, fswpenln
    global hin_max, c_hi_range, cgrid, icgrid, swgrid
    global grow_net, PP_net, hbri, chl_net, NO_net, upNO, upNH
    global Zoo, zfswin, iDi, iki, bphi, bTiz
    global ocean_bio, fbio_snoice, fbio_atmice, ocean_bio_all
    global ice_bio_net, snow_bio_net, algal_peak, bgrid, igrid
    global floe_rad_l, floe_rad_c, floe_binwidth, c_fsd_range
    global wavefreq, dwavefreq, wave_sig_ht, wave_spectrum
    global d_afsd_newi, d_afsd_latg, d_afsd_latm, d_afsd_wave, d_afsd_weld

    ncat = int(ice_domain_size.ncat)
    nilyr = int(ice_domain_size.nilyr)
    nslyr = int(ice_domain_size.nslyr)
    nblyr = int(ice_domain_size.nblyr)
    nfsd = int(ice_domain_size.nfsd)
    nfreq = int(icepack_parameters.nfreq)
    if ncat < 1 or nilyr < 1 or nslyr < 1 or nblyr < 1 or nfsd < 1:
        raise ValueError("(alloc_arrays_column) category counts are not set")

    horiz = ice_domain_size.horizontal_shape()
    cat = horiz + (ncat,)
    ice_cat = horiz + (nilyr, ncat)
    snow_cat = horiz + (nslyr, ncat)
    pen_cat = horiz + (nilyr + 1, ncat)
    bio_cat = horiz + (nblyr + 1, ncat)
    bio_iface = horiz + (nblyr + 2, ncat)

    for name in (
        "Cdn_atm", "Cdn_ocn", "hfreebd", "hdraft", "hridge", "distrdg",
        "hkeel", "dkeel", "lfloe", "dfloe",
        "Cdn_atm_skin", "Cdn_atm_floe", "Cdn_atm_pond", "Cdn_atm_rdg",
        "Cdn_ocn_skin", "Cdn_ocn_floe", "Cdn_ocn_keel", "Cdn_atm_ratio",
        "grow_net", "PP_net", "hbri", "chl_net", "NO_net", "upNO", "upNH",
        "meltsliq",
    ):
        globals()[name] = _zero(horiz)

    for name in (
        "meltsliqn", "dhsn", "ffracn",
        "alvdrn", "alidrn", "alvdfn", "alidfn",
        "albicen", "albsnon", "albpndn", "apeffn", "snowfracn",
        "fswsfcn", "fswthrun",
        "fswthrun_vdr", "fswthrun_vdf", "fswthrun_idr", "fswthrun_idf",
        "fswthrun_uvrdr", "fswthrun_uvrdf", "fswthrun_pardr", "fswthrun_pardf",
        "fswintn", "first_ice_real",
        "dhbr_top", "dhbr_bot", "darcy_V", "sice_rho",
    ):
        globals()[name] = _zero(cat)

    first_ice = np.zeros(cat, dtype=log_kind)
    Iswabsn = _zero(ice_cat)
    Sswabsn = _zero(snow_cat)
    fswpenln = _zero(pen_cat)

    for name in ("Zoo", "zfswin", "iDi", "iki"):
        globals()[name] = _zero(bio_cat)
    for name in ("bphi", "bTiz"):
        globals()[name] = _zero(bio_iface)
    bio = horiz + (max_nbtrcr,)
    for name in (
        "ocean_bio", "fbio_snoice", "fbio_atmice", "ocean_bio_all",
        "ice_bio_net", "snow_bio_net",
    ):
        globals()[name] = _zero(bio)
    algal_peak = _zero(horiz + (max_algae,))

    hin_max = _zero(ncat + 1)
    hin_max[...] = c0
    c_hi_range = [""] * ncat
    bgrid = _zero(nblyr + 2)
    igrid = _zero(nblyr + 1)
    cgrid = _zero(nilyr + 1)
    icgrid = _zero(nilyr + 1)
    swgrid = _zero(nilyr + 1)

    floe_rad_l = _zero(nfsd)
    floe_rad_c = _zero(nfsd)
    floe_binwidth = _zero(nfsd)
    c_fsd_range = [""] * nfsd
    wavefreq = _zero(nfreq)
    dwavefreq = _zero(nfreq)
    wave_sig_ht = _zero(horiz)
    wave_spectrum = _zero(horiz + (nfreq,))
    fsd = horiz + (nfsd,)
    for name in (
        "d_afsd_newi", "d_afsd_latg", "d_afsd_latm", "d_afsd_wave", "d_afsd_weld",
    ):
        globals()[name] = _zero(fsd)
