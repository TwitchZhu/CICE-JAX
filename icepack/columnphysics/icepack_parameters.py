"""Flags and parameters for the column package.

author for JAX version: Xinyuan Zhu
XZ: converted from F90 to Python

authors: Elizabeth C. Hunke, LANL

Scalars use the same width as Fortran dbl_kind. icepack_recompute_constants
fills the derived constants; namelist values are applied later by input_data.
"""

import math

import numpy as np

from icepack.columnphysics.icepack_kinds import dbl_kind, int_kind, log_kind

#-----------------------------------------------------------------
# control options
#-----------------------------------------------------------------

argcheck = "first"  # optional argument checks, 'never', 'first', 'always'

#-----------------------------------------------------------------
# parameter constants
#-----------------------------------------------------------------

c0 = dbl_kind(0.0)
c1 = dbl_kind(1.0)
c1p5 = dbl_kind(1.5)
c2 = dbl_kind(2.0)
c3 = dbl_kind(3.0)
c4 = dbl_kind(4.0)
c5 = dbl_kind(5.0)
c6 = dbl_kind(6.0)
c8 = dbl_kind(8.0)
c10 = dbl_kind(10.0)
c15 = dbl_kind(15.0)
c16 = dbl_kind(16.0)
c20 = dbl_kind(20.0)
c25 = dbl_kind(25.0)
c100 = dbl_kind(100.0)
c180 = dbl_kind(180.0)
c1000 = dbl_kind(1000.0)
p001 = dbl_kind(0.001)
p01 = dbl_kind(0.01)
p1 = dbl_kind(0.1)
p2 = dbl_kind(0.2)
p4 = dbl_kind(0.4)
p5 = dbl_kind(0.5)
p6 = dbl_kind(0.6)
p05 = dbl_kind(0.05)
p15 = dbl_kind(0.15)
p25 = dbl_kind(0.25)
p75 = dbl_kind(0.75)
p333 = c1 / c3
p666 = c2 / c3
spval_const = dbl_kind(-1.0e36)

secday = dbl_kind(86400.0)  # seconds in calendar day
puny = dbl_kind(1.0e-11)
bignum = dbl_kind(1.0e30)
pi = dbl_kind(3.14159265358979323846)

#-----------------------------------------------------------------
# derived physical constants
#    Lfresh = Lsub - Lvap
#    cprho  = cp_ocn * rhow
#    Cp     = 0.5 * gravit * (rhow - rhoi) * rhoi / rhow
#-----------------------------------------------------------------

pih = spval_const         # 0.5 * pi
piq = spval_const         # 0.25 * pi
pi2 = spval_const         # 2 * pi
rad_to_deg = spval_const  # conversion factor, radians to degrees
Lfresh = spval_const      # latent heat of melting of fresh ice (J/kg)
cprho = spval_const       # for ocean mixed layer (J kg / K m^3)
Cp = spval_const          # proport const for PE

#-----------------------------------------------------------------
# Densities
#-----------------------------------------------------------------

rhos = dbl_kind(330.0)      # density of snow (kg/m^3)
rhoi = dbl_kind(917.0)      # density of ice (kg/m^3)
rhosi = dbl_kind(940.0)     # average sea ice density
                            # Cox and Weeks, 1982: 919-974 kg/m^2
rhow = dbl_kind(1026.0)     # density of seawater (kg/m^3)
rhofresh = dbl_kind(1000.0) # density of fresh water (kg/m^3)

#-----------------------------------------------------------------------
# Parameters for thermodynamics
#-----------------------------------------------------------------------

hfrazilmin = dbl_kind(0.05)          # min thickness of new frazil ice (m)
cp_ice = dbl_kind(2106.0)            # specific heat of fresh ice (J/kg/K)
cp_ocn = dbl_kind(4218.0)            # specific heat of ocn (J/kg/K)
                                     # freshwater value needed for enthalpy
depressT = dbl_kind(0.054)           # Tf:brine salinity ratio (C/ppt)
viscosity_dyn = dbl_kind(1.79e-3)    # dynamic viscosity of brine (kg/m/s)
tscale_pnd_drain = c10               # mushy macroscopic drainage timescale (days)
Tocnfrz = dbl_kind(-1.8)             # freezing temp of seawater (C),
                                     # used as Tsfcn for open water
Tffresh = dbl_kind(273.15)           # freezing temp of fresh ice (K)
Lsub = dbl_kind(2.835e6)             # latent heat, sublimation freshwater (J/kg)
Lvap = dbl_kind(2.501e6)             # latent heat, vaporization freshwater (J/kg)
Timelt = dbl_kind(0.0)               # melting temperature, ice top surface (C)
Tsmelt = dbl_kind(0.0)               # melting temperature, snow top surface (C)
ice_ref_salinity = dbl_kind(4.0)     # (ppt)
kice = dbl_kind(2.03)                # thermal conductivity of fresh ice (W/m/deg)
                                     # kice is only used with ktherm=1 (BL99) and conduct='MU71'
ksno = dbl_kind(0.30)                # thermal conductivity of snow (W/m/deg)
hs_min = dbl_kind(1.0e-4)            # min snow thickness for computing zTsn (m)
snowpatch = dbl_kind(0.02)           # parameter for fractional snow area (m)
saltmax = dbl_kind(3.2)              # max salinity at ice base for BL99 (ppt)
                                     # phi_init, dSin0_frazil are for mushy thermo
phi_init = dbl_kind(0.75)            # initial liquid fraction of frazil
min_salin = p1                       # threshold for brine pocket treatment
Tliquidus_max = c0                   # maximum liquidus temperature of mush (C)
dSin0_frazil = c3                    # bulk salinity reduction of newly formed frazil
ustar_min = dbl_kind(0.005)          # minimum friction velocity for ocean heat flux (m/s)
hi_min = p01                         # minimum ice thickness allowed (m) for thermo
# mushy thermo
a_rapid_mode = dbl_kind(0.5e-3)      # channel radius for rapid drainage mode (m)
Rac_rapid_mode = dbl_kind(10.0)      # critical Rayleigh number
aspect_rapid_mode = dbl_kind(1.0)    # aspect ratio (larger is wider)
dSdt_slow_mode = dbl_kind(-1.5e-7)   # slow mode drainage strength (m s-1 K-1)
phi_c_slow_mode = dbl_kind(0.05)     # critical liquid fraction porosity cutoff
phi_i_mushy = dbl_kind(0.85)         # liquid fraction of congelation ice

ktherm = int_kind(1)  # type of thermodynamics
                      # -1 none
                      # 1 = Bitz and Lipscomb 1999
                      # 2 = mushy layer theory
conduct = "bubbly"                 # 'MU71' or 'bubbly'
fbot_xfer_type = "constant"        # transfer coefficient type for ice-ocean heat flux
cpl_frazil = "fresh_ice_correction"  # type of coupling for frazil ice
calc_Tsfc = log_kind(True)         # if true, calculate surface temperature
                                   # if false, Tsfc is computed elsewhere and
                                   # atmos-ice fluxes are provided to CICE
semi_implicit_Tsfc = log_kind(False)       # surface temperature coupling option
vapor_flux_correction = log_kind(False)    # compute mass/enthalpy correction for evaporation/sublimation
update_ocn_f = log_kind(False)             # include fresh water and salt fluxes for frazil
modal_aero = log_kind(False)               # if true, use modal aerosal optical properties
                                           # only for use with tr_aero or tr_zaero
conserv_check = log_kind(False)            # if true, do conservations checks and abort
congel_freeze = "two-step"  # congelation computation
                            # 'two-step' = original formulation
                            # 'one-step' = Plante et al, The Cryosphere, 2024
tfrz_option = "mushy"  # form of ocean freezing temperature
                       # 'minus1p8' = -1.8 C
                       # 'constant' = Tocnfrz
                       # 'linear_salt' = -depressT * sss
                       # 'mushy' conforms with ktherm=2
saltflux_option = "constant"  # Salt flux computation
                              # 'constant' reference value of ice_ref_salinity
                              # 'prognostic' prognostic salt flux

#-----------------------------------------------------------------------
# Parameters for radiation
#-----------------------------------------------------------------------

# (Briegleb JGR 97 11475-11485  July 1992)
emissivity = dbl_kind(0.985)             # emissivity of snow and ice
albocn = dbl_kind(0.06)                  # ocean albedo
vonkar = dbl_kind(0.4)                   # von Karman constant
stefan_boltzmann = dbl_kind(567.0e-10)   # W/m^2/K^4
# (Ebert, Schramm and Curry JGR 100 15965-15975 Aug 1995)
kappav = dbl_kind(1.4)                   # vis extnctn coef in ice, wvlngth<700nm (1/m)
hi_ssl = dbl_kind(0.050)                 # ice surface scattering layer thickness (m)
hs_ssl = dbl_kind(0.040)                 # snow surface scattering layer thickness (m)
hs_ssl_min = dbl_kind(5.0e-4)            # minimum snow surface scattering layer thickness for aerosol (m)
# baseline albedos for ccsm3 shortwave, set in namelist
albicev = dbl_kind(0.78)                 # visible ice albedo for h > ahmax
albicei = dbl_kind(0.36)                 # near-ir ice albedo for h > ahmax
albsnowv = dbl_kind(0.98)                # cold snow albedo, visible
albsnowi = dbl_kind(0.70)                # cold snow albedo, near IR
ahmax = dbl_kind(0.3)                    # thickness above which ice albedo is constant (m)
# dEdd tuning parameters, set in namelist
R_ice = c0                               # sea ice tuning parameter; +1 > 1sig increase in albedo
R_pnd = c0                               # ponded ice tuning parameter; +1 > 1sig increase in albedo
R_snw = c1p5                             # snow tuning parameter; +1 > ~.01 change in broadband albedo
dT_mlt = c1p5                            # change in temp for non-melt to melt snow grain
                                         # radius change (C)
rsnw_mlt = dbl_kind(1500.0)              # maximum melting snow grain radius (10^-6 m)
kalg = dbl_kind(0.60)                    # algae absorption coefficient for 0.5 m thick layer
                                         # 0.5 m path of 75 mg Chl a / m2
R_gC2molC = dbl_kind(12.0107)            # g carbon per mol carbon
# weights for albedos
# 4 Jan 2007 BPB  Following are appropriate for complete cloud
# in a summer polar atmosphere with 1.5m bare sea ice surface:
# .636/.364 vis/nir with only 0.5% direct for each band.
# currently used only for history and diagnostics
awtvdr = dbl_kind(0.00318)  # visible, direct
awtidr = dbl_kind(0.00182)  # near IR, direct
awtvdf = dbl_kind(0.63282)  # visible, diffuse
awtidf = dbl_kind(0.36218)  # near IR, diffuse
shortwave = "dEdd"    # shortwave method, 'ccsm3' or 'dEdd' or 'dEdd_snicar_ad'
albedo_type = "ccsm3" # albedo parameterization, 'ccsm3' or 'constant'
                      # shortwave='dEdd' overrides this parameter
# Parameters for shortwave redistribution
sw_redist = log_kind(False)
sw_frac = dbl_kind(0.9)      # Fraction of internal shortwave moved to surface
sw_dtemp = dbl_kind(0.02)    # temperature difference from melting
# Parameters for dEdd_snicar_ad
snw_ssp_table = "test"  # lookup table: 'snicar' or 'test'
# Parameters for the impact of pond depth on shortwave
hpmin = dbl_kind(0.005)  # minimum allowed melt pond depth (m)
hp0 = dbl_kind(0.200)    # pond depth below which transition to bare ice

#-----------------------------------------------------------------------
# Parameters for dynamics, including ridging and strength
#-----------------------------------------------------------------------

# defined in namelist
kstrength = int_kind(1)    # 0 for simple Hibler (1979) formulation
                           # 1 for Rothrock (1975) pressure formulation
krdg_partic = int_kind(1)  # 0 for Thorndike et al. (1975) formulation
                           # 1 for exponential participation function
krdg_redist = int_kind(1)  # 0 for Hibler (1980) formulation
                           # 1 for exponential redistribution function
itd_area_min = dbl_kind(1.0e-11)       # zap residual ice below a minimum area
itd_mass_min = dbl_kind(1.0e-10)       # zap residual ice below a minimum mass
Cf = dbl_kind(17.0)                    # ratio of ridging work to PE change in ridging
Pstar = dbl_kind(2.75e4)               # constant in Hibler strength formula
                                       # (kstrength = 0)
Cstar = dbl_kind(20.0)                 # constant in Hibler strength formula
                                       # (kstrength = 0)
dragio = dbl_kind(0.00536)             # ice-ocn drag coefficient
thickness_ocn_layer1 = dbl_kind(2.0)   # thickness of first ocean level (m)
iceruf_ocn = dbl_kind(0.03)            # under-ice roughness (m)
gravit = dbl_kind(9.80616)             # gravitational acceleration (m/s^2)
mu_rdg = dbl_kind(3.0)                 # e-folding scale of ridged ice, krdg_partic=1 (m^0.5)
                                       # (krdg_redist = 1)
calc_dragio = log_kind(False)  # if true, calculate dragio from iceruf_ocn and thickness_ocn_layer1

#-----------------------------------------------------------------------
# Parameters for atmosphere
#-----------------------------------------------------------------------

cp_air = dbl_kind(1005.0)        # specific heat of air (J/kg/K)
cp_wv = dbl_kind(1.81e3)         # specific heat of water vapor (J/kg/K)
zvir = dbl_kind(0.606)           # rh2o/rair - 1.0
zref = dbl_kind(10.0)            # reference height for stability (m)
iceruf = dbl_kind(0.0005)        # ice surface roughness (m)
qqqice = dbl_kind(11637800.0)    # for qsat over ice
TTTice = dbl_kind(5897.8)        # for qsat over ice
qqqocn = dbl_kind(627572.4)      # for qsat over ocn
TTTocn = dbl_kind(5107.4)        # for qsat over ocn
senscoef = dbl_kind(0.0012)      # Sensible heat flux coefficient for constant-based boundary layer
latncoef = dbl_kind(0.0015)      # Latent heat flux coefficient for constant-based boundary layer
atmbndy = "similarity"           # atmo boundary method, 'similarity', 'constant' or 'mixed'
calc_strair = log_kind(True)     # if true, calculate wind stress
formdrag = log_kind(False)       # if true, calculate form drag
highfreq = log_kind(False)       # if true, calculate high frequency coupling
natmiter = int_kind(5)           # number of iterations for atm boundary layer calcs
# Flux convergence tolerance
atmiter_conv = c0

#-----------------------------------------------------------------------
# Parameters for the ice thickness distribution
#-----------------------------------------------------------------------

kitd = int_kind(1)       # type of itd conversions
                         #   0 = delta function
                         #   1 = linear remap
kcatbound = int_kind(1)  #   0 = old category boundary formula
                         #   1 = new formula giving round numbers
                         #   2 = WMO standard
                         #   3 = asymptotic formula

#-----------------------------------------------------------------------
# Parameters for the floe size distribution
#-----------------------------------------------------------------------

nfreq = int_kind(25)             # number of frequencies
floeshape = dbl_kind(0.66)       # constant from Rothrock 1984 (unitless)
floediam = dbl_kind(300.0)       # effective floe diameter for lateral melt (m)
wave_spec = log_kind(False)      # if true, use wave forcing
wave_spec_type = "constant"      # 'none', 'constant', or 'random'
wave_height_type = "internal"    # 'none', 'internal', 'coupled'

#-----------------------------------------------------------------------
# Parameters for melt ponds
#-----------------------------------------------------------------------

hs0 = dbl_kind(0.03)          # snow depth for transition to bare sea ice (m)
hs1 = dbl_kind(0.03)          # snow depth for transition to bare pond ice (m)
dpscale = dbl_kind(0.001)     # alter e-folding time scale for flushing (ktherm=1)
rfracmin = dbl_kind(0.15)     # minimum retained fraction of meltwater
rfracmax = dbl_kind(0.85)     # maximum retained fraction of meltwater
pndaspect = dbl_kind(0.8)     # ratio of pond depth to area fraction
frzpnd = "cesm"               # pond refreezing parameterization
# sealvl ponds
apnd_sl = dbl_kind(0.27)      # equilibrium pond fraction for sea level parameterization
pndhyps = "sealevel"          # pond hypsometry option
pndfrbd = "floor"             # over what domain to calculate freeboard constraint
pndhead = "perched"           # geometry for computing pond pressure head
pndmacr = "lambda"            # driving force for macro-flaw pond drainage
# topo ponds
hp1 = dbl_kind(0.01)          # critical pond lid thickness for topo ponds

#-----------------------------------------------------------------------
# Parameters for snow redistribution, metamorphosis
#-----------------------------------------------------------------------

snwredist = "none"            # type of snow redistribution
snw_aging_table = "test"      # lookup table: 'snicar' or 'test' or 'file'
use_smliq_pnd = log_kind(False)  # use liquid in snow for ponds
snwgrain = log_kind(False)       # snow metamorphosis
rsnw_fall = dbl_kind(54.526)     # radius of new snow (10^-6 m)
rsnw_tmax = dbl_kind(1500.0)     # maximum snow radius (10^-6 m)
rhosnew = dbl_kind(100.0)        # new snow density (kg/m^3)
rhosmin = dbl_kind(100.0)        # minimum snow density (kg/m^3)
rhosmax = dbl_kind(450.0)        # maximum snow density (kg/m^3)
windmin = dbl_kind(10.0)         # minimum wind speed to compact snow (m/s)
drhosdwind = dbl_kind(27.3)      # wind compaction factor for snow (kg s/m^4)
snwlvlfac = dbl_kind(0.3)        # fractional increase in snow
                                 # depth for bulk redistribution
snw_growth_wet = dbl_kind(4.22e5)  # wet metamorphism parameter (um^3/s)
                                   # 1.e18 * 4.22e-13 (Oleson 2010)
drsnw_min = dbl_kind(0.0)        # minimum snow grain growth factor
snwliq_max = dbl_kind(0.033)     # irreducible saturation fraction
                                 # 0.033 (Anderson 1976)
                                 # 0.09 to 0.1  (Denoth et al, 1979 & Brun 1989)

# indices for aging lookup table
# Fortran leaves these undefined until icepack_init_parameters.
isnw_T = int_kind(0)       # maximum temperature index
isnw_Tgrd = int_kind(0)    # maximum temperature gradient index
isnw_rhos = int_kind(0)    # maximum snow density index

# dry snow aging parameters
# None stands for an unallocated Fortran array.
snowage_rhos = None    # snowage table dimension data for rhos (kg/m^3)
snowage_Tgrd = None    # snowage table dimension data for temp gradient (deg K/m)
snowage_T = None       # snowage table dimension data for temperature (deg K)
snowage_tau = None     # snowage table 3D data for tau (10^-6 m)
snowage_kappa = None   # snowage table 3D data for kappa (10^-6 m)
snowage_drdt0 = None   # snowage table 3D data for drdt0 (10^-6 m/hr)

#-----------------------------------------------------------------------
# Parameters for biogeochemistry
#-----------------------------------------------------------------------

# skl biology parameters
bgc_flux_type = "Jin2006"  # type of ocean-ice piston velocity (or 'constant')

z_tracers = log_kind(False)   # if true, bgc or aerosol tracers are vertically resolved
scale_bgc = log_kind(False)   # if true, initialize bgc tracers proportionally with salinity
solve_zbgc = log_kind(False)  # if true, solve vertical biochemistry portion of code
dEdd_algae = log_kind(False)  # if true, algal absorption of shortwave is computed in the
skl_bgc = log_kind(False)     # if true, solve skeletal biochemistry
use_macromolecules = log_kind(False)  # if true, ocean DOC already split into
                                      # polysaccharids, lipid and protein fractions
use_atm_dust_iron = log_kind(False)   # if true, compute iron contribution from dust
restartbgc = log_kind(False)

phi_snow = dbl_kind(-1.0)       # snow porosity (compute from snow density if negative)
grid_o = dbl_kind(0.006)        # for bottom flux
initbio_frac = c1               # fraction of ocean trcr concentration in bio trcrs
l_sk = dbl_kind(2.0)            # characteristic diffusive scale brine (m)
algal_vel = dbl_kind(1.0e-7)    # 0.5 cm/d(m/s) Lavoie 2005  1.5 cm/day
R_dFe2dust = dbl_kind(0.035)    # g/g (3.5% content) Tagliabue 2009
dustFe_sol = dbl_kind(0.005)    # solubility fraction
frazil_scav = dbl_kind(0.8)     # fraction or multiple of bgc concentrated in frazil ice
sk_l = dbl_kind(0.03)           # skeletal layer thickness (m)
min_bgc = dbl_kind(0.01)        # fraction of ocean bgc concentration in surface melt
T_max = c0                      # maximum temperature (C)
fsal = c1                       # Salinity limitation (1)
op_dep_min = p1                 # light attenuates for optical depths exceeding min
fr_graze_s = p5                 # fraction of grazing spilled or slopped
fr_graze_e = p5                 # fraction of assimilation excreted
fr_mort2min = dbl_kind(0.9)     # fractionation of mortality to Am
fr_dFe = dbl_kind(1.0)          # fraction of remineralized nitrogen
                                # (in units of algal iron)
k_nitrif = dbl_kind(0.046)      # nitrification rate (1/day)
t_iron_conv = dbl_kind(3065.0)  # desorption loss pFe to dFe (day)
max_loss = dbl_kind(0.9)        # restrict uptake to % of remaining value
max_dfe_doc1 = dbl_kind(0.2)    # max ratio of dFe to saccharides in the ice
                                # (nM Fe/muM C)
fr_resp = dbl_kind(0.05)        # fraction of algal growth lost due to respiration
fr_resp_s = dbl_kind(0.90)      # DMSPd fraction of respiration loss as DMSPd
y_sk_DMS = dbl_kind(0.7)        # fraction conversion given high yield
t_sk_conv = dbl_kind(5.0)       # Stefels conversion time (d)
t_sk_ox = dbl_kind(12.0)        # DMS oxidation time (d)
grid_o_t = dbl_kind(0.006)      # ice surface molecular sublayer thickness (m)
ratio_Si2N_diatoms = dbl_kind(1.8)   # algal Si to N (mol/mol)
ratio_Si2N_sp = c0
ratio_Si2N_phaeo = c0
ratio_S2N_diatoms = dbl_kind(0.03)   # algal S to N (mol/mol)
ratio_S2N_sp = dbl_kind(0.03)
ratio_S2N_phaeo = dbl_kind(0.03)
ratio_Fe2C_diatoms = dbl_kind(0.0033)  # algal Fe to C (umol/mol)
ratio_Fe2C_sp = dbl_kind(0.0033)
ratio_Fe2C_phaeo = dbl_kind(0.1)
ratio_Fe2N_diatoms = dbl_kind(0.023)   # algal Fe to N (umol/mol)
ratio_Fe2N_sp = dbl_kind(0.023)
ratio_Fe2N_phaeo = dbl_kind(0.7)
ratio_Fe2DON = dbl_kind(0.023)         # Fe to N of DON (nmol/umol)
ratio_Fe2DOC_s = dbl_kind(0.1)         # Fe to C of DOC (nmol/umol) saccharids
ratio_Fe2DOC_l = dbl_kind(0.033)       # Fe to C of DOC (nmol/umol) lipids
tau_min = dbl_kind(3600.0)             # rapid timescale for mobile to stationary exchanges (s)
tau_max = dbl_kind(604800.0)           # short timescale for mobile to stationary exchanges (s)
chlabs_diatoms = dbl_kind(0.03)        # absorptivity for diatoms (1/m/(mg/m3))
chlabs_sp = dbl_kind(0.01)             # absorptivity for small plankton (1/m/(mg/m3))
chlabs_phaeo = dbl_kind(0.05)          # absorptivity for phaeocystis (1/m/(mg/m3))
alpha2max_low_diatoms = dbl_kind(0.3)  # light limitation for diatoms (1/(W/m2))
alpha2max_low_sp = dbl_kind(0.2)       # light limitation for small plankton (1/(W/m2))
alpha2max_low_phaeo = dbl_kind(0.17)   # light limitation for phaeocystis (1/(W/m2))
beta2max_diatoms = dbl_kind(0.001)     # light inhibition (1/(W/m^2))
beta2max_sp = dbl_kind(0.001)
beta2max_phaeo = dbl_kind(0.04)
mu_max_diatoms = dbl_kind(1.44)        # maximum growth rate (1/day)
mu_max_sp = dbl_kind(0.41)
mu_max_phaeo = dbl_kind(0.63)
grow_Tdep_diatoms = dbl_kind(0.063)    # Temperature dependence of growth (1/C)
grow_Tdep_sp = dbl_kind(0.063)
grow_Tdep_phaeo = dbl_kind(0.063)
fr_graze_diatoms = dbl_kind(0.19)      # Fraction grazed
fr_graze_sp = dbl_kind(0.19)
fr_graze_phaeo = dbl_kind(0.19)
mort_pre_diatoms = dbl_kind(0.007)     # Mortality (1/day)
mort_pre_sp = dbl_kind(0.007)
mort_pre_phaeo = dbl_kind(0.007)
mort_Tdep_diatoms = dbl_kind(0.03)     # T dependence of mortality (1/C)
mort_Tdep_sp = dbl_kind(0.03)
mort_Tdep_phaeo = dbl_kind(0.03)
k_exude_diatoms = c0                   # algal exudation (1/d)
k_exude_sp = c0
k_exude_phaeo = c0
K_Nit_diatoms = c1                     # nitrate half saturation (mmol/m^3)
K_Nit_sp = c1
K_Nit_phaeo = c1
K_Am_diatoms = dbl_kind(0.3)           # ammonium half saturation (mmol/m^3)
K_Am_sp = dbl_kind(0.3)
K_Am_phaeo = dbl_kind(0.3)
K_Sil_diatoms = dbl_kind(4.0)          # silicate half saturation (mmol/m^3)
K_Sil_sp = c0
K_Sil_phaeo = c0
K_Fe_diatoms = c1                      # iron half saturation (nM)
K_Fe_sp = dbl_kind(0.2)
K_Fe_phaeo = dbl_kind(0.1)
f_don_protein = dbl_kind(0.6)          # fraction of spilled grazing to proteins
kn_bac_protein = dbl_kind(0.2)         # Bacterial degredation of DON (1/d)
f_don_Am_protein = c1                  # fraction of remineralized DON to ammonium
f_doc_s = dbl_kind(0.5)                # fraction of mortality to DOC
f_doc_l = dbl_kind(0.5)
f_exude_s = c1                         # fraction of exudation to DOC
f_exude_l = c1
k_bac_s = dbl_kind(0.03)               # Bacterial degredation of DOC (1/d)
k_bac_l = dbl_kind(0.03)
algaltype_diatoms = c0                 # mobility type
algaltype_sp = c0                      # algal groups
algaltype_phaeo = c0
nitratetype = dbl_kind(-1.0)           # nitrate
ammoniumtype = c0                      # ammonium
silicatetype = dbl_kind(-1.0)          # silicate
dmspptype = dbl_kind(0.5)              # DMS
dmspdtype = c0
humtype = c0                           # humics
doctype_s = c0                         # DOC
doctype_l = c0
dictype_1 = dbl_kind(-1.0)             # DIC
dontype_protein = c0                   # DON
fedtype_1 = c0                         # Iron
feptype_1 = dbl_kind(0.5)
zaerotype_bc1 = dbl_kind(-1.0)         # Aerosols
zaerotype_bc2 = dbl_kind(-1.0)
zaerotype_dust1 = dbl_kind(-1.0)
zaerotype_dust2 = dbl_kind(-1.0)
zaerotype_dust3 = dbl_kind(-1.0)
zaerotype_dust4 = dbl_kind(-1.0)
ratio_C2N_diatoms = dbl_kind(7.0)      # algal carbon to nitrogen mole ratio
ratio_C2N_sp = dbl_kind(7.0)
ratio_C2N_phaeo = dbl_kind(7.0)
ratio_chl2N_diatoms = dbl_kind(2.1)    # algal chlorophyll to nitrogen ratio (g chla/mol)
ratio_chl2N_sp = dbl_kind(1.1)
ratio_chl2N_phaeo = dbl_kind(0.84)
ratio_C2N_proteins = dbl_kind(5.0)     # Ratio of carbon to nitrogen in proteins
F_abs_chl_diatoms = dbl_kind(2.0)      # scales absorbed radiation for dEdd
F_abs_chl_sp = dbl_kind(4.0)
F_abs_chl_phaeo = dbl_kind(5.0)


def set_values(**kwargs):
    unknown = [name for name in kwargs if name not in globals()]
    if unknown:
        raise KeyError(unknown)
    for name, value in kwargs.items():
        current = globals()[name]
        if isinstance(current, np.floating):
            value = dbl_kind(value)
        elif isinstance(current, np.integer):
            value = int_kind(value)
        elif isinstance(current, np.bool_):
            value = log_kind(value)
        elif isinstance(current, np.ndarray):
            value = np.array(value, dtype=dbl_kind, copy=True)
        globals()[name] = value


def icepack_recompute_constants():
    """Reinitialize some derived constants."""
    global cprho, Lfresh, Cp, pih, piq, pi2, rad_to_deg, dragio
    cprho = dbl_kind(cp_ocn * rhow)
    Lfresh = dbl_kind(Lsub - Lvap)
    Cp = dbl_kind(0.5) * gravit * (rhow - rhoi) * rhoi / rhow
    pih = p5 * pi
    piq = p5 * p5 * pi
    pi2 = c2 * pi
    rad_to_deg = c180 / pi
    if calc_dragio:
        dragio = (vonkar / math.log(p5 * thickness_ocn_layer1 / iceruf_ocn)) ** 2  # dragio at half first layer
        lam = (thickness_ocn_layer1 - iceruf_ocn) / (
            thickness_ocn_layer1
            * (
                math.sqrt(dragio) / vonkar
                * (math.log(c2) - c1 + iceruf_ocn / thickness_ocn_layer1)
                + c1
            )
        )
        dragio = dbl_kind(dragio * lam ** 2)


def icepack_init_parameters(**kwargs):
    """Set the column package internal parameters.

    Each argument is optional. A trailing ``_in`` matches the Fortran
    keyword, as in ``ktherm_in``. Only names that are present are changed.
    Snow-age tables are checked against ``isnw_rhos``, ``isnw_Tgrd``, and
    ``isnw_T``, then stored. Derived constants are recomputed at the end.
    """
    updates = {}
    for key, value in kwargs.items():
        name = key[:-3] if key.endswith("_in") else key
        updates[name] = value

    snowage_vectors = {
        "snowage_rhos": "isnw_rhos",
        "snowage_Tgrd": "isnw_Tgrd",
        "snowage_T": "isnw_T",
    }
    snowage_tables = ("snowage_tau", "snowage_kappa", "snowage_drdt0")
    vectors = {name: updates.pop(name) for name in snowage_vectors if name in updates}
    tables = {name: updates.pop(name) for name in snowage_tables if name in updates}
    set_values(**updates)

    for name, values in vectors.items():
        length = int(globals()[snowage_vectors[name]])
        arr = np.asarray(values)
        if arr.shape != (length,):
            raise ValueError(f"(icepack_init_parameters) incorrect size of {name}_in")
        globals()[name] = np.array(arr, dtype=dbl_kind, copy=True)
    table_shape = (int(isnw_rhos), int(isnw_Tgrd), int(isnw_T))
    for name, values in tables.items():
        arr = np.asarray(values)
        if arr.shape != table_shape:
            raise ValueError(f"(icepack_init_parameters) incorrect size of {name}_in")
        globals()[name] = np.array(arr, dtype=dbl_kind, copy=True)

    errors = []
    if solve_zbgc or skl_bgc:
        for name in (
            "ratio_Fe2N_diatoms",
            "ratio_Fe2N_sp",
            "ratio_Fe2N_phaeo",
            "K_Nit_diatoms",
            "K_Nit_sp",
            "K_Nit_phaeo",
            "K_Am_diatoms",
            "K_Am_sp",
            "K_Am_phaeo",
        ):
            if globals()[name] <= c0:
                errors.append(f"(icepack_init_parameters) WARNING: {name} < = 0")
    if argcheck not in ("never", "first", "always"):
        errors.append(
            "(icepack_init_parameters) argcheck must be never, first, or always"
        )

    icepack_recompute_constants()
    if errors:
        raise ValueError("; ".join(errors))


def icepack_query_parameters(*names, **kwargs):
    """Query the column package internal parameters.

    Names may use the Fortran ``_out`` suffix. Returns a dict of the
    requested values. An unallocated snow-age table is None.
    """
    requested = [*(names), *kwargs]
    found = {}
    module = globals()
    for key in requested:
        name = key[:-4] if key.endswith("_out") else key
        if name not in module:
            raise KeyError(name)
        value = module[name]
        if isinstance(value, np.ndarray):
            value = value.copy()
        found[name] = value
    return found


def icepack_write_parameters(iounit):
    """Write the column package internal parameters.

    ``iounit`` is the diagnostic stream. CICE passes ``nu_diag``, which is
    stdout or the file named by ``diag_file``.
    """
    fields = (
        ("  rhos       = ", rhos),
        ("  rhoi       = ", rhoi),
        ("  rhow       = ", rhow),
        ("  cp_air     = ", cp_air),
        ("  emissivity = ", emissivity),
        ("  floediam   = ", floediam),
        ("  hfrazilmin = ", hfrazilmin),
        ("  cp_ice     = ", cp_ice),
        ("  cp_ocn     = ", cp_ocn),
        ("  depressT   = ", depressT),
        ("  dragio     = ", dragio),
        ("  calc_dragio= ", calc_dragio),
        ("  iceruf_ocn = ", iceruf_ocn),
        ("  thickness_ocn_layer1 = ", thickness_ocn_layer1),
        ("  albocn     = ", albocn),
        ("  gravit     = ", gravit),
        ("  viscosity_dyn = ", viscosity_dyn),
        ("  tscale_pnd_drain = ", tscale_pnd_drain),
        ("  Tocnfrz    = ", Tocnfrz),
        ("  rhofresh   = ", rhofresh),
        ("  zvir       = ", zvir),
        ("  vonkar     = ", vonkar),
        ("  cp_wv      = ", cp_wv),
        ("  stefan_boltzmann = ", stefan_boltzmann),
        ("  Tffresh    = ", Tffresh),
        ("  Lsub       = ", Lsub),
        ("  Lvap       = ", Lvap),
        ("  Timelt     = ", Timelt),
        ("  Tsmelt     = ", Tsmelt),
        ("  ice_ref_salinity = ", ice_ref_salinity),
        ("  iceruf     = ", iceruf),
        ("  itd_area_min = ", itd_area_min),
        ("  itd_mass_min = ", itd_mass_min),
        ("  Cf         = ", Cf),
        ("  Pstar      = ", Pstar),
        ("  Cstar      = ", Cstar),
        ("  kappav     = ", kappav),
        ("  kice       = ", kice),
        ("  ksno       = ", ksno),
        ("  zref       = ", zref),
        ("  hs_min     = ", hs_min),
        ("  snowpatch  = ", snowpatch),
        ("  rhosi      = ", rhosi),
        ("  sk_l       = ", sk_l),
        ("  saltmax    = ", saltmax),
        ("  phi_init   = ", phi_init),
        ("  min_salin  = ", min_salin),
        ("  Tliquidus_max = ", Tliquidus_max),
        ("  min_bgc    = ", min_bgc),
        ("  dSin0_frazil = ", dSin0_frazil),
        ("  hi_ssl     = ", hi_ssl),
        ("  hs_ssl     = ", hs_ssl),
        ("  hs_ssl_min = ", hs_ssl_min),
        ("  awtvdr     = ", awtvdr),
        ("  awtidr     = ", awtidr),
        ("  awtvdf     = ", awtvdf),
        ("  awtidf     = ", awtidf),
        ("  qqqice     = ", qqqice),
        ("  TTTice     = ", TTTice),
        ("  qqqocn     = ", qqqocn),
        ("  TTTocn     = ", TTTocn),
        ("  argcheck   = ", argcheck.strip()),
        ("  puny       = ", puny),
        ("  bignum     = ", bignum),
        ("  secday     = ", secday),
        ("  pi         = ", pi),
        ("  pih        = ", pih),
        ("  piq        = ", piq),
        ("  pi2        = ", pi2),
        ("  rad_to_deg = ", rad_to_deg),
        ("  Lfresh     = ", Lfresh),
        ("  cprho      = ", cprho),
        ("  Cp         = ", Cp),
        ("  ktherm     = ", ktherm),
        ("  conduct    = ", conduct.strip()),
        ("  fbot_xfer_type = ", fbot_xfer_type.strip()),
        ("  calc_Tsfc  = ", calc_Tsfc),
        ("  semi_implicit_Tsfc = ", semi_implicit_Tsfc),
        ("  vapor_flux_correction = ", vapor_flux_correction),
        ("  cpl_frazil = ", cpl_frazil.strip()),
        ("  update_ocn_f = ", update_ocn_f),
        ("  ustar_min  = ", ustar_min),
        ("  hi_min     = ", hi_min),
        ("  a_rapid_mode = ", a_rapid_mode),
        ("  Rac_rapid_mode = ", Rac_rapid_mode),
        ("  aspect_rapid_mode = ", aspect_rapid_mode),
        ("  dSdt_slow_mode = ", dSdt_slow_mode),
        ("  phi_c_slow_mode = ", phi_c_slow_mode),
        ("  phi_i_mushy= ", phi_i_mushy),
        ("  shortwave  = ", shortwave.strip()),
        ("  albedo_type= ", albedo_type.strip()),
        ("  albicev    = ", albicev),
        ("  albicei    = ", albicei),
        ("  albsnowv   = ", albsnowv),
        ("  albsnowi   = ", albsnowi),
        ("  ahmax      = ", ahmax),
        ("  R_ice      = ", R_ice),
        ("  R_pnd      = ", R_pnd),
        ("  R_snw      = ", R_snw),
        ("  dT_mlt     = ", dT_mlt),
        ("  rsnw_mlt   = ", rsnw_mlt),
        ("  kalg       = ", kalg),
        ("  R_gC2molC  = ", R_gC2molC),
        ("  kstrength  = ", kstrength),
        ("  krdg_partic= ", krdg_partic),
        ("  krdg_redist= ", krdg_redist),
        ("  mu_rdg     = ", mu_rdg),
        ("  atmbndy    = ", atmbndy.strip()),
        ("  calc_strair= ", calc_strair),
        ("  formdrag   = ", formdrag),
        ("  highfreq   = ", highfreq),
        ("  natmiter   = ", natmiter),
        ("  atmiter_conv = ", atmiter_conv),
        ("  congel_freeze = ", congel_freeze.strip()),
        ("  tfrz_option= ", tfrz_option.strip()),
        ("  saltflux_option = ", saltflux_option.strip()),
        ("  kitd       = ", kitd),
        ("  kcatbound  = ", kcatbound),
        ("  floeshape  = ", floeshape),
        ("  wave_spec  = ", wave_spec),
        ("  wave_spec_type = ", wave_spec_type.strip()),
        ("  wave_height_type = ", wave_height_type.strip()),
        ("  nfreq      = ", nfreq),
        ("  hs0        = ", hs0),
        ("  frzpnd     = ", frzpnd.strip()),
        ("  dpscale    = ", dpscale),
        ("  rfracmin   = ", rfracmin),
        ("  rfracmax   = ", rfracmax),
        ("  pndaspect  = ", pndaspect),
        ("  apnd_sl    = ", apnd_sl),
        ("  hs1        = ", hs1),
        ("  hp1        = ", hp1),
        ("  snwredist  = ", snwredist.strip()),
        ("  snw_aging_table = ", snw_aging_table.strip()),
        ("  snwgrain   = ", snwgrain),
        ("  use_smliq_pnd = ", use_smliq_pnd),
        ("  rsnw_fall  = ", rsnw_fall),
        ("  rsnw_tmax  = ", rsnw_tmax),
        ("  rhosnew    = ", rhosnew),
        ("  rhosmin    = ", rhosmin),
        ("  rhosmax    = ", rhosmax),
        ("  windmin    = ", windmin),
        ("  drhosdwind = ", drhosdwind),
        ("  snwlvlfac  = ", snwlvlfac),
        ("  snw_growth_wet = ", snw_growth_wet),
        ("  drsnw_min  = ", drsnw_min),
        ("  snwliq_max = ", snwliq_max),
        ("  isnw_T     = ", isnw_T),
        ("  isnw_Tgrd  = ", isnw_Tgrd),
        ("  isnw_rhos  = ", isnw_rhos),
        # write(iounit,*) "  snowage_rhos  = ", snowage_rhos(1)
        # write(iounit,*) "  snowage_Tgrd  = ", snowage_Tgrd(1)
        # write(iounit,*) "  snowage_T     = ", snowage_T(1)
        # write(iounit,*) "  snowage_tau   = ", snowage_tau(1,1,1)
        # write(iounit,*) "  snowage_kappa = ", snowage_kappa(1,1,1)
        # write(iounit,*) "  snowage_drdt0 = ", snowage_drdt0(1,1,1)
        ("  snw_ssp_table = ", snw_ssp_table.strip()),
        ("  bgc_flux_type = ", bgc_flux_type.strip()),
        ("  z_tracers  = ", z_tracers),
        ("  scale_bgc  = ", scale_bgc),
        ("  solve_zbgc = ", solve_zbgc),
        ("  dEdd_algae = ", dEdd_algae),
        ("  modal_aero = ", modal_aero),
        ("  use_macromolecules = ", use_macromolecules),
        ("  use_atm_dust_iron  = ", use_atm_dust_iron),
        ("  restartbgc = ", restartbgc),
        ("  conserv_check = ", conserv_check),
        ("  skl_bgc    = ", skl_bgc),
        ("  grid_o     = ", grid_o),
        ("  l_sk       = ", l_sk),
        ("  grid_o_t   = ", grid_o_t),
        ("  initbio_frac = ", initbio_frac),
        ("  frazil_scav= ", frazil_scav),
        ("  phi_snow   = ", phi_snow),
        ("  ratio_Si2N_diatoms = ", ratio_Si2N_diatoms),
        ("  ratio_Si2N_sp      = ", ratio_Si2N_sp),
        ("  ratio_Si2N_phaeo   = ", ratio_Si2N_phaeo),
        ("  ratio_S2N_diatoms  = ", ratio_S2N_diatoms),
        ("  ratio_S2N_sp       = ", ratio_S2N_sp),
        ("  ratio_S2N_phaeo    = ", ratio_S2N_phaeo),
        ("  ratio_Fe2C_diatoms = ", ratio_Fe2C_diatoms),
        ("  ratio_Fe2C_sp      = ", ratio_Fe2C_sp),
        ("  ratio_Fe2C_phaeo   = ", ratio_Fe2C_phaeo),
        ("  ratio_Fe2N_diatoms = ", ratio_Fe2N_diatoms),
        ("  ratio_Fe2N_sp      = ", ratio_Fe2N_sp),
        ("  ratio_Fe2N_phaeo   = ", ratio_Fe2N_phaeo),
        ("  ratio_Fe2DON       = ", ratio_Fe2DON),
        ("  ratio_Fe2DOC_s     = ", ratio_Fe2DOC_s),
        ("  ratio_Fe2DOC_l     = ", ratio_Fe2DOC_l),
        ("  tau_min            = ", tau_min),
        ("  tau_max            = ", tau_max),
        ("  chlabs_diatoms     = ", chlabs_diatoms),
        ("  chlabs_sp          = ", chlabs_sp),
        ("  chlabs_phaeo       = ", chlabs_phaeo),
        ("  alpha2max_low_diatoms   = ", alpha2max_low_diatoms),
        ("  alpha2max_low_sp        = ", alpha2max_low_sp),
        ("  alpha2max_low_phaeo     = ", alpha2max_low_phaeo),
        ("  beta2max_diatoms    = ", beta2max_diatoms),
        ("  beta2max_sp         = ", beta2max_sp),
        ("  beta2max_phaeo      = ", beta2max_phaeo),
        ("  mu_max_diatoms      = ", mu_max_diatoms),
        ("  mu_max_sp           = ", mu_max_sp),
        ("  mu_max_phaeo        = ", mu_max_phaeo),
        ("  grow_Tdep_diatoms   = ", grow_Tdep_diatoms),
        ("  grow_Tdep_sp        = ", grow_Tdep_sp),
        ("  grow_Tdep_phaeo     = ", grow_Tdep_phaeo),
        ("  fr_graze_diatoms    = ", fr_graze_diatoms),
        ("  fr_graze_sp         = ", fr_graze_sp),
        ("  fr_graze_phaeo      = ", fr_graze_phaeo),
        ("  mort_pre_diatoms    = ", mort_pre_diatoms),
        ("  mort_pre_sp         = ", mort_pre_sp),
        ("  mort_pre_phaeo      = ", mort_pre_phaeo),
        ("  mort_Tdep_diatoms   = ", mort_Tdep_diatoms),
        ("  mort_Tdep_sp        = ", mort_Tdep_sp),
        ("  mort_Tdep_phaeo     = ", mort_Tdep_phaeo),
        ("  k_exude_diatoms     = ", k_exude_diatoms),
        ("  k_exude_sp          = ", k_exude_sp),
        ("  k_exude_phaeo       = ", k_exude_phaeo),
        ("  K_Nit_diatoms       = ", K_Nit_diatoms),
        ("  K_Nit_sp            = ", K_Nit_sp),
        ("  K_Nit_phaeo         = ", K_Nit_phaeo),
        ("  K_Am_diatoms        = ", K_Am_diatoms),
        ("  K_Am_sp             = ", K_Am_sp),
        ("  K_Am_phaeo          = ", K_Am_phaeo),
        ("  K_Sil_diatoms       = ", K_Sil_diatoms),
        ("  K_Sil_sp            = ", K_Sil_sp),
        ("  K_Sil_phaeo         = ", K_Sil_phaeo),
        ("  K_Fe_diatoms        = ", K_Fe_diatoms),
        ("  K_Fe_sp             = ", K_Fe_sp),
        ("  K_Fe_phaeo          = ", K_Fe_phaeo),
        ("  f_don_protein       = ", f_don_protein),
        ("  kn_bac_protein      = ", kn_bac_protein),
        ("  f_don_Am_protein    = ", f_don_Am_protein),
        ("  f_doc_s             = ", f_doc_s),
        ("  f_doc_l             = ", f_doc_l),
        ("  f_exude_s           = ", f_exude_s),
        ("  f_exude_l           = ", f_exude_l),
        ("  k_bac_s             = ", k_bac_s),
        ("  k_bac_l             = ", k_bac_l),
        ("  algaltype_diatoms   = ", algaltype_diatoms),
        ("  algaltype_sp        = ", algaltype_sp),
        ("  algaltype_phaeo     = ", algaltype_phaeo),
        ("  nitratetype         = ", nitratetype),
        ("  ammoniumtype        = ", ammoniumtype),
        ("  silicatetype        = ", silicatetype),
        ("  dmspptype           = ", dmspptype),
        ("  dmspdtype           = ", dmspdtype),
        ("  humtype             = ", humtype),
        ("  doctype_s           = ", doctype_s),
        ("  doctype_l           = ", doctype_l),
        ("  dictype_1           = ", dictype_1),
        ("  dontype_protein     = ", dontype_protein),
        ("  fedtype_1           = ", fedtype_1),
        ("  feptype_1           = ", feptype_1),
        ("  zaerotype_bc1       = ", zaerotype_bc1),
        ("  zaerotype_bc2       = ", zaerotype_bc2),
        ("  zaerotype_dust1     = ", zaerotype_dust1),
        ("  zaerotype_dust2     = ", zaerotype_dust2),
        ("  zaerotype_dust3     = ", zaerotype_dust3),
        ("  zaerotype_dust4     = ", zaerotype_dust4),
        ("  ratio_C2N_diatoms   = ", ratio_C2N_diatoms),
        ("  ratio_C2N_sp        = ", ratio_C2N_sp),
        ("  ratio_C2N_phaeo     = ", ratio_C2N_phaeo),
        ("  ratio_chl2N_diatoms = ", ratio_chl2N_diatoms),
        ("  ratio_chl2N_sp      = ", ratio_chl2N_sp),
        ("  ratio_chl2N_phaeo   = ", ratio_chl2N_phaeo),
        ("  F_abs_chl_diatoms   = ", F_abs_chl_diatoms),
        ("  F_abs_chl_sp        = ", F_abs_chl_sp),
        ("  F_abs_chl_phaeo     = ", F_abs_chl_phaeo),
        ("  ratio_C2N_proteins  = ", ratio_C2N_proteins),
        ("  fr_resp    = ", fr_resp),
        ("  algal_vel  = ", algal_vel),
        ("  R_dFe2dust = ", R_dFe2dust),
        ("  dustFe_sol = ", dustFe_sol),
        ("  T_max      = ", T_max),
        ("  fsal       = ", fsal),
        ("  op_dep_min = ", op_dep_min),
        ("  fr_graze_s = ", fr_graze_s),
        ("  fr_graze_e = ", fr_graze_e),
        ("  fr_mort2min= ", fr_mort2min),
        ("  fr_dFe     = ", fr_dFe),
        ("  k_nitrif   = ", k_nitrif),
        ("  t_iron_conv= ", t_iron_conv),
        ("  max_loss   = ", max_loss),
        ("  max_dfe_doc1 = ", max_dfe_doc1),
        ("  fr_resp_s  = ", fr_resp_s),
        ("  y_sk_DMS   = ", y_sk_DMS),
        ("  t_sk_conv  = ", t_sk_conv),
        ("  t_sk_ox    = ", t_sk_ox),
        ("  sw_redist  = ", sw_redist),
        ("  sw_frac    = ", sw_frac),
        ("  sw_dtemp   = ", sw_dtemp),
    )
    iounit.write("\n")
    iounit.write("(icepack_write_parameters)\n")
    for label, value in fields:
        if isinstance(value, (bool, np.bool_)):
            value = "T" if value else "F"
        iounit.write(f"{label}{value}\n")
    iounit.write("\n")


def icepack_chkoptargflag(first_call):
    return log_kind(argcheck == "always" or (argcheck == "first" and first_call))
