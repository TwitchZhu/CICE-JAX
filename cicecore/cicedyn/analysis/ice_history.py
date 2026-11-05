"""Driver for core history output.

author for JAX version: Xinyuan Zhu
XZ: converted from F90 to Python

2010 Alison McLaren and ECH: Added 3D capability
2009 D Bailey and ECH: Generalized for multiple frequency output
2006 ECH: Accepted some CESM code into mainstream CICE
          Added ice_present, aicen, vicen; removed aice1...10, vice1...1.
          Added histfreq_n and histfreq='h' options, removed histfreq='w'
          Converted to free source form (F90)
          Added option for binary output instead of netCDF
2004 WHL: Block structure added

The following variables are currently hard-wired as snapshots
  (instantaneous rather than time-averages):
  divu, shear, vort, sig1, sig2, sigP, trsig, mlt_onset,
  frz_onset, hisnap, aisnap

Options for histfreq: '1','h','d','m','y','x', where x means that
  output stream will not be used (recommended for efficiency).
histfreq_n can be any nonnegative integer, where 0 means that the
  corresponding histfreq frequency will not be used.
The flags (f_<field>) can be set to '1','h','d','m','y' or 'x', where
  n means the field will not be written.  To output the same field at
  more than one frequency, for instance monthy and daily, set
  f_<field> = 'md'.

This file registers the standard 2D history fields and the category
and vertical ice/snow profiles. Pond, snow-physics, drag, mechanical
redistribution, floe-size, and biogeochemistry catalogs are registered
by their own modules. ice_read_hist reads every allocated rank.
The icefields_nml group is optional here: a missing group keeps the
Fortran default frequencies. An 'x' in a default still drops that field.

authors Tony Craig and Bruce Briegleb, NCAR
        Elizabeth C. Hunke and William H. Lipscomb, LANL
        C. M. Bitz, UW
"""

from contextlib import contextmanager
import time

import jax
import jax.numpy as jnp

from cicecore.cicedyn.analysis import ice_history_bgc
from cicecore.cicedyn.analysis import ice_history_drag
from cicecore.cicedyn.analysis import ice_history_fsd
from cicecore.cicedyn.analysis import ice_history_mechred
from cicecore.cicedyn.analysis import ice_history_pond
from cicecore.cicedyn.analysis import ice_history_shared
from cicecore.cicedyn.analysis import ice_history_snow
from cicecore.cicedyn.dynamics import ice_dyn_shared
from cicecore.cicedyn.dynamics import ice_dyn_eap
from cicecore.cicedyn.general import ice_flux
from cicecore.cicedyn.general import ice_flux_bgc
from cicecore.cicedyn.general import ice_init
from cicecore.cicedyn.general import ice_state
from cicecore.cicedyn.infrastructure.comm.serial import ice_timers
from cicecore.cicedyn.infrastructure.io.io_netcdf import ice_history_write
from cicecore.cicedyn.infrastructure import ice_grid
from cicecore.shared import ice_arrays_column
from cicecore.shared import ice_calendar
from cicecore.shared import ice_domain_size
from cicecore.shared import ice_fileunits
from cicecore.shared.ice_constants import c0, c1, c100, c180, c360, kg_to_g, mps_to_cmpdy, p25
from cicecore.shared.ice_kinds_mod import dbl_kind
from icepack.columnphysics import icepack_parameters
from icepack.columnphysics import icepack_therm_shared
from icepack.columnphysics import icepack_mushy_physics
from icepack.columnphysics import icepack_tracers

# Defaults match ice_history_shared.F90. 'm' is written on a monthly stream.
# 'x' is not written.
_F = {
    "hi": "m",
    "hs": "m",
    "snowfrac": "x",
    "Tsfc": "m",
    "aice": "m",
    "aice_init": "x",
    "uvel": "m",
    "vvel": "m",
    "icespd": "m",
    "icedir": "m",
    "uvelE": "x", "vvelE": "x", "icespdE": "x", "icedirE": "x",
    "uvelN": "x", "vvelN": "x", "icespdN": "x", "icedirN": "x",
    "uatm": "m",
    "vatm": "m",
    "atmspd": "m",
    "atmdir": "m",
    "fswup": "m",
    "fswdn": "m",
    "flwdn": "m",
    "snow": "m",
    "snow_ai": "m",
    "rain": "m",
    "rain_ai": "m",
    "sst": "m",
    "sss": "m",
    "uocn": "m",
    "vocn": "m",
    "ocnspd": "m",
    "ocndir": "m",
    "sice": "m",
    "frzmlt": "m",
    "fswfac": "m",
    "fswint_ai": "x",
    "fswabs": "m",
    "fswabs_ai": "m",
    "albsni": "m",
    "alvdr": "x",
    "alidr": "x",
    "alvdf": "x",
    "alidf": "x",
    "alvdr_ai": "m",
    "alidr_ai": "m",
    "alvdf_ai": "m",
    "alidf_ai": "m",
    "albice": "m",
    "albsno": "m",
    "albpnd": "m",
    "coszen": "m",
    "flat": "m",
    "flat_ai": "m",
    "fsens": "m",
    "fsens_ai": "m",
    "flwup": "m",
    "flwup_ai": "m",
    "evap": "m",
    "evap_ai": "m",
    "Tair": "m",
    "Tref": "m",
    "Qref": "m",
    "congel": "m",
    "frazil": "m",
    "snoice": "m",
    "dsnow": "m",
    "meltt": "m",
    "melts": "m",
    "meltb": "m",
    "meltl": "m",
    "fresh": "m",
    "fresh_ai": "m",
    "fsalt": "m",
    "fsalt_ai": "m",
    "fbot": "m",
    "fhocn": "m",
    "fhocn_ai": "m",
    "fswthru": "m",
    "fswthru_ai": "m",
    "strairx": "m",
    "strairy": "m",
    "strtltx": "m",
    "strtlty": "m",
    "strcorx": "m",
    "strcory": "m",
    "strocnx": "m",
    "strocny": "m",
    "strintx": "m",
    "strinty": "m",
    "taubx": "m",
    "tauby": "m",
    "strairxE": "x", "strairyE": "x", "strairxN": "x", "strairyN": "x",
    "strtltxE": "x", "strtltyE": "x", "strtltxN": "x", "strtltyN": "x",
    "strcorxE": "x", "strcoryE": "x", "strcorxN": "x", "strcoryN": "x",
    "strocnxE": "x", "strocnyE": "x", "strocnxN": "x", "strocnyN": "x",
    "strintxE": "x", "strintyE": "x", "strintxN": "x", "strintyN": "x",
    "taubxE": "x", "taubyE": "x", "taubxN": "x", "taubyN": "x",
    "strength": "m",
    "vort": "m",
    "divu": "m",
    "shear": "m",
    "sig1": "m",
    "sig2": "m",
    "sigP": "m",
    "dvidtt": "m",
    "dvidtd": "m",
    "daidtt": "m",
    "daidtd": "m",
    "dagedtt": "m",
    "dagedtd": "m",
    "mlt_onset": "m",
    "frz_onset": "m",
    "iage": "m",
    "FY": "m",
    "hisnap": "m",
    "aisnap": "m",
    "trsig": "m",
    "icepresent": "m",
    "fsurf_ai": "m",
    "fcondtop_ai": "m",
    "fmeltt_ai": "m",
    "a11": "x", "a12": "x", "e11": "x", "e12": "x", "e22": "x",
    "s11": "x", "s12": "x", "s22": "x",
    "yieldstress11": "x", "yieldstress12": "x", "yieldstress22": "x",
    "aicen": "x",
    "vicen": "x",
    "vsnon": "x",
    "snowfracn": "x",
    "fsurfn_ai": "x",
    "fcondtopn_ai": "x",
    "fmelttn_ai": "x",
    "flatn_ai": "x",
    "fsensn_ai": "x",
    "keffn_top": "x",
    "siitdconc": "x",
    "siitdsnconc": "x",
    "siitdsnthick": "x",
    "siitdthick": "x",
    "Tinz": "x",
    "Sinz": "x",
    "Tsnz": "x",
}

# CMIP two-dimensional fields default to off in ice_history_shared.F90.
_F.update({name: "x" for name in (
    "sithick", "sisnthick", "siage", "siconc", "sisnconc", "sisnmass",
    "sitemptop", "sitempsnic", "sitempbot", "sivol", "sispeed",
    "siu", "siv", "sidmasstranx", "sidmasstrany", "sistrxdtop",
    "sistrydtop", "sistrxubot", "sistryubot", "sicompstren", "sisali",
    "sihc", "sisnhc", "sidconcth", "sidconcdyn", "sidmassth",
    "sidmassdyn", "sidmassgrowthwat", "sidmassgrowthbot", "simass",
    "sisaltmass", "sidmassgrowthsi", "sidmassevapsubl", "sisndmasssubl",
    "sidmassmelttop", "sidmassmeltbot", "sidmassmeltlat",
    "sisndmasssnf", "sisndmassmelt", "sisndmassdyn", "sisndmasssi",
    "siflswdtop", "siflswutop", "siflswdbot", "sifllwdtop",
    "sifllwutop", "siflsenstop", "siflsensbot", "sifllattop",
    "siflcondtop", "siflcondbot", "sipr", "sifb", "siflsaltbot",
    "siflfwbot", "siflfwdrain", "siforcetiltx", "siforcetilty",
    "siforcecoriolx", "siforcecorioly", "siforceintstrx",
    "siforceintstry", "sidragbot", "sidragtop", "sistressave",
    "sistressmax", "sidivvel", "sishearvel", "sitimefrac",
)})


def init_hist(dt):
    """Initialize history files.

    authors Tony Craig, NCAR
            Elizabeth C. Hunke, LANL
            C.M. Bitz, UW
            Bruce P. Briegleb, NCAR
            William H. Lipscomb, LANL
    """
    subname = "(init_hist)"
    ice_history_shared.reset_history()

    # histfreq options ('1','h','d','m','y')
    # nstreams is initialized to zero in init_calendar.
    nstreams = 0
    histfreq = ice_calendar.histfreq
    max_nstrm = int(ice_domain_size.max_nstrm)
    for ns in range(max_nstrm):
        freq = str(histfreq[ns]).strip()
        if freq in ("1", "h", "d", "m", "y", "n"):
            nstreams = nstreams + 1
            if ns >= 1 and str(histfreq[ns - 1]).strip() == "x":
                raise ValueError(
                    subname + " ERROR: histfreq all non x must be at start of array"
                )
        elif freq != "x":
            ice_fileunits.nu_diag.write(f"{subname} ns,histfreq = {ns + 1} {freq}\n")
            raise ValueError(subname + " ERROR: histfreq contains illegal element")
    if nstreams == 0:
        ice_fileunits.nu_diag.write(subname + " WARNING: No history output\n")
    for ns1 in range(nstreams):
        for ns2 in range(nstreams):
            if str(histfreq[ns1]).strip() == str(histfreq[ns2]).strip() and ns1 != ns2:
                raise ValueError(subname + " ERROR: histfreq elements must be unique")
    ice_calendar.nstreams = nstreams

    # Turn on one-time grid output file
    if bool(ice_init.grid_outfile):
        nstreams = nstreams + 1
        if nstreams > max_nstrm:
            raise ValueError(subname + " ERROR: no room for the one-time grid file")
        ice_calendar.nstreams = nstreams
        ice_calendar.histfreq[nstreams - 1] = "g"
        ice_history_shared.hist_avg[nstreams - 1] = False
        ice_fileunits.nu_diag.write(subname + " Writing one-time grid file\n")

    ice_history_shared.note_nml(subname, "icefields_nml")
    ice_history_shared.apply_history_flags(_F)
    if bool(icepack_parameters.update_ocn_f):
        for name in ("siflsaltbot", "siflfwbot"):
            if _F[name][:1] != "x":
                raise NotImplementedError(
                    name + " with update_ocn_f=true uses an unassigned dfresh "
                    "in the Fortran history diagnostic")
    if int(ice_init.kdyn) != 2:
        for name in ("a11", "a12", "e11", "e12", "e22", "s11", "s12", "s22",
                     "yieldstress11", "yieldstress12", "yieldstress22"):
            _F[name] = "x"
    if _F["fmeltt_ai"][:1] != "x":
        _F["fmelttn_ai"] = _F["fmeltt_ai"]
    if str(ice_init.grid_ice).strip() in ("C", "CD"):
        for name in ("uvel", "vvel", "icespd", "icedir", "strairx", "strairy",
                     "strtltx", "strtlty", "strcorx", "strcory", "strocnx",
                     "strocny", "strintx", "strinty", "taubx", "tauby"):
            _F[name + "E"] = _F[name]
            _F[name + "N"] = _F[name]
    if not bool(icepack_tracers.tr_iage):
        _F["iage"] = "x"
        _F["dagedtt"] = "x"
        _F["dagedtd"] = "x"
    if not bool(icepack_tracers.tr_FY):
        _F["FY"] = "x"

    # these must be output at the same frequency because of
    # cos(zenith angle) averaging
    if _F["albice"][:1] != "x" and _F["albsni"][:1] != "x":
        _F["albice"] = _F["albsni"]
    if _F["albsno"][:1] != "x":
        _F["albsno"] = _F["albice"]
    if _F["albpnd"][:1] != "x":
        _F["albpnd"] = _F["albice"]
    if _F["coszen"][:1] != "x" and _F["albice"][:1] != "x":
        _F["coszen"] = _F["albice"]
    if _F["coszen"][:1] != "x" and _F["albsni"][:1] != "x":
        _F["coszen"] = _F["albsni"]

    if nstreams == 0:
        return

    # ncat_hist = ncat. nzilyr = nilyr. nzblyr = nblyr+2. nzalyr = nblyr+4.
    ice_history_shared.set_history_dimensions()

    queried = icepack_parameters.icepack_query_parameters(
        "secday", "Tffresh", "rhofresh", "puny", "rad_to_deg",
        "awtvdr", "awtidr", "awtvdf", "awtidf",
    )
    secday = queried["secday"]
    tffresh = queried["Tffresh"]
    rhofresh = queried["rhofresh"]
    water = mps_to_cmpdy / rhofresh
    per_step = mps_to_cmpdy / dbl_kind(dt)
    strain = secday * c100
    age = c1 / (secday * dbl_kind(ice_calendar.days_per_year))

    def state(name):
        def gather():
            return getattr(ice_state, name)
        return gather

    def flux(name):
        def gather():
            return getattr(ice_flux, name)
        return gather

    def times(left, right):
        def gather():
            return left() * right()
        return gather

    def speed(u_name, v_name, module):
        def gather():
            u = getattr(module, u_name)
            v = getattr(module, v_name)
            return jnp.sqrt(u * u + v * v)
        return gather

    def direction(u_name, v_name, module):
        # atan2 is applied only when the vector is larger than puny.
        # The 180 degree offset is applied on every cell, matching
        # the Fortran statement that sits outside that test.
        def gather():
            u = getattr(module, u_name)
            v = getattr(module, v_name)
            puny = queried["puny"]
            rad_to_deg = queried["rad_to_deg"]
            moving = (jnp.abs(u) > puny) | (jnp.abs(v) > puny)
            work = jnp.where(moving, jnp.arctan2(u, v) * rad_to_deg, c0) + c180
            return work
        return gather

    def sice():
        # volume-weighted layers are stored consecutively from nt_sice.
        # The written value is their unweighted mean.
        nt = int(icepack_tracers.nt_sice)
        nilyr = int(ice_domain_size.nilyr)
        work = jnp.zeros(ice_state.aice.shape, dtype=dbl_kind)
        if nt < 0 or nilyr < 1:
            return work
        for k in range(nilyr):
            work = work + ice_state.trcr[:, :, nt + k]
        return work / dbl_kind(nilyr)

    def tracer(index_name):
        def gather():
            nt = int(getattr(icepack_tracers, index_name))
            if nt < 0:
                return jnp.zeros(ice_state.aice.shape, dtype=dbl_kind)
            return ice_state.trcr[:, :, nt]
        return gather

    def ice_present():
        puny = queried["puny"]
        return jnp.where(ice_state.aice > puny, c1, c0)

    def ocndir():
        # going to, folded into [0, 360), unlike icedir and atmdir
        puny = queried["puny"]
        rad_to_deg = queried["rad_to_deg"]
        u = ice_flux.uocn
        v = ice_flux.vocn
        moving = (jnp.abs(u) > puny) | (jnp.abs(v) > puny)
        work = jnp.where(moving, jnp.arctan2(u, v) * rad_to_deg, c0)
        return jnp.where(work < c0, work + c360, work)

    def albsni():
        return (queried["awtvdr"] * ice_flux.alvdr_ai
                + queried["awtidr"] * ice_flux.alidr_ai
                + queried["awtvdf"] * ice_flux.alvdf_ai
                + queried["awtidf"] * ice_flux.alidf_ai)

    def trsig():
        if int(ice_init.kdyn) == 2:  # for EAP dynamics different time of output
            return ice_state.strength
        return p25 * (ice_flux.stressp_1 + ice_flux.stressp_2
                      + ice_flux.stressp_3 + ice_flux.stressp_4)

    def principal(index):
        # compute sig1 and sig2
        def gather():
            if str(ice_init.grid_ice).strip() == "B":
                stresses = (ice_flux.stressp_1, ice_flux.stressm_1, ice_flux.stress12_1)
            else:
                stresses = (ice_flux.stresspT, ice_flux.stressmT, ice_flux.stress12T)
            sig = ice_dyn_shared.principal_stress(*stresses, ice_state.strength)
            return sig[index]
        return gather

    if str(ice_init.grid_ice).strip() == "B":
        description = ", on U grid  (NE corner values)"
    else:
        description = ", on T grid"

    def add(name, unit, desc, comment, cona, conb, gather, snapshot=False, rank="2D", avg_ice_present="none", mask_ice_free_points=False, flag=None, grid="T"):
        # flag is the f_<flag> namelist name when it differs from the
        # written name (f_fswfac -> scale_factor, f_icepresent -> ice_present).
        # call define_hist_field
        ice_history_shared.define_hist_field(
            name, unit, desc, comment, cona, conb, _F[flag or name], gather, snapshot,
            rank=rank, avg_ice_present=avg_ice_present,
            mask_ice_free_points=mask_ice_free_points, grid=grid,
        )

    aice = state("aice")
    aice_init = state("aice_init")

    add("hi", "m", "grid cell mean ice thickness", "ice volume per unit grid cell area", c1, c0, state("vice"))
    add("hs", "m", "grid cell mean snow thickness", "snow volume per unit grid cell area", c1, c0, state("vsno"))
    add("snowfrac", "1", "snow fraction of sea ice", "none", c1, c0, flux("snowfrac"))
    add("Tsfc", "C", "snow/ice surface temperature", "averaged with Tf if no ice is present", c1, c0, tracer("nt_Tsfc"))
    add("aice", "1", "ice area  (aggregate)", "none", c1, c0, aice)
    add("aice_init", "1", "ice area  (aggregate) initial", "none", c1, c0, aice_init)
    add("uvel", "m/s", "ice velocity (x)", "positive is x direction on U grid", c1, c0, state("uvel"), grid="U")
    add("vvel", "m/s", "ice velocity (y)", "positive is y direction on U grid", c1, c0, state("vvel"), grid="U")
    add("icespd", "m/s", "sea ice speed", "vector magnitude", c1, c0, speed("uvel", "vvel", ice_state), grid="U")
    add("icedir", "deg", "sea ice direction", "vector direction - coming from", c1, c0, direction("uvel", "vvel", ice_state), grid="U")
    for grid in ("E", "N"):
        add("uvel" + grid, "m/s", "ice velocity (x)", "positive is x direction on " + grid + " grid",
            c1, c0, state("uvel" + grid), grid=grid)
        add("vvel" + grid, "m/s", "ice velocity (y)", "positive is y direction on " + grid + " grid",
            c1, c0, state("vvel" + grid), grid=grid)
        add("icespd" + grid, "m/s", "sea ice speed", "vector magnitude",
            c1, c0, speed("uvel" + grid, "vvel" + grid, ice_state), grid=grid)
        add("icedir" + grid, "deg", "sea ice direction", "vector direction - coming from",
            c1, c0, direction("uvel" + grid, "vvel" + grid, ice_state), grid=grid)
    add("uatm", "m/s", "atm velocity (x)", "positive is x direction on U grid", c1, c0, flux("uatm"))
    add("vatm", "m/s", "atm velocity (y)", "positive is y direction on U grid", c1, c0, flux("vatm"))
    add("atmspd", "m/s", "atmosphere wind speed", "vector magnitude", c1, c0, speed("uatm", "vatm", ice_flux))
    add("atmdir", "deg", "atmosphere wind direction", "vector direction - coming from", c1, c0, direction("uatm", "vatm", ice_flux))
    add("sice", "ppt", "bulk ice salinity", "none", c1, c0, sice)
    add("fswup", "W/m^2", "upward solar flux", "positive upward", c1, c0, flux("fswup"))
    add("fswdn", "W/m^2", "down solar flux", "positive downward", c1, c0, flux("fsw"))
    add("flwdn", "W/m^2", "down longwave flux", "positive downward", c1, c0, flux("flw"))
    add("snow", "cm/day", "snowfall rate (cpl)", "none", water, c0, flux("fsnow"))
    add("snow_ai", "cm/day", "snowfall rate", "weighted by ice area", water, c0, times(flux("fsnow"), aice_init))
    add("rain", "cm/day", "rainfall rate (cpl)", "none", water, c0, flux("frain"))
    add("rain_ai", "cm/day", "rainfall rate", "weighted by ice area", water, c0, times(flux("frain"), aice_init))
    add("sst", "C", "sea surface temperature", "none", c1, c0, flux("sst"))
    add("sss", "ppt", "sea surface salinity", "none", c1, c0, flux("sss"))
    add("uocn", "m/s", "ocean current (x)", "positive is x direction on U grid", c1, c0, flux("uocn"))
    add("vocn", "m/s", "ocean current (y)", "positive is y direction on U grid", c1, c0, flux("vocn"))
    add("ocnspd", "m/s", "ocean current speed", "vector magnitude", c1, c0, speed("uocn", "vocn", ice_flux))
    add("ocndir", "deg", "ocean current direction", "vector direction - going to", c1, c0, ocndir)
    add("frzmlt", "W/m^2", "freeze/melt potential", "if >0, new ice forms; if <0, ice melts", c1, c0, flux("frzmlt_init"))
    add("scale_factor", "1", "shortwave scaling factor", "ratio of netsw new:old", c1, c0, flux("scale_factor"), flag="fswfac")
    add("fswint_ai", "W/m^2", "shortwave absorbed in ice interior", "does not include surface", c1, c0, flux("fswint_ai"))
    add("fswabs", "W/m^2", "snow/ice/ocn absorbed solar flux (cpl)", "positive downward", c1, c0, flux("fswabs"))
    add("fswabs_ai", "W/m^2", "snow/ice/ocn absorbed solar flux", "weighted by ice area", c1, c0, times(flux("fswabs"), aice))
    add("albsni", "%", "snow/ice broad band albedo", "averaged for coszen>0, weighted by aice", c100, c0, albsni)
    add("alvdr", "%", "visible direct albedo", "scaled (divided) by aice", c100, c0, flux("alvdr"))
    add("alidr", "%", "near IR direct albedo", "scaled (divided) by aice", c100, c0, flux("alidr"))
    add("alvdf", "%", "visible diffuse albedo", "scaled (divided) by aice", c100, c0, flux("alvdf"))
    add("alidf", "%", "near IR diffuse albedo", "scaled (divided) by aice", c100, c0, flux("alidf"))
    add("alvdr_ai", "%", "visible direct albedo", " ", c100, c0, flux("alvdr_ai"))
    add("alidr_ai", "%", "near IR direct albedo", " ", c100, c0, flux("alidr_ai"))
    add("alvdf_ai", "%", "visible diffuse albedo", " ", c100, c0, flux("alvdf_ai"))
    add("alidf_ai", "%", "near IR diffuse albedo", " ", c100, c0, flux("alidf_ai"))
    add("albice", "%", "bare ice albedo", "averaged for coszen>0, weighted by aice", c100, c0, flux("albice"))
    add("albsno", "%", "snow albedo", "averaged for coszen>0, weighted by aice", c100, c0, flux("albsno"))
    add("albpnd", "%", "melt pond albedo", "averaged for coszen>0, weighted by aice", c100, c0, flux("albpnd"))
    add("coszen", "radian", "cosine of the zenith angle", "negative below horizon", c1, c0, flux("coszen"))
    add("flat", "W/m^2", "latent heat flux (cpl)", "positive downward", c1, c0, flux("flat"))
    add("flat_ai", "W/m^2", "latent heat flux", "weighted by ice area", c1, c0, times(flux("flat"), aice))
    add("fsens", "W/m^2", "sensible heat flux (cpl)", "positive downward", c1, c0, flux("fsens"))
    add("fsens_ai", "W/m^2", "sensible heat flux", "weighted by ice area", c1, c0, times(flux("fsens"), aice))
    add("flwup", "W/m^2", "upward longwave flux (cpl)", "positive downward", c1, c0, flux("flwout"))
    add("flwup_ai", "W/m^2", "upward longwave flux", "weighted by ice area", c1, c0, times(flux("flwout"), aice))
    add("evap", "cm/day", "evaporative water flux (cpl)", "none", water, c0, flux("evap"))
    add("evap_ai", "cm/day", "evaporative water flux", "weighted by ice area", water, c0, times(flux("evap"), aice))
    add("Tair", "C", "air temperature", "none", c1, -tffresh, flux("Tair"))
    add("Tref", "C", "2m reference temperature", "none", c1, -tffresh, flux("Tref"))
    add("Qref", "g/kg", "2m reference specific humidity", "none", kg_to_g, c0, flux("Qref"))
    add("congel", "cm/day", "congelation ice growth", "none", per_step, c0, flux("congel"))
    add("frazil", "cm/day", "frazil ice growth", "none", per_step, c0, flux("frazil"))
    add("snoice", "cm/day", "snow-ice formation", "none", per_step, c0, flux("snoice"))
    add("dsnow", "cm/day", "snow formation", "none", per_step, c0, flux("dsnow"))
    add("meltt", "cm/day", "top ice melt", "none", per_step, c0, flux("meltt"))
    add("melts", "cm/day", "top snow melt", "none", per_step, c0, flux("melts"))
    add("meltb", "cm/day", "basal ice melt", "none", per_step, c0, flux("meltb"))
    add("meltl", "cm/day", "lateral ice melt", "none", per_step, c0, flux("meltl"))
    add("fresh", "cm/day", "freshwtr flx ice to ocn (cpl)", "if positive, ocean gains fresh water", water, c0, flux("fresh"))
    add("fresh_ai", "cm/day", "freshwtr flx ice to ocn", "weighted by ice area", water, c0, flux("fresh_ai"))
    add("fsalt", "kg/m^2/s", "salt flux ice to ocn (cpl)", "if positive, ocean gains salt", c1, c0, flux("fsalt"))
    add("fsalt_ai", "kg/m^2/s", "salt flux ice to ocean", "weighted by ice area", c1, c0, flux("fsalt_ai"))
    add("fbot", "W/m^2", "heat flux ice to ocean (fbot)", "if positive, ocean gains heat", c1, c0, flux("fbot"))
    add("fhocn", "W/m^2", "heat flux ice to ocn (cpl)", "if positive, ocean gains heat", c1, c0, flux("fhocn"))
    add("fhocn_ai", "W/m^2", "heat flux ice to ocean (fhocn_ai)", "weighted by ice area", c1, c0, flux("fhocn_ai"))
    add("fswthru", "W/m^2", "SW thru ice to ocean (cpl)", "if positive, ocean gains heat", c1, c0, flux("fswthru"))
    add("fswthru_ai", "W/m^2", "SW flux thru ice to ocean", "weighted by ice area", c1, c0, flux("fswthru_ai"))
    add("strairx", "N/m^2", "atm/ice stress (x)", "positive is x direction on U grid", c1, c0, flux("strairxU"), grid="U")
    add("strairy", "N/m^2", "atm/ice stress (y)", "positive is y direction on U grid", c1, c0, flux("strairyU"), grid="U")
    add("strtltx", "N/m^2", "sea sfc tilt stress (x)", "none", c1, c0, flux("strtltxU"), grid="U")
    add("strtlty", "N/m^2", "sea sfc tilt stress (y)", "none", c1, c0, flux("strtltyU"), grid="U")
    add("strcorx", "N/m^2", "Coriolis stress (x)", "none", c1, c0,
        times(flux("fmU"), state("vvel")), grid="U")
    add("strcory", "N/m^2", "Coriolis stress (y)", "none", -c1, c0,
        times(flux("fmU"), state("uvel")), grid="U")
    add("strocnx", "N/m^2", "ocean/ice stress (x)", "positive is x direction on U grid", c1, c0, flux("strocnxU"), grid="U")
    add("strocny", "N/m^2", "ocean/ice stress (y)", "positive is y direction on U grid", c1, c0, flux("strocnyU"), grid="U")
    add("strintx", "N/m^2", "internal ice stress (x)", "positive is x direction on U grid", c1, c0, flux("strintxU"), grid="U")
    add("strinty", "N/m^2", "internal ice stress (y)", "positive is y direction on U grid", c1, c0, flux("strintyU"), grid="U")
    add("taubx", "N/m^2", "seabed (basal) stress (x)", "positive is x direction on U grid", c1, c0, flux("taubxU"), grid="U")
    add("tauby", "N/m^2", "seabed (basal) stress (y)", "positive is y direction on U grid", c1, c0, flux("taubyU"), grid="U")
    for grid in ("E", "N"):
        for name, desc in (("strairx", "atm/ice stress (x)"),
                           ("strairy", "atm/ice stress (y)"),
                           ("strtltx", "sea sfc tilt stress (x)"),
                           ("strtlty", "sea sfc tilt stress (y)"),
                           ("strocnx", "ocean/ice stress (x)"),
                           ("strocny", "ocean/ice stress (y)"),
                           ("strintx", "internal ice stress (x)"),
                           ("strinty", "internal ice stress (y)"),
                           ("taubx", "seabed (basal) stress (x)"),
                           ("tauby", "seabed (basal) stress (y)")):
            add(name + grid, "N/m^2", desc, "positive on " + grid + " grid",
                c1, c0, flux(name + grid), grid=grid)
        add("strcorx" + grid, "N/m^2", "Coriolis stress (x)", "positive on " + grid + " grid",
            c1, c0, times(flux("fm" + grid), state("vvel" + grid)), grid=grid)
        add("strcory" + grid, "N/m^2", "Coriolis stress (y)", "positive on " + grid + " grid",
            -c1, c0, times(flux("fm" + grid), state("uvel" + grid)), grid=grid)
    add("strength", "N/m", "compressive ice strength", "none", c1, c0, state("strength"))
    # The following fields (divu, shear, vort, sig1, and sig2) will be smeared
    #  if averaged over more than a few days.
    # Snapshots may be more useful (see below).
    add("divu", "%/day", "strain rate (divergence)", "divu is instantaneous, on T grid", strain, c0, state("divu"), snapshot=True)
    add("shear", "%/day", "strain rate (shear)", "shear is instantaneous, on T grid", strain, c0, state("shear"), snapshot=True)
    add("vort", "%/day", "strain rate (vorticity)", "vort is instantaneous, on T grid", strain, c0, state("vort"), snapshot=True)
    add("sig1", "1", "norm. principal stress 1", "sig1 is instantaneous" + description, c1, c0, principal(0), snapshot=True)
    add("sig2", "1", "norm. principal stress 2", "sig2 is instantaneous" + description, c1, c0, principal(1), snapshot=True)
    add("sigP", "N/m", "ice pressure", "sigP is instantaneous" + description, c1, c0, principal(2), snapshot=True)
    add("dvidtt", "cm/day", "volume tendency thermo", "none", mps_to_cmpdy, c0, flux("dvidtt"))
    add("dvidtd", "cm/day", "volume tendency dynamics", "none", mps_to_cmpdy, c0, flux("dvidtd"))
    add("daidtt", "%/day", "area tendency thermo", "none", strain, c0, flux("daidtt"))
    add("daidtd", "%/day", "area tendency dynamics", "none", strain, c0, flux("daidtd"))
    add("dagedtt", "day/day", "age tendency thermo", "excludes time step increment", c1, c0, flux("dagedtt"))
    add("dagedtd", "day/day", "age tendency dynamics", "excludes time step increment", c1, c0, flux("dagedtd"))
    add("mlt_onset", "day of year", "melt onset date", "midyear restart gives erroneous dates", c1, c0, flux("mlt_onset"), snapshot=True)
    add("frz_onset", "day of year", "freeze onset date", "midyear restart gives erroneous dates", c1, c0, flux("frz_onset"), snapshot=True)
    add("hisnap", "m", "ice volume snapshot", "none", c1, c0, state("vice"), snapshot=True)
    add("aisnap", "1", "ice area snapshot", "none", c1, c0, aice, snapshot=True)
    add("trsig", "N/m", "internal stress tensor trace", "ice strength approximation", c1, c0, trsig, snapshot=True)
    add("ice_present", "1", "fraction of time-avg interval that ice is present", "ice extent flag", c1, c0, ice_present, flag="icepresent")
    add("fsurf_ai", "W/m^2", "net surface heat flux", "positive downward, excludes conductive flux, weighted by ice area", c1, c0, times(flux("fsurf"), aice))
    add("fcondtop_ai", "W/m^2", "top surface conductive heat flux", "positive downward, weighted by ice area", c1, c0, times(flux("fcondtop"), aice))
    add("fmeltt_ai", "W/m^2", "net surface heat flux causing melt", "weighted by ice area", c1, c0,
        lambda: jnp.sum(jnp.maximum(ice_flux.fsurfn - ice_flux.fcondtopn, c0) *
                       ice_state.aicen_init, axis=2))
    for name, unit, desc in (
        ("a11", " ", "component a11 of the structure tensor"),
        ("a12", " ", "component a12 of the structure tensor"),
        ("e11", "1/s", "component e11 of the strain rate tensor"),
        ("e12", "1/s", "component e12 of the strain rate tensor"),
        ("e22", "1/s", "component e22 of the strain rate tensor"),
        ("s11", "kg/s^2", "component s11 of the stress tensor"),
        ("s12", "kg/s^2", "component s12 of the stress tensor"),
        ("s22", "kg/s^2", "component s12 of the stress tensor"),
        ("yieldstress11", "kg/s^2", "component 11 of the yieldstress tensor"),
        ("yieldstress12", "kg/s^2", "component 12 of the yieldstress tensor"),
        ("yieldstress22", "kg/s^2", "component 12 of the yieldstress tensor"),
    ):
        add(name, unit, desc, "none", c1, c0,
            lambda field=name: getattr(ice_dyn_eap, field))

    # Tracers
    # iage and FYarea are written as snapshots; they are never accumulated.

    # Ice Age
    add("iage", "years", "sea ice age", "none", age, c0, tracer("nt_iage"), snapshot=True)

    # First Year Ice Area
    add("FYarea", " ", "first-year ice area", "weighted by ice area", c1, c0, tracer("nt_FY"), snapshot=True, flag="FY")

    # CMIP 2D variables. Intensive quantities are accumulated with their
    # ice-area factor, then divided by the area sampled over the interval.
    # The initial/final choice follows the factors in ice_history.F90.
    def cmip(name, unit, desc, gather, average="none", snapshot=False,
             scale=c1, offset=c0, grid="T"):
        add(name, unit, desc, "CMIP sea-ice diagnostic", scale, offset,
            gather, snapshot=snapshot, avg_ice_present=average,
            mask_ice_free_points=average != "none", grid=grid)

    def cmip_density():
        if int(icepack_parameters.ktherm) == 2:
            nq = int(icepack_tracers.nt_qice)
            ns = int(icepack_tracers.nt_sice)
            nz = int(ice_history_shared.nzilyr)
            return ice_history_shared.ice_brine_density(
                ice_state.trcr[:, :, nq:nq + nz],
                ice_state.trcr[:, :, ns:ns + nz], ice_flux.sss)
        shape = ice_state.aice.shape
        return (jnp.full(shape, icepack_parameters.rhoi, dtype=dbl_kind),
                jnp.full(shape, icepack_parameters.rhow, dtype=dbl_kind),
                jnp.full(shape, icepack_parameters.ice_ref_salinity,
                        dtype=dbl_kind))

    rhoi = icepack_parameters.rhoi
    rhos = icepack_parameters.rhos
    puny = queried["puny"]
    area = lambda: ice_state.aice
    initial_area = lambda: ice_state.aice_init

    cmip("sitimefrac", "1", "fraction of time steps with sea ice",
         lambda: (area() > puny).astype(dbl_kind))
    cmip("sithick", "m", "sea-ice thickness", state("vice"), "final")
    cmip("sivol", "m", "sea-ice volume per area", state("vice"))
    cmip("simass", "kg m-2", "sea-ice mass per area",
         lambda: cmip_density()[0] * ice_state.vice)
    cmip("sisaltmass", "kg m-2", "mass of salt in sea-ice per area",
         lambda: cmip_density()[0] * cmip_density()[2] * ice_state.vice)
    cmip("sisali", "ppt", "sea-ice salinity",
         lambda: area() * cmip_density()[2], "final")
    cmip("siconc", "%", "sea-ice area percentage", area, scale=c100)
    cmip("sisnconc", "%", "snow area percentage", flux("snowfrac"), scale=c100)
    cmip("sisnmass", "kg m-2", "snow mass per area",
         lambda: rhos * ice_state.vsno)
    cmip("siage", "s", "age of sea ice",
         lambda: area() * ice_state.trcr[:, :, int(icepack_tracers.nt_iage)], "final")
    cmip("sisnthick", "m", "snow thickness", state("vsno"), "final")
    cmip("sitemptop", "K", "surface temperature of sea ice",
         lambda: area() * ice_state.trcr[:, :, int(icepack_tracers.nt_Tsfc)],
         "final", offset=tffresh)
    # Tsnice is already multiplied by aicen in Icepack.
    cmip("sitempsnic", "K", "temperature at snow-ice interface",
         flux("Tsnice"), "init", offset=tffresh)
    cmip("sitempbot", "K", "temperature at ice-ocean interface",
         lambda: initial_area() * ice_flux.Tbot, "init", offset=tffresh)
    cmip("siu", "m/s", "x-component of sea-ice velocity",
         times(area, state("uvel")), "final", grid="U")
    cmip("siv", "m/s", "y-component of sea-ice velocity",
         times(area, state("vvel")), "final", grid="U")
    cmip("sispeed", "m/s", "sea-ice speed",
         lambda: area() * jnp.hypot(ice_state.uvel, ice_state.vvel),
         "final", grid="U")

    # The transport uses neighboring T-point volume and U-point velocity.
    def mass_transport_x():
        result = jnp.zeros_like(ice_state.aice)
        v = ice_state.vice
        s = ice_state.vsno
        u = ice_state.uvel
        result = result.at[1:-1, 1:-1].set(
            (cmip_density()[0][1:-1, 1:-1] * .5 *
             (v[2:, 1:-1] + v[1:-1, 1:-1])
             + rhos * .5 * (s[2:, 1:-1] + s[1:-1, 1:-1]))
            * ice_grid.dyU[1:-1, 1:-1]
            * .5 * (u[1:-1, :-2] + u[1:-1, 1:-1]))
        return result

    def mass_transport_y():
        result = jnp.zeros_like(ice_state.aice)
        v = ice_state.vice
        s = ice_state.vsno
        w = ice_state.vvel
        result = result.at[1:-1, 1:-1].set(
            (cmip_density()[0][1:-1, 1:-1] * .5 *
             (v[1:-1, 2:] + v[1:-1, 1:-1])
             + rhos * .5 * (s[1:-1, 2:] + s[1:-1, 1:-1]))
            * ice_grid.dxU[1:-1, 1:-1]
            * .5 * (w[:-2, 1:-1] + w[1:-1, 1:-1]))
        return result

    cmip("sidmasstranx", "kg/s", "x-component of sea-ice mass transport",
         mass_transport_x, grid="U")
    cmip("sidmasstrany", "kg/s", "y-component of sea-ice mass transport",
         mass_transport_y, grid="U")
    for name, desc, gather in (
        ("sistrxdtop", "x-component of atmospheric stress on sea ice", flux("strairxU")),
        ("sistrydtop", "y-component of atmospheric stress on sea ice", flux("strairyU")),
        ("sistrxubot", "x-component of ocean stress on sea ice", flux("strocnxU")),
        ("sistryubot", "y-component of ocean stress on sea ice", flux("strocnyU")),
    ):
        cmip(name, "N m-2", desc, times(area, gather), "final", grid="U")
    cmip("sicompstren", "N m-1", "compressive sea-ice strength",
         times(area, state("strength")), "final")
    cmip("sihc", "J m-2", "sea-ice heat content",
         lambda: (jnp.mean(ice_state.trcr[:, :, int(icepack_tracers.nt_qice):
                                      int(icepack_tracers.nt_qice) + ice_history_shared.nzilyr],
                          axis=2) * ice_state.vice))
    cmip("sisnhc", "J m-2", "snow heat content",
         lambda: (jnp.mean(ice_state.trcr[:, :, int(icepack_tracers.nt_qsno):
                                      int(icepack_tracers.nt_qsno) + ice_history_shared.nzslyr],
                          axis=2) * ice_state.vsno))
    cmip("sidconcth", "1/s", "sea-ice area tendency from thermodynamics", flux("daidtt"))
    cmip("sidconcdyn", "1/s", "sea-ice area tendency from dynamics", flux("daidtd"))
    cmip("sidmassth", "kg m-2 s-1", "sea-ice mass tendency from thermodynamics",
         lambda: cmip_density()[0] * ice_flux.dvidtt)
    cmip("sidmassdyn", "kg m-2 s-1", "sea-ice mass tendency from dynamics",
         lambda: cmip_density()[0] * ice_flux.dvidtd)
    for name, desc, value in (
        ("sidmassgrowthwat", "mass growth in supercooled water", flux("frazil")),
        ("sidmassgrowthbot", "mass growth at ice base", flux("congel")),
        ("sidmassgrowthsi", "mass growth from snow-to-ice conversion", flux("snoice")),
    ):
        cmip(name, "kg m-2 s-1", desc,
             lambda value=value: cmip_density()[0] * value() / dt)
    cmip("sisndmasssi", "kg m-2 s-1", "snow loss by snow-to-ice conversion",
         lambda: -rhos * ice_flux.snoice / dt)
    cmip("sidmassevapsubl", "kg m-2 s-1", "ice mass change by evaporation",
         lambda: cmip_density()[0] * ice_flux.evapi)
    for name, desc, value in (
        ("sidmassmelttop", "sea-ice mass loss by surface melting", flux("meltt")),
        ("sidmassmeltbot", "sea-ice mass loss by bottom melting", flux("meltb")),
        ("sidmassmeltlat", "sea-ice mass loss by lateral melting", flux("meltl")),
    ):
        cmip(name, "kg m-2 s-1", desc,
             lambda value=value: -cmip_density()[0] * value() / dt)
    cmip("sisndmasssubl", "kg m-2 s-1", "snow mass change by sublimation", flux("evaps"))
    cmip("sisndmasssnf", "kg m-2 s-1", "snow mass change by snowfall", flux("fsnow"))
    cmip("sisndmassmelt", "kg m-2 s-1", "snow mass loss by melting",
         lambda: -rhos * ice_flux.melts / dt)
    cmip("sisndmassdyn", "kg m-2 s-1", "snow mass tendency from dynamics",
         lambda: rhos * ice_flux.dvsdtd)
    for name, desc, gather, average in (
        ("siflswdtop", "downwelling shortwave flux over sea ice", flux("fsw"), "init"),
        ("siflswdbot", "downwelling shortwave flux under sea ice", flux("fswthru"), "init"),
        ("sifllwdtop", "downwelling longwave flux over sea ice", flux("flw"), "init"),
        ("sifllwutop", "upwelling longwave flux over sea ice", flux("flwout"), "final"),
        ("siflsenstop", "net sensible heat flux over sea ice", flux("fsens"), "final"),
        ("siflsensbot", "net sensible heat flux under sea ice", flux("fhocn"), "final"),
        ("sifllattop", "net latent heat flux over sea ice", flux("flat"), "final"),
        ("siflcondtop", "conductive heat flux at ice surface", flux("fcondtop"), "final"),
        ("siflcondbot", "conductive heat flux at ice base", flux("fcondbot"), "final"),
    ):
        factor = initial_area if average == "init" else area
        # Fortran divides these fields by initial ice area even where the
        # numerator uses the final area (e.g. upwelling longwave flux).
        divisor = "init" if name in (
            "sifllwutop", "siflsenstop", "siflsensbot", "sifllattop",
            "siflcondtop", "siflcondbot") else average
        cmip(name, "W/m2", desc, times(factor, gather), divisor)
    cmip("siflswutop", "W/m2", "upwelling shortwave flux over sea ice",
         flux("fswup"), "init")
    cmip("sipr", "kg m-2 s-1", "rainfall rate over sea ice",
         (times(initial_area, flux("frain")) if icepack_tracers.tr_pond_topo
          else lambda: jnp.zeros_like(ice_state.aice)), "init")
    cmip("sifb", "m", "sea-ice freeboard",
         lambda: jnp.where(area() > puny,
                          jnp.maximum(((cmip_density()[1] - cmip_density()[0])
                                      * ice_state.vice - rhos * ice_state.vsno)
                                     / cmip_density()[1], c0), c0), "final")
    # Fortran adds the frazil correction only when update_ocn_f is false.
    # The update_ocn_f=true branch is rejected above because dfresh has
    # no assignment in the original routine.
    frazil_correction = (int(icepack_parameters.ktherm) == 2 and
                         str(icepack_parameters.cpl_frazil).strip() ==
                         "fresh_ice_correction")
    if str(icepack_parameters.saltflux_option).strip() == "prognostic":
        cmip("siflsaltbot", "kg m-2 s-1", "salt flux from sea ice",
             lambda: jnp.where(area() > puny,
                              area() * (ice_flux.fsalt +
                                        (jnp.mean(ice_state.trcr[:, :,
                                                 int(icepack_tracers.nt_sice):
                                                 int(icepack_tracers.nt_sice) +
                                                 ice_history_shared.nzilyr], axis=2)
                                         * ice_state.vice) * .001 *
                                        (-rhoi * (ice_flux.frazil -
                                                 (ice_flux.frazil_diag if frazil_correction else c0)) / dt)),
                              c0), "final")
    else:
        cmip("siflsaltbot", "kg m-2 s-1", "salt flux from sea ice",
             lambda: jnp.where(area() > puny,
                              area() * (ice_flux.fsalt +
                                        icepack_parameters.ice_ref_salinity * .001
                                        * (-rhoi * (ice_flux.frazil -
                                                   (ice_flux.frazil_diag if frazil_correction else c0)) / dt)),
                              c0), "final")
    cmip("siflfwbot", "kg m-2 s-1", "freshwater flux from sea ice",
         lambda: jnp.where(area() > puny,
                          area() * (ice_flux.fresh - rhoi *
                                    (ice_flux.frazil -
                                     (ice_flux.frazil_diag if frazil_correction else c0)) / dt),
                          c0), "final")
    cmip("siflfwdrain", "kg m-2 s-1", "freshwater flux from ice surface",
         lambda: area() * (ice_flux.melts + ice_flux.meltt), "final")
    cmip("sidragtop", "1", "atmospheric drag coefficient",
         lambda: initial_area() * ice_arrays_column.Cdn_atm, "init")
    cmip("sidragbot", "1", "ocean drag coefficient",
         lambda: initial_area() * ice_arrays_column.Cdn_ocn, "init")
    for name, desc, gather in (
        ("siforcetiltx", "x sea-surface tilt force", flux("strtltxU")),
        ("siforcetilty", "y sea-surface tilt force", flux("strtltyU")),
    ):
        cmip(name, "N m-2", desc, times(initial_area, gather), "init")
    cmip("siforcecoriolx", "N m-2", "x Coriolis force",
         lambda: area() * ice_flux.fmU * ice_state.vvel, "final")
    cmip("siforcecorioly", "N m-2", "y Coriolis force",
         lambda: -area() * ice_flux.fmU * ice_state.uvel, "final")
    cmip("siforceintstrx", "N m-2", "x internal stress force",
         times(area, flux("strintxU")), "final")
    cmip("siforceintstry", "N m-2", "y internal stress force",
         times(area, flux("strintyU")), "final")

    # Fortran writes these four fields as instantaneous snapshots.
    cmip("sistressave", "N m-1", "average normal stress in sea ice",
         lambda: .5 * (principal(0)() + principal(1)()), snapshot=True, grid="U")
    cmip("sistressmax", "N m-1", "maximum shear stress in sea ice",
         lambda: .5 * (principal(0)() - principal(1)()), snapshot=True, grid="U")
    cmip("sidivvel", "s-1", "divergence of ice velocity",
         state("divu"), snapshot=True, grid="U")
    cmip("sishearvel", "s-1", "maximum shear of ice velocity",
         state("shear"), snapshot=True, grid="U")

    # other 2D history variables

    # mechanical redistribution
    ice_history_mechred.init_hist_mechred_2D()

    # melt ponds
    ice_history_pond.init_hist_pond_2D()

    # biogeochemistry
    ice_history_bgc.init_hist_bgc_2D()

    # form drag
    ice_history_drag.init_hist_drag_2D()

    # floe size distribution
    ice_history_fsd.init_hist_fsd_2D()

    # advanced snow physics
    ice_history_snow.init_hist_snow_2D(dt)

    #-----------------------------------------------------------------
    # 3D (category) variables looped separately for ordering
    #-----------------------------------------------------------------
    ncat = ice_history_shared.ncat_hist

    def categories(array):
        def gather():
            return jnp.asarray(array())[:, :, :ncat]
        return gather

    def category_product(left, right):
        def gather():
            return jnp.asarray(left())[:, :, :ncat] * jnp.asarray(right())[:, :, :ncat]
        return gather

    def fmelttn():
        # net surface heat flux that causes melt
        melt = jnp.maximum(
            jnp.asarray(ice_flux.fsurfn)[:, :, :ncat] - jnp.asarray(ice_flux.fcondtopn)[:, :, :ncat],
            c0,
        )
        return melt * jnp.asarray(ice_state.aicen_init)[:, :, :ncat]

    add("aicen", "1", "ice area, categories", "none", c1, c0, categories(state("aicen")), rank="3Dc")
    add("vicen", "m", "ice volume, categories", "none", c1, c0, categories(state("vicen")), rank="3Dc")
    add("vsnon", "m", "snow depth on ice, categories", "volume per unit area of snow", c1, c0, categories(state("vsnon")), rank="3Dc")
    add("snowfracn", "1", "category mean snow fraction", "snow fraction per unit grid cell area", c1, c0, categories(lambda: ice_arrays_column.snowfracn), rank="3Dc")
    add("fsurfn_ai", "W/m^2", "net surface heat flux, categories", "weighted by ice area", c1, c0, category_product(lambda: ice_flux.fsurfn, state("aicen_init")), rank="3Dc")
    add("fcondtopn_ai", "W/m^2", "top sfc conductive heat flux, cat", "weighted by ice area", c1, c0, category_product(lambda: ice_flux.fcondtopn, state("aicen_init")), rank="3Dc")
    add("fmelttn_ai", "W/m^2", "net sfc heat flux causing melt, cat", "weighted by ice area", c1, c0, fmelttn, rank="3Dc")
    add("flatn_ai", "W/m^2", "latent heat flux, category", "weighted by ice area", c1, c0, category_product(lambda: ice_flux.flatn, state("aicen_init")), rank="3Dc")
    add("fsensn_ai", "W/m^2", "sensible heat flux, category", "weighted by ice area", c1, c0, category_product(lambda: ice_flux.fsensn, state("aicen_init")), rank="3Dc")
    add("keffn_top", "W/m^2/K", "effective thermal conductivity of the top ice layer, categories", "multilayer scheme", c1, c0, categories(lambda: ice_flux.keffn_top), rank="3Dc")
    # CMIP 3D
    add("siitdconc", "%", "sea-ice area percentages in ice thickness categories", "percentage of grid cell covered by each ice thickness category", c100, c0, categories(state("aicen")), rank="3Dc", avg_ice_present="none")
    add("siitdsnconc", "%", "snow area percentages in ice thickness categories", "percentage of grid cell covered by snow in each ice thickness category", c100, c0, categories(lambda: ice_arrays_column.snowfracn), rank="3Dc")
    add("siitdsnthick", "m", "snow thickness in ice thickness categories", "actual thickness of snow in each ice thickness category", c1, c0, categories(state("vsnon")), rank="3Dc", avg_ice_present="final")
    add("siitdthick", "m", "sea-ice thickness in ice thickness categories", "actual (floe) thickness of sea ice in each thickness category", c1, c0, categories(state("vicen")), rank="3Dc", avg_ice_present="final")

    # other 3D (category) history variables
    ice_history_mechred.init_hist_mechred_3Dc()
    ice_history_pond.init_hist_pond_3Dc()
    ice_history_bgc.init_hist_bgc_3Dc()
    ice_history_snow.init_hist_snow_3Dc()

    # biogeochemistry vertical profiles, then floe size
    ice_history_bgc.init_hist_bgc_3Db()
    ice_history_bgc.init_hist_bgc_3Da()
    ice_history_fsd.init_hist_fsd_3Df(dt)

    #-----------------------------------------------------------------
    # 4D (categories, vertical) variables must be looped separately
    #-----------------------------------------------------------------
    def tinz():
        nz = ice_history_shared.nzilyr
        nt_q = int(icepack_tracers.nt_qice)
        nt_s = int(icepack_tracers.nt_sice)
        work = jnp.zeros(ice_state.aice.shape + (nz, ncat), dtype=dbl_kind)
        if nt_q < 0 or nt_s < 0:
            return work
        qn = ice_state.trcrn[:, :, nt_q:nt_q + nz, :ncat]
        sn = ice_state.trcrn[:, :, nt_s:nt_s + nz, :ncat]
        temperature, _ = icepack_therm_shared.icepack_ice_temperature_jax(
            qn, sn, int(icepack_parameters.ktherm),
            bool(icepack_therm_shared.l_brine), vars(icepack_parameters))
        return temperature

    def sinz():
        # salinity is copied only where the category has ice
        nz = ice_history_shared.nzilyr
        nt_s = int(icepack_tracers.nt_sice)
        work = jnp.zeros(ice_state.aice.shape + (nz, ncat), dtype=dbl_kind)
        if nt_s < 0:
            return work
        puny = queried["puny"]
        thick = ice_state.vicen[:, :, :ncat] > puny
        slab = ice_state.trcrn[:, :, nt_s : nt_s + nz, :ncat]
        return jnp.where(thick[:, :, None, :], slab, c0)

    def tsnz():
        nz = ice_history_shared.nzslyr
        nt_q = int(icepack_tracers.nt_qsno)
        work = jnp.zeros(ice_state.aice.shape + (nz, ncat), dtype=dbl_kind)
        if nt_q < 0:
            return work
        qn = ice_state.trcrn[:, :, nt_q:nt_q + nz, :ncat]
        if int(icepack_parameters.ktherm) == 2:
            return icepack_mushy_physics.temperature_snow_jax(qn, vars(icepack_parameters))
        return icepack_therm_shared.icepack_snow_temperature_jax(qn, vars(icepack_parameters))

    add("Tinz", "C", "ice internal temperatures on CICE grid", "vertical profile", c1, c0, tinz, rank="4Di")
    add("Sinz", "ppt", "ice internal bulk salinity", "vertical profile", c1, c0, sinz, rank="4Di")
    add("Tsnz", "C", "snow internal temperatures", "vertical profile", c1, c0, tsnz, rank="4Ds")

    # 4D floe size by thickness category
    ice_history_fsd.init_hist_fsd_4Df()
    ice_history_shared.write_history_catalog()

    ice_history_shared.allocate_hist_fields()


@contextmanager
def device_fields(pytree):
    """Lend device arrays to the legacy output catalogs, without copying them.

    This host-only scope must not be traced by JIT. Restore the writable CPU
    input buffers even if output fails; forcing readers still use those buffers.
    """
    saved = []
    try:
        for name, module in (
            ("ice_state", ice_state), ("ice_flux", ice_flux),
            ("ice_flux_bgc", ice_flux_bgc),
            ("ice_arrays_column", ice_arrays_column), ("ice_dyn_eap", ice_dyn_eap),
        ):
            for field, value in pytree["state"][name].items():
                saved.append((module, field, getattr(module, field)))
                setattr(module, field, value)
        for field, value in pytree["constants"]["ice_grid"].items():
            saved.append((ice_grid, field, getattr(ice_grid, field)))
            setattr(ice_grid, field, value)
        for field, value in pytree["parameters"]["icepack_parameters"].items():
            if field == "secday":  # Calendar configuration stays on the host.
                continue
            saved.append((icepack_parameters, field, getattr(icepack_parameters, field)))
            setattr(icepack_parameters, field, value)
        yield
    finally:
        for module, field, value in reversed(saved):
            setattr(module, field, value)


def accum_hist(dt, pytree=None, timing=None):
    """Accumulate average ice quantities or snapshots.

    author:   Elizabeth C. Hunke, LANL

    dt is the thermodynamics timestep. Unit conversions were stored
    when the fields were defined. Call this once per thermodynamics
    step. A stream is written when ice_calendar.write_history[ns] is
    true, and stream 0 is written once when write_ic is true.
    With a pytree, call inside device_fields and return device counter/onset
    resets. Restart output synchronizes the returned state only when needed.
    """
    subname = "(accum_hist)"
    accumulate_started = time.perf_counter()
    nstreams = ice_calendar.nstreams
    if nstreams < 1 or not ice_history_shared.avail_hist_fields:
        nstreams = 0

    # Calendar metadata remains on the host, including inside device_fields.
    secday = icepack_parameters.secday
    if pytree is not None:
        state = dict(pytree["state"])
        flux = dict(state["ice_flux"])

    #---------------------------------------------------------------
    # increment step counter
    #---------------------------------------------------------------
    for ns in range(nstreams):
        if not ice_history_shared.hist_avg[ns]:
            # write snapshots
            for field in ice_history_shared.avail_hist_fields:
                if pytree is None and field.stream == ns and not field.snapshot:
                    ice_history_shared.zero_hist_slot(field)
            ice_history_shared.avgct[ns] = c1
        else:
            # write averages over time histfreq
            ice_history_shared.avgct[ns] = ice_history_shared.avgct[ns] + c1
        if ice_history_shared.avgct[ns] == c1:
            timedbl = (dbl_kind(ice_calendar.timesecs) - dbl_kind(dt)) / secday
            ice_history_shared.time_beg[ns] = timedbl

    if pytree is not None and nstreams:
        ice_history_shared.accumulate_device_fields()
    else:
        for field in ice_history_shared.avail_hist_fields:
            if field.snapshot or nstreams == 0:
                continue
            ice_history_shared.accum_hist_field(field, field.gather())
    history_arrays = [getattr(ice_history_shared, name)
                      for _, name, _ in ice_history_shared._RANKS.values()]
    ice_timers.record_stage(timing, "history.accumulate_gpu", accumulate_started, history_arrays)

    #---------------------------------------------------------------
    # Write output files at prescribed intervals
    #---------------------------------------------------------------
    nstrm = nstreams
    if ice_calendar.write_ic and nstreams > 0:
        nstrm = 1

    wrote_ic = False
    for ns in range(nstrm):
        if ice_calendar.write_history[ns] or ice_calendar.write_ic:
            timedbl = dbl_kind(ice_calendar.timesecs) / secday
            ice_history_shared.time_end[ns] = timedbl
            ice_timers.ice_timer_start(ice_timers.timer_readwrite)  # reading/writing
            ice_history_write.ice_write_hist(ns, timing=timing)
            ice_timers.ice_timer_stop(ice_timers.timer_readwrite)  # reading/writing
            reset_started = time.perf_counter()

            #---------------------------------------------------------------
            # reset to zero
            #------------------------------------------------------------
            if ice_calendar.write_ic:
                wrote_ic = True
            else:
                ice_history_shared.avgct[ns] = c0
                for field in ice_history_shared.avail_hist_fields:
                    if field.stream == ns:
                        ice_history_shared.zero_hist_slot(field)
                for name in ("albcnt", "snwcnt"):
                    if pytree is not None:
                        flux[name] = flux[name].at[:, :, ns].set(c0)
                        setattr(ice_flux, name, flux[name])
                    else:
                        getattr(ice_flux, name)[:, :, ns] = c0
            history_arrays = [getattr(ice_history_shared, name)
                              for _, name, _ in ice_history_shared._RANKS.values()]
            ice_timers.record_stage(timing, "history.reset_gpu", reset_started,
                                    (history_arrays, ice_flux.albcnt, ice_flux.snwcnt))

    reset_started = time.perf_counter()
    if wrote_ic:
        for _rank, (_count, array_name, _axes) in ice_history_shared._RANKS.items():
            array = getattr(ice_history_shared, array_name)
            if array is not None:
                if pytree is not None:
                    setattr(ice_history_shared, array_name, jnp.zeros_like(array))
                else:
                    array[...] = c0
        ice_history_shared.avgct[:] = c0
        for name in ("albcnt", "snwcnt"):
            if pytree is not None:
                flux[name] = jnp.zeros_like(flux[name])
                setattr(ice_flux, name, flux[name])
            else:
                getattr(ice_flux, name)[...] = c0
        ice_calendar.write_ic = False  # write initial condition once at most
        ice_fileunits.nu_diag.write(subname + " wrote initial condition history\n")

    # Reset after history output and before restart output, as in Fortran.
    january = bool(ice_calendar.new_year)
    july = ice_calendar.mmonth == 7 and bool(ice_calendar.new_month)
    if january or july:
        ilo, ihi, jlo, jhi = ice_domain_size.physical_bounds()
        physical = jnp.zeros_like(ice_grid.lmask_n, dtype=bool)
        physical = physical.at[ilo:ihi + 1, jlo:jhi + 1].set(True)
        melt_mask = physical & ((january & ice_grid.lmask_n)
                                | (july & ice_grid.lmask_s))
        freeze_mask = physical & ((january & ice_grid.lmask_s)
                                  | (july & ice_grid.lmask_n))
        for name, mask in (("mlt_onset", melt_mask), ("frz_onset", freeze_mask)):
            if pytree is not None:
                flux[name] = jnp.where(jnp.asarray(mask), c0, flux[name])
                setattr(ice_flux, name, flux[name])
            else:
                getattr(ice_flux, name)[jax.device_get(mask)] = c0
    history_arrays = [getattr(ice_history_shared, name)
                      for _, name, _ in ice_history_shared._RANKS.values()]
    ice_timers.record_stage(timing, "history.reset_gpu", reset_started,
                            (history_arrays, ice_flux.albcnt, ice_flux.snwcnt,
                             ice_flux.mlt_onset, ice_flux.frz_onset))
    if ice_calendar.write_histrest and ice_calendar.write_restart == 1:
        ice_timers.ice_timer_start(ice_timers.timer_readwrite)
        try:
            for ns in range(nstrm):
                if ice_history_shared.hist_avg[ns] and ice_history_shared.avgct[ns] > c0:
                    ice_history_write.ice_write_hist(ns, histrest=True, timing=timing)
        finally:
            ice_timers.ice_timer_stop(ice_timers.timer_readwrite)
    if pytree is not None:
        state["ice_flux"] = flux
        pytree = dict(pytree)
        pytree["state"] = state
    return pytree
