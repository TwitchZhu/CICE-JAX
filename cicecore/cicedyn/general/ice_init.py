"""Parameter and variable initializations.

author for JAX version: Xinyuan Zhu
XZ: converted from F90 to Python

2006 ECH: Added namelist variables, warnings.
          Replaced old default initial ice conditions with 3.14 version.
          Converted to free source form (F90).
2004 WHL: Block structure added
authors: Elizabeth C. Hunke and William H. Lipscomb, LANL
         C. M. Bitz, UW

gx3 configuration read by input_data.

The namelist groups read by standalone ``input_data`` are accepted, including
their checks and the configuration overview. CESMCOUPLED file naming and the
MPI broadcast are not in this port. ``restart`` is set true when ``ice_ic``
is a file.

Call ``init_constants_tracers_namelist`` to run the CICE sequence
``icepack_configure``, ``input_data``, ``input_zbgc``, ``count_tracers``.

``init_state`` and ``set_state_var`` are ported. ``icepack_aggregate`` is
called from ``init_state`` and raises until the later step. ``bound_state``
updates ghosts. The C and CD face velocities are averaged only for those grids.
"""

import copy
import math
import sys
import re

import numpy as np

from cicecore.cicedyn.general import ice_flux
from cicecore.cicedyn.general import ice_state
from cicecore.cicedyn.infrastructure import ice_grid
from cicecore.cicedyn.infrastructure.comm.serial.ice_boundary import (
    global_indices,
    ice_halo_update,
)
from cicecore.shared import ice_arrays_column
from cicecore.shared import ice_domain_size
from cicecore.shared import ice_fileunits
from cicecore.shared import ice_init_column
from cicecore.shared.ice_constants import (
    c0,
    c1,
    c2,
    c3,
    c5,
    c12,
    cm_to_m,
    field_loc_Eface,
    field_loc_Nface,
    field_type_scalar,
    p166,
    p2,
    p3,
    p5,
    p75,
)
from cicecore.shared.ice_kinds_mod import dbl_kind
from icepack.columnphysics import icepack_intfc
from icepack.columnphysics import icepack_itd
from icepack.columnphysics import icepack_parameters
from icepack.columnphysics import icepack_therm_shared
from icepack.columnphysics import icepack_tracers

MAX_NSTRM = 6
itd_min_unset = -999.0

# Fortran defaults from input_data, before ice_in is read.
histfreq = ["x"] * MAX_NSTRM
histfreq[0] = "1"
histfreq[1] = "h"
histfreq[2] = "d"
histfreq[3] = "m"
histfreq[4] = "y"
dumpfreq = ["x"] * MAX_NSTRM
dumpfreq[0] = "y"
namelist_defaults = {
        "days_per_year": 365,
        "use_leap_years": False,
        "year_init": 0,
        "month_init": 1,
        "day_init": 1,
        "sec_init": 0,
        "istep0": 0,
        "dt": 3600.0,
        "npt": 99999,
        "npt_unit": "1",
        "ndtd": 1,
        "diagfreq": 24,
        "diag_type": "stdout",
        "diag_file": "ice_diag.d",
        "numin": 11,  # min allowed unit number
        "numax": 99,  # max allowed unit number
        "debug_model": False,  # debug output
        "debug_model_step": 0,  # debug model after this step number
        "debug_model_ponds": False,  # pond states at debug checkpoints (JAX extension)
        "debug_model_ponds_inputs": False,  # therm1 pond inputs/outputs (JAX extension)
        "debug_model_ponds_i": [-1] * 16,  # paired global 1-based pond diagnostic indices
        "debug_model_ponds_j": [-1] * 16,
        "debug_model_i": -1,  # debug model local i index
        "debug_model_j": -1,  # debug model local j index
        "debug_model_iblk": -1,  # debug model local iblk number
        "debug_model_task": -1,  # debug model local task number
        "timer_stats": False,  # if true, print out detailed timer statistics
        "memory_stats": False,  # if true, print out memory information
        "bfbflag": "off",  # off = optimized. Only off is accepted.
        "runid": "unknown",  # run ID used in CESM and for machine 'bering'
        "version_name": "unknown_version_name",
        "history_format": "cdf1",  # history file format
        "history_root": -99,  # history iotasks, root, stride sets pes for pio
        "history_stride": -99,  # history iotasks, root, stride sets pes for pio
        "history_iotasks": -99,  # history iotasks, root, stride sets pes for pio
        "history_rearranger": "default",  # history rearranger for pio
        "hist_time_axis": "end",  # History file time axis averaging interval position
        "history_dir": "./",  # write to executable dir for default
        "history_file": "iceh",  # history file name prefix
        "history_precision": 4,  # precision of history files
        "history_deflate": 0,  # compression level for netcdf4
        "history_chunksize": [0, 0],  # chunksize for netcdf4
        "write_histrest": True,  # write out history restart files if needed
        "cpl_bgc": False,  # couple bgc thru driver
        "incond_dir": "./",  # write to history dir for default
        "incond_file": "iceh_ic",  # file prefix
        "hist_suffix": ["x"] * MAX_NSTRM,  # appended to 'history_file' in filename when not 'x'
        "print_points": False,
        "print_global": True,
        "histfreq": histfreq,
        "histfreq_n": [1] * MAX_NSTRM,
        "histfreq_base": ["zero"] * MAX_NSTRM,
        "hist_avg": [True] * MAX_NSTRM,
        "dumpfreq": dumpfreq,
        "dumpfreq_n": [1] * MAX_NSTRM,
        "dumpfreq_base": ["init"] * MAX_NSTRM,
        "dump_last": False,
        "restart_dir": "./",
        "restart_file": "iced",
        "restart_ext": False,
        "restart_format": "cdf1",  # restart file format
        "restart_root": -99,  # restart iotasks, root, stride sets pes for pio
        "restart_stride": -99,  # restart iotasks, root, stride sets pes for pio
        "restart_iotasks": -99,  # restart iotasks, root, stride sets pes for pio
        "restart_rearranger": "default",  # restart rearranger for pio
        "restart_deflate": 0,  # compression level for netcdf4
        "restart_chunksize": [0, 0],  # chunksize for netcdf4
        "restart_mod": "none",  # restart modification option
        "lcdf64": False,  # 64 bit offset for netCDF. deprecated, backwards compatibility
        "pointer_file": "ice.restart_file",
        "ice_ic": "default",
        "runtype": "initial",
        "restart": False,
        "use_restart_time": False,
        "write_ic": False,
        "conserv_check": False,
        "latpnt": [90.0, -65.0],
        "lonpnt": [0.0, -45.0],
        "debug_forcing": False,
        "grid_format": "bin",
        "grid_type": "rectangular",
        "grid_file": "unknown_grid_file",
        "grid_ice": "B",
        "grid_atm": "A",
        "grid_ocn": "A",  # underlying atm forcing/coupling grid
        "gridcpl_file": "unknown_gridcpl_file",
        "close_boundaries": False,  # true = set land on edges of grid
        "orca_halogrid": False,  # orca haloed grid - deprecated
        "kmt_type": "file",
        "kmt_file": "unknown_kmt_file",
        "bathymetry_file": "unknown_bathymetry_file",
        "bathymetry_format": "default",
        "use_bathymetry": False,
        "ncat": 0,
        "nfsd": 1,
        "nilyr": 0,
        "nslyr": 0,
        "nblyr": 0,
        "kcatbound": 1,
        "dxrect": 0.0,
        "dyrect": 0.0,
        "lonrefrect": -156.50,
        "latrefrect": 71.35,
        "scale_dxdy": False,
        "dxscale": 1.0,
        "dyscale": 1.0,
        "grid_outfile": False,
        "grid_ice_thrm": "T",
        "grid_ice_dynu": "U",
        "grid_ice_dynv": "U",
        "grid_atm_thrm": "T",
        "grid_atm_dynu": "T",
        "grid_atm_dynv": "T",
        "grid_ocn_thrm": "T",
        "grid_ocn_dynu": "T",
        "grid_ocn_dynv": "T",
        "save_ghte_ghtn": False,
        # extra tracers
        "tr_iage": False,  # ice age
        "restart_age": False,  # ice age restart
        "tr_FY": False,  # ice age
        "restart_FY": False,  # ice age restart
        "tr_lvl": False,  # level ice
        "restart_lvl": False,  # level ice restart
        "tr_pond_lvl": False,  # level-ice melt ponds
        "restart_pond_lvl": False,  # melt ponds restart
        "tr_pond_sealvl": False,  # Sea level melt ponds
        "restart_pond_sealvl": False,  # Sea level melt ponds restart
        "tr_pond_topo": False,  # explicit melt ponds (topographic)
        "restart_pond_topo": False,  # melt ponds restart
        "tr_snow": False,  # advanced snow physics
        "restart_snow": False,  # advanced snow physics restart
        "tr_iso": False,  # isotopes
        "restart_iso": False,  # isotopes restart
        "tr_aero": False,  # aerosols
        "restart_aero": False,  # aerosols restart
        "tr_fsd": False,  # floe size distribution
        "restart_fsd": False,  # floe size distribution restart
        "tr_pond": False,
        "n_iso": 0,
        "n_aero": 0,
        "n_zaero": 0,
        "n_algae": 0,
        "n_doc": 0,
        "n_dic": 0,
        "n_don": 0,
        "n_fed": 0,
        "n_fep": 0,
        "kitd": 1,
        "ktherm": 1,
        "conduct": "bubbly",
        "ksno": 0.3,
        "a_rapid_mode": 0.5e-3,
        "Rac_rapid_mode": 10.0,
        "aspect_rapid_mode": 1.0,
        "dSdt_slow_mode": -1.5e-7,
        "phi_c_slow_mode": 0.05,
        "phi_i_mushy": 0.85,
        "floediam": 300.0,
        "hfrazilmin": 0.05,
        "Tliquidus_max": 0.0,
        "hi_min": 0.01,
        "itd_area_min": itd_min_unset,
        "itd_mass_min": itd_min_unset,
        "tscale_pnd_drain": 10.0,
        "kdyn": 1,
        "ndte": 120,
        "revised_evp": False,
        "evp_algorithm": "standard_2d",
        "elasticDamp": 0.36,
        "brlx": 300.0,
        "arlx": 300.0,
        "ssh_stress": "geostrophic",
        "advection": "remap",
        "coriolis": "latitude",
        "kridge": 1,
        "ktransport": 1,
        "kstrength": 1,
        "krdg_partic": 1,
        "krdg_redist": 1,
        "mu_rdg": 3.0,
        "e_yieldcurve": 2.0,
        "e_plasticpot": 2.0,
        "visc_method": "avg_zeta",
        "deltaminEVP": 1.0e-11,  # minimum delta for viscosities (EVP, Hunke 2001)
        "deltaminVP": 2.0e-9,  # minimum delta for viscosities (VP, Hibler 1979)
        "capping_method": "max",  # method for capping of viscosities (max=Hibler 1979,sum=Kreyscher2000)
        "seabed_stress": False,  # if true, seabed stress for landfast is on
        "seabed_stress_method": "LKD",  # LKD = Lemieux et al 2015, probabilistic = Dupont et al. 2022
        "k1": 7.5,  # 1st free parameter for landfast parameterization
        "k2": 15.0,  # 2nd free parameter (N/m^3) for landfast parametrization
        "alphab": 20.0,  # alphab=Cb factor in Lemieux et al 2015
        "threshold_hw": 30.0,  # max water depth for grounding
        # preconditioner for fgmres: 'ident' (identity), 'diag' (diagonal),
        # 'pgmres' (Jacobi-preconditioned GMRES)
        "precond": "pgmres",
        "dim_fgmres": 50,  # size of fgmres Krylov subspace
        "dim_pgmres": 5,  # size of pgmres Krylov subspace
        "maxits_nonlin": 10,  # max nb of iteration for nonlinear solver
        "maxits_fgmres": 50,  # max nb of iteration for fgmres
        "maxits_pgmres": 5,  # max nb of iteration for pgmres
        "monitor_nonlin": False,  # print nonlinear residual norm
        "monitor_fgmres": False,  # print fgmres residual norm
        "monitor_pgmres": False,  # print pgmres residual norm
        "ortho_type": "mgs",  # orthogonalization procedure 'cgs' or 'mgs'
        "reltol_nonlin": 1.0e-8,  # nonlinear stopping criterion: reltol_nonlin*res(k=0)
        "reltol_fgmres": 1.0e-1,  # fgmres stopping criterion: reltol_fgmres*res(k)
        "reltol_pgmres": 1.0e-6,  # pgmres stopping criterion: reltol_pgmres*res(k)
        "algo_nonlin": "picard",  # nonlinear algorithm: 'picard' (Picard iteration), 'anderson' (Anderson acceleration)
        # fixed point function for Anderson acceleration:
        # 1: g(x) = FMGRES(A(x),b(x)), 2: g(x) = x - A(x)x + b(x)
        "fpfunc_andacc": 1,
        "dim_andacc": 5,  # size of Anderson minimization matrix (number of saved previous residuals)
        "reltol_andacc": 1.0e-6,  # relative tolerance for Anderson acceleration
        "damping_andacc": 0,  # damping factor for Anderson acceleration
        "start_andacc": 0,  # acceleration delay factor (acceleration starts at this iteration)
        "use_mean_vrel": True,  # use mean of previous 2 iterates to compute vrel
        "capping": -9.99e30,
        "Cf": 17.0,
        "Pstar": 2.75e4,
        "Cstar": 20.0,
        "Ktens": 0.0,
        "dyn_area_min": 1.0e-11,
        "dyn_mass_min": 1.0e-10,
        "yield_curve": "ellipse",
        "shortwave": "ccsm3",
        "albedo_type": "ccsm3",
        "snw_ssp_table": "test",
        "albicev": 0.78,
        "albicei": 0.36,
        "albsnowv": 0.98,
        "albsnowi": 0.70,
        "ahmax": 0.3,
        "R_ice": 0.0,
        "R_pnd": 0.0,
        "R_snw": 1.5,
        "sw_redist": False,
        "sw_frac": 0.9,
        "sw_dtemp": 0.02,
        "dT_mlt": 1.5,
        "rsnw_mlt": 1500.0,
        "kalg": 0.60,
        "hs0": 0.03,
        "dpscale": 1.0,
        "frzpnd": "cesm",
        "rfracmin": 0.15,
        "rfracmax": 0.85,
        "pndaspect": 0.8,
        "hs1": 0.03,
        "hp1": 0.01,
        "apnd_sl": 0.27,
        "snwredist": "none",
        "snwgrain": False,
        "rsnw_fall": 100.0,
        "rsnw_tmax": 1500.0,
        "rhosnew": 100.0,
        "rhosmin": 100.0,
        "rhosmax": 450.0,
        "snwlvlfac": 0.3,
        "windmin": 10.0,
        "drhosdwind": 27.3,
        "use_smliq_pnd": False,
        "snw_aging_table": "test",  # snow aging lookup table
        "snw_filename": "unknown",  # snowtable filename
        "snw_tau_fname": "unknown",  # snowtable file tau fieldname
        "snw_kappa_fname": "unknown",  # snowtable file kappa fieldname
        "snw_drdt0_fname": "unknown",  # snowtable file drdt0 fieldname
        "snw_rhos_fname": "unknown",  # snowtable file rhos fieldname
        "snw_Tgrd_fname": "unknown",  # snowtable file Tgrd fieldname
        "snw_T_fname": "unknown",  # snowtable file T fieldname
        "formdrag": False,
        "atmbndy": "similarity",
        "calc_strair": True,
        "calc_Tsfc": True,
        "highfreq": False,
        "natmiter": 5,
        "atmiter_conv": 0.0,
        "calc_dragio": False,
        "ustar_min": 0.005,
        "emissivity": 0.985,
        "iceruf": 0.0005,
        "iceruf_ocn": 0.03,
        "fbot_xfer_type": "constant",
        "update_ocn_f": False,
        "l_mpond_fresh": False,
        "tfrz_option": "mushy",
        "saltflux_option": "constant",
        "ice_ref_salinity": 4.0,
        "cpl_frazil": "fresh_ice_correction",
        "congel_freeze": "two-step",
        "oceanmixed_ice": False,
        "restore_ice": False,
        "restore_ocn": False,
        "trestore": 90.0,
        "precip_units": "mks",
        "default_season": "winter",
        "wave_spec_type": "none",  # type of wave spectrum forcing
        "nfreq": 25,  # number of wave frequencies
        "wave_height_type": "internal",  # type of wave height forcing
        "wave_spec": False,
        "wave_spec_file": " ",  # wave forcing file name
        "bgc_data_type": "default",
        "fe_data_type": "default",
        "bgc_data_dir": "unknown_bgc_data_dir",
        "oceanmixed_file": "unknown_oceanmixed_file",  # ocean forcing data
        "atm_data_type": "default",
        "ocn_data_type": "default",
        "ice_data_type": "default",
        "ice_data_conc": "default",
        "ice_data_dist": "default",
        "fyear_init": 1900,
        "ycycle": 1,
        "restart_coszen": False,
        "atm_data_dir": " ",
        "ocn_data_dir": "unknown_ocn_data_dir",
        "atm_data_format": "bin",
        "ocn_data_format": "bin",
        "rotate_wind": True,
        "atm_data_version": "_undef",
        "semi_implicit_Tsfc": False,
        "vapor_flux_correction": False,
    }


def _fortran_g(value, width=13, digits=4):
    return f"{float(value):{width}.{digits}g}"


def _fortran_d(value, width=13, digits=6):
    number = float(value)
    sign = "-" if number < 0.0 else ""
    magnitude = abs(number)
    if magnitude == 0.0:
        text = f"0.{'0' * digits}D+00"
    else:
        exponent = math.floor(math.log10(magnitude)) + 1
        mantissa = magnitude / 10.0 ** exponent
        text = f"{sign}{mantissa:.{digits}f}D{exponent:+03d}"
    return f"{text:>{width}}"


def write_model_configuration(state, errors):
    """Write the input_data overview of the model configuration."""
    out = ice_fileunits.nu_diag
    puny = float(icepack_parameters.puny)
    tocnfrz = float(icepack_parameters.Tocnfrz)

    def line(text=" "):
        out.write(text + "\n")

    def logical(value):
        return f"{'T' if value else 'F':>6}"

    def f1020(label, value, note=""):
        text = f"{label}{' ' * 8}{int(value):6d}"
        if note:
            text += " " + note
        line(text)

    def f1021(label, value):
        line(f"{label} {int(value):6d}")

    def f1022(label, value):
        line(f"{label} {int(value):12d}")

    def f1002(label, value, note=""):
        text = f"{label}{' ' * 5}{float(value):9.2f}"
        if note:
            text += " " + note
        line(text)

    def f1000(label, value, note=""):
        text = f"{label} {float(value):13.6f}"
        if note:
            text += " " + note
        line(text)

    def f1003(label, value, note=""):
        text = f"{label} {_fortran_g(value)}"
        if note:
            text += " " + note
        line(text)

    def f1004(label, first, second):
        line(f"{label} {_fortran_g(first)}{_fortran_g(second)}")

    def f1009(label, value, note=""):
        text = f"{label} {_fortran_d(value)}"
        if note:
            text += " " + note
        line(text)

    def f1010(label, value, note=""):
        text = f"{label}{' ' * 8}{logical(value)}"
        if note:
            text += " " + note
        line(text)

    def f1011(label, value):
        line(f"{label} {logical(value)}")

    def f1030(label, value="", note=""):
        text = f"{label}{str(value)[:14]:>14}"
        if note:
            text += " " + note
        line(text)

    def f1031(label, value, note=""):
        line(f"{label} {value}{note}")

    def f_ints(label, values):
        body = "".join(f"{int(item):6d}" for item in values)
        line(f"{label} {body}")

    def f_chars(label, values):
        body = "".join(f"{str(item):>6}" for item in values)
        line(f"{label} {body}")

    def f_logicals(label, values):
        body = "".join(f"{'T' if item else 'F':>3}" for item in values)
        line(f"{label} {body}")

    streams = slice(0, MAX_NSTRM - 1)
    line(" Overview of model configuration with relevant parameters")
    line("=========================================================")
    line("For details, compare namelist output below with the")
    line("Case Settings section in the model documentation.")
    line(" ")
    line(" Calendar")
    line("--------------------------------")
    f1020(" days_per_year    = ", state["days_per_year"], " : number of days in a model year")
    if state["use_leap_years"]:
        note = " : leap days are included"
    else:
        note = " : leap days are not included"
    f1010(" use_leap_years   = ", state["use_leap_years"], note)
    f1002(" dt               = ", state["dt"], " : model time step")

    line(" ")
    line(" Grid, Discretization")
    line("--------------------------------")
    f1030(" grid_format      = ", state["grid_format"].strip())
    note = " "
    if state["grid_type"].strip() == "rectangular":
        note = " : internally defined, rectangular grid"
    if state["grid_type"].strip() == "regional":
        note = " : grid file, regional grid"
    if state["grid_type"].strip() == "displaced_pole":
        note = " : grid file with rotated north pole"
    if state["grid_type"].strip() == "tripole":
        note = " : grid file with northern hemisphere zipper"
    if state["grid_type"].strip() == "latlon":
        note = " : cesm latlon domain file"
    f1030(" grid_type        = ", state["grid_type"].strip(), note)
    f1030(" grid_ice         = ", state["grid_ice"].strip())
    f1030("   grid_ice_thrm  = ", state["grid_ice_thrm"].strip())
    f1030("   grid_ice_dynu  = ", state["grid_ice_dynu"].strip())
    f1030("   grid_ice_dynv  = ", state["grid_ice_dynv"].strip())
    f1030(" grid_atm         = ", state["grid_atm"].strip())
    f1030("   grid_atm_thrm  = ", state["grid_atm_thrm"].strip())
    f1030("   grid_atm_dynu  = ", state["grid_atm_dynu"].strip())
    f1030("   grid_atm_dynv  = ", state["grid_atm_dynv"].strip())
    f1030(" grid_ocn         = ", state["grid_ocn"].strip())
    f1030("   grid_ocn_thrm  = ", state["grid_ocn_thrm"].strip())
    f1030("   grid_ocn_dynu  = ", state["grid_ocn_dynu"].strip())
    f1030("   grid_ocn_dynv  = ", state["grid_ocn_dynv"].strip())
    f1030(" kmt_type         = ", state["kmt_type"].strip())
    if state["grid_type"].strip() == "rectangular":
        f1004("lon/lat refrect   = ", state["lonrefrect"], state["latrefrect"])
        f1004("dx/dy rect (cm)   = ", state["dxrect"], state["dyrect"])
        f1010("scale_dxdy        = ", state["scale_dxdy"])
        f1004("dx/dy scale       = ", state["dxscale"], state["dyscale"])
    else:
        if state["use_bathymetry"]:
            note = " : bathymetric input data is used"
        else:
            note = " : bathymetric input data is not used"
        f1010(" use_bathymetry   = ", state["use_bathymetry"], note)
        f1030(" bathymetry_format= ", state["bathymetry_format"].strip())
    f1020(" nilyr            = ", state["nilyr"], " : number of ice layers (equal thickness)")
    f1020(" nslyr            = ", state["nslyr"], " : number of snow layers (equal thickness)")
    f1020(" nblyr            = ", state["nblyr"], " : number of bio layers (equal thickness)")
    if state["shortwave"][:4] == "dEdd":
        line("dEdd interior and sfc scattering layers are used in both ice, snow (unequal)")
    f1020(" ncat             = ", state["ncat"], " : number of ice categories")
    if state["kcatbound"] == 0:
        note = " : original ITD category bounds"
    elif state["kcatbound"] == 1:
        note = " : round-number category bounds"
    elif state["kcatbound"] == 2:
        note = " : WMO standard ITD categories"
    elif state["kcatbound"] == -1:
        note = " : one thickness category"
    else:
        note = " : unknown value"
    f1020(" kcatbound        = ", state["kcatbound"], note)
    if state["kitd"] == 0:
        note = " : delta function ITD approx"
    else:
        note = " : linear remapping ITD approx"
    f1020(" kitd             = ", state["kitd"], note)
    if state["tr_fsd"]:
        note = " : floe size distribution is enabled"
    else:
        note = " : floe size distribution is disabled"
    f1010(" tr_fsd           = ", state["tr_fsd"], note)
    f1020(" nfsd             = ", state["nfsd"], " : number of floe size categories")

    line(" ")
    line(" Horizontal Dynamics")
    line("--------------------------------")
    if state["kdyn"] == 1:
        note = " : elastic-viscous-plastic dynamics"
    elif state["kdyn"] == 2:
        note = " : elastic-anisotropic-plastic dynamics"
    elif state["kdyn"] == 3:
        note = " : viscous-plastic dynamics"
    elif state["kdyn"] < 1:
        note = " : dynamics disabled"
    else:
        note = " : unknown value"
    f1020(" kdyn             = ", state["kdyn"], note)
    f1003(
        " dyn_area_min     = ",
        state["dyn_area_min"],
        " : min ice area concentration to activate dynamics",
    )
    f1003(
        " dyn_mass_min     = ",
        state["dyn_mass_min"],
        " : min ice mass to activate dynamics (kg/m2)",
    )
    if state["kdyn"] >= 1:
        if state["kdyn"] in (1, 2):
            if state["revised_evp"]:
                note = " : revised EVP formulation used"
                f1002(" arlx             = ", state["arlx"], " : stress equation factor alpha")
                f1002(" brlx             = ", state["brlx"], " : stress equation factor beta")
            else:
                note = " : revised EVP formulation not used"
            f1010(" revised_evp      = ", state["revised_evp"], note)
            if state["evp_algorithm"] == "standard_2d":
                note = " : standard 2d EVP solver"
            elif state["evp_algorithm"] == "shared_mem_1d":
                note = " : vectorized 1d EVP solver"
            else:
                note = " : unknown value"
            f1031(" evp_algorithm    = ", state["evp_algorithm"].strip(), note)
            f1020(
                " ndtd             = ",
                state["ndtd"],
                " : number of dynamics/advection/ridging/steps per thermo timestep",
            )
            f1020(" ndte             = ", state["ndte"], " : number of EVP or EAP subcycles")
        if state["kdyn"] in (1, 3):
            f1030(" yield_curve      = ", state["yield_curve"].strip(), " : yield curve")
            if state["yield_curve"].strip() == "ellipse":
                f1002(
                    " e_yieldcurve     = ",
                    state["e_yieldcurve"],
                    " : aspect ratio of yield curve",
                )
            f1002(
                " e_plasticpot     = ",
                state["e_plasticpot"],
                " : aspect ratio of plastic potential",
            )
        if state["kdyn"] == 1:
            f1003(" deltamin     = ", state["deltaminEVP"], " : minimum delta for viscosities")
            f1030(
                " capping_meth = ",
                state["capping_method"].strip(),
                " : capping method for viscosities",
            )
        elif state["kdyn"] == 3:
            f1003(" deltamin     = ", state["deltaminVP"], " : minimum delta for viscosities")
            f1030(
                " capping_meth = ",
                state["capping_method"].strip(),
                " : capping method for viscosities",
            )
        #write(nu_diag,1002) ' capping      = ', capping, ' : capping value for viscosities'
        f1002(
            " elasticDamp  = ",
            state["elasticDamp"],
            " : coefficient for calculating the parameter E",
        )
        if state["coriolis"].strip() == "latitude":
            note = " : latitude-dependent Coriolis parameter"
        elif state["coriolis"].strip() == "contant":
            note = " : = 1.46e-4/s"
        elif state["coriolis"].strip() == "zero":
            note = " : = 0.0"
        else:
            note = ": unknown value"
        f1030(" coriolis         = ", state["coriolis"].strip(), note)
        if state["ssh_stress"].strip() == "geostrophic":
            note = " : from ocean velocity"
        elif state["ssh_stress"].strip() == "coupled":
            note = " : from coupled sea surface height gradients"
        else:
            note = " : unknown value"
        f1030(" ssh_stress       = ", state["ssh_stress"].strip(), note)
        if state["advection"].strip() == "remap":
            note = " : linear remapping advection"
        elif state["advection"].strip() == "upwind":
            note = " : donor cell (upwind) advection"
        elif state["advection"].strip() == "none":
            note = " : advection disabled by ktransport namelist"
        else:
            note = " : unknown value"
        f1030(" advection        = ", state["advection"].strip(), note)
        if state["seabed_stress"]:
            note = " : use seabed stress parameterization for landfast ice"
        else:
            note = " : no seabed stress parameterization"
        f1010(" seabed_stress    = ", state["seabed_stress"], note)
        if state["seabed_stress"]:
            f1030(" seabed method    = ", state["seabed_stress_method"].strip())
            if state["seabed_stress_method"] == "LKD":
                f1002(" k1               = ", state["k1"], " : free parameter for landfast ice")
                f1002(" k2               = ", state["k2"], " : free parameter for landfast ice")
                f1002(" alphab           = ", state["alphab"], " : factor for landfast ice")
                f1002(
                    " threshold_hw     = ",
                    state["threshold_hw"],
                    " : max water depth for grounding ice",
                )
            elif state["seabed_stress_method"] == "probabilistic":
                f1002(" alphab           = ", state["alphab"], " : factor for landfast ice")
        if state["grid_ice"] in ("C", "CD"):
            f1030(
                " visc_method= ",
                state["visc_method"].strip(),
                " : viscosities method (U point)",
            )
        f1002(" Ktens            = ", state["Ktens"], " : tensile strength factor")
        if state["kdyn"] == 3:
            f1020(
                " maxits_nonlin    = ",
                state["maxits_nonlin"],
                " : max nb of iteration for nonlinear solver",
            )
            f1030(" precond          = ", state["precond"].strip(), " : preconditioner for FGMRES")
            f1020(" dim_fgmres       = ", state["dim_fgmres"], " : size of FGMRES Krylov subspace")
            f1020(" dim_pgmres       = ", state["dim_pgmres"], " : size of PGMRES Krylov subspace")
            f1020(" maxits_fgmres    = ", state["maxits_fgmres"], " : max nb of iteration for FGMRES")
            f1020(" maxits_pgmres    = ", state["maxits_pgmres"], " : max nb of iteration for PGMRES")
            f1010(" monitor_nonlin   = ", state["monitor_nonlin"], " : print nonlinear residual norm")
            f1010(" monitor_fgmres   = ", state["monitor_fgmres"], " : print FGMRES residual norm")
            f1010(" monitor_pgmres   = ", state["monitor_pgmres"], " : print PGMRES residual norm")
            f1030(
                " ortho_type       = ",
                state["ortho_type"].strip(),
                " : type of orthogonalization for FGMRES",
            )
            f1009(" reltol_nonlin    = ", state["reltol_nonlin"], " : nonlinear stopping criterion")
            f1009(" reltol_fgmres    = ", state["reltol_fgmres"], " : FGMRES stopping criterion")
            f1009(" reltol_pgmres    = ", state["reltol_pgmres"], " : PGMRES stopping criterion")
            f1030(" algo_nonlin      = ", state["algo_nonlin"].strip(), " : nonlinear algorithm")
            f1010(
                " use_mean_vrel    = ",
                state["use_mean_vrel"],
                " : use mean of previous 2 iterates to compute vrel",
            )
            if state["algo_nonlin"] == "anderson":
                f1020(
                    " fpfunc_andacc    = ",
                    state["fpfunc_andacc"],
                    " : fixed point function for Anderson acceleration",
                )
                f1020(
                    " dim_andacc       = ",
                    state["dim_andacc"],
                    " : size of Anderson minimization matrix",
                )
                f1009(
                    " reltol_andacc    = ",
                    state["reltol_andacc"],
                    " : relative tolerance for Anderson acceleration",
                )
                f1000(
                    " damping_andacc   = ",
                    state["damping_andacc"],
                    " : damping factor for Anderson acceleration",
                )
                f1020(
                    " start_andacc     = ",
                    state["start_andacc"],
                    " : nonlinear iteration at which acceleration starts",
                )

    line(" ")
    line(" Mechanical Deformation (Ridging) and Ice Strength")
    line("--------------------------------------------------")
    if state["kridge"] == 1:
        note = " : ridging enabled"
    else:
        note = " : ridging disabled"
    f1010(" tr_lvl           = ", state["tr_lvl"], " : ridging related tracers")
    f1020(" kridge           = ", state["kridge"], note)
    if state["kridge"] == 1:
        if state["krdg_partic"] == 1:
            note = " : new participation function"
        else:
            note = " : old participation function"
        f1020(" krdg_partic      = ", state["krdg_partic"], note)
        if state["krdg_partic"] == 1:
            f1002(" mu_rdg           = ", state["mu_rdg"], " : e-folding scale of ridged ice")
        if state["krdg_redist"] == 1:
            note = " : new redistribution function"
        else:
            note = " : old redistribution function"
        f1020(" krdg_redist      = ", state["krdg_redist"], note)
    if state["kstrength"] == 0:
        note = " : Hibler (1979)"
    elif state["kstrength"] == 1:
        note = " : Rothrock (1975)"
    else:
        note = " : unknown value"
    f1020(" kstrength        = ", state["kstrength"], note)
    if state["kstrength"] == 0:
        f1009(" Pstar            = ", state["Pstar"], " : P* strength factor")
        f1002(" Cstar            = ", state["Cstar"], " : C* strength exponent factor")
    elif state["kstrength"] == 1:
        f1002(" Cf               = ", state["Cf"], " : ratio of ridging work to PE change")

    line(" ")
    line(" Thermodynamics")
    line("--------------------------------")
    if state["ktherm"] == 1:
        note = " : Bitz and Lipscomb 1999 thermo"
    elif state["ktherm"] == 2:
        note = " : mushy-layer thermo"
    elif state["ktherm"] < 0:
        note = " : Thermodynamics disabled"
    else:
        note = " : unknown value"
    f1020(" ktherm           = ", state["ktherm"], note)
    if state["ktherm"] >= 0:
        f1002(" dt               = ", state["dt"], " : thermodynamic time step")
        f1002(" ksno             = ", state["ksno"], " : snow thermal conductivity")
        if state["ktherm"] == 1:
            f1030(" conduct          = ", state["conduct"].strip(), " : ice thermal conductivity")
        if state["ktherm"] == 2:
            f1002(" a_rapid_mode     = ", state["a_rapid_mode"], " : brine channel diameter")
            f1002(" Rac_rapid_mode   = ", state["Rac_rapid_mode"], " : critical Rayleigh number")
            f1002(" aspect_rapid_mode= ", state["aspect_rapid_mode"], " : brine convection aspect ratio")
            f1009(" dSdt_slow_mode   = ", state["dSdt_slow_mode"], " : drainage strength parameter")
            f1002(" phi_c_slow_mode  = ", state["phi_c_slow_mode"], " : critical liquid fraction")
            f1002(" phi_i_mushy      = ", state["phi_i_mushy"], " : solid fraction at lower boundary")
            f1002(" Tliquidus_max    = ", state["Tliquidus_max"], " : max mush liquidus temperature")
        f1002(" hfrazilmin       = ", state["hfrazilmin"], " : minimum new frazil ice thickness")

        line(" ")
        line(" Radiation")
        line("--------------------------------")
        if state["shortwave"].strip() == "dEdd":
            note = " : delta-Eddington multiple-scattering method"
        elif state["shortwave"].strip() == "dEdd_snicar_ad":
            note = " : delta-Eddington multiple-scattering method with SNICAR AD"
        elif state["shortwave"].strip() == "ccsm3":
            note = " : NCAR CCSM3 distribution method"
        else:
            note = " : unknown value"
        f1030(" shortwave        = ", state["shortwave"].strip(), note)
        if state["shortwave"][:4] == "dEdd":
            f1002(" R_ice            = ", state["R_ice"], " : tuning parameter for sea ice albedo")
            f1002(" R_pnd            = ", state["R_pnd"], " : tuning parameter for ponded sea ice albedo")
            f1002(" R_snw            = ", state["R_snw"], " : tuning parameter for snow broadband albedo")
            f1002(
                " dT_mlt           = ",
                state["dT_mlt"],
                " : change in temperature per change in snow grain radius",
            )
            f1002(" rsnw_mlt         = ", state["rsnw_mlt"], " : maximum melting snow grain radius")
            f1002(" kalg             = ", state["kalg"], " : absorption coefficient for algae")
            if state["shortwave"].strip() == "dEdd_snicar_ad":
                f1030(" snw_ssp_table    = ", state["snw_ssp_table"].strip())
        else:
            if state["albedo_type"].strip() == "ccsm3":
                note = " : NCAR CCSM3 albedos"
            elif state["albedo_type"].strip() == "constant":
                note = " : four constant albedos"
            else:
                note = " : unknown value"
                errors.append(f"unknown albedo_type={state['albedo_type']}")
            f1030(" albedo_type     = ", state["albedo_type"].strip(), note)
            if state["albedo_type"].strip() == "ccsm3":
                f1002(" albicev          = ", state["albicev"], " : visible  ice albedo for thicker ice")
                f1002(
                    " albicei          = ",
                    state["albicei"],
                    " : near infrared ice albedo for thicker ice",
                )
                f1002(" albsnowv         = ", state["albsnowv"], " : visible, cold snow albedo")
                f1002(" albsnowi         = ", state["albsnowi"], " : near infrared, cold snow albedo")
                f1002(" ahmax            = ", state["ahmax"], " : albedo is constant above this thickness")
                f1002(" ahmax            = ", state["ahmax"], " : albedo is constant above this thickness")
        f1000(" emissivity       = ", state["emissivity"], " : emissivity of snow and ice")
        f1010(
            " sw_redist        = ",
            state["sw_redist"],
            " : redistribute internal shortwave to surface",
        )
        if state["sw_redist"]:
            f1002(" sw_frac          = ", state["sw_frac"], " : fraction redistributed")
            f1002(
                " sw_dtemp         = ",
                state["sw_dtemp"],
                " : temperature difference from freezing to redistribute",
            )

    line(" ")
    line(" Atmospheric Forcing / Coupling")
    line("--------------------------------")
    f1010(
        " calc_Tsfc        = ",
        state["calc_Tsfc"],
        " : calculate surface temperature as part of thermo",
    )
    f1010(" calc_strair      = ", state["calc_strair"], " : calculate wind stress and speed")
    f1010(" rotate_wind      = ", state["rotate_wind"], " : rotate wind/stress to computational grid")
    f1010(" formdrag         = ", state["formdrag"], " : use form drag parameterization")
    f1000(" iceruf           = ", state["iceruf"], " : ice surface roughness at atmosphere interface (m)")
    f1010(
        " semi_implicit_Tsfc    = ",
        state["semi_implicit_Tsfc"],
        " : surface temperature coupling option based on d(hf)/dTs",
    )
    f1010(
        " vapor_flux_correction = ",
        state["vapor_flux_correction"],
        " : mass/enthalpy correction for evaporation/sublimation",
    )
    if state["atmbndy"].strip() == "constant":
        note = " : constant-based boundary layer"
    elif state["atmbndy"].strip() in ("similarity", "mixed"):
        f1010(" highfreq         = ", state["highfreq"], " : high-frequency atmospheric coupling")
        f1020(" natmiter         = ", state["natmiter"], " : number of atmo boundary layer iterations")
        f1002(" atmiter_conv     = ", state["atmiter_conv"], " : convergence criterion for ustar")
        if state["atmbndy"].strip() == "similarity":
            note = " : stability-based boundary layer"
        else:
            note = (
                " : stability-based boundary layer for wind stress, "
                "constant-based for sensible+latent heat fluxes"
            )
    else:
        note = " : unknown value"
    f1030(" atmbndy          = ", state["atmbndy"].strip(), note)

    line(" ")
    line(" Oceanic Forcing / Coupling")
    line("--------------------------------")
    if state["oceanmixed_ice"]:
        note = " : ocean mixed layer calculation (SST) enabled"
    else:
        note = " : ocean mixed layer calculation (SST) disabled"
    f1010(" oceanmixed_ice   = ", state["oceanmixed_ice"], note)
    if state["oceanmixed_ice"]:
        line("     WARNING: ocean mixed layer ON")
        line("     WARNING: will impact ocean forcing interaction")
        line("     WARNING: coupled forcing will be modified by mixed layer routine")
    f1030(" saltflux_option  = ", state["saltflux_option"].strip())
    if state["saltflux_option"].strip() == "constant":
        f1002(" ice_ref_salinity = ", state["ice_ref_salinity"])
    if state["tfrz_option"].strip() == "constant":
        note = " : constant ocean freezing temperature (Tocnfrz)"
    elif state["tfrz_option"].strip() == "minus1p8":
        note = " : constant ocean freezing temperature (-1.8C) (to be deprecated)"
    elif state["tfrz_option"].strip() == "linear_salt":
        note = " : linear function of salinity (use with ktherm=1)"
    elif state["tfrz_option"].strip() == "mushy":
        note = " : Assur (1958) as in mushy-layer thermo (ktherm=2)"
    else:
        note = " : unknown value"
    f1030(" tfrz_option      = ", state["tfrz_option"].strip(), note)
    if state["tfrz_option"].strip() == "constant":
        f1002(" Tocnfrz          = ", tocnfrz)
    f1030(" congel_freeze    = ", state["congel_freeze"].strip())
    if state["update_ocn_f"]:
        note = " : frazil water/salt fluxes included in ocean fluxes"
    else:
        note = " : frazil water/salt fluxes not included in ocean fluxes"
    f1010(" update_ocn_f     = ", state["update_ocn_f"], note)
    f1030(" cpl_frazil       = ", state["cpl_frazil"].strip())
    if state["l_mpond_fresh"] and state["tr_pond_topo"]:
        note = " : retain (topo) pond water until ponds drain"
    else:
        note = " : pond water not retained on ice (virtual only)"
    f1010(" l_mpond_fresh    = ", state["l_mpond_fresh"], note)
    if state["fbot_xfer_type"].strip() == "constant":
        note = " : ocean heat transfer coefficient is constant"
    elif state["fbot_xfer_type"].strip() == "Cdn_ocn":
        note = " : variable ocean heat transfer coefficient"  # only used with form_drag=T?
    else:
        note = " : unknown value"
    f1030(" fbot_xfer_type   = ", state["fbot_xfer_type"].strip(), note)
    f1000(" ustar_min        = ", state["ustar_min"], " : minimum value of ocean friction velocity")
    f1000(" hi_min           = ", state["hi_min"], " : minimum ice thickness allowed (m)")
    f1000(" puny             = ", puny, " : general-use minimum value")
    line(" Ice thickness category areas smaller than puny are always removed.")
    f1003(" itd_area_min     = ", state["itd_area_min"], " : zap residual ice below a minimum area")
    f1003(" itd_mass_min     = ", state["itd_mass_min"], " : zap residual ice below a minimum mass")
    if state["calc_dragio"]:
        note = " : dragio computed from iceruf_ocn"
    else:
        note = " : dragio hard-coded"
    f1010(" calc_dragio   = ", state["calc_dragio"], note)
    if state["calc_dragio"]:
        f1002(" iceruf_ocn       = ", state["iceruf_ocn"], " : under-ice roughness length")

    line(" ")
    line(" Floe size distribution and waves")
    line("---------------------------------")
    f1002(" floediam         = ", state["floediam"], " : constant floe diameter")
    if state["tr_fsd"]:
        if state["wave_spec"]:
            note = " : use wave spectrum for floe size distribution"
        else:
            note = "WARNING : floe size distribution does not use wave spectrum"
        f1010(" wave_spec          = ", state["wave_spec"], note)
        if state["wave_spec"]:
            if state["wave_spec_type"].strip() == "none":
                note = " : no wave data provided, no wave-ice interactions"
            elif state["wave_spec_type"].strip() == "profile":
                note = (
                    " : use fixed dummy wave spectrum for testing, sea surface height generated "
                    "using constant phase (1 iteration of wave fracture)"
                )
            elif state["wave_spec_type"].strip() == "constant":
                note = (
                    " : wave spectrum data file provided, sea surface height generated "
                    "using constant phase (1 iteration of wave fracture)"
                )
            elif state["wave_spec_type"].strip() == "random":
                note = (
                    " : wave spectrum data file provided, sea surface height generated using "
                    "random number (multiple iterations of wave fracture to convergence)"
                )
            else:
                note = " : unknown value"
            f1030(" wave_spec_type   = ", state["wave_spec_type"].strip(), note)
        f1020(" nfreq            = ", state["nfreq"], " : number of wave spectral forcing frequencies")
        if state["wave_height_type"].strip() == "internal":
            note = " : use internally generated wave height"
        elif state["wave_height_type"].strip() == "coupled":
            note = " : use wave height from external coupled model"
        elif state["wave_height_type"].strip() == "none":
            note = " : no wave height data available, default==0"
        else:
            note = " "
        f1030(" wave_height_type = ", state["wave_height_type"].strip(), note)

    line(" ")
    line(" Age related tracers")
    line("--------------------------------")
    f1010(" tr_iage          = ", state["tr_iage"], " : chronological ice age")
    f1010(" tr_FY            = ", state["tr_FY"], " : first-year ice area")

    line(" ")
    line(" Melt ponds")
    line("--------------------------------")
    if state["tr_pond_lvl"]:
        f1010(" tr_pond_lvl      = ", state["tr_pond_lvl"], " : level-ice pond formulation")
        f1002(" pndaspect        = ", state["pndaspect"], " : ratio of pond depth to area fraction")
        f1000(" dpscale          = ", state["dpscale"], " : time scale for flushing in permeable ice")
        if state["frzpnd"].strip() == "hlid":
            note = " : Stefan refreezing with pond ice thickness"
        elif state["frzpnd"].strip() == "cesm":
            note = " : CESM refreezing empirical formula"
        else:
            note = " : unknown value"
        f1030(" frzpnd           = ", state["frzpnd"].strip(), note)
        f1002(" hs1              = ", state["hs1"], " : snow depth of transition to pond ice")
    elif state["tr_pond_topo"]:
        f1010(" tr_pond_topo     = ", state["tr_pond_topo"], " : topo pond formulation")
        line("     WARNING: dpnd history fields are turned off for topo ponds")
        f1002(" hp1              = ", state["hp1"], " : critical ice lid thickness for topo ponds")
    elif state["tr_pond_sealvl"]:
        f1010(" tr_pond_sealvl   = ", state["tr_pond_sealvl"], " : sealvl pond formulation")
        f1002(" apnd_sl          = ", state["apnd_sl"], " : equilibrium pond fraction")
    elif state["shortwave"].strip() == "ccsm3":
        line("Pond effects on radiation are treated implicitly in the ccsm3 shortwave scheme")
    else:
        line("Using default dEdd melt pond scheme for testing only")
    if state["shortwave"][:4] == "dEdd":
        f1002(" hs0              = ", state["hs0"], " : snow depth of transition to bare sea ice")
    f1002(" rfracmin         = ", state["rfracmin"], " : minimum fraction of melt water added to ponds")
    f1002(" rfracmax         = ", state["rfracmax"], " : maximum fraction of melt water added to ponds")

    line(" ")
    line(" Snow redistribution/metamorphism tracers")
    line("-----------------------------------------")
    if state["tr_snow"]:
        f1010(" tr_snow          = ", state["tr_snow"], " : advanced snow physics")
        if state["snwredist"][:4] == "none":
            f1030(
                " snwredist        = ",
                state["snwredist"].strip(),
                " : Snow redistribution scheme turned off",
            )
        else:
            if state["snwredist"][:4] == "bulk":
                f1030(
                    " snwredist        = ",
                    state["snwredist"].strip(),
                    " : Using bulk snow redistribution scheme",
                )
            elif state["snwredist"][:6] == "ITDrdg":
                f1030(
                    " snwredist        = ",
                    state["snwredist"].strip(),
                    " : Using ridging based snow redistribution scheme",
                )
                f1002(" rhosnew          = ", state["rhosnew"], " : new snow density (kg/m^3)")
                f1002(" rhosmin          = ", state["rhosmin"], " : minimum snow density (kg/m^3)")
                f1002(" rhosmax          = ", state["rhosmax"], " : maximum snow density (kg/m^3)")
                f1002(" windmin          = ", state["windmin"], " : minimum wind speed to compact snow (m/s)")
                f1002(" drhosdwind       = ", state["drhosdwind"], " : wind compaction factor (kg s/m^4)")
            f1002(
                " snwlvlfac        = ",
                state["snwlvlfac"],
                " : fractional increase in snow depth for redistribution on ridges",
            )
        if not state["snwgrain"]:
            f1010(" snwgrain         = ", state["snwgrain"], " : Snow metamorphosis turned off")
        else:
            f1010(" snwgrain         = ", state["snwgrain"], " : Using snow metamorphosis scheme")
            f1002(" rsnw_tmax        = ", state["rsnw_tmax"], " : maximum snow radius (10^-6 m)")
        f1002(" rsnw_fall        = ", state["rsnw_fall"], " : radius of new snow (10^-6 m)")
        if state["snwgrain"]:
            if state["use_smliq_pnd"]:
                note = " : Using liquid water in snow for melt ponds"
            else:
                note = " : NOT using liquid water in snow for melt ponds"
            f1010(" use_smliq_pnd    = ", state["use_smliq_pnd"], note)
            if state["snw_aging_table"] == "test":
                note = " : Using 5x5x1 test matrix of internallly defined snow aging parameters"
                f1030(" snw_aging_table  = ", state["snw_aging_table"].strip(), note)
            elif state["snw_aging_table"] == "snicar":
                note = " : Reading 3D snow aging parameters from SNICAR file"
                f1030(" snw_aging_table  = ", state["snw_aging_table"].strip(), note)
                f1031(" snw_filename     = ", state["snw_filename"].strip())
                f1031(" snw_tau_fname    = ", state["snw_tau_fname"].strip())
                f1031(" snw_kappa_fname  = ", state["snw_kappa_fname"].strip())
                f1031(" snw_drdt0_fname  = ", state["snw_drdt0_fname"].strip())
            elif state["snw_aging_table"] == "file":
                note = (
                    " : Reading 1D and 3D snow aging dimensions and parameters from external file"
                )
                f1030(" snw_aging_table  = ", state["snw_aging_table"].strip(), note)
                f1031(" snw_filename     = ", state["snw_filename"].strip())
                f1031(" snw_rhos_fname   = ", state["snw_rhos_fname"].strip())
                f1031(" snw_Tgrd_fname   = ", state["snw_Tgrd_fname"].strip())
                f1031(" snw_T_fname      = ", state["snw_T_fname"].strip())
                f1031(" snw_tau_fname    = ", state["snw_tau_fname"].strip())
                f1031(" snw_kappa_fname  = ", state["snw_kappa_fname"].strip())
                f1031(" snw_drdt0_fname  = ", state["snw_drdt0_fname"].strip())

    line(" ")
    line(" Primary state variables, tracers")
    line("   (excluding biogeochemistry)")
    line("---------------------------------")
    line("Conserved properties (all tracers are conserved):")
    line("ice concentration, volume and enthalpy")
    line("snow volume and enthalpy")
    if state["ktherm"] == 2:
        f1030(" ice salinity")
    if state["tr_fsd"]:
        f1010(" tr_fsd           = ", state["tr_fsd"], " : floe size distribution")
    if state["tr_lvl"]:
        f1010(" tr_lvl           = ", state["tr_lvl"], " : ridging related tracers")
    if state["tr_pond_lvl"]:
        f1010(" tr_pond_lvl      = ", state["tr_pond_lvl"], " : level-ice pond formulation")
    if state["tr_pond_sealvl"]:
        f1010(" tr_pond_sealvl   = ", state["tr_pond_sealvl"], " : sea level pond formulation")
    if state["tr_pond_topo"]:
        f1010(" tr_pond_topo     = ", state["tr_pond_topo"], " : topo pond formulation")
    if state["tr_snow"]:
        f1010(" tr_snow          = ", state["tr_snow"], " : advanced snow physics")
    if state["tr_iage"]:
        f1010(" tr_iage          = ", state["tr_iage"], " : chronological ice age")
    if state["tr_FY"]:
        f1010(" tr_FY            = ", state["tr_FY"], " : first-year ice area")
    if state["tr_iso"]:
        f1010(" tr_iso           = ", state["tr_iso"], " : diagnostic isotope tracers")
    if state["tr_aero"]:
        f1010(" tr_aero          = ", state["tr_aero"], " : CESM aerosol tracers")
    line("Non-conserved properties:")
    line("ice surface temperature")
    line("ice velocity components and internal stress")

    line(" ")
    line(" Other ice_in namelist parameters:")
    line("===================================== ")
    if state["runid"].strip() != "unknown":
        f1031(" runid            = ", state["runid"].strip())
    f1031(" version_name     = ", state["version_name"].strip())
    f1031(" runtype          = ", state["runtype"].strip())
    f1021(" year_init        = ", state["year_init"])
    f1021(" month_init       = ", state["month_init"])
    f1021(" day_init         = ", state["day_init"])
    f1021(" sec_init         = ", state["sec_init"])
    f1021(" istep0           = ", state["istep0"])
    f1031(" npt_unit         = ", state["npt_unit"].strip())
    f1021(" npt              = ", state["npt"])
    f1021(" diagfreq         = ", state["diagfreq"])
    f1011(" print_global     = ", state["print_global"])
    f1011(" print_points     = ", state["print_points"])
    f1011(" debug_model      = ", state["debug_model"])
    f1022(" debug_model_step = ", state["debug_model_step"])
    f1011(" debug_model_ponds = ", state["debug_model_ponds"])
    f1011(" debug_model_ponds_inputs = ", state["debug_model_ponds_inputs"])
    line(f" debug_model_ponds_i = {state['debug_model_ponds_i']}")
    line(f" debug_model_ponds_j = {state['debug_model_ponds_j']}")
    f1021(" debug_model_i    = ", state["debug_model_i"])
    f1021(" debug_model_i    = ", state["debug_model_j"])
    f1021(" debug_model_iblk = ", state["debug_model_iblk"])
    f1021(" debug_model_task = ", state["debug_model_task"])
    f1011(" timer_stats      = ", state["timer_stats"])
    f1011(" memory_stats     = ", state["memory_stats"])
    f1031(" bfbflag          = ", state["bfbflag"].strip())
    f1021(" numin            = ", state["numin"])
    f1021(" numax            = ", state["numax"])
    f1011(" grid_outfile     = ", state["grid_outfile"])
    f_chars(" histfreq         = ", state["histfreq"][streams])
    f_ints(" histfreq_n       = ", state["histfreq_n"][streams])
    f_chars(" histfreq_base    = ", state["histfreq_base"][streams])
    f_logicals(" hist_avg         = ", state["hist_avg"][streams])
    f_chars(" hist_suffix      = ", state["hist_suffix"][streams])
    f1031(" history_dir      = ", state["history_dir"].strip())
    f1031(" history_file     = ", state["history_file"].strip())
    f1021(" history_precision= ", state["history_precision"])
    f1031(" history_format   = ", state["history_format"].strip())
    f1031(" history_rearranger = ", state["history_rearranger"].strip())
    f1021(" history_iotasks  = ", state["history_iotasks"])
    f1021(" history_root     = ", state["history_root"])
    f1021(" history_stride   = ", state["history_stride"])
    f1031(" hist_time_axis   = ", state["hist_time_axis"].strip())
    f1021(" history_deflate  = ", state["history_deflate"])
    f_ints(" history_chunksize= ", state["history_chunksize"])
    if state["write_ic"]:
        line(f" Initial condition will be written in  {state['incond_dir'].strip()}")
    f_chars(" dumpfreq         = ", state["dumpfreq"][streams])
    f_ints(" dumpfreq_n       = ", state["dumpfreq_n"][streams])
    f_chars(" dumpfreq_base    = ", state["dumpfreq_base"][streams])
    f1011(" dump_last        = ", state["dump_last"])
    f1011(" restart          = ", state["restart"])
    f1031(" restart_dir      = ", state["restart_dir"].strip())
    f1011(" restart_ext      = ", state["restart_ext"])
    f1031(" restart_mod      = ", state["restart_mod"].strip())
    f1011(" restart_coszen   = ", state["restart_coszen"])
    f1031(" restart_format   = ", state["restart_format"].strip())
    f1021(" restart_deflate  = ", state["restart_deflate"])
    f_ints(" restart_chunksize= ", state["restart_chunksize"])
    #         write(nu_diag,1011) ' lcdf64           = ', lcdf64   ! deprecated
    f1031(" restart_rearranger = ", state["restart_rearranger"].strip())
    f1021(" restart_iotasks  = ", state["restart_iotasks"])
    f1021(" restart_root     = ", state["restart_root"])
    f1021(" restart_stride   = ", state["restart_stride"])
    f1031(" restart_file     = ", state["restart_file"].strip())
    f1031(" pointer_file     = ", state["pointer_file"].strip())
    f1011(" use_restart_time = ", state["use_restart_time"])
    f1011(" write_histrest   = ", state["write_histrest"])
    f1031(" ice_ic           = ", state["ice_ic"].strip())
    if state["grid_type"].strip() != "rectangular" or state["grid_type"].strip() != "column":
        f1031(" grid_file        = ", state["grid_file"].strip())
        f1031(" gridcpl_file     = ", state["gridcpl_file"].strip())
        f1031(" bathymetry_file  = ", state["bathymetry_file"].strip())
        if state["kmt_type"].strip() == "file":
            f1031(" kmt_file         = ", state["kmt_file"].strip())
    f1011(" conserv_check    = ", state["conserv_check"])
    f1021(" fyear_init       = ", state["fyear_init"])
    f1021(" ycycle           = ", state["ycycle"])
    f1031(" atm_data_type    = ", state["atm_data_type"].strip())
    f1031(" atm_data_version = ", state["atm_data_version"].strip())
    if state["atm_data_type"].strip() != "default":
        f1031(" atm_data_dir     = ", state["atm_data_dir"].strip())
        f1031(" precip_units     = ", state["precip_units"].strip())
    elif state["atm_data_type"].strip() == "default":
        f1031(" default_season   = ", state["default_season"].strip())
    if state["wave_spec"]:
        f1031(" wave_spec_file   = ", state["wave_spec_file"].strip())
    if state["bgc_data_type"].strip() == "ncar" or state["ocn_data_type"].strip() == "ncar":
        f1031(" oceanmixed_file  = ", state["oceanmixed_file"].strip())
    if state["cpl_bgc"]:
        line("BGC coupling is switched ON")
    else:
        line("BGC coupling is switched OFF")
    f1031(" bgc_data_type    = ", state["bgc_data_type"].strip())
    f1031(" fe_data_type     = ", state["fe_data_type"].strip())
    f1031(" ice_data_type    = ", state["ice_data_type"].strip())
    f1031(" ice_data_conc    = ", state["ice_data_conc"].strip())
    f1031(" ice_data_dist    = ", state["ice_data_dist"].strip())
    f1031(" bgc_data_dir     = ", state["bgc_data_dir"].strip())
    f1031(" ocn_data_type    = ", state["ocn_data_type"].strip())
    if (
        state["bgc_data_type"].strip() != "default"
        or state["ocn_data_type"].strip() != "default"
    ):
        f1031(" ocn_data_dir     = ", state["ocn_data_dir"].strip())
        f1011(" restore_ocn      = ", state["restore_ocn"])
    f1011(" restore_ice      = ", state["restore_ice"])
    if state["restore_ice"] or state["restore_ocn"]:
        f1021(" trestore         = ", int(state["trestore"]))
    line(" ")
    line(
        f"{'Diagnostic point 1: lat, lon =':31}"
        f"{state['latpnt'][0]:8.2f}{state['lonpnt'][0]:8.2f}"
    )
    line(
        f"{'Diagnostic point 2: lat, lon =':31}"
        f"{state['latpnt'][1]:8.2f}{state['lonpnt'][1]:8.2f}"
    )
    line(" ")
    # tracer restarts
    f1011(" restart_age      = ", state["restart_age"])
    f1011(" restart_FY       = ", state["restart_FY"])
    f1011(" restart_lvl      = ", state["restart_lvl"])
    f1011(" restart_pond_lvl = ", state["restart_pond_lvl"])
    f1011(" restart_pond_sealvl = ", state["restart_pond_sealvl"])
    f1011(" restart_pond_topo= ", state["restart_pond_topo"])
    f1011(" restart_snow     = ", state["restart_snow"])
    f1011(" restart_iso      = ", state["restart_iso"])
    f1011(" restart_aero     = ", state["restart_aero"])
    f1011(" restart_fsd      = ", state["restart_fsd"])
    f1021(" n_iso            = ", state["n_iso"])
    f1021(" n_aero           = ", state["n_aero"])
    f1021(" n_zaero          = ", state["n_zaero"])
    f1021(" n_algae          = ", state["n_algae"])
    f1021(" n_doc            = ", state["n_doc"])
    f1021(" n_dic            = ", state["n_dic"])
    f1021(" n_don            = ", state["n_don"])
    f1021(" n_fed            = ", state["n_fed"])
    f1021(" n_fep            = ", state["n_fep"])
    line(" ")


def input_data(nml_path="configuration/ice_in"):
    """Read gx3 namelist values and push column parameters into icepack."""
    global nml_filename
    nml_filename = nml_path
    state = copy.deepcopy(namelist_defaults)
    required_groups = (
        "setup_nml",
        "grid_nml",
        "tracer_nml",
        "thermo_nml",
        "dynamics_nml",
        "shortwave_nml",
        "ponds_nml",
        "snow_nml",
        "forcing_nml",
    )

    #-----------------------------------------------------------------
    # read from input file
    # namelist read starts: ice_in groups through the assignments below
    #-----------------------------------------------------------------
    with open(nml_path, encoding="utf-8") as handle:
        lines = handle.readlines()
    groups = {}
    current = None
    for line in lines:
        out = []
        quote = None
        for char in line:
            if quote:
                out.append(char)
                if char == quote:
                    quote = None
                continue
            if char in "'\"":
                quote = char
                out.append(char)
                continue
            if char == "!":
                break
            out.append(char)
        text = "".join(out).strip()
        if not text:
            continue
        if text.startswith("&"):
            current = text[1:].split()[0]
            groups.setdefault(current, [])
            continue
        if text == "/":
            current = None
            continue
        if current is None:
            continue
        groups[current].append(text)

    parsed = {}
    for name, chunks in groups.items():
        assignments = []
        pending = None
        for text in chunks:
            if "=" in text:
                if pending is not None:
                    assignments.append(pending)
                pending = text
            else:
                if pending is None:
                    raise ValueError(f"{name}: cannot parse {text}")
                pending = pending + " " + text
        if pending is not None:
            assignments.append(pending)
        pairs = []
        for item in assignments:
            key, raw = item.split("=", 1)
            pairs.append((key.strip(), raw.strip()))
        parsed[name] = pairs
    groups = parsed

    missing = [name for name in required_groups if name not in groups]
    if missing:
        raise ValueError(f"missing namelist groups: {', '.join(missing)}")

    # Keys not in namelist_defaults are skipped.
    # domain_nml is read by init_domain_blocks.
    group_names = list(required_groups)
    for name in group_names:
        ice_fileunits.nu_diag.write(f" (input_data) Reading {name}\n")
        for key, raw in groups[name]:
            match = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)(?:\((\d+)\))?", key.strip())
            if match is None or match.group(1) not in state:
                continue
            field = match.group(1)
            index = int(match.group(2)) if match.group(2) else None
            current = state[field]
            tokens = []
            buf = []
            quote = None
            for char in raw:
                if quote:
                    buf.append(char)
                    if char == quote:
                        quote = None
                    continue
                if char in "'\"":
                    quote = char
                    buf.append(char)
                    continue
                if char == ",":
                    token = "".join(buf).strip()
                    if token:
                        tokens.append(token)
                    buf = []
                    continue
                buf.append(char)
            token = "".join(buf).strip()
            if token:
                tokens.append(token)

            prototype = current[0] if isinstance(current, list) else current
            values = []
            for token in tokens:
                token = token.strip()
                if isinstance(prototype, bool):
                    lowered = token.lower()
                    if lowered in (".true.", "true"):
                        values.append(True)
                    elif lowered in (".false.", "false"):
                        values.append(False)
                    else:
                        raise ValueError(f"expected a Fortran logical, got {token}")
                elif isinstance(prototype, str):
                    if len(token) >= 2 and token[0] == token[-1] and token[0] in "'\"":
                        values.append(token[1:-1])
                    else:
                        values.append(token)
                elif isinstance(prototype, float):
                    text = re.sub(r"([0-9.])d([+-]?\d+)", r"\1e\2", token.strip().lower())
                    values.append(float(text))
                elif isinstance(prototype, int):
                    values.append(int(token))
                else:
                    raise TypeError(type(prototype))

            if isinstance(current, list):
                if index is None:
                    if len(values) > len(current):
                        raise ValueError(f"{field}: {len(values)} values, room for {len(current)}")
                    current[: len(values)] = values
                else:
                    if len(values) != 1:
                        raise ValueError(f"{field}({index}) expects one value")
                    current[index - 1] = values[0]
            else:
                if index is not None:
                    raise ValueError(f"{field} is not an array")
                if len(values) != 1:
                    raise ValueError(f"{field} expects one value, got {tokens}")
                state[field] = values[0]
    #-----------------------------------------------------------------
    # namelist read ends
    #-----------------------------------------------------------------

    errors = []
    puny = icepack_parameters.puny

    def diag(text):
        ice_fileunits.nu_diag.write(f" {text}\n")

    if state["debug_model"] and (state["debug_model_ponds"] or state["debug_model_ponds_inputs"]):
        points = list(zip(state["debug_model_ponds_i"], state["debug_model_ponds_j"]))
        if any((i, j) != (-1, -1) and (i < 1 or j < 1) for i, j in points):
            errors.append("debug_model_ponds_i/j must be positive paired indices or -1/-1")
        if not any(i > 0 and j > 0 for i, j in points):
            errors.append("pond diagnostics require at least one debug_model_ponds_i/j pair")

    # itd_* left unset in the namelist copy the dynamics zap thresholds.
    if state["itd_area_min"] != itd_min_unset or state["itd_mass_min"] != itd_min_unset:
        diag("(input_data) WARNING: zap_residual parameters are reset in namelist")
    elif state["itd_area_min"] == 0.0 or state["itd_mass_min"] == 0.0:
        diag("(input_data) WARNING: zap_residual is turned off")
    else:
        state["itd_area_min"] = state["dyn_area_min"]
        state["itd_mass_min"] = state["dyn_mass_min"]

    if state["ice_ic"] == "default":
        state["ice_ic"] = "internal"
    if state["ice_data_conc"] == "default":
        state["ice_data_conc"] = "parabolic"
    if state["ice_data_dist"] == "default":
        state["ice_data_dist"] = "uniform"
    if state["ice_data_type"] == "default":
        state["ice_data_type"] = "latsst"
    if state["grid_format"] == "nc":
        state["grid_format"] = "pop_nc"
    elif state["grid_format"] == "nc_ext":
        state["grid_format"] = "pop_nc_ext"

    diag_type = state["diag_type"]
    diag_file = state["diag_file"]
    if ice_fileunits.nu_diag is not sys.stdout:
        ice_fileunits.nu_diag.close()
    if diag_type == "file":
        print(f"Diagnostic output will be in file {diag_file}")
        ice_fileunits.nu_diag = open(diag_file, "w", encoding="utf-8")
    else:
        ice_fileunits.nu_diag = sys.stdout
    ice_fileunits.nu_diag.write("--------------------------------\n")
    ice_fileunits.nu_diag.write("   (input_data)\n")
    ice_fileunits.nu_diag.write("  CICE model diagnostic output  \n")
    ice_fileunits.nu_diag.write("--------------------------------\n")
    ice_fileunits.nu_diag.write(" \n")
    ice_fileunits.nu_diag.flush()

    if state["runtype"] == "continue":
        diag("(input_data)NOTE: runtype=continue, setting restart=.true.")
        if not state["use_restart_time"]:
            diag("(input_data)NOTE: runtype=continue, setting use_restart_time=.true.")
        diag("")
        state["restart"] = True
        state["use_restart_time"] = True
    elif state["runtype"] == "initial":
        if state["ice_ic"] in ("none", "internal"):
            diag("(input_data)NOTE: ice_ic = none or internal, setting restart flags to .false.")
            if not state["use_restart_time"]:
                diag("(input_data)NOTE: ice_ic = none or internal, setting use_restart_time=.false.")
            diag("")
            state["use_restart_time"] = False
            state["restart"] = False
            for name in (
                "restart_iso",
                "restart_aero",
                "restart_fsd",
                "restart_age",
                "restart_FY",
                "restart_lvl",
                "restart_pond_lvl",
                "restart_pond_topo",
                "restart_snow",
            ):
                state[name] = False
        else:
            diag("(input_data)NOTE: ice_ic /= none or internal, setting restart=.true.")
            diag("")
            state["restart"] = True
    else:
        diag(f"(input_data) ERROR: runtype unknown = {state['runtype']}")
        errors.append(f"unknown runtype={state['runtype']}")

    io_formats = (
        "cdf1",
        "cdf2",
        "cdf5",
        "hdf5",
        "pnetcdf1",
        "pnetcdf2",
        "pnetcdf5",
        "pio_netcdf",  # backwards compatibility
        "pio_pnetcdf",  # backwards compatibility
        "binary",
        "default",  # backwards compatibility
    )
    if state["history_format"] not in io_formats:
        diag(f"(input_data) ERROR: history_format unknown = {state['history_format']}")
        errors.append(f"unknown history_format={state['history_format']}")
    if state["restart_format"] not in io_formats:
        diag(f"(input_data) ERROR: restart_format unknown = {state['restart_format']}")
        errors.append(f"unknown restart_format={state['restart_format']}")

    # backwards compatibility for history and restart formats, lcdf64
    if state["history_format"] in ("pio_pnetcdf", "pio_netcdf"):
        diag(
            "(input_data) WARNING: history_format="
            f"{state['history_format']} is deprecated, please update namelist settings"
        )
    if state["restart_format"] in ("pio_pnetcdf", "pio_netcdf"):
        diag(
            "(input_data) WARNING: restart_format="
            f"{state['restart_format']} is deprecated, please update namelist settings"
        )

    if state["lcdf64"]:
        diag("(input_data) WARNING: lcdf64 is deprecated, please update namelist settings")
        if state["history_format"] in ("default", "pio_netcdf"):
            state["history_format"] = "cdf2"
        elif state["history_format"] == "pio_pnetcdf":
            state["history_format"] = "pnetcdf2"
        else:
            diag(
                "(input_data) ERROR: lcdf64 is T and history_format not supported for "
                f"{state['history_format']}"
            )
            errors.append(
                "lcdf64 is T and history_format not supported for "
                f"{state['history_format']}"
            )
        if state["restart_format"] in ("default", "pio_netcdf"):
            state["restart_format"] = "cdf2"
        elif state["restart_format"] == "pio_pnetcdf":
            state["restart_format"] = "pnetcdf2"
        else:
            diag(
                "(input_data) ERROR: lcdf64 is T and restart_format not supported for "
                f"{state['restart_format']}"
            )
            errors.append(
                "lcdf64 is T and restart_format not supported for "
                f"{state['restart_format']}"
            )
    else:
        if state["history_format"] in ("default", "pio_netcdf"):
            state["history_format"] = "cdf1"
        elif state["history_format"] == "pio_pnetcdf":
            state["history_format"] = "pnetcdf1"
        if state["restart_format"] in ("default", "pio_netcdf"):
            state["restart_format"] = "cdf1"
        elif state["restart_format"] == "pio_pnetcdf":
            state["restart_format"] = "pnetcdf1"

    if state["ktransport"] <= 0:
        state["advection"] = "none"
    elif state["advection"] not in ("remap", "upwind"):
        diag(f"(input_data) ERROR: invalid advection={state['advection']}")
        errors.append(f"invalid advection={state['advection']}")

    if state["ncat"] == 1 and state["kitd"] == 1:
        diag("(input_data) ERROR: kitd incompatability: ncat=1 and kitd=1")
        diag("(input_data) ERROR:   Remapping the ITD is not allowed for ncat=1.")
        diag("(input_data) ERROR:   Use kitd = 0 (delta function ITD) with kcatbound = 0")
        diag("(input_data) ERROR:   or for column configurations use kcatbound = -1")
        errors.append("kitd=1 is not allowed when ncat=1")
    if state["ncat"] != 1 and state["kcatbound"] == -1:
        diag("(input_data) ERROR: ITD required for ncat > 1")
        diag(f"(input_data) ERROR:   ncat={state['ncat']} kcatbound={state['kcatbound']}")
        diag("(input_data) ERROR:   Please review user guide")
        errors.append("kcatbound=-1 requires ncat=1")
    if state["ncat"] < 1:
        diag("(input_data) ERROR: ncat < 1")
        errors.append("ncat < 1")
    if state["nilyr"] < 1:
        diag("(input_data) ERROR: nilyr < 1")
        errors.append("nilyr < 1")
    if state["nslyr"] < 1:
        diag("(input_data) ERROR: nslyr < 1")
        errors.append("nslyr < 1")
    if state["nblyr"] < 1:
        diag("(input_data) ERROR: nblyr < 1")
        diag("(input_data) ERROR:   not allowed due to history implementation.")
        errors.append("nblyr < 1")
    if state["nfsd"] < 1:
        diag("(input_data) ERROR: nfsd < 1")
        diag("(input_data) ERROR:   not allowed due to history implementation.")
        errors.append("nfsd < 1")

    if state["kdyn"] > 3:
        diag("(input_data) WARNING: kdyn out of range")
        errors.append("kdyn out of range")
    if state["kdyn"] == 1 and state["evp_algorithm"] == "shared_mem_1d":
        state["save_ghte_ghtn"] = True
    if state["kdyn"] == 1 and state["evp_algorithm"] not in ("standard_2d", "shared_mem_1d"):
        diag(f"(input_data) ERROR: unknown evp_algorithm={state['evp_algorithm']}")
        errors.append(f"unknown evp_algorithm={state['evp_algorithm']}")

    if state["kdyn"] == 2 and state["revised_evp"]:
        diag("(input_data) WARNING: revised_evp = T with EAP dynamics")
        diag("(input_data) WARNING:   revised_evp is ignored")
        state["revised_evp"] = False

    if state["seabed_stress"]:
        if state["seabed_stress_method"] not in ("LKD", "probabilistic"):
            diag("(input_data) ERROR: invalid seabed stress method")
            diag("(input_data) ERROR: seabed_stress_method should be LKD or probabilistic")
            errors.append("invalid seabed stress method")

    if state["close_boundaries"]:
        diag(
            "(input_data) ERROR: close_boundaries deprecated, "
            "use ew_boundary_type=closed and/or ns_boundary_type=closed"
        )
        errors.append("close_boundaries deprecated")

    state["capping"] = -9.99e30
    if state["kdyn"] in (1, 3):
        if state["capping_method"] == "max":
            state["capping"] = 1.0
        elif state["capping_method"] == "sum":
            state["capping"] = 0.0
        else:
            diag("(input_data) ERROR: invalid method for capping viscosities")
            diag("(input_data) ERROR: capping_method should be equal to max or sum")
            errors.append("capping_method must be max or sum")

    rplvl = 0
    rptopo = 0
    rpsealvl = 0
    if state["tr_pond_lvl"]:  # level-ice melt ponds
        rplvl = 1
    if state["tr_pond_sealvl"]:  # Sea level melt ponds
        rpsealvl = 1
    if state["tr_pond_topo"]:  # explicit melt ponds (topographic)
        rptopo = 1

    state["tr_pond"] = False  # explicit melt ponds
    if rplvl + rptopo + rpsealvl > 0:
        state["tr_pond"] = True

    if rplvl + rptopo + rpsealvl > 1:
        diag("(input_data) ERROR: Must use only one melt pond scheme")
        errors.append("must use only one melt pond scheme")
    if state["tr_pond_lvl"] and not state["tr_lvl"]:
        diag("(input_data) ERROR: tr_pond_lvl=T but tr_lvl=F")
        errors.append("tr_pond_lvl requires tr_lvl")
    # tcraig - this was originally implemented by resetting hs0=0. EH says it might be OK
    # to not reset it but extra calculations are done and it might not be bfb.  In our
    # testing, we should explicitly set hs0 to 0. when setting tr_pond_lvl=T, and otherwise
    # this will abort (safest option until additional testing is done)
    if state["tr_pond_lvl"] and abs(state["hs0"]) > puny:
        diag("(input_data) ERROR: tr_pond_lvl=T and hs0 /= 0")
        errors.append("tr_pond_lvl requires hs0 = 0")
    if state["semi_implicit_Tsfc"] and state["tr_pond_topo"]:
        diag("ERROR: semi_implicit_Tsfc and tr_pond_topo not supported together")
        errors.append("semi_implicit_Tsfc and tr_pond_topo not supported together")
    if state["shortwave"][:4] != "dEdd" and state["tr_pond"] and state["calc_Tsfc"]:
        diag("(input_data) ERROR: tr_pond=T, calc_tsfc=T, invalid shortwave")
        diag("(input_data) ERROR:   Must use shortwave=dEdd or dEdd_snicar_ad")
        errors.append("ponds with calc_Tsfc require shortwave dEdd")

    if state["snwredist"][:3] == "ITD" and not state["tr_snow"]:
        diag("ERROR: snwredist on but tr_snow=F")
        diag("ERROR: Use tr_snow=T for snow redistribution")
        errors.append("snwredist ITD requires tr_snow")
    if state["snwredist"][:4] == "bulk" and not state["tr_lvl"]:
        diag("ERROR: snwredist=bulk but tr_lvl=F")
        diag("ERROR: Use tr_lvl=T for snow redistribution")
        errors.append("snwredist bulk requires tr_lvl")
    if state["snwredist"][:6] == "ITDrdg" and not state["tr_lvl"]:
        diag("ERROR: snwredist=ITDrdg but tr_lvl=F")
        diag("ERROR: Use tr_lvl=T for snow redistribution")
        errors.append("snwredist ITDrdg requires tr_lvl")
    if state["use_smliq_pnd"] and not state["snwgrain"]:
        diag("ERROR: use_smliq_pnd = T but")
        diag("ERROR: snow metamorphosis not used")
        diag("ERROR: Use snwgrain=T with smliq for ponds")
        errors.append("use_smliq_pnd requires snwgrain")
    if state["use_smliq_pnd"] and not state["tr_snow"]:
        diag("ERROR: use_smliq_pnd = T but")
        diag("ERROR: snow tracers are not active")
        diag("ERROR: Use tr_snow=T with smliq for ponds")
        errors.append("use_smliq_pnd requires tr_snow")
    if state["snwgrain"] and not state["tr_snow"]:
        diag("ERROR: snwgrain=T but tr_snow=F")
        diag("ERROR: Use tr_snow=T for snow metamorphosis")
        errors.append("snwgrain requires tr_snow")
    if state["snw_aging_table"] not in ("test", "snicar", "file"):
        diag(f"ERROR: unknown snw_aging_table = {state['snw_aging_table']}")
        errors.append(f"unknown snw_aging_table={state['snw_aging_table']}")

    if state["tr_iso"] and state["n_iso"] == 0:
        diag("(input_data) ERROR: isotopes activated but")
        diag("(input_data) ERROR:   not allocated in tracer array.")
        diag("(input_data) ERROR:   if tr_iso, n_iso must be > 0.")
        errors.append("tr_iso requires n_iso > 0")
    if state["tr_aero"] and state["n_aero"] == 0:
        diag("(input_data) ERROR: aerosols activated but")
        diag("(input_data) ERROR:   not allocated in tracer array.")
        diag("(input_data) ERROR:   if tr_aero, n_aero must be > 0.")
        errors.append("tr_aero requires n_aero > 0")

    if state["shortwave"][:4] != "dEdd" and state["tr_aero"]:
        diag("(input_data) ERROR: tr_aero=T, invalid shortwave")
        diag("(input_data) ERROR:   Must use shortwave=dEdd or dEdd_snicar_ad")
        errors.append("tr_aero requires shortwave dEdd")
    if state["shortwave"][:4] != "dEdd" and state["snwgrain"]:
        diag("(input_data) ERROR: snow grain radius is activated")
        diag("(input_data) ERROR:   Must use shortwave=dEdd or dEdd_snicar_ad")
        errors.append("snwgrain requires shortwave dEdd")

    rfracmin = state["rfracmin"]
    rfracmax = state["rfracmax"]
    if (
        rfracmin < -puny
        or rfracmin > 1.0 + puny
        or rfracmax < -puny
        or rfracmax > 1.0 + puny
        or rfracmin > rfracmax
    ):
        diag("(input_data) ERROR: rfracmin, rfracmax must be between 0 and 1")
        diag("(input_data) ERROR:   and rfracmax >= rfracmin")
        errors.append("rfracmin and rfracmax must lie in [0, 1] with rfracmax >= rfracmin")
    state["rfracmin"] = min(max(rfracmin, 0.0), 1.0)
    state["rfracmax"] = min(max(rfracmax, 0.0), 1.0)

    if state["ktherm"] == 2 and not state["calc_Tsfc"]:
        diag("(input_data) ERROR: ktherm = 2 and calc_Tsfc=F")
        errors.append("ktherm=2 requires calc_Tsfc")

    # ech: allow inconsistency for testing sensitivities.  It's not recommended for science runs
    if state["ktherm"] == 1 and state["tfrz_option"] != "linear_salt":
        diag(f"(input_data) WARNING: ktherm = 1 and tfrz_option = {state['tfrz_option']}")
        diag("(input_data) WARNING:   For consistency, set tfrz_option = linear_salt")
    if state["ktherm"] == 2 and state["tfrz_option"] != "mushy":
        diag(f"(input_data) WARNING: ktherm = 2 and tfrz_option = {state['tfrz_option']}")
        diag("(input_data) WARNING:   For consistency, set tfrz_option = mushy")
    if state["ktherm"] == 1 and state["saltflux_option"] != "constant":
        diag(
            "(input_data) WARNING: ktherm = 1 and saltflux_option = "
            f"{state['saltflux_option']}"
        )
        diag("(input_data) WARNING:   For consistency, set saltflux_option = constant")
    if state["ktherm"] == 1 and not state["sw_redist"]:
        flag = "T" if state["sw_redist"] else "F"
        diag(f"(input_data) WARNING: ktherm = 1 and sw_redist = {flag}")
        diag("(input_data) WARNING:   For consistency, set sw_redist = .true.")

    if state["atmbndy"] == "default":
        diag("(input_data) WARNING: atmbndy = default is deprecated")
        diag("(input_data) WARNING:   setting atmbndy = similarity")
        state["atmbndy"] = "similarity"

    if state["formdrag"]:
        if state["atmbndy"] == "constant":
            diag("(input_data) ERROR: formdrag=T and atmbndy=constant")
            errors.append("formdrag=T and atmbndy=constant")
        if not state["calc_strair"]:
            diag("(input_data) ERROR: formdrag=T and calc_strair=F")
            errors.append("formdrag=T and calc_strair=F")
        if not state["tr_pond"]:
            diag("(input_data) ERROR: formdrag=T and tr_pond=F")
            errors.append("formdrag=T and tr_pond=F")
        if not state["tr_lvl"]:
            diag("(input_data) ERROR: formdrag=T and tr_lvl=F")
            errors.append("formdrag=T and tr_lvl=F")
    if state["fbot_xfer_type"] == "Cdn_ocn" and not state["formdrag"]:
        diag("(input_data) ERROR: formdrag=F and fbot_xfer_type=Cdn_ocn")
        errors.append("formdrag=F and fbot_xfer_type=Cdn_ocn")

    if state["atm_data_type"] == "monthly" and state["calc_strair"]:
        diag("(input_data) ERROR: atm_data_type=monthly and calc_strair=T")
        errors.append("atm_data_type=monthly and calc_strair=T")

    if state["history_precision"] not in (4, 8):
        diag("(input_data) ERROR: bad value for history_precision, allowed values: 4, 8")
        errors.append(f"bad history_precision={state['history_precision']}")

    state["wave_spec"] = False
    if state["tr_fsd"] and state["wave_spec_type"] != "none":
        if state["wave_height_type"] != "none":
            state["wave_spec"] = True
        # wave_height_type=coupled is not yet implemented in CICE
        if state["wave_height_type"] not in ("none", "internal"):
            diag("WARNING: set wave_height_type=internal")
            errors.append("wave_height_type=coupled is not implemented")
    if state["tr_fsd"] and not state["wave_spec"]:
        diag("WARNING: tr_fsd=T but wave_spec=F - not recommended")
        if state["wave_height_type"] != "none":
            diag("WARNING: Wave_spec=F, wave_height_type/=none, wave_sig_ht = 0")

    if state["grid_ice"] == "C_override_D":
        diag("(input_data) using grid_ice = CD (C_override_D alias)")
        state["grid_ice"] = "CD"
    if state["grid_ice"] in ("C", "CD"):
        if state["kdyn"] > 1 or (
            state["kdyn"] == 1 and state["evp_algorithm"] != "standard_2d"
        ):
            diag(
                "(input_data) ERROR: grid_ice = C | CD only supported with "
                "kdyn=1 and evp_algorithm=standard_2d"
            )
            diag("(input_data) ERROR: kdyn and/or evp_algorithm and grid_ice inconsistency")
            errors.append("grid_ice = C | CD inconsistent with kdyn or evp_algorithm")
        if state["visc_method"] not in ("avg_zeta", "avg_strength"):
            diag("(input_data) ERROR: invalid method for viscosities")
            diag("(input_data) ERROR: visc_method should be avg_zeta or avg_strength")
            errors.append("invalid visc_method")

    for label, value in (
        ("grid_ice", state["grid_ice"]),
        ("grid_atm", state["grid_atm"]),
        ("grid_ocn", state["grid_ocn"]),
    ):
        if value == "A":
            dynu, dynv = "T", "T"
        elif value == "B":
            dynu, dynv = "U", "U"
        elif value == "C":
            dynu, dynv = "E", "N"
        elif value == "CD":
            dynu, dynv = "NE", "NE"
        else:
            diag(f"(input_data) ERROR: unknown {label}: {value}")
            errors.append(f"unknown {label}={value}")
            continue
        state[f"{label}_thrm"] = "T"
        state[f"{label}_dynu"] = dynu
        state[f"{label}_dynv"] = dynv

    for index in range(MAX_NSTRM):
        if state["histfreq_base"][index] not in ("init", "zero"):
            diag(
                "(input_data) ERROR: bad value for histfreq_base, allowed values: "
                f"init, zero: {state['histfreq_base'][index]}"
            )
            errors.append(f"bad histfreq_base={state['histfreq_base'][index]}")
        if state["dumpfreq_base"][index] not in ("init", "zero"):
            diag(
                "(input_data) ERROR: bad value for dumpfreq_base, allowed values: "
                f"init, zero: {state['dumpfreq_base'][index]}"
            )
            errors.append(f"bad dumpfreq_base={state['dumpfreq_base'][index]}")
        freq = state["dumpfreq"][index]
        first = freq[0] if freq else ""
        second = freq[1] if len(freq) > 1 else " "
        if first not in "ymdhx1YMDHX" or second not in ("1", " "):
            diag(f"(input_data) WARNING: unrecognized dumpfreq={freq}")
            diag("(input_data) WARNING:   No restarts files will be written for this stream")
            diag("(input_data) WARNING:   Allowed values : y,m,d,h,x,1 followed by an optional 1")
            state["dumpfreq"][index] = "x"

    if state["hist_time_axis"] not in ("begin", "middle", "end"):
        diag(f"(input_data) ERROR: hist_time_axis value not valid = {state['hist_time_axis']}")
        errors.append(f"bad hist_time_axis={state['hist_time_axis']}")

    # history_format not used by nuopc driver
    if state["history_format"] != "hdf5" and state["history_deflate"] != 0:
        diag(
            "(input_data) WARNING: history_deflate not compatible with "
            f"{state['history_format']}"
        )
        diag('(input_data) WARNING: netcdf compression only possible with history_type="hdf5" ')
    if state["history_format"] != "hdf5" and (
        state["history_chunksize"][0] != 0 or state["history_chunksize"][1] != 0
    ):
        diag(
            "(input_data) WARNING: history_chunksize not compatible with "
            f"{state['history_format']}"
        )
        diag('(input_data) WARNING: netcdf chunking only possible with history_type="hdf5" ')
    if state["restart_format"] != "hdf5" and state["restart_deflate"] != 0:
        diag(
            "(input_data) WARNING: restart_deflate not compatible with "
            f"{state['restart_format']}"
        )
        diag('(input_data) WARNING: netcdf compression only possible with restart_type="hdf5" ')
    if state["restart_format"] != "hdf5" and (
        state["restart_chunksize"][0] != 0 or state["restart_chunksize"][1] != 0
    ):
        diag(
            "(input_data) WARNING: restart_chunksize not compatible with "
            f"{state['restart_format']}"
        )
        diag('(input_data) WARNING: netcdf chunking only possible with restart_type="hdf5" ')
    if state["history_deflate"] < 0 or state["history_deflate"] > 9:
        diag(
            "(input_data) ERROR: history_deflate value not valid. "
            "Allowed range: integers from 0 to 9 "
        )
        errors.append(f"bad history_deflate={state['history_deflate']}")
    if state["restart_deflate"] < 0 or state["restart_deflate"] > 9:
        diag(
            "(input_data) ERROR: restart_deflate value not valid. "
            "Allowed range: integers from 0 to 9 "
        )
        errors.append(f"bad restart_deflate={state['restart_deflate']}")

    # Implicit solver input validation
    if state["kdyn"] == 3:
        for name in ("dim_fgmres", "dim_pgmres", "maxits_fgmres", "maxits_pgmres"):
            if state[name] < 1:
                errors.append(f"{name} must be positive")
        for name in ("maxits_nonlin", "dim_andacc", "start_andacc"):
            if state[name] < 0:
                errors.append(f"{name} must be nonnegative")
        if state["fpfunc_andacc"] != 1:
            errors.append("VP supports fpfunc_andacc=1; Fortran g_2(x) is unimplemented")
        if state["algo_nonlin"] not in ("picard", "anderson"):
            diag(f"(input_data) ERROR: unknown algo_nonlin: {state['algo_nonlin']}")
            diag("(input_data) ERROR:   allowed values: 'picard', 'anderson'")
            errors.append(f"unknown algo_nonlin={state['algo_nonlin']}")
        if state["algo_nonlin"] == "picard":
            # Picard solver is implemented in the Anderson solver; reset number of saved residuals to zero
            state["dim_andacc"] = 0
        if state["precond"] not in ("ident", "diag", "pgmres"):
            diag(f"(input_data) ERROR: unknown precond: {state['precond']}")
            diag("(input_data) ERROR:   allowed values: 'ident', 'diag', 'pgmres'")
            errors.append(f"unknown precond={state['precond']}")
        if state["ortho_type"] not in ("cgs", "mgs"):
            diag(f"(input_data) ERROR: unknown ortho_type: {state['ortho_type']}")
            diag("(input_data) ERROR:   allowed values: 'cgs', 'mgs'")
            errors.append(f"unknown ortho_type={state['ortho_type']}")

    # reprosum, lsum8, lsum16, and ddpdd are cross-process sums and are not ported.
    if state["bfbflag"].strip() != "off":
        diag(
            "(input_data) ERROR: bfbflag must be off, got "
            + state["bfbflag"].strip()
        )
        errors.append(f"bfbflag={state['bfbflag'].strip()} is not allowed")

    if state["orca_halogrid"]:
        diag("(input_data) ERROR: orca_halogrid has been deprecated")
        errors.append("orca_halogrid has been deprecated")
    if state["grid_type"] == "cpom_grid":
        diag("(input_data) ERROR: grid_type = 'cpom_grid' has been deprecated")
        errors.append("grid_type = cpom_grid has been deprecated")

    icepack_parameters.icepack_init_parameters(Cf=state["Cf"], ksno=state["ksno"])

    write_model_configuration(state, errors)

    if state["grid_format"] not in ("pop_nc", "pop_nc_ext", "mom_nc", "geosnc", "meshnc", "bin"):
        diag(f"(input_data) ERROR: unknown grid_format={state['grid_type']}")
        errors.append(f"unknown grid_format={state['grid_format']}")
    if state["grid_type"] not in (
        "displaced_pole",
        "tripole",
        "column",
        "rectangular",
        "regional",
        "latlon",
    ):
        diag(f"(input_data) ERROR: unknown grid_type={state['grid_type']}")
        errors.append(f"unknown grid_type={state['grid_type']}")
    if state["grid_ice"] not in ("B", "C", "CD"):
        diag(f"(input_data) ERROR: unknown grid_ice={state['grid_ice']}")
        errors.append(f"unknown grid_ice={state['grid_ice']}")
    if state["kmt_type"] not in (
        "file",
        "channel",
        "channel_oneeast",
        "channel_onenorth",
        "wall",
        "default",
        "boxislands",
        "none",
    ):
        diag(f"(input_data) ERROR: unknown kmt_type={state['kmt_type']}")
        errors.append(f"unknown kmt_type={state['kmt_type']}")
    if state["grid_type"] not in ("column", "rectangular") and state["kmt_type"] not in (
        "file",
        "none",
    ):
        diag(f"(input_data) ERROR: need kmt file, kmt_type={state['kmt_type']}")
        errors.append(f"need a kmt file, kmt_type={state['kmt_type']}")

    if errors:
        ice_fileunits.nu_diag.flush()
        raise ValueError("input_data: " + "; ".join(errors))

    globals().update(state)
    ice_flux.l_mpond_fresh = bool(state["l_mpond_fresh"])
    # use_smliq_pnd is checked above and is not passed into icepack.
    icepack_parameters.icepack_init_parameters(
        Cf=state["Cf"],
        ksno=state["ksno"],
        ustar_min=state["ustar_min"],
        albicev=state["albicev"],
        albicei=state["albicei"],
        albsnowv=state["albsnowv"],
        albsnowi=state["albsnowi"],
        natmiter=state["natmiter"],
        atmiter_conv=state["atmiter_conv"],
        emissivity=state["emissivity"],
        snw_ssp_table=state["snw_ssp_table"],
        hi_min=state["hi_min"],
        ahmax=state["ahmax"],
        shortwave=state["shortwave"],
        albedo_type=state["albedo_type"],
        R_ice=state["R_ice"],
        R_pnd=state["R_pnd"],
        R_snw=state["R_snw"],
        dT_mlt=state["dT_mlt"],
        rsnw_mlt=state["rsnw_mlt"],
        kstrength=state["kstrength"],
        krdg_partic=state["krdg_partic"],
        krdg_redist=state["krdg_redist"],
        mu_rdg=state["mu_rdg"],
        atmbndy=state["atmbndy"],
        calc_strair=state["calc_strair"],
        formdrag=state["formdrag"],
        highfreq=state["highfreq"],
        kitd=state["kitd"],
        kcatbound=state["kcatbound"],
        hs0=state["hs0"],
        dpscale=state["dpscale"],
        frzpnd=state["frzpnd"],
        rfracmin=state["rfracmin"],
        rfracmax=state["rfracmax"],
        pndaspect=state["pndaspect"],
        hs1=state["hs1"],
        hp1=state["hp1"],
        apnd_sl=state["apnd_sl"],
        itd_area_min=state["itd_area_min"],
        itd_mass_min=state["itd_mass_min"],
        ktherm=state["ktherm"],
        calc_Tsfc=state["calc_Tsfc"],
        conduct=state["conduct"],
        semi_implicit_Tsfc=state["semi_implicit_Tsfc"],
        a_rapid_mode=state["a_rapid_mode"],
        Rac_rapid_mode=state["Rac_rapid_mode"],
        vapor_flux_correction=state["vapor_flux_correction"],
        floediam=state["floediam"],
        hfrazilmin=state["hfrazilmin"],
        Tliquidus_max=state["Tliquidus_max"],
        aspect_rapid_mode=state["aspect_rapid_mode"],
        dSdt_slow_mode=state["dSdt_slow_mode"],
        phi_c_slow_mode=state["phi_c_slow_mode"],
        phi_i_mushy=state["phi_i_mushy"],
        conserv_check=state["conserv_check"],
        wave_spec_type=state["wave_spec_type"],
        wave_spec=state["wave_spec"],
        nfreq=state["nfreq"],
        wave_height_type=state["wave_height_type"],
        update_ocn_f=state["update_ocn_f"],
        cpl_frazil=state["cpl_frazil"],
        congel_freeze=state["congel_freeze"],
        tfrz_option=state["tfrz_option"],
        kalg=state["kalg"],
        fbot_xfer_type=state["fbot_xfer_type"],
        saltflux_option=state["saltflux_option"],
        ice_ref_salinity=state["ice_ref_salinity"],
        Pstar=state["Pstar"],
        Cstar=state["Cstar"],
        iceruf=state["iceruf"],
        iceruf_ocn=state["iceruf_ocn"],
        calc_dragio=state["calc_dragio"],
        windmin=state["windmin"],
        drhosdwind=state["drhosdwind"],
        rsnw_fall=state["rsnw_fall"],
        rsnw_tmax=state["rsnw_tmax"],
        rhosnew=state["rhosnew"],
        snwlvlfac=state["snwlvlfac"],
        rhosmin=state["rhosmin"],
        rhosmax=state["rhosmax"],
        snwredist=state["snwredist"],
        snwgrain=state["snwgrain"],
        snw_aging_table=state["snw_aging_table"],
        sw_redist=state["sw_redist"],
        sw_frac=state["sw_frac"],
        sw_dtemp=state["sw_dtemp"],
        tscale_pnd_drain=state["tscale_pnd_drain"],
    )
    icepack_tracers.icepack_init_tracer_flags(
        tr_iage=state["tr_iage"],
        tr_FY=state["tr_FY"],
        tr_lvl=state["tr_lvl"],
        tr_iso=state["tr_iso"],
        tr_aero=state["tr_aero"],
        tr_fsd=state["tr_fsd"],
        tr_snow=state["tr_snow"],
        tr_pond=state["tr_pond"],
        tr_pond_lvl=state["tr_pond_lvl"],
        tr_pond_sealvl=state["tr_pond_sealvl"],
        tr_pond_topo=state["tr_pond_topo"],
    )
    icepack_tracers.icepack_init_tracer_sizes(
        ncat=state["ncat"],
        nilyr=state["nilyr"],
        nslyr=state["nslyr"],
        nblyr=state["nblyr"],
        nfsd=state["nfsd"],
        n_algae=state["n_algae"],
        n_iso=state["n_iso"],
        n_aero=state["n_aero"],
        n_doc=state["n_doc"],
        n_don=state["n_don"],
        n_dic=state["n_dic"],
        n_fed=state["n_fed"],
        n_fep=state["n_fep"],
        n_zaero=state["n_zaero"],
    )
    icepack_parameters.icepack_write_parameters(ice_fileunits.nu_diag)
    ice_fileunits.nu_diag.flush()


def count_tracers():
    """Count tracers, then zero the driver aerosol count when aerosols are off.

    Icepack keeps the namelist ``n_aero`` (1 in gx3). CICE clears the separate
    driver copy because ``tr_aero`` is false.
    """
    icepack_tracers.count_tracers()
    global n_aero
    if not icepack_tracers.tr_aero:
        n_aero = 0
    # Printed indices are 1-based, matching count_tracers in ice_init_column.F90.
    # Stored nt_* stays 0-based.
    nu = ice_fileunits.nu_diag
    nu.write(" \n")
    rows = (
        (" ntrcr                     = ", int(icepack_tracers.ntrcr)),
        (" nbtrcr                    = ", int(icepack_tracers.nbtrcr)),
        (" nbtrcr_sw                 = ", int(icepack_tracers.nbtrcr_sw)),
        (" nt_sice                   = ", int(icepack_tracers.nt_sice) + 1),
        (" nt_qice                   = ", int(icepack_tracers.nt_qice) + 1),
        (" nt_qsno                   = ", int(icepack_tracers.nt_qsno) + 1),
    )
    nu.write(f"{rows[0][0]:<30}  {rows[0][1]:6d}\n")
    nu.write(f"{rows[1][0]:<30}  {rows[1][1]:6d}\n")
    nu.write(f"{rows[2][0]:<30}  {rows[2][1]:6d}\n")
    nu.write(" \n")
    nu.write(f"{rows[3][0]:<30}  {rows[3][1]:6d}\n")
    nu.write(f"{rows[4][0]:<30}  {rows[4][1]:6d}\n")
    nu.write(f"{rows[5][0]:<30}  {rows[5][1]:6d}\n")
    nu.write(" \n")
    icepack_tracers.icepack_write_tracer_flags(nu)
    icepack_tracers.icepack_write_tracer_sizes(nu)
    icepack_tracers.icepack_write_tracer_indices(nu)
    nu.flush()


def init_constants_tracers_namelist(nml_path="configuration/ice_in"):
    """``icepack_configure``, then ``input_data``, then ``count_tracers``."""
    icepack_intfc.icepack_configure()
    input_data(nml_path)
    ice_init_column.input_zbgc(nml_path)
    count_tracers()


globals().update(copy.deepcopy(namelist_defaults))


def init_state():
    """Initialize state for the itd model.

    authors: C. M. Bitz, UW
             William H. Lipscomb, LANL
    """
    subname = "(init_state)"
    nilyr = int(ice_domain_size.nilyr)
    nslyr = int(ice_domain_size.nslyr)
    ntrcr = int(icepack_tracers.ntrcr)
    nt_Tsfc = int(icepack_tracers.nt_Tsfc)
    nt_sice = int(icepack_tracers.nt_sice)
    nt_qice = int(icepack_tracers.nt_qice)
    nt_qsno = int(icepack_tracers.nt_qsno)
    nt_iage = int(icepack_tracers.nt_iage)
    nt_FY = int(icepack_tracers.nt_FY)
    nt_alvl = int(icepack_tracers.nt_alvl)
    nt_vlvl = int(icepack_tracers.nt_vlvl)
    nt_apnd = int(icepack_tracers.nt_apnd)
    nt_hpnd = int(icepack_tracers.nt_hpnd)
    nt_ipnd = int(icepack_tracers.nt_ipnd)
    nt_smice = int(icepack_tracers.nt_smice)
    nt_smliq = int(icepack_tracers.nt_smliq)
    nt_rhos = int(icepack_tracers.nt_rhos)
    nt_rsnw = int(icepack_tracers.nt_rsnw)
    nt_isosno = int(icepack_tracers.nt_isosno)
    nt_isoice = int(icepack_tracers.nt_isoice)
    nt_aero = int(icepack_tracers.nt_aero)
    nt_fsd = int(icepack_tracers.nt_fsd)
    tr_iage = bool(icepack_tracers.tr_iage)
    tr_FY = bool(icepack_tracers.tr_FY)
    tr_lvl = bool(icepack_tracers.tr_lvl)
    tr_iso = bool(icepack_tracers.tr_iso)
    tr_aero = bool(icepack_tracers.tr_aero)
    tr_pond_lvl = bool(icepack_tracers.tr_pond_lvl)
    tr_pond_topo = bool(icepack_tracers.tr_pond_topo)
    tr_pond_sealvl = bool(icepack_tracers.tr_pond_sealvl)
    tr_snow = bool(icepack_tracers.tr_snow)
    tr_fsd = bool(icepack_tracers.tr_fsd)
    # call icepack_warnings_flush / abort_ice are not ported.

    #-----------------------------------------------------------------
    # Check number of layers in ice and snow.
    #-----------------------------------------------------------------
    # if (my_task == master_task): one process, so the check always runs.
    nu = ice_fileunits.nu_diag
    if nilyr < 1:
        nu.write(subname + " ERROR: Must have at least one ice layer\n")
        nu.write(subname + f" ERROR:   nilyr = {nilyr}\n")
        nu.flush()
        raise ValueError(subname + " Not enough ice layers")
    if nslyr < 1:
        nu.write(subname + " ERROR: Must have at least one snow layer\n")
        nu.write(subname + f" ERROR:   nslyr = {nslyr}\n")
        nu.flush()
        raise ValueError(subname + " Not enough snow layers")
    if nt_Tsfc < 0:
        raise ValueError(subname + " nt_Tsfc is not set")

    #-----------------------------------------------------------------
    # Set tracer types
    #-----------------------------------------------------------------
    depend = ice_state.trcr_depend
    base = ice_state.trcr_base
    strata_n = ice_state.n_trcr_strata
    strata = ice_state.nt_strata
    base[...] = c0

    # 0, 1, and 2 stay the area, ice-volume, and snow-volume kinds.
    # A parent tracer is stored as 3 + its 0-based index, which is
    # Fortran's 2 + the 1-based index.
    depend[nt_Tsfc] = 0  # ice/snow surface temperature
    for k in range(nilyr):
        depend[nt_sice + k] = 1  # volume-weighted ice salinity
        depend[nt_qice + k] = 1  # volume-weighted ice enthalpy
    for k in range(nslyr):
        depend[nt_qsno + k] = 2  # volume-weighted snow enthalpy
    if tr_iage:
        depend[nt_iage] = 1   # volume-weighted ice age
    if tr_FY:
        depend[nt_FY] = 0     # area-weighted first-year ice area
    if tr_lvl:
        depend[nt_alvl] = 0   # level ice area
    if tr_lvl:
        depend[nt_vlvl] = 1   # level ice volume
    if tr_pond_lvl:
        depend[nt_apnd] = 3 + nt_alvl   # melt pond area
        depend[nt_hpnd] = 3 + nt_apnd   # melt pond depth
        depend[nt_ipnd] = 3 + nt_apnd   # refrozen pond lid
    if tr_pond_topo or tr_pond_sealvl:
        depend[nt_apnd] = 0             # melt pond area
        depend[nt_hpnd] = 3 + nt_apnd   # melt pond depth
        depend[nt_ipnd] = 3 + nt_apnd   # refrozen pond lid
    if tr_snow:  # snow-volume-weighted snow tracers
        for k in range(nslyr):
            depend[nt_smice + k] = 2  # ice mass in snow
            depend[nt_smliq + k] = 2  # liquid mass in snow
            depend[nt_rhos + k] = 2   # effective snow density
            depend[nt_rsnw + k] = 2   # snow radius
    if tr_fsd:
        for it in range(int(ice_domain_size.nfsd)):
            depend[nt_fsd + it] = 0  # area-weighted floe size distribution
    if tr_iso:  # isotopes
        for it in range(int(n_iso)):
            depend[nt_isosno + it] = 2  # snow
            depend[nt_isoice + it] = 1  # ice
    if tr_aero:  # volume-weighted aerosols
        for it in range(int(n_aero)):
            depend[nt_aero + it * 4] = 2      # snow
            depend[nt_aero + it * 4 + 1] = 2  # snow
            depend[nt_aero + it * 4 + 2] = 1  # ice
            depend[nt_aero + it * 4 + 3] = 1  # ice

    for it in range(ntrcr):
        # mask for base quantity on which tracers are carried
        kind = int(depend[it])
        if kind == 0:      # area
            base[it, 0] = c1
        elif kind == 1:    # ice volume
            base[it, 1] = c1
        elif kind == 2:    # snow volume
            base[it, 2] = c1
        else:
            base[it, 0] = c1  # default: ice area
            base[it, 1] = c0
            base[it, 2] = c0

        # initialize number of underlying tracer layers
        strata_n[it] = 0
        # default indices of underlying tracer layers
        # -1 is unused. 0 is surface temperature.
        strata[it, 0] = -1
        strata[it, 1] = -1

    if tr_pond_lvl:
        strata_n[nt_apnd] = 1        # melt pond area
        strata[nt_apnd, 0] = nt_alvl  # on level ice area
        strata_n[nt_hpnd] = 2        # melt pond depth
        strata[nt_hpnd, 1] = nt_apnd  # on melt pond area
        strata[nt_hpnd, 0] = nt_alvl  # on level ice area
        strata_n[nt_ipnd] = 2        # refrozen pond lid
        strata[nt_ipnd, 1] = nt_apnd  # on melt pond area
        strata[nt_ipnd, 0] = nt_alvl  # on level ice area
    if tr_pond_topo or tr_pond_sealvl:
        strata_n[nt_hpnd] = 1        # melt pond depth
        strata[nt_hpnd, 0] = nt_apnd  # on melt pond area
        strata_n[nt_ipnd] = 1        # refrozen pond lid
        strata[nt_ipnd, 0] = nt_apnd  # on melt pond area

    #-----------------------------------------------------------------
    # Set state variables
    #-----------------------------------------------------------------
    set_state_var()

    #-----------------------------------------------------------------
    # ghost cell updates
    #-----------------------------------------------------------------
    ice_state.bound_state(
        ice_state.aicen,
        ice_state.vicen,
        ice_state.vsnon,
        ntrcr,
        ice_state.trcrn,
    )

    grid = str(grid_ice).strip()
    if grid in ("CD", "C"):
        ice_grid.grid_average_X2Y("A", ice_state.uvel, "U", ice_state.uvelN, "N")
        ice_grid.grid_average_X2Y("A", ice_state.vvel, "U", ice_state.vvelN, "N")
        ice_grid.grid_average_X2Y("A", ice_state.uvel, "U", ice_state.uvelE, "E")
        ice_grid.grid_average_X2Y("A", ice_state.vvel, "U", ice_state.vvelE, "E")

        # Halo update on North, East faces
        ice_halo_update(
            ice_state.uvelN, field_loc_Nface, field_type_scalar, fill_value=c0
        )
        ice_halo_update(
            ice_state.vvelN, field_loc_Nface, field_type_scalar, fill_value=c0
        )
        ice_halo_update(
            ice_state.uvelE, field_loc_Eface, field_type_scalar, fill_value=c0
        )
        ice_halo_update(
            ice_state.vvelE, field_loc_Eface, field_type_scalar, fill_value=c0
        )

    #-----------------------------------------------------------------
    # compute aggregate ice state and open water area
    #-----------------------------------------------------------------
    ice_state.aice[...] = c0
    ice_state.vice[...] = c0
    ice_state.vsno[...] = c0
    ice_state.trcr[...] = c0
    tmask = ice_grid.tmask
    ny = ice_state.aice.shape[1]
    nx = ice_state.aice.shape[0]
    for j in range(ny):
        for i in range(nx):
            if tmask[i, j]:
                (
                    ice_state.aice[i, j],
                    ice_state.trcr[i, j, :],
                    ice_state.vice[i, j],
                    ice_state.vsno[i, j],
                    ice_state.aice0[i, j],
                ) = icepack_itd.icepack_aggregate(
                    aicen=ice_state.aicen[i, j, :],
                    trcrn=ice_state.trcrn[i, j, :, :],
                    vicen=ice_state.vicen[i, j, :],
                    vsnon=ice_state.vsnon[i, j, :],
                    trcr_depend=depend,
                    trcr_base=base,
                    n_trcr_strata=strata_n,
                    nt_strata=strata,
                    Tf=ice_flux.Tf[i, j],
                )
            ice_state.aice_init[i, j] = ice_state.aice[i, j]
    # call icepack_warnings_flush / abort_ice are not ported.


def set_state_var():
    """Initialize state in each ice thickness category.

    authors: C. M. Bitz
             William H. Lipscomb, LANL

    One grid. The Fortran block bounds and per-block slices are the whole
    domain, including ghost cells where the Fortran loop is ``1:nx_block``.
    """
    subname = "(set_state_var)"
    nilyr = int(ice_domain_size.nilyr)
    nslyr = int(ice_domain_size.nslyr)
    ncat = int(ice_domain_size.ncat)
    nx = int(ice_domain_size.nx_global)
    ny = int(ice_domain_size.ny_global)
    ntrcr = int(icepack_tracers.ntrcr)
    nt_Tsfc = int(icepack_tracers.nt_Tsfc)
    nt_qice = int(icepack_tracers.nt_qice)
    nt_qsno = int(icepack_tracers.nt_qsno)
    nt_sice = int(icepack_tracers.nt_sice)
    nt_fbri = int(icepack_tracers.nt_fbri)
    nt_alvl = int(icepack_tracers.nt_alvl)
    nt_vlvl = int(icepack_tracers.nt_vlvl)
    nt_smice = int(icepack_tracers.nt_smice)
    nt_smliq = int(icepack_tracers.nt_smliq)
    nt_rhos = int(icepack_tracers.nt_rhos)
    nt_rsnw = int(icepack_tracers.nt_rsnw)
    tr_brine = bool(icepack_tracers.tr_brine)
    tr_lvl = bool(icepack_tracers.tr_lvl)
    tr_snow = bool(icepack_tracers.tr_snow)
    rhos = icepack_parameters.rhos
    Lfresh = icepack_parameters.Lfresh
    puny = icepack_parameters.puny
    rad_to_deg = icepack_parameters.rad_to_deg
    rsnw_fall = icepack_parameters.rsnw_fall
    # call icepack_warnings_flush / abort_ice are not ported.

    # initial snow thickness (m)
    hsno_init = dbl_kind(0.20)
    edge_init_nh = dbl_kind(70.0)   # initial ice edge, N.Hem. (deg)
    edge_init_sh = dbl_kind(-60.0)  # initial ice edge, S.Hem. (deg)

    aicen = ice_state.aicen
    vicen = ice_state.vicen
    vsnon = ice_state.vsnon
    trcrn = ice_state.trcrn
    tmask = ice_grid.tmask
    Tf = ice_flux.Tf
    salinz = ice_flux.salinz
    Tmltz = ice_flux.Tmltz

    # Initialize state variables.
    # If restarting, these values are overwritten.
    aicen[...] = c0
    vicen[...] = c0
    vsnon[...] = c0
    # surface temperature
    trcrn[:, :, nt_Tsfc, :] = np.where(tmask[:, :, None], Tf[:, :, None], c0)
    if ntrcr >= 2:
        # tracers 2:ntrcr. This also clears Tsfc when nt_Tsfc is not 0.
        trcrn[:, :, 1:, :] = c0
    if tr_lvl:
        trcrn[:, :, nt_alvl, :] = c1
    if tr_lvl:
        trcrn[:, :, nt_vlvl, :] = c1
    if tr_brine:
        trcrn[:, :, nt_fbri, :] = c1
    for k in range(nilyr):
        trcrn[:, :, nt_sice + k, :] = salinz[:, :, k, None]
    for k in range(nslyr):
        trcrn[:, :, nt_qsno + k, :] = -rhos * Lfresh
    if tr_snow:
        for k in range(nslyr):
            trcrn[:, :, nt_rsnw + k, :] = rsnw_fall
            trcrn[:, :, nt_rhos + k, :] = rhos
            trcrn[:, :, nt_smice + k, :] = rhos
            trcrn[:, :, nt_smliq + k, :] = c0

    ice_ic_name = str(ice_ic).strip()
    if ice_ic_name != "internal":
        return

    #---------------------------------------------------------
    # ice concentration/thickness
    #---------------------------------------------------------
    ice_data_conc_name = str(ice_data_conc).strip()
    hinit = np.zeros(ncat, dtype=dbl_kind)
    ainit = np.zeros(ncat, dtype=dbl_kind)
    hin_max = ice_arrays_column.hin_max
    if ice_data_conc_name in ("p5", "p8", "p9", "c1", "box2001"):
        if ice_data_conc_name == "p5":
            hbar = c2  # initial ice thickness
            abar = p5  # initial ice concentration
        elif ice_data_conc_name == "p8":
            hbar = c1  # initial ice thickness
            abar = dbl_kind(0.8)  # initial ice concentration
        elif ice_data_conc_name == "p9":
            hbar = c1  # initial ice thickness
            abar = dbl_kind(0.9)  # initial ice concentration
        elif ice_data_conc_name == "c1":
            hbar = c1  # initial ice thickness
            abar = c1  # initial ice concentration
        else:  # box2001
            hbar = c2  # initial ice thickness
            abar = p5  # initial ice concentration
        for n in range(1, ncat + 1):
            hinit[n - 1] = c0
            ainit[n - 1] = c0
            if hbar > hin_max[n - 1] and hbar <= hin_max[n]:
                hinit[n - 1] = hbar
                ainit[n - 1] = abar
    elif ice_data_conc_name == "parabolic":
        # initial category areas in cells with ice
        hbar = c3  # initial ice thickness with greatest area
                   # Note: the resulting average ice thickness
                   # tends to be less than hbar due to the
                   # nonlinear distribution of ice thicknesses
        asum = c0
        for n in range(1, ncat + 1):
            if n < ncat:
                hinit[n - 1] = p5 * (hin_max[n - 1] + hin_max[n])  # m
            else:  # n=ncat
                hinit[n - 1] = hin_max[n - 1] + c1  # m
            # parabola, max at h=hbar, zero at h=0, 2*hbar
            ainit[n - 1] = max(c0, (c2 * hbar * hinit[n - 1] - hinit[n - 1] ** 2))
            asum = asum + ainit[n - 1]
        for n in range(ncat):
            ainit[n] = ainit[n] / (asum + puny / ncat)  # normalize
    else:
        raise ValueError(
            subname + "ERROR: ice_data_conc setting = " + ice_data_conc_name
        )

    #---------------------------------------------------------
    # location of ice
    #---------------------------------------------------------
    ilo, ihi, jlo, jhi = ice_domain_size.physical_bounds()
    ig, jg = global_indices()
    ice_data_type_name = str(ice_data_type).strip()
    cells = []

    if ice_data_type_name == "boxslotcyl":
        # Geometric configuration of the slotted cylinder
        dx_rect = dbl_kind(dxrect)
        dy_rect = dbl_kind(dyrect)
        diam = p3 * dx_rect * dbl_kind(nx - 1)
        center_x = p5 * dx_rect * dbl_kind(nx - 1)
        center_y = p75 * dy_rect * dbl_kind(ny - 1)
        radius = p5 * diam
        width = p166 * diam
        length = c5 * p166 * diam
        for j in range(jlo, jhi + 1):
            for i in range(ilo, ihi + 1):
                if tmask[i, j]:
                    # check if grid point is inside slotted cylinder
                    x = dx_rect * dbl_kind(int(ig[i]) - 1)
                    y = dy_rect * dbl_kind(int(jg[j]) - 1)
                    in_slot = (
                        (x >= center_x - width / c2)
                        and (x <= center_x + width / c2)
                        and (y >= center_y - radius)
                        and (y <= center_y + (length - radius))
                    )
                    in_cyl = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2) <= radius
                    if in_cyl and not in_slot:
                        cells.append((i, j))
    elif ice_data_type_name in ("uniform", "box2001"):
        # all cells not land mask are ice
        # box2001 used to have a check for west of 50W, this was changed, so now box2001 is
        # the same as uniform.  keep box2001 option for backwards compatibility.
        for j in range(jlo, jhi + 1):
            for i in range(ilo, ihi + 1):
                if tmask[i, j]:
                    cells.append((i, j))
    elif ice_data_type_name[:7] == "channel":
        # channel ice in center of domain in i direction
        # The j upper bound uses nx_global, as written in the Fortran.
        for j in range(jlo, jhi + 1):
            for i in range(ilo, ihi + 1):
                if int(jg[j]) > ny // 4 and int(jg[j]) < (3 * nx) // 4:
                    cells.append((i, j))
    elif ice_data_type_name == "block":
        # ice in 50% of domain, not at edges
        iedge = int(dbl_kind(nx) * 0.25) + 1
        jedge = int(dbl_kind(ny) * 0.25) + 1
        for j in range(jlo, jhi + 1):
            for i in range(ilo, ihi + 1):
                if (int(ig[i]) > iedge and int(ig[i]) < nx - iedge + 1) and (
                    int(jg[j]) > jedge and int(jg[j]) < ny - jedge + 1
                ):
                    cells.append((i, j))
    elif ice_data_type_name == "eastblock":
        # block on east half of domain in center of domain
        # The j upper bound uses nx_global, as written in the Fortran.
        for j in range(jlo, jhi + 1):
            for i in range(ilo, ihi + 1):
                if (
                    int(jg[j]) > ny // 4
                    and int(jg[j]) < (3 * nx) // 4
                    and int(ig[i]) >= nx // 2
                ):
                    cells.append((i, j))
    elif ice_data_type_name == "latsst":
        #-----------------------------------------------------------------
        # Place ice where ocean surface is cold.
        # Note: If SST is not read from a file, then the ocean is assumed
        #       to be at its freezing point everywhere, and ice will
        #       extend to the prescribed edges.
        #-----------------------------------------------------------------
        # ifdef CESMCOUPLED
        # Option to use Tair instead.
        # if (Tair(i,j) <= Tffresh)
        # else: standalone uses sst
        sst = ice_flux.sst
        TLAT = ice_grid.TLAT
        for j in range(jlo, jhi + 1):
            for i in range(ilo, ihi + 1):
                if tmask[i, j]:
                    # place ice in high latitudes where ocean sfc is cold
                    if (sst[i, j] <= Tf[i, j] + p2) and (
                        TLAT[i, j] < edge_init_sh / rad_to_deg
                        or TLAT[i, j] > edge_init_nh / rad_to_deg
                    ):
                        cells.append((i, j))
    else:
        raise ValueError(
            subname + "ERROR: ice_data_type setting = " + ice_data_type_name
        )

    #---------------------------------------------------------
    # ice distribution
    #---------------------------------------------------------
    ice_data_dist_name = str(ice_data_dist).strip()
    for n in range(ncat):
        # ice volume, snow volume
        for i, j in cells:
            aicen[i, j, n] = ainit[n]
            if ice_data_dist_name == "box2001":
                if hinit[n] > c0:
                    # varies linearly from 0 to 1 in x direction
                    aicen[i, j, n] = (dbl_kind(int(ig[i])) - p5) / dbl_kind(nx)
                    # constant slope from 0 to 0.5 in x direction
                    # aicen(i,j,n) = (real(iglob(i), kind=dbl_kind)-p5)
                    #              / (real(nx_global,kind=dbl_kind)) * p5
                    # quadratic
                    # aicen(i,j,n) = max(c0,(real(iglob(i), kind=dbl_kind)-p5)
                    #                     / (real(nx_global,kind=dbl_kind))
                    #                     * (real(jglob(j), kind=dbl_kind)-p5)
                    #                     / (real(ny_global,kind=dbl_kind)) * p5)
                    # aicen(i,j,n) = max(c0,(real(nx_global, kind=dbl_kind)
                    #                     -  real(iglob(i), kind=dbl_kind)-p5)
                    #                     / (real(nx_global,kind=dbl_kind))
                    #                     * (real(ny_global, kind=dbl_kind)
                    #                     -  real(jglob(j), kind=dbl_kind)-p5)
                    #                     / (real(ny_global,kind=dbl_kind)) * p5)
            elif ice_data_dist_name == "gauss":
                if hinit[n] > c0:
                    dist_ratio = dbl_kind(8.0) * np.sqrt(
                        (dbl_kind(int(ig[i])) - dbl_kind(nx + 1) / c2) ** 2
                        + (dbl_kind(int(jg[j])) - dbl_kind(ny + 1) / c2) ** 2
                    ) / np.sqrt(dbl_kind(nx) ** 2 + dbl_kind(ny) ** 2)
                    aicen[i, j, n] = ainit[n] * np.exp(-dist_ratio)
            elif ice_data_dist_name == "uniform":
                # nothing extra to do
                pass
            else:
                raise ValueError(
                    subname + "ERROR: ice_data_dist setting = " + ice_data_dist_name
                )

            vicen[i, j, n] = hinit[n] * aicen[i, j, n]  # m
            vsnon[i, j, n] = min(aicen[i, j, n] * hsno_init, p2 * vicen[i, j, n])

            Tsfc, qin, qsn = icepack_therm_shared.icepack_init_trcr(
                Tair=ice_flux.Tair[i, j],
                Tf=Tf[i, j],
                Sprofile=salinz[i, j, :],
                Tprofile=Tmltz[i, j, :],
            )
            # surface temperature
            trcrn[i, j, nt_Tsfc, n] = Tsfc  # deg C
            # ice enthalpy, salinity
            for k in range(nilyr):
                trcrn[i, j, nt_qice + k, n] = qin[k]
                trcrn[i, j, nt_sice + k, n] = salinz[i, j, k]
            # snow enthalpy
            for k in range(nslyr):
                trcrn[i, j, nt_qsno + k, n] = qsn[k]
            # brine fraction
            if tr_brine:
                trcrn[i, j, nt_fbri, n] = c1

    #---------------------------------------------------------
    # ice velocity
    # these velocites are defined on B-grid
    #---------------------------------------------------------
    if ice_data_type_name == "boxslotcyl":
        secday = icepack_parameters.secday
        pi = icepack_parameters.pi
        domain_length = dbl_kind(dxrect) * cm_to_m * dbl_kind(nx)
        period = c12 * secday  # 12 days rotational period
        max_vel = pi * domain_length / period
        umask = ice_grid.umask
        j_term = (
            c2 * max_vel * (dbl_kind(jg) - p5) / dbl_kind(ny - 1) - max_vel
        )
        i_term = (
            -c2 * max_vel * (dbl_kind(ig) - p5) / dbl_kind(nx - 1) + max_vel
        )
        u_full = np.broadcast_to(j_term[np.newaxis, :], ice_state.uvel.shape)
        v_full = np.broadcast_to(i_term[:, np.newaxis], ice_state.vvel.shape)
        ice_state.uvel[...] = np.where(umask, u_full, c0)
        ice_state.vvel[...] = np.where(umask, v_full, c0)
    else:
        ice_state.uvel[...] = c0
        ice_state.vvel[...] = c0
