"""Elastic-viscous-plastic sea ice dynamics model code shared with other
approaches

author for JAX version: Xinyuan Zhu
XZ: converted from F90 to Python

author: Elizabeth C. Hunke, LANL

2013: Split from ice_dyn_evp.F90 by Elizabeth Hunke

One grid, no block index. Namelist values live on ``ice_init``
until ``init_dyn_shared`` copies them here, which is where the Fortran
module holds them.

Ported: ``alloc_dyn_shared``, ``init_dyn_shared``, ``set_evp_parameters``,
``principal_stress`` (called by accum_hist), ``visc_replpress``,
the B/C/CD-grid preparation, momentum and strain formulas, and both
seabed-stress schemes. C/CD deformation completion lives in ice_dyn_evp_cgrid.
JAX halo updates replace dyn_haloUpdate and the field-bundling helpers.
"""

import numpy as np
import jax
import jax.numpy as jnp

from cicecore.cicedyn.general import ice_flux
from cicecore.cicedyn.general import ice_init
from cicecore.cicedyn.general import ice_state
from cicecore.cicedyn.infrastructure import ice_domain
from cicecore.cicedyn.infrastructure import ice_grid
from cicecore.cicedyn.infrastructure.comm.serial.ice_boundary import (
    ice_halo_update,
)
from cicecore.shared import ice_domain_size
from cicecore.shared import ice_fileunits
from cicecore.shared.ice_constants import (
    c0,
    c1,
    c1p5,
    c2,
    c4,
    field_loc_center,
    field_type_vector,
    omega,
    p5,
    spval_dbl,
)
from cicecore.shared.ice_kinds_mod import dbl_kind, log_kind
from icepack.columnphysics import icepack_parameters

# namelist parameters. Filled from ice_init by init_dyn_shared.

kdyn = None        # type of dynamics ( -1, 0 = off, 1 = evp, 2 = eap, 3 = vp )
kridge = None      # set to "-1" to turn off ridging
ndte = None        # number of subcycles
grid_ice = None    # 'B', 'C', or 'CD'. Namelist value, also read by the grid.

coriolis = None    # 'constant', 'zero', or 'latitude'
ssh_stress = None  # 'geostrophic' or 'coupled'

revised_evp = None  # if true, use revised evp procedure

evp_algorithm = None  # standard_2d = 2D org version (standard)
                      # shared_mem_1d = 1d without mpi call and refactorization to 1d

dyn_area_min = None  # minimum ice area concentration to activate dynamics
dyn_mass_min = None  # minimum ice mass to activate dynamics (kg/m^2)
elasticDamp = None   # coefficient for calculating the parameter E, elastic damping parameter

# other EVP parameters

yield_curve = None       # 'ellipse' ('teardrop' needs further testing)
visc_method = None       # method for viscosity calc at U points (C, CD grids)
seabed_stress_method = None  # method for seabed stress calculation
                             # LKD: Lemieux et al. 2015, probabilistic: Dupont et al. 2022

u0 = dbl_kind(5e-5)  # residual velocity for seabed stress (m/s)
cosw = c1            # cos(ocean turning angle)  ! turning angle = 0
sinw = c0            # sin(ocean turning angle)  ! turning angle = 0

revp = None          # 0 for classic EVP, 1 for revised EVP
e_yieldcurve = None  # VP aspect ratio of elliptical yield curve
e_plasticpot = None  # VP aspect ratio of elliptical plastic potential
epp2i = None         # 1/(e_plasticpot)^2
e_factor = None      # (e_yieldcurve)^2/(e_plasticpot)^4
ecci = None          # temporary for 1d evp
deltaminEVP = None   # minimum delta for viscosities (EVP)
deltaminVP = None    # minimum delta for viscosities (VP)
capping = None       # capping of viscosities (1=Hibler79, 0=Kreyscher2000)
dtei = None          # ndte/dt, where dt/ndte is subcycling timestep (1/s)
denom1 = None        # constants for stress equation

# Bouillon et al relaxation constants
arlx = None    # alpha for stressp
arlx1i = None  # (inverse of alpha) for stressp
brlx = None    # beta   for momentum

fcor_blk = None  # Coriolis parameter (1/s)

fcorE_blk = None  # Coriolis parameter at E points (1/s)
fcorN_blk = None  # Coriolis parameter at N points  (1/s)

fld2 = None  # 2 bundled fields
fld3 = None  # 3 bundled fields
fld4 = None  # 4 bundled fields

uvel_init = None  # x-component of velocity (m/s), beginning of timestep
vvel_init = None  # y-component of velocity (m/s), beginning of timestep

uvelN_init = None  # x-component of velocity (m/s), beginning of timestep
vvelN_init = None  # y-component of velocity (m/s), beginning of timestep

uvelE_init = None  # x-component of velocity (m/s), beginning of timestep
vvelE_init = None  # y-component of velocity (m/s), beginning of timestep

iceTmask = None  # ice extent mask (T-cell)
iceUmask = None  # ice extent mask (U-cell)
iceNmask = None  # ice extent mask (N-cell)
iceEmask = None  # ice extent mask (E-cell)

DminTarea = None  # deltamin * tarea (m^2/s)

cyp = None   # 1.5*HTE(i,j)-0.5*HTW(i,j) = 1.5*HTE(i,j)-0.5*HTE(i-1,j)
cxp = None   # 1.5*HTN(i,j)-0.5*HTS(i,j) = 1.5*HTN(i,j)-0.5*HTN(i,j-1)
cym = None   # 0.5*HTE(i,j)-1.5*HTW(i,j) = 0.5*HTE(i,j)-1.5*HTE(i-1,j)
cxm = None   # 0.5*HTN(i,j)-1.5*HTS(i,j) = 0.5*HTN(i,j)-1.5*HTN(i,j-1)
dxhy = None  # 0.5*(HTE(i,j) - HTW(i,j)) = 0.5*(HTE(i,j) - HTE(i-1,j))
dyhx = None  # 0.5*(HTN(i,j) - HTS(i,j)) = 0.5*(HTN(i,j) - HTN(i,j-1))

# ice isotropic tensile strength parameter
Ktens = None  # T=Ktens*P (tensile strength: see Konig and Holland, 2010)

# seabed (basal) stress parameters and settings
seabed_stress = None  # if true, seabed stress for landfast on

k1 = None            # 1st free parameter for seabed1 grounding parameterization
k2 = None            # second free parameter (N/m^3) for seabed1 grounding parametrization
alphab = None        # alphab=Cb factor in Lemieux et al 2015
threshold_hw = None  # max water depth for grounding
                     # see keel data from Amundrud et al. 2004 (JGR)

# Set in init_dyn_shared. Empirical; not a namelist value.
halo_dynbundle = None


def _load_namelist():
    """Copy the dynamics namelist from ``ice_init`` into this module."""
    global kdyn, kridge, ndte, grid_ice, coriolis, ssh_stress, revised_evp, evp_algorithm
    global dyn_area_min, dyn_mass_min, elasticDamp
    global yield_curve, visc_method, seabed_stress_method
    global e_yieldcurve, e_plasticpot, deltaminEVP, deltaminVP, capping
    global Ktens, seabed_stress, k1, k2, alphab, threshold_hw
    global arlx, brlx
    kdyn = int(ice_init.kdyn)
    kridge = int(ice_init.kridge)
    ndte = int(ice_init.ndte)
    grid_ice = str(ice_init.grid_ice).strip()
    coriolis = str(ice_init.coriolis).strip()
    ssh_stress = str(ice_init.ssh_stress).strip()
    revised_evp = bool(ice_init.revised_evp)
    evp_algorithm = str(ice_init.evp_algorithm).strip()
    dyn_area_min = dbl_kind(ice_init.dyn_area_min)
    dyn_mass_min = dbl_kind(ice_init.dyn_mass_min)
    elasticDamp = dbl_kind(ice_init.elasticDamp)
    yield_curve = str(ice_init.yield_curve).strip()
    visc_method = str(ice_init.visc_method).strip()
    seabed_stress_method = str(ice_init.seabed_stress_method).strip()
    e_yieldcurve = dbl_kind(ice_init.e_yieldcurve)
    e_plasticpot = dbl_kind(ice_init.e_plasticpot)
    deltaminEVP = dbl_kind(ice_init.deltaminEVP)
    deltaminVP = dbl_kind(ice_init.deltaminVP)
    capping = dbl_kind(ice_init.capping)
    Ktens = dbl_kind(ice_init.Ktens)
    seabed_stress = log_kind(ice_init.seabed_stress)
    k1 = dbl_kind(ice_init.k1)
    k2 = dbl_kind(ice_init.k2)
    alphab = dbl_kind(ice_init.alphab)
    threshold_hw = dbl_kind(ice_init.threshold_hw)
    arlx = dbl_kind(ice_init.arlx)
    brlx = dbl_kind(ice_init.brlx)


def _c_or_cd():
    return str(grid_ice).strip() in ("C", "CD")


def _zero(shape):
    return np.zeros(shape, dtype=dbl_kind)


def alloc_dyn_shared():
    """Allocate space for all variables."""
    global uvel_init, vvel_init, iceTmask, iceUmask, fcor_blk, DminTarea
    global fld2, fld3, fld4
    global cyp, cxp, cym, cxm, dxhy, dyhx
    global uvelE_init, vvelE_init, uvelN_init, vvelN_init
    global iceEmask, iceNmask, fcorE_blk, fcorN_blk

    subname = "(alloc_dyn_shared)"
    horiz = ice_domain_size.horizontal_shape()
    if horiz[0] < 1 or horiz[1] < 1:
        raise ValueError(f"{subname}: horizontal size is not set")

    uvel_init = _zero(horiz)  # x-component of velocity (m/s), beginning of timestep
    vvel_init = _zero(horiz)  # y-component of velocity (m/s), beginning of timestep
    iceTmask = np.zeros(horiz, dtype=bool)  # T mask for dynamics
    iceUmask = np.zeros(horiz, dtype=bool)  # U mask for dynamics
    fcor_blk = _zero(horiz)  # Coriolis
    DminTarea = _zero(horiz)

    fld2 = _zero(horiz + (2,))
    fld3 = _zero(horiz + (3,))
    fld4 = _zero(horiz + (4,))

    cyp = _zero(horiz)  # 1.5*HTE - 0.5*HTW
    cxp = _zero(horiz)  # 1.5*HTN - 0.5*HTS
    cym = _zero(horiz)  # 0.5*HTE - 1.5*HTW
    cxm = _zero(horiz)  # 0.5*HTN - 1.5*HTS

    dxhy = None
    dyhx = None
    if str(grid_ice).strip() == "B":
        dxhy = _zero(horiz)  # 0.5*(HTE - HTW)
        dyhx = _zero(horiz)  # 0.5*(HTN - HTS)

    uvelE_init = None
    vvelE_init = None
    uvelN_init = None
    vvelN_init = None
    iceEmask = None
    iceNmask = None
    fcorE_blk = None
    fcorN_blk = None
    if _c_or_cd():
        uvelE_init = _zero(horiz)  # x-component of velocity (m/s), beginning of timestep
        vvelE_init = _zero(horiz)  # y-component of velocity (m/s), beginning of timestep
        uvelN_init = _zero(horiz)  # x-component of velocity (m/s), beginning of timestep
        vvelN_init = _zero(horiz)  # y-component of velocity (m/s), beginning of timestep
        iceEmask = np.zeros(horiz, dtype=bool)  # T mask for dynamics
        iceNmask = np.zeros(horiz, dtype=bool)  # U mask for dynamics
        fcorE_blk = _zero(horiz)  # Coriolis
        fcorN_blk = _zero(horiz)  # Coriolis


def init_dyn_shared(dt):
    """Initialize parameters and variables needed for the dynamics.

    author: Elizabeth C. Hunke, LANL
    """
    global halo_dynbundle
    subname = "(init_dyn_shared)"
    _load_namelist()

    # checks
    if (
        kdyn == 1
        and evp_algorithm == "shared_mem_1d"
        and str(ice_domain.ns_boundary_type).strip() in ("tripole", "tripoleT")
    ):
        raise ValueError(
            subname + " ERROR: evp_alg shared mem 1d not supported with tripole"
        )

    set_evp_parameters(dt)
    # allocate dyn shared (init_uvel,init_vvel)
    alloc_dyn_shared()
    # Set halo_dynbundle, this is empirical at this point, could become namelist
    # get_num_procs is MPI and is not ported. A serial run has one process.
    halo_dynbundle = True
    nprocs = 1
    nx_block, ny_block = ice_domain_size.horizontal_shape()
    if (nx_block * ny_block) // nprocs > 100:
        halo_dynbundle = False
    # The namelist flag lives on ice_domain, the same storage as the Fortran.
    ice_domain.halo_dynbundle = halo_dynbundle

    # if (my_task == master_task): one process, so the writes always run.
    nu = ice_fileunits.nu_diag
    nu.write(f" dt  =  {dt}\n")
    nu.write(f" dt_subcyle =  {dbl_kind(dt) / dbl_kind(ndte)}\n")
    nu.write(f" tdamp = {elasticDamp * dbl_kind(dt)}\n")
    nu.write(f" halo_dynbundle = {halo_dynbundle}\n")
    nu.flush()

    # velocity
    ice_state.uvel[...] = c0  # m/s
    ice_state.vvel[...] = c0  # m/s
    if _c_or_cd():  # extra velocity variables
        ice_state.uvelE[...] = c0
        ice_state.vvelE[...] = c0
        ice_state.uvelN[...] = c0
        ice_state.vvelN[...] = c0

    # Coriolis parameter
    if coriolis == "constant":
        fcor_blk[...] = dbl_kind(1.46e-4)  # Hibler 1979, N. Hem; 1/s
    elif coriolis == "zero":
        fcor_blk[...] = c0
    else:
        fcor_blk[...] = c2 * omega * np.sin(ice_grid.ULAT)  # 1/s

    if _c_or_cd():
        if coriolis == "constant":
            fcorE_blk[...] = dbl_kind(1.46e-4)  # Hibler 1979, N. Hem; 1/s
            fcorN_blk[...] = dbl_kind(1.46e-4)  # Hibler 1979, N. Hem; 1/s
        elif coriolis == "zero":
            fcorE_blk[...] = c0
            fcorN_blk[...] = c0
        else:
            fcorE_blk[...] = c2 * omega * np.sin(ice_grid.ELAT)  # 1/s
            fcorN_blk[...] = c2 * omega * np.sin(ice_grid.NLAT)  # 1/s

    # stress tensor,  kg/s^2
    ice_flux.stressp_1[...] = c0
    ice_flux.stressp_2[...] = c0
    ice_flux.stressp_3[...] = c0
    ice_flux.stressp_4[...] = c0
    ice_flux.stressm_1[...] = c0
    ice_flux.stressm_2[...] = c0
    ice_flux.stressm_3[...] = c0
    ice_flux.stressm_4[...] = c0
    ice_flux.stress12_1[...] = c0
    ice_flux.stress12_2[...] = c0
    ice_flux.stress12_3[...] = c0
    ice_flux.stress12_4[...] = c0

    if _c_or_cd():
        ice_flux.stresspT[...] = c0
        ice_flux.stressmT[...] = c0
        ice_flux.stress12T[...] = c0
        ice_flux.stresspU[...] = c0
        ice_flux.stressmU[...] = c0
        ice_flux.stress12U[...] = c0

    if kdyn == 1:
        DminTarea[...] = deltaminEVP * ice_grid.tarea
    elif kdyn == 3:
        DminTarea[...] = deltaminVP * ice_grid.tarea

    # ice extent mask on velocity points
    iceUmask[...] = False
    if _c_or_cd():
        iceEmask[...] = False
        iceNmask[...] = False

    ilo, ihi, jlo, jhi = ice_domain_size.physical_bounds()
    if str(grid_ice).strip() == "B":
        ip = slice(ilo, ihi + 1)
        jp = slice(jlo, jhi + 1)
        dxhy[ip, jp] = p5 * (
            ice_grid.HTE[ip, jp] - ice_grid.HTE[ilo - 1:ihi, jp]
        )
        dyhx[ip, jp] = p5 * (
            ice_grid.HTN[ip, jp] - ice_grid.HTN[ip, jlo - 1:jhi]
        )

        ice_halo_update(dxhy, field_loc_center, field_type_vector, fill_value=c1)
        ice_halo_update(dyhx, field_loc_center, field_type_vector, fill_value=c1)

    # ilo:ihi+1 and jlo:jhi+1, one point past the physical domain
    ii = slice(ilo, ihi + 2)
    jj = slice(jlo, jhi + 2)
    im = slice(ilo - 1, ihi + 1)
    jm = slice(jlo - 1, jhi + 1)
    hte = ice_grid.HTE
    htn = ice_grid.HTN
    cyp[ii, jj] = c1p5 * hte[ii, jj] - p5 * hte[im, jj]
    cxp[ii, jj] = c1p5 * htn[ii, jj] - p5 * htn[ii, jm]
    # match order of operations in cyp, cxp for tripole grids
    cym[ii, jj] = -(c1p5 * hte[im, jj] - p5 * hte[ii, jj])
    cxm[ii, jj] = -(c1p5 * htn[ii, jm] - p5 * htn[ii, jj])


def set_evp_parameters(dt):
    """Set parameters needed for the evp dynamics.
    Note: This subroutine is currently called only during initialization.
          If the dynamics time step can vary during runtime, it should
           be called whenever the time step changes.

    author: Elizabeth C. Hunke, LANL
    """
    global dtei, epp2i, e_factor, ecci, revp, denom1, arlx, arlx1i, brlx
    # elastic time step
    dtei = dbl_kind(ndte) / dbl_kind(dt)

    # variables for elliptical yield curve and plastic potential
    epp2i = c1 / e_plasticpot**2
    e_factor = e_yieldcurve**2 / e_plasticpot**4
    ecci = c1 / e_yieldcurve**2  # temporary for 1d evp

    if revised_evp:  # Bouillon et al, Ocean Mod 2013
        revp = c1
        denom1 = c1
        arlx1i = c1 / arlx
    else:  # Hunke, JCP 2013 with modified stress eq
        revp = c0
        arlx = c2 * elasticDamp * dbl_kind(ndte)
        arlx1i = c1 / arlx
        brlx = dbl_kind(ndte)
        denom1 = c1 / (c1 + arlx1i)
    # The namelist variables arlx and brlx are the same storage in Fortran.
    ice_init.arlx = float(arlx)
    ice_init.brlx = float(brlx)
    # if (my_task == master_task): one process, so the write always runs.
    ice_fileunits.nu_diag.write(
        f" arlx, arlxi, brlx, denom1 {arlx} {arlx1i} {brlx} {denom1}\n"
    )
    ice_fileunits.nu_diag.flush()


def principal_stress(stressp, stressm, stress12, strength):
    """Computes principal stresses for comparison with the theoretical
    yield curve

    author: Elizabeth C. Hunke, LANL

    stressp   : sigma11 + sigma22
    stressm   : sigma11 - sigma22
    stress12  : sigma12
    strength  : for normalization of sig1 and sig2

    Returns sig1, sig2 (normalized principal stress components) and
    sigP (internal ice pressure (N/m)).
    """
    puny = icepack_parameters.puny
    ice = strength > puny
    divisor = jnp.where(ice, strength, c1)
    sigP = jnp.where(ice, -p5 * stressp, spval_dbl)
    root = jnp.sqrt(stressm ** 2 + c4 * stress12 ** 2)
    sig1 = jnp.where(ice, p5 * (stressp + root) / divisor, spval_dbl)
    sig2 = jnp.where(ice, p5 * (stressp - root) / divisor, spval_dbl)
    return sig1, sig2, sigP


def visc_replpress(strength, DminArea, Delta, parameters):
    """Twice the bulk/shear viscosity and replacement pressure.

    The two capping choices and tensile strength follow the Fortran
    momentum solver's ``visc_replpress`` exactly.
    """
    capping = parameters["capping"]
    Ktens = parameters["Ktens"]
    epp2i = parameters["epp2i"]
    tmpcalc = (capping * strength / jnp.maximum(Delta, DminArea)
               + (1.0 - capping) * strength / (Delta + DminArea))
    zetax2 = (1.0 + Ktens) * tmpcalc
    rep_prs = (1.0 - Ktens) * tmpcalc * Delta
    etax2 = epp2i * zetax2
    return zetax2, etax2, rep_prs


def dyn_prep1(aice, vice, vsno, Tmask, bounds, parameters, dyn_parameters):
    """Compute T-point mass and the one-cell-extended ice dynamics mask."""
    Tmass = jnp.where(
        Tmask, parameters["rhoi"] * vice + parameters["rhos"] * vsno,
        0.0)
    ice_present = (Tmask & (aice > dyn_parameters["dyn_area_min"])
                   & (Tmass > dyn_parameters["dyn_mass_min"]))
    # Fortran extends the ice mask to all eight neighboring T cells,
    # then restricts it to the physical ocean domain.
    neighbors = jnp.zeros_like(ice_present)
    padded = jnp.pad(ice_present, ((1, 1), (1, 1)),
                     constant_values=False)
    for di in range(3):
        for dj in range(3):
            neighbors = neighbors | padded[
                di:di + aice.shape[0], dj:dj + aice.shape[1]]
    ilo, ihi, jlo, jhi = bounds
    physical = jnp.zeros_like(Tmask).at[
        ilo:ihi + 1, jlo:jhi + 1].set(True)
    iceTmask = neighbors & Tmask & physical
    return Tmass, iceTmask


def dyn_prep2(fields, iceTmask, Xmask, bounds, dt,
              parameters, dyn_parameters, options):
    """Initialize X-point momentum inputs and velocities as in dyn_prep2.

    ``fields`` has the Fortran argument names. The boolean masks replace
    the compressed ``indxTi``/``indxXi`` lists on the single GPU.
    """
    result = dict(fields)
    ilo, ihi, jlo, jhi = bounds
    physical = jnp.zeros_like(Xmask).at[
        ilo:ihi + 1, jlo:jhi + 1].set(True)
    iceXmask = (physical & Xmask
                & (fields["aiX"] > dyn_parameters["dyn_area_min"])
                & (fields["Xmass"] > dyn_parameters["dyn_mass_min"]))
    old_mask = fields["iceXmask"]
    result["iceXmask"] = jnp.where(physical, iceXmask, old_mask)
    for name in ("waterx", "watery", "forcex", "forcey",
                 "Xmassdti", "TbU", "taubx", "tauby"):
        result[name] = jnp.zeros_like(fields[name])
    for stem in ("stressp", "stressm", "stress12"):
        for corner in range(1, 5):
            name = f"{stem}_{corner}"
            result[name] = jnp.where(iceTmask, fields[name], 0.0)

    new_ice = iceXmask & ~old_mask
    for name, ocean_name in (("uvel", "uocn"), ("vvel", "vocn")):
        result[name] = jnp.where(
            physical,
            jnp.where(iceXmask,
                      jnp.where(new_ice, fields[ocean_name], fields[name]),
                      0.0), fields[name])
        result[name + "_init"] = jnp.where(
            physical, result[name], 0.0)
    for name in ("strintx", "strinty", "strocnx", "strocny"):
        result[name] = jnp.where(physical & ~iceXmask, 0.0, fields[name])

    Xmassdti = fields["Xmass"] / dt
    fm = fields["fcor"] * fields["Xmass"]
    sign_fm = jnp.where(fm >= 0.0, 1.0, -1.0)
    # CICE fixes the turning angle at zero (cosw=1, sinw=0).
    waterx = fields["uocn"] * 1.0 - fields["vocn"] * 0.0 * sign_fm
    watery = fields["vocn"] * 1.0 + fields["uocn"] * 0.0 * sign_fm
    ssh_stress = str(options["ssh_stress"]).strip()
    if ssh_stress == "geostrophic":
        strtltx = -fm * fields["vocn"]
        strtlty = fm * fields["uocn"]
    elif ssh_stress == "coupled":
        strtltx = -parameters["gravit"] * fields["Xmass"] * fields["ss_tltx"]
        strtlty = -parameters["gravit"] * fields["Xmass"] * fields["ss_tlty"]
    else:
        raise ValueError("unknown ssh_stress: " + ssh_stress)
    result["Xmassdti"] = jnp.where(iceXmask, Xmassdti, 0.0)
    result["fm"] = jnp.where(iceXmask, fm, fields["fm"])
    result["waterx"] = jnp.where(iceXmask, waterx, 0.0)
    result["watery"] = jnp.where(iceXmask, watery, 0.0)
    result["strtltx"] = jnp.where(iceXmask, strtltx, fields["strtltx"])
    result["strtlty"] = jnp.where(iceXmask, strtlty, fields["strtlty"])
    result["forcex"] = jnp.where(iceXmask,
                                  fields["strairx"] + strtltx, 0.0)
    result["forcey"] = jnp.where(iceXmask,
                                  fields["strairy"] + strtlty, 0.0)
    return result


def dyn_finish(uvel, vvel, uocn, vocn, aiX, fm, Cw,
               strocnx, strocny, iceXmask, parameters):
    """Update the ocean stress from the final ice velocity."""
    urel = uocn - uvel
    vrel = vocn - vvel
    drag = parameters["rhow"] * Cw * jnp.sqrt(urel**2 + vrel**2) * aiX
    # The configured turning angle is zero (cosw=1, sinw=0).
    strocnx = jnp.where(iceXmask, drag * urel, strocnx)
    strocny = jnp.where(iceXmask, drag * vrel, strocny)
    return strocnx, strocny


def shift(field, di=0, dj=0):
    """Neighbor values on the allocated grid; physical boundaries use halos."""
    return jnp.roll(field, (-di, -dj), axis=(0, 1))


def strain_rates_T(uvelE, vvelE, uvelN, vvelN, grid, mask, parameters):
    """Compute the dt (div, tension) strain rates at the T point.

    author: JF Lemieux, ECCC
    Nov 2021
    """
    dxN, dyE, dxT, dyT = (grid[n] for n in ("dxN", "dyE", "dxT", "dyT"))
    # strain rates
    # NOTE these are actually strain rates * area  (m^2/s)
    # divergence  =  e_11 + e_22
    divT = (dyE * uvelE - shift(dyE * uvelE, -1)
            + dxN * vvelN - shift(dxN * vvelN, 0, -1))
    # tension strain rate  =  e_11 - e_22
    tensionT = (dyT**2 * (uvelE / dyE - shift(uvelE / dyE, -1))
                - dxT**2 * (vvelN / dxN - shift(vvelN / dxN, 0, -1)))
    # shearing strain rate  =  2*e_12
    shearT = (dxT**2 * (uvelN / dxN - shift(uvelN / dxN, 0, -1))
              + dyT**2 * (vvelE / dyE - shift(vvelE / dyE, -1)))
    # Delta (in the denominator of zeta, eta)
    DeltaT = jnp.sqrt(divT**2 + parameters["e_factor"] * (tensionT**2 + shearT**2))
    return tuple(jnp.where(mask, x, 0.0) for x in (divT, tensionT, shearT, DeltaT))


def strain_rates_U(uvelE, vvelE, uvelN, vvelN, uvelU, vvelU,
                   grid, ratios, mask, parameters):
    """Compute strain rates at the U point including boundary conditions.

    author: JF Lemieux, ECCC
    Nov 2021
    """
    epm, npm = grid["epm"], grid["npm"]
    ep, np_ = shift(epm, 0, 1), shift(npm, 1)
    # strain rates
    # NOTE these are actually strain rates * area  (m^2/s)
    uNip1j = (shift(uvelN, 1) * np_
              + (npm - np_) * npm * ratios["ratiodxN"] * uvelN)
    uNij = (uvelN * npm
            + (np_ - npm) * np_ * ratios["ratiodxNr"] * shift(uvelN, 1))
    vEijp1 = (shift(vvelE, 0, 1) * ep
              + (epm - ep) * epm * ratios["ratiodyE"] * vvelE)
    vEij = (vvelE * epm
            + (ep - epm) * ep * ratios["ratiodyEr"] * shift(vvelE, 0, 1))
    dxE, dyN, dxU, dyU = (grid[n] for n in ("dxE", "dyN", "dxU", "dyU"))
    # divergence  =  e_11 + e_22
    divergU = (dyU * (uNip1j - uNij) + uvelU * (shift(dyN, 1) - dyN)
               + dxU * (vEijp1 - vEij) + vvelU * (shift(dxE, 0, 1) - dxE))
    # tension strain rate  =  e_11 - e_22
    tensionU = (dyU * (uNip1j - uNij) - uvelU * (shift(dyN, 1) - dyN)
                - dxU * (vEijp1 - vEij) + vvelU * (shift(dxE, 0, 1) - dxE))
    uEijp1 = (shift(uvelE, 0, 1) * ep
              + (epm - ep) * epm * ratios["ratiodyE"] * uvelE)
    uEij = (uvelE * epm
            + (ep - epm) * ep * ratios["ratiodyEr"] * shift(uvelE, 0, 1))
    vNip1j = (shift(vvelN, 1) * np_
              + (npm - np_) * npm * ratios["ratiodxN"] * vvelN)
    vNij = (vvelN * npm
            + (np_ - npm) * np_ * ratios["ratiodxNr"] * shift(vvelN, 1))
    # shearing strain rate  =  2*e_12
    shearU = (dxU * (uEijp1 - uEij) - uvelU * (shift(dxE, 0, 1) - dxE)
              + dyU * (vNip1j - vNij) - vvelU * (shift(dyN, 1) - dyN))
    # Delta (in the denominator of zeta, eta)
    DeltaU = jnp.sqrt(divergU**2 + parameters["e_factor"] * (tensionU**2 + shearU**2))
    return tuple(jnp.where(mask, x, 0.0) for x in (divergU, tensionU, shearU, DeltaU))


def stepuv_CD(fields, Cw, parameters, dyn_parameters, component="both"):
    """Integration of the momentum equation at E/N locations on C/CD grids.

    component='u' and 'v' are Fortran stepu_C and stepv_C, respectively.
    """
    result = dict(fields)
    uold, vold = fields["uvel"], fields["vvel"]
    # integrate the momentum equation
    # (magnitude of relative ocean current)*rhow*drag*aice
    vrel = (fields["aiX"] * parameters["rhow"] * Cw
            * jnp.sqrt((fields["uocn"] - uold)**2 + (fields["vocn"] - vold)**2))
    # ice/ocean stress
    taux = vrel * fields["waterx"]  # NOTE this is not the entire
    tauy = vrel * fields["watery"]  # ocn stress term
    Cb = fields["TbU"] / (jnp.sqrt(uold**2 + vold**2) + 5.0e-5)  # for seabed stress
    # revp = 0 for classic evp, 1 for revised evp
    brlx, revp = dyn_parameters["brlx"], dyn_parameters["revp"]
    cca = (brlx + revp) * fields["Xmassdti"] + vrel + Cb
    ccb = fields["fm"]  # cosw=1, sinw=0
    # compute the velocity components
    cc1 = (fields["strintx"] + fields["forcex"] + taux
           + fields["Xmassdti"] * (brlx * uold + revp * fields["uvel_init"]))
    cc2 = (fields["strinty"] + fields["forcey"] + tauy
           + fields["Xmassdti"] * (brlx * vold + revp * fields["vvel_init"]))
    mask = fields["iceXmask"]
    if component == "both":
        ab2 = jnp.where(mask, cca**2 + ccb**2, 1.0)
        uvel, vvel = (cca * cc1 + ccb * cc2) / ab2, (cca * cc2 - ccb * cc1) / ab2
    elif component == "u":
        uvel, vvel = (ccb * vold + cc1) / jnp.where(mask, cca, 1.0), vold
    else:
        uvel, vvel = uold, (-ccb * uold + cc2) / jnp.where(mask, cca, 1.0)
    # calculate seabed stress component for outputs
    # only needed on last iteration.
    for name, value in (("uvel", uvel), ("vvel", vvel)):
        result[name] = jnp.where(mask, value, fields[name])
    if component != "v":
        result["taubx"] = jnp.where(mask, -uvel * Cb, fields["taubx"])
    if component != "u":
        result["tauby"] = jnp.where(mask, -vvel * Cb, fields["tauby"])
    return result


def stepu(fields, stress_terms, Cw, uarear, iceUmask,
          parameters, dyn_parameters):
    """Integrate the B-grid momentum equation on active U points."""
    uold = fields["uvel"]
    vold = fields["vvel"]
    urel = fields["uocn"] - uold
    vrel = fields["vocn"] - vold
    # Fortran stepu calls this drag magnitude vrel; waterx/watery give its
    # x/y direction before the momentum solve.
    drag = (fields["aiX"] * parameters["rhow"] * Cw
            * jnp.sqrt(urel**2 + vrel**2))
    taux = drag * fields["waterx"]
    tauy = drag * fields["watery"]
    # Fortran Cb is the speed-regularized seabed drag coefficient.
    basal = fields["TbU"] / (jnp.sqrt(uold**2 + vold**2) + 5.0e-5)
    cca = ((dyn_parameters["brlx"] + dyn_parameters["revp"])
           * fields["Umassdti"] + drag + basal)
    ccb = fields["fm"]  # sinw=0 in this CICE configuration.
    denominator = cca**2 + ccb**2
    # Fortran stepu combines str(i,j,1:8) from four neighboring T cells
    # to form the two components of internal-stress divergence at each U point.
    terms = jnp.pad(stress_terms, ((0, 1), (0, 1), (0, 0)))
    divergence_x = uarear * (
        terms[:-1, :-1, 0] + terms[1:, :-1, 1]
        + terms[:-1, 1:, 2] + terms[1:, 1:, 3])
    divergence_y = uarear * (
        terms[:-1, :-1, 4] + terms[:-1, 1:, 5]
        + terms[1:, :-1, 6] + terms[1:, 1:, 7])
    cc1 = (divergence_x + fields["forcex"] + taux
           + fields["Umassdti"] * (
               dyn_parameters["brlx"] * uold
               + dyn_parameters["revp"] * fields["uvel_init"]))
    cc2 = (divergence_y + fields["forcey"] + tauy
           + fields["Umassdti"] * (
               dyn_parameters["brlx"] * vold
               + dyn_parameters["revp"] * fields["vvel_init"]))
    safe_denominator = jnp.where(iceUmask & (denominator != 0.0),
                                 denominator, 1.0)
    # Solve the same coupled x/y momentum equations as Fortran stepu.
    uvel = (cca * cc1 + ccb * cc2) / safe_denominator
    vvel = (cca * cc2 - ccb * cc1) / safe_denominator
    result = dict(fields)
    result["strintx"] = jnp.where(iceUmask, divergence_x, fields["strintx"])
    result["strinty"] = jnp.where(iceUmask, divergence_y, fields["strinty"])
    result["uvel"] = jnp.where(iceUmask, uvel, uold)
    result["vvel"] = jnp.where(iceUmask, vvel, vold)
    result["taubx"] = jnp.where(iceUmask, -uvel * basal, fields["taubx"])
    result["tauby"] = jnp.where(iceUmask, -vvel * basal, fields["tauby"])
    return result


def seabed_stress_factor_LKD(vice, aice, hwater, TbU,
                             iceXmask, grid_location, dyn_parameters):
    """Compute the linear-keel seabed stress coefficient at U, E or N."""
    if grid_location not in ("U", "E", "N"):
        raise ValueError("unknown grid_location: " + grid_location)
    hwater_pad = jnp.pad(hwater, ((0, 1), (0, 1)), mode="edge")
    aice_pad = jnp.pad(aice, ((0, 1), (0, 1)), mode="edge")
    vice_pad = jnp.pad(vice, ((0, 1), (0, 1)), mode="edge")
    if grid_location == "U":
        offsets = ((0, 0), (1, 0), (0, 1), (1, 1))
    elif grid_location == "E":
        offsets = ((0, 0), (1, 0))
    else:
        offsets = ((0, 0), (0, 1))
    water_neighbors = []
    area_neighbors = []
    volume_neighbors = []
    for di, dj in offsets:
        water_neighbors.append(hwater_pad[
            di:di + hwater.shape[0], dj:dj + hwater.shape[1]])
        area_neighbors.append(aice_pad[
            di:di + aice.shape[0], dj:dj + aice.shape[1]])
        volume_neighbors.append(vice_pad[
            di:di + vice.shape[0], dj:dj + vice.shape[1]])
    hwu = jnp.min(jnp.stack(water_neighbors), axis=0)
    au = jnp.max(jnp.stack(area_neighbors), axis=0)
    hu = jnp.max(jnp.stack(volume_neighbors), axis=0)
    critical = au * hwu / dyn_parameters["k1"]
    stress = jnp.where(hwu < dyn_parameters["threshold_hw"],
                       dyn_parameters["k2"] * jnp.maximum(0.0, hu - critical)
                       * jnp.exp(-dyn_parameters["alphab"] * (1.0 - au)),
                       0.0)
    return jnp.where(iceXmask, stress, TbU)


def strain_rates(uvel, vvel, dxT, dyT, cxp, cyp, cxm, cym,
                 iceTmask, dyn_parameters):
    """Four B-grid corner strains, in the Fortran NE/NW/SW/SE order.

    The quantities still include cell area, as in strain_rates.F90.
    A trailing length-four axis replaces the four scalar output groups.
    """
    u = jnp.pad(uvel, ((1, 0), (1, 0)))
    v = jnp.pad(vvel, ((1, 0), (1, 0)))
    uc, uw, us, usw = u[1:, 1:], u[:-1, 1:], u[1:, :-1], u[:-1, :-1]
    vc, vw, vs, vsw = v[1:, 1:], v[:-1, 1:], v[1:, :-1], v[:-1, :-1]
    div = jnp.stack((
        cyp * uc - dyT * uw + cxp * vc - dxT * vs,
        cym * uw + dyT * uc + cxp * vw - dxT * vsw,
        cym * usw + dyT * us + cxm * vsw + dxT * vw,
        cyp * us - dyT * usw + cxm * vs + dxT * vc,
    ), axis=-1)
    tension = jnp.stack((
        -cym * uc - dyT * uw + cxm * vc + dxT * vs,
        -cyp * uw + dyT * uc + cxm * vw + dxT * vsw,
        -cyp * usw + dyT * us + cxp * vsw - dxT * vw,
        -cym * us - dyT * usw + cxp * vs - dxT * vc,
    ), axis=-1)
    shear = jnp.stack((
        -cym * vc - dyT * vw - cxm * uc - dxT * us,
        -cyp * vw + dyT * vc - cxm * uw - dxT * usw,
        -cyp * vsw + dyT * vs - cxp * usw + dxT * uw,
        -cym * vs - dyT * vsw - cxp * us + dxT * uc,
    ), axis=-1)
    delta = jnp.sqrt(div**2 + dyn_parameters["e_factor"]
                     * (tension**2 + shear**2))
    return tuple(jnp.where(iceTmask[..., None], value, 0.0)
                 for value in (div, tension, shear, delta))


def deformations(uvel, vvel, grid, shared, iceTmask, dyn_parameters):
    """B-grid divergence, shear, vorticity and ridging deformation."""
    div, tension, shearing, delta = strain_rates(
        uvel, vvel, grid["dxT"], grid["dyT"],
        shared["cxp"], shared["cyp"], shared["cxm"], shared["cym"],
        iceTmask, dyn_parameters)
    # Fortran sums NE, NW, SE, SW; storage uses NE, NW, SW, SE.
    divu = 0.25 * (div[..., 0] + div[..., 1] + div[..., 3]
                   + div[..., 2]) * grid["tarear"]
    tmp = 0.25 * (delta[..., 0] + delta[..., 1] + delta[..., 3]
                  + delta[..., 2]) * grid["tarear"]
    rdg_conv = -jnp.minimum(divu, 0.0)
    rdg_shear = 0.5 * (tmp - jnp.abs(divu))
    shear = 0.25 * grid["tarear"] * jnp.sqrt(
        (tension[..., 0] + tension[..., 1] + tension[..., 3]
         + tension[..., 2])**2
        + (shearing[..., 0] + shearing[..., 1] + shearing[..., 3]
           + shearing[..., 2])**2)
    u = jnp.pad(uvel, ((1, 0), (1, 0)))
    v = jnp.pad(vvel, ((1, 0), (1, 0)))
    dyU = jnp.pad(grid["dyU"], ((1, 0), (1, 0)))
    dxU = jnp.pad(grid["dxU"], ((1, 0), (1, 0)))
    dvdxn = dyU[1:, 1:] * v[1:, 1:] - dyU[:-1, 1:] * v[:-1, 1:]
    dvdxs = dyU[1:, :-1] * v[1:, :-1] - dyU[:-1, :-1] * v[:-1, :-1]
    dudye = dxU[1:, 1:] * u[1:, 1:] - dxU[1:, :-1] * u[1:, :-1]
    dudyw = dxU[:-1, 1:] * u[:-1, 1:] - dxU[:-1, :-1] * u[:-1, :-1]
    vort = 0.5 * grid["tarear"] * (dvdxn + dvdxs - dudye - dudyw)
    return tuple(jnp.where(iceTmask, value, 0.0) for value in (
        vort, shear, divu, rdg_conv, rdg_shear))


def seabed_stress_factor_prob(aicen, vicen, hwater, hin_max,
                              iceTmask, iceXmask, grid_location,
                              parameters, dyn_parameters):
    """Probabilistic grounded-keel stress at T, then U/E/N points."""
    if grid_location not in ("U", "E", "N"):
        raise ValueError("unknown grid_location: " + grid_location)
    ncat_i = 100
    ncat_b = 100
    max_depth = 50.0
    mu_s = 0.1
    sigma_b = 2.5
    wid_i = max_depth / ncat_i
    wid_b = 6.0 * sigma_b / ncat_b
    x_k = wid_i * (jnp.arange(ncat_i, dtype=aicen.dtype) + 0.5)
    y_offset = -3.0 * sigma_b + wid_b * (
        jnp.arange(ncat_b, dtype=aicen.dtype) + 0.5)
    P_y = (jnp.exp(-y_offset**2 / (2.0 * sigma_b**2))
           * wid_b / (sigma_b * jnp.sqrt(2.0 * parameters["pi"])))
    atot = jnp.sum(aicen, axis=-1)
    m_i = jnp.sum(vicen, axis=-1)
    puny = parameters["puny"]
    v_i = jnp.sum(vicen**2 / jnp.maximum(aicen, puny), axis=-1)
    v_i = jnp.maximum(v_i - m_i**2, puny)
    safe_m = jnp.where(m_i > 0.0, m_i, 1.0)
    mu_i = jnp.log(safe_m / jnp.sqrt(1.0 + v_i / safe_m**2))
    sigma_i = jnp.sqrt(jnp.log(1.0 + v_i / safe_m**2))
    x_kmax = jnp.exp(mu_i + jnp.sqrt(2.0 * sigma_i) * 1.9430)
    # The baseline's DO n = ncat,-1,1 has zero iterations for positive ncat.
    # Preserve its fixed cutoff rather than silently correcting that loop.
    x_kmax = jnp.minimum(x_k[-1], x_kmax)
    rho_ratio = parameters["rhoi"] / parameters["rhow"]
    prefactor = mu_s * parameters["gravit"]

    # Scan avoids materializing the cell x 100 ice x 100 bathymetry cube.
    def thickness_bin(n, total):
        x = x_k[n]
        P_x = (jnp.exp(-(jnp.log(x) - mu_i)**2 / (2.0 * sigma_i**2))
               * wid_i / (x * sigma_i * jnp.sqrt(2.0 * parameters["pi"])))
        P_x = jnp.where(x <= x_kmax, P_x, 0.0)
        y_n = hwater[..., None] + y_offset
        grounded = y_n <= rho_ratio * x
        force = jnp.sum(jnp.where(
            grounded, P_y * (parameters["rhoi"] * x
                             - parameters["rhow"] * y_n), 0.0), axis=-1)
        return total + jnp.maximum(prefactor * P_x * force, 0.0)

    Tbt = jax.lax.fori_loop(0, ncat_i, thickness_bin,
                             jnp.zeros_like(hwater))
    Tbt = jnp.where(
        iceTmask & (atot > 0.05) & (hwater < max_depth),
        Tbt * jnp.exp(-dyn_parameters["alphab"] * (1.0 - atot)), 0.0)
    padded = jnp.pad(Tbt, ((0, 1), (0, 1)), mode="edge")
    if grid_location == "U":
        neighbors = (padded[:-1, :-1], padded[1:, :-1],
                     padded[:-1, 1:], padded[1:, 1:])
    elif grid_location == "E":
        neighbors = (padded[:-1, :-1], padded[1:, :-1])
    else:
        neighbors = (padded[:-1, :-1], padded[:-1, 1:])
    return jnp.where(iceXmask, jnp.max(jnp.stack(neighbors), axis=0), 0.0)
