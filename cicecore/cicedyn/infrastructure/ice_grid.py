"""Spatial grids, masks, and boundary conditions

author for JAX version: Xinyuan Zhu
XZ: converted from F90 to Python
XZ: grid_average_X2Y_NEversion and grid_average_X2Y_2

2021: Add N (center of north face) and E (center of east face) grids
      to support C and CD solvers.  Defining T at center of cells, U at
      NE corner, N at center of top face, E at center of right face.
      All cells are quadrilaterals with NE, E, and N associated with
      directions relative to logical grid.
2007: Option to read from netcdf files (A. Keen, Met Office)
      Grid reading routines reworked by E. Hunke for boundary values
2006: Converted to free source form (F90) by Elizabeth Hunke
2004: Block structure added by William Lipscomb
      init_grid split into two parts as in POP 2.0
      Boundary update routines replaced by POP versions
authors: Elizabeth C. Hunke and William H. Lipscomb, LANL
         Tony Craig, NCAR

One array holds the whole domain plus ghost cells. ``init_grid1`` keeps
the tripole checks. The ULAT and KMT read that weights the block
distribution is not ported. ``init_grid2`` follows the Fortran ``grid_type`` and
``grid_format`` selection. Fortran computes Coriolis in
``init_dyn_shared``; it is computed here because this step owns the
latitude arrays. History gridbox corners are not written.
``latlongrid`` is compiled only with CESMCOUPLED, so a latlon
``grid_type`` uses ``rectgrid``, the same branch as a standalone build.
"""

import numpy as np
import jax.numpy as jnp

from cicecore.cicedyn.general import ice_flux
from cicecore.cicedyn.general import ice_init
from cicecore.shared import ice_fileunits
from cicecore.cicedyn.general import ice_state
from cicecore.cicedyn.infrastructure import ice_read_write
from cicecore.cicedyn.infrastructure.comm.serial import ice_boundary
from cicecore.cicedyn.infrastructure.comm.serial.ice_boundary import (
    global_indices,
    ice_halo_extrapolate,
    ice_halo_update,
    scatter_global,
    scatter_global_ext,
)
from cicecore.shared import ice_arrays_column
from cicecore.shared.ice_constants import (
    c0,
    c1,
    c1p5,
    c2,
    c180,
    c20,
    c4,
    cm_to_m,
    field_loc_Eface,
    field_loc_NEcorner,
    field_loc_Nface,
    field_loc_center,
    field_type_angle,
    field_type_scalar,
    omega,
    p25,
    p5,
    radius,
)
from cicecore.shared import ice_domain_size
from cicecore.shared.ice_domain_size import (
    horizontal_shape,
    nghost,
    physical_bounds,
    sync_vertical_sizes,
)
from cicecore.shared.ice_kinds_mod import dbl_kind, log_kind
from icepack.columnphysics import icepack_tracers
from icepack.columnphysics.icepack_parameters import (
    icepack_init_parameters,
    icepack_query_parameters,
)

_HORIZ = (
    "dxT",    # width of T-cell through the middle (m)
    "dyT",    # height of T-cell through the middle (m)
    "dxU",    # width of U-cell through the middle (m)
    "dyU",    # height of U-cell through the middle (m)
    "dxN",    # width of N-cell through the middle (m)
    "dyN",    # height of N-cell through the middle (m)
    "dxE",    # width of E-cell through the middle (m)
    "dyE",    # height of E-cell through the middle (m)
    "HTE",    # length of eastern edge of T-cell (m)
    "HTN",    # length of northern edge of T-cell (m)
    "tarea",  # area of T-cell (m^2), valid in halo
    "uarea",  # area of U-cell (m^2), valid in halo
    "narea",  # area of N-cell (m^2), valid in halo
    "earea",  # area of E-cell (m^2), valid in halo
    "tarear",  # 1/tarea, valid in halo
    "uarear",  # 1/uarea, valid in halo
    "narear",  # 1/narea, valid in halo
    "earear",  # 1/earea, valid in halo
    "tarean",  # area of NH T-cells
    "tareas",  # area of SH T-cells
    "ULON",    # longitude of velocity pts, NE corner of T pts (radians)
    "ULAT",    # latitude of velocity pts, NE corner of T pts (radians)
    "TLON",    # longitude of temp (T) pts (radians)
    "TLAT",    # latitude of temp (T) pts (radians)
    "NLON",    # longitude of center of north face of T pts (radians)
    "NLAT",    # latitude of center of north face of T pts (radians)
    "ELON",    # longitude of center of east face of T pts (radians)
    "ELAT",    # latitude of center of east face of T pts (radians)
    "ANGLE",   # for conversions between POP grid and lat/lon
    "ANGLET",  # ANGLE converted to T-cells, valid in halo
    "bathymetry",  # ocean depth, for grounding keels and bergs (m)
    "ocn_gridcell_frac",  # ocean gridcell fraction
                          # gridcell value of [1 - (land fraction)] (T-cell)
                          # only relevant for lat-lon grids
    "hm",   # land/boundary mask, thickness (T-cell)
    "bm",   # task/block id
    "uvm",  # land/boundary mask (U-cell)
    "npm",  # land/boundary mask (N-cell)
    "epm",  # land/boundary mask (E-cell)
    "kmt",  # ocean topography mask for bathymetry (T-cell)
    "rndex_global",  # global index for local subdomain (dbl)
    "fcor",   # Coriolis parameter (1/s)
    "fcorE",  # Coriolis parameter at E points (1/s)
    "fcorN",  # Coriolis parameter at N points  (1/s)
)
# masks
_MASKS = (
    "tmask",    # land/boundary mask, thickness (T-cell)
    "umask",    # land/boundary mask  (U-cell) (1 if all surrounding T cells are ocean)
    "umaskCD",  # land/boundary mask  (U-cell) (1 if at least two surrounding T cells are ocean)
    "nmask",    # land/boundary mask, (N-cell)
    "emask",    # land/boundary mask, (E-cell)
    "opmask",   # land/boundary orphan mask, ocean cells in atmosphere but not ocean/ice
    "lmask_n",  # northern hemisphere mask
    "lmask_s",  # southern hemisphere mask
)
# Declared here so each field is a module name. ``alloc_grid`` fills them.
dxT = None
dyT = None
dxU = None
dyU = None
dxN = None
dyN = None
dxE = None
dyE = None
HTE = None
HTN = None
tarea = None
uarea = None
narea = None
earea = None
tarear = None
uarear = None
narear = None
earear = None
tarean = None
tareas = None
ULON = None
ULAT = None
TLON = None
TLAT = None
NLON = None
NLAT = None
ELON = None
ELAT = None
ANGLE = None
ANGLET = None
bathymetry = None
ocn_gridcell_frac = None
hm = None
bm = None
uvm = None
npm = None
epm = None
kmt = None
rndex_global = None
fcor = None
fcorE = None
fcorN = None
tmask = None
umask = None
umaskCD = None
nmask = None
emask = None
opmask = None
lmask_n = None
lmask_s = None
G_HTN = None  # length of northern edge of T-cell (global ext.)
G_HTE = None  # length of eastern edge of T-cell (global ext.)

l_readCenter = False  # If anglet exist in grid file read it otherwise calculate it

# Constructs the shift based on the grid
_SHIFT = {
    ("T", "U"): "NE",
    ("T", "E"): "E",
    ("T", "N"): "N",
    ("U", "T"): "SW",
    ("U", "E"): "S",
    ("U", "N"): "W",
    ("E", "T"): "W",
    ("E", "U"): "N",
    ("E", "N"): "NW",
    ("N", "T"): "S",
    ("N", "U"): "E",
    ("N", "E"): "SE",
}
_OFFSETS = {
    "NE": ((0, 0), (1, 0), (0, 1), (1, 1)),
    "SW": ((0, 0), (-1, 0), (0, -1), (-1, -1)),
    "NW": ((-1, 0), (0, 0), (-1, 1), (0, 1)),
    "SE": ((0, -1), (1, -1), (0, 0), (1, 0)),
    "E": ((0, 0), (1, 0)),
    "W": ((-1, 0), (0, 0)),
    "N": ((0, 0), (0, 1)),
    "S": ((0, -1), (0, 0)),
}


def _zero(shape):
    return np.zeros(shape, dtype=dbl_kind)


def init_grid1():
    """Distribute blocks across processors.  The distribution is optimized
    based on latitude and topography, contained in the ULAT and KMT arrays.

    authors: William Lipscomb and Phil Jones, LANL

    Only the tripole checks are ported. The ULAT and KMT read that weights
    the block distribution is not ported.
    """
    #-----------------------------------------------------------------
    # Get global ULAT and KMT arrays used for block decomposition.
    #-----------------------------------------------------------------
    # check tripole flags here
    # can't check in init_data because ns_boundary_type is not yet read
    # can't check in init_domain_blocks because grid_type is not accessible due to circular logic

    subname = "(init_grid1)"
    grid_type = str(ice_init.grid_type).strip()
    grid_format = str(ice_init.grid_format).strip()
    ns_boundary_type = str(ice_boundary.ns_boundary_type).strip()
    nx_global = int(ice_domain_size.nx_global)
    if (
        grid_type == "tripole"
        and ns_boundary_type != "tripole"
        and ns_boundary_type != "tripoleT"
    ):
        raise ValueError(
            subname + " ERROR: grid_type tripole needs tripole ns_boundary_type"
        )
    if grid_type == "tripole" and (nx_global % 2) != 0:
        raise ValueError(
            subname + " ERROR: grid_type tripole requires even nx_global number"
        )
    if grid_format == "mom_nc" and ns_boundary_type == "tripoleT":
        raise ValueError(
            subname + " ERROR: ns_boundary_type='tripoleT' not implemented "
            "for grid_format='mom_nc'. Use 'tripole' instead."
        )


def alloc_grid():
    """Allocate space for all variables."""
    shape = horizontal_shape()
    for name in _HORIZ:
        globals()[name] = _zero(shape)
    for name in _MASKS:
        globals()[name] = np.zeros(shape, dtype=log_kind)
    # special value to start, will be ignored unless set elsewhere
    ocn_gridcell_frac[...] = -c1
    if ice_init.save_ghte_ghtn:
        globals()["G_HTN"] = _zero(shape)  # length of northern edge of T-cell (global ext.)
        globals()["G_HTE"] = _zero(shape)  # length of eastern edge of T-cell (global ext.)


def _place(array, global_field, field_loc, field_type):
    scatter_global(array, global_field, field_loc, field_type)
    ice_halo_extrapolate(array)


def _view(array, di, dj):
    ilo, ihi, jlo, jhi = physical_bounds()
    return array[ilo + di:ihi + 1 + di, jlo + dj:jhi + 1 + dj]


def popgrid():
    """POP displaced pole grid (or tripole).

    Grid record number, field and units are:
    (1) ULAT  (radians)
    (2) ULON  (radians)
    (3) HTN   (cm)
    (4) HTE   (cm)
    (5) HUS   (cm)
    (6) HUW   (cm)
    (7) ANGLE (radians)

    author: Elizabeth C. Hunke, LANL

    HUS and HUW are not used.
    """
    path = ice_init.grid_file
    #-----------------------------------------------------------------
    # lat, lon, angle
    #-----------------------------------------------------------------
    _place(
        ULAT,
        ice_read_write.ice_read_global(path, 1, "rda8", True),  # ULAT
        field_loc_NEcorner,
        field_type_scalar,
    )
    _place(
        ULON,
        ice_read_write.ice_read_global(path, 2, "rda8", True),  # ULON
        field_loc_NEcorner,
        field_type_scalar,
    )
    _place(
        ANGLE,
        ice_read_write.ice_read_global(path, 7, "rda8", True),  # ANGLE
        field_loc_NEcorner,
        field_type_angle,
    )
    #-----------------------------------------------------------------
    # cell dimensions
    # calculate derived quantities from global arrays to preserve
    # information on boundaries
    #-----------------------------------------------------------------
    primary_grid_lengths_HTN(
        ice_read_write.ice_read_global(path, 3, "rda8", True)  # HTN
    )  # dxU, dxT, dxN, dxE
    primary_grid_lengths_HTE(
        ice_read_write.ice_read_global(path, 4, "rda8", True)  # HTE
    )  # dyU, dyT, dyN, dyE


def primary_grid_lengths_HTN(work_g):
    """Calculate dxU and dxT from HTN on the global grid, to preserve
    ghost cell and/or land values that might otherwise be lost. Scatter
    dxU, dxT and HTN to all processors.

    author: Elizabeth C. Hunke, LANL

    ``work_g`` is the global array holding HTN.
    """
    # HTN, dxU = average of 2 neighbor HTNs in i
    work_g *= cm_to_m  # HTN
    # i+1
    # assume cyclic; noncyclic will be handled during extrapolate
    ip1 = (np.arange(work_g.shape[0]) + 1) % work_g.shape[0]
    dxu = p5 * (work_g + work_g[ip1, :])  # dxU
    _place(HTN, work_g, field_loc_Nface, field_type_scalar)
    if ice_init.save_ghte_ghtn:
        G_HTN[...] = HTN
    _place(dxU, dxu, field_loc_NEcorner, field_type_scalar)

    # dxT = average of 2 neighbor HTNs in j
    dxt = np.empty_like(work_g)
    dxt[:, 1:] = p5 * (work_g[:, 1:] + work_g[:, :-1])  # dxT
    # extrapolate to obtain dxT along j=1
    dxt[:, 0] = c2 * work_g[:, 1] - work_g[:, 2]  # dxT
    _place(dxT, dxt, field_loc_center, field_type_scalar)

    # dxN = HTN
    dxN[...] = HTN  # dxN

    # dxE = average of 4 surrounding HTNs
    dxe = np.empty_like(work_g)
    # assume cyclic; noncyclic will be handled during extrapolate
    dxe[:, 1:] = p25 * (
        work_g[:, 1:] + work_g[ip1, 1:] + work_g[:, :-1] + work_g[ip1, :-1]
    )  # dxE
    # extrapolate to obtain dxT along j=1
    # assume cyclic; noncyclic will be handled during extrapolate
    dxe[:, 0] = p5 * (
        c2 * work_g[:, 1] - work_g[:, 2] + c2 * work_g[ip1, 1] - work_g[ip1, 2]
    )  # dxE
    _place(dxE, dxe, field_loc_center, field_type_scalar)


def primary_grid_lengths_HTE(work_g):
    """Calculate dyU and dyT from HTE on the global grid, to preserve
    ghost cell and/or land values that might otherwise be lost. Scatter
    dyU, dyT and HTE to all processors.

    author: Elizabeth C. Hunke, LANL

    ``work_g`` is the global array holding HTE.
    """
    # HTE, dyU = average of 2 neighbor HTE in j
    work_g *= cm_to_m  # HTE
    # i-1
    # assume cyclic; noncyclic will be handled during extrapolate
    im1 = (np.arange(work_g.shape[0]) - 1) % work_g.shape[0]
    dyu = np.empty_like(work_g)
    dyu[:, :-1] = p5 * (work_g[:, :-1] + work_g[:, 1:])  # dyU
    # extrapolate to obtain dyU along j=ny_global
    dyu[:, -1] = c2 * work_g[:, -2] - work_g[:, -3]  # dyU
    _place(HTE, work_g, field_loc_Eface, field_type_scalar)
    if ice_init.save_ghte_ghtn:
        G_HTE[...] = HTE
    _place(dyU, dyu, field_loc_NEcorner, field_type_scalar)

    # dyT = average of 2 neighbor HTE in i
    dyt = p5 * (work_g + work_g[im1, :])  # dyT
    _place(dyT, dyt, field_loc_center, field_type_scalar)

    # dyN = average of 4 neighbor HTEs
    dyn = np.empty_like(work_g)
    # assume cyclic; noncyclic will be handled during extrapolate
    dyn[:, :-1] = p25 * (
        work_g[:, :-1] + work_g[im1, :-1] + work_g[:, 1:] + work_g[im1, 1:]
    )  # dyN
    # extrapolate to obtain dyN along j=ny_global
    # assume cyclic; noncyclic will be handled during extrapolate
    dyn[:, -1] = p5 * (
        c2 * work_g[:, -2] - work_g[:, -3]
        + c2 * work_g[im1, -2] - work_g[im1, -3]
    )  # dyN
    _place(dyN, dyn, field_loc_center, field_type_scalar)

    # dyE = HTE
    dyE[...] = HTE


def kmtmask(filetype):
    """POP land mask.

    Land mask record number and field is (1) KMT.

    ``filetype`` is 'nc' or 'bin'.
    """
    puny = icepack_query_parameters("puny")["puny"]
    kmt[...] = c0
    hm[...] = c0
    path = ice_init.kmt_file
    if filetype == "bin":
        ice_read_write.ice_read(
            path, 1, kmt, "ida4", True,  # write diagnostic info
            field_loc_center, field_type_scalar,
        )
    elif filetype == "nc":
        name = ice_read_write.mask_fieldname(path)
        ice_read_write.ice_read_nc(
            path, name, kmt, True,  # write diagnostic info
            field_loc_center, field_type_scalar,
        )
    elif filetype == "nc_ext":
        name = ice_read_write.mask_fieldname(path)
        ice_read_write.ice_read_nc(
            path, name, kmt, True,  # write diagnostic info
            field_loc_center, field_type_scalar, extended=True,
        )
    else:
        raise ValueError(f"(kmtmask) invalid filetype={filetype}")
    # beginning and end of physical domain
    ilo, ihi, jlo, jhi = physical_bounds()
    frac = ocn_gridcell_frac[ilo:ihi + 1, jlo:jhi + 1]
    kmt_phys = kmt[ilo:ihi + 1, jlo:jhi + 1]
    # force grid cells to land if ocn_gridcell_frac is defined
    kmt_phys[(frac >= c0) & (frac < puny)] = c0
    hm[kmt >= p5] = c1


def _reciprocal(area):
    out = np.zeros(area.shape, dtype=dbl_kind)  # possible on boundaries
    good = area > c0
    out[good] = c1 / area[good]
    return out


def _corner_xyz(di, dj):
    ulat = _view(ULAT, di, dj)
    ulon = _view(ULON, di, dj)
    coslat = np.cos(ulat)
    return np.cos(ulon) * coslat, np.sin(ulon) * coslat, np.sin(ulat)


def makemask():
    """Sets the boundary values for the T cell land mask (hm) and
    makes the logical land masks for T and U cells (tmask, umask)
    and N and E cells (nmask, emask).
    Also creates hemisphere masks (mask-n northern, mask-s southern)

    author: Elizabeth C. Hunke, LANL
    """
    puny = icepack_query_parameters("puny")["puny"]
    ice_halo_update(kmt, field_loc_center, field_type_scalar)
    ice_halo_update(hm, field_loc_center, field_type_scalar)

    #-----------------------------------------------------------------
    # construct T-cell and U-cell masks
    #-----------------------------------------------------------------
    # beginning and end of physical domain
    ilo, ihi, jlo, jhi = physical_bounds()
    uvm[ilo:ihi + 1, jlo:jhi + 1] = np.minimum(
        np.minimum(_view(hm, 0, 0), _view(hm, 1, 0)),
        np.minimum(_view(hm, 0, 1), _view(hm, 1, 1)),
    )
    npm[ilo:ihi + 1, jlo:jhi + 1] = np.minimum(_view(hm, 0, 0), _view(hm, 0, 1))
    epm[ilo:ihi + 1, jlo:jhi + 1] = np.minimum(_view(hm, 0, 0), _view(hm, 1, 0))
    bm[ilo:ihi + 1, jlo:jhi + 1] = dbl_kind(0.01)
    uvm_cd = _zero(hm.shape)
    uvm_cd[ilo:ihi + 1, jlo:jhi + 1] = (
        _view(hm, 0, 0) + _view(hm, 1, 0) + _view(hm, 0, 1) + _view(hm, 1, 1)
    )

    ice_halo_update(uvm, field_loc_NEcorner, field_type_scalar)
    ice_halo_update(uvm_cd, field_loc_NEcorner, field_type_scalar)
    ice_halo_update(npm, field_loc_Nface, field_type_scalar)
    ice_halo_update(epm, field_loc_Eface, field_type_scalar)
    ice_halo_update(bm, field_loc_center, field_type_scalar)
    ice_halo_update(ocn_gridcell_frac, field_loc_center, field_type_scalar)

    # needs to cover halo (no halo update for logicals)
    tmask[...] = hm > p5
    umask[...] = uvm > p5
    umaskCD[...] = uvm_cd > c1p5
    nmask[...] = npm > p5
    emask[...] = epm > p5
    opmask[...] = (ocn_gridcell_frac > puny) & np.logical_not(tmask)

    #-----------------------------------------------------------------
    # create hemisphere masks
    #-----------------------------------------------------------------
    lmask_n[...] = False
    lmask_s[...] = False
    tarean[...] = c0
    tareas[...] = c0
    north = _view(ULAT, 0, 0) >= -puny
    lmask_n[ilo:ihi + 1, jlo:jhi + 1] = north  # N. Hem.
    lmask_s[ilo:ihi + 1, jlo:jhi + 1] = np.logical_not(north)  # S. Hem.
    # N hemisphere area mask (m^2)
    tarean[ilo:ihi + 1, jlo:jhi + 1] = np.where(north, _view(tarea, 0, 0) * _view(hm, 0, 0), c0)
    # S hemisphere area mask (m^2)
    tareas[ilo:ihi + 1, jlo:jhi + 1] = np.where(
        np.logical_not(north), _view(tarea, 0, 0) * _view(hm, 0, 0), c0
    )


def Tlatlon():
    """Initializes latitude and longitude on T grid

    author: Elizabeth C. Hunke, LANL; code originally based on POP grid
    generation routine
    """
    ice_fileunits.nu_diag.write(" (Tlatlon) called\n")
    TLON[...] = c0
    TLAT[...] = c0
    # beginning and end of physical domain
    ilo, ihi, jlo, jhi = physical_bounds()
    # ---------
    # TLON/TLAT 4 pt computation (pts 1, 2, 3, 4)
    # ---------
    # TLON in radians East
    # TLAT in radians North
    points = (
        _corner_xyz(-1, -1),
        _corner_xyz(0, -1),
        _corner_xyz(-1, 0),
        _corner_xyz(0, 0),
    )
    tx = sum(point[0] for point in points) / c4
    ty = sum(point[1] for point in points) / c4
    tz = sum(point[2] for point in points) / c4
    tz = tz / np.sqrt(tx * tx + ty * ty + tz * tz)
    # in radians East
    lon = np.zeros(tx.shape, dtype=dbl_kind)
    use = (tx != c0) | (ty != c0)
    lon[use] = np.arctan2(ty[use], tx[use])
    # in radians North
    lat = np.arcsin(tz)
    TLON[ilo:ihi + 1, jlo:jhi + 1] = lon
    TLAT[ilo:ihi + 1, jlo:jhi + 1] = lat
    # for W boundary extrapolate from interior
    if str(ice_init.grid_type).strip() in ("regional", "rectangular"):
        ig, _jg = global_indices()
        if int(ig[ilo]) == 1:
            rows = slice(jlo, jhi + 1)
            TLON[ilo, rows] = c2 * TLON[ilo + 1, rows] + (-c1) * TLON[ilo + 2, rows]
            TLAT[ilo, rows] = c2 * TLAT[ilo + 1, rows] + (-c1) * TLAT[ilo + 2, rows]
    ice_halo_update(TLON, field_loc_center, field_type_scalar, fill_value=c1)
    ice_halo_update(TLAT, field_loc_center, field_type_scalar, fill_value=c1)
    ice_halo_extrapolate(TLON)
    ice_halo_extrapolate(TLAT)


def NElatlon():
    """Initializes latitude and longitude on N, E grid

    author: T. Craig from Tlatlon
    """
    ice_fileunits.nu_diag.write(" (NElatlon) called\n")
    NLON[...] = c0
    NLAT[...] = c0
    ELON[...] = c0
    ELAT[...] = c0
    # beginning and end of physical domain
    ilo, ihi, jlo, jhi = physical_bounds()
    # ---------
    # NLON/NLAT 2 pt computation (pts 3, 4)
    # ---------
    # NLON in radians East
    # NLAT in radians North
    n_points = (_corner_xyz(-1, 0), _corner_xyz(0, 0))
    ntx = sum(point[0] for point in n_points) / c2
    nty = sum(point[1] for point in n_points) / c2
    ntz = sum(point[2] for point in n_points) / c2
    ntz = ntz / np.sqrt(ntx * ntx + nty * nty + ntz * ntz)
    # in radians East
    nlon = np.zeros(ntx.shape, dtype=dbl_kind)
    n_use = (ntx != c0) | (nty != c0)
    nlon[n_use] = np.arctan2(nty[n_use], ntx[n_use])
    # in radians North
    nlat = np.arcsin(ntz)
    # ---------
    # ELON/ELAT 2 pt computation (pts 2, 4)
    # ---------
    # ELON in radians East
    # ELAT in radians North
    e_points = (_corner_xyz(0, -1), _corner_xyz(0, 0))
    etx = sum(point[0] for point in e_points) / c2
    ety = sum(point[1] for point in e_points) / c2
    etz = sum(point[2] for point in e_points) / c2
    etz = etz / np.sqrt(etx * etx + ety * ety + etz * etz)
    # in radians East
    elon = np.zeros(etx.shape, dtype=dbl_kind)
    e_use = (etx != c0) | (ety != c0)
    elon[e_use] = np.arctan2(ety[e_use], etx[e_use])
    # in radians North
    elat = np.arcsin(etz)
    NLON[ilo:ihi + 1, jlo:jhi + 1] = nlon
    NLAT[ilo:ihi + 1, jlo:jhi + 1] = nlat
    ELON[ilo:ihi + 1, jlo:jhi + 1] = elon
    ELAT[ilo:ihi + 1, jlo:jhi + 1] = elat
    # for W boundary extrapolate from interior
    if str(ice_init.grid_type).strip() in ("regional", "rectangular"):
        ig, _jg = global_indices()
        if int(ig[ilo]) == 1:
            rows = slice(jlo, jhi + 1)
            NLON[ilo, rows] = c1p5 * NLON[ilo + 1, rows] + (-p5) * NLON[ilo + 2, rows]
            NLAT[ilo, rows] = c1p5 * NLAT[ilo + 1, rows] + (-p5) * NLAT[ilo + 2, rows]
    ice_halo_update(NLON, field_loc_Nface, field_type_scalar, fill_value=c1)
    ice_halo_update(NLAT, field_loc_Nface, field_type_scalar, fill_value=c1)
    ice_halo_update(ELON, field_loc_Eface, field_type_scalar, fill_value=c1)
    ice_halo_update(ELAT, field_loc_Eface, field_type_scalar, fill_value=c1)
    ice_halo_extrapolate(NLON)
    ice_halo_extrapolate(NLAT)
    ice_halo_extrapolate(ELON)
    ice_halo_extrapolate(ELAT)

    # min/max over the physical domain, converted to degrees
    rad_to_deg = icepack_query_parameters("rad_to_deg")["rad_to_deg"]

    def _span(field, mask):
        values = np.asarray(field[ilo:ihi + 1, jlo:jhi + 1])
        use = np.asarray(mask[ilo:ihi + 1, jlo:jhi + 1])
        kept = values[use]
        return float(np.min(kept)) * rad_to_deg, float(np.max(kept)) * rad_to_deg

    y1, y2 = _span(ULON, umask)
    y3, y4 = _span(ULAT, umask)
    x1, x2 = _span(TLON, tmask)
    x3, x4 = _span(TLAT, tmask)
    nu = ice_fileunits.nu_diag
    nu.write(" \n")
    nu.write(f" (NElatlon) min/max ULON: {y1} {y2}\n")
    nu.write(f" (NElatlon) min/max ULAT: {y3} {y4}\n")
    nu.write(f" (NElatlon) min/max TLON: {x1} {x2}\n")
    nu.write(f" (NElatlon) min/max TLAT: {x3} {x4}\n")
    x1, x2 = _span(NLON, nmask)
    x3, x4 = _span(NLAT, nmask)
    y1, y2 = _span(ELON, emask)
    y3, y4 = _span(ELAT, emask)
    nu.write(" \n")
    nu.write(f" (NElatlon) min/max NLON: {x1} {x2}\n")
    nu.write(f" (NElatlon) min/max NLAT: {x3} {x4}\n")
    nu.write(f" (NElatlon) min/max ELON: {y1} {y2}\n")
    nu.write(f" (NElatlon) min/max ELAT: {y3} {y4}\n")
    nu.write(" \n")


def _clamp_angle(array, pi):
    # fix ANGLE: roundoff error due to single precision
    array[array > pi] = pi
    array[array < -pi] = -pi


def get_bathymetry():
    """ocean bathymetry for grounded sea ice (seabed stress) or icebergs
    currently hardwired for 40 levels (gx3, gx1 grids)
    should be read from a file instead (see subroutine read_seabedstress_bathy)
    """
    params = icepack_query_parameters("calc_dragio")
    if ice_init.use_bathymetry:
        # Read bathymetry data for seabed stress calculation (grounding scheme for
        # landfast ice) in CICE stand-alone mode. When CICE is in coupled mode
        # (e.g. CICE-NEMO), hwater should be uptated at each time level so that
        # it varies with ocean dynamics.
        #
        # author: Fred Dupont, CMC
        ice_fileunits.nu_diag.write(f"\nBathymetry file: {ice_init.bathymetry_file}\n")
        # input data file names
        ice_read_write.ice_read_nc(
            ice_init.bathymetry_file,
            "Bathymetry",
            bathymetry,
            True,
            field_loc_center,
            field_type_scalar,
        )
        return
    nlevel = 40  # number of layers (gx3 grid)
    # ocean layer thickness, m
    thick = np.array(
        [
            10.01244, 10.11258, 10.31682, 10.63330, 11.07512, 11.66145,
            12.41928, 13.38612, 14.61401, 16.17561, 18.17368, 20.75558,
            24.13680, 28.63821, 34.74644, 43.20857, 55.16812, 72.30458,
            96.74901, 130.0392, 170.0489, 207.9933, 233.5694, 245.2719,
            248.9804, 249.8322, 249.9787, 249.9979, 249.9998, 250.0000,
            250.0000, 250.0000, 250.0000, 250.0000, 250.0000, 250.0000,
            250.0000, 250.0000, 250.0000, 250.0000,
        ],
        dtype=dbl_kind,
    )
    # total depth, m
    # convert to total depth
    depth = np.cumsum(thick)
    k = np.minimum(np.rint(kmt).astype(np.int32), nlevel)
    bathymetry[...] = c0
    ocean = k > 0
    bathymetry[ocean] = depth[k[ocean] - 1]
    # For consistency, set thickness_ocn_layer1 in Icepack if 'calc_dragio' is active
    if params["calc_dragio"]:
        icepack_init_parameters(thickness_ocn_layer1=thick[0])


def get_bathymetry_popfile():
    """with use_bathymetry = false, vertical depth profile generated for max KMT
    with use_bathymetry = true, expects to read in pop vert_grid file
    """
    params = icepack_query_parameters("calc_dragio")
    nlevel = int(np.max(np.rint(kmt)))  # number of levels (max KMT)
    if nlevel < 1:
        bathymetry[...] = c0
        return
    ice_fileunits.nu_diag.write(f"(get_bathymetry_popfile) KMT max = {nlevel}\n")
    if ice_init.use_bathymetry:
        ice_fileunits.nu_diag.write(
            f"(get_bathymetry_popfile) Bathymetry file = {ice_init.bathymetry_file}\n"
        )
        thick = np.empty(nlevel, dtype=dbl_kind)
        with open(ice_init.bathymetry_file, encoding="utf-8") as stream:
            for k in range(nlevel):
                line = stream.readline()
                if line == "":
                    raise ValueError("(get_bathymetry_popfile) read error")
                thick[k] = dbl_kind(float(line.split()[0]))
    else:
        # create thickness profile
        thick = np.empty(nlevel, dtype=dbl_kind)
        k1 = min(5, nlevel)  # levels
        thick[:k1] = max(dbl_kind(10000.0) / dbl_kind(nlevel), dbl_kind(500.0))
        for k in range(k1, nlevel):
            thick[k] = min(thick[k - 1] * dbl_kind(1.2), dbl_kind(20000.0))
    # convert thick from cm to m
    # layer thickness, cm -> m
    thick = thick / dbl_kind(100.0)
    # total depth, m
    # convert to total depth
    depth = np.cumsum(thick)
    if np.any(depth < c0):
        raise RuntimeError("(get_bathymetry_popfile) negative depth error")
    for k in range(nlevel):
        ice_fileunits.nu_diag.write(
            f"(get_bathymetry_popfile)   k, thick(m), depth(m) = "
            f"{k + 1:6d} {thick[k]:13.7f} {depth[k]:13.7f}\n"
        )
    level = np.rint(kmt).astype(np.int32)
    if np.any(level > nlevel):
        raise RuntimeError("(get_bathymetry_popfile) kmt gt nlevel error")
    bathymetry[...] = c0
    ocean = level > 0
    bathymetry[ocean] = depth[level[ocean] - 1]
    # For consistency, set thickness_ocn_layer1 in Icepack if 'calc_dragio' is active
    if params["calc_dragio"]:
        icepack_init_parameters(thickness_ocn_layer1=thick[0])


def set_coriolis():
    """Coriolis parameter. U points always; E and N points on a C or CD grid.

    Fortran computes this in ``init_dyn_shared``; it is computed here
    because this step owns the latitude arrays.
    """
    kind = str(ice_init.coriolis).strip()
    # Coriolis parameter
    if kind == "constant":
        fcor[...] = dbl_kind(1.46e-4)  # Hibler 1979, N. Hem; 1/s
    elif kind == "zero":
        fcor[...] = c0
    else:
        fcor[...] = c2 * omega * np.sin(ULAT)  # 1/s
    if str(ice_init.grid_ice).strip() not in ("C", "CD"):
        return
    if kind == "constant":
        fcorE[...] = dbl_kind(1.46e-4)  # Hibler 1979, N. Hem; 1/s
        fcorN[...] = dbl_kind(1.46e-4)  # Hibler 1979, N. Hem; 1/s
    elif kind == "zero":
        fcorE[...] = c0
        fcorN[...] = c0
    else:
        fcorE[...] = c2 * omega * np.sin(ELAT)  # 1/s
        fcorN[...] = c2 * omega * np.sin(NLAT)  # 1/s


def grid_average_X2Y_jax(avg_type, work1, grid1, work2, grid2, constants, bounds):
    """JAX version of the physical-cell grid shift used by step_prep."""
    if grid1 == grid2:
        return work1
    direction = _SHIFT[(grid1, grid2)]
    area1 = constants[{"T": "tarea", "U": "uarea", "N": "narea", "E": "earea"}[grid1]]
    mask1 = constants[{"T": "hm", "U": "uvm", "N": "npm", "E": "epm"}[grid1]]
    area2 = constants[{"T": "tarea", "U": "uarea", "N": "narea", "E": "earea"}[grid2]]
    ilo, ihi, jlo, jhi = bounds
    acc = jnp.zeros_like(work2[ilo:ihi + 1, jlo:jhi + 1])
    acc_w = jnp.zeros_like(acc)
    for di, dj in _OFFSETS[direction]:
        cells = (slice(ilo + di, ihi + 1 + di), slice(jlo + dj, jhi + 1 + dj))
        values = work1[cells]
        weight = area1[cells]
        if avg_type == "F":
            acc = acc + values * weight
        elif avg_type == "S":
            mask = mask1[cells]
            acc_w = acc_w + mask * weight
            acc = acc + mask * values * weight
        elif avg_type == "A":
            acc_w = acc_w + weight
            acc = acc + values * weight
        else:
            raise ValueError(f"(grid_average_X2Y_jax) unknown type {avg_type}")
    if avg_type == "F":
        values = (1.0 / len(_OFFSETS[direction])) * acc / area2[ilo:ihi + 1, jlo:jhi + 1]
    else:
        values = jnp.where(acc_w != 0, acc / jnp.where(acc_w != 0, acc_w, 1), 0)
    return jnp.zeros_like(work2).at[ilo:ihi + 1, jlo:jhi + 1].set(values)


def grid_average_X2Y(avg_type, work1, grid1, work2, grid2):
    """Shifts quantities from one grid to another
    Constructs the shift based on the grid
    NOTE: Input array includes ghost cells that must be updated before
          calling this routine.

    author: T. Craig

    Only physical points of ``work2`` are written. Same-grid arguments
    copy the whole array.
    """
    if grid1 == grid2:
        work2[...] = work1
        return work2
    try:
        direction = _SHIFT[(grid1, grid2)]
    except KeyError as exc:
        raise ValueError(f"(grid_average_X2Y) unknown shift {grid1} to {grid2}") from exc
    if grid1 == "T":
        area1, mask1 = tarea, hm
    elif grid1 == "U":
        area1, mask1 = uarea, uvm
    elif grid1 == "N":
        area1, mask1 = narea, npm
    elif grid1 == "E":
        area1, mask1 = earea, epm
    else:
        raise ValueError(f"(grid_average_X2Y) unknown grid {grid1}")
    if grid2 == "T":
        area2 = tarea
    elif grid2 == "U":
        area2 = uarea
    elif grid2 == "N":
        area2 = narea
    elif grid2 == "E":
        area2 = earea
    else:
        raise ValueError(f"(grid_average_X2Y) unknown grid {grid2}")
    work2[...] = c0
    # beginning and end of physical domain
    ilo, ihi, jlo, jhi = physical_bounds()
    shape = (ihi - ilo + 1, jhi - jlo + 1)
    acc = np.zeros(shape, dtype=dbl_kind)
    acc_w = np.zeros(shape, dtype=dbl_kind)
    for di, dj in _OFFSETS[direction]:
        values = work1[ilo + di:ihi + 1 + di, jlo + dj:jhi + 1 + dj]
        weight = area1[ilo + di:ihi + 1 + di, jlo + dj:jhi + 1 + dj]
        if avg_type == "F":
            # Flux masked, original implementation based on earlier t2u and u2t versions
            # flux unmasked
            acc += values * weight
        elif avg_type == "S":
            # State masked version, simple area weighted averager
            # state masked
            mask = mask1[ilo + di:ihi + 1 + di, jlo + dj:jhi + 1 + dj]
            acc_w += mask * weight
            acc += mask * values * weight
        elif avg_type == "A":
            # State unmasked version, simple weighted averager
            # state unmasked
            acc_w += weight
            acc += values * weight
        else:
            raise ValueError(f"(grid_average_X2Y) unknown type {avg_type}")
    dest = work2[ilo:ihi + 1, jlo:jhi + 1]
    if avg_type == "F":
        dest[...] = (c1 / dbl_kind(len(_OFFSETS[direction]))) * acc / area2[ilo:ihi + 1, jlo:jhi + 1]
    else:
        good = acc_w != c0
        dest[good] = acc[good] / acc_w[good]
    return work2


def grid_average_X2Y_NE(avg_type, work1a, grid1a, work1b, grid1b, work2, grid2):
    """Shifts quantities from one grid to another
    NOTE: Input array includes ghost cells that must be updated before
          calling this routine.

    author: T. Craig

    Dual-input NE/EN average. ``grid_average_X2Y_2`` keeps the N field
    on ``work1a`` and the E field on ``work1b``; an EN name swaps them.
    """
    # X2Y = trim(grid1a)//trim(grid1b)//'2'//trim(grid2)//trim(type)
    x2y = f"{grid1a}{grid1b}2{grid2}{avg_type}"
    # state masked
    # state unmasked
    if x2y in ("EN2US", "EN2TS", "EN2UA", "EN2TA"):
        work_n, work_e = work1b, work1a
        x2y = "NE" + x2y[2:]
    elif x2y in ("NE2US", "NE2TS", "NE2UA", "NE2TA"):
        work_n, work_e = work1a, work1b
    else:
        raise ValueError(f"(grid_average_X2Y_NEversion) ERROR: unknown X2Y {x2y}")
    # NE2US / NE2UA: (i,j) and (i+1,j) from N, (i,j) and (i,j+1) from E
    # NE2TS / NE2TA: (i,j-1) and (i,j) from N, (i-1,j) and (i,j) from E
    if x2y in ("NE2US", "NE2UA"):
        terms = (
            (work_n, narea, npm, 0, 0),
            (work_n, narea, npm, 1, 0),
            (work_e, earea, epm, 0, 0),
            (work_e, earea, epm, 0, 1),
        )
    else:
        terms = (
            (work_n, narea, npm, 0, -1),
            (work_n, narea, npm, 0, 0),
            (work_e, earea, epm, -1, 0),
            (work_e, earea, epm, 0, 0),
        )
    masked = x2y in ("NE2US", "NE2TS")
    ilo, ihi, jlo, jhi = physical_bounds()
    work2[...] = c0
    acc = np.zeros((ihi - ilo + 1, jhi - jlo + 1), dtype=dbl_kind)
    acc_w = np.zeros(acc.shape, dtype=dbl_kind)
    for work, weight_all, mask_all, di, dj in terms:
        sl = (slice(ilo + di, ihi + 1 + di), slice(jlo + dj, jhi + 1 + dj))
        # first field is N (narea, npm); second field is E (earea, epm)
        weight = weight_all[sl]
        mask = mask_all[sl]
        if masked:
            acc_w += mask * weight
            acc += mask * work[sl] * weight
        else:
            acc_w += weight
            acc += work[sl] * weight
    # if (wtmp /= c0) work2 = acc / wtmp
    dest = work2[ilo:ihi + 1, jlo:jhi + 1]
    good = acc_w != c0
    dest[good] = acc[good] / acc_w[good]
    return work2


def _nx_ny():
    return int(ice_domain_size.nx_global), int(ice_domain_size.ny_global)


def popgrid_nc():
    """POP displaced pole grid and land mask.

    Grid record number, field and units are:
    (1) ULAT  (radians)
    (2) ULON  (radians)
    (3) HTN   (cm)
    (4) HTE   (cm)
    (5) HUS   (cm)
    (6) HUW   (cm)
    (7) ANGLE (radians)

    author: Elizabeth C. Hunke, LANL
    Revised for netcdf input: Ann Keen, Met Office, May 2007

    Lengths are still centimeters in the file.
    """
    global l_readCenter
    path = ice_init.grid_file
    pi = icepack_query_parameters("pi")["pi"]
    #-----------------------------------------------------------------
    # lat, lon, angle
    #-----------------------------------------------------------------
    _place(ULAT, ice_read_write.ice_read_global_nc(path, "ulat"), field_loc_NEcorner, field_type_scalar)  # ULAT
    _place(ULON, ice_read_write.ice_read_global_nc(path, "ulon"), field_loc_NEcorner, field_type_scalar)  # ULON
    _place(ANGLE, ice_read_write.ice_read_global_nc(path, "angle"), field_loc_NEcorner, field_type_angle)  # ANGLE
    _clamp_angle(ANGLE, pi)
    # if grid file includes anglet then read instead
    if ice_read_write.nc_has_var(path, "anglet"):
        ice_fileunits.nu_diag.write(
            "(popgrid_nc) angleT, TLON and TLAT is read from grid file\n"
        )
        l_readCenter = True
        _place(
            ANGLET,
            ice_read_write.ice_read_global_nc(path, "anglet"),
            field_loc_center,
            field_type_angle,
        )
        _clamp_angle(ANGLET, pi)
        _place(TLON, ice_read_write.ice_read_global_nc(path, "tlon"), field_loc_center, field_type_scalar)
        _place(TLAT, ice_read_write.ice_read_global_nc(path, "tlat"), field_loc_center, field_type_scalar)
    else:
        ice_fileunits.nu_diag.write("(popgrid_nc) CICE will calculate angleT, TLON and TLAT\n")
    #-----------------------------------------------------------------
    # cell dimensions
    # calculate derived quantities from global arrays to preserve
    # information on boundaries
    #-----------------------------------------------------------------
    primary_grid_lengths_HTN(ice_read_write.ice_read_global_nc(path, "htn"))  # HTN; dxU, dxT, dxN, dxE
    # dxu, dxt
    primary_grid_lengths_HTE(ice_read_write.ice_read_global_nc(path, "hte"))  # HTE; dyU, dyT, dyN, dyE
    # dyu, dyt


def geosgrid_nc():
    """GEOS MOM grid
    Grid record number, field and units are:
    (1) ULAT  (radians)
    (2) ULON  (radians)
    (3) ANGLE (radians)
    (4) ANGLET (radians)
    (5) HTN   (cm)
    (6) HTE   (cm)

    Land mask record number and field is (1) KMT.

    Same variables as ``popgrid_nc``.
    """
    global l_readCenter
    path = ice_init.grid_file
    pi = icepack_query_parameters("pi")["pi"]
    #-----------------------------------------------------------------
    # lat, lon, angle
    #-----------------------------------------------------------------
    _place(ULAT, ice_read_write.ice_read_global_nc(path, "ulat"), field_loc_NEcorner, field_type_scalar)  # ULAT
    _place(ULON, ice_read_write.ice_read_global_nc(path, "ulon"), field_loc_NEcorner, field_type_scalar)  # ULON
    _place(ANGLE, ice_read_write.ice_read_global_nc(path, "angle"), field_loc_NEcorner, field_type_angle)  # ANGLE
    _clamp_angle(ANGLE, pi)
    # if grid file includes anglet then read instead
    if ice_read_write.nc_has_var(path, "anglet"):
        ice_fileunits.nu_diag.write(
            "(popgrid_nc) angleT, TLON and TLAT is read from grid file\n"
        )
        l_readCenter = True
        _place(
            ANGLET,
            ice_read_write.ice_read_global_nc(path, "anglet"),
            field_loc_center,
            field_type_angle,
        )
        _clamp_angle(ANGLET, pi)
        _place(TLON, ice_read_write.ice_read_global_nc(path, "tlon"), field_loc_center, field_type_scalar)
        _place(TLAT, ice_read_write.ice_read_global_nc(path, "tlat"), field_loc_center, field_type_scalar)
    else:
        ice_fileunits.nu_diag.write("(popgrid_nc) CICE will calculate angleT, TLON and TLAT\n")
    #-----------------------------------------------------------------
    # cell dimensions
    # calculate derived quantities from global arrays to preserve
    # information on boundaries
    #-----------------------------------------------------------------
    primary_grid_lengths_HTN(ice_read_write.ice_read_global_nc(path, "htn"))  # HTN; dxU, dxT, dxN, dxE
    # dxu, dxt
    primary_grid_lengths_HTE(ice_read_write.ice_read_global_nc(path, "hte"))  # HTE; dyU, dyT, dyN, dyE
    # dyu, dyt


def popgrid_nc_ext():
    """POP extended displaced pole grid and land mask.

    Grid record number, field and units are:
    (1) ULAT  (radians)
    (2) ULON  (radians)
    (3) HTN   (cm)
    (4) HTE   (cm)
    (5) HUS   (cm)
    (6) HUW   (cm)
    (7) ANGLE (radians)

    author: T. Craig
    Revised for netcdf input: Ann Keen, Met Office, May 2007

    The file already contains the ghost rim.
    """
    global l_readCenter
    pi = icepack_query_parameters("pi")["pi"]
    ghost = int(nghost)
    nx, ny = _nx_ny()
    shape = (nx + 2 * ghost, ny + 2 * ghost)  # temporary global extended array
    path = ice_init.grid_file

    def _read(name):
        return ice_read_write.ice_read_global_nc(path, name, shape=shape)

    #-----------------------------------------------------------------
    # lat, lon, angle
    #-----------------------------------------------------------------
    scatter_global_ext(ULAT, _read("ulat"))  # ULAT
    scatter_global_ext(ULON, _read("ulon"))  # ULON
    scatter_global_ext(ANGLE, _read("angle"))  # ANGLE
    _clamp_angle(ANGLE, pi)
    # if grid file includes anglet then read instead
    if ice_read_write.nc_has_var(path, "anglet"):
        ice_fileunits.nu_diag.write(
            "(popgrid_nc_ext) angleT, TLON and TLAT is read from grid file\n"
        )
        l_readCenter = True
        scatter_global_ext(ANGLET, _read("anglet"))
        _clamp_angle(ANGLET, pi)
        scatter_global_ext(TLON, _read("tlon"))
        scatter_global_ext(TLAT, _read("tlat"))
    else:
        ice_fileunits.nu_diag.write("(popgrid_nc_ext) CICE will calculate angleT, TLON and TLAT\n")

    #-----------------------------------------------------------------
    # cell dimensions
    # calculate derived quantities from global arrays to preserve
    # information on boundaries
    #-----------------------------------------------------------------
    # call primary_grid_lengths_HTN(work_g1)                  ! dxU, dxT, dxN, dxE
    htn = _read("htn") * cm_to_m  # HTN
    scatter_global_ext(HTN, htn)
    if ice_init.save_ghte_ghtn:
        G_HTN[...] = HTN
    dxN[...] = HTN
    # beginning and end of physical domain
    ilo, ihi, jlo, jhi = physical_bounds()
    here = _view(HTN, 0, 0)
    at_u = _view(HTN, 1, 0)
    at_t = _view(HTN, 0, -1)
    diag = _view(HTN, 1, -1)
    dxU[ilo:ihi + 1, jlo:jhi + 1] = p5 * (here + at_u)
    dxT[ilo:ihi + 1, jlo:jhi + 1] = p5 * (here + at_t)
    dxE[ilo:ihi + 1, jlo:jhi + 1] = p25 * (here + at_u + at_t + diag)
    for field in (dxU, dxT, dxE):
        ice_halo_update(field, field_loc_center, field_type_scalar)
        ice_halo_extrapolate(field)

    # call primary_grid_lengths_HTE(work_g1)                  ! dyU, dyT, dyN, dyE
    hte = _read("hte") * cm_to_m  # HTE
    # HTN
    scatter_global_ext(HTE, hte)
    if ice_init.save_ghte_ghtn:
        G_HTE[...] = HTE
    dyE[...] = HTE
    # beginning and end of physical domain
    ilo, ihi, jlo, jhi = physical_bounds()
    here = _view(HTE, 0, 0)
    at_u = _view(HTE, 0, 1)
    at_t = _view(HTE, -1, 0)
    diag = _view(HTE, -1, 1)
    dyU[ilo:ihi + 1, jlo:jhi + 1] = p5 * (here + at_u)
    dyT[ilo:ihi + 1, jlo:jhi + 1] = p5 * (here + at_t)
    dyN[ilo:ihi + 1, jlo:jhi + 1] = p25 * (here + at_u + at_t + diag)
    for field in (dyU, dyT, dyN):
        ice_halo_update(field, field_loc_center, field_type_scalar)
        ice_halo_extrapolate(field)


def grid_boxislands_kmt(work):
    """Complex land mask for testing box cases
    Requires nx_global, ny_global > 20
    Assumes work array has been initialized to 1 (ocean) and north and
    south land boundaries have been applied (ew_boundary_type='cyclic')
    """
    nx, ny = work.shape
    # number of cells in 5% of global grid x and y lengths
    # convenient cell-block sizes for building the mask
    nxb = int(dbl_kind(nx) / c20)
    nyb = int(dbl_kind(ny) / c20)
    if nxb < 1 or nyb < 1:
        raise RuntimeError("(grid_boxislands_kmt) requires larger grid size")
    # initialize work area as all ocean (c1).
    work[...] = c1

    def land(i0, i1, j0, j1):
        # Fortran loops are inclusive and 1-based.
        if i0 > i1 or j0 > j1:
            return
        work[i0 - 1:i1, j0 - 1:j1] = c0

    # now add land points (c0)
    # northeast triangle
    k = 0
    for j in range(ny, ny - 3 * nyb - 1, -1):
        k += 1
        land(nx - 3 * nxb + k, nx, j, j)
    # northwest docks
    land(1, 1, ny - 3 * nyb, ny)
    for i in range(1, 2 * nxb + 1):
        land(i, i, ny - 3 * nyb, ny - nyb - 2)
        land(i, i, ny - nyb, ny - nyb + 1)
    # southwest docks
    land(1, 1, 2 * nyb, 3 * nyb)
    for j in range(1, 2 * nyb + 1):
        land(2, nxb, j, j)
        land(2 * nxb - 1, 2 * nxb, j, j)
        land(2 * nxb + 2, 4 * nxb, j, j)
    # tiny island
    land(14 * nxb, 14 * nxb + 1, 14 * nyb, 14 * nyb + 1)
    # X islands
    # left triangle
    k = 0
    for i in range(2 * nxb, 4 * nxb + 1):
        k += 1
        land(i, i, 10 * nyb + k, 14 * nyb - k)
    # upper triangle
    k = 0
    for j in range(14 * nyb, 12 * nyb - 1, -1):
        k += 1
        land(2 * nxb + 2 + k, 6 * nxb - 2 - k, j, j)
    # diagonal
    k = 0
    for j in range(10 * nyb, 14 * nyb + 1):
        k += 1
        land(2 * nxb + 4 + k, 2 * nxb + 6 + k, j, j)
    # lower right triangle
    k = 0
    for j in range(12 * nyb, 10 * nyb - 1, -1):
        k += 1
        land(5 * nxb + k, 8 * nxb, j, j)
    # bar islands
    for i in range(10 * nxb, 16 * nxb + 1):
        land(i, i, 4 * nyb, 5 * nyb)
        land(i, i, 6 * nyb + 2, 8 * nyb)
        land(i, i, 8 * nyb + 2, 8 * nyb + 3)


def rectgrid_scale_dxdy():
    """generate a variable spaced rectangluar grid.
    extend spacing from center of grid outward.

    Variable dx and dy, grown from the center by ``dxscale`` and ``dyscale``.
    """
    nx, ny = _nx_ny()
    rad_to_deg = icepack_query_parameters("rad_to_deg")["rad_to_deg"]
    # determine dx spacing
    # strategy: initialize with dxrect.
    # if want to scale the grid, work from center outwards,
    # multplying neighbor cell by scale factor.
    # this assumes dx varies in x direction only.
    # (i.e, dx is the same across same y location)
    # initialize with initial dxrect
    work = np.full((nx, ny), dbl_kind(ice_init.dxrect), dtype=dbl_kind)
    # check if nx is even or odd
    # if even, middle 2 columns are center
    # of odd,  middle 1 column is center
    # array centers for expanding dx, dy
    if nx % 2 == 0:  # nx_global is even
        # with even number of x locatons,
        # the center two y columns are center
        center1 = nx // 2  # integer math
        center2 = center1 + 1  # integer math
    else:  # nx_global = odd
        # only one center index. set center2=center1
        center1 = nx // 2
        center2 = center1
    # note loop over only half the x grid points (center1)-1
    # working from the center outward.
    for i in range(1, center1):
        # work from center1 to left
        work[center1 - i - 1, :] = dbl_kind(ice_init.dxscale) * work[center1 - i, :]
        # work from center2 to right
        work[center2 + i - 1, :] = dbl_kind(ice_init.dxscale) * work[center2 + i - 2, :]
    # note work_g1 is converted to meters in primary_grid_lengths_HTN
    primary_grid_lengths_HTN(work)  # dxU, dxT, dxN, dxE
    # make ULON array
    # make first column reference lon in radians.
    # the remaining work_g1 is still dx in meters
    work[0, :] = dbl_kind(ice_init.lonrefrect) / rad_to_deg  # radians
    # loop over remaining points and add spacing to successive
    # x locations
    for i in range(1, nx):  # start from i=2. i=1 is lonrefrect
        length = work[i, :] / radius  # grid spacing in radians
        work[i, :] = work[i - 1, :] + length  # ULON
    _place(ULON, work, field_loc_NEcorner, field_type_scalar)

    # determine dy spacing
    # strategy: initialize with dyrect.
    # if want to scale the grid, work from center outwards,
    # multplying neighbor cell by scale factor.
    # this assumes dy varies in y direction only.
    # (i.e, dy is the same across same x location)
    # initialize with initial dxrect
    work = np.full((nx, ny), dbl_kind(ice_init.dyrect), dtype=dbl_kind)
    # check if ny is even or odd
    # if even, middle 2 rows are center
    # of odd,  middle 1 row is center
    if ny % 2 == 0:  # ny_global is even
        # with even number of x locatons,
        # the center two y columns are center
        center1 = ny // 2  # integer math
        center2 = center1 + 1  # integer math
    else:  # ny_global = odd
        # only one center index. set center2=center1
        center1 = ny // 2
        center2 = center1
    # note loop over only half the y grid points (center1)-1
    # working from the center outward.
    for j in range(1, center1):
        # work from center1 to bottom
        work[:, center1 - j - 1] = dbl_kind(ice_init.dyscale) * work[:, center1 - j]
        # work from center2 to top
        work[:, center2 + j - 1] = dbl_kind(ice_init.dyscale) * work[:, center2 + j - 2]
    # note work_g1 is converted to meters primary_grid_lengths_HTE
    primary_grid_lengths_HTE(work)  # dyU, dyT, dyN, dyE
    # make ULAT array
    # make first row reference lat in radians.
    # the remaining work_g1 is still dy in meters
    work[:, 0] = dbl_kind(ice_init.latrefrect) / rad_to_deg  # radians
    # loop over remaining points and add spacing to successive
    # x locations
    for j in range(1, ny):  # start from j=2. j=1 is latrefrect
        length = work[:, j] / radius  # grid spacing in radians
        work[:, j] = work[:, j - 1] + length  # ULAT
    _place(ULAT, work, field_loc_NEcorner, field_type_scalar)


def rectgrid():
    """Regular rectangular grid and mask

    author: Elizabeth C. Hunke, LANL

    Latitudes and longitudes are in radians.
    """
    nx, ny = _nx_ny()
    rad_to_deg = icepack_query_parameters("rad_to_deg")["rad_to_deg"]
    #-----------------------------------------------------------------
    # Calculate various geometric 2d arrays
    #-----------------------------------------------------------------
    hm[...] = c0
    kmt[...] = c0
    ANGLE[...] = c0  # "square with the world"
    if ice_init.scale_dxdy:
        # scale grid spacing from center outward.
        # this different than original method in it
        # needs to define grid spacing before lat/lon.
        # original rectgrid defines latlon first
        rectgrid_scale_dxdy()
    else:
        # rectgrid no grid spacing.
        # original method with addition to use namelist lat/lon reference
        length = dbl_kind(ice_init.dxrect) * cm_to_m / radius * rad_to_deg
        ulon = np.empty((nx, ny), dtype=dbl_kind)
        ulon[0, :] = dbl_kind(ice_init.lonrefrect)  # reference lon from namelist
        for i in range(1, nx):
            ulon[i, :] = ulon[i - 1, :] + length  # ULON
        ulon /= rad_to_deg
        _place(ULON, ulon, field_loc_NEcorner, field_type_scalar)

        length = dbl_kind(ice_init.dyrect) * cm_to_m / radius * rad_to_deg
        ulat = np.empty((nx, ny), dtype=dbl_kind)
        ulat[:, 0] = dbl_kind(ice_init.latrefrect)  # reference latitude from namelist
        for j in range(1, ny):
            ulat[:, j] = ulat[:, j - 1] + length  # ULAT
        ulat /= rad_to_deg
        _place(ULAT, ulat, field_loc_NEcorner, field_type_scalar)
        primary_grid_lengths_HTN(
            np.full((nx, ny), dbl_kind(ice_init.dxrect), dtype=dbl_kind)  # HTN
        )  # dxU, dxT, dxN, dxE
        primary_grid_lengths_HTE(
            np.full((nx, ny), dbl_kind(ice_init.dyrect), dtype=dbl_kind)  # HTE
        )  # dyU, dyT, dyN, dyE

    #-----------------------------------------------------------------
    # Construct T-cell land mask
    # Keyed on ew_boundary_type; ns_boundary_type should be 'open'.
    #-----------------------------------------------------------------
    work = np.zeros((nx, ny), dtype=dbl_kind)
    kind = str(ice_init.kmt_type).strip()
    work[...] = c0  # initialize hm as land
    if kind == "boxislands":
        grid_boxislands_kmt(work)
    elif kind == "none":
        work[...] = c1  # initialize hm as ocean
    elif kind == "channel":
        # closed top and bottom
        # open sides
        # NOTE nx_global > 5
        work[:, 2:ny - 2] = c1
    elif kind == "channel_oneeast":
        # one channel wide
        # open sides
        # NOTE nx_global > 5
        work[:, ny // 2 - 1] = c1
    elif kind == "channel_onenorth":
        # open sides
        # one channel wide
        # NOTE nx_global > 5
        work[nx // 2 - 1, :] = c1
    elif kind == "wall":
        # open except
        # closed east edge
        work[:nx - 2, :] = c1
    elif kind == "default":
        # land in the upper left and lower right corners,
        # otherwise open boundaries
        imid = int(np.rint(np.trunc(dbl_kind(nx) / c2)))
        jmid = int(np.rint(np.trunc(dbl_kind(ny) / c2)))
        work[2:nx - 2, 2:ny - 2] = c1  # open central domain
        if nx > 5 and ny > 5:  # > 5x5 grid
            work[:imid + 2, :jmid + 2] = c1  # open lower left corner
            work[max(imid - 2, 1) - 1:nx, max(jmid - 2, 1) - 1:ny] = c1  # open upper right corner
    else:
        raise RuntimeError(f"(rectgrid) unknown kmt_type {kind}")
    if ice_boundary.ew_boundary_type == "closed":
        work[:2, :] = c0
        work[nx - 2:nx, :] = c0
    if ice_boundary.ns_boundary_type == "closed":
        work[:, :2] = c0
        work[:, ny - 2:ny] = c0
    scatter_global(hm, work, field_loc_center, field_type_scalar)


def mom_dx(work):
    """mom supergrid has four cells for every model cell, sum the sidelengths to get model dx"""
    nx, ny = _nx_ny()
    # i & j for mom supergrid
    dxt = np.zeros((nx, ny), dtype=dbl_kind)
    dxn = np.zeros((nx, ny), dtype=dbl_kind)
    dxe = np.zeros((nx, ny), dtype=dbl_kind)
    dxu = np.zeros((nx, ny), dtype=dbl_kind)
    # left ; center column of first t-cell
    # left column of second T-cell, (ie right column of U-cell)
    # middle , top of first row
    for i in range(nx - 1):
        dxt[i, :] = work[2 * i, 1:2 * ny:2] + work[2 * i + 1, 1:2 * ny:2]  # dxT
        dxn[i, :] = work[2 * i, 2:2 * ny + 1:2] + work[2 * i + 1, 2:2 * ny + 1:2]  # dxN
        dxe[i, :] = work[2 * i + 1, 1:2 * ny:2] + work[2 * i + 2, 1:2 * ny:2]  # dxE
        dxu[i, :] = work[2 * i + 1, 2:2 * ny + 1:2] + work[2 * i + 2, 2:2 * ny + 1:2]  # dxU
    # fill the last col
    # middle , top of first row
    dxt[nx - 1, :] = work[2 * nx - 2, 1:2 * ny:2] + work[2 * nx - 1, 1:2 * ny:2]  # dxT
    dxn[nx - 1, :] = work[2 * nx - 2, 2:2 * ny + 1:2] + work[2 * nx - 1, 2:2 * ny + 1:2]  # dxN
    if ice_boundary.ew_boundary_type == "cyclic":
        dxe[nx - 1, :] = work[2 * nx - 1, 1:2 * ny:2] + work[0, 1:2 * ny:2]  # dxE
        dxu[nx - 1, :] = work[2 * nx - 1, 2:2 * ny + 1:2] + work[0, 2:2 * ny + 1:2]  # dxU
    elif ice_boundary.ew_boundary_type == "open":
        dxe[nx - 1, :] = 4 * work[2 * nx - 1, 1:2 * ny:2] - 2 * work[2 * nx - 2, 1:2 * ny:2]  # dxE
        dxu[nx - 1, :] = 4 * work[2 * nx - 1, 2:2 * ny + 1:2] - 2 * work[2 * nx - 2, 2:2 * ny + 1:2]  # dxU
    _place(dxT, dxt, field_loc_center, field_type_scalar)
    _place(HTN, dxn, field_loc_Nface, field_type_scalar)
    if ice_init.save_ghte_ghtn:
        G_HTN[...] = HTN
    dxN[...] = HTN
    _place(dxE, dxe, field_loc_center, field_type_scalar)
    _place(dxU, dxu, field_loc_NEcorner, field_type_scalar)


def mom_dy(work):
    """mom supergrid has four cells for every model cell, sum the sidelengths to get model dy"""
    nx, ny = _nx_ny()
    # i & j for mom supergrid
    dyt = np.zeros((nx, ny), dtype=dbl_kind)
    dyn = np.zeros((nx, ny), dtype=dbl_kind)
    dye = np.zeros((nx, ny), dtype=dbl_kind)
    dyu = np.zeros((nx, ny), dtype=dbl_kind)
    # middle , right edge of first T-cell
    for i in range(nx):
        dyt[i, :ny - 1] = work[2 * i + 1, 0:2 * (ny - 1):2] + work[2 * i + 1, 1:2 * (ny - 1):2]  # dyT
        dyn[i, :ny - 1] = work[2 * i + 1, 1:2 * (ny - 1):2] + work[2 * i + 1, 2:2 * ny - 1:2]  # dyN
        dye[i, :ny - 1] = work[2 * i + 2, 0:2 * (ny - 1):2] + work[2 * i + 2, 1:2 * (ny - 1):2]  # dyE
        dyu[i, :ny - 1] = work[2 * i + 2, 1:2 * (ny - 1):2] + work[2 * i + 2, 2:2 * ny - 1:2]  # dyU
        # fill the top row
        # middle , right edge of first column
        dyt[i, ny - 1] = work[2 * i + 1, 2 * ny - 2] + work[2 * i + 1, 2 * ny - 1]  # dyT
        dye[i, ny - 1] = work[2 * i + 2, 2 * ny - 2] + work[2 * i + 2, 2 * ny - 1]  # dyE
    if ice_boundary.ns_boundary_type == "tripole":
        for i in range(nx):
            im1 = 2 * i + 1
            im2 = 2 * i + 2
            dyn[i, ny - 1] = work[im1, 2 * ny - 1] + work[2 * nx - im1, 2 * ny - 1]  # dyN
            dyu[i, ny - 1] = work[im2, 2 * ny - 1] + work[2 * nx - im2, 2 * ny - 1]  # dyU
    elif ice_boundary.ns_boundary_type == "cyclic":
        for i in range(nx):
            dyn[i, ny - 1] = work[2 * i + 1, 2 * ny - 1] + work[2 * i + 1, 0]  # dyN
            dyu[i, ny - 1] = work[2 * i + 2, 2 * ny - 1] + work[2 * i + 2, 0]  # dyU
    elif ice_boundary.ns_boundary_type == "open":
        for i in range(nx):
            dyn[i, ny - 1] = 4 * work[2 * i + 1, 2 * ny - 1] - 2 * work[2 * i + 1, 2 * ny - 2]  # dyN
            dyu[i, ny - 1] = 4 * work[2 * i + 2, 2 * ny - 1] - 2 * work[2 * i + 2, 2 * ny - 2]  # dyU
    _place(dyT, dyt, field_loc_center, field_type_scalar)
    _place(dyN, dyn, field_loc_Nface, field_type_scalar)
    _place(HTE, dye, field_loc_center, field_type_scalar)
    if ice_init.save_ghte_ghtn:
        G_HTE[...] = HTE
    dyE[...] = HTE
    _place(dyU, dyu, field_loc_NEcorner, field_type_scalar)


def _mom_super_sum(work, i0, j0):
    """Sum the four supergrid cells whose lower-left is (i0, j0), 0-based."""
    return work[i0, j0] + work[i0, j0 + 1] + work[i0 + 1, j0] + work[i0 + 1, j0 + 1]


def mom_area(work):
    """mom supergrid has four cells for every model cell, sum these
    to get uarea and tarea
    earea and narea are calculated from dx & dy - see https://github.com/NOAA-GFDL/MOM6/issues/740
    """
    nx, ny = _nx_ny()
    # calculate narea and earea
    narea[...] = dxN * dyN
    earea[...] = dxE * dyE
    tarea_g = np.zeros((nx, ny), dtype=dbl_kind)
    uarea_g = np.zeros((nx, ny), dtype=dbl_kind)
    # load tarea and uarea
    # left/right -half of first column
    # right of first U - cell
    # bottom/top -half of first row
    # top of first U - cell
    # i & j for mom supergrid
    for i in range(nx - 1):
        for j in range(ny - 1):
            tarea_g[i, j] = _mom_super_sum(work, 2 * i, 2 * j)
            uarea_g[i, j] = _mom_super_sum(work, 2 * i + 1, 2 * j + 1)
    # fill last column
    for j in range(ny - 1):
        tarea_g[nx - 1, j] = _mom_super_sum(work, 2 * nx - 2, 2 * j)
        if ice_boundary.ew_boundary_type == "cyclic":
            uarea_g[nx - 1, j] = (
                work[2 * nx - 1, 2 * j + 1] + work[2 * nx - 1, 2 * j + 2]
                + work[0, 2 * j + 1] + work[0, 2 * j + 2]
            )
        elif ice_boundary.ew_boundary_type == "open":
            uarea_g[nx - 1, j] = (
                4 * work[2 * nx - 1, 2 * j + 1] + 4 * work[2 * nx - 1, 2 * j + 2]
                - 2 * work[2 * nx - 2, 2 * j + 1] - 2 * work[2 * nx - 2, 2 * j + 2]
            )
    # fill last row
    for i in range(nx - 1):
        tarea_g[i, ny - 1] = _mom_super_sum(work, 2 * i, 2 * ny - 2)
        im2 = 2 * i + 1
        im3 = 2 * i + 2
        if ice_boundary.ns_boundary_type == "tripole":
            uarea_g[i, ny - 1] = (
                work[im2, 2 * ny - 1] + work[2 * nx - 1 - im2, 2 * ny - 1]
                + work[im3, 2 * ny - 1] + work[2 * nx - 1 - im3, 2 * ny - 1]
            )
        elif ice_boundary.ns_boundary_type == "cyclic":
            # Partner row is the first supergrid row, as in mom_dy.
            uarea_g[i, ny - 1] = (
                work[im2, 2 * ny - 1] + work[im2, 0] + work[im3, 2 * ny - 1] + work[im3, 0]
            )
        elif ice_boundary.ns_boundary_type == "open":
            uarea_g[i, ny - 1] = (
                4 * work[im2, 2 * ny - 1] + 4 * work[im3, 2 * ny - 1]
                - 2 * work[im2, 2 * ny - 2] - 2 * work[im3, 2 * ny - 2]
            )
    # the top right corner
    tarea_g[nx - 1, ny - 1] = _mom_super_sum(work, 2 * nx - 2, 2 * ny - 2)
    im2 = 2 * nx - 1
    jm2 = 2 * ny - 1
    jm1 = 2 * ny - 2
    ew = ice_boundary.ew_boundary_type
    ns = ice_boundary.ns_boundary_type
    if ns == "tripole":
        uarea_g[nx - 1, ny - 1] = c2 * (work[im2, jm2] + work[0, jm2])
    elif ns == "cyclic" and ew == "cyclic":
        uarea_g[nx - 1, ny - 1] = work[im2, jm2] + work[0, jm2] + work[im2, 0] + work[0, 0]
    elif ns == "cyclic" and ew == "open":
        uarea_g[nx - 1, ny - 1] = (
            4 * work[im2, jm2] + 4 * work[im2, 0] - 2 * work[2 * nx - 2, jm2] - 2 * work[2 * nx - 2, 0]
        )
    elif ns == "open" and ew == "cyclic":
        uarea_g[nx - 1, ny - 1] = (
            4 * work[im2, jm2] + 4 * work[0, jm2] - 2 * work[im2, jm1] - 2 * work[0, jm1]
        )
    elif ns == "open" and ew == "open":
        uarea_g[nx - 1, ny - 1] = (
            8 * work[im2, jm2] - 2 * work[im2, jm1] - 2 * work[2 * nx - 2, jm2]
        )
    _place(tarea, tarea_g, field_loc_center, field_type_scalar)
    _place(uarea, uarea_g, field_loc_NEcorner, field_type_scalar)


def mom_grid():
    """Create the CICE grid from the MOM supergrid netcdf file.
    CICE fields and units are:
    ULAT, ULON, TLAT, TLON, ELAT, ELON, NLAT, NLON (radians)
    HTN, HTE   (m)
    dxT, dyT, dxU, dyU, dxN, dyN, dxE, dyE,   (m)
    ANGLE, ANGLET (radians)
    tarea, uarea, narea, earea (m^2)
    lont_bounds, latt_bounds, etc (degrees)

    Latitudes in the file are degrees. History gridbox corners are not written.
    """
    global l_readCenter
    path = ice_init.grid_file
    nx, ny = _nx_ny()
    #-----------------------------------------------------------------
    # lat, lon, angle
    #-----------------------------------------------------------------
    # populate all LAT fields
    # field name in netCDF file
    super_yx = ice_read_write.ice_read_global_nc(path, "y", shape=(2 * nx + 1, 2 * ny + 1))
    # mom supergrid has four cells for every model cell
    # we need to select the correct edges to get lat & lon for a model cell
    # we include left/bottom edges for U-points, and top/right edges for T-points
    # and close per ew_boundary_type & ns_boundary_type
    # i & j for mom supergrid
    gu = np.zeros((nx + 1, ny + 1), dtype=dbl_kind)
    gt = np.zeros((nx + 1, ny + 1), dtype=dbl_kind)
    ge = np.zeros((nx + 1, ny + 1), dtype=dbl_kind)
    gn = np.zeros((nx + 1, ny + 1), dtype=dbl_kind)
    # lh , middle  hand edge of first col
    # bottom, middle of first row
    gu[:nx, :ny] = super_yx[0:2 * nx:2, 0:2 * ny:2]  # ULAT/LON
    gn[:nx, :ny] = super_yx[1:2 * nx:2, 0:2 * ny:2]  # NLAT/LON
    ge[:nx, :ny] = super_yx[0:2 * nx:2, 1:2 * ny:2]  # ELAT/LON
    gt[:nx, :ny] = super_yx[1:2 * nx:2, 1:2 * ny:2]  # TLAT/LON
    # fill last col
    # bottom, middle of first row
    gu[nx, :ny] = super_yx[2 * nx, 0:2 * ny:2]
    ge[nx, :ny] = super_yx[2 * nx, 1:2 * ny:2]
    if ice_boundary.ew_boundary_type == "cyclic":
        gt[nx, :] = gt[0, :]
        gn[nx, :] = gn[0, :]
    elif ice_boundary.ew_boundary_type == "open":
        gt[nx, :] = c2 * gt[nx - 1, :] - gt[nx - 2, :]
        gn[nx, :] = c2 * gn[nx - 1, :] - gn[nx - 2, :]
    # fill last row
    gu[:nx + 1, ny] = super_yx[0:2 * nx + 1:2, 2 * ny]
    # im2 runs 2, 4, ... and the last sample, 2*nx+2, is outside the supergrid.
    # The extra column follows the same east-west rule as the rows above.
    gn[:nx, ny] = super_yx[1:2 * nx:2, 2 * ny]
    if ice_boundary.ew_boundary_type == "cyclic":
        gn[nx, ny] = gn[0, ny]
    elif ice_boundary.ew_boundary_type == "open":
        gn[nx, ny] = c2 * gn[nx - 1, ny] - gn[nx - 2, ny]
    if ice_boundary.ns_boundary_type == "tripole":
        # i and (nx+1-i) on the last T/E row. i=nx+1 asks for Fortran index 0;
        # that corner is the i=1 column.
        src = nx + 1 - np.arange(1, nx + 2)
        src = np.where(src < 1, 1, src)
        gt[:, ny] = gt[src - 1, ny - 1]
        ge[:, ny] = ge[src - 1, ny - 1]
    elif ice_boundary.ns_boundary_type == "cyclic":
        gt[:, ny] = gt[:, 0]
        ge[:, ny] = ge[:, 0]
    elif ice_boundary.ns_boundary_type == "open":
        gt[:, ny] = c2 * gt[:, ny - 1] - gt[:, ny - 2]
        ge[:, ny] = c2 * ge[:, ny - 1] - ge[:, ny - 2]
    # distribute global array to local
    # with a global array of corner points in degrees, convert to rad and scatter to workers
    pi = icepack_query_parameters("pi")["pi"]
    scale = pi / c180
    # convert to rad
    gu *= scale
    gt *= scale
    ge *= scale
    gn *= scale
    # distribute to processors
    # subset G_T to active cells by dropping right/top halo
    _place(TLAT, gt[:nx, :ny], field_loc_center, field_type_scalar)
    # subset G_U/G_E/G_N to active cells by dropping left/bottom edge
    _place(ULAT, gu[1:nx + 1, 1:ny + 1], field_loc_NEcorner, field_type_scalar)
    _place(NLAT, gn[:nx, 1:ny + 1], field_loc_Nface, field_type_scalar)
    _place(ELAT, ge[1:nx + 1, :ny], field_loc_Eface, field_type_scalar)
    ulat_c = gu
    tlat_c = gt
    # populate all LON fields
    super_yx = ice_read_write.ice_read_global_nc(path, "x", shape=(2 * nx + 1, 2 * ny + 1))
    # mom supergrid has four cells for every model cell
    # we need to select the correct edges to get lat & lon for a model cell
    # we include left/bottom edges for U-points, and top/right edges for T-points
    # and close per ew_boundary_type & ns_boundary_type
    # i & j for mom supergrid
    gu = np.zeros((nx + 1, ny + 1), dtype=dbl_kind)
    gt = np.zeros((nx + 1, ny + 1), dtype=dbl_kind)
    ge = np.zeros((nx + 1, ny + 1), dtype=dbl_kind)
    gn = np.zeros((nx + 1, ny + 1), dtype=dbl_kind)
    # lh , middle  hand edge of first col
    # bottom, middle of first row
    gu[:nx, :ny] = super_yx[0:2 * nx:2, 0:2 * ny:2]  # ULAT/LON
    gn[:nx, :ny] = super_yx[1:2 * nx:2, 0:2 * ny:2]  # NLAT/LON
    ge[:nx, :ny] = super_yx[0:2 * nx:2, 1:2 * ny:2]  # ELAT/LON
    gt[:nx, :ny] = super_yx[1:2 * nx:2, 1:2 * ny:2]  # TLAT/LON
    # fill last col
    # bottom, middle of first row
    gu[nx, :ny] = super_yx[2 * nx, 0:2 * ny:2]
    ge[nx, :ny] = super_yx[2 * nx, 1:2 * ny:2]
    if ice_boundary.ew_boundary_type == "cyclic":
        gt[nx, :] = gt[0, :]
        gn[nx, :] = gn[0, :]
    elif ice_boundary.ew_boundary_type == "open":
        gt[nx, :] = c2 * gt[nx - 1, :] - gt[nx - 2, :]
        gn[nx, :] = c2 * gn[nx - 1, :] - gn[nx - 2, :]
    # fill last row
    gu[:nx + 1, ny] = super_yx[0:2 * nx + 1:2, 2 * ny]
    # im2 runs 2, 4, ... and the last sample, 2*nx+2, is outside the supergrid.
    # The extra column follows the same east-west rule as the rows above.
    gn[:nx, ny] = super_yx[1:2 * nx:2, 2 * ny]
    if ice_boundary.ew_boundary_type == "cyclic":
        gn[nx, ny] = gn[0, ny]
    elif ice_boundary.ew_boundary_type == "open":
        gn[nx, ny] = c2 * gn[nx - 1, ny] - gn[nx - 2, ny]
    if ice_boundary.ns_boundary_type == "tripole":
        # i and (nx+1-i) on the last T/E row. i=nx+1 asks for Fortran index 0;
        # that corner is the i=1 column.
        src = nx + 1 - np.arange(1, nx + 2)
        src = np.where(src < 1, 1, src)
        gt[:, ny] = gt[src - 1, ny - 1]
        ge[:, ny] = ge[src - 1, ny - 1]
    elif ice_boundary.ns_boundary_type == "cyclic":
        gt[:, ny] = gt[:, 0]
        ge[:, ny] = ge[:, 0]
    elif ice_boundary.ns_boundary_type == "open":
        gt[:, ny] = c2 * gt[:, ny - 1] - gt[:, ny - 2]
        ge[:, ny] = c2 * ge[:, ny - 1] - ge[:, ny - 2]
    # with a global array of corner points in degrees, convert to rad and scatter to workers
    pi = icepack_query_parameters("pi")["pi"]
    scale = pi / c180
    # convert to rad
    gu *= scale
    gt *= scale
    ge *= scale
    gn *= scale
    # distribute to processors
    # subset G_T to active cells by dropping right/top halo
    _place(TLON, gt[:nx, :ny], field_loc_center, field_type_scalar)
    # subset G_U/G_E/G_N to active cells by dropping left/bottom edge
    _place(ULON, gu[1:nx + 1, 1:ny + 1], field_loc_NEcorner, field_type_scalar)
    _place(NLON, gn[:nx, 1:ny + 1], field_loc_Nface, field_type_scalar)
    _place(ELON, ge[1:nx + 1, :ny], field_loc_Eface, field_type_scalar)
    # array for angle field
    # populate angle fields, angle is u-points, angleT is t-points
    # even though mom supergrid files contain angle_dx, mom6 calculates internally
    # create angles in the same way mom6 creates the angle
    # based on https://github.com/mom-ocean/MOM6/blob/129e1bda02d454fb280819d1d87ae16347fd044c/src/initialization/MOM_shared_initialization.F90#L535
    # the angle is between logical north on the grid and true north.
    lon_cnr = gu
    lat_cnr = ulat_c
    lon_cen = gt[:nx, :ny]
    anglet = np.zeros(lon_cen.shape, dtype=dbl_kind)  # anglet
    # The periodic range of longitudes, usually 2pi.
    len_lon = np.max(lon_cnr) - np.min(lon_cnr)
    for j in range(lon_cen.shape[1]):
        for i in range(lon_cen.shape[0]):
            lon_adj = lon_cen[i, j] - p5 * len_lon
            # shift 4 lon corner points to be similar range to centre point
            # e.g. upper limit of 0 might be shifted to 2*pi
            lonb = np.mod(lon_cnr[i:i + 2, j:j + 2] - lon_adj, len_lon) + lon_adj
            # The trigonometric scaling factor converting changes in longitude to equivalent distances in latitudes [nondim]
            lon_scale = np.cos(
                p25 * (lat_cnr[i, j] + lat_cnr[i + 1, j + 1] + lat_cnr[i + 1, j] + lat_cnr[i, j + 1])
            )
            anglet[i, j] = np.arctan2(
                lon_scale * ((lonb[0, 1] - lonb[1, 0] + lonb[1, 1] - lonb[0, 0])),
                lat_cnr[i, j + 1] - lat_cnr[i + 1, j] + lat_cnr[i + 1, j + 1] - lat_cnr[i, j],
            )
    _place(ANGLET, anglet, field_loc_center, field_type_angle)
    lon_cnr = gt
    lat_cnr = tlat_c
    lon_cen = gu[1:nx + 1, 1:ny + 1]
    angle = np.zeros(lon_cen.shape, dtype=dbl_kind)  # angle
    # The periodic range of longitudes, usually 2pi.
    len_lon = np.max(lon_cnr) - np.min(lon_cnr)
    for j in range(lon_cen.shape[1]):
        for i in range(lon_cen.shape[0]):
            lon_adj = lon_cen[i, j] - p5 * len_lon
            # shift 4 lon corner points to be similar range to centre point
            # e.g. upper limit of 0 might be shifted to 2*pi
            lonb = np.mod(lon_cnr[i:i + 2, j:j + 2] - lon_adj, len_lon) + lon_adj
            # The trigonometric scaling factor converting changes in longitude to equivalent distances in latitudes [nondim]
            lon_scale = np.cos(
                p25 * (lat_cnr[i, j] + lat_cnr[i + 1, j + 1] + lat_cnr[i + 1, j] + lat_cnr[i, j + 1])
            )
            angle[i, j] = np.arctan2(
                lon_scale * ((lonb[0, 1] - lonb[1, 0] + lonb[1, 1] - lonb[0, 0])),
                lat_cnr[i, j + 1] - lat_cnr[i + 1, j] + lat_cnr[i + 1, j + 1] - lat_cnr[i, j],
            )
    _place(ANGLE, angle, field_loc_NEcorner, field_type_angle)
    #-----------------------------------------------------------------
    # cell dimensions
    #-----------------------------------------------------------------
    # dx uses the cells in x, edges in y, reallocate work_mom to this size
    mom_dx(ice_read_write.ice_read_global_nc(path, "dx", shape=(2 * nx, 2 * ny + 1)))
    # dy uses the edges in x, cells in y, reallocate work_mom to this size
    mom_dy(ice_read_write.ice_read_global_nc(path, "dy", shape=(2 * nx + 1, 2 * ny)))
    #-----------------------------------------------------------------
    # cell areas
    #-----------------------------------------------------------------
    mom_area(ice_read_write.ice_read_global_nc(path, "area", shape=(2 * nx, 2 * ny)))
    #-----------------------------------------------------------------
    # fin
    #-----------------------------------------------------------------
    l_readCenter = True  # we have read t quantities


def init_grid2():
    """Horizontal grid initialization:

        U{LAT,LONG} = true {latitude,longitude} of U points
        HT{N,E} = cell widths on {N,E} sides of T cell
        ANGLE = angle between local x direction and true east
        hm = land mask (c1 for ocean points, c0 for land points)
        D{X,Y}{T,U} = {x,y} spacing centered at {T,U} points
        T-grid and ghost cell values
        Various grid quantities needed for dynamics and transport

    author: Elizabeth C. Hunke, LANL

    Latitudes, cell widths, land mask, and the Coriolis parameter.
    Coriolis is computed here.
    """
    global l_readCenter
    l_readCenter = False
    pi = icepack_query_parameters("pi")["pi"]
    grid_type = str(ice_init.grid_type).strip()
    grid_format = str(ice_init.grid_format).strip()
    #-----------------------------------------------------------------
    # lat, lon, cell widths, angle, land mask
    #-----------------------------------------------------------------
    if grid_type in ("displaced_pole", "tripole", "regional"):
        if grid_format == "mom_nc":
            mom_grid()  # derive cice grid from MOM supergrid nc file
        elif grid_format == "pop_nc":
            popgrid_nc()  # read POP grid lengths from nc file
        elif grid_format == "pop_nc_ext":
            popgrid_nc_ext()  # read POP extended grid lengths from nc file
        elif grid_format == "geosnc":
            geosgrid_nc()  # read GEOS MOM grid used from nc file
        else:
            popgrid()  # read POP grid lengths directly
    else:
        # lat lon grid for sequential CESM (CAM mode)
        # latlon is latlongrid only in a CESMCOUPLED build. Standalone
        # CICE falls through to the rectangular grid.
        rectgrid()  # regular rectangular grid

    kmt_type = str(ice_init.kmt_type).strip()
    if kmt_type == "none":
        kmt[...] = c1
        hm[...] = c1
    elif kmt_type == "file":
        if grid_format in ("mom_nc", "pop_nc", "geosnc"):
            kmtmask("nc")
        elif grid_format == "pop_nc_ext":
            kmtmask("nc_ext")
        else:
            kmtmask("bin")
    # the other types are handled by rectgrid

    #-----------------------------------------------------------------
    # T-grid cell and U-grid cell quantities
    # Fill halo data locally where possible to avoid missing
    # data associated with land block elimination
    # Note: HTN, HTE, dx*, dy* are all defined from global arrays
    # at halos.
    #-----------------------------------------------------------------
    if grid_format != "mom_nc":
        tarea[...] = dxT * dyT
        uarea[...] = dxU * dyU
        narea[...] = dxN * dyN
        earea[...] = dxE * dyE
    tarear[...] = _reciprocal(tarea)
    uarear[...] = _reciprocal(uarea)
    narear[...] = _reciprocal(narea)
    earear[...] = _reciprocal(earea)
    #-----------------------------------------------------------------
    # Ghost cell updates
    # On the tripole grid, one must be careful with updates of
    #  quantities that involve a difference of cell lengths.
    # For example, dyhx and dxhy are cell-centered vector components.
    # Also note that on the tripole grid, cxp and cxm would swap places,
    #  as would cyp and cym.  These quantities are computed only
    #  in north and east ghost cells (above), not south and west.
    #-----------------------------------------------------------------
    # Update just on the tripole seam to ensure bit-for-bit symmetry across seam
    for field, loc in (
        (tarea, field_loc_center),
        (uarea, field_loc_NEcorner),
        (narea, field_loc_Nface),
        (earea, field_loc_Eface),
        (tarear, field_loc_center),
        (uarear, field_loc_NEcorner),
        (narear, field_loc_Nface),
        (earear, field_loc_Eface),
    ):
        ice_halo_update(field, loc, field_type_scalar, fill_value=c1, tripole_only=True)

    #-----------------------------------------------------------------
    # Ensure that -pi <= ANGLE <= pi
    #-----------------------------------------------------------------
    if np.any(ANGLE < -pi) or np.any(ANGLE > pi):
        raise RuntimeError(
            f"(init_grid2) ANGLE out of expected range: "
            f"{np.min(ANGLE)} {np.max(ANGLE)}"
        )
    if l_readCenter:
        #-----------------------------------------------------------------
        # Ensure that -pi <= ANGLE <= pi
        #-----------------------------------------------------------------
        if np.any(ANGLET < -pi) or np.any(ANGLET > pi):
            raise RuntimeError(
                f"(init_grid2) ANGLET out of expected range: "
                f"{np.min(ANGLET)} {np.max(ANGLET)}"
            )
    else:
        #-----------------------------------------------------------------
        # Compute ANGLE on T-grid
        #-----------------------------------------------------------------
        ANGLET[...] = c0
        # beginning and end of physical domain
        ilo, ihi, jlo, jhi = physical_bounds()
        samples = (
            _view(ANGLE, 0, 0),   # w----0
            _view(ANGLE, -1, 0),  # |    |
            _view(ANGLE, 0, -1),  # |    |
            _view(ANGLE, -1, -1),  # sw---s
        )
        ANGLET[ilo:ihi + 1, jlo:jhi + 1] = np.arctan2(
            p25 * sum(np.sin(sample) for sample in samples),
            p25 * sum(np.cos(sample) for sample in samples),
        )
        # for W boundary extrapolate from interior
        if str(ice_init.grid_type).strip() in ("regional", "rectangular"):
            ig, _jg = global_indices()
            if int(ig[ilo]) == 1:
                rows = slice(jlo, jhi + 1)
                ANGLET[ilo, rows] = c2 * ANGLET[ilo + 1, rows] + (-c1) * ANGLET[ilo + 2, rows]
    ice_halo_update(ANGLET, field_loc_center, field_type_angle, fill_value=c1)

    makemask()  # velocity mask, hemisphere masks
    #----------------------------------------------------------------
    # Coordinates for all T/N/E cells
    #----------------------------------------------------------------
    if grid_format != "mom_nc":
        if not l_readCenter:
            Tlatlon()  # get lat, lon on the T grid
        NElatlon()  # get lat, lon on the N, E grid
        # corners for CF-compliant output
        # History gridbox corners are not written.

    #-----------------------------------------------------------------
    # bathymetry
    #-----------------------------------------------------------------
    bathy = str(ice_init.bathymetry_format).strip()
    if bathy == "default":
        get_bathymetry()
    elif bathy == "pop":
        get_bathymetry_popfile()
    else:
        raise RuntimeError(
            "(init_grid2) bathymetry_format value must be default or pop"
        )

    #-----------------------------------------------------------------
    # Compute global index (used for unpacking messages from coupler)
    #-----------------------------------------------------------------
    nx, ny = _nx_ny()
    index = np.empty((nx, ny), dtype=dbl_kind)
    index[...] = (
        np.arange(ny, dtype=dbl_kind)[None, :] * nx
        + np.arange(1, nx + 1, dtype=dbl_kind)[:, None]
    )
    scatter_global(rndex_global, index, field_loc_center, field_type_scalar)
    # Coriolis parameter. Computed here because this step owns the latitude arrays.
    set_coriolis()


def init_domain_grid_state():
    """Allocate the grid and state, then build the grid.

    Call ``init_constants_tracers_namelist`` first. Replaces
    the ``alloc_*`` calls and ``init_grid2``. ``init_grid1`` is the
    tripole checks only.
    ``init_domain_blocks`` is not ported; it reads ``domain_nml`` and
    sets the domain size and boundary types before this routine.

    Distribute blocks across processors.  The distribution is optimized
    based on latitude and topography, contained in the ULAT and KMT arrays.

    authors: William Lipscomb and Phil Jones, LANL
    """
    if int(ice_init.ncat) < 1 or int(icepack_tracers.ntrcr) < 1:
        raise ValueError("(init_domain_grid_state) run the namelist step first")
    if int(ice_domain_size.nx_global) < 1 or int(ice_domain_size.ny_global) < 1:
        raise ValueError("(init_domain_grid_state) domain size is set by init_domain_blocks")
    sync_vertical_sizes(
        ice_init.ncat,
        ice_init.nilyr,
        ice_init.nslyr,
        ice_init.nblyr,
        ice_init.nfsd,
        icepack_tracers.ntrcr,
    )
    out = ice_fileunits.nu_diag
    out.write("\nDomain Information\n\n")
    out.write(f"  Horizontal domain: nx = {int(ice_domain_size.nx_global):6d}\n")
    out.write(f"                     ny = {int(ice_domain_size.ny_global):6d}\n")
    out.write(f"  No. of categories: nc = {int(ice_domain_size.ncat):6d}\n")
    out.write(f"  No. of ice layers: ni = {int(ice_domain_size.nilyr):6d}\n")
    out.write(f"  No. of snow layers:ns = {int(ice_domain_size.nslyr):6d}\n")
    out.write(f"  ew_boundary_type      = {ice_boundary.ew_boundary_type}\n")
    out.write(f"  ns_boundary_type      = {ice_boundary.ns_boundary_type}\n")
    out.write(f"  Number of ghost cells = {int(nghost):6d}\n\n")
    out.flush()

    alloc_grid()
    ice_arrays_column.alloc_arrays_column()
    ice_state.alloc_state()
    ice_flux.alloc_flux()
    init_grid2()
