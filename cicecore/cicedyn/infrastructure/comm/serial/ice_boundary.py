"""This module contains data types and routines for updating halo
regions (ghost cells)

author for JAX version: Xinyuan Zhu
XZ: converted from F90 to Python
XZ: cyclic corner ghosts, tripole ranks above 2D, integer nint,
    stress seam without the degenerate-point average

2023-03-09: Tony Craig updated the implementation to fix bug in
            tripoleT and reduce number of copies in tripole overall.
            Because all blocks are local, can fill the tripole
            buffer from "north" copies.  This is not true for
            the MPI version.
2008-01-28: Elizabeth Hunke replaced old routines with new POP
            infrastructure
2007-07-19: Phil Jones, Yoshi Yoshida, John Dennis
            new naming conventions, optimizations during
            initialization, true multi-dimensional updates
            (rather than serial call to two-dimensional updates),
            fixes for non-existent blocks

One domain, no block decomposition. ``ew_boundary_type`` and
``ns_boundary_type`` come from ``domain_nml``. Cyclic edges copy the
opposite side. Open and closed edges are left alone unless a fill value
is set; ``ice_halo_extrapolate`` then fills a non-cyclic edge. A tripole
or tripoleT north edge is folded with the same offsets and sign as the
Fortran halo update.
"""

import numpy as np
import jax.numpy as jnp

from cicecore.shared.ice_constants import (
    c0,
    c2,
    field_loc_Eface,
    field_loc_NEcorner,
    field_loc_Nface,
    field_loc_center,
    field_loc_noupdate,
    field_loc_unknown,
    field_type_angle,
    field_type_noupdate,
    field_type_scalar,
    field_type_unknown,
    field_type_vector,
    p5,
)
from cicecore.shared import ice_domain_size
from cicecore.shared.ice_domain_size import nghost, physical_bounds
from cicecore.shared.ice_kinds_mod import dbl_kind

# type of boundary to use in logical ew dir
ew_boundary_type = "cyclic"
# type of boundary to use in logical ns dir
ns_boundary_type = "open"


def set_boundary_types(ew, ns):
    """Store the namelist boundary types."""
    # type of boundary to use in logical ns dir
    # type of boundary to use in logical ew dir
    global ew_boundary_type, ns_boundary_type
    ew_boundary_type = str(ew).strip()
    ns_boundary_type = str(ns).strip()


# integer mask saved by ice_HaloMask
halo_mask = None


def ice_HaloMask(mask):
    """Create a halo description from the base halo and a mask.

    Serial ice_boundary: halos are not masked for local copies.
    The mask is stored. A later ghost update uses the same local copy
    as the base halo.
    """
    global halo_mask
    halo_mask = np.array(mask, copy=True)


def global_indices():
    """1-based global i and j for every local point, including ghosts.

    Cyclic indices wrap. A tripole north ghost is stored as a negative
    j, the same convention as ``ice_blocks``. 0 means padding.

    global i index for each point in each block
    global j index for each point in each block
    global domain location for each point.
    valid values between 1:nx_global, 1:ny_global.
    outside that range may occur in the halo with
    open or closed bcs or on the tripole.
    by definition, tripole is only on the north
    boundary and in that case, the j_glob values
    will be valid j_glob values with minus sign.
    """
    ghost = int(nghost)
    nx = int(ice_domain_size.nx_global)
    ny = int(ice_domain_size.ny_global)
    nxb = nx + 2 * ghost
    nyb = ny + 2 * ghost
    ig = np.arange(1, nxb + 1) - ghost  # left to right counting
    jg = np.arange(1, nyb + 1) - ghost  # simple lower to upper counting
    #*** western ghost cells
    # case ('open')
    #    left to right
    # case ('closed')
    #    left to right
    if ew_boundary_type == "cyclic":
        ig = np.where(ig < 1, ig + nx, ig)
        #*** eastern ghost cells
        ig = np.where(ig > nx, ig - nx, ig)
    #*** eastern ghost cells
    # case ('open')
    #    left to right
    # case ('closed')
    #    left to right
    #*** southern ghost cells
    # case ('open')
    #    lower to upper
    # case ('closed')
    #    lower to upper
    # case ('tripole')
    #    lower to upper
    # case ('tripoleT')
    #    lower to upper
    if ns_boundary_type == "cyclic":
        jg = np.where(jg < 1, jg + ny, jg)
        #*** northern ghost cells
        jg = np.where(jg > ny, jg - ny, jg)
    #*** padded domain - fill padded region with zero
    ig = np.where(ig > nx + ghost, 0, ig)
    #*** padding required
    jg = np.where(jg > ny + ghost, 0, jg)  # padding
    #*** northern ghost cells
    # case ('open')
    #    lower to upper
    # case ('closed')
    #    lower to upper
    # case ('tripole')
    #    negative
    # case ('tripoleT')
    #    negative
    if ns_boundary_type in ("tripole", "tripoleT"):
        jg = np.where(jg > ny, -jg, jg)  # negative
    return ig.astype(np.int32), jg.astype(np.int32)


def ice_halo_update_center_jax(array, static, iglob):
    """JAX scalar T-cell halo update on the single full grid.

    This is the center/scalar case of ice_HaloUpdate, used by
    bound_state for aicen, trcrn, vicen and vsnon. Trailing category
    and tracer axes pass through the same edge operations.
    """
    ilo, ihi, jlo, jhi = static["ice_domain_size"]["physical_bounds"]
    ghost = int(static["ice_domain_size"]["nghost"])
    ew = static["ice_domain"]["ew_boundary_type"].strip()
    ns = static["ice_domain"]["ns_boundary_type"].strip()

    if ew == "cyclic":
        array = array.at[ilo - ghost:ilo, jlo:jhi + 1, ...].set(
            array[ihi - ghost + 1:ihi + 1, jlo:jhi + 1, ...]
        )
        array = array.at[ihi + 1:ihi + 1 + ghost, jlo:jhi + 1, ...].set(
            array[ilo:ilo + ghost, jlo:jhi + 1, ...]
        )
    if ns == "cyclic":
        array = array.at[ilo:ihi + 1, jlo - ghost:jlo, ...].set(
            array[ilo:ihi + 1, jhi - ghost + 1:jhi + 1, ...]
        )
        array = array.at[ilo:ihi + 1, jhi + 1:jhi + 1 + ghost, ...].set(
            array[ilo:ihi + 1, jlo:jlo + ghost, ...]
        )
    if ew == "cyclic" and ns == "cyclic":
        array = array.at[ilo - ghost:ilo, jlo - ghost:jlo, ...].set(
            array[ihi - ghost + 1:ihi + 1, jhi - ghost + 1:jhi + 1, ...]
        )
        array = array.at[ihi + 1:ihi + 1 + ghost, jlo - ghost:jlo, ...].set(
            array[ilo:ilo + ghost, jhi - ghost + 1:jhi + 1, ...]
        )
        array = array.at[ilo - ghost:ilo, jhi + 1:jhi + 1 + ghost, ...].set(
            array[ihi - ghost + 1:ihi + 1, jlo:jlo + ghost, ...]
        )
        array = array.at[ihi + 1:ihi + 1 + ghost, jhi + 1:jhi + 1 + ghost, ...].set(
            array[ilo:ilo + ghost, jlo:jlo + ghost, ...]
        )

    if ns in ("tripole", "tripoleT"):
        nx = int(static["ice_domain_size"]["nx_global"])
        rows = ghost + 1 + (ns == "tripoleT")
        buf = jnp.zeros((nx, rows) + array.shape[2:], dtype=array.dtype)
        buf = buf.at[iglob[ilo:ihi + 1] - 1, ...].set(
            array[ilo:ihi + 1, jhi - rows + 1:jhi + 1, ...]
        )
        if ns == "tripoleT":
            # The upper physical T row is degenerate: paired cells are
            # averaged before folding across the northern seam.
            left = jnp.arange(1, nx // 2)
            right = nx - left
            average = (buf[left, rows - 1, ...] + buf[right, rows - 1, ...]) / 2
            buf = buf.at[left, rows - 1, ...].set(average)
            buf = buf.at[right, rows - 1, ...].set(average)
            mirrored = (nx - iglob + 1) % nx
            array = array.at[:, jhi, ...].set(buf[mirrored, rows - 1, ...])
            array = array.at[:, jhi + 1, ...].set(buf[mirrored, rows - 2, ...])
        else:
            mirrored = (nx - iglob) % nx
            array = array.at[:, jhi + 1, ...].set(buf[mirrored, rows - 1, ...])
    return array


def ice_halo_update_jax(array, field_loc, field_type, static, iglob,
                        fill_value=None, tripole_only=False, source=None,
                        seam_average=True):
    """Single-grid JAX translation of ice_HaloUpdate for all grid locations.

    ``field_loc`` is T, U, E or N; ``field_type`` is scalar or vector.
    The tripole fold uses the same global-index offsets as the serial
    Fortran halo implementation. Trailing category axes are preserved.
    """
    ilo, ihi, jlo, jhi = static["ice_domain_size"]["physical_bounds"]
    ghost = int(static["ice_domain_size"]["nghost"])
    ew = static["ice_domain"]["ew_boundary_type"].strip()
    ns = static["ice_domain"]["ns_boundary_type"].strip()
    value = 0.0 if fill_value is None else fill_value
    if not tripole_only:
        ew_fill = ew == "cyclic" or fill_value is not None
        ns_fill = ns in ("cyclic", "tripole", "tripoleT") or fill_value is not None
        if ew_fill:
            array = array.at[ilo - ghost:ilo, jlo:jhi + 1, ...].set(value)
            array = array.at[ihi + 1:ihi + ghost + 1, jlo:jhi + 1, ...].set(value)
        if ns_fill:
            array = array.at[ilo:ihi + 1, jlo - ghost:jlo, ...].set(value)
            array = array.at[ilo:ihi + 1, jhi + 1:jhi + ghost + 1, ...].set(value)
        if ew_fill and ns_fill:
            for ii in (slice(ilo - ghost, ilo), slice(ihi + 1, ihi + ghost + 1)):
                for jj in (slice(jlo - ghost, jlo), slice(jhi + 1, jhi + ghost + 1)):
                    array = array.at[ii, jj, ...].set(value)
        if ew == "cyclic":
            array = array.at[ilo - ghost:ilo, jlo:jhi + 1, ...].set(
                array[ihi - ghost + 1:ihi + 1, jlo:jhi + 1, ...])
            array = array.at[ihi + 1:ihi + ghost + 1, jlo:jhi + 1, ...].set(
                array[ilo:ilo + ghost, jlo:jhi + 1, ...])
        if ns == "cyclic":
            array = array.at[ilo:ihi + 1, jlo - ghost:jlo, ...].set(
                array[ilo:ihi + 1, jhi - ghost + 1:jhi + 1, ...])
            array = array.at[ilo:ihi + 1, jhi + 1:jhi + ghost + 1, ...].set(
                array[ilo:ihi + 1, jlo:jlo + ghost, ...])
        if ew == "cyclic" and ns == "cyclic":
            array = array.at[ilo - ghost:ilo, jlo - ghost:jlo, ...].set(
                array[ihi - ghost + 1:ihi + 1, jhi - ghost + 1:jhi + 1, ...])
            array = array.at[ihi + 1:ihi + ghost + 1, jlo - ghost:jlo, ...].set(
                array[ilo:ilo + ghost, jhi - ghost + 1:jhi + 1, ...])
            array = array.at[ilo - ghost:ilo, jhi + 1:jhi + ghost + 1, ...].set(
                array[ihi - ghost + 1:ihi + 1, jlo:jlo + ghost, ...])
            array = array.at[ihi + 1:ihi + ghost + 1, jhi + 1:jhi + ghost + 1, ...].set(
                array[ilo:ilo + ghost, jlo:jlo + ghost, ...])
    if ns not in ("tripole", "tripoleT"):
        return array

    nx = int(static["ice_domain_size"]["nx_global"])
    rows = ghost + 1 + int(ns == "tripoleT")
    buf = jnp.zeros((nx, rows) + array.shape[2:], dtype=array.dtype)
    buf_src = array if source is None else source
    buf = buf.at[iglob[ilo:ihi + 1] - 1, ...].set(
        buf_src[ilo:ihi + 1, jhi - rows + 1:jhi + 1, ...])
    sign = -1 if field_type == "vector" else 1
    if ns == "tripoleT":
        if field_loc == "T":
            ioffset, joffset = -1, 0
            pairs = ((i, nx - i + 2) for i in range(2, nx // 2 + 1))
        elif field_loc == "U":
            ioffset, joffset, pairs = 0, 1, ()
        elif field_loc == "E":
            ioffset, joffset = 0, 0
            pairs = ((i, nx + 1 - i) for i in range(1, nx // 2 + 1))
        elif field_loc == "N":
            ioffset, joffset, pairs = -1, 1, ()
        else:
            raise ValueError("unknown field location")
    else:
        if field_loc == "T":
            ioffset, joffset, pairs = 0, 0, ()
        elif field_loc == "U":
            ioffset, joffset = 1, 1
            pairs = ((i, nx - i) for i in range(1, nx // 2))
        elif field_loc == "E":
            ioffset, joffset, pairs = 1, 0, ()
        elif field_loc == "N":
            ioffset, joffset = 0, 1
            pairs = ((i, nx + 1 - i) for i in range(1, nx // 2 + 1))
        else:
            raise ValueError("unknown field location")
    pairs = tuple(pairs)
    if seam_average and pairs:
        # The fold pairs are disjoint, so their averages can be updated together.
        indices = jnp.asarray(pairs, dtype=jnp.int32) - 1
        left, right = indices[:, 0], indices[:, 1]
        average = 0.5 * (buf[left, -1, ...] + sign * buf[right, -1, ...])
        buf = buf.at[left, -1, ...].set(average)
        buf = buf.at[right, -1, ...].set(sign * average)
    columns = jnp.arange(array.shape[0])
    folded = (nx - iglob[columns] - ioffset) % nx
    for j in range(1, ghost + 2):
        source_row = ghost + 3 - j - joffset
        # Fortran's first destination is the top physical row. Depending
        # on the grid location, its source row may be out of range and
        # the copy is then skipped.
        destination_row = jhi + j - 1
        if 0 < source_row <= rows and destination_row < array.shape[1]:
            array = array.at[:, destination_row, ...].set(
                sign * buf[folded, source_row - 1, ...])
    return array


def ice_halo_update(
    array, field_loc, field_type, fill_value=None, tripole_only=False, source=None, seam_average=True
):
    """This routine updates ghost cells for an input array and is a
    member of a group of routines under the generic interface
    ice_HaloUpdate.  This routine is the specific interface
    for 2d horizontal arrays of double precision.

    array containing field for which halo
    needs to be updated
    id for type of field (scalar, vector, angle)
    id for location on horizontal grid
     (center, NEcorner, Nface, Eface)
    optional value to put in ghost cells
     where neighbor points are unknown
     (e.g. eliminated land blocks or
      closed boundaries)
    optional flag to execute halo only across tripole seam
    local tripoleOnly value

    ``tripole_only`` updates only the tripole seam. On any other boundary
    that call changes nothing. An open or closed edge is left as it is,
    unless ``fill_value`` is set.
    """
    #-----------------------------------------------------------------------
    #
    #  abort or return on unknown or noupdate field_loc or field_type
    #
    #-----------------------------------------------------------------------
    if field_loc == field_loc_unknown or field_type == field_type_unknown:
        raise ValueError("field_loc/type unknown is not allowed in ice_halo_update")
    if field_loc == field_loc_noupdate or field_type == field_type_noupdate:
        return array
    if not tripole_only:
        # tripoleOnly skip fill, do not overwrite any values in interior as they may
        # already be set and filling tripole is not necessary
        #-----------------------------------------------------------------------
        #
        #  initialize error code and fill value
        #
        #-----------------------------------------------------------------------
        ghost = int(nghost)
        # block sizes for fill
        # begin, end indices for physical domain
        ilo, ihi, jlo, jhi = physical_bounds()
        # fill outer boundary ew
        # fill outer boundary if cyclic or tripole
        ew_fill = ew_boundary_type == "cyclic"
        # fill outer boundary ns
        ns_fill = ns_boundary_type in ("cyclic", "tripole", "tripoleT")
        if fill_value is not None:
            # always fill outer boundary if fillValue is passed
            ew_fill = True
            ns_fill = True
        # value to use for unknown points
        value = c0 if fill_value is None else fill_value
        #-----------------------------------------------------------------------
        #
        #  Fill out halo region
        #  Needed for masked halos to ensure halo values are filled for
        #  halo grid cells that are not updated
        #  In general, do NOT fill outer boundary for open boundary conditions
        #  because do not want to overwrite existing data
        #
        #-----------------------------------------------------------------------
        # fill outer boundary as needed
        # only fill corners if both edges are being filled
        if ew_fill:
            array[ilo - ghost:ilo, jlo:jhi + 1] = value  # west edge
            array[ihi + 1:ihi + 1 + ghost, jlo:jhi + 1] = value  # east edge
        if ns_fill:
            array[ilo:ihi + 1, jlo - ghost:jlo] = value  # south edge
            array[ilo:ihi + 1, jhi + 1:jhi + 1 + ghost] = value  # north edge
        if ew_fill and ns_fill:
            array[ilo - ghost:ilo, jlo - ghost:jlo] = value  # southwest corner
            array[ilo - ghost:ilo, jhi + 1:jhi + 1 + ghost] = value  # northwest corner
            array[ihi + 1:ihi + 1 + ghost, jlo - ghost:jlo] = value  # southeast corner
            array[ihi + 1:ihi + 1 + ghost, jhi + 1:jhi + 1 + ghost] = value  # northeast corner
        ghost = int(nghost)
        # begin, end indices for physical domain
        ilo, ihi, jlo, jhi = physical_bounds()
        if ew_boundary_type == "cyclic":
            #*** copy easternmost physical domain of src
            #*** into westernmost halo of dst
            array[ilo - ghost:ilo, jlo:jhi + 1] = array[ihi - ghost + 1:ihi + 1, jlo:jhi + 1]
            #*** copy westernmost physical domain of src
            #*** into easternmost halo of dst
            array[ihi + 1:ihi + 1 + ghost, jlo:jhi + 1] = array[ilo:ilo + ghost, jlo:jhi + 1]
        if ns_boundary_type == "cyclic":
            #*** copy northern physical domain of src
            #*** into southern halo of dst
            array[ilo:ihi + 1, jlo - ghost:jlo] = array[ilo:ihi + 1, jhi - ghost + 1:jhi + 1]
            #*** copy southern physical domain of src
            #*** into northern halo of dst
            array[ilo:ihi + 1, jhi + 1:jhi + 1 + ghost] = array[ilo:ihi + 1, jlo:jlo + ghost]
        # Edge copies stop at the physical row or column, so the four
        # corner ghosts are still the fill. Both boundaries cyclic means
        # each corner is the opposite physical corner.
        if ew_boundary_type == "cyclic" and ns_boundary_type == "cyclic":
            array[ilo - ghost:ilo, jlo - ghost:jlo] = array[
                ihi - ghost + 1:ihi + 1, jhi - ghost + 1:jhi + 1
            ]
            array[ihi + 1:ihi + 1 + ghost, jlo - ghost:jlo] = array[
                ilo:ilo + ghost, jhi - ghost + 1:jhi + 1
            ]
            array[ilo - ghost:ilo, jhi + 1:jhi + 1 + ghost] = array[
                ihi - ghost + 1:ihi + 1, jlo:jlo + ghost
            ]
            array[ihi + 1:ihi + 1 + ghost, jhi + 1:jhi + 1 + ghost] = array[
                ilo:ilo + ghost, jlo:jlo + ghost
            ]
    if ns_boundary_type in ("tripole", "tripoleT"):
        # Fold the northern physical rows into the northern ghosts.
        #-----------------------------------------------------------------------
        #
        #  take care of northern boundary in tripole case
        #  bufTripole array contains the top nghost+1 rows (u-fold) or nghost+2 rows
        #  (T-fold) of physical domain for entire (global) top row
        #
        #-----------------------------------------------------------------------
        #*** set tripole flag and add two copies for inserting
        #*** and extracting info from the tripole buffer
        ghost = int(nghost)
        # global domain size in x (tripole)
        nx = int(ice_domain_size.nx_global)
        # number of rows in tripole buffer
        rows = ghost + 1
        if ns_boundary_type == "tripoleT":
            rows += 1
        # begin, end indices for physical domain
        ilo, ihi, _jlo, jhi = physical_bounds()
        ig, _jg = global_indices()
        #-----------------------------------------------------------------------
        #
        #  global buffers for tripole boundary
        #
        #-----------------------------------------------------------------------
        #*** allocate tripole message buffers if not already done
        # Trailing dimensions stay with the field. A 2D buffer would
        # drop every layer but the first.
        buf = np.zeros((nx, rows, *array.shape[2:]), dtype=array.dtype)
        # Physical columns of the top rows, in global-i order.
        #*** if a tripole boundary block, also create a local
        #*** message into and out of tripole buffer
        #*** copy in
        #*** tripole grid - copy info into tripole buffer
        #*** copy physical domain of top halo+1 rows
        #*** into global buffer at src location
        #*** for tripole grids, send a north tripole message to
        #*** the east block to make sure enough information is
        #*** available for tripole manipulations
        # tcx,tcraig, 3/2023, this is not needed
        #      if (tripoleBlock) then
        #         call ice_HaloMsgCreate(halo, dist, iblock, -eastBlock, 'north')
        #      endif
        #*** for tripole grids, send a north tripole message to
        #*** the west block to make sure enough information is
        #*** available for tripole manipulations
        # tcx,tcraig, 3/2023, this is not needed
        #      if (tripoleBlock) then
        #         call ice_HaloMsgCreate(halo, dist, iblock, -westBlock, 'north')
        #      endif
        # case ('northeast')
        # tcx,tcraig, 3/2023, this is not needed
        #         else
        #
        #            !*** tripole grid - copy entire top halo+1
        #            !*** rows into global buffer at src location
        #
        #            msgIndx = halo%numLocalCopies
        #
        #            do j=1,nghost+1
        #            do i=1,ieSrc-ibSrc+1
        #
        #               msgIndx = msgIndx + 1
        #
        #               halo%srcLocalAddr(1,msgIndx) = ibSrc + i - 1
        #               halo%srcLocalAddr(2,msgIndx) = jeSrc-1-nghost+j
        #               halo%srcLocalAddr(3,msgIndx) = srcLocalID
        #
        #               halo%dstLocalAddr(1,msgIndx) = iGlobal(ibSrc + i - 1)
        #               halo%dstLocalAddr(2,msgIndx) = j
        #               halo%dstLocalAddr(3,msgIndx) = -dstLocalID
        #
        #            end do
        #            end do
        #
        #            halo%numLocalCopies = msgIndx
        # case ('northwest')
        # tcx,tcraig, 3/2023, this is not needed
        #         else
        #
        #            !*** tripole grid - copy entire top halo+1
        #            !*** rows into global buffer at src location
        #
        #            msgIndx = halo%numLocalCopies
        #
        #            do j=1,nghost+1
        #            do i=1,ieSrc-ibSrc+1
        #
        #               msgIndx = msgIndx + 1
        #
        #               halo%srcLocalAddr(1,msgIndx) = ibSrc + i - 1
        #               halo%srcLocalAddr(2,msgIndx) = jeSrc-1-nghost+j
        #               halo%srcLocalAddr(3,msgIndx) = srcLocalID
        #
        #               halo%dstLocalAddr(1,msgIndx) = iGlobal(ibSrc + i - 1)
        #               halo%dstLocalAddr(2,msgIndx) = j
        #               halo%dstLocalAddr(3,msgIndx) = -dstLocalID
        #
        #            end do
        #            end do
        #
        #            halo%numLocalCopies = msgIndx
        # dummy loop indices
        # source addresses for message
        # dest   addresses for message
        ib = ilo + 1
        je = jhi + 1
        ie = ihi + 1
        # ice_HaloUpdate_stress reads the paired stress component into the buffer.
        buf_src = array if source is None else source
        for j in range(1, rows + 1):
            src_j = je - rows + j
            for i in range(1, ie - ib + 2):
                src_i = ib + i - 1
                # tripole copy into buffer
                buf[int(ig[src_i - 1]) - 1, j - 1] = buf_src[src_i - 1, src_j - 1]

        # sign factor for tripole grids
        # sign factor for tripole boundary conditions
        if int(field_type) in (int(field_type_vector), int(field_type_angle)):
            sign = -1
        elif int(field_type) in (int(field_type_scalar), int(field_type_noupdate)):
            # field_type_noupdate: ghost cells not needed - use cell center
            sign = 1
        else:
            raise ValueError("unknown field type")
        # address shifts for tripole
        loc = int(field_loc)
        if ns_boundary_type == "tripoleT":
            if loc == int(field_loc_center):  # cell center location
                #*** top row is degenerate, so must enforce symmetry
                #***   use average of two degenerate points for value
                ioffset, joffset = -1, 0
                pairs = [(i, nx - i + 2) for i in range(2, nx // 2 + 1)]
            elif loc == int(field_loc_NEcorner):  # cell corner location
                ioffset, joffset, pairs = 0, 1, []
            elif loc == int(field_loc_Eface):  # cell center location
                #*** top row is degenerate, so must enforce symmetry
                #***   use average of two degenerate points for value
                ioffset, joffset = 0, 0
                pairs = [(i, nx + 1 - i) for i in range(1, nx // 2 + 1)]
            elif loc == int(field_loc_Nface):  # cell corner (velocity) location
                ioffset, joffset, pairs = -1, 1, []
            else:
                raise ValueError("unknown field location in ice_halo_update")
        else:  # tripole u-fold
            if loc == int(field_loc_center):  # cell center location
                ioffset, joffset, pairs = 0, 0, []
            elif loc == int(field_loc_NEcorner):  # cell corner location
                #*** top row is degenerate, so must enforce symmetry
                #***   use average of two degenerate points for value
                ioffset, joffset = 1, 1
                pairs = [(i, nx - i) for i in range(1, nx // 2)]
            elif loc == int(field_loc_Eface):  # cell center location
                ioffset, joffset, pairs = 1, 0, []
            elif loc == int(field_loc_Nface):  # cell corner (velocity) location
                #*** top row is degenerate, so must enforce symmetry
                #***   use average of two degenerate points for value
                ioffset, joffset = 0, 1
                pairs = [(i, nx + 1 - i) for i in range(1, nx // 2 + 1)]
            else:
                raise ValueError("unknown field location in ice_halo_update")
        last = rows - 1
        #*** top row is degenerate, so must enforce symmetry
        #***   use average of two degenerate points for value
        # ice_HaloUpdate_stress copies the seam and does not average it.
        if seam_average:
            for i_left, i_right in pairs:
                # scalars for enforcing symmetry at U pts
                x1 = buf[i_left - 1, last]
                x2 = buf[i_right - 1, last]
                average = p5 * (x1 + dbl_kind(sign) * x2)
                # I4 halo stores nint of the average; a float stored in an
                # integer buffer would truncate toward zero.
                if np.issubdtype(buf.dtype, np.integer):
                    average = np.sign(average) * np.floor(np.abs(average) + 0.5)
                buf[i_left - 1, last] = average
                buf[i_right - 1, last] = dbl_kind(sign) * average

        # if srcBlock is less than zero, the message is a copy out of the
        #   tripole buffer and will be treated later
        #*** copy out of tripole buffer - includes halo
        #*** copy out of global tripole buffer into local
        #*** ghost cells
        #*** look through local copies to find the copy out
        #*** messages (srcBlock < 0)
        #*** tripole grid - set up for copying out of
        #*** tripole buffer into ghost cell domains
        #*** include e-w ghost cells
        for j in range(1, rows + 1):
            j_dst = je + j - 1 if j <= ghost + 1 else 0  # never used
            #*** correct for offsets
            j_src = ghost + 3 - j - joffset
            for i in range(1, ie + ghost + 1):
                # tripole buffer addr
                i_src = nx - int(ig[i - 1]) + 1 - ioffset
                if i_src < 1:
                    i_src += nx
                if i_src > nx:
                    i_src -= nx
                #*** for center and Eface on u-fold, and NE corner and Nface
                #*** on T-fold, do not need to replace
                #*** top row of physical domain, so jSrc should be
                #*** out of range and skipped
                #*** otherwise do the copy
                # local block addr
                if 0 < j_src <= rows and j_dst > 0:
                    array[i - 1, j_dst - 1] = dbl_kind(sign) * buf[i_src - 1, j_src - 1]
    return array


def ice_HaloUpdate_stress(array1, array2, field_loc, field_type, fill_value=None):
    """Update the tripole seam of ``array1`` from the paired component ``array2``.

    author Elizabeth C. Hunke, LANL

    do NOT zero the halo out, this halo update just updates
    the tripole zipper as needed for stresses.  if you zero
    it out, all halo values will be wiped out.

    One grid. ``tripole_only`` skips the fill. The buffer is filled from
    ``array2`` and written into ``array1``.
    """
    if ns_boundary_type not in ("tripole", "tripoleT"):
        return array1
    return ice_halo_update(
        array1,
        field_loc,
        field_type,
        fill_value=fill_value,
        tripole_only=True,
        source=array2,
        seam_average=False,
    )


def ice_halo_extrapolate(array):
    """This subroutine extrapolates ARRAY values into the ghost cells,
    and is intended for grid variables whose ghost cells
    would otherwise be set using the default boundary conditions (Dirichlet
    or Neumann).

    This is the specific interface for double precision arrays
    corresponding to the generic interface ice_HaloExtrapolate

    T.Craig, Oct 2025 - extend to nghost > 1

    generic interface
    ice_HaloExtrapolate2DR8
    ice_HaloExtrapolate2DR4  ! not yet
    ice_HaloExtrapolate2DI4  ! implemented

    array containing distributed field

    Linear extrapolation into ghost cells on a non-cyclic edge.

    Each ghost layer uses the two cells just inside it, so the layers are
    filled from the interior outward. North and south are applied after
    east and west, and they overwrite the corner ghosts. A tripole north
    edge is left to the halo fold.
    """
    ghost = int(nghost)
    #-----------------------------------------------------------------------
    #
    #  Linear extrapolation
    #
    #-----------------------------------------------------------------------
    # active block indices
    # begin, end indices for physical domain
    # dummy loop indices
    ilo, ihi, jlo, jhi = physical_bounds()

    if ew_boundary_type != "cyclic":
        # west edge
        for n in range(1, ghost + 1):
            ii = ilo - n  # gridcell to extrapolate to
            array[ii, :] = c2 * array[ii + 1, :] - array[ii + 2, :]
        # east edge
        for n in range(1, ghost + 1):
            ii = ihi + n  # gridcell to extrapolate to
            array[ii, :] = c2 * array[ii - 1, :] - array[ii - 2, :]

    if ns_boundary_type != "cyclic":
        # south edge
        for n in range(1, ghost + 1):
            jj = jlo - n  # gridcell to extrapolate to
            array[:, jj] = c2 * array[:, jj + 1] - array[:, jj + 2]
        if ns_boundary_type not in ("tripole", "tripoleT"):
            # north edge
            for n in range(1, ghost + 1):
                jj = jhi + n  # gridcell to extrapolate to
                array[:, jj] = c2 * array[:, jj - 1] - array[:, jj - 2]
    return array


def scatter_global(array, global_field, field_loc, field_type):
    """This subroutine scatters a global-sized array on the processor
    src_task to a distribution of blocks given by dst_dist.

    This is the specific interface for double precision arrays
    corresponding to the generic interface scatter_global.  It is shown
    to provide information on the generic interface (the generic
    interface is identical, but chooses a specific interface based
    on the data type of the input argument).

    array containing global field on src_task
    array containing distributed field
    id for type of field (scalar, vector, angle)
    id for location on horizontal grid
     (center, NEcorner, Nface, Eface)

    Copy a global (nx, ny) field onto the halo array.

    Points whose global index lies inside the field are copied. Cyclic
    ghosts use the wrapped index. Tripole northern ghosts are reflected
    across the top row with the field location offset and the vector sign.
    ``field_loc_noupdate`` then clears every ghost cell.
    """
    if field_loc == field_loc_unknown or field_type == field_type_unknown:
        raise ValueError("field_loc/type unknown is not allowed in scatter_global")

    nx = global_field.shape[0]
    ny = global_field.shape[1]
    ig, jg = global_indices()
    #-----------------------------------------------------------------------
    #
    #  initialize return array to zero and set up tripole quantities
    #
    #-----------------------------------------------------------------------
    array[...] = c0
    tripole = ns_boundary_type in ("tripole", "tripoleT")
    normal = jg >= 1
    if tripole:
        normal = jg >= 1
    #-----------------------------------------------------------------------
    #
    #  copy blocks of global array into local block distribution
    #
    #-----------------------------------------------------------------------
    # dummy loop indices
    # source addresses
    # normal block
    ii, jj = np.meshgrid(ig, jg, indexing="ij")
    valid = normal[None, :] & (ii >= 1) & (ii <= nx) & (jj >= 1) & (jj <= ny)
    array[valid] = global_field[ii[valid] - 1, jj[valid] - 1]

    if tripole and int(field_loc) != int(field_loc_noupdate):
        # offsets for tripole boundary conditions
        # for the tripoleTflag - all blocks have it
        loc = int(field_loc)
        if ns_boundary_type == "tripoleT" and loc != int(field_loc_noupdate):
            table = {
                int(field_loc_center): (2, 0),  # cell center location
                int(field_loc_NEcorner): (1, -1),  # cell corner (velocity) location
                int(field_loc_Eface): (1, 0),  # cell face location
                int(field_loc_Nface): (2, -1),  # cell face location
            }
        else:
            table = {
                int(field_loc_center): (1, 1),  # cell center location
                int(field_loc_NEcorner): (0, 0),  # cell corner (velocity) location
                int(field_loc_Eface): (0, 1),  # cell face location
                int(field_loc_Nface): (1, 0),  # cell face location
                int(field_loc_noupdate): (1, 1),  # ghost cells never used - use cell center
            }
        if loc not in table:
            raise ValueError("unknown field location in scatter_global")
        xoffset, yoffset = table[loc]
        # sign factor for tripole grids
        # sign factor for tripole boundary conditions
        if int(field_type) in (int(field_type_vector), int(field_type_angle)):
            sign = dbl_kind(-1)
        elif int(field_type) in (int(field_type_scalar), int(field_type_noupdate)):
            # field_type_noupdate: ghost cells not needed - use cell center
            sign = dbl_kind(1)
        else:
            raise ValueError("unknown field type")
        for j_py, j_glob in enumerate(jg):
            if int(j_glob) >= 0:
                continue
            # tripole is top block with j_glob < 0
            # for yoffset=0 or 1, yoffset2=0,0
            # for yoffset=-1, yoffset2=0,1, for u-rows on T-fold grid
            for yoffset2 in range(0, max(yoffset, 0) - yoffset + 1):
                jsrc = ny + yoffset + yoffset2 + (int(j_glob) + ny)
                jdst = j_py - yoffset2
                if jdst < 0 or not (1 <= jsrc <= ny):
                    continue
                for i_py, i_glob in enumerate(ig):
                    if int(i_glob) == 0:
                        continue
                    isrc = nx + xoffset - int(i_glob)
                    if isrc < 1:
                        isrc += nx
                    if isrc > nx:
                        isrc -= nx
                    array[i_py, jdst] = sign * global_field[isrc - 1, jsrc - 1]

    if int(field_loc) == int(field_loc_noupdate):
        #-----------------------------------------------------------------
        # Set ghost cell values to 0 for noupdate
        #-----------------------------------------------------------------
        ghost = int(nghost)
        array[:ghost, :] = c0  # west edge
        array[-ghost:, :] = c0  # east edge
        array[:, :ghost] = c0  # south edge
        array[:, -ghost:] = c0  # north edge
    return array


def scatter_global_ext(array, global_ext):
    """This subroutine scatters a global-sized array on the processor
    src_task to a distribution of blocks given by dst_dist.

    This is the specific interface for double precision arrays
    corresponding to the generic interface scatter_global.  It is shown
    to provide information on the generic interface (the generic
    interface is identical, but chooses a specific interface based
    on the data type of the input argument).

    array containing global field on src_task
    array containing distributed field

    initialize return array to zero

    Copy an extended global field, ghosts included, onto the halo array.

    This is one block covering the whole domain, so it is both the first
    and the last block. The edge copies in ``scatter_global_ext`` then
    put the file's ghost rim back, and the halo array matches the record.
    """
    ghost = int(nghost)
    nx = int(ice_domain_size.nx_global)
    ny = int(ice_domain_size.ny_global)
    expect = (nx + 2 * ghost, ny + 2 * ghost)
    if global_ext.shape != expect:
        raise ValueError(
            f"(scatter_global_ext) expected {expect}, got {global_ext.shape}"
        )
    #-----------------------------------------------------------------------
    #
    #  copy blocks of global array into local block distribution
    #
    #-----------------------------------------------------------------------
    # source addresses
    # interior
    # south edge
    # southwest corner
    # southeast corner
    # north edge
    # northwest corner
    # northeast corner
    # west edge
    # northwest corner
    # southwest corner
    # east edge
    # northeast corner
    # southeast corner
    array[...] = global_ext
    return array
