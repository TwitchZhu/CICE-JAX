"""Restore points on the boundaries of the global grid to their initial values.
These are saved at the time of initialization in the main program.
They must be restored each time step.

author for JAX version: Xinyuan Zhu
XZ: converted from F90 to Python

authors: Elizabeth C. Hunke, LANL

One grid, no block index. ``ice_HaloRestore_init`` copies the initial
edge state and multiplies by the land mask. ``ice_HaloRestore`` performs
the runtime JAX update. ``restore_ic`` is ``initial``; ``set_restore_var``
for the defined-state branch is not ported and raises.
"""

import numpy as np

from cicecore.cicedyn.general import ice_init
from cicecore.cicedyn.general import ice_state
from cicecore.cicedyn.infrastructure import ice_domain
from cicecore.cicedyn.infrastructure import ice_grid
from cicecore.cicedyn.infrastructure.comm.serial.ice_boundary import global_indices
from cicecore.shared import ice_domain_size
from cicecore.shared import ice_fileunits
from cicecore.shared.ice_constants import c0
from cicecore.shared.ice_kinds_mod import dbl_kind
from icepack.columnphysics import icepack_tracers

aicen_rest = None
vicen_rest = None
vsnon_rest = None
trcrn_rest = None


def _edge_limit(glob, other):
    """Last index that is not padding, matching the ibc walk.

    A column or row whose global index is 0 is padding when the other
    direction still has a real point.
    """
    ibc = len(glob) - 1
    other_sum = int(np.sum(other))
    for index in range(len(glob) - 1, -1, -1):
        npad = 0
        if int(glob[index]) == 0:
            npad = other_sum
        if npad != 0:
            ibc = ibc - 1
    return ibc


def ice_HaloRestore_init():
    """Initialize restoring on the boundaries of the global grid.

    Returns immediately when ``restore_ice`` is false.
    """
    global aicen_rest, vicen_rest, vsnon_rest, trcrn_rest

    subname = "(ice_HaloRestore_init)"
    # restore_ic = 'defined' ! otherwise restore to initial ice state
    restore_ic = "initial"  # restore to initial ice state

    if not bool(ice_init.restore_ice):
        return

    ntrcr = int(icepack_tracers.ntrcr)
    ncat = int(icepack_tracers.ncat)
    ew_boundary_type = str(ice_domain.ew_boundary_type).strip()
    ns_boundary_type = str(ice_domain.ns_boundary_type).strip()

    if (ew_boundary_type == "open" or ns_boundary_type == "open") and (
        not bool(ice_init.restart_ext)
    ):
        ice_fileunits.nu_diag.write(" ERROR: restart_ext=F and open boundaries\n")
        raise RuntimeError(subname + "open boundary and restart_ext=F")

    horiz = ice_domain_size.horizontal_shape()
    aicen_rest = np.zeros(horiz + (ncat,), dtype=dbl_kind)
    vicen_rest = np.zeros(horiz + (ncat,), dtype=dbl_kind)
    vsnon_rest = np.zeros(horiz + (ncat,), dtype=dbl_kind)
    trcrn_rest = np.zeros(horiz + (ntrcr, ncat), dtype=dbl_kind)

    aicen_rest[...] = c0
    vicen_rest[...] = c0
    vsnon_rest[...] = c0
    trcrn_rest[...] = c0

    #-----------------------------------------------------------------------
    # initialize
    # halo cells have to be filled manually at this stage
    # these arrays could be set to values read from a file...
    #-----------------------------------------------------------------------

    if restore_ic.strip() == "defined":
        # restore to defined ice state
        # set_restore_var is the dead branch. restore_ic is 'initial'.
        raise NotImplementedError(subname + " set_restore_var")
    else:  # restore_ic
        # restore to initial ice state

        # the easy way
        # aicen_rest(:,:,:,:) = aicen(:,:,:,:)
        # vicen_rest(:,:,:,:) = vicen(:,:,:,:)
        # vsnon_rest(:,:,:,:) = vsnon(:,:,:,:)
        # trcrn_rest(:,:,:,:,:) = trcrn(:,:,:,:,:)

        # the more precise way
        # One grid is both the west and east edge, and the south and north edge.
        ilo, ihi, jlo, jhi = ice_domain_size.physical_bounds()
        iglob, jglob = global_indices()
        aicen = ice_state.aicen
        vicen = ice_state.vicen
        vsnon = ice_state.vsnon
        trcrn = ice_state.trcrn

        if ew_boundary_type != "cyclic":  # west edge
            # do i = 1, ilo
            aicen_rest[: ilo + 1, :, :] = aicen[ilo : ilo + 1, :, :]
            vicen_rest[: ilo + 1, :, :] = vicen[ilo : ilo + 1, :, :]
            vsnon_rest[: ilo + 1, :, :] = vsnon[ilo : ilo + 1, :, :]
            trcrn_rest[: ilo + 1, :, :, :] = trcrn[ilo : ilo + 1, :, :, :]

        if ew_boundary_type != "cyclic":  # east edge
            # locate ghost cell column (avoid padding)
            ibc = _edge_limit(iglob, jglob)
            # do i = ihi, ibc
            aicen_rest[ihi : ibc + 1, :, :] = aicen[ihi : ihi + 1, :, :]
            vicen_rest[ihi : ibc + 1, :, :] = vicen[ihi : ihi + 1, :, :]
            vsnon_rest[ihi : ibc + 1, :, :] = vsnon[ihi : ihi + 1, :, :]
            trcrn_rest[ihi : ibc + 1, :, :, :] = trcrn[ihi : ihi + 1, :, :, :]

        if ns_boundary_type != "cyclic":  # south edge
            # do j = 1, jlo
            aicen_rest[:, : jlo + 1, :] = aicen[:, jlo : jlo + 1, :]
            vicen_rest[:, : jlo + 1, :] = vicen[:, jlo : jlo + 1, :]
            vsnon_rest[:, : jlo + 1, :] = vsnon[:, jlo : jlo + 1, :]
            trcrn_rest[:, : jlo + 1, :, :] = trcrn[:, jlo : jlo + 1, :, :]

        if (
            ns_boundary_type != "cyclic"
            and ns_boundary_type != "tripole"
            and ns_boundary_type != "tripoleT"
        ):  # north edge
            # locate ghost cell row (avoid padding)
            jbc = _edge_limit(jglob, iglob)
            # do j = jhi, ibc
            aicen_rest[:, jhi : jbc + 1, :] = aicen[:, jhi : jhi + 1, :]
            vicen_rest[:, jhi : jbc + 1, :] = vicen[:, jhi : jhi + 1, :]
            vsnon_rest[:, jhi : jbc + 1, :] = vsnon[:, jhi : jhi + 1, :]
            trcrn_rest[:, jhi : jbc + 1, :, :] = trcrn[:, jhi : jhi + 1, :, :]

    #-----------------------------------------------------------------
    # Impose land mask
    #-----------------------------------------------------------------
    hm = ice_grid.hm
    aicen_rest[...] = aicen_rest * hm[:, :, None]
    vicen_rest[...] = vicen_rest * hm[:, :, None]
    vsnon_rest[...] = vsnon_rest * hm[:, :, None]
    trcrn_rest[...] = trcrn_rest * hm[:, :, None, None]

    ice_fileunits.nu_diag.write(
        f" ice restoring timescale =  {ice_init.trestore}  days\n"
    )


def ice_HaloRestore(pytree, static):
    """Relax noncyclic edge cells toward their saved initial values.

    This is ice_HaloRestore from ice_restoring.F90 for the single full grid.
    The west/south ranges include the first physical cell, as in Fortran.
    """
    state = dict(pytree["state"])
    ice = dict(state["ice_state"])
    rest = state["ice_restoring"]
    bounds = static["ice_domain_size"]["physical_bounds"]
    ilo, ihi, jlo, jhi = bounds
    domain = static["ice_domain"]
    dt = static["ice_calendar"]["dt"]
    trestore = static["ice_init"]["trestore"]
    secday = pytree["parameters"]["icepack_parameters"]["secday"]

    # Fortran uses dt when trestore=0: data are used instantaneously.
    trest = dt if trestore == 0 else trestore * secday
    ctime = dt / trest

    if domain["ew_boundary_type"].strip() != "cyclic":
        for name in ("aicen", "vicen", "vsnon", "trcrn"):
            current = ice[name]
            target = rest[name + "_rest"]
            current = current.at[:ilo + 1, ...].set(
                current[:ilo + 1, ...] + (target[:ilo + 1, ...] - current[:ilo + 1, ...]) * ctime
            )
            # The ibc padding walk was resolved once from fixed grid indices.
            ibc = domain["east_edge_limit"]
            current = current.at[ihi:ibc + 1, ...].set(
                current[ihi:ibc + 1, ...]
                + (target[ihi:ibc + 1, ...] - current[ihi:ibc + 1, ...]) * ctime
            )
            ice[name] = current

    if domain["ns_boundary_type"].strip() != "cyclic":
        for name in ("aicen", "vicen", "vsnon", "trcrn"):
            current = ice[name]
            target = rest[name + "_rest"]
            current = current.at[:, :jlo + 1, ...].set(
                current[:, :jlo + 1, ...] + (target[:, :jlo + 1, ...] - current[:, :jlo + 1, ...]) * ctime
            )
            if domain["ns_boundary_type"].strip() not in ("tripole", "tripoleT"):
                jbc = domain["north_edge_limit"]
                current = current.at[:, jhi:jbc + 1, ...].set(
                    current[:, jhi:jbc + 1, ...]
                    + (target[:, jhi:jbc + 1, ...] - current[:, jhi:jbc + 1, ...]) * ctime
                )
            ice[name] = current

    state["ice_state"] = ice
    result = dict(pytree)
    result["state"] = state
    return result
