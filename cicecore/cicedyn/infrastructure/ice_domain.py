"""This module contains the model domain and routines for initializing
the domain.  It also initializes the decompositions and
distributions across processors/threads by calling relevant
routines in the block, distribution modules.

author for JAX version: Xinyuan Zhu
XZ: converted from F90 to Python

author: Phil Jones, LANL
Oct. 2004: Adapted from POP by William H. Lipscomb, LANL
Feb. 2007: E. Hunke removed NE and SW boundary options (they were buggy
 and not used anyhow).

One grid. ``init_domain_blocks`` reads ``domain_nml`` and stores the
boundary types, domain size, and ``maskhalo_*`` flags.
``create_blocks`` and ``init_domain_distribution`` are not ported.
"""

import re

from cicecore.cicedyn.general import ice_init
from cicecore.cicedyn.infrastructure.comm.serial.ice_boundary import set_boundary_types
from cicecore.shared import ice_domain_size
from cicecore.shared import ice_fileunits
from icepack.columnphysics import icepack_tracers

nblocks = None  # actual number of blocks on this processor

# deprecated Nov, 2025
close_boundaries = None

blocks_ice = None  # block ids for local blocks
distrb_info = None  # block distribution info
halo_info = None    # ghost cell update info

# type of domain bndy in each logical direction (ew is i, ns is j)
ew_boundary_type = None
ns_boundary_type = None

# if true, use masked halo updates for dynamics
maskhalo_dyn = None
# if true, use masked halo updates for transport
maskhalo_remap = None
# if true, use masked halo updates for bound_state
maskhalo_bound = None
# if true, bundle halo update in dynamics
halo_dynbundle = None
# if true, land block elimination is on
landblockelim = None

# method to use for distributing blocks
distribution_type = None
# method for weighting work per block
distribution_wght = None
# file for distribution_wght=file
distribution_wght_file = None

# num of processors
nprocs = None
processor_shape = None
add_mpi_barriers = None
debug_blocks = None
max_blocks = None
block_size_x = None
block_size_y = None


def init_domain_blocks(nml_path="configuration/ice_in"):
    """Read domain information and set up the block decomposition.

    Block decomposition is not ported. The namelist read and the
    domain-size check follow the Fortran subroutine. JAX uses one
    process, so a namelist nprocs other than 1 is not an error.
    """
    global nblocks, close_boundaries, ew_boundary_type, ns_boundary_type
    global maskhalo_dyn, maskhalo_remap, maskhalo_bound, halo_dynbundle
    global landblockelim, distribution_type, distribution_wght
    global distribution_wght_file, nprocs, processor_shape
    global add_mpi_barriers, debug_blocks, max_blocks, block_size_x, block_size_y

    subname = "(init_domain_blocks)"
    #----------------------------------------------------------------------
    # read domain information from namelist input
    #----------------------------------------------------------------------
    nprocs = -1
    processor_shape = "slenderX2"
    distribution_type = "cartesian"
    distribution_wght = "latitude"
    distribution_wght_file = "unknown"
    ew_boundary_type = "cyclic"
    ns_boundary_type = "open"
    maskhalo_dyn = False       # if true, use masked halos for dynamics
    maskhalo_remap = False     # if true, use masked halos for transport
    maskhalo_bound = False     # if true, use masked halos for bound_state
    halo_dynbundle = True      # if true, bundle halo updates in dynamics
    add_mpi_barriers = False   # if true, throttle communication
    debug_blocks = False       # if true, print verbose block information
    max_blocks = -1            # max number of blocks per processor
    block_size_x = -1          # size of block in first horiz dimension
    block_size_y = -1          # size of block in second horiz dimension
    nx_global = -1             # NXGLOB,  i-axis size
    ny_global = -1             # NYGLOB,  j-axis size
    landblockelim = True       # on by default
    close_boundaries = False

    state = {
        "nprocs": nprocs,
        "max_blocks": max_blocks,
        "block_size_x": block_size_x,
        "block_size_y": block_size_y,
        "nx_global": nx_global,
        "ny_global": ny_global,
        "processor_shape": processor_shape,
        "distribution_type": distribution_type,
        "distribution_wght": distribution_wght,
        "distribution_wght_file": distribution_wght_file,
        "ew_boundary_type": ew_boundary_type,
        "ns_boundary_type": ns_boundary_type,
        "maskhalo_dyn": maskhalo_dyn,
        "maskhalo_remap": maskhalo_remap,
        "maskhalo_bound": maskhalo_bound,
        "add_mpi_barriers": add_mpi_barriers,
        "debug_blocks": debug_blocks,
    }

    # if (my_task == master_task): one process, so the read always runs.
    nu = ice_fileunits.nu_diag
    nml_name = "domain_nml"
    nu.write(f" {subname} Reading {nml_name}\n")
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
    if nml_name not in groups:
        raise ValueError(subname + " ERROR: searching for " + nml_name)

    assignments = []
    pending = None
    for text in groups[nml_name]:
        if "=" in text:
            if pending is not None:
                assignments.append(pending)
            pending = text
        else:
            if pending is None:
                raise ValueError(subname + " ERROR: " + nml_name + " reading " + text)
            pending = pending + " " + text
    if pending is not None:
        assignments.append(pending)

    for item in assignments:
        key, raw = item.split("=", 1)
        key = key.strip()
        if key not in state:
            raise ValueError(
                subname + " ERROR: " + nml_name + " reading " + item.strip()
            )
        prototype = state[key]
        tokens = []
        buf = []
        quote = None
        for char in raw.strip():
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
        if len(tokens) != 1:
            raise ValueError(
                subname + " ERROR: " + nml_name + " reading " + item.strip()
            )
        token = tokens[0].strip()
        if isinstance(prototype, bool):
            lowered = token.lower()
            if lowered in (".true.", "true"):
                state[key] = True
            elif lowered in (".false.", "false"):
                state[key] = False
            else:
                raise ValueError(
                    subname + " ERROR: " + nml_name + " reading " + item.strip()
                )
        elif isinstance(prototype, str):
            if len(token) >= 2 and token[0] == token[-1] and token[0] in "'\"":
                state[key] = token[1:-1]
            else:
                state[key] = token
        elif isinstance(prototype, int):
            state[key] = int(token)
        else:
            text = re.sub(r"([0-9.])d([+-]?\d+)", r"\1e\2", token.strip().lower())
            state[key] = float(text)

    nprocs = state["nprocs"]
    processor_shape = state["processor_shape"]
    distribution_type = state["distribution_type"]
    distribution_wght = state["distribution_wght"]
    distribution_wght_file = state["distribution_wght_file"]
    ew_boundary_type = state["ew_boundary_type"]
    ns_boundary_type = state["ns_boundary_type"]
    maskhalo_dyn = state["maskhalo_dyn"]
    maskhalo_remap = state["maskhalo_remap"]
    maskhalo_bound = state["maskhalo_bound"]
    add_mpi_barriers = state["add_mpi_barriers"]
    debug_blocks = state["debug_blocks"]
    max_blocks = state["max_blocks"]
    block_size_x = state["block_size_x"]
    block_size_y = state["block_size_y"]
    nx_global = state["nx_global"]
    ny_global = state["ny_global"]

    # call broadcast_scalar is MPI and is not ported.

    #----------------------------------------------------------------------
    # Set nprocs if not explicitly set to valid value in namelist
    #----------------------------------------------------------------------
    # ifdef CESMCOUPLED
    # nprocs = get_num_procs()
    # else
    #   if (nprocs < 0) then
    #      nprocs = get_num_procs()
    #   else if (nprocs /= get_num_procs()) then
    #      write(nu_diag,*) subname,' ERROR: nprocs, get_num_procs = ',nprocs,get_num_procs()
    #      call abort_ice(subname//' ERROR: Input nprocs not same as system (e.g MPI) request', ...)
    #   endif
    # endif
    # get_num_procs is MPI and is not ported. JAX always has one process.
    nprocs = 1

    ice_domain_size.sync_vertical_sizes(
        ice_init.ncat,
        ice_init.nilyr,
        ice_init.nslyr,
        ice_init.nblyr,
        ice_init.nfsd,
        icepack_tracers.ntrcr,
    )
    ncat = int(ice_domain_size.ncat)
    nilyr = int(ice_domain_size.nilyr)
    nslyr = int(ice_domain_size.nslyr)

    #----------------------------------------------------------------------
    # perform some basic checks on domain
    #----------------------------------------------------------------------
    if nx_global < 1 or ny_global < 1 or ncat < 1:
        # domain size zero or negative
        raise ValueError(subname + " ERROR: Invalid domain: size < 1")
    if int(ice_domain_size.nghost) < 1:
        # must have at least 1 layer of ghost cells
        raise ValueError(subname + " ERROR: Not enough ghost cells allocated")

    ice_domain_size.set_horizontal(nx_global, ny_global)
    set_boundary_types(ew_boundary_type, ns_boundary_type)

    #----------------------------------------------------------------------
    # compute block decomposition and details
    #----------------------------------------------------------------------
    # call create_blocks(nx_global, ny_global, trim(ew_boundary_type),
    #                                          trim(ns_boundary_type))
    # One grid. Block decomposition is not ported.
    nblocks = 1

    #----------------------------------------------------------------------
    # Now we need grid info before proceeding further
    # Print some domain information
    #----------------------------------------------------------------------
    # if (my_task == master_task): one process, so the writes always run.
    nu.write("\nDomain Information\n\n")
    nu.write(f"  Horizontal domain: nx = {nx_global:6d}\n")
    nu.write(f"                     ny = {ny_global:6d}\n")
    nu.write(f"  No. of categories: nc = {ncat:6d}\n")
    nu.write(f"  No. of ice layers: ni = {nilyr:6d}\n")
    nu.write(f"  No. of snow layers:ns = {nslyr:6d}\n")
    nu.write(f"  Processors:  total    = {nprocs:6d}\n")
    nu.write(f"  Processor shape       = {processor_shape.strip()}\n")
    nu.write(f"  Distribution type     = {distribution_type.strip()}\n")
    nu.write(f"  Distribution weight   = {distribution_wght.strip()}\n")
    nu.write(f"  Distribution wght file= {distribution_wght_file.strip()}\n")
    nu.write(f"  ew_boundary_type      = {ew_boundary_type.strip()}\n")
    nu.write(f"  ns_boundary_type      = {ns_boundary_type.strip()}\n")
    nu.write(f"  maskhalo_dyn          = {'T' if maskhalo_dyn else 'F':>6}\n")
    nu.write(f"  maskhalo_remap        = {'T' if maskhalo_remap else 'F':>6}\n")
    nu.write(f"  maskhalo_bound        = {'T' if maskhalo_bound else 'F':>6}\n")
    nu.write(f"  add_mpi_barriers      = {'T' if add_mpi_barriers else 'F':>6}\n")
    nu.write(f"  debug_blocks          = {'T' if debug_blocks else 'F':>6}\n")
    nu.write(f"  block_size_x,_y       = {block_size_x:6d}{block_size_y:6d}\n")
    nu.write(f"  max_blocks            = {max_blocks:6d}\n")
    nu.write(f"  Number of ghost cells = {int(ice_domain_size.nghost):6d}\n\n")
    nu.flush()
