"""Defines the global domain size and number of categories and layers.
Code originally based on domain_size.F in POP.

author for JAX version: Xinyuan Zhu
XZ: converted from F90 to Python

2006: Converted to free source form (F90) by Elizabeth Hunke
      Removed hardwired sizes (NX...can now be set in compile scripts)
2004: Block structure and snow parameters added by William Lipscomb
      Renamed (used to be ice_model_size)
author Elizabeth C. Hunke, LANL

Block decomposition is not ported, so the horizontal arrays are the
whole domain plus ``nghost`` cells on each side. ``nx_global`` and
``ny_global`` come from ``domain_nml`` through ``set_horizontal``.
Category counts come from ``sync_vertical_sizes`` after ``input_data``.
``nghost`` is fixed at 1 in ``ice_blocks``, not by the namelist.
"""

from cicecore.shared.ice_kinds_mod import int_kind

# namelist
nx_global = int_kind(-1)  # i-axis size
ny_global = int_kind(-1)  # j-axis size
nghost = int_kind(1)      # number of ghost cells around each block

ncat = int_kind(0)    # number of thickness categories
nfsd = int_kind(0)    # number of floe size categories
nilyr = int_kind(0)   # number of ice layers per category
nslyr = int_kind(0)   # number of snow layers per category
nblyr = int_kind(0)   # number of bio/brine layers per category
ntrcr = int_kind(0)   # tracers in use; set by sync_vertical_sizes

max_nstrm = int_kind(6)  # max number of history output streams
                         # 5 streams plus 1 one-time grid output file


def set_horizontal(nx, ny):
    """Store the domain size from ``domain_nml``."""
    global nx_global, ny_global
    if int(nx) < 1 or int(ny) < 1:
        raise ValueError(
            f"(set_horizontal) nx_global={int(nx)} ny_global={int(ny)}; both must be >= 1"
        )
    nx_global = int_kind(nx)
    ny_global = int_kind(ny)


def horizontal_shape():
    """Shape of one horizontal field, including ghost cells. (i, j)."""
    ghost = int(nghost)
    return (int(nx_global) + 2 * ghost, int(ny_global) + 2 * ghost)


def physical_bounds():
    """0-based inclusive i/j limits of the physical domain.

    Fortran ``ilo = nghost + 1`` is ``ilo`` here. A local index ``i`` holds
    global index ``i - nghost`` before the east-west wrap.
    """
    ghost = int(nghost)
    ilo = ghost
    ihi = ghost + int(nx_global) - 1
    jlo = ghost
    jhi = ghost + int(ny_global) - 1
    return ilo, ihi, jlo, jhi


def sync_vertical_sizes(ncat_in, nilyr_in, nslyr_in, nblyr_in, nfsd_in, ntrcr_in):
    """Copy the namelist category counts and the tracer count."""
    global ncat, nilyr, nslyr, nblyr, nfsd, ntrcr
    ncat = int_kind(ncat_in)
    nilyr = int_kind(nilyr_in)
    nslyr = int_kind(nslyr_in)
    nblyr = int_kind(nblyr_in)
    nfsd = int_kind(nfsd_in)
    ntrcr = int_kind(ntrcr_in)
