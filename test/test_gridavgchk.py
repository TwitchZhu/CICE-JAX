"""Grid-average checks from cicecore/drivers/unittest/gridavgchk/gridavgchk.F90.

author for JAX version: Xinyuan Zhu
XZ: converted from F90 to Python

One block. Grids are gx3 (reads the case ice_in) and gbox128 (rectangular,
kmt_type default, no data file). gx1 waits for its data. Land-block
elimination is not in this port, so the F group is included: 16 A + 16 S
+ 16 F + 4 NE/EN, each on a constant, an i/j index, and an area field.

The Fortran area test also calls the user-weight form of grid_average_X2Y.
That form is not ported; the area item checks output / destination area.
"""

import io
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cicecore.cicedyn.general import ice_init
from cicecore.cicedyn.infrastructure import ice_domain
from cicecore.cicedyn.infrastructure import ice_grid
from cicecore.cicedyn.infrastructure.comm.serial.ice_boundary import (
    global_indices,
    set_boundary_types,
)
from cicecore.shared import ice_domain_size
from cicecore.shared import ice_fileunits
from cicecore.shared import ice_init_column
from cicecore.shared.ice_domain_size import physical_bounds
from cicecore.shared.ice_kinds_mod import dbl_kind
from icepack.columnphysics import icepack_intfc

# real(dbl_kind), parameter :: testconst = 100
TESTCONST = 100.0


def _names_and_tol(nx, ny):
    # tolerances from gridavgchk.F90. The smaller pair applies when both
    # dimensions exceed 200. Neither gx3 nor gbox128 does.
    big = nx > 200 and ny > 200
    names = []
    t_const = []
    t_ij = []
    t_area = []
    groups = (
        ("A", 1.0e-5, 0.10, 0.04, 0.03, 0.003),
        ("S", 1.0e-5, 0.51, 0.19, None, 0.06),
        ("F", 0.0065, 0.65, 0.04, 0.22, 0.004),
    )
    for kind, ctol, ij, area, ij_big, area_big in groups:
        for src in "TUNE":
            for dst in "TUNE":
                names.append(f"{src}2{dst}{kind}")
                t_const.append(ctol)
                t_ij.append(ij_big if big and ij_big is not None else ij)
                t_area.append(area_big if big else area)
    ne_ij = 0.26 if big else 0.51
    ne_area = 0.03 if big else 0.12
    for name in ("NE2TS", "EN2TS", "NE2US", "EN2US"):
        names.append(name)
        t_const.append(1.0e-5)
        t_ij.append(ne_ij)
        t_area.append(ne_area)
    return names, t_const, t_ij, t_area


def _mask(name):
    letter = name[-2]
    grids = {
        "T": ice_grid.tmask,
        "U": ice_grid.umask,
        "N": ice_grid.nmask,
        "E": ice_grid.emask,
    }
    return grids[letter]


def _deltas(name):
    dx = 0.0
    dy = 0.0
    if name[:3] in ("T2U", "T2E", "N2U", "N2E") or name[:4] in ("NE2U", "EN2U"):
        dx = 0.5
    elif name[:3] in ("U2T", "U2N", "E2T", "E2N"):
        dx = -0.5
    if name[:3] in ("T2U", "T2N", "E2U", "E2N"):
        dy = 0.5
    elif name[:3] in ("U2T", "U2E", "N2T", "N2E") or name[:4] in ("NE2T", "EN2T"):
        dy = -0.5
    return dx, dy


def _call(name, field, second, swap_en):
    kind = name[-1]
    out = np.zeros(field.shape, dtype=dbl_kind)
    if len(name) == 4:
        ice_grid.grid_average_X2Y(kind, field, name[0], out, name[2])
    elif name[:2] == "NE" or not swap_en:
        # area fills array1 / arrays for EN already, so it does not swap
        ice_grid.grid_average_X2Y_NE(
            kind, field, name[0], second, name[1], out, name[3]
        )
    else:
        # const and ij pass the E field first for an EN name
        ice_grid.grid_average_X2Y_NE(
            kind, second, name[0], field, name[1], out, name[3]
        )
    return out


def _active_bad(err, result, mask, tol):
    ilo, ihi, jlo, jhi = physical_bounds()
    sl = (slice(ilo, ihi + 1), slice(jlo, jhi + 1))
    # if array2 is c0, there are no valid surrounding points and ignore it
    use = np.asarray(mask[sl]) & (result[sl] != 0.0)
    return bool(np.any(use & (np.abs(err[sl]) > tol)))


def _const_items(names, tols):
    shape = ice_grid.hm.shape
    field = np.full(shape, TESTCONST, dtype=dbl_kind)
    second = np.full(shape, TESTCONST, dtype=dbl_kind)
    rows = []
    for name, tol in zip(names, tols):
        result = _call(name, field, second, swap_en=True)
        err = (result - TESTCONST) / TESTCONST
        err = np.where(result == 0.0, 0.0, err)
        rows.append((not _active_bad(err, result, _mask(name), tol), f"{name} const"))
    return rows


def _ij_fields():
    ig, jg = global_indices()
    ilo, ihi, jlo, jhi = physical_bounds()
    shape = ice_grid.hm.shape
    ax = np.full(shape, -999.0, dtype=dbl_kind)
    ay = np.full(shape, -999.0, dtype=dbl_kind)
    ax[ilo:ihi + 1, jlo:jhi + 1] = ig[ilo:ihi + 1, None]
    ay[ilo:ihi + 1, jlo:jhi + 1] = jg[None, jlo:jhi + 1]
    # Fill in ghost cells with locally appropriate value. Skip corners, then
    # set the east and west ghosts, which also fills the corners.
    ax[ilo:ihi + 1, jlo - 1] = ax[ilo:ihi + 1, jlo]
    ay[ilo:ihi + 1, jlo - 1] = ay[ilo:ihi + 1, jlo] - 1.0
    ax[ilo:ihi + 1, jhi + 1] = ax[ilo:ihi + 1, jhi]
    ay[ilo:ihi + 1, jhi + 1] = ay[ilo:ihi + 1, jhi] + 1.0
    ax[ilo - 1, :] = ax[ilo, :] - 1.0
    ay[ilo - 1, :] = ay[ilo, :]
    ax[ihi + 1, :] = ax[ihi, :] + 1.0
    ay[ihi + 1, :] = ay[ihi, :]
    return ax, ay


def _ij_items(names, tols):
    ax, ay = _ij_fields()
    sx = ax + 0.5
    sy = ay - 0.5
    rows = []
    for name, tol in zip(names, tols):
        dx, dy = _deltas(name)
        rx = _call(name, ax, sx, swap_en=True)
        ry = _call(name, ay, sy, swap_en=True)
        errx = np.where(rx == 0.0, 0.0, rx - ax - dx)
        erry = np.where(ry == 0.0, 0.0, ry - ay - dy)
        bad = _active_bad(errx, rx, _mask(name), tol) or _active_bad(erry, ry, _mask(name), tol)
        rows.append((not bad, f"{name} ijind"))
    return rows


def _areas(name):
    if name[:3] == "NE2":
        return ice_grid.narea, ice_grid.earea
    if name[:3] == "EN2":
        return ice_grid.earea, ice_grid.narea
    src = {"T": ice_grid.tarea, "U": ice_grid.uarea, "N": ice_grid.narea, "E": ice_grid.earea}
    return src[name[0]], None


def _dest_area(name):
    key = name[1:3] if len(name) == 4 else name[2:4]
    return {"2T": ice_grid.tarea, "2U": ice_grid.uarea, "2E": ice_grid.earea, "2N": ice_grid.narea}[key]


def _area_items(names, tols):
    rows = []
    for name, tol in zip(names, tols):
        field, second = _areas(name)
        # EN area arguments are already E then N; do not swap again
        result = _call(name, np.array(field, copy=True), None if second is None else np.array(second, copy=True), swap_en=False)
        exact = _dest_area(name)
        with np.errstate(divide="ignore", invalid="ignore"):
            err = result / exact - 1.0
        err = np.where(result == 0.0, 0.0, err)
        rows.append((not _active_bad(err, result, _mask(name), tol), f"{name} area"))
    return rows


def _build_gbox128():
    ice_fileunits.nu_diag = io.StringIO()
    icepack_intfc.icepack_configure()
    ice_init.grid_type = "rectangular"
    ice_init.kmt_type = "default"
    ice_init.nx_global = 128
    ice_init.ny_global = 128
    ice_init.dxrect = 3.0e5
    ice_init.dyrect = 3.0e5
    ice_init.scale_dxdy = False
    ice_domain_size.set_horizontal(128, 128)
    set_boundary_types("cyclic", "open")
    ice_grid.init_grid1()
    ice_grid.alloc_grid()
    ice_grid.init_grid2()


def _build_gx3():
    os.chdir(ROOT / "cases" / "gx3_test")
    ice_fileunits.nu_diag = io.StringIO()
    icepack_intfc.icepack_configure()
    ice_init.input_data("ice_in")
    ice_fileunits.nu_diag = io.StringIO()
    ice_init_column.input_zbgc("ice_in")
    ice_init.count_tracers()
    ice_domain.init_domain_blocks("ice_in")
    ice_grid.init_grid1()
    ice_grid.alloc_grid()
    ice_grid.init_grid2()


def _run_grid(label, nx, ny):
    names, t_const, t_ij, t_area = _names_and_tol(nx, ny)
    rows = []
    rows.extend(_const_items(names, t_const))
    rows.extend(_ij_items(names, t_ij))
    rows.extend(_area_items(names, t_area))
    return [(ok, f"{label} {text}") for ok, text in rows]


def main():
    npass = 0
    nfail = 0
    builds = (
        ("gbox128", 128, 128, _build_gbox128),
        ("gx3", 100, 116, _build_gx3),
    )
    saved = os.getcwd()
    try:
        for label, nx, ny, build in builds:
            try:
                build()
            except Exception as exc:
                names, *_rest = _names_and_tol(nx, ny)
                for name in names:
                    for kind in ("const", "ijind", "area"):
                        nfail += 1
                        print(f"FAIL {label} {name} {kind}")
                print(f"{label} grid init failed: {exc}")
                continue
            for ok, text in _run_grid(label, nx, ny):
                if ok:
                    npass += 1
                    print(f"PASS {text}")
                else:
                    nfail += 1
                    print(f"FAIL {text}")
    finally:
        os.chdir(saved)
    print(f"total pass = {npass}")
    print(f"total fail = {nfail}")
    return 1 if nfail else 0


if __name__ == "__main__":
    sys.exit(main())
