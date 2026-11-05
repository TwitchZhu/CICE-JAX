"""Halo checks from cicecore/drivers/unittest/halochk/halochk.F90.

author for JAX version: Xinyuan Zhu
XZ: converted from F90 to Python

One block for the whole grid. Indices only: no grid file and no
CICE_Initialize. tx1 here is 360 x 300, the size named in the port
task. cice_decomp.csh uses 360 x 240 for the Fortran suite.
field_loc_unknown and field_type_unknown are not tested; Fortran
comments them out because they abort.
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cicecore.cicedyn.infrastructure.comm.serial import ice_boundary
from cicecore.cicedyn.infrastructure.comm.serial.ice_boundary import (
    global_indices,
    ice_HaloUpdate_stress,
    ice_halo_update,
    set_boundary_types,
)
from cicecore.shared.ice_constants import (
    field_loc_Eface,
    field_loc_NEcorner,
    field_loc_Nface,
    field_loc_center,
    field_loc_noupdate,
    field_type_angle,
    field_type_noupdate,
    field_type_scalar,
    field_type_vector,
)
from cicecore.shared import ice_domain_size
from cicecore.shared.ice_domain_size import set_horizontal

# fillval = -88888, dhalofillval = -999, nz1 = 3, nz2 = 4
FILL = -88888.0
DHALO = -999.0
NZ1 = 3
NZ2 = 4

# gbox80 and gx3 default is ew cyclic, ns open (ice_in, no set_nml boundary).
# cyclic / open come from set_nml.cyclic and set_nml.open.
# tx1 is set_nml.tx1 (ns tripole). tripolet is set_nml.tripolet.
CASES = (
    ("gbox80", 80, 80, "cyclic", "open"),
    ("gbox80", 80, 80, "cyclic", "cyclic"),
    ("gbox80", 80, 80, "open", "open"),
    ("gx3", 100, 116, "cyclic", "open"),
    ("gx3", 100, 116, "cyclic", "cyclic"),
    ("gx3", 100, 116, "open", "open"),
    ("tx1", 360, 300, "cyclic", "tripole"),
    ("tx1", 360, 300, "cyclic", "tripoleT"),
)

FIELDS = (
    ("2DR8", 2, np.float64),
    ("3DR8", 3, np.float64),
    ("4DR8", 4, np.float64),
    ("2DR4", 2, np.float32),
    ("3DR4", 3, np.float32),
    ("4DR4", 4, np.float32),
    ("2DI4", 2, np.int32),
    ("3DI4", 3, np.int32),
    ("4DI4", 4, np.int32),
    ("2DL1", 2, bool),
    ("STRESS", 2, np.float64),
)
LOCS = (
    ("center", field_loc_center),
    ("NEcorn", field_loc_NEcorner),
    ("Nface", field_loc_Nface),
    ("Eface", field_loc_Eface),
    ("none", field_loc_noupdate),
)
TYPES = (
    ("scalar", field_type_scalar),
    ("vector", field_type_vector),
    ("angle", field_type_angle),
    ("none", field_type_noupdate),
)
FILLS = (("fill", True), ("nofill", False))


def _bounds():
    # Read the module each time. A `from ice_domain_size import nx_global`
    # keeps the value from import, which is -1 until set_horizontal.
    ghost = int(ice_domain_size.nghost)
    nx = int(ice_domain_size.nx_global)
    ny = int(ice_domain_size.ny_global)
    ilo = ghost
    ihi = ghost + nx - 1
    jlo = ghost
    jhi = ghost + ny - 1
    return ilo, ihi, jlo, jhi


def _shape(ndim):
    ghost = int(ice_domain_size.nghost)
    shape = (
        int(ice_domain_size.nx_global) + 2 * ghost,
        int(ice_domain_size.ny_global) + 2 * ghost,
    )
    if ndim == 3:
        return shape + (NZ1,)
    if ndim == 4:
        return shape + (NZ1, NZ2)
    return shape


def _plane(array, k1, k2):
    if array.ndim == 2:
        return array
    if array.ndim == 3:
        return array[:, :, k1 - 1]
    return array[:, :, k1 - 1, k2 - 1]


def _ks(ndim):
    if ndim == 2:
        return (1,), (1,)
    if ndim == 3:
        return tuple(range(1, NZ1 + 1)), (1,)
    return tuple(range(1, NZ1 + 1)), tuple(range(1, NZ2 + 1))


def _put(array, ig, jg, which, ilo, ihi, jlo, jhi):
    """Physical domain gets the global index. Ghosts stay at fillval.

    cidata = i_glob + k1*1000 + k2*10000. Logical ghosts are false on the
    i array and true on the j array. Elsewhere logical is mod(nint, 2) == 1.
    """
    k1s, k2s = _ks(array.ndim)
    index = ig if which == "i" else jg
    for k1 in k1s:
        for k2 in k2s:
            off = k1 * 1000 + k2 * 10000
            plane = _plane(array, k1, k2)
            if which == "i":
                base = index[ilo:ihi + 1, None] + off
            else:
                base = index[None, jlo:jhi + 1] + off
            if array.dtype == bool:
                plane[ilo:ihi + 1, jlo:jhi + 1] = np.mod(np.rint(base), 2) == 1
            elif np.issubdtype(array.dtype, np.integer):
                plane[ilo:ihi + 1, jlo:jhi + 1] = np.rint(base).astype(array.dtype)
            else:
                plane[ilo:ihi + 1, jlo:jhi + 1] = base


def _blank(dtype, ndim, which):
    shape = _shape(ndim)
    if dtype is bool:
        # where (array == fillval) i is false and j is true
        value = False if which == "i" else True
        return np.full(shape, value, dtype=bool)
    if np.issubdtype(dtype, np.integer):
        return np.full(shape, int(FILL), dtype=dtype)
    return np.full(shape, FILL, dtype=dtype)


def _expected(ig, jg, k1, k2, loc, typ, halofill, stress, pre, logical, which):
    """Expected halo from the halochk formulas.

    Baseline is i_glob / j_glob everywhere, including ghosts. Non-cyclic
    outer edges become the fill value (or 0 on a tripole edge when the
    update is not given a fill). The tripole north edge is folded by hand.
    A stress update only changes the tripole zipper, so it starts from the
    array copied before the call.
    """
    ilo, ihi, jlo, jhi = _bounds()
    nx = int(ice_domain_size.nx_global)
    ny = int(ice_domain_size.ny_global)
    off = k1 * 1000.0 + k2 * 10000.0
    if which == "i":
        bas = ig.astype(np.float64)[:, None] + off
    else:
        bas = jg.astype(np.float64)[None, :] + off
    bas = np.broadcast_to(bas, (ig.size, jg.size)).copy()
    ew = ice_boundary.ew_boundary_type
    ns = ice_boundary.ns_boundary_type
    fillexpected = DHALO if halofill else FILL
    if loc == field_loc_noupdate or typ == field_type_noupdate:
        exp = bas.copy()
        ghost = np.ones(exp.shape, dtype=bool)
        ghost[ilo:ihi + 1, jlo:jhi + 1] = False
        exp[ghost] = FILL
        return exp
    if stress:
        exp = pre.copy()
    else:
        exp = bas.copy()
        # if ew_boundary_type is not cyclic we expect just fill values on outer boundary
        if ew != "cyclic":
            if int(ig[ilo]) == 1:
                exp[:ilo, :] = fillexpected
            if int(ig[ihi]) == nx:
                exp[ihi + 1:, :] = fillexpected
        # tripole north is haloed below; south is fill or the internal default c0
        if ns != "cyclic":
            south_value = 0.0 if ns in ("tripole", "tripoleT") and not halofill else fillexpected
            if int(jg[jlo]) == 1:
                exp[:, :jlo] = south_value
            if int(jg[jhi]) == ny:
                exp[:, jhi + 1:] = south_value
    if int(jg[jhi]) == ny and ns in ("tripole", "tripoleT"):
        _tripole(exp, bas, ig, jg, loc, typ, stress, logical, which, k1, k2, ns)
    return exp


def _tripole(exp, bas, ig, jg, loc, typ, stress, logical, which, k1, k2, ns):
    _ilo, _ihi, _jlo, jhi = _bounds()
    nx = int(ice_domain_size.nx_global)
    # flip sign for vector/angle except for logical halo updates
    rsign = 1.0
    if typ in (field_type_vector, field_type_angle) and not logical:
        rsign = -1.0
    je = jhi + 1  # 1-based
    if ns == "tripole":
        js = [
            j
            for j in range(jg.size)
            if (j + 1) > je or (j + 1 == je and loc in (field_loc_Nface, field_loc_NEcorner))
        ]
    else:
        js = [j for j in range(jg.size) if (j + 1) >= je]
    off = k1 * 1000.0 + k2 * 10000.0
    for j in js:
        j_f = j + 1
        for i in range(ig.size):
            i_glob = int(ig[i])
            average = False
            pole = False
            ioffset = 0
            joffset = 0
            if ns == "tripole":
                # standard center tripole u-fold
                itrip = nx - i_glob + 1
                jtrip = max(je - (j_f - je) + 1, je)
                if loc in (field_loc_NEcorner, field_loc_Nface):
                    joffset = -1
                    if j_f == je:
                        average = True
                if loc in (field_loc_NEcorner, field_loc_Eface):
                    ioffset = -1
                    # CICE does not average the ends of the tripole seam
                    if j_f == je and i_glob in (nx // 2, nx):
                        pole = True
            else:
                # standard center tripoleT t-fold
                itrip = nx - i_glob + 2
                jtrip = je - (j_f - je)
                if loc in (field_loc_NEcorner, field_loc_Eface):
                    ioffset = -1
                if loc in (field_loc_NEcorner, field_loc_Nface):
                    joffset = -1
                if loc in (field_loc_center, field_loc_Eface) and j_f == je:
                    average = True
                if loc == field_loc_center and j_f == je and i_glob in (1, nx // 2 + 1):
                    pole = True
            itrip = (itrip + ioffset + nx - 1) % nx + 1
            jtrip = jtrip + joffset
            rival = itrip + off
            rjval = float(jg[jtrip - 1]) + off
            folded = rival if which == "i" else rjval
            base = bas[i, j]
            if stress:
                # ends of the seam are not averaged; sign follows darray*str
                if pole:
                    exp[i, j] = -rsign * base
                else:
                    exp[i, j] = -rsign * folded
            elif pole:
                exp[i, j] = rsign * base
            elif average:
                if logical:
                    # logical math doesn't work this way, force to correct answer
                    pass
                else:
                    exp[i, j] = 0.5 * (base + rsign * folded)
            else:
                exp[i, j] = rsign * folded


def _nint(values):
    # Fortran nint: ties round away from zero. numpy rint ties to even.
    values = np.asarray(values, dtype=np.float64)
    return np.sign(values) * np.floor(np.abs(values) + 0.5)


def _finish(exp, got, logical, integer, which):
    if integer:
        exp = _nint(exp)
    if logical:
        # logical ghosts that stayed at a fill become 0 (i) or 1 (j)
        fill = (exp == DHALO) | (exp == FILL)
        out = np.mod(_nint(exp), 2)
        out[fill] = 0.0 if which == "i" else 1.0
        exp = out
    return np.array_equal(np.asarray(got, dtype=np.float64), np.asarray(exp, dtype=np.float64))


def _run_case(grid, nx, ny, ew, ns):
    set_horizontal(nx, ny)
    set_boundary_types(ew, ns)
    ig, jg = global_indices()
    ilo, ihi, jlo, jhi = _bounds()
    lines = []
    for halofld, ndim, dtype in FIELDS:
        for loc_name, loc in LOCS:
            for typ_name, typ in TYPES:
                for fill_name, halofill in FILLS:
                    name = f"{grid} {halofld} {loc_name} {typ_name} {fill_name} {ew} {ns}"
                    ok = _one(ig, jg, ilo, ihi, jlo, jhi, halofld, ndim, dtype, loc, typ, halofill)
                    lines.append((ok, name))
    return lines


def _one(ig, jg, ilo, ihi, jlo, jhi, halofld, ndim, dtype, loc, typ, halofill):
    logical = halofld == "2DL1"
    stress = halofld == "STRESS"
    integer = "I4" in halofld
    array_i = _blank(dtype, ndim, "i")
    array_j = _blank(dtype, ndim, "j")
    _put(array_i, ig, jg, "i", ilo, ihi, jlo, jhi)
    _put(array_j, ig, jg, "j", ilo, ihi, jlo, jhi)
    pre_i = np.array(array_i, dtype=np.float64, copy=True)
    pre_j = np.array(array_j, dtype=np.float64, copy=True)
    fill_i = DHALO if halofill else None
    fill_j = DHALO if halofill else None
    if logical and halofill:
        fill_i, fill_j = 0, 1
    if stress:
        # flip sign for testing; the stress update reads the second array
        src_i = -np.array(array_i, dtype=np.float64, copy=True)
        src_j = -np.array(array_j, dtype=np.float64, copy=True)
    try:
        if stress:
            out_i = ice_HaloUpdate_stress(array_i, src_i, loc, typ, fill_value=fill_i)
            out_j = ice_HaloUpdate_stress(array_j, src_j, loc, typ, fill_value=fill_j)
        else:
            out_i = ice_halo_update(array_i, loc, typ, fill_value=fill_i)
            out_j = ice_halo_update(array_j, loc, typ, fill_value=fill_j)
    except Exception:
        return False
    if out_i is None or out_j is None:
        return False
    k1s, k2s = _ks(ndim)
    for k1 in k1s:
        for k2 in k2s:
            got_i = _plane(out_i, k1, k2)
            got_j = _plane(out_j, k1, k2)
            if logical:
                got_i = np.where(got_i, 1.0, 0.0)
                got_j = np.where(got_j, 1.0, 0.0)
            pre_i_k = _plane(pre_i, k1, k2) if pre_i.ndim > 2 else pre_i
            pre_j_k = _plane(pre_j, k1, k2) if pre_j.ndim > 2 else pre_j
            exp_i = _expected(ig, jg, k1, k2, loc, typ, halofill, stress, pre_i_k, logical, "i")
            exp_j = _expected(ig, jg, k1, k2, loc, typ, halofill, stress, pre_j_k, logical, "j")
            if logical:
                # force averaged logical points to the updated value, as Fortran does
                _force_logical(exp_i, got_i, ig, jg, loc)
                _force_logical(exp_j, got_j, ig, jg, loc)
            if not _finish(exp_i, got_i, logical, integer, "i"):
                return False
            if not _finish(exp_j, got_j, logical, integer, "j"):
                return False
    return True


def _force_logical(exp, got, ig, jg, loc):
    """tripole_average on a logical field copies the updated value.

    Ends of the seam are tripole_pole and stay on the folded formula.
    """
    ns = ice_boundary.ns_boundary_type
    if ns not in ("tripole", "tripoleT"):
        return
    _ilo, _ihi, _jlo, jhi = _bounds()
    if int(jg[jhi]) != int(ice_domain_size.ny_global):
        return
    nx = int(ice_domain_size.nx_global)
    row = np.asarray(got, dtype=np.float64)[:, jhi]
    cols = np.ones(ig.size, dtype=bool)
    if ns == "tripole":
        if loc not in (field_loc_Nface, field_loc_NEcorner):
            return
        if loc == field_loc_NEcorner:
            cols[(ig == nx // 2) | (ig == nx)] = False
    else:
        if loc not in (field_loc_center, field_loc_Eface):
            return
        if loc == field_loc_center:
            cols[(ig == 1) | (ig == nx // 2 + 1)] = False
    exp[cols, jhi] = row[cols]


def main():
    npass = 0
    nfail = 0
    for grid, nx, ny, ew, ns in CASES:
        for ok, name in _run_case(grid, nx, ny, ew, ns):
            if ok:
                npass += 1
                print(f"PASS {name}")
            else:
                nfail += 1
                print(f"FAIL {name}")
    print(f"total pass = {npass}")
    print(f"total fail = {nfail}")
    return 1 if nfail else 0


if __name__ == "__main__":
    sys.exit(main())
