"""Write and read a history stream in netCDF.

author for JAX version: Xinyuan Zhu
XZ: converted from F90 to Python

USE_NETCDF is the only path. One process writes the whole grid, so
there is no gather. The written arrays are the physical domain:
axis 0 is i (ni), axis 1 is j (nj).
"""

import os
import re
from time import perf_counter

import numpy as np
import jax
from netCDF4 import Dataset

from cicecore.cicedyn.analysis import ice_history_shared
from cicecore.cicedyn.general import ice_flux
from cicecore.cicedyn.general import ice_init
from cicecore.cicedyn.infrastructure import ice_grid
from cicecore.shared import ice_arrays_column
from cicecore.cicedyn.infrastructure.comm.serial import ice_boundary
from cicecore.cicedyn.infrastructure.comm.serial import ice_timers
from cicecore.shared import ice_calendar
from cicecore.shared import ice_domain_size
from cicecore.shared import ice_fileunits
from cicecore.shared.ice_constants import field_loc_noupdate, field_type_noupdate, spval_dbl
from cicecore.shared.ice_kinds_mod import dbl_kind
from icepack.columnphysics.icepack_parameters import icepack_query_parameters

subname = "(ice_write_hist)"


def ice_write_hist(ns, histrest=False, timing=None):
    """Write stream ns (0-based), or its unscaled restart accumulators."""
    fields = [field for field in ice_history_shared.avail_hist_fields if field.stream == ns]
    if not fields and str(ice_calendar.histfreq[ns]).strip() != "g":
        return
    write_started = perf_counter()
    measured_before = (sum(timing["seconds"].get(name, 0.0) for name in (
        "history.write.prepare_gpu", "history.write.d2h")) if timing is not None else 0.0)
    secday = icepack_query_parameters("secday")["secday"]
    rad_to_deg = _history_host_array(
        icepack_query_parameters("rad_to_deg")["rad_to_deg"], np.float64, timing).item()
    ncfile = ice_history_shared.construct_filename(
        "nc", ns, option="histrest" if histrest else "history",
    )
    if histrest:
        os.makedirs(os.path.dirname(ncfile) or ".", exist_ok=True)
    fmt = ice_history_shared.history_format
    if fmt == "cdf1":
        format_code = "NETCDF3_CLASSIC"
    elif fmt == "cdf2":
        format_code = "NETCDF3_64BIT_OFFSET"
    elif fmt == "cdf5":
        format_code = "NETCDF3_64BIT_DATA"
    elif fmt == "hdf5":
        format_code = "NETCDF4"
    else:
        raise ValueError(
            subname + " ERROR: history_format not allowed for " + fmt
        )
    precision = np.float64 if histrest or ice_history_shared.history_precision == 8 else np.float32
    nx = int(ice_domain_size.nx_global)
    ny = int(ice_domain_size.ny_global)
    ilo, ihi, jlo, jhi = ice_domain_size.physical_bounds()
    averaged = ice_history_shared.hist_avg[ns] and not ice_calendar.write_ic
    freq = str(ice_calendar.histfreq[ns]).strip()
    grid_flags = {}
    with open(ice_init.nml_filename, encoding="utf-8") as handle:
        for line in handle:
            match = re.match(r"\s*f_(\w+)\s*=\s*\.(true|false)\.", line, re.I)
            if match:
                grid_flags[match.group(1).lower()] = match.group(2).lower() == "true"

    with Dataset(ncfile, "w", format=format_code) as nc:
        if averaged and freq != "g":
            nc.createDimension("nbnd", 2)
        nc.createDimension("ni", nx)
        nc.createDimension("nj", ny)
        nc.createDimension("time", None)

        # do not write time axis on grid output file
        if freq != "g":
            idate0 = int(ice_calendar.idate0)
            cdate = f"{idate0:08d}"
            cal_units = (
                "days since "
                f"{cdate[0:4]}-{cdate[4:6]}-{cdate[6:8]} "
                f"{int(ice_calendar.hh_init):02d}:"
                f"{int(ice_calendar.mm_init):02d}:"
                f"{int(ice_calendar.ss_init):02d}"
            )
            days_per_year = int(ice_calendar.days_per_year)
            if days_per_year == 360:
                cal_att = "360_day"
            elif days_per_year == 365 and not ice_calendar.use_leap_years:
                cal_att = "noleap"
            elif ice_calendar.use_leap_years:
                cal_att = "proleptic_gregorian"
            else:
                raise ValueError(subname + " ERROR: invalid calendar settings")
            time = nc.createVariable("time", np.float64, ("time",))
            time.units = cal_units
            time.calendar = cal_att
            time.long_name = "model time"
            timedbl = float(ice_calendar.timesecs) / float(secday)
            if ice_history_shared.hist_avg[ns]:
                if ice_init.hist_time_axis == "begin":
                    timedbl = float(ice_history_shared.time_beg[ns])
                elif ice_init.hist_time_axis == "middle":
                    timedbl = 0.5 * float(ice_history_shared.time_beg[ns] + ice_history_shared.time_end[ns])
            time[0] = timedbl
            if averaged:
                bounds = nc.createVariable("time_bounds", np.float64, ("time", "nbnd"))
                bounds.units = cal_units
                bounds.calendar = cal_att
                bounds[0, 0] = float(ice_history_shared.time_beg[ns])
                bounds[0, 1] = float(ice_history_shared.time_end[ns])
                time.bounds = "time_bounds"

        for name, long_name, units, values in (
            ("TLON", "T grid center longitude", "degrees_east", ice_grid.TLON),
            ("TLAT", "T grid center latitude", "degrees_north", ice_grid.TLAT),
            ("ULON", "U grid center longitude", "degrees_east", ice_grid.ULON),
            ("ULAT", "U grid center latitude", "degrees_north", ice_grid.ULAT),
            ("NLON", "N grid center longitude", "degrees_east", ice_grid.NLON),
            ("NLAT", "N grid center latitude", "degrees_north", ice_grid.NLAT),
            ("ELON", "E grid center longitude", "degrees_east", ice_grid.ELON),
            ("ELAT", "E grid center latitude", "degrees_north", ice_grid.ELAT),
        ):
            if grid_flags.get(name.lower(), True) or freq == "g":
                _coord(nc, name, long_name, units, values, ilo, ihi, jlo, jhi, rad_to_deg, precision, timing)

        for name, long_name, units, coordinates, values in (
            ("tmask", "mask of T grid cells, 0 = land, 1 = ocean", "1", "TLON TLAT", ice_grid.hm),
            ("umask", "mask of U grid cells, 0 = land, 1 = ocean", "1", "ULON ULAT", ice_grid.uvm),
            ("nmask", "mask of N grid cells, 0 = land, 1 = ocean", "1", "NLON NLAT", ice_grid.npm),
            ("emask", "mask of E grid cells, 0 = land, 1 = ocean", "1", "ELON ELAT", ice_grid.epm),
            ("tarea", "area of T grid cells", "m^2", "TLON TLAT", ice_grid.tarea),
            ("uarea", "area of U grid cells", "m^2", "ULON ULAT", ice_grid.uarea),
            ("narea", "area of N grid cells", "m^2", "NLON NLAT", ice_grid.narea),
            ("earea", "area of E grid cells", "m^2", "ELON ELAT", ice_grid.earea),
            ("blkmask", "block id of T grid cells, mytask + iblk/100", "1", "TLON TLAT", ice_grid.bm),
            ("dxt", "T cell width through middle", "m", "TLON TLAT", ice_grid.dxT),
            ("dyt", "T cell height through middle", "m", "TLON TLAT", ice_grid.dyT),
            ("dxu", "U cell width through middle", "m", "ULON ULAT", ice_grid.dxU),
            ("dyu", "U cell height through middle", "m", "ULON ULAT", ice_grid.dyU),
            ("dxn", "N cell width through middle", "m", "NLON NLAT", ice_grid.dxN),
            ("dyn", "N cell height through middle", "m", "NLON NLAT", ice_grid.dyN),
            ("dxe", "E cell width through middle", "m", "ELON ELAT", ice_grid.dxE),
            ("dye", "E cell height through middle", "m", "ELON ELAT", ice_grid.dyE),
            ("HTN", "T cell width on North side", "m", "TLON TLAT", ice_grid.HTN),
            ("HTE", "T cell width on East side", "m", "TLON TLAT", ice_grid.HTE),
            ("ANGLE", "angle grid makes with latitude line on U grid", "radians", "ULON ULAT", ice_grid.ANGLE),
            ("ANGLET", "angle grid makes with latitude line on T grid", "radians", "TLON TLAT", ice_grid.ANGLET),
        ):
            if grid_flags.get(name.lower(), True) or freq == "g":
                var = nc.createVariable(name, precision, ("nj", "ni"))
                var.long_name = long_name
                var.units = units
                var.coordinates = coordinates
                var[:, :] = _history_host_array(
                    values[ilo:ihi + 1, jlo:jhi + 1], precision, timing).T

        # Extra dimensions used by fields or enabled grid coordinates.
        axis_dim = {
            "ncat": "nc",
            "nzilyr": "nkice",
            "nzslyr": "nksnow",
            "nzblyr": "nkbio",
            "nzalyr": "nkaer",
            "nfsd": "nf",
        }
        needed = []
        for field in fields:
            for axis in ice_history_shared._RANKS[field.rank][2]:
                if axis not in needed:
                    needed.append(axis)
        coord_names = {
            "ncat": "NCAT", "nzilyr": "VGRDi", "nzslyr": "VGRDs",
            "nzblyr": "VGRDb", "nzalyr": "VGRDa", "nfsd": "NFSD",
        }
        for axis, coord_name in coord_names.items():
            if (grid_flags.get(coord_name.lower(), axis != "nfsd") or freq == "g") and axis not in needed:
                needed.append(axis)
        for axis in needed:
            length = {
                "ncat": ice_history_shared.ncat_hist,
                "nzilyr": ice_history_shared.nzilyr,
                "nzslyr": ice_history_shared.nzslyr,
                "nzblyr": ice_history_shared.nzblyr,
                "nzalyr": ice_history_shared.nzalyr,
                "nfsd": ice_history_shared.nfsd_hist,
            }[axis]
            nc.createDimension(axis_dim[axis], int(length))
            coord_name = coord_names[axis]
            if not grid_flags.get(coord_name.lower(), axis != "nfsd") and freq != "g":
                continue
            if axis == "ncat":
                var = nc.createVariable("NCAT", precision, ("nc",))
                var.long_name = "category maximum thickness"
                var.units = "m"
                var[:] = _history_host_array(ice_arrays_column.hin_max[1:int(length) + 1], precision, timing)
            elif axis == "nfsd":
                var = nc.createVariable("NFSD", precision, ("nf",))
                var.long_name = "category floe size (center)"
                var.units = "m"
                var[:] = _history_host_array(ice_arrays_column.floe_rad_c[:int(length)], precision, timing)
            else:
                var = nc.createVariable(coord_name, precision, (axis_dim[axis],))
                var.long_name = "vertical levels"
                var.units = "1"
                var[:] = np.arange(1, int(length) + 1, dtype=precision)

        if histrest:
            for name in ("time_beg", "avgct"):
                var = nc.createVariable(name, np.float64)
                var.assignValue(getattr(ice_history_shared, name)[ns])
            for name in ("albcnt", "snwcnt"):
                var = nc.createVariable(name + str(ns + 1), np.float64, ("time", "nj", "ni"))
                var[0, :, :] = _history_host_array(
                    getattr(ice_flux, name)[ilo:ihi + 1, jlo:jhi + 1, ns], np.float64, timing).T

        fill = np.dtype(precision).type(spval_dbl)
        for field in fields:
            dims = tuple(axis_dim[axis] for axis in ice_history_shared._RANKS[field.rank][2])
            var = nc.createVariable(
                field.vname,
                precision,
                ("time",) + dims[::-1] + ("nj", "ni"),
                fill_value=fill,
            )
            var.units = field.vunit
            var.long_name = field.vdesc
            var.comment = field.vcomment
            var.coordinates = f"{field.grid}LON {field.grid}LAT"
            var.missing_value = fill
            prepare_started = perf_counter()
            if histrest:
                array = getattr(ice_history_shared, ice_history_shared._RANKS[field.rank][1])
                # Preserve raw sums, including zero snapshot slots and land cells.
                values = array[ilo:ihi + 1, jlo:jhi + 1, ..., field.slot]
            else:
                values = ice_history_shared.scale_field(field, ice_history_shared.avgct[ns])
            ice_timers.record_stage(timing, "history.write.prepare_gpu", prepare_started,
                                    values, emit=False)
            # Normalize and crop on device; transfer only this output field.
            values = _history_host_array(values, precision, timing)
            order = tuple(range(values.ndim - 1, 1, -1)) + (1, 0)
            var[0, ...] = np.transpose(values, order)

    ice_fileunits.nu_diag.write(f"{subname} {ncfile}\n")
    if timing is not None:
        measured_after = sum(timing["seconds"].get(name, 0.0) for name in (
            "history.write.prepare_gpu", "history.write.d2h"))
        io_seconds = max(0.0, perf_counter() - write_started - (measured_after - measured_before))
        name = "history.write.io_metadata"
        timing["seconds"][name] = timing["seconds"].get(name, 0.0) + io_seconds
    ice_timers.record_stage(timing, "history.write", write_started)


def _history_host_array(values, precision, timing):
    started = perf_counter()
    ice_timers.record_stage(timing, "history.write.prepare_gpu", started, values, emit=False)
    started = perf_counter()
    host = np.asarray(jax.device_get(values), dtype=precision)
    ice_timers.record_stage(timing, "history.write.d2h", started, emit=False)
    return host


def _coord(nc, name, long_name, units, values, ilo, ihi, jlo, jhi, rad_to_deg, precision, timing=None):
    var = nc.createVariable(name, precision, ("nj", "ni"))
    var.long_name = long_name
    var.units = units
    piece = _history_host_array(values[ilo:ihi + 1, jlo:jhi + 1], dbl_kind, timing) * float(rad_to_deg)
    if name == "TLON":
        # CICE stores T longitudes in [-180, 180] but writes [0, 360].
        piece = piece + 360.0
        piece = np.where(piece > 360.0, piece - 360.0, piece)
        piece = np.where(piece < 0.0, piece + 360.0, piece)
    var[:, :] = np.asarray(piece, dtype=precision).T


def ice_read_hist():
    """Read history restarts so averaged fields do not start again at zero.

    A missing file is reported and skipped. A missing variable is left at
    its current value. Category, vertical, floe-size, and 4D accumulators
    are read when that rank was allocated.
    """
    subname = "(ice_read_hist)"
    readstr_t = " read ok:"
    readstr_f = " DID NOT READ:"
    nu = ice_fileunits.nu_diag
    nx = int(ice_domain_size.nx_global)
    ny = int(ice_domain_size.ny_global)

    ice_timers.ice_timer_start(ice_timers.timer_readwrite)  # reading/writing
    for ns in range(int(ice_calendar.nstreams)):
        if not ice_history_shared.hist_avg[ns]:
            continue

        cns = f"{ns + 1:01d}"
        ncfile = ice_history_shared.construct_filename("nc", ns, option="histrest")
        # add local directory path name to ncfile
        if not os.path.isfile(ncfile):
            nu.write(f" {subname} file not found {ncfile}\n")
            nu.flush()
            continue

        nu.write(f" {subname} reading file {ncfile}\n")
        nu.flush()
        with Dataset(ncfile, "r") as nc:
            # read variable data
            _read_scalar(nc, "time_beg", ice_history_shared.time_beg, ns, nu, subname, readstr_t, readstr_f)
            # Fortran prints the label time_beg on this avgct line as well.
            _read_scalar(nc, "avgct", ice_history_shared.avgct, ns, nu, subname, readstr_t, readstr_f)
            _read_counter(nc, "albcnt" + cns, ice_flux.albcnt[:, :, ns], nx, ny, nu, subname, readstr_t, readstr_f)
            _read_counter(nc, "snwcnt" + cns, ice_flux.snwcnt[:, :, ns], nx, ny, nu, subname, readstr_t, readstr_f)

            freq = str(ice_calendar.histfreq[ns]).strip()
            for field in ice_history_shared.avail_hist_fields:
                # vhistfreq == histfreq(ns)
                if str(field.vhistfreq).strip() != freq:
                    continue
                array_name = ice_history_shared._RANKS[field.rank][1]
                array = getattr(ice_history_shared, array_name)
                if array is None:
                    continue
                extra = array.ndim - 3
                dest = array[(slice(None), slice(None)) + (slice(None),) * extra + (field.slot,)]
                _read_field(nc, field.vname, dest, nx, ny, nu, subname, readstr_t, readstr_f)
        nu.write(f" {subname} Finished reading {ncfile}\n")
        nu.flush()
    ice_timers.ice_timer_stop(ice_timers.timer_readwrite)  # reading/writing


def _read_scalar(nc, name, dest, ns, nu, subname, readstr_t, readstr_f):
    readstr = readstr_f
    if name in nc.variables:
        dest[ns] = dbl_kind(np.array(nc.variables[name][:]).item())
        readstr = readstr_t
    nu.write(f"{subname}{readstr} {name}\n")
    nu.flush()


def _read_counter(nc, name, dest, nx, ny, nu, subname, readstr_t, readstr_f):
    readstr = readstr_f
    if name in nc.variables:
        slab = _as_ni_nj(np.array(nc.variables[name][:], dtype=dbl_kind), nx, ny)
        if slab is not None:
            ice_boundary.scatter_global(dest, slab, field_loc_noupdate, field_type_noupdate)
            readstr = readstr_t
    nu.write(f"{subname}{readstr} {name}\n")
    nu.flush()


def _read_field(nc, name, dest, nx, ny, nu, subname, readstr_t, readstr_f):
    readstr = readstr_f
    if name in nc.variables:
        axes = tuple(int(n) for n in dest.shape[2:])
        slab = _as_stored(np.array(nc.variables[name][:], dtype=dbl_kind), nx, ny, axes)
        if slab is not None:
            for index in np.ndindex(axes):
                ice_boundary.scatter_global(
                    dest[(slice(None), slice(None)) + index],
                    slab[(slice(None), slice(None)) + index],
                    field_loc_noupdate,
                    field_type_noupdate,
                )
            readstr = readstr_t
    nu.write(f"{subname}{readstr} {name}\n")
    nu.flush()


def _as_ni_nj(data, nx, ny):
    """Return a global field with axis 0 = i and axis 1 = j."""
    return _as_stored(data, nx, ny, ())


def _as_stored(data, nx, ny, axes):
    """Return a global field shaped (ni, nj, *axes).

    Fortran history restarts store (ni, nj, *axes) with an optional
    trailing time of length 1. Files written here store
    (time, *reversed axes, nj, ni).
    """
    data = np.asarray(data)
    if data.ndim == 1 + len(axes) + 2 and data.shape[0] == 1:
        data = data[0]
    if data.ndim == 2 + len(axes) + 1 and data.shape[-1] == 1:
        data = data[..., 0]
    want = (nx, ny) + tuple(axes)
    if tuple(data.shape) == want:
        return data
    rev = tuple(axes[::-1]) + (ny, nx)
    if tuple(data.shape) == rev:
        nextra = len(axes)
        order = (nextra + 1, nextra) + tuple(range(nextra - 1, -1, -1))
        return np.transpose(data, order)
    return None
