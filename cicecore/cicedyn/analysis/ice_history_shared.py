"""History field registry, accumulation, and filenames.

author for JAX version: Xinyuan Zhu
XZ: converted from F90 to Python

2012 Elizabeth Hunke split code from ice_history.F90

Output files: netCDF. One horizontal grid, so there is no block index.
The following variables are currently hard-wired as snapshots
  (instantaneous rather than time-averages):
  divu, shear, vort, sig1, sig2, sigP, mlt_onset,
  frz_onset, hisnap, aisnap

Options for histfreq: '1','h','d','m','y','x', where x means that
  output stream will not be used (recommended for efficiency).
histfreq_n can be any nonnegative integer, where 0 means that the
  corresponding histfreq frequency will not be used.
The flags (f_<field>) can be set to '1','h','d','m','y' or 'x', where
  x means the field will not be written.  To output the same field at
  more than one frequency, for instance monthly and daily, set
  f_<field> = 'md'.
"""

import os
import re

import numpy as np
import jax
import jax.numpy as jnp

from cicecore.cicedyn.general import ice_init
from cicecore.cicedyn.infrastructure import ice_grid
from cicecore.shared import ice_calendar
from cicecore.shared import ice_domain_size
from cicecore.shared import ice_fileunits
from cicecore.shared.ice_constants import c0, c1, spval_dbl
from cicecore.shared.ice_kinds_mod import dbl_kind, real_kind
from icepack.columnphysics import icepack_parameters
from icepack.columnphysics import icepack_mushy_physics

# if true, write averaged data instead of snapshots
hist_avg = None

history_file = "iceh"  # output file for history
incond_file = "iceh_ic"  # output file for snapshot initial conditions
history_dir = "./"  # directory name for history file
incond_dir = "./"  # directory for snapshot initial conditions
history_precision = 4
history_format = "cdf1"
hist_suffix = None  # appended to history_file in filename

# Field counts per rank. There is no block axis: one grid, ghosts included.
num_avail_hist_fields_2D = 0
num_avail_hist_fields_3Dc = 0  # thickness categories
num_avail_hist_fields_3Dz = 0  # vertical ice
num_avail_hist_fields_3Db = 0  # vertical biology
num_avail_hist_fields_3Da = 0  # snow + biology + aerosols
num_avail_hist_fields_3Df = 0  # floe size
num_avail_hist_fields_4Di = 0  # ice layers x categories
num_avail_hist_fields_4Ds = 0  # snow layers x categories
num_avail_hist_fields_4Df = 0  # floe size x categories

# Written sizes. ncat_hist <= ncat, nfsd_hist <= nfsd.
ncat_hist = 0
nfsd_hist = 0
nzilyr = 0  # vertical dimension (allows alternative grids)
nzslyr = 0  # snow
nzblyr = 0  # bio grid
nzalyr = 0  # aerosols (2 snow & nblyr+2 bio)

avail_hist_fields = []

a2D = None   # (i, j, field)
a3Dc = None  # (i, j, ncat, field)
a3Dz = None  # (i, j, nzilyr, field)
a3Db = None  # (i, j, nzblyr, field)
a3Da = None  # (i, j, nzalyr, field)
a3Df = None  # (i, j, nfsd, field)
a4Di = None  # (i, j, nzilyr, ncat, field)
a4Ds = None  # (i, j, nzslyr, ncat, field)
a4Df = None  # (i, j, nfsd, ncat, field)
avgct = None  # number of samples in the current average, one per stream

# rank -> (counter global name, array global name, axis names in storage order)
_RANKS = {
    "2D": ("num_avail_hist_fields_2D", "a2D", ()),
    "3Dc": ("num_avail_hist_fields_3Dc", "a3Dc", ("ncat",)),
    "3Dz": ("num_avail_hist_fields_3Dz", "a3Dz", ("nzilyr",)),
    "3Db": ("num_avail_hist_fields_3Db", "a3Db", ("nzblyr",)),
    "3Da": ("num_avail_hist_fields_3Da", "a3Da", ("nzalyr",)),
    "3Df": ("num_avail_hist_fields_3Df", "a3Df", ("nfsd",)),
    "4Di": ("num_avail_hist_fields_4Di", "a4Di", ("nzilyr", "ncat")),
    "4Ds": ("num_avail_hist_fields_4Ds", "a4Ds", ("nzslyr", "ncat")),
    "4Df": ("num_avail_hist_fields_4Df", "a4Df", ("nfsd", "ncat")),
}

time_beg = None  # bounds for averaging
time_end = None

_ncfile_last = None

subname_define = "(define_hist_field)"
subname_file = "(construct_filename)"


class HistField:
    """One history variable on one stream."""

    def __init__(self, base, vname, vunit, vdesc, vcomment, cona, conb, vhistfreq, stream, slot, gather, snapshot, rank, avg_ice_present, mask_ice_free_points, grid):
        self.base = base  # name before the stream-frequency suffix
        self.vname = vname
        self.vunit = vunit
        self.vdesc = vdesc
        self.vcomment = vcomment
        self.cona = cona  # multiplicative conversion factor
        self.conb = conb  # additive conversion factor
        self.vhistfreq = vhistfreq  # frequency of history output
        self.stream = stream
        self.slot = slot
        self.gather = gather
        self.snapshot = snapshot
        self.rank = rank
        # 'none', 'final', 'init', 'pond', or 'ridge'
        self.avg_ice_present = avg_ice_present
        self.mask_ice_free_points = mask_ice_free_points
        self.grid = grid


def set_history_dimensions():
    """Record how many category and vertical levels history will write.

    ncat_hist = ncat. nzilyr = nilyr. nzblyr = nblyr+2. nzalyr = nblyr+4.
    """
    global ncat_hist, nfsd_hist, nzilyr, nzslyr, nzblyr, nzalyr
    ncat_hist = int(ice_domain_size.ncat)
    nfsd_hist = int(ice_domain_size.nfsd)
    nzilyr = int(ice_domain_size.nilyr)
    nzslyr = int(ice_domain_size.nslyr)
    nzblyr = int(ice_domain_size.nblyr) + 2
    nzalyr = int(ice_domain_size.nblyr) + 4


def reset_history():
    """Clear the field list before init_hist defines it again."""
    global num_avail_hist_fields_2D, avail_hist_fields, a2D, avgct, time_beg, time_end
    global num_avail_hist_fields_3Dc, num_avail_hist_fields_3Dz, num_avail_hist_fields_3Db
    global num_avail_hist_fields_3Da, num_avail_hist_fields_3Df
    global num_avail_hist_fields_4Di, num_avail_hist_fields_4Ds, num_avail_hist_fields_4Df
    global a3Dc, a3Dz, a3Db, a3Da, a3Df, a4Di, a4Ds, a4Df
    global hist_avg, history_file, incond_file, history_dir, incond_dir
    global history_precision, history_format, hist_suffix, _ncfile_last
    num_avail_hist_fields_2D = 0
    num_avail_hist_fields_3Dc = 0
    num_avail_hist_fields_3Dz = 0
    num_avail_hist_fields_3Db = 0
    num_avail_hist_fields_3Da = 0
    num_avail_hist_fields_3Df = 0
    num_avail_hist_fields_4Di = 0
    num_avail_hist_fields_4Ds = 0
    num_avail_hist_fields_4Df = 0
    avail_hist_fields = []
    a2D = None
    a3Dc = None
    a3Dz = None
    a3Db = None
    a3Da = None
    a3Df = None
    a4Di = None
    a4Ds = None
    a4Df = None
    nstrm = int(ice_domain_size.max_nstrm)
    avgct = np.zeros(nstrm, dtype=dbl_kind)
    time_beg = np.zeros(nstrm, dtype=real_kind)
    time_end = np.zeros(nstrm, dtype=real_kind)
    hist_avg = [bool(value) for value in ice_init.hist_avg]
    history_file = str(ice_init.history_file).strip()
    incond_file = str(ice_init.incond_file).strip()
    history_dir = str(ice_init.history_dir).strip()
    incond_dir = str(ice_init.incond_dir).strip()
    history_precision = int(ice_init.history_precision)
    history_format = str(ice_init.history_format).strip()
    hist_suffix = [str(value).strip() for value in ice_init.hist_suffix]
    _ncfile_last = ["UnDefineD"] * nstrm


def apply_history_flags(flags):
    """Overlay f_<name> from the icefields namelists onto flags.

    Fortran reads these in each ice_history_* module before the tracer
    switches turn fields off. Defaults stay when a name is absent.
    """
    path = getattr(ice_init, "nml_filename", "ice_in")
    found = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            code = line.split("!", 1)[0]
            match = re.match(r"\s*f_(\w+)\s*=\s*'([^']*)'", code)
            if match:
                found[match.group(1)] = match.group(2).strip()
    for name in flags:
        if name in found:
            flags[name] = found[name]


def note_nml(subname, nml_name):
    """Echo the namelist group about to be read, as init_hist does."""
    ice_fileunits.nu_diag.write(f" {subname} Reading {nml_name}\n")


def write_history_catalog():
    """Print the history-tape table.

    Format follows ice_history.F90: description 40, units 16, name 12,
    one frequency character, then histfreq_n. A stream that received no
    fields has histfreq_n set to 0.
    """
    nu = ice_fileunits.nu_diag
    total = len(avail_hist_fields)
    nu.write(" \n")
    nu.write(f" total number of history fields = {total:12d}\n")
    nu.write(f" max number of history fields   = {1200:12d}\n")
    nu.write(" The following variables will be written to the history tape: \n")
    nu.write(
        f"  {'description':<19}{' ' * 10}{'units':<16}{' ' * 9}"
        f"{'variable':<12}  frequency   x\n"
    )
    if total == 0:
        nu.write(" *** WARNING: NO HISTORY FIELDS WILL BE WRITTEN ***\n")
    nstreams = int(ice_calendar.nstreams)
    counts = [0] * nstreams
    for field in avail_hist_fields:
        every = int(ice_calendar.histfreq_n[field.stream])
        if every != 0:
            nu.write(
                f" {field.vdesc[:40]:<40}  {field.vunit[:16]:<16}  "
                f"{field.vname[:12]:<12} {field.vhistfreq}  {every:6d}\n"
            )
        for ns in range(nstreams):
            if field.vhistfreq == str(ice_calendar.histfreq[ns]).strip():
                counts[ns] = counts[ns] + 1
    nu.write(" \n")
    for ns in range(nstreams):
        if counts[ns] == 0:
            ice_calendar.histfreq_n[ns] = 0


def define_hist_field(vname, vunit, vdesc, vcomment, cona, conb, vhistfreq, gather, snapshot=False, rank="2D", avg_ice_present="none", mask_ice_free_points=False, grid="T"):
    """Register a field on every stream whose frequency is in vhistfreq.

    Initializes description of an available field and returns location
    in the available fields array for use in later calls.
    2009 Created by D. Bailey following POP

    rank selects the accumulator: 2D, 3Dc, 3Dz, 3Db, 3Da, 3Df, 4Di, 4Ds, 4Df.
    The block index is not stored.
    """
    if rank not in _RANKS:
        raise ValueError(subname_define + f" ERROR: unknown history rank {rank}")
    freq_text = str(vhistfreq).strip()
    if "x" in freq_text and freq_text != "x":
        raise ValueError(
            subname_define
            + " ERROR: history frequency variable f_"
            + vname
            + " can't contain 'x' along with active frequencies"
        )
    if freq_text == "x" or freq_text == "":
        return
    counter_name = _RANKS[rank][0]
    for ns in range(ice_calendar.nstreams):
        freq = str(ice_calendar.histfreq[ns]).strip()
        if freq == "x":
            raise ValueError(subname_define + " ERROR: define_hist_fields has histfreq x")
        if freq not in freq_text:
            continue
        count = globals()[counter_name] + 1
        globals()[counter_name] = count
        if count > 1200:
            ice_fileunits.nu_diag.write(f"{subname_define} {counter_name} = {count}\n")
            raise ValueError(subname_define + " ERROR: Need in computation of max_avail_hist_fields")
        stmp = vname
        if ns > 0:
            stmp = f"{stmp}_{freq}"
        avail_hist_fields.append(
            HistField(
                vname,
                stmp,
                vunit,
                vdesc,
                vcomment,
                dbl_kind(cona),
                dbl_kind(conb),
                freq,
                ns,
                count - 1,
                gather,
                snapshot,
                rank,
                str(avg_ice_present).strip(),
                bool(mask_ice_free_points),
                grid,
            )
        )


def _axis_sizes(rank):
    """Lengths of the non-horizontal axes for one rank, in storage order."""
    sizes = {
        "ncat": ncat_hist,
        "nzilyr": nzilyr,
        "nzslyr": nzslyr,
        "nzblyr": nzblyr,
        "nzalyr": nzalyr,
        "nfsd": nfsd_hist,
    }
    return tuple(int(sizes[name]) for name in _RANKS[rank][2])


def allocate_hist_fields():
    """Allocate accumulators. One grid, including ghost cells. No block index."""
    shape = ice_domain_size.horizontal_shape()
    for rank, (counter_name, array_name, _axes) in _RANKS.items():
        count = globals()[counter_name]
        if count < 1:
            globals()[array_name] = None
            continue
        axes = _axis_sizes(rank)
        if any(length < 1 for length in axes):
            raise ValueError(
                "(allocate_hist_fields) ERROR: history rank "
                + rank
                + " has a zero axis"
            )
        globals()[array_name] = np.zeros(shape + axes + (count,), dtype=dbl_kind)


def zero_hist_slot(field):
    """Set one field's accumulator back to zero."""
    array_name = _RANKS[field.rank][1]
    array = globals()[array_name]
    if array is None:
        return
    if isinstance(array, jax.Array):
        globals()[array_name] = array.at[..., field.slot].set(c0)
    else:
        array[..., field.slot] = c0


def device_gather(compute, inputs):
    """Keep host reads separate from arithmetic that can join accumulation JIT."""
    def gather():
        return compute(*inputs())

    gather.device_inputs = inputs
    gather.device_compute = compute
    return gather


def start_device_history():
    """Move restored sums once, and compile a masked update for this catalog."""
    global _device_accumulate, _device_sources
    fields_by_base = {f.base: f for f in reversed(avail_hist_fields) if not f.snapshot}
    _device_sources = {
        base: getattr(f.gather, "device_inputs", lambda f=f: (f.gather(),))
        for base, f in fields_by_base.items()
    }
    groups = {}
    for rank, (_count, name, _axes) in _RANKS.items():
        array = globals()[name]
        if array is not None:
            globals()[name] = jnp.asarray(array)
            groups[name] = tuple(sorted(
                (f for f in avail_hist_fields if f.rank == rank), key=lambda f: f.slot))
    ilo, ihi, jlo, jhi = ice_domain_size.physical_bounds()
    cell = (slice(ilo, ihi + 1), slice(jlo, jhi + 1))
    mask = jnp.asarray(ice_grid.tmask[cell])
    averaged = tuple(bool(value) for value in hist_avg)

    @jax.jit
    def accumulate(buffers, inputs):
        sources = {}
        for base, field in fields_by_base.items():
            compute = getattr(field.gather, "device_compute", None)
            value = inputs[base][0] if compute is None else compute(*inputs[base])
            sources[base] = jnp.asarray(value, dtype=dbl_kind)
        result = {}
        for name, fields in groups.items():
            array = buffers[name]
            ocean = mask.reshape(mask.shape + (1,) * (array.ndim - 2))
            increments = jnp.stack([
                jnp.zeros_like(array[cell + (Ellipsis, f.slot)]) if f.snapshot
                else sources[f.base][cell] for f in fields
            ], axis=-1)
            keep = jnp.asarray([averaged[f.stream] or f.snapshot for f in fields])
            previous = jnp.where(keep, array, c0)
            accumulate_field = jnp.asarray([not f.snapshot for f in fields])
            result[name] = previous.at[cell].set(jnp.where(
                ocean & accumulate_field, previous[cell] + increments, previous[cell]))
        return result

    _device_accumulate = accumulate


def accumulate_device_fields():
    """Gather only enabled fields and update all ranks without a host transfer."""
    sources = {base: inputs() for base, inputs in _device_sources.items()}
    buffers = {name: globals()[name] for _, name, _ in _RANKS.values()
               if globals()[name] is not None}
    for name, array in _device_accumulate(buffers, sources).items():
        globals()[name] = array


def accum_hist_field(field, source):
    """Add one field on ocean cells.

    2009 Created by D. Bailey following POP
    2010 Generalized dimension of variables by N. Jeffery, E. Hunke
    """
    if field is None:
        return
    array = globals()[_RANKS[field.rank][1]]
    if array is None:
        return
    ilo, ihi, jlo, jhi = ice_domain_size.physical_bounds()
    # CICE's accum_hist_field uses tmask for fields on every grid.
    mask = ice_grid.tmask[ilo : ihi + 1, jlo : jhi + 1]
    extra = array.ndim - 3
    view = array[(slice(ilo, ihi + 1), slice(jlo, jhi + 1)) + (slice(None),) * extra + (field.slot,)]
    piece = np.asarray(source, dtype=dbl_kind)
    piece = piece[(slice(ilo, ihi + 1), slice(jlo, jhi + 1)) + (slice(None),) * (piece.ndim - 2)]
    ocean = np.broadcast_to(mask.reshape(mask.shape + (1,) * (view.ndim - 2)), view.shape)
    view[ocean] = view[ocean] + piece[ocean]


def construct_filename(suffix, ns, option="history"):
    """Build the history or initial-condition filename for stream ns.

    ns is 0-based. Averaged files are labeled with the end of the
    averaging window, which is the previous day, month, or year when
    the calendar has just stepped onto a new period. ``option``
    ``histrest`` names the history-restart file in ``restart_dir``.
    """
    global _ncfile_last
    iyear = int(ice_calendar.myear)
    imonth = int(ice_calendar.mmonth)
    iday = int(ice_calendar.mday)
    msec = int(ice_calendar.msec)
    dt = float(ice_calendar.dt)
    isec = int(msec - dt)
    freq = str(ice_calendar.histfreq[ns]).strip()
    every = int(ice_calendar.histfreq_n[ns])
    cstream = ""
    if hist_suffix[ns] != "x":
        cstream = hist_suffix[ns]
    if option not in ("history", "histrest"):
        raise ValueError(subname_file + f" ERROR: option invalid = {option}")

    # construct filename
    if ice_calendar.write_ic:
        isec = msec
        ncfile = f"{incond_file}.{iyear:04d}-{imonth:02d}-{iday:02d}-{isec:05d}.{suffix}"
    elif option == "histrest":
        freq = str(ice_calendar.histfreq[ns]).strip()
        ncfile = (
            f"{history_file}{cstream}_r{freq}."
            f"{iyear:04d}-{imonth:02d}-{iday:02d}-{msec:05d}.{suffix}"
        )
    elif hist_avg[ns]:
        if freq in ("1", "h", "H"):
            pass
        elif ice_calendar.new_year:
            iyear = iyear - 1
            imonth = 12
            iday = int(ice_calendar.daymo[imonth - 1])
        elif ice_calendar.new_month:
            imonth = int(ice_calendar.mmonth) - 1
            iday = int(ice_calendar.daymo[imonth - 1])
        elif ice_calendar.new_day:
            iday = iday - 1
        if freq == "1" and every == 1:
            ncfile = (
                f"{history_file}{cstream}_inst."
                f"{iyear:04d}-{imonth:02d}-{iday:02d}-{msec:05d}.{suffix}"
            )
        elif freq == "1" and every > 1:
            ncfile = (
                f"{history_file}{cstream}."
                f"{iyear:04d}-{imonth:02d}-{iday:02d}-{msec:05d}.{suffix}"
            )
        elif freq in ("h", "H"):
            ncfile = (
                f"{history_file}{cstream}_{every:02d}h."
                f"{iyear:04d}-{imonth:02d}-{iday:02d}-{msec:05d}.{suffix}"
            )
        elif freq in ("d", "D"):
            ncfile = f"{history_file}{cstream}.{iyear:04d}-{imonth:02d}-{iday:02d}.{suffix}"
        elif freq in ("m", "M"):
            ncfile = f"{history_file}{cstream}.{iyear:04d}-{imonth:02d}.{suffix}"
        elif freq in ("y", "Y"):
            ncfile = f"{history_file}{cstream}.{iyear:04d}.{suffix}"
        elif freq == "g":
            ncfile = f"{history_file}_grid.{suffix}"
        else:
            raise ValueError(subname_file + f" ERROR: histfreq unknown = {freq}")
    else:
        # instantaneous
        if freq == "g":
            ncfile = f"{history_file}_grid.{suffix}"
        else:
            ncfile = (
                f"{history_file}{cstream}_inst."
                f"{iyear:04d}-{imonth:02d}-{iday:02d}-{msec:05d}.{suffix}"
            )

    # Check whether the filename is already in use.
    # Same filename in multiple history streams leads to files being overwritten.
    # History restarts are not part of that check.
    if option != "histrest":
        for n, previous in enumerate(_ncfile_last):
            if ncfile == previous:
                ice_fileunits.nu_diag.write(f"{subname_file} history stream = {ns + 1}\n")
                ice_fileunits.nu_diag.write(f"{subname_file} history filename = {ncfile}\n")
                ice_fileunits.nu_diag.write(f"{subname_file} filename in use for stream {n + 1}\n")
                raise ValueError(
                    subname_file
                    + " ERROR: history filename already used for another history stream "
                    + ncfile
                )
        _ncfile_last[ns] = ncfile

    if option == "histrest":
        folder = str(ice_init.restart_dir).strip()
    elif ice_calendar.write_ic:
        folder = incond_dir
    else:
        folder = history_dir
    if folder == "":
        folder = "."
    if option != "histrest":
        os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, ncfile)


def _reference_field(base, stream):
    for other in avail_hist_fields:
        if other.base == base and other.stream == stream:
            return other
    return None


def scale_field(field, avgct):
    """Convert one accumulator to the value written in the file.

    Land and ghost cells are the special value. Snapshots ignore the
    average and copy the current array. Averaged fields are divided
    by the number of samples, then scaled by cona and conb.
    avg_ice_present 'final', 'init', 'pond', and 'ridge' divide by the
    accumulated reference field instead of the sample count. Category
    fields use the accumulated aicen of the same category.
    """
    puny = icepack_parameters.puny
    # Initial-condition output can still use NumPy accumulators. Output from
    # the time loop keeps normalization on the same device as the sums.
    array = globals()[_RANKS[field.rank][1]]
    xp = jnp if isinstance(array, jax.Array) else np

    ilo, ihi, jlo, jhi = ice_domain_size.physical_bounds()
    mask = xp.asarray(ice_grid.tmask[ilo : ihi + 1, jlo : jhi + 1])
    extra = 0 if array is None else array.ndim - 3
    if field.snapshot:
        raw = xp.asarray(field.gather(), dtype=dbl_kind)
        source = raw[(slice(ilo, ihi + 1), slice(jlo, jhi + 1)) + (slice(None),) * (raw.ndim - 2)]
    else:
        if array is None:
            raise ValueError("(accum_hist) ERROR: accumulator is not allocated")
        source = array[(slice(ilo, ihi + 1), slice(jlo, jhi + 1)) + (slice(None),) * extra + (field.slot,)]
    ocean = mask.reshape(mask.shape + (1,) * (source.ndim - 2))
    present = field.avg_ice_present
    if field.snapshot or present == "none":
        if not field.snapshot and avgct == c0:
            raise ValueError("(accum_hist) ERROR: avgct is zero")
        factor = c1 if field.snapshot else c1 / dbl_kind(avgct)
        return xp.where(ocean, field.cona * source * factor + field.conb, spval_dbl)
    if field.rank == "3Dc":
        reference = _reference_field("aicen", field.stream)
        if reference is None:
            raise ValueError("(accum_hist) ERROR: f_aicen must be defined")
        ref = a3Dc[(slice(ilo, ihi + 1), slice(jlo, jhi + 1), slice(None), reference.slot)]
    else:
        base = {"final": "aice", "init": "aice_init", "pond": "apond_ai", "ridge": "ardg"}[present]
        reference = _reference_field(base, field.stream)
        if reference is None:
            raise ValueError("(accum_hist) ERROR: f_" + base + " must be defined")
        ref = a2D[ilo : ihi + 1, jlo : jhi + 1, reference.slot]
        while ref.ndim < source.ndim:
            ref = ref[..., None]
    alive = ref > puny
    weight = xp.where(alive, c1 / xp.where(alive, ref, c1), c0)
    out = xp.where(ocean, field.cona * source * weight + field.conb, spval_dbl)
    if field.mask_ice_free_points:
        bare = ocean & (weight == c0)
        out = xp.where(bare, spval_dbl, out)
    return out


def ice_brine_density(qice, sice, sss):
    """Combined ice/brine density, ocean density, and bulk ice salinity.

    Port of the CMIP history helper in ice_history_shared.F90. The layer
    dimension is last, and horizontal arrays are returned without blocks.
    """
    parameters = {name: getattr(icepack_parameters, name) for name in (
        "rhoi", "rhow", "cp_ice", "cp_ocn", "Lfresh", "puny")}
    return _ice_brine_density(qice, sice, sss, parameters)


@jax.jit
def _ice_brine_density(qice, sice, sss, parameters):
    rhoi = parameters["rhoi"]
    nlayer = qice.shape[-1]
    rho_ocn = 1000.3 + 0.78237 * sss + 2.8008e-4 * sss**2
    rho_ice = jnp.zeros_like(sss, dtype=dbl_kind)
    salt_ice = jnp.zeros_like(sss, dtype=dbl_kind)
    for k in range(nlayer):
        Sbr = sice[:, :, k]
        Tice = icepack_mushy_physics.icepack_mushy_temperature_mush_jax(
            qice[:, :, k], Sbr, parameters)
        salt_ice += Sbr / nlayer
        phi = icepack_mushy_physics.icepack_mushy_liquid_fraction_jax(Tice, Sbr, parameters)
        rhob = 1000.3 + 0.78237 * Sbr + 2.8008e-4 * Sbr**2
        rho_ice += jnp.minimum(phi * rhob + (1.0 - phi) * rhoi, rho_ocn)
    return rho_ice / nlayer, rho_ocn, salt_ice
