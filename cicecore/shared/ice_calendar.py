"""Calendar routines for managing time.

author for JAX version: Xinyuan Zhu
XZ: converted from F90 to Python

2020 TC : Significant refactor to move away from time as prognostic
           Note that the reference date is arbitrarily set to
           0000-01-01-00000 and dates cannot be less than that.
           The implementation is also limited by some integer
           math to myear_max which is a parameter in this module.
2010 CM : Fixed support for Gregorian calendar: subroutines
           sec2time, time2sec and set_calendar added.
2006 ECH: Removed 'w' option for history; added 'h' and histfreq_n.
           Converted to free form source (F90).
authors: Elizabeth C. Hunke, LANL
         Tony Craig, NCAR
         Craig MacLachlan, UK Met Office

Namelist dates are copied from ice_init in init_calendar. calc_timesteps
and the first advance_timestep run after the restart is read.
"""

from cicecore.cicedyn.general import ice_init
from cicecore.shared import ice_fileunits
from cicecore.shared.ice_domain_size import max_nstrm
from cicecore.shared.ice_kinds_mod import dbl_kind
from icepack.columnphysics.icepack_parameters import icepack_query_parameters

# INTERFACES
# init_calendar            initialize calendar
# calc_timesteps           initialize number of timesteps (after namelist and restart are read)
# advance_timestep         advance model 1 timestep and update calendar
# calendar                 update model internal calendar/time information
# set_date_from_timesecs   set model date from time in seconds
#                          (relative to init date)
#                          needed for binary restarts
# semi-private, only used directly by unit tester
# compute_elapsed_days     compute elapsed days since 0000-01-01
# compute_days_between     compute elapsed days between two dates
# update_date              input date and delta date, compute new date
# calendar_date2time       convert date to time relative to init date
# calendar_time2date       convert time to date relative to init date
# calendar_sec2hms         convert seconds to hour, minute, seconds
# compute_calendar_data    compute info about calendar for a given year
# private functions
# set_calendar             sets model calendar type (noleap, etc)
# compute_relative_elapsed compute relative elapsed years, months, days, hours

# PUBLIC

ice_calendar_proleptic_gregorian = "proleptic_gregorian"  # calendar name
ice_calendar_gregorian = "Gregorian"  # actually proleptic gregorian here, legacy name
ice_calendar_noleap = "NO_LEAP"       # 365 day per year calendar
ice_calendar_360day = "360day"        # 360 day calendar with 30 days per month

months_per_year = 12      # months per year
hours_per_day = 24        # hours per day
minutes_per_hour = 60     # minutes per hour
seconds_per_minute = 60   # seconds per minute

seconds_per_day = 0                    # seconds per day
seconds_per_hour = 0                   # seconds per hour
days_per_year = 0                      # number of days in one year
daymo = [0] * months_per_year          # number of days in each month
daycal = [0] * (months_per_year + 1)   # accumulated days in year to end of prior month

# step counters
istep = 0     # local step counter for current run in time loop
istep0 = 0    # counter, number of steps at start of run
istep1 = 0    # counter, number of steps at current timestep
# basic model time variables
myear = 0     # year number
mmonth = 0    # month number, 1 to months_per_year
mday = 0      # day of the month
msec = 0      # elapsed seconds into date
# initial time
year_init = 0    # initial year
month_init = 0   # initial month
day_init = 0     # initial day of month
sec_init = 0     # initial seconds
# other stuff
hh_init = 0          # initial hour derived from sec_init
mm_init = 0          # initial minute derived from sec_init
ss_init = 0          # initial second derived from sec_init
idate = 0            # date (yyyymmdd)
idate0 = 0           # initial date (yyyymmdd), associated with year_init, month_init, day_init
dayyr = 0            # number of days in the current year
npt = 0              # total number of time steps (dt)
npt0 = 0             # original npt value in npt0_unit
ndtd = 1             # number of dynamics subcycles: dt_dyn=dt/ndtd
stop_now = 0         # if 1, end program execution
write_restart = 0    # if 1, write restart now
diagfreq = 0         # diagnostic output frequency (10 = once per 10 dt)
nstreams = 0         # number of history output streams
dumpfreq_n = [1] * int(max_nstrm)   # restart output frequency (10 = once per 10 d,m,y)
histfreq_n = [1] * int(max_nstrm)   # history output frequency

new_year = False     # new year = .true.
new_month = False    # new month = .true.
new_day = False      # new day = .true.
new_hour = False     # new hour = .true.

dt = dbl_kind(0.0)             # thermodynamics timestep (s)
dt_dyn = dbl_kind(0.0)         # dynamics/transport/ridging timestep (s)
timesecs = dbl_kind(0.0)       # total elapsed time (s)
yday = dbl_kind(0.0)           # day of the year
nextsw_cday = dbl_kind(0.0)    # julian day of next shortwave calculation

use_leap_years = False         # use leap year functionality if true
write_ic = False               # write initial condition now
write_histrest = True          # write history restarts if needed
dump_last = False              # write restart file on last time step
force_restart_now = False      # force a restart now
write_history = [False] * int(max_nstrm)  # write history now

npt_unit = "1"     # run length unit, 'y', 'm', 'd', 'h', 's', '1'
npt0_unit = "1"    # original run length unit, 'y', 'm', 'd', 'h', 's', '1'
histfreq = ["x"] * int(max_nstrm)   # history output frequency, 'y','m','d','h','1','x'
dumpfreq = ["x"] * int(max_nstrm)   # restart frequency, 'y','m','d', h', '1', 'x' followed by optional 1

# This is needed for the calendar unit tester
dumpfreq_base = ["init"] * int(max_nstrm)   # restart frequency basetime ('zero', 'init')
histfreq_base = ["zero"] * int(max_nstrm)   # history frequency basetime ('zero', 'init')
calendar_type = ""                          # define calendar type

# PRIVATE

hour = 0   # hour of the day

myear_max = 200000   # maximum year, limited by integer overflow in elapsed_hours

# 360-day year data
daymo360 = [30] * months_per_year   # number of days in each month
# 365-day year data
daymo365 = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]   # number of days in each month
# 366-day year data (leap year)
daymo366 = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]   # number of days in each month


def _abort(subname, message):
    ice_fileunits.nu_diag.write(f"{subname} {message}\n")
    ice_fileunits.nu_diag.flush()
    raise RuntimeError(f"{subname}{message}")


def _nint(value):
    """Fortran nint: half away from zero."""
    number = float(value)
    if number >= 0.0:
        return int(number + 0.5)
    return int(number - 0.5)


def _secday():
    # seconds per day
    return icepack_query_parameters("secday")["secday"]


#=======================================================================
# Compute hours, minutes, seconds from seconds

def calendar_sec2hms(seconds):
    # seconds: calendar seconds in day
    # hh, mm, ss: output hours, minutes, seconds
    subname = "(calendar_sec2hms)"
    if seconds >= seconds_per_day:
        _abort(subname, f" ERROR seconds >= seconds_per_day, {seconds} {seconds_per_day}")
    hh = seconds // seconds_per_hour
    mm = (seconds - hh * seconds_per_hour) // seconds_per_minute
    ss = seconds - hh * seconds_per_hour - mm * seconds_per_minute
    return hh, mm, ss


#=======================================================================
# compute calendar data based on year

def compute_calendar_data(ayear):
    # ayear: year
    # adaymo: days per month
    # adaycal: day count per month
    # adayyr: days per year
    subname = "(compute_calendar_data)"
    if ayear < 0:
        _abort(subname, f" ERROR in ayear = {ayear}")
    if calendar_type in (ice_calendar_gregorian, ice_calendar_proleptic_gregorian):
        isleap = False  # not a leap year
        if ayear % 4 == 0:
            isleap = True
        if ayear % 100 == 0:
            isleap = False
        if ayear % 400 == 0:
            isleap = True
        adaymo = list(daymo366 if isleap else daymo365)
    elif calendar_type == ice_calendar_360day:
        adaymo = list(daymo360)
    else:
        adaymo = list(daymo365)
    adaycal = [0] * (months_per_year + 1)
    for n in range(months_per_year):
        adaycal[n + 1] = adaycal[n] + adaymo[n]
    return adaymo, adaycal, adaycal[months_per_year]


#=======================================================================
# Set the model calendar data for year

def set_calendar(year):
    # year: current year
    global daymo, daycal, dayyr
    daymo, daycal, dayyr = compute_calendar_data(year)


#=======================================================================
# Add and reconcile date
# delta time arguments are optional

def update_date(ayear, amon, aday, asec, dyear=None, dmon=None, dday=None, dsec=None):
    # ayear, amon, aday, asec: year, month, day, sec
    # dyear, dmon, dday, dsec: delta year, month, day, sec
    # tdaymo: days per month
    # tdaycal: day count per month
    # tdayyr: days in year
    # isecday: seconds per day
    subname = "(update_date)"
    isecday = _nint(_secday())

    # order matters.  think about adding 1 month and 10 days to the 25th of a month
    # what is the right order?
    # will add all deltas then reconcile years then months then days then seconds
    if dyear is not None:
        ayear = ayear + dyear
    if dmon is not None:
        amon = amon + dmon
    if dday is not None:
        aday = aday + dday
    if dsec is not None:
        asec = asec + dsec

    # adjust negative data first
    # reconcile months - years
    while amon <= 0:
        delta = abs(amon) // months_per_year + 1
        ayear = ayear - delta
        amon = amon + delta * months_per_year
    _tdaymo, _tdaycal, _tdayyr = compute_calendar_data(ayear)
    tdaymo = _tdaymo

    # reconcile days - months - years
    while aday <= 0:
        amon = amon - 1
        while amon <= 0:
            delta = abs(amon) // months_per_year + 1
            ayear = ayear - delta
            amon = amon + delta * months_per_year
            tdaymo, _tdaycal, _tdayyr = compute_calendar_data(ayear)
        aday = aday + tdaymo[amon - 1]

    # reconcile seconds - days - months - years
    if asec < 0:
        delta = abs(asec) // isecday + 1
        aday = aday - delta
        asec = asec + delta * isecday
    while aday <= 0:
        amon = amon - 1
        while amon <= 0:
            delta = abs(amon) // months_per_year + 1
            ayear = ayear - delta
            amon = amon + delta * months_per_year
            tdaymo, _tdaycal, _tdayyr = compute_calendar_data(ayear)
        aday = aday + tdaymo[amon - 1]

    # check for negative data
    if ayear < 0 or amon <= 0 or aday <= 0 or asec < 0:
        _abort(subname, f" ERROR in dateA, {ayear} {amon} {aday} {asec}")

    # reconcile months - years
    while amon > months_per_year:
        delta = (amon - 1) // months_per_year
        ayear = ayear + delta
        amon = amon - delta * months_per_year
    tdaymo, _tdaycal, _tdayyr = compute_calendar_data(ayear)

    # reconcile days - months - years
    while aday > tdaymo[amon - 1]:
        aday = aday - tdaymo[amon - 1]
        amon = amon + 1
        while amon > months_per_year:
            delta = (amon - 1) // months_per_year
            ayear = ayear + delta
            amon = amon - delta * months_per_year
            tdaymo, _tdaycal, _tdayyr = compute_calendar_data(ayear)

    # reconcile seconds - days - months - years
    if asec >= isecday:
        delta = asec // isecday
        aday = aday + delta
        asec = asec - delta * isecday
    while aday > tdaymo[amon - 1]:
        aday = aday - tdaymo[amon - 1]
        amon = amon + 1
        while amon > months_per_year:
            delta = (amon - 1) // months_per_year
            ayear = ayear + delta
            amon = amon - delta * months_per_year
            tdaymo, _tdaycal, _tdayyr = compute_calendar_data(ayear)

    # check for negative data, just in case
    if ayear < 0 or amon <= 0 or aday <= 0 or asec < 0:
        _abort(subname, f" ERROR in dateB, {ayear} {amon} {aday} {asec}")
    return ayear, amon, aday, asec


#=======================================================================
# Compute elapsed days from 0000-01-01 to year1,month1,day1
# 0000-01-01 is 0 elapsed days

def compute_elapsed_days(ayear, amonth, aday):
    # ayear: year
    # amonth: month
    # aday: day
    # tdaymo: days per month
    # tdaycal: day count per month
    # tdayyr: days in year
    subname = "(compute_elapsed_days)"

    # use 0000-01-01 as base, year 0 is a leap year
    # this must be implemented consistent with set_calendar
    lyear = ayear
    lmonth = amonth
    lday = aday
    lsec = 0
    if lyear < 0 or lmonth <= 0 or lday <= 0:
        _abort(subname, f" ERROR for year,month,day = {lyear} {lmonth} {lday}")
    elif lmonth > months_per_year:
        lyear, lmonth, lday, lsec = update_date(lyear, lmonth, lday, lsec)

    # compute days from year 0000-01-01 to year-01-01
    # don't loop thru years for performance reasons
    if calendar_type in (ice_calendar_gregorian, ice_calendar_proleptic_gregorian):
        if lyear == 0:
            ced_nday = 0
        else:
            ced_nday = (
                lyear * 365
                + 1
                + (lyear - 1) // 4
                - (lyear - 1) // 100
                + (lyear - 1) // 400
            )
    else:
        ced_nday = lyear * daycal[months_per_year]

    # now compute days in this year
    tdaymo, _tdaycal, _tdayyr = compute_calendar_data(lyear)
    for n in range(lmonth - 1):
        ced_nday = ced_nday + tdaymo[n]
    if lday <= tdaymo[lmonth - 1]:
        ced_nday = ced_nday + lday - 1
    else:
        _abort(subname, f" ERROR for year,month,day = {ayear} {amonth} {aday}")
    return ced_nday


#=======================================================================
# Compute elapsed days from year0,month0,day0 to year1,month1,day1
# Same day results in 0 elapsed days

def compute_days_between(year0, month0, day0, year1, month1, day1):
    # year0: start year
    # month0: start month
    # day0: start day
    # year1: end year
    # month1: end month
    # day1: end day
    nday0 = compute_elapsed_days(year0, month0, day0)
    nday1 = compute_elapsed_days(year1, month1, day1)
    return nday1 - nday0


#=======================================================================
# Compute time in seconds from input calendar date
# relative to year_init, month_init, day_init, sec_init unless _ref values passed in
# For santity, must pass all four ref values or none

def calendar_date2time(
    ayear, amon, aday, asec, year_ref=None, mon_ref=None, day_ref=None, sec_ref=None
):
    # ayear, amon, aday, asec: year, month, day, sec of ttimesecs
    # atimesecs: seconds since init date
    # year_ref, mon_ref, day_ref, sec_ref: year, month, day, sec reference time
    # elapsed_days: since beginning this run
    # lyear, lmon, lday, lsec: local reference year, month, day, sec
    subname = "(calendar_date2time)"
    # set reference date and check that 0 or 4 optional arguments are passed
    cnt = 0
    if year_ref is not None:
        lyear = year_ref
        cnt = cnt + 1
    else:
        lyear = year_init
    if mon_ref is not None:
        lmon = mon_ref
        cnt = cnt + 1
    else:
        lmon = month_init
    if day_ref is not None:
        lday = day_ref
        cnt = cnt + 1
    else:
        lday = day_init
    if sec_ref is not None:
        lsec = sec_ref
        cnt = cnt + 1
    else:
        lsec = sec_init
    if cnt != 0 and cnt != 4:
        _abort(subname, " ERROR in ref args, must pass 0 or 4 ")
    secday = _secday()
    elapsed_days = compute_days_between(lyear, lmon, lday, ayear, amon, aday)
    return (
        dbl_kind(elapsed_days) * secday
        + dbl_kind(asec)
        - dbl_kind(lsec)
    )


#=======================================================================
# Compute calendar date from input time in seconds
# relative to year_init, month_init, day_init, sec_init or ref data if passed.
# For sanity, require all four or no ref values.
# Implemented to minimize accumulating errors and avoid overflows
# and perform well.

def calendar_time2date(
    atimesecs, year_ref=None, mon_ref=None, day_ref=None, sec_ref=None
):
    # atimesecs: seconds since init date
    # ayear, amon, aday, asec: year, month, day, sec of timesecs
    # tyear, tmon, tday, tsec: temporaries
    # tdaymo: days per month
    # tdaycal: day count per month
    # tdayyr: days in year
    subname = "(calendar_time2date)"
    secday = _secday()

    # we could allow negative atimesecs, but this shouldn't be needed
    if atimesecs < 0.0:
        _abort(subname, f" ERROR in atimesecs {atimesecs}")
    # set reference date and check that 0 or 4 optional arguments are passed
    # year_ref, mon_ref, day_ref, sec_ref: year, month, day, sec reference time
    # lyear, lmon, lday, lsec: local reference year, month, day, sec
    cnt = 0
    if year_ref is not None:
        lyear = year_ref
        cnt = cnt + 1
    else:
        lyear = year_init
    if mon_ref is not None:
        lmon = mon_ref
        cnt = cnt + 1
    else:
        lmon = month_init
    if day_ref is not None:
        lday = day_ref
        cnt = cnt + 1
    else:
        lday = day_init
    if sec_ref is not None:
        lsec = sec_ref
        cnt = cnt + 1
    else:
        lsec = sec_init
    if cnt != 0 and cnt != 4:
        _abort(subname, " ERROR in ref args, must pass 0 or 4 ")

    # -------------------------------------------------------------------
    # tcraig, this is risky because atimesecs is real and could be very large
    #     ayear = lyear_ref
    #     amon = lmon_ref
    #     aday = lday_ref
    #     asec = lsec_ref
    #
    #     call update_date(ayear,amon,aday,asec,dsec=nint(atimesecs))
    #     return
    # -------------------------------------------------------------------

    # initial guess
    tyear = lyear
    tmon = 1
    tday = 1
    # tsec = 0

    # add initial seconds to timesecs and treat lsec_ref as zero
    _tdaymo, _tdaycal, tdayyr = compute_calendar_data(tyear)
    ltimesecs = dbl_kind(atimesecs) + dbl_kind(lsec)

    # first estimate of tyear
    rdays = ltimesecs / secday
    tyear = tyear + int(rdays) // tdayyr

    # reduce estimate of tyear if ndays > rdays
    ndays = compute_days_between(lyear, lmon, lday, tyear, tmon, tday)
    if ndays > int(rdays):
        tyear = tyear - (ndays - int(rdays)) // tdayyr - 1
        ndays = compute_days_between(lyear, lmon, lday, tyear, tmon, tday)
    tdaymo, _tdaycal, _tdayyr = compute_calendar_data(tyear)

    # compute residual days, switch to integers, compute date
    rdays = ltimesecs / secday
    tday = int(rdays) - ndays + 1
    while tday > tdaymo[tmon - 1]:
        tday = tday - tdaymo[tmon - 1]
        tmon = tmon + 1
        while tmon > months_per_year:
            tmon = tmon - months_per_year
            tyear = tyear + 1
            tdaymo, _tdaycal, _tdayyr = compute_calendar_data(tyear)

    ndays = compute_days_between(lyear, lmon, lday, tyear, tmon, tday)
    tsec = int(ltimesecs - dbl_kind(ndays) * secday)
    if tsec > secday:
        _abort(subname, f" ERROR in seconds, {tyear} {tmon} {tday} {tsec}")
    return tyear, tmon, tday, tsec


#=======================================================================
# Set internal calendar date from timesecs input
# Needed for binary restarts where only timesecs is on the restart file

def set_date_from_timesecs(ttimesecs):
    # ttimesecs: seconds since init date
    global timesecs, myear, mmonth, mday, msec
    timesecs = dbl_kind(ttimesecs)
    myear, mmonth, mday, msec = calendar_time2date(
        ttimesecs, year_init, month_init, day_init, sec_init
    )


#=======================================================================
# Compute relative elapsed years, months, days, hours from base time

def compute_relative_elapsed(base):
    # ey, em, ed, eh: relative elapsed year, month, day, hour
    subname = "(compute_relative_elapsed)"
    if base == "zero":
        elapsed_years = myear
        elapsed_months = elapsed_years * months_per_year + mmonth - 1
        elapsed_days = compute_days_between(0, 1, 1, myear, mmonth, mday)
        elapsed_hours = elapsed_days * hours_per_day + hour
    elif base == "init":
        elapsed_years = myear - year_init
        elapsed_months = elapsed_years * months_per_year + (mmonth - month_init)
        elapsed_days = compute_days_between(year_init, month_init, day_init, myear, mmonth, mday)
        elapsed_hours = elapsed_days * hours_per_day + hour
    else:
        _abort(subname, f" ERROR base not recognized, {base}")
    # if (my_task == master_task) then
    #    write(nu_diag,*) subname,' ey',ey
    #    write(nu_diag,*) subname,' em',em
    #    write(nu_diag,*) subname,' ed',ed
    #    write(nu_diag,*) subname,' eh',eh
    # endif
    return elapsed_years, elapsed_months, elapsed_days, elapsed_hours


#=======================================================================
# Update the calendar and time manager info

def calendar():
    # This sets a bunch of internal calendar stuff including history and
    # restart frequencies.  These frequencies are relative to the start
    # of time which is arbitrarily set to year=0, month=1, day=1, sec=0
    # or to the model init time (year_init, month_init, day_init) depending
    # on histfreq_base or dumpfreq_base settings.
    # Using 'zero' means that the frequencies are repeatable between runs
    # regardless of the initial model date.
    # One thing to watch for is the size of elapsed hours.  This will
    # become a large integer and will overflow if the year is ever
    # greater than about 200,000 years.  A check has been added just
    # to make sure this doesn't happen.
    # The largest integer*4 is about 2^31 = 2*2^10^3 =~ 2*1000^3 = 2.e9
    # 2.e9 (hours) / (365*24 hours/year) =~ 228,000 years
    # The elapsed hours will overflow integers at some point after that.

    # real (kind=dbl_kind), intent(in), optional :: &
    #    ttime                          ! time variable

    # local variables
    # ns: loop index
    # yearp, monthp, dayp, hourp: previous year, month, day, hour
    # elapsed_years: relative elapsed years
    # elapsed_months: relative elapsed months
    # elapsed_days: relative elapsed days
    # elapsed_hours: relative elapsed hours
    global myear, mmonth, mday, msec, hour
    global new_year, new_month, new_day, new_hour
    global write_restart, idate, yday, timesecs, stop_now
    subname = "(calendar)"
    yearp = myear
    monthp = mmonth
    dayp = mday
    hourp = hour
    new_year = False
    new_month = False
    new_day = False
    new_hour = False
    for ns in range(len(write_history)):
        write_history[ns] = False
    write_restart = 0

    myear, mmonth, mday, msec = update_date(myear, mmonth, mday, msec)
    set_calendar(myear)
    if myear > myear_max:
        _abort(subname, f" ERROR year too large, {myear} {myear_max}")

    idate = myear * 10000 + mmonth * 100 + mday  # date (yyyymmdd)
    yday = dbl_kind(daycal[mmonth - 1] + mday)   # day of the year
    hour = msec // seconds_per_hour
    timesecs = calendar_date2time(myear, mmonth, mday, msec)

    #--- compute other stuff
    # CESMCOUPLED / GEOSCOUPLED skips setting stop_now and dump_last write_restart
    if istep >= npt + 1:
        stop_now = 1
    if istep == npt and dump_last:
        write_restart = 1  # last timestep
    if myear != yearp:
        new_year = True
    if mmonth != monthp:
        new_month = True
    if mday != dayp:
        new_day = True
    if hour != hourp:
        new_hour = True

    # History writing flags
    for ns in range(nstreams):
        elapsed_years, elapsed_months, elapsed_days, elapsed_hours = compute_relative_elapsed(
            histfreq_base[ns]
        )
        freq = str(histfreq[ns]).strip()
        every = histfreq_n[ns]
        if freq in ("y", "Y") and new_year and every != 0 and elapsed_years % every == 0:
            write_history[ns] = True
        elif freq in ("m", "M") and new_month and every != 0 and elapsed_months % every == 0:
            write_history[ns] = True
        elif freq in ("d", "D") and new_day and every != 0 and elapsed_days % every == 0:
            write_history[ns] = True
        elif freq in ("n", "N") and new_day and every != 0 and mday == every:
            write_history[ns] = True
        elif freq in ("h", "H") and new_hour and every != 0 and elapsed_hours % every == 0:
            write_history[ns] = True
        elif freq == "1" and every != 0 and istep1 % every == 0:
            write_history[ns] = True
        elif freq == "g":
            # grid file
            write_history[ns] = True

    # Restart writing flag, set dumpfreq to 'x" if stream is written once
    for ns in range(int(max_nstrm)):
        elapsed_years, elapsed_months, elapsed_days, elapsed_hours = compute_relative_elapsed(
            dumpfreq_base[ns]
        )
        token = (str(dumpfreq[ns]) + "  ")[:2]
        first = token[0]
        second = token[1]
        every = dumpfreq_n[ns]
        hit = False
        if first in ("y", "Y") and new_year and elapsed_years % every == 0:
            hit = True
        elif first in ("m", "M") and new_month and elapsed_months % every == 0:
            hit = True
        elif first in ("d", "D") and new_day and elapsed_days % every == 0:
            hit = True
        elif first in ("h", "H") and new_hour and elapsed_hours % every == 0:
            hit = True
        elif first == "1" and istep1 % every == 0:
            hit = True
        if hit:
            write_restart = 1
            if second == "1":
                dumpfreq[ns] = "x"

    if force_restart_now:
        write_restart = 1

    if diagfreq != 0 and istep1 % diagfreq == 0 and stop_now != 1:
        ice_fileunits.nu_diag.write(" \n")
        ice_fileunits.nu_diag.write(
            f"istep1:{istep1:10d}    idate:{idate:10d}    sec:{msec:10d}\n"
        )
        ice_fileunits.nu_diag.flush()


#=======================================================================
# Initialize calendar variables

def init_calendar():
    # secday: seconds per day
    global seconds_per_day, seconds_per_hour
    global istep, myear, mmonth, mday, msec, hour, istep1, idate0
    global stop_now, dt_dyn, force_restart_now, nstreams, calendar_type
    global year_init, month_init, day_init, sec_init, istep0
    global dt, ndtd, npt, npt_unit, days_per_year, use_leap_years
    global diagfreq, dump_last, write_ic, write_histrest
    global histfreq, histfreq_n, histfreq_base
    global dumpfreq, dumpfreq_n, dumpfreq_base
    global hh_init, mm_init, ss_init
    subname = "(init_calendar)"

    secday = _secday()
    seconds_per_day = _nint(secday)
    if abs(float(seconds_per_day) / float(secday)) - 1.0 > 1.0e-7:
        _abort(subname, f"ERROR: improper secday {secday}")
    seconds_per_hour = _nint(float(secday) / float(hours_per_day))
    if abs(seconds_per_hour * hours_per_day - seconds_per_day) > 0:
        _abort(subname, "ERROR: improper seconds_per_hour")

    year_init = int(ice_init.year_init)
    month_init = int(ice_init.month_init)
    day_init = int(ice_init.day_init)
    sec_init = int(ice_init.sec_init)
    istep0 = int(ice_init.istep0)
    dt = dbl_kind(ice_init.dt)
    ndtd = int(ice_init.ndtd)
    npt = int(ice_init.npt)
    npt_unit = str(ice_init.npt_unit).strip()
    days_per_year = int(ice_init.days_per_year)
    use_leap_years = bool(ice_init.use_leap_years)
    diagfreq = int(ice_init.diagfreq)
    dump_last = bool(ice_init.dump_last)
    write_ic = bool(ice_init.write_ic)
    write_histrest = bool(getattr(ice_init, "write_histrest", True))
    histfreq = [str(value) for value in ice_init.histfreq]
    histfreq_n = [int(value) for value in ice_init.histfreq_n]
    histfreq_base = [str(value) for value in ice_init.histfreq_base]
    dumpfreq = [str(value) for value in ice_init.dumpfreq]
    dumpfreq_n = [int(value) for value in ice_init.dumpfreq_n]
    dumpfreq_base = [str(value) for value in ice_init.dumpfreq_base]

    istep = 0             # local timestep number
    myear = year_init     # year
    mmonth = month_init   # month
    mday = day_init       # day of the month
    msec = sec_init       # seconds into date
    hh_init, mm_init, ss_init = calendar_sec2hms(sec_init)  # initialize hh,mm,ss _init
    hour = 0              # computed in calendar, but needs some reasonable initial value
    istep1 = istep0       # number of steps at current timestep
                          # real (dumped) or imagined (use to set calendar)
    idate0 = myear * 10000 + mmonth * 100 + mday  # date (yyyymmdd)
    stop_now = 0          # end program execution if stop_now=1
    dt_dyn = dt / dbl_kind(ndtd)  # dynamics et al timestep
    force_restart_now = False

    # initialize nstreams to zero (will be initialized from namelist in 'init_hist')
    # this avoids using it uninitialzed in 'calendar' below
    nstreams = 0

    # CESMCOUPLED / GEOSCOUPLED: calendar_type set by coupling
    calendar_type = ""
    if use_leap_years:
        if days_per_year == 365:
            calendar_type = ice_calendar_proleptic_gregorian
        else:
            _abort(subname, "ERROR: use_leap_years is true, must set days_per_year to 365")
    elif days_per_year == 365:
        calendar_type = ice_calendar_noleap
    elif days_per_year == 360:
        calendar_type = ice_calendar_360day
    else:
        _abort(subname, "ERROR: days_per_year only 365 or 360 supported")

    # prevent coupling setting invalid calendars
    if calendar_type not in (
        ice_calendar_gregorian,
        ice_calendar_proleptic_gregorian,
        ice_calendar_noleap,
        ice_calendar_360day,
    ):
        _abort(subname, f"Error: {calendar_type} not supported")

    set_calendar(myear)
    calendar()


#=======================================================================
# Initialize timestep counter
# This converts npt_unit and npt to a number of timesteps stored in npt
# npt0 and npt0_unit remember the original values
# It is safe to call this more than once, but it should be called only after
# the initial model run date is known (from namelist or restart) and before
# the first timestep

def calc_timesteps():
    # secday: seconds per day
    # dtimesecs: time in seconds of run
    # yeare, monthe, daye, sece: time at end of run
    global npt, npt_unit, npt0, npt0_unit
    subname = "(calc_timesteps)"
    secday = _secday()
    yeare = myear
    monthe = mmonth
    daye = mday
    sece = msec
    npt0 = npt
    npt0_unit = npt_unit

    if npt_unit == "y":
        yeare, monthe, daye, sece = update_date(yeare, monthe, daye, sece, dyear=npt)
        dtimesecs = calendar_date2time(yeare, monthe, daye, sece, myear, mmonth, mday, msec)
    elif npt_unit == "m":
        yeare, monthe, daye, sece = update_date(yeare, monthe, daye, sece, dmon=npt)
        dtimesecs = calendar_date2time(yeare, monthe, daye, sece, myear, mmonth, mday, msec)
    elif npt_unit == "d":
        dtimesecs = dbl_kind(npt) * secday
        yeare, monthe, daye, sece = update_date(yeare, monthe, daye, sece, dday=npt)
    elif npt_unit == "h":
        dtimesecs = dbl_kind(npt) * secday / dbl_kind(hours_per_day)
        yeare, monthe, daye, sece = update_date(
            yeare, monthe, daye, sece, dsec=_nint(dtimesecs)
        )
    elif npt_unit == "s":
        yeare, monthe, daye, sece = update_date(yeare, monthe, daye, sece, dsec=npt)
        dtimesecs = dbl_kind(npt)
    elif npt_unit == "1":
        dtimesecs = dt * dbl_kind(npt)
        yeare, monthe, daye, sece = update_date(
            yeare, monthe, daye, sece, dsec=_nint(dtimesecs)
        )
    else:
        _abort(subname, f" ERROR invalid npt_unit = {npt_unit}")

    npt = _nint(dtimesecs / dt)
    npt_unit = "1"
    ice_init.npt = npt
    ice_init.npt_unit = npt_unit

    out = ice_fileunits.nu_diag
    out.write(" \n")
    out.write(
        f" {subname} modified npt from {npt0:9d} {npt0_unit} with dt= {float(dt):13.2f}\n"
    )
    out.write(
        f" {subname}                to {npt:9d} {npt_unit} with dt= {float(dt):13.2f}\n"
    )
    out.write(
        f" {subname} start time is{f'{myear:04d}'.rjust(6)}-{mmonth:02d}-{mday:02d}:{msec:05d}\n"
    )
    out.write(
        f" {subname}   end time is{f'{yeare:04d}'.rjust(6)}-{monthe:02d}-{daye:02d}:{sece:05d}\n"
    )
    out.write(" \n")
    out.flush()

    # check that npt is very close to an integer
    if abs(float(npt) * float(dt) / float(dtimesecs)) - 1.0 > 1.0e-7:
        _abort(subname, f" ERROR dt and npt not consistent {npt} {dt}")


#=======================================================================
# Determine the date at the end of the time step

def advance_timestep():
    # local variables
    # idt: integer dt
    global istep, istep1, msec
    subname = "(advance_timestep)"
    if str(npt_unit).strip() != "1":
        _abort(
            subname,
            f" ERROR npt_unit should be converted to timesteps by now {npt_unit}",
        )
    istep = istep + 1
    istep1 = istep1 + 1
    idt = _nint(dt)
    # dt is historically a real but it should be an integer
    # make sure dt is very close to an integer
    if abs(float(idt) / float(dt)) - 1.0 > 1.0e-7:
        _abort(subname, f" ERROR dt error, needs to be integer number of seconds, dt={dt}")
    msec = msec + idt
    calendar()
