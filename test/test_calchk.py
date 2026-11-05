"""Calendar checks from cicecore/drivers/unittest/calchk/calchk.F90.

author for JAX version: Xinyuan Zhu
XZ: converted from F90 to Python

Fortran defaults yearmax to 100000. This script uses 1000, the other
value whose day count is written next to that default (calchk.F90 around
line 161): noleap 365364, 360day 360359, gregorian 365607.
The absolute dates checked after the add/subtract loop do not depend on
yearmax; they start from year 1000.
"""

import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cicecore.cicedyn.general import ice_init
from cicecore.shared import ice_calendar as cal
from cicecore.shared import ice_fileunits

# Fortran default is 100000. Both day counts are in calchk.F90 near line 161.
yearmax = 1000

TESTNAMES = (
    "compute_elapsed_days",
    "set_date_from_timesecs",
    "calendar advance",
    "date2time time2date",
    "big add/sub update_date",
    "small add/sub update_date",
    "special checks",
    "calc_timesteps",
    "seconds_to_hms",
)

# nfa == 1 date after adding the chunks through calendar().
# Order is noleap, 360day, gregorian.
ADVANCE_DATE = {
    "noleap": (1487, 1, 21, 22825),
    "360day": (1488, 1, 13, 22825),
    "gregorian": (1487, 1, 5, 22825),
}
# update_date of the summed deltas from 1000-01-01.
BIG_DATE = {
    "noleap": (1487, 1, 24, 22825),
    "360day": (1488, 1, 13, 22825),
    "gregorian": (1487, 1, 7, 22825),
}
NDAY_1000 = {"noleap": 365364, "360day": 360359, "gregorian": 365607}
NPT_MONTH = {"noleap": 7272, "360day": 7200, "gregorian": 7296}
NPT_YEAR = {"noleap": 87600, "360day": 86400, "gregorian": 87672}


def _configure(use_leap_years, days_per_year):
    # istep0 = 1000, year_init = 0, dt = 86400, then init_calendar.
    # Python init_calendar reads these from the namelist module.
    ice_init.use_leap_years = use_leap_years
    ice_init.days_per_year = days_per_year
    ice_init.istep0 = 1000
    ice_init.year_init = 0
    ice_init.month_init = 1
    ice_init.day_init = 1
    ice_init.sec_init = 0
    ice_init.dt = 86400.0
    ice_init.diagfreq = 99999999
    ice_fileunits.nu_diag = io.StringIO()
    cal.myear = -1
    cal.mmonth = -1
    cal.mday = -1
    cal.init_calendar()


def _xadd(nf1, nf2):
    if nf2 == 1 or nf2 == 2:
        return nf1 * nf1
    if nf2 == 3:
        return nf1 * nf1 * nf1 * nf1
    return nf1 ** 7


def _run_one(calstr, use_leap_years, days_per_year):
    fail = [False] * 9
    _configure(use_leap_years, days_per_year)

    # check days increment by 1, then set_date_from_timesecs
    ndayc = -1
    ndayv = None
    for ny in range(0, yearmax + 1):
        for nm in range(1, cal.months_per_year + 1):
            # daymo is the current year; calendar() refreshes it
            for nd in range(1, cal.daymo[nm - 1] + 1):
                ndayv = cal.compute_elapsed_days(ny, nm, nd)
                if ndayv - ndayc != 1:
                    fail[0] = True
                cal.myear = ny
                cal.mmonth = nm
                cal.mday = nd
                cal.msec = 0
                cal.calendar()
                timesecsv = cal.timesecs
                cal.myear = -1
                cal.mmonth = -1
                cal.mday = -1
                cal.msec = -1
                cal.timesecs = -1
                cal.set_date_from_timesecs(timesecsv)
                if (
                    cal.myear != ny
                    or cal.mmonth != nm
                    or cal.mday != nd
                    or cal.msec != 0
                    or cal.timesecs != timesecsv
                ):
                    fail[1] = True
                ndayc = ndayv
    if ndayv != NDAY_1000[calstr]:
        fail[0] = True

    # add arbitrary amounts, then subtract the same amounts in reverse
    cal.myear = 1000
    cal.mmonth = 1
    cal.mday = 1
    cal.msec = 0
    cal.calendar()
    dyear = dmon = dday = dsec = 0
    for nfa in (1, -1):
        for nfb in range(1, 11):
            for nfc in range(1, 5):
                if nfa == 1:
                    nf1, nf2 = nfb, nfc
                else:
                    nf1, nf2 = 11 - nfb, 5 - nfc
                fyear = fmon = fday = fsec = 0
                xadd = _xadd(nf1, nf2)
                if nf2 == 1:
                    cal.myear = cal.myear + nfa * xadd
                    if nfa == 1:
                        dyear = dyear + nfa * xadd
                    fyear = nfa * xadd
                elif nf2 == 2:
                    cal.mmonth = cal.mmonth + nfa * xadd
                    if nfa == 1:
                        dmon = dmon + nfa * xadd
                    fmon = nfa * xadd
                elif nf2 == 3:
                    cal.mday = cal.mday + nfa * xadd
                    if nfa == 1:
                        dday = dday + nfa * xadd
                    fday = nfa * xadd
                else:
                    cal.msec = cal.msec + nfa * xadd
                    if nfa == 1:
                        dsec = dsec + nfa * xadd
                    fsec = nfa * xadd
                cal.calendar()

                # update_date add then subtract returns the original date
                yearc, monc, dayc, secc = cal.myear, cal.mmonth, cal.mday, cal.msec
                timesecsc = cal.timesecs
                yearv, monv, dayv, secv = cal.update_date(
                    yearc, monc, dayc, secc, dyear=fyear, dmon=fmon, dday=fday, dsec=fsec
                )
                # ERROR6a: the add must change the date. Fortran also compares
                # timesecsv(6), which this block never sets.
                if (yearv, monv, dayv, secv) == (yearc, monc, dayc, secc):
                    fail[5] = True
                yearv, monv, dayv, secv = cal.update_date(
                    yearv, monv, dayv, secv, dyear=-fyear, dmon=-fmon, dday=-fday, dsec=-fsec
                )
                timesecsv = cal.calendar_date2time(yearc, monc, dayc, secc)
                if (yearv, monv, dayv, secv) != (yearc, monc, dayc, secc) or timesecsc != timesecsv:
                    fail[5] = True

                # date2time then time2date, reference is half the date
                yi = cal.myear // 2
                mi = max(cal.mmonth // 2, 1)
                di = max(cal.mday * 7 // 8, 1)
                si = max(cal.msec * 7 // 8, 1)
                timesecsv = cal.calendar_date2time(
                    cal.myear, cal.mmonth, cal.mday, cal.msec, yi, mi, di, si
                )
                yearv, monv, dayv, secv = cal.calendar_time2date(timesecsv, yi, mi, di, si)
                if (yearv, monv, dayv, secv) != (cal.myear, cal.mmonth, cal.mday, cal.msec):
                    fail[3] = True
        if nfa == 1:
            expect = ADVANCE_DATE[calstr]
        else:
            expect = (1000, 1, 1, 0)
        if (cal.myear, cal.mmonth, cal.mday, cal.msec) != expect:
            fail[2] = True

    # one update_date of the summed deltas, then the reverse
    yearv, monv, dayv, secv = cal.update_date(
        1000, 1, 1, 0, dyear=dyear, dmon=dmon, dday=dday, dsec=dsec
    )
    if (yearv, monv, dayv, secv) != BIG_DATE[calstr]:
        fail[4] = True
    yearv, monv, dayv, secv = cal.update_date(
        yearv, monv, dayv, secv, dyear=-dyear, dmon=-dmon, dday=-dday, dsec=-dsec
    )
    if (yearv, monv, dayv, secv) != (1000, 1, 1, 0):
        fail[4] = True

    # add one month to the last day of each month
    for year in (1900, 1999, 2000, 2004, 2005):
        for nm in range(1, cal.months_per_year + 1):
            tdaymo, _tdaycal, _tdayyr = cal.compute_calendar_data(year)
            dayv = tdaymo[nm - 1]
            next_len = tdaymo[nm % cal.months_per_year]
            if next_len >= tdaymo[nm - 1]:
                monc = nm % cal.months_per_year + 1
                dayc = dayv
            else:
                monc = (nm + 1) % cal.months_per_year + 1
                dayc = tdaymo[nm - 1] - next_len
            yearc = year + 1 if monc < nm else year
            yearv, monv, dayv, secv = cal.update_date(year, nm, dayv, 0, dmon=1)
            if (yearv, monv, dayv, secv) != (yearc, monc, dayc, 0):
                fail[6] = True
    for ns1 in range(0, cal.seconds_per_day + 1, cal.seconds_per_day // 4):
        for ns2 in range(0, cal.seconds_per_day + 1, cal.seconds_per_day // 4):
            tdaymo, _tdaycal, _tdayyr = cal.compute_calendar_data(2002)
            dayv = tdaymo[2]
            got = cal.calendar_date2time(2002, 3, dayv, ns2, 2002, 3, dayv, ns1)
            if got != ns2 - ns1:
                fail[6] = True
            got = cal.calendar_date2time(2002, 4, 1, ns2, 2002, 3, dayv, ns1)
            if got != ns2 - ns1 + cal.seconds_per_day:
                fail[6] = True

    # calc_timesteps
    cal.myear = 2000
    cal.mmonth = 2
    cal.mday = 1
    cal.msec = 0
    units = (
        ("1", 10, 10),
        ("s", 36000, 10),
        ("h", 10, 10),
        ("d", 10, 240),
        ("m", 10, NPT_MONTH[calstr]),
        ("y", 10, NPT_YEAR[calstr]),
    )
    for npt_unit, npt, nptc in units:
        cal.npt = npt
        cal.npt_unit = npt_unit
        cal.dt = 3600.0
        cal.calc_timesteps()
        if cal.npt != nptc:
            fail[7] = True

    # only the last second must be 23:59:59
    hh = mm = ss = None
    for ns1 in range(86400):
        hh, mm, ss = cal.calendar_sec2hms(ns1)
    if (hh, mm, ss) != (23, 59, 59):
        fail[8] = True
    return fail


def main():
    calendars = (
        ("noleap", False, 365),
        ("360day", False, 360),
        ("gregorian", True, 365),
    )
    npass = 0
    nfail = 0
    for calstr, leap, dpy in calendars:
        flags = _run_one(calstr, leap, dpy)
        for name, bad in zip(TESTNAMES, flags):
            if bad:
                nfail += 1
                print(f"FAIL {calstr} {name}")
            else:
                npass += 1
                print(f"PASS {calstr} {name}")
    print(f"total pass = {npass}")
    print(f"total fail = {nfail}")
    return 1 if nfail else 0


if __name__ == "__main__":
    sys.exit(main())
