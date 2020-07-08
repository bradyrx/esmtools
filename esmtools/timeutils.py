import cftime
import numpy as np
import pandas as pd
import xarray as xr
from xarray.core.common import contains_cftime_datetimes, is_np_datetime_like

from .constants import DAYS_PER_MONTH, DAYS_PER_YEAR


@xr.register_dataarray_accessor('timeutils')
class TimeUtilAccessor:
    """Accessor for cftime, datetime, and timeoffset indexed DataArrays. This aids
    in modifying time axes for slope correction and for converting to numeric time."""

    def __init__(self, xarray_obj):
        self._obj = xarray_obj
        self._is_temporal = contains_datetime_like_objects(self._obj)

    @property
    def annual_factor(self):
        return DAYS_PER_YEAR[self.calendar]

    @property
    def calendar(self):
        return get_calendar(self._obj)

    @property
    def freq(self):
        if self.is_temporal:
            return infer_freq(self._obj)
        else:
            return None

    @property
    def is_cftime_like(self):
        if (self.is_temporal) and (contains_cftime_datetimes(self._obj)):
            return True
        else:
            return False

    @property
    def is_datetime_like(self):
        if (self.is_temporal) and (is_np_datetime_like(self._obj)):
            return True
        else:
            return False

    @property
    def is_temporal(self):
        return self._is_temporal

    @property
    def slope_factor(self):
        if self.freq is None:
            return 1.0
        else:
            slope_factors = self.construct_slope_factors()
            return slope_factors[self.freq]

    def return_numeric_time(self):
        """Returns numeric time."""
        if self.is_datetime_like:
            x = (self._obj - np.datetime64('1990-01-01')) / np.timedelta64(1, 'D')
            return x
        elif self.is_cftime_like:
            x = cftime.date2num(
                self._obj, 'days since 1990-01-01', calendar=self.calendar
            )
            x = xr.DataArray(x, dims=self._obj.dims, coords=self._obj.coords)
            return x
        else:
            raise ValueError('DataArray is not a time array of datetimes or cftimes.')

    @staticmethod
    def construct_quarterly_aliases():
        quarters = ['Q', 'BQ', 'QS', 'BQS']
        for month in [
            'JAN',
            'FEB',
            'MAR',
            'APR',
            'MAY',
            'JUN',
            'JUL',
            'AUG',
            'SEP',
            'OCT',
            'NOV',
            'DEC',
        ]:
            quarters.append(f'Q-{month}')
            quarters.append(f'BQ-{month}')
            quarters.append(f'BQS-{month}')
            quarters.append(f'QS-{month}')
        return quarters

    @staticmethod
    def construct_annual_aliases():
        years = ['A', 'Y', 'BA', 'BY', 'AS', 'YS', 'BAS', 'BYS', 'Q']
        for month in [
            'JAN',
            'FEB',
            'MAR',
            'APR',
            'MAY',
            'JUN',
            'JUL',
            'AUG',
            'SEP',
            'OCT',
            'NOV',
            'DEC',
        ]:
            years.append(f'A-{month}')
            years.append(f'BA-{month}')
            years.append(f'BAS-{month}')
            years.append(f'AS-{month}')
        return years

    def construct_slope_factors(self):
        years = self.construct_annual_aliases()
        years = {k: self.annual_factor for k in years}

        quarters = self.construct_quarterly_aliases()
        quarters = {k: self.annual_factor / 4 for k in quarters}

        months = ('M', 'BM', 'CBM', 'MS', 'BMS', 'CBMS')
        months = {k: self.annual_factor / 12 for k in months}

        semimonths = {k: 15 for k in ('SM', 'SMS')}

        weeks = ('W', 'W-SUN', 'W-MON', 'W-TUE', 'W-WED', 'W-THU', 'W-FRI', 'W-SAT')
        weeks = {k: 7 for k in weeks}

        days = {k: 1 for k in ('B', 'C', 'D')}
        hours = {k: 1 / 24 for k in ('BH', 'H')}
        mins = {k: 1 / (24 * 60) for k in ('T', 'min')}
        secs = {k: 1 / (24 * 60 * 60) for k in ('S')}
        millisecs = {k: 1 / (24 * 60 * 60 * 1e3) for k in ('ms', 'L')}
        microsecs = {k: 1 / (24 * 60 * 60 * 1e6) for k in ('U', 'us')}
        nanosecs = {k: 1 / (24 * 60 * 60 * 1e9) for k in ('N')}

        DATETIME_FACTOR = {}
        for d in (
            years,
            quarters,
            months,
            semimonths,
            weeks,
            days,
            hours,
            mins,
            secs,
            millisecs,
            microsecs,
            nanosecs,
        ):
            DATETIME_FACTOR.update(d)
        return DATETIME_FACTOR


def contains_datetime_like_objects(var):
    """Check if a variable contains datetime like objects (either
    np.datetime64, np.timedelta64, or cftime.datetime)
    """
    return is_np_datetime_like(var.dtype) or contains_cftime_datetimes(var)


def get_calendar(dates):
    """Attempt to pull calendar type automatically using ``cftime`` objects.

    .. note::
        This relies upon ``xarray``'s automatic conversion of time to ``cftime``.

    Args:
        dates (xarray.DataArray): Dates from which to retrieve calendar.

    Returns:
        str: Name of calendar in NetCDF convention.

    Raises:
        ValueError: If inferred calendar is not in our list of supported calendars.
    """
    if np.asarray(dates).dtype == 'datetime64[ns]':
        return 'proleptic_gregorian'
    else:
        return np.asarray(dates).ravel()[0].calendar


def get_days_per_month(time, calendar='standard'):
    """Return an array of days per month corresponding to a given calendar.

    Args:
        x (xarray index): Time index from xarray object.
        calendar (optional str): Calendar type.

    Returns:
        ndarray: Array of number of days for each monthly time step provided.

    Raises:
        ValueError: If input time index is not a CFTimeIndex or DatetimeIndex.
    """
    is_time_index(time, 'time index')
    month_length = np.zeros(len(time), dtype=np.int)

    cal_days = DAYS_PER_MONTH[calendar]

    for i, (month, year) in enumerate(zip(time.month, time.year)):
        month_length[i] = cal_days[month]
        if leap_year(year, calendar=calendar):
            month_length[i] += 1
    return month_length


def is_time_index(xobj, kind):
    """
    Checks that xobj coming through is a DatetimeIndex or CFTimeIndex.

    This checks that `esmtools` is converting the DataArray to an index,
    i.e. through .to_index()
    """
    xtype = type(xobj).__name__
    if xtype not in ['CFTimeIndex', 'DatetimeIndex']:
        raise ValueError(
            f'Your {kind} object must be either an xr.CFTimeIndex or '
            f'pd.DatetimeIndex.'
        )
    return True


def leap_year(year, calendar='standard'):
    """Determine if year is a leap year.

    Args:
        year (int): Year to assess.
        calendar (optional str): Calendar type.

    Returns:
        bool: True if year is a leap year.
    """
    leap = False
    if (calendar in ['standard', 'gregorian', 'proleptic_gregorian', 'julian']) and (
        year % 4 == 0
    ):
        leap = True
        if (
            (calendar == 'proleptic_gregorian')
            and (year % 100 == 0)
            and (year % 400 != 0)
        ):
            leap = False
        elif (
            (calendar in ['standard', 'gregorian'])
            and (year % 100 == 0)
            and (year % 400 != 0)
            and (year < 1583)
        ):
            leap = False
    return leap


def infer_freq(index):
    """NOTE: This is pulled from xarray v0.15.2, which isn't released yet. I want
    to avoid making a requirement the master unreleased branch. We'll switch this
    simply to `xr.infer_freq()` once it's released."""
    from xarray.core.dataarray import DataArray

    if isinstance(index, (DataArray, pd.Series)):
        dtype = np.asarray(index).dtype
        if dtype == 'datetime64[ns]':
            index = pd.DatetimeIndex(index.values)
        elif dtype == 'timedelta64[ns]':
            index = pd.TimedeltaIndex(index.values)
        else:
            index = xr.CFTimeIndex(index.values)

    if isinstance(index, xr.CFTimeIndex):
        inferer = _CFTimeFrequencyInferer(index)
        return inferer.get_freq()

    return pd.infer_freq(index)


_ONE_MICRO = 1
_ONE_MILLI = _ONE_MICRO * 1000
_ONE_SECOND = _ONE_MILLI * 1000
_ONE_MINUTE = 60 * _ONE_SECOND
_ONE_HOUR = 60 * _ONE_MINUTE
_ONE_DAY = 24 * _ONE_HOUR
_MONTH_ABBREVIATIONS = {
    1: 'JAN',
    2: 'FEB',
    3: 'MAR',
    4: 'APR',
    5: 'MAY',
    6: 'JUN',
    7: 'JUL',
    8: 'AUG',
    9: 'SEP',
    10: 'OCT',
    11: 'NOV',
    12: 'DEC',
}


class _CFTimeFrequencyInferer:
    def __init__(self, index):
        self.index = index
        self.values = index.asi8

        if len(index) < 3:
            raise ValueError('Need at least 3 dates to infer frequency')

        self.is_monotonic = (
            self.index.is_monotonic_decreasing or self.index.is_monotonic_increasing
        )

        self._deltas = None
        self._year_deltas = None
        self._month_deltas = None

    def get_freq(self):
        if not self.is_monotonic or not self.index.is_unique:
            return None

        delta = self.deltas[0]  # Smallest delta
        if _is_multiple(delta, _ONE_DAY):
            return self._infer_daily_rule()
        # There is no possible intraday frequency with a non-unique delta
        # Different from pandas: we don't need to manage DST and business offsets
        # in cftime
        elif not len(self.deltas) == 1:
            return None

        if _is_multiple(delta, _ONE_HOUR):
            return _maybe_add_count('H', delta / _ONE_HOUR)
        elif _is_multiple(delta, _ONE_MINUTE):
            return _maybe_add_count('T', delta / _ONE_MINUTE)
        elif _is_multiple(delta, _ONE_SECOND):
            return _maybe_add_count('S', delta / _ONE_SECOND)
        elif _is_multiple(delta, _ONE_MILLI):
            return _maybe_add_count('L', delta / _ONE_MILLI)
        else:
            return _maybe_add_count('U', delta / _ONE_MICRO)

    def _infer_daily_rule(self):
        annual_rule = self._get_annual_rule()
        if annual_rule:
            nyears = self.year_deltas[0]
            month = _MONTH_ABBREVIATIONS[self.index[0].month]
            alias = f'{annual_rule}-{month}'
            return _maybe_add_count(alias, nyears)

        quartely_rule = self._get_quartely_rule()
        if quartely_rule:
            nquarters = self.month_deltas[0] / 3
            mod_dict = {0: 12, 2: 11, 1: 10}
            month = _MONTH_ABBREVIATIONS[mod_dict[self.index[0].month % 3]]
            alias = f'{quartely_rule}-{month}'
            return _maybe_add_count(alias, nquarters)

        monthly_rule = self._get_monthly_rule()
        if monthly_rule:
            return _maybe_add_count(monthly_rule, self.month_deltas[0])

        if len(self.deltas) == 1:
            # Daily as there is no "Weekly" offsets with CFTime
            days = self.deltas[0] / _ONE_DAY
            return _maybe_add_count('D', days)

        # CFTime has no business freq and no "week of month" (WOM)
        return None

    def _get_annual_rule(self):
        if len(self.year_deltas) > 1:
            return None

        if len(np.unique(self.index.month)) > 1:
            return None

        return {'cs': 'AS', 'ce': 'A'}.get(month_anchor_check(self.index))

    def _get_quartely_rule(self):
        if len(self.month_deltas) > 1:
            return None

        if not self.month_deltas[0] % 3 == 0:
            return None

        return {'cs': 'QS', 'ce': 'Q'}.get(month_anchor_check(self.index))

    def _get_monthly_rule(self):
        if len(self.month_deltas) > 1:
            return None

        return {'cs': 'MS', 'ce': 'M'}.get(month_anchor_check(self.index))

    @property
    def deltas(self):
        """Sorted unique timedeltas as microseconds."""
        if self._deltas is None:
            self._deltas = _unique_deltas(self.values)
        return self._deltas

    @property
    def year_deltas(self):
        """Sorted unique year deltas."""
        if self._year_deltas is None:
            self._year_deltas = _unique_deltas(self.index.year)
        return self._year_deltas

    @property
    def month_deltas(self):
        """Sorted unique month deltas."""
        if self._month_deltas is None:
            self._month_deltas = _unique_deltas(self.index.year * 12 + self.index.month)
        return self._month_deltas


def _is_multiple(us, mult: int):
    """Whether us is a multiple of mult"""
    return us % mult == 0


def _maybe_add_count(base: str, count: float):
    """If count is greater than 1, add it to the base offset string"""
    if count != 1:
        assert count == int(count)
        count = int(count)
        return f'{count}{base}'
    else:
        return base


def month_anchor_check(dates):
    """Return the monthly offset string.
    Return "cs" if all dates are the first days of the month,
    "ce" if all dates are the last day of the month,
    None otherwise.
    Replicated pandas._libs.tslibs.resolution.month_position_check
    but without business offset handling.
    """
    calendar_end = True
    calendar_start = True

    for date in dates:
        if calendar_start:
            calendar_start &= date.day == 1

        if calendar_end:
            cal = date.day == date.daysinmonth
            if calendar_end:
                calendar_end &= cal
        elif not calendar_start:
            break

    if calendar_end:
        return 'ce'
    elif calendar_start:
        return 'cs'
    else:
        return None


def _unique_deltas(arr):
    """Sorted unique deltas of numpy array"""
    return np.sort(np.unique(np.diff(arr)))
