import cftime
import numpy as np
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
            return xr.infer_freq(self._obj)
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
