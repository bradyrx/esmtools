import cftime
import numpy as np
import xarray as xr

from .checks import is_time_index
from .constants import CALENDARS, CFTIME_TO_NETCDF, DAYS_PER_MONTH


@xr.register_dataarray_accessor('timeutils')
class TimeUtilAccessor:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj
        try:
            self._type = type(self._obj.to_index()).__name__
        except ValueError:
            self._type = 'gridded'

    @property
    def annual_factor(self):
        if self.calendar == 'all_leap':
            return 366.0
        elif self.calendar == 'noleap':
            return 365.0
        else:
            return 365.25

    @property
    def calendar(self):
        if self.type == 'CFTimeIndex':
            return get_calendar(self._obj)
        else:
            return 'gregorian'

    @property
    def freq(self):
        if self.type in ['DatetimeIndex', 'CFTimeIndex']:
            return xr.infer_freq(self._obj)
        else:
            return None

    @property
    def slope_factor(self):
        if self.freq is None:
            return 1.0
        else:
            slope_factors = self.construct_slope_factors()
            return slope_factors[self.freq]

    @property
    def type(self):
        return self._type

    def return_numeric_time(self):
        """Returns numeric time."""
        if self.type == 'DatetimeIndex':
            x = (self._obj - np.datetime64('1990-01-01')) / np.timedelta64(1, 'D')
            return x
        elif self.type == 'CFTimeIndex':
            calendar = get_calendar(self._obj)
            x = cftime.date2num(self._obj, 'days since 1990-01-01', calendar=calendar)
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


def get_calendar(ds, dim=None):
    """Attempt to pull calendar type automatically using ``cftime`` objects.

    .. note::
        This relies upon ``xarray``'s automatic conversion of time to ``cftime``.

    Args:
        ds (xarray object): Dataset to retrieve calendar from.
        dim (optional str): Name of time dimension.

    Returns:
        str: Name of calendar in NetCDF convention.

    Raises:
        ValueError: If inferred calendar is not in our list of supported calendars.
    """
    if dim is not None:
        example_time = ds[dim].values[0]
    else:
        example_time = ds.values[0]
    # Type of variable being used for time.
    var_type = type(example_time).__name__
    if var_type in CFTIME_TO_NETCDF:
        # If this is a `cftime` object, infer what type of calendar it is.
        return CFTIME_TO_NETCDF[var_type]
    else:
        raise ValueError(f'Please submit a calendar from {CALENDARS}')


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
