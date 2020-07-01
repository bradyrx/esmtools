import warnings

import cftime
import numpy as np
import xarray as xr
from pandas.core.indexes.datetimes import DatetimeIndex

from .checks import is_time_index
from .constants import CALENDARS, CFTIME_TO_NETCDF, DAYS_PER_MONTH


def convert_time(x, dim):
    """Convert independent time axis into a numeric axis for statistics.

    .. note::
        If the independent axis is a datetime object, this converts it to
        days since a reference period. This allows for fitting to irregularly spaced
        time steps. Otherwise, the axis is left as is.

    Args:
        x (xarray DataArray): Independent axis that statistics are being applied to.
        dim (str): Name of time dimension.

    Returns:
        xarray object: If a datetime index, this returns it modified into numerical
        values for applying statistics. If a numeric index, this returns it unmodified.
    """
    # NOTE: Can use the new `xr.infer_freq` function here to change the "days since"
    # of np.timedelta64 part. Then can return in proper slope units per time step
    # (could add attribute with units for this).
    if isinstance(x.to_index(), DatetimeIndex):
        x = (x - np.datetime64('1990-01-01')) / np.timedelta64(1, 'D')
    elif isinstance(x.to_index(), xr.CFTimeIndex):
        calendar = get_calendar(x, dim)
        warnings.warn(f'Assuming calendar type {calendar}.')
        x = cftime.date2num(x, 'days since 1990-01-01', calendar=calendar)
    else:  # Numeric case, e.g. integer years.
        x = x
    return x


def get_calendar(ds, dim='time'):
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
    example_time = ds[dim].values[0]
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


def get_dims(da):
    """
    Simple function to retrieve dimensions from a given dataset/datarray.
    Currently returns as a list, but can add keyword to select tuple or
    list if desired for any reason.
    """
    return list(da.dims)


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
