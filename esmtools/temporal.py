# Most of this module comes from the following example:
# http://xarray.pydata.org/en/stable/examples/monthly-means.html
import numpy as np
import xarray as xr


# This supports all calendars used in netCDF.
dpm = {
    "noleap": [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
    "365_day": [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
    "standard": [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
    "gregorian": [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
    "proleptic_gregorian": [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
    "all_leap": [0, 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
    "366_day": [0, 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
    "360_day": [0, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30],
    "julian": [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
}
# Converts from `cftime` class name to netCDF convention for calendar
cftime_to_netcdf = {
    "DatetimeJulian": "julian",
    "DatetimeProlepticGregorian": "proleptic_gregorian",
    "DatetimeNoLeap": "noleap",
    "DatetimeAllLeap": "all_leap",
    "DatetimeGregorian": "gregorian",
}
CALENDARS = [k for k in dpm]
resolutions = {"annual": "time.year"}
RESOLUTIONS = [k for k in resolutions]


def _leap_year(year, calendar="standard"):
    """Determine if year is a leap year"""
    leap = False
    if (calendar in ["standard", "gregorian", "proleptic_gregorian", "julian"]) and (
        year % 4 == 0
    ):
        leap = True
        if (
            (calendar == "proleptic_gregorian")
            and (year % 100 == 0)
            and (year % 400 != 0)
        ):
            leap = False
        elif (
            (calendar in ["standard", "gregorian"])
            and (year % 100 == 0)
            and (year % 400 != 0)
            and (year < 1583)
        ):
            leap = False
    return leap


def _get_dpm(time, calendar="standard"):
    """
    return a array of days per month corresponding to the months provided in `months`
    """
    month_length = np.zeros(len(time), dtype=np.int)

    cal_days = dpm[calendar]

    for i, (month, year) in enumerate(zip(time.month, time.year)):
        month_length[i] = cal_days[month]
        if _leap_year(year, calendar=calendar):
            month_length[i] += 1
    return month_length


def _retrieve_calendar(ds, dim="time"):
    """Attempt to pull calendar type automatically using cftime objects.

    .. note::
        This relies upon ``xarray``'s automatic conversion of time to ``cftime``.

    Args:
        ds (xarray object): Dataset being resampled.
        dim (optional str): Time dimension.
    """
    example_time = ds[dim].values[0]
    # Type of variable being used for time.
    var_type = type(example_time).__name__
    if var_type in cftime_to_netcdf:
        # If this is a `cftime` object, infer what type of calendar it is.
        return cftime_to_netcdf[var_type]
    else:
        raise ValueError(f"Please submit a calendar from {CALENDARS}")


def _weighted_resample(ds, calendar=None, dim="time", resample_resolution=None):
    """Generalized function for time-weighted resampling.

    Args:
        ds (xarray object): Dataset to resample.
        calendar (str): Calendar type (see wrapper functions).
        dim (str): Name of time dimension.
        resample_resolution (str): Temporal resolution to resample to
            * 'annual'

    Returns:
        ds_weighted (xarray object): Variable(s) resampled to the desired temporal
                                     resolution with weighting.
    """
    if (calendar is None) or (calendar not in CALENDARS):
        calendar = _retrieve_calendar(ds, dim=dim)

    if resample_resolution not in RESOLUTIONS:
        raise ValueError(f"Please submit a temporal resolution from {RESOLUTIONS}")

    time_length = xr.DataArray(
        _get_dpm(ds.time.to_index(), calendar=calendar),
        coords=[ds.time],
        name="time_length",
    )

    time_res = resolutions[resample_resolution]
    # Get weights of each time unit (e.g., daily, monthly)
    weights = time_length.groupby(time_res) / time_length.groupby(time_res).sum()

    # Assert that the sum of the weights for each year is 1.0.
    weights_sum = weights.groupby(time_res).sum().values
    np.testing.assert_allclose(weights_sum, np.ones(len(weights_sum)))

    # Calculate the weighted average
    ds_weighted = (ds * weights).groupby(time_res).sum(dim=dim)
    return ds_weighted


def to_annual(ds, calendar=None, how="mean", dim="time"):
    """Resample sub-annual temporal resolution to annual resolution with weighting.

    .. note::
        Using ``pandas.groupby()`` still performs an arithmetic mean. This function
        properly weights, e.g., February is weighted at 28/365 if going from monthly
        to annual.

    Args:
        ds (xarray object): Dataset or DataArray with data to be temporally averaged.
        calendar (str): Calendar type for data. If None and `ds` is in `cftime`, infer
                        calendar type.

            * 'noleap'/'365_day': Gregorian calendar without leap years
                                  (all are 365 days long).
            * 'gregorian'/'standard': Mixed Gregorian/Julian calendar. 1582-10-05 to
                                      1582-10-14 don't exist, because people are crazy.
                                      Nor does year 0.
            * 'proleptic_gregorian': A Gregorian calendar extended to dates before
                                     1582-10-15.
            * 'all_leap'/'366_day': Gregorian calendar with every year being a leap
                                    year (all years are 366 days long).
            * '360_day': All years are 360 days divided into 30 day months.
            * 'julian': Standard Julian calendar.
        how (optional str): How to convert to annual. Currently only `mean` is
                            supported, but we plan to add `sum` as well.
        dim (optional str): Dimension to apply resampling over (default 'time').

    Returns:
        ds_weighted (xarray object): Dataset or DataArray resampled to annual resolution
    """
    if how != "mean":
        raise NotImplementedError(
            "Only annual-weighted averaging is currently"
            + "supported. Please change `how` keyword to 'mean'"
        )
    return _weighted_resample(
        ds, calendar=calendar, dim=dim, resample_resolution="annual"
    )
