# Most of this module comes from the following example:
# http://xarray.pydata.org/en/stable/examples/monthly-means.html
import numpy as np
import xarray as xr

from .constants import CALENDARS
from .timeutils import get_calendar, get_days_per_month

GROUPBY_TIMES = {"annual": "time.year"}
TIME_RESOLUTIONS = [k for k in GROUPBY_TIMES]


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
        calendar = get_calendar(ds[dim])

    if resample_resolution not in TIME_RESOLUTIONS:
        raise ValueError(f"Please submit a temporal resolution from {TIME_RESOLUTIONS}")

    time_length = xr.DataArray(
        get_days_per_month(ds.time.to_index(), calendar=calendar),
        coords=[ds.time],
        name="time_length",
    )

    time_res = GROUPBY_TIMES[resample_resolution]
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
