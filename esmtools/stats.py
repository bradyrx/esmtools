import warnings

import climpred.stats as st
import numpy as np
import numpy.polynomial.polynomial as poly
import scipy
import xarray as xr

from .checks import is_xarray
from .timeutils import TimeUtilAccessor
from .utils import get_dims


@is_xarray(0)
def standardize(ds, dim='time'):
    """Standardize Dataset/DataArray

    .. math::
        \\frac{x - \\mu_{x}}{\\sigma_{x}}

    Args:
        ds (xarray object): Dataset or DataArray with variable(s) to standardize.
        dim (optional str): Which dimension to standardize over (default 'time').

    Returns:
        stdized (xarray object): Standardized variable(s).
    """
    stdized = (ds - ds.mean(dim)) / ds.std(dim)
    return stdized


@is_xarray(0)
def nanmean(ds, dim='time'):
    """Compute mean NaNs and suppress warning from numpy"""
    if 'time' in ds.dims:
        mask = ds.isnull().isel(time=0)
    else:
        mask = ds.isnull()
    ds = ds.fillna(0).mean(dim)
    ds = ds.where(~mask)
    return ds


@is_xarray(0)
def cos_weight(da, lat_coord='lat', lon_coord='lon', one_dimensional=True):
    """
    Area-weights data on a regular (e.g. 360x180) grid that does not come with
    cell areas. Uses cosine-weighting.

    Parameters
    ----------
    da : DataArray with longitude and latitude
    lat_coord : str (optional)
        Name of latitude coordinate
    lon_coord : str (optional)
        Name of longitude coordinate
    one_dimensional : bool (optional)
        If true, assumes that lat and lon are 1D (i.e. not a meshgrid)

    Returns
    -------
    aw_da : Area-weighted DataArray

    Examples
    --------
    import esmtools as et
    da_aw = et.stats.reg_aw(SST)
    """
    non_spatial = [i for i in get_dims(da) if i not in [lat_coord, lon_coord]]
    filter_dict = {}
    while len(non_spatial) > 0:
        filter_dict.update({non_spatial[0]: 0})
        non_spatial.pop(0)
    if one_dimensional:
        lon, lat = np.meshgrid(da[lon_coord], da[lat_coord])
    else:
        lat = da[lat_coord]
    # NaN out land to not go into area-weighting
    lat = lat.astype('float')
    nan_mask = np.asarray(da.isel(filter_dict).isnull())
    lat[nan_mask] = np.nan
    cos_lat = np.cos(np.deg2rad(lat))
    aw_da = (da * cos_lat).sum(lat_coord).sum(lon_coord) / np.nansum(cos_lat)
    return aw_da


@is_xarray(0)
def area_weight(da, area_coord='area'):
    """
    Returns an area-weighted time series from the input xarray dataarray. This
    automatically figures out spatial dimensions vs. other dimensions. I.e.,
    this function works for just a single realization or for many realizations.
    See `reg_aw` if you have a regular (e.g. 360x180) grid that does not
    contain cell areas.

    It also looks like xarray is implementing a feature like this.

    NOTE: This currently does not support datasets (of multiple variables)
    The user can alleviate this by using the .apply() function.

    Parameters
    ----------
    da : DataArray
    area_coord : str (defaults to 'area')
        Name of area coordinate if different from 'area'

    Returns
    -------
    aw_da : Area-weighted DataArray
    """
    area = da[area_coord]
    # Mask the area coordinate in case you've got a bunch of NaNs, e.g. a mask
    # or land.
    dimlist = get_dims(da)
    # Pull out coordinates that aren't spatial. Time, ensemble members, etc.
    non_spatial = [i for i in dimlist if i not in get_dims(area)]
    filter_dict = {}
    while len(non_spatial) > 0:
        filter_dict.update({non_spatial[0]: 0})
        non_spatial.pop(0)
    masked_area = area.where(da.isel(filter_dict).notnull())
    # Compute area-weighting.
    dimlist = get_dims(masked_area)
    aw_da = da * masked_area
    # Sum over arbitrary number of dimensions.
    while len(dimlist) > 0:
        print(f'Summing over {dimlist[0]}')
        aw_da = aw_da.sum(dimlist[0])
        dimlist.pop(0)
    # Finish area-weighting by dividing by sum of area coordinate.
    aw_da = aw_da / masked_area.sum()
    return aw_da


def _preprocess_x_and_y(x, y, dim):
    """Preprocesses ``x`` and ``y`` for regression functions.

    This checks that ``y`` is not the independent variable and converts the time
    dimension to numeric units to account for differences in, e.g., lengths of months.
    """
    if isinstance(y, xr.DataArray) and (y.name == dim):
        raise ValueError(
            f'Dependent variable y should not be the same as the dim {dim} being '
            'applied over. Change your y variable to x.'
        )

    # If independent variable is a datetime axis (computing stats with reference to
    # time), this converts to numeric time to account for differences in length of
    # months, leap years, etc.
    slope_factor = 1.0
    if isinstance(x, xr.DataArray):
        x_type = x.timeutils.type
        if x_type in ['DatetimeIndex', 'CFTimeIndex']:
            slope_factor = x.timeutils.slope_factor
            x = x.timeutils.return_numeric_time()
    return x, y, slope_factor


def warn_if_not_converted_to_original_time_units(x):
    if isinstance(x, xr.DataArray):
        x_type = x.timeutils.type
        if x_type in ['DatetimeIndex', 'CFTimeIndex']:
            freq = x.timeutils.freq
            if freq is None:
                warnings.warn(
                    'Datetime frequency not detected. Slope and std. errors will be '
                    'in original units per day (e.g., degC per day). Multiply by '
                    'e.g., 365.25 to convert to original units per year.'
                )


@is_xarray([0, 1])
def linear_slope(x, y, dim='time'):
    """Returns the linear slope with y regressed onto x.

    Args:
        x (xarray object): Independent variable (predictor) for linear regression.
        y (xarray object): Dependent variable (predictand) for linear regression.
        dim (str, optional): Dimension to apply linear regression over.
            Defaults to "time".

    Returns:
        xarray object: Slopes computed through a least-squares linear regression.
    """
    x, y, slope_factor = _preprocess_x_and_y(x, y, dim)

    slopes = xr.apply_ufunc(
        lambda x, y: np.polyfit(x, y, 1)[0],
        x,
        y,
        vectorize=True,
        dask='parallelized',
        input_core_dims=[[dim], [dim]],
        output_dtypes=['float64'],
    )
    warn_if_not_converted_to_original_time_units(x)
    return slopes * slope_factor


@is_xarray([0, 1])
def linregress(x, y, dim='time'):
    """Applies the `scipy.stats` linregress to a grid."""
    x, y, slope_factor = _preprocess_x_and_y(x, y, dim)

    def _linregress(x, y, slope_factor):
        m, b, r, p, e = scipy.stats.linregress(x, y)
        # Multiply slope by factor. If time indices converted to numeric units, this
        # gets them back to the original units.
        m *= slope_factor
        e *= slope_factor
        return np.array([m, b, r, p, e])

    results = xr.apply_ufunc(
        _linregress,
        x,
        y,
        slope_factor,
        vectorize=True,
        dask='parallelized',
        input_core_dims=[[dim], [dim], []],
        output_core_dims=[['parameter']],
        output_dtypes=['float64'],
        output_sizes={'parameter': 5},
    )
    results = results.assign_coords(
        parameter=['slope', 'intercept', 'rvalue', 'pvalue', 'stderr']
    )
    warn_if_not_converted_to_original_time_units(x)
    return results


@is_xarray(0)
def polyfit(x, y, order, dim='time'):
    """Returns the fitted polynomial line of order N

    .. note::
        This automatically interpolates across NaNs to make the fit.

    Args:
        ds (xarray object): Time series to fit  polynomial to.
        order (int): Order of polynomial fit.
        dim (optional str): Dimension over which to fit hte polynomial.

    Returns:
        xarray object with polynomial fit.

    References:
        This is a modification of @ahuang11's script `rm_poly` in `climpred`.
    """
    x, y, _ = _preprocess_x_and_y(x, y, dim)

    def _polyfit(x, y, order):
        coefs = poly.polyfit(x, y, order)
        fit = poly.polyval(x, coefs)
        return fit

    return xr.apply_ufunc(
        _polyfit,
        x,
        y,
        order,
        vectorize=True,
        dask='parallelized',
        input_core_dims=[[dim], [dim], []],
        output_core_dims=[[dim]],
        output_dtypes=['float64'],
    )


@is_xarray(0)
def corr(x, y, dim='time', lead=0, return_p=False):
    """
    Computes the Pearson product-momment coefficient of linear correlation.

    This version calculates the effective degrees of freedom, accounting
    for autocorrelation within each time series that could fluff the
    significance of the correlation.

    Parameters
    ----------
    x, y : xarray DataArray
        time series being correlated (can be multi-dimensional)
    dim : str (default 'time')
        Correlation dimension
    lead : int (default 0)
        If lead > 0, x leads y by that many time steps.
        If lead < 0, x lags y by that many time steps.
    return_p : boolean (default False)
        If true, return both r and p

    Returns
    -------
    r : correlation coefficient
    p : p-value accounting for autocorrelation (if return_p True)

    References (for dealing with autocorrelation):
    ----------
    1. Wilks, Daniel S. Statistical methods in the atmospheric sciences.
    Vol. 100. Academic press, 2011.
    2. Lovenduski, Nicole S., and Nicolas Gruber. "Impact of the Southern
    Annular Mode on Southern Ocean circulation and biology." Geophysical
    Research Letters 32.11 (2005).
    3. Brady, R. X., Lovenduski, N. S., Alexander, M. A., Jacox, M., and
    Gruber, N.: On the role of climate modes in modulating the air-sea CO2
    fluxes in Eastern Boundary Upwelling Systems, Biogeosciences Discuss.,
    https://doi.org/10.5194/bg-2018-415, in review, 2018.
    """
    # Broadcasts a time series to the same coordinates/size as the grid. If they
    # are both grids, this function does nothing and isn't expensive.
    x, y = xr.broadcast(x, y)

    # Negative lead should have y lead x.
    if lead < 0:
        lead = np.abs(lead)
        return st.corr(y, x, dim=dim, lag=lead, return_p=return_p)
    else:
        return st.corr(x, y, dim=dim, lag=lead, return_p=return_p)


@is_xarray(0)
def rm_poly(x, y, order, dim='time'):
    """
    Update description.
    """
    x, y, _ = _preprocess_x_and_y(x, y, dim)

    def _rm_poly(x, y, order):
        coefs = poly.polyfit(x, y, order)
        fit = poly.polyval(x, coefs)
        detrended = y - fit
        return detrended

    return xr.apply_ufunc(
        _rm_poly,
        x,
        y,
        order,
        vectorize=True,
        dask='parallelized',
        input_core_dims=[[dim], [dim], []],
        output_core_dims=[[dim]],
        output_dtypes=['float64'],
    )


@is_xarray(0)
def rm_trend(x, y, dim='time'):
    """
    Update description.
    """
    return rm_poly(x, y, 1, dim=dim)


@is_xarray(0)
def autocorr(ds, lag=1, dim='time', return_p=False):
    """
    Calculated lagged correlation of a xr.Dataset.
    Parameters
    ----------
    ds : xarray dataset/dataarray
    lag : int (default 1)
        number of time steps to lag correlate.
    dim : str (default 'time')
        name of time dimension/dimension to autocorrelate over
    return_p : boolean (default False)
        if false, return just the correlation coefficient.
        if true, return both the correlation coefficient and p-value.
    Returns
    -------
    r : Pearson correlation coefficient
    p : (if return_p True) p-value
    """
    return st.autocorr(ds, lag=lag, dim=dim, return_p=return_p)


@is_xarray(0)
def ACF(ds, dim='time', nlags=None):
    """
    Compute the ACF of a time series to a specific lag.

    Args:
      ds (xarray object): dataset/dataarray containing the time series.
      dim (str): dimension to apply ACF over.
      nlags (optional int): number of lags to compute ACF over. If None,
                            compute for length of `dim` on `ds`.

    Returns:
      Dataset or DataArray with ACF results.

    Notes:
      This is preferred over ACF functions from MATLAB/scipy, since it doesn't
      use FFT methods.
    """
    # Drop variables that don't have requested dimension, so this can be
    # applied over the full dataset.
    if isinstance(ds, xr.Dataset):
        dropVars = [i for i in ds if dim not in ds[i].dims]
        ds = ds.drop(dropVars)

    # Loop through every step in `dim`
    if nlags is None:
        nlags = ds[dim].size

    acf = []
    # The 2 factor accounts for fact that time series reduces in size for
    # each lag.
    for i in range(nlags - 2):
        res = autocorr(ds, lag=i, dim=dim)
        acf.append(res)
    acf = xr.concat(acf, dim=dim)
    return acf
