"""
Stats module. This serves as a wrapper for the most desired functions
from climpred.

Area-weighting
------------
`xr_cos_weight`: Area-weights output or observations without grid cell area
                 information using cosine weighting.
`xr_area_weight`: Area-weights output with grid cell area information.

Time Series
-----------
`xr_smooth_series` : Returns a smoothed time series.
`xr_linregress` : Returns results of linear regression over input dataarray.
`xr_rm_poly` : Returns time series with polynomial fit removed.
`xr_rm_trend` : Returns detrended (first order) time series.
`xr_autocorr` : Calculates the autocorrelation of time series over some lag.
`xr_corr` : Computes pearsonr between two time series accounting for
            autocorrelation.
"""
import climpred.stats as st
import numpy as np
import xarray as xr
from scipy.stats import linregress


# --------------------------------------------#
# HELPER FUNCTIONS
# Should only be used internally by esmtools.
# --------------------------------------------#
def _check_xarray(x):
    """
    Check if the object being submitted to a given function is either a
    Dataset or DataArray. This is important since `esmtools` is built as an
    xarray wrapper.
    TODO: Move this to a generalized util.py module with any other functions
    that are being called in other submodules.
    """
    if not (isinstance(x, xr.DataArray) or isinstance(x, xr.Dataset)):
        typecheck = type(x)
        raise IOError(f"""The input data is not an xarray object (an xarray
            DataArray or Dataset). esmtools is built to wrap xarray to make
            use of its awesome features. Please input an xarray object and
            retry the function.
            Your input was of type: {typecheck}""")


def _get_coords(da):
    """
    Simple function to retrieve dimensions from a given dataset/dataarray.
    Currently returns as a list, but can add keyword to select tuple or
    list if desired for any reason.
    """
    return list(da.coords)


def _get_dims(da):
    """
    Simple function to retrieve dimensions from a given dataset/datarray.
    Currently returns as a list, but can add keyword to select tuple or
    list if desired for any reason.
    """
    return list(da.dims)


# --------------------------
# AREA-WEIGHTING DEFINITIONS
# --------------------------
def xr_cos_weight(da, lat_coord='lat', lon_coord='lon', one_dimensional=True):
    """
    Area-weights data on a regular (e.g. 360x180) grid that does not come with
    cell areas. Uses cosine-weighting.
    NOTE: Currently explicitly writing `xr` as a prefix for xarray-specific
    definitions. Since `esmtools` is supposed to be a wrapper for xarray,
    this might be altered in the future.
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
    _check_xarray(da)
    non_spatial = [i for i in _get_dims(da) if i not in [lat_coord, lon_coord]]
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
    aw_da = (da * cos_lat).sum(lat_coord).sum(lon_coord) / \
        np.nansum(cos_lat)
    return aw_da


def xr_area_weight(da, area_coord='area'):
    """
    Returns an area-weighted time series from the input xarray dataarray. This
    automatically figures out spatial dimensions vs. other dimensions. I.e.,
    this function works for just a single realization or for many realizations.
    See `reg_aw` if you have a regular (e.g. 360x180) grid that does not
    contain cell areas.
    NOTE: This currently does not support datasets (of multiple variables)
    The user can alleviate this by using the .apply() function.
    NOTE: Currently explicitly writing `xr` as a prefix for xarray-specific
    definitions. Since `esmtools` is supposed to be a wrapper for xarray,
    this might be altered in the future.
    Parameters
    ----------
    da : DataArray
    area_coord : str (defaults to 'area')
        Name of area coordinate if different from 'area'
    Returns
    -------
    aw_da : Area-weighted DataArray
    """
    _check_xarray(da)
    area = da[area_coord]
    # Mask the area coordinate in case you've got a bunch of NaNs, e.g. a mask
    # or land.
    dimlist = _get_dims(da)
    # Pull out coordinates that aren't spatial. Time, ensemble members, etc.
    non_spatial = [i for i in dimlist if i not in _get_dims(area)]
    filter_dict = {}
    while len(non_spatial) > 0:
        filter_dict.update({non_spatial[0]: 0})
        non_spatial.pop(0)
    masked_area = area.where(da.isel(filter_dict).notnull())
    # Compute area-weighting.
    dimlist = _get_dims(masked_area)
    aw_da = da * masked_area
    # Sum over arbitrary number of dimensions.
    while len(dimlist) > 0:
        print(f'Summing over {dimlist[0]}')
        aw_da = aw_da.sum(dimlist[0])
        dimlist.pop(0)
    # Finish area-weighting by dividing by sum of area coordinate.
    aw_da = aw_da / masked_area.sum()
    return aw_da


# -----------
# TIME SERIES
# -----------
def xr_smooth_series(da, dim, length, center=True):
    """
    Returns a smoothed version of the input timeseries.
    NOTE: Currently explicitly writing `xr` as a prefix for xarray-specific
    definitions. Since `esmtools` is supposed to be a wrapper for xarray,
    this might be altered in the future.
    Parameters
    ----------
    da : xarray DataArray
    dim : str
        dimension to smooth over (e.g. 'time')
    length : int
        number of steps to smooth over for the given dim
    center : boolean (default to True)
        whether to center the smoothing filter or start from the beginning
    Returns
    -------
    smoothed : smoothed DataArray object
    """
    _check_xarray(da)
    return da.rolling({dim: length}, center=center).mean()


def xr_linregress(da, dim='time', compact=True):
    """
    Computes the least-squares linear regression of a dataarray over some
    dimension (typically time).
    Parameters
    ----------
    da : xarray DataArray
    dim : str (default to 'time')
        dimension over which to compute the linear regression.
    compact : boolean (default to True)
        If true, return all results of linregress as a single dataset.
        If false, return results as five separate DataArrays.
    Returns
    -------
    ds : xarray Dataset
        Dataset containing slope, intercept, rvalue, pvalue, stderr from
        the linear regression. Excludes the dimension the regression was
        computed over. If compact is False, these five parameters are
        returned separately.
    """
    _check_xarray(da)
    results = xr.apply_ufunc(linregress, da[dim], da,
                             input_core_dims=[[dim], [dim]],
                             output_core_dims=[[], [], [], [], []],
                             vectorize=True, dask='parallelized')
    # Force into a cleaner dataset. The above function returns a dataset
    # with no clear labeling.
    ds = xr.Dataset()
    labels = ['slope', 'intercept', 'rvalue', 'pvalue', 'stderr']
    for i, l in enumerate(labels):
        results[i].name = l
        ds = xr.merge([ds, results[i]])
    if compact:
        return ds
    else:
        return ds['slope'], ds['intercept'], ds['rvalue'], ds['pvalue'], \
               ds['stderr']


def xr_corr(x, y, dim='time', lag=0, two_sided=True, return_p=False):
    """
    Computes the Pearson product-momment coefficient of linear correlation.
    (See xr_autocorr for autocorrelation/lag for one time series)
    This version calculates the effective degrees of freedom, accounting
    for autocorrelation within each time series that could fluff the
    significance of the correlation.
    NOTE: If lag is not zero, x predicts y. In other words, the time series for
    x is stationary, and y slides to the left. Or, y stays in place and x
    slides to the right.
    This function is written to accept a dataset of arbitrary number of
    dimensions (e.g., lat, lon, depth).
    TODO: Add functionality for an ensemble.
    Parameters
    ----------
    x, y : xarray DataArray
        time series being correlated (can be multi-dimensional)
    dim : str (default 'time')
        Correlation dimension
    lag : int (default 0)
        Lag to apply to correlation, with x predicting y.
    two_sided : boolean (default True)
        If true, compute a two-sided t-test
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
    return st.xr_corr(x, y, dim=dim, lag=lag, two_sided=two_sided,
                      return_p=return_p)


def xr_rm_poly(da, order, dim='time'):
    """
    Returns xarray object with nth-order fit removed from every time series.
    Input
    -----
    da : xarray DataArray
        Single time series or many gridded time series of object to be
        detrended
    order : int
        Order of polynomial fit to be removed. If 1, this is functionally
        the same as calling `xr_rm_trend`
    dim : str (default 'time')
        Dimension over which to remove the polynomial fit.
    Returns
    -------
    detrended_ts : xarray DataArray
        DataArray with detrended time series.
    """
    return st.xr_rm_poly(da, order, dim=dim)


def xr_rm_trend(da, dim='time'):
    """
    Calls xr_rm_poly with an order 1 argument.
    """
    return st.xr_rm_trend(da, dim=dim)


def xr_autocorr(ds, lag=1, dim='time', return_p=False):
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
    return st.xr_autocorr(ds, lag=lag, dim=dim, return_p=return_p)
