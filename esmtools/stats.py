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

# --------------------------
# AREA-WEIGHTING DEFINITIONS
# --------------------------


def xr_cos_weight(ds, lat_coord='lat', lon_coord='lon', one_dimensional=True):
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
    return st.xr_cos_weight(ds, lat_coord=lat_coord, lon_coord=lon_coord,
                            one_dimensional=one_dimensional)


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
    return st.xr_area_weight(da, area_coord=area_coord)


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
    return st.xr_smooth_series(da, dim, length, center=center)


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
    return st.xr_linregress(da, dim=dim, compact=compact)


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
