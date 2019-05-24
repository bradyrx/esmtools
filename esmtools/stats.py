import climpred.stats as st
import numpy as np
import xarray as xr
from scipy.stats import linregress as lreg
from scipy.stats import ttest_ind as tti
from scipy.stats import ttest_ind_from_stats as tti_from_stats

from .utils import (check_xarray, get_dims)


# --------------------------
# AREA-WEIGHTING DEFINITIONS
# --------------------------
@check_xarray(0)
def cos_weight(da, lat_coord='lat', lon_coord='lon', one_dimensional=True):
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
    aw_da = (da * cos_lat).sum(lat_coord).sum(lon_coord) / \
        np.nansum(cos_lat)
    return aw_da


@check_xarray(0)
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


# -----------
# TIME SERIES
# -----------
@check_xarray(0)
def smooth_series(da, dim, length, center=True):
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
    return da.rolling({dim: length}, center=center).mean()


@check_xarray(0)
def linregress(da, dim='time', compact=True):
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
    results = xr.apply_ufunc(lreg, da[dim], da,
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


@check_xarray(0)
def corr(x, y, dim='time', lag=0, two_sided=True, return_p=False):
    """
    Computes the Pearson product-momment coefficient of linear correlation.
    (See autocorr for autocorrelation/lag for one time series)
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
    return st.corr(x, y, dim=dim, lag=lag, two_sided=two_sided,
                   return_p=return_p)


@check_xarray(0)
def rm_poly(da, order, dim='time'):
    """
    Returns xarray object with nth-order fit removed from every time series.
    Input
    -----
    da : xarray DataArray
        Single time series or many gridded time series of object to be
        detrended
    order : int
        Order of polynomial fit to be removed. If 1, this is functionally
        the same as calling `rm_trend`
    dim : str (default 'time')
        Dimension over which to remove the polynomial fit.
    Returns
    -------
    detrended_ts : xarray DataArray
        DataArray with detrended time series.
    """
    return st.rm_poly(da, order, dim=dim)


@check_xarray(0)
def rm_trend(da, dim='time'):
    """
    Calls rm_poly with an order 1 argument.
    """
    return st.rm_trend(da, dim=dim)


@check_xarray(0)
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


@check_xarray(0)
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


def ttest_ind(a, b, dim='time'):
    """Parallelize scipy.stats.ttest_ind."""
    return xr.apply_ufunc(tti, a, b, input_core_dims=[[dim], [dim]],
                          output_core_dims=[[], []],
                          vectorize=True, dask='parallelized')


def ttest_ind_from_stats(mean1, std1, nobs1, mean2, std2, nobs2):
    """Parallelize scipy.stats.ttest_ind_from_stats."""
    return xr.apply_ufunc(tti_from_stats, mean1, std1, nobs1, mean2, std2,
                          nobs2,
                          input_core_dims=[[], [], [], [], [], []],
                          output_core_dims=[[], []],
                          vectorize=True, dask='parallelized')


def standardize(ds, dim='time'):
    return (ds - ds.mean(dim)) / ds.std(dim)


def composite_analysis(field,
                       timeseries,
                       threshold=1,
                       plot=False,
                       ttest=True,
                       psig=0.05,
                       **plot_kwargs):
    """Short summary.

    Args:
        field (xr.object): contains dims: 'time', 2 spatial.
        timeseries (xr.object): Create composite based on timeseries.
        threshold (float): threshold value for positive composite.
                        Defaults to 1.
        plot (bool): quick plot and no returns. Defaults to False.
        ttest (bool): Apply `ttest` whether pos/neg different from mean.
                      Defaults to True.
        psig (float): Significance level for ttest. Defaults to 0.05.
        **plot_kwargs (type): Description of parameter `**plot_kwargs`.

    Returns:
        composite (xr.object): pos and negative composite if `not plot`.

    """
    index = standardize(timeseries)
    field = field - field.mean('time')

    def _create_composites(anomaly_field, timeseries, threshold=1, dim='time'):
        index_comp = xr.full_like(timeseries, 'none', dtype='U4')
        index_comp[timeseries >= threshold] = 'pos'
        index_comp[timeseries <= -threshold] = 'neg'
        composite = anomaly_field.groupby(
            index_comp.rename('index'))
        return composite

    composite = _create_composites(field, index, threshold=threshold)
    if ttest:
        # test if pos different from none
        index = 'pos'
        m1 = composite.mean('time').sel(index=index)
        s1 = composite.std('time').sel(index=index)
        n1 = len(composite.groups[index])
        index = 'none'
        m2 = composite.mean('time').sel(index=index)
        s2 = composite.std('time').sel(index=index)
        n2 = len(composite.groups[index])

        t, p = ttest_ind_from_stats(m1, s1, n1, m2, s2, n2)
        comp_pos = composite.mean('time').sel(index='pos').where(p < psig)

        # test if neg different from none
        index = 'neg'
        m1 = composite.mean('time').sel(index=index)
        s1 = composite.std('time').sel(index=index)
        n1 = len(composite.groups[index])
        index = 'none'
        m2 = composite.mean('time').sel(index=index)
        s2 = composite.std('time').sel(index=index)
        n2 = len(composite.groups[index])

        t, p = ttest_ind_from_stats(m1, s1, n1, m2, s2, n2)
        comp_neg = composite.mean('time').sel(index='neg').where(p < psig)

        composite = xr.concat([comp_pos, comp_neg], dim='index')
    else:
        composite = composite.mean('time').sel(index=['pos', 'neg'])

    composite['index'] = ['positive', 'negative']

    if plot:
        composite.plot(col='index', **plot_kwargs)
    else:
        return composite
