import climpred.stats as st
import numpy as np
import numpy.polynomial.polynomial as poly
import xarray as xr
from scipy.stats import linregress as lreg
from scipy.stats import ttest_ind_from_stats as tti_from_stats

from .checks import has_dims
from .utils import check_xarray, get_dims


# ------------------
# GENERAL STATISTICS
# ------------------
@check_xarray(0)
def standardize(ds, dim="time"):
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


@check_xarray(0)
def nanmean(ds, dim="time"):
    """Compute mean NaNs and suppress warning from numpy"""
    if "time" in ds.dims:
        mask = ds.isnull().isel(time=0)
    else:
        mask = ds.isnull()
    ds = ds.fillna(0).mean(dim)
    ds = ds.where(~mask)
    return ds


# --------------------------
# AREA-WEIGHTING DEFINITIONS
# --------------------------
@check_xarray(0)
def cos_weight(da, lat_coord="lat", lon_coord="lon", one_dimensional=True):
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
    lat = lat.astype("float")
    nan_mask = np.asarray(da.isel(filter_dict).isnull())
    lat[nan_mask] = np.nan
    cos_lat = np.cos(np.deg2rad(lat))
    aw_da = (da * cos_lat).sum(lat_coord).sum(lon_coord) / np.nansum(cos_lat)
    return aw_da


@check_xarray(0)
def area_weight(da, area_coord="area"):
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
        print(f"Summing over {dimlist[0]}")
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
def fit_poly(ds, order, dim="time"):
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
    has_dims(ds, dim, "dataset")

    # handle both datasets and dataarray
    if isinstance(ds, xr.Dataset):
        da = ds.to_array()
        return_ds = True
    else:
        da = ds.copy()
        return_ds = False

    da_dims_orig = list(da.dims)  # orig -> original
    if len(da_dims_orig) > 1:
        # want independent axis to be the leading dimension
        da_dims_swap = da_dims_orig.copy()  # copy to prevent contamination

        # https://stackoverflow.com/questions/1014523/
        # simple-syntax-for-bringing-a-list-element-to-the-front-in-python
        da_dims_swap.insert(0, da_dims_swap.pop(da_dims_swap.index(dim)))
        da = da.transpose(*da_dims_swap)

        # hide other dims into a single dim
        da = da.stack({"other_dims": da_dims_swap[1:]})
        dims_swapped = True
    else:
        dims_swapped = False

    # NaNs will make the polyfit fail--interpolate any NaNs in
    # the provided dim to prevent poor fit, while other dims' NaNs
    # will be filled with 0s; however, all NaNs will be replaced
    # in the final output
    nan_locs = np.isnan(da.values)

    # any(nan_locs.sum(axis=0)) fails if not 2D
    if nan_locs.ndim == 1:
        nan_locs = nan_locs.reshape(len(nan_locs), 1)

    # check if there's any NaNs in the provided dim because
    # interpolate_na is computationally expensive to run regardless of NaNs
    if any(nan_locs.sum(axis=0)) > 0:
        # Could do a check to see if there's any NaNs that aren't bookended.
        # [0, np.nan, 2], can interpolate.
        da = da.interpolate_na(dim)
        if any(nan_locs[0, :]):
            # [np.nan, 1, 2], no first value to interpolate from; back fill
            da = da.bfill(dim)
        if any(nan_locs[-1, :]):
            # [0, 1, np.nan], no last value to interpolate from; forward fill
            da = da.ffill(dim)

    # this handles the other axes; doesn't matter since it won't affect the fit
    da = da.fillna(0)

    # the actual operation of detrending
    y = da.values
    x = np.arange(0, len(y), 1)
    coefs = poly.polyfit(x, y, order)
    fit = poly.polyval(x, coefs)
    fit = fit.transpose()
    fit = xr.DataArray(fit, dims=da.dims, coords=da.coords)

    if dims_swapped:
        # revert the other dimensions to its original form and ordering
        fit = fit.unstack("other_dims").transpose(*da_dims_orig)

    if return_ds:
        # revert back into a dataset
        return xr.merge(
            fit.sel(variable=var).rename(var).drop("variable")
            for var in fit["variable"].values
        )
    else:
        return fit


@check_xarray(0)
def linear_regression(da, dim="time", interpolate_na=False, compact=True, psig=None):
    """
    Computes the least-squares linear regression of an xr.DataArray x against
    another xr.DataArray y.

    Note: This only returns the statistics from the LSR (slope, intercept, rvalue,
    pvalue, stderr). Use `fit_poly` to get the fitted line returned.

    Parameters
    ----------
    da : xarray DataArray
    dim : str (default to 'time')
        dimension over which to compute the linear regression.
    interpolate_na : bool (default to False)
        If True, linearly interpolate NaNs (and backfill/forwardfill). Note that if
        this is False, the linear regression will return NaN if there are *any*
        NaNs in the time series.
    compact : boolean (default to True)
        If true, return all results of linregress as a single dataset.
        If false, return results as five separate DataArrays.
    psig : double (default to None)
        Alpha level for correlation significance. If not None, NaNs out any grid cells
        that are larger than psig.

    Returns
    -------
    ds : xarray Dataset
        Dataset containing slope, intercept, rvalue, pvalue, stderr from
        the linear regression. Excludes the dimension the regression was
        computed over. If compact is False, these five parameters are
        returned separately.
    """
    # Check if dataset.
    if isinstance(da, xr.Dataset):
        raise NotImplementedError(
            "Datasets are not yet supported for this function. "
            + "Please retry with a DataArray."
        )

    x = da[dim]
    # Linear regression doesn't like to work with actual datetime. So temporarily
    # convert to integers.
    # `np.object` covers cftime.
    if np.issubdtype(x, np.datetime64) or isinstance(x, np.object):
        x = np.arange(len(x))

    if interpolate_na:
        # borrowed from @ahuang11's implementation in `climpred`
        da_dims_orig = list(da.dims)  # orig -> original
        if len(da_dims_orig) > 1:
            # want independent axis to be the leading dimension
            da_dims_swap = da_dims_orig.copy()  # copy to prevent contamination

            # https://stackoverflow.com/questions/1014523/
            # simple-syntax-for-bringing-a-list-element-to-the-front-in-python
            da_dims_swap.insert(0, da_dims_swap.pop(da_dims_swap.index(dim)))
            da = da.transpose(*da_dims_swap)

            # hide other dims into a single dim
            da = da.stack({"other_dims": da_dims_swap[1:]})
            dims_swapped = True
        else:
            dims_swapped = False

        # This is borrowed from @ahuang11's implementation in `climpred` to handle
        # NaNs in the time series.
        nan_locs = np.isnan(da.values)

        # any(nan_locs.sum(axis=0)) fails if not 2D
        if nan_locs.ndim == 1:
            nan_locs = nan_locs.reshape(len(nan_locs), 1)

        # check if there's any NaNs in the provided dim because
        # interpolate_na is computationally expensive to run regardless of NaNs
        # if nan_locs.sum(dim=dim).any():
        if any(nan_locs.sum(axis=0)) > 0:
            da = da.interpolate_na(dim)
            if any(nan_locs[0, :]):
                # [np.nan, 1, 2], no first value to interpolate from; back fill
                da = da.bfill(dim)
            if any(nan_locs[-1, :]):
                # [0, 1, np.nan], no last value to interpolate from; forward fill
                da = da.ffill(dim)

        # this handles the other axes; doesn't matter since it won't affect the fit
        da = da.fillna(0)
        if dims_swapped:
            da = da.unstack("other_dims").transpose(*da_dims_orig)

    results = xr.apply_ufunc(
        lreg,
        x,
        da,
        input_core_dims=[[dim], [dim]],
        output_core_dims=[[], [], [], [], []],
        vectorize=True,
        dask="allowed",
    )

    # Force into a cleaner dataset. The above function returns a dataset
    # with no clear labeling.
    ds = xr.Dataset()
    labels = ["slope", "intercept", "rvalue", "pvalue", "stderr"]
    for i, l in enumerate(labels):
        results[i].name = l
        ds = xr.merge([ds, results[i]])
    if psig is not None:
        ds = ds.where(ds["pvalue"] < psig)
    if compact:
        return ds
    else:
        return ds["slope"], ds["intercept"], ds["rvalue"], ds["pvalue"], ds["stderr"]


@check_xarray(0)
def corr(x, y, dim="time", lead=0, return_p=False):
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


@check_xarray(0)
def rm_poly(da, order, dim="time"):
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
def rm_trend(da, dim="time"):
    """
    Calls rm_poly with an order 1 argument.
    """
    return st.rm_trend(da, dim=dim)


@check_xarray(0)
def autocorr(ds, lag=1, dim="time", return_p=False):
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
def ACF(ds, dim="time", nlags=None):
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


def ttest_ind_from_stats(mean1, std1, nobs1, mean2, std2, nobs2):
    """Parallelize scipy.stats.ttest_ind_from_stats."""
    return xr.apply_ufunc(
        tti_from_stats,
        mean1,
        std1,
        nobs1,
        mean2,
        std2,
        nobs2,
        input_core_dims=[[], [], [], [], [], []],
        output_core_dims=[[], []],
        vectorize=True,
        dask="parallelized",
    )
