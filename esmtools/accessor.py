import xarray as xr

from .grid import convert_lon
from .stats import corr, linear_slope, linregress, polyfit, rm_poly, rm_trend


@xr.register_dataarray_accessor("grid")
@xr.register_dataset_accessor("grid")
class GridAccessor:
    """Allows functions to be called directly on an xarray dataset for the grid module.

    This implementation was heavily inspired by @ahuang11's implementation to
    xskillscore."""

    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def convert_lon(self, coord="lon"):
        """Converts longitude grid from -180to180 to 0to360 and vice versa.

        .. note::
            Longitudes are not sorted after conversion (i.e., spanning -180 to 180 or
            0 to 360 from index 0, ..., N) if it is 2D.

        Args:
            ds (xarray object): Dataset to be converted.
            coord (optional str): Name of longitude coordinate, defaults to 'lon'.

        Returns:
            xarray object: Dataset with converted longitude grid.

        Raises:
            ValueError: If ``coord`` does not exist in the dataset.
        """
        return convert_lon(self._obj, coord=coord)


@xr.register_dataarray_accessor('stats')
class StatsAccessor:
    """Allows functions to be called directly on an xarray dataset for the stats module.

    This implementation was heavily inspired by @ahuang11's implementation to
    xskillscore."""

    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def corr(self, arr, dim='time', lead=0, return_p=False):
        """Computes the Pearson product-moment coefficient of linear correlation.

        Args:
            arr (xarray object): Time series being correlated against DataArray
            dim (str, optional): Dimension to calculate correlation over. Defaults to
                'time'.
            lead (int, optional): If lead > 0, the data array leads ``arr`` by that
                many time steps. If lead < 0, the data array lags ``arr`` by that
                many time steps. Defaults to 0.
            return_p (bool, optional). If True, return both the correlation coefficient
                and p value. Otherwise, just returns the correlation coefficient.

        Returns:
            corrcoef (xarray object): Pearson correlation coefficient.
            pval (xarray object): p value, if ``return_p`` is True.

        """
        x = self._obj
        y = arr

        return corr(x, y, dim=dim, lead=lead, return_p=return_p)

    def linregress(self, dim='time', nan_policy='none'):
        """Vectorized applciation of ``scipy.stats.linregress``.

        .. note::

            This function will try to infer the time freqency of sampling if ``dim`` is
            in datetime units. The final slope and standard error will be returned in
            the original units per that frequency (e.g. SST per year). If the frequency
            cannot be inferred (e.g. because the sampling is irregular), it will return
            in the original units per day (e.g. SST per day).

        .. note::
            If you would like to regress against a different variable that is not a
            dimension in the data array see the linregress function in the stats module

        Args:
            dim (str, optional): Dimension to apply linear regression over.
                Defaults to "time".
            nan_policy (str, optional): Policy to use when handling nans. Defaults to
                "none".

                * 'none', 'propagate': If a NaN exists anywhere on the given dimension,
                    return nans for that whole dimension.
                * 'raise': If a NaN exists at all in the datasets, raise an error.
                * 'omit', 'drop': If a NaN exists in the data array, drop that index
                    and compute the slope without it.

        Returns:
            xarray object: Slope, intercept, correlation, p value, and standard error
                for the linear regression. These 5 parameters are added as a new
                dimension "parameter".
        """
        return linregress(self._obj, dim=dim, nan_policy=nan_policy)

    def linear_slope(self, dim='time', nan_policy='none'):
        """Returns the linear slope with data array regressed onto dimension.

        .. note::

            This function will try to infer the time freqency of sampling if ``dim`` is
            in datetime units. The final slope will be returned in the original units
            per that frequency (e.g. SST per year). If the frequency cannot be inferred
            (e.g. because the sampling is irregular), it will return in the original
            units per day (e.g. SST per day).

        .. note::
            If you would like to regress against a different variable that is not a
            dimension in the data array see the linear_slope function in the stats
            module.

        Args:
            dim (str, optional): Dimension to apply linear regression over.
                Defaults to "time".
            nan_policy (str, optional): Policy to use when handling nans. Defaults to
                "none".

                * 'none', 'propagate': If a NaN exists anywhere on the given dimension,
                    return nans for that whole dimension.
                * 'raise': If a NaN exists at all in the datasets, raise an error.
                * 'omit', 'drop': If a NaN exists in the data array, drop that index and
                    compute the slope without it.

        Returns:
            xarray object: Slopes computed through a least-squares linear regression.
        """
        return linear_slope(self._obj, dim=dim, nan_policy=nan_policy)

    def rm_trend(self, dim='time', nan_policy='none'):
        """Removes a linear trend from ``array`` regressed onto ``dim`.

        .. note::
            If you would like to regress against a different variable that is not a
            dimension in the data array see the rm_trend function in the stats module.

        Args:
            dim (str, optional): Dimension to apply linear fit over.
                Defaults to "time".
            nan_policy (str, optional): Policy to use when handling nans. Defaults to
                "none".

                * 'none', 'propagate': If a NaN exists anywhere on the given dimension,
                    return nans for that whole dimension.
                * 'raise': If a NaN exists at all in the datasets, raise an error.
                * 'omit', 'drop': If a NaN exists in the data array, drop that index
                    and compute the slope without it.

        Returns:
            xarray object: ``array`` with linear trend removed.
        """
        return rm_trend(self._obj, dim=dim, nan_policy=nan_policy)

    def polyfit(self, order=1, dim='time', nan_policy='none'):
        """Returns the fitted polynomial line of ``array`` regressed onto ``dim``.

        .. note::
            This will be released as a standard ``xarray`` func in 0.15.2.

        .. note::
            If you would like to regress against a different variable that is not a
            dimension in the data array see the polyfit function in the stats module.

        Args:
            order (int): Order of polynomial fit to perform.
            dim (str, optional): Dimension to apply polynomial fit over.
                Defaults to "time".
            nan_policy (str, optional): Policy to use when handling nans. Defaults to
                "none".

                * 'none', 'propagate': If a NaN exists anywhere on the given dimension,
                    return nans for that whole dimension.
                * 'raise': If a NaN exists at all in the datasets, raise an error.
                * 'omit', 'drop': If a NaN exists in the data array, drop that index
                    and compute the slope without it.

        Returns:
            xarray object: The polynomial fit for ``array`` regressed onto ``dim``.
                Has the same dimensions as ``array``.
        """
        return polyfit(self._obj, dim=dim, nan_policy=nan_policy)

    def rm_poly(self, order=1, dim='time', nan_policy='none'):
        """Removes a polynomial fit from ``array`` regressed onto ``dim`.

        .. note::
            If you would like to regress against a different variable that is not a
            dimension in the data array see the rm_poly function in the stats module.

        Args:
            order (int): Order of polynomial fit to perform.
            dim (str, optional): Dimension to apply polynomial fit over.
                Defaults to "time".
            nan_policy (str, optional): Policy to use when handling nans. Defaults to
                "none".

                * 'none', 'propagate': If a NaN exists anywhere on the given dimension,
                    return nans for that whole dimension.
                * 'raise': If a NaN exists at all in the datasets, raise an error.
                * 'omit', 'drop': If a NaN exists in the data array, drop that index
                    and compute the slope without it.

        Returns:
            xarray object: ``array`` with polynomial fit of order ``order`` removed.
        """
        return rm_poly(self._obj, order=order, dim=dim, nan_policy=nan_policy)
