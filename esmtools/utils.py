import numpy as np

from .checks import has_missing


def match_nans(x, y):
    """Performs pairwise matching of nans between ``x`` and ``y``.

    Args:
        x, y (ndarray or xarray object): Array-like objects to pairwise match nans over.

    Returns:
        x, y (ndarray or xarray object): If either ``x`` or ``y`` has missing data,
            adds nans in the same places between the two arrays.
    """
    if has_missing(x) or has_missing(y):
        # Need to copy to avoid mutating original objects and to avoid writeable errors
        # with ``xr.apply_ufunc`` with vectorize turned on.
        x, y = x.copy(), y.copy()
        idx = np.logical_or(np.isnan(x), np.isnan(y))
        # NaNs cannot be added to `int` arrays.
        if x.dtype == 'int':
            x = x.astype('float')
        if y.dtype == 'int':
            y = y.astype('float')
        x[idx], y[idx] = np.nan, np.nan
    return x, y
