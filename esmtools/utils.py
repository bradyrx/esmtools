import numpy as np

from .checks import has_missing


def get_dims(da):
    """
    Simple function to retrieve dimensions from a given dataset/datarray.
    Currently returns as a list, but can add keyword to select tuple or
    list if desired for any reason.
    """
    return list(da.dims)


def match_nans(x, y):
    if has_missing(x) or has_missing(y):
        x = x.copy()
        y = y.copy()
        idx = np.logical_or(np.isnan(x), np.isnan(y))
        # NaNs cannot be added to `int` arrays.
        if x.dtype == "int":
            x = x.astype("float")
        if y.dtype == "int":
            y = y.astype("float")
        # Need to do the following two lines, only if ndarrays.
        x.setflags(write=1)
        y.setflags(write=1)
        x[idx] = np.nan
        y[idx] = np.nan
    return x, y
