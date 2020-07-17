from functools import wraps

import numpy as np
import xarray as xr


def has_missing(data):
    """Returns ``True`` if any NaNs in ``data`` and ``False`` otherwise.

    Args:
        data (ndarray or xarray object): Array to check for missing data.
    """
    return np.isnan(data).any()


# https://stackoverflow.com/questions/10610824/
# python-shortcut-for-writing-decorators-which-accept-arguments
def dec_args_kwargs(wrapper):
    return lambda *dec_args, **dec_kwargs: lambda func: wrapper(
        func, *dec_args, **dec_kwargs
    )


def has_dims(xobj, dims, kind):
    """
    Checks that at the minimum, the object has provided dimensions.

    Args:
        xobj (xarray object): Dataset or DataArray to check dimensions on.
        dims (list or str): Dimensions being checked.
        kind (str): String to precede "object" in the error message.
    """
    if isinstance(dims, str):
        dims = [dims]

    if not all(dim in xobj.dims for dim in dims):
        raise ValueError(
            f'Your {kind} object must contain the '
            f'following dimensions at the minimum: {dims}'
        )
    return True


@dec_args_kwargs
# Lifted from climpred. This was originally written by Andrew Huang.
def is_xarray(func, *dec_args):
    """
    Decorate a function to ensure the first arg being submitted is
    either a Dataset or DataArray.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            ds_da_locs = dec_args[0]
            if not isinstance(ds_da_locs, list):
                ds_da_locs = [ds_da_locs]

            for loc in ds_da_locs:
                if isinstance(loc, int):
                    ds_da = args[loc]
                elif isinstance(loc, str):
                    ds_da = kwargs[loc]

                is_ds_da = isinstance(ds_da, (xr.Dataset, xr.DataArray))
                if not is_ds_da:
                    typecheck = type(ds_da)
                    raise IOError(
                        f"""The input data is not an xarray DataArray or
                        Dataset.

                        Your input was of type: {typecheck}"""
                    )
        except IndexError:
            pass
        # this is outside of the try/except so that the traceback is relevant
        # to the actual function call rather than showing a simple Exception
        # (probably IndexError from trying to subselect an empty dec_args list)
        return func(*args, **kwargs)

    return wrapper
