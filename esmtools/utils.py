import xarray as xr


# https://stackoverflow.com/questions/10610824/
# python-shortcut-for-writing-decorators-which-accept-arguments
def dec_args_kwargs(wrapper):
    return (
        lambda *dec_args, **dec_kwargs:
            lambda func:
                wrapper(func, *dec_args, **dec_kwargs)
    )

# --------------------------------------#
# CHECKS
# --------------------------------------#
@dec_args_kwargs
# Lifted from climpred. This was originally written by Andrew Huang.
def check_xarray(func, *dec_args):
    """
    Decorate a function to ensure the first arg being submitted is
    either a Dataset or DataArray.
    """
    def wrapper(*args, **kwargs):
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
                raise IOError(f"""The input data is not an xarray DataArray or
                    Dataset. Please convert to an xarray object for this
                    function to work properly.
                    Your input was of type: {typecheck}""")

        return func(*args, **kwargs)
    return wrapper


def get_dims(da):
    """
    Simple function to retrieve dimensions from a given dataset/datarray.
    Currently returns as a list, but can add keyword to select tuple or
    list if desired for any reason.
    """
    return list(da.dims)
