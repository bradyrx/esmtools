def get_dims(da):
    """
    Simple function to retrieve dimensions from a given dataset/datarray.
    Currently returns as a list, but can add keyword to select tuple or
    list if desired for any reason.
    """
    return list(da.dims)
