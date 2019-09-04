from .exceptions import DimensionError


def has_dims(xobj, dims, kind):
    """
    Checks that at the minimum, the object has provided dimensions.
    """
    if isinstance(dims, str):
        dims = [dims]

    if not all(dim in xobj.dims for dim in dims):
        raise DimensionError(
            f"Your {kind} object must contain the "
            f"following dimensions at the minimum: {dims}"
        )
    return True
