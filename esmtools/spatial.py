import numpy as np

from .checks import is_xarray


@is_xarray(0)
def extract_region(ds, xgrid, ygrid, coords, lat_dim='lat', lon_dim='lon'):
    """Extract a subset of some larger spatial data.

    Args:
        ds (xarray object): Data to be subset.
        xgrid (array_like): Meshgrid of longitudes.
        ygrid (array_like): Meshgrid of latitudes.
        coords (1-D array or list): [x0, x1, y0, y1] pertaining to the corners of the
                                    box to extract.
        lat_dim (optional str): Latitude dimension name (default 'lat').
        lon_dim (optional str): Longitude dimension name (default 'lon')

    Returns:
        subset_data (xarray object): Data subset to domain of interest.

    Examples:
        >>> import esmtools as et
        >>> import numpy as np
        >>> import xarray as xr
        >>> x = np.linspace(0, 360, 37)
        >>> y = np.linspace(-90, 90, 19)
        >>> xx, yy = np.meshgrid(x, y)
        >>> ds = xr.DataArray(np.random.rand(19, 37), dims=['lat', 'lon'])
        >>> ds['latitude'] = (('lat', 'lon'), yy)
        >>> ds['longitude'] = (('lat', 'lon'), xx)
        >>> coords = [0, 30, -20, 20]
        >>> subset = et.spatial.extract_region(ds, xx, yy, coords)
    """
    # Extract the corners of the box.
    x0, x1, y0, y1 = coords
    # Find indices on meshgrid for the box corners.
    a, c = find_indices(xgrid, ygrid, x0, y0)
    b, d = find_indices(xgrid, ygrid, x1, y1)
    # Slice is not inclusive, so need to add one to end.
    subset_data = ds.isel({lat_dim: slice(a, b + 1), lon_dim: slice(c, d + 1)})
    return subset_data


def find_indices(xgrid, ygrid, xpoint, ypoint):
    """Returns the i, j index for a latitude/longitude point on a grid.

    .. note::
        Longitude and latitude points (``xpoint``/``ypoint``) should be in the same
        range as the grid itself (e.g., if the longitude grid is 0-360, should be
        200 instead of -160).

    Args:
        xgrid (array_like): Longitude meshgrid (shape `M`, `N`)
        ygrid (array_like): Latitude meshgrid (shape `M`, `N`)
        xpoint (int or double): Longitude of point searching for on grid.
        ypoint (int or double): Latitude of point searching for on grid.

    Returns:
        i, j (int):
            Keys for the inputted grid that lead to the lat/lon point the user is
            seeking.

    Examples:
        >>> import esmtools as et
        >>> import numpy as np
        >>> x = np.linspace(0, 360, 37)
        >>> y = np.linspace(-90, 90, 19)
        >>> xx, yy = np.meshgrid(x, y)
        >>> xp = 20
        >>> yp = -20
        >>> i, j = et.spatial.find_indices(xx, yy, xp, yp)
        >>> print(xx[i, j])
        20.0
        >>> print(yy[i, j])
        -20.0
    """
    dx = xgrid - xpoint
    dy = ygrid - ypoint
    reduced_grid = abs(dx) + abs(dy)
    min_ix = np.nanargmin(reduced_grid)
    i, j = np.unravel_index(min_ix, reduced_grid.shape)
    return i, j
