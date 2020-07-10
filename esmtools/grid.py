from .checks import is_xarray


@is_xarray(0)
def _convert_lon_to_180to180(ds, coord='lon'):
    """Convert from 0 to 360 (degrees E) grid to -180 to 180 (W-E) grid.

    .. note::
        Longitudes are not sorted after conversion (i.e., spanning -180 to 180 or
        0 to 360 from index 0, ..., N), as it is expected that the user will plot
        via ``cartopy``, ``basemap``, or ``xarray`` plot functions.

    Args:
        ds (xarray object): Dataset to be converted.
        coord (optional str): Name of longitude coordinate.

    Returns:
        xarray object: Dataset with converted longitude grid.
    """
    ds = ds.copy()
    lon = ds[coord].values
    # Convert everything over 180 back to the negative (degrees W) values.
    lon[lon > 180] = lon[lon > 180] - 360
    # Need to account for clarifying dimensions if the grid is 2D.
    ds.coords[coord] = (ds[coord].dims, lon)
    return ds


@is_xarray(0)
def _convert_lon_to_0to360(ds, coord='lon'):
    """Convert from -180 to 180 (W-E) to 0 to 360 (degrees E) grid.

    .. note::
        Longitudes are not sorted after conversion (i.e., spanning -180 to 180 or
        0 to 360 from index 0, ..., N), as it is expected that the user will plot
        via ``cartopy``, ``basemap``, or ``xarray`` plot functions.

    Args:
        ds (xarray object): Dataset to be converted.
        coord (optional str): Name of longitude coordinate.

    Returns:
        xarray object: Dataset with converted longitude grid.
    """
    ds = ds.copy()
    lon = ds[coord].values
    # Convert -180 to 0 into scale reaching 360.
    lon[lon < 0] = lon[lon < 0] + 360
    # Need to account for clarifying dimensions if the grid is 2D.
    ds.coords[coord] = (ds[coord].dims, lon)
    return ds


# NOTE: Check weird POP grid that goes up to 240 or something. How do we deal with
# that?
@is_xarray(0)
def convert_lon(ds, coord='lon'):
    """Converts longitude grid from -180to180 to 0to360 and vice versa.

    .. note::
        Longitudes are not sorted after conversion (i.e., spanning -180 to 180 or
        0 to 360 from index 0, ..., N) if it is 2D.,

    Args:
        ds (xarray object): Dataset to be converted.
        coord (optional str): Name of longitude coordinate.

    Returns:
        xarray object: Dataset with converted longitude grid.

    Raises:
        ValueError: If ``coord`` does not exist in the dataset.

    Examples:
       >>> import numpy as np
       >>> import xarray as xr
       >>> from esmtools.grid import convert_lon
       >>> lat = np.linspace(-89.5, 89.5, 180)
       >>> lon = np.linspace(0.5, 359.5, 360)
       >>> empty = xr.DataArray(np.empty((180, 360)), dims=['lat', 'lon'])
       >>> data = xr.DataArray(np.linspace(-180, 180, 360), dims=['lon'],)
       >>> data, _ = xr.broadcast(data, empty)
       >>> data = data.T
       >>> data['lon'] = lon
       >>> data['lat'] = lat
       >>> converted = convert_lon(data, coord='lon')
    """
    if coord not in ds.coords:
        raise ValueError(f'{coord} not found in coordinates.')
    if ds[coord].min() < 0:
        ds = _convert_lon_to_0to360(ds, coord=coord)
    else:
        ds = _convert_lon_to_180to180(ds, coord=coord)
    # If 1-D, need to sort by lon (rearrange it) to allow it to be plotted with
    # xarray.
    if len(ds[coord].dims) == 1:
        ds = ds.sortby(coord)
    return ds
