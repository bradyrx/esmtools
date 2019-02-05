"""Definitions related to climate model grids."""


def convert_lon_to_180to180(ds, lon_coord='lon'):
    """Convert from 0 to 360 grid to -180 to 180 grid."""
    ds = ds.copy()
    lon = ds[lon_coord].values
    lon[lon > 180] = lon[lon > 180] - 360
    ds.coords[lon_coord] = lon
    ds = ds.sortby(lon_coord)
    return ds


def convert_lon_to_0to360(ds, lon_coord='lon'):
    """Convert from -180 to 180 grid to 0 to 360 grid."""
    ds = ds.copy()
    lon = ds[lon_coord].values
    lon[lon < 0] = lon[lon < 0] + 360
    ds.coords[lon_coord] = lon
    ds = ds.sortby(lon_coord)
    return ds
