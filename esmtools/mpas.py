import numpy as np


def xyz_to_lat_lon(x, y, z, radians=False):
    """
    Convert xyz output (e.g. from Lagrangian Particle Tracking) to conventional
    lat/lon coordinates.
    Input
    -----
    x : array_like
        Array of x values
    y : array_like
        Array of y values
    z : array_like
        Array of z values
    radians : boolean (optional)
        If true, return lat/lon as radians
    Returns
    -------
    lon : array_like
        Array of longitude values
    lat : array_like
        Array of latitude values
    Examples
    --------
    from esmtools.mpas import xyz_to_lat_lon
    import xarray as xr
    ds = xr.open_dataset('particle_output.nc')
    lon, lat = xyz_to_lat_lon(ds.xParticle, ds.yParticle, ds.zParticle)
    """
    x, y, z = np.asarray(x), np.asarray(y), np.asarray(z)
    plat = np.arcsin(z / np.sqrt(x ** 2 + y ** 2 + z ** 2))
    plon = np.arctan2(y, x)
    # longitude adjustment
    plon[plon < 0.0] = 2 * np.pi + plon[plon < 0.0]
    if radians:
        return plon, plat
    else:
        return plon * 180.0 / np.pi, plat * 180.0 / np.pi


def convert_rad_to_deg(ds):
    """Quick converter from radians to degrees.

    Just set up for LIGHT for now.
    """
    ds['latParticle'] = ds['latParticle'] * (180 / np.pi)
    ds['lonParticle'] = ds['lonParticle'] * (180 / np.pi)
    return ds


def convert_deg_to_rad(ds):
    """Quick converter from degrees to radians.

    Just set up for LIGHT for now.
    """
    ds['latParticle'] = ds['latParticle'] * (np.pi / 180)
    ds['lonParticle'] = ds['lonParticle'] * (np.pi / 180)
    return ds
