import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from .vis import make_cartopy


def scatter(
    lon,
    lat,
    data,
    cmap,
    vmin,
    vmax,
    stride=5,
    projection=ccrs.Robinson(),
    colorbar=True,
):
    """Create map of MPAS output on the native unstructured grid.

    .. note::
      ``pcolormesh`` and ``contourf`` can't be used with the native output, since
      it's on an unstructured grid. To visualize the unstructured grid in the most
      straight forward fashion, it's best to use ParaView.

    Args:
        lon (xarray object): 1D da of longitudes (``lonCell``)
        lat (xarray object): 1D da of latitudes (``latCell``)
        data (xarray object): Data to plot
        cmap (str): Colormap str.
        vmin (float): Minimum color bound.
        vmax (float): Maximum color bound.
        stride (optional int):
            Stride in plotting data to avoid plotting too much. Defaults to 5.
        projection (cartopy map projection):
            Map projection to use. Defaults to Robinson.
        colorbar (optional bool):
            Whether or not to add a colorbar to the figure. Generally want to set this
            to off and do it manually if you need more advanced changes to it.

    Examples:
        >>> from esmtools.mpas import scatter
        >>> import xarray as xr
        >>> ds = xr.open_dataset('some_mpas_BGC_output.nc')
        >>> scatter(ds.lonCell, ds.latCell, ds.FG_CO2, "RdBu_r",
                    -5, 5)
    """
    f, ax = make_cartopy(projection=projection, grid_lines=False, frameon=False)
    lon = lon[0::stride]
    lat = lat[0::stride]
    data = data[0::stride]

    p = ax.scatter(
        lon,
        lat,
        s=1,
        c=data,
        cmap=cmap,
        transform=ccrs.PlateCarree(),
        vmin=vmin,
        vmax=vmax,
    )
    if colorbar:
        plt.colorbar(p, orientation="horizontal", pad=0.05, fraction=0.08)


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
    ds["latParticle"] = ds["latParticle"] * (180 / np.pi)
    ds["lonParticle"] = ds["lonParticle"] * (180 / np.pi)
    return ds


def convert_deg_to_rad(ds):
    """Quick converter from degrees to radians.

    Just set up for LIGHT for now.
    """
    ds["latParticle"] = ds["latParticle"] * (np.pi / 180)
    ds["lonParticle"] = ds["lonParticle"] * (np.pi / 180)
    return ds
