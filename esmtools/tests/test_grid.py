import numpy as np
import pytest
import xarray as xr

from esmtools.grid import convert_lon


@pytest.fixture()
def da_1D():
    """Gridded DataArray with one-dimensional lat and lon coordinates."""
    # Wrapper so fixture can be called with arguments.
    # https://alysivji.github.io/pytest-fixures-with-function-arguments.html
    def _gen_data(degreesEast=None):
        """Generate the DataArray.

        degreesEast (bool): If True, create 0to360 longitude. If False, -180to180.
        """
        # Generate standard 1x1 deg lat/lon.
        lat = np.linspace(-89.5, 89.5, 180)
        if degreesEast:
            lon = np.linspace(0.5, 359.5, 360)
        elif not degreesEast:
            lon = np.linspace(-179.5, 179.5, 360)
        else:
            raise ValueError('Please specify `degreesEast` as either True or False.')
        # Template for broadcasting to
        empty = xr.DataArray(
            np.empty((180, 360)), dims=['lat', 'lon'], coords=[lat, lon]
        )
        # Data is roughly the longitude at each grid cell.
        data = xr.DataArray(np.linspace(-180, 180, 360), dims=['lon'], coords=[lon])
        # Simple broadcasting up to 180x360 dimensions.
        data, _ = xr.broadcast(data, empty)
        data = data.T
        return data

    return _gen_data


@pytest.fixture()
def da_2D():
    """Gridded DataArray with two-dimensional lat and lon coordinates."""
    # Wrapper so fixture can be called with arguments.
    # https://alysivji.github.io/pytest-fixures-with-function-arguments.html
    def _gen_data(degreesEast=None):
        """Generate the DataArray.

        degreesEast (bool): If True, create 0to360 longitude. If False, -180to180.
        """
        # Generate standard 1x1 deg lat/lon.
        y = np.linspace(-89.5, 89.5, 180)
        if degreesEast:
            x = np.linspace(0.5, 359.5, 360)
        elif not degreesEast:
            x = np.linspace(-179.5, 179.5, 360)
        else:
            raise ValueError('Please specify `degreesEast` as either True or False.')
        # Meshgrid into a 2-dimensional lon/lat.
        lon, lat = np.meshgrid(x, y)
        # Template for broadcasting to
        empty = xr.DataArray(np.empty((180, 360)), dims=['y', 'x'])
        # Data is roughly the longitude at each grid cell.
        data = xr.DataArray(np.linspace(-180, 180, 360), dims=['x'])
        # Simple broadcasting up to 180x360 dimensions.
        data, _ = xr.broadcast(data, empty)
        data = data.T
        # Add 2D coordinates.
        data['lon'] = (('y', 'x'), lon)
        data['lat'] = (('y', 'x'), lat)
        return data

    return _gen_data


def test_1D_0to360_to_180to180(da_1D):
    """Tests that a 1D 0to360 grid converts to -180to180."""
    data = da_1D(degreesEast=True)
    converted = convert_lon(data)
    lonmin = converted.lon.min()
    lonmax = converted.lon.max()
    # Checks that it was appropriately converted, not dipping below -180 or above 180.
    # But also accounts for coarser grids.
    assert (lonmin >= -180) & (lonmin <= 0) & (lonmax <= 180) & (lonmax >= 0)
    # Checks that data isn't changed.
    assert np.allclose(data.mean(), converted.mean())


def test_1D_180to180_to_0to360(da_1D):
    """Tests that a 1D -180to180 grid converts to 0to360."""
    data = da_1D(degreesEast=False)
    converted = convert_lon(data)
    lonmin = converted.lon.min()
    lonmax = converted.lon.max()
    # Checks that it was appropriately converted, not going below 0 or above 360.
    assert (lonmin >= 0) & (lonmax <= 360)
    # Checks that data isn't changed.
    assert np.allclose(data.mean(), converted.mean())


def test_2D_0to360_to_180to180(da_2D):
    """Tests that a 2D 0to360 grid converts to -180to180."""
    data = da_2D(degreesEast=True)
    converted = convert_lon(data)
    lonmin = converted.lon.min()
    lonmax = converted.lon.max()
    # Checks that it was appropriately converted, not dipping below -180 or above 180.
    # But also accounts for coarser grids.
    assert (lonmin >= -180) & (lonmin <= 0) & (lonmax <= 180) & (lonmax >= 0)
    # Checks that data isn't changed.
    assert np.allclose(data.mean(), converted.mean())


def test_2D_180to180_to_0to360(da_2D):
    """Tests that a 2D -180to180 grid converts to 0to360."""
    data = da_2D(degreesEast=False)
    converted = convert_lon(data)
    lonmin = converted.lon.min()
    lonmax = converted.lon.max()
    # Checks that it was appropriately converted, not going below 0 or above 360.
    assert (lonmin >= 0) & (lonmax <= 360)
    # Checks that data isn't changed.
    assert np.allclose(data.mean(), converted.mean())


def test_coordinate_error(da_1D):
    """Tests that coordinate error is thrown when convert_lon is called and the
    coordinate doesn't exist."""
    data = da_1D(degreesEast=True)
    with pytest.raises(ValueError) as e:
        # Purposefully call nonexistant coordinate.
        convert_lon(data, coord='foo')
    assert 'not found in coordinates' in str(e.value)
