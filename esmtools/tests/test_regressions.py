import numpy as np
import pytest
from xarray.tests import assert_allclose

from esmtools.stats import linear_slope

TIME_TYPES = ['datetime', 'cftime', 'float']


@pytest.mark.parametrize('gridded', (True, False))
@pytest.mark.parametrize('time_type', TIME_TYPES)
def test_linear_slope_time_da(
    gridded_da_datetime, gridded_da_cftime, gridded_da_float, time_type, gridded
):
    """Tests that linear slope can be computed on an in-memory DataArray with respect to
    time."""
    data = eval(f'gridded_da_{time_type}')()
    x = data['time']
    y = data
    if not gridded:
        y = y.isel(lat=0, lon=0)
    slope = linear_slope(x, y, 'time')
    assert not slope.isnull().any()
    expected_shape = (3, 3) if gridded else ()
    assert slope.shape == expected_shape


@pytest.mark.parametrize('gridded', (True, False))
@pytest.mark.parametrize('time_type', TIME_TYPES)
def test_linear_slope_singular_data_da(
    gridded_da_datetime, gridded_da_cftime, gridded_da_float, time_type, gridded
):
    """Tests that ``linear_slope`` works between a grid or time series of data and
    another singular data (e.g., a Nino index)."""
    data = eval(f'gridded_da_{time_type}')()
    x = data.isel(lat=0, lon=0)
    y = data
    if not gridded:
        y = y.isel(lat=0, lon=0)
    slope = linear_slope(x, y, 'time')
    assert not slope.isnull().any()
    expected_shape = (3, 3) if gridded else ()
    assert slope.shape == expected_shape


@pytest.mark.parametrize('time_type', TIME_TYPES)
def test_linear_slope_gridded_to_gridded(
    gridded_da_datetime, gridded_da_cftime, gridded_da_float, time_type
):
    """Tests that ``linear_slope`` works between one grid of data and another grid
    of data."""
    data1 = eval(f'gridded_da_{time_type}')()
    data2 = eval(f'gridded_da_{time_type}')()
    slope = linear_slope(data1, data2, 'time')
    assert not slope.isnull().any()
    assert slope.shape == (3, 3)


@pytest.mark.parametrize('gridded', (True, False))
@pytest.mark.parametrize('time_type', TIME_TYPES)
def test_linear_slope_time_ds(
    gridded_ds_datetime, gridded_ds_cftime, gridded_ds_float, time_type, gridded
):
    """Tests that linear slope can be computed on an in-memory Dataset with respect to
    time."""
    ds = eval(f'gridded_ds_{time_type}')
    x = ds['time']
    if not gridded:
        ds = ds.isel(lat=0, lon=0)
    slope = linear_slope(x, ds, 'time')
    assert len(ds.data_vars) > 1
    assert not slope['foo'].isnull().any()
    assert not slope['bar'].isnull().any()
    expected_shape = (3, 3) if gridded else ()
    assert slope['foo'].shape == expected_shape
    assert slope['bar'].shape == expected_shape


@pytest.mark.parametrize('gridded', (True, False))
@pytest.mark.parametrize('time_type', TIME_TYPES)
def test_linear_slope_da_dask(
    gridded_da_datetime, gridded_da_cftime, gridded_da_float, time_type, gridded
):
    """Tests that linear slope can be computed on a dask DataArray."""
    data = eval(f'gridded_da_{time_type}')().chunk()
    x = data['time']
    y = data
    if not gridded:
        y = y.isel(lat=0, lon=0)
    expected = linear_slope(x.load(), y.load(), 'time')
    actual = linear_slope(x, y, 'time').compute()
    assert_allclose(expected, actual)


@pytest.mark.parametrize('gridded', (True, False))
@pytest.mark.parametrize('time_type', TIME_TYPES)
def test_linear_slope_ds_dask(
    gridded_ds_datetime, gridded_ds_cftime, gridded_ds_float, time_type, gridded
):
    """Tests that linear slope can be computed on a dask Dataset."""
    ds = eval(f'gridded_ds_{time_type}').chunk()
    x = ds['time']
    if not gridded:
        ds = ds.isel(lat=0, lon=0)
    expected = linear_slope(x.load(), ds.load(), 'time')
    actual = linear_slope(x, ds, 'time').compute()
    assert_allclose(expected, actual)


def test_linear_slope_time_conversion_accurate(ts_annual_da):
    """Tests that computed slope with cftime/datetime conversion is accurate."""
    data = ts_annual_da()
    data2 = data.copy()
    # Just create numeric time, e.g. time in years.
    data2['time'] = np.arange(data2.time.size)
    slope_cftime = linear_slope(data['time'], data, 'time')
    slope_ints = linear_slope(data2['time'], data2, 'time')
    diff = slope_cftime * 365.25 - slope_ints
    assert np.abs(diff) < 1e-4


def test_linear_slope_raises_error(ts_annual_da):
    """Tests that error is raised for `linear_slope` when independent variable is
    in y position."""
    with pytest.raises(ValueError):
        data = ts_annual_da()
        x = data['time']
        y = data
        linear_slope(y, x, 'time')
