import numpy as np
import pytest
from xarray.tests import assert_allclose

from esmtools.stats import linear_slope, linregress

TIME_TYPES = ['datetime', 'cftime', 'float']


@pytest.mark.parametrize('gridded', (True, False))
@pytest.mark.parametrize('time_type', TIME_TYPES)
@pytest.mark.parametrize('func', (linear_slope, linregress))
@pytest.mark.parametrize('single_arg', (True, False))
def test_linear_regression_time_da(
    gridded_da_datetime,
    gridded_da_cftime,
    gridded_da_float,
    time_type,
    gridded,
    func,
    single_arg,
):
    """Tests that linear slope can be computed on an in-memory DataArray with respect to
    time."""
    data = eval(f'gridded_da_{time_type}')()
    x = data['time']
    y = data if gridded else data.isel(lat=0, lon=0)
    if single_arg:
        result = func(y, dim='time')
    else:
        result = func(x, y, 'time')
    assert not result.isnull().any()
    if func == linear_slope:
        expected_shape = (3, 3) if gridded else ()
    elif func == linregress:
        expected_shape = (3, 3, 5) if gridded else (5,)
    assert result.shape == expected_shape


@pytest.mark.parametrize('gridded', (True, False))
@pytest.mark.parametrize('time_type', TIME_TYPES)
@pytest.mark.parametrize('func', (linear_slope, linregress))
@pytest.mark.parametrize('single_arg', (True, False))
def test_linear_regression_singular_data_da(
    gridded_da_datetime,
    gridded_da_cftime,
    gridded_da_float,
    time_type,
    gridded,
    func,
    single_arg,
):
    """Tests that ``linear_slope`` works between a grid or time series of data and
    another singular data (e.g., a Nino index)."""
    data = eval(f'gridded_da_{time_type}')()
    x = data.isel(lat=0, lon=0)
    y = data if gridded else data.isel(lat=0, lon=0)
    if single_arg:
        result = func(y, dim='time')
    else:
        result = func(x, y, 'time')
    assert not result.isnull().any()
    if func == linear_slope:
        expected_shape = (3, 3) if gridded else ()
    elif func == linregress:
        expected_shape = (3, 3, 5) if gridded else (5,)
    assert result.shape == expected_shape


@pytest.mark.parametrize('time_type', TIME_TYPES)
@pytest.mark.parametrize('func', (linear_slope, linregress))
def test_linear_regression_gridded_to_gridded(
    gridded_da_datetime, gridded_da_cftime, gridded_da_float, time_type, func
):
    """Tests that ``linear_slope`` works between one grid of data and another grid
    of data."""
    data1 = eval(f'gridded_da_{time_type}')()
    data2 = eval(f'gridded_da_{time_type}')()
    result = func(data1, data2, 'time')
    assert not result.isnull().any()
    expected_shape = (3, 3) if func == linear_slope else (3, 3, 5)
    assert result.shape == expected_shape


@pytest.mark.parametrize('gridded', (True, False))
@pytest.mark.parametrize('time_type', TIME_TYPES)
@pytest.mark.parametrize('func', (linear_slope, linregress))
@pytest.mark.parametrize('single_arg', (True, False))
def test_linear_regression_time_ds(
    gridded_ds_datetime,
    gridded_ds_cftime,
    gridded_ds_float,
    time_type,
    gridded,
    func,
    single_arg,
):
    """Tests that linear slope can be computed on an in-memory Dataset with respect to
    time."""
    ds = eval(f'gridded_ds_{time_type}')
    x = ds['time']
    if not gridded:
        ds = ds.isel(lat=0, lon=0)
    if single_arg:
        result = func(ds, dim='time')
    else:
        result = func(x, ds, 'time')
    assert len(ds.data_vars) > 1
    assert not result['foo'].isnull().any()
    assert not result['bar'].isnull().any()
    if func == linear_slope:
        expected_shape = (3, 3) if gridded else ()
    elif func == linregress:
        expected_shape = (3, 3, 5) if gridded else (5,)
    assert result['foo'].shape == expected_shape
    assert result['bar'].shape == expected_shape


@pytest.mark.parametrize('gridded', (True, False))
@pytest.mark.parametrize('time_type', TIME_TYPES)
@pytest.mark.parametrize('func', (linear_slope, linregress))
@pytest.mark.parametrize('single_arg', (True, False))
def test_linear_regression_da_dask(
    gridded_da_datetime,
    gridded_da_cftime,
    gridded_da_float,
    time_type,
    gridded,
    func,
    single_arg,
):
    """Tests that linear slope can be computed on a dask DataArray."""
    data = eval(f'gridded_da_{time_type}')().chunk()
    x = data['time']
    y = data if gridded else data.isel(lat=0, lon=0)
    if single_arg:
        expected = func(y.load(), dim='time')
        actual = func(y, dim='time').compute()
    else:
        expected = func(x.load(), y.load(), 'time')
        actual = func(x, y, 'time').compute()
    assert_allclose(expected, actual)


@pytest.mark.parametrize('gridded', (True, False))
@pytest.mark.parametrize('time_type', TIME_TYPES)
@pytest.mark.parametrize('func', (linear_slope, linregress))
@pytest.mark.parametrize('single_arg', (True, False))
def test_linear_slope_ds_dask(
    gridded_ds_datetime,
    gridded_ds_cftime,
    gridded_ds_float,
    time_type,
    gridded,
    func,
    single_arg,
):
    """Tests that linear slope can be computed on a dask Dataset."""
    ds = eval(f'gridded_ds_{time_type}').chunk()
    x = ds['time']
    if not gridded:
        ds = ds.isel(lat=0, lon=0)
    if single_arg:
        expected = func(ds.load(), dim='time')
        actual = func(ds, dim='time').compute()
    else:
        expected = func(x.load(), ds.load(), 'time')
        actual = func(x, ds, 'time').compute()
    assert_allclose(expected, actual)


def test_linear_slope_time_conversion_accurate(ts_annual_da):
    """Tests that computed slope with cftime/datetime conversion is accurate."""
    data = ts_annual_da()
    data2 = data.copy()
    # Just create numeric time, e.g. time in years.
    data2['time'] = np.arange(data2.time.size)
    slope_cftime = linear_slope(data['time'], data, 'time')
    slope_ints = linear_slope(data2['time'], data2, 'time')
    diff = slope_cftime - slope_ints
    assert np.abs(diff) < 1e-4


def test_linear_slope_raises_error(ts_annual_da):
    """Tests that error is raised for `linear_slope` when independent variable is
    in y position."""
    with pytest.raises(ValueError):
        data = ts_annual_da()
        x = data['time']
        y = data
        linear_slope(y, x, 'time')


def test_linear_slope_linregress_same_result(ts_annual_da):
    """Tests that ``linear_slope`` and ``linregress`` returns the same result
    for the slope parameter."""
    data = ts_annual_da()
    x = data['time']
    y = data
    m1 = linear_slope(x, y, 'time')
    m2 = linregress(x, y, 'time').sel(parameter='slope')
    assert np.abs(m1.values - m2.values) < 1e-15
