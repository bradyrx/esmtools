import dask
import numpy as np
import xarray as xr
from xarray.tests import assert_allclose
from esmtools.stats import compute_slope
import pytest


TIME_TYPES = ["datetime", "cftime", "float"]
# TODO:
# * Fix and test time series
# * Test linear_regression function and see that it's approx the same.
@pytest.fixture()
def gridded_da_datetime():
    """Mock data of gridded time series in numpy datetime."""
    # Wrapper so fixture can be called multiple times.
    # https://alysivji.github.io/pytest-fixures-with-function-arguments.html
    def _gen_data():
        data = np.random.rand(60, 3, 3)
        da = xr.DataArray(data, dims=["time", "lat", "lon"])
        # Monthly resolution time axis for 5 years.
        da["time"] = np.arange("1990-01", "1995-01", dtype="datetime64[M]")
        return da

    return _gen_data


@pytest.fixture()
def gridded_da_cftime():
    """Mock data of gridded time series in cftime."""
    # Wrapper so fixture can be called multiple times.
    # https://alysivji.github.io/pytest-fixures-with-function-arguments.html
    def _gen_data():
        data = np.random.rand(60, 3, 3)
        da = xr.DataArray(data, dims=["time", "lat", "lon"])
        # Monthly resolution time axis for 5 years.
        da["time"] = xr.cftime_range("1990-01", "1995-01", freq="M")
        return da

    return _gen_data


@pytest.fixture()
def gridded_da_float():
    """Mock data of gridded time series in float time."""
    # Wrapper so fixture can be called multiple times.
    # https://alysivji.github.io/pytest-fixures-with-function-arguments.html
    def _gen_data():
        data = np.random.rand(60, 3, 3)
        da = xr.DataArray(data, dims=["time", "lat", "lon"])
        # Annual resolution time axis for 60 years.
        da["time"] = np.arange(1900, 1960)
        return da

    return _gen_data


@pytest.fixture()
def time_series_da():
    """Mock data of a single time series in numpy datetime."""
    # Wrapper so fixture can be called multiple times.
    # https://alysivji.github.io/pytest-fixures-with-function-arguments.html
    def _gen_data():
        data = np.random.rand(60)
        da = xr.DataArray(data, dims=["time"])
        # Monthly resolution time axis for 5 years.
        da["time"] = np.arange("1990-01", "1995-01", dtype="datetime64[M]")
        return da

    return _gen_data


def _create_dataset(data_func):
    """Creates an xarray Dataset from combining two instances of an above DataArray
    generator.

    Args:
        data_func (pytest fixture): One of the above pytest fixture functions.

    Returns:
        xarray Dataset with variables 'A' and 'B'.
    """
    a = data_func()
    a.name = "A"
    a = a.to_dataset()

    b = data_func()
    b.name = "B"
    b = b.to_dataset()
    return xr.merge([a, b])


@pytest.mark.parametrize("gridded", (True, False))
@pytest.mark.parametrize("time_type", TIME_TYPES)
def test_compute_slope_time_da(
    gridded_da_datetime, gridded_da_cftime, gridded_da_float, time_type, gridded
):
    """Tests that linear slope can be computed on an in-memory DataArray with respect to
    time."""
    gen = eval(f"gridded_da_{time_type}")
    y = gen()
    if not gridded:
        y = y.isel(lat=0, lon=0)
    slope = compute_slope(y["time"], y)
    assert not slope.isnull().any()


@pytest.mark.parametrize("gridded", (True, False))
@pytest.mark.parametrize("time_type", TIME_TYPES)
def test_compute_slope_time_ds(
    gridded_da_datetime, gridded_da_cftime, gridded_da_float, time_type, gridded
):
    """Tests that linear slope can be computed on an in-memory Dataset with respect to
    time."""
    gen = eval(f"gridded_da_{time_type}")
    ds = _create_dataset(gen)
    if not gridded:
        ds = ds.isel(lat=0, lon=0)
    slope = compute_slope(ds["time"], ds)
    assert len(ds.data_vars) > 1
    assert not slope["A"].isnull().any()
    assert not slope["B"].isnull().any()


# @pytest.mark.skip(reason="Dask breaks with 1D right now for some reason")
@pytest.mark.parametrize("gridded", (True, False))
@pytest.mark.parametrize("time_type", TIME_TYPES)
def test_compute_slope_time_da_dask(
    gridded_da_datetime, gridded_da_cftime, gridded_da_float, time_type, gridded
):
    """Tests that linear slope can be computed on out-of-memory DataArray with respect
    to time."""
    gen = eval(f"gridded_da_{time_type}")
    y = gen().chunk()
    if not gridded:
        y = y.isel(lat=0, lon=0)
    slope_dask = compute_slope(y["time"], y)
    slope_loaded = compute_slope(y["time"].load(), y.load())
    assert dask.is_dask_collection(slope_dask)
    assert_allclose(slope_dask.compute(), slope_loaded)


# @pytest.mark.skip(reason="Dask breaks with 1D right now for some reason")
@pytest.mark.parametrize("gridded", (True, False))
@pytest.mark.parametrize("time_type", TIME_TYPES)
def test_compute_slope_time_ds_dask(
    gridded_da_datetime, gridded_da_cftime, gridded_da_float, time_type, gridded
):
    """Tests that linear slope can be computed on out-of-memory Dataset with respect to
    time."""
    gen = eval(f"gridded_da_{time_type}")
    ds = _create_dataset(gen).chunk()
    if not gridded:
        ds = ds.isel(lat=0, lon=0)
    slope_dask = compute_slope(ds["time"], ds)
    slope_loaded = compute_slope(ds["time"].load(), ds.load())
    assert len(slope_dask.data_vars) > 1
    assert dask.is_dask_collection(slope_dask)
    assert_allclose(slope_dask.compute(), slope_loaded)


@pytest.mark.parametrize("gridded", (True, False))
@pytest.mark.parametrize("time_type", TIME_TYPES)
def test_compute_slope_x_y_da(
    gridded_da_datetime, gridded_da_cftime, gridded_da_float, time_type, gridded
):
    """Tests that linear slope can be computed between two different in-memory
    DataArrays."""
    gen = eval(f"gridded_da_{time_type}")
    x = gen()
    y = gen()
    if not gridded:
        x, y = x.isel(lat=0, lon=0), y.isel(lat=0, lon=0)
    slope = compute_slope(x, y)
    assert not slope.isnull().any()


# @pytest.mark.skip(reason="Dask breaks with 1D right now for some reason")
@pytest.mark.parametrize("gridded", (True, False))
@pytest.mark.parametrize("time_type", TIME_TYPES)
def test_compute_slope_x_y_da_dask(
    gridded_da_datetime, gridded_da_cftime, gridded_da_float, time_type, gridded
):
    """Tests that linear slope can be computed between two different out-of-memory
    DataArrays."""
    gen = eval(f"gridded_da_{time_type}")
    x = gen().chunk()
    y = gen().chunk()
    if not gridded:
        x, y = x.isel(lat=0, lon=0), y.isel(lat=0, lon=0)
    slope_dask = compute_slope(x, y)
    slope_loaded = compute_slope(x.load(), y.load())
    assert dask.is_dask_collection(slope_dask)
    assert_allclose(slope_dask.compute(), slope_loaded)


@pytest.mark.parametrize("gridded", (True, False))
@pytest.mark.parametrize("time_type", TIME_TYPES)
def test_compute_slope_x_y_ds(
    gridded_da_datetime, gridded_da_cftime, gridded_da_float, time_type, gridded
):
    """Tests that linear slope can be computed between two different in-memory
    Datasets."""
    gen = eval(f"gridded_da_{time_type}")
    x, y = _create_dataset(gen), _create_dataset(gen)
    if not gridded:
        x, y = x.isel(lat=0, lon=0), y.isel(lat=0, lon=0)
    slope = compute_slope(x, y)
    assert len(x.data_vars) > 1
    assert len(y.data_vars) > 1
    assert not slope["A"].isnull().any()
    assert not slope["B"].isnull().any()


# @pytest.mark.skip(reason="Dask breaks with 1D right now for some reason")
@pytest.mark.parametrize("gridded", (True, False))
@pytest.mark.parametrize("time_type", TIME_TYPES)
def test_compute_slope_x_y_ds_dask(
    gridded_da_datetime, gridded_da_cftime, gridded_da_float, time_type, gridded
):
    """Tests that linear slope can be computed between two different out-of-memory
    Datasets."""
    gen = eval(f"gridded_da_{time_type}")
    x, y = _create_dataset(gen).chunk(), _create_dataset(gen).chunk()
    if not gridded:
        x, y = x.isel(lat=0, lon=0), y.isel(lat=0, lon=0)
    slope_dask = compute_slope(x, y)
    slope_loaded = compute_slope(x.load(), y.load())
    assert len(x.data_vars) > 1
    assert len(y.data_vars) > 1
    assert dask.is_dask_collection(slope_dask)
    assert_allclose(slope_dask.compute(), slope_loaded)
