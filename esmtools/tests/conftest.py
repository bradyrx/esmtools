import numpy as np
import pytest
import xarray as xr


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
        da["time"] = xr.cftime_range(start="1990-01", freq="M", periods=da.time.size)
        return da

    return _gen_data


@pytest.fixture()
def gridded_ds_float(gridded_da_float):
    """Mock data of a gridded time series as an xarray Dataset."""
    data = xr.Dataset()
    data["foo"] = gridded_da_float()
    data["bar"] = gridded_da_float()
    return data


@pytest.fixture()
def gridded_ds_datetime(gridded_da_datetime):
    """Mock data of a gridded time series as an xarray Dataset."""
    data = xr.Dataset()
    data["foo"] = gridded_da_datetime()
    data["bar"] = gridded_da_datetime()
    return data


@pytest.fixture()
def gridded_ds_cftime(gridded_da_cftime):
    """Mock data of a gridded time series as an xarray Dataset."""
    data = xr.Dataset()
    data["foo"] = gridded_da_cftime()
    data["bar"] = gridded_da_cftime()
    return data


@pytest.fixture()
def ts_monthly_da():
    """Mock time series at monthly resolution for ten years."""
    # Wrapper so fixture can be called multiple times.
    # https://alysivji.github.io/pytest-fixures-with-function-arguments.html
    def _gen_data():
        data = np.random.rand(60)
        da = xr.DataArray(data, dims=["time"])
        # Monthly resolution time axis for 10 years.
        da["time"] = np.arange("1990-01", "1995-01", dtype="datetime64[M]")
        return da

    return _gen_data


@pytest.fixture()
def ts_annual_da():
    """Mock time series at annual resolution for twenty years."""

    def _gen_data():
        data = np.random.rand(20)
        da = xr.DataArray(data, dims=["time"])
        da["time"] = xr.cftime_range("1990", freq="YS", periods=da.time.size)
        return da

    return _gen_data
