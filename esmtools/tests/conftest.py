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
def gridded_da_landmask(gridded_da_float):
    """Mock data for gridded time series in float time with land mask (nans over
    all time in some grid cells)."""
    data = gridded_da_float()
    # Mask arbitrary chunk through all time to simulate land mask.
    data = data.where((data.lat < 2) & (data.lon > 0))
    return data


@pytest.fixture()
def gridded_da_missing_data(gridded_da_float):
    """Mock data for gridded time series in float time with missing data (nans over
    some time points in different grid cells)."""
    data = gridded_da_float()
    # Add nan to arbitrary time steps.
    data[5, 0, 0] = np.nan
    data[3, 1, 0] = np.nan
    return data


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
    """Mock data of a gridded time series as an xarray Dataset with float time."""
    data = xr.Dataset()
    data["foo"] = gridded_da_float()
    data["bar"] = gridded_da_float()
    return data


@pytest.fixture()
def gridded_ds_datetime(gridded_da_datetime):
    """Mock data of a gridded time series as an xarray Dataset with numpy datetime."""
    data = xr.Dataset()
    data["foo"] = gridded_da_datetime()
    data["bar"] = gridded_da_datetime()
    return data


@pytest.fixture()
def gridded_ds_cftime(gridded_da_cftime):
    """Mock data of a gridded time series as an xarray Dataset with cftime."""
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


@pytest.fixture()
def annual_all_leap():
    """Mock annual 12-year time series with an all leap calendar."""
    data = xr.DataArray(np.random.rand(12,), dims=["time"])
    data["time"] = xr.cftime_range(
        "1990", freq="YS", periods=data.time.size, calendar="all_leap"
    )
    return data


@pytest.fixture()
def annual_no_leap():
    """Mock annual 12-year time series with a no leap calendar."""
    data = xr.DataArray(np.random.rand(12,), dims=["time"])
    data["time"] = xr.cftime_range(
        "1990", freq="YS", periods=data.time.size, calendar="noleap"
    )
    return data


@pytest.fixture()
def annual_gregorian():
    """Mock annual 12-year time series with a Gregorian calendar."""
    data = xr.DataArray(np.random.rand(12,), dims=["time"])
    data["time"] = xr.cftime_range(
        "1990", freq="YS", periods=data.time.size, calendar="gregorian"
    )
    return data


@pytest.fixture()
def annual_julian():
    """Mock annual 12-year time series with a Julian calendar."""
    data = xr.DataArray(np.random.rand(12,), dims=["time"])
    data["time"] = xr.cftime_range(
        "1990", freq="YS", periods=data.time.size, calendar="julian"
    )
    return data
