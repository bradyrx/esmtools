import numpy as np
import xarray as xr
from esmtools.stats import corr
import pytest


@pytest.fixture()
def gridded_ts_da():
    """Mock data of gridded time series."""
    # Wrapper so fixture can be called multiple times.
    # https://alysivji.github.io/pytest-fixures-with-function-arguments.html
    def _gen_data():
        data = np.random.rand(120, 10, 10)
        da = xr.DataArray(data, dims=["time", "lat", "lon"])
        # Monthly resolution time axis for 10 years.
        da["time"] = np.arange("1990-01", "2000-01", dtype="datetime64[M]")
        return da

    return _gen_data


@pytest.fixture()
def ts_monthly_da():
    """Mock time series at monthly resolution for ten years."""
    # Wrapper so fixture can be called multiple times.
    # https://alysivji.github.io/pytest-fixures-with-function-arguments.html
    def _gen_data():
        data = np.random.rand(120)
        da = xr.DataArray(data, dims=["time"])
        # Monthly resolution time axis for 10 years.
        da["time"] = np.arange("1990-01", "2000-01", dtype="datetime64[M]")
        return da

    return _gen_data


def test_corr_two_grids(gridded_ts_da):
    """Tests that correlations between two grids work."""
    x = gridded_ts_da()
    y = gridded_ts_da()
    corrcoeff = corr(x, y, dim="time")
    # check that there's no NaNs in the resulting output.
    assert not corrcoeff.isnull().any()


def test_corr_grid_and_ts(ts_monthly_da, gridded_ts_da):
    """Tests that correlation works between a grid and a time series."""
    x = ts_monthly_da()
    y = gridded_ts_da()
    corrcoeff = corr(x, y, dim="time")
    # check that there's no NaNs in the resulting output.
    assert not corrcoeff.isnull().any()


def test_corr_lead(gridded_ts_da):
    """Tests positive lead for correlation."""
    x = gridded_ts_da()
    y = gridded_ts_da()
    corrcoeff = corr(x, y, dim="time", lead=3)
    # check that there's no NaNs in the resulting output.
    assert not corrcoeff.isnull().any()


def test_corr_lag(gridded_ts_da):
    """Tests negative lead for correlation."""
    x = gridded_ts_da()
    y = gridded_ts_da()
    corrcoeff = corr(x, y, dim="time", lead=-3)
    # check that there's no NaNs in the resulting output.
    assert not corrcoeff.isnull().any()


def test_corr_return_p(gridded_ts_da):
    """Tests that p-value is returned properly for correlation."""
    x = gridded_ts_da()
    y = gridded_ts_da()
    corrcoeff, pval = corr(x, y, dim="time", return_p=True)
    # check that there's no NaNs in the resulting output.
    assert not (corrcoeff.isnull().any()) & (pval.isnull().any())
