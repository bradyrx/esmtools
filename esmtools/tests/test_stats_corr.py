import xarray as xr

from esmtools.stats import autocorr, corr


def test_corr_two_grids(gridded_da_float):
    """Tests that correlations between two grids work."""
    x = gridded_da_float()
    y = gridded_da_float()
    corrcoeff = corr(x, y, dim='time')
    assert not corrcoeff.isnull().any()


def test_corr_two_grids_dask(gridded_da_float):
    """Tests that correlations between two grids work with dask."""
    x = gridded_da_float().chunk()
    y = gridded_da_float().chunk()
    expected = corr(x.load(), y.load(), dim='time')
    actual = corr(x, y, dim='time').compute()
    assert (expected == actual).all()


def test_corr_grid_and_ts(ts_monthly_da, gridded_da_datetime):
    """Tests that correlation works between a grid and a time series."""
    x = ts_monthly_da()
    y = gridded_da_datetime()
    corrcoeff = corr(x, y, dim='time')
    assert not corrcoeff.isnull().any()


def test_corr_grid_and_ts_dask(ts_monthly_da, gridded_da_datetime):
    """Tests that correlation works between a grid and a time series with dask."""
    x = ts_monthly_da().chunk()
    y = gridded_da_datetime().chunk()
    expected = corr(x.load(), y.load(), dim='time')
    actual = corr(x, y, dim='time').compute()
    assert (expected == actual).all()


def test_corr_lead(gridded_da_float):
    """Tests positive lead for correlation."""
    x = gridded_da_float()
    y = gridded_da_float()
    corrcoeff = corr(x, y, dim='time', lead=3)
    assert not corrcoeff.isnull().any()


def test_corr_lag(gridded_da_float):
    """Tests negative lead for correlation."""
    x = gridded_da_float()
    y = gridded_da_float()
    corrcoeff = corr(x, y, dim='time', lead=-3)
    assert not corrcoeff.isnull().any()


def test_corr_return_p(gridded_da_float):
    """Tests that p-value is returned properly for correlation."""
    x = gridded_da_float()
    y = gridded_da_float()
    corrcoeff, pval = corr(x, y, dim='time', return_p=True)
    assert not (corrcoeff.isnull().any()) & (pval.isnull().any())


def test_autocorr(gridded_da_float):
    """Tests that ``autocorr`` functions properly."""
    x = gridded_da_float()
    actual = autocorr(x, nlags=None)
    actual = autocorr(x, nlags=10)
    expected = xr.concat([corr(x, x, lead=i) for i in range(10)], 'lead')
    assert (actual.values == expected.values).all()
