from esmtools.stats import corr


def test_corr_two_grids(gridded_monthly_da):
    """Tests that correlations between two grids work."""
    x = gridded_monthly_da()
    y = gridded_monthly_da()
    corrcoeff = corr(x, y, dim='time')
    # check that there's no NaNs in the resulting output.
    assert not corrcoeff.isnull().any()


def test_corr_grid_and_ts(ts_monthly_da, gridded_monthly_da):
    """Tests that correlation works between a grid and a time series."""
    x = ts_monthly_da()
    y = gridded_monthly_da()
    corrcoeff = corr(x, y, dim='time')
    # check that there's no NaNs in the resulting output.
    assert not corrcoeff.isnull().any()


def test_corr_lead(gridded_monthly_da):
    """Tests positive lead for correlation."""
    x = gridded_monthly_da()
    y = gridded_monthly_da()
    corrcoeff = corr(x, y, dim='time', lead=3)
    # check that there's no NaNs in the resulting output.
    assert not corrcoeff.isnull().any()


def test_corr_lag(gridded_monthly_da):
    """Tests negative lead for correlation."""
    x = gridded_monthly_da()
    y = gridded_monthly_da()
    corrcoeff = corr(x, y, dim='time', lead=-3)
    # check that there's no NaNs in the resulting output.
    assert not corrcoeff.isnull().any()


def test_corr_return_p(gridded_monthly_da):
    """Tests that p-value is returned properly for correlation."""
    x = gridded_monthly_da()
    y = gridded_monthly_da()
    corrcoeff, pval = corr(x, y, dim='time', return_p=True)
    # check that there's no NaNs in the resulting output.
    assert not (corrcoeff.isnull().any()) & (pval.isnull().any())
