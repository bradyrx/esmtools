import numpy as np

from esmtools.utils import match_nans


def test_match_nans_no_nans(ts_annual_da):
    """Tests that `match_nans` doesn't modify arrays without nans."""
    x = ts_annual_da()
    y = ts_annual_da()
    x, y = match_nans(x, y)
    assert x.notnull().all()
    assert y.notnull().all()


def test_match_nans_pairwise(ts_annual_da):
    """Tests that `match_nans` actually matches nans pairwise."""
    x = ts_annual_da()
    y = ts_annual_da()
    x[3] = np.nan
    y[0] = np.nan
    x, y = match_nans(x, y)
    assert x[3].isnull()
    assert x[0].isnull()
    assert y[3].isnull()
    assert y[0].isnull()


def test_match_nans_doesnt_modify_original(ts_annual_da):
    """Tests that `match_nans` does not mutate original time series."""
    x = ts_annual_da()
    y = ts_annual_da()
    x[3] = np.nan
    y[0] = np.nan
    x_mod, y_mod = match_nans(x, y)
    assert x[0].notnull()
    assert y[3].notnull()


def test_int_arrays_apply_nans():
    """Tests that match nans converts int arrays into floats when adding nans to avoid
    returning an int nan (which is a crazy number)."""
    x = np.array([1, 2, 3, 4, 5]).astype("int")
    y = np.array([3, np.nan, 4, 5, 6])
    x, y = match_nans(x, y)
    assert x.dtype == "float"
    assert np.isnan(x).sum() > 0
