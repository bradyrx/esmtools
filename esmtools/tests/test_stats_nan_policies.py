import numpy as np
import pytest

from esmtools.stats import linear_slope, linregress, polyfit, rm_poly, rm_trend


@pytest.mark.parametrize("nan_policy", ("drop", "omit"))
@pytest.mark.parametrize("func", (polyfit, linear_slope, linregress, rm_poly, rm_trend))
def test_full_data_omit_nans(gridded_da_float, func, nan_policy):
    """Tests that the `drop`/`omit` nan policy works with a dataset that has no
    missing data."""
    data = gridded_da_float()
    x, y = data["time"], data
    args = (
        {"order": 1, "nan_policy": nan_policy}
        if func in [polyfit, rm_poly]
        else {"nan_policy": nan_policy}
    )
    result = func(x, y, **args)
    assert result.notnull().all()


@pytest.mark.parametrize("nan_policy", ("none", "propagate"))
@pytest.mark.parametrize("func", (polyfit, linear_slope, linregress, rm_poly, rm_trend))
def test_full_data_propagate_nans(gridded_da_float, func, nan_policy):
    """Tests that the `none`/`propagate` nan policy works with a dataset that has no
    missing data."""
    data = gridded_da_float()
    x, y = data["time"], data
    args = (
        {"order": 1, "nan_policy": nan_policy}
        if func in [polyfit, rm_poly]
        else {"nan_policy": nan_policy}
    )
    result = func(x, y, **args)
    assert result.notnull().all()


@pytest.mark.parametrize("func", (polyfit, linear_slope, linregress, rm_poly, rm_trend))
def test_full_data_raise_nans(gridded_da_float, func):
    """Tests that the `raise` nan policy does not raise an error with a dataset that
    has no missing data."""
    data = gridded_da_float()
    x, y = data["time"], data
    args = (
        {"order": 1, "nan_policy": "raise"}
        if func in [polyfit, rm_poly]
        else {"nan_policy": "raise"}
    )
    result = func(x, y, **args)
    assert result.notnull().all()


@pytest.mark.parametrize("nan_policy", ("drop", "omit"))
@pytest.mark.parametrize("func", (polyfit, linear_slope, linregress, rm_poly, rm_trend))
def test_landmask_omit_nans(gridded_da_landmask, func, nan_policy):
    """Tests that the `drop`/`omit` nan policy works with a dataset that simulates
    having land masking."""
    data = gridded_da_landmask
    x, y = data["time"], data
    args = (
        {"order": 1, "nan_policy": nan_policy}
        if func in [polyfit, rm_poly]
        else {"nan_policy": nan_policy}
    )
    result = func(x, y, **args)
    assert result.isel(lat=0, lon=0).isnull().all()
    assert result.isel(lat=1, lon=1).notnull().all()


@pytest.mark.parametrize("nan_policy", ("none", "propagate"))
@pytest.mark.parametrize("func", (polyfit, linear_slope, linregress, rm_poly, rm_trend))
def test_landmask_propagate_nans(gridded_da_landmask, func, nan_policy):
    """Tests that the `none`/`propagate` nan policy works with a dataset that simulates
    having land masking."""
    data = gridded_da_landmask
    x, y = data["time"], data
    args = (
        {"order": 1, "nan_policy": nan_policy}
        if func in [polyfit, rm_poly]
        else {"nan_policy": nan_policy}
    )
    result = func(x, y, **args)
    assert result.isel(lat=0, lon=0).isnull().all()
    assert result.isel(lat=1, lon=1).notnull().all()


@pytest.mark.parametrize("func", (polyfit, linear_slope, linregress, rm_poly, rm_trend))
def test_landmask_raise_nans(gridded_da_landmask, func):
    """Tests that error is raised for `raise` with a dataset that simulates having nan
    masking."""
    with pytest.raises(ValueError):
        data = gridded_da_landmask
        x, y = data["time"], data
        args = (
            {"order": 1, "nan_policy": "raise"}
            if func in [polyfit, rm_poly]
            else {"nan_policy": "raise"}
        )
        func(x, y, **args)


@pytest.mark.parametrize("nan_policy", ("drop", "omit"))
@pytest.mark.parametrize("func", (polyfit, linear_slope, linregress, rm_poly, rm_trend))
def test_missing_data_omit_nans(gridded_da_missing_data, func, nan_policy):
    """Tests that the `drop`/`omit` nan policy works with a dataset that simulates
    missing data."""
    data = gridded_da_missing_data
    x, y = data["time"], data
    args = (
        {"order": 1, "nan_policy": nan_policy}
        if func in [polyfit, rm_poly]
        else {"nan_policy": nan_policy}
    )
    result = func(x, y, **args)
    if func not in [rm_poly, rm_trend]:
        assert result.notnull().all()
    else:
        assert result.isel(lat=0, lon=0, time=5).isnull()


@pytest.mark.parametrize("nan_policy", ("none", "propagate"))
@pytest.mark.parametrize("func", (polyfit, linear_slope, linregress, rm_poly, rm_trend))
def test_missing_data_propagate_nans(gridded_da_missing_data, func, nan_policy):
    """Tests that the `none`/`propagate` nan policy works with a dataset that simulates
    missing data."""
    data = gridded_da_missing_data
    x, y = data["time"], data
    args = (
        {"order": 1, "nan_policy": nan_policy}
        if func in [polyfit, rm_poly]
        else {"nan_policy": nan_policy}
    )
    result = func(x, y, **args)
    assert result.isel(lat=0, lon=0).isnull().all()
    assert result.isel(lat=1, lon=0).isnull().all()
    assert result.isel(lat=0, lon=2).notnull().all()


@pytest.mark.parametrize("func", (polyfit, linear_slope, linregress, rm_poly, rm_trend))
def test_missing_data_raise_nans(gridded_da_missing_data, func):
    """Tests that error is raised for `raise` with a dataset that simulates missing
    data."""
    with pytest.raises(ValueError):
        data = gridded_da_missing_data
        x, y = data["time"], data
        args = (
            {"order": 1, "nan_policy": "raise"}
            if func in [polyfit, rm_poly]
            else {"nan_policy": "raise"}
        )
        func(x, y, **args)


@pytest.mark.parametrize("func", (polyfit, linear_slope, linregress, rm_poly, rm_trend))
def test_missing_independent_data_propagate_nans(gridded_da_float, func):
    """Tests that functions still work with nans in independent axis, in the case
    that you're fitting to something lik ENSO data with missing values. `polyfit`
    functions break with missing data in `x` but not in `y`."""
    x = gridded_da_float()
    y = gridded_da_float()
    x[3, 0, 0] = np.nan
    args = (
        {"order": 1, "nan_policy": "propagate"}
        if func in [polyfit, rm_poly]
        else {"nan_policy": "propagate"}
    )
    result = func(x, y, **args)
    assert result.isel(lat=0, lon=0).isnull().all()
