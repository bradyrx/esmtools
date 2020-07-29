import numpy as np
import pytest
import xarray as xr

from esmtools.temporal import to_annual


@pytest.mark.parametrize(
    'dataset',
    (
        pytest.lazy_fixture('gridded_da_datetime'),
        pytest.lazy_fixture('gridded_da_cftime'),
    ),
)
def test_to_annual(dataset):
    """General checks that `to_annual` time conversion is working as expected."""
    data = dataset()
    result = to_annual(data)
    assert result.notnull().all()
    assert 'year' in result.dims


def test_to_annual_accuracy(ts_monthly_da):
    """Tests that weighted sum correctly takes the annual mean."""
    data = ts_monthly_da().isel(time=slice(0, 12))
    MONTH_LENGTHS = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    manual_sum = []
    for i in range(len(data)):
        manual_sum.append(data[i].values * MONTH_LENGTHS[i] / 365)
    expected = sum(manual_sum)
    actual = to_annual(data)
    assert np.abs(actual.values - expected) < 1e-5


def test_to_annual_retains_nans(gridded_da_landmask):
    """Tests that `to_annual` function retains nans where the original dataset had nans

    .. note::
        Previous versions of `esmtools` did not do this, since xarray automatically
        skips nans with the grouped sum operator, returning zeroes where there used
        to be nans.
    """
    data = gridded_da_landmask
    data['time'] = xr.cftime_range(
        start='1990-01', freq='MS', periods=data['time'].size
    )
    result = to_annual(data)
    assert result.isel(lat=0, lon=0).isnull().all()
