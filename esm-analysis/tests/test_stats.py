import climpred
import pytest
from esmtools.stats import linear_regression


@pytest.fixture
def single_ts():
    da = climpred.tutorial.load_dataset('FOSI-SST')
    da = da['SST']
    return da


def test_linear_regression(single_ts):
    """Tests that linear regression works on a single time series."""
    results = linear_regression(single_ts, dim='time')
    for v in results.data_vars:
        assert results[v]
