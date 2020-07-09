import pandas as pd
import pytest

from esmtools.timeutils import TimeUtilAccessor


def test_timeutils_accessor(annual_gregorian):
    """Test that the `timeutils` accessor can be called."""
    # Calling the TimeUtilAccessor directly in the first instance to not trigger
    # F401 (importing unused module) from flake8.
    assert TimeUtilAccessor(annual_gregorian)._obj.notnull().all()


def test_annual_factor(
    annual_all_leap, annual_no_leap, annual_gregorian, annual_julian
):
    """Tests that the annual factor returned by timeutils is accurate."""
    assert annual_all_leap['time'].timeutils.annual_factor == 366.0
    assert annual_no_leap['time'].timeutils.annual_factor == 365.0
    assert annual_gregorian['time'].timeutils.annual_factor == 365.25
    assert annual_julian['time'].timeutils.annual_factor == 365.25


def test_calendar(annual_all_leap, annual_no_leap, annual_gregorian, annual_julian):
    """Tests that the calendar returned by timeutils is accurate."""
    assert annual_all_leap['time'].timeutils.calendar == 'all_leap'
    assert annual_no_leap['time'].timeutils.calendar == 'noleap'
    assert annual_gregorian['time'].timeutils.calendar == 'gregorian'
    assert annual_julian['time'].timeutils.calendar == 'julian'


@pytest.mark.parametrize('frequency', ('L', 'D', 'AS-NOV', 'W-TUE'))
def test_freq(annual_gregorian, frequency):
    """Tests that the calendar frequency returned by timeutils is accurate."""
    data = annual_gregorian
    data['time'] = pd.date_range('1990', freq=frequency, periods=data.time.size)
    assert data['time'].timeutils.freq == frequency


expected_slopes = {
    'L': 1 / (24 * 60 * 60 * 1e3),
    'D': 1,
    'AS-NOV': 365.25,
    'W-TUE': 7.0,
}


@pytest.mark.parametrize('frequency', ('L', 'D', 'AS-NOV', 'W-TUE'))
def test_slope_factor(annual_gregorian, frequency):
    """Tests that the slope factor returned by timeutils is accurate."""
    data = annual_gregorian
    data['time'] = pd.date_range('1990', freq=frequency, periods=data.time.size)
    assert data['time'].timeutils.slope_factor == expected_slopes[frequency]


def test_return_numeric_time_datetime(gridded_da_datetime):
    """Tests that datetimes are properly converted to numeric time by timeutils."""
    data = gridded_da_datetime()['time']
    assert data.timeutils.return_numeric_time().dtype == 'float64'


def test_return_numeric_time_cftime(gridded_da_cftime):
    """Tests that cftimes are properly converted to numeric time by timeutils."""
    data = gridded_da_cftime()['time']
    assert data.timeutils.return_numeric_time().dtype == 'float64'
