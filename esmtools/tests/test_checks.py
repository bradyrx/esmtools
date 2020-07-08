from esmtools.checks import has_missing


def test_has_missing(gridded_da_float, gridded_da_landmask, gridded_da_missing_data):
    """Tests that `has_missing` function works with various NaN configurations."""
    assert not has_missing(gridded_da_float())
    assert has_missing(gridded_da_landmask)
    assert has_missing(gridded_da_missing_data)
