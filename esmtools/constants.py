MULTIPLE_TESTS = [
    "bonferroni",
    "sidak",
    "holm-sidak",
    "holm",
    "simes-hochberg",
    "hommel",
    "fdr_bh",
    "fdr_by",
    "fdr_tsbh",
    "fdr_tsbky",
]

# Number of days per month for each `cftime` calendar.
# Taken from xarray example:
# http://xarray.pydata.org/en/stable/examples/monthly-means.html
DAYS_PER_MONTH = {
    'noleap': [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
    '365_day': [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
    'standard': [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
    'gregorian': [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
    'proleptic_gregorian': [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
    'all_leap': [0, 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
    '366_day': [0, 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
    '360_day': [0, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30],
    'julian': [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
}
CALENDARS = [k for k in DAYS_PER_MONTH]

# Converts from `cftime` class name to netCDF convention for calendar
CFTIME_TO_NETCDF = {
    'DatetimeJulian': 'julian',
    'DatetimeProlepticGregorian': 'proleptic_gregorian',
    'DatetimeNoLeap': 'noleap',
    'DatetimeAllLeap': 'all_leap',
    'DatetimeGregorian': 'gregorian',
}
