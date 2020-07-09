MULTIPLE_TESTS = [
    'bonferroni',
    'sidak',
    'holm-sidak',
    'holm',
    'simes-hochberg',
    'hommel',
    'fdr_bh',
    'fdr_by',
    'fdr_tsbh',
    'fdr_tsbky',
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
# Not as simple as summing since there's nuance here in which years get a quarter year.
DAYS_PER_YEAR = {
    'noleap': 365,
    '365_day': 365.25,
    'standard': 365.25,
    'gregorian': 365.25,
    'proleptic_gregorian': 365.25,
    'all_leap': 366,
    '366_day': 366,
    '360_day': 360,
    'julian': 365.25,
}
CALENDARS = [k for k in DAYS_PER_MONTH]

# Useful when concatting list comprehension results. Speeds things up.
CONCAT_KWARGS = {'coords': 'minimal', 'compat': 'override'}
