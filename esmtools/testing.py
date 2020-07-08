import numpy as np
import xarray as xr
from scipy.stats import ttest_ind_from_stats as tti_from_stats
from statsmodels.stats.multitest import multipletests as statsmodels_multipletests

from .checks import is_xarray
from .constants import MULTIPLE_TESTS

__all__ = ['ttest_ind_from_stats', 'multipletests']


@is_xarray(0)
def multipletests(p, alpha=0.05, method=None, **multipletests_kwargs):
    """Apply statsmodels.stats.multitest.multipletests for multi-dimensional
    xr.objects.

    Args:
        p (xr.object): uncorrected p-values.
        alpha (optional float): FWER, family-wise error rate. Defaults to 0.05.
        method (str): Method used for testing and adjustment of pvalues. Can be
            either the full name or initial letters.  Available methods are:
            - bonferroni : one-step correction
            - sidak : one-step correction
            - holm-sidak : step down method using Sidak adjustments
            - holm : step-down method using Bonferroni adjustments
            - simes-hochberg : step-up method (independent)
            - hommel : closed method based on Simes tests (non-negative)
            - fdr_bh : Benjamini/Hochberg (non-negative)
            - fdr_by : Benjamini/Yekutieli (negative)
            - fdr_tsbh : two stage fdr correction (non-negative)
            - fdr_tsbky : two stage fdr correction (non-negative)
        **multipletests_kwargs (optional dict): is_sorted, returnsorted
           see statsmodels.stats.multitest.multitest

    Returns:
        reject (xr.object): true for hypothesis that can be rejected for given
            alpha
        pvals_corrected (xr.object): p-values corrected for multiple tests

    Example:
        >>> from esmtools.testing import multipletests
        >>> reject, xpvals_corrected = multipletests(p, method='fdr_bh')

    """
    if method is None:
        raise ValueError(
            f"Please indicate a method using the 'method=...' keyword. "
            f'Select from {MULTIPLE_TESTS}'
        )
    elif method not in MULTIPLE_TESTS:
        raise ValueError(
            f"Your method '{method}' is not in the accepted methods: {MULTIPLE_TESTS}"
        )

    # stack all to 1d array
    p_stacked = p.stack(s=p.dims)

    # mask only where not nan:
    # https://github.com/statsmodels/statsmodels/issues/2899
    mask = np.isfinite(p_stacked)
    pvals_corrected = xr.full_like(p_stacked, np.nan)
    reject = xr.full_like(p_stacked, np.nan)

    # apply test where mask
    reject[mask], pvals_corrected[mask], *_ = statsmodels_multipletests(
        p_stacked[mask], alpha=alpha, method=method, **multipletests_kwargs
    )

    return reject.unstack('s'), pvals_corrected.unstack('s')


def ttest_ind_from_stats(mean1, std1, nobs1, mean2, std2, nobs2, equal_var=True):
    """Parallelize scipy.stats.ttest_ind_from_stats and make dask-compatible.

    Args:
        mean1, mean2 (array_like): The means of samples 1 and 2.
        std1, std2 (array_like): The standard deviations of samples 1 and 2.
        nobs1, nobs2 (array_like): The number of observations for samples 1 and 2.
        equal_var (bool, optional): If True (default), perform a standard independent
            2 sample test that assumes equal population variances. If False, perform
            Welch's t-test, which does not assume equal population variance.

    Returns:
        statistic (float or array): The calculated t-statistics.
        pvalue (float or array): The two-tailed p-value.
    """
    return xr.apply_ufunc(
        tti_from_stats,
        mean1,
        std1,
        nobs1,
        mean2,
        std2,
        nobs2,
        equal_var,
        input_core_dims=[[], [], [], [], [], [], []],
        output_core_dims=[[], []],
        vectorize=True,
        dask='parallelized',
    )
