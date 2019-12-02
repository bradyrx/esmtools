import numpy as np
import xarray as xr
from scipy.stats import ttest_ind_from_stats as tti_from_stats
from statsmodels.stats.multitest import multipletests
from .constants import MULTIPLE_TESTS
from .utils import check_xarray


def ttest_ind_from_stats(mean1, std1, nobs1, mean2, std2, nobs2):
    """Parallelize scipy.stats.ttest_ind_from_stats."""
    return xr.apply_ufunc(
        tti_from_stats,
        mean1,
        std1,
        nobs1,
        mean2,
        std2,
        nobs2,
        input_core_dims=[[], [], [], [], [], []],
        output_core_dims=[[], []],
        vectorize=True,
        dask="parallelized",
    )


@check_xarray(0)
def multipletest(p, alpha=0.05, method="fdr_bh", **multipletests_kwargs):
    """Apply statsmodels.stats.multitest.multipletests for multi-dimensional
    xr.objects.

    Args:
        p (xr.object): uncorrected p-values.
        alpha (type): FWER, family-wise error rate. Defaults to 0.05.
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
           Defaults to 'fdr_bh'.
        **multipletests_kwargs (optional dict): is_sorted, returnsorted
           see statsmodels.stats.multitest.multitest

    Returns:
        reject (xr.object): true for hypothesis that can be rejected for given
            alpha
        pvals_corrected (xr.object): p-values corrected for multiple tests

    Example:
        reject, xpvals_corrected = xr_multipletest(p, method='fdr_bh')
    """
    if method not in MULTIPLE_TESTS:
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
    reject[mask], pvals_corrected[mask], *_ = multipletests(
        p_stacked[mask], alpha=alpha, method=method, **multipletests_kwargs
    )

    return reject.unstack("s"), pvals_corrected.unstack("s")
