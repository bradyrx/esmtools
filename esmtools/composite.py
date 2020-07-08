import warnings

import xarray as xr

from .checks import is_xarray
from .stats import standardize
from .testing import ttest_ind_from_stats


def _create_composites(anomaly_field, index, threshold=1, dim="time"):
    """Creates composite from some variable's anomaly field and a climate
    index"""
    index_comp = xr.full_like(index, "none", dtype="U4")
    index_comp[index >= threshold] = "pos"
    index_comp[index <= -threshold] = "neg"
    composite = anomaly_field.groupby(index_comp.rename("index"))
    return composite


@is_xarray([0, 1])
def composite_analysis(
    field, index, threshold=1, plot=False, ttest=False, psig=0.05, **plot_kwargs
):
    """Create composite maps based on some variable's response to a climate
    index.

    .. note::
        Make sure that the field and index are detrended prior to using this
        function if needed.

    Args:
        field (xr.object): Variable to create composites for. Contains dims
                           `time` and 2 spatial.
        index (xr.object): Climate index time series.
        threshold (float): Threshold value for standardized composite.
                           Defaults to 1.
        plot (bool): Quick plot and no returns. Defaults to False.
        ttest (bool): Apply `ttest` whether pos/neg different from mean.
                      Defaults to False.
        psig (float): Significance level for ttest. Defaults to 0.05.
        **plot_kwargs (type): kwargs to pass to xarray's plot function.

    Returns:
        composite (xr.object): Positive and negative composite if `not plot`.

    References:
        * Motivated from Ryan Abernathy's notebook here:
          https://rabernat.github.io/research_computing/xarray.html

    """

    def compute_ttest_for_composite(composite, index, psig):
        """
        Computes the ttest for the composite relative to neutral years and
        returns a masked map based on some alpha level.

        Args:
            composite: The grouped composite object
            index: 'pos' or 'neg'
            psig: Significance level for ttest
        """
        # Suppress NaN of empty slice warning.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            m1 = composite.mean("time").sel(index=index)
            s1 = composite.std("time").sel(index=index)
            n1 = len(composite.groups[index])
            m2 = composite.mean("time").sel(index="none")
            s2 = composite.std("time").sel(index="none")
            n2 = len(composite.groups["none"])
            t, p = ttest_ind_from_stats(m1, s1, n1, m2, s2, n2)
            return composite.mean("time").sel(index=index).where(p < psig)

    # Raise error if time slices are different.
    if field.time.size != index.time.size:
        raise ValueError(
            f"""Time periods for field and index do not match.
        field: {field.time.size}
        index: {index.time.size}"""
        )

    index = standardize(index)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        field = field - field.mean("time")
    composite = _create_composites(field, index, threshold=threshold)

    if ttest:
        # test if pos different from none
        comp_pos = compute_ttest_for_composite(composite, "pos", psig)
        # test if neg different from none
        comp_neg = compute_ttest_for_composite(composite, "neg", psig)
        composite = xr.concat([comp_pos, comp_neg], dim="index")
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            composite = composite.mean("time").sel(index=["pos", "neg"])

    composite["index"] = ["positive", "negative"]

    if plot:
        composite.plot(col="index", **plot_kwargs)
    else:
        return composite
