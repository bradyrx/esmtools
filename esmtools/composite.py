import xarray as xr
from scipy.stats import ttest_ind as tti
from scipy.stats import ttest_ind_from_stats as tti_from_stats


def ttest_ind(a, b, dim='time'):
    """Parallelize scipy.stats.ttest_ind."""
    return xr.apply_ufunc(tti, a, b, input_core_dims=[[dim], [dim]],
                          output_core_dims=[[], []],
                          vectorize=True, dask='parallelized')


def ttest_ind_from_stats(mean1, std1, nobs1, mean2, std2, nobs2):
    """Parallelize scipy.stats.ttest_ind_from_stats."""
    return xr.apply_ufunc(tti_from_stats, mean1, std1, nobs1, mean2, std2,
                          nobs2,
                          input_core_dims=[[], [], [], [], [], []],
                          output_core_dims=[[], []],
                          vectorize=True, dask='parallelized')


def standardize(ds, dim='time'):
    return (ds - ds.mean(dim)) / ds.std(dim)


def _create_composites(anomaly_field, timeseries, threshold=1, dim='time'):
    index_comp = xr.full_like(timeseries, 'none', dtype='U4')
    index_comp[timeseries >= threshold] = 'pos'
    index_comp[timeseries <= -threshold] = 'neg'
    composite = anomaly_field.groupby(
        index_comp.rename('index'))
    return composite


def composite_analysis(field,
                       timeseries,
                       threshold=1,
                       plot=False,
                       ttest=True,
                       psig=0.05,
                       **plot_kwargs):
    """Short summary.

    Args:
        field (xr.object): contains dims: 'time', 2 spatial.
        timeseries (xr.object): Create composite based on timeseries.
        threshold (float): threshold value for positive composite.
                        Defaults to 1.
        plot (bool): quick plot and no returns. Defaults to False.
        ttest (bool): Apply `ttest` whether pos/neg different from mean.
                      Defaults to True.
        psig (float): Significance level for ttest. Defaults to 0.05.
        **plot_kwargs (type): Description of parameter `**plot_kwargs`.

    Returns:
        composite (xr.object): pos and negative composite if `not plot`.

    """
    index = standardize(timeseries)
    field = field - field.mean('time')
    composite = _create_composites(field, index, threshold=threshold)

    if ttest:
        # test if pos different from none
        index = 'pos'
        m1 = composite.mean('time').sel(index=index)
        s1 = composite.std('time').sel(index=index)
        n1 = len(composite.groups[index])
        index = 'none'
        m2 = composite.mean('time').sel(index=index)
        s2 = composite.std('time').sel(index=index)
        n2 = len(composite.groups[index])

        t, p = ttest_ind_from_stats(m1, s1, n1, m2, s2, n2)
        comp_pos = composite.mean('time').sel(index='pos').where(p < psig)

        # test if neg different from none
        index = 'neg'
        m1 = composite.mean('time').sel(index=index)
        s1 = composite.std('time').sel(index=index)
        n1 = len(composite.groups[index])
        index = 'none'
        m2 = composite.mean('time').sel(index=index)
        s2 = composite.std('time').sel(index=index)
        n2 = len(composite.groups[index])

        t, p = ttest_ind_from_stats(m1, s1, n1, m2, s2, n2)
        comp_neg = composite.mean('time').sel(index='neg').where(p < psig)

        composite = xr.concat([comp_pos, comp_neg], dim='index')
    else:
        composite = composite.mean('time').sel(index=['pos', 'neg'])

    composite['index'] = ['positive', 'negative']

    if plot:
        composite.plot(col='index', **plot_kwargs)
    else:
        return composite
