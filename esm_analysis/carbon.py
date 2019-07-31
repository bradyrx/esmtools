import warnings

import numpy as np
import xarray as xr
from tqdm import tqdm

from .stats import linear_regression, nanmean, rm_poly
from .utils import check_xarray


@check_xarray([0, 1])
def co2_sol(t, s):
    """Compute CO2 solubility per the equation used in CESM.

    .. note::
        See ``co2calc.F90`` for the calculation of CO2 solubility in CESM.

    Args:
        t (xarray object): SST (degC)
        s (xarray object): SSS (PSU)

    Return:
        ff (xarray object): Value of solubility in mol/kg/atm

    References:
        Weiss & Price (1980, Mar. Chem., 8, 347-359; Eq 13 with table 6 values)

    Examples:
        >>> from esm_analysis.carbon import co2_sol
        >>> import numpy as np
        >>> import xarray as xr
        >>> t = xr.DataArray(np.random.randint(10, 25, size=(100, 10, 10)),
                dims=['time', 'lat', 'lon'])
        >>> s = xr.DataArray(np.random.randint(30, 35, size=(100, 10, 10)),
                dims=['time', 'lat', 'lon'])
        >>> ff = co2_sol(t, s)
    """

    def sol_calc(t, s):
        a = [-162.8301, 218.2968, 90.9241, -1.47696]
        b = [0.025695, -0.025225, 0.0049867]
        t = (t + 273.15) * 0.01
        t_sq = t ** 2
        t_inv = 1.0 / t
        log_t = np.log(t)
        d0 = b[2] * t_sq + b[1] * t + b[0]
        # Compute solubility in mol.kg^{-1}.atm^{-1}
        ff = np.exp(a[0] + a[1] * t_inv + a[2] * log_t + a[3] * t_sq + d0 * s)
        return ff

    ff = xr.apply_ufunc(
        sol_calc, t, s, input_core_dims=[[], []], vectorize=True, dask='allowed'
    )
    ff.attrs['units'] = 'mol/kg/atm'
    return ff


@check_xarray(0)
def schmidt(t):
    """Computes the dimensionless Schmidt number.

    .. note::
        The polynomials used are for SST ranges between 0 and 30C and a salinity of 35.

    Args:
        t (xarray object): SST (degC)

    Return:
        Sc (xarray object): Schmidt number (dimensionless)

    References:
        Sarmiento and Gruber (2006). Ocean Biogeochemical Dynamics. Table 3.3.1

    Examples:
        >>> from esm_analysis.carbon import schmidt
        >>> import numpy as np
        >>> import xarray as xr
        >>> t = xr.DataArray(np.random.randint(10, 25, size=(100, 10, 10)),
                    dims=['time', 'lat', 'lon'])
        >>> Sc = schmidt(t)
    """

    def calc_schmidt(t):
        c = [2073.1, 125.62, 3.6276, 0.043219]
        Sc = c[0] - c[1] * t + c[2] * (t ** 2) - c[3] * (t ** 3)
        return Sc

    Sc = xr.apply_ufunc(
        calc_schmidt, t, input_core_dims=[[]], vectorize=True, dask='allowed'
    )
    return Sc


@check_xarray(0)
def temp_decomp_takahashi(ds, time_dim='time', temperature='tos', pco2='spco2'):
    """Decompose surface pCO2 into thermal and non-thermal components.

    .. note::
        This expects cmorized variable names. You can pass keywords to change that
        or rename your variables accordingly.

    Args:
        ds (xarray.Dataset): Contains two variables:
            * `tos` (sea surface temperature in degC)
            * `spco2` (surface pCO2 in uatm)

    Return:
        decomp (xr.Dataset): Decomposed thermal and non-thermal components.

    References:
        Takahashi, Taro, Stewart C. Sutherland, Colm Sweeney, Alain Poisson, Nicolas
        Metzl, Bronte Tilbrook, Nicolas Bates, et al. “Global Sea–Air CO2 Flux
        Based on Climatological Surface Ocean PCO2, and Seasonal Biological and
        Temperature Effects.” Deep Sea Research Part II: Topical Studies in
        Oceanography, The Southern Ocean I: Climatic Changes in the Cycle of
        Carbon in the Southern Ocean, 49, no. 9 (January 1,2002): 1601–22.
        https://doi.org/10/dmk4f2.

    Examples:
        >>> from esm_analysis.carbon import temp_decomp_takahashi
        >>> import numpy as np
        >>> import xarray as xr
        >>> t = xr.DataArray(np.random.randint(10, 25, size=(100, 10, 10)),
                dims=['time', 'lat', 'lon']).rename('tos')
        >>> pco2 = xr.DataArray(np.random.randint(350, 400, size=(100, 10, 10)),
                dims=['time', 'lat', 'lon']).rename('spco2')
        >>> ds = xr.merge([t, pco2])
        >>> decomp = temp_decomp_takahashi(ds)
    """
    if temperature not in ds.data_vars:
        raise ValueError(f'{temperature} is not a variable in your dataset.')
    if pco2 not in ds.data_vars:
        raise ValueError(f'{pco2} is not a variable in your dataset.')

    fac = 0.0432
    tos_mean = ds[temperature].mean(time_dim)
    tos_diff = ds[temperature] - tos_mean
    thermal = (ds[pco2].mean(time_dim) * (np.exp(tos_diff * fac))).rename('thermal')
    non_thermal = (ds[pco2] * (np.exp(tos_diff * -fac))).rename('non_thermal')
    decomp = xr.merge([thermal, non_thermal])
    decomp.attrs[
        'description'
    ] = 'Takahashi decomposition of pCO2 into thermal and non-thermal components.'
    return decomp


@check_xarray([0, 1])
def potential_pco2(t_insitu, pco2_insitu):
    """Calculate potential pCO2 in the interior ocean.

    .. note::
        Requires the first index of depth to be at the surface.

    Args:
        t_insitu (xarray object): Temperature with depth [degC]
        pco2_insitu (xarray object): pCO2 with depth [uatm]

    Return:
        pco2_potential (xarray object): potential pCO2 with depth

    Reference:
        Sarmiento, Jorge Louis, and Nicolas Gruber. Ocean Biogeochemical Dynamics.
        Princeton, NJ: Princeton Univ. Press, 2006., p.421, eq. (10:3:1)

    Examples:
        >>> from esm_analysis.carbon import potential_pco2
        >>> import numpy as np
        >>> import xarray as xr
        >>> t_insitu = xr.DataArray(np.random.randint(0, 20, size=(100, 10, 30)),
            dims=['time', 'lat', 'depth'])
        >>> pco2_insitu = xr.DataArray(np.random.randint(350, 500, size=(100, 10, 30)),
            dims=['time', 'lat', 'depth'])
        >>> pco2_potential = potential_pco2(t_insitu, pco2_insitu)
    """
    t_sfc = t_insitu.isel(depth=0)
    pco2_potential = pco2_insitu * (1 + 0.0423 * (t_sfc - t_insitu))
    return pco2_potential


@check_xarray(0)
def spco2_sensitivity(ds):
    """Compute sensitivity of surface pCO2 to changes in driver variables.

    Args:
        ds (xr.Dataset): containing cmorized variables:
                         * spco2 [uatm]: ocean pCO2 at surface
                         * talkos[mmol m-3]: Alkalinity at ocean surface
                         * dissicos[mmol m-3]: DIC at ocean surface
                         * tos [C] : temperature at ocean surface
                         * sos [psu] : salinity at ocean surface

    Returns:
        sensitivity (xr.Dataset):

    References:
        * Lovenduski, Nicole S., Nicolas Gruber, Scott C. Doney, and Ivan D. Lima.
          “Enhanced CO2 Outgassing in the Southern Ocean from a Positive Phase of
          the Southern Annular Mode.” Global Biogeochemical Cycles 21, no. 2
          (2007). https://doi.org/10/fpv2wt.
        * Sarmiento, Jorge Louis, and Nicolas Gruber. Ocean Biogeochemical Dynamics.
          Princeton, NJ: Princeton Univ. Press, 2006., p.421, eq. (10:3:1)

    Examples:
        >>> from esm_analysis.carbon import spco2_sensitivity
        >>> import numpy as np
        >>> import xarray as xr
        >>> tos = xr.DataArray(np.random.randint(15, 30, size=(100, 10, 10)),
                dims=['time', 'lat', 'lon']).rename('tos')
        >>> sos = xr.DataArray(np.random.randint(30, 35, size=(100, 10, 10)),
                dims=['time', 'lat', 'lon']).rename('sos')
        >>> spco2 = xr.DataArray(np.random.randint(350, 400, size=(100, 10, 10)),
                dims=['time', 'lat', 'lon']).rename('spco2')
        >>> dissicos = xr.DataArray(np.random.randint(1900, 2100, size=(100, 10, 10)),
                dims=['time', 'lat', 'lon']).rename('dissicos')
        >>> talkos = xr.DataArray(np.random.randint(2100, 2300, size=(100, 10, 10)),
                dims=['time', 'lat', 'lon']).rename('talkos')
        >>> ds = xr.merge([tos, sos, spco2, dissicos, talkos])
        >>> sensitivity = spco2_sensitivity(ds)
    """

    def _check_variables(ds):
        requiredVars = ['spco2', 'tos', 'sos', 'talkos', 'dissicos']
        if not all(i in ds.data_vars for i in requiredVars):
            missingVars = [i for i in requiredVars if i not in ds.data_vars]
            raise ValueError(
                f"""Missing variables needed for calculation:
            {missingVars}"""
            )

    _check_variables(ds)
    # Sensitivities are based on the time-mean for each field. This computes
    # sensitivities at each grid cell.
    # TODO: Add keyword for sliding mean, as in N year chunks of time to
    # account for trends.
    DIC = ds['dissicos']
    ALK = ds['talkos']
    SALT = ds['sos']
    pCO2 = ds['spco2']

    buffer_factor = dict()
    buffer_factor['ALK'] = -ALK ** 2 / ((2 * DIC - ALK) * (ALK - DIC))
    buffer_factor['DIC'] = (3 * ALK * DIC - 2 * DIC ** 2) / (
        (2 * DIC - ALK) * (ALK - DIC)
    )

    # Compute sensitivities
    sensitivity = dict()
    sensitivity['tos'] = 0.0423
    sensitivity['sos'] = 1 / SALT
    sensitivity['talkos'] = (1 / ALK) * buffer_factor['ALK']
    sensitivity['dissicos'] = (1 / DIC) * buffer_factor['DIC']
    sensitivity = xr.Dataset(sensitivity) * pCO2
    return sensitivity


@check_xarray(0)
def co2_flx_ocean_sensitivities(ds):
    """Compute sensitivity of oceanic co2_flux to changes in driver variables.

    .. math: co2_flx_ocean = co2trans * (co2atm - co2ocean) * (1 - ice)

    Args:
        ds (xr.Dataset): containing cmorized variables:
                         * co2_flx_ocean [kg m-2 s-1]: surface air-sea CO2 flux
                         * co2ocean[ppm CO2]: partial pressure of CO2 in
                                              surface sea water
                         * CO2[ppm]: partial pressure of atmospheric CO2 at the
                                     surface
                         * co2trans [10-9 mol s m-4] : transfer velocity of
                                                       ocean/atmosphere CO2 flux
                         * seaice [%]: fraction of grid cell covered by ice

    Returns:
        sensitivity (xr.Dataset):

    References:
        * Lovenduski, Nicole S., Nicolas Gruber, Scott C. Doney, and Ivan D. Lima.
          “Enhanced CO2 Outgassing in the Southern Ocean from a Positive Phase of
          the Southern Annular Mode.” Global Biogeochemical Cycles 21, no. 2
          (2007). https://doi.org/10/fpv2wt.
        * Sarmiento, Jorge Louis, and Nicolas Gruber. Ocean Biogeochemical Dynamics.
          Princeton, NJ: Princeton Univ. Press, 2006., p.421, eq. (10:3:1)

    """

    def _check_variables(ds):
        requiredVars = ['co2_flx_ocean', 'CO2', 'co2ocean', 'co2trans', 'seaice']
        if not all(i in ds.data_vars for i in requiredVars):
            missingVars = [i for i in requiredVars if i not in ds.data_vars]
            raise ValueError(
                f"""Missing variables needed for calculation:
            {missingVars}"""
            )

    _check_variables(ds)

    # Sensitivities are based on the time-mean for each field. This computes
    # sensitivities at each grid cell.
    # TODO: Add keyword for sliding mean, as in N year chunks of time to
    # account for trends.
    CO2FLUX = ds['co2_flx_ocean']
    CO2OCEAN = ds['co2ocean']
    CO2ATM = ds['CO2']
    ICE = ds['seaice']
    CO2TRANS = ds['co2trans']

    # Compute sensitivities
    sensitivity = dict()
    sensitivity['ice'] = -1 / ICE
    sensitivity['co2trans'] = 1 / CO2TRANS
    sensitivity['co2atm'] = 1 / CO2ATM
    sensitivity['co2ocean'] = -1 / CO2OCEAN
    sensitivity = xr.Dataset(sensitivity) * CO2FLUX
    return sensitivity


# TODO: adapt for CESM and MPI output.
@check_xarray([0, 1])
def decomposition_index(
    ds_terms,
    index,
    detrend=True,
    order=1,
    deseasonalize=False,
    plot=False,
    sliding_window=10,
    decompose='spco2',
    **plot_kwargs,
):
    """Decompose a spco2 or co2_flx_ocean in a first order Taylor-expansion.

    Args:
        ds (xr.Dataset): containing cmorized variables as required for
                         sensitivity
        index (xarray object): Climate index to regress onto.
        detrend (bool): Whether to detrend time series prior to regression.
                        Defaults to a linear (order 1) regression.
        order (int): If detrend is True, what order polynomial to remove.
        deseasonalize (bool): Whether to deseasonalize time series prior to
                              regression.
        plot (bool): quick plot. Defaults to False.
        sliding_window (int): Number of years to apply sliding window to for
                              calculation. Defaults to 10.
        decompose (str): variable to be decomposed. Choose from:
                         ['co2_flx_ocean', 'spco2']
        **plot_kwargs (type): `**plot_kwargs`.

    Returns:
        terms_in_decompose_units (xr.Dataset): terms of spco2 decomposition,
                                          if `not plot`

    References:
        * Lovenduski, Nicole S., Nicolas Gruber, Scott C. Doney, and Ivan D. Lima.
          “Enhanced CO2 Outgassing in the Southern Ocean from a Positive Phase of
          the Southern Annular Mode.” Global Biogeochemical Cycles 21, no. 2
          (2007). https://doi.org/10/fpv2wt.
        * Brady, Riley X., et al. "On the role of climate modes in modulating the
          air–sea CO 2 fluxes in eastern boundary upwelling systems." Biogeosciences
          16.2 (2019): 329-346.
    """

    def regression_against_index(ds, index, psig=None):
        terms = dict()
        for term in ds.data_vars:
            if term != decompose:
                reg = linear_regression(index, ds[term], psig=psig)
                terms[term] = reg['slope']
        terms = xr.Dataset(terms)
        return terms

    if decompose == 'spco2':
        decompose_sensitivity = spco2_sensitivity
    elif decompose == 'co2_flx_ocean':
        decompose_sensitivity = co2_flx_ocean_sensitivities
    else:
        raise ValueError(
            'Please provide `decompose` from \
                         ["spco2", "co2_flx_ocean"]'
        )
    sensitivity = decompose_sensitivity(ds_terms)

    if detrend and not order:
        raise KeyError(
            """Please provide the order of polynomial to remove from
                       your time series if you are using detrend."""
        )
    elif detrend:
        ds_terms_anomaly = rm_poly(ds_terms, order=order, dim='time')
    else:
        warnings.warn('Your data are not being detrended.')
        ds_terms_anomaly = ds_terms - nanmean(ds_terms)

    if deseasonalize:
        clim = ds_terms_anomaly.groupby('time.month').mean('time')
        ds_terms_anomaly = ds_terms_anomaly.groupby('time.month') - clim
    else:
        warnings.warn('Your data are not being deseasonalized.')

    # Apply sliding window to regressions. I.e., compute in N year chunks
    # then average the resulting dpCO2/dX.
    if sliding_window is None:
        terms = regression_against_index(ds_terms_anomaly, index)
        terms_in_decompose_units = terms * nanmean(sensitivity)
    else:
        years = [y for y in index.groupby('time.year').groups]
        y_end = index['time.year'][-1]
        res = []
        for y1 in tqdm(years):
            y2 = y1 + sliding_window
            if y2 <= y_end:
                ds = ds_terms_anomaly.sel(time=slice(str(y1), str(y2)))
                ind = index.sel(time=slice(str(y1), str(y2)))
                terms = regression_against_index(ds, ind)
                sens = sensitivity.sel(time=slice(str(y1), str(y2)))
                res.append(terms * nanmean(sens))
        terms_in_decompose_units = xr.concat(res, dim='time').mean('time')

    if plot:
        terms_in_decompose_units.to_array().plot(
            col='variable', cmap='RdBu_r', robust=True, **plot_kwargs
        )
    else:
        return terms_in_decompose_units


@check_xarray(0)
def decomposition(
    ds_terms, detrend=True, order=1, deseasonalize=False, decompose='spco2'
):
    """Decompose spco2 or co2_flx_ocean in a first order Taylor-expansion.

    Args:
        ds_terms (xr.Dataset): containing cmorized variables as required by
                               sensitivities
        detrend (bool): If True, detrend when generating anomalies. Default to
                        a linear (order 1) regression.
        order (int): If detrend is true, the order polynomial to remove from
                     your time series.
        deseasonalize (bool): If True, deseasonalize when generating anomalies.
        decompose (str): variable to be decomposed. Choose from:
                         ['co2_flx_ocean', 'spco2']

    Return:
        terms_in_decompose_units (xr.Dataset): terms of spco2 decomposition

    References:
    * Lovenduski, Nicole S., Nicolas Gruber, Scott C. Doney, and Ivan D. Lima.
        “Enhanced CO2 Outgassing in the Southern Ocean from a Positive Phase of
        the Southern Annular Mode.” Global Biogeochemical Cycles 21, no. 2
        (2007). https://doi.org/10/fpv2wt.
    """
    if decompose == 'spco2':
        decompose_sensitivity = spco2_sensitivity
    elif decompose == 'co2_flx_ocean':
        decompose_sensitivity = co2_flx_ocean_sensitivities
    else:
        raise ValueError(
            'Please provide `decompose` from \
                         ["spco2", "co2_flx_ocean"]'
        )
    sensitivity = decompose_sensitivity(ds_terms)

    if detrend and not order:
        raise KeyError(
            """Please provide the order of polynomial you would like
                       to remove from your time series."""
        )
    elif detrend:
        ds_terms_anomaly = rm_poly(ds_terms, order=order, dim='time')
    else:
        warnings.warn('Your data are not being detrended.')
        ds_terms_anomaly = ds_terms - ds_terms.mean('time')

    if deseasonalize:
        clim = ds_terms_anomaly.groupby('time.month').mean('time')
        ds_terms_anomaly = ds_terms_anomaly.groupby('time.month') - clim
    else:
        warnings.warn('Your data are not being deseasonalized.')

    terms_in_decompose_units = sensitivity.mean('time') * ds_terms_anomaly
    return terms_in_decompose_units
