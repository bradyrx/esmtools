import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

from .checks import is_xarray
from .constants import CONCAT_KWARGS
from .stats import linregress, nanmean, rm_poly


def calculate_compatible_emissions(global_co2_flux, co2atm_forcing):
    """Calculate compatible emissions.

    Args:
        global_co2_flux (xr.object): global co2_flux in PgC/yr.
        co2atm_forcing (xr.object): prescribed atm. CO2 forcing in ppm.

    Returns:
        xr.object: compatible emissions in PgC/yr.

    Reference:
        * Jones, Chris, Eddy Robertson, Vivek Arora, Pierre Friedlingstein, Elena
          Shevliakova, Laurent Bopp, Victor Brovkin, et al. “Twenty-First-Century
          Compatible CO2 Emissions and Airborne Fraction Simulated by CMIP5 Earth
          System Models under Four Representative Concentration Pathways.”
          Journal of Climate 26, no. 13 (February 1, 2013): 4398–4413.
          https://doi.org/10/f44bbn.

    """
    compatible_emissions = co2atm_forcing.diff('time') * 2.12 - global_co2_flux
    compatible_emissions.name = 'compatible_emissions'
    return compatible_emissions


@is_xarray([0, 1])
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
        >>> from esmtools.carbon import co2_sol
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


def get_iam_emissions():
    """Download IAM emissions from PIK website.

    Returns:
        iam_emissions (xr.object): emissions from the IAMs in PgC/yr.
    """
    ds = []
    member = ['rcp26', 'rcp45', 'rcp85']
    for r in member:
        if r == 'rcp26':
            r = 'rcp3pd'
        r = r.upper()
        link = f'http://www.pik-potsdam.de/~mmalte/rcps/data/{r}_EMISSIONS.xls'
        e = pd.read_excel(link, sheet_name=f'{r}_EMISSIONS', skiprows=35, header=2)
        e = e.set_index(e.columns[0])
        e.index.name = 'Year'
        ds.append(e[['FossilCO2', 'OtherCO2']].to_xarray())
    ds = xr.concat(ds, 'member', **CONCAT_KWARGS)
    ds = ds.sel(Year=slice(1850, 2100)).rename({'Year': 'time'})
    ds['member'] = member
    ds['IAM_emissions'] = ds['FossilCO2'] + ds['OtherCO2']
    return ds['IAM_emissions']


def plot_compatible_emissions(
    compatible_emissions, global_co2_flux, iam_emissions=None, ax=None
):
    """Plot combatible emissions.

    Args:
        compatible_emissions (xr.object): compatible_emissions in PgC/yr from
            `calculate_compatible_emissions`.
        global_co2_flux (xr.object): Global CO2 flux in PgC/yr.
        iam_emissions (xr.object): (optional) Emissions from the IAMs in PgC/yr.
            Defaults to None.
        ax (plt.ax): (optional) matplotlib axis to plot onto. Defaults to None.

    Returns:
        ax (plt.ax): matplotlib axis.

    References:
        * Jones, Chris, Eddy Robertson, Vivek Arora, Pierre Friedlingstein, Elena
          Shevliakova, Laurent Bopp, Victor Brovkin, et al. “Twenty-First-Century
          Compatible CO2 Emissions and Airborne Fraction Simulated by CMIP5 Earth
          System Models under Four Representative Concentration Pathways.”
          Journal of Climate 26, no. 13 (February 1, 2013): 4398–4413.
          https://doi.org/10/f44bbn. Fig. 5a
        * IPCC AR5 Fig. 6.25

    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    # hist
    alpha = 0.1
    c = 'gray'
    compatible_emissions.isel(member=0).sel(
        time=slice(None, 2005)
    ).to_dataframe().unstack()['compatible_emissions'].plot(
        ax=ax, legend=False, color=c, alpha=alpha
    )
    compatible_emissions.isel(member=0).sel(time=slice(None, 2005)).mean(
        'initialization'
    ).plot(ax=ax, color='w', lw=3)
    compatible_emissions.isel(member=0).sel(time=slice(None, 2005)).mean(
        'initialization'
    ).plot(ax=ax, color=c, lw=2)
    # rcps
    colors = ['royalblue', 'orange', 'red'][::-1]
    for i, m in enumerate(global_co2_flux.member.values[::-1]):
        c = colors[i]
        compatible_emissions.sel(member=m).sel(
            time=slice(2005, None)
        ).to_dataframe().unstack()['compatible_emissions'].plot(
            ax=ax, legend=False, color=c, alpha=alpha
        )
        compatible_emissions.sel(member=m).sel(time=slice(2005, None)).mean(
            'initialization'
        ).plot(ax=ax, color='w', lw=3)
        compatible_emissions.sel(member=m).sel(time=slice(2005, None)).mean(
            'initialization'
        ).plot(ax=ax, color=c, lw=2)

    if iam_emissions is not None:
        ls = (0, (5, 5))
        iam_emissions.isel(member=0).sel(time=slice(None, 2005)).plot(
            ax=ax, color='white', lw=3
        )
        iam_emissions.isel(member=0).sel(time=slice(None, 2005)).plot(
            ax=ax, color='gray', lw=2, ls=ls
        )
        for i, m in enumerate(global_co2_flux.member.values[::-1]):
            c = colors[i]
            iam_emissions.sel(member=m).sel(time=slice(2005, None)).plot(
                ax=ax, color='white', lw=3
            )
            iam_emissions.sel(member=m).sel(time=slice(2005, None)).plot(
                ax=ax, color=c, lw=2, ls=ls
            )

    # fig aestetics
    ax.axhline(y=0, ls=':', c='gray')
    ax.set_ylabel('Compatible emissions [PgC/yr]')
    ax.set_xlabel('Time [year]')
    ax.set_title('Compatible emissions')
    return ax


@is_xarray([0, 1])
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
        * Sarmiento, Jorge Louis, and Nicolas Gruber. Ocean Biogeochemical Dynamics.
          Princeton, NJ: Princeton Univ. Press, 2006., p.421, eq. (10:3:1)

    Examples:
        >>> from esmtools.carbon import potential_pco2
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


@is_xarray(0)
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
        >>> from esmtools.carbon import schmidt
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


@is_xarray(0)
def spco2_decomposition(ds_terms, detrend=True, order=1, deseasonalize=False):
    """Decompose oceanic surface pco2 in a first order Taylor-expansion.

    Args:
        ds_terms (xr.Dataset): containing cmorized variables:
                               spco2 [ppm]: ocean pCO2 at surface
                               talkos[mmol m-3]: Alkalinity at ocean surface
                               dissicos[mmol m-3]: DIC at ocean surface
                               tos [C] : temperature at ocean surface
                               sos [psu] : salinity at ocean surface
        detrend (bool): If True, detrend when generating anomalies. Default to
                        a linear (order 1) regression.
        order (int): If detrend is true, the order polynomial to remove from
                     your time series.
        deseasonalize (bool): If True, deseasonalize when generating anomalies.

    Return:
        terms_in_pCO2_units (xr.Dataset): terms of spco2 decomposition

    References:
        * Lovenduski, Nicole S., Nicolas Gruber, Scott C. Doney, and Ivan D. Lima.
          “Enhanced CO2 Outgassing in the Southern Ocean from a Positive Phase of
          the Southern Annular Mode.” Global Biogeochemical Cycles 21, no. 2
          (2007). https://doi.org/10/fpv2wt.

    """
    pco2_sensitivity = spco2_sensitivity(ds_terms)

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

    terms_in_pCO2_units = pco2_sensitivity.mean('time') * ds_terms_anomaly
    return terms_in_pCO2_units


# TODO: adapt for CESM and MPI output.
@is_xarray([0, 1])
def spco2_decomposition_index(
    ds_terms,
    index,
    detrend=True,
    order=1,
    deseasonalize=False,
    plot=False,
    sliding_window=10,
    **plot_kwargs,
):
    """Decompose oceanic surface pco2 in a first order Taylor-expansion.

    Args:
        ds (xr.Dataset): containing cmorized variables:
                            spco2 [ppm]: ocean pCO2 at surface
                            talkos[mmol m-3]: Alkalinity at ocean surface
                            dissicos[mmol m-3]: DIC at ocean surface
                            tos [C] : temperature at ocean surface
                            sos [psu] : salinity at ocean surface
        index (xarray object): Climate index to regress onto.
        detrend (bool): Whether to detrend time series prior to regression.
                        Defaults to a linear (order 1) regression.
        order (int): If detrend is True, what order polynomial to remove.
        deseasonalize (bool): Whether to deseasonalize time series prior to
                              regression.
        plot (bool): quick plot. Defaults to False.
        sliding_window (int): Number of years to apply sliding window to for
                              calculation. Defaults to 10.
        **plot_kwargs (type): `**plot_kwargs`.

    Returns:
        terms_in_pCO2_units (xr.Dataset): terms of spco2 decomposition,
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
            if term != 'spco2':
                reg = linregress(index, ds[term], psig=psig)
                terms[term] = reg['slope']
        terms = xr.Dataset(terms)
        return terms

    pco2_sensitivity = spco2_sensitivity(ds_terms)
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
        terms_in_pCO2_units = terms * nanmean(pco2_sensitivity)
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
                sens = pco2_sensitivity.sel(time=slice(str(y1), str(y2)))
                res.append(terms * nanmean(sens))
        terms_in_pCO2_units = xr.concat(res, dim='time').mean('time')

    if plot:
        terms_in_pCO2_units.to_array().plot(
            col='variable', cmap='RdBu_r', robust=True, **plot_kwargs
        )
    else:
        return terms_in_pCO2_units


@is_xarray(0)
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
        >>> from esmtools.carbon import spco2_sensitivity
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
    buffer_factor['ALK'] = -(ALK ** 2) / ((2 * DIC - ALK) * (ALK - DIC))
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


@is_xarray(0)
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
        * Takahashi, Taro, Stewart C. Sutherland, Colm Sweeney, Alain Poisson, Nicolas
          Metzl, Bronte Tilbrook, Nicolas Bates, et al. “Global Sea–Air CO2 Flux
          Based on Climatological Surface Ocean PCO2, and Seasonal Biological and
          Temperature Effects.” Deep Sea Research Part II: Topical Studies in
          Oceanography, The Southern Ocean I: Climatic Changes in the Cycle of
          Carbon in the Southern Ocean, 49, no. 9 (January 1,2002): 1601–22.
          https://doi.org/10/dmk4f2.

    Examples:
        >>> from esmtools.carbon import temp_decomp_takahashi
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
