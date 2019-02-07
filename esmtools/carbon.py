"""
Objects dealing with the carbon cycle and carbonate chemistry.

Seawater Chemistry
------------------
`co2_sol` : Compute the solubility of CO2 in seawater based on temperature and salinity.
`schmidt` : Computes the Schmidt number for CO2.
"""
import numpy as np


def co2_sol(t, s):
    """
    Compute CO2 sollubility per the equation used in CESM. The mean will be taken over
    the time series provided to produce the average solubility over this time period.
    Thus, if you want more accurate solubility you can feed in smaller time periods.

    Input
    -----
    t : SST time series (degC)
    s : SSS time series (PSU)

    Return
    ------
    ff : Value of solubility in mol/kg/atm

    References
    ----------
    Weiss & Price (1980, Mar. Chem., 8, 347-359;
    Eq 13 with table 6 values)
    """
    a = [-162.8301, 218.2968, 90.9241, -1.47696]
    b = [0.025695, -0.025225, 0.0049867]
    t = (np.mean(t) + 273.15) * 0.01
    s = np.mean(s)
    t_sq = t**2
    t_inv = 1.0 / t
    log_t = np.log(t)
    d0 = b[2] * t_sq + b[1] * t + b[0]
    # Compute solubility in mol.kg^{-1}.atm^{-1}
    ff = np.exp( a[0] + a[1] * t_inv + a[2] * log_t + \
        a[3] * t_sq + d0 * s )
    return ff


def schmidt(t):
    """
    Computes the dimensionless Schmidt number. The mean will be taken over the
    time series provided to produce the average Schmidt number over this time period.
    The polynomials used are for SST ranges between 0 and 30C and a salinity of 35.

    Input
    -----
    t : SST time series (degC)

    Return
    ------
    Sc : Schmidt number (dimensionless)

    Reference
    --------
    Sarmiento and Gruber (2006). Ocean Biogeochemical Dynamics.
    Table 3.3.1
    """
    c = [2073.1, 125.62, 3.6276, 0.043219]
    t = np.mean(t)
    Sc = c[0] - c[1] * t + c[2] * (t ** 2) - c[3] * (t ** 3)
    return Sc


def temp_decomp_takahashi(ds, time_dim='time', temperature='tos', pco2='spco2'):
    """
    Decompose spco2 into thermal and non-thermal component.

    Reference
    ---------
    Takahashi, Taro, Stewart C. Sutherland, Colm Sweeney, Alain Poisson, Nicolas
        Metzl, Bronte Tilbrook, Nicolas Bates, et al. “Global Sea–Air CO2 Flux
        Based on Climatological Surface Ocean PCO2, and Seasonal Biological and
        Temperature Effects.” Deep Sea Research Part II: Topical Studies in
        Oceanography, The Southern Ocean I: Climatic Changes in the Cycle of
        Carbon in the Southern Ocean, 49, no. 9 (January 1,2002): 1601–22.
        https://doi.org/10/dmk4f2.

    Input
    -----
    ds : xr.Dataset containing spco2[ppm] and tos[C or K]

    Output
    ------
    thermal, non_thermal : xr.DataArray
        thermal and non-thermal components in ppm units

    """
    fac = 0.0432
    tos_mean = ds[temperature].mean(time_dim)
    tos_diff = ds[temperature] - tos_mean
    thermal = ds[pco2].mean(time_dim) * (np.exp(tos_diff * fac))
    non_thermal = ds[pco2] * (np.exp(tos_diff * -fac))
    return thermal, non_thermal


def potential_pco2(t_insitu, pco2_insitu):
    """
    Calculate potential pco2 in the inner ocean. Requires the first index of
    depth to be at the surface.

    Input
    -----
    t_insitu : xr object
        SST with depth [C or K]
    pco2_insitu : xr object
        pCO2 with depth [ppm]

    Output
    ------
    pco2_potential : xr object
        potential pco2 with depth

    Reference:
    - Sarmiento, Jorge Louis, and Nicolas Gruber. Ocean Biogeochemical Dynamics.
        Princeton, NJ: Princeton Univ. Press, 2006., p.421, eq. (10:3:1)

    """
    t_sfc = t_insitu.isel(depth=0)
    pco2_potential = pco2_insitu * (1 + 0.0423 * (t_sfc - t_insitu))
    return pco2_potential
