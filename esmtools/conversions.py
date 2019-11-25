from .checks import is_xarray


@is_xarray(0)
def convert_mpas_fgco2(mpas_fgco2):
    """Convert native MPAS CO2 flux (mmol m-3 m s-1) to (molC m-2 yr-1)

    Args:
        mpas_fgco2 (xarray object): Dataset or DataArray containing native MPAS-O
                                    CO2 flux output.

    Returns:
        conv_fgco2 (xarray object): MPAS-O CO2 flux in mol/m2/yr.
    """
    # The -1 term ensures that negative is uptake of CO2 by the ocean.
    # MPAS defaults to gregorian noleap calendar (thus, 365).
    conv_fgco2 = mpas_fgco2 * -1 * (60 * 60 * 24 * 365) * (1 / 10 ** 3)
    return conv_fgco2
