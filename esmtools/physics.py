import numpy as np
import xarray as xr


def stress_to_speed(x, y):
    """Convert ocean wind stress to wind speed at 10 m over the ocean.

    This expects that tau is in dyn/cm2.

    .. math::
        \\tau = 0.0027 * U + 0.000142 * U2 + 0.0000764 * U3

    .. note::
        This is useful for looking at wind speed on the native ocean grid, rather than
        trying to interpolate between atmospheric and oceanic grids.

    This is based on the conversion used in Lovenduski et al. (2007), which is related
    to the CESM coupler conversion:
    http://www.cesm.ucar.edu/models/ccsm3.0/cpl6/users_guide/node20.html

    Args:
        x (xr.DataArray): ``TAUX`` or ``TAUX2``.
        y (xr.DataArray): ``TAUY`` or ``TAUY2``.

    Returns:
        U10 (xr.DataArray): Approximated U10 wind speed.
    """
    tau = (
        (np.sqrt(x ** 2 + y ** 2)) / 1.2 * 100 ** 2 / 1e5
    )  # Convert from dyn/cm2 to m2/s2
    U10 = np.zeros(len(tau))
    for t in range(len(tau)):
        c_tau = tau[t]
        p = np.array([0.0000764, 0.000142, 0.0027, -1 * c_tau])
        r = np.roots(p)
        i = np.imag(r)
        good = np.where(i == 0)
        U10[t] = np.real(r[good])
    U10 = xr.DataArray(U10, dims=['time'], coords=[tau['time']])
    return U10
