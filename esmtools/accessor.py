import xarray as xr

from .grid import convert_lon


@xr.register_dataarray_accessor('grid')
@xr.register_dataset_accessor('grid')
class GridAccessor:
    """Allows functions to be called directly on an xarray dataset for the grid module.

    This implementation was heavily inspired by @ahuang11's implementation to
    xskillscore."""

    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def convert_lon(self, coord='lon'):
        """Converts longitude grid from -180to180 to 0to360 and vice versa.

        .. note::
            Longitudes are not sorted after conversion (i.e., spanning -180 to 180 or
            0 to 360 from index 0, ..., N) if it is 2D.

        Args:
            ds (xarray object): Dataset to be converted.
            coord (optional str): Name of longitude coordinate, defaults to 'lon'.

        Returns:
            xarray object: Dataset with converted longitude grid.

        Raises:
            ValueError: If ``coord`` does not exist in the dataset.
        """
        return convert_lon(self._obj, coord=coord)
