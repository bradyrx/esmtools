from pkg_resources import DistributionNotFound, get_distribution

from .accessor import GridAccessor
from . import carbon, composite, conversions, spatial, grid, physics, stats, temporal
from .versioning.print_versions import show_versions

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    # package is not installed
    pass
