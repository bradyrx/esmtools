from pkg_resources import DistributionNotFound, get_distribution

from . import (
    carbon,
    composite,
    conversions,
    spatial,
    grid,
    loadutils,
    mpas,
    physics,
    stats,
    temporal,
    vis,
)
from .versioning.print_versions import show_versions

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    # package is not installed
    pass
