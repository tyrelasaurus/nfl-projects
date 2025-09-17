"""NCAA spread model package derived from the NFL implementation."""

from .config_manager import get_ncaa_config  # noqa: F401
from .spread_model import SpreadCalculator, MatchupResult  # noqa: F401
