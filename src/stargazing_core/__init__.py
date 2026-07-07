"""stargazing-core — shared astronomical computation.

Provides :class:`TelescopeConfig`, :class:`TelescopeOptics`, equipment presets,
and deep-sky catalog loading used by both ``mcp-stargazing`` and
``stargazing-place-finder``.
"""

from stargazing_core._catalog import DeepSkyCatalog, load_objects
from stargazing_core._celestial_models import (
    CelestialPosition,
    MoonInfo,
    RiseSet,
    VisiblePlanet,
)
from stargazing_core._constellation import identify_constellation
from stargazing_core._coord import validate_coordinates
from stargazing_core._filtering import (
    filter_candidates_by_lst,
    match_telescope_targets,
    score_deep_sky_objects,
)
from stargazing_core._geo import GeoBounds, GeoPoint, TimeInfo
from stargazing_core._moon import calculate_moon_info, get_moon_altaz
from stargazing_core._pagination import PaginatedResult
from stargazing_core._planets import get_visible_planets
from stargazing_core._shooting_plan import (
    ShootingPlan,
    ShootingSlot,
    generate_shooting_schedule,
)
from stargazing_core._telescope import (
    TELESCOPE_PRESETS,
    TelescopeConfig,
    TelescopeOptics,
)
from stargazing_core._timegrid import find_rise_set_indices, generate_time_grid

__all__ = [
    'TelescopeConfig',
    'TelescopeOptics',
    'TELESCOPE_PRESETS',
    'DeepSkyCatalog',
    'load_objects',
    'GeoPoint',
    'GeoBounds',
    'TimeInfo',
    'CelestialPosition',
    'RiseSet',
    'MoonInfo',
    'VisiblePlanet',
    'PaginatedResult',
    'validate_coordinates',
    'calculate_moon_info',
    'get_moon_altaz',
    'get_visible_planets',
    'filter_candidates_by_lst',
    'match_telescope_targets',
    'score_deep_sky_objects',
    'identify_constellation',
    'generate_time_grid',
    'find_rise_set_indices',
    'ShootingPlan',
    'ShootingSlot',
    'generate_shooting_schedule',
]
__version__ = '0.1.0a1'
