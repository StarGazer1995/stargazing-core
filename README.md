# stargazing-core

[![PyPI version](https://img.shields.io/pypi/v/stargazing-core)](https://pypi.org/project/stargazing-core/)
[![Python](https://img.shields.io/pypi/pyversions/stargazing-core)](https://pypi.org/project/stargazing-core/)

Shared astronomical computation package for the stargazing toolchain —
telescope optics, deep-sky catalog, and equipment presets used by both
[mcp-stargazing](https://github.com/StarGazer1995/mcp-stargazing) and
[stargazing-place-finder](https://github.com/StarGazer1995/stargazing-place-finder).

## Installation

```bash
pip install stargazing-core
```

Requires Python ≥ 3.11.

## Quick Start

```python
from stargazing_core import TelescopeConfig, TelescopeOptics, TELESCOPE_PRESETS

# Look up a preset
config = TELESCOPE_PRESETS['seestar-s50']

# Compute derived optics
optics = config.compute_optics()
print(f'FOV: {optics.fov_width_deg:.1f}° × {optics.fov_height_deg:.1f}°')
# → FOV: 1.7° × 1.3°
```

## Modules

| Module | Purpose |
|--------|---------|
| `_catalog.py` | 10,000+ Messier/NGC deep-sky object catalog |
| `_celestial_models.py` | Pydantic models: CelestialPosition, RiseSet, MoonInfo, VisiblePlanet |
| `_constellation.py` | Constellation center lookup |
| `_coord.py` | Coordinate validation (`validate_coordinates`) |
| `_ephemeris.py` | Ephemeris calculations for observing windows |
| `_filtering.py` | Deep-sky object filtering pipeline |
| `_geo.py` | GeoBounds / GeoPoint shared geometry types |
| `_moon.py` | Moon phase / altitude / illumination |
| `_mosaic.py` | Mosaic grid computation for wide-field targets |
| `_pagination.py` | Generic PaginatedResult[T] |
| `_planets.py` | Solar-system planet visibility |
| `_shooting_plan.py` | ShootingPlan / ShootingSlot — imaging schedule engine |
| `_telescope.py` | TelescopeConfig, TELESCOPE_PRESETS, match_telescope_targets |
| `_timegrid.py` | Altitude curve time-grid generation |

## Consumers

| Project | Usage |
|---------|-------|
| [mcp-stargazing](https://github.com/StarGazer1995/mcp-stargazing) | `get_telescope_targets`, `get_shooting_plan` tools; shared CelestialPosition/RiseSet/MoonInfo models |
| [stargazing-place-finder](https://github.com/StarGazer1995/stargazing-place-finder) | TelescopeConfig, GeoPoint, telescope target matching endpoint |

## Public API

All symbols are importable from the top-level package:

```python
from stargazing_core import (
    # Telescope
    TelescopeConfig, TelescopeOptics, TELESCOPE_PRESETS,
    # Deep-sky matching
    DeepSkyCatalog, load_objects,
    match_telescope_targets, filter_candidates_by_lst, score_deep_sky_objects,
    # Shooting plan
    ShootingPlan, ShootingSlot, generate_shooting_schedule,
    # Mosaic
    MosaicPanel, MosaicGrid, compute_mosaic_grid,
    # Celestial models
    CelestialPosition, RiseSet, MoonInfo, VisiblePlanet,
    # Astronomy
    calculate_moon_info, get_moon_altaz, get_visible_planets, identify_constellation,
    # Geometry
    GeoPoint, GeoBounds, TimeInfo, validate_coordinates,
    # Time grid
    generate_time_grid, find_rise_set_indices,
    # Pagination
    PaginatedResult,
)
```

## Compatibility

| Consumer | Min version | Status |
|----------|-------------|--------|
| [mcp-stargazing](https://github.com/StarGazer1995/mcp-stargazing) | `>=0.1.0` | ✅ Active |
| [stargazing-place-finder](https://github.com/StarGazer1995/stargazing-place-finder) | `>=0.1.0` | ✅ Active |

## Release

Version: **0.1.0** ([PyPI](https://pypi.org/project/stargazing-core/))

Release process:
1. Bump version in `pyproject.toml`
2. Merge to `master` → auto-tag workflow creates `vX.Y.Z` tag
3. Tag push triggers `release-pypi.yml` → OIDC Trusted Publishing to PyPI
4. GitHub Release auto-generated

## Development

```bash
uv sync
uv run pytest -v
uv run ruff format --check . && uv run ruff check .
```
