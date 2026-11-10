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

`generate_shooting_schedule` interprets historical naive dusk/dawn strings as UTC.
An explicit ISO UTC offset preserves the instant and is converted to UTC for scheduling,
including altitude-curve alignment and moonset delay. The original input strings remain
echoed in `dusk`/`dawn`; slot timestamps are UTC. A nonpositive interval yields an empty
plan with zero duration and a warning. These source fixes need a package release before
installed consumers receive them; no version publication is included in this change.

Target ranking computes positions for the requested observing night. Missing/non-finite
photometry earns no surface-brightness bonus; telescope results represent unknown magnitudes
as `null` and keep suitability scores finite. The filter matcher recognizes common SIMBAD
type codes such as `OpC`, `HII`, `G`, and `PN` as well as English object types. These source
corrections require a new core package release before installed consumers receive them.

| Consumer | Min version | Status |
|----------|-------------|--------|
| [mcp-stargazing](https://github.com/StarGazer1995/mcp-stargazing) | `>=0.1.0` | ✅ Active |
| [stargazing-place-finder](https://github.com/StarGazer1995/stargazing-place-finder) | `>=0.1.0` | ✅ Active |

## Cross-repository CI

Changes to core run both consumer suites against a locally built core wheel.
The shared baseline/candidate policy is documented in
[mcp-stargazing/docs/dependency-chain-ci.md](https://github.com/StarGazer1995/mcp-stargazing/blob/main/docs/dependency-chain-ci.md).

## Release

Version: **0.2.3** ([PyPI](https://pypi.org/project/stargazing-core/))

Release process:
1. Bump version in `pyproject.toml`
2. Merge to `main` → auto-tag workflow creates `vX.Y.Z` tag
3. Tag push triggers `release-pypi.yml` → OIDC Trusted Publishing to PyPI
4. GitHub Release auto-generated

## Development

```bash
uv sync
uv run pytest -v
uv run ruff format --check . && uv run ruff check .
```


## Imaging target ranking

Equipment presets live in `TELESCOPE_PRESETS`, including `hope_d60_asi2600mc_air`.
MCP keeps its existing public preset IDs and projects the shared HOPE configuration;
SPF exposes the same catalog at `/api/telescope/presets` and loads it in the browser.
Preset dimensions describe the existing configured setup, not a fresh hardware measurement.

Imaging candidates must have at least 30 consecutive sampled minutes with altitude
at least 25° and the Sun at or below −18°. Samples are 15 minutes apart; both ends
of each interval must pass. A bright Moon within 15° excludes that sample when
illumination exceeds 0.1 and the Moon is up. Midnight and dawn altitude no longer
exclude otherwise usable targets. The visual-only NGC magnitude cutoff and visual
aperture magnitude limit do not constrain sensor-equipped imaging.

The base score remains 40% FOV fit + 30% surface brightness + 20% filter match +
10% mean altitude over usable samples. Multiply by `0.5 + 0.5 × min(1, longest_run_minutes / 120)`,
then subtract the Moon penalty (when illumination >0.3: illumination × fraction of
usable samples with Moon above the horizon ×0.25), clamp to zero, and multiply by 100.
Scores sort descending, then dawn altitude ascending. These are heuristic choices,
not calibrated exposure, SNR, weather, or success estimates. Missing photometry gets
zero brightness contribution. High-latitude nights without astronomical darkness
produce no imaging recommendations.

FOV fit ramps to full credit at 10% area fill. Oversized axes divide that credit
by `max(1, target_long/sensor_long) × max(1, target_short/sensor_short)`, allowing
camera rotation. Tiny targets below 1% area fill are excluded; large targets remain
eligible as mosaic options. Civil dusk/dawn bounds and curve fields retain their
existing meanings; usable imaging time is stricter than the plotted civil night.

### Catalog data and attribution

`src/stargazing_core/data/objects.json` remains the generic catalog authority.
Imaging alone overlays pure open-cluster optical sizes from
`data/imaging_cluster_sizes.json`; this avoids treating large membership regions
as bright photographic footprints. The overlay excludes cluster+nebula entries,
so it does not shrink extended nebulae to their central cluster. A missing minor
axis stays null; area calculations use the documented circular approximation.
Other catalog classifications, dimensions and photometry still have source limitations.
AGN/Seyfert galaxy codes use the galaxy filter category, rather than an unknown-type fallback.

The overlay is an adaptation of **OpenNGC**, © Mattia Verga and contributors,
licensed **CC BY-SA 4.0** separately from this project's code. It records the exact
upstream commit URL and CSV SHA-256. Transformation: select `Type == OCl` with
positive `MajAx`, retain arcminute major/minor axes (blank minor → null), normalize
NGC/IC IDs, and add Messier aliases from `M`. No coordinates or photometry are imported.
The complete license is distributed in `data/OPENNGC-LICENSE.txt`.
