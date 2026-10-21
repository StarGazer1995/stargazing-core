# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Package Management

**Always use `uv`** — never `pip`.

- Use `uv sync` / `uv add` / `uv remove`
- Use `uv run <command>` to execute anything within the project venv
- `stargazing-core` is published on PyPI (v0.1.0) — consumers use `stargazing-core>=0.1.0`

## Quick Commands

| Action | Command |
|--------|---------|
| Install deps | `uv sync` |
| Run all tests | `uv run pytest -v` |
| Lint + format | `uv run ruff format --check src/ tests/ && uv run ruff check src/ tests/` |
| Build package | `uv run python -m build` |
| Publish (local check) | `uv run twine check dist/*` |

## Architecture

```
stargazing_core/          ← installable package (flat namespace)
├── __init__.py           ← public re-exports (30 symbols)
├── _catalog.py           ← DeepSkyCatalog + load_objects (10k objects)
├── _celestial_models.py  ← Pydantic v2 models (CelestialPosition, MoonInfo, RiseSet, VisiblePlanet)
├── _constellation.py     ← Constellation center lookup
├── _coord.py             ← validate_coordinates()
├── _ephemeris.py         ← Ephemeris / observing window calculations
├── _filtering.py         ← DSO filtering + match_telescope_targets + scoring
├── _geo.py               ← GeoBounds, GeoPoint, TimeInfo
├── _moon.py              ← Moon phase / altitude / illumination
├── _mosaic.py            ← compute_mosaic_grid (MosaicGrid, MosaicPanel)
├── _pagination.py        ← PaginatedResult[T]
├── _planets.py           ← get_visible_planets()
├── _shooting_plan.py     ← generate_shooting_schedule (ShootingPlan, ShootingSlot)
├── _telescope.py         ← TelescopeConfig, TelescopeOptics, TELESCOPE_PRESETS
├── _timegrid.py          ← generate_time_grid, find_rise_set_indices
└── data/                 ← objects.json + constellation_centers.json
```

**No circular dependencies** — each `_*.py` module is self-contained. Public API is via `__init__.py` re-exports only.

## Key Design Patterns

1. **Private modules, public `__init__`** — All implementation modules are `_prefixed`. Consumers import from `stargazing_core`, not from internal modules. `__all__` is maintained for explicit export control.

2. **Pydantic v2 models are the contract** — `CelestialPosition`, `RiseSet`, `MoonInfo`, `TelescopeConfig`, `ShootingPlan` etc. are Pydantic `BaseModel` instances used by both MCP and SPF as the shared data contract.

3. **NumPy-vectorised where possible** — Filtering, scoring, and time-grid operations use NumPy arrays. No per-object loops in hot paths.

4. **Coordinates are always validated** — `validate_coordinates(lat, lon)` raises descriptive errors before any computation.

5. **`__version__` in `__init__.py`** — Single source of truth for the package version, also set in `pyproject.toml`. Keep them in sync on release.

## Testing

- Tests in `tests/`, run with `uv run pytest -v`
- CI: lint + test on Python 3.11/3.12/3.13
- CI enforces **100% diff-cover** on PRs
- No FastMCP or server dependencies — tests are pure unit/integration

## Release Process

1. Bump version in `pyproject.toml` AND `src/stargazing_core/__init__.py`
2. PR → review → merge to `main`
3. Auto-tag workflow creates `vX.Y.Z` tag
4. Tag push triggers `release-pypi.yml` → OIDC Trusted Publishing to PyPI
5. GitHub Release auto-generated

**Never** force-push or re-push the same tag. PyPI does not allow overwriting versions.

## Consumers

| Project | What they use |
|---------|---------------|
| `mcp-stargazing` | TelescopeConfig, match_telescope_targets, ShootingPlan, CelestialPosition, RiseSet, MoonInfo, GeoPoint, PaginatedResult, validate_coordinates |
| `stargazing-place-finder` | TelescopeConfig, GeoPoint, GeoBounds, compute_mosaic_grid |

## Commit Policy

- Never commit internal planning documents or code review findings.
- Use `git status` to verify only expected files are staged.
