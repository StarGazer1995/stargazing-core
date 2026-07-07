# Changelog

## 0.1.0 (2026-07-07) — First stable release

### Added

- **Telescope optics**: `TelescopeConfig`, `TelescopeOptics`, `TELESCOPE_PRESETS` (Seestar S50, RedCat 51, etc.)
- **Deep-sky catalog**: 10,000+ Messier/NGC objects with angular sizes from SIMBAD
- **Target matching**: `match_telescope_targets` — ranked DSO list by FOV fit, surface brightness, filter match
- **Shooting plan engine**: `generate_shooting_schedule` — minute-by-minute imaging sequence with meridian flip warnings
- **Mosaic planning**: `compute_mosaic_grid` — multi-panel grid for large targets
- **Shared models**: `CelestialPosition`, `RiseSet`, `MoonInfo`, `VisiblePlanet`, `GeoPoint`, `GeoBounds`, `PaginatedResult`
- **Astronomy utilities**: `calculate_moon_info`, `get_moon_altaz`, `get_visible_planets`, `identify_constellation`
- **CI/CD**: auto-tag on version bump, OIDC Trusted Publishing to PyPI, GitHub Release

### Changed

- Version `0.1.0a1` → `0.1.0` (first stable, published to PyPI)
- NumPy vectorisation for `filter_candidates_by_lst` + single-pass extraction (~97× speedup)
- NumPy vectorisation for `match_telescope_targets` pipeline

## 0.1.0a1 (pre-release)

Initial alpha with core telescope optics, catalog loading, and basic celestial models.
