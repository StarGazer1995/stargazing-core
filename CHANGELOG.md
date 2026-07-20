# Changelog

## 0.2.0 (2026-07-20) — Surface brightness precision & rise/set semantics

### Added

- **Elliptical FOV scoring**: `_score_fov_fit()` now uses ellipse area (π × a × b) instead of circular, for more accurate FOV-fit scores on elongated galaxies
- **`optimal_rotation_deg`**: Computed from PA to align camera sensor long side with target major axis
- **Explicit rise/set/transit**: `rise_time`, `set_time`, `transit_time`, `transit_alt` on every target — no more inferring direction from altitude curves
- **`moonrise`/`moonset`**: Explicit horizon-crossing timestamps in moon response and `MoonInfo` model
- **Angular size fallbacks**: Added `err`, `?`, empty-type fallback entries — 100% catalog coverage (was 99.3%)

### Changed

- `_score_fov_fit()` signature: `(maj_arcmin, min_arcmin, fov_w, fov_h)` — second arg added
- `fov_fill_ratio` now uses ellipse area instead of circular
- `mosaic_recommended` now checks both major and minor axes against FOV
- `_shooting_plan.py` uses explicit `moon['moonset']` instead of scanning altitude curve
- Weather normalisers return `(normalised, valid_mask)` tuple — eliminates double `isnan`

### Fixed

- **#72**: Merged `valid` computation into weather normalisers (removed TODO)

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
