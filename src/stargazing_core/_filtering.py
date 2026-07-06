"""Deep-sky object filtering and scoring.

Pure computation — uses astropy for coordinate transforms but no network I/O.
Provides the telescope target matching pipeline.
"""

from __future__ import annotations

import math
from typing import Any

import astropy.units as u
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time

from . import _ephemeris  # noqa: F401 — ensure ephemeris is configured

# ── Filter-type → object-type scoring table ──────────────────────────────

_FILTER_MATCH: dict[str | None, dict[str, float]] = {
    'Hα': {
        'emission nebula': 1.0,
        'planetary nebula': 0.8,
        'supernova remnant': 0.8,
        'star-forming region': 1.0,
        'galaxy': 0.1,
        'globular cluster': 0.0,
        'open cluster': 0.0,
    },
    'OIII': {
        'planetary nebula': 1.0,
        'supernova remnant': 0.8,
        'emission nebula': 0.5,
        'galaxy': 0.0,
        'globular cluster': 0.0,
        'open cluster': 0.0,
    },
    'SII': {
        'supernova remnant': 1.0,
        'emission nebula': 0.8,
        'planetary nebula': 0.5,
        'galaxy': 0.0,
        'globular cluster': 0.0,
        'open cluster': 0.0,
    },
    None: {  # LRGB / no filter
        'galaxy': 1.0,
        'globular cluster': 1.0,
        'open cluster': 1.0,
        'reflection nebula': 1.0,
        'emission nebula': 0.6,
        'planetary nebula': 0.6,
        'supernova remnant': 0.5,
    },
}

_FALLBACK_FILTER_SCORE = 0.3


def filter_candidates_by_lst(
    raw_objects: list[dict[str, Any]], lst_deg: float
) -> list[dict[str, Any]]:
    """Filter deep-sky objects to those near the meridian (±8h RA from LST)."""
    candidates: list[dict[str, Any]] = []

    for obj in raw_objects:
        mag = obj.get('magnitude', 99.9)
        catalog = obj.get('catalog', 'Unknown')

        if catalog == 'NGC' and mag > 10.0:
            continue

        obj_ra = obj['ra']
        diff = abs(obj_ra - lst_deg)
        if diff > 180:
            diff = 360 - diff

        if diff > 120:  # ~8 hours
            continue

        candidates.append(obj)

    return candidates


def score_deep_sky_objects(
    candidates: list[dict[str, Any]],
    time: Time,
    observer_location: EarthLocation,
    moon_coord: SkyCoord,
    moon_illum: float,
) -> list[dict[str, Any]]:
    """Score each candidate by altitude, moon interference, and catalog prestige."""
    scored: list[dict[str, Any]] = []

    altaz_frame = AltAz(obstime=time, location=observer_location)
    moon_altaz = moon_coord.transform_to(altaz_frame)
    moon_up = moon_illum > 0.1 and moon_altaz.alt.deg > 0

    for obj in candidates:
        try:
            ra_val = float(obj['ra'])
            dec_val = float(obj['dec'])
        except (ValueError, TypeError):
            continue

        coord = SkyCoord(ra=ra_val * u.deg, dec=dec_val * u.deg, frame='icrs')
        altaz = coord.transform_to(altaz_frame)
        alt = altaz.alt.deg

        if alt < 20:
            continue

        mag = obj.get('magnitude', 99.9)
        effective_mag = mag

        if moon_up:
            sep = coord.separation(moon_coord).deg
            if sep < 15:
                continue
            elif sep < 60:
                effective_mag += (60 - sep) * 0.1

        alt_bonus = (alt / 90.0) * 2.0
        score = effective_mag - alt_bonus

        if obj.get('catalog') == 'Messier':
            score -= 5.0

        scored.append(
            {
                'name': obj['name'],
                'type': obj['type'],
                'magnitude': mag,
                'altitude': round(alt, 1),
                'azimuth': round(altaz.az.deg, 1),
                'catalog': obj.get('catalog', 'Unknown'),
                'score': score,
            }
        )

    scored.sort(key=lambda x: x['score'])
    return scored


# ── Telescope matching pipeline ─────────────────────────────────────────


def _calc_surface_brightness(
    mag: float,
    maj_arcmin: float | None,
    min_arcmin: float | None,
) -> float | None:
    """Calculate surface brightness in mag/arcmin².

    SB = m + 2.5 × log₁₀(π × a × b)

    Uses ellipse area when both axes are available, otherwise circular
    approximation from maj_arcmin alone.  Returns None when no size data
    is available.
    """
    if maj_arcmin is None:
        return None
    a = maj_arcmin / 2.0  # semi-major axis
    b = (min_arcmin / 2.0) if min_arcmin is not None else a  # semi-minor
    area_sq_arcmin = math.pi * a * b
    if area_sq_arcmin <= 0:
        return None
    return mag + 2.5 * math.log10(area_sq_arcmin)


def _score_fov_fit(
    obj_size_arcmin: float | None,
    fov_width_deg: float,
    fov_height_deg: float,
) -> float:
    """Score how well an object fits in the FOV (0.0–1.0).

    - < 1% fill → near 0 (too small for detail)
    - 10%–60% fill → 1.0 (ideal)
    - > 60% fill → stays at 1.0 (Phase 4 mosaic handles large targets)
    """
    if obj_size_arcmin is None or obj_size_arcmin <= 0:
        return 0.0

    fov_area_sq_deg = fov_width_deg * fov_height_deg
    if fov_area_sq_deg <= 0:
        return 0.0

    obj_area_sq_deg = math.pi * (obj_size_arcmin / 2.0 / 60.0) ** 2
    fill_ratio = obj_area_sq_deg / fov_area_sq_deg

    if fill_ratio >= 0.10:
        return 1.0
    # Linear ramp: 0% → 0.0, 10% → 1.0
    return fill_ratio / 0.10


def _score_filter_match(obj_type: str, filter_type: str | None) -> float:
    """Score how well a filter matches an object type (0.0–1.0)."""
    table = _FILTER_MATCH.get(filter_type, _FILTER_MATCH[None])
    obj_lower = obj_type.lower()
    for key, score in table.items():
        if key in obj_lower:
            return score
    return _FALLBACK_FILTER_SCORE


def match_telescope_targets(
    config: Any,
    observer: EarthLocation,
    time: Time,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Recommend astrophotography targets for a telescope setup.

    Args:
        config: TelescopeConfig instance.
        observer: Observer location (astropy EarthLocation).
        time: Observation time (astropy Time).
        limit: Maximum number of results to return.

    Returns:
        Sorted list of target dicts, best first.
    """
    # Lazy imports to avoid circular deps at module level
    from ._catalog import load_objects
    from ._moon import calculate_moon_info, get_moon_altaz

    optics = config.compute_optics()

    # ── Stage 1: coarse filtering ──────────────────────────────────
    lst_deg = time.sidereal_time('mean', longitude=observer.lon).deg
    all_objects = load_objects()
    candidates = filter_candidates_by_lst(all_objects, lst_deg)

    # ── Stage 2: moon + alt/az base scoring ────────────────────────
    moon_info = calculate_moon_info(time)
    moon_alt, moon_az = get_moon_altaz(observer, time.to_datetime())
    moon_coord = SkyCoord(
        moon_az * u.deg,
        moon_alt * u.deg,
        frame='altaz',
        obstime=time,
        location=observer,
    ).icrs

    base_scored = score_deep_sky_objects(
        candidates,
        time,
        observer,
        moon_coord,
        moon_info['illumination'],
    )

    # ── Stage 3: device-aware scoring ──────────────────────────────
    fov_w = optics.fov_width_deg
    fov_h = optics.fov_height_deg
    lim_mag = optics.limiting_magnitude

    results: list[dict[str, Any]] = []
    for obj in base_scored:
        mag = obj.get('magnitude', 99.9)

        # --- hard filter: magnitude ---
        if lim_mag is not None and mag > lim_mag:
            continue

        # --- find original catalog entry for angular size ---
        orig = next(
            (o for o in all_objects if o['name'] == obj['name']),
            None,
        )
        maj = orig.get('angular_size_maj_arcmin') if orig else None
        min_ = orig.get('angular_size_min_arcmin') if orig else None
        obj_type = obj['type']

        # --- scores ---
        fov_fit = _score_fov_fit(maj, fov_w or 0, fov_h or 0)
        sb = _calc_surface_brightness(mag, maj, min_)
        filter_score = _score_filter_match(obj_type, config.filter_type)

        # Normalize surface brightness: typical range 10–25 mag/arcmin²
        sb_score = 0.0
        if sb is not None:
            sb_score = max(0.0, min(1.0, (25.0 - sb) / 15.0))

        alt_score = obj['altitude'] / 90.0

        total = fov_fit * 0.40 + sb_score * 0.30 + filter_score * 0.20 + alt_score * 0.10

        mosaic_recommended = maj is not None and fov_w is not None and (maj / 60.0) > fov_w * 1.5

        results.append(
            {
                'name': obj['name'],
                'ra': orig.get('ra') if orig else None,
                'dec': orig.get('dec') if orig else None,
                'type': obj_type,
                'magnitude': mag,
                'surface_brightness': round(sb, 2) if sb is not None else None,
                'angular_size_arcmin': maj,
                'altitude': obj['altitude'],
                'azimuth': obj['azimuth'],
                'fov_fill_ratio': round(
                    (math.pi * (maj / 2.0 / 60.0) ** 2) / (fov_w * fov_h),
                    4,
                )
                if (maj and fov_w and fov_h)
                else None,
                'fov_fit_score': round(fov_fit, 2),
                'surface_brightness_score': round(sb_score, 2),
                'filter_match_score': round(filter_score, 2),
                'altitude_score': round(alt_score, 2),
                'suitability_score': round(total * 100, 1),
                'mosaic_recommended': mosaic_recommended,
                'catalog': obj.get('catalog', 'Unknown'),
            }
        )

    results.sort(key=lambda x: x['suitability_score'], reverse=True)
    return results[:limit]
