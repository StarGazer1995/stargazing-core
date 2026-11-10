"""Deep-sky object filtering and scoring.

Pure computation — uses astropy for coordinate transforms but no network I/O.
Provides the telescope target matching pipeline.
"""

from __future__ import annotations

import math
from typing import Any

import astropy.units as u
import numpy as np
from astropy.coordinates import AltAz, EarthLocation, SkyCoord, get_body, get_sun
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
    n = len(raw_objects)
    if n == 0:
        return []

    # Single-pass extract into numpy arrays
    ra = np.empty(n, dtype=np.float64)
    mag = np.empty(n, dtype=np.float64)
    is_ngc = np.empty(n, dtype=bool)
    for i, obj in enumerate(raw_objects):
        ra[i] = obj['ra']
        mag[i] = obj.get('magnitude', 99.9)
        is_ngc[i] = obj.get('catalog', '') == 'NGC'

    diff = np.abs(ra - lst_deg)
    diff = np.where(diff > 180, 360 - diff, diff)

    keep = (diff <= 120) & ~(is_ngc & (mag > 10.0))
    return [raw_objects[i] for i in np.flatnonzero(keep)]


def score_deep_sky_objects(
    candidates: list[dict[str, Any]],
    time: Time,
    observer_location: EarthLocation,
    moon_coord: SkyCoord,
    moon_illum: float,
) -> list[dict[str, Any]]:
    """Score each candidate by altitude, moon interference, and catalog prestige.

    Uses vectorised SkyCoord + AltAz transforms: a single SkyCoord from
    arrays replaces ~3 600 individual SkyCoord constructions, and a single
    ``transform_to`` replaces the per-object loop.
    """
    n = len(candidates)
    if n == 0:
        return []

    # ── Single-pass extraction into typed numpy arrays ──────────────
    ra = np.empty(n, dtype=np.float64)
    dec = np.empty(n, dtype=np.float64)
    mag = np.empty(n, dtype=np.float64)
    is_messier = np.zeros(n, dtype=bool)
    names: list[str] = [''] * n
    types: list[str] = [''] * n
    catalogs: list[str] = [''] * n
    valid = np.ones(n, dtype=bool)

    for i, obj in enumerate(candidates):
        try:
            ra[i] = float(obj['ra'])
            dec[i] = float(obj['dec'])
        except (ValueError, TypeError):
            valid[i] = False
            continue
        mag[i] = obj.get('magnitude', 99.9)
        is_messier[i] = obj.get('catalog') == 'Messier'
        names[i] = obj['name']
        types[i] = obj.get('type', '')
        catalogs[i] = obj.get('catalog', 'Unknown')

    if not valid.any():
        return []

    # ── Vectorised coordinate transform (single call for all objects) ─
    coords = SkyCoord(ra=ra[valid] * u.deg, dec=dec[valid] * u.deg, frame='icrs')
    altaz_frame = AltAz(obstime=time, location=observer_location)
    altaz = coords.transform_to(altaz_frame)
    alt = altaz.alt.deg  # shape (n_valid,)

    # ── Altitude filter ──────────────────────────────────────────────
    above = alt >= 20.0

    # ── Moon interference (vectorised) ───────────────────────────────
    moon_altaz = moon_coord.transform_to(altaz_frame)
    moon_up = moon_illum > 0.1 and moon_altaz.alt.deg > 0

    # Missing photometry must not poison sorting with NaN scores.
    effective_mag = np.where(np.isfinite(mag[valid]), mag[valid], 99.9)
    moon_skip: np.ndarray = np.zeros(valid.sum(), dtype=bool)

    if moon_up:
        sep = coords.separation(moon_coord).deg  # shape (n_valid,)
        moon_skip = sep < 15.0
        near_moon = np.logical_not(moon_skip) & (sep < 60.0)
        effective_mag = np.where(near_moon, effective_mag + (60.0 - sep) * 0.1, effective_mag)

    keep = above & np.logical_not(moon_skip)

    if not keep.any():
        return []

    # ── Scoring (vectorised) ─────────────────────────────────────────
    alt_bonus = (alt / 90.0) * 2.0
    score = effective_mag - alt_bonus
    score = np.where(is_messier[valid], score - 5.0, score)

    # ── Build result dicts only for survivors ────────────────────────
    # Map from valid-local indices back to valid indices for name/type lookup
    valid_idx = np.flatnonzero(valid)  # original → valid-local
    keep_local = np.flatnonzero(keep)  # indices into valid-local arrays

    scored: list[dict[str, Any]] = []
    for kl in keep_local:
        orig_i = valid_idx[kl]
        scored.append(
            {
                'name': names[orig_i],
                'type': types[orig_i],
                'magnitude': float(mag[orig_i]),
                'altitude': round(float(alt[kl]), 1),
                'azimuth': round(float(altaz.az.deg[kl]), 1),
                'catalog': catalogs[orig_i],
                'score': float(score[kl]),
                '_ra': float(ra[orig_i]),
                '_dec': float(dec[orig_i]),
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
    if maj_arcmin is None or not math.isfinite(mag) or not math.isfinite(maj_arcmin):
        return None
    if min_arcmin is not None and not math.isfinite(min_arcmin):
        return None
    a = maj_arcmin / 2.0  # semi-major axis
    b = (min_arcmin / 2.0) if min_arcmin is not None else a  # semi-minor
    area_sq_arcmin = math.pi * a * b
    if area_sq_arcmin <= 0:
        return None
    return mag + 2.5 * math.log10(area_sq_arcmin)


def _score_fov_fit(
    maj_arcmin: float | None,
    min_arcmin: float | None,
    fov_width_deg: float,
    fov_height_deg: float,
) -> float:
    """Score how well an object fits in the FOV (0.0–1.0).

    Uses ellipse area (π × a × b) when both axes are available, falling
    back to circular approximation from maj_arcmin alone.

    - < 1% fill → near 0 (too small for detail)
    - 10%–60% fill → 1.0 (ideal)
    - Oversized axes reduce the score by the required rectangular coverage.
    - Sensor rotation is allowed when comparing major/minor extents.
    """
    if maj_arcmin is None or maj_arcmin <= 0:
        return 0.0

    fov_area_sq_deg = fov_width_deg * fov_height_deg
    if fov_area_sq_deg <= 0:
        return 0.0

    # Semi-axes in degrees
    a_deg = maj_arcmin / 2.0 / 60.0
    b_deg = (min_arcmin / 2.0 / 60.0) if min_arcmin else a_deg
    obj_area_sq_deg = math.pi * a_deg * b_deg
    fill_ratio = obj_area_sq_deg / fov_area_sq_deg

    sensor_long, sensor_short = sorted((fov_width_deg, fov_height_deg), reverse=True)
    target_long, target_short = sorted((2 * a_deg, 2 * b_deg), reverse=True)
    coverage = max(1.0, target_long / sensor_long) * max(1.0, target_short / sensor_short)
    return min(1.0, fill_ratio / 0.10) / coverage


def _score_filter_match(obj_type: str, filter_type: str | None) -> float:
    """Score how well a filter matches an object type (0.0–1.0)."""
    table = _FILTER_MATCH.get(filter_type, _FILTER_MATCH[None])
    # The packaged SIMBAD catalog uses codes rather than these English labels.
    aliases = {
        'OpC': 'open cluster',
        'GlC': 'globular cluster',
        'G': 'galaxy',
        'Gx': 'galaxy',
        'AGN': 'galaxy',
        'SyG': 'galaxy',
        'Sy1': 'galaxy',
        'Sy2': 'galaxy',
        'GiG': 'galaxy',
        'GiP': 'galaxy',
        'GiC': 'galaxy',
        'LIN': 'galaxy',
        'BLL': 'galaxy',
        'SBG': 'galaxy',
        'H2G': 'galaxy',
        'PaG': 'galaxy',
        'IG': 'galaxy',
        'rG': 'galaxy',
        'EmG': 'galaxy',
        'HII': 'emission nebula',
        'RNe': 'reflection nebula',
        'PN': 'planetary nebula',
        'SNR': 'supernova remnant',
    }
    obj_lower = aliases.get(obj_type, obj_type).lower()
    for key, score in table.items():
        if key in obj_lower:
            return score
    return _FALLBACK_FILTER_SCORE


def _imaging_catalog(objects: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep valid coordinates without visual-only NGC magnitude or midnight gates."""
    from ._catalog import _imaging_cluster_sizes

    cluster_sizes = _imaging_cluster_sizes()
    result = []
    for obj in objects:
        try:
            ra, dec = float(obj['ra']), float(obj['dec'])
        except (KeyError, TypeError, ValueError):
            continue
        if not (math.isfinite(ra) and math.isfinite(dec) and -90 <= dec <= 90):
            continue
        obj = {**obj, '_ra': ra, '_dec': dec}
        size = cluster_sizes.get(obj.get('name', ''))
        if size is not None:
            obj.update(
                angular_size_maj_arcmin=size['major_arcmin'],
                angular_size_min_arcmin=size['minor_arcmin'],
                angular_size_pa_deg=None,
            )
        result.append(obj)
    return result


def _longest_window_minutes(mask: np.ndarray, step_minutes: float = 15.0) -> float:
    """Conservative duration: both endpoints of each sampled interval must pass."""
    longest = run = 0
    for usable in mask:
        run = run + 1 if usable else 0
        longest = max(longest, run)
    return max(0, longest - 1) * step_minutes


def match_telescope_targets(
    config: Any,
    observer: EarthLocation,
    time: Time,
    limit: int = 20,
) -> dict[str, Any]:
    """Recommend astrophotography targets for a telescope setup.

    Args:
        config: TelescopeConfig instance.
        observer: Observer location (astropy EarthLocation).
        time: Observation time (astropy Time).
        limit: Maximum number of results to return.

    Returns:
        Dict with keys:
        - targets: sorted list of target dicts, best first
        - moon: dict with illumination, phase, altitude_curve,
          always_down, always_up, dark_fraction
    """
    # Lazy imports to avoid circular deps at module level
    from ._catalog import load_objects
    from ._moon import calculate_moon_info

    optics = config.compute_optics()

    # ── Find civil dusk (sun < -6°) via two-phase search ────────────
    # Phase 1: coarse scan at 1h intervals (49 steps vs original 192)
    # to bracket the -6° crossings, then Phase 2 refines each bracket.
    # This reduces get_sun()+AltAz calls from ~192 to ~49 + 2×20 ≈ 89.
    _coarse_step = 1.0  # hours
    _fine_step = 0.05  # hours (~3 min) for refinement
    _twilight_alt = -6.0  # civil dusk/dawn threshold

    coarse_offsets = np.arange(-12, 36 + _coarse_step / 2, _coarse_step)
    coarse_times = Time([time + h * u.hour for h in coarse_offsets])
    coarse_frames = AltAz(obstime=coarse_times, location=observer)
    coarse_alts = get_sun(coarse_times).transform_to(coarse_frames).alt.deg

    dusk_h = None
    dawn_h = None
    for i in range(1, len(coarse_alts)):
        if (
            dusk_h is None
            and coarse_alts[i - 1] > _twilight_alt
            and coarse_alts[i] <= _twilight_alt
        ):
            dusk_h = (coarse_offsets[i - 1], coarse_offsets[i])
        elif (
            dusk_h is not None
            and coarse_alts[i - 1] < _twilight_alt
            and coarse_alts[i] >= _twilight_alt
        ):
            dawn_h = (coarse_offsets[i - 1], coarse_offsets[i])
            break

    def _refine_crossing(h0: float, h1: float, crossing_down: bool) -> float | None:
        """Refine a -6° crossing within [h0, h1] using fine steps.

        crossing_down=True → sun goes from above to below (dusk).
        crossing_down=False → sun goes from below to above (dawn).
        """
        n_fine = max(2, int((h1 - h0) / _fine_step))
        fine_offsets = np.linspace(h0, h1, n_fine + 1)
        fine_times = Time([time + fh * u.hour for fh in fine_offsets])
        fine_frames = AltAz(obstime=fine_times, location=observer)
        fine_alts = get_sun(fine_times).transform_to(fine_frames).alt.deg
        for j in range(1, len(fine_alts)):
            if crossing_down and fine_alts[j - 1] > _twilight_alt and fine_alts[j] <= _twilight_alt:
                return float(fine_offsets[j])
            if (
                not crossing_down
                and fine_alts[j - 1] < _twilight_alt
                and fine_alts[j] >= _twilight_alt
            ):
                return float(fine_offsets[j])
        return None  # pragma: no cover — defensive, bracket always contains crossing

    if dusk_h is not None:
        refined_dusk = _refine_crossing(dusk_h[0], dusk_h[1], crossing_down=True)
        civil_dusk = time + (refined_dusk if refined_dusk is not None else dusk_h[0]) * u.hour
    else:  # pragma: no cover — polar day, no civil dusk in 48 h window
        civil_dusk = time

    if dawn_h is not None:
        refined_dawn = _refine_crossing(dawn_h[0], dawn_h[1], crossing_down=False)
        civil_dawn = time + (refined_dawn if refined_dawn is not None else dawn_h[1]) * u.hour
    else:  # pragma: no cover — polar day, no civil dawn in 48 h window
        civil_dawn = time + 12 * u.hour

    civil_midnight = civil_dusk + (civil_dawn - civil_dusk) / 2.0

    # Imaging considers the full catalog. A midnight-only or dawn-only gate
    # would remove targets with perfectly usable early/late-night windows.
    all_objects = load_objects()
    base_scored = _imaging_catalog(all_objects)
    moon_info = calculate_moon_info(civil_midnight)

    # ── Altitude curves: 15-min steps from dusk to dawn ──────────────
    night_hours = (civil_dawn - civil_dusk).to(u.hour).value
    n_steps = max(2, int(night_hours / 0.25))
    curve_times = [civil_dusk + i * 0.25 * u.hour for i in range(n_steps + 1)]
    curve_frames = [AltAz(obstime=ct, location=observer) for ct in curve_times]

    # Moon altitude curve over the observation window (batch get_body + transform)
    moon_times = Time(list(curve_times))
    moon_positions = get_body('moon', moon_times, location=observer).transform_to(
        AltAz(obstime=moon_times, location=observer)
    )
    moon_alts_arr = moon_positions.alt.deg
    moon_curve = []
    moon_always_down = True
    moon_always_up = True
    _moon_rise_time = None
    _moon_set_time = None
    _prev_moon_alt = float(moon_alts_arr[0]) if len(moon_alts_arr) > 0 else 0.0

    for i, ct in enumerate(curve_times):
        moon_alt_t = float(moon_alts_arr[i])
        moon_curve.append(
            {
                'time': ct.utc.unix,
                'alt': round(moon_alt_t, 1),
            }
        )
        if moon_alt_t > 0:
            moon_always_down = False
        if moon_alt_t <= 0:
            moon_always_up = False

        # Detect horizon crossings (linear interpolation for sub-step precision)
        if i > 0 and _prev_moon_alt <= 0 < moon_alt_t:  # pragma: no cover — needs moon-rise window
            # Moon rises — interpolate crossing time
            frac = (
                (0 - _prev_moon_alt) / (moon_alt_t - _prev_moon_alt)
                if moon_alt_t != _prev_moon_alt
                else 0.5
            )
            prev_ct = curve_times[i - 1]
            cross_t = prev_ct.utc.unix + frac * (ct.utc.unix - prev_ct.utc.unix)
            if _moon_rise_time is None:
                _moon_rise_time = round(cross_t, 0)
        elif i > 0 and _prev_moon_alt > 0 >= moon_alt_t:
            # Moon sets — interpolate crossing time
            frac = (
                (_prev_moon_alt - 0) / (_prev_moon_alt - moon_alt_t)
                if moon_alt_t != _prev_moon_alt
                else 0.5
            )
            prev_ct = curve_times[i - 1]
            cross_t = prev_ct.utc.unix + frac * (ct.utc.unix - prev_ct.utc.unix)
            _moon_set_time = round(cross_t, 0)  # keep last set (may be after midnight)

        _prev_moon_alt = moon_alt_t
    # Fraction of observation window with moon below horizon
    moon_down_count = sum(1 for p in moon_curve if p['alt'] <= 0)
    moon_dark_fraction = round(moon_down_count / max(len(moon_curve), 1), 2)

    # ── Device-aware scoring over the astronomical-dark window ─────────────────
    dusk_frame = AltAz(obstime=civil_dusk, location=observer)
    dawn_frame = AltAz(obstime=civil_dawn, location=observer)

    fov_w = optics.fov_width_deg
    fov_h = optics.fov_height_deg
    lim_mag = optics.limiting_magnitude

    # ── Vectorised dusk / dawn altitude for ALL base-scored objects ─
    # Create one SkyCoord from all _ra/_dec arrays, then two batch
    # transforms — replaces ~3 600 × 2 individual transform_to calls.
    n_scored = len(base_scored)
    _ra_arr = np.array([o.get('_ra', float('nan')) for o in base_scored], dtype=np.float64)
    _dec_arr = np.array([o.get('_dec', float('nan')) for o in base_scored], dtype=np.float64)
    _mag_arr = np.array([o.get('magnitude', 99.9) for o in base_scored], dtype=np.float64)

    # Filter out objects without valid _ra/_dec (e.g. orphaned test objects)
    _has_coord = np.isfinite(_ra_arr) & np.isfinite(_dec_arr)

    coords_stage3 = SkyCoord(
        ra=_ra_arr[_has_coord] * u.deg, dec=_dec_arr[_has_coord] * u.deg, frame='icrs'
    )
    dusk_alts_all = np.full(n_scored, -99.0, dtype=np.float64)
    dawn_alts_all = np.full(n_scored, -99.0, dtype=np.float64)
    dusk_alts_all[_has_coord] = coords_stage3.transform_to(dusk_frame).alt.deg
    dawn_alts_all[_has_coord] = coords_stage3.transform_to(dawn_frame).alt.deg

    # Require at least 30 continuous sampled minutes above 25 degrees in
    # astronomical darkness. Keep civil bounds/curves for public compatibility.
    curve_altitudes = np.empty((n_scored, len(curve_times)), dtype=np.float64)
    curve_azimuths = np.empty(n_scored, dtype=np.float64)
    sun_alts = (
        get_sun(moon_times).transform_to(AltAz(obstime=moon_times, location=observer)).alt.deg
    )
    dark = sun_alts <= -18.0

    moon_up = moon_positions.alt.deg > 0
    usable = np.zeros_like(curve_altitudes, dtype=bool)
    for j, frame in enumerate(curve_frames):
        sky = coords_stage3.transform_to(frame)
        curve_altitudes[:, j] = sky.alt.deg
        usable[:, j] = (sky.alt.deg >= 25.0) & dark[j]
        if moon_info['illumination'] > 0.1 and moon_up[j]:
            usable[:, j] &= sky.separation(moon_positions[j]).deg >= 15.0
    midnight_sky = coords_stage3.transform_to(AltAz(obstime=civil_midnight, location=observer))
    curve_azimuths[:] = midnight_sky.az.deg
    window_minutes = np.array([_longest_window_minutes(row) for row in usable])
    _mag_ok = np.ones(n_scored, dtype=bool)
    # Visual limiting magnitude is not an imaging exposure limit.
    if lim_mag is not None and fov_w is None:
        _mag_ok = (_mag_arr <= lim_mag) | np.isnan(_mag_arr)
    _stage3_keep = _has_coord & _mag_ok & (window_minutes >= 30.0)

    results: list[dict[str, Any]] = []
    for idx in np.flatnonzero(_stage3_keep):
        obj = base_scored[idx]
        mag = float(_mag_arr[idx])
        dusk_alt_val = float(dusk_alts_all[idx])
        dawn_alt_val = float(dawn_alts_all[idx])

        orig = obj
        maj = orig.get('angular_size_maj_arcmin')
        min_ = orig.get('angular_size_min_arcmin')

        obj_type = obj.get('type', '')

        # --- scores ---
        fov_fit = _score_fov_fit(maj, min_, fov_w or 0, fov_h or 0)
        sb = _calc_surface_brightness(mag, maj, min_)
        filter_score = _score_filter_match(obj_type, config.filter_type)

        # Reject tiny targets, but retain oversized targets as mosaic options.
        if maj is not None and fov_w and fov_h:
            fill = math.pi * (maj / 120.0) * ((min_ or maj) / 120.0) / (fov_w * fov_h)
            if fill < 0.01:
                continue

        # Normalize surface brightness: typical range 10–25 mag/arcmin²
        sb_score = 0.0
        if sb is not None:
            sb_score = max(0.0, min(1.0, (25.0 - sb) / 15.0))

        alt_score = float(np.mean(curve_altitudes[idx, usable[idx]])) / 90.0
        # Reward longer runs up to two hours without claiming SNR prediction.
        window_factor = 0.5 + 0.5 * min(1.0, window_minutes[idx] / 120.0)
        moon_penalty_factor = 0.0
        if moon_info['illumination'] > 0.3:
            moon_penalty_factor = (
                moon_info['illumination'] * float(np.mean(moon_up[usable[idx]])) * 0.25
            )

        total = (
            fov_fit * 0.40 + sb_score * 0.30 + filter_score * 0.20 + alt_score * 0.10
        ) * window_factor - moon_penalty_factor
        total = max(total, 0.0)  # moon penalty can't push below zero

        # Mosaic recommended when the target's elliptical extent exceeds FOV
        _maj_deg = (maj / 60.0) if maj else 0
        _min_deg = (min_ / 60.0) if min_ else _maj_deg
        _fits_in_fov = (
            fov_w is not None
            and fov_h is not None
            and max(_maj_deg, _min_deg) <= max(fov_w, fov_h)
            and min(_maj_deg, _min_deg) <= min(fov_w, fov_h)
        )
        mosaic_recommended = bool(maj and fov_w and not _fits_in_fov)

        # Optimal camera rotation: align sensor long side with target major axis
        _pa = orig.get('angular_size_pa_deg')
        _optimal_rot = round(((_pa or 0) + 90) % 360, 0) if _pa is not None else None

        results.append(
            {
                'name': obj['name'],
                'ra': obj['_ra'],
                'dec': obj['_dec'],
                'type': obj_type,
                'magnitude': mag if math.isfinite(mag) else None,
                'surface_brightness': round(sb, 2) if sb is not None else None,
                'angular_size_arcmin': maj,
                'angular_size_min_arcmin': min_,
                'angular_size_pa_deg': orig.get('angular_size_pa_deg'),
                'optimal_rotation_deg': _optimal_rot,
                'altitude': round(dusk_alt_val, 1),  # altitude at dusk
                'azimuth': round(float(curve_azimuths[idx]), 1),  # azimuth at midnight
                'dawn_altitude': round(dawn_alt_val, 1),  # altitude at dawn
                'fov_fill_ratio': round(
                    (math.pi * _maj_deg / 2.0 * _min_deg / 2.0) / (fov_w * fov_h),
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
                'altitude_curve': [
                    {'time': ct.utc.unix, 'alt': round(float(alt), 1)}
                    for ct, alt in zip(curve_times, curve_altitudes[idx], strict=True)
                ],
                'observation_time': civil_dusk.isot,
                'civil_dusk': civil_dusk.isot,
                'civil_dawn': civil_dawn.isot,
            }
        )

    # Sort by suitability (desc) then dawn altitude (asc):
    # best targets first; among equals, those setting sooner go first.
    results.sort(key=lambda x: (-x['suitability_score'], x['dawn_altitude']))

    top_targets = results[:limit]

    if top_targets:
        # ── Detect rise / set / transit for each top-N target ─────────
        for target in top_targets:
            curve = target['altitude_curve']
            _rise = None
            _set = None
            _peak = None
            _peak_alt = -999.0
            _prev_alt = curve[0]['alt'] if curve else 0.0

            for j, pt in enumerate(curve):
                alt_j = pt['alt']
                t_j = pt['time']

                # Track peak altitude (culmination / 中天)
                if alt_j > _peak_alt:
                    _peak_alt = alt_j
                    _peak = t_j

                # Detect horizon crossings
                if j > 0:
                    if _prev_alt <= 0 < alt_j:
                        # Rise: crossing above horizon
                        frac = (0 - _prev_alt) / (alt_j - _prev_alt) if alt_j != _prev_alt else 0.5
                        _rise = round(curve[j - 1]['time'] + frac * (t_j - curve[j - 1]['time']), 0)
                    elif _prev_alt > 0 >= alt_j:  # pragma: no cover — target must set in window
                        # Set: crossing below horizon
                        frac = (_prev_alt - 0) / (_prev_alt - alt_j) if _prev_alt != alt_j else 0.5
                        _set = round(curve[j - 1]['time'] + frac * (t_j - curve[j - 1]['time']), 0)

                _prev_alt = alt_j

            target['transit_time'] = _peak
            target['transit_alt'] = round(_peak_alt, 1)
            target['rise_time'] = _rise
            target['set_time'] = _set

    return {
        'targets': top_targets,
        'moon': {
            'illumination': moon_info['illumination'],
            'phase': moon_info['phase_name'],
            'altitude_curve': moon_curve,
            'always_down': moon_always_down,
            'always_up': moon_always_up,
            'dark_fraction': moon_dark_fraction,
            'moonrise': _moon_rise_time,
            'moonset': _moon_set_time,
        },
    }
