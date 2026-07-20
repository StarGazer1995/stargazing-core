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

    effective_mag = mag[valid].copy()
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
    if maj_arcmin is None:
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
    - > 60% fill → stays at 1.0 (Phase 4 mosaic handles large targets)
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
    from ._moon import calculate_moon_info, get_moon_altaz

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

    # ── Stage 1: coarse filtering (LST at civil midnight) ───────────
    lst_deg = civil_midnight.sidereal_time('mean', longitude=observer.lon).deg
    all_objects = load_objects()
    candidates = filter_candidates_by_lst(all_objects, lst_deg)

    # Build name→object index for O(1) lookup in Stage 3
    _object_index: dict[str, dict[str, Any]] = {o['name']: o for o in all_objects}

    # ── Stage 2: moon + alt/az base scoring at midnight ─────────────
    moon_info = calculate_moon_info(civil_midnight)
    moon_alt, moon_az = get_moon_altaz(observer, civil_midnight.to_datetime())
    moon_coord = SkyCoord(
        moon_az * u.deg,
        moon_alt * u.deg,
        frame='altaz',
        obstime=civil_midnight,
        location=observer,
    ).icrs

    base_scored = score_deep_sky_objects(
        candidates,
        civil_midnight,
        observer,
        moon_coord,
        moon_info['illumination'],
    )

    # ── Altitude curves: 15-min steps from dusk to dawn ──────────────
    night_hours = (civil_dawn - civil_dusk).to(u.hour).value
    n_steps = max(2, int(night_hours / 0.25))
    curve_times = [civil_dusk + i * 0.25 * u.hour for i in range(n_steps + 1)]
    curve_frames = [AltAz(obstime=ct, location=observer) for ct in curve_times]

    # Moon altitude curve over the observation window (batch get_body + transform)
    moon_times = Time(list(curve_times))
    moon_alts_arr = (
        get_body('moon', moon_times)
        .transform_to(AltAz(obstime=moon_times, location=observer))
        .alt.deg
    )
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
        if i > 0 and _prev_moon_alt <= 0 < moon_alt_t:
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

    # ── Dynamic moon penalty: bright moon above horizon washes out the sky ──
    # Penalty ∝ illumination × (fraction of night moon is up).
    if not moon_always_down and moon_info['illumination'] > 0.3:
        moon_penalty_factor = moon_info['illumination'] * (1 - moon_dark_fraction) * 0.25
        for obj in base_scored:
            obj['score'] = obj['score'] + moon_info['illumination'] * 5.0 * (1 - moon_dark_fraction)
    else:
        moon_penalty_factor = 0.0

    # ── Stage 3: device-aware scoring at civil dusk ─────────────────
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

    # Pre-compute filter mask.
    # Match the original NaN behaviour: ``mag > lim_mag`` is False for NaN,
    # so NaN-magnitude objects pass the magnitude filter.  ``<=`` is also
    # False for NaN, so we explicitly include NaN magnitudes.
    _mag_ok = np.ones(n_scored, dtype=bool)
    if lim_mag is not None:
        _mag_ok = (_mag_arr <= lim_mag) | np.isnan(_mag_arr)
    _dawn_ok = dawn_alts_all >= 20.0
    _stage3_keep = _has_coord & _mag_ok & _dawn_ok

    results: list[dict[str, Any]] = []
    for idx in np.flatnonzero(_stage3_keep):
        obj = base_scored[idx]
        mag = float(_mag_arr[idx])
        dusk_alt_val = float(dusk_alts_all[idx])
        dawn_alt_val = float(dawn_alts_all[idx])

        # --- find original catalog entry for angular size ---
        orig = _object_index.get(obj['name'])
        if orig is None:
            continue
        maj = orig.get('angular_size_maj_arcmin')
        min_ = orig.get('angular_size_min_arcmin')

        obj_type = obj['type']

        # --- scores ---
        fov_fit = _score_fov_fit(maj, min_, fov_w or 0, fov_h or 0)
        sb = _calc_surface_brightness(mag, maj, min_)
        filter_score = _score_filter_match(obj_type, config.filter_type)

        # --- hard filter: FOV too small for imaging ---
        if maj is not None and fov_fit < 0.1:
            continue

        # Normalize surface brightness: typical range 10–25 mag/arcmin²
        sb_score = 0.0
        if sb is not None:
            sb_score = max(0.0, min(1.0, (25.0 - sb) / 15.0))

        alt_score = obj['altitude'] / 90.0

        total = (
            fov_fit * 0.40
            + sb_score * 0.30
            + filter_score * 0.20
            + alt_score * 0.10
            - moon_penalty_factor
        )
        total = max(total, 0.0)  # moon penalty can't push below zero

        # Mosaic recommended when the target's elliptical extent exceeds FOV
        _maj_deg = (maj / 60.0) if maj else 0
        _min_deg = (min_ / 60.0) if min_ else _maj_deg
        _fits_in_fov = (
            fov_w is not None
            and fov_h is not None
            and _maj_deg <= fov_w * 1.2
            and _min_deg <= fov_h * 1.2
        )
        mosaic_recommended = bool(maj and fov_w and not _fits_in_fov)

        # Optimal camera rotation: align sensor long side with target major axis
        _pa = orig.get('angular_size_pa_deg')
        _optimal_rot = round(((_pa or 0) + 90) % 360, 0) if _pa is not None else None

        results.append(
            {
                'name': obj['name'],
                'ra': orig.get('ra') if orig else None,
                'dec': orig.get('dec') if orig else None,
                'type': obj_type,
                'magnitude': mag,
                'surface_brightness': round(sb, 2) if sb is not None else None,
                'angular_size_arcmin': maj,
                'angular_size_min_arcmin': min_,
                'angular_size_pa_deg': orig.get('angular_size_pa_deg'),
                'optimal_rotation_deg': _optimal_rot,
                'altitude': round(dusk_alt_val, 1),  # altitude at dusk
                'azimuth': obj['azimuth'],  # azimuth at midnight
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
                'altitude_curve': [],  # computed after filtering (lazy, only for top-N)
                'observation_time': civil_dusk.isot,
                'civil_dusk': civil_dusk.isot,
                'civil_dawn': civil_dawn.isot,
            }
        )

    # Sort by suitability (desc) then dawn altitude (asc):
    # best targets first; among equals, those setting sooner go first.
    results.sort(key=lambda x: (-x['suitability_score'], x['dawn_altitude']))

    top_targets = results[:limit]

    # ── Compute altitude curves only for final top-N targets ──────
    # Batch transform: one SkyCoord for all targets, then per-timestep
    # transforms instead of per-target×per-step.  For 7 targets × 56
    # steps this replaces 392 individual transform_to calls with 56
    # batch calls (7 objects each).
    if top_targets:
        top_ras = np.array([t['ra'] for t in top_targets], dtype=np.float64)
        top_decs = np.array([t['dec'] for t in top_targets], dtype=np.float64)
        coords_top = SkyCoord(ra=top_ras * u.deg, dec=top_decs * u.deg, frame='icrs')
        for target in top_targets:
            target['altitude_curve'] = []
        for ct, cf in zip(curve_times, curve_frames, strict=True):
            alts = coords_top.transform_to(cf).alt.deg
            for i, target in enumerate(top_targets):
                target['altitude_curve'].append(
                    {
                        'time': ct.utc.unix,
                        'alt': round(float(alts[i]), 1),
                    }
                )

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
                    elif _prev_alt > 0 >= alt_j:
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
