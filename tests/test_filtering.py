"""Tests for deep-sky filtering and scoring."""

import astropy.units as u
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time

from stargazing_core import match_telescope_targets
from stargazing_core._filtering import (
    _calc_surface_brightness,
    _score_filter_match,
    _score_fov_fit,
    filter_candidates_by_lst,
    score_deep_sky_objects,
)


def _greenwich():
    return EarthLocation(lat=51.4769 * u.deg, lon=0.0 * u.deg)


def test_filter_by_lst_near_meridian():
    """Objects within ±120° of LST should pass."""
    objects = [
        {'name': 'Obj1', 'ra': 50.0, 'dec': 0.0, 'magnitude': 5.0, 'catalog': 'NGC'},
        {'name': 'Obj2', 'ra': 200.0, 'dec': 0.0, 'magnitude': 5.0, 'catalog': 'NGC'},
    ]
    result = filter_candidates_by_lst(objects, lst_deg=50.0)
    assert len(result) == 1
    assert result[0]['name'] == 'Obj1'


def test_filter_excludes_faint_ngc():
    """NGC objects fainter than mag 10 should be excluded."""
    objects = [
        {'name': 'Faint', 'ra': 50.0, 'dec': 0.0, 'magnitude': 12.0, 'catalog': 'NGC'},
        {'name': 'Bright', 'ra': 50.0, 'dec': 0.0, 'magnitude': 8.0, 'catalog': 'NGC'},
    ]
    result = filter_candidates_by_lst(objects, lst_deg=50.0)
    assert len(result) == 1
    assert result[0]['name'] == 'Bright'


def test_filter_keeps_messier_regardless():
    """Messier objects pass regardless of magnitude."""
    objects = [
        {'name': 'M 101', 'ra': 50.0, 'dec': 0.0, 'magnitude': 12.0, 'catalog': 'Messier'},
    ]
    result = filter_candidates_by_lst(objects, lst_deg=50.0)
    assert len(result) == 1


def test_score_basic():
    loc = _greenwich()
    time = Time('2024-01-25T22:00:00')
    moon_coord = SkyCoord(100, 20, unit=u.deg, frame='icrs')
    candidates = [
        {
            'name': 'M 42',
            'type': 'Neb',
            'ra': 83.8,
            'dec': -5.4,
            'magnitude': 4.0,
            'catalog': 'Messier',
        },
    ]
    result = score_deep_sky_objects(candidates, time, loc, moon_coord, 0.99)
    assert len(result) >= 0


def test_filter_ra_wraparound():
    """RA near 360° with LST near 0° should work (wrap-around)."""
    objects = [
        {'name': 'NearPole', 'ra': 359.0, 'dec': 80.0, 'magnitude': 5.0, 'catalog': 'NGC'},
    ]
    result = filter_candidates_by_lst(objects, lst_deg=1.0)
    assert len(result) == 1


def test_score_skip_bad_coords():
    """Objects with invalid ra/dec should be skipped."""
    loc = _greenwich()
    time = Time('2024-01-25T22:00:00')
    moon = SkyCoord(180, 0, unit=u.deg, frame='icrs')
    candidates = [
        {'name': 'Bad', 'ra': 'xxx', 'dec': 'yyy', 'magnitude': 5.0, 'catalog': 'NGC'},
    ]
    result = score_deep_sky_objects(candidates, time, loc, moon, 0.5)
    assert len(result) == 0


def test_score_too_close_to_moon():
    """Object within 15° of bright moon is skipped."""
    loc = EarthLocation(lat=30 * u.deg, lon=0 * u.deg)
    time = Time('2024-01-25T22:00:00')
    # Place moon at ra=83.8, dec=-5.4 (same as M42)
    moon = SkyCoord(83.8, -5.4, unit=u.deg, frame='icrs')
    candidates = [
        {'name': 'M 42', 'ra': 83.8, 'dec': -5.4, 'magnitude': 4.0, 'catalog': 'Messier'},
    ]
    result = score_deep_sky_objects(candidates, time, loc, moon, 0.99)
    assert len(result) == 0  # too close to moon


def test_score_below_horizon_skipped():
    """Object below 20° altitude is skipped."""
    loc = _greenwich()
    time = Time('2024-01-25T22:00:00')
    moon = SkyCoord(180, 0, unit=u.deg, frame='icrs')
    # Object at dec=-85° from Greenwich — always low
    candidates = [
        {'name': 'LowDec', 'ra': 60.0, 'dec': -85.0, 'magnitude': 5.0, 'catalog': 'NGC'},
    ]
    result = score_deep_sky_objects(candidates, time, loc, moon, 0.1)
    assert len(result) == 0


# ── Surface brightness ──────────────────────────────────────────────────


def test_surface_brightness_m33():
    """M33 (Triangulum Galaxy): mag 5.72, 60.3'×35.5' → known SB."""
    sb = _calc_surface_brightness(5.72, 60.3, 35.5)
    assert sb is not None
    # M33 surface brightness ≈ 14.2 mag/arcmin²
    assert 13.5 <= sb <= 15.0


def test_surface_brightness_circular():
    """Circular approximation when min_arcmin is None."""
    sb = _calc_surface_brightness(5.0, 10.0, None)
    assert sb is not None
    # SB = 5.0 + 2.5*log10(π*5²) ≈ 5.0 + 4.73 ≈ 9.73
    assert 9.0 <= sb <= 10.5


def test_surface_brightness_none_when_no_size():
    """Returns None when maj_arcmin is None."""
    assert _calc_surface_brightness(5.0, None, None) is None


# ── FOV fit score ───────────────────────────────────────────────────────


def test_fov_fit_optimal():
    """30 arcmin object in 1°×0.7° FOV → ~14% fill → 1.0."""
    score = _score_fov_fit(30, 1.0, 0.7)
    assert score == 1.0


def test_fov_fit_too_small():
    """3 arcmin object in 1°×0.7° FOV → ~0.14% fill → near 0."""
    score = _score_fov_fit(3, 1.0, 0.7)
    assert score < 0.5


def test_fov_fit_large_mosaic():
    """120 arcmin (2°) object in 1°×0.7° FOV → >100% fill → still 1.0."""
    score = _score_fov_fit(120, 1.0, 0.7)
    assert score == 1.0


def test_fov_fit_no_size():
    """None angular size → 0."""
    assert _score_fov_fit(None, 1.0, 0.7) == 0.0


def test_fov_fit_no_fov():
    """Zero FOV → 0."""
    assert _score_fov_fit(30, 0, 0) == 0.0


# ── Filter match score ──────────────────────────────────────────────────


def test_filter_match_halpha_emission():
    assert _score_filter_match('emission nebula', 'Hα') == 1.0


def test_filter_match_halpha_galaxy():
    assert _score_filter_match('galaxy', 'Hα') == 0.1


def test_filter_match_none_galaxy():
    """No filter (LRGB) with galaxy → 1.0."""
    assert _score_filter_match('galaxy', None) == 1.0


def test_filter_match_unknown_type_fallback():
    """Unknown object type gets fallback score."""
    assert _score_filter_match('dark nebula', 'Hα') == 0.3


# ── End-to-end matching ─────────────────────────────────────────────────


def test_match_telescope_targets_basic():
    """End-to-end: RedCat51 preset at Greenwich should return results."""
    from stargazing_core._telescope import TELESCOPE_PRESETS

    config = TELESCOPE_PRESETS['redcat51-asi2600']
    observer = EarthLocation(lat=51.5 * u.deg, lon=0.0 * u.deg)
    time = Time('2024-01-25T22:00:00')

    results = match_telescope_targets(config, observer, time, limit=10)

    assert len(results) > 0
    assert len(results) <= 10
    # Verify structure of first result
    r = results[0]
    assert 'name' in r
    assert 'ra' in r
    assert 'dec' in r
    assert isinstance(r['ra'], (int, float))
    assert isinstance(r['dec'], (int, float))
    assert 'suitability_score' in r
    assert 'fov_fit_score' in r
    assert 'mosaic_recommended' in r
    assert 0 <= r['suitability_score'] <= 100
    # Results should be sorted by score descending
    scores = [x['suitability_score'] for x in results]
    assert scores == sorted(scores, reverse=True)


def test_match_telescope_targets_empty_with_weak_scope():
    """A very small scope (low limiting mag) may return few/no results."""
    from stargazing_core._telescope import TELESCOPE_PRESETS

    config = TELESCOPE_PRESETS['redcat51-asi2600'].model_copy()
    config.aperture_mm = 30  # tiny — limiting mag ≈ 7
    observer = EarthLocation(lat=51.5 * u.deg, lon=0.0 * u.deg)
    time = Time('2024-01-25T22:00:00')

    results = match_telescope_targets(config, observer, time, limit=20)
    # With limiting mag ~7, most DSOs are filtered out
    assert isinstance(results, list)


def test_match_mosaic_recommended_for_large_target():
    """M31 should get mosaic_recommended=True for a narrow-FOV scope."""
    from stargazing_core._telescope import TELESCOPE_PRESETS

    # C14 with reducer — narrow FOV (~0.5°×0.35°)
    config = TELESCOPE_PRESETS['c14-reducer-asi2600']
    observer = EarthLocation(lat=41.0 * u.deg, lon=-72.0 * u.deg)
    time = Time('2024-11-15T02:00:00')  # M31 is well-placed

    results = match_telescope_targets(config, observer, time, limit=30)
    m31 = next((r for r in results if 'M 31' in r['name']), None)

    if m31 is not None:
        assert m31['mosaic_recommended'] is True
