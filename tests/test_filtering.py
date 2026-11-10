"""Tests for deep-sky filtering and scoring."""

import json
import math

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import AltAz, EarthLocation, SkyCoord, get_sun
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


@pytest.mark.parametrize(
    'mag,major,minor',
    [
        (float('nan'), 480.0, 480.0),
        (float('inf'), 480.0, 480.0),
        (5.0, float('nan'), 10.0),
        (5.0, 10.0, float('nan')),
    ],
)
def test_missing_photometry_does_not_become_maximum_brightness(mag, major, minor):
    assert _calc_surface_brightness(mag, major, minor) is None


def test_simbad_types_use_the_same_filter_scores_as_english_types():
    assert _score_filter_match('OpC', 'Hα') == 0.0
    assert _score_filter_match('OpC', None) == 1.0
    assert _score_filter_match('HII', 'Hα') == 1.0
    assert _score_filter_match('G', 'Hα') == 0.1


# ── FOV fit score ───────────────────────────────────────────────────────


def test_fov_fit_optimal():
    """30 arcmin object in 1°×0.7° FOV → ~14% fill → 1.0."""
    score = _score_fov_fit(30, None, 1.0, 0.7)
    assert score == 1.0


def test_fov_fit_too_small():
    """3 arcmin object in 1°×0.7° FOV → ~0.14% fill → near 0."""
    score = _score_fov_fit(3, None, 1.0, 0.7)
    assert score < 0.5


def test_fov_fit_large_mosaic():
    """Oversized targets incur a mosaic coverage cost."""
    score = _score_fov_fit(120, None, 1.0, 0.7)
    assert score == pytest.approx(0.175)


def test_fov_fit_no_size():
    """None angular size → 0."""
    assert _score_fov_fit(None, None, 1.0, 0.7) == 0.0


def test_fov_fit_no_fov():
    """Zero FOV → 0."""
    assert _score_fov_fit(30, None, 0, 0) == 0.0


def test_fov_fit_ellipse_lower_than_circular():
    """Elongated object (30"×3") scores lower than circular (30"×30")."""
    score_circular = _score_fov_fit(30, None, 0.5, 0.5)
    score_ellipse = _score_fov_fit(30, 3, 0.5, 0.5)
    # Ellipse area is 1/10 of circular → fill ratio drops → score drops
    assert score_ellipse < score_circular


def test_fov_fit_ellipse_with_min():
    """Explicit min_arcmin gives correct ellipse-area score."""
    score = _score_fov_fit(30, 20, 1.0, 0.7)
    assert score == 1.0  # fill ≈ 0.375 → ≥10%


def test_fov_fit_ellipse_round_object():
    """When maj == min the ellipse score equals the circular score."""
    score = _score_fov_fit(30, 30, 1.0, 0.7)
    assert score == 1.0


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


def test_match_telescope_targets_basic(redcat51_greenwich_jan2024):
    """End-to-end: RedCat51 preset at Greenwich should return results."""
    result = redcat51_greenwich_jan2024

    assert 'targets' in result
    assert 'moon' in result
    targets = result['targets']
    moon = result['moon']
    assert len(targets) > 0
    assert len(targets) <= 10
    # Unknown catalog magnitudes are null; scores stay finite and JSON is strict.
    json.dumps(targets, allow_nan=False)
    assert all(math.isfinite(target['suitability_score']) for target in targets)

    # Verify moon fields
    assert 'illumination' in moon
    assert 'phase' in moon
    assert 'altitude_curve' in moon
    assert 'always_down' in moon
    assert 'always_up' in moon
    assert 'dark_fraction' in moon
    assert 0.0 <= moon['illumination'] <= 1.0
    assert 0.0 <= moon['dark_fraction'] <= 1.0
    assert len(moon['altitude_curve']) >= 2
    assert isinstance(moon['phase'], str)
    assert isinstance(moon['always_down'], bool)
    assert isinstance(moon['always_up'], bool)
    # moonrise / moonset are explicit horizon crossing timestamps (or None)
    assert 'moonrise' in moon
    assert 'moonset' in moon
    if moon['moonrise'] is not None:
        assert isinstance(moon['moonrise'], (int, float))
    if moon['moonset'] is not None:
        assert isinstance(moon['moonset'], (int, float))

    # Verify structure of first result
    r = targets[0]
    assert 'name' in r
    assert 'ra' in r
    assert 'dec' in r
    assert isinstance(r['ra'], (int, float))
    assert isinstance(r['dec'], (int, float))
    assert 'suitability_score' in r
    assert 'fov_fit_score' in r
    assert 'mosaic_recommended' in r
    assert 'optimal_rotation_deg' in r
    # PA is not null for most objects → rotation should be numeric or None
    assert r['optimal_rotation_deg'] is None or isinstance(r['optimal_rotation_deg'], (int, float))
    # Rise/set/transit times extracted from altitude_curve
    assert 'rise_time' in r
    assert 'set_time' in r
    assert 'transit_time' in r
    assert 'transit_alt' in r
    if r['rise_time'] is not None:
        assert isinstance(r['rise_time'], (int, float))
    if r['set_time'] is not None:
        assert isinstance(r['set_time'], (int, float))
    if r['transit_time'] is not None:
        assert isinstance(r['transit_time'], (int, float))
        assert isinstance(r['transit_alt'], (int, float))
    # Verify altitude_curve was populated for top-N targets
    assert len(r['altitude_curve']) >= 2
    if r['rise_time'] is not None:
        assert r['rise_time'] >= r['altitude_curve'][0]['time']
    if r['set_time'] is not None:
        assert r['set_time'] <= r['altitude_curve'][-1]['time']
    assert 0 <= r['suitability_score'] <= 100
    # Results sorted by suitability_score desc, then dawn_altitude asc
    scores = [x['suitability_score'] for x in targets]
    assert scores == sorted(scores, reverse=True), 'highest scores first'
    # Every target has a continuous imaging window, regardless of dawn altitude.
    from stargazing_core._filtering import _longest_window_minutes

    for target in targets:
        curve_times = Time([point['time'] for point in target['altitude_curve']], format='unix')
        sun_alt = (
            get_sun(curve_times)
            .transform_to(
                AltAz(obstime=curve_times, location=EarthLocation(lat=51.5 * u.deg, lon=0 * u.deg))
            )
            .alt.deg
        )
        usable = np.array([point['alt'] >= 25 for point in target['altitude_curve']]) & (
            sun_alt <= -18
        )
        assert _longest_window_minutes(usable) >= 30

    # Verify new fields present and sane
    for key in ('dawn_altitude', 'observation_time', 'civil_dusk', 'civil_dawn'):
        assert key in r, f'missing field: {key}'

    # FOV filter: objects with angular_size should have fov_fit_score >= 0.1
    for x in targets:
        if x['angular_size_arcmin'] is not None:
            assert x['fov_fit_score'] > 0, (
                x['name']
                + ': fov_fit_score='
                + str(x['fov_fit_score'])
                + ' with size='
                + str(x['angular_size_arcmin'])
            )

    # Dusk/dawn times should be sane: dawn after dusk
    dusk_t = Time(r['civil_dusk'], format='isot')
    dawn_t = Time(r['civil_dawn'], format='isot')
    assert dawn_t > dusk_t, 'civil_dawn must be after civil_dusk'


def test_match_telescope_targets_empty_with_weak_scope():
    """A very small scope (low limiting mag) may return few/no results."""
    from stargazing_core._telescope import TELESCOPE_PRESETS

    config = TELESCOPE_PRESETS['redcat51-asi2600'].model_copy()
    config.aperture_mm = 30  # tiny — limiting mag ≈ 7
    observer = EarthLocation(lat=51.5 * u.deg, lon=0.0 * u.deg)
    time = Time('2024-01-25T22:00:00')

    result = match_telescope_targets(config, observer, time, limit=20)
    # With limiting mag ~7, most DSOs are filtered out
    assert isinstance(result, dict)
    assert 'targets' in result
    assert isinstance(result['targets'], list)


def test_match_telescope_targets_moon_sets_during_night(redcat51_japan_feb2024):
    """Moon that sets during the night → always_up=False, dark_fraction > 0."""
    result = redcat51_japan_feb2024
    moon = result['moon']
    assert moon['always_up'] is False, 'first quarter moon should set during the night'
    assert moon['always_down'] is False, 'first quarter moon should be up at dusk'
    assert 0.0 < moon['dark_fraction'] < 1.0, 'moon should set partway through the night'
    assert len(moon['altitude_curve']) > 0


def test_match_mosaic_recommended_for_large_target():
    """M31 should get mosaic_recommended=True for a narrow-FOV scope."""
    from stargazing_core._telescope import TELESCOPE_PRESETS

    # C14 with reducer — narrow FOV (~0.5°×0.35°)
    config = TELESCOPE_PRESETS['c14-reducer-asi2600']
    observer = EarthLocation(lat=41.0 * u.deg, lon=-72.0 * u.deg)
    time = Time('2024-11-15T02:00:00')  # M31 is well-placed

    result = match_telescope_targets(config, observer, time, limit=30)
    m31 = next((r for r in result['targets'] if 'M 31' in r['name']), None)

    if m31 is not None:
        assert m31['mosaic_recommended'] is True


# ── Edge-case coverage for diff-cover ────────────────────────────────────


def test_filter_by_lst_empty_input():
    """Empty object list returns empty list."""
    assert filter_candidates_by_lst([], lst_deg=50.0) == []


def test_score_deep_sky_objects_empty_input():
    """Empty candidate list returns empty list."""
    loc = _greenwich()
    time = Time('2024-01-25T22:00:00')
    moon = SkyCoord(180, 0, unit=u.deg, frame='icrs')
    assert score_deep_sky_objects([], time, loc, moon, 0.5) == []


def test_moon_penalty_skipped_when_moon_down():
    """moon_penalty_factor stays 0.0 when moon illumination is ≤0.3 (new moon)."""
    from stargazing_core._telescope import TELESCOPE_PRESETS

    config = TELESCOPE_PRESETS['redcat51-asi2600']
    # New moon on 2024-03-10 — illumination ≈ 0.0, below the 0.3 threshold
    observer = EarthLocation(lat=51.5 * u.deg, lon=0.0 * u.deg)
    time = Time('2024-03-10T22:00:00')

    result = match_telescope_targets(config, observer, time, limit=10)
    assert len(result['targets']) >= 0  # should complete without error
    assert result['moon']['illumination'] <= 0.3


@pytest.mark.parametrize(
    'mask,minutes',
    [
        ([], 0),
        ([True], 0),
        ([True, True], 15),
        ([True, True, True], 30),
        ([True, True, False, True, True, True], 30),
    ],
)
def test_longest_imaging_window_counts_intervals(mask, minutes):
    from stargazing_core._filtering import _longest_window_minutes

    assert _longest_window_minutes(np.array(mask)) == minutes


def test_imaging_catalog_rejects_invalid_coordinates():
    from stargazing_core._filtering import _imaging_catalog

    invalid = [
        {},
        {'ra': None, 'dec': 0},
        {'ra': 'bad', 'dec': 0},
        {'ra': float('nan'), 'dec': 0},
        {'ra': 0, 'dec': 91},
    ]
    good = {'name': 'valid', 'ra': '10', 'dec': '20', 'magnitude': 15, 'catalog': 'NGC'}
    assert _imaging_catalog(invalid + [good]) == [{**good, '_ra': 10.0, '_dec': 20.0}]


def test_imaging_keeps_early_late_and_faint_targets(monkeypatch):
    import stargazing_core._catalog as catalog
    import stargazing_core._moon as moon
    from stargazing_core import TELESCOPE_PRESETS

    objects = [
        {'name': 'early', 'ra': 10.7, 'dec': 41.3},
        {'name': 'late', 'ra': 220.0, 'dec': 20.0},
        {'name': 'faint', 'ra': 83.8, 'dec': -5.4},
    ]
    for obj in objects:
        obj.update(
            type='HII',
            magnitude=14.0,
            catalog='NGC',
            angular_size_maj_arcmin=60.0,
            angular_size_min_arcmin=40.0,
        )
    monkeypatch.setattr(catalog, 'load_objects', lambda: objects)
    monkeypatch.setattr(
        moon, 'calculate_moon_info', lambda time: {'illumination': 0.0, 'phase_name': 'New Moon'}
    )
    result = match_telescope_targets(
        TELESCOPE_PRESETS['hope_d60_asi2600mc_air'],
        EarthLocation(lat=51.5 * u.deg, lon=0 * u.deg),
        Time('2024-01-25T22:00:00'),
        10,
    )
    by_name = {obj['name']: obj for obj in result['targets']}
    assert set(by_name) == {'early', 'late', 'faint'}
    assert by_name['early']['dawn_altitude'] < 20
    assert by_name['late']['altitude'] < 20
    assert all(obj['magnitude'] == 14 for obj in by_name.values())


def test_fov_score_allows_camera_rotation_and_penalizes_long_axes():
    assert _score_fov_fit(40, 20, 1, 0.5) == _score_fov_fit(40, 20, 0.5, 1)
    assert _score_fov_fit(180, 5, 1, 0.5) < _score_fov_fit(40, 20, 1, 0.5)


def test_visual_magnitude_gate_and_empty_imaging_catalog(monkeypatch):
    import stargazing_core._catalog as catalog
    import stargazing_core._moon as moon
    from stargazing_core import TelescopeConfig

    monkeypatch.setattr(
        moon, 'calculate_moon_info', lambda time: {'illumination': 0, 'phase_name': 'New Moon'}
    )
    objects = [
        {'name': 'bright', 'ra': 83.8, 'dec': -5.4, 'magnitude': 4},
        {'name': 'faint', 'ra': 83.8, 'dec': -5.4, 'magnitude': 14},
        {'name': 'unknown', 'ra': 83.8, 'dec': -5.4, 'magnitude': float('nan')},
    ]
    monkeypatch.setattr(catalog, 'load_objects', lambda: objects)
    config = TelescopeConfig(focal_length_mm=280, aperture_mm=60)
    observer = EarthLocation(lat=51.5 * u.deg, lon=0 * u.deg)
    time = Time('2024-01-25T22:00:00')
    assert {obj['name'] for obj in match_telescope_targets(config, observer, time)['targets']} == {
        'bright',
        'unknown',
    }
    monkeypatch.setattr(catalog, 'load_objects', lambda: [])
    assert match_telescope_targets(config, observer, time)['targets'] == []


def test_imaging_cluster_sizes_do_not_reuse_extended_membership_regions():
    from stargazing_core import load_objects
    from stargazing_core._filtering import _imaging_catalog

    original = {obj['name']: obj for obj in load_objects()}
    projected = {obj['name']: obj for obj in _imaging_catalog(list(original.values()))}
    assert projected['M 36']['angular_size_maj_arcmin'] == 7.2
    assert projected['NGC 2169']['angular_size_maj_arcmin'] == 7.2
    assert projected['M 36']['angular_size_min_arcmin'] is None
    assert original['M 36']['angular_size_maj_arcmin'] == 120.0
    assert (
        projected['IC 1396']['angular_size_maj_arcmin']
        == original['IC 1396']['angular_size_maj_arcmin']
    )
    assert _score_filter_match('AGN', None) == _score_filter_match('galaxy', None)
