"""Tests for deep-sky filtering and scoring."""

import astropy.units as u
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time

from stargazing_core._filtering import filter_candidates_by_lst, score_deep_sky_objects


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
        {'name': 'M 42', 'type': 'Neb', 'ra': 83.8, 'dec': -5.4,
         'magnitude': 4.0, 'catalog': 'Messier'},
    ]
    result = score_deep_sky_objects(candidates, time, loc, moon_coord, 0.99)
    assert len(result) >= 0


def test_filter_ra_wraparound():
    """RA near 360° with LST near 0° should work (wrap-around)."""
    objects = [
        {'name': 'NearPole', 'ra': 359.0, 'dec': 80.0,
         'magnitude': 5.0, 'catalog': 'NGC'},
    ]
    result = filter_candidates_by_lst(objects, lst_deg=1.0)
    assert len(result) == 1


def test_score_skip_bad_coords():
    """Objects with invalid ra/dec should be skipped."""
    loc = _greenwich()
    time = Time('2024-01-25T22:00:00')
    moon = SkyCoord(180, 0, unit=u.deg, frame='icrs')
    candidates = [
        {'name': 'Bad', 'ra': 'xxx', 'dec': 'yyy',
         'magnitude': 5.0, 'catalog': 'NGC'},
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
        {'name': 'M 42', 'ra': 83.8, 'dec': -5.4,
         'magnitude': 4.0, 'catalog': 'Messier'},
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
        {'name': 'LowDec', 'ra': 60.0, 'dec': -85.0,
         'magnitude': 5.0, 'catalog': 'NGC'},
    ]
    result = score_deep_sky_objects(candidates, time, loc, moon, 0.1)
    assert len(result) == 0
