"""Tests for visible planet calculation."""

from datetime import datetime

import astropy.units as u
import pytz
from astropy.coordinates import EarthLocation

from stargazing_core import get_visible_planets


def _greenwich():
    return EarthLocation(lat=51.4769 * u.deg, lon=0.0 * u.deg)


def test_jupiter_visible():
    """Jupiter was high in London on Jan 25, 2024, 22:00 UTC."""
    loc = _greenwich()
    dt = datetime(2024, 1, 25, 22, 0, tzinfo=pytz.UTC)
    planets = get_visible_planets(loc, dt)
    names = [p['name'] for p in planets]
    assert 'Jupiter' in names


def test_structure():
    """Each visible planet dict has expected keys."""
    loc = _greenwich()
    dt = datetime(2024, 1, 25, 22, 0, tzinfo=pytz.UTC)
    planets = get_visible_planets(loc, dt)
    assert len(planets) > 0
    for p in planets:
        assert 'altitude' in p
        assert 'azimuth' in p
        assert p['altitude'] > 0


def test_all_above_horizon():
    """All returned planets should be above horizon."""
    loc = EarthLocation(lat=0 * u.deg, lon=0 * u.deg)
    dt = datetime(2024, 1, 1, 12, 0, tzinfo=pytz.UTC)
    planets = get_visible_planets(loc, dt)
    for p in planets:
        assert p['altitude'] > 0


def test_naive_datetime_raises():
    loc = _greenwich()
    with __import__('pytest').raises(ValueError, match='timezone-aware'):
        get_visible_planets(loc, datetime(2024, 1, 1))
