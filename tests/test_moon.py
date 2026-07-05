"""Tests for moon phase calculation."""

from datetime import datetime

import pytz

from stargazing_core import calculate_moon_info


def test_full_moon():
    """Known Full Moon: Jan 25, 2024, ~17:54 UTC."""
    dt = datetime(2024, 1, 25, 17, 54, tzinfo=pytz.UTC)
    info = calculate_moon_info(dt)
    assert info['phase_name'] == 'Full Moon'
    assert info['illumination'] > 0.99


def test_new_moon():
    """Known New Moon: Jan 11, 2024, ~11:57 UTC."""
    dt = datetime(2024, 1, 11, 11, 57, tzinfo=pytz.UTC)
    info = calculate_moon_info(dt)
    assert info['phase_name'] == 'New Moon'
    assert info['illumination'] < 0.01


def test_first_quarter():
    """First Quarter ~50% illumination."""
    dt = datetime(2024, 1, 18, 4, 0, tzinfo=pytz.UTC)
    info = calculate_moon_info(dt)
    assert 0.4 < info['illumination'] < 0.6


def test_naive_datetime_raises():
    """Naive datetime should raise ValueError."""
    with __import__('pytest').raises(ValueError, match='timezone-aware'):
        calculate_moon_info(datetime(2024, 1, 1))


def test_last_quarter():
    """Last Quarter: Jan 4, 2024 ~03:30 UTC."""
    dt = datetime(2024, 1, 4, 3, 30, tzinfo=pytz.UTC)
    info = calculate_moon_info(dt)
    assert info['phase_name'] == 'Last Quarter'
    assert 0.4 < info['illumination'] < 0.6


def test_waxing_crescent():
    """A few days after New Moon (Jan 13, 2024)."""
    dt = datetime(2024, 1, 13, 12, 0, tzinfo=pytz.UTC)
    info = calculate_moon_info(dt)
    assert info['phase_name'] == 'Waxing Crescent'
    assert 0.05 < info['illumination'] < 0.3


def test_waxing_gibbous():
    """Between First Quarter and Full Moon (Jan 20, 2024)."""
    dt = datetime(2024, 1, 20, 12, 0, tzinfo=pytz.UTC)
    info = calculate_moon_info(dt)
    assert info['phase_name'] == 'Waxing Gibbous'
    assert 0.7 < info['illumination'] < 0.95


def test_waning_gibbous():
    """A few days after Full Moon (Jan 28, 2024)."""
    dt = datetime(2024, 1, 28, 12, 0, tzinfo=pytz.UTC)
    info = calculate_moon_info(dt)
    assert info['phase_name'] == 'Waning Gibbous'
    assert 0.7 < info['illumination'] < 0.95


def test_waning_crescent():
    """A few days before New Moon (Jan 8, 2024)."""
    dt = datetime(2024, 1, 8, 12, 0, tzinfo=pytz.UTC)
    info = calculate_moon_info(dt)
    assert info['phase_name'] == 'Waning Crescent'
    assert 0.05 < info['illumination'] < 0.3


def test_moon_altaz_returns_tuple():
    """get_moon_altaz returns (alt, az) floats."""
    from astropy.coordinates import EarthLocation
    import astropy.units as uu
    from stargazing_core import get_moon_altaz

    loc = EarthLocation(lat=40.0 * uu.deg, lon=116.0 * uu.deg)
    alt, az = get_moon_altaz(loc, datetime(2024, 1, 25, 22, 0, tzinfo=pytz.UTC))
    assert isinstance(alt, float)
    assert isinstance(az, float)
    assert -90 <= alt <= 90
    assert 0 <= az <= 360


def test_all_fields_present():
    dt = datetime(2024, 1, 25, 17, 54, tzinfo=pytz.UTC)
    info = calculate_moon_info(dt)
    for key in ('illumination', 'phase_name', 'age_days', 'elongation', 'earth_distance'):
        assert key in info, f'Missing key: {key}'
