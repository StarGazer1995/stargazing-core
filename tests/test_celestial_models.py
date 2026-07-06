"""Tests for celestial Pydantic model validation."""

import pytest
from pydantic import ValidationError

from stargazing_core import CelestialPosition, MoonInfo, RiseSet, VisiblePlanet


class TestCelestialPosition:
    def test_valid_position(self):
        pos = CelestialPosition(altitude=45.0, azimuth=180.0)
        assert pos.altitude == 45.0
        assert pos.azimuth == 180.0

    def test_missing_field(self):
        with pytest.raises(ValidationError):
            CelestialPosition(altitude=45.0)  # missing azimuth

    def test_extra_field_ignored(self):
        pos = CelestialPosition(altitude=30.0, azimuth=90.0)
        assert not hasattr(pos, 'name')


class TestRiseSet:
    def test_valid_rise_set(self):
        rs = RiseSet(rise_time='2024-01-15T07:30:00+08:00', set_time='2024-01-15T17:45:00+08:00')
        assert rs.rise_time == '2024-01-15T07:30:00+08:00'
        assert rs.set_time == '2024-01-15T17:45:00+08:00'

    def test_defaults_to_none(self):
        rs = RiseSet()
        assert rs.rise_time is None
        assert rs.set_time is None


class TestMoonInfo:
    def test_valid_moon_info(self):
        info = MoonInfo(
            illumination=0.5,
            phase_name='First Quarter',
            age_days=7.4,
            elongation=90.0,
            earth_distance=384400.0,
        )
        assert info.phase_name == 'First Quarter'
        assert info.altitude is None
        assert info.azimuth is None

    def test_int_coerced_to_float(self):
        info = MoonInfo(
            illumination=1,
            phase_name='Full Moon',
            age_days=14.8,
            elongation=180.0,
            earth_distance=384400.0,
        )
        assert isinstance(info.illumination, float)

    def test_with_altitude_azimuth(self):
        info = MoonInfo(
            illumination=0.8,
            phase_name='Waxing Gibbous',
            age_days=10.5,
            elongation=120.0,
            earth_distance=390000.0,
            altitude=45.0,
            azimuth=180.0,
        )
        assert info.altitude == 45.0
        assert info.azimuth == 180.0

    def test_missing_required_field(self):
        with pytest.raises(ValidationError):
            # missing age_days, elongation, earth_distance
            MoonInfo(illumination=0.5, phase_name='First Quarter')


class TestVisiblePlanet:
    def test_valid_visible_planet(self):
        planet = VisiblePlanet(name='Mars', altitude=30.0, azimuth=90.0)
        assert planet.name == 'Mars'
        assert planet.altitude == 30.0
        assert planet.azimuth == 90.0
        assert planet.constellation is None

    def test_with_constellation(self):
        planet = VisiblePlanet(name='Jupiter', altitude=60.0, azimuth=270.0, constellation='Taurus')
        assert planet.constellation == 'Taurus'

    def test_inherits_celestial_position(self):
        planet = VisiblePlanet(name='Venus', altitude=15.0, azimuth=45.0)
        assert isinstance(planet, CelestialPosition)

    def test_missing_name(self):
        with pytest.raises(ValidationError):
            VisiblePlanet(altitude=30.0, azimuth=90.0)  # missing name
