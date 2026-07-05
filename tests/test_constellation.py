"""Tests for constellation identification."""

import astropy.units as u
from astropy.coordinates import SkyCoord

from stargazing_core import identify_constellation


def test_polaris_in_ursa_minor():
    """Polaris is in Ursa Minor."""
    coord = SkyCoord(37.95, 89.26, unit=u.deg, frame='icrs')
    const = identify_constellation(coord)
    assert const == 'Ursa Minor'


def test_betelgeuse_in_orion():
    """Betelgeuse is in Orion."""
    coord = SkyCoord(88.79, 7.41, unit=u.deg, frame='icrs')
    const = identify_constellation(coord)
    assert const == 'Orion'
