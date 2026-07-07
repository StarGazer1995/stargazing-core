"""Shared fixtures for stargazing-core tests."""

import astropy.units as u
import pytest
from astropy.coordinates import EarthLocation
from astropy.time import Time

from stargazing_core import match_telescope_targets
from stargazing_core._telescope import TELESCOPE_PRESETS, TelescopeConfig


@pytest.fixture
def seestar() -> TelescopeConfig:
    """Seestar S50 — a typical smart-telescope config."""
    return TELESCOPE_PRESETS['seestar-s50']


@pytest.fixture
def redcat51() -> TelescopeConfig:
    """RedCat 51 + ASI2600 — a typical wide-field astrograph."""
    return TELESCOPE_PRESETS['redcat51-asi2600']


@pytest.fixture
def c8_reducer() -> TelescopeConfig:
    """Celestron C8 + reducer + ASI2600."""
    return TELESCOPE_PRESETS['c8-reducer-asi2600']


@pytest.fixture(scope='session')
def redcat51_greenwich_jan2024():
    """RedCat51 at Greenwich, Jan 2024 — shared by basic filtering tests."""
    config = TELESCOPE_PRESETS['redcat51-asi2600']
    observer = EarthLocation(lat=51.5 * u.deg, lon=0.0 * u.deg)
    time = Time('2024-01-25T22:00:00')
    return match_telescope_targets(config, observer, time, limit=10)


@pytest.fixture(scope='session')
def redcat51_japan_feb2024():
    """RedCat51 at Japan, Feb 2024 (first quarter moon)."""
    config = TELESCOPE_PRESETS['redcat51-asi2600']
    observer = EarthLocation(lat=35.0 * u.deg, lon=139.0 * u.deg)
    time = Time('2024-02-16T22:00:00')
    return match_telescope_targets(config, observer, time, limit=10)
