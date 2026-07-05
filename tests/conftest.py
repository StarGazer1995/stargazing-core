"""Shared fixtures for stargazing-core tests."""

import pytest

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
