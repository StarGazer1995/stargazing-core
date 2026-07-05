"""Tests for pure coordinate utilities."""

import pytest

from stargazing_core._coord import validate_coordinates


@pytest.mark.parametrize(
    'lat,lon,expected',
    [
        (0.0, 0.0, True),
        (90.0, 180.0, True),
        (-90.0, -180.0, True),
        (40.0, 116.0, True),
        (90.1, 0.0, False),
        (-90.1, 0.0, False),
        (0.0, 180.1, False),
        (0.0, -180.1, False),
        (100.0, 200.0, False),
    ],
)
def test_validate_coordinates(lat: float, lon: float, expected: bool) -> None:
    assert validate_coordinates(lat, lon) is expected
