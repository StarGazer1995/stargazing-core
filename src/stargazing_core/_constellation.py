"""Constellation identification.

Copied from ``mcp-stargazing/src/celestial.py``.  Pure computation — uses
astropy's built-in constellation boundaries.
"""

from astropy.coordinates import SkyCoord, get_constellation

from . import _ephemeris  # noqa: F401 — ensure ephemeris is configured


def identify_constellation(sky_coord: SkyCoord) -> str:
    """Identify which constellation a coordinate belongs to."""
    return get_constellation(sky_coord)
