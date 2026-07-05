"""Visible planet calculation.

Copied from ``mcp-stargazing/src/celestial.py``.  Pure computation — no
network I/O (uses astropy's built-in ephemeris).
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import pytz
from astropy.coordinates import AltAz, EarthLocation, get_body
from astropy.time import Time

from . import _ephemeris  # noqa: F401 — ensure ephemeris is configured

logger = logging.getLogger(__name__)


def get_visible_planets(
    observer_location: EarthLocation, time: Time | datetime
) -> list[dict[str, Any]]:
    """Get a list of planets currently above the horizon.

    Args:
        observer_location: Observer's EarthLocation.
        time: Observation time.

    Returns:
        List of dicts with planet name, altitude, azimuth, and constellation.
    """
    if isinstance(time, datetime):
        if time.tzinfo is None:
            raise ValueError('Input datetime must be timezone-aware for local time.')
        time = Time(time.astimezone(pytz.UTC))

    planets = ['mercury', 'venus', 'mars', 'jupiter', 'saturn', 'uranus', 'neptune']
    visible_planets = []

    for planet in planets:
        obj_coord = get_body(planet, time)
        altaz_frame = AltAz(obstime=time, location=observer_location)
        altaz = obj_coord.transform_to(altaz_frame)

        if altaz.alt.deg > 0:
            visible_planets.append(
                {
                    'name': planet.capitalize(),
                    'altitude': float(altaz.alt.deg),
                    'azimuth': float(altaz.az.deg),
                    'constellation': None,
                }
            )

    return visible_planets
