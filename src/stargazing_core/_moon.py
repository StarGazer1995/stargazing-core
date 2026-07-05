"""Moon phase, illumination, and distance calculations.

Copied from ``mcp-stargazing/src/celestial.py``.  Pure computation — no
network I/O (uses astropy's built-in ephemeris).
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import astropy.units as u
import numpy as np
import pytz
from astropy.coordinates import GeocentricTrueEcliptic, get_body, get_sun
from astropy.time import Time

from . import _ephemeris  # noqa: F401 — ensure ephemeris is configured

logger = logging.getLogger(__name__)


def calculate_moon_info(time: Time | datetime) -> dict[str, Any]:
    """Calculate detailed information about the Moon's phase and position.

    Args:
        time: Observation time (Astropy Time or timezone-aware datetime).

    Returns:
        Dict containing:
        - illumination: Fraction of the moon illuminated (0.0 to 1.0)
        - phase_name: String description of the phase (e.g. "Waxing Gibbous")
        - age_days: Approximate age of the moon in days (since New Moon)
        - elongation: Angular separation from Sun in degrees
        - earth_distance: Distance from Earth in km
    """
    if isinstance(time, datetime):
        if time.tzinfo is None:
            raise ValueError('Input datetime must be timezone-aware for local time.')
        time = Time(time.astimezone(pytz.UTC))

    sun = get_sun(time)
    moon = get_body('moon', time)

    elongation = sun.separation(moon)
    illumination = (1 - np.cos(elongation.rad)) / 2.0

    sun_ecl = sun.transform_to(GeocentricTrueEcliptic(obstime=time))
    moon_ecl = moon.transform_to(GeocentricTrueEcliptic(obstime=time))
    lon_diff = (moon_ecl.lon.deg - sun_ecl.lon.deg) % 360

    if lon_diff < 1 or lon_diff > 359:
        phase_name = 'New Moon'
    elif 1 <= lon_diff < 89:
        phase_name = 'Waxing Crescent'
    elif 89 <= lon_diff <= 91:
        phase_name = 'First Quarter'
    elif 91 < lon_diff < 179:
        phase_name = 'Waxing Gibbous'
    elif 179 <= lon_diff <= 181:
        phase_name = 'Full Moon'
    elif 181 < lon_diff < 269:
        phase_name = 'Waning Gibbous'
    elif 269 <= lon_diff <= 271:
        phase_name = 'Last Quarter'
    else:
        phase_name = 'Waning Crescent'

    age_days = (lon_diff / 360.0) * 29.53059

    return {
        'illumination': float(illumination),
        'phase_name': phase_name,
        'age_days': float(age_days),
        'elongation': float(elongation.deg),
        'earth_distance': float(moon.distance.to(u.km).value),
    }
