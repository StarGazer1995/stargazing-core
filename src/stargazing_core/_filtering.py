"""Deep-sky object filtering and scoring.

Copied from ``mcp-stargazing/src/celestial.py``.  Pure computation — uses
astropy for coordinate transforms but no network I/O.
"""

from __future__ import annotations

from typing import Any

import astropy.units as u
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time

from . import _ephemeris  # noqa: F401 — ensure ephemeris is configured


def filter_candidates_by_lst(
    raw_objects: list[dict[str, Any]], lst_deg: float
) -> list[dict[str, Any]]:
    """Filter deep-sky objects to those near the meridian (±8h RA from LST)."""
    candidates: list[dict[str, Any]] = []

    for obj in raw_objects:
        mag = obj.get('magnitude', 99.9)
        catalog = obj.get('catalog', 'Unknown')

        if catalog == 'NGC' and mag > 10.0:
            continue

        obj_ra = obj['ra']
        diff = abs(obj_ra - lst_deg)
        if diff > 180:
            diff = 360 - diff

        if diff > 120:  # ~8 hours
            continue

        candidates.append(obj)

    return candidates


def score_deep_sky_objects(
    candidates: list[dict[str, Any]],
    time: Time,
    observer_location: EarthLocation,
    moon_coord: SkyCoord,
    moon_illum: float,
) -> list[dict[str, Any]]:
    """Score each candidate by altitude, moon interference, and catalog prestige."""
    scored: list[dict[str, Any]] = []

    altaz_frame = AltAz(obstime=time, location=observer_location)
    moon_altaz = moon_coord.transform_to(altaz_frame)
    moon_up = moon_illum > 0.1 and moon_altaz.alt.deg > 0

    for obj in candidates:
        try:
            ra_val = float(obj['ra'])
            dec_val = float(obj['dec'])
        except (ValueError, TypeError):
            continue

        coord = SkyCoord(ra=ra_val * u.deg, dec=dec_val * u.deg, frame='icrs')
        altaz = coord.transform_to(altaz_frame)
        alt = altaz.alt.deg

        if alt < 20:
            continue

        mag = obj.get('magnitude', 99.9)
        effective_mag = mag

        if moon_up:
            sep = coord.separation(moon_coord).deg
            if sep < 15:
                continue
            elif sep < 60:
                effective_mag += (60 - sep) * 0.1

        alt_bonus = (alt / 90.0) * 2.0
        score = effective_mag - alt_bonus

        if obj.get('catalog') == 'Messier':
            score -= 5.0

        scored.append(
            {
                'name': obj['name'],
                'type': obj['type'],
                'magnitude': mag,
                'altitude': round(alt, 1),
                'azimuth': round(altaz.az.deg, 1),
                'catalog': obj.get('catalog', 'Unknown'),
                'score': score,
            }
        )

    scored.sort(key=lambda x: x['score'])
    return scored
