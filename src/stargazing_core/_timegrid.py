"""Time-grid utilities for rise/set calculations.

Copied from ``mcp-stargazing/src/celestial.py``.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
from astropy.time import Time


def generate_time_grid(date: datetime) -> Time:
    """Generate a grid of Time objects for *date* at 5-minute intervals (288 steps)."""
    start = Time(date.replace(hour=0, minute=0, second=0))
    end = Time(date.replace(hour=23, minute=59, second=59))
    return start + np.linspace(0, 1, 288) * (end - start)


def find_rise_set_indices(altitudes: np.ndarray, horizon: float) -> tuple[int | None, int | None]:
    """Find indices where *altitudes* cross *horizon*.

    Returns ``(rise_idx, set_idx)`` or ``None`` for each when no crossing is found.
    """
    above = altitudes > horizon
    diff = np.diff(above.astype(int))
    rise_candidates = np.where(diff == 1)[0]
    set_candidates = np.where(diff == -1)[0]
    rise_idx = int(rise_candidates[0]) if len(rise_candidates) > 0 else None
    set_idx = int(set_candidates[0]) if len(set_candidates) > 0 else None
    return rise_idx, set_idx
