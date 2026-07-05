"""Pure coordinate utilities — no astropy or other heavy deps required.

Originally from ``mcp-stargazing/src/utils.py``.  Kept dependency-free so
both projects can use these without pulling in astropy.
"""


def validate_coordinates(lat: float, lon: float) -> bool:
    """Return ``True`` when *lat* and *lon* are within valid geographic ranges.

    >>> validate_coordinates(40.0, 116.0)
    True
    >>> validate_coordinates(100.0, 0.0)
    False
    """
    return -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0
