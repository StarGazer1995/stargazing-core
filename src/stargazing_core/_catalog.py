"""Deep-sky object catalog loader.

Reads the embedded ``objects.json`` (11 000+ Messier / NGC / IC objects with
angular-size data) via :mod:`importlib.resources`.  Callers get plain Python
``list[dict]`` — no astropy required.
"""

from __future__ import annotations

import json
import threading
from importlib import resources
from typing import Any, Optional

# ── Fallback angular sizes by object type ──────────────────────
# Same values as used by the mcp-stargazing download_data.py script.

_FALLBACK_SIZES: dict[str, tuple[float, float]] = {
    # galaxies (elongated — min ≈ 0.5 × maj)
    'Gx': (2.0, 1.0),
    'GiG': (2.0, 1.0),
    'GiP': (2.0, 1.0),
    'GiC': (1.5, 0.7),
    'SyG': (1.5, 0.7),
    'Sy1': (1.5, 0.7),
    'Sy2': (1.5, 0.7),
    'LIN': (1.5, 0.7),
    'AGN': (1.5, 0.7),
    'BLL': (1.5, 0.7),
    'SBG': (1.5, 0.7),
    'H2G': (1.5, 0.7),
    'PaG': (1.5, 0.7),
    'G': (2.0, 1.0),
    'AG?': (1.5, 0.7),
    'IG': (2.0, 1.0),
    'rG': (1.5, 1.0),
    'EmG': (2.0, 1.0),
    'BiC': (2.0, 1.0),
    'G?': (2.0, 1.0),
    'PoG': (1.0, 1.0),
    'mul': (2.0, 2.0),
    # clusters (round)
    'GlC': (5.0, 5.0),
    'Gb': (4.0, 4.0),
    'OpC': (15.0, 15.0),
    'Cl*': (10.0, 10.0),
    'Cl?': (10.0, 10.0),
    'ClG': (10.0, 10.0),
    'OC': (10.0, 10.0),
    # nebulae
    'PN': (0.5, 0.5),
    'HII': (15.0, 10.0),
    'RNe': (15.0, 10.0),
    'Nb': (15.0, 10.0),
    'C+N': (15.0, 10.0),
    'ISM': (20.0, 12.0),
    'sh': (15.0, 10.0),
    'Em*': (5.0, 5.0),
    'GNe': (15.0, 10.0),
    'EmO': (10.0, 10.0),
    'HH': (0.5, 0.5),
    'Opt': (1.0, 1.0),
    # supernova remnants
    'SNR': (8.0, 8.0),
}

# thread-safe cache
_catalog_cache: Optional[list[dict[str, Any]]] = None
_catalog_lock = threading.Lock()


def _load_data_resource(filename: str) -> list[dict[str, Any]]:
    """Read packaged JSON data from ``stargazing_core/data/``."""
    pkg = resources.files('stargazing_core')
    return json.loads((pkg / 'data' / filename).read_text(encoding='utf-8'))


def load_objects() -> list[dict[str, Any]]:
    """Return the full deep-sky object catalog (thread-safe, cached)."""
    global _catalog_cache

    if _catalog_cache is not None:
        return _catalog_cache

    with _catalog_lock:
        if _catalog_cache is not None:  # pragma: no cover — thread-safety
            return _catalog_cache
        _catalog_cache = _load_data_resource('objects.json')

    return _catalog_cache


def get_angular_size_fallback(obj_type: str) -> Optional[tuple[float, float]]:
    """Return ``(maj_arcmin, min_arcmin)`` fallback for *obj_type*, or ``None``."""
    return _FALLBACK_SIZES.get(obj_type)


class DeepSkyCatalog:
    """Lightweight wrapper around the deep-sky object catalog.

    Loads the 11 000+ object catalog from packaged JSON data on first use.
    """

    def __init__(self) -> None:
        self._objects = load_objects()

    def __len__(self) -> int:
        return len(self._objects)

    def all(self) -> list[dict[str, Any]]:
        """Return every object in the catalog."""
        return self._objects

    def by_types(self, types: set[str]) -> list[dict[str, Any]]:
        """Filter objects whose ``type`` field is in *types*."""
        return [o for o in self._objects if o.get('type') in types]

    def with_angular_size(self) -> list[dict[str, Any]]:
        """Return objects that have measured (non-fallback) angular-size data."""
        return [o for o in self._objects if o.get('angular_size_maj_arcmin') is not None]

    def type_counts(self) -> dict[str, int]:
        """Return ``{type: count}`` for the full catalog."""
        counts: dict[str, int] = {}
        for o in self._objects:
            t = o.get('type', '?')
            counts[t] = counts.get(t, 0) + 1
        return counts
