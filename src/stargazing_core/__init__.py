"""stargazing-core — shared astronomical computation.

Provides :class:`TelescopeConfig`, :class:`TelescopeOptics`, equipment presets,
and deep-sky catalog loading used by both ``mcp-stargazing`` and
``stargazing-place-finder``.
"""

from stargazing_core._catalog import DeepSkyCatalog, load_objects
from stargazing_core._telescope import (
    TELESCOPE_PRESETS,
    TelescopeConfig,
    TelescopeOptics,
)

__all__ = [
    'TelescopeConfig',
    'TelescopeOptics',
    'TELESCOPE_PRESETS',
    'DeepSkyCatalog',
    'load_objects',
]
__version__ = '0.1.0a1'
