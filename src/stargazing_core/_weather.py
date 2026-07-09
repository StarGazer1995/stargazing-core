"""Spatial weather data reader and tile renderer.

Reads Open-Meteo spatial weather data (`OM file format`_) via the ``omfiles``
package and renders weather-variable overlays as PNG tile images.

Provides:
  - :class:`WeatherModel` / :class:`WeatherVariable` enums
  - :class:`OmWeatherReader` for window-based data extraction
  - :func:`weather_tile_bounds` for XYZ ↔ lat/lon conversion
  - :func:`render_weather_tile` for NumPy → RGBA PNG rendering
  - :class:`WeatherTileCache` for thread-safe LRU caching

.. _OM file format: https://github.com/open-meteo/om-file-format
"""

from __future__ import annotations

import json
import math
import os
import tempfile
import threading
import time
import urllib.request
from collections import OrderedDict
from enum import StrEnum
from io import BytesIO
from typing import Callable

import fsspec
import numpy as np
from omfiles import OmFileReader as _OmFileReader
from PIL import Image

# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

_SPATIAL_BASE_URL: str = 'https://map-tiles.open-meteo.com/data_spatial'
_S3_BASE: str = 's3://openmeteo/data_spatial'
_DEFAULT_BLOCK_CACHE: str = os.path.join(tempfile.gettempdir(), 'om_block_cache')
_TILE_SIZE: int = 256
_METADATA_FETCH_TIMEOUT: float = 15.0  # seconds
_PNG_COMPRESS_LEVEL: int = 1  # speed-optimised (default is 6)


# ═══════════════════════════════════════════════════════════════════════════
# Enums
# ═══════════════════════════════════════════════════════════════════════════


class WeatherModel(StrEnum):
    """Supported numerical weather prediction models with spatial tile data."""

    DWD_ICON = 'dwd_icon'
    """DWD ICON Global — 0.1° (~11 km), hourly, 7.5-day forecast, every 6h."""

    DWD_ICON_EU = 'dwd_icon_eu'
    """DWD ICON-EU nest — 0.0625° (~7 km), hourly, 5-day forecast, every 3h."""

    DWD_ICON_D2 = 'dwd_icon_d2'
    """DWD ICON-D2 — 0.02° (~2 km), hourly, 2-day forecast, Central Europe."""

    ECMWF_IFS025 = 'ecmwf_ifs025'
    """ECMWF IFS — 0.25° (~25 km), 3-hourly, 15-day forecast, every 6h."""

    NCEP_GFS013 = 'ncep_gfs013'
    """NOAA NCEP GFS — 0.11° (~13 km), hourly, 16-day forecast, every 6h."""


class WeatherVariable(StrEnum):
    """Weather variables available through the spatial tile service."""

    CLOUD_COVER = 'cloud_cover'
    CLOUD_COVER_LOW = 'cloud_cover_low'
    CLOUD_COVER_MID = 'cloud_cover_mid'
    CLOUD_COVER_HIGH = 'cloud_cover_high'
    PRECIPITATION = 'precipitation'
    WIND_SPEED_10M = 'wind_speed_10m'
    TEMPERATURE_2M = 'temperature_2m'
    CAPE = 'cape'


# ═══════════════════════════════════════════════════════════════════════════
# Colormap definitions
# ═══════════════════════════════════════════════════════════════════════════

#: (value, (R, G, B, A)) stops for cloud cover — optimised for stargazing.
#: Low cloud → transparent (light-pollution data visible underneath).
#: High cloud → opaque white (obscures the map, signalling "don't go here").
_CLOUD_CMAP_STOPS: list[tuple[float, tuple[int, int, int, int]]] = [
    (0.00, (0, 0, 0, 0)),
    (0.10, (0, 0, 180, 60)),
    (0.30, (0, 100, 255, 90)),
    (0.50, (180, 180, 180, 120)),
    (0.70, (220, 220, 220, 160)),
    (0.90, (255, 255, 255, 200)),
    (1.00, (255, 255, 255, 240)),
]

_PRECIP_CMAP_STOPS: list[tuple[float, tuple[int, int, int, int]]] = [
    (0.00, (0, 0, 0, 0)),
    (0.05, (100, 149, 237, 100)),
    (0.20, (30, 60, 180, 160)),
    (0.50, (0, 0, 255, 200)),
    (1.00, (75, 0, 130, 240)),
]

_WIND_CMAP_STOPS: list[tuple[float, tuple[int, int, int, int]]] = [
    (0.00, (0, 0, 0, 0)),
    (0.10, (50, 205, 50, 80)),
    (0.30, (255, 255, 0, 130)),
    (0.60, (255, 165, 0, 180)),
    (1.00, (255, 0, 0, 220)),
]

# Pre-built LUT caches (lazy-initialised, one per colormap).
_lut_cache: dict[str, np.ndarray] = {}
_lut_lock = threading.Lock()

_DEFAULT_CMAP = _CLOUD_CMAP_STOPS


def _build_lut(stops: list[tuple[float, tuple[int, int, int, int]]]) -> np.ndarray:
    """Build a 256×4 RGBA lookup table from colormap *stops*."""
    # Pre-allocate and compute via vectorised search
    values = np.linspace(0.0, 1.0, 256)
    stop_vals = np.array([s[0] for s in stops], dtype=np.float64)
    stop_cols = np.array([s[1] for s in stops], dtype=np.uint8)

    # For each value, find the right stop interval
    idx = np.searchsorted(stop_vals, values, side='right') - 1
    idx = np.clip(idx, 0, len(stops) - 2)

    t0 = stop_vals[idx]
    t1 = stop_vals[idx + 1]
    # Avoid division by zero for identical stops
    denom = np.where(t1 > t0, t1 - t0, 1.0)
    f = np.where(t1 > t0, (values - t0) / denom, 0.0)

    c0 = stop_cols[idx].astype(np.float64)
    c1 = stop_cols[idx + 1].astype(np.float64)
    f4 = f[:, np.newaxis]
    rgba = (c0 + (c1 - c0) * f4).astype(np.uint8)

    return rgba


def get_colormap(variable: WeatherVariable) -> np.ndarray:
    """Return a 256×4 RGBA lookup table for *variable*.

    Thread-safe; each colormap is built once and cached.
    """
    key = variable.value
    if key not in _lut_cache:
        with _lut_lock:
            if key not in _lut_cache:
                stops = _VARIABLE_CMAPS.get(variable, _DEFAULT_CMAP)
                _lut_cache[key] = _build_lut(stops)
    return _lut_cache[key]


_VARIABLE_CMAPS: dict[WeatherVariable, list] = {
    WeatherVariable.CLOUD_COVER: _CLOUD_CMAP_STOPS,
    WeatherVariable.CLOUD_COVER_LOW: _CLOUD_CMAP_STOPS,
    WeatherVariable.CLOUD_COVER_MID: _CLOUD_CMAP_STOPS,
    WeatherVariable.CLOUD_COVER_HIGH: _CLOUD_CMAP_STOPS,
    WeatherVariable.PRECIPITATION: _PRECIP_CMAP_STOPS,
    WeatherVariable.WIND_SPEED_10M: _WIND_CMAP_STOPS,
}


# ═══════════════════════════════════════════════════════════════════════════
# Per-variable normaliser dispatch (pre-built to avoid branch in hot path)
# ═══════════════════════════════════════════════════════════════════════════

_NORMALISERS: dict[WeatherVariable, Callable[[np.ndarray], np.ndarray]] = {}


def _register(*variables: WeatherVariable):
    """Decorator: register a normaliser function for one or more variables."""

    def deco(fn):
        for v in variables:
            _NORMALISERS[v] = fn
        return fn

    return deco


@_register(
    WeatherVariable.CLOUD_COVER,
    WeatherVariable.CLOUD_COVER_LOW,
    WeatherVariable.CLOUD_COVER_MID,
    WeatherVariable.CLOUD_COVER_HIGH,
)
def _normalise_cloud(data: np.ndarray) -> np.ndarray:
    valid = ~np.isnan(data)
    result = np.where(valid, data, 0.0)
    np.multiply(result, 0.01, out=result)  # / 100 in-place
    np.clip(result, 0.0, 1.0, out=result)
    return result


@_register(WeatherVariable.PRECIPITATION)
def _normalise_precip(data: np.ndarray) -> np.ndarray:
    valid = ~np.isnan(data)
    result = np.where(valid, data, 0.0)
    np.log1p(result, out=result)
    np.multiply(result, 1.0 / np.log1p(50), out=result)
    np.clip(result, 0.0, 1.0, out=result)
    return result


@_register(WeatherVariable.WIND_SPEED_10M)
def _normalise_wind(data: np.ndarray) -> np.ndarray:
    valid = ~np.isnan(data)
    result = np.where(valid, data, 0.0)
    np.multiply(result, 1.0 / 150.0, out=result)
    np.clip(result, 0.0, 1.0, out=result)
    return result


@_register(WeatherVariable.TEMPERATURE_2M)
def _normalise_temperature(data: np.ndarray) -> np.ndarray:
    valid = ~np.isnan(data)
    result = np.where(valid, data, 0.0)
    np.add(result, 40.0, out=result)
    np.multiply(result, 1.0 / 80.0, out=result)
    np.clip(result, 0.0, 1.0, out=result)
    return result


@_register(WeatherVariable.CAPE)
def _normalise_cape(data: np.ndarray) -> np.ndarray:
    valid = ~np.isnan(data)
    result = np.where(valid, data, 0.0)
    np.log1p(result, out=result)
    np.multiply(result, 1.0 / np.log1p(5000), out=result)
    np.clip(result, 0.0, 1.0, out=result)
    return result


def _normalise_for_colormap(data: np.ndarray, variable: WeatherVariable) -> np.ndarray:
    """Dispatch to the registered per-variable normaliser."""
    normaliser = _NORMALISERS.get(variable)
    if normaliser is None:
        # Fallback: treat like cloud (0–100 % linear)
        normaliser = _normalise_cloud
    return normaliser(data)


# ═══════════════════════════════════════════════════════════════════════════
# XYZ tile ↔ geographic coordinate helpers
# ═══════════════════════════════════════════════════════════════════════════


def weather_tile_bounds(z: int, x: int, y: int) -> tuple[float, float, float, float]:
    """Convert XYZ tile coordinates to Web Mercator geographic bounds.

    Returns ``(north, south, east, west)`` in decimal degrees.
    """
    n_tiles = 1 << z
    inv_n = 1.0 / n_tiles
    two_pi = 2.0 * math.pi

    n = math.pi - two_pi * y * inv_n
    north = math.degrees(math.atan(math.sinh(n)))
    south = math.degrees(math.atan(math.sinh(n - two_pi * inv_n)))
    west = x * 360.0 * inv_n - 180.0
    east = (x + 1) * 360.0 * inv_n - 180.0
    return north, south, east, west


# ═══════════════════════════════════════════════════════════════════════════
# Weather tile rendering
# ═══════════════════════════════════════════════════════════════════════════


def render_weather_tile(
    data: np.ndarray,
    variable: WeatherVariable,
) -> bytes:
    """Render a 2-D NumPy weather-data array as an RGBA PNG tile.

    Args:
        data: 2-D float32 array of weather values.
        variable: Weather variable, which determines the colormap and
            normalisation.

    Returns:
        PNG-encoded bytes.
    """
    if data.ndim != 2:
        raise ValueError(f'Expected 2-D array, got {data.ndim}-D')

    lut = get_colormap(variable)

    # ── single-pass: normalise → LUT index → RGBA ──
    valid = ~np.isnan(data)
    normalised = _normalise_for_colormap(data, variable)  # reuses valid internally (sigh)
    # TODO: merge valid computation into normalisers to avoid double isnan

    idx = np.clip((normalised * 255.0).astype(np.uint8), 0, 255)
    rgba = lut[idx]  # view — no copy needed since we overwrite NaN pixels next
    rgba[~valid] = (0, 0, 0, 0)

    # Fast PNG encode: compress_level=1 for throughput over size
    img = Image.fromarray(rgba, 'RGBA')
    buf = BytesIO()
    img.save(buf, format='PNG', compress_level=_PNG_COMPRESS_LEVEL)
    return buf.getvalue()


# ═══════════════════════════════════════════════════════════════════════════
# Tile resampling (PIL NEAREST — C-optimised, beats NumPy indexing at tile sizes)
# ═══════════════════════════════════════════════════════════════════════════


def _resample_2d(data: np.ndarray, valid: np.ndarray, out_shape: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbour resample of *data* with NaN-mask preservation."""
    out_w, out_h = out_shape[1], out_shape[0]
    data_filled = np.where(valid, data, 0.0)
    img = Image.fromarray(data_filled.astype(np.float32))
    img = img.resize((out_w, out_h), Image.Resampling.NEAREST)
    resampled = np.array(img, dtype=np.float32)
    mask_img = Image.fromarray(valid.astype(np.float32))
    mask_img = mask_img.resize((out_w, out_h), Image.Resampling.NEAREST)
    mask_arr = np.array(mask_img, dtype=np.float32) > 0.5
    resampled[~mask_arr] = np.nan
    return resampled


# ═══════════════════════════════════════════════════════════════════════════
# OM Weather Reader
# ═══════════════════════════════════════════════════════════════════════════


class OmWeatherReader:
    """Read spatial weather data from Open-Meteo OM files.

    Wraps ``omfiles`` and ``fsspec`` to provide window-based access to
    weather variables from the Open-Meteo spatial data S3 bucket.

    Uses the `Capture API`_ (``latest.json``) to auto-resolve the latest
    model run when *time* is ``None``.  OM files are read via HTTPS with
    ``fsspec``, which enables byte-range requests — only the chunks that
    cover the requested geographic window are downloaded and decompressed.

    Thread-safe: all reading methods may be called from any thread (e.g.
    wrapped in :func:`asyncio.to_thread`).

    Typical usage::

        reader = OmWeatherReader(WeatherModel.DWD_ICON)
        data = reader.read_window(
            WeatherVariable.CLOUD_COVER,
            north=35.0, south=30.0, east=120.0, west=115.0,
        )
        # → np.ndarray (float32, shape ≈ (50, 50))

    .. _Capture API: https://map-tiles.open-meteo.com/data_spatial/
    """

    __slots__ = (
        '_model',
        '_timeout',
        '_reference_time',
        '_valid_times',
        '_available_variables',
        '_completed',
        '_run_path',
        '_cache_dir',
    )

    def __init__(
        self,
        model: WeatherModel,
        time: str | None = None,
        *,
        timeout: float = _METADATA_FETCH_TIMEOUT,
        cache_dir: str | None = None,
    ) -> None:
        """Initialise a reader for *model* at *time*.

        Args:
            model: Weather model to query.
            time: ISO-8601 datetime string (e.g. ``"2026-07-09T06:00:00Z"``).
                When ``None``, the Capture API resolves the latest run.
            timeout: HTTP request timeout in seconds.

        Raises:
            OSError: If the metadata fetch fails.
            ValueError: If the model has no completed data available.
        """
        self._model = model
        self._timeout = timeout
        _ = time  # reserved for explicit-time lookup (future)
        self._cache_dir = cache_dir

        metadata = self._fetch_metadata()
        self._reference_time: str = metadata['reference_time']
        self._valid_times: list[str] = list(metadata.get('valid_times', []))
        self._available_variables: list[str] = list(metadata.get('variables', []))
        self._completed: bool = bool(metadata.get('completed', False))

        if not self._completed:
            raise ValueError(
                f'Model {model.value} has no completed data available '
                f'(reference_time={self._reference_time})'
            )

        self._run_path = self._reference_time_to_run_path(self._reference_time)

    # ── public properties ──────────────────────────────────────────────

    @property
    def model(self) -> WeatherModel:
        """The weather model this reader is configured for."""
        return self._model

    @property
    def reference_time(self) -> str:
        """Model initialisation time (ISO-8601)."""
        return self._reference_time

    @property
    def valid_times(self) -> list[str]:
        """Sorted list of forecast valid times (ISO-8601)."""
        return self._valid_times

    @property
    def available_variables(self) -> list[str]:
        """Variable names available in this model run."""
        return self._available_variables

    # ── data access ────────────────────────────────────────────────────

    def read_window(
        self,
        variable: WeatherVariable,
        north: float,
        south: float,
        east: float,
        west: float,
        *,
        shape: tuple[int, int] | None = None,
        valid_time_index: int = 0,
    ) -> np.ndarray:
        """Read a geographic window of weather data.

        Args:
            variable: Weather variable to extract.
            north, south, east, west: Geographic bounding box (degrees).
            shape: If given, resample output to ``(height, width)`` via
                nearest-neighbour.  When ``None`` return at native resolution.
            valid_time_index: Index into ``self.valid_times``.

        Returns:
            2-D ``float32`` NumPy array.  Out-of-domain cells are ``NaN``.

        Raises:
            ValueError: Invalid variable or time index.
            OSError: Network error fetching the OM file.
        """
        if variable.value not in self._available_variables:
            raise ValueError(
                f'Variable "{variable.value}" not available in {self._model.value}. '
                f'Available: {self._available_variables}'
            )
        if valid_time_index < 0 or valid_time_index >= len(self._valid_times):
            raise ValueError(
                f'valid_time_index {valid_time_index} out of range '
                f'[0, {len(self._valid_times) - 1}]'
            )

        s3_uri = self._build_s3_uri(valid_time_index)

        reader = self._open_om_file(s3_uri)

        try:
            child = reader.get_child_by_name(variable.value)
        except Exception as exc:
            raise ValueError(
                f'Variable "{variable.value}" not found in OM file at {s3_uri}'
            ) from exc

        if not child.is_array:
            raise ValueError(
                f'Variable "{variable.value}" is not an array (type: {type(child).__name__})'
            )

        grid_shape = child.shape
        if len(grid_shape) != 2:
            raise ValueError(f'Expected 2-D grid, got shape {grid_shape}')

        # ── Map lat/lon → grid indices ──
        rows, cols = grid_shape
        inv_cell_lat = rows / 180.0
        inv_cell_lon = cols / 360.0

        # Normalise and order lat bounds
        n = max(-90.0, min(90.0, north))
        s = max(-90.0, min(90.0, south))
        if n < s:
            n, s = s, n

        r_top = int((90.0 - n) * inv_cell_lat)
        r_bot = int((90.0 - s) * inv_cell_lat)
        r_top = max(0, r_top)
        r_bot = min(rows, max(r_top + 1, r_bot))

        # Normalise lon to 0–360; detect meridian crossing
        w_norm = west % 360.0
        e_norm = east % 360.0
        crosses_zero = w_norm > e_norm

        if crosses_zero:
            # Window wraps across 0°: read two strips and concatenate.
            c_left = int(w_norm * inv_cell_lon)
            c_right = int(e_norm * inv_cell_lon)

            c_left = max(0, c_left)
            c_right = min(cols, max(c_left + 1, c_right))

            try:
                data_left = child.read_array((slice(r_top, r_bot), slice(c_left, cols)))
                data_right = child.read_array((slice(r_top, r_bot), slice(0, c_right)))
                data = np.hstack((data_left, data_right))
            except Exception as exc:
                raise OSError(f'Failed to read lon-wrapped window from {s3_uri}: {exc}') from exc
        else:
            c_w = int(w_norm * inv_cell_lon)
            c_e = int(e_norm * inv_cell_lon)

            c_w = max(0, c_w)
            c_e = min(cols, max(c_w + 1, c_e))

            try:
                data: np.ndarray = child.read_array((slice(r_top, r_bot), slice(c_w, c_e)))
            except Exception as exc:
                raise OSError(f'Failed to read data window from {s3_uri}: {exc}') from exc

        # ── Optional resample ──
        if shape is not None:
            valid_mask = ~np.isnan(data)
            data = _resample_2d(data, valid_mask, shape)
            return data

        return data

    def read_point(
        self,
        variable: WeatherVariable,
        lat: float,
        lon: float,
        *,
        valid_time_index: int = 0,
    ) -> float:
        """Read the weather value at a single geographic point."""
        delta = 0.05
        data = self.read_window(
            variable,
            north=lat + delta,
            south=lat - delta,
            east=lon + delta,
            west=lon - delta,
            valid_time_index=valid_time_index,
        )
        if data.size == 0:
            return float('nan')
        valid = data[~np.isnan(data)]
        if valid.size == 0:
            return float('nan')
        return float(valid.mean())

    # ── internal helpers ───────────────────────────────────────────────

    def _open_om_file(self, s3_uri: str):
        """Open the OM file via S3 + blockcache for byte-range random access.

        Uses the public Open-Meteo S3 bucket (``s3://openmeteo/``) with
        anonymous access.  The ``blockcache`` wrapper caches fetched blocks
        on disk, so repeated reads of the same file avoid re-downloading
        metadata and chunk data.

        Falls back to HTTPS + local cache if ``s3fs`` is not available.
        """
        cache_dir = self._cache_dir or _DEFAULT_BLOCK_CACHE

        try:
            backend = fsspec.open(
                f'blockcache::{s3_uri}',
                mode='rb',
                s3={'anon': True, 'default_block_size': 65536},
                blockcache={'cache_storage': cache_dir},
            )
            return _OmFileReader(backend)
        except (ImportError, ModuleNotFoundError):
            # s3fs not available — fall back to HTTPS download
            om_url = s3_uri.replace(_S3_BASE, _SPATIAL_BASE_URL)
            import hashlib
            import os
            import urllib.request as _urllib

            cache_path = os.path.join(
                cache_dir,
                hashlib.md5(om_url.encode(), usedforsecurity=False).hexdigest() + '.om',
            )
            os.makedirs(cache_dir, exist_ok=True)

            if not os.path.exists(cache_path):
                _urllib.urlretrieve(om_url, cache_path)  # nosec B310 — only https:// URLs

            return _OmFileReader.from_path(cache_path)

    @staticmethod
    def _reference_time_to_run_path(reference_time: str) -> str:
        """Convert ``"2026-07-09T00:00:00Z"`` → ``"2026/07/09/0000Z"``."""
        # reference_time is ISO-8601: YYYY-MM-DDTHH:MM:SSZ
        date_part, time_part = reference_time.replace('Z', '').split('T')
        year, month, day = date_part.split('-')
        hour = time_part.split(':')[0]
        return f'{year}/{month}/{day}/{hour}00Z'

    @staticmethod
    def _valid_time_to_filename(valid_time: str) -> str:
        """Convert ``"2026-07-09T06:00Z"`` → ``"2026-07-09T0600"``."""
        # Handles both "HH:MMZ" and "HH:MM:SSZ" formats.
        date_str, time_str = valid_time.rstrip('Z').split('T')
        parts = time_str.split(':')
        hhmm = parts[0] + parts[1]  # "HH" + "MM"
        return f'{date_str}T{hhmm}'

    def _build_s3_uri(self, valid_time_index: int) -> str:
        """Construct the S3 URI for the OM file at a forecast step."""
        fn = self._valid_time_to_filename(self._valid_times[valid_time_index])
        return f'{_S3_BASE}/{self._model.value}/{self._run_path}/{fn}.om'

    def _build_om_url(self, valid_time_index: int) -> str:
        """Construct the HTTPS URL for the OM file at a forecast step.

        .. deprecated:: 0.2.0
            Use :meth:`_build_s3_uri` for byte-range S3 access instead.
        """
        fn = self._valid_time_to_filename(self._valid_times[valid_time_index])
        return f'{_SPATIAL_BASE_URL}/{self._model.value}/{self._run_path}/{fn}.om'

    def _fetch_metadata(self) -> dict:
        """Fetch model metadata from the Capture API."""
        url = f'{_SPATIAL_BASE_URL}/{self._model.value}/latest.json'
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                body = resp.read()
        except urllib.error.URLError as exc:
            raise OSError(f'Failed to fetch metadata from {url}: {exc}') from exc
        except urllib.error.HTTPError as exc:
            raise OSError(f'HTTP {exc.code} fetching metadata from {url}') from exc

        try:
            metadata = json.loads(body)
        except json.JSONDecodeError as exc:
            raise OSError(f'Invalid JSON from {url}: {exc}') from exc

        if not isinstance(metadata, dict):
            raise OSError(f'Unexpected metadata format from {url}: {type(metadata).__name__}')

        return metadata


# ═══════════════════════════════════════════════════════════════════════════
# Tile cache
# ═══════════════════════════════════════════════════════════════════════════


class WeatherTileCache:
    """Thread-safe LRU in-memory cache for rendered weather tiles.

    Typical usage::

        cache = WeatherTileCache(max_items=500, ttl=900)  # 15-min TTL
        png = cache.get('dwd_icon/cloud_cover/latest/12/3456/2345')
        if png is None:
            png = render_weather_tile(data, variable)
            cache.set('dwd_icon/cloud_cover/latest/12/3456/2345', png)
    """

    __slots__ = ('_max_items', '_ttl', '_store', '_lock')

    def __init__(self, max_items: int = 500, ttl: float = 900.0) -> None:
        if max_items < 1:
            raise ValueError('max_items must be >= 1')
        if ttl <= 0:
            raise ValueError('ttl must be > 0')
        self._max_items = max_items
        self._ttl = ttl
        self._store: OrderedDict[str, tuple[bytes, float]] = OrderedDict()
        self._lock = threading.RLock()

    def get(self, key: str) -> bytes | None:
        """Return cached PNG bytes for *key*, or ``None`` on miss/expiry."""
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            data, created_at = entry
            if time.time() - created_at < self._ttl:
                self._store.move_to_end(key)
                return data
            del self._store[key]
        return None

    def set(self, key: str, data: bytes) -> None:
        """Store *data* under *key*, evicting oldest entries if full."""
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
            else:
                while len(self._store) >= self._max_items:
                    self._store.popitem(last=False)
            self._store[key] = (data, time.time())

    def clear(self) -> None:
        """Remove all cached entries."""
        with self._lock:
            self._store.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)

    def __contains__(self, key: str) -> bool:
        return self.get(key) is not None
