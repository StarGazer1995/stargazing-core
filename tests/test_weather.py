"""Tests for stargazing_core._weather — weather model, tile rendering, cache, reader."""

from __future__ import annotations

import json
import math
import threading
import time
import urllib.request
from io import BytesIO
from unittest import mock

import numpy as np
import pytest
from PIL import Image

from stargazing_core._weather import (
    _CLOUD_CMAP_STOPS,
    _SPATIAL_BASE_URL,
    WeatherModel,
    WeatherTileCache,
    WeatherVariable,
    _build_lut,
    _normalise_for_colormap,
    get_colormap,
    render_weather_tile,
    weather_tile_bounds,
)
from stargazing_core._weather import OmWeatherReader as _OmWeatherReader

# ═══════════════════════════════════════════════════════════════════════════
# Enums
# ═══════════════════════════════════════════════════════════════════════════


class TestWeatherModel:
    def test_all_models_have_values(self):
        models = list(WeatherModel)
        assert len(models) == 5
        for m in models:
            assert isinstance(m.value, str)
            assert len(m.value) > 0

    def test_model_string_equality(self):
        assert WeatherModel.DWD_ICON == 'dwd_icon'
        assert WeatherModel.ECMWF_IFS025 == 'ecmwf_ifs025'

    def test_model_from_string(self):
        assert WeatherModel('dwd_icon') == WeatherModel.DWD_ICON


class TestWeatherVariable:
    def test_all_variables_have_values(self):
        variables = list(WeatherVariable)
        assert len(variables) == 8
        for v in variables:
            assert isinstance(v.value, str)
            assert len(v.value) > 0

    def test_variable_string_equality(self):
        assert WeatherVariable.CLOUD_COVER == 'cloud_cover'
        assert WeatherVariable.PRECIPITATION == 'precipitation'

    def test_variable_from_string(self):
        assert WeatherVariable('temperature_2m') == WeatherVariable.TEMPERATURE_2M


# ═══════════════════════════════════════════════════════════════════════════
# weather_tile_bounds
# ═══════════════════════════════════════════════════════════════════════════


class TestWeatherTileBounds:
    def test_zoom_0_single_tile_covers_world(self):
        north, south, east, west = weather_tile_bounds(0, 0, 0)
        assert north == pytest.approx(85.0511, rel=1e-4)
        assert south == pytest.approx(-85.0511, rel=1e-4)
        assert east == pytest.approx(180.0, rel=1e-4)
        assert west == pytest.approx(-180.0, rel=1e-4)

    def test_zoom_1_top_left(self):
        north, south, east, west = weather_tile_bounds(1, 0, 0)
        assert north == pytest.approx(85.0511, rel=1e-4)
        assert south == pytest.approx(0.0, abs=0.01)
        assert east == pytest.approx(0.0, abs=0.01)
        assert west == pytest.approx(-180.0, rel=1e-4)

    def test_higher_zoom_smaller_bounds(self):
        n1, s1, e1, w1 = weather_tile_bounds(1, 0, 0)
        n2, s2, e2, w2 = weather_tile_bounds(2, 0, 0)
        span1 = n1 - s1
        span2 = n2 - s2
        assert span2 < span1

    def test_bounds_are_ordered(self):
        north, south, east, west = weather_tile_bounds(5, 10, 10)
        assert north > south
        assert east > west

    def test_lon_wraps_correctly(self):
        z = 3
        max_x = (1 << z) - 1
        _, _, east, west = weather_tile_bounds(z, max_x, 0)
        assert east == pytest.approx(180.0, rel=1e-4)
        assert west < east


# ═══════════════════════════════════════════════════════════════════════════
# Colormap
# ═══════════════════════════════════════════════════════════════════════════


class TestBuildLut:
    def test_output_shape(self):
        lut = _build_lut(_CLOUD_CMAP_STOPS)
        assert lut.shape == (256, 4)
        assert lut.dtype == np.uint8

    def test_first_entry_is_first_stop_color(self):
        lut = _build_lut(_CLOUD_CMAP_STOPS)
        np.testing.assert_array_equal(lut[0], _CLOUD_CMAP_STOPS[0][1])

    def test_last_entry_is_last_stop_color(self):
        lut = _build_lut(_CLOUD_CMAP_STOPS)
        np.testing.assert_array_equal(lut[255], _CLOUD_CMAP_STOPS[-1][1])

    def test_middle_entry_is_interpolated(self):
        lut = _build_lut(_CLOUD_CMAP_STOPS)
        idx = 127
        r, g, b, a = lut[idx]
        assert g > 0
        assert a > 0


class TestGetColormap:
    def test_cloud_returns_rgba(self):
        lut = get_colormap(WeatherVariable.CLOUD_COVER)
        assert lut.shape == (256, 4)

    def test_precip_returns_rgba(self):
        lut = get_colormap(WeatherVariable.PRECIPITATION)
        assert lut.shape == (256, 4)

    def test_wind_returns_rgba(self):
        lut = get_colormap(WeatherVariable.WIND_SPEED_10M)
        assert lut.shape == (256, 4)

    def test_cloud_subtypes_use_cloud_cmap(self):
        cloud_lut = get_colormap(WeatherVariable.CLOUD_COVER)
        low_lut = get_colormap(WeatherVariable.CLOUD_COVER_LOW)
        mid_lut = get_colormap(WeatherVariable.CLOUD_COVER_MID)
        high_lut = get_colormap(WeatherVariable.CLOUD_COVER_HIGH)
        np.testing.assert_array_equal(low_lut, cloud_lut)
        np.testing.assert_array_equal(mid_lut, cloud_lut)
        np.testing.assert_array_equal(high_lut, cloud_lut)

    def test_different_variables_have_different_colormaps(self):
        cloud_lut = get_colormap(WeatherVariable.CLOUD_COVER)
        precip_lut = get_colormap(WeatherVariable.PRECIPITATION)
        assert not np.array_equal(cloud_lut, precip_lut)

    def test_unregistered_variable_gets_default(self):
        cape_lut = get_colormap(WeatherVariable.CAPE)
        assert cape_lut.shape == (256, 4)

    def test_colormap_alpha_zero_for_clear(self):
        lut = get_colormap(WeatherVariable.CLOUD_COVER)
        assert lut[0][3] == 0

    def test_colormap_alpha_high_for_overcast(self):
        lut = get_colormap(WeatherVariable.CLOUD_COVER)
        assert lut[255][3] > 200


# ═══════════════════════════════════════════════════════════════════════════
# Normalise for colormap
# ═══════════════════════════════════════════════════════════════════════════


class TestNormaliseForColormap:
    def test_cloud_0_normalises_to_0(self):
        data = np.array([[0.0]], dtype=np.float32)
        result, _valid = _normalise_for_colormap(data, WeatherVariable.CLOUD_COVER)
        assert result[0, 0] == 0.0

    def test_cloud_100_normalises_to_1(self):
        data = np.array([[100.0]], dtype=np.float32)
        result, _valid = _normalise_for_colormap(data, WeatherVariable.CLOUD_COVER)
        assert result[0, 0] == 1.0

    def test_cloud_50_normalises_to_0_5(self):
        data = np.array([[50.0]], dtype=np.float32)
        result, _valid = _normalise_for_colormap(data, WeatherVariable.CLOUD_COVER)
        assert result[0, 0] == 0.5

    def test_nan_remains_0(self):
        data = np.array([[np.nan]], dtype=np.float32)
        result, valid = _normalise_for_colormap(data, WeatherVariable.CLOUD_COVER)
        assert result[0, 0] == 0.0
        assert not valid[0, 0]  # NaN → invalid mask

    def test_precip_uses_log_scale(self):
        data = np.array([[50.0]], dtype=np.float32)
        result, _valid = _normalise_for_colormap(data, WeatherVariable.PRECIPITATION)
        assert 0.0 < result[0, 0] <= 1.0

    def test_wind_linear(self):
        data = np.array([[75.0]], dtype=np.float32)
        result, _valid = _normalise_for_colormap(data, WeatherVariable.WIND_SPEED_10M)
        assert result[0, 0] == 0.5

    def test_temperature_maps_range(self):
        data = np.array([[0.0]], dtype=np.float32)
        result, _valid = _normalise_for_colormap(data, WeatherVariable.TEMPERATURE_2M)
        assert result[0, 0] == 0.5

    def test_unregistered_normaliser_fallback(self):
        """A variable with no registered normaliser falls back to cloud."""
        from stargazing_core._weather import _NORMALISERS

        saved = _NORMALISERS.pop(WeatherVariable.CAPE, None)
        try:
            data = np.array([[500.0]], dtype=np.float32)
            result, _valid = _normalise_for_colormap(data, WeatherVariable.CAPE)
            # CAPE 500 J/kg → fallback treats as cloud % → 500/100 clipped to 1.0
            assert result[0, 0] == 1.0
        finally:
            if saved is not None:
                _NORMALISERS[WeatherVariable.CAPE] = saved


# ═══════════════════════════════════════════════════════════════════════════
# render_weather_tile
# ═══════════════════════════════════════════════════════════════════════════


class TestRenderWeatherTile:
    def test_returns_bytes(self):
        data = np.full((64, 64), 50.0, dtype=np.float32)
        png = render_weather_tile(data, WeatherVariable.CLOUD_COVER)
        assert isinstance(png, bytes)
        assert len(png) > 0

    def test_output_is_valid_png(self):
        data = np.full((64, 64), 30.0, dtype=np.float32)
        png = render_weather_tile(data, WeatherVariable.CLOUD_COVER)
        img = Image.open(BytesIO(png))
        assert img.format == 'PNG'
        assert img.size == (64, 64)
        assert img.mode == 'RGBA'

    def test_full_clear_is_transparent(self):
        data = np.zeros((64, 64), dtype=np.float32)
        png = render_weather_tile(data, WeatherVariable.CLOUD_COVER)
        img = Image.open(BytesIO(png))
        arr = np.array(img)
        assert np.mean(arr[:, :, 3]) < 10

    def test_full_overcast_is_opaque(self):
        data = np.full((64, 64), 100.0, dtype=np.float32)
        png = render_weather_tile(data, WeatherVariable.CLOUD_COVER)
        img = Image.open(BytesIO(png))
        arr = np.array(img)
        assert np.mean(arr[:, :, 3]) > 200

    def test_precipitation_renders(self):
        data = np.full((64, 64), 10.0, dtype=np.float32)
        png = render_weather_tile(data, WeatherVariable.PRECIPITATION)
        assert isinstance(png, bytes)
        assert len(png) > 0

    def test_wind_renders(self):
        data = np.full((64, 64), 50.0, dtype=np.float32)
        png = render_weather_tile(data, WeatherVariable.WIND_SPEED_10M)
        assert isinstance(png, bytes)
        assert len(png) > 0

    def test_temperature_renders(self):
        data = np.full((64, 64), 20.0, dtype=np.float32)
        png = render_weather_tile(data, WeatherVariable.TEMPERATURE_2M)
        assert isinstance(png, bytes)

    def test_cape_renders(self):
        data = np.full((64, 64), 1000.0, dtype=np.float32)
        png = render_weather_tile(data, WeatherVariable.CAPE)
        assert isinstance(png, bytes)

    def test_nan_renders_as_transparent(self):
        data = np.full((64, 64), np.nan, dtype=np.float32)
        png = render_weather_tile(data, WeatherVariable.CLOUD_COVER)
        img = Image.open(BytesIO(png))
        arr = np.array(img)
        assert np.all(arr[:, :, 3] == 0)

    def test_3d_array_raises(self):
        data = np.zeros((64, 64, 3), dtype=np.float32)
        with pytest.raises(ValueError, match='Expected 2-D'):
            render_weather_tile(data, WeatherVariable.CLOUD_COVER)

    def test_mixed_nan_and_value(self):
        data = np.full((64, 64), 50.0, dtype=np.float32)
        data[0:10, 0:10] = np.nan
        png = render_weather_tile(data, WeatherVariable.CLOUD_COVER)
        img = Image.open(BytesIO(png))
        arr = np.array(img)
        assert np.all(arr[0:10, 0:10, 3] == 0)
        assert np.mean(arr[30:40, 30:40, 3]) > 0

    def test_output_consistent_size(self):
        data = np.full((32, 32), 50.0, dtype=np.float32)
        png1 = render_weather_tile(data, WeatherVariable.CLOUD_COVER)
        png2 = render_weather_tile(data, WeatherVariable.CLOUD_COVER)
        assert png1 == png2


# ═══════════════════════════════════════════════════════════════════════════
# WeatherTileCache
# ═══════════════════════════════════════════════════════════════════════════


class TestWeatherTileCache:
    def test_init_defaults(self):
        cache = WeatherTileCache()
        assert len(cache) == 0

    def test_init_custom_params(self):
        cache = WeatherTileCache(max_items=100, ttl=30)
        assert len(cache) == 0

    def test_init_invalid_max_items(self):
        with pytest.raises(ValueError, match='max_items'):
            WeatherTileCache(max_items=0)

    def test_init_invalid_ttl(self):
        with pytest.raises(ValueError, match='ttl'):
            WeatherTileCache(ttl=0)

    def test_set_and_get(self):
        cache = WeatherTileCache()
        cache.set('key1', b'data1')
        assert cache.get('key1') == b'data1'

    def test_get_miss(self):
        cache = WeatherTileCache()
        assert cache.get('nonexistent') is None

    def test_contains(self):
        cache = WeatherTileCache()
        cache.set('k', b'v')
        assert 'k' in cache
        assert 'nonexistent' not in cache

    def test_set_overwrites(self):
        cache = WeatherTileCache()
        cache.set('k', b'old')
        cache.set('k', b'new')
        assert cache.get('k') == b'new'
        assert len(cache) == 1

    def test_set_moves_to_end(self):
        cache = WeatherTileCache(max_items=3, ttl=60)
        cache.set('a', b'a')
        cache.set('b', b'b')
        cache.set('c', b'c')
        cache.get('a')
        cache.set('d', b'd')
        assert cache.get('b') is None
        assert cache.get('a') == b'a'
        assert cache.get('c') == b'c'
        assert cache.get('d') == b'd'

    def test_ttl_expiry(self):
        cache = WeatherTileCache(ttl=0.01)
        cache.set('k', b'data')
        assert cache.get('k') == b'data'
        time.sleep(0.02)
        assert cache.get('k') is None

    def test_lru_eviction(self):
        cache = WeatherTileCache(max_items=2, ttl=3600)
        cache.set('a', b'a')
        cache.set('b', b'b')
        cache.set('c', b'c')
        assert cache.get('a') is None
        assert cache.get('b') == b'b'
        assert cache.get('c') == b'c'
        assert len(cache) == 2

    def test_clear(self):
        cache = WeatherTileCache()
        cache.set('a', b'a')
        cache.set('b', b'b')
        cache.clear()
        assert len(cache) == 0
        assert cache.get('a') is None

    def test_len(self):
        cache = WeatherTileCache()
        assert len(cache) == 0
        cache.set('a', b'a')
        assert len(cache) == 1
        cache.set('b', b'b')
        assert len(cache) == 2

    def test_thread_safety_write(self):
        cache = WeatherTileCache(max_items=200, ttl=3600)
        errors = []

        def writer(start: int):
            try:
                for i in range(start, start + 50):
                    cache.set(f'k{i}', f'data{i}'.encode())
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(i * 50,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(cache) <= 200

    def test_thread_safety_read_write(self):
        cache = WeatherTileCache(max_items=500, ttl=3600)
        for i in range(200):
            cache.set(f'k{i}', f'data{i}'.encode())

        errors = []

        def worker():
            try:
                for i in range(200):
                    _ = cache.get(f'k{i}')
                    cache.set(f'w{i}', b'worker')
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


# ═══════════════════════════════════════════════════════════════════════════
# OmWeatherReader (with mock)
# ═══════════════════════════════════════════════════════════════════════════

_MOCK_METADATA = {
    'completed': True,
    'reference_time': '2026-07-09T00:00:00Z',
    'valid_times': [
        '2026-07-09T00:00Z',
        '2026-07-09T01:00Z',
        '2026-07-09T02:00Z',
    ],
    'variables': [
        'cloud_cover',
        'cloud_cover_low',
        'precipitation',
        'temperature_2m',
    ],
}


def _mock_urlopen_success():
    """Return a mock response for the Capture API."""
    resp = mock.MagicMock()
    resp.read.return_value = json.dumps(_MOCK_METADATA).encode()
    resp.__enter__.return_value = resp
    resp.__exit__.return_value = False
    return resp


def _make_mock_om_child(is_array=True, shape=(1801, 3600), read_array_data=None):
    """Create a mock OM variable child."""
    child = mock.MagicMock()
    child.is_array = is_array
    child.shape = shape
    if read_array_data is not None:
        child.read_array.return_value = read_array_data
    return child


def _make_mock_om_reader(children=None):
    """Create a mock OM file reader with optional children dict.

    *children* is a dict mapping variable name → MagicMock child,
    or variable name → exception to raise.
    """
    r = mock.MagicMock()

    if children:

        def _get_child(name):
            result = children[name]
            if isinstance(result, Exception):
                raise result
            return result

        r.get_child_by_name.side_effect = _get_child
    else:
        r.get_child_by_name.return_value = _make_mock_om_child(
            read_array_data=np.full((10, 10), 25.0, dtype=np.float32)
        )
    r.__enter__.return_value = r
    r.__exit__.return_value = False
    return r


class TestOmWeatherReaderMetadata:
    def test_fetch_metadata_success(self):
        with mock.patch('urllib.request.urlopen', return_value=_mock_urlopen_success()):
            reader = _OmWeatherReader(WeatherModel.DWD_ICON)
        assert reader.reference_time == '2026-07-09T00:00:00Z'
        assert len(reader.valid_times) == 3
        assert reader.available_variables == [
            'cloud_cover',
            'cloud_cover_low',
            'precipitation',
            'temperature_2m',
        ]

    def test_model_property(self):
        with mock.patch('urllib.request.urlopen', return_value=_mock_urlopen_success()):
            reader = _OmWeatherReader(WeatherModel.DWD_ICON)
        assert reader.model == WeatherModel.DWD_ICON

    def test_different_models(self):
        with mock.patch('urllib.request.urlopen', return_value=_mock_urlopen_success()):
            reader = _OmWeatherReader(WeatherModel.ECMWF_IFS025)
        assert reader.model == WeatherModel.ECMWF_IFS025

    def test_url_contains_model(self):
        captured_urls = []

        def side_effect(req, timeout=None):
            captured_urls.append(req.full_url if hasattr(req, 'full_url') else str(req))
            return _mock_urlopen_success()

        with mock.patch('urllib.request.urlopen', side_effect=side_effect):
            _OmWeatherReader(WeatherModel.DWD_ICON_D2)
        assert 'dwd_icon_d2' in captured_urls[0]
        assert captured_urls[0].endswith('/latest.json')

    def test_not_completed_raises(self):
        meta = dict(_MOCK_METADATA, completed=False)
        resp = mock.MagicMock()
        resp.read.return_value = json.dumps(meta).encode()
        resp.__enter__.return_value = resp
        resp.__exit__.return_value = False

        with (
            mock.patch('urllib.request.urlopen', return_value=resp),
            pytest.raises(ValueError, match='no completed data'),
        ):
            _OmWeatherReader(WeatherModel.DWD_ICON)

    def test_network_error_raises_oserror(self):
        with (
            mock.patch(
                'urllib.request.urlopen',
                side_effect=urllib.error.URLError('connection refused'),
            ),
            pytest.raises(OSError, match='Failed to fetch metadata'),
        ):
            _OmWeatherReader(WeatherModel.DWD_ICON)

    def test_http_error_raises_oserror(self):
        with (
            mock.patch(
                'urllib.request.urlopen',
                side_effect=urllib.error.HTTPError('http://fake', 500, 'Internal Error', {}, None),
            ),
            pytest.raises(OSError, match=r'HTTP.*500'),
        ):
            _OmWeatherReader(WeatherModel.DWD_ICON)

    def test_invalid_json_raises_oserror(self):
        resp = mock.MagicMock()
        resp.read.return_value = b'not json'
        resp.__enter__.return_value = resp
        resp.__exit__.return_value = False

        with (
            mock.patch('urllib.request.urlopen', return_value=resp),
            pytest.raises(OSError, match='Invalid JSON'),
        ):
            _OmWeatherReader(WeatherModel.DWD_ICON)

    def test_non_dict_metadata_raises_oserror(self):
        resp = mock.MagicMock()
        resp.read.return_value = b'[]'
        resp.__enter__.return_value = resp
        resp.__exit__.return_value = False

        with (
            mock.patch('urllib.request.urlopen', return_value=resp),
            pytest.raises(OSError, match='Unexpected metadata format'),
        ):
            _OmWeatherReader(WeatherModel.DWD_ICON)

    def test_run_path_format(self):
        result = _OmWeatherReader._reference_time_to_run_path('2026-07-09T06:00:00Z')
        assert result == '2026/07/09/0600Z'

    def test_run_path_midnight(self):
        result = _OmWeatherReader._reference_time_to_run_path('2026-07-09T00:00:00Z')
        assert result == '2026/07/09/0000Z'


class TestOmWeatherReaderBuildUrl:
    def test_build_om_url(self):
        with mock.patch('urllib.request.urlopen', return_value=_mock_urlopen_success()):
            reader = _OmWeatherReader(WeatherModel.DWD_ICON)
        url = reader._build_om_url(0)
        assert url.startswith(_SPATIAL_BASE_URL)
        assert '/dwd_icon/' in url
        assert url.endswith('.om')
        assert '2026/07/09/0000Z' in url

    def test_build_om_url_different_time_step(self):
        with mock.patch('urllib.request.urlopen', return_value=_mock_urlopen_success()):
            reader = _OmWeatherReader(WeatherModel.DWD_ICON)
        url1 = reader._build_om_url(0)
        url2 = reader._build_om_url(1)
        assert url1 != url2


class TestOmWeatherReaderReadWindow:
    @staticmethod
    def _make_reader():
        with mock.patch('urllib.request.urlopen', return_value=_mock_urlopen_success()):
            return _OmWeatherReader(WeatherModel.DWD_ICON)

    def test_invalid_variable_raises(self):
        reader = self._make_reader()
        with pytest.raises(ValueError, match='not available'):
            reader.read_window(
                WeatherVariable.WIND_SPEED_10M,
                north=50,
                south=40,
                east=10,
                west=0,
            )

    def test_invalid_time_index_raises(self):
        reader = self._make_reader()
        with pytest.raises(ValueError, match='out of range'):
            reader.read_window(
                WeatherVariable.CLOUD_COVER,
                north=50,
                south=40,
                east=10,
                west=0,
                valid_time_index=99,
            )

    def test_read_window_opens_correct_url(self):
        reader = self._make_reader()
        mock_om = _make_mock_om_reader()

        with (
            mock.patch('fsspec.open', return_value=mock_om),
            mock.patch('stargazing_core._weather._OmFileReader', return_value=mock_om),
        ):
            data = reader.read_window(
                WeatherVariable.CLOUD_COVER,
                north=50,
                south=40,
                east=10,
                west=0,
            )
        assert isinstance(data, np.ndarray)

    def test_read_window_with_shape_resamples(self):
        reader = self._make_reader()
        mock_om = _make_mock_om_reader(
            children={
                'cloud_cover': _make_mock_om_child(
                    read_array_data=np.full((50, 50), 60.0, dtype=np.float32)
                )
            }
        )

        with (
            mock.patch('fsspec.open', return_value=mock_om),
            mock.patch('stargazing_core._weather._OmFileReader', return_value=mock_om),
        ):
            data = reader.read_window(
                WeatherVariable.CLOUD_COVER,
                north=50,
                south=40,
                east=10,
                west=0,
                shape=(32, 32),
            )
        assert data.shape == (32, 32)

    def test_read_window_not_array_raises(self):
        reader = self._make_reader()
        mock_om = _make_mock_om_reader(
            children={'cloud_cover': _make_mock_om_child(is_array=False)}
        )

        with (
            mock.patch('fsspec.open', return_value=mock_om),
            mock.patch('stargazing_core._weather._OmFileReader', return_value=mock_om),
            pytest.raises(ValueError, match='not an array'),
        ):
            reader.read_window(
                WeatherVariable.CLOUD_COVER,
                north=50,
                south=40,
                east=10,
                west=0,
            )

    def test_read_window_missing_variable_raises(self):
        reader = self._make_reader()
        mock_om = _make_mock_om_reader(children={'cloud_cover': RuntimeError('no such child')})

        with (
            mock.patch('fsspec.open', return_value=mock_om),
            mock.patch('stargazing_core._weather._OmFileReader', return_value=mock_om),
            pytest.raises(ValueError, match='not found in OM file'),
        ):
            reader.read_window(
                WeatherVariable.CLOUD_COVER,
                north=50,
                south=40,
                east=10,
                west=0,
            )

    def test_read_window_fsspec_failure_raises_oserror(self):
        reader = self._make_reader()
        with (
            mock.patch('fsspec.open', side_effect=OSError('network down')),
            pytest.raises(OSError, match='network down'),
        ):
            reader.read_window(
                WeatherVariable.CLOUD_COVER,
                north=50,
                south=40,
                east=10,
                west=0,
            )

    def test_read_window_lon_wraps_zero_meridian(self):
        """When lon crosses 0° (e.g., west=358°, east=2°), read two strips."""
        reader = self._make_reader()

        # Grid: 1801×3600 → 0° in middle of col range
        # west=358° → w_norm=358, c_left≈3579
        # east=2° → e_norm=2, c_right≈20
        # crosses_zero=True → read left:[3579,3600) + right:[0,20)
        left_data = np.full((20, 21), 30.0, dtype=np.float32)  # cols 3579-3599
        right_data = np.full((20, 20), 40.0, dtype=np.float32)  # cols 0-19

        call_count = [0]

        def read_array_side_effect(slices):
            call_count[0] += 1
            if call_count[0] == 1:
                return left_data
            else:
                return right_data

        mock_child = _make_mock_om_child(shape=(1801, 3600))
        mock_child.read_array.side_effect = read_array_side_effect

        mock_om = _make_mock_om_reader_with_child(mock_child)

        with (
            mock.patch('fsspec.open', return_value=mock_om),
            mock.patch('stargazing_core._weather._OmFileReader', return_value=mock_om),
            mock.patch('numpy.hstack', return_value=np.hstack((left_data, right_data))),
        ):
            data = reader.read_window(
                WeatherVariable.CLOUD_COVER,
                north=50,
                south=48,
                east=2,
                west=-2,
            )
        assert data.shape == (20, 41)
        # Left side should be 30, right side 40
        assert data[0, 0] == 30.0
        assert data[0, -1] == 40.0

    def test_read_window_swapped_lat_bounds(self):
        """north < south should auto-swap (user-friendly)."""
        reader = self._make_reader()
        mock_om = _make_mock_om_reader()

        with (
            mock.patch('fsspec.open', return_value=mock_om),
            mock.patch('stargazing_core._weather._OmFileReader', return_value=mock_om),
        ):
            data = reader.read_window(
                WeatherVariable.CLOUD_COVER,
                north=30,
                south=50,
                east=10,
                west=0,  # reversed!
            )
            assert isinstance(data, np.ndarray)

    def test_read_window_out_of_bounds_poles(self):
        """Lat values beyond ±90° should be clamped."""
        reader = self._make_reader()
        mock_child = _make_mock_om_child(read_array_data=np.full((50, 50), 50.0, dtype=np.float32))
        mock_om = _make_mock_om_reader_with_child(mock_child)

        with (
            mock.patch('fsspec.open', return_value=mock_om),
            mock.patch('stargazing_core._weather._OmFileReader', return_value=mock_om),
        ):
            data = reader.read_window(
                WeatherVariable.CLOUD_COVER,
                north=100,
                south=-100,
                east=10,
                west=0,  # beyond poles
            )
            assert isinstance(data, np.ndarray)

    def test_read_window_3d_grid_raises(self):
        """A 3-D grid should raise ValueError."""
        reader = self._make_reader()
        mock_child = _make_mock_om_child(is_array=True, shape=(10, 20, 30))
        mock_om = _make_mock_om_reader_with_child(mock_child)

        with (
            mock.patch('fsspec.open', return_value=mock_om),
            mock.patch('stargazing_core._weather._OmFileReader', return_value=mock_om),
            pytest.raises(ValueError, match='Expected 2-D'),
        ):
            reader.read_window(
                WeatherVariable.CLOUD_COVER,
                north=50,
                south=40,
                east=10,
                west=0,
            )

    def test_read_window_read_array_oserror(self):
        """read_array raising an exception should be wrapped in OSError."""
        reader = self._make_reader()
        mock_child = _make_mock_om_child()
        mock_child.read_array.side_effect = RuntimeError('chunk decode error')
        mock_om = _make_mock_om_reader_with_child(mock_child)

        with (
            mock.patch('fsspec.open', return_value=mock_om),
            mock.patch('stargazing_core._weather._OmFileReader', return_value=mock_om),
            pytest.raises(OSError, match='Failed to read data window'),
        ):
            reader.read_window(
                WeatherVariable.CLOUD_COVER,
                north=50,
                south=40,
                east=10,
                west=0,
            )

    def test_read_window_lon_wrap_read_oserror(self):
        """read_array raising in lon-wrapped mode should be OSError."""
        reader = self._make_reader()
        mock_child = _make_mock_om_child(shape=(1801, 3600))
        mock_child.read_array.side_effect = RuntimeError('chunk error')
        mock_om = _make_mock_om_reader_with_child(mock_child)

        with (
            mock.patch('fsspec.open', return_value=mock_om),
            mock.patch('stargazing_core._weather._OmFileReader', return_value=mock_om),
            pytest.raises(OSError, match='Failed to read lon-wrapped window'),
        ):
            reader.read_window(
                WeatherVariable.CLOUD_COVER,
                north=50,
                south=48,
                east=2,
                west=-2,  # crosses 0°
            )

    def test_open_om_file_https_fallback(self):
        """When s3fs is unavailable, fall back to HTTPS download."""
        reader = self._make_reader()

        mock_child = _make_mock_om_child(read_array_data=np.full((10, 10), 50.0, dtype=np.float32))
        mock_om = _make_mock_om_reader_with_child(mock_child)

        def _fake_open_fail(uri, mode=None, s3=None, blockcache=None):
            raise ImportError('No module named s3fs')

        with (
            mock.patch('fsspec.open', side_effect=_fake_open_fail),
            mock.patch(
                'urllib.request.urlretrieve',
                return_value=None,
            ) as mock_retrieve,
            mock.patch(
                'stargazing_core._weather._OmFileReader.from_path',
                return_value=mock_om,
            ),
            mock.patch('os.path.exists', return_value=False),
            mock.patch('os.makedirs'),
        ):
            data = reader.read_window(
                WeatherVariable.CLOUD_COVER,
                north=50,
                south=40,
                east=10,
                west=0,
            )
            assert isinstance(data, np.ndarray)
            mock_retrieve.assert_called_once()

    def test_read_point_empty_data(self):
        """read_point with an empty array should return NaN."""
        reader = self._make_reader()
        mock_child = _make_mock_om_child(read_array_data=np.array([[]], dtype=np.float32))
        mock_child.shape = (1801, 3600)
        mock_om = _make_mock_om_reader_with_child(mock_child)

        with (
            mock.patch('fsspec.open', return_value=mock_om),
            mock.patch('stargazing_core._weather._OmFileReader', return_value=mock_om),
        ):
            result = reader.read_point(
                WeatherVariable.CLOUD_COVER,
                lat=90,
                lon=0,
            )
            assert math.isnan(result)


def _make_mock_om_reader_with_child(mock_child):
    """Helper: create an OM reader mock with a specific child variable."""
    r = mock.MagicMock()
    r.get_child_by_name.return_value = mock_child
    r.__enter__.return_value = r
    r.__exit__.return_value = False
    return r


class TestOmWeatherReaderReadPoint:
    @staticmethod
    def _make_reader():
        with mock.patch('urllib.request.urlopen', return_value=_mock_urlopen_success()):
            return _OmWeatherReader(WeatherModel.DWD_ICON)

    def test_read_point_returns_float(self):
        reader = self._make_reader()
        mock_om = _make_mock_om_reader(
            children={
                'cloud_cover': _make_mock_om_child(
                    read_array_data=np.full((10, 10), 42.0, dtype=np.float32)
                )
            }
        )

        with (
            mock.patch('fsspec.open', return_value=mock_om),
            mock.patch('stargazing_core._weather._OmFileReader', return_value=mock_om),
        ):
            result = reader.read_point(
                WeatherVariable.CLOUD_COVER,
                lat=35.0,
                lon=115.0,
            )
        assert isinstance(result, float)
        assert result == pytest.approx(42.0)

    def test_read_point_all_nan(self):
        reader = self._make_reader()
        mock_om = _make_mock_om_reader(
            children={
                'cloud_cover': _make_mock_om_child(
                    read_array_data=np.full((5, 5), np.nan, dtype=np.float32)
                )
            }
        )

        with (
            mock.patch('fsspec.open', return_value=mock_om),
            mock.patch('stargazing_core._weather._OmFileReader', return_value=mock_om),
        ):
            result = reader.read_point(
                WeatherVariable.CLOUD_COVER,
                lat=35.0,
                lon=115.0,
            )
        assert math.isnan(result)
