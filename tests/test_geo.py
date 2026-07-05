"""Tests for shared geographic models."""

from datetime import datetime, timezone

import pydantic
import pytest

from stargazing_core._geo import GeoBounds, GeoPoint, TimeInfo


class TestGeoPoint:
    def test_valid_point(self) -> None:
        p = GeoPoint(lat=40.0, lon=116.0)
        assert p.lat == 40.0
        assert p.lon == 116.0
        assert p.elevation_m is None

    def test_with_elevation(self) -> None:
        p = GeoPoint(lat=40.0, lon=116.0, elevation_m=50.0)
        assert p.elevation_m == 50.0

    def test_lat_out_of_range_raises(self) -> None:
        with pytest.raises(pydantic.ValidationError):
            GeoPoint(lat=100.0, lon=0.0)
        with pytest.raises(pydantic.ValidationError):
            GeoPoint(lat=-100.0, lon=0.0)

    def test_lon_out_of_range_raises(self) -> None:
        with pytest.raises(pydantic.ValidationError):
            GeoPoint(lat=0.0, lon=200.0)
        with pytest.raises(pydantic.ValidationError):
            GeoPoint(lat=0.0, lon=-200.0)

    def test_boundary_values(self) -> None:
        GeoPoint(lat=90.0, lon=180.0)
        GeoPoint(lat=-90.0, lon=-180.0)

    def test_negative_elevation_raises(self) -> None:
        with pytest.raises(pydantic.ValidationError):
            GeoPoint(lat=0.0, lon=0.0, elevation_m=-1.0)


class TestGeoBounds:
    def test_valid_bounds(self) -> None:
        b = GeoBounds(south=30.0, west=100.0, north=40.0, east=120.0)
        assert b.south == 30.0
        assert b.north == 40.0

    def test_north_lt_south_raises(self) -> None:
        with pytest.raises(pydantic.ValidationError):
            GeoBounds(south=40.0, west=100.0, north=30.0, east=120.0)

    def test_east_lt_west_raises(self) -> None:
        with pytest.raises(pydantic.ValidationError):
            GeoBounds(south=30.0, west=120.0, north=40.0, east=100.0)

    def test_equal_edges_ok(self) -> None:
        GeoBounds(south=0.0, west=0.0, north=0.0, east=0.0)


class TestTimeInfo:
    def test_aware_datetime(self) -> None:
        dt = datetime(2024, 1, 1, tzinfo=timezone.utc)
        t = TimeInfo(dt=dt, timezone='UTC')
        assert t.dt == dt

    def test_naive_datetime_raises(self) -> None:
        with pytest.raises(pydantic.ValidationError):
            TimeInfo(dt=datetime(2024, 1, 1), timezone='Asia/Shanghai')
