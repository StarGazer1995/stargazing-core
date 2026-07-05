"""Shared geographic models — GeoPoint, GeoBounds, TimeInfo.

Originally defined separately in ``mcp-stargazing`` (``schemas/base.py``) and
``stargazing-place-finder`` (``models/geo.py``).  Unified here so both projects
share a single source of truth.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class GeoPoint(BaseModel):
    """A geographic coordinate point.

    Mirrors ``mcp-stargazing.schemas.base.GeoPoint`` and subsumes
    ``stargazing-place-finder.models.geo.GeoCoordinate``.
    """

    lat: float = Field(ge=-90.0, le=90.0, description='Latitude in decimal degrees')
    lon: float = Field(ge=-180.0, le=180.0, description='Longitude in decimal degrees')
    elevation_m: Optional[float] = Field(
        default=None, ge=0.0, description='Elevation in meters above sea level'
    )


class GeoBounds(BaseModel):
    """A bounding box defined by south/west/north/east edges.

    Mirrors ``mcp-stargazing.schemas.base.GeoBounds``.
    """

    south: float = Field(ge=-90.0, le=90.0, description='Southern latitude boundary')
    west: float = Field(ge=-180.0, le=180.0, description='Western longitude boundary')
    north: float = Field(ge=-90.0, le=90.0, description='Northern latitude boundary')
    east: float = Field(ge=-180.0, le=180.0, description='Eastern longitude boundary')

    @field_validator('north')
    @classmethod
    def _north_gte_south(cls, v: float, info) -> float:
        if 'south' in info.data and v < info.data['south']:
            raise ValueError(f'north ({v}) must be >= south ({info.data["south"]})')
        return v

    @field_validator('east')
    @classmethod
    def _east_gte_west(cls, v: float, info) -> float:
        if 'west' in info.data and v < info.data['west']:
            raise ValueError(f'east ({v}) must be >= west ({info.data["west"]})')
        return v


class TimeInfo(BaseModel):
    """A timezone-aware datetime with IANA timezone identifier.

    Mirrors ``mcp-stargazing.schemas.base.TimeInfo``.
    """

    dt: datetime = Field(description='Timezone-aware datetime object')
    timezone: str = Field(description="IANA timezone identifier, e.g. 'Asia/Shanghai'")

    @field_validator('dt')
    @classmethod
    def _dt_must_be_aware(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError(f'Datetime must be timezone-aware, got naive datetime: {v}')
        return v
