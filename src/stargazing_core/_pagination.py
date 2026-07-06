"""Generic paginated result model."""

from __future__ import annotations

from typing import Generic, TypeVar

from pydantic import BaseModel, Field

T = TypeVar('T')


class PaginatedResult(BaseModel, Generic[T]):
    """A generic paginated result container.

    Usage:
        result = PaginatedResult[MyModel](
            items=[...], total=42, page=1, page_size=10
        )
        result.model_dump()  # serializes with typed items
    """

    items: list[T] = Field(description='Paginated items for the current page')
    total: int = Field(ge=0, description='Total number of items across all pages')
    page: int = Field(ge=1, description='Current page number (1-based)')
    page_size: int = Field(ge=1, description='Number of items per page')
    total_pages: int = Field(ge=0, description='Total number of pages')
    resource_id: str | None = Field(default=None, description='Optional cache/resource identifier')
