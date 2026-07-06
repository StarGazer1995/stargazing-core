"""Tests for generic PaginatedResult Pydantic model."""

import pytest
from pydantic import ValidationError

from stargazing_core import PaginatedResult


class TestPaginatedResult:
    def test_valid_empty_result(self):
        result = PaginatedResult[int](
            items=[],
            total=0,
            page=1,
            page_size=10,
            total_pages=0,
        )
        assert result.items == []
        assert result.total == 0
        assert result.page == 1
        assert result.page_size == 10
        assert result.total_pages == 0
        assert result.resource_id is None

    def test_valid_result_with_items(self):
        result = PaginatedResult[str](
            items=['a', 'b', 'c'],
            total=3,
            page=1,
            page_size=10,
            total_pages=1,
            resource_id='abc123',
        )
        assert len(result.items) == 3
        assert result.resource_id == 'abc123'

    def test_total_must_be_non_negative(self):
        with pytest.raises(ValidationError):
            PaginatedResult[int](
                items=[],
                total=-1,
                page=1,
                page_size=10,
                total_pages=0,
            )

    def test_page_must_be_positive(self):
        with pytest.raises(ValidationError):
            PaginatedResult[int](
                items=[],
                total=0,
                page=0,
                page_size=10,
                total_pages=0,
            )

    def test_page_size_must_be_positive(self):
        with pytest.raises(ValidationError):
            PaginatedResult[int](
                items=[],
                total=0,
                page=1,
                page_size=0,
                total_pages=0,
            )

    def test_total_pages_must_be_non_negative(self):
        with pytest.raises(ValidationError):
            PaginatedResult[int](
                items=[],
                total=0,
                page=1,
                page_size=10,
                total_pages=-1,
            )

    def test_model_dump_serializable(self):
        import json

        result = PaginatedResult[int](
            items=[1, 2, 3],
            total=3,
            page=1,
            page_size=10,
            total_pages=1,
        )
        dumped = result.model_dump()
        json.dumps(dumped)  # should not raise

    def test_generic_with_custom_type(self):
        from pydantic import BaseModel

        class Item(BaseModel):
            name: str
            value: int

        result = PaginatedResult[Item](
            items=[Item(name='a', value=1), Item(name='b', value=2)],
            total=2,
            page=1,
            page_size=10,
            total_pages=1,
        )
        assert result.items[0].name == 'a'
        assert result.items[1].value == 2
