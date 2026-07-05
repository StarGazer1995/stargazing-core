"""Tests for time-grid utilities."""

from datetime import datetime

import numpy as np

from stargazing_core._timegrid import find_rise_set_indices, generate_time_grid


def test_generate_time_grid():
    grid = generate_time_grid(datetime(2024, 1, 15))
    assert len(grid) == 288  # 24h / 5min
    assert grid[0].datetime.hour == 0
    assert grid[-1].datetime.hour == 23


def test_find_rise_set_both():
    """Signal that goes from below to above and back."""
    altitudes = np.array([-1.0, -0.5, 0.5, 1.0, 0.5, -0.5, -1.0])
    rise, fall = find_rise_set_indices(altitudes, 0.0)
    assert rise == 1  # transition at index 1→2
    assert fall == 4  # transition at index 4→5


def test_find_rise_only():
    altitudes = np.array([-1.0, 0.5, 1.0, 2.0])
    rise, fall = find_rise_set_indices(altitudes, 0.0)
    assert rise == 0
    assert fall is None


def test_find_set_only():
    altitudes = np.array([2.0, 1.0, 0.5, -1.0])
    rise, fall = find_rise_set_indices(altitudes, 0.0)
    assert rise is None
    assert fall == 2


def test_always_below():
    altitudes = np.array([-5.0, -4.0, -3.0])
    rise, fall = find_rise_set_indices(altitudes, 0.0)
    assert rise is None
    assert fall is None
