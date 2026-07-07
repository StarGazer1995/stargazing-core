"""Tests for mosaic planning."""

import pytest

from stargazing_core import compute_mosaic_grid

# ── Single-panel (target fits in one FOV) ──────────────────────────────


def test_single_panel_when_target_smaller_than_fov():
    """Target smaller than FOV → 1 panel, centred on target."""
    grid = compute_mosaic_grid(
        target_ra=10.0,
        target_dec=41.0,
        maj_arcmin=30,
        fov_width_deg=2.0,
        fov_height_deg=1.5,
    )
    assert grid.total_panels == 1
    assert grid.rows == 1
    assert grid.cols == 1
    p = grid.panels[0]
    assert p.row == 1 and p.col == 1
    assert p.ra_center == pytest.approx(10.0)
    assert p.dec_center == pytest.approx(41.0)
    assert len(p.corners) == 4


def test_single_panel_when_target_equals_fov():
    """Target exactly equals FOV → 1 panel."""
    grid = compute_mosaic_grid(
        target_ra=0,
        target_dec=0,
        maj_arcmin=60,  # 1 degree
        fov_width_deg=1.0,
        fov_height_deg=1.0,
    )
    assert grid.total_panels == 1
    assert grid.cols == 1
    assert grid.rows == 1


# ── Multi-panel ────────────────────────────────────────────────────────


def test_two_column_mosaic():
    """Target 1.5× FOV width → 2 columns, 1 row."""
    grid = compute_mosaic_grid(
        target_ra=0,
        target_dec=45,
        maj_arcmin=120,  # 2 degrees
        min_arcmin=50,  # narrow in height → stays 1 row
        fov_width_deg=1.5,
        fov_height_deg=1.0,
        overlap=0.15,
    )
    assert grid.cols == 2
    assert grid.rows == 1
    assert grid.total_panels == 2
    # First panel left of centre, second right of centre
    assert grid.panels[0].ra_center < grid.panels[1].ra_center


def test_m31_redcat51():
    """M31 (199'×71') with RedCat51 (~5.4°×3.6°) → single panel (200' < 5.4°)."""
    grid = compute_mosaic_grid(
        target_ra=10.68,
        target_dec=41.27,
        maj_arcmin=199.5,
        min_arcmin=70.8,
        fov_width_deg=5.38,
        fov_height_deg=3.60,
    )
    assert grid.total_panels == 1
    assert grid.cols == 1


def test_m31_c8_reducer():
    """M31 (199'×71') with C8+reducer (~1.05°×0.70°) → ~4 panels."""
    grid = compute_mosaic_grid(
        target_ra=10.68,
        target_dec=41.27,
        maj_arcmin=199.5,
        min_arcmin=70.8,
        fov_width_deg=1.05,
        fov_height_deg=0.70,
        overlap=0.15,
    )
    # 199/60 = 3.32°, step_x = 1.05×0.85 = 0.8925
    # cols = 1 + ceil((3.32-1.05)/0.8925) = 1 + ceil(2.27/0.8925) = 1+3 = 4
    # 71/60 = 1.18°, step_y = 0.70×0.85 = 0.595
    # rows = 1 + ceil((1.18-0.70)/0.595) = 1 + ceil(0.48/0.595) = 1+1 = 2
    assert grid.cols == 4
    assert grid.rows == 2
    assert grid.total_panels == 8
    assert grid.overlap == 0.15


# ── Corners ────────────────────────────────────────────────────────────


def test_corners_form_rectangle():
    """Corners are 4 points forming a rectangle."""
    grid = compute_mosaic_grid(
        target_ra=0,
        target_dec=0,
        maj_arcmin=120,
        fov_width_deg=2.0,
        fov_height_deg=1.0,
        overlap=0.2,
    )
    for p in grid.panels:
        assert len(p.corners) == 4
        # Width ≈ fov_width (in RA, adjusted by cos(dec))
        w = abs(p.corners[1][0] - p.corners[0][0])
        assert w == pytest.approx(2.0, abs=0.01)
        # Height ≈ fov_height
        h = abs(p.corners[0][1] - p.corners[3][1])
        assert h == pytest.approx(1.0, abs=0.01)


# ── min_arcmin defaults ────────────────────────────────────────────────


def test_min_arcmin_defaults_to_maj():
    """When min_arcmin is None, circular approximation is used."""
    grid1 = compute_mosaic_grid(
        target_ra=0,
        target_dec=0,
        maj_arcmin=30,
        fov_width_deg=1.0,
        fov_height_deg=1.0,
    )
    grid2 = compute_mosaic_grid(
        target_ra=0,
        target_dec=0,
        maj_arcmin=30,
        min_arcmin=30,
        fov_width_deg=1.0,
        fov_height_deg=1.0,
    )
    assert grid1.rows == grid2.rows
    assert grid1.cols == grid2.cols
    assert grid1.total_panels == 1


# ── Error cases ────────────────────────────────────────────────────────


def test_zero_fov_raises():
    with pytest.raises(ValueError, match='fov_width_deg'):
        compute_mosaic_grid(
            target_ra=0,
            target_dec=0,
            maj_arcmin=60,
            fov_width_deg=0,
            fov_height_deg=1.0,
        )


def test_negative_overlap_raises():
    with pytest.raises(ValueError, match='overlap'):
        compute_mosaic_grid(
            target_ra=0,
            target_dec=0,
            maj_arcmin=60,
            fov_width_deg=1.0,
            fov_height_deg=1.0,
            overlap=-0.1,
        )


def test_overlap_exceeds_max_panels():
    """Low overlap with tiny FOV → too many panels → raises."""
    with pytest.raises(ValueError, match='max 10'):
        compute_mosaic_grid(
            target_ra=0,
            target_dec=0,
            maj_arcmin=600,  # 10° object
            fov_width_deg=0.5,
            fov_height_deg=0.5,  # tiny FOV
            overlap=0.0,  # no overlap
            max_panels=10,
        )
