"""Tests for TelescopeConfig, TelescopeOptics, and equipment presets."""

import math

import pydantic
import pytest

from stargazing_core._telescope import TELESCOPE_PRESETS, TelescopeConfig, TelescopeOptics


def test_seestar_fov():
    """Seestar S50 has a known FOV: 250 mm FL × 7.6×5.7 mm sensor → ~1.7°×1.3°."""
    cfg = TELESCOPE_PRESETS['seestar-s50']
    optics = cfg.compute_optics()

    assert optics.fov_width_deg == pytest.approx(1.74, abs=0.05)
    assert optics.fov_height_deg == pytest.approx(1.31, abs=0.05)
    assert optics.effective_focal_length_mm == 250
    assert optics.limiting_magnitude is not None
    assert optics.sampling_arcsec_per_pixel is not None


def test_redcat51_fov():
    """RedCat 51 + ASI2600: 250 mm × 23.5×15.7 → ~5.4°×3.6°."""
    cfg = TELESCOPE_PRESETS['redcat51-asi2600']
    optics = cfg.compute_optics()

    assert optics.fov_width_deg == pytest.approx(5.38, abs=0.1)
    assert optics.fov_height_deg == pytest.approx(3.60, abs=0.1)


def test_c8_f10_fov():
    """C8 native: 2032 mm × 23.5×15.7 → tiny FOV ~0.66°×0.44°."""
    cfg = TELESCOPE_PRESETS['c8-asi2600']
    optics = cfg.compute_optics()

    assert optics.fov_width_deg == pytest.approx(0.66, abs=0.02)
    assert optics.fov_height_deg == pytest.approx(0.44, abs=0.02)


def test_reducer_doubles_fov():
    """Reducer factor 0.63 should halve effective focal length → double FOV."""
    cfg = TelescopeConfig(
        focal_length_mm=1000, sensor_width_mm=36, sensor_height_mm=24, reducer_factor=0.63
    )
    optics = cfg.compute_optics()
    assert optics.effective_focal_length_mm == 630
    assert optics.fov_width_deg is not None and optics.fov_width_deg > 1.0


def test_limiting_magnitude():
    """200 mm aperture → limiting mag ≈ 2 + 5×log₁₀(200) ≈ 13.5."""
    cfg = TelescopeConfig(
        focal_length_mm=1000, sensor_width_mm=36, sensor_height_mm=24, aperture_mm=200
    )
    optics = cfg.compute_optics()
    assert optics.limiting_magnitude == pytest.approx(13.5, abs=0.2)


def test_dawes_limit():
    """Dawes limit for 50 mm aperture → 116/50 = 2.32 arcsec."""
    cfg = TELESCOPE_PRESETS['seestar-s50']
    optics = cfg.compute_optics()
    assert optics.dawes_limit_arcsec == pytest.approx(2.32, abs=0.05)


def test_sampling():
    """Sampling for Seestar: 206.265 × 2.9 µm / 250 mm ≈ 2.39 arcsec/px."""
    cfg = TELESCOPE_PRESETS['seestar-s50']
    optics = cfg.compute_optics()
    assert optics.sampling_arcsec_per_pixel == pytest.approx(2.39, abs=0.05)


def test_central_obstruction():
    """30% obstruction → effective aperture = aperture × sqrt(1-0.3²) ≈ 0.95×."""
    cfg = TelescopeConfig(
        focal_length_mm=1000,
        sensor_width_mm=36,
        sensor_height_mm=24,
        aperture_mm=200,
        central_obstruction_pct=30,
    )
    assert cfg.effective_aperture_mm == pytest.approx(200 * math.sqrt(0.91), abs=0.1)


def test_no_sensor():
    """Visual-only config (no sensor) → FOV fields are None."""
    cfg = TelescopeConfig(focal_length_mm=1200, aperture_mm=200)
    optics = cfg.compute_optics()
    assert optics.fov_width_deg is None
    assert optics.fov_height_deg is None
    assert optics.fov_area_sq_deg is None
    assert optics.limiting_magnitude is not None  # still computed


def test_mismatched_sensor_dims_raises():
    """Setting only one sensor dimension should fail validation."""
    with pytest.raises(pydantic.ValidationError):
        TelescopeConfig(focal_length_mm=1000, sensor_width_mm=36)


def test_all_presets_compute():
    """Every preset should produce valid TelescopeOptics."""
    for name, cfg in TELESCOPE_PRESETS.items():
        optics = cfg.compute_optics()
        assert optics.effective_focal_length_mm > 0, f'{name}: bad EFL'
        if cfg.sensor_width_mm:
            assert optics.fov_width_deg is not None, f'{name}: missing FOV width'
            assert optics.fov_width_deg > 0, f'{name}: non-positive FOV width'


def test_compute_optics_via_config_method():
    """TelescopeConfig.compute_optics() returns TelescopeOptics."""
    cfg = TELESCOPE_PRESETS['seestar-s50']
    optics = cfg.compute_optics()
    assert isinstance(optics, TelescopeOptics)
    assert optics == TelescopeOptics.compute(cfg)


def test_effective_aperture_none_when_no_aperture():
    """effective_aperture_mm returns None when aperture is not set."""
    cfg = TelescopeConfig(focal_length_mm=1000)
    assert cfg.effective_aperture_mm is None
    assert cfg.limiting_magnitude is None


def test_preset_count():
    """We should have a reasonable number of presets."""
    assert len(TELESCOPE_PRESETS) >= 15
