"""Telescope configuration, derived optics, and equipment presets.

Provides pure-math optical calculations — no astropy required.
"""

from __future__ import annotations

import math

from pydantic import BaseModel, Field, model_validator


class TelescopeConfig(BaseModel):
    """User telescope + camera configuration.

    All linear dimensions are in **mm**.  Angular results are in **degrees**.
    """

    # ── optical ───────────────────────────────────────────────────
    focal_length_mm: float = Field(gt=0, description='Focal length (mm)')
    aperture_mm: float | None = Field(default=None, gt=0, description='Aperture (mm)')
    central_obstruction_pct: float = Field(
        default=0, ge=0, le=50, description='Central obstruction (%)'
    )
    reducer_factor: float = Field(default=1.0, gt=0, description='Focal reducer factor')
    barlow_factor: float = Field(default=1.0, gt=0, description='Barlow / extender factor')

    # ── camera (optional — pure visual observing when absent) ─────
    sensor_width_mm: float | None = Field(default=None, gt=0, description='Sensor width (mm)')
    sensor_height_mm: float | None = Field(default=None, gt=0, description='Sensor height (mm)')
    sensor_pixel_size_um: float | None = Field(default=None, gt=0, description='Pixel size (µm)')

    # ── mount ─────────────────────────────────────────────────────
    mount_type: str = Field(default='equatorial', description='"equatorial" | "altaz"')
    guiding_supported: bool = Field(default=False)

    # ── filter ────────────────────────────────────────────────────
    filter_type: str | None = Field(
        default=None,
        description='Broadband / narrowband filter type',
    )

    # ── helpers ───────────────────────────────────────────────────

    @property
    def effective_focal_length_mm(self) -> float:
        """Focal length after reducer / Barlow."""
        return self.focal_length_mm * self.reducer_factor * self.barlow_factor

    @property
    def focal_ratio(self) -> float | None:
        """Focal ratio (f/).  ``None`` when aperture is unknown."""
        if self.aperture_mm is None or self.aperture_mm <= 0:
            return None
        return self.effective_focal_length_mm / self.aperture_mm

    @property
    def effective_aperture_mm(self) -> float | None:
        """Clear aperture after obstruction loss."""
        if self.aperture_mm is None:
            return None
        obstruction_ratio = self.central_obstruction_pct / 100.0
        return self.aperture_mm * math.sqrt(1.0 - obstruction_ratio**2)

    @property
    def limiting_magnitude(self) -> float | None:
        """Approximate visual limiting magnitude."""
        if self.aperture_mm is None:
            return None
        return 2.0 + 5.0 * math.log10(self.aperture_mm)

    # ── validation ────────────────────────────────────────────────

    @model_validator(mode='after')
    def _check_sensor_dimensions(self) -> TelescopeConfig:
        sw = self.sensor_width_mm
        sh = self.sensor_height_mm
        if (sw is None) != (sh is None):
            raise ValueError('sensor_width_mm and sensor_height_mm must both be set or both None')
        return self

    def compute_optics(self) -> TelescopeOptics:
        """Derive FOV, sampling, and limiting magnitude from this config."""
        return TelescopeOptics.compute(self)


# ── derived optics ─────────────────────────────────────────────────


class TelescopeOptics(BaseModel):
    """Optical parameters derived from a :class:`TelescopeConfig`."""

    effective_focal_length_mm: float
    fov_width_deg: float | None = None
    fov_height_deg: float | None = None
    fov_area_sq_deg: float | None = None
    focal_ratio: float | None = None
    limiting_magnitude: float | None = None
    sampling_arcsec_per_pixel: float | None = None
    dawes_limit_arcsec: float | None = None

    @classmethod
    def compute(cls, config: TelescopeConfig) -> TelescopeOptics:
        """Compute derived optics from *config*."""
        efl = config.effective_focal_length_mm
        result = cls(effective_focal_length_mm=efl)

        # FOV (only with sensor)
        if config.sensor_width_mm and config.sensor_height_mm:
            result.fov_width_deg = math.degrees(
                2.0 * math.atan(config.sensor_width_mm / (2.0 * efl))
            )
            result.fov_height_deg = math.degrees(
                2.0 * math.atan(config.sensor_height_mm / (2.0 * efl))
            )
            result.fov_area_sq_deg = round(result.fov_width_deg * result.fov_height_deg, 4)

        result.focal_ratio = config.focal_ratio
        result.limiting_magnitude = config.limiting_magnitude

        # Sampling (arcsec / pixel)
        if config.sensor_pixel_size_um is not None and efl > 0:
            result.sampling_arcsec_per_pixel = round(206.265 * config.sensor_pixel_size_um / efl, 2)

        # Dawes limit (only with aperture)
        if config.aperture_mm is not None and config.aperture_mm > 0:
            result.dawes_limit_arcsec = round(116.0 / config.aperture_mm, 2)

        return result


# ── equipment presets ──────────────────────────────────────────────


def _mk(
    focal: float,
    sw: float,
    sh: float,
    aperture: float | None = None,
    pixel: float | None = None,
    mount: str = 'equatorial',
    filter_type: str | None = None,
) -> TelescopeConfig:
    return TelescopeConfig(
        focal_length_mm=focal,
        sensor_width_mm=sw,
        sensor_height_mm=sh,
        aperture_mm=aperture,
        sensor_pixel_size_um=pixel,
        mount_type=mount,
        filter_type=filter_type,
    )


TELESCOPE_PRESETS: dict[str, TelescopeConfig] = {
    # ── smart telescopes ────────────────────────────────────────
    'seestar-s50': _mk(250, 7.6, 5.7, aperture=50, pixel=2.9, mount='altaz'),
    'seestar-s30': _mk(150, 7.6, 5.7, aperture=30, pixel=2.9, mount='altaz'),
    'dwarf-ii': _mk(100, 7.4, 5.6, aperture=24, pixel=3.75, mount='altaz'),
    'vespera': _mk(200, 7.6, 5.7, aperture=50, pixel=2.9, mount='altaz'),
    # ── wide-field refractors ───────────────────────────────────
    'redcat51-asi2600': _mk(250, 23.5, 15.7, aperture=51, pixel=3.76),
    'redcat51-asi533': _mk(250, 11.3, 11.3, aperture=51, pixel=3.76),
    'redcat71-asi2600': _mk(350, 23.5, 15.7, aperture=71, pixel=3.76),
    'askar-fma180-asi2600': _mk(180, 23.5, 15.7, aperture=40, pixel=3.76),
    'rokinon-135-asi2600': _mk(135, 23.5, 15.7, aperture=67.5, pixel=3.76),
    # ── medium refractors ───────────────────────────────────────
    'askar-103apo-asi2600': _mk(700, 23.5, 15.7, aperture=103, pixel=3.76),
    'sharpstar-61edphii-asi533': _mk(275, 11.3, 11.3, aperture=61, pixel=3.76),
    # ── full-frame DSLR lenses ──────────────────────────────────
    'fullframe-200mm': _mk(200, 36, 24, pixel=5.9),
    'fullframe-50mm': _mk(50, 36, 24, pixel=5.9),
    'fullframe-85mm': _mk(85, 36, 24, pixel=5.9),
    # ── SCT / long focal length ─────────────────────────────────
    'c8-asi2600': _mk(2032, 23.5, 15.7, aperture=203, pixel=3.76),
    'c8-reducer-asi2600': _mk(1280, 23.5, 15.7, aperture=203, pixel=3.76),
    'c11-asi2600': _mk(2800, 23.5, 15.7, aperture=280, pixel=3.76),
    'c11-reducer-asi2600': _mk(1960, 23.5, 15.7, aperture=280, pixel=3.76),
    'c14-asi2600': _mk(3910, 23.5, 15.7, aperture=356, pixel=3.76),
    'c14-reducer-asi2600': _mk(2740, 23.5, 15.7, aperture=356, pixel=3.76),
    # ── RASA (fast astrographs) ─────────────────────────────────
    'rasa8-asi2600': _mk(400, 23.5, 15.7, aperture=203, pixel=3.76),
    'rasa11-asi2600': _mk(620, 23.5, 15.7, aperture=279, pixel=3.76),
    # ── Newtonians ──────────────────────────────────────────────
    'f4-newt-8inch-asi2600': _mk(800, 23.5, 15.7, aperture=203, pixel=3.76),
    'f4-newt-10inch-asi2600': _mk(1000, 23.5, 15.7, aperture=254, pixel=3.76),
}
