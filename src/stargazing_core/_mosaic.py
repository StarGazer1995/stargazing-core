"""Mosaic (multi-panel) planning for large deep-sky objects.

Pure computation — no astropy required for the grid math (the caller provides
already-transformed angular coordinates).

Provides ``compute_mosaic_grid`` which, given a target's angular size and a
telescope field-of-view, calculates a regular grid of overlapping panels that
cover the entire object.
"""

from __future__ import annotations

import math
from typing import Optional

from pydantic import BaseModel, Field


class MosaicPanel(BaseModel):
    """A single panel in a mosaic grid."""

    row: int = Field(ge=1, description='1-based row index')
    col: int = Field(ge=1, description='1-based column index')
    ra_center: float = Field(description='Panel centre RA (degrees)')
    dec_center: float = Field(description='Panel centre Dec (degrees)')
    corners: list[tuple[float, float]] = Field(
        description='Four corners as [(ra, dec), …] in clockwise order from top-left'
    )


class MosaicGrid(BaseModel):
    """Complete mosaic plan for a single target."""

    target_name: str
    rows: int = Field(ge=1)
    cols: int = Field(ge=1)
    total_panels: int = Field(ge=1)
    overlap: float = Field(ge=0.0, le=1.0)
    fov_width_deg: float = Field(gt=0)
    fov_height_deg: float = Field(gt=0)
    panels: list[MosaicPanel]


def _panels_needed(target_deg: float, fov_deg: float, overlap: float) -> int:
    """How many panels are needed to cover *target_deg* with a FOV of *fov_deg*?

    A single frame covers the target when ``target_deg ≤ fov_deg``.
    Otherwise the first panel covers one full FOV, and each subsequent panel
    adds ``fov_deg × (1 − overlap)`` of new coverage.
    """
    if target_deg <= fov_deg:
        return 1
    remaining = target_deg - fov_deg
    return 1 + math.ceil(remaining / (fov_deg * (1.0 - overlap)))


def compute_mosaic_grid(
    target_ra: float,
    target_dec: float,
    maj_arcmin: float,
    *,
    min_arcmin: Optional[float] = None,
    fov_width_deg: float = 0,
    fov_height_deg: float = 0,
    overlap: float = 0.15,
    max_panels: int = 36,
) -> MosaicGrid:
    """Compute a regular overlapping grid covering a deep-sky object.

    Args:
        target_ra: Target centre Right Ascension in **degrees**.
        target_dec: Target centre Declination in **degrees**.
        maj_arcmin: Major-axis angular size in **arcminutes**.
        min_arcmin: Minor-axis angular size.  Defaults to *maj_arcmin*
            (circular approximation).
        fov_width_deg: Telescope field-of-view width in degrees.
        fov_height_deg: Telescope field-of-view height in degrees.
        overlap: Fractional overlap between adjacent panels (0–1).
        max_panels: Safety cap — raises ``ValueError`` if the grid would
            exceed this many panels.

    Returns:
        ``MosaicGrid`` with panel centre coordinates and corner positions
        suitable for rendering as FOV footprints (e.g. via Aladin Lite).

    Raises:
        ValueError: If *fov_width_deg* or *fov_height_deg* is ≤ 0, or if the
            computed panel count exceeds *max_panels*.
    """
    if fov_width_deg <= 0 or fov_height_deg <= 0:
        raise ValueError('fov_width_deg and fov_height_deg must be > 0')
    if not (0.0 <= overlap < 1.0):
        raise ValueError('overlap must be in [0, 1)')

    target_w = maj_arcmin / 60.0
    target_h = (min_arcmin if min_arcmin is not None else maj_arcmin) / 60.0

    cols = _panels_needed(target_w, fov_width_deg, overlap)
    rows = _panels_needed(target_h, fov_height_deg, overlap)
    total = cols * rows
    if total > max_panels:
        raise ValueError(
            f'Mosaic would need {total} panels (max {max_panels}). '
            f'Increase overlap or use a wider FOV.'
        )

    step_x = fov_width_deg * (1.0 - overlap)
    step_y = fov_height_deg * (1.0 - overlap)

    dec_rad = math.radians(target_dec)
    cos_dec = math.cos(dec_rad)
    if abs(cos_dec) < 1e-12:  # pragma: no cover — pole edge case
        cos_dec = 1e-12

    half_w = fov_width_deg / 2.0
    half_h = fov_height_deg / 2.0

    panels: list[MosaicPanel] = []
    for r in range(rows):
        for c in range(cols):
            # Offset from target centre (grid is centred)
            dx = (c - (cols - 1) / 2.0) * step_x
            dy = (r - (rows - 1) / 2.0) * step_y

            ra_c = target_ra + dx / cos_dec
            dec_c = target_dec + dy

            # Four corners (top-left, top-right, bottom-right, bottom-left)
            # "Top" = higher Dec, "Left" = lower RA
            corners: list[tuple[float, float]] = [
                (ra_c - half_w / cos_dec, dec_c + half_h),
                (ra_c + half_w / cos_dec, dec_c + half_h),
                (ra_c + half_w / cos_dec, dec_c - half_h),
                (ra_c - half_w / cos_dec, dec_c - half_h),
            ]

            panels.append(
                MosaicPanel(
                    row=r + 1,
                    col=c + 1,
                    ra_center=ra_c,
                    dec_center=dec_c,
                    corners=corners,
                )
            )

    return MosaicGrid(
        target_name='',  # filled by caller
        rows=rows,
        cols=cols,
        total_panels=total,
        overlap=overlap,
        fov_width_deg=fov_width_deg,
        fov_height_deg=fov_height_deg,
        panels=panels,
    )
