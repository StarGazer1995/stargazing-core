"""Automatic single-night shooting schedule generator.

Takes match_telescope_targets output (targets + moon) and produces a
time-allocated shooting plan — what to shoot when, considering moon
windows, altitude curves, and target quality.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from pydantic import BaseModel, Field


class ShootingSlot(BaseModel):
    """A single imaging time slot for one target."""

    start_time: str = Field(description='ISO datetime when imaging starts')
    end_time: str = Field(description='ISO datetime when imaging ends')
    target_name: str
    target_type: str
    start_alt: float = Field(description='Altitude at start (degrees)')
    end_alt: float = Field(description='Altitude at end (degrees)')
    duration_min: int = Field(ge=0, description='Slot duration in minutes')
    fov_fit_score: float = Field(ge=0, le=1)
    suitability_score: float = Field(ge=0, le=100)
    mosaic_recommended: bool = False
    notes: list[str] = Field(default_factory=list)


class ShootingPlan(BaseModel):
    """Complete single-night shooting plan."""

    date: str
    dusk: str
    dawn: str
    moon_phase: str
    moon_illumination: float = Field(ge=0, le=1)
    moon_delay_min: int = Field(ge=0, description='Minutes to wait for moon to set')
    slots: list[ShootingSlot]
    total_dark_min: int = Field(ge=0)
    used_min: int = Field(ge=0)
    warnings: list[str] = Field(default_factory=list)


def generate_shooting_schedule(
    targets: list[dict],
    moon: dict,
    dusk: str,
    dawn: str,
    min_alt: float = 25.0,
    min_duration_min: int = 30,
) -> ShootingPlan:
    """Generate a single-night shooting schedule.

    Args:
        targets: match_telescope_targets 'targets' list (sorted by dawn_altitude).
        moon: match_telescope_targets 'moon' dict.
        dusk: Civil dusk ISO datetime string.
        dawn: Civil dawn ISO datetime string.
        min_alt: Minimum acceptable altitude for imaging (degrees).
        min_duration_min: Minimum slot duration to include a target (minutes).

    Returns:
        ShootingPlan with time-allocated slots.
    """

    # ── Parse dusk/dawn (force UTC) ────────────────────────────────────
    dusk_dt = datetime.fromisoformat(dusk).replace(tzinfo=timezone.utc)
    dawn_dt = datetime.fromisoformat(dawn).replace(tzinfo=timezone.utc)
    total_dark_sec = (dawn_dt - dusk_dt).total_seconds()
    total_dark_min = int(total_dark_sec / 60)
    if total_dark_min <= 0:
        return ShootingPlan(
            date=dusk_dt.strftime('%Y-%m-%d'),
            dusk=dusk,
            dawn=dawn,
            moon_phase=moon.get('phase', 'Unknown'),
            moon_illumination=moon.get('illumination', 0),
            moon_delay_min=0,
            slots=[],
            total_dark_min=total_dark_min,
            used_min=0,
            warnings=['No dark time — cannot schedule'],
        )

    # ── Determine moon delay ──────────────────────────────────────────
    moon_delay_min = 0
    warnings: list[str] = []

    if not moon.get('always_down') and moon.get('illumination', 0) > 0.3:
        # Moon is up at dusk with significant illumination — find when it sets
        moon_curve = moon.get('altitude_curve', [])
        moon_set_time = None
        for pt in moon_curve:
            if pt['alt'] > 0:
                continue
            # Moon just went below horizon — find where it crosses
            moon_set_time = pt['time']
            break

        if moon_set_time is not None:
            moon_set_dt = datetime.fromtimestamp(moon_set_time, tz=timezone.utc)
            delay_sec = (moon_set_dt - dusk_dt).total_seconds()
            if delay_sec > 0:
                moon_delay_min = max(0, int(delay_sec / 60) + 10)  # +10 min buffer
                warnings.append(
                    f'Moon ({moon["illumination"] * 100:.0f}% {moon["phase"]}) sets '
                    f'at {moon_set_dt.strftime("%H:%M")} — wait {moon_delay_min} min'
                )

    if moon.get('always_up') and moon.get('illumination', 0) > 0.3:
        warnings.append(
            f'Moon up all night ({moon["illumination"] * 100:.0f}% {moon["phase"]}) '
            '— narrowband recommended'
        )

    # ── Allocate slots ────────────────────────────────────────────────
    available_start_sec = moon_delay_min * 60  # seconds after dusk
    available_end_sec = total_dark_sec
    slots: list[ShootingSlot] = []
    current_time_sec = available_start_sec

    for t in targets:
        if current_time_sec >= available_end_sec:
            warnings.append(f'Skipped {t["name"]} — dark window exhausted')
            break

        curve = t.get('altitude_curve', [])
        if not curve or len(curve) < 2:
            continue

        # Find the continuous segment where alt > min_alt, starting from
        # the earliest time where alt crosses above min_alt.
        slot_start_sec = None
        slot_end_sec = None
        dusk_ts = dusk_dt.timestamp()
        for pt in curve:
            t_sec = pt['time'] - dusk_ts
            if t_sec < current_time_sec:
                continue
            if pt['alt'] >= min_alt:
                if slot_start_sec is None:
                    slot_start_sec = t_sec
                slot_end_sec = t_sec
            elif slot_start_sec is not None:
                # Fell below min_alt — end the window
                break

        if slot_start_sec is None or slot_end_sec is None:
            continue
        window_duration = slot_end_sec - slot_start_sec
        window_min = int(window_duration / 60)
        if window_min < min_duration_min:
            continue

        # Don't start before the target is actually above min_alt
        if current_time_sec < slot_start_sec:
            current_time_sec = slot_start_sec

        # Clamp to remaining dark time
        remaining_sec = available_end_sec - current_time_sec
        if window_duration > remaining_sec:
            window_duration = remaining_sec
            window_min = int(window_duration / 60)
            if window_min < min_duration_min:
                break

        start_dt = dusk_dt + timedelta(seconds=current_time_sec)
        end_dt = dusk_dt + timedelta(seconds=current_time_sec + window_duration)

        # Find actual alt values at start/end
        start_ts = start_dt.timestamp()
        end_ts = end_dt.timestamp()
        start_alt = next(
            (p['alt'] for p in curve if abs(p['time'] - start_ts) < 60),
            t.get('altitude', 0),
        )
        end_alt = next(
            (p['alt'] for p in reversed(curve) if abs(p['time'] - end_ts) < 60),
            t.get('dawn_altitude', 0),
        )

        notes: list[str] = []
        if t.get('mosaic_recommended'):
            notes.append('Mosaic recommended')
        if t.get('filter_match_score', 0) >= 0.8:
            notes.append(f'Filter match {t["filter_match_score"] * 100:.0f}%')

        slots.append(
            ShootingSlot(
                start_time=start_dt.isoformat(),
                end_time=end_dt.isoformat(),
                target_name=t['name'],
                target_type=t['type'],
                start_alt=round(start_alt, 1),
                end_alt=round(end_alt, 1),
                duration_min=window_min,
                fov_fit_score=t.get('fov_fit_score', 0),
                suitability_score=t.get('suitability_score', 0),
                mosaic_recommended=t.get('mosaic_recommended', False),
                notes=notes,
            )
        )

        current_time_sec += window_duration

    used_min = sum(s.duration_min for s in slots)

    # ── Remaining time warning ────────────────────────────────────────
    remaining_min = total_dark_min - used_min - moon_delay_min
    if remaining_min > 60:
        warnings.append(f'{remaining_min} min of dark time unallocated')

    return ShootingPlan(
        date=dusk_dt.strftime('%Y-%m-%d'),
        dusk=dusk,
        dawn=dawn,
        moon_phase=moon.get('phase', 'Unknown'),
        moon_illumination=moon.get('illumination', 0),
        moon_delay_min=moon_delay_min,
        slots=slots,
        total_dark_min=total_dark_min,
        used_min=used_min,
        warnings=warnings,
    )
