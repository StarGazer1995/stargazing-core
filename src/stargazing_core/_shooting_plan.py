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
    max_slot_min: int = 720,
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

    # ── Per-minute best-target table + consecutive merge ────────────────
    available_start_sec = moon_delay_min * 60
    total_avail_min = total_dark_min - moon_delay_min
    if total_avail_min <= 0:
        return ShootingPlan(
            date=dusk_dt.strftime('%Y-%m-%d'),
            dusk=dusk,
            dawn=dawn,
            moon_phase=moon.get('phase', 'Unknown'),
            moon_illumination=moon.get('illumination', 0),
            moon_delay_min=moon_delay_min,
            slots=[],
            total_dark_min=total_dark_min,
            used_min=0,
            warnings=['Moon up all night — no dark window'],
        )

    dusk_ts = dusk_dt.timestamp()

    # Pre-compute per-target per-minute score table
    # Curve points are 15-min apart; fill gaps so every minute has a score
    target_minutes: list[dict] = []
    for t in targets:
        curve = t.get('altitude_curve', [])
        if not curve or len(curve) < 2:
            continue
        fov_s = (t.get('fov_fit_score', 0) or 0) * 40
        sb_s = (t.get('surface_brightness_score', 0) or 0) * 30
        flt_s = (t.get('filter_match_score', 0) or 0) * 20
        score_static = fov_s + sb_s + flt_s

        minute_score: dict[int, float] = {}
        for i, pt in enumerate(curve):
            t_sec = pt['time'] - dusk_ts
            minute = int((t_sec - available_start_sec) / 60)
            if not (0 <= minute < total_avail_min):
                continue
            if pt['alt'] < min_alt:
                continue
            # Fill this minute and up to 14 more (15-min gap to next point)
            end_minute = total_avail_min
            if i + 1 < len(curve):
                next_min = int((curve[i + 1]['time'] - dusk_ts - available_start_sec) / 60)
                end_minute = min(total_avail_min, next_min)
            score = score_static + (pt['alt'] / 90) * 10
            for m in range(minute, min(minute + 15, end_minute)):
                if m < total_avail_min and (m not in minute_score or score > minute_score[m]):
                    minute_score[m] = score

        if minute_score:
            target_minutes.append(
                {
                    'target': t,
                    'scores': minute_score,
                    'name': t['name'],
                }
            )

    if not target_minutes:
        return ShootingPlan(
            date=dusk_dt.strftime('%Y-%m-%d'),
            dusk=dusk,
            dawn=dawn,
            moon_phase=moon.get('phase', 'Unknown'),
            moon_illumination=moon.get('illumination', 0),
            moon_delay_min=moon_delay_min,
            slots=[],
            total_dark_min=total_dark_min,
            used_min=0,
            warnings=['No viable imaging windows'],
        )

    # For each minute, pick the best target
    schedule: list[int | None] = [None] * total_avail_min  # target index or None
    for minute in range(total_avail_min):
        best_score = -1.0
        best_idx = None
        for idx, tm in enumerate(target_minutes):
            s = tm['scores'].get(minute)
            if s is not None and s > best_score:
                best_score = s
                best_idx = idx
        schedule[minute] = best_idx

    # Merge consecutive minutes of the same target, applying min_duration and max_slot
    slots: list[ShootingSlot] = []
    m = 0
    while m < total_avail_min:
        cur = schedule[m]
        if cur is None:
            m += 1
            continue
        # Find run of same target
        run_end = m
        while run_end < total_avail_min and schedule[run_end] == cur:
            run_end += 1
        run_len = run_end - m
        # Clamp to max_slot_min
        if run_len > max_slot_min:
            run_end = m + max_slot_min
            run_len = max_slot_min
        if run_len < min_duration_min:
            m = run_end
            continue

        tm = target_minutes[cur]
        t = tm['target']
        start_dt = dusk_dt + timedelta(seconds=available_start_sec + m * 60)
        end_dt = dusk_dt + timedelta(seconds=available_start_sec + run_end * 60)

        curve = t.get('altitude_curve', [])
        start_ts = start_dt.timestamp()
        end_ts = end_dt.timestamp()
        start_alt = next(
            (p['alt'] for p in curve if abs(p['time'] - start_ts) < 60), t.get('altitude', 0)
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
                duration_min=run_len,
                fov_fit_score=t.get('fov_fit_score', 0),
                suitability_score=t.get('suitability_score', 0),
                mosaic_recommended=t.get('mosaic_recommended', False),
                notes=notes,
            )
        )
        m = run_end

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
