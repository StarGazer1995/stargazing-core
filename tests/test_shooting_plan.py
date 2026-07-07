"""Tests for automatic shooting schedule generation."""

from datetime import datetime, timezone

from stargazing_core import ShootingPlan, generate_shooting_schedule

DUSK = '2024-01-15T18:00:00'
DAWN = '2024-01-16T06:00:00'
NEW_MOON = {
    'phase': 'New Moon',
    'illumination': 0.0,
    'always_down': True,
    'always_up': False,
    'altitude_curve': [],
}


def _mk_curve(alts, dusk=None):
    """Build altitude_curve from dusk + 15-min steps."""
    ts = datetime.fromisoformat(dusk or DUSK).replace(tzinfo=timezone.utc).timestamp()
    return [{'time': ts + i * 900, 'alt': a} for i, a in enumerate(alts)]


def _mk_target(name, curve, **kw):
    return {
        'name': name,
        'type': kw.get('type', 'galaxy'),
        'altitude': kw.get('altitude', 45),
        'dawn_altitude': kw.get('dawn_altitude', 30),
        'fov_fit_score': kw.get('fov_fit_score', 0.8),
        'suitability_score': kw.get('suitability_score', 80),
        'mosaic_recommended': kw.get('mosaic_recommended', False),
        'filter_match_score': kw.get('filter_match_score', 0.5),
        'altitude_curve': curve,
    }


# ── Real data tests ─────────────────────────────────────────────────────


def test_basic(redcat51_japan_feb2024):
    targets = redcat51_japan_feb2024['targets']
    moon = redcat51_japan_feb2024['moon']
    plan = generate_shooting_schedule(
        targets, moon, targets[0]['civil_dusk'], targets[0]['civil_dawn']
    )
    assert isinstance(plan, ShootingPlan)
    assert len(plan.slots) > 0
    assert plan.total_dark_min > 0
    assert 0 <= plan.moon_illumination <= 1
    for slot in plan.slots:
        assert slot.target_name
        assert slot.duration_min >= 30
        assert slot.end_alt >= 0
        assert slot.start_time < slot.end_time


def test_moon_delay(redcat51_japan_feb2024):
    moon = redcat51_japan_feb2024['moon']
    assert moon['illumination'] > 0.3 or moon['always_down']
    targets = redcat51_japan_feb2024['targets']
    plan = generate_shooting_schedule(
        targets, moon, targets[0]['civil_dusk'], targets[0]['civil_dawn']
    )
    assert plan.moon_delay_min >= 0
    assert plan.used_min + plan.moon_delay_min <= plan.total_dark_min + 10


def test_no_dark():
    moon = {'phase': 'New Moon', 'illumination': 0.0, 'always_down': True, 'altitude_curve': []}
    plan = generate_shooting_schedule([], moon, '2024-06-21T12:00:00', '2024-06-21T12:00:00')
    assert len(plan.slots) == 0
    assert len(plan.warnings) >= 1


def test_no_targets(redcat51_japan_feb2024):
    t = redcat51_japan_feb2024['targets']
    plan = generate_shooting_schedule(
        [], redcat51_japan_feb2024['moon'], t[0]['civil_dusk'], t[0]['civil_dawn']
    )
    assert len(plan.slots) == 0
    assert plan.used_min == 0


def test_fields(redcat51_japan_feb2024):
    t = redcat51_japan_feb2024['targets']
    plan = generate_shooting_schedule(
        t, redcat51_japan_feb2024['moon'], t[0]['civil_dusk'], t[0]['civil_dawn']
    )
    assert plan.date and plan.dusk and plan.dawn and plan.moon_phase
    assert isinstance(plan.moon_illumination, float)
    assert isinstance(plan.slots, list)
    assert isinstance(plan.total_dark_min, int) and isinstance(plan.used_min, int)
    assert isinstance(plan.warnings, list)


def test_slot_fields(redcat51_japan_feb2024):
    t = redcat51_japan_feb2024['targets']
    plan = generate_shooting_schedule(
        t, redcat51_japan_feb2024['moon'], t[0]['civil_dusk'], t[0]['civil_dawn'], min_alt=0
    )
    assert len(plan.slots) > 0
    s = plan.slots[0]
    assert s.target_name and s.start_time and s.end_time and s.start_time < s.end_time
    assert s.duration_min > 0
    assert isinstance(s.target_type, str)
    assert isinstance(s.fov_fit_score, float) and isinstance(s.suitability_score, float)
    assert isinstance(s.mosaic_recommended, bool) and isinstance(s.notes, list)


# ── Synthetic edge case tests ────────────────────────────────────────────


def test_moon_always_up():
    moon = {
        'phase': 'Full Moon',
        'illumination': 0.98,
        'always_down': False,
        'always_up': True,
        'altitude_curve': [],
    }
    curve = _mk_curve([30, 50, 60, 40])
    plan = generate_shooting_schedule([_mk_target('M42', curve)], moon, DUSK, DAWN)
    assert any('narrowband' in w.lower() for w in plan.warnings)


def test_no_curve_skipped():
    plan = generate_shooting_schedule([_mk_target('NoCurve', [])], NEW_MOON, DUSK, DAWN)
    assert len(plan.slots) == 0


def test_below_min_alt():
    plan = generate_shooting_schedule(
        [_mk_target('Low', _mk_curve([10, 15, 12, 8]))], NEW_MOON, DUSK, DAWN
    )
    assert len(plan.slots) == 0


def test_window_too_short():
    # ~30 min above 25° → filtered by min_duration_min=60
    curve = _mk_curve([30, 31, 29])
    plan = generate_shooting_schedule(
        [_mk_target('Brief', curve)], NEW_MOON, DUSK, DAWN, min_duration_min=60
    )
    assert len(plan.slots) == 0


def test_filter_match_note():
    curve = _mk_curve([30, 50, 60, 40])
    plan = generate_shooting_schedule(
        [_mk_target('Hα', curve, filter_match_score=0.9)], NEW_MOON, DUSK, DAWN
    )
    assert len(plan.slots) == 1
    assert any('Filter match' in n for n in plan.slots[0].notes)


def test_mosaic_note():
    curve = _mk_curve([30, 50, 60, 40])
    plan = generate_shooting_schedule(
        [_mk_target('M31', curve, mosaic_recommended=True)], NEW_MOON, DUSK, DAWN
    )
    assert len(plan.slots) == 1
    assert any('Mosaic' in n for n in plan.slots[0].notes)


def test_window_clamped_by_remaining_time():
    """Window longer than remaining dark time → clamped and skipped if too short."""
    moon = {
        'phase': 'New Moon',
        'illumination': 0.0,
        'always_down': True,
        'always_up': False,
        'altitude_curve': [],
    }
    # Long curve (135 min above 25°) in a 20-min night → clamp + skip
    curve = _mk_curve([30, 40, 50, 60, 70, 60, 50, 40, 30], '2024-01-15T18:00:00')
    plan = generate_shooting_schedule(
        [_mk_target('Big', curve)],
        moon,
        '2024-01-15T18:00:00',
        '2024-01-15T18:20:00',
        min_duration_min=30,
    )
    # Window clamped to 20 min < 30 min min_duration → skipped
    assert len(plan.slots) == 0


def test_curve_peaks_then_drops():
    """Altitude goes above min_alt, peaks, then falls below → window ends at drop."""
    curve = _mk_curve([20, 30, 50, 40, 20, 10])  # peaks then drops below 25°
    plan = generate_shooting_schedule([_mk_target('Peak', curve)], NEW_MOON, DUSK, DAWN)
    assert len(plan.slots) == 1
    # Window should end where alt drops, not continue to dawn
    assert plan.slots[0].duration_min < 360  # less than full night
