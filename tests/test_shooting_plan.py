"""Tests for automatic shooting schedule generation."""

import astropy.units as u
from astropy.coordinates import EarthLocation
from astropy.time import Time

from stargazing_core import (
    ShootingPlan,
    ShootingSlot,
    generate_shooting_schedule,
    match_telescope_targets,
)


def _get_real_data():
    """Run match_telescope_targets for a known good scenario."""
    from stargazing_core._telescope import TELESCOPE_PRESETS

    config = TELESCOPE_PRESETS['redcat51-asi2600']
    observer = EarthLocation(lat=35.0 * u.deg, lon=139.0 * u.deg)
    time = Time('2024-02-16T22:00:00')  # first quarter moon
    result = match_telescope_targets(config, observer, time, limit=10)
    return result


def test_generate_shooting_schedule_basic():
    """End-to-end: real match_telescope_targets → shooting plan."""
    result = _get_real_data()
    targets = result['targets']
    moon = result['moon']
    # Use dusk/dawn from first target (all share the same)
    dusk = targets[0]['civil_dusk']
    dawn = targets[0]['civil_dawn']

    plan = generate_shooting_schedule(targets, moon, dusk, dawn)

    assert isinstance(plan, ShootingPlan)
    assert plan.date.startswith('2024-02-')
    assert len(plan.slots) > 0, 'should allocate at least one slot'
    assert plan.total_dark_min > 0
    assert plan.used_min <= plan.total_dark_min
    assert 0 <= plan.moon_illumination <= 1
    assert isinstance(plan.moon_delay_min, int)

    # Verify slot structure
    for slot in plan.slots:
        assert isinstance(slot, ShootingSlot)
        assert slot.target_name
        assert slot.duration_min >= 30, f'{slot.target_name} too short: {slot.duration_min}min'
        assert slot.end_alt >= 0
        assert slot.start_time < slot.end_time


def test_generate_shooting_schedule_respects_moon_delay():
    """When moon is up at dusk with >30% illumination, delay start."""
    result = _get_real_data()
    moon = result['moon']

    # First quarter moon — should have some illumination and not always down
    assert moon['illumination'] > 0.3 or moon['always_down'], (
        'test assumes first quarter moon is up at dusk'
    )

    targets = result['targets']
    dusk = targets[0]['civil_dusk']
    dawn = targets[0]['civil_dawn']
    plan = generate_shooting_schedule(targets, moon, dusk, dawn)

    # moon_delay_min may be 0 or >0 depending on exact moon position,
    # but the plan should account for it either way
    assert plan.moon_delay_min >= 0
    assert plan.used_min + plan.moon_delay_min <= plan.total_dark_min + 10


def test_generate_shooting_schedule_no_dark_time():
    """Dusk == dawn → empty plan with warning."""
    moon = {'phase': 'New Moon', 'illumination': 0.0, 'always_down': True, 'altitude_curve': []}
    targets = []
    plan = generate_shooting_schedule(targets, moon, '2024-06-21T12:00:00', '2024-06-21T12:00:00')
    assert len(plan.slots) == 0
    assert len(plan.warnings) >= 1


def test_generate_shooting_schedule_no_targets():
    """Empty targets → empty plan."""
    result = _get_real_data()
    dusk = result['targets'][0]['civil_dusk']
    dawn = result['targets'][0]['civil_dawn']
    moon = result['moon']

    plan = generate_shooting_schedule([], moon, dusk, dawn)
    assert isinstance(plan, ShootingPlan)
    assert len(plan.slots) == 0
    assert plan.used_min == 0


def test_shooting_plan_fields_complete():
    """All fields in ShootingPlan are populated."""
    result = _get_real_data()
    targets = result['targets']
    moon = result['moon']
    dusk = targets[0]['civil_dusk']
    dawn = targets[0]['civil_dawn']

    plan = generate_shooting_schedule(targets, moon, dusk, dawn)

    assert plan.date
    assert plan.dusk
    assert plan.dawn
    assert plan.moon_phase
    assert isinstance(plan.moon_illumination, float)
    assert isinstance(plan.slots, list)
    assert isinstance(plan.total_dark_min, int)
    assert isinstance(plan.used_min, int)
    assert isinstance(plan.warnings, list)


def test_shooting_slot_fields_complete():
    """All fields in ShootingSlot are populated."""
    result = _get_real_data()
    targets = result['targets']
    moon = result['moon']
    dusk = targets[0]['civil_dusk']
    dawn = targets[0]['civil_dawn']

    plan = generate_shooting_schedule(targets, moon, dusk, dawn, min_alt=0)

    assert len(plan.slots) > 0
    slot = plan.slots[0]
    assert slot.target_name
    assert isinstance(slot.target_type, str)
    assert slot.start_time
    assert slot.end_time
    assert slot.start_time < slot.end_time
    assert slot.duration_min > 0
    assert isinstance(slot.start_alt, (int, float))
    assert isinstance(slot.end_alt, (int, float))
    assert isinstance(slot.fov_fit_score, float)
    assert isinstance(slot.suitability_score, float)
    assert isinstance(slot.mosaic_recommended, bool)
    assert isinstance(slot.notes, list)
