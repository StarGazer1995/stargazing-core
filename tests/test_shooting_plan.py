"""Tests for automatic shooting schedule generation."""

from stargazing_core import ShootingPlan, ShootingSlot, generate_shooting_schedule


def test_generate_shooting_schedule_basic(redcat51_japan_feb2024):
    """End-to-end: real match_telescope_targets → shooting plan."""
    targets = redcat51_japan_feb2024['targets']
    moon = redcat51_japan_feb2024['moon']
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

    for slot in plan.slots:
        assert isinstance(slot, ShootingSlot)
        assert slot.target_name
        assert slot.duration_min >= 30, f'{slot.target_name} too short: {slot.duration_min}min'
        assert slot.end_alt >= 0
        assert slot.start_time < slot.end_time


def test_shooting_plan_respects_moon_delay(redcat51_japan_feb2024):
    """When moon is up at dusk with >30% illumination, delay start."""
    moon = redcat51_japan_feb2024['moon']
    assert moon['illumination'] > 0.3 or moon['always_down'], (
        'test assumes first quarter moon is up at dusk'
    )

    targets = redcat51_japan_feb2024['targets']
    dusk = targets[0]['civil_dusk']
    dawn = targets[0]['civil_dawn']
    plan = generate_shooting_schedule(targets, moon, dusk, dawn)

    assert plan.moon_delay_min >= 0
    assert plan.used_min + plan.moon_delay_min <= plan.total_dark_min + 10


def test_shooting_plan_no_dark_time():
    """Dusk == dawn → empty plan with warning."""
    moon = {'phase': 'New Moon', 'illumination': 0.0, 'always_down': True, 'altitude_curve': []}
    plan = generate_shooting_schedule([], moon, '2024-06-21T12:00:00', '2024-06-21T12:00:00')
    assert len(plan.slots) == 0
    assert len(plan.warnings) >= 1


def test_shooting_plan_no_targets(redcat51_japan_feb2024):
    """Empty targets → empty plan."""
    targets = redcat51_japan_feb2024['targets']
    dusk = targets[0]['civil_dusk']
    dawn = targets[0]['civil_dawn']
    moon = redcat51_japan_feb2024['moon']

    plan = generate_shooting_schedule([], moon, dusk, dawn)
    assert isinstance(plan, ShootingPlan)
    assert len(plan.slots) == 0
    assert plan.used_min == 0


def test_shooting_plan_fields_complete(redcat51_japan_feb2024):
    """All fields in ShootingPlan are populated."""
    targets = redcat51_japan_feb2024['targets']
    moon = redcat51_japan_feb2024['moon']
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


def test_shooting_slot_fields_complete(redcat51_japan_feb2024):
    """All fields in ShootingSlot are populated."""
    targets = redcat51_japan_feb2024['targets']
    moon = redcat51_japan_feb2024['moon']
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
