from dataclasses import replace

import pytest

from game.mission import (ACTIVE, COMPLETE, PENDING, Mission, MissionControl, PlanError)
from physics import Burn, Craft, ephemeris
from physics import planning

DAY = 86_400.0
ET = ephemeris.str2et("2026-11-20")
CRAFT = Craft(dry_mass=2000, fuel=12_000, isp=320, thrust=50_000)


@pytest.fixture
def moon_mission():
    return Mission.to_moon("Luna 1", CRAFT, ET, 4 * DAY, periapsis_alt=100)


def test_moon_mission_reaches_the_moon(moon_mission):
    entries = [e for e in moon_mission.trajectory.events if e.kind == "soi_entry"]
    assert entries and entries[0].body == "Moon"


def test_passing_the_target_without_braking_is_a_flyby(moon_mission):
    out = moon_mission.outcome
    assert out.kind == "flyby" and out.body == "Moon"
    assert out.altitude == pytest.approx(100, abs=2)
    assert not out.ends_mission
    assert "Flies past Moon at 100 km" in out.describe()
    # The closest approach happens inside the visit
    visit = planning.first_visit(moon_mission.trajectory, "Moon")
    assert visit.start.et <= out.et <= visit.end_et


def test_earth_orbit_mission_is_never_a_flyby():
    m = Mission.earth_orbit("Escape", CRAFT, ET)
    m.set_plan([Burn(ET + 600, prograde=3.5)], horizon_et=ET + 30 * DAY)
    assert m.outcome.kind == "in_flight" and m.outcome.body == "Sun"


def test_capture_turns_it_into_an_orbit_outcome(moon_mission):
    cap = planning.capture_burn(moon_mission.trajectory, "Moon")
    moon_mission.set_plan(moon_mission.burns + [cap], horizon_et=cap.et + 2 * DAY)
    out = moon_mission.outcome
    assert out.kind == "orbit" and out.body == "Moon" and out.ends_mission


def test_status_follows_the_clock(moon_mission):
    assert moon_mission.status == PENDING
    assert not moon_mission.update(ET - 1)
    assert moon_mission.update(ET + 60) and moon_mission.status == ACTIVE


def test_past_burns_lock_after_launch(moon_mission):
    tli = moon_mission.burns[0]
    now = ET + 3600
    moon_mission.update(now)
    assert moon_mission.is_locked(tli, now)
    with pytest.raises(PlanError):
        moon_mission.set_plan([])                                   # can't delete a burn that happened
    with pytest.raises(PlanError):
        moon_mission.set_plan([tli, Burn(ET + 600, prograde=0.1)], now=now)   # nor add one in the past
    with pytest.raises(PlanError):
        moon_mission.set_craft(replace(CRAFT, fuel=1))
    # Future burns are fine
    moon_mission.set_plan([tli, Burn(ET + DAY, prograde=0.01)], now=now)
    assert len(moon_mission.burns) == 2


def test_pending_missions_can_change_everything(moon_mission):
    moon_mission.set_craft(replace(CRAFT, fuel=20_000))
    moon_mission.set_plan([])
    assert moon_mission.outcome.kind == "orbit"                     # never left the parking orbit


def test_impact_completes_automatically():
    m = Mission.earth_orbit("Deorbit", CRAFT, ET)
    m.set_plan([Burn(ET + 600, prograde=-0.5)])
    out = m.outcome
    assert out.kind == "returned"                                   # came down on Earth
    m.update(ET + 60)
    assert m.status == ACTIVE
    m.update(out.et + 1)
    assert m.status == COMPLETE and m.end_note == "Returns to Earth"


def test_declaring_an_orbit_complete_needs_a_full_lap():
    m = Mission.earth_orbit("Sat", CRAFT, ET)
    m.update(ET + 60)
    assert not m.can_declare_complete(ET + 60)
    assert m.can_declare_complete(ET + 2 * 3600)
    with pytest.raises(PlanError):
        m.declare_complete(ET + 60)
    m.declare_complete(ET + 2 * 3600)
    assert m.status == COMPLETE and "Earth" in m.end_note


def test_cant_declare_with_burns_still_to_come():
    m = Mission.earth_orbit("Sat", CRAFT, ET)
    m.set_plan([Burn(ET + 5 * DAY, prograde=0.1)], horizon_et=ET + 10 * DAY)
    m.update(ET + DAY)
    assert not m.can_declare_complete(ET + DAY)


def test_active_mission_projection_keeps_up_with_the_clock():
    m = Mission.earth_orbit("Sat", CRAFT, ET)
    m.update(ET + 100 * DAY)
    assert m.horizon_et > ET + 100 * DAY
    assert m.state_at(ET + 100 * DAY).body == "Earth"


def test_mission_control():
    mc = MissionControl()
    a = Mission.earth_orbit("Sat", CRAFT, ET)
    mc.add(a)
    with pytest.raises(PlanError):
        mc.add(Mission.earth_orbit("Sat", CRAFT, ET))
    assert mc.unique_name("Sat") == "Sat 2"
    assert mc.update(ET + 60) == [a]
    assert mc.by_status(ACTIVE) == [a]
    with pytest.raises(PlanError):
        mc.remove(a)                                                # already launched


def test_burn_time_uses_the_fuel_left():
    ion = Craft(dry_mass=1000, fuel=100, isp=3000, thrust=0.5)
    m = Mission.earth_orbit("Ion", ion, ET)
    b = Burn(ET + 600, prograde=0.5)
    m.set_plan([b])
    assert m.burn_time(b) == pytest.approx(ion.burn_time(0.5))
    assert m.burn_time(b) > DAY
