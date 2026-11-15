import numpy as np
import pytest

from game.base import sites_at
from game.design import CraftDesign
from game.legs import apply_leg_start, leg_targets, next_leg_name, next_leg_start
from game.mission import COMPLETE, IDLE, PENDING, Flight, Mission, MissionControl, PlanError
from physics import BODIES, ephemeris, planning, porkchop

DAY = 86_400.0
ET = ephemeris.str2et("2026-11-20")
CREWED = {"capsule2": 1, "shelter2": 1, "supplies": 1, "solar_s": 1}


def flight_with(design, flight):
    flight.design = design
    return flight


def landed_on_moon(stage="methalox", launcher="heavy", fuel=None):
    d = CraftDesign(launcher, stage, fuel or 30_000, dict(CREWED), 2)
    f = flight_with(d, Flight.to_moon("Lander", d.physics_craft(), ET, 4 * DAY, periapsis_alt=100))
    f.add_landing(ET, "Moon", ET + 4 * DAY)
    mc = MissionControl()
    m = Mission("Luna", [f])
    mc.add(m)
    mc.update(f.outcome.et + DAY)
    return mc, m, f


def fly_on(mc, m, parent, make, et):
    """Plan a next leg with `make(design, start)` -> Flight, add it, and return it."""
    start = next_leg_start(mc, m, parent, et)
    leg = make(start.design.physics_craft(), start)
    apply_leg_start(leg, start, parent.name)
    leg.name = next_leg_name(m, parent)
    m.add_leg(leg)
    return leg, start


# ---------- Idle ----------
def test_landing_makes_the_craft_idle_and_a_base():
    mc, m, f = landed_on_moon()
    assert f.status == IDLE and f.location() == ("Moon", "surface")
    assert m.status == "active"                        # idle craft keep the mission going
    assert sites_at(mc, f.idle_since + 1)[0].crew == 2


def test_stable_orbit_becomes_idle_after_a_lap():
    d = CraftDesign("medium", "storable", 5000, dict(CREWED), 2)
    f = flight_with(d, Flight.earth_orbit("Sat", d.physics_craft(), ET))
    m = Mission("Sat", [f])
    m.update(ET + 60)
    assert f.status == "active"
    m.update(ET + 3 * 3600)
    assert f.status == IDLE and f.location() == ("Earth", "orbit")


# ---------- Flying on from a surface ----------
def test_moon_lander_comes_home():
    mc, m, f = landed_on_moon(stage="torch", launcher="heavy", fuel=20_000)
    leave = f.idle_since + 10 * DAY
    leg, start = fly_on(mc, m, f, lambda craft, s: Flight.to_parent("x", craft, "Moon", leave, 4 * DAY), leave)
    assert start.where == "surface" and start.ascent_dv == pytest.approx(1.9)
    assert start.design.fuel_kg < f.craft_at_arrival_fuel()          # take-off fuel paid
    assert start.design.crew == 2 and start.food > 0
    assert leg.name == "Lander leg 2" and leg.leg_of == "Lander"
    assert leg.outcome.kind == "returned"

    mc.update(leave + 60)
    assert f.status == COMPLETE and f.end_note == "Flew on as Lander leg 2"
    site = sites_at(mc, leave + 60)[0]
    assert site.crew == 0 and not site.modules and site.fuel == pytest.approx(0, abs=1e-6)
    assert "left with 2 crew" in site.log[-1].text

    mc.update(leg.outcome.et + 60)
    assert leg.status == COMPLETE and leg.end_note == "Returns to Earth"
    assert m.status == COMPLETE


def test_cant_take_off_without_a_relaunch_stage():
    mc, m, f = landed_on_moon(stage="storable", launcher="medium", fuel=12_000)
    with pytest.raises(PlanError, match="can't take off again"):
        next_leg_start(mc, m, f, f.idle_since + DAY)


def test_cant_take_off_without_enough_fuel():
    mc, m, f = landed_on_moon(fuel=13_000)           # enough to land, not enough to take off again
    with pytest.raises(PlanError, match="Taking off from Moon needs"):
        next_leg_start(mc, m, f, f.idle_since + DAY)


def test_dead_crew_cant_fly():
    d = CraftDesign("heavy", "methalox", 30_000, {"capsule2": 1, "supplies": 1, "solar_s": 1}, 2)  # no shelter
    f = flight_with(d, Flight.to_moon("Lander", d.physics_craft(), ET, 4 * DAY, periapsis_alt=100))
    f.add_landing(ET, "Moon", ET + 4 * DAY)
    mc = MissionControl()
    m = Mission("Luna", [f])
    mc.add(m)
    mc.update(f.outcome.et + DAY)
    with pytest.raises(PlanError, match="Nobody is left alive"):
        next_leg_start(mc, m, f, f.idle_since + DAY)


# ---------- Flying on from orbit ----------
def test_from_mars_orbit_back_to_earth():
    # The next Mars -> Earth window can be well over a year away: bring plenty of supplies
    d = CraftDesign("superheavy", "torch", 20_000, {**CREWED, "supplies_l": 1}, 2)
    dep = ephemeris.str2et("2026-09-01") + np.arange(0, 120, 2) * DAY
    pc = porkchop("Earth", "Mars", dep, np.arange(150, 360, 3) * DAY)
    i, j = pc.best("fuel")
    depart, arrive = pc.depart_et[i], pc.depart_et[i] + pc.tof[j]
    f = flight_with(d, Flight.to_planet("Ares", "Mars", d.physics_craft(), depart, arrive, pc.v_inf_depart[i, j]))
    exit_et = planning.soi_exit_time(f.trajectory, "Earth")
    before = f.trajectory.state_at(exit_et + DAY)
    aim, _ = planning.aim_burn(before, f.trajectory.craft, "Mars", arrive, 300)
    f.set_plan(f.burns + [aim])
    f.set_plan(f.burns + [planning.capture_burn(f.trajectory, "Mars")], horizon_et=arrive + 10 * DAY)
    mc = MissionControl()
    m = Mission("Ares", [f])
    mc.add(m)
    mc.update(arrive + 5 * DAY)
    assert f.status == IDLE and f.location() == ("Mars", "orbit")

    # Next Mars -> Earth window, from Mars orbit, landing on Earth
    deps = f.idle_since + np.arange(0, 800, 4) * DAY
    back = porkchop("Mars", "Earth", deps, np.arange(150, 360, 3) * DAY)
    i, j = back.best("fuel")
    leave, home = back.depart_et[i], back.depart_et[i] + back.tof[j]

    def make(craft, s):
        leg = Flight.to_planet("x", "Earth", craft, leave, home, back.v_inf_depart[i, j], origin="Mars")
        out = planning.soi_exit_time(leg.trajectory, "Mars")
        leg.add_landing(out + DAY, "Earth", home)
        return leg
    leg, start = fly_on(mc, m, f, make, leave)
    assert start.where == "orbit" and start.ascent_dv == 0
    assert leg.start.body == "Mars"
    assert leg.outcome.kind == "returned"


# ---------- Rules ----------
def test_add_leg_rules():
    mc, m, f = landed_on_moon()
    leave = f.idle_since + DAY
    start = next_leg_start(mc, m, f, leave)
    early = apply_leg_start(Flight.to_parent("Early", start.design.physics_craft(), "Moon",
                                             f.idle_since - DAY, 4 * DAY), start, f.name)
    with pytest.raises(PlanError, match="can't leave before"):
        m.add_leg(early)
    ok = apply_leg_start(Flight.to_parent("Home", start.design.physics_craft(), "Moon", leave, 4 * DAY),
                         start, f.name)
    m.add_leg(ok)
    again = apply_leg_start(Flight.to_parent("Again", start.design.physics_craft(), "Moon", leave, 4 * DAY),
                            start, f.name)
    with pytest.raises(PlanError, match="already has a next leg"):
        m.add_leg(again)


def test_leg_targets():
    assert ("Earth (back down)", "parent", "Earth") in leg_targets("Moon")
    mars = {t[2] for t in leg_targets("Mars") if t[1] == "planet"}
    assert mars == {"Mercury", "Venus", "Earth"}
    assert ("Moon", "moon", "Moon") in leg_targets("Earth")
    assert leg_targets("Venus")[-1][1] == "park"


def test_leg_names():
    mc, m, f = landed_on_moon()
    assert next_leg_name(m, f) == "Lander leg 2"
    f.name = "Lander leg 2"
    assert next_leg_name(m, f) == "Lander leg 3"
