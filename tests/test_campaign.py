import json

import pytest

from game.campaign import (FAILED, PLAYING, WON, BaseGoal, CampaignState, CampaignTracker, LandGoal, OrbitGoal,
                           ReturnGoal, Scenario, ScienceGoal, goal_from_spec, load_scenarios, scenarios)
from game.design import CraftDesign
from game.mission import Flight, Mission, MissionControl
from physics import ephemeris
from tests.test_legs import CREWED, fly_on, landed_on_moon

DAY = 86_400.0
ET = ephemeris.str2et("2026-11-20")


def earth_orbit(crew=2, modules=CREWED):
    d = CraftDesign("medium", "storable", 5000, dict(modules), crew)
    f = Flight.earth_orbit("Sat", d.physics_craft(), ET)
    f.design = d
    mc = MissionControl()
    mc.add(Mission("Sat", [f]))
    return mc, f


def state(mc, now, science=0.0):
    mc.update(now)
    return CampaignState.build(mc, now, science)


# ---------- Goals ----------
def test_orbit_goal_needs_the_craft_to_settle():
    mc, f = earth_orbit()
    goal = OrbitGoal("Earth")
    assert not goal.progress(state(mc, ET + 60)).done
    p = goal.progress(state(mc, ET + 3 * 3600))
    assert p.done and p.et == f.idle_since
    assert not OrbitGoal("Earth", crew=3).progress(state(mc, ET + 3 * 3600)).done
    assert not OrbitGoal("Mars").progress(state(mc, ET + 3 * 3600)).done


def test_landing_counts_only_survivors():
    mc, m, f = landed_on_moon()
    now = f.idle_since + 1
    assert LandGoal("Moon", crew=2).progress(state(mc, now)).done
    p = LandGoal("Moon", crew=3).progress(state(mc, now))
    assert not p.done and p.fraction == pytest.approx(2 / 3)


def test_landing_without_a_habitat_doesnt_count():
    d = CraftDesign("heavy", "methalox", 30_000, {"capsule2": 1, "supplies": 1, "solar_s": 1}, 2)
    f = Flight.to_moon("Lander", d.physics_craft(), ET, 4 * DAY, periapsis_alt=100)
    f.design = d
    f.add_landing(ET, "Moon", ET + 4 * DAY)
    mc = MissionControl()
    mc.add(Mission("Luna", [f]))
    p = LandGoal("Moon").progress(state(mc, f.outcome.et + DAY))
    assert not p.done and "0 crew" in p.detail


def test_return_goal_needs_a_leg_home():
    mc, m, f = landed_on_moon(stage="torch", launcher="heavy", fuel=20_000)
    goal = ReturnGoal("Moon")
    p = goal.progress(state(mc, f.idle_since + DAY))
    assert not p.done and p.fraction == 0.5
    leave = f.idle_since + 10 * DAY
    leg, _ = fly_on(mc, m, f, lambda craft, s: Flight.to_parent("x", craft, "Moon", leave, 4 * DAY), leave)
    p = goal.progress(state(mc, leg.outcome.et + 60))
    assert p.done and p.et == leg.outcome.et
    assert not ReturnGoal("Moon", where="orbit").progress(state(mc, leg.outcome.et + 60)).done


def test_base_goal_must_hold_now():
    mc, m, f = landed_on_moon()
    goal = BaseGoal("Moon", crew=2, supply_days=30, modules={"solar_s": 1})
    assert not goal.milestone
    p = goal.progress(state(mc, f.idle_since + 1))
    assert p.done, p.detail
    assert not BaseGoal("Moon", crew=2, supply_days=10_000).progress(state(mc, f.idle_since + 1)).done
    # Once the supplies run out the crew die and the base no longer counts
    (site,) = state(mc, f.idle_since + 1).sites
    later = f.idle_since + (min(site.days_left()) + 1) * DAY
    p = BaseGoal("Moon", crew=2).progress(state(mc, later))
    assert not p.done and p.detail.startswith("0/2 crew")


def test_science_goal_reads_the_points_passed_in():
    mc, _ = earth_orbit()
    goal = ScienceGoal(100)
    assert not goal.progress(state(mc, ET, science=40)).done
    assert goal.progress(state(mc, ET, science=40)).fraction == pytest.approx(0.4)
    assert goal.progress(state(mc, ET, science=100)).done


# ---------- Scenarios ----------
def test_goals_from_specs():
    g = goal_from_spec({"type": "base", "body": "Mars", "crew": 4, "supply_days": 365})
    assert g == BaseGoal("Mars", crew=4, supply_days=365)
    assert "4 crew" in g.title and "365 days" in g.title
    with pytest.raises(ValueError, match="Unknown goal"):
        goal_from_spec({"type": "conquer"})


def test_bundled_scenarios_load():
    sc = scenarios()
    assert "sandbox" in sc and sc["sandbox"].sandbox
    for s in sc.values():
        for g in s.goals:
            assert g.title


def test_scenario_file(tmp_path):
    path = tmp_path / "s.json"
    path.write_text(json.dumps({"x": {"title": "X", "goals": [{"type": "orbit", "body": "Earth"}],
                                      "deadline": "2027-01-01"}}))
    (x,) = load_scenarios(path).values()
    assert x.id == "x" and x.deadline == pytest.approx(ephemeris.str2et("2027-01-01"))


# ---------- Tracker ----------
def test_sandbox_is_never_decided():
    mc, f = earth_orbit()
    t = CampaignTracker(Scenario("sandbox", "Sandbox"))
    mc.update(ET + DAY)
    assert t.check(mc, ET + DAY).status == PLAYING


def test_winning_sticks():
    mc, m, f = landed_on_moon()
    t = CampaignTracker(Scenario("base", "Base", goals=[BaseGoal("Moon", crew=2)]))
    now = f.idle_since + 1
    mc.update(now)
    v = t.check(mc, now)
    assert v.status == WON and v.et == now and v.met == 1
    # The base dies later, but the scenario stays won
    later = now + 10_000 * DAY
    mc.update(later)
    v = t.check(mc, later)
    assert v.status == WON and v.met == 0


def test_every_goal_at_once():
    mc, f = earth_orbit()
    t = CampaignTracker(Scenario("s", "S", goals=[OrbitGoal("Earth"), ScienceGoal(10)]))
    mc.update(ET + DAY)
    assert t.check(mc, ET + DAY, science=5).status == PLAYING
    assert t.check(mc, ET + DAY, science=10).status == WON


def test_over_budget_loses():
    mc, f = earth_orbit()
    t = CampaignTracker(Scenario("s", "S", goals=[OrbitGoal("Mars")], budget=f.design.cost() - 1))
    assert t.check(mc, ET - DAY).status == PLAYING          # scheduled launches don't count yet
    mc.update(ET + 60)
    v = t.check(mc, ET + 60)
    assert v.status == FAILED and "Over budget" in v.reason


def test_deadline_loses_but_not_if_won_first():
    mc, f = earth_orbit()
    late = Scenario("s", "S", goals=[OrbitGoal("Mars")], deadline=ET + DAY)
    t = CampaignTracker(late)
    mc.update(ET + 2 * DAY)
    v = t.check(mc, ET + 2 * DAY)
    assert v.status == FAILED and v.reason == "Out of time" and v.et == ET + DAY

    t = CampaignTracker(Scenario("s", "S", goals=[OrbitGoal("Earth")], deadline=ET + 2 * DAY))
    assert t.check(mc, ET + 2 * DAY).status == WON
