import pytest

from game.design import CraftDesign
from game.mission import IDLE, Flight, Mission, MissionControl, PlanError
from game.science import Science
from physics import ephemeris

DAY = 86_400.0
ET = ephemeris.str2et("2026-11-20")
LANDER = {"capsule2": 1, "shelter2": 1, "supplies": 1, "solar_s": 1}


def moon_landing(name, modules=LANDER, crew=2):
    d = CraftDesign("medium", "storable", 12_000, dict(modules), crew)
    f = Flight.to_moon(name, d.physics_craft(), ET, 4 * DAY, periapsis_alt=100)
    f.design = d
    f.add_landing(ET, "Moon", ET + 4 * DAY)
    return f


def parked(name, body, modules):
    d = CraftDesign("medium", "storable", 1_000, {"supplies": 1, **modules}, 1)
    f = Flight.park(name, d.physics_craft(), body, ET)
    f.design = d
    return f


def control_with(*flights):
    mc = MissionControl()
    for f in flights:
        mc.add(Mission(f.name, [f]))
    return mc


def kinds(sci):
    return {(f.kind, f.body) for f in sci.firsts}


def test_starts_with_the_start_points_and_nothing_earned():
    sci = Science()
    assert sci.update(MissionControl(), ET) == []
    assert sci.balance() == sci.rules.start_points
    assert sci.per_day() == 0


def test_crewed_moon_landing_earns_visit_landing_and_crew_firsts_once():
    mc = control_with(moon_landing("Luna"))
    sci = Science()
    sci.update(mc, ET)
    assert sci.firsts == []                          # nothing before it gets there
    end = mc.missions[0].flights[0].outcome.et + DAY
    mc.update(end)
    new = sci.update(mc, end)
    assert {("visit", "Moon"), ("landing", "Moon"), ("crewed", "Moon")} <= kinds(sci)
    assert {(f.kind, f.body) for f in new} == kinds(sci)
    assert sci.update(mc, end + DAY) == []           # firsts are only news once
    r = sci.rules.firsts
    assert sci.balance() == pytest.approx(sci.rules.start_points + r["visit"]["points"]
                                          + r["landing"]["points"] + r["crewed"]["points"])


def test_crew_who_die_on_landing_dont_count():
    mc = control_with(moon_landing("Luna", {"capsule2": 1, "supplies": 1, "solar_s": 1}))
    end = mc.missions[0].flights[0].outcome.et + DAY
    mc.update(end)
    sci = Science()
    sci.update(mc, end)
    assert ("landing", "Moon") in kinds(sci) and ("crewed", "Moon") not in kinds(sci)


def test_a_science_kit_base_pays_a_first_and_a_daily_trickle():
    mc = control_with(moon_landing("Luna", {**LANDER, "science": 1}))
    end = mc.missions[0].flights[0].outcome.et + DAY
    mc.update(end)
    sci = Science()
    sci.update(mc, end)
    assert ("base", "Moon") in kinds(sci)
    rate = sci.per_day()
    assert rate == pytest.approx(sci.rules.kit_per_day * sci.rules.crewed_factor)   # crewed, Moon x1
    before = sci.balance()
    sci.update(mc, end + 10 * DAY)
    assert sci.balance() == pytest.approx(before + 10 * rate, rel=1e-6)


def test_satellite_deploys_only_from_orbit_and_earns():
    f = parked("Sat", "Moon", {"capsule2": 1, "satellite": 2})
    mc = control_with(f)
    sci = Science()
    assert sci.deploy_problem(f) is not None         # not settled yet
    mc.update(ET + 3 * DAY)
    assert f.status == IDLE and f.location() == ("Moon", "orbit")
    sci.deploy(f, ET + 3 * DAY)
    assert sci.satellites_on_board(f) == 1
    sci.update(mc, ET + 3 * DAY)
    assert {("orbit", "Moon"), ("satellite", "Moon")} <= kinds(sci)
    assert sci.per_day() == pytest.approx(sci.rules.satellite_per_day)
    # The next leg leaves the deployed satellite behind
    assert sci.without_deployed(f, f.design).modules["satellite"] == 1
    sci.deploy(f, ET + 3 * DAY)
    with pytest.raises(PlanError):
        sci.deploy(f, ET + 3 * DAY)                  # none left


def test_earth_orbit_satellite_is_a_small_first():
    f = parked("Sat", "Earth", {"capsule2": 1, "satellite": 1})
    mc = control_with(f)
    mc.update(ET + DAY)
    sci = Science()
    sci.deploy(f, ET + DAY)
    sci.update(mc, ET + DAY)
    (first,) = sci.firsts
    assert (first.kind, first.body, first.points) == ("satellite", "Earth", sci.rules.earth_satellite_points)


def test_tech_tree_locks_parts_until_unlocked():
    sci = Science()
    assert sci.is_unlocked("storable") and sci.is_unlocked("torch") and sci.is_unlocked("heavy")
    assert not sci.is_unlocked("methalox") and not sci.is_unlocked("habitat4")
    d = CraftDesign("heavy", "methalox", 1000, {"capsule2": 1, "habitat4": 1}, 1)
    assert len(sci.locked_parts(d)) == 2
    assert sci.unlock_problem("hydrolox").startswith("Needs Methalox")
    sci.unlock("methalox", ET)
    assert sci.is_unlocked("methalox")
    assert sci.balance() == sci.rules.start_points - sci.rules.tech["methalox"].cost
    with pytest.raises(PlanError):
        sci.unlock("hydrolox", ET)                   # not enough points yet


def test_moon_firsts_pay_for_all_of_tier_one():
    sci = Science()
    moon = sum(f["points"] for f in sci.rules.firsts.values()) * sci.multiplier("Moon")
    tier1 = sum(n.cost for n in sci.rules.tech.values() if not n.needs)
    assert sci.rules.start_points + moon >= tier1
