from game.campaign import CampaignTracker, SatelliteGoal, Scenario, WON
from game.design import CraftDesign
from game.mission import COMPLETE, LOST, Flight, Mission, MissionControl
from game.report import mission_report
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


def finished(mc, f, days_after=1):
    end = f.outcome.et + days_after * DAY
    mc.update(end)
    f.declare_complete(end)
    mc.update(end)
    return end


def test_successful_landing_report_scores_high_and_lists_firsts():
    f = moon_landing("Lander")
    mc = MissionControl()
    m = Mission("Luna", [f])
    mc.add(m)
    end = finished(mc, f)
    sci = Science()
    sci.update(mc, end)
    r = mission_report(m, mc, sci)
    assert m.status == COMPLETE and r.status == "Success"
    assert r.score >= 90
    assert {x.kind for x in r.firsts} >= {"landing", "crewed"}
    text = r.as_text()
    assert "Crew: 2 of 2 alive" in text and "Fuel left" in text


def test_landing_without_habitat_places_explains_the_deaths():
    f = moon_landing("Lander", {"capsule2": 1, "supplies": 1, "solar_s": 1})
    mc = MissionControl()
    m = Mission("Luna", [f])
    mc.add(m)
    finished(mc, f)
    r = mission_report(m, mc)
    (craft,) = r.crafts
    assert craft.died == 2
    assert any("died on arrival" in what for _off, what, _tip in craft.problems)
    assert any("habitat place" in tip for tip in r.tips)
    assert r.score <= 70


def test_lost_mission_report_says_why_and_how_early():
    # One day of supplies for a four-day trip: the crew starve on the way
    d = CraftDesign("medium", "storable", 12_000, {"capsule2": 1, "shelter2": 1}, 2)
    f = Flight.to_moon("Doomed", d.physics_craft(), ET, 4 * DAY, periapsis_alt=100)
    f.design = d
    f.add_landing(ET, "Moon", ET + 4 * DAY)
    mc = MissionControl()
    m = Mission("Hungry", [f])
    mc.add(m)
    mc.update(ET + 10 * DAY)
    assert f.status == LOST
    r = mission_report(m, mc)
    assert r.status == "Lost" and r.score <= 30
    (off, what, tip) = r.problems[0]
    assert "supplies ran out" in what and "before it would have arrived" in what
    assert "supply pods" in tip


def test_satellite_goal_and_efficiency_score():
    d = CraftDesign("medium", "storable", 1_000, {"capsule2": 1, "supplies": 1, "satellite": 1}, 1)
    f = Flight.park("Sat", d.physics_craft(), "Earth", ET)
    f.design = d
    mc = MissionControl()
    mc.add(Mission("Sat", [f]))
    mc.update(ET + DAY)
    sci = Science()
    sc = Scenario("t", "Tutorial", goals=[SatelliteGoal("Earth")], budget=1000)
    tracker = CampaignTracker(sc)
    assert tracker.check(mc, ET + DAY, satellites=sci.deployments).status == "playing"
    sci.deploy(f, ET + DAY)
    v = tracker.check(mc, ET + DAY, satellites=sci.deployments)
    assert v.status == WON
    spent = mc.costs().spent
    assert v.score == round(40 + 60 * (1 - spent / 1000))
