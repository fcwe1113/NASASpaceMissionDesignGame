import pytest

from game.base import ORBIT, SURFACE, SiteState
from game.campaign import CampaignTracker, Scenario, TerraformGoal, WON, scenarios
from game.mission import MissionControl
from game.terraform import Terraforming
from physics import ephemeris

DAY = 86_400.0
ET = ephemeris.str2et("2030-01-01")


def mars(crew=2, power=10.0, **modules):
    return SiteState("Mars", SURFACE, crew=crew, modules=dict(modules), power_kw=power)


def mars_orbit(**modules):
    return SiteState("Mars", ORBIT, modules=dict(modules))


def run(t, sites, days, start=ET, step=10):
    """Advance the tracker `days` days in `step`-day updates with the same sites."""
    et = start
    t.update(sites, et)
    finished = []
    for k in range(1, int(days / step) + 1):
        et = start + k * step * DAY
        finished += t.update(sites, et)
    return et, finished


def test_nothing_happens_without_equipment():
    t = Terraforming()
    run(t, [mars()], 1000)
    assert t.fraction == 0 and t.current == 0
    (s1, s2, s3) = t.status([mars()])
    assert s1.state == "stopped" and "Bring Atmosphere factory" in s1.text
    assert s2.state == s3.state == "waiting"


def test_equipment_needs_crew_and_power():
    t = Terraforming()
    run(t, [mars(crew=0, atmo_factory=2)], 500)
    assert t.fraction == 0
    assert "nobody is there" in t.status([mars(crew=0, atmo_factory=2)])[0].text
    run(t, [mars(power=-5, atmo_factory=2)], 500)
    assert t.fraction == 0
    assert "short of power" in t.status([mars(power=-5, atmo_factory=2)])[0].text


def test_more_equipment_is_faster_and_stages_run_in_order():
    one, three = Terraforming(), Terraforming()
    work = one.stages[0].work_days
    run(one, [mars(atmo_factory=1)], 365)
    run(three, [mars(atmo_factory=3)], 365)
    assert three.work[0] == pytest.approx(3 * one.work[0])
    # Finish stage 1 with three factories; mirrors that were already there only count from then on
    sites = [mars(atmo_factory=3), mars_orbit(mirror=2)]
    et, finished = run(three, sites, work / 3)
    assert [s.id for s in finished] == ["atmosphere"] and three.current == 1
    assert three.done_et[0] == pytest.approx(ET + work / 3 * DAY, abs=DAY)
    assert three.work[1] < 2 * 20                          # at most one update's worth on the mirrors yet
    # Orbital mirrors need no crew; factories don't help stage 2
    et, finished = run(three, [mars(crew=0, atmo_factory=3), mars_orbit(mirror=2)], 1460 / 2 + 20, start=et)
    assert [s.id for s in finished] == ["ice"]
    et, finished = run(three, [mars(biolab=4)], 1460 / 4 + 20, start=et)
    assert [s.id for s in finished] == ["life"] and three.complete and three.fraction == 1


def test_terraform_goal_and_speed_score():
    t = Terraforming()
    sc = Scenario("t", "Test", goals=[TerraformGoal("Mars")], deadline=ET + 3650 * DAY, scoring="speed")
    tracker = CampaignTracker(sc)
    mc = MissionControl()
    assert tracker.check(mc, ET, terraform=t).status == "playing"
    for k, stage in enumerate(t.stages):         # pretend every stage got done in a year
        t.work[k], t.done_et[k] = stage.work_days, ET + 365 * DAY
    v = tracker.check(mc, ET + 365 * DAY, terraform=t)
    assert v.status == WON
    assert v.score == round(40 + 60 * 0.9)       # 90% of the time left


def test_every_difficulty_has_a_scenario():
    levels = {s.difficulty for s in scenarios().values()}
    assert {"Tutorial", "Easy", "Medium", "Hard", "Sandbox"} <= levels
    hard = scenarios()["terraform_mars"]
    assert hard.scoring == "speed" and any(isinstance(g, TerraformGoal) for g in hard.goals)
