import math

import numpy as np
import pytest

from game.mission import Mission
from physics import BODIES, Craft, ephemeris, fly
from physics import planning
from physics.assist import MIN_FLYBY_ALT, assist_burn, assist_search, flyby_geometry
from physics.planning import PlanningError

DAY = 86_400.0
CRAFT = Craft(dry_mass=2000, fuel=30_000, isp=320, thrust=50_000)


# ---------- Flyby geometry ----------
def test_flyby_geometry_matches_the_hyperbola_formula():
    v = 5.0
    delta = math.radians(40)
    v_in = np.array([v, 0, 0])
    v_out = v * np.array([math.cos(delta), math.sin(delta), 0])
    rp, turn, direction = flyby_geometry(v_in, v_out, "Venus")
    assert turn == pytest.approx(delta)
    gm = BODIES["Venus"].gm
    # Turn angle of a hyperbola: sin(delta/2) = 1 / (1 + rp v^2 / mu)
    assert math.sin(delta / 2) == pytest.approx(1 / (1 + rp * v ** 2 / gm))
    # Pass across the approach, on the side away from where the velocity turns
    assert abs(direction @ v_in) < 1e-12
    assert direction @ (v_out - v_in) < 0


def test_bigger_bends_need_closer_passes():
    v_in = np.array([5.0, 0, 0])
    rps = [flyby_geometry(v_in, 5 * np.array([math.cos(a), math.sin(a), 0]), "Venus")[0]
           for a in np.radians([10, 30, 60])]
    assert rps[0] > rps[1] > rps[2]


# ---------- Search ----------
@pytest.fixture(scope="module")
def evm_2030():
    flyby = ephemeris.str2et("2030-01-01") + np.arange(0, 730, 4) * DAY
    return assist_search("Earth", "Venus", "Mars", flyby,
                         np.linspace(60, 300, 40) * DAY, np.linspace(80, 500, 40) * DAY)


def test_search_finds_the_2030_venus_route(evm_2030):
    f, m, k = evm_2030.best("fuel")
    depart, flyby, arrive = evm_2030.route(f, m, k)
    assert ephemeris.et2str(flyby).startswith("2030")
    assert depart < flyby < arrive
    assert evm_2030.dv_total[f, m, k] < 8.5
    assert evm_2030.flyby_alt[f, m, k] >= MIN_FLYBY_ALT
    assert evm_2030.dv_flyby[f, m, k] < 0.3            # little or no top-up needed


def test_search_speed_option(evm_2030):
    cheap = evm_2030.best("fuel")
    fast = evm_2030.best("speed", max_dv=evm_2030.dv_total[cheap] * 1.3)
    t = lambda i: evm_2030.tof1[i[1]] + evm_2030.tof2[i[2]]
    assert t(fast) <= t(cheap)
    assert evm_2030.best("fuel", max_dv=1.0) is None


def test_search_needs_three_different_bodies():
    with pytest.raises(ValueError):
        assist_search("Earth", "Earth", "Mars", [0.0], [DAY], [DAY])


# ---------- Flying it ----------
@pytest.fixture(scope="module")
def evm_mission(evm_2030):
    f, m, k = evm_2030.best("fuel")
    depart, flyby, arrive = evm_2030.route(f, m, k)
    return Mission.via_assist("Venus assist", "Mars", CRAFT, depart, flyby, arrive,
                              evm_2030.v_inf_depart[f, m], "Venus"), (depart, flyby, arrive)


def test_assist_mission_passes_venus_then_reaches_mars(evm_mission):
    mission, (_, flyby, arrive) = evm_mission
    entries = [(e.body, e.et) for e in mission.trajectory.events if e.kind == "soi_entry"]
    assert [b for b, _ in entries][:2] == ["Venus", "Mars"]
    assert abs(entries[1][1] - arrive) < 5 * DAY
    venus = planning.first_visit(mission.trajectory, "Venus")
    assert venus.start.elements.periapsis - BODIES["Venus"].radius == pytest.approx(
        mission.assist.planned_alt, abs=5)
    assert mission.outcome.kind == "flyby" and mission.outcome.body == "Mars"


def test_designed_flyby_does_the_work(evm_mission):
    """Only small burns are needed besides departure: Venus provides the bend."""
    mission, _ = evm_mission
    setup, touch_up = mission.burns[1], mission.burns[2]
    assert setup.dv < 0.3 and touch_up.dv < 0.3
    assert mission.assist.turn_deg > 10


def test_flyby_bends_the_path_towards_mars(evm_mission):
    """The outgoing velocity relative to Venus points where leg 2 needs it, before any touch-up."""
    mission, (_, flyby, arrive) = evm_mission
    traj = fly(mission.start, mission.craft, mission.burns[:2], arrive)
    venus = planning.first_visit(traj, "Venus")
    v_in = venus.start.v
    v_out = venus.state_at(venus.end_et).v
    turn = math.degrees(math.acos(v_in @ v_out / (np.linalg.norm(v_in) * np.linalg.norm(v_out))))
    assert turn == pytest.approx(mission.assist.turn_deg, abs=3)


def test_impossible_bend_is_refused(evm_2030):
    f, m, k = evm_2030.best("fuel")
    depart, flyby, arrive = evm_2030.route(f, m, k)
    start, dep = planning.departure("Earth", depart, evm_2030.v_inf_depart[f, m], 200)
    first = fly(start, CRAFT, [dep], depart + 10 * DAY)
    before = fly(start, CRAFT, [dep], planning.soi_exit_time(first, "Earth") + DAY)
    # Arriving at Mars only 20 days after Venus would need a hairpin turn
    with pytest.raises(PlanningError, match="can't bend|needs a"):
        assist_burn(before.end, before.craft, "Venus", flyby, "Mars", flyby + 20 * DAY)


def test_aim_reports_missing_fuel_plainly(evm_2030):
    f, m, k = evm_2030.best("fuel")
    depart, flyby, _ = evm_2030.route(f, m, k)
    start, dep = planning.departure("Earth", depart, evm_2030.v_inf_depart[f, m], 200)
    first = fly(start, CRAFT, [dep], depart + 10 * DAY)
    before = fly(start, CRAFT, [dep], planning.soi_exit_time(first, "Earth") + DAY)
    nearly_empty = Craft(2000, 5, 320, 50_000)
    with pytest.raises(PlanningError, match="only has"):
        planning.aim_burn(before.end, nearly_empty, "Mars", flyby + 100 * DAY, 300)


def test_hand_built_assist_with_automatic_arrival(evm_2030):
    """A plain trip to Venus, then 'fly past Venus to Mars' with the arrival left to auto."""
    f, m, k = evm_2030.best("fuel")
    depart, flyby, arrive = evm_2030.route(f, m, k)
    mission = Mission.to_planet("Hand built", "Venus", CRAFT, depart, flyby, evm_2030.v_inf_depart[f, m])
    exit_et = planning.soi_exit_time(mission.trajectory, "Earth")
    note = mission.add_assist(exit_et + DAY, "Venus", flyby, "Mars")
    assert "bends the path" in note and "touch-up" in note
    bodies_entered = [e.body for e in mission.trajectory.events if e.kind == "soi_entry"]
    assert bodies_entered[:2] == ["Venus", "Mars"]
    # The automatic arrival lands near the route the 3-date search found
    mars = next(e.et for e in mission.trajectory.events if e.kind == "soi_entry" and e.body == "Mars")
    assert abs(mars - arrive) < 30 * DAY


def test_automatic_arrival_explains_when_nothing_works(evm_2030):
    f, m, k = evm_2030.best("fuel")
    depart, flyby, _ = evm_2030.route(f, m, k)
    start, dep = planning.departure("Earth", depart, evm_2030.v_inf_depart[f, m], 200)
    first = fly(start, CRAFT, [dep], depart + 10 * DAY)
    before = fly(start, CRAFT, [dep], planning.soi_exit_time(first, "Earth") + DAY)
    from physics.assist import best_arrival
    with pytest.raises(PlanningError, match="No arrival"):
        best_arrival(before.end, "Venus", flyby, "Mars", tof2=np.linspace(5, 15, 5) * DAY)


def test_outcome_names_the_last_body_passed(evm_2030):
    """A mission set up as a Venus trip, extended past Venus to Mars, reports the Mars pass."""
    f, m, k = evm_2030.best("fuel")
    depart, flyby, _ = evm_2030.route(f, m, k)
    mission = Mission.to_planet("Hand built", "Venus", CRAFT, depart, flyby, evm_2030.v_inf_depart[f, m])
    mission.add_assist(planning.soi_exit_time(mission.trajectory, "Earth") + DAY, "Venus", flyby, "Mars")
    out = mission.outcome
    assert out.kind == "flyby" and out.body == "Mars"
    assert out.altitude == pytest.approx(300, abs=5)
