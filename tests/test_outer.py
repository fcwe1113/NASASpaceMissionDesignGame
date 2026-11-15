"""The outer planets and Pluto: transfers reach them, capture works, gas giants have no surface."""
import numpy as np
import pytest

from game.base import SURFACE, site_at, Arrival
from game.design import CraftDesign, assess
from game.mission import Flight
from physics import BODIES, GAS_GIANTS, PLANETS, Craft, ephemeris, fly, porkchop
from physics import planning

DAY = 86_400.0
TORCH = Craft(dry_mass=8_000, fuel=20_000, isp=6000, thrust=2_000_000)
START = ephemeris.str2et("2027-01-01")


def test_every_planet_and_pluto_is_in_the_game():
    assert PLANETS == ["Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune", "Pluto"]
    assert GAS_GIANTS == {"Jupiter", "Saturn", "Uranus", "Neptune"}
    for name in PLANETS:
        r = np.linalg.norm(ephemeris.state(BODIES[name].spice_id, START, "10")[0])
        assert 0.5 < r / BODIES[name].a < 1.5, name        # where its orbit says it should be


@pytest.mark.parametrize("planet", ["Jupiter", "Saturn", "Uranus", "Neptune", "Pluto"])
def test_transfer_then_capture_into_orbit(planet):
    a = (BODIES["Earth"].a + BODIES[planet].a) / 2
    hohmann = np.pi * np.sqrt(a ** 3 / BODIES["Sun"].gm) / DAY
    dep = START + np.arange(0, 400, 10) * DAY
    pc = porkchop("Earth", planet, dep, np.linspace(0.5, 1.5, 30) * hohmann * DAY,
                  park_alt_target=planning.PARK_ALT[planet])
    i, j = pc.best("fuel")
    depart, arrive = pc.depart_et[i], pc.depart_et[i] + pc.tof[j]
    f = Flight.to_planet("Probe", planet, TORCH, depart, arrive, pc.v_inf_depart[i, j])
    # Like the planner: aim properly a day after leaving Earth, then capture at closest approach
    exit_et = planning.soi_exit_time(f.trajectory, "Earth")
    before = fly(f.start, TORCH, f.burns, exit_et + DAY)
    burn, _ = planning.aim_burn(before.end, before.craft, planet, arrive, 5_000)
    f.set_plan(f.burns + [burn], horizon_et=arrive + 60 * DAY)
    assert planning.first_visit(f.trajectory, planet) is not None
    cap = planning.capture_burn(f.trajectory, planet)
    f.set_plan(f.burns + [cap], horizon_et=cap.et + 60 * DAY)
    assert f.outcome.kind == "orbit" and f.outcome.body == planet


def test_landing_in_a_gas_giant_kills_the_crew():
    a = Arrival(START, "Jupiter", SURFACE, "Dive: Probe", 2, 100, 100, 0, {"capsule2": 1, "shelter2": 1},
                "storable")
    s = site_at("Jupiter", SURFACE, [a], START + DAY)
    assert s.crew == 0 and s.deaths == 2 and "no surface" in s.log[-1].text


def test_planner_health_check_warns_about_gas_giant_landings():
    d = CraftDesign("medium", "storable", 1000, {"capsule2": 1, "shelter2": 1}, 2)

    class Landing:                    # just enough of a Flight for assess()
        burns = []
        launch_et, end_et = START, START + DAY
        outcome = type("O", (), {"kind": "impact", "body": "Saturn"})()
        trajectory = type("T", (), {"end": type("E", (), {"body": "Saturn"})()})()
        target = "Saturn"
    crew = next(i for i in assess(d, Landing()) if i.name == "Crew")
    assert crew.level == "bad" and "no surface" in crew.text
