import numpy as np
import pytest

from game.mission import Flight
from physics import BODIES, Craft, ephemeris, porkchop
from physics import planning
from physics.planning import PlanningError

DAY = 86_400.0
BIG = Craft(dry_mass=2000, fuel=60_000, isp=380, thrust=200_000)     # plenty of fuel for any test
ET = ephemeris.str2et("2026-11-20")


def assert_lands_on(flight, body):
    out = flight.outcome
    assert out.kind in ("impact", "returned") and out.body == body, out.describe()
    assert np.linalg.norm(flight.trajectory.end.r) == pytest.approx(BODIES[body].radius, rel=1e-6)


# ---------- From the parent's orbit (aimed arrival) ----------
def test_land_on_the_moon_from_a_moon_bound_parking_orbit():
    """The Moon route launches into a parking orbit lined up with the Moon: an ordinary TLI."""
    f = Flight.to_moon("Lander", BIG, ET, 4 * DAY, periapsis_alt=100)
    note = f.add_landing(ET, "Moon", ET + 4 * DAY)
    assert_lands_on(f, "Moon")
    assert "Lands on Moon" in note
    assert 2.9 < f.burns[0].dv < 3.6


def test_land_on_the_moon_from_any_parking_orbit_waits_for_the_right_moment():
    """From an arbitrary orbit it still works, burning at the cheapest point of the lap.
    (This orbit is in the ecliptic, so a plane change makes it dearer than a lined-up one.)"""
    f = Flight.earth_orbit("Lander", BIG, ET)
    f.add_landing(ET, "Moon", ET + 4 * DAY)
    assert_lands_on(f, "Moon")
    assert ET <= f.burns[0].et < ET + 2 * 3600              # within the first lap
    assert f.burns[0].dv < 4.5
    # ...and burning at the cheapest point beats burning straight away
    from physics import fly
    now = fly(f.start, BIG, [], ET).end
    v1 = planning.correction_burn(now, "Moon", ET + 4 * DAY)
    assert f.burns[0].dv < v1.dv


@pytest.mark.parametrize("planet, start", [("Mars", "2026-09-01"), ("Venus", "2026-06-01"),
                                           ("Mercury", "2026-06-01")])
def test_land_on_each_planet_from_a_transfer(planet, start):
    dep = ephemeris.str2et(start) + np.arange(0, 500, 4) * DAY
    lo, hi = {"Mars": (150, 360), "Venus": (80, 250), "Mercury": (60, 200)}[planet]
    pc = porkchop("Earth", planet, dep, np.arange(lo, hi, 3) * DAY)
    i, j = pc.best("fuel")
    depart, arrive = pc.depart_et[i], pc.depart_et[i] + pc.tof[j]
    f = Flight.to_planet("Lander", planet, BIG, depart, arrive, pc.v_inf_depart[i, j])
    exit_et = planning.soi_exit_time(f.trajectory, "Earth")
    f.add_landing(exit_et + DAY, planet, arrive)
    assert_lands_on(f, planet)
    assert abs(f.outcome.et - arrive) < 5 * DAY


# ---------- From orbit around the body (de-orbit) ----------
def test_land_from_a_parking_orbit_around_the_body():
    f = Flight.earth_orbit("Shuttle", BIG, ET)
    f.add_landing(ET + 3600, "Earth")                       # de-orbit from low Earth orbit
    assert_lands_on(f, "Earth")
    assert f.outcome.kind == "returned"                     # coming down on Earth counts as a return
    assert f.burns[-1].prograde < 0                         # a retrograde burn
    assert f.burns[-1].dv < 0.3                             # dropping from 200 km is cheap


def test_land_after_capture_into_moon_orbit():
    f = Flight.to_moon("Orbiter", BIG, ET, 4 * DAY, periapsis_alt=100)
    cap = planning.capture_burn(f.trajectory, "Moon")
    f.set_plan(f.burns + [cap], horizon_et=cap.et + 5 * DAY)
    assert f.outcome.kind == "orbit"
    f.add_landing(cap.et + DAY, "Moon")
    assert_lands_on(f, "Moon")
    assert len(f.burns) == 3


# ---------- Refusals ----------
def test_refuses_when_not_near_the_body():
    f = Flight.earth_orbit("Sat", BIG, ET)
    with pytest.raises(PlanningError, match="orbiting Mars or Sun"):
        f.add_landing(ET + 600, "Mars", ET + 200 * DAY)
    with pytest.raises(PlanningError, match="Sun"):
        planning.deorbit_burn(f.trajectory.state_at(ET).to("Sun"))


def test_refuses_without_enough_fuel_and_keeps_the_plan():
    tiny = Craft(dry_mass=2000, fuel=50, isp=320, thrust=50_000)
    f = Flight.earth_orbit("Sat", tiny, ET)
    before = list(f.burns)
    with pytest.raises(PlanningError, match="only has"):
        f.add_landing(ET, "Moon", ET + 4 * DAY)
    assert f.burns == before


def test_already_coming_down_needs_no_burn():
    f = Flight.earth_orbit("Sat", BIG, ET)
    f.add_landing(ET + 600, "Earth")
    with pytest.raises(PlanningError, match="already"):
        planning.deorbit_burn(f.trajectory.state_at(f.burns[-1].et + 60))
