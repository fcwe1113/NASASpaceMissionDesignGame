import numpy as np
import pytest

from physics import BODIES, Craft, ephemeris, escape_dv, fly, is_stable_orbit, porkchop
from physics import planning
from physics.planning import PlanningError

DAY = 86_400.0
CRAFT = Craft(dry_mass=2000, fuel=12_000, isp=320, thrust=50_000)


@pytest.fixture(scope="module", params=["2026-09-01", "2028-09-01"])
def mars(request):
    dep = ephemeris.str2et(request.param) + np.arange(0, 120, 2) * DAY
    tof = np.arange(150, 360, 3) * DAY
    pc = porkchop("Earth", "Mars", dep, tof)
    i, j = pc.best("fuel")
    return pc.depart_et[i], pc.depart_et[i] + pc.tof[j], pc.v_inf_depart[i, j], pc.dv_depart[i, j]


def test_departure_starts_in_a_parking_orbit_with_a_prograde_burn(mars):
    depart, _, v_inf, dv = mars
    start, burn = planning.departure("Earth", depart, v_inf, 200)
    assert is_stable_orbit(start)
    assert np.linalg.norm(start.r) == pytest.approx(BODIES["Earth"].radius + 200)
    assert burn.et == depart and burn.prograde == pytest.approx(dv) and burn.dv == pytest.approx(dv)
    traj = fly(start, CRAFT, [burn], depart + 10 * DAY)
    assert planning.soi_exit_time(traj, "Earth") is not None


def test_full_mars_mission_ends_in_orbit_at_the_chosen_altitude(mars):
    depart, arrive, v_inf, _ = mars
    start, dep = planning.departure("Earth", depart, v_inf, 200)
    first = fly(start, CRAFT, [dep], depart + 10 * DAY)
    st = first.state_at(planning.soi_exit_time(first, "Earth") + DAY)
    aim, alt = planning.aim_burn(st, first.craft, "Mars", arrive, 300)
    assert alt == pytest.approx(300, abs=5)
    assert aim.dv < 0.5

    traj = fly(start, CRAFT, [dep, aim], arrive + 60 * DAY)
    # Every SOI exit lands exactly on the boundary (long hyperbolic arcs once didn't)
    for seg in traj.segments:
        if seg.end_reason == "soi_exit":
            assert np.linalg.norm(seg.state_at(seg.end_et).r) == pytest.approx(BODIES[seg.body].soi, rel=1e-9)
    cap = planning.capture_burn(traj, "Mars")
    assert cap.prograde < 0                                     # slowing down to stay
    # ...and costs what the arrival speed says it should
    visit = planning.first_visit(traj, "Mars")
    v_inf = np.sqrt(2 * visit.start.elements.energy)
    assert cap.dv == pytest.approx(escape_dv(v_inf, "Mars", 300), rel=0.01)
    traj = fly(start, CRAFT, [dep, aim, cap], arrive + 60 * DAY)
    assert traj.stable_orbit() == "Mars"
    assert traj.end.elements.periapsis - BODIES["Mars"].radius == pytest.approx(300, abs=5)
    assert traj.craft.fuel > 0


def test_aim_requires_orbiting_the_targets_parent(mars):
    depart, arrive, v_inf, _ = mars
    start, _ = planning.departure("Earth", depart, v_inf, 200)
    with pytest.raises(PlanningError, match="orbiting Sun"):
        planning.aim_burn(start, CRAFT, "Mars", arrive, 300)       # still around Earth


def test_moon_trip_from_parking_orbit():
    et = ephemeris.str2et("2026-11-20")
    start, tli = planning.moon_departure(et, 4 * DAY)
    assert is_stable_orbit(start)
    assert 2.9 < tli.dv < 3.6
    aim, alt = planning.aim_burn(start, CRAFT, "Moon", et + 4 * DAY, 100)
    assert alt == pytest.approx(100, abs=2)
    traj = fly(start, CRAFT, [aim], et + 6 * DAY)
    cap = planning.capture_burn(traj, "Moon")
    traj = fly(start, CRAFT, [aim, cap], et + 8 * DAY)
    assert traj.stable_orbit() == "Moon"


def test_capture_needs_a_visit(mars):
    depart, _, v_inf, _ = mars
    start, dep = planning.departure("Earth", depart, v_inf, 200)
    traj = fly(start, CRAFT, [dep], depart + 30 * DAY)
    with pytest.raises(PlanningError, match="never reaches"):
        planning.capture_burn(traj, "Mars")
