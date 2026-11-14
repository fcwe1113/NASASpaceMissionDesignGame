import math

import numpy as np
import pytest

from physics import (BODIES, Burn, Craft, State, bodies, capture_dv, correction_burn,
                     departure_state, ephemeris, escape_dv, fly, is_stable_orbit, normalize,
                     parking_orbit, porkchop)
from physics.trajectory import BurnSegment, CoastSegment, local_frame

DAY = 86_400.0
ET = ephemeris.str2et("2026-11-14")
CHEMICAL = Craft(dry_mass=2000, fuel=8000, isp=320, thrust=50_000)     # short burns
ION = Craft(dry_mass=1000, fuel=100, isp=3000, thrust=0.5)             # burns last weeks


def kinds(traj):
    return [e.kind for e in traj.events]


# ---------- Craft and burns ----------
def test_rocket_equation():
    dv = CHEMICAL.dv_remaining
    assert dv == pytest.approx(320 * 9.80665 / 1000 * math.log(10_000 / 2000))
    assert CHEMICAL.fuel_for(dv) == pytest.approx(8000)


def test_local_frame_is_orthonormal_and_radial_points_out():
    r, v = np.array([7000.0, 0, 0]), np.array([0, 7.5, 0.5])
    pro, nrm, rad = local_frame(r, v)
    m = np.array([pro, nrm, rad])
    assert np.allclose(m @ m.T, np.eye(3))
    assert rad @ r > 0


def test_instant_burn_changes_speed_and_uses_fuel():
    s = parking_orbit("Earth", ET, 400)
    traj = fly(s, CHEMICAL, [Burn(ET + 600, prograde=0.5)], ET + 1200)
    before = s.elements
    after = traj.end.elements
    assert after.energy > before.energy
    assert traj.craft.fuel == pytest.approx(8000 - CHEMICAL.fuel_for(0.5))
    assert "burn" in kinds(traj)


def test_burn_limited_by_fuel():
    small = Craft(dry_mass=2000, fuel=10, isp=320, thrust=50_000)
    traj = fly(parking_orbit("Earth", ET, 400), small, [Burn(ET + 60, prograde=2.0)], ET + 120)
    assert "out_of_fuel" in kinds(traj)
    assert traj.craft.fuel == pytest.approx(0, abs=1e-9)


def test_long_burn_spiral_matches_low_thrust_theory():
    """A slow tangential burn spirals between circular orbits: dv = v_circ_start - v_circ_end."""
    s = parking_orbit("Earth", ET, 400)
    assert ION.burn_time(1.0) > DAY                      # so it is simulated, not instant
    traj = fly(s, ION, [Burn(ET + 3600, prograde=1.0)], ET + 60 * DAY)
    assert any(isinstance(seg, BurnSegment) for seg in traj.segments)
    mu = BODIES["Earth"].gm
    expected_a = mu / (math.sqrt(mu / np.linalg.norm(s.r)) - 1.0) ** 2
    assert traj.end.elements.a == pytest.approx(expected_a, rel=1e-3)
    assert traj.end.elements.e < 1e-3
    assert traj.craft.fuel == pytest.approx(100 - ION.fuel_for(1.0))


def test_burn_segment_states_between_samples_follow_the_orbit():
    traj = fly(parking_orbit("Earth", ET, 400), ION, [Burn(ET, prograde=0.3)], ET + 20 * DAY)
    seg = next(s for s in traj.segments if isinstance(s, BurnSegment))
    r10, r11 = np.linalg.norm(seg.rs[10]), np.linalg.norm(seg.rs[11])
    r_mid = np.linalg.norm(seg.state_at((seg.ets[10] + seg.ets[11]) / 2).r)
    assert min(r10, r11) - 0.1 < r_mid < max(r10, r11) + 0.1     # no cutting across the curve
    # Landing exactly on a sample returns that sample
    assert np.allclose(seg.state_at(seg.ets[11]).r, seg.rs[11])


# ---------- Orbits, impacts and SOI changes ----------
def test_parking_orbit_is_a_stable_orbit_after_a_full_lap():
    s = parking_orbit("Earth", ET, 400)
    assert is_stable_orbit(s)
    assert fly(s, CHEMICAL, [], ET + DAY).stable_orbit() == "Earth"
    assert fly(s, CHEMICAL, [], ET + 600).stable_orbit() is None      # not a full lap yet


def test_orbit_test_rejects_escape_and_surface_grazing():
    s = parking_orbit("Earth", ET, 400)
    escaping = State("Earth", ET, s.r, s.v * 1.5)
    assert not is_stable_orbit(escaping)
    grazing = State("Earth", ET, s.r, s.v * 0.9)            # periapsis below the surface
    assert not is_stable_orbit(grazing)


def test_deorbit_burn_impacts_the_surface():
    s = parking_orbit("Earth", ET, 400)
    traj = fly(s, CHEMICAL, [Burn(ET + 60, prograde=-0.5)], ET + DAY)
    assert traj.end_reason == "impact"
    assert np.linalg.norm(traj.end.r) == pytest.approx(BODIES["Earth"].radius, rel=1e-6)
    assert traj.stable_orbit() is None


def test_escape_hands_over_to_the_sun_continuously():
    s = parking_orbit("Earth", ET, 400)
    traj = fly(s, CHEMICAL, [Burn(ET + 60, prograde=3.6)], ET + 30 * DAY)
    assert "soi_exit" in kinds(traj)
    assert traj.end.body == "Sun"
    # Just before and after the boundary, the heliocentric position/velocity agree
    exit_et = next(e.et for e in traj.events if e.kind == "soi_exit")
    k = next(i for i, seg in enumerate(traj.segments) if seg.end_reason == "soi_exit")
    before = traj.segments[k].state_at(exit_et).heliocentric()
    after = traj.segments[k + 1].state_at(exit_et).heliocentric()
    assert np.allclose(before.r, after.r, atol=1e-3)
    assert np.allclose(before.v, after.v, atol=1e-9)
    assert np.linalg.norm(traj.segments[k].state_at(exit_et).r) == pytest.approx(BODIES["Earth"].soi)


def test_normalize_puts_a_state_in_the_right_soi():
    r_moon, v_moon = bodies.body_state("Moon", ET)
    near_moon = State("Earth", ET, r_moon + [5000, 0, 0], v_moon)
    assert normalize(near_moon).body == "Moon"
    far = State("Earth", ET, [2e6, 0, 0], [0, 1, 0])
    assert normalize(far).body == "Sun"


def test_trip_to_the_moon_enters_its_soi_and_can_be_captured():
    """Lambert-aimed transfer from low Earth orbit to the Moon, then a capture burn."""
    arrive = ET + 4 * DAY
    r_moon, _ = bodies.body_state("Moon", arrive)
    # Leave from the side of Earth opposite where the Moon will be (a little off 180 degrees)
    away = -r_moon / np.linalg.norm(r_moon)
    side = np.cross([0, 0, 1], away)
    start_dir = math.cos(math.radians(10)) * away + math.sin(math.radians(10)) * side
    rp = BODIES["Earth"].radius + 300
    s = State("Earth", ET, rp * start_dir, math.sqrt(BODIES["Earth"].gm / rp) * np.cross([0, 0, 1], start_dir))
    # Aim well wide: the Lambert path ignores the Moon's own pull, which bends the
    # craft inward, so the actual closest approach is much nearer than the aim point
    tli = correction_burn(s, "Moon", arrive, aim_offset=[0, 0, 8000])
    assert 2.9 < tli.dv < 3.5                                  # classic trans-lunar injection
    traj = fly(s, CHEMICAL, [tli], arrive + DAY)
    assert "soi_entry" in kinds(traj)
    entry = next(e for e in traj.events if e.kind == "soi_entry")
    assert entry.body == "Moon"

    # Burn at closest approach to stay
    seg = next(seg for seg in traj.segments if seg.body == "Moon")
    el = seg.start.elements
    assert el.periapsis > BODIES["Moon"].radius                 # aimed past, not into, the Moon
    t_peri = seg.start.et + _time_to_periapsis(seg.start)
    peri = seg.state_at(t_peri)
    capture = Burn(t_peri, prograde=-capture_dv(peri))
    traj2 = fly(s, CHEMICAL, [tli, capture], t_peri + 2 * DAY)
    assert traj2.end.body == "Moon"
    assert traj2.stable_orbit() == "Moon"


def _time_to_periapsis(state):
    from physics import kepler
    el = state.elements
    n = math.sqrt(state.mu / abs(el.a) ** 3)
    return -kepler.mean_anomaly(el.nu, el.e) / n


# ---------- Interplanetary ----------
@pytest.fixture(scope="module")
def mars_transfer():
    dep = ephemeris.str2et("2026-09-01") + np.arange(0, 120, 2) * DAY
    tof = np.arange(150, 360, 3) * DAY
    pc = porkchop("Earth", "Mars", dep, tof)
    i, j = pc.best("fuel")
    return pc.depart_et[i], pc.depart_et[i] + pc.tof[j], pc.v_inf_depart[i, j]


def test_departure_state_leaves_with_the_requested_excess_velocity(mars_transfer):
    depart, _, v_inf = mars_transfer
    s = departure_state("Earth", depart, v_inf, 200)
    assert np.linalg.norm(s.r) == pytest.approx(BODIES["Earth"].radius + 200)
    traj = fly(s, CHEMICAL, [], depart + 10 * DAY)
    exit_et = next(e.et for e in traj.events if e.kind == "soi_exit")
    k = next(i for i, seg in enumerate(traj.segments) if seg.end_reason == "soi_exit")
    v_out = traj.segments[k].state_at(exit_et).v
    # At the SOI edge the craft is nearly on its asymptote, heading along v_inf;
    # Earth's pull there isn't quite zero, so the speed still has a bit of it in
    cos = v_out @ v_inf / (np.linalg.norm(v_out) * np.linalg.norm(v_inf))
    assert cos > 0.999
    vi = np.linalg.norm(v_inf)
    expected = math.sqrt(vi ** 2 + 2 * BODIES["Earth"].gm / BODIES["Earth"].soi)
    assert np.linalg.norm(v_out) == pytest.approx(expected, rel=1e-9)
    assert escape_dv(np.linalg.norm(v_inf), "Earth", 200) == pytest.approx(
        np.linalg.norm(s.v) - math.sqrt(BODIES["Earth"].gm / np.linalg.norm(s.r)))


def test_mid_course_correction_reaches_mars(mars_transfer):
    """Patched conics leave a small miss from the SOI offset; one correction burn fixes it."""
    depart, arrive, v_inf = mars_transfer
    s = departure_state("Earth", depart, v_inf, 200)
    uncorrected = fly(s, CHEMICAL, [], arrive + 30 * DAY)
    exit_et = next(e.et for e in uncorrected.events if e.kind == "soi_exit")

    tcm = correction_burn(uncorrected.state_at(exit_et + DAY), "Mars", arrive, aim_offset=[0, 0, 20_000])
    assert tcm.dv < 0.5                                         # a small fix, not a new transfer
    traj = fly(s, CHEMICAL, [tcm], arrive + 30 * DAY)
    entries = [e for e in traj.events if e.kind == "soi_entry"]
    assert entries and entries[0].body == "Mars"
    assert abs(entries[0].et - arrive) < 5 * DAY
    # Aimed 20,000 km off-centre: a flyby (or capture), not an impact
    assert traj.end_reason != "impact"


def test_venus_flyby_is_a_gravity_assist():
    """In Venus's frame the flyby keeps its speed; around the Sun the speed changes."""
    dep = ephemeris.str2et("2026-06-01") + np.arange(0, 400, 4) * DAY
    tof = np.arange(80, 250, 4) * DAY
    pc = porkchop("Earth", "Venus", dep, tof)
    i, j = pc.best("fuel")
    depart, arrive = pc.depart_et[i], pc.depart_et[i] + pc.tof[j]
    s = departure_state("Earth", depart, pc.v_inf_depart[i, j], 200)
    first = fly(s, CHEMICAL, [], depart + 10 * DAY)
    exit_et = next(e.et for e in first.events if e.kind == "soi_exit")
    tcm = correction_burn(first.state_at(exit_et + DAY), "Venus", arrive, aim_offset=[0, 0, 30_000])
    traj = fly(s, CHEMICAL, [tcm], arrive + 60 * DAY)

    k = next(i for i, seg in enumerate(traj.segments) if seg.body == "Venus")
    seg = traj.segments[k]
    assert seg.end_reason == "soi_exit"
    v_in, v_out = seg.start.v, seg.state_at(seg.end_et).v
    assert np.linalg.norm(v_out) == pytest.approx(np.linalg.norm(v_in), rel=1e-9)
    assert seg.start.elements.periapsis > BODIES["Venus"].radius

    helio_in = seg.start.heliocentric().v
    helio_out = seg.state_at(seg.end_et).heliocentric().v
    assert abs(np.linalg.norm(helio_out) - np.linalg.norm(helio_in)) > 0.5    # km/s gained or lost


def test_sample_gives_a_continuous_path(mars_transfer):
    depart, arrive, v_inf = mars_transfer
    traj = fly(departure_state("Earth", depart, v_inf, 200), CHEMICAL, [], depart + 100 * DAY)
    ets, pts = traj.sample(50)
    assert pts.shape == (len(ets), 3)
    assert np.all(np.diff(ets) >= 0)
    jumps = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    assert jumps.max() < 0.1 * 1.496e8          # no teleporting between samples
