import math

import numpy as np
import pytest

from physics import kepler

MU_EARTH = 398_600.435507


def test_circular_orbit_returns_after_one_period():
    r0 = np.array([7000.0, 0, 0])
    v0 = np.array([0, math.sqrt(MU_EARTH / 7000), 0])
    period = 2 * math.pi * math.sqrt(7000 ** 3 / MU_EARTH)
    r, v = kepler.propagate(r0, v0, period, MU_EARTH)
    assert np.allclose(r, r0, atol=1e-6)
    assert np.allclose(v, v0, atol=1e-9)
    # Half a period puts it on the other side
    r, _ = kepler.propagate(r0, v0, period / 2, MU_EARTH)
    assert np.allclose(r, -r0, atol=1e-6)


def test_curtis_example_3_7():
    """Curtis, Orbital Mechanics for Engineering Students, example 3.7 (60 minutes later)."""
    r, v = kepler.propagate([7000, -12124, 0], [2.6679, 4.6210, 0], 3600, 398_600)
    assert np.allclose(r, [-3297.8, 7413.4, 0], atol=0.5)
    assert np.allclose(v, [-8.2977, -0.96309, 0], atol=1e-3)


@pytest.mark.parametrize("v0", [
    [0, 8.5, 1.0],      # ellipse
    [0, 12.0, 0.5],     # hyperbola
    [3.0, 9.0, 2.0],    # eccentric, inclined
])
def test_propagation_conserves_energy_and_momentum_and_reverses(v0):
    r0 = np.array([7000.0, 1000.0, -500.0])
    v0 = np.asarray(v0, float)
    r, v = kepler.propagate(r0, v0, 20_000, MU_EARTH)
    energy = lambda r, v: v @ v / 2 - MU_EARTH / np.linalg.norm(r)
    assert energy(r, v) == pytest.approx(energy(r0, v0), rel=1e-9)
    assert np.allclose(np.cross(r, v), np.cross(r0, v0), rtol=1e-9)
    rb, vb = kepler.propagate(r, v, -20_000, MU_EARTH)
    assert np.allclose(rb, r0, atol=1e-5)
    assert np.allclose(vb, v0, atol=1e-8)


def test_many_periods_stays_accurate():
    r0 = np.array([7000.0, 0, 0])
    v0 = np.array([0, 8.5, 0.3])
    el = kepler.elements(r0, v0, MU_EARTH)
    r, v = kepler.propagate(r0, v0, 1000 * el.period(MU_EARTH), MU_EARTH)
    assert np.allclose(r, r0, atol=1e-3)


def test_elements_of_known_orbit():
    """Curtis example 4.3 (state vector to elements)."""
    el = kepler.elements([-6045, -3490, 2500], [-3.457, 6.618, 2.533], 398_600)
    assert el.e == pytest.approx(0.1712, abs=1e-4)
    assert math.degrees(el.i) == pytest.approx(153.2, abs=0.1)
    assert math.degrees(el.raan) % 360 == pytest.approx(255.3, abs=0.1)
    assert math.degrees(el.argp) % 360 == pytest.approx(20.07, abs=0.1)
    assert math.degrees(el.nu) % 360 == pytest.approx(28.45, abs=0.1)
    assert el.a == pytest.approx(8788, abs=1)


def test_periapsis_apoapsis_and_open_orbits():
    el = kepler.elements([7000, 0, 0], [0, 8.5, 0], MU_EARTH)
    assert el.periapsis == pytest.approx(7000, rel=1e-9)
    assert el.apoapsis > 7000
    hyp = kepler.elements([7000, 0, 0], [0, 12, 0], MU_EARTH)
    assert hyp.e > 1 and hyp.apoapsis == math.inf and hyp.period(MU_EARTH) == math.inf


@pytest.mark.parametrize("r0, v0, radius, outward", [
    ([8000, 0, 0], [0, 8.5, 0], 9000, True),       # ellipse climbing past 9000 km
    ([8000, 0, 0], [0, 8.5, 0], 9000, False),      # ...and coming back down through it
    ([8000, 0, 0], [0, 12, 0], 500_000, True),     # hyperbola leaving
    ([7000, 0, 0], [-1, 5, 0], 6371, False),       # sub-orbital, falling to the surface
])
def test_time_to_radius_lands_on_that_radius(r0, v0, radius, outward):
    dt = kepler.time_to_radius(r0, v0, MU_EARTH, radius, outward)
    assert dt is not None and dt > 0
    r, v = kepler.propagate(r0, v0, dt, MU_EARTH)
    assert np.linalg.norm(r) == pytest.approx(radius, rel=1e-8)
    radial_speed = r @ v / np.linalg.norm(r)
    assert (radial_speed > 0) == outward


def test_time_to_radius_none_when_never_reached():
    # Circular orbit never changes radius; bound ellipse never reaches 1e6 km
    assert kepler.time_to_radius([7000, 0, 0], [0, math.sqrt(MU_EARTH / 7000), 0],
                                 MU_EARTH, 6371, False) is None
    assert kepler.time_to_radius([7000, 0, 0], [0, 8.5, 0], MU_EARTH, 1e6, True) is None
    # Hyperbola already heading out never comes back down
    assert kepler.time_to_radius([7000, 0, 0], [3, 12, 0], MU_EARTH, 6371, False) is None
