import numpy as np
import pytest

from physics import kepler, ephemeris
from physics.lambert import lambert, porkchop, escape_dv

DAY = 86_400.0
MU_SUN = 132_712_440_041.279419


def test_curtis_example_5_2():
    v1, v2 = lambert([5000, 10000, 2100], [-14600, 2500, 7000], 3600, 398_600)
    assert np.allclose(v1, [-5.9925, 1.9254, 3.2456], atol=1e-3)
    assert np.allclose(v2, [-3.3125, -4.1966, -0.38529], atol=1e-3)


@pytest.mark.parametrize("r2, tof_days", [
    ([0.0, 1.5e8, 1e6], 100),         # quarter-ish turn, slow
    ([-1.2e8, 1.6e8, 0.0], 250),      # long transfer
    ([1.0e8, 1.0e8, 0.0], 10),        # fast (hyperbolic)
])
def test_solution_actually_reaches_r2(r2, tof_days):
    r1 = np.array([1.5e8, 0.0, 0.0])
    v1, v2 = lambert(r1, r2, tof_days * DAY, MU_SUN)
    r, v = kepler.propagate(r1, v1, tof_days * DAY, MU_SUN)
    assert np.allclose(r, r2, rtol=1e-6, atol=10)
    assert np.allclose(v, v2, rtol=1e-6)


def test_prograde_transfers_go_anticlockwise():
    v1, _ = lambert([1.5e8, 0, 0], [-1.2e8, -1.6e8, 0], 200 * DAY, MU_SUN)
    assert np.cross([1.5e8, 0, 0], v1)[2] > 0
    v1r, _ = lambert([1.5e8, 0, 0], [-1.2e8, -1.6e8, 0], 200 * DAY, MU_SUN, prograde=False)
    assert np.cross([1.5e8, 0, 0], v1r)[2] < 0


def test_vectorised_matches_scalar():
    r1 = np.array([1.5e8, 0.0, 0.0])
    r2s = np.array([[0.0, 1.5e8, 0.0], [-1.0e8, 1.0e8, 2e6], [1.0e8, 1.2e8, 0.0]])
    tofs = np.array([90, 150, 40]) * DAY
    v1s, v2s = lambert(r1, r2s, tofs, MU_SUN)
    for k in range(3):
        v1, v2 = lambert(r1, r2s[k], tofs[k], MU_SUN)
        assert np.allclose(v1s[k], v1) and np.allclose(v2s[k], v2)


def test_hohmann_cost_from_escape_dv():
    # Leaving LEO with v_inf = 0 costs exactly escape minus circular speed
    v_esc = np.sqrt(2 * 398_600.435507 / 6571) - np.sqrt(398_600.435507 / 6571)
    assert escape_dv(0.0, "Earth", 200) == pytest.approx(v_esc)


@pytest.fixture(scope="module")
def mars_2026():
    dep = ephemeris.str2et("2026-08-01") + np.arange(0, 180, 2) * DAY
    tof = np.arange(120, 400, 4) * DAY
    return porkchop("Earth", "Mars", dep, tof)


def test_porkchop_finds_the_2026_mars_window(mars_2026):
    """The real late-2026 Earth->Mars window: C3 around 9-10 km^2/s^2, leaving Oct-Dec 2026."""
    i, j = mars_2026.best("fuel")
    depart = ephemeris.et2str(mars_2026.depart_et[i])
    assert depart.startswith("2026") and depart[5:8] in ("OCT", "NOV", "DEC")
    assert 8 < mars_2026.c3[i, j] < 12
    assert 5 < mars_2026.dv_total[i, j] < 7


def test_porkchop_speed_trades_fuel_for_time(mars_2026):
    i, j = mars_2026.best("fuel")
    budget = mars_2026.dv_total[i, j] * 1.5
    fi, fj = mars_2026.best("speed", max_dv=budget)
    assert mars_2026.tof[fj] < mars_2026.tof[j]
    assert mars_2026.dv_total[fi, fj] <= budget
    assert mars_2026.best("fuel", max_dv=1.0) is None
    assert mars_2026.best("speed", max_dv=1.0) is None


def test_porkchop_rejects_bodies_without_a_shared_parent():
    with pytest.raises(ValueError):
        porkchop("Earth", "Moon", [0.0], [DAY])
