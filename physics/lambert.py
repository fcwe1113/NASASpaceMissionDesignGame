"""Lambert's problem and launch-window (porkchop) searches.

Lambert: which orbit takes you from r1 to r2 in exactly `tof` seconds?
The solver is the universal-variable method (Curtis algorithm 5.2), solved by
bisection on z so it is robust, and vectorised so a whole porkchop grid of
departure/arrival dates is solved in one go. Single-revolution transfers only.
"""
from dataclasses import dataclass

import numpy as np

from . import bodies
from .bodies import BODIES

TWO_PI_SQ = 4 * np.pi ** 2


def _stumpff(z):
    """Vectorised Stumpff C(z), S(z)."""
    z = np.asarray(z, float)
    c, s = np.empty_like(z), np.empty_like(z)
    pos, neg = z > 1e-8, z < -1e-8
    mid = ~(pos | neg)
    sp = np.sqrt(z[pos])
    c[pos] = (1 - np.cos(sp)) / z[pos]
    s[pos] = (sp - np.sin(sp)) / sp ** 3
    sn = np.sqrt(-z[neg])
    c[neg] = (np.cosh(sn) - 1) / -z[neg]
    s[neg] = (np.sinh(sn) - sn) / sn ** 3
    c[mid] = 0.5 - z[mid] / 24
    s[mid] = 1 / 6 - z[mid] / 120
    return c, s


def lambert(r1, r2, tof, mu, prograde=True, iterations=100):
    """Velocities (v1, v2) at r1 and r2 for a transfer taking `tof` seconds.

    Broadcasts over leading dimensions: r1, r2 are (..., 3), tof is (...).
    Unsolvable cases (e.g. exactly 180 degree transfers) come back as NaN.
    """
    r1, r2 = np.asarray(r1, float), np.asarray(r2, float)
    tof = np.asarray(tof, float)
    r1, r2 = np.broadcast_arrays(r1, r2)
    shape = np.broadcast_shapes(r1.shape[:-1], tof.shape)
    r1 = np.broadcast_to(r1, shape + (3,))
    r2 = np.broadcast_to(r2, shape + (3,))
    tof = np.broadcast_to(tof, shape)

    r1n, r2n = np.linalg.norm(r1, axis=-1), np.linalg.norm(r2, axis=-1)
    cos_dth = np.clip(np.sum(r1 * r2, axis=-1) / (r1n * r2n), -1, 1)
    cross_z = np.cross(r1, r2)[..., 2]
    # Prograde transfers go anticlockwise seen from +z (the way the planets orbit)
    long_way = (cross_z < 0) if prograde else (cross_z >= 0)
    sin_dth = np.sqrt(1 - cos_dth ** 2) * np.where(long_way, -1, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        A = sin_dth * np.sqrt(r1n * r2n / (1 - cos_dth))

    def y_of(z):
        c, s = _stumpff(z)
        with np.errstate(invalid="ignore", divide="ignore"):
            return r1n + r2n + A * (z * s - 1) / np.sqrt(c), c, s

    def tof_of(z):
        y, c, s = y_of(z)
        with np.errstate(invalid="ignore"):
            t = ((y / c) ** 1.5 * s + A * np.sqrt(y)) / np.sqrt(mu)
        return np.where(y > 0, t, -np.inf)     # y <= 0: shorter than any real transfer

    # Time of flight rises monotonically with z on (-inf, 4 pi^2): bisect for it.
    lo = np.full(shape, -TWO_PI_SQ)
    for _ in range(60):                         # widen the bracket for fast hyperbolas
        too_slow = tof_of(lo) > tof
        if not too_slow.any():
            break
        lo = np.where(too_slow, lo * 2, lo)
    hi = np.full(shape, TWO_PI_SQ * (1 - 1e-12))
    for _ in range(iterations):
        mid = (lo + hi) / 2
        short = tof_of(mid) < tof
        lo, hi = np.where(short, mid, lo), np.where(short, hi, mid)
    z = (lo + hi) / 2

    y, _c, _s = y_of(z)
    f = 1 - y / r1n
    g = A * np.sqrt(y / mu)
    gdot = 1 - y / r2n
    with np.errstate(invalid="ignore", divide="ignore"):
        v1 = (r2 - f[..., None] * r1) / g[..., None]
        v2 = (gdot[..., None] * r2 - r1) / g[..., None]
    return v1, v2


def escape_dv(v_inf, body, altitude):
    """Burn from a circular parking orbit at `altitude` onto a hyperbola with excess speed v_inf.

    The same number is the capture burn into such an orbit on arrival.
    """
    b = BODIES[body]
    rp = b.radius + altitude
    return np.sqrt(v_inf ** 2 + 2 * b.gm / rp) - np.sqrt(b.gm / rp)


@dataclass
class Porkchop:
    """A grid of transfers: rows are departure dates, columns are flight times."""
    origin: str
    target: str
    depart_et: np.ndarray        # (N,)
    tof: np.ndarray              # (M,) seconds
    v1: np.ndarray               # (N, M, 3) transfer velocity leaving, relative to the shared parent
    v2: np.ndarray               # (N, M, 3) transfer velocity arriving
    v_inf_depart: np.ndarray     # (N, M, 3) excess velocity relative to the origin body
    v_inf_arrive: np.ndarray     # (N, M, 3) excess velocity relative to the target body
    dv_depart: np.ndarray        # (N, M) burn from the origin parking orbit
    dv_arrive: np.ndarray        # (N, M) burn into the target parking orbit

    @property
    def dv_total(self):
        return self.dv_depart + self.dv_arrive

    @property
    def c3(self):
        """Launch energy, km^2/s^2 (what mission planners quote for departures)."""
        return np.sum(self.v_inf_depart ** 2, axis=-1)

    def best(self, by="fuel", max_dv=np.inf):
        """(i, j) of the cheapest transfer, or the fastest one costing at most max_dv.

        by="fuel" minimises total delta-v; by="speed" minimises flight time.
        Returns None if nothing on the grid fits.
        """
        dv = np.where(np.isfinite(self.dv_total), self.dv_total, np.inf)
        if by == "fuel":
            idx = np.unravel_index(np.argmin(dv), dv.shape)
            return idx if dv[idx] <= max_dv else None
        if by == "speed":
            ok = dv <= max_dv
            if not ok.any():
                return None
            tof_grid = np.where(ok, self.tof[None, :], np.inf)
            # Among the fastest, take the cheapest
            fastest = tof_grid == tof_grid.min()
            return np.unravel_index(np.argmin(np.where(fastest, dv, np.inf)), dv.shape)
        raise ValueError(f"by must be 'fuel' or 'speed', not {by!r}")


def porkchop(origin, target, depart_et, tof, park_alt_origin=200.0, park_alt_target=200.0):
    """Solve every (departure date, flight time) pair between two planets.

    Both bodies must orbit the same parent (e.g. Earth -> Mars around the Sun).
    The Moon is reachable almost any day, so Earth -> Moon trips don't need a
    window search; plan them with burns from a parking orbit instead.
    Departure and arrival burns assume circular parking orbits at the given altitudes.
    """
    o, t = BODIES[origin], BODIES[target]
    if o.parent is None or o.parent != t.parent:
        raise ValueError(f"No direct transfer from {origin} to {target}")
    center = BODIES[o.parent]
    depart_et, tof = np.asarray(depart_et, float), np.asarray(tof, float)
    arrive_et = depart_et[:, None] + tof[None, :]

    r1, vb1 = bodies.body_state(origin, depart_et)
    r2, vb2 = bodies.body_state(target, arrive_et.ravel())
    r2, vb2 = r2.reshape(arrive_et.shape + (3,)), vb2.reshape(arrive_et.shape + (3,))

    v1, v2 = lambert(r1[:, None, :], r2, tof[None, :], center.gm)
    vinf_d = v1 - vb1[:, None, :]
    vinf_a = v2 - vb2
    dv_d = escape_dv(np.linalg.norm(vinf_d, axis=-1), origin, park_alt_origin)
    dv_a = escape_dv(np.linalg.norm(vinf_a, axis=-1), target, park_alt_target)
    return Porkchop(origin, target, depart_et, tof, v1, v2, vinf_d, vinf_a, dv_d, dv_a)
