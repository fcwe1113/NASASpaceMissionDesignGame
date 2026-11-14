"""Two-body (Kepler) orbits: propagation, orbital elements and event times.

Propagation uses the universal-variable formulation (Curtis, "Orbital Mechanics
for Engineering Students", algorithms 3.3/3.4), so it handles ellipses and
hyperbolas alike. Units: km, s, km/s.
"""
from dataclasses import dataclass
import math

import numpy as np


def stumpff_c(z):
    if z > 1e-8:
        return (1 - math.cos(math.sqrt(z))) / z
    if z < -1e-8:
        return (math.cosh(math.sqrt(-z)) - 1) / -z
    return 0.5 - z / 24


def stumpff_s(z):
    if z > 1e-8:
        s = math.sqrt(z)
        return (s - math.sin(s)) / s ** 3
    if z < -1e-8:
        s = math.sqrt(-z)
        return (math.sinh(s) - s) / s ** 3
    return 1 / 6 - z / 120


def propagate(r0, v0, dt, mu):
    """State after coasting `dt` seconds (may be negative) on a Kepler orbit."""
    r0, v0 = np.asarray(r0, float), np.asarray(v0, float)
    if dt == 0:
        return r0.copy(), v0.copy()
    rn, vn = np.linalg.norm(r0), np.linalg.norm(v0)
    vr = r0 @ v0 / rn
    alpha = 2 / rn - vn ** 2 / mu          # 1/a
    sqmu = math.sqrt(mu)

    # Ellipses repeat, so drop whole periods first (keeps Newton fast and accurate)
    if alpha > 1e-12:
        period = 2 * math.pi / math.sqrt(mu * alpha ** 3)
        dt = math.fmod(dt, period)

    chi = sqmu * abs(alpha) * dt if abs(alpha) > 1e-12 else sqmu * dt / rn
    for _ in range(100):
        z = alpha * chi ** 2
        c, s = stumpff_c(z), stumpff_s(z)
        f = (rn * vr / sqmu * chi ** 2 * c + (1 - alpha * rn) * chi ** 3 * s
             + rn * chi - sqmu * dt)
        df = (rn * vr / sqmu * chi * (1 - alpha * chi ** 2 * s)
              + (1 - alpha * rn) * chi ** 2 * c + rn)
        step = f / df
        chi -= step
        if abs(step) < 1e-9 * max(1.0, abs(chi)):
            break

    z = alpha * chi ** 2
    c, s = stumpff_c(z), stumpff_s(z)
    f = 1 - chi ** 2 / rn * c
    g = dt - chi ** 3 / sqmu * s
    r = f * r0 + g * v0
    rnorm = np.linalg.norm(r)
    fdot = sqmu / (rnorm * rn) * (alpha * chi ** 3 * s - chi)
    gdot = 1 - chi ** 2 / rnorm * c
    return r, fdot * r0 + gdot * v0


@dataclass
class Elements:
    a: float          # semi-major axis, km (negative for hyperbolas, inf for parabolas)
    e: float          # eccentricity
    i: float          # inclination, rad
    raan: float       # longitude of ascending node, rad
    argp: float       # argument of periapsis, rad
    nu: float         # true anomaly, rad in (-pi, pi]
    p: float          # semi-latus rectum, km
    energy: float     # specific orbital energy, km^2/s^2

    @property
    def periapsis(self):
        return self.p / (1 + self.e)

    @property
    def apoapsis(self):
        """inf for open (parabolic/hyperbolic) orbits."""
        return self.p / (1 - self.e) if self.e < 1 else math.inf

    def period(self, mu):
        return 2 * math.pi * math.sqrt(self.a ** 3 / mu) if self.e < 1 else math.inf


def elements(r, v, mu):
    r, v = np.asarray(r, float), np.asarray(v, float)
    rn, vn = np.linalg.norm(r), np.linalg.norm(v)
    h = np.cross(r, v)
    hn = np.linalg.norm(h)
    evec = np.cross(v, h) / mu - r / rn
    e = np.linalg.norm(evec)
    energy = vn ** 2 / 2 - mu / rn
    a = -mu / (2 * energy) if abs(energy) > 1e-12 else math.inf
    p = hn ** 2 / mu

    i = math.acos(max(-1.0, min(1.0, h[2] / hn))) if hn > 0 else 0.0
    n = np.array([-h[1], h[0], 0.0])            # node line (z x h)
    nn = np.linalg.norm(n)
    raan = math.atan2(n[1], n[0]) if nn > 1e-12 else 0.0

    # In-plane angles measured from the node line (or the x axis for flat orbits)
    ref = n / nn if nn > 1e-12 else np.array([1.0, 0.0, 0.0])
    hhat = h / hn if hn > 0 else np.array([0.0, 0.0, 1.0])
    def angle_from_ref(vec):
        return math.atan2(np.cross(ref, vec) @ hhat, ref @ vec)

    if e > 1e-10:
        argp = angle_from_ref(evec)
        nu = math.atan2(np.cross(evec, r) @ hhat, evec @ r)
    else:                                        # circular: periapsis undefined, use the node
        argp = 0.0
        nu = angle_from_ref(r)
    return Elements(a, e, i, raan, argp, nu, p, energy)


def mean_anomaly(nu, e):
    """Mean anomaly for true anomaly `nu` (elliptic or hyperbolic orbits)."""
    if e < 1:
        E = 2 * math.atan(math.sqrt((1 - e) / (1 + e)) * math.tan(nu / 2))
        return E - e * math.sin(E)
    F = 2 * math.atanh(math.sqrt((e - 1) / (e + 1)) * math.tan(nu / 2))
    return e * math.sinh(F) - F


def time_between(el, nu_to, mu):
    """Seconds from the current true anomaly to `nu_to`, going forward (None if never)."""
    if el.e < 1:
        n = math.sqrt(mu / el.a ** 3)
        dm = (mean_anomaly(nu_to, el.e) - mean_anomaly(el.nu, el.e)) % (2 * math.pi)
        return dm / n
    nu_inf = math.acos(-1 / el.e)            # hyperbola only exists for |nu| < nu_inf
    if not -nu_inf < nu_to < nu_inf:
        return None
    n = math.sqrt(mu / (-el.a) ** 3)
    dt = (mean_anomaly(nu_to, el.e) - mean_anomaly(el.nu, el.e)) / n
    return dt if dt > 0 else None


def time_to_radius(r, v, mu, radius, outward):
    """Seconds until the orbit next crosses `radius` going outward (or inward).

    None if it never does. Exact, so it never misses a crossing between samples.
    """
    el = elements(r, v, mu)
    if el.e < 1e-10:
        return None                              # circular: radius never changes
    x = (el.p / radius - 1) / el.e               # cos(nu) where r == radius
    if not -1 <= x <= 1:
        return None
    nu = math.acos(x)
    return time_between(el, nu if outward else -nu, mu)
