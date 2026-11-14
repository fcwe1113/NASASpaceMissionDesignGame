"""Gravity assists: finding origin -> flyby -> target routes, and aiming the flyby.

A flyby can't change the craft's speed relative to the planet it passes, only
its direction: the planet bends the path by an angle that depends on how close
the craft passes. Around the Sun that bend is a change of velocity for free.

Planning one is a three-date problem (leave, pass, arrive). Two Lambert legs give
the velocity arriving at the flyby planet (v_in) and the one needed leaving it
(v_out), both relative to the planet. The flyby works if
  * |v_in| == |v_out|   (any difference is made up with a burn: "powered flyby"), and
  * the angle between them can be turned without passing below the minimum altitude.
"""
from dataclasses import dataclass
import math

import numpy as np

from . import bodies
from .bodies import BODIES
from .lambert import escape_dv, lambert
from .planning import PlanningError, aim_burn

MIN_FLYBY_ALT = 200.0        # km; closer passes aren't allowed


def flyby_geometry(v_in, v_out, body):
    """(periapsis radius km, turn angle rad, unit aim-offset direction) for a flyby.

    The planet pulls the craft towards itself, so the velocity turns *towards*
    the planet: to turn v_in into v_out the craft passes on the side opposite to
    the change (v_out - v_in), measured across the approach direction.
    """
    v_in, v_out = np.asarray(v_in, float), np.asarray(v_out, float)
    vi, vo = np.linalg.norm(v_in), np.linalg.norm(v_out)
    delta = math.acos(max(-1.0, min(1.0, v_in @ v_out / (vi * vo))))
    v = (vi + vo) / 2
    gm = BODIES[body].gm
    rp = math.inf if delta < 1e-9 else gm / v ** 2 * (1 / math.sin(delta / 2) - 1)
    u = v_in / vi
    change = v_out - v_in
    across = change - (change @ u) * u
    n = np.linalg.norm(across)
    direction = -across / n if n > 1e-12 else np.cross(u, [0.0, 0.0, 1.0])
    return rp, delta, direction / np.linalg.norm(direction)


@dataclass
class AssistSearch:
    """A grid of origin -> via -> target routes, indexed (flyby date, leg-1 time, leg-2 time)."""
    origin: str
    via: str
    target: str
    flyby_et: np.ndarray        # (F,)
    tof1: np.ndarray            # (M,) seconds, origin -> via
    tof2: np.ndarray            # (K,) seconds, via -> target
    v_inf_depart: np.ndarray    # (F, M, 3) leaving the origin
    dv_depart: np.ndarray       # (F, M)
    dv_flyby: np.ndarray        # (F, M, K) burn to make up |v_out| - |v_in|
    dv_arrive: np.ndarray       # (F, K)
    flyby_alt: np.ndarray       # (F, M, K) km; inf for no bend at all

    @property
    def feasible(self):
        return np.isfinite(self.dv_total)

    @property
    def dv_total(self):
        t = self.dv_depart[:, :, None] + self.dv_flyby + self.dv_arrive[:, None, :]
        return np.where((self.flyby_alt >= MIN_FLYBY_ALT) & np.isfinite(t), t, np.inf)

    def best(self, by="fuel", max_dv=np.inf):
        """(f, m, k) of the cheapest route, or the fastest one costing at most max_dv; None if none fits."""
        dv = self.dv_total
        if by == "fuel":
            idx = np.unravel_index(np.argmin(dv), dv.shape)
            return idx if dv[idx] <= max_dv else None
        if by == "speed":
            ok = dv <= max_dv
            if not ok.any():
                return None
            total_time = np.where(ok, self.tof1[None, :, None] + self.tof2[None, None, :], np.inf)
            fastest = total_time == total_time.min()
            return np.unravel_index(np.argmin(np.where(fastest, dv, np.inf)), dv.shape)
        raise ValueError(f"by must be 'fuel' or 'speed', not {by!r}")

    def route(self, f, m, k):
        """(depart, flyby, arrive) times for a grid cell."""
        fb = self.flyby_et[f]
        return fb - self.tof1[m], fb, fb + self.tof2[k]


def assist_search(origin, via, target, flyby_et, tof1, tof2, park_alt_origin=200.0, park_alt_target=200.0):
    """Every (flyby date, leg-1 time, leg-2 time) route from origin past `via` to target.

    All three bodies must orbit the same parent (the Sun). Costs: departure burn
    from a parking orbit, the powered-flyby top-up, and capture into a parking
    orbit at the target. Routes needing a pass below MIN_FLYBY_ALT count as impossible.
    """
    o, v, t = BODIES[origin], BODIES[via], BODIES[target]
    if not (o.parent == v.parent == t.parent) or o.parent is None:
        raise ValueError(f"{origin}, {via} and {target} must all orbit the same body")
    if len({origin, via, target}) < 3:
        raise ValueError("The flyby body must differ from where you start and where you go")
    mu = BODIES[o.parent].gm
    flyby_et, tof1, tof2 = (np.asarray(x, float) for x in (flyby_et, tof1, tof2))
    rf, vf = bodies.body_state(via, flyby_et)

    dep = flyby_et[:, None] - tof1[None, :]
    r0, v0 = bodies.body_state(origin, dep.ravel())
    r0, v0 = r0.reshape(dep.shape + (3,)), v0.reshape(dep.shape + (3,))
    arr = flyby_et[:, None] + tof2[None, :]
    r2, v2 = bodies.body_state(target, arr.ravel())
    r2, v2 = r2.reshape(arr.shape + (3,)), v2.reshape(arr.shape + (3,))

    a1, b1 = lambert(r0, rf[:, None, :], tof1[None, :], mu)
    a2, b2 = lambert(rf[:, None, :], r2, tof2[None, :], mu)
    v_inf_dep = a1 - v0
    vin, vout = b1 - vf[:, None, :], a2 - vf[:, None, :]
    vi = np.linalg.norm(vin, axis=-1)[:, :, None]
    vo = np.linalg.norm(vout, axis=-1)[:, None, :]
    with np.errstate(invalid="ignore", divide="ignore"):
        cosd = np.einsum("fmi,fki->fmk", vin, vout) / (vi * vo)
        delta = np.arccos(np.clip(cosd, -1, 1))
        vm = (vi + vo) / 2
        rp = v.gm / vm ** 2 * (1 / np.sin(delta / 2) - 1)
    return AssistSearch(origin, via, target, flyby_et, tof1, tof2, v_inf_dep,
                        escape_dv(np.linalg.norm(v_inf_dep, axis=-1), origin, park_alt_origin),
                        np.abs(vo - vi),
                        escape_dv(np.linalg.norm(b2 - v2, axis=-1), target, park_alt_target),
                        np.where(np.isfinite(rp), rp - v.radius, np.inf))


def best_arrival(state, via, et_flyby, target, tof2=None, park_alt_target=200.0):
    """The cheapest possible arrival time at `target` for a flyby of `via` at `et_flyby`.

    `state` is the craft just before the set-up burn (orbiting the Sun). Tries
    every leg-2 time in `tof2` (default 80-500 days) and keeps those whose bend
    is possible; cost = powered-flyby top-up + capture at the target.
    """
    parent = BODIES[via].parent
    if state.body != parent:
        raise PlanningError(f"The craft must be orbiting {parent} to set up a {via} flyby "
                            f"(it is orbiting {state.body} then)")
    if et_flyby <= state.et:
        raise PlanningError("The flyby must come after the set-up burn")
    tof2 = np.linspace(80, 500, 200) * 86_400 if tof2 is None else np.asarray(tof2, float)
    mu = BODIES[parent].gm
    rf, vf = bodies.body_state(via, et_flyby)
    _, b1 = lambert(state.r, rf, et_flyby - state.et, mu)
    v_in = b1 - vf
    arrive = et_flyby + tof2
    rt, vt = bodies.body_state(target, arrive)
    a2, b2 = lambert(rf, rt, tof2, mu)
    v_out = a2 - vf
    vi, vo = np.linalg.norm(v_in), np.linalg.norm(v_out, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        delta = np.arccos(np.clip(v_out @ v_in / (vo * vi), -1, 1))
        rp = BODIES[via].gm / ((vi + vo) / 2) ** 2 * (1 / np.sin(delta / 2) - 1)
    cost = np.abs(vo - vi) + escape_dv(np.linalg.norm(b2 - vt, axis=-1), target, park_alt_target)
    cost = np.where((rp - BODIES[via].radius >= MIN_FLYBY_ALT) & np.isfinite(cost), cost, np.inf)
    k = int(np.argmin(cost))
    if not np.isfinite(cost[k]):
        raise PlanningError(f"No arrival at {target} within {tof2[0] / 86_400:.0f}-{tof2[-1] / 86_400:.0f} "
                            f"days works with a {via} flyby on that date; try another flyby date")
    return float(arrive[k])


@dataclass
class AssistPlan:
    burn: object             # the burn that sets up the flyby
    flyby_alt: float         # km, achieved closest approach
    planned_alt: float       # km, what the geometry asked for
    turn_deg: float          # how far the planet bends the path
    v_in: float              # km/s relative to the flyby planet
    v_out_needed: float      # km/s relative to the planet, leaving towards the target
    dv_flyby: float          # |v_out| - |v_in| still to make up with a burn


def assist_burn(state, craft, via, et_flyby, target, et_arrive):
    """The burn at `state` that flies past `via` so the planet bends the path towards `target`.

    `state` is just before the burn and must be orbiting the Sun (or whatever
    `via` and `target` orbit). After the flyby a small correction burn usually
    remains (see planning.aim_burn), plus any powered-flyby top-up.
    """
    parent = BODIES[via].parent
    if state.body != parent:
        raise PlanningError(f"The craft must be orbiting {parent} to set up a {via} flyby "
                            f"(it is orbiting {state.body} then)")
    if not state.et < et_flyby < et_arrive:
        raise PlanningError("The burn, the flyby and the arrival must come in that order")
    mu = BODIES[parent].gm
    rf, vf = bodies.body_state(via, et_flyby)
    rt, _ = bodies.body_state(target, et_arrive)
    _, b1 = lambert(state.r, rf, et_flyby - state.et, mu)
    a2, _ = lambert(rf, rt, et_arrive - et_flyby, mu)
    v_in, v_out = b1 - vf, a2 - vf
    rp, delta, direction = flyby_geometry(v_in, v_out, via)
    alt = rp - BODIES[via].radius
    if alt < MIN_FLYBY_ALT:
        raise PlanningError(f"{via} can't bend the path that far: it would mean passing "
                            f"{alt:,.0f} km above it (at least {MIN_FLYBY_ALT:.0f} km is needed). "
                            f"Try other dates")
    if not math.isfinite(alt):
        raise PlanningError(f"These dates need no bend at all at {via}; no flyby is needed")
    burn, got = aim_burn(state, craft, via, et_flyby, alt, offset_dir=direction)
    return AssistPlan(burn, got, alt, math.degrees(delta), float(np.linalg.norm(v_in)),
                      float(np.linalg.norm(v_out)), float(abs(np.linalg.norm(v_out) - np.linalg.norm(v_in))))
