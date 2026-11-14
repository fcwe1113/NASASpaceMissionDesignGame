"""Planning helpers: the burns a mission planner offers to work out for the player.

Every mission starts in a circular parking orbit around Earth at launch time
(getting there is the launch vehicle's job). From there the plan is just a list
of burns, so the departure burn is fuel-accounted and locked like any other.
"""
from dataclasses import dataclass
import math

import numpy as np

from . import bodies, kepler
from .bodies import BODIES
from .lambert import lambert, escape_dv
from .trajectory import (Burn, CoastSegment, State, capture_dv, correction_burn, departure_state,
                         fly, local_frame)


class PlanningError(ValueError):
    """A helper can't produce a burn for this situation (the message says why)."""


def departure(body, et, v_inf, altitude):
    """(parking-orbit state, departure burn) that leaves `body` with excess velocity v_inf.

    The parking orbit is placed so a single prograde burn at `et` puts the craft
    on the escape hyperbola whose asymptote is v_inf (see trajectory.departure_state).
    """
    s = departure_state(body, et, v_inf, altitude)
    rn = np.linalg.norm(s.r)
    v_circ = math.sqrt(BODIES[body].gm / rn) * s.v / np.linalg.norm(s.v)
    dv = escape_dv(np.linalg.norm(v_inf), body, altitude)
    return State(body, et, s.r, v_circ), Burn(et, prograde=float(dv))


def moon_departure(et, tof, altitude=200.0, lead_angle=10.0):
    """(parking-orbit state, trans-lunar burn) reaching the Moon's centre after `tof` s.

    Leaves from roughly the opposite side of Earth to where the Moon will be
    (exactly opposite is a degenerate Lambert case, hence the small lead angle).
    Use aim_burn afterwards to swap the centre aim for a proper flyby or orbit.
    """
    earth = BODIES["Earth"]
    r_moon, _ = bodies.body_state("Moon", et + tof)
    away = -r_moon / np.linalg.norm(r_moon)
    side = np.cross([0.0, 0.0, 1.0], away)
    side /= np.linalg.norm(side)
    a = math.radians(lead_angle)
    u = math.cos(a) * away + math.sin(a) * side
    rp = earth.radius + altitude
    v1, _ = lambert(rp * u, r_moon, tof, earth.gm)
    # Park in the transfer's own plane, circling the same way, so the burn is all prograde
    along = v1 - (v1 @ u) * u
    along /= np.linalg.norm(along)
    start = State("Earth", et, rp * u, math.sqrt(earth.gm / rp) * along)
    return start, _as_burn(start, v1 - start.v)


def _as_burn(state, dv_vec):
    pro, nrm, rad = local_frame(state.r, state.v)
    return Burn(state.et, float(dv_vec @ pro), float(dv_vec @ nrm), float(dv_vec @ rad))


def first_visit(traj, target):
    """The first coast segment around `target` in a trajectory, or None."""
    return next((s for s in traj.segments if s.body == target and isinstance(s, CoastSegment)), None)


def aim_burn(state, craft, target, et_arrive, periapsis_alt, side=1, iterations=8, offset_dir=None):
    """An instant burn at `state` that reaches `target` passing `periapsis_alt` km above it.

    `state` is the craft just before the burn; it must be orbiting the target's
    parent (e.g. past Earth's SOI, around the Sun, for Mars). The aim point is
    offset from the target's centre across the approach direction, then refined
    by flying the result until the actual closest approach matches.
    `side` (+1/-1) picks which side of the target to pass, in the ecliptic;
    `offset_dir` (a vector, km/s frame) overrides it with any side at all, which
    is what a gravity assist needs (see assist.assist_burn).
    Returns (burn, achieved periapsis altitude in km).
    """
    t = BODIES[target]
    if state.body != t.parent:
        raise PlanningError(f"The craft must be orbiting {t.parent} to aim at {target} "
                            f"(it is orbiting {state.body} then)")
    if et_arrive <= state.et:
        raise PlanningError("The arrival must be after the burn")
    rt, vt = bodies.body_state(target, et_arrive)
    _, v2 = lambert(state.r, rt, et_arrive - state.et, BODIES[t.parent].gm)
    v_inf = v2 - vt
    vi = np.linalg.norm(v_inf)
    u = v_inf / vi
    if offset_dir is not None:
        perp = np.asarray(offset_dir, float)
        perp = perp - (perp @ u) * u                 # only the part across the approach counts
    else:
        perp = side * np.cross(u, [0.0, 0.0, 1.0])
        if np.linalg.norm(perp) < 1e-6:
            perp = side * np.cross(u, [1.0, 0.0, 0.0])
    perp = perp / np.linalg.norm(perp)

    # Impact parameter that gives this periapsis for a hyperbola with excess speed vi
    rp_goal = t.radius + periapsis_alt
    b = rp_goal * math.sqrt(1 + 2 * t.gm / (rp_goal * vi ** 2))
    look = et_arrive + 0.25 * (et_arrive - state.et) + 86_400
    burn, rp = None, None
    for _ in range(iterations):
        burn = correction_burn(state, target, et_arrive, aim_offset=b * perp)
        if burn.dv > craft.dv_remaining:
            raise PlanningError(f"Reaching {target} then needs a {burn.dv:.2f} km/s burn, but the craft "
                                f"only has {max(0.0, craft.dv_remaining):.2f} km/s left")
        seg = first_visit(fly(state, craft, [burn], look), target)
        if seg is None:
            raise PlanningError(f"Couldn't reach {target}'s sphere of influence from here by that "
                                f"arrival time; try a different arrival time or burn time")
        rp = seg.start.elements.periapsis
        if abs(rp - rp_goal) < 1.0:
            break
        # rp grows like b^2 when gravity bends the path a lot, like b when it doesn't;
        # the square-root step is safe for both
        b *= math.sqrt(rp_goal / max(rp, 1.0))
    return burn, rp - t.radius


LANDING_DEPTH = 0.5      # arrivals aim their lowest point this deep (fraction of the radius): a sure touchdown
DEORBIT_DEPTH = 0.95     # from orbit, just below the surface is enough (and much cheaper)


def deorbit_burn(state):
    """Retrograde burn, right now, that brings the lowest point of the orbit below the surface.

    Slows the craft along its direction of travel just enough that the new
    periapsis sits at DEORBIT_DEPTH x the radius of the body it is orbiting.
    """
    b = BODIES[state.body]
    if b.parent is None:
        raise PlanningError("Can't land on the Sun")
    goal = DEORBIT_DEPTH * b.radius
    if state.elements.periapsis < b.radius:
        raise PlanningError(f"The craft already comes down on {state.body} without a burn")
    # Periapsis grows with speed (for a fixed direction): bisect the speed factor
    lo, hi = 0.0, 1.0
    for _ in range(60):
        k = (lo + hi) / 2
        if kepler.elements(state.r, state.v * k, b.gm).periapsis > goal:
            hi = k
        else:
            lo = k
    dv = (1 - lo) * np.linalg.norm(state.v)
    return Burn(state.et, prograde=-float(dv))


def best_departure_point(state, target, et_arrive, samples=72, max_period=10 * 86_400):
    """Where in its current short orbit the craft should burn for `target` (it waits for it).

    From a parking orbit the place you leave from matters a lot (leaving from the
    wrong side of Earth for the Moon costs several times more). Scans one lap with
    a quick Lambert estimate and returns the coasting state at the cheapest point.
    Long orbits (e.g. around the Sun) are returned unchanged.
    """
    el = state.elements
    period = el.period(state.mu)
    if el.e >= 1 or period > max_period:
        return state
    rt, _ = bodies.body_state(target, et_arrive)
    best, best_dv = state, math.inf
    for k in range(samples):
        t = state.et + period * k / samples
        if t >= et_arrive - 0.5 * 86_400:
            break
        r, v = kepler.propagate(state.r, state.v, t - state.et, state.mu)
        v1, _ = lambert(r, rt, et_arrive - t, state.mu)
        dv = float(np.linalg.norm(v1 - v))
        if dv < best_dv:
            best, best_dv = State(state.body, t, r, v), dv
    return best


def landing_burn(state, craft, body, et_arrive=None):
    """The burn at `state` that ends with the craft touching down on `body`.

    Orbiting `body` already: a de-orbit burn. Orbiting its parent (e.g. the Sun
    for Mars, Earth for the Moon): aims the arrival at the surface, reaching it
    around `et_arrive`. Raises PlanningError otherwise, or if fuel is short.
    """
    b = BODIES[body]
    if state.body == body:
        burn = deorbit_burn(state)
    elif state.body == b.parent:
        if et_arrive is None:
            raise PlanningError(f"Say when to arrive at {body}")
        state = best_departure_point(state, body, et_arrive)
        burn, _ = aim_burn(state, craft, body, et_arrive, (LANDING_DEPTH - 1) * b.radius)
    else:
        raise PlanningError(f"To land on {body} the craft must be orbiting {body} or {b.parent} "
                            f"when the burn starts (it is orbiting {state.body} then)")
    if burn.dv > craft.dv_remaining:
        raise PlanningError(f"Landing on {body} from there needs a {burn.dv:.2f} km/s burn, but the "
                            f"craft only has {max(0.0, craft.dv_remaining):.2f} km/s left")
    return burn


def capture_burn(traj, target):
    """Retrograde burn at the first closest approach to `target` that leaves a circular orbit."""
    seg = first_visit(traj, target)
    if seg is None:
        raise PlanningError(f"The plan never reaches {target}")
    dt = kepler.time_to_periapsis(seg.start.r, seg.start.v, seg.start.mu)
    if dt is None or seg.start.et + dt > seg.end_et:
        raise PlanningError(f"The plan leaves or hits {target} before its closest approach")
    peri = seg.state_at(seg.start.et + dt)
    if peri.elements.periapsis <= BODIES[target].radius:
        raise PlanningError(f"The plan hits {target}; aim for a higher pass first")
    return Burn(peri.et, prograde=-capture_dv(peri))


FLYBY_BODIES = ("Mercury", "Venus", "Earth", "Mars", "Moon")


@dataclass
class Approach:
    body: str
    et: float
    distance: float          # km between centres
    at_end: bool = False     # still closing in when the trajectory (projection) ends

    @property
    def altitude(self):
        return self.distance - BODIES[self.body].radius

    @property
    def inside_soi(self):
        return self.distance < BODIES[self.body].soi


def closest_approaches(traj, names=FLYBY_BODIES, samples=2000):
    """The closest the craft gets to each body over the whole trajectory, by time.

    The body the flight starts at only counts once the craft has left its SOI
    (otherwise every mission would "pass" Earth at launch). Coarse samples find
    each minimum; a golden-section search then pins it down.
    """
    t_start, t_end = traj.segments[0].start.et, traj.segments[-1].end_et
    ets = np.linspace(t_start, t_end, samples)
    craft = traj.positions(ets)
    launch_body = traj.segments[0].body
    out = []
    for name in names:
        t_from = t_start
        if name == launch_body:
            t_from = soi_exit_time(traj, name)
            if t_from is None:
                continue
        mask = ets >= t_from
        if not mask.any():
            continue
        rb, _ = bodies.heliocentric_state(name, ets[mask])
        d = np.linalg.norm(craft[mask] - rb, axis=1)
        i = int(np.argmin(d))
        if i == 0:
            continue                 # only getting further away: where we started, not a pass
        local = ets[mask]
        if i == len(local) - 1:
            out.append(Approach(name, local[-1], float(d[-1]), at_end=True))
            continue
        lo, hi = local[max(i - 1, 0)], local[i + 1]
        et = _golden_min(lambda t: _distance(traj, name, t), lo, hi)
        out.append(Approach(name, et, _distance(traj, name, et)))
    return sorted(out, key=lambda a: a.et)


def _distance(traj, name, et):
    return float(np.linalg.norm(traj.positions([et])[0] - bodies.heliocentric_state(name, et)[0]))


def _golden_min(f, lo, hi, tol=1.0):
    g = (math.sqrt(5) - 1) / 2
    a, b = lo + (1 - g) * (hi - lo), lo + g * (hi - lo)
    fa, fb = f(a), f(b)
    while hi - lo > tol:
        if fa < fb:
            hi, b, fb = b, a, fa
            a = lo + (1 - g) * (hi - lo)
            fa = f(a)
        else:
            lo, a, fa = a, b, fb
            b = lo + g * (hi - lo)
            fb = f(b)
    return (lo + hi) / 2


def soi_exit_time(traj, body):
    """When the trajectory first leaves `body`'s SOI, or None."""
    return next((e.et for e in traj.events if e.kind == "soi_exit" and e.body == body), None)
