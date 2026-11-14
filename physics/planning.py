"""Planning helpers: the burns a mission planner offers to work out for the player.

Every mission starts in a circular parking orbit around Earth at launch time
(getting there is the launch vehicle's job). From there the plan is just a list
of burns, so the departure burn is fuel-accounted and locked like any other.
"""
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


def aim_burn(state, craft, target, et_arrive, periapsis_alt, side=1, iterations=8):
    """An instant burn at `state` that reaches `target` passing `periapsis_alt` km above it.

    `state` is the craft just before the burn; it must be orbiting the target's
    parent (e.g. past Earth's SOI, around the Sun, for Mars). The aim point is
    offset from the target's centre across the approach direction, then refined
    by flying the result until the actual closest approach matches.
    `side` (+1/-1) picks which side of the target to pass.
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
    perp = np.cross(u, [0.0, 0.0, 1.0])
    if np.linalg.norm(perp) < 1e-6:
        perp = np.cross(u, [1.0, 0.0, 0.0])
    perp = side * perp / np.linalg.norm(perp)

    # Impact parameter that gives this periapsis for a hyperbola with excess speed vi
    rp_goal = t.radius + periapsis_alt
    b = rp_goal * math.sqrt(1 + 2 * t.gm / (rp_goal * vi ** 2))
    look = et_arrive + 0.25 * (et_arrive - state.et) + 86_400
    burn, rp = None, None
    for _ in range(iterations):
        burn = correction_burn(state, target, et_arrive, aim_offset=b * perp)
        seg = first_visit(fly(state, craft, [burn], look), target)
        if seg is None:
            raise PlanningError(f"Couldn't reach {target}'s sphere of influence from here")
        rp = seg.start.elements.periapsis
        if abs(rp - rp_goal) < 1.0:
            break
        # rp grows like b^2 when gravity bends the path a lot, like b when it doesn't;
        # the square-root step is safe for both
        b *= math.sqrt(rp_goal / max(rp, 1.0))
    return burn, rp - t.radius


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


def soi_exit_time(traj, body):
    """When the trajectory first leaves `body`'s SOI, or None."""
    return next((e.et for e in traj.events if e.kind == "soi_exit" and e.body == body), None)
