"""Patched-conic flight: coasting, sphere-of-influence (SOI) changes, burns and fuel.

A craft always orbits exactly one body: the one whose SOI it is in. Between
events it follows an exact Kepler orbit around that body; at an SOI boundary its
state is re-expressed relative to the next body and it carries on. Events:

    soi_exit    left the current body's SOI (now orbiting its parent)
    soi_entry   entered a child body's SOI (e.g. Earth -> Moon)
    impact      hit the surface (counts as a landing in this game)
    burn        a planned burn ran
    out_of_fuel a burn was cut short

Burns are given in the craft's local frame (prograde, normal, radial-out),
relative to the body it is orbiting when the burn starts. Burns that take at
most LONG_BURN seconds are instant velocity changes; longer ones are simulated
step by step under the current body's gravity plus thrust.
"""
from dataclasses import dataclass, field, replace
import math

import numpy as np

from . import bodies, kepler
from .bodies import BODIES, CHILDREN
from .lambert import lambert, escape_dv

G0 = 9.80665                 # m/s^2, for Isp
LONG_BURN = 86_400.0         # s; shorter burns are treated as instant
MIN_STEP, MAX_STEP = 1.0, 5 * 86_400.0   # s, child-SOI search step limits
TIME_TOL = 1e-3              # s, how precisely SOI crossings are located
BURN_STEP = 6 * 3600.0       # s, max simulation step during long burns
BURN_STEP_FRACTION = 0.1     # of the local orbital period, the other step limit


# ---------- State and craft ----------
@dataclass
class State:
    """Position/velocity (km, km/s) relative to `body` at ephemeris time `et`."""
    body: str
    et: float
    r: np.ndarray
    v: np.ndarray

    def __post_init__(self):
        self.r, self.v = np.asarray(self.r, float), np.asarray(self.v, float)

    def to(self, body):
        r, v = bodies.convert(self.r, self.v, self.et, self.body, body)
        return State(body, self.et, r, v)

    def heliocentric(self):
        return self.to("Sun")

    @property
    def mu(self):
        return BODIES[self.body].gm

    @property
    def elements(self):
        return kepler.elements(self.r, self.v, self.mu)


@dataclass(frozen=True)
class Craft:
    dry_mass: float      # kg
    fuel: float          # kg
    isp: float           # s
    thrust: float        # N

    @property
    def mass(self):
        return self.dry_mass + self.fuel

    @property
    def exhaust_velocity(self):
        """km/s"""
        return self.isp * G0 / 1000

    @property
    def mass_flow(self):
        """kg/s at full thrust"""
        return self.thrust / (self.isp * G0)

    @property
    def dv_remaining(self):
        """km/s, from the rocket equation"""
        return self.exhaust_velocity * math.log(self.mass / self.dry_mass)

    def fuel_for(self, dv):
        """kg of fuel needed for a burn of `dv` km/s."""
        return self.mass * (1 - math.exp(-dv / self.exhaust_velocity))

    def burn_time(self, dv):
        return self.fuel_for(dv) / self.mass_flow


@dataclass(frozen=True)
class Burn:
    et: float
    prograde: float = 0.0    # km/s
    normal: float = 0.0
    radial: float = 0.0

    @property
    def dv(self):
        return math.sqrt(self.prograde ** 2 + self.normal ** 2 + self.radial ** 2)

    def direction(self, r, v):
        """Unit thrust direction in the inertial frame for this state."""
        pro, nrm, rad = local_frame(r, v)
        d = self.prograde * pro + self.normal * nrm + self.radial * rad
        return d / np.linalg.norm(d)


def local_frame(r, v):
    """Unit (prograde, normal, radial-out) vectors for a state."""
    pro = v / np.linalg.norm(v)
    h = np.cross(r, v)
    nrm = h / np.linalg.norm(h)
    return pro, nrm, np.cross(pro, nrm)


@dataclass
class Event:
    et: float
    kind: str
    body: str
    note: str = ""


# ---------- Segments ----------
@dataclass
class CoastSegment:
    """A Kepler arc around one body."""
    start: State
    end_et: float
    end_reason: str

    @property
    def body(self):
        return self.start.body

    def state_at(self, et):
        r, v = kepler.propagate(self.start.r, self.start.v, et - self.start.et, self.start.mu)
        return State(self.body, et, r, v)


@dataclass
class BurnSegment:
    """A long burn, stored as integration samples (relative to `body`)."""
    body: str
    ets: np.ndarray
    rs: np.ndarray
    vs: np.ndarray
    end_reason: str

    @property
    def start(self):
        return State(self.body, self.ets[0], self.rs[0], self.vs[0])

    @property
    def end_et(self):
        return self.ets[-1]

    def state_at(self, et):
        """Coast from the sample before `et`: exact gravity, ignoring one step's worth of thrust."""
        i = int(np.clip(np.searchsorted(self.ets, et, side="right") - 1, 0, len(self.ets) - 1))
        r, v = kepler.propagate(self.rs[i], self.vs[i], et - self.ets[i], BODIES[self.body].gm)
        return State(self.body, et, r, v)


@dataclass
class Trajectory:
    segments: list
    craft: Craft                     # after all burns
    events: list = field(default_factory=list)

    @property
    def end(self):
        return self.segments[-1].state_at(self.segments[-1].end_et)

    @property
    def end_reason(self):
        return self.segments[-1].end_reason

    def segment_at(self, et):
        for seg in self.segments:
            if et <= seg.end_et:
                return seg
        return self.segments[-1]

    def state_at(self, et):
        return self.segment_at(et).state_at(et)

    def sample(self, n_per_segment=100, frame="Sun"):
        """(ets, positions) along the whole flight, relative to `frame`, for drawing."""
        ets, pts = [], []
        for seg in self.segments:
            for et in np.linspace(seg.start.et, seg.end_et, n_per_segment):
                ets.append(et)
                pts.append(seg.state_at(et).to(frame).r)
        return np.array(ets), np.array(pts)

    def stable_orbit(self):
        """Name of the body the craft ends up stably orbiting, or None.

        Stable = inside that body's SOI, bound, periapsis above the surface,
        apoapsis inside the SOI, and coasting like that for at least a full lap.
        """
        last = self.segments[-1]
        if not isinstance(last, CoastSegment) or last.end_reason != "time":
            return None
        s = last.start
        if not is_stable_orbit(s):
            return None
        return s.body if last.end_et - s.et >= s.elements.period(s.mu) else None


def is_stable_orbit(state):
    """The orbit test, minus the 'held for a full lap' part (see Trajectory.stable_orbit)."""
    b = BODIES[state.body]
    el = state.elements
    return el.e < 1 and el.periapsis > b.radius and el.apoapsis < b.soi


# ---------- Coasting ----------
def normalize(state):
    """Re-express a state relative to the body whose SOI it is actually in."""
    while True:
        b = BODIES[state.body]
        if np.linalg.norm(state.r) > b.soi:
            state = state.to(b.parent)
            continue
        for c in CHILDREN[state.body]:
            rc, _ = bodies.body_state(c, state.et)
            if np.linalg.norm(state.r - rc) < BODIES[c].soi:
                state = state.to(c)
                break
        else:
            return state


def _child_gaps(seg, et, children):
    """Distance outside each child's SOI (negative = inside) and relative speeds."""
    s = seg.state_at(et)
    gaps, speeds = [], []
    for c in children:
        rc, vc = bodies.body_state(c, et)
        gaps.append(np.linalg.norm(s.r - rc) - BODIES[c].soi)
        speeds.append(np.linalg.norm(s.v - vc))
    return np.array(gaps), np.array(speeds)


def _first_child_entry(seg, t_stop):
    """(et, child) of the first child-SOI entry before t_stop, or None.

    Children can't be solved analytically (they move), so this steps forward,
    never further than half the time the craft would need to reach any SOI, and
    then bisects the crossing.
    """
    children = CHILDREN[seg.body]
    if not children:
        return None
    t = seg.start.et
    gaps, speeds = _child_gaps(seg, t, children)
    active = gaps > 0     # a child we've only just left can't be re-entered until we're clear
    while t < t_stop:
        dt = MAX_STEP
        for k in np.flatnonzero(active):
            dt = min(dt, 0.5 * gaps[k] / max(speeds[k], 1e-9))
        t_next = min(t + max(dt, MIN_STEP), t_stop)
        new_gaps, speeds = _child_gaps(seg, t_next, children)
        hit = np.flatnonzero(active & (new_gaps < 0))
        if hit.size:
            k = hit[np.argmin(new_gaps[hit])]
            lo, hi = t, t_next
            while hi - lo > TIME_TOL:
                mid = (lo + hi) / 2
                if _child_gaps(seg, mid, [children[k]])[0][0] < 0:
                    hi = mid
                else:
                    lo = mid
            return hi, children[k]
        active |= new_gaps > 0
        t, gaps = t_next, new_gaps
    return None


def coast(state, et_end, events):
    """Coast from `state` until `et_end` or an impact. Returns (segments, final state)."""
    segments = []
    while True:
        b = BODIES[state.body]
        seg = CoastSegment(state, et_end, "time")

        t_event, reason = et_end, "time"
        dt = kepler.time_to_radius(state.r, state.v, b.gm, b.radius, outward=False)
        if dt is not None and state.et + dt < t_event:
            t_event, reason = state.et + dt, "impact"
        if np.isfinite(b.soi):
            dt = kepler.time_to_radius(state.r, state.v, b.gm, b.soi, outward=True)
            if dt is not None and state.et + dt < t_event:
                t_event, reason = state.et + dt, "soi_exit"
        entry = _first_child_entry(seg, t_event)
        if entry is not None:
            t_event, reason = entry[0], "soi_entry"

        seg.end_et, seg.end_reason = t_event, reason
        segments.append(seg)
        end = seg.state_at(t_event)
        if reason == "time":
            return segments, end
        if reason == "impact":
            events.append(Event(t_event, "impact", b.name))
            return segments, end
        if reason == "soi_exit":
            events.append(Event(t_event, "soi_exit", b.name))
            state = end.to(b.parent)
        else:
            events.append(Event(t_event, "soi_entry", entry[1]))
            state = end.to(entry[1])


# ---------- Burns ----------
def _could_be_near(child, r_from_parent):
    """Cheap pre-check: only near a child's SOI if our distance from the parent is
    near the child's orbit radius (25% margin covers Mercury's eccentric orbit)."""
    c = BODIES[child]
    return abs(r_from_parent - c.a) < 0.25 * c.a + c.soi



def _impulsive_burn(state, craft, burn, events):
    dv = burn.dv
    if craft.fuel_for(dv) > craft.fuel:
        events.append(Event(state.et, "out_of_fuel", state.body,
                            f"wanted {dv:.3f} km/s, had {craft.dv_remaining:.3f}"))
        dv = craft.dv_remaining
    v = state.v + burn.direction(state.r, state.v) * dv
    craft = replace(craft, fuel=craft.fuel - craft.fuel_for(dv))
    events.append(Event(state.et, "burn", state.body, f"{dv:.3f} km/s instant"))
    return State(state.body, state.et, state.r, v), craft


def _finite_burn(state, craft, burn, events):
    """Simulate a long burn. Returns (segments, state, craft, impacted).

    Each step is half a thrust kick, an exact Kepler coast, then the other half
    kick. Gravity is therefore exact at any step size; only the slowly turning
    thrust direction is approximated. Kicks use the rocket equation, so the fuel
    used always matches the delta-v achieved.
    """
    fuel_needed = craft.fuel_for(burn.dv)
    fuel_used = min(fuel_needed, craft.fuel)
    if fuel_needed > craft.fuel:
        events.append(Event(state.et, "out_of_fuel", state.body,
                            f"wanted {burn.dv:.3f} km/s, had {craft.dv_remaining:.3f}"))
    duration = fuel_used / craft.mass_flow
    t_end = state.et + duration
    m_end = craft.mass - fuel_used
    events.append(Event(state.et, "burn", state.body, f"{burn.dv:.3f} km/s over {duration / 86_400:.1f} days"))

    segments = []
    t, r, v, m = state.et, state.r, state.v, craft.mass
    body = state.body
    ets, rs, vs = [t], [r], [v]

    def kick(r, v, m, dt):
        m_next = m - craft.mass_flow * dt
        return v + burn.direction(r, v) * craft.exhaust_velocity * math.log(m / m_next), m_next

    impacted = False
    while t < t_end - 1e-9:
        b = BODIES[body]
        rn = np.linalg.norm(r)
        # Small enough to follow the turning thrust direction, and never long enough
        # to jump right over a child's SOI (boundaries are only checked between steps)
        h = min(BURN_STEP, BURN_STEP_FRACTION * 2 * math.pi * math.sqrt(rn ** 3 / b.gm), t_end - t)
        for c in CHILDREN[body]:
            if _could_be_near(c, rn):
                rc, vc = bodies.body_state(c, t)
                gap = np.linalg.norm(r - rc) - BODIES[c].soi
                h = min(h, max(MIN_STEP, 0.5 * gap / max(np.linalg.norm(v - vc), 1e-9)))

        v, m = kick(r, v, m, h / 2)
        dt_hit = kepler.time_to_radius(r, v, b.gm, b.radius, outward=False)
        if dt_hit is not None and dt_hit < h:
            r, v = kepler.propagate(r, v, dt_hit, b.gm)
            t += dt_hit
            impacted = True
        else:
            r, v = kepler.propagate(r, v, h, b.gm)
            v, m = kick(r, v, m, h / 2)
            t += h
        ets.append(t), rs.append(r), vs.append(v)

        new_body = None
        if impacted:
            pass
        elif np.linalg.norm(r) > b.soi:
            new_body = b.parent
        else:
            for c in CHILDREN[body]:
                if not _could_be_near(c, np.linalg.norm(r)):
                    continue
                rc, _ = bodies.body_state(c, t)
                if np.linalg.norm(r - rc) < BODIES[c].soi:
                    new_body = c
                    break
        if impacted or new_body:
            reason = "impact" if impacted else ("soi_exit" if new_body == b.parent else "soi_entry")
            segments.append(BurnSegment(body, np.array(ets), np.array(rs), np.array(vs), reason))
            events.append(Event(t, reason, b.name if reason != "soi_entry" else new_body))
            if impacted:
                break
            s = State(body, t, r, v).to(new_body)
            body, r, v = new_body, s.r, s.v
            ets, rs, vs = [t], [r], [v]
    if not impacted and len(ets) > 1:
        segments.append(BurnSegment(body, np.array(ets), np.array(rs), np.array(vs), "burn"))
    # Fuel actually used up to here (all of it unless we hit the ground first)
    burned = craft.mass - m if impacted else fuel_used
    craft = replace(craft, fuel=craft.fuel - burned)
    return segments, State(body, t, r, v), craft, impacted


def fly(start, craft, burns, et_end):
    """Fly a craft from `start` through a list of burns until `et_end`.

    Stops early on impact. A burn scheduled before the previous one finished
    starts as soon as it can (noted in the events).
    """
    events = []
    segments = []
    state = normalize(start)
    for burn in sorted(burns, key=lambda b: b.et):
        if burn.et >= et_end:
            break
        if burn.et > state.et:
            segs, state = coast(state, burn.et, events)
            segments += segs
            if segs[-1].end_reason == "impact":
                return Trajectory(segments, craft, events)
        elif burn.et < state.et:
            events.append(Event(state.et, "burn_delayed", state.body,
                                f"by {(state.et - burn.et) / 86_400:.2f} days"))
        if burn.dv == 0:
            continue
        if craft.burn_time(burn.dv) <= LONG_BURN or craft.fuel <= 0:
            state, craft = _impulsive_burn(state, craft, burn, events)
        else:
            segs, state, craft, impacted = _finite_burn(state, craft, burn, events)
            segments += segs
            if impacted:
                return Trajectory(segments, craft, events)
    if et_end > state.et or not segments:
        segs, state = coast(state, et_end, events)
        segments += segs
    return Trajectory(segments, craft, events)


# ---------- Planning helpers ----------
def parking_orbit(body, et, altitude, inclination=0.0, phase=0.0):
    """A circular orbit around `body` (ecliptic-referenced, angles in radians)."""
    b = BODIES[body]
    rn = b.radius + altitude
    vn = math.sqrt(b.gm / rn)
    ci, si = math.cos(inclination), math.sin(inclination)
    cp, sp = math.cos(phase), math.sin(phase)
    r = rn * np.array([cp, sp * ci, sp * si])
    v = vn * np.array([-sp, cp * ci, cp * si])
    return State(body, et, r, v)


def departure_state(body, et, v_inf, altitude):
    """Periapsis of the escape hyperbola whose outgoing asymptote is `v_inf`.

    This is the state right after the departure burn from a parking orbit at
    `altitude` (the game's launch abstraction); the burn costs escape_dv(|v_inf|).
    The hyperbola lies in the plane containing v_inf and the ecliptic pole where possible.
    """
    b = BODIES[body]
    v_inf = np.asarray(v_inf, float)
    vi = np.linalg.norm(v_inf)
    u = v_inf / vi
    rp = b.radius + altitude
    e = 1 + rp * vi ** 2 / b.gm
    nu_inf = math.acos(-1 / e)

    n = np.array([0.0, 0.0, 1.0]) - u[2] * u            # pole, made perpendicular to u
    if np.linalg.norm(n) < 1e-6:
        n = np.array([1.0, 0.0, 0.0]) - u[0] * u
    n /= np.linalg.norm(n)
    # Periapsis direction sits nu_inf "behind" the asymptote, in the plane normal to n
    p = math.cos(nu_inf) * u - math.sin(nu_inf) * np.cross(n, u)
    q = np.cross(n, p)
    vp = math.sqrt(vi ** 2 + 2 * b.gm / rp)
    return State(body, et, rp * p, vp * q)


def correction_burn(state, target, et_arrive, aim_offset=(0.0, 0.0, 0.0)):
    """The instant burn that puts `state` on a Lambert path to `target` at `et_arrive`.

    Aims at the target's centre plus `aim_offset` (km, e.g. to set up a flyby
    instead of an impact). `state` must be orbiting the target's parent (e.g.
    heliocentric, aiming at Mars). Returns a Burn in the local frame at state.et.
    """
    parent = BODIES[target].parent
    s = state.to(parent)
    rt, _ = bodies.body_state(target, et_arrive)
    v1, _ = lambert(s.r, rt + np.asarray(aim_offset, float), et_arrive - s.et, BODIES[parent].gm)
    dv = v1 - s.v
    pro, nrm, rad = local_frame(s.r, s.v)
    return Burn(state.et, float(dv @ pro), float(dv @ nrm), float(dv @ rad))


def capture_dv(state):
    """Instant burn at periapsis that turns an arrival into a circular orbit there."""
    el = state.elements
    rp = el.periapsis
    v_peri = math.sqrt(max(0.0, 2 * (el.energy + state.mu / rp)))
    return v_peri - math.sqrt(state.mu / rp)


__all__ = ["State", "Craft", "Burn", "Event", "CoastSegment", "BurnSegment", "Trajectory",
           "fly", "coast", "normalize", "is_stable_orbit", "local_frame", "parking_orbit",
           "departure_state", "correction_burn", "capture_dv", "escape_dv", "LONG_BURN"]
