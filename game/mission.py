"""Missions: a craft, a start in a parking orbit, and a plan of burns.

Life cycle:  pending -> active (at launch) -> complete | lost
  - pending: everything can change.
  - active:  the craft and launch are fixed; burns already carried out are
             locked; future burns can be added, edited or removed.
  - complete: ended by an impact (landing), returning to Earth, or the
             player declaring a stable orbit.

Missions are deterministic: the whole flight is re-flown from launch with the
current plan, and since past burns can't change, neither can the past path.
"""
from dataclasses import dataclass, field
import math

from physics import BODIES, Burn, Craft, State, fly, is_stable_orbit, parking_orbit
from physics import kepler, planning

PENDING, ACTIVE, COMPLETE, LOST = "pending", "active", "complete", "lost"
PARKING_ALT = 200.0          # km, where every mission starts
DAY = 86_400.0


class PlanError(ValueError):
    """A plan change that isn't allowed (the message says why)."""


@dataclass
class Outcome:
    kind: str        # "impact", "returned", "orbit", "flyby" or "in_flight"
    body: str        # flyby: the target passed; in_flight: the body orbited at the end
    et: float        # when it happens (orbits: final orbit starts; flybys: closest approach)
    altitude: float | None = None     # flybys: closest approach, km above the surface

    def describe(self):
        if self.kind == "flyby":
            return f"Flies past {self.body} at {self.altitude:,.0f} km and keeps going"
        return {"impact": f"Lands on {self.body}",
                "returned": "Returns to Earth",
                "orbit": f"Stable orbit around {self.body}",
                "in_flight": f"Still travelling (around {self.body}) when the projection ends"}[self.kind]

    @property
    def ends_mission(self):
        return self.kind not in ("in_flight", "flyby")


@dataclass
class Mission:
    name: str
    target: str
    craft: Craft
    start: State                       # parking orbit at launch
    burns: list = field(default_factory=list)
    horizon_et: float = 0.0            # how far ahead the plan is flown/projected
    status: str = PENDING
    ended_et: float | None = None
    end_note: str = ""

    def __post_init__(self):
        self.burns = sorted(self.burns, key=lambda b: b.et)
        if self.horizon_et <= self.launch_et:
            self.horizon_et = self.launch_et + 30 * DAY
        self._traj = None
        self.now = -math.inf           # sim time of the last update()

    # ---------- Construction ----------
    @classmethod
    def to_planet(cls, name, target, craft, depart_et, arrive_et, v_inf):
        """Departure burn onto a porkchop transfer (correction/capture burns are added separately)."""
        start, dep = planning.departure("Earth", depart_et, v_inf, PARKING_ALT)
        return cls(name, target, craft, start, [dep], horizon_et=arrive_et + 30 * DAY)

    @classmethod
    def to_moon(cls, name, craft, depart_et, tof, periapsis_alt=100.0):
        start, _ = planning.moon_departure(depart_et, tof, PARKING_ALT)
        tli, _ = planning.aim_burn(start, craft, "Moon", depart_et + tof, periapsis_alt)
        return cls(name, "Moon", craft, start, [tli], horizon_et=depart_et + tof + 5 * DAY)

    @classmethod
    def earth_orbit(cls, name, craft, launch_et, altitude=PARKING_ALT):
        start = parking_orbit("Earth", launch_et, altitude)
        return cls(name, "Earth", craft, start, [], horizon_et=launch_et + DAY)

    # ---------- Flight ----------
    @property
    def launch_et(self):
        return self.start.et

    @property
    def trajectory(self):
        if self._traj is None:
            self._traj = fly(self.start, self.craft, self.burns, self.horizon_et)
        return self._traj

    def _changed(self):
        self._traj = None

    def state_at(self, et):
        return self.trajectory.state_at(min(et, self.trajectory.segments[-1].end_et))

    @property
    def outcome(self):
        traj = self.trajectory
        end = traj.end
        if traj.end_reason == "impact":
            # Coming down on Earth after the flight is a return home; the launch itself isn't
            if end.body == "Earth":
                return Outcome("returned", "Earth", end.et)
            return Outcome("impact", end.body, end.et)
        body = traj.stable_orbit()
        if body:
            return Outcome("orbit", body, traj.segments[-1].start.et)
        flyby = self._target_flyby()
        if flyby:
            return flyby
        return Outcome("in_flight", end.body, end.et)

    def _target_flyby(self):
        """The first pass through the target's SOI that leaves it again, as an Outcome."""
        if self.target == self.start.body:
            return None                                  # leaving the launch body isn't a flyby
        seg = planning.first_visit(self.trajectory, self.target)
        if seg is None or seg.end_reason != "soi_exit":
            return None
        s = seg.start
        dt = kepler.time_to_periapsis(s.r, s.v, s.mu)
        et = s.et + dt if dt is not None and s.et + dt <= seg.end_et else s.et
        return Outcome("flyby", self.target, et, s.elements.periapsis - BODIES[self.target].radius)

    @property
    def end_et(self):
        """When the flight physically stops (impact), else the projection horizon."""
        return self.trajectory.segments[-1].end_et

    # ---------- Editing ----------
    def is_locked(self, burn, now):
        return self.status != PENDING and burn.et <= now

    def set_plan(self, burns, horizon_et=None, now=None):
        """Replace the burn list. Burns already carried out (at or before `now`) must stay as they are.

        `now` defaults to the time of the last update().
        """
        now = self.now if now is None else now
        burns = sorted(burns, key=lambda b: b.et)
        if self.status in (COMPLETE, LOST):
            raise PlanError("This mission is over")
        if self.status == ACTIVE:
            done_before = [b for b in self.burns if b.et <= now]
            done_after = [b for b in burns if b.et <= now]
            if done_before != done_after:
                raise PlanError("Burns that have already happened can't be changed, "
                                "and new burns can't be placed in the past")
        for b in burns:
            if b.et < self.launch_et:
                raise PlanError("A burn can't happen before launch")
        self.burns = burns
        if horizon_et is not None:
            self.horizon_et = max(horizon_et, self.launch_et + 1)
        self._changed()

    def set_craft(self, craft):
        if self.status != PENDING:
            raise PlanError("The craft and payload can't change after launch")
        self.craft = craft
        self._changed()

    def set_start(self, start, burns=None, horizon_et=None):
        """New launch (and its plan). Only before launch."""
        if self.status != PENDING:
            raise PlanError("The launch can't change after it has happened")
        self.start = start
        if burns is not None:
            self.burns = sorted(burns, key=lambda b: b.et)
        if horizon_et is not None:
            self.horizon_et = horizon_et
        self._changed()

    # ---------- Progress ----------
    def update(self, now):
        """Advance the status to `now`. Returns True if it changed."""
        old = self.status
        self.now = now
        if self.status == PENDING and now >= self.launch_et:
            self.status = ACTIVE
        if self.status == ACTIVE:
            # Make sure the projection always reaches the present
            if now > self.horizon_et and self.trajectory.end_reason != "impact":
                self.horizon_et = now + 30 * DAY
                self._changed()
            out = self.outcome
            if out.kind in ("impact", "returned") and now >= out.et:
                self.status = COMPLETE
                self.ended_et = out.et
                self.end_note = out.describe()
        return self.status != old

    def can_declare_complete(self, now):
        """In a stable orbit, with no burns still to come and a full lap since the last one."""
        if self.status != ACTIVE or any(b.et > now for b in self.burns):
            return False
        s = self.state_at(now)
        if not is_stable_orbit(s):
            return False
        seg = self.trajectory.segment_at(now)
        period = s.elements.period(s.mu)
        return now - seg.start.et >= period

    def declare_complete(self, now):
        if not self.can_declare_complete(now):
            raise PlanError("A mission can only be ended in a stable orbit held for a full lap "
                            "(or by landing / returning to Earth)")
        s = self.state_at(now)
        self.status = COMPLETE
        self.ended_et = now
        self.end_note = f"Stable orbit around {s.body}"
        self.burns = [b for b in self.burns if b.et <= now]
        self.horizon_et = now
        self._changed()

    def burn_time(self, burn):
        """Seconds the burn lasts with the fuel the craft has left by then."""
        before = fly(self.start, self.craft, [b for b in self.burns if b.et < burn.et], burn.et)
        return before.craft.burn_time(burn.dv) if before.craft.fuel > 0 else 0.0


class MissionControl:
    """Every mission in the campaign."""

    def __init__(self):
        self.missions = []

    def add(self, mission):
        if any(m.name == mission.name for m in self.missions):
            raise PlanError(f"There is already a mission called {mission.name!r}")
        self.missions.append(mission)

    def remove(self, mission):
        if mission.status != PENDING:
            raise PlanError("Only missions that haven't launched can be cancelled")
        self.missions.remove(mission)

    def update(self, now):
        """Returns the missions whose status changed."""
        return [m for m in self.missions if m.update(now)]

    def by_status(self, *statuses):
        return [m for m in self.missions if m.status in statuses]

    def unique_name(self, base):
        names = {m.name for m in self.missions}
        if base not in names:
            return base
        k = 2
        while f"{base} {k}" in names:
            k += 1
        return f"{base} {k}"
