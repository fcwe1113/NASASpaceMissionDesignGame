"""Missions and flights.

A Mission is one launch: one or more craft leaving Earth on the same date. Each
craft has its own Flight: a start in a parking orbit and a plan of burns, with
its own destination. A craft that has arrived somewhere can fly on: its next
leg is another Flight, starting from a parking orbit around where it is.

Flight life cycle:  pending -> active (at launch) -> idle -> complete | lost
  - pending: everything can change.
  - active:  the craft and launch are fixed; burns already carried out are
             locked; future burns can be added, edited or removed.
  - idle:    arrived: landed, or settled in a stable orbit with no burns left
             (held for a full lap). Can fly on (a next leg) or be ended there.
  - complete: returned to Earth, ended by the player, or continued as a next leg.
  - lost:    the crew died (ran out of food or water on the way).

Missions are deterministic: the whole flight is re-flown from launch with the
current plan, and since past burns can't change, neither can the past path.
"""
from dataclasses import dataclass, field
from itertools import groupby
import math

from physics import BODIES, Burn, Craft, State, fly, is_stable_orbit, parking_orbit
from physics import assist, kepler, planning
from physics.trajectory import CoastSegment

PENDING, ACTIVE, IDLE, COMPLETE, LOST = "pending", "active", "idle", "complete", "lost"
PARKING_ALT = 200.0          # km, where every mission starts
DAY = 86_400.0


def park_alt(body):
    """Parking-orbit altitude (km) for starting a flight at `body`."""
    return planning.PARK_ALT.get(body, PARKING_ALT)


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
class Flight:
    """One craft's flight: its physics craft, parking-orbit start and burn plan."""
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
        self.assist = None             # the last gravity assist planned (assist.AssistPlan)
        self.design = None             # the CraftDesign the craft was built from, if any
        self.food0 = self.water0 = None    # supplies at the start, if not the design's full pods (legs)
        self.leg_of = None             # name of the flight this one continues, for next legs
        self.idle_since = None         # when it arrived (went idle)

    # ---------- Construction ----------
    @classmethod
    def to_planet(cls, name, target, craft, depart_et, arrive_et, v_inf, origin="Earth"):
        """Departure burn onto a porkchop transfer (correction/capture burns are added separately)."""
        start, dep = planning.departure(origin, depart_et, v_inf, park_alt(origin))
        return cls(name, target, craft, start, [dep], horizon_et=arrive_et + 30 * DAY)

    @classmethod
    def park(cls, name, craft, body, et, altitude=None):
        """Sit in a circular parking orbit around `body` (no burns)."""
        start = parking_orbit(body, et, park_alt(body) if altitude is None else altitude)
        return cls(name, body, craft, start, [], horizon_et=et + DAY)

    @classmethod
    def to_parent(cls, name, craft, origin, depart_et, tof):
        """From a moon back down to its planet (e.g. Moon -> Earth), landing there."""
        start, dep = planning.parent_return(origin, depart_et, tof)
        return cls(name, BODIES[origin].parent, craft, start, [dep], horizon_et=depart_et + tof + 5 * DAY)

    @classmethod
    def via_assist(cls, name, target, craft, depart_et, flyby_et, arrive_et, v_inf, via, arrive_alt=300.0,
                   origin="Earth"):
        """Departure, the burn that sets up a gravity assist at `via`, and the touch-up after it.

        Raises PlanningError if the craft can't afford the burns or the flyby is impossible.
        """
        start, dep = planning.departure(origin, depart_et, v_inf, park_alt(origin))
        m = cls(name, target, craft, start, [dep], horizon_et=arrive_et + 30 * DAY)
        exit_et = planning.soi_exit_time(fly(start, craft, [dep], depart_et + 30 * DAY), origin)
        if exit_et is None or exit_et + DAY >= flyby_et:
            raise planning.PlanningError(f"The craft doesn't leave {origin} in time for the flyby")
        m.add_assist(exit_et + DAY, via, flyby_et, target, arrive_et, arrive_alt)
        return m

    def add_assist(self, burn_et, via, flyby_et, target, arrive_et=None, arrive_alt=300.0):
        """Plan a gravity assist from `burn_et`: a burn setting up the flyby of `via`, and a
        touch-up a day after leaving it, aimed at `target`. Later burns are replaced.
        With arrive_et=None, the cheapest workable arrival time is chosen.

        Returns a note on what happened (e.g. if the touch-up couldn't be worked out).
        Raises PlanningError if the flyby itself can't be set up.
        """
        earlier = [b for b in self.burns if b.et < burn_et]
        before = fly(self.start, self.craft, earlier, burn_et)
        if before.end_reason == "impact":
            raise planning.PlanningError("The craft has already landed by then")
        if arrive_et is None:
            arrive_et = assist.best_arrival(before.end, via, flyby_et, target)
        setup = assist.assist_burn(before.end, before.craft, via, flyby_et, target, arrive_et)
        self.set_plan(earlier + [setup.burn], horizon_et=max(self.horizon_et, arrive_et + 30 * DAY))
        self.assist = setup
        note = (f"{via} bends the path {setup.turn_deg:.0f}° with a pass {setup.flyby_alt:,.0f} km up, "
                f"arriving {(arrive_et - self.launch_et) / DAY:.0f} days after launch")
        # Patched conics leave a small error; aim properly a day after leaving the flyby planet
        out_et = planning.soi_exit_time(self.trajectory, via)
        if out_et is None:
            return note + f"; but the craft doesn't come back out of {via}'s sphere of influence"
        before = fly(self.start, self.craft, self.burns, out_et + DAY)
        try:
            touch_up, alt = planning.aim_burn(before.end, before.craft, target, arrive_et, arrive_alt)
        except planning.PlanningError as exc:
            return note + f"; no touch-up towards {target}: {exc}"
        self.set_plan(self.burns + [touch_up])
        note += f"; touch-up of {touch_up.dv * 1000:.0f} m/s after it to pass {target} at {alt:,.0f} km"
        if setup.dv_flyby > 0.05:
            note += f" (includes making up {setup.dv_flyby * 1000:.0f} m/s the flyby can't give)"
        return note

    def add_landing(self, burn_et, body, arrive_et=None):
        """Plan a touchdown on `body` with a burn at `burn_et`. Later burns are replaced.

        Orbiting `body` then: a de-orbit burn. Orbiting its parent: an arrival aimed
        at the surface around `arrive_et`. Returns a note; raises PlanningError if
        it can't be done (wrong place, not enough fuel) and leaves the plan as it was.
        """
        earlier = [b for b in self.burns if b.et < burn_et]
        before = fly(self.start, self.craft, earlier, burn_et)
        if before.end_reason == "impact":
            raise planning.PlanningError("The craft has already landed by then")
        burn = planning.landing_burn(before.end, before.craft, body, arrive_et)
        old = (self.burns, self.horizon_et)
        self.set_plan(earlier + [burn], horizon_et=max(arrive_et or burn_et, burn_et) + 30 * DAY)
        out = self.outcome
        if out.kind not in ("impact", "returned") or out.body != body:     # an Earth touchdown is a return
            self.set_plan(*old)
            raise planning.PlanningError(f"That burn doesn't bring the craft down on {body} "
                                         f"({out.describe().lower()})")
        return f"Lands on {body} {(out.et - self.launch_et) / DAY:.1f} days after launch"

    @classmethod
    def to_moon(cls, name, craft, depart_et, tof, periapsis_alt=100.0):
        start, _ = planning.moon_departure(depart_et, tof, PARKING_ALT)
        tli, _ = planning.aim_burn(start, craft, "Moon", depart_et + tof, periapsis_alt)
        return cls(name, "Moon", craft, start, [tli], horizon_et=depart_et + tof + 5 * DAY)

    @classmethod
    def earth_orbit(cls, name, craft, launch_et, altitude=PARKING_ALT):
        return cls.park(name, craft, "Earth", launch_et, altitude)

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
        flyby = self._final_flyby()
        if flyby:
            return flyby
        return Outcome("in_flight", end.body, end.et)

    def _final_flyby(self):
        """The last body the craft passes and leaves again (e.g. Mars after a Venus assist).

        The launch body and the Sun don't count: leaving Earth isn't a flyby.
        """
        visits = [(body, list(segs)) for body, segs in groupby(self.trajectory.segments, key=lambda s: s.body)]
        visits = [(b, segs) for b, segs in visits if b not in ("Sun", self.start.body)]
        if not visits:
            return None
        body, segs = visits[-1]
        coast = next((s for s in segs if isinstance(s, CoastSegment)), None)
        if coast is None or segs[-1].end_reason != "soi_exit":
            return None
        s = coast.start
        dt = kepler.time_to_periapsis(s.r, s.v, s.mu)
        et = s.et + dt if dt is not None and s.et + dt <= coast.end_et else s.et
        return Outcome("flyby", body, et, s.elements.periapsis - BODIES[body].radius)

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

    def set_design(self, design):
        """Build the craft from a CraftDesign (and remember the design)."""
        self.set_craft(design.physics_craft())
        self.design = design

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
    def idle_et(self):
        """When the craft arrives and goes idle: touchdown (not on Earth: that's a return home),
        or a full lap into its final stable orbit with no burns left. None if it never does."""
        out = self.outcome
        if out.kind == "impact":
            return out.et
        if out.kind == "orbit":
            seg = self.trajectory.segments[-1]
            return seg.start.et + seg.start.elements.period(seg.start.mu)
        return None

    def craft_at_arrival_fuel(self):
        """Fuel (kg) left once every planned burn is done, i.e. on arrival."""
        return self.trajectory.craft.fuel

    def location(self):
        """(body, "surface" or "orbit") where an idle or ended craft sits, else None."""
        out = self.outcome
        if out.kind == "impact":
            return out.body, "surface"
        if out.kind == "orbit":
            return out.body, "orbit"
        return None

    def update(self, now):
        """Advance the status to `now`. Returns True if it changed."""
        old = self.status
        self.now = now
        if self.status == PENDING and now >= self.launch_et:
            self.status = ACTIVE
        if self.status in (ACTIVE, IDLE):
            # Make sure the projection always reaches the present
            if now > self.horizon_et and self.trajectory.end_reason != "impact":
                self.horizon_et = now + 30 * DAY
                self._changed()
        if self.status == ACTIVE:
            out = self.outcome
            starve = self.starve_et()
            returned = out.et if out.kind == "returned" else math.inf
            idle = self.idle_et() or math.inf
            if starve is not None and starve < min(returned, idle) and now >= starve:
                # No grace period: the crew die the moment the food or water runs out
                self.status = LOST
                self.ended_et = starve
                self.end_note = f"Lost: the crew ran out of supplies {(starve - self.launch_et) / DAY:.0f} days in"
            elif now >= returned:
                self.status = COMPLETE
                self.ended_et = out.et
                self.end_note = out.describe()
            elif now >= idle:
                body, where = self.location()
                self.status = IDLE
                self.idle_since = idle
                self.end_note = f"Idle on {body} (surface)" if where == "surface" else f"Idle in {body} orbit"
        return self.status != old

    def starve_et(self):
        """When the crew on board run out of food or water (None if never, or no design known)."""
        if self.design is None or self.design.crew <= 0:
            return None
        days = self.design.supply_days(food=self.food0, water=self.water0)
        return None if math.isinf(days) else self.launch_et + days * DAY

    def can_declare_complete(self, now):
        """Idle; or in a stable orbit, with no burns still to come and a full lap since the last one."""
        if self.status == IDLE:
            return True
        if self.status != ACTIVE or any(b.et > now for b in self.burns):
            return False
        s = self.state_at(now)
        if not is_stable_orbit(s):
            return False
        seg = self.trajectory.segment_at(now)
        period = s.elements.period(s.mu)
        return now - seg.start.et >= period

    def declare_complete(self, now):
        """End the flight where it is (its crew and hardware stay there)."""
        if not self.can_declare_complete(now):
            raise PlanError("A flight can only be ended once it has arrived: landed, or in a stable "
                            "orbit held for a full lap (or by returning to Earth)")
        where = self.location()
        if self.status != IDLE:
            self.idle_since = now
        self.status = COMPLETE
        self.ended_et = now
        if where and where[1] == "surface":
            self.end_note = f"Ended on {where[0]} (surface)"
        else:
            self.end_note = f"Stable orbit around {self.state_at(now).body}"
            self.burns = [b for b in self.burns if b.et <= now]
            self.horizon_et = now
            self._changed()

    def burn_time(self, burn):
        """Seconds the burn lasts with the fuel the craft has left by then."""
        before = fly(self.start, self.craft, [b for b in self.burns if b.et < burn.et], burn.et)
        return before.craft.burn_time(burn.dv) if before.craft.fuel > 0 else 0.0


class Mission:
    """One launch: several craft (flights) leaving Earth on the same date, plus any later legs.

    Status: pending before launch; active while any craft is still flying, idle
    or waiting to fly on; complete once every craft has ended (and at least one
    ended well); lost if every craft was lost.
    """

    def __init__(self, name, flights):
        if not flights:
            raise PlanError("A mission needs at least one craft")
        names = [f.name for f in flights]
        if len(set(names)) != len(names):
            raise PlanError("Each craft in a mission needs its own name")
        if len({f.launch_et for f in flights if f.leg_of is None}) > 1:
            raise PlanError("All craft in a mission launch together, on the same date")
        self.name = name
        self.flights = list(flights)

    @property
    def first_legs(self):
        return [f for f in self.flights if f.leg_of is None]

    @property
    def launch_et(self):
        return self.first_legs[0].launch_et

    @property
    def target(self):
        """Where the first craft is going (for lists)."""
        return self.first_legs[0].target

    @property
    def status(self):
        states = {f.status for f in self.flights}
        if states == {PENDING} and all(f.leg_of is None for f in self.flights):
            return PENDING
        if states & {PENDING, ACTIVE, IDLE}:
            return ACTIVE
        return LOST if states == {LOST} else COMPLETE

    @property
    def ended_et(self):
        if self.status not in (COMPLETE, LOST):
            return None
        ends = [f.ended_et for f in self.flights if f.ended_et is not None]
        return max(ends) if ends else None

    @property
    def end_note(self):
        return "; ".join(f"{f.name}: {f.end_note}" for f in self.flights) if len(self.flights) > 1 \
            else self.flights[0].end_note

    def flight(self, name):
        return next(f for f in self.flights if f.name == name)

    def cost(self):
        return sum(f.design.cost() for f in self.first_legs if f.design is not None)

    def update(self, now):
        """Advance every flight. Returns True if the mission's status changed."""
        old = self.status
        for f in self.flights:
            f.update(now)
            if f.leg_of is not None and f.status != PENDING:
                parent = self.flight(f.leg_of)
                if parent.status == IDLE:
                    # Flying on: the earlier leg ends where the new one leaves
                    parent.status = COMPLETE
                    parent.ended_et = f.launch_et
                    parent.end_note = f"Flew on as {f.name}"
        return self.status != old

    def set_flights(self, flights):
        """Replace all craft (only before launch)."""
        if self.status != PENDING:
            raise PlanError("Craft can't be added or removed after launch")
        Mission(self.name, flights)            # validates
        self.flights = list(flights)

    def add_leg(self, flight):
        """Add a next leg for an idle craft (`flight.leg_of` names it)."""
        parent = self.flight(flight.leg_of)
        if parent.status != IDLE:
            raise PlanError(f"{parent.name} has to have arrived (be idle) before it can fly on")
        if any(f.leg_of == parent.name and f.status == PENDING for f in self.flights):
            raise PlanError(f"{parent.name} already has a next leg planned")
        if flight.launch_et < parent.idle_since:
            raise PlanError(f"The next leg can't leave before {parent.name} has arrived")
        if any(f.name == flight.name for f in self.flights):
            raise PlanError(f"There is already a craft called {flight.name!r} in this mission")
        self.flights.append(flight)


@dataclass
class CampaignCost:
    spent: float          # $M on missions that have launched
    committed: float      # $M on scheduled launches not yet flown
    by_mission: dict      # mission name -> $M

    @property
    def total(self):
        return self.spent + self.committed


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

    def costs(self):
        """Campaign spending, in $M.

        spent: missions that have launched (whatever happened to them since,
        lost ones included: the money is gone). committed: scheduled launches
        not yet flown (cancelling one gets its money back). Next legs are free:
        they fly hardware that's already paid for.
        """
        spent = sum(m.cost() for m in self.missions if m.status != PENDING)
        committed = sum(m.cost() for m in self.missions if m.status == PENDING)
        return CampaignCost(spent, committed, {m.name: m.cost() for m in self.missions})

    def unique_name(self, base):
        names = {m.name for m in self.missions}
        if base not in names:
            return base
        k = 2
        while f"{base} {k}" in names:
            k += 1
        return f"{base} {k}"
