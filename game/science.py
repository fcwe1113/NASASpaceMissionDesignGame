"""Science points and the tech tree.

Science rewards exploring. Most points are "firsts": a one-off payout the first
time any craft reaches, orbits or lands on a body, lands crew there who live,
leaves a satellite there or sets up a base with a science kit. Bodies that are
harder to reach are worth more (data/science.json "multiplier"). On top of that
satellites and science bases earn a small trickle every day.

Points are spent on the tech tree, which unlocks the more advanced stages and
modules. Parts that aren't in any tech node (tier 0 parts, every launcher and
the torch drive) are always available.

Firsts are worked out again from the missions every update (like base.py's
sites), so they always match what has actually happened. The daily trickle is
added up as the clock runs.
"""
from dataclasses import dataclass
from itertools import groupby
import json
from pathlib import Path
import math
import time

from . import catalogue as cat_module
from .base import SURFACE, arrivals, site_at
from .mission import IDLE, PENDING, PlanError

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "data" / "science.json"
DAY = 86_400.0
REFRESH_S = 1.0                # bases are re-checked at least this often (real seconds)
SATELLITE = "satellite"        # module id of the science satellite
KIT = "science"                # module id of the science kit


@dataclass(frozen=True)
class TechNode:
    id: str
    name: str
    cost: float
    needs: tuple = ()
    parts: tuple = ()


@dataclass
class First:
    """A one-off payout for doing something first at a body."""
    kind: str          # visit, orbit, landing, crewed, satellite or base
    body: str
    et: float
    points: float
    text: str


@dataclass
class Deployment:
    """A satellite left in orbit by a craft."""
    flight: object     # the Flight it came from
    body: str
    et: float


@dataclass
class Source:
    """Something earning science every day."""
    what: str
    body: str
    per_day: float


@dataclass
class Rules:
    start_points: float
    multiplier: dict
    firsts: dict                  # kind -> {"points", "text"}
    earth_satellite_points: float
    satellite_per_day: float
    kit_per_day: float
    crewed_factor: float
    tech: dict                    # node id -> TechNode, in display order

    @classmethod
    def load(cls, path=DEFAULT_PATH):
        raw = json.loads(Path(path).read_text())
        tech = {}
        for row in raw["tech"]:
            node = TechNode(row["id"], row["name"], row["cost"], tuple(row["needs"]), tuple(row["parts"]))
            if node.id in tech:
                raise ValueError(f"Duplicate tech id {node.id!r}")
            tech[node.id] = node
        return cls(raw["start_points"], raw["multiplier"], raw["firsts"], raw["earth_satellite_points"],
                   raw["satellite_points_per_day"], raw["base_kit_points_per_day"],
                   raw["crewed_base_factor"], tech)


class Science:
    """The player's science: points, unlocked tech and deployed satellites."""

    def __init__(self, rules=None, catalogue=None):
        self.rules = rules or Rules.load()
        self.catalogue = catalogue or cat_module.default()
        self.unlocked = {}            # tech id -> sim time it was unlocked
        self.deployments = []         # Deployment, in time order
        self.passive = 0.0            # points earned by the daily trickle so far
        self.firsts = []              # First, in time order (from the last update)
        self.sources = []             # Source, earning now (from the last update)
        self._seen = set()            # (kind, body) of firsts already returned by update()
        self._last_et = None
        self._visit_cache = {}        # id(flight) -> (flight, trajectory, visits)
        # The slow part (where every craft ended up, and the bases) is only redone when a
        # flight changes, or every REFRESH_S, so the game doesn't stutter with many missions
        self._sig = None
        self._checked_at = -math.inf
        self._arr = []                # base.arrivals()
        self._places = []             # (flight, body, where, idle_since) for craft that have arrived
        self._part_node = {p: n for n in self.rules.tech.values() for p in n.parts}

    # ---------- Points ----------
    def spent(self):
        return sum(self.rules.tech[t].cost for t in self.unlocked)

    def balance(self):
        return self.rules.start_points + sum(f.points for f in self.firsts) + self.passive - self.spent()

    def per_day(self):
        return sum(s.per_day for s in self.sources)

    def multiplier(self, body):
        return self.rules.multiplier.get(body, 0)

    def update(self, control, now):
        """Bring firsts and the daily trickle up to `now`. Returns the firsts that are new since last time."""
        if self._last_et is not None and now > self._last_et:
            self.passive += self.per_day() * (now - self._last_et) / DAY
        self._last_et = now if self._last_et is None else max(now, self._last_et)
        sig = (len(self.deployments), tuple((id(f), f.status, f.idle_since, id(f.trajectory))
                                            for m in control.missions for f in m.flights))
        if sig != self._sig or time.monotonic() - self._checked_at >= REFRESH_S:
            self._sig, self._checked_at = sig, time.monotonic()
            self._arr = arrivals(control, self.catalogue)
            self._places = [(f, *f.location(), f.idle_since) for m in control.missions for f in m.flights
                            if f.idle_since is not None and f.location() is not None]
            self.sources = self._find_sources(now)
        self.firsts = self._find_firsts(control, self._arr, now)
        new = [f for f in self.firsts if (f.kind, f.body) not in self._seen]
        self._seen.update((f.kind, f.body) for f in new)
        return new

    def _find_firsts(self, control, arr, now):
        found = {}                     # (kind, body) -> et of the earliest

        def note(kind, body, et):
            if et <= now and body not in ("Sun", "Earth") and self.multiplier(body) > 0:
                key = (kind, body)
                if key not in found or et < found[key]:
                    found[key] = et

        for m in control.missions:
            for f in m.flights:
                if f.status == PENDING or now < f.launch_et:
                    continue
                for body, et in self._visits(f):
                    note("visit", body, et)
        for _f, body, where, idle_since in self._places:
            note("landing" if where == SURFACE else "orbit", body, idle_since)

        for a in arr:
            if a.leaving or a.where != SURFACE or a.et > now:
                continue
            if a.modules.get(KIT, 0) > 0:
                note("base", a.body, a.et)
            if a.crew > 0 and ("crewed", a.body) not in found:
                site = [x for x in arr if (x.body, x.where) == (a.body, SURFACE)]
                if site_at(a.body, SURFACE, site, a.et, self.catalogue).crew > 0:
                    note("crewed", a.body, a.et)

        out = []
        for d in self.deployments:
            if d.et > now:
                continue
            if d.body == "Earth":
                if ("satellite", "Earth") not in found:
                    found[("satellite", "Earth")] = d.et
            else:
                note("satellite", d.body, d.et)
        for (kind, body), et in found.items():
            rule = self.rules.firsts[kind]
            if body == "Earth":
                points = self.rules.earth_satellite_points
            else:
                points = rule["points"] * self.multiplier(body)
            out.append(First(kind, body, et, points, rule["text"].format(body=body)))
        return sorted(out, key=lambda f: f.et)

    def _visits(self, flight):
        """[(body, et)] each time the craft enters a body's sphere of influence (cached per trajectory)."""
        traj = flight.trajectory
        hit = self._visit_cache.get(id(flight))
        if hit is not None and hit[0] is flight and hit[1] is traj:
            return hit[2]
        groups = groupby(traj.segments, key=lambda s: s.body)
        next(groups, None)             # where it starts isn't a visit
        visits = [(body, next(segs).start.et) for body, segs in groups]
        self._visit_cache[id(flight)] = (flight, traj, visits)
        return visits

    def _sites(self, now):
        """Every surface or orbit site at `now` (like base.sites_at, from the arrivals already found)."""
        groups = {}
        for a in self._arr:
            if a.et <= now:
                groups.setdefault((a.body, a.where), []).append(a)
        return [site_at(body, where, arr, now, self.catalogue) for (body, where), arr in groups.items()]

    def _find_sources(self, now):
        out = []
        for d in self.deployments:
            if d.et <= now and self.multiplier(d.body) > 0:
                out.append(Source("Satellite", d.body, self.rules.satellite_per_day * self.multiplier(d.body)))
        for s in self._sites(now):
            kits = s.modules.get(KIT, 0)
            if s.where != SURFACE or s.body == "Earth" or not kits or self.multiplier(s.body) <= 0:
                continue
            if s.power_kw < 0:
                out.append(Source(f"Base ({kits} kit, no power)", s.body, 0.0))
                continue
            rate = self.rules.kit_per_day * kits * self.multiplier(s.body)
            if s.crew > 0:
                rate *= self.rules.crewed_factor
            out.append(Source(f"Base ({kits} kit{'s' if kits > 1 else ''}"
                              f"{', crewed' if s.crew else ''})", s.body, rate))
        return out

    # ---------- Tech tree ----------
    def node_for(self, part_id):
        """The tech node that unlocks a part, or None if it is always available."""
        return self._part_node.get(part_id)

    def is_unlocked(self, part_id):
        node = self.node_for(part_id)
        return node is None or node.id in self.unlocked

    def unlock_problem(self, node_id):
        """Why a node can't be unlocked right now, or None if it can."""
        node = self.rules.tech[node_id]
        if node_id in self.unlocked:
            return "Already unlocked"
        missing = [self.rules.tech[n].name for n in node.needs if n not in self.unlocked]
        if missing:
            return "Needs " + ", ".join(missing) + " first"
        if self.balance() < node.cost:
            return f"Needs {node.cost:,.0f} science (you have {self.balance():,.0f})"
        return None

    def unlock(self, node_id, now):
        problem = self.unlock_problem(node_id)
        if problem:
            raise PlanError(problem)
        self.unlocked[node_id] = now

    def locked_parts(self, design):
        """Names of the parts in a design that are still locked."""
        c = self.catalogue
        parts = [(design.launcher, c.launchers), (design.stage, c.stages)]
        parts += [(mid, c.modules) for mid, n in design.modules.items() if n > 0]
        return [f"{table[pid].name} (needs {self.node_for(pid).name})"
                for pid, table in parts if not self.is_unlocked(pid)]

    def newly_locked(self, old, new):
        """Locked parts that `new` adds compared with `old` (designs)."""
        before = set(self.locked_parts(old))
        return [p for p in self.locked_parts(new) if p not in before]

    # ---------- Satellites ----------
    def deployed_from(self, flight):
        return sum(1 for d in self.deployments if d.flight is flight)

    def satellites_on_board(self, flight):
        if flight.design is None:
            return 0
        return flight.design.modules.get(SATELLITE, 0) - self.deployed_from(flight)

    def deploy_problem(self, flight):
        """Why `flight` can't deploy a satellite now, or None if it can."""
        if self.satellites_on_board(flight) <= 0:
            return "No satellites on board"
        if flight.status != IDLE or flight.location() is None or flight.location()[1] == SURFACE:
            return "The craft has to have settled into orbit first (idle in orbit)"
        if flight.location()[0] == "Sun":
            return "Satellites can't be left orbiting the Sun"
        return None

    def deploy(self, flight, now):
        problem = self.deploy_problem(flight)
        if problem:
            raise PlanError(problem)
        self.deployments.append(Deployment(flight, flight.location()[0], max(now, flight.idle_since)))

    def without_deployed(self, flight, design):
        """`design` minus the satellites `flight` has already left behind (for its next leg)."""
        n = self.deployed_from(flight)
        if not n or SATELLITE not in design.modules:
            return design
        modules = dict(design.modules)
        modules[SATELLITE] = max(0, modules[SATELLITE] - n)
        if not modules[SATELLITE]:
            del modules[SATELLITE]
        return type(design)(design.launcher, design.stage, design.fuel_kg, modules, design.crew)
