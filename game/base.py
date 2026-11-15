"""What is where: crew, supplies and hardware at each site (a body's surface, or orbit around it).

Sites fill up as craft arrive:
  * a craft that lands leaves everything on the surface of that body;
  * a craft that settles into a stable orbit becomes part of that orbit's site.
A craft flying on (a next leg) takes its own hardware, fuel, surviving crew and
its share of the supplies with it; everything else stays. Crew use food and
water every day; farms and recyclers help. Crew die, with no grace period:
  * on landing, if there aren't enough habitat places for them (a capsule isn't one);
  * the moment the site's food or water runs out.

Everything is replayed from the arrivals and departures in time order, so the
state at any moment is deterministic and needs no saved running totals.
"""
from dataclasses import dataclass, field
import math

from . import catalogue as cat_module
from .design import distance_au
from .mission import PENDING

DAY = 86_400.0
SURFACE, ORBIT = "surface", "orbit"


@dataclass
class Arrival:
    """A delivery to a site (or, with leaving=True, a craft flying on and taking things away)."""
    et: float
    body: str
    where: str                 # SURFACE or ORBIT
    source: str                # "Mission: craft"
    crew: int
    food: float                # kg left after the trip
    water: float
    fuel: float
    modules: dict              # module id -> count
    stage: str
    leaving: bool = False


@dataclass
class LogEntry:
    et: float
    text: str
    deaths: int = 0


@dataclass
class SiteState:
    body: str
    where: str
    crew: int = 0
    habitat_places: int = 0
    food: float = 0.0
    water: float = 0.0
    fuel: float = 0.0
    modules: dict = field(default_factory=dict)
    stages: dict = field(default_factory=dict)
    food_per_day: float = 0.0          # net use (after farms)
    water_per_day: float = 0.0         # net use (after recycling)
    power_kw: float = 0.0
    log: list = field(default_factory=list)

    @property
    def name(self):
        return f"{self.body} ({self.where})"

    @property
    def deaths(self):
        return sum(e.deaths for e in self.log)

    def days_left(self):
        """(food days, water days) at current use; inf when not being used up."""
        f = self.food / self.food_per_day if self.food_per_day > 0 else math.inf
        w = self.water / self.water_per_day if self.water_per_day > 0 else math.inf
        return f, w


# ---------- Collecting arrivals from the campaign ----------
def transit_supplies(design, days, catalogue):
    """(food, water) left after `days` of travel with this design's crew, farms and recyclers."""
    food_use, water_use = use_rates(design.crew, design.modules, catalogue)
    items = design.items(catalogue)
    food = sum(m.food_kg * n for m, n in items) - food_use * days
    water = sum(m.water_kg * n for m, n in items) - water_use * days
    return max(0.0, food), max(0.0, water)


def use_rates(crew, modules, catalogue):
    """(food kg/day, water kg/day) used by `crew` with these modules."""
    if crew <= 0:
        return 0.0, 0.0
    mods = [(catalogue.modules[mid], n) for mid, n in modules.items() if n > 0]
    grown = sum(m.food_kg_per_day * n for m, n in mods)
    recycling = min(0.95, sum(m.water_recycling * n for m, n in mods))
    return (max(0.0, crew * catalogue.food_per_crew_day - grown),
            crew * catalogue.water_per_crew_day * (1 - recycling))


def flight_supplies(flight, et, catalogue):
    """(food, water) kg a flight's crew still has at `et`, from what it set out with."""
    d = flight.design
    items = d.items(catalogue)
    food = sum(m.food_kg * n for m, n in items) if flight.food0 is None else flight.food0
    water = sum(m.water_kg * n for m, n in items) if flight.water0 is None else flight.water0
    food_use, water_use = use_rates(d.crew, d.modules, catalogue)
    days = (et - flight.launch_et) / DAY
    return max(0.0, food - food_use * days), max(0.0, water - water_use * days)


def arrivals(control, catalogue=None):
    """Every delivery to, and departure from, a site so far: from craft that have arrived
    (idle or ended there) and next legs that have left. Only craft with a known design count."""
    c = catalogue or cat_module.default()
    out = []
    for m in control.missions:
        for f in m.flights:
            if f.design is None:
                continue
            if f.idle_since is not None and f.location() is not None:
                body, where = f.location()
                food, water = flight_supplies(f, f.idle_since, c)
                out.append(Arrival(f.idle_since, body, where, f"{m.name}: {f.name}", f.design.crew, food,
                                   water, max(0.0, f.craft_at_arrival_fuel()), dict(f.design.modules),
                                   f.design.stage))
            if f.leg_of is not None and f.status != PENDING:
                parent = m.flight(f.leg_of)
                loc = parent.location()
                if loc is None:
                    continue
                out.append(Arrival(f.launch_et, loc[0], loc[1], f"{m.name}: {f.name}", f.design.crew,
                                   f.food0 or 0.0, f.water0 or 0.0, max(0.0, parent.craft_at_arrival_fuel()),
                                   dict(f.design.modules), f.design.stage, leaving=True))
    return sorted(out, key=lambda a: a.et)


# ---------- Replaying a site ----------
def site_at(body, where, site_arrivals, et, catalogue=None):
    """The state of one site at time `et`, from the arrivals there (in any order)."""
    c = catalogue or cat_module.default()
    s = SiteState(body, where)
    t = None
    for a in sorted(site_arrivals, key=lambda a: a.et):
        if a.et > et:
            break
        if t is not None:
            _consume(s, t, a.et, c)
        t = a.et
        if a.leaving:
            _leave(s, a, c)
            continue
        for mid, n in a.modules.items():
            s.modules[mid] = s.modules.get(mid, 0) + n
        s.stages[a.stage] = s.stages.get(a.stage, 0) + 1
        s.food += a.food
        s.water += a.water
        s.fuel += a.fuel
        s.habitat_places = sum(c.modules[mid].habitat_places * n for mid, n in s.modules.items())
        s.crew += a.crew
        s.log.append(LogEntry(a.et, f"{a.source} arrived with {a.crew} crew"))
        if where == SURFACE and s.crew > s.habitat_places:
            dead = s.crew - s.habitat_places
            s.crew = s.habitat_places
            s.log.append(LogEntry(a.et, f"{dead} crew died on landing: no habitat place", dead))
        _rates(s, c)
        if s.crew and (s.food <= 0 or s.water <= 0):
            dead, s.crew = s.crew, 0
            s.log.append(LogEntry(a.et, f"{dead} crew died: no {'food' if s.food <= 0 else 'water'}", dead))
            _rates(s, c)
    if t is not None:
        _consume(s, t, et, c)
    _rates(s, c)
    s.power_kw = _power(s, body, c)
    return s


def _leave(s, a, c):
    """A craft flies on, taking its own hardware, fuel, crew and supplies."""
    for mid, n in a.modules.items():
        s.modules[mid] = max(0, s.modules.get(mid, 0) - n)
        if not s.modules[mid]:
            del s.modules[mid]
    s.stages[a.stage] = max(0, s.stages.get(a.stage, 0) - 1)
    if not s.stages[a.stage]:
        del s.stages[a.stage]
    s.crew = max(0, s.crew - a.crew)
    s.food = max(0.0, s.food - a.food)
    s.water = max(0.0, s.water - a.water)
    s.fuel = max(0.0, s.fuel - a.fuel)
    s.habitat_places = sum(c.modules[mid].habitat_places * n for mid, n in s.modules.items())
    s.log.append(LogEntry(a.et, f"{a.source} left with {a.crew} crew"))


def _rates(s, c):
    s.food_per_day, s.water_per_day = use_rates(s.crew, s.modules, c)


def _consume(s, t0, t1, c):
    """Use supplies from t0 to t1; if one runs out, the crew die at that moment."""
    _rates(s, c)
    days = (t1 - t0) / DAY
    if days <= 0 or s.crew == 0:
        return
    f_days, w_days = s.days_left()
    empty = min(f_days, w_days)
    if empty < days:
        what = "food" if f_days <= w_days else "water"
        s.food = max(0.0, s.food - s.food_per_day * empty)
        s.water = max(0.0, s.water - s.water_per_day * empty)
        dead, s.crew = s.crew, 0
        s.log.append(LogEntry(t0 + empty * DAY, f"{dead} crew died: the {what} ran out", dead))
        _rates(s, c)
        return
    s.food -= s.food_per_day * days
    s.water -= s.water_per_day * days


def _power(s, body, c):
    total = 0.0
    for mid, n in s.modules.items():
        m = c.modules[mid]
        total += m.power_kw * n / distance_au(body) ** 2 if m.kind == "solar" else m.power_kw * n
    return total


def sites_at(control, et, catalogue=None):
    """Every site that has had an arrival by `et`, sorted by body then surface/orbit."""
    groups = {}
    for a in arrivals(control, catalogue):
        if a.et <= et:
            groups.setdefault((a.body, a.where), []).append(a)
    order = {name: k for k, name in enumerate(["Mercury", "Venus", "Earth", "Moon", "Mars"])}
    return [site_at(body, where, arr, et, catalogue)
            for (body, where), arr in sorted(groups.items(), key=lambda kv: (order.get(kv[0][0], 9), kv[0][1]))]
