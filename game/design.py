"""Craft designs (launcher + stage + fuel + modules + crew) and the health checks shown to players.

A design becomes the physics Craft the flight simulation uses; its health
indicators turn raw stats into the questions a new player actually has:
does it fit on the rocket, does it have the fuel, the supplies and the power,
and will the crew survive where it's going.
"""
from dataclasses import dataclass, field
import math

from physics import BODIES, GAS_GIANTS, Craft, LONG_BURN

from . import catalogue as cat_module

AU_KM = 149_597_870.7
DAY = 86_400.0
OK, WARN, BAD, INFO = "ok", "warn", "bad", "info"
MARGIN = 0.10          # less than 10% spare counts as a warning


@dataclass
class CraftDesign:
    launcher: str
    stage: str
    fuel_kg: float
    modules: dict = field(default_factory=dict)      # module id -> count
    crew: int = 1

    def _cat(self, catalogue):
        return catalogue or cat_module.default()

    def items(self, catalogue=None):
        """[(Module, count)] for the modules on board."""
        c = self._cat(catalogue)
        return [(c.modules[mid], n) for mid, n in self.modules.items() if n > 0]

    def dry_mass(self, catalogue=None):
        c = self._cat(catalogue)
        return c.stages[self.stage].dry_kg + sum(m.mass_kg * n for m, n in self.items(c))

    def total_mass(self, catalogue=None):
        return self.dry_mass(catalogue) + self.fuel_kg

    def physics_craft(self, catalogue=None):
        c = self._cat(catalogue)
        s = c.stages[self.stage]
        return Craft(self.dry_mass(c), self.fuel_kg, s.isp_s, s.thrust_n)

    def seats(self, catalogue=None):
        return sum(m.seats * n for m, n in self.items(catalogue))

    def habitat_places(self, catalogue=None):
        return sum(m.habitat_places * n for m, n in self.items(catalogue))

    def can_relaunch(self, catalogue=None):
        return self._cat(catalogue).stages[self.stage].relaunch

    def cost(self, catalogue=None):
        c = self._cat(catalogue)
        return (c.launchers[self.launcher].cost_musd + c.stages[self.stage].cost_musd
                + sum(m.cost_musd * n for m, n in self.items(c))
                + self.fuel_kg / 1000 * c.fuel_cost_musd_per_tonne)

    def power_kw(self, distance_au=1.0, catalogue=None):
        """Net power (kW): solar output falls with distance from the Sun squared."""
        total = 0.0
        for m, n in self.items(catalogue):
            if m.kind == "solar":
                total += m.power_kw * n / distance_au ** 2
            else:
                total += m.power_kw * n
        return total

    def supply_days(self, catalogue=None, food=None, water=None):
        """How long the food and water on board last the crew (inf if they never run out).

        `food`/`water` (kg) override the full supply pods, e.g. for a next leg
        that has already eaten into them.
        """
        c = self._cat(catalogue)
        items = self.items(c)
        if self.crew <= 0:
            return math.inf
        food = sum(m.food_kg * n for m, n in items) if food is None else food
        water = sum(m.water_kg * n for m, n in items) if water is None else water
        recycling = min(0.95, sum(m.water_recycling * n for m, n in items))
        grown = sum(m.food_kg_per_day * n for m, n in items)
        food_use = self.crew * c.food_per_crew_day - grown
        water_use = self.crew * c.water_per_crew_day * (1 - recycling)
        days = [food / food_use if food_use > 0 else math.inf,
                water / water_use if water_use > 0 else math.inf]
        return min(days)

    def problems(self, catalogue=None):
        """Hard errors that stop the design from launching at all."""
        c = self._cat(catalogue)
        out = []
        if self.launcher not in c.launchers:
            out.append(f"Unknown launcher {self.launcher!r}")
        if self.stage not in c.stages:
            out.append(f"Unknown stage {self.stage!r}")
        if out:
            return out
        if not 0 <= self.fuel_kg <= c.stages[self.stage].fuel_capacity_kg:
            out.append(f"Fuel must be between 0 and {c.stages[self.stage].fuel_capacity_kg:,.0f} kg "
                       f"(the stage's tank)")
        if self.total_mass(c) > c.launchers[self.launcher].payload_kg:
            out.append(f"Too heavy for the {c.launchers[self.launcher].name}: "
                       f"{self.total_mass(c):,.0f} kg, limit {c.launchers[self.launcher].payload_kg:,.0f} kg")
        if self.crew < 1:
            out.append("Every craft needs at least one crew member to fly it")
        if self.crew > self.seats(c):
            out.append(f"{self.crew} crew but only {self.seats(c)} seats (add a capsule)")
        return out


@dataclass
class Indicator:
    name: str
    level: str          # ok / warn / bad / info
    text: str


def distance_au(body):
    """Mean distance of a body (or its planet) from the Sun, in AU."""
    if body is None or body == "Sun":
        return 1.0
    b = BODIES[body]
    while b.parent not in (None, "Sun"):
        b = BODIES[b.parent]
    return b.a / AU_KM if b.parent else 1.0


def _margin_level(have, need):
    if have < need:
        return BAD
    return WARN if have < need * (1 + MARGIN) else OK


def assess(design, flight=None, catalogue=None, food=None, water=None):
    """Health indicators for a design, using its planned flight when there is one.

    `food`/`water` (kg): what is actually on board, if not the design's full supply pods.
    """
    c = catalogue or cat_module.default()
    out = []
    launcher = c.launchers[design.launcher]
    mass, limit = design.total_mass(c), launcher.payload_kg
    out.append(Indicator("Mass", BAD if mass > limit else WARN if mass > 0.95 * limit else OK,
                         f"{mass / 1000:,.1f} t of {limit / 1000:,.0f} t on the {launcher.name}"))

    craft = design.physics_craft(c)
    have = craft.dv_remaining
    if flight is not None and flight.burns:
        need = sum(b.dv for b in flight.burns)
        out.append(Indicator("Δv", _margin_level(have, need), f"needs {need:.2f} km/s, has {have:.2f} km/s"))
        longest = max(flight.burn_time(b) for b in flight.burns)
        out.append(Indicator("Longest burn", INFO,
                             "instant" if longest <= LONG_BURN else f"{longest / DAY:.1f} days"))
    else:
        out.append(Indicator("Δv", INFO, f"has {have:.2f} km/s (no burns planned yet)"))

    days = design.supply_days(c, food=food, water=water)
    days_text = "never run out" if math.isinf(days) else f"last {days:,.0f} days"
    if flight is not None:
        needed = (flight.end_et - flight.launch_et) / DAY
        out.append(Indicator("Supplies", _margin_level(days, needed),
                             f"{days_text}; the planned flight takes {needed:,.0f} days"))
    else:
        out.append(Indicator("Supplies", INFO, f"{days_text} for {design.crew} crew"))

    target = None
    if flight is not None:
        target = flight.trajectory.end.body if flight.trajectory.end.body != "Sun" else flight.target
    here, there = design.power_kw(1.0, c), design.power_kw(distance_au(target), c)
    worst = min(here, there)
    text = f"{here:+.1f} kW near Earth"
    if target and target not in ("Earth", "Moon"):
        text += f", {there:+.1f} kW at {target}"
    out.append(Indicator("Power", BAD if worst < 0 else OK, text))

    crew_level, crew_text = OK, f"{design.crew} crew, {design.seats(c)} seats"
    if design.crew < 1 or design.crew > design.seats(c):
        crew_level = BAD
    elif flight is not None and flight.outcome.kind == "impact" and flight.outcome.body in GAS_GIANTS:
        crew_level = BAD
        crew_text += f"; {flight.outcome.body} has no surface: the craft sinks into it and the crew die"
    elif flight is not None and flight.outcome.kind == "impact":
        places = design.habitat_places(c)
        if places < design.crew:
            crew_level = BAD
            crew_text += (f"; lands on {flight.outcome.body} with {places} habitat place(s): "
                          f"crew without one die on landing")
        else:
            crew_text += f"; {places} habitat place(s) on landing"
    out.append(Indicator("Crew", crew_level, crew_text))

    out.append(Indicator("Cost", INFO, f"${design.cost(c):,.0f}M"))
    return out


def starter_design():
    """What a new craft starts as: a 2-seat capsule on a Medium launcher with a full storable stage."""
    return CraftDesign("medium", "storable", 12_000, {"capsule2": 1, "supplies": 1, "solar_s": 1}, crew=2)
