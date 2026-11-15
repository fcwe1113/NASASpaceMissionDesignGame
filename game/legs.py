"""Flying on: what an idle craft takes with it on its next leg, and where that leg can go.

A next leg starts like a launch, in a parking orbit around wherever the craft
is. From a surface that needs a stage that can relaunch, and the take-off
(catalogue "surface_ascent_dv_km_s") is paid from the craft's own fuel.
The leg takes the craft's own hardware and fuel, its crew that are still alive
there, and its share of the supplies (what it brought, less what its crew ate,
and never more than the site still has).
"""
from dataclasses import dataclass, replace
import math

from physics import BODIES, CHILDREN, PLANETS

from . import catalogue as cat_module
from .base import SURFACE, arrivals, flight_supplies, site_at
from .mission import IDLE, PlanError

G0 = 9.80665


@dataclass
class LegStart:
    body: str            # where it leaves from
    where: str           # "surface" or "orbit"
    design: object       # CraftDesign with the fuel and crew it actually has
    food: float
    water: float
    ascent_dv: float     # km/s spent taking off (0 from orbit)
    earliest: float      # can't leave before this


def next_leg_start(control, mission, flight, et, catalogue=None):
    """What `flight`'s craft takes on a next leg leaving at `et`. Raises PlanError if it can't go."""
    c = catalogue or cat_module.default()
    if flight.status != IDLE:
        raise PlanError(f"{flight.name} has to arrive somewhere (be idle) before it can fly on")
    if flight.design is None:
        raise PlanError(f"{flight.name} has no known design")
    et = max(et, flight.idle_since)
    body, where = flight.location()
    site = site_at(body, where, [a for a in arrivals(control, c) if (a.body, a.where) == (body, where)], et, c)
    d = flight.design
    crew = min(d.crew, site.crew)
    if crew < 1:
        raise PlanError(f"Nobody is left alive at {body} to fly {flight.name}")
    food, water = flight_supplies(flight, et, c)
    food, water = min(food, site.food), min(water, site.water)
    fuel = flight.craft_at_arrival_fuel()
    ascent = 0.0
    if where == SURFACE:
        stage = c.stages[d.stage]
        if not stage.relaunch:
            raise PlanError(f"{flight.name}'s {stage.name} can't take off again: flying on from a "
                            f"surface needs a stage that can relaunch")
        ascent = c.ascent_dv.get(body, math.inf)
        ve = stage.isp_s * G0 / 1000
        mass = d.dry_mass(c) + fuel
        needed = mass * (1 - math.exp(-ascent / ve)) if math.isfinite(ascent) else math.inf
        if needed > fuel:
            have = ve * math.log(mass / d.dry_mass(c)) if fuel > 0 else 0.0
            raise PlanError(f"Taking off from {body} needs {ascent:.1f} km/s, but {flight.name} only has "
                            f"{have:.2f} km/s of fuel left")
        fuel -= needed
    return LegStart(body, where, replace(d, fuel_kg=fuel, crew=crew), food, water, ascent, et)


def apply_leg_start(flight, start, leg_of):
    """Stamp a new leg's flight with what it carries (design, supplies) and what it continues."""
    flight.design = start.design
    flight.food0, flight.water0 = start.food, start.water
    flight.leg_of = leg_of
    return flight


def next_leg_name(mission, flight):
    base, _, n = flight.name.rpartition(" leg ")
    base, k = (base, int(n) + 1) if n.isdigit() and base else (flight.name, 2)
    names = {f.name for f in mission.flights}
    while f"{base} leg {k}" in names:
        k += 1
    return f"{base} leg {k}"


def leg_targets(origin):
    """Where a leg starting at `origin` can go: list of (label, kind, body).

    kind: "planet" (launch window to a body orbiting the same parent), "moon"
    (Earth -> Moon), "parent" (a moon back down to its planet), "park" (stay in orbit).
    """
    b = BODIES[origin]
    out = []
    if b.parent == "Sun":
        out += [(p, "planet", p) for p in PLANETS if p != origin]
        out += [(c, "moon", c) for c in CHILDREN[origin]]
    elif b.parent is not None:
        out.append((f"{b.parent} (back down)", "parent", b.parent))
    out.append((f"{origin} orbit (stay)", "park", origin))
    return out
