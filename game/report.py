"""Mission reports: what happened on a finished mission, a score out of 100, and how to do better.

Every mission that has ended (complete or lost) gets one. It goes craft by
craft: when it flew, how it ended, the crew, fuel and supplies it had left,
what went wrong (crew lost to empty supplies, deaths on landing, running out of
fuel, never arriving) and the science firsts it earned. Problems take points
off and each comes with a tip.
"""
from dataclasses import dataclass, field

from physics import GAS_GIANTS

from . import catalogue as cat_module
from .base import arrivals, site_at
from .mission import COMPLETE, LOST

DAY = 86_400.0


@dataclass
class CraftReport:
    name: str
    lines: list = field(default_factory=list)      # facts, in order
    problems: list = field(default_factory=list)   # (points off, what went wrong, tip)
    crew: int = 0
    died: int = 0


@dataclass
class MissionReport:
    name: str
    status: str                  # "Success", "Partly lost" or "Lost"
    score: int
    summary: str
    crafts: list
    firsts: list                 # science.First earned during the mission
    cost: float

    @property
    def problems(self):
        return [p for c in self.crafts for p in c.problems]

    @property
    def tips(self):
        seen, out = set(), []
        for _off, _what, tip in self.problems:
            if tip and tip not in seen:
                seen.add(tip)
                out.append(tip)
        return out

    def as_text(self, date=lambda et: f"{et:.0f}"):
        out = [f"{self.name}: {self.status}", f"Score: {self.score}/100", self.summary, ""]
        for c in self.crafts:
            out.append(c.name)
            out += [f"  {line}" for line in c.lines]
            for off, what, _tip in c.problems:
                out.append(f"  ✗ {what} (-{off})")
            out.append("")
        if self.firsts:
            out.append("Science firsts")
            out += [f"  {date(f.et)}  {f.text}  +{f.points:,.0f}" for f in self.firsts]
            out.append("")
        if self.tips:
            out.append("How to do better")
            out += [f"  • {t}" for t in self.tips]
        else:
            out.append("Nothing went wrong: a clean mission.")
        return "\n".join(out).rstrip()


def _deaths_on_arrival(flight, mission, arr, catalogue):
    """(crew who died the moment this craft arrived, why), from the base replay."""
    source = f"{mission.name}: {flight.name}"
    mine = [a for a in arr if a.source == source and not a.leaving]
    if not mine:
        return 0, ""
    a = mine[0]
    site_arr = [x for x in arr if (x.body, x.where) == (a.body, a.where)]
    s = site_at(a.body, a.where, site_arr, a.et, catalogue)
    deaths = [e for e in s.log if e.deaths and abs(e.et - a.et) < 1]
    return sum(e.deaths for e in deaths), (deaths[0].text if deaths else "")


def _visited(flight):
    bodies = {seg.body for seg in flight.trajectory.segments}
    return bodies - {"Sun"}


def craft_report(mission, flight, arr, catalogue, date):
    c = catalogue
    d = flight.design
    r = CraftReport(flight.name, crew=d.crew if d else 0)
    end = flight.ended_et if flight.ended_et is not None else flight.end_et
    days = (end - flight.launch_et) / DAY
    leg = f" (next leg of {flight.leg_of})" if flight.leg_of else ""
    r.lines.append(f"Flew {date(flight.launch_et)} to {date(end)}: {days:,.0f} days{leg}")
    r.lines.append(f"Ended: {flight.end_note or flight.outcome.describe()}")
    if d is not None:
        r.lines.append(f"Craft: {c.launchers[d.launcher].name}, {c.stages[d.stage].name}, "
                       f"{sum(d.modules.values())} module(s), {d.crew} crew")

    # Fuel
    events = [e for e in flight.trajectory.events if e.et <= end]
    if any(e.kind == "out_of_fuel" for e in events):
        e = next(e for e in events if e.kind == "out_of_fuel")
        r.problems.append((15, f"Ran out of fuel during a burn on {date(e.et)} ({e.note})",
                           "Carry more fuel, or a stage with a better Isp (unlock them in the Tech tree)"))
    if d is not None:
        left = max(0.0, flight.trajectory.craft.fuel if flight.status != LOST else flight.craft_at_arrival_fuel())
        cap = c.stages[d.stage].fuel_capacity_kg
        r.lines.append(f"Fuel left: {left:,.0f} kg ({left / cap:.0%} of the tank)")
        if flight.status == COMPLETE and 0 < left / cap < 0.03 and not any(p[0] == 15 for p in r.problems):
            r.problems.append((5, "Finished with almost no fuel to spare",
                               "Leave a margin: small errors need small correction burns"))

    # Crew and supplies
    if flight.status == LOST:
        r.died = r.crew
        starve = flight.starve_et()
        arrive = flight.idle_et() or flight.outcome.et
        late = f", {(arrive - starve) / DAY:,.0f} days before it would have arrived" \
            if starve is not None and arrive and arrive > starve else ""
        r.problems.append((60, f"Lost: {r.crew} crew died when the supplies ran out{late}",
                           "Check the Supplies line in Craft check before launch: add supply pods, a "
                           "water recycler or a farm, or pick a faster route"))
    elif d is not None and d.crew:
        died, why = _deaths_on_arrival(flight, mission, arr, c)
        if died:
            r.died = died
            tip = ("Gas giants have no surface: orbit them instead of landing"
                   if flight.location() and flight.location()[0] in GAS_GIANTS else
                   "Bring a habitat place (surface shelter or habitat) for every crew member who lands")
            r.problems.append((15 * died, f"{died} crew died on arrival: {why}", tip))
        if flight.starve_et() is not None and flight.idle_since is None and flight.outcome.kind != "returned":
            margin = (flight.starve_et() - end) / DAY
            r.lines.append(f"Supplies left at the end: {margin:,.0f} days")
            if 0 <= margin < 30:
                r.problems.append((10, f"Only {margin:,.0f} days of supplies left at the end",
                                   "Pack more supplies than the trip needs"))
        r.lines.append(f"Crew: {d.crew - r.died} of {d.crew} alive at the end"
                       + (" (came home)" if flight.outcome.kind == "returned" else ""))
    if flight.status == COMPLETE and flight.outcome.kind in ("flyby", "in_flight") and flight.idle_since is None:
        r.problems.append((20, "Never arrived anywhere: it was still drifting when the flight ended",
                           "Add a capture burn (Capture into orbit…) or a landing so the craft stays put"))
    return r


def mission_report(mission, control, science=None, catalogue=None, date=lambda et: f"{et:.0f}"):
    """The report for a mission that has ended (also works, as 'so far', for one still flying)."""
    c = catalogue or cat_module.default()
    arr = arrivals(control, c)
    crafts = [craft_report(mission, f, arr, c, date) for f in mission.flights if f.status != "pending"]
    lost = [r for r in crafts if any(off >= 60 for off, _w, _t in r.problems)]
    if crafts and len(lost) == len(crafts):
        status = "Lost"
    elif lost:
        status = "Partly lost"
    else:
        status = "Success"
    score = 100 - sum(off for r in crafts for off, _w, _t in r.problems)
    if status == "Lost":
        score = min(score, 30)
    score = max(0, min(100, score))
    start = min(f.launch_et for f in mission.flights)
    end = mission.ended_et or max((f.ended_et or f.end_et) for f in mission.flights)
    visited = set().union(*(_visited(f) for f in mission.flights if f.status != "pending"))
    firsts = [f for f in (science.firsts if science else []) if start <= f.et <= end and f.body in visited]
    crew = sum(f.design.crew for f in mission.first_legs if f.design is not None)   # legs carry the same crew
    died = sum(r.died for r in crafts)
    summary = (f"{len(crafts)} craft, {crew} crew ({died} died), {(end - start) / DAY:,.0f} days, "
               f"cost ${mission.cost():,.0f}M"
               + (f", {len(firsts)} science first(s)" if firsts else ""))
    return MissionReport(mission.name, status, score, summary, crafts, firsts, mission.cost())
