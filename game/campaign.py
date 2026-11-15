"""Win conditions: what a scenario asks of the player, and checking it.

A Scenario is a list of goals (all must be met at once to win) plus optional
limits that lose it when broken: a budget and a deadline. Sandbox has no
goals, so it is never won (and has no limits, so never lost).

Goals read a CampaignState: a snapshot of the campaign at one moment (the
missions, the sites replayed at that moment, the money spent and the science
points). There are two kinds:
  * milestones ("land crew on the Moon") are reached once and stay reached;
  * standing goals ("4 crew living on Mars") must hold when checked.
Everything is replayed from the missions, so only the final result needs
saving: a CampaignTracker keeps the moment the scenario was won or lost.

CampaignState.science is the science earned so far (game/science.py), satellites
the ones deployed, and terraform the Mars terraforming progress (game/terraform.py);
the caller passes them in.

Each scenario has a difficulty (Tutorial, Easy, Medium, Hard or Sandbox) and,
once won, a score out of 100 (score_win): "efficiency" scenarios score how little
of the budget was spent and how few crew died, "speed" ones how early they were won.

Scenarios live in data/scenarios.json. Goals are written as specs such as
{"type": "land", "body": "Moon", "crew": 2}; GOAL_TYPES maps "type" to a class
and the other keys are its arguments.
"""
from dataclasses import dataclass, field
import json
import math
from pathlib import Path

from . import catalogue as cat_module
from .base import ORBIT, SURFACE, arrivals, site_at, sites_at
from .mission import COMPLETE

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "data" / "scenarios.json"
PLAYING, WON, FAILED = "playing", "won", "failed"


@dataclass
class CampaignState:
    """Everything goals look at, at one moment."""
    control: object            # MissionControl
    now: float                 # ET
    sites: list                # SiteState at `now`
    spent: float               # $M on launched missions
    committed: float           # $M on scheduled launches
    science: float = 0.0       # points earned
    catalogue: object = None
    _arrivals: list = None
    satellites: list = field(default_factory=list)     # science.Deployment
    terraform: object = None                            # terraform.Terraforming

    @classmethod
    def build(cls, control, now, science=0.0, catalogue=None, satellites=(), terraform=None, sites=None):
        c = catalogue or cat_module.default()
        costs = control.costs()
        if sites is None:
            sites = sites_at(control, now, c)
        return cls(control, now, sites, costs.spent, costs.committed, science, c,
                   satellites=list(satellites), terraform=terraform)

    @property
    def arrivals(self):
        """Deliveries and departures up to now (computed once)."""
        if self._arrivals is None:
            self._arrivals = [a for a in arrivals(self.control, self.catalogue) if a.et <= self.now]
        return self._arrivals

    def site(self, body, where):
        return next((s for s in self.sites if s.body == body and s.where == where), None)


@dataclass
class Progress:
    done: bool
    detail: str                # what's there so far, e.g. "2 of 4 crew"
    fraction: float = 0.0      # 0..1, for a progress bar
    et: float | None = None    # milestones: when it was reached


# ---------- Goals ----------
class Goal:
    """One thing a scenario asks for. Subclasses set `milestone` and implement title and progress."""
    milestone = True

    @property
    def title(self):
        raise NotImplementedError

    def progress(self, state):
        raise NotImplementedError


def _crew(n):
    return f"{n} crew" if n != 1 else "1 crew member"


@dataclass
class OrbitGoal(Goal):
    """A craft with living crew settles into a stable orbit around `body`."""
    body: str
    crew: int = 1

    @property
    def title(self):
        return f"Put {_crew(self.crew)} in orbit around {self.body}"

    def progress(self, state):
        for a in state.arrivals:
            if a.body == self.body and a.where == ORBIT and not a.leaving and a.crew >= self.crew:
                return Progress(True, f"{a.source} arrived", 1.0, a.et)
        return Progress(False, "Not yet")


@dataclass
class LandGoal(Goal):
    """Crew land on `body` and survive the landing (they need habitat places)."""
    body: str
    crew: int = 1

    @property
    def title(self):
        return f"Land {_crew(self.crew)} on {self.body}"

    def progress(self, state):
        here = [a for a in state.arrivals if a.body == self.body and a.where == SURFACE]
        best = 0
        for a in here:
            if a.leaving:
                continue
            # Replay the site just after this landing: anyone without a habitat place is dead
            survivors = site_at(self.body, SURFACE, [b for b in here if b.et <= a.et], a.et, state.catalogue).crew
            if survivors >= self.crew:
                return Progress(True, f"{a.source} landed", 1.0, a.et)
            best = max(best, survivors)
        return Progress(False, f"Best so far: {_crew(best)}" if here else "Not yet", best / self.crew)


@dataclass
class ReturnGoal(Goal):
    """A mission reaches `body` (surface, orbit or either) and a later leg of it returns to Earth."""
    body: str
    where: str = SURFACE       # SURFACE, ORBIT or "any"

    @property
    def title(self):
        how = {SURFACE: "Land on", ORBIT: "Orbit"}.get(self.where, "Reach")
        return f"{how} {self.body} and return to Earth"

    def progress(self, state):
        half = False
        for m in state.control.missions:
            visits = [f.idle_since for f in m.flights
                      if f.idle_since is not None and f.idle_since <= state.now and self._here(f.location())]
            if not visits:
                continue
            half = True
            for f in m.flights:
                out = f.outcome
                if (f.status == COMPLETE and out.kind == "returned" and out.et <= state.now
                        and f.launch_et >= min(visits)):
                    return Progress(True, f"{m.name}: {f.name} came home", 1.0, out.et)
        return Progress(False, f"Reached {self.body}, not home yet" if half else "Not yet", 0.5 if half else 0.0)

    def _here(self, loc):
        return loc is not None and loc[0] == self.body and self.where in ("any", loc[1])


@dataclass
class BaseGoal(Goal):
    """A standing base: at least `crew` living crew at a site, optionally with
    supplies for `supply_days` and some required modules (module id -> count)."""
    body: str
    crew: int = 1
    where: str = SURFACE
    supply_days: float = 0.0
    modules: dict = field(default_factory=dict)
    milestone = False

    @property
    def title(self):
        text = f"Keep {_crew(self.crew)} living {'on' if self.where == SURFACE else 'in orbit around'} {self.body}"
        if self.supply_days:
            text += f" with {self.supply_days:,.0f} days of supplies"
        if self.modules:
            text += " and " + ", ".join(f"{n}x {mid}" for mid, n in self.modules.items())
        return text

    def progress(self, state):
        s = state.site(self.body, self.where)
        if s is None:
            return Progress(False, "No base yet")
        parts = [min(1.0, s.crew / self.crew)]
        detail = [f"{s.crew}/{self.crew} crew"]
        if self.supply_days:
            days = min(s.days_left()) if s.crew else 0.0
            parts.append(min(1.0, days / self.supply_days))
            detail.append("supplies never run out" if math.isinf(days) else f"{days:,.0f}/{self.supply_days:,.0f} days")
        for mid, n in self.modules.items():
            have = s.modules.get(mid, 0)
            parts.append(min(1.0, have / n))
            detail.append(f"{have}/{n} {mid}")
        fraction = sum(parts) / len(parts)
        return Progress(fraction >= 1.0, ", ".join(detail), fraction)


@dataclass
class ScienceGoal(Goal):
    """Earn `points` science points (science isn't in the game yet; see CampaignState.science)."""
    points: float
    milestone = False          # standing, but points only go up, so it behaves like one

    @property
    def title(self):
        return f"Earn {self.points:,.0f} science points"

    def progress(self, state):
        return Progress(state.science >= self.points, f"{state.science:,.0f}/{self.points:,.0f}",
                        min(1.0, state.science / self.points) if self.points else 1.0)


@dataclass
class SatelliteGoal(Goal):
    """Leave a satellite in orbit around `body` (Deploy satellite in the planner)."""
    body: str
    count: int = 1

    @property
    def title(self):
        what = "a satellite" if self.count == 1 else f"{self.count} satellites"
        return f"Deploy {what} in orbit around {self.body}"

    def progress(self, state):
        here = sorted(d.et for d in state.satellites if d.body == self.body and d.et <= state.now)
        if len(here) >= self.count:
            return Progress(True, "Deployed", 1.0, here[self.count - 1])
        return Progress(False, f"{len(here)}/{self.count}" if here else "Not yet", len(here) / self.count)


@dataclass
class TerraformGoal(Goal):
    """Finish every terraforming stage (game/terraform.py)."""
    body: str = "Mars"

    @property
    def title(self):
        return f"Terraform {self.body} (all 3 stages)"

    def progress(self, state):
        t = state.terraform
        if t is None:
            return Progress(False, "Not started", 0.0)
        if t.complete:
            return Progress(True, "Terraformed", 1.0, t.done_et[-1])
        stage = t.stages[t.current]
        return Progress(False, f"Stage {t.current + 1}: {stage.name}, {t.fraction:.0%} overall", t.fraction)


GOAL_TYPES = {"orbit": OrbitGoal, "land": LandGoal, "return": ReturnGoal, "base": BaseGoal,
              "science": ScienceGoal, "satellite": SatelliteGoal, "terraform": TerraformGoal}


def goal_from_spec(spec):
    spec = dict(spec)
    kind = spec.pop("type")
    if kind not in GOAL_TYPES:
        raise ValueError(f"Unknown goal type {kind!r} (known: {', '.join(GOAL_TYPES)})")
    return GOAL_TYPES[kind](**spec)


# ---------- Scenarios ----------
@dataclass
class Scenario:
    id: str
    title: str
    description: str = ""
    goals: list = field(default_factory=list)
    budget: float | None = None        # $M; going over (on launched missions) loses
    deadline: float | None = None      # ET; still playing at this moment loses
    difficulty: str = "Sandbox"        # Tutorial, Easy, Medium, Hard or Sandbox
    scoring: str = "efficiency"        # how a win is scored: "efficiency" or "speed"
    unlocked: list = field(default_factory=list)    # tech ids unlocked from the start

    @property
    def sandbox(self):
        return not self.goals

    @classmethod
    def from_spec(cls, sid, spec):
        from physics import ephemeris
        deadline = spec.get("deadline")
        return cls(sid, spec["title"], spec.get("description", ""),
                   [goal_from_spec(g) for g in spec.get("goals", [])], spec.get("budget"),
                   ephemeris.str2et(deadline) if deadline else None,
                   spec.get("difficulty", "Sandbox"), spec.get("scoring", "efficiency"),
                   list(spec.get("unlocked", [])))


def load_scenarios(path=DEFAULT_PATH):
    """Scenario id -> Scenario, in file order."""
    raw = json.loads(Path(path).read_text())
    return {sid: Scenario.from_spec(sid, spec) for sid, spec in raw.items()}


_default = None


def scenarios():
    """The scenarios in data/scenarios.json (loaded once)."""
    global _default
    if _default is None:
        _default = load_scenarios()
    return _default


# ---------- Checking ----------
DIFFICULTIES = ["Tutorial", "Easy", "Medium", "Hard", "Sandbox"]


def crew_deaths(state):
    """Crew who died so far: at bases, and on board lost craft."""
    at_sites = sum(s.deaths for s in state.sites)
    on_board = sum(f.design.crew for m in state.control.missions for f in m.flights
                   if f.status == "lost" and f.design is not None)
    return at_sites + on_board


def score_win(scenario, state, started_et):
    """(score out of 100, [why]) for a scenario won at state.now."""
    notes = []
    if scenario.scoring == "speed" and scenario.deadline is not None:
        span = max(scenario.deadline - started_et, 1.0)
        left = max(0.0, min(1.0, (scenario.deadline - state.now) / span))
        score = 40 + 60 * left
        notes.append(f"Speed: won with {left:.0%} of the time left (+{60 * left:.0f})")
    elif scenario.budget:
        left = max(0.0, min(1.0, 1 - state.spent / scenario.budget))
        score = 40 + 60 * left
        notes.append(f"Efficiency: {left:.0%} of the budget unspent (+{60 * left:.0f})")
    else:
        score = 100
    dead = crew_deaths(state)
    if dead:
        score -= 10 * dead
        notes.append(f"{dead} crew died (-{10 * dead})")
    return max(0, min(100, round(score))), notes


@dataclass
class Verdict:
    status: str                # PLAYING, WON or FAILED
    goals: list                # (goal, Progress) in scenario order
    reason: str = ""
    et: float | None = None    # when it was won or lost
    score: int | None = None   # out of 100, once won
    score_notes: list = field(default_factory=list)

    @property
    def met(self):
        return sum(p.done for _, p in self.goals)


class CampaignTracker:
    """Checks a scenario as the clock runs. Once won or lost, the result stays."""

    def __init__(self, scenario):
        self.scenario = scenario
        self.result = None         # (status, reason, et) once decided; this is what a save keeps
        self.started_et = None     # first check: when the scenario began (for speed scores)
        self.score = None          # (score, notes) once won

    @property
    def status(self):
        return self.result[0] if self.result else PLAYING

    def check(self, control, now, science=0.0, catalogue=None, satellites=(), terraform=None, sites=None):
        """Verdict at `now`. The first check that finds every goal met (or a limit broken) decides it."""
        if self.started_et is None:
            self.started_et = now
        state = CampaignState.build(control, now, science, catalogue, satellites, terraform, sites)
        goals = [(g, g.progress(state)) for g in self.scenario.goals]
        if self.result is None:
            self.result = self._decide(goals, state)
            if self.result is not None and self.result[0] == WON:
                self.score = score_win(self.scenario, state, self.started_et)
        if self.result is None:
            return Verdict(PLAYING, goals)
        status, reason, et = self.result
        score, notes = self.score if self.score else (None, [])
        return Verdict(status, goals, reason, et, score, notes)

    def _decide(self, goals, state):
        sc = self.scenario
        if sc.budget is not None and state.spent > sc.budget:
            return FAILED, f"Over budget: spent ${state.spent:,.0f}M of ${sc.budget:,.0f}M", state.now
        if goals and all(p.done for _, p in goals):
            return WON, "Every goal met", state.now
        if sc.deadline is not None and state.now >= sc.deadline:
            return FAILED, "Out of time", sc.deadline
        return None
