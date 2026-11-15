"""Terraforming Mars: three stages, one after another, each run by its own equipment.

  1. Thicken the atmosphere: atmosphere factories on the surface.
  2. Melt the polar ice: orbital mirrors in Mars orbit.
  3. Seed a biosphere: biosphere labs on the surface.

A stage moves forward while its equipment is where it has to be (data/terraform.json
"where"); surface equipment also needs living crew at that base and power that
isn't negative. Progress is equipment-days: two factories go twice as fast as
one. Equipment only counts for its own stage (the doc: "equipment from one stage
can't be used in another"), and the stages run in order.

Like the science trickle, progress is added up as the clock runs; update() gets
the sites (base.sites_at) at the current moment.
"""
from dataclasses import dataclass
import json
from pathlib import Path

from . import catalogue as cat_module

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "data" / "terraform.json"
DAY = 86_400.0


@dataclass(frozen=True)
class Stage:
    id: str
    name: str
    module: str
    where: str
    needs_crew: bool
    work_days: float
    done_text: str


@dataclass
class StageStatus:
    stage: Stage
    fraction: float          # 0..1
    state: str               # "done", "running", "stopped", "waiting" (an earlier stage isn't done)
    text: str                # what's happening, or what's holding it back
    units: int = 0           # equipment working on it right now
    done_et: float | None = None
    days_left: float | None = None


def load_rules(path=DEFAULT_PATH):
    raw = json.loads(Path(path).read_text())
    return raw["body"], [Stage(**row) for row in raw["stages"]]


class Terraforming:
    def __init__(self, path=DEFAULT_PATH, catalogue=None):
        self.body, self.stages = load_rules(path)
        self.catalogue = catalogue or cat_module.default()
        self.work = [0.0] * len(self.stages)       # equipment-days done per stage
        self.done_et = [None] * len(self.stages)
        self._last_et = None
        self._rate = 0.0                           # equipment working on the current stage

    @property
    def current(self):
        """Index of the stage being worked on (len(stages) once every stage is done)."""
        return next((k for k, t in enumerate(self.done_et) if t is None), len(self.stages))

    @property
    def complete(self):
        return self.current == len(self.stages)

    @property
    def fraction(self):
        """Overall progress, 0..1 (every stage counts the same)."""
        return sum(min(1.0, w / s.work_days) for w, s in zip(self.work, self.stages)) / len(self.stages)

    def _site(self, sites, where):
        return next((s for s in sites if s.body == self.body and s.where == where), None)

    def _working(self, stage, sites):
        """(units working, reason they aren't) for a stage, with the sites as they are now."""
        name = self.catalogue.modules[stage.module].name
        site = self._site(sites, stage.where)
        units = site.modules.get(stage.module, 0) if site else 0
        place = f"the {self.body} surface" if stage.where == "surface" else f"{self.body} orbit"
        if not units:
            return 0, f"Bring {name}s to {place}"
        if stage.needs_crew and site.crew < 1:
            return 0, f"{units} {name}(s) on {self.body}, but nobody is there to run them (needs living crew)"
        if stage.where == "surface" and site.power_kw < 0:
            return 0, (f"{units} {name}(s) on {self.body}, but the base is short of power "
                       f"({site.power_kw:+.1f} kW): add solar arrays or a reactor")
        return units, ""

    def update(self, sites, now):
        """Add progress up to `now`. Returns the stages finished since the last update."""
        finished = []
        if self._last_et is not None and now > self._last_et:
            k = self.current
            if k < len(self.stages) and self._rate > 0:
                stage = self.stages[k]
                need = stage.work_days - self.work[k]
                gained = self._rate * (now - self._last_et) / DAY
                if gained >= need:
                    self.work[k] = stage.work_days
                    self.done_et[k] = self._last_et + need / self._rate * DAY
                    finished.append(stage)
                else:
                    self.work[k] += gained
        self._last_et = now if self._last_et is None else max(now, self._last_et)
        k = self.current
        self._rate = self._working(self.stages[k], sites)[0] if k < len(self.stages) else 0.0
        return finished

    def status(self, sites):
        out = []
        current = self.current
        for k, stage in enumerate(self.stages):
            frac = min(1.0, self.work[k] / stage.work_days)
            if self.done_et[k] is not None:
                out.append(StageStatus(stage, 1.0, "done", stage.done_text, done_et=self.done_et[k]))
            elif k > current:
                out.append(StageStatus(stage, frac, "waiting", f"Starts once '{self.stages[current].name}' is done"))
            else:
                units, why = self._working(stage, sites)
                name = self.catalogue.modules[stage.module].name
                if units:
                    left = (stage.work_days - self.work[k]) / units
                    out.append(StageStatus(stage, frac, "running",
                                           f"{units} {name}(s) working: about {left:,.0f} days to go at this rate",
                                           units, days_left=left))
                else:
                    out.append(StageStatus(stage, frac, "stopped", why))
        return out
