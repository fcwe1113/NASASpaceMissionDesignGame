"""The equipment catalogue: launchers, propulsion stages and payload modules.

Loaded from data/catalogue.json so balancing never needs a code change.
"""
from dataclasses import dataclass, field
import json
from pathlib import Path

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "data" / "catalogue.json"


@dataclass(frozen=True)
class Launcher:
    """Lifts one craft from Earth's surface to a parking orbit."""
    id: str
    name: str
    payload_kg: float
    cost_musd: float
    tier: int = 0
    description: str = ""


@dataclass(frozen=True)
class Stage:
    """The craft's own propulsion: every burn after the parking orbit."""
    id: str
    name: str
    isp_s: float
    thrust_n: float
    dry_kg: float
    fuel_capacity_kg: float
    cost_musd: float
    relaunch: bool = False
    tier: int = 0
    secret: bool = False            # a hidden unlock (the tech tree can keep it out of sight)
    description: str = ""


@dataclass(frozen=True)
class Module:
    id: str
    name: str
    kind: str                       # capsule, habitat, supplies, solar, life, farm, science...
    mass_kg: float
    power_kw: float = 0.0           # negative = uses power; solar: output at 1 AU
    cost_musd: float = 0.0
    seats: int = 0
    habitat_places: int = 0
    food_kg: float = 0.0
    water_kg: float = 0.0
    food_kg_per_day: float = 0.0    # farms
    water_recycling: float = 0.0    # fraction of crew water use recovered
    tier: int = 0
    description: str = ""


@dataclass
class Catalogue:
    launchers: dict
    stages: dict
    modules: dict
    food_per_crew_day: float
    water_per_crew_day: float
    fuel_cost_musd_per_tonne: float
    ascent_dv: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path=DEFAULT_PATH):
        raw = json.loads(Path(path).read_text())

        def table(kind, key):
            items = [kind(**row) for row in raw[key]]
            ids = [i.id for i in items]
            if len(ids) != len(set(ids)):
                raise ValueError(f"Duplicate ids in {key}")
            return {i.id: i for i in items}

        return cls(table(Launcher, "launchers"), table(Stage, "stages"), table(Module, "modules"),
                   raw["consumption"]["food_kg_per_crew_day"], raw["consumption"]["water_kg_per_crew_day"],
                   raw["fuel_cost_musd_per_tonne"], raw.get("surface_ascent_dv_km_s", {}))


_default = None


def default():
    """The catalogue in data/catalogue.json (loaded once)."""
    global _default
    if _default is None:
        _default = Catalogue.load()
    return _default
