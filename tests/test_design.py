import math
from dataclasses import replace

import pytest

from game import catalogue
from game.design import BAD, OK, WARN, CraftDesign, assess, distance_au, starter_design

CAT = catalogue.default()
FOOD, WATER = CAT.food_per_crew_day, CAT.water_per_crew_day      # kg per crew member per day


def by_name(indicators):
    return {i.name: i for i in indicators}


# ---------- Catalogue ----------
def test_catalogue_loads_with_a_starter_set():
    assert {"light", "medium"} <= {k for k, v in CAT.launchers.items() if v.tier == 0}
    assert any(s.tier == 0 for s in CAT.stages.values())
    starter_kinds = {m.kind for m in CAT.modules.values() if m.tier == 0}
    assert {"capsule", "habitat", "supplies", "solar"} <= starter_kinds


def test_catalogue_values_are_sane():
    for s in CAT.stages.values():
        assert s.isp_s > 0 and s.thrust_n > 0 and s.dry_kg > 0 and s.fuel_capacity_kg > 0
    for m in CAT.modules.values():
        assert m.mass_kg > 0
        if m.kind == "solar":
            assert m.power_kw > 0
        else:
            assert m.power_kw <= 0             # everything but solar uses power
    assert any(s.relaunch for s in CAT.stages.values())


# ---------- Designs ----------
def test_masses_and_physics_craft():
    d = starter_design()
    stage = CAT.stages["storable"]
    modules = 2000 + 600 + 200
    assert d.dry_mass() == stage.dry_kg + modules
    assert d.total_mass() == stage.dry_kg + modules + 12_000
    c = d.physics_craft()
    assert (c.dry_mass, c.fuel, c.isp, c.thrust) == (d.dry_mass(), 12_000, stage.isp_s, stage.thrust_n)


def test_supply_days():
    d = starter_design()                      # 2 crew, 200 kg food, 300 kg water
    assert d.supply_days() == pytest.approx(min(200 / (2 * FOOD), 300 / (2 * WATER)))
    with_recycler = replace(d, modules={**d.modules, "recycler": 1})
    assert with_recycler.supply_days() == pytest.approx(200 / (2 * FOOD))     # now food runs out first
    # A farm feeds one crew member forever; the water (80% recycled) then sets the limit
    farm = replace(d, crew=1, modules={**d.modules, "farm": 1, "recycler": 1})
    assert farm.supply_days() == pytest.approx(300 / (1 * WATER * 0.2))
    assert replace(farm, modules={**farm.modules, "supplies": 0}).supply_days() == 0    # no water at all


def test_solar_power_falls_with_distance():
    d = starter_design()                      # 3 kW solar, 1 kW capsule
    assert d.power_kw(1.0) == pytest.approx(2.0)
    mars = distance_au("Mars")
    assert mars == pytest.approx(1.52, abs=0.01)
    assert d.power_kw(mars) == pytest.approx(3 / mars ** 2 - 1)
    assert distance_au("Moon") == pytest.approx(1.0, abs=0.01)


def test_cost_adds_up():
    d = starter_design()
    expected = 65 + 15 + 40 + 2 + 5 + 12 * CAT.fuel_cost_musd_per_tonne
    assert d.cost() == pytest.approx(expected)


def test_problems():
    assert starter_design().problems() == []
    assert any("Too heavy" in p for p in replace(starter_design(), launcher="light").problems())
    assert any("seats" in p for p in replace(starter_design(), crew=3).problems())
    assert any("at least one crew" in p for p in replace(starter_design(), crew=0).problems())
    assert any("tank" in p for p in replace(starter_design(), fuel_kg=1e6).problems())


# ---------- Health ----------
def test_moon_landing_sanity_check():
    """Easy mission: 2 crew land on the Moon with a shelter; fits a Medium but not a Light rocket."""
    from game.mission import Flight
    from physics import ephemeris
    et = ephemeris.str2et("2026-12-01")
    d = CraftDesign("medium", "storable", 11_500, {"capsule2": 1, "shelter2": 1, "supplies": 1, "solar_s": 1}, crew=2)
    assert d.problems() == []
    assert replace(d, launcher="light").problems()
    flight = Flight.to_moon("Lander", d.physics_craft(), et, 4 * 86_400, periapsis_alt=-100)   # aimed into the Moon
    assert flight.outcome.kind == "impact" and flight.outcome.body == "Moon"
    h = by_name(assess(d, flight))
    assert h["Mass"].level == OK and h["Crew"].level == OK and h["Δv"].level in (OK, WARN)

    no_shelter = replace(d, modules={"capsule2": 1, "supplies": 1, "solar_s": 1})
    crew = by_name(assess(no_shelter, flight))["Crew"]
    assert crew.level == BAD and "die" in crew.text


def test_health_flags_over_mass_and_missing_power():
    heavy = replace(starter_design(), launcher="light")
    assert by_name(assess(heavy))["Mass"].level == BAD
    dark = replace(starter_design(), modules={"capsule2": 1, "supplies": 1})
    assert by_name(assess(dark))["Power"].level == BAD
