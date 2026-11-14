import pytest

from game import catalogue
from game.base import ORBIT, SURFACE, arrivals, site_at, sites_at, transit_supplies
from game.design import CraftDesign
from game.mission import COMPLETE, LOST, Flight, Mission, MissionControl
from physics import ephemeris

DAY = 86_400.0
ET = ephemeris.str2et("2026-11-20")
CAT = catalogue.default()
FOOD, WATER = CAT.food_per_crew_day, CAT.water_per_crew_day      # kg per crew member per day
LANDER = {"capsule2": 1, "shelter2": 1, "supplies": 1, "solar_s": 1}


def moon_landing(name, modules=LANDER, crew=2, launch=ET):
    d = CraftDesign("medium", "storable", 12_000, dict(modules), crew)
    f = Flight.to_moon(name, d.physics_craft(), launch, 4 * DAY, periapsis_alt=100)
    f.design = d
    f.add_landing(launch, "Moon", launch + 4 * DAY)
    return f


def control_with(*flights_by_mission):
    mc = MissionControl()
    for name, flights in flights_by_mission:
        mc.add(Mission(name, flights))
    return mc


def landed(mc, days_after=0):
    """Run the clock until every flight has ended, plus `days_after`."""
    end = max(f.outcome.et for m in mc.missions for f in m.flights) + days_after * DAY
    mc.update(end)
    return end


# ---------- Arrivals ----------
def test_landing_delivers_everything_to_the_surface():
    mc = control_with(("Luna", [moon_landing("Lander")]))
    now = landed(mc)
    (a,) = arrivals(mc)
    assert (a.body, a.where, a.crew) == ("Moon", SURFACE, 2)
    assert a.modules == LANDER and a.stage == "storable"
    # Food and water were eaten on the way
    trip = (a.et - ET) / DAY
    assert a.food == pytest.approx(200 - 2 * FOOD * trip)
    assert a.water == pytest.approx(300 - 2 * WATER * trip)

    (site,) = sites_at(mc, now)
    assert site.name == "Moon (surface)" and site.crew == 2 and site.habitat_places == 2
    assert site.deaths == 0
    assert site.power_kw == pytest.approx(3 / 1.0 ** 2 - 1 - 1, abs=0.1)     # solar - capsule - shelter


def test_nothing_is_delivered_before_landing():
    mc = control_with(("Luna", [moon_landing("Lander")]))
    mc.update(ET + DAY)
    assert sites_at(mc, ET + DAY) == []


def test_supplies_run_down_and_then_the_crew_die():
    mc = control_with(("Luna", [moon_landing("Lander")]))
    now = landed(mc)
    site = sites_at(mc, now)[0]
    food_days, water_days = site.days_left()
    assert water_days < food_days                       # 300 kg water lasts less than 200 kg food here
    later = sites_at(mc, now + (water_days - 1) * DAY)[0]
    assert later.crew == 2 and later.water == pytest.approx(2 * WATER * 1, rel=1e-6)
    after = sites_at(mc, now + (water_days + 1) * DAY)[0]
    assert after.crew == 0 and after.deaths == 2
    death = after.log[-1]
    assert "water ran out" in death.text
    assert death.et == pytest.approx(now + water_days * DAY, abs=1)


def test_no_habitat_place_means_death_on_landing():
    mc = control_with(("Luna", [moon_landing("Lander", {"capsule2": 1, "supplies": 1, "solar_s": 1})]))
    now = landed(mc)
    site = sites_at(mc, now)[0]
    assert site.crew == 0 and site.deaths == 2
    assert "no habitat place" in site.log[-1].text


def test_a_second_delivery_adds_to_the_site():
    """A cargo run with a shelter and supplies lands after the crew: they live longer."""
    first = moon_landing("Lander")
    cargo = moon_landing("Cargo", {"capsule2": 1, "shelter2": 1, "supplies": 2, "solar_s": 1}, crew=1,
                         launch=ET + 10 * DAY)
    mc = control_with(("Luna", [first]), ("Luna cargo", [cargo]))
    now = landed(mc, 1)
    site = sites_at(mc, now)[0]
    assert site.crew == 3 and site.habitat_places == 4
    assert site.modules["supplies"] == 3 and site.stages["storable"] == 2
    assert len([e for e in site.log if "arrived" in e.text]) == 2


def test_order_of_arrivals_matters_for_shelter():
    """Crew first with no shelter die, even if a shelter lands a week later."""
    crew_first = moon_landing("Crew", {"capsule2": 1, "supplies": 1, "solar_s": 1})
    shelter_later = moon_landing("Shelter", {"capsule2": 1, "shelter2": 1, "supplies": 1}, crew=1,
                                 launch=ET + 7 * DAY)
    mc = control_with(("A", [crew_first]), ("B", [shelter_later]))
    now = landed(mc)
    site = sites_at(mc, now)[0]
    assert site.deaths == 2 and site.crew == 1


def test_orbit_sites_from_declared_orbits():
    f = Flight.earth_orbit("Station", CraftDesign("medium", "storable", 5000, dict(LANDER), 2).physics_craft(), ET)
    f.design = CraftDesign("medium", "storable", 5000, dict(LANDER), 2)
    mc = control_with(("Station", [f]))
    mc.update(ET + 2 * 3600)
    f.declare_complete(ET + 2 * 3600)
    (site,) = sites_at(mc, ET + 3 * 3600)
    assert (site.body, site.where, site.crew) == ("Earth", ORBIT, 2)


def test_returning_home_leaves_nothing_behind():
    f = Flight.earth_orbit("Hop", CraftDesign("medium", "storable", 5000, dict(LANDER), 2).physics_craft(), ET)
    f.design = CraftDesign("medium", "storable", 5000, dict(LANDER), 2)
    f.add_landing(ET + 3600, "Earth")
    mc = control_with(("Hop", [f]))
    landed(mc)
    assert f.status == COMPLETE and arrivals(mc) == []


# ---------- Starving in transit ----------
def test_crew_who_run_out_in_transit_are_lost():
    d = CraftDesign("medium", "storable", 5000, {"capsule2": 1, "solar_s": 1}, 2)     # no supplies at all
    f = Flight.earth_orbit("Doomed", d.physics_craft(), ET)
    f.design = d
    mc = control_with(("Doomed", [f]))
    mc.update(ET + 60)
    assert f.status == LOST and "ran out of supplies" in f.end_note
    assert mc.missions[0].status == LOST


def test_landing_before_supplies_run_out_is_fine():
    f = moon_landing("Lander")
    assert f.starve_et() > f.outcome.et
    mc = control_with(("Luna", [f]))
    landed(mc)
    assert f.status == COMPLETE


def test_transit_supplies_helper():
    d = CraftDesign("medium", "storable", 5000, {"capsule2": 1, "supplies": 1}, 2)
    assert transit_supplies(d, 10, CAT) == pytest.approx((200 - 2 * FOOD * 10, 300 - 2 * WATER * 10))
    assert transit_supplies(d, 100_000, CAT) == (0.0, 0.0)
