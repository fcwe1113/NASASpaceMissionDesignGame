import numpy as np
import pytest

from physics import bodies, ephemeris
from physics.bodies import BODIES

ET = ephemeris.str2et("2026-11-14")


@pytest.mark.parametrize("name, soi", [
    ("Earth", 924_000), ("Moon", 66_100), ("Mars", 577_000), ("Venus", 616_000), ("Mercury", 112_000),
])
def test_soi_radii_match_textbook_values(name, soi):
    assert BODIES[name].soi == pytest.approx(soi, rel=0.01)


def test_sun_soi_is_infinite():
    assert BODIES["Sun"].soi == np.inf


def test_soi_tree():
    assert bodies.CHILDREN["Sun"] == ["Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune",
                                      "Pluto"]
    assert bodies.CHILDREN["Earth"] == ["Moon"]
    assert bodies.path_to_root("Moon") == ["Moon", "Earth", "Sun"]


def test_earth_is_about_one_au_from_the_sun():
    r, v = bodies.body_state("Earth", ET)
    assert np.linalg.norm(r) == pytest.approx(1.496e8, rel=0.02)
    assert np.linalg.norm(v) == pytest.approx(29.8, rel=0.02)


def test_heliocentric_moon_is_earth_plus_moon_offset():
    r_moon, _ = bodies.heliocentric_state("Moon", ET)
    r_earth, _ = bodies.body_state("Earth", ET)
    r_rel, _ = bodies.body_state("Moon", ET)
    assert np.allclose(r_moon, r_earth + r_rel)


def test_convert_round_trips():
    r, v = np.array([1e5, 2e5, 3e3]), np.array([1.0, -0.5, 0.1])
    r2, v2 = bodies.convert(r, v, ET, "Moon", "Sun")
    r3, v3 = bodies.convert(r2, v2, ET, "Sun", "Moon")
    assert np.allclose(r3, r, atol=1e-6) and np.allclose(v3, v, atol=1e-9)


def test_ephemeris_covers_the_campaign():
    assert ephemeris.COVER_MIN < ET < ephemeris.COVER_MAX
    if ephemeris.EPHEMERIS.name == "de440.bsp":
        assert ephemeris.COVER_MAX > ephemeris.str2et("2600-01-01")
