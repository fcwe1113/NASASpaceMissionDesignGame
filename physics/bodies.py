"""The bodies in the game and their sphere-of-influence (SOI) tree.

Sun -> Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune, Pluto;  Earth -> Moon.

The outer planets (and Pluto) are their system barycentres in DE440, so their GM
is the whole system's (planet plus moons). Their moons aren't in the ephemeris
file the game uses, so they aren't in the game.
"""
from dataclasses import dataclass

import numpy as np

from . import ephemeris


@dataclass(frozen=True)
class Body:
    name: str
    spice_id: str
    gm: float            # km^3/s^2 (DE440 values)
    radius: float        # km, mean
    parent: str | None
    a: float = 0.0       # km, semi-major axis of the orbit around the parent

    @property
    def soi(self):
        """Laplace sphere-of-influence radius, km (infinite for the Sun)."""
        if self.parent is None:
            return np.inf
        return self.a * (self.gm / BODIES[self.parent].gm) ** 0.4


BODIES = {b.name: b for b in [
    Body("Sun",     "10",  132_712_440_041.279419, 695_700.0, None),
    Body("Mercury", "199",          22_031.868551,   2_439.7, "Sun",   57_909_050.0),
    Body("Venus",   "299",         324_858.592000,   6_051.8, "Sun",  108_208_000.0),
    Body("Earth",   "399",         398_600.435507,   6_371.0, "Sun",  149_598_023.0),
    # Mars barycenter (499 isn't in de440); GM is the whole Mars system
    Body("Mars",    "4",            42_828.375816,   3_389.5, "Sun",  227_939_200.0),
    Body("Moon",    "301",           4_902.800118,   1_737.4, "Earth",    384_399.0),
    Body("Jupiter", "5",       126_712_764.100000,  69_911.0, "Sun",   778_570_000.0),
    Body("Saturn",  "6",        37_940_584.841800,  58_232.0, "Sun", 1_433_530_000.0),
    Body("Uranus",  "7",         5_794_556.400000,  25_362.0, "Sun", 2_872_460_000.0),
    Body("Neptune", "8",         6_836_527.100580,  24_622.0, "Sun", 4_495_060_000.0),
    Body("Pluto",   "9",               975.500000,   1_188.3, "Sun", 5_906_380_000.0),
]}

# Planets in order from the Sun (everything orbiting the Sun directly)
PLANETS = [b.name for b in BODIES.values() if b.parent == "Sun"]
INNER_PLANETS = ["Mercury", "Venus", "Earth", "Mars"]
# No solid surface: "landing" means sinking into the atmosphere, and nobody survives
GAS_GIANTS = {"Jupiter", "Saturn", "Uranus", "Neptune"}

CHILDREN = {name: [b.name for b in BODIES.values() if b.parent == name] for name in BODIES}


def body_state(name, et):
    """(r, v) of a body relative to its parent. The Sun is the origin."""
    b = BODIES[name]
    if b.parent is None:
        z = np.zeros(np.shape(et) + (3,))
        return z, z.copy()
    return ephemeris.state(b.spice_id, et, BODIES[b.parent].spice_id)


def path_to_root(name):
    """[name, parent, grandparent, ..., 'Sun']"""
    path = [name]
    while BODIES[path[-1]].parent is not None:
        path.append(BODIES[path[-1]].parent)
    return path


def heliocentric_state(name, et):
    """(r, v) of a body relative to the Sun."""
    r, v = np.zeros(3), np.zeros(3)
    for n in path_to_root(name)[:-1]:
        dr, dv = body_state(n, et)
        r, v = r + dr, v + dv
    return r, v


def convert(r, v, et, from_body, to_body):
    """Re-express a state relative to `from_body` as one relative to `to_body`."""
    if from_body == to_body:
        return np.asarray(r, float), np.asarray(v, float)
    r0, v0 = heliocentric_state(from_body, et)
    r1, v1 = heliocentric_state(to_body, et)
    return np.asarray(r) + r0 - r1, np.asarray(v) + v0 - v1
