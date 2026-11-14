"""Physics core: patched-conic spaceflight using real JPL planet positions.

    bodies      the Sun, planets and Moon, their GM, radii and SOI tree
    ephemeris   loads the SPICE kernels; body states from DE440
    kepler      two-body propagation, orbital elements, event times
    lambert     Lambert solver and porkchop (launch window) searches
    trajectory  coasting through SOIs, burns, fuel, the orbit test

Units everywhere: km, km/s, kg, seconds of ephemeris time (ET).
"""
from .bodies import BODIES, CHILDREN
from .lambert import lambert, porkchop, escape_dv, Porkchop
from .trajectory import (State, Craft, Burn, Event, Trajectory, fly, coast, normalize,
                         is_stable_orbit, parking_orbit, departure_state, correction_burn,
                         capture_dv, LONG_BURN)
