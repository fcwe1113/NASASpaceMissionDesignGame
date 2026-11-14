"""Loads the JPL kernels once and gives body states from them.

Everything is in km, km/s and ephemeris time (ET, seconds past J2000), in the
ecliptic J2000 frame.
"""
from pathlib import Path

import numpy as np
import spiceypy as spice

ROOT = Path(__file__).resolve().parent.parent
# de440.bsp covers 1550-2650; the smaller de440s.bsp (1849-2150) is the fallback
EPHEMERIS = next((ROOT / f for f in ("de440.bsp", "de440s.bsp") if (ROOT / f).exists()),
                 ROOT / "de440.bsp")
spice.furnsh(str(ROOT / "naif0012.tls"))
spice.furnsh(str(EPHEMERIS))

FRAME = "ECLIPJ2000"
DAY = 86_400.0

# Exact time span the ephemeris file covers, less a day of margin at each end
_cov = spice.wnfetd(spice.spkcov(str(EPHEMERIS), 3), 0)
COVER_MIN, COVER_MAX = _cov[0] + DAY, _cov[1] - DAY


def state(target, et, center):
    """(r, v) of SPICE id `target` relative to `center` at `et`. Vectorised over et."""
    s, _ = spice.spkezr(target, et, FRAME, "NONE", center)
    s = np.asarray(s)
    return s[..., :3], s[..., 3:]


def str2et(date):
    return spice.str2et(date)


def et2str(et):
    return spice.et2utc(et, "C", 0)
