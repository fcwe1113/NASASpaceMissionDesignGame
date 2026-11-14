"""
3D inner solar system viewer using real JPL ephemeris data.

Setup:
    pip install ursina spiceypy numpy
    Put naif0012.tls and de440s.bsp in the same folder as this script.

Controls:
    Right-drag      orbit camera      Scroll       zoom
    0-5             lock view on Sun, Mercury, Venus, Earth, Mars, Moon
    F               free camera (then middle-drag pans)
    Up / Down       speed up / slow down time
    Space           pause / resume    R            reverse time
"""
from ursina import *
import spiceypy as spice
import numpy as np

# ---------- Ephemeris ----------
spice.furnsh("naif0012.tls")
spice.furnsh("de440s.bsp")

AU_KM = 149_597_870.7
KM_PER_UNIT = 1000.0             # 1 game unit = 1,000 km (everything at true scale)
SCALE = AU_KM / KM_PER_UNIT      # game units per AU (~149,598)
MOON_EXAG = 1.0                  # true Moon distance
DAY = 86_400.0

# Exact time span the ephemeris file covers (trails are clipped to this)
_cov = spice.wnfetd(spice.spkcov("de440s.bsp", 3), 0)
COVER_MIN, COVER_MAX = _cov[0] + DAY, _cov[1] - DAY

ET_MIN = spice.str2et("1850-01-01")   # sim clock stays inside the coverage
ET_MAX = spice.str2et("2150-01-01")

#           SPICE id, period (days), body colour,  orbit-line colour,       mean radius (km)
PLANETS = {
    "Mercury": ("199",  88, color.light_gray, color.light_gray.tint(-0.4), 2_439.7),
    "Venus":   ("299", 225, color.orange,     color.orange,                6_051.8),
    "Earth":   ("399", 365, color.azure,      color.azure.tint(-0.4),      6_371.0),
    "Mars":    ("4",   687, color.red,        color.red.tint(-0.4),        3_389.5),  # barycenter (499 isn't in de440s)
}
SUN_RADIUS_KM = 695_700.0
MOON_RADIUS_KM = 1_737.4


def spice_pos(target, et, center="10"):
    """Position of target relative to center, in AU, ecliptic J2000 frame."""
    p, _ = spice.spkpos(target, et, "ECLIPJ2000", "NONE", center)
    return np.asarray(p) / AU_KM


def to_game(p_au):
    """Ecliptic (x, y, z-up) -> Ursina (x, y-up, z)."""
    x, y, z = p_au * SCALE
    return Vec3(x, z, y)


class Trail:
    """Orbit line covering half a lap behind to half a lap ahead of the current time.

    Orbits change slightly every lap (planets pull on each other; the Sun pulls hard
    on the Moon), so a fixed loop drawn once drifts away from the real body.
    This rebuilds the line whenever the sim clock moves on by 1/100 of a lap.
    """

    def __init__(self, target, center, period_days, et, col, thickness=1.5,
                 exag=1.0, parent=scene, n_points=300):
        self.target, self.center, self.exag = target, center, exag
        self.half = period_days / 2 * DAY
        self.refresh = period_days / 100 * DAY
        self.n = n_points
        self.entity = Entity(parent=parent, color=col,
                             model=Mesh(vertices=self._points(et), mode="line",
                                        thickness=thickness))
        self.built_at = et

    def _points(self, et):
        ets = np.clip(np.linspace(et - self.half, et + self.half, self.n),
                      COVER_MIN, COVER_MAX)
        p, _ = spice.spkpos(self.target, ets, "ECLIPJ2000", "NONE", self.center)  # vectorised
        p = np.asarray(p) / AU_KM * self.exag * SCALE
        return [Vec3(x, z, y) for x, y, z in p]

    def update(self, et):
        if abs(et - self.built_at) > self.refresh:
            self.entity.model.vertices = self._points(et)
            self.entity.model.generate()
            self.built_at = et


# ---------- Scene ----------
app = Ursina(title="Inner Solar System (JPL DE440)")
window.color = color.black

sim_et = spice.str2et("2026-11-14")
days_per_second = 10.0
paused = False

def diameter(radius_km):
    """Sphere model has diameter 1, so scale = true diameter in game units."""
    return 2 * radius_km / KM_PER_UNIT


sun = Entity(model="sphere", color=color.yellow, scale=diameter(SUN_RADIUS_KM), unlit=True)

bodies = {}
trails = []
for name, (sid, period, col, line_col, radius_km) in PLANETS.items():
    trails.append(Trail(sid, "10", period, sim_et, line_col))
    bodies[name] = Entity(model="sphere", color=col, scale=diameter(radius_km))

# Moon: its trail is relative to Earth, drawn on a pivot that follows Earth
moon_pivot = Entity()
trails.append(Trail("301", "399", 27.3, sim_et, color.gray, thickness=1,
                    exag=MOON_EXAG, parent=moon_pivot, n_points=200))
moon = Entity(model="sphere", color=color.white, scale=diameter(MOON_RADIUS_KM))

date_text = Text(position=(-0.85, 0.47), scale=1.2)


# ---------- Location markers ----------
# Real-scale bodies are usually far smaller than a pixel, so each one gets a
# screen-space ring and name label drawn on the UI layer. The ring keeps a
# minimum on-screen size, and grows to enclose the body when you zoom in close.
MARKER_MIN_RADIUS = 0.012   # UI units (screen height = 1)
MARKER_PADDING = 1.3        # ring radius vs the body's on-screen radius when zoomed in


def project(cam_space):
    """Camera-space point -> UI-space Vec2 (same maths as Entity.screen_position)."""
    p = camera.lens.getProjectionMat().xform(Vec4(cam_space[0], cam_space[1], cam_space[2], 1))
    return Vec2(p[0] / p[3] * camera.aspect_ratio / 2, p[1] / p[3] / 2)


class Marker:
    def __init__(self, body, name, col, radius_km):
        self.body = body
        self.radius = radius_km / KM_PER_UNIT
        self.ring = Entity(parent=camera.ui, color=col,
                           model=Circle(48, mode="line", thickness=1.5))
        self.label = Text(name, parent=camera.ui, color=col, origin=(-0.5, 0), scale=0.8)
        self.screen_pos = Vec2(0, 0)
        self.ring_r = MARKER_MIN_RADIUS

    def update(self):
        rel = camera.getRelativePoint(self.body, Vec3(0, 0, 0))   # y = depth in front of camera
        visible = rel[1] > 0
        self.ring.enabled = self.label.enabled = visible
        if not visible:
            return
        self.screen_pos = project(rel)
        edge = project((rel[0] + self.radius, rel[1], rel[2]))
        body_r = abs(edge.x - self.screen_pos.x)
        self.ring_r = max(MARKER_MIN_RADIUS, body_r * MARKER_PADDING)
        self.ring.position = self.screen_pos
        self.ring.scale = self.ring_r * 2                         # Circle mesh has radius 0.5
        self.label.position = (self.screen_pos.x + self.ring_r + 0.006, self.screen_pos.y)

    def hide(self):
        self.ring.enabled = self.label.enabled = False


markers = {"Sun": Marker(sun, "Sun", color.yellow, SUN_RADIUS_KM)}
for name, (_sid, _p, col, _lc, radius_km) in PLANETS.items():
    markers[name] = Marker(bodies[name], name, col, radius_km)
markers["Moon"] = Marker(moon, "Moon", color.white, MOON_RADIUS_KM)


def update_markers():
    for m in markers.values():
        m.update()
    # From far away the Moon's marker sits on top of Earth's; hide it until they separate
    e, mo = markers["Earth"], markers["Moon"]
    if mo.ring.enabled and e.ring.enabled and \
            distance_2d(e.screen_pos, mo.screen_pos) < e.ring_r + mo.ring_r:
        mo.hide()

# At true scale the scene spans ~400,000 units but planets are only a few units
# wide, so the camera needs a very long view distance and a small near plane.
camera.clip_plane_near = 0.05
camera.clip_plane_far = 2_000_000

# EditorCamera is a pivot entity; the camera orbits around its position.
# Move the pivot and the view centre moves with it. Its zoom is proportional
# to distance, so scrolling works from whole-system view down to a single planet.
PAN_SPEED = Vec2(20, 20)
cam = EditorCamera(rotation=(30, 0, 0), pan_speed=Vec2(0, 0))  # panning off while following
camera.z = -700_000          # start with the whole inner solar system in view
cam.target_z = camera.z      # EditorCamera eases camera.z toward target_z, so set both

FOCUS_KEYS = {"0": sun, "1": bodies["Mercury"], "2": bodies["Venus"],
              "3": bodies["Earth"], "4": bodies["Mars"], "5": moon}
focus_target = sun   # None = free camera
GLIDE_TIME = 0.6     # seconds to fly between targets
glide_t = 1.0        # 0 -> 1 during a glide; 1 = locked on
glide_from = Vec3(0, 0, 0)


def set_focus(target):
    global focus_target, glide_t, glide_from
    focus_target = target
    if target is None:
        cam.pan_speed = PAN_SPEED          # free camera: middle-drag pans
    else:
        cam.pan_speed = Vec2(0, 0)         # locked: panning disabled
        glide_from = Vec3(cam.position)
        glide_t = 0.0


def update():
    global sim_et
    if not paused:
        sim_et = min(max(sim_et + time.dt * days_per_second * DAY, ET_MIN), ET_MAX)

    for name, (sid, *_rest) in PLANETS.items():
        bodies[name].position = to_game(spice_pos(sid, sim_et))

    earth = bodies["Earth"].position
    moon_pivot.position = earth
    moon.position = earth + to_game(spice_pos("301", sim_et, center="399") * MOON_EXAG)

    for trail in trails:
        trail.update(sim_et)

    # Keep the view centre exactly on the selected body.
    # During a switch, ease from the old centre to the (moving) new target,
    # then lock on with no lag.
    global glide_t
    if focus_target is not None:
        target = focus_target.world_position
        if glide_t < 1.0:
            glide_t = min(1.0, glide_t + time.dt / GLIDE_TIME)
            s = glide_t * glide_t * (3 - 2 * glide_t)   # smoothstep
            cam.position = lerp(glide_from, target, s)
        else:
            cam.position = target

    date_text.text = (f"{spice.et2utc(sim_et, 'C', 0)[:11]}   "
                      f"{days_per_second:+.1f} days/s{'  (paused)' if paused else ''}")


def input(key):
    global days_per_second, paused
    if key in FOCUS_KEYS:
        set_focus(FOCUS_KEYS[key])
    elif key == "f":                   # release to free camera
        set_focus(None)
    elif key in ("up arrow", "up arrow hold"):
        days_per_second *= 1.1
    elif key in ("down arrow", "down arrow hold"):
        days_per_second /= 1.1
    elif key == "space":
        paused = not paused
    elif key == "r":
        days_per_second = -days_per_second


# Ursina runs this script's update() first, then each entity's update() in the
# order the entities were created. EditorCamera pans/rotates/zooms in its own
# update(), so markers placed in update() used last frame's camera and jittered
# whenever the camera moved. Updating them from an entity created *after* the
# EditorCamera makes them run last, once the camera has settled for this frame.
late_updater = Entity()
late_updater.update = update_markers

app.run()