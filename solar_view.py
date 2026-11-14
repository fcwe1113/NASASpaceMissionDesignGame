"""
3D inner solar system view using real JPL ephemeris data, embedded in a host window.

The host (see main.py) owns the event loop: it creates a SolarSystemView inside one
of its native windows and calls step() repeatedly. Keyboard controls are replaced
by methods (set_focus, faster, slower, toggle_pause) the host can call.

Mouse controls inside the view:
    Right-drag      orbit camera      Scroll       zoom
    Middle-drag     pan (free camera only)
"""
from panda3d.core import WindowProperties, loadPrcFileData
from ursina import *
import spiceypy as spice
import numpy as np

# ---------- Ephemeris ----------
# physics.ephemeris loads the kernels and knows the time span they cover
from physics.ephemeris import COVER_MIN, COVER_MAX

AU_KM = 149_597_870.7
C_KM_S = 299_792.458
KM_PER_UNIT = 1000.0             # 1 game unit = 1,000 km (everything at true scale)
SCALE = AU_KM / KM_PER_UNIT      # game units per AU (~149,598)
MOON_EXAG = 1.0                  # true Moon distance
DAY = 86_400.0

ET_MIN, ET_MAX = COVER_MIN, COVER_MAX   # sim clock stays inside the coverage (trails are clipped to it too)

#           SPICE id, period (days), body colour,  orbit-line colour,       mean radius (km)
PLANETS = {
    "Mercury": ("199",  88, color.light_gray, color.light_gray.tint(-0.4), 2_439.7),
    "Venus":   ("299", 225, color.orange,     color.orange,                6_051.8),
    "Earth":   ("399", 365, color.azure,      color.azure.tint(-0.4),      6_371.0),
    "Mars":    ("4",   687, color.red,        color.red.tint(-0.4),        3_389.5),  # barycenter (499 isn't in de440/de440s)
}
SUN_RADIUS_KM = 695_700.0
MOON_RADIUS_KM = 1_737.4

# SPICE ids of every body shown, in display order (used by the info tables)
BODY_IDS = {"Sun": "10", **{n: v[0] for n, v in PLANETS.items()}, "Moon": "301"}


def spice_pos(target, et, center="10"):
    """Position of target relative to center, in AU, ecliptic J2000 frame."""
    p, _ = spice.spkpos(target, et, "ECLIPJ2000", "NONE", center)
    return np.asarray(p) / AU_KM


def to_game(p_au):
    """Ecliptic (x, y, z-up) -> Ursina (x, y-up, z)."""
    x, y, z = p_au * SCALE
    return Vec3(x, z, y)


def diameter(radius_km):
    """Sphere model has diameter 1, so scale = true diameter in game units."""
    return 2 * radius_km / KM_PER_UNIT


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
        rel = camera.getRelativePoint(self.body, Vec3(0, 0, 0))   # z = depth in front of camera (y-up)
        visible = rel[2] > 0
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


class SolarSystemView:
    """The Ursina scene, rendered into the native window `window_handle`.

    Ursina is a singleton, so only one of these can exist per process.
    """

    GLIDE_TIME = 0.6     # seconds to fly between targets
    PAN_SPEED = Vec2(20, 20)

    def __init__(self, window_handle, width, height, start_date="2026-11-14"):
        # Make Panda3D open its window as a child of the host's widget
        loadPrcFileData("", f"parent-window-handle {window_handle}")
        self.app = Ursina(title="Inner Solar System (JPL DE440)", size=(width, height),
                          fullscreen=False, borderless=True, development_mode=False)
        window.color = color.black
        self.resize(width, height)   # Ursina centres its window on the monitor; undo that

        self.sim_et = spice.str2et(start_date)
        self.days_per_second = 10.0
        self.paused = False

        # Runs the simulation step each frame. Created before the EditorCamera so it
        # runs first, like a script-level update() would.
        self._sim_updater = Entity()
        self._sim_updater.update = self._update_sim

        self.sun = Entity(model="sphere", color=color.yellow,
                          scale=diameter(SUN_RADIUS_KM), unlit=True)

        self.bodies = {}
        self.trails = []
        for name, (sid, period, col, line_col, radius_km) in PLANETS.items():
            self.trails.append(Trail(sid, "10", period, self.sim_et, line_col))
            self.bodies[name] = Entity(model="sphere", color=col, scale=diameter(radius_km))

        # Moon: its trail is relative to Earth, drawn on a pivot that follows Earth
        self.moon_pivot = Entity()
        self.trails.append(Trail("301", "399", 27.3, self.sim_et, color.gray, thickness=1,
                                 exag=MOON_EXAG, parent=self.moon_pivot, n_points=200))
        self.moon = Entity(model="sphere", color=color.white, scale=diameter(MOON_RADIUS_KM))

        # Anchored to the view's left edge; Ursina keeps UI x in proportion on resize
        self.date_text = Text(position=window.top_left + Vec2(0.02, -0.03), scale=1.2)

        self.markers = {"Sun": Marker(self.sun, "Sun", color.yellow, SUN_RADIUS_KM)}
        for name, (_sid, _p, col, _lc, radius_km) in PLANETS.items():
            self.markers[name] = Marker(self.bodies[name], name, col, radius_km)
        self.markers["Moon"] = Marker(self.moon, "Moon", color.white, MOON_RADIUS_KM)

        # At true scale the scene spans ~400,000 units but planets are only a few units
        # wide, so the camera needs a very long view distance and a small near plane.
        camera.clip_plane_near = 0.05
        camera.clip_plane_far = 2_000_000

        # EditorCamera is a pivot entity; the camera orbits around its position.
        # Move the pivot and the view centre moves with it. Its zoom is proportional
        # to distance, so scrolling works from whole-system view down to a single planet.
        self.cam = EditorCamera(rotation=(30, 0, 0), pan_speed=Vec2(0, 0))  # panning off while following
        camera.z = -700_000            # start with the whole inner solar system in view
        self.cam.target_z = camera.z   # EditorCamera eases camera.z toward target_z, so set both

        self.focus_targets = {"Sun": self.sun, **self.bodies, "Moon": self.moon}
        self.focus_target = self.sun   # None = free camera
        self.glide_t = 1.0             # 0 -> 1 during a glide; 1 = locked on
        self.glide_from = Vec3(0, 0, 0)

        # Ursina runs each entity's update() in the order the entities were created.
        # EditorCamera pans/rotates/zooms in its own update(), so markers placed
        # before it used last frame's camera and jittered whenever the camera moved.
        # Updating them from an entity created *after* the EditorCamera makes them
        # run last, once the camera has settled for this frame.
        self._late_updater = Entity()
        self._late_updater.update = self._update_markers

    # ---------- Host interface ----------
    def step(self):
        """Advance and render one frame."""
        self.app.taskMgr.step()

    def resize(self, width, height):
        props = WindowProperties()
        props.setOrigin(0, 0)
        props.setSize(max(1, int(width)), max(1, int(height)))
        self.app.win.requestProperties(props)

    def destroy(self):
        self.app.destroy()

    def set_focus(self, name):
        """Lock the view on a body by name, or pass None for a free camera."""
        self.focus_target = None if name is None else self.focus_targets[name]
        if self.focus_target is None:
            self.cam.pan_speed = self.PAN_SPEED   # free camera: middle-drag pans
        else:
            self.cam.pan_speed = Vec2(0, 0)       # locked: panning disabled
            self.glide_from = Vec3(self.cam.position)
            self.glide_t = 0.0

    def faster(self):
        self.days_per_second *= 1.1

    def slower(self):
        self.days_per_second /= 1.1

    def toggle_pause(self):
        self.paused = not self.paused

    def date_str(self):
        return spice.et2utc(self.sim_et, "C", 0)[:11]

    def heliocentric_rows(self):
        """(name, distance from Sun in AU, speed relative to Sun in km/s) per body."""
        rows = []
        for name, sid in BODY_IDS.items():
            if name == "Sun":
                continue
            s, _ = spice.spkezr(sid, self.sim_et, "ECLIPJ2000", "NONE", "10")
            rows.append((name, np.linalg.norm(s[:3]) / AU_KM, np.linalg.norm(s[3:])))
        return rows

    def geocentric_rows(self):
        """(name, distance from Earth in km, one-way light time in minutes) per body."""
        rows = []
        for name, sid in BODY_IDS.items():
            if name == "Earth":
                continue
            d = np.linalg.norm(spice_pos(sid, self.sim_et, center="399")) * AU_KM
            rows.append((name, d, d / C_KM_S / 60))
        return rows

    # ---------- Per-frame updates ----------
    def _update_sim(self):
        if not self.paused:
            self.sim_et = min(max(self.sim_et + time.dt * self.days_per_second * DAY, ET_MIN), ET_MAX)

        for name, (sid, *_rest) in PLANETS.items():
            self.bodies[name].position = to_game(spice_pos(sid, self.sim_et))

        earth = self.bodies["Earth"].position
        self.moon_pivot.position = earth
        self.moon.position = earth + to_game(spice_pos("301", self.sim_et, center="399") * MOON_EXAG)

        for trail in self.trails:
            trail.update(self.sim_et)

        # Keep the view centre exactly on the selected body.
        # During a switch, ease from the old centre to the (moving) new target,
        # then lock on with no lag.
        if self.focus_target is not None:
            target = self.focus_target.world_position
            if self.glide_t < 1.0:
                self.glide_t = min(1.0, self.glide_t + time.dt / self.GLIDE_TIME)
                s = self.glide_t * self.glide_t * (3 - 2 * self.glide_t)   # smoothstep
                self.cam.position = lerp(self.glide_from, target, s)
            else:
                self.cam.position = target

        self.date_text.text = (f"{self.date_str()}   "
                               f"{self.days_per_second:+.1f} days/s{'  (paused)' if self.paused else ''}")

    def _update_markers(self):
        for m in self.markers.values():
            m.update()
        # From far away the Moon's marker sits on top of Earth's; hide it until they separate
        e, mo = self.markers["Earth"], self.markers["Moon"]
        if mo.ring.enabled and e.ring.enabled and \
                distance_2d(e.screen_pos, mo.screen_pos) < e.ring_r + mo.ring_r:
            mo.hide()
