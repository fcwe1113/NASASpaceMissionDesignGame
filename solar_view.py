"""
3D inner solar system view using real JPL ephemeris data, embedded in a host window.

The host (see main.py) creates a SolarSystemView inside one of its native windows,
then calls run(), which hands the main loop to Ursina and calls the host back once
per frame so it can handle its own events. Keyboard controls are replaced
by methods (set_focus, faster, slower, toggle_pause) the host can call.

Mouse controls inside the view:
    Right-drag      orbit camera      Scroll       zoom
    Middle-drag     pan (while following a body, the view keeps moving with it;
                    press the body's camera button again to re-centre)
"""
from itertools import groupby

from panda3d.core import WindowProperties, loadPrcFileData
from ursina import *
import spiceypy as spice
import numpy as np

# ---------- Ephemeris ----------
# physics.ephemeris loads the kernels and knows the time span they cover
from physics.ephemeris import COVER_MIN, COVER_MAX
from physics.bodies import BODIES, body_state

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
BODY_COLORS = {"Sun": color.yellow, **{n: v[2] for n, v in PLANETS.items()}, "Moon": color.white}


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
        visible = rel[2] > 0 and self.body.enabled
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

    def destroy(self):
        destroy(self.ring)
        destroy(self.label)


# ---------- Mission paths ----------
def km_to_game(r):
    """Ecliptic km vector (x, y, z-up) -> Ursina (x, y-up, z) game units."""
    return Vec3(r[0] / KM_PER_UNIT, r[2] / KM_PER_UNIT, r[1] / KM_PER_UNIT)


class Visit:
    """A run of consecutive trajectory segments around one body, drawn on one anchor.

    The anchor sits where the body is *while the craft is there*: before the
    visit, where the body will be on arrival (with a dim "ghost" of the body to
    show it); during it, on the moving body; afterwards, where the body was when
    the craft left. So the arc into a planet's SOI meets the pass around it, and
    snaps onto the real planet as the clock catches up.
    """

    def __init__(self, view, body, segments, show_ghost):
        self.view, self.body = view, body
        self.t0, self.t1 = segments[0].start.et, segments[-1].end_et
        self.parent = BODIES[body].parent
        if self.parent:
            self.off0 = km_to_game(body_state(body, self.t0)[0])
            self.off1 = km_to_game(body_state(body, self.t1)[0])
        self.anchor = Entity()
        self.ghost = self.ghost_marker = None
        if show_ghost:
            col = BODY_COLORS[body]
            self.ghost = Entity(model="sphere", color=col.tint(-0.55), scale=diameter(BODIES[body].radius))
            self.ghost_marker = Marker(self.ghost, f"{body} on arrival", col.tint(-0.2), BODIES[body].radius)
        self.update(view.sim_et)

    def update(self, et):
        if self.parent is None:
            pos = Vec3(0, 0, 0)
        elif et < self.t0:
            pos = self.view.pivots[self.parent].world_position + self.off0
        elif et > self.t1:
            pos = self.view.pivots[self.parent].world_position + self.off1
        else:
            pos = self.view.pivots[self.body].world_position
        self.anchor.position = pos
        if self.ghost:
            self.ghost.enabled = et < self.t0
            self.ghost.position = pos

    def destroy(self):
        destroy(self.anchor)
        if self.ghost:
            destroy(self.ghost)
            self.ghost_marker.destroy()


class FlightPath:
    """A mission's path and (optionally) its craft.

    Each trajectory segment is drawn relative to the body it orbits (see Visit
    for where that body's frame is placed), so e.g. a parking orbit moves along
    with Earth while the craft is in it.
    """
    MAX_POINTS = 300

    def __init__(self, view, label, trajectory, col, craft_window=None, thickness=1.5):
        self.view, self.trajectory = view, trajectory
        self.craft_window = craft_window          # (from_et, to_et) when the craft is shown
        self.visits, self.lines = [], []
        for k, (body, segs) in enumerate(groupby(trajectory.segments, key=lambda s: s.body)):
            segs = list(segs)
            visit = Visit(view, body, segs, show_ghost=k > 0 and body != "Sun")
            self.visits.append(visit)
            self.lines += [Entity(parent=visit.anchor, color=col,
                                  model=Mesh(vertices=self._points(seg), mode="line", thickness=thickness))
                           for seg in segs]
        self.craft = Entity(model="sphere", color=col, scale=0.05, enabled=False)
        self.marker = Marker(self.craft, label, col, 0.0)

    @property
    def markers(self):
        return [self.marker] + [v.ghost_marker for v in self.visits if v.ghost_marker]

    def _points(self, seg):
        t0, t1 = seg.start.et, seg.end_et
        el = seg.start.elements
        if hasattr(seg, "ets"):                     # long burn: use its own samples
            step = max(1, len(seg.ets) // self.MAX_POINTS)
            pts = list(seg.rs[::step]) + [seg.rs[-1]]
        else:
            if el.e < 1:                            # closed orbit: one lap is enough
                t1 = min(t1, t0 + el.period(seg.start.mu))
            pts = [seg.state_at(t).r for t in np.linspace(t0, t1, self.MAX_POINTS)]
        return [km_to_game(p) for p in pts]

    def update(self, et):
        for visit in self.visits:
            visit.update(et)
        show = self.craft_window is not None and self.craft_window[0] <= et <= self.craft_window[1]
        self.craft.enabled = show
        if show:
            s = self.trajectory.state_at(et)
            self.craft.position = self.view.pivots[s.body].world_position + km_to_game(s.r)

    def destroy(self):
        for line in self.lines:
            destroy(line)
        for visit in self.visits:
            visit.destroy()
        destroy(self.craft)
        self.marker.destroy()


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
        self.paused = True             # the game starts paused

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

        # One pivot per body, following it, for drawing things in that body's frame
        self.pivots = {name: Entity() for name in BODY_IDS}
        self.paths = {}            # key -> FlightPath

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
        # Middle-drag pans at any time; while following a body the pan is kept as an offset from it
        self.cam = EditorCamera(rotation=(30, 0, 0), pan_speed=self.PAN_SPEED)
        camera.z = -700_000            # start with the whole inner solar system in view
        self.cam.target_z = camera.z   # EditorCamera eases camera.z toward target_z, so set both

        self.focus_targets = {"Sun": self.sun, **self.bodies, "Moon": self.moon}
        self.focus_target = self.sun   # None = free camera
        self.glide_t = 1.0             # 0 -> 1 during a glide; 1 = locked on
        self.glide_from = Vec3(0, 0, 0)
        self.focus_offset = Vec3(0, 0, 0)        # panned away from the followed body by this much
        self._last_follow_pos = Vec3(0, 0, 0)    # where the view centre was put last frame

        # Ursina runs each entity's update() in the order the entities were created.
        # EditorCamera pans/rotates/zooms in its own update(), so markers placed
        # before it used last frame's camera and jittered whenever the camera moved.
        # Updating them from an entity created *after* the EditorCamera makes them
        # run last, once the camera has settled for this frame.
        self._late_updater = Entity()
        self._late_updater.update = self._update_markers

    # ---------- Host interface ----------
    def run(self, on_frame):
        """Hand the main loop to Ursina. Never returns.

        on_frame() is called once per frame (the host uses it to let Tk handle its
        own events). When it returns False the host has closed, so the app exits.
        """
        def pump(task):
            if on_frame() is False:
                self.app.userExit()      # closes the 3D window and exits the program
            return task.cont
        self.app.taskMgr.add(pump, "host-pump")
        self.app.run(info=False)

    def resize(self, width, height):
        props = WindowProperties()
        props.setOrigin(0, 0)
        props.setSize(max(1, int(width)), max(1, int(height)))
        self.app.win.requestProperties(props)

    def destroy(self):
        self.app.destroy()

    def show_path(self, key, label, trajectory, col, craft_window=None, thickness=1.5):
        """Draw (or redraw) a flight path. craft_window = (from_et, to_et) to show the craft."""
        self.remove_path(key)
        self.paths[key] = FlightPath(self, label, trajectory, col, craft_window, thickness)

    def remove_path(self, key):
        path = self.paths.pop(key, None)
        if path is not None:
            if self.focus_target is path.craft:
                self.set_focus("Sun")
            path.destroy()

    def focus_craft(self, key):
        """Follow a mission's craft, if it is currently flying."""
        path = self.paths.get(key)
        if path is not None and path.craft.enabled:
            self._focus_on(path.craft)
            return True
        return False

    def set_focus(self, name):
        """Lock the view on a body by name, or pass None for a free camera."""
        self._focus_on(None if name is None else self.focus_targets[name])

    def _focus_on(self, entity):
        """Follow `entity` (None = free camera). Choosing a body again re-centres on it."""
        self.focus_target = entity
        self.focus_offset = Vec3(0, 0, 0)
        if self.focus_target is not None:
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

        for name, body in {**self.bodies, "Moon": self.moon}.items():
            self.pivots[name].position = body.position
        for path in self.paths.values():
            path.update(self.sim_et)

        # Keep the view centre on the selected body (plus any panning).
        # During a switch, ease from the old centre to the (moving) new target,
        # then lock on with no lag. The EditorCamera pans by moving its position
        # in its own update (after this one), so whatever it moved since last
        # frame is panning: keep it as an offset from the body.
        if self.focus_target is not None:
            target = self.focus_target.world_position
            if self.glide_t < 1.0:
                self.glide_t = min(1.0, self.glide_t + time.dt / self.GLIDE_TIME)
                s = self.glide_t * self.glide_t * (3 - 2 * self.glide_t)   # smoothstep
                self.cam.position = lerp(self.glide_from, target, s)
            else:
                self.focus_offset += self.cam.position - self._last_follow_pos
                self.cam.position = target + self.focus_offset
            self._last_follow_pos = Vec3(self.cam.position)

        self.date_text.text = (f"{self.date_str()}   "
                               f"{self.days_per_second:+.1f} days/s{'  (paused)' if self.paused else ''}")

    def _update_markers(self):
        for m in self.markers.values():
            m.update()
        for path in self.paths.values():
            for m in path.markers:
                m.update()
        # From far away the Moon's marker sits on top of Earth's; hide it until they separate
        e, mo = self.markers["Earth"], self.markers["Moon"]
        if mo.ring.enabled and e.ring.enabled and \
                distance_2d(e.screen_pos, mo.screen_pos) < e.ring_r + mo.ring_r:
            mo.hide()
