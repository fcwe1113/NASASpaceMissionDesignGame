"""Top-down 2D system map for the mission planner.

Shows a planned trajectory over the ecliptic, with a time slider that places the
craft and every body at the same moment, so you can see what the craft passes
and when. "Centre on" a body redraws everything relative to it (the craft's
path then shows its real shape around that body, e.g. a flyby hyperbola).
A table lists the closest approach to each body; selecting a row jumps there.

Mouse: scroll to zoom (around the cursor), drag to pan.
"""
import math
import tkinter as tk
from tkinter import ttk

import numpy as np

from physics import BODIES, PLANETS, ephemeris
from physics import bodies as phys_bodies
from physics.planning import FLYBY_BODIES, closest_approaches

DAY = 86_400.0
AU = 149_597_870.7
FRAMES = ["Sun"] + PLANETS + ["Moon"]
COLOURS = {"Sun": "#ffd84a", "Mercury": "#bbbbbb", "Venus": "#ff9a2e", "Earth": "#3a8dff",
           "Mars": "#ff4a3a", "Moon": "#dddddd", "Jupiter": "#d8a46a", "Saturn": "#e6cf8a",
           "Uranus": "#8fe3e8", "Neptune": "#4f6dff", "Pluto": "#c9a98f"}
BG, PATH, CRAFT, BURN, GRID = "#0d0d12", "#33d6ff", "#ffffff", "#ffa040", "#2a2a35"
MAX_PX = 30_000          # X11 draws in 16-bit coordinates; drop points much further out than this


def date_str(et):
    return ephemeris.et2str(et)[:17]


def fmt_km(d):
    if d >= 0.1 * AU:
        return f"{d / AU:.3f} AU"
    if d >= 1e6:
        return f"{d / 1e6:.2f} million km"
    return f"{d:,.0f} km"


class SystemMap(ttk.Frame):
    def __init__(self, parent, on_aim=None, on_assist=None, width=740, height=430):
        super().__init__(parent)
        self.on_aim = on_aim                  # callback(body, et) for "Aim to pass here…"
        self.on_assist = on_assist            # callback(body, et) for "Use as flyby…"
        self.W, self.H = width, height
        self.traj = None
        self.burn_ets = []
        self.frame_body = "Sun"
        self.center = np.zeros(2)             # km, in the frame body's coordinates
        self.scale = 1.8 * AU / (height / 2)  # km per pixel
        self._dirty = False
        self._drag = None
        self._fitted = False

        top = ttk.Frame(self)
        top.pack(fill="x")
        ttk.Label(top, text="Centre on").pack(side="left")
        self.frame_var = tk.StringVar(value="Sun")
        box = ttk.Combobox(top, textvariable=self.frame_var, values=FRAMES, state="readonly", width=9)
        box.pack(side="left", padx=4)
        box.bind("<<ComboboxSelected>>", lambda e: self.set_frame(self.frame_var.get()))
        ttk.Button(top, text="Fit path", command=self.fit).pack(side="left", padx=4)
        ttk.Label(top, text="Scroll to zoom, drag to pan", foreground="#777").pack(side="right")

        self.canvas = tk.Canvas(self, width=width, height=height, bg=BG, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, pady=(4, 0))
        self.canvas.bind("<ButtonPress-1>", self._press)
        self.canvas.bind("<B1-Motion>", self._move)
        self.canvas.bind("<MouseWheel>", lambda e: self._zoom(e.x, e.y, 1 / 1.25 if e.delta > 0 else 1.25))
        self.canvas.bind("<Button-4>", lambda e: self._zoom(e.x, e.y, 1 / 1.25))
        self.canvas.bind("<Button-5>", lambda e: self._zoom(e.x, e.y, 1.25))
        self.canvas.bind("<Configure>", self._resized)

        row = ttk.Frame(self)
        row.pack(fill="x", pady=(4, 0))
        self.t_var = tk.DoubleVar(value=0)
        self.slider = ttk.Scale(row, from_=0, to=1, variable=self.t_var, command=lambda _: self._draw_moment())
        self.slider.pack(side="left", fill="x", expand=True)
        self.readout = ttk.Label(self, foreground="#333", font=("TkDefaultFont", 9))
        self.readout.pack(anchor="w")

        lower = ttk.Frame(self)
        lower.pack(fill="x", pady=(4, 0))
        self.table = ttk.Treeview(lower, columns=("body", "dist", "tplus", "date", "note"), show="headings",
                                  height=5, selectmode="browse")
        for key, text, w in [("body", "Closest approach to", 120), ("dist", "Distance", 130),
                             ("tplus", "T+ days", 70), ("date", "Date (UTC)", 130), ("note", "", 200)]:
            self.table.heading(key, text=text)
            self.table.column(key, width=w, anchor="e" if key == "tplus" else "w")
        self.table.pack(side="left", fill="x", expand=True)
        self.table.bind("<<TreeviewSelect>>", lambda e: self._show_selected())
        side = ttk.Frame(lower)
        side.pack(side="left", padx=(6, 0), anchor="n")
        ttk.Button(side, text="Show on map", command=self._show_selected).pack(fill="x")
        ttk.Button(side, text="Aim to pass here…", command=self._aim_selected).pack(fill="x", pady=4)
        ttk.Button(side, text="Use as flyby…", command=self._assist_selected).pack(fill="x")
        self.approaches = []

        self.bind("<Map>", lambda e: self._compute_if_dirty())
        self._draw_empty()

    # ---------- Data ----------
    def set_trajectory(self, traj, burn_ets=()):
        """New plan to show. Heavy work waits until the map is actually visible."""
        self.traj = traj
        self.burn_ets = list(burn_ets)
        self._dirty = True
        if self.winfo_ismapped():
            self._compute_if_dirty()

    def clear(self):
        self.traj = None
        self._dirty = False
        self.table.delete(*self.table.get_children())
        self._draw_empty()

    def _compute_if_dirty(self):
        if not self._dirty or self.traj is None:
            return
        self._dirty = False
        traj = self.traj
        self.t0, self.t1 = traj.segments[0].start.et, traj.segments[-1].end_et
        # Samples: dense enough per segment; closed orbits get one dense lap, then a coarse rest
        # (flagged so it isn't drawn in that body's own frame, where it would just retrace the lap)
        self.samples = []                      # (segment body, ets, repeat-flag)
        for seg in traj.segments:
            a, b = seg.start.et, seg.end_et
            el = seg.start.elements
            period = el.period(seg.start.mu)
            if el.e < 1 and b - a > period:
                self.samples.append((seg.body, np.linspace(a, a + period, 240), False))
                self.samples.append((seg.body, np.linspace(a + period, b, 160), True))
            else:
                self.samples.append((seg.body, np.linspace(a, b, 300), False))
        self.helio = [traj.positions(ets) for _, ets, _ in self.samples]
        # Bodies along the same times (for drawing relative to a body)
        self.body_at = {name: [phys_bodies.heliocentric_state(name, ets)[0] for _, ets, _ in self.samples]
                        for name in FRAMES if name != "Sun"}
        # Planet orbit outlines (heliocentric), one period around launch
        self.orbits = {}
        for name in PLANETS:
            period = 2 * math.pi * math.sqrt(BODIES[name].a ** 3 / BODIES["Sun"].gm)
            # Centred on launch, and kept inside the ephemeris (one Pluto lap is 248 years)
            ets = np.clip(self.t0 + np.linspace(-period / 2, period / 2, 240),
                          ephemeris.COVER_MIN, ephemeris.COVER_MAX)
            self.orbits[name] = phys_bodies.heliocentric_state(name, ets)[0]
        moon_ets = self.t0 + np.linspace(0, 27.4 * DAY, 120)
        self.moon_orbit = phys_bodies.body_state("Moon", moon_ets)[0]
        self.approaches = closest_approaches(traj)
        self._fill_table()
        if not self._fitted:
            self.fit()
            self._fitted = True
        else:
            self.redraw()

    def _fill_table(self):
        self.table.delete(*self.table.get_children())
        for k, a in enumerate(self.approaches):
            if a.at_end:
                note = "still closing when the projection ends"
            elif a.altitude <= 0:
                note = "hits the surface"
            elif a.inside_soi:
                note = f"flyby, {fmt_km(a.altitude)} above surface"
            else:
                note = f"outside SOI ({fmt_km(BODIES[a.body].soi)})"
            self.table.insert("", "end", iid=str(k), values=(
                a.body, fmt_km(a.distance), f"{(a.et - self.t0) / DAY:.1f}", date_str(a.et), note))

    # ---------- View ----------
    def set_frame(self, body, keep_view=False):
        self.frame_body = body
        self.frame_var.set(body)
        if not keep_view:
            self.center = np.zeros(2)
            soi = BODIES[body].soi
            self.scale = (1.8 * AU if body == "Sun" else 2.5 * soi) / (min(self.W, self.H) / 2)
        self.redraw()

    def fit(self):
        """Zoom to the whole path in the current frame."""
        if self.traj is None or self._dirty:
            return
        pts = np.vstack([self._rel(k) for k in range(len(self.samples))])[:, :2]
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        self.center = (lo + hi) / 2
        span = max((hi - lo).max(), 1.0)
        self.scale = span * 1.15 / min(self.W, self.H)
        self.redraw()

    def _rel(self, k):
        """Sample block k relative to the current frame body."""
        if self.frame_body == "Sun":
            return self.helio[k]
        return self.helio[k] - self.body_at[self.frame_body][k]

    def _to_screen(self, xy):
        xy = np.atleast_2d(xy)
        sx = self.W / 2 + (xy[:, 0] - self.center[0]) / self.scale
        sy = self.H / 2 - (xy[:, 1] - self.center[1]) / self.scale
        return sx, sy

    def _polyline(self, pts, **kw):
        """Draw a polyline, split wherever points are too far off-screen for X11."""
        sx, sy = self._to_screen(pts[:, :2])
        ok = (np.abs(sx) < MAX_PX) & (np.abs(sy) < MAX_PX)
        run = []
        for x, y, good in zip(sx, sy, ok):
            if good:
                run += [x, y]
            else:
                if len(run) >= 4:
                    self.canvas.create_line(*run, **kw)
                run = []
        if len(run) >= 4:
            self.canvas.create_line(*run, **kw)

    def _dot(self, xy, r, fill, tag, label=None, outline="", left=False):
        sx, sy = self._to_screen(xy)
        x, y = float(sx[0]), float(sy[0])
        if abs(x) > MAX_PX or abs(y) > MAX_PX:
            return
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill=fill, outline=outline, tags=tag)
        # Skip a label that would sit on top of one already drawn (e.g. Moon on Earth)
        if label and any(abs(x - lx) < 40 and abs(y - ly) < 10 for lx, ly in self._labels):
            label = None
        if label:
            self._labels.append((x, y))
            self.canvas.create_text(x - r - 4 if left else x + r + 4, y, text=label, fill=fill,
                                    anchor="e" if left else "w", font=("TkDefaultFont", 8), tags=tag)

    def redraw(self):
        c = self.canvas
        c.delete("all")
        if self.traj is None or self._dirty:
            self._draw_empty()
            return
        f = self.frame_body
        if f == "Sun":
            for name, pts in self.orbits.items():
                self._polyline(pts, fill=self._dim(COLOURS[name]))
        elif f == "Earth":
            self._polyline(self.moon_orbit, fill=self._dim(COLOURS["Moon"]))
        if f != "Sun":
            # The frame body's sphere of influence, and the body itself at true size
            soi = BODIES[f].soi
            sx, sy = self._to_screen(np.zeros(2))
            r = soi / self.scale
            if r < MAX_PX:
                c.create_oval(sx[0] - r, sy[0] - r, sx[0] + r, sy[0] + r, outline=GRID, dash=(3, 4))
                c.create_text(sx[0], sy[0] - r - 8, text=f"{f}'s sphere of influence", fill="#666",
                              font=("TkDefaultFont", 8))
            rb = max(BODIES[f].radius / self.scale, 3)
            if rb < MAX_PX:
                c.create_oval(sx[0] - rb, sy[0] - rb, sx[0] + rb, sy[0] + rb, fill=COLOURS[f], outline="")

        for k, (body, _ets, repeat) in enumerate(self.samples):
            if repeat and body == f:
                continue
            self._polyline(self._rel(k), fill=PATH, width=2)

        # Burns along the path
        # Burn labels go on the left, out of the way of body labels (which go on the right)
        self._labels = []
        if self.burn_ets:
            for n, et in enumerate(self.burn_ets, 1):
                if self.t0 <= et <= self.t1:
                    p = self._positions_in_frame([et])
                    self._dot(p, 4, BURN, "static", label=f"burn {n}", left=True)
        self._scale_bar()
        self._draw_moment()

    def _positions_in_frame(self, ets):
        p = self.traj.positions(ets)
        if self.frame_body != "Sun":
            p = p - phys_bodies.heliocentric_state(self.frame_body, np.asarray(ets, float))[0]
        return p[:, :2]

    def _draw_moment(self):
        """Craft and bodies at the slider's time."""
        self.canvas.delete("moment")
        if self.traj is None or self._dirty:
            return
        self._labels = []
        t = self.t0 + self.t_var.get() * (self.t1 - self.t0)
        f = self.frame_body
        origin = np.zeros(3) if f == "Sun" else phys_bodies.heliocentric_state(f, t)[0]
        for name in FRAMES:
            if name == f:
                continue
            p = (np.zeros(3) if name == "Sun" else phys_bodies.heliocentric_state(name, t)[0]) - origin
            r = max(BODIES[name].radius / self.scale, 4 if name == "Sun" else 3)
            self._dot(p[:2], r, COLOURS[name], "moment", label=name)
            soi = BODIES[name].soi / self.scale
            if name != "Sun" and 6 < soi < MAX_PX:
                sx, sy = self._to_screen(p[:2])
                self.canvas.create_oval(sx[0] - soi, sy[0] - soi, sx[0] + soi, sy[0] + soi,
                                        outline=self._dim(COLOURS[name]), dash=(2, 4), tags="moment")
        craft = self._positions_in_frame([t])
        self._dot(craft, 5, CRAFT, "moment", outline=PATH)

        s = self.traj.state_at(t)
        nearest = min((a for a in self._distances_at(t)), key=lambda x: x[1])
        self.readout.config(text=(
            f"T+{(t - self.t0) / DAY:.1f} days   {date_str(t)}   orbiting {s.body}   "
            f"nearest: {nearest[0]} at {fmt_km(nearest[1])}"))

    def _distances_at(self, t):
        p = self.traj.positions([t])[0]
        for name in FLYBY_BODIES:
            yield name, float(np.linalg.norm(p - phys_bodies.heliocentric_state(name, t)[0]))

    def _scale_bar(self):
        target = 120 * self.scale                      # km for ~120 px
        mag = 10 ** math.floor(math.log10(target))
        nice = next(m * mag for m in (5, 2, 1) if m * mag <= target)
        px = nice / self.scale
        y = self.H - 14
        self.canvas.create_line(12, y, 12 + px, y, fill="#aaa", width=2)
        self.canvas.create_text(12 + px + 6, y, text=fmt_km(nice), fill="#aaa", anchor="w",
                                font=("TkDefaultFont", 8))

    def _draw_empty(self):
        self.canvas.delete("all")
        self.canvas.create_text(self.W / 2, self.H / 2, fill="#777",
                                text="Plan a launch to see it on the map")

    @staticmethod
    def _dim(col):
        r, g, b = (int(col[k:k + 2], 16) for k in (1, 3, 5))
        return f"#{r * 2 // 5:02x}{g * 2 // 5:02x}{b * 2 // 5:02x}"

    # ---------- Interaction ----------
    def _resized(self, e):
        self.W, self.H = e.width, e.height
        if self.traj is not None and not self._dirty:
            self.redraw()

    def _press(self, e):
        self._drag = (e.x, e.y, self.center.copy())

    def _move(self, e):
        if self._drag is None:
            return
        x0, y0, c0 = self._drag
        self.center = c0 + np.array([-(e.x - x0), e.y - y0]) * self.scale
        self.redraw()

    def _zoom(self, x, y, factor):
        # Keep the point under the cursor fixed
        wx = self.center[0] + (x - self.W / 2) * self.scale
        wy = self.center[1] - (y - self.H / 2) * self.scale
        self.scale *= factor
        self.center = np.array([wx - (x - self.W / 2) * self.scale, wy + (y - self.H / 2) * self.scale])
        self.redraw()

    def _selected_approach(self):
        sel = self.table.selection()
        return self.approaches[int(sel[0])] if sel else None

    def _show_selected(self):
        a = self._selected_approach()
        if a is None:
            return
        self.t_var.set((a.et - self.t0) / (self.t1 - self.t0) if self.t1 > self.t0 else 0)
        self.set_frame(a.body)
        # Zoom so the pass and the SOI both fit
        span = max(a.distance * 2.5, BODIES[a.body].soi * 1.2)
        self.scale = span / min(self.W, self.H)
        self.redraw()

    def _aim_selected(self):
        a = self._selected_approach()
        if a is not None and self.on_aim:
            self.on_aim(a.body, a.et)

    def _assist_selected(self):
        a = self._selected_approach()
        if a is not None and self.on_assist:
            self.on_assist(a.body, a.et)
