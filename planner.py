"""Mission planner window (new missions, and viewing/changing ones already planned or flying).

The window edits a copy of the mission ("draft"). Nothing changes in the
campaign until Schedule/Save, and saving an active mission goes through
Mission.set_plan, which refuses changes to burns that have already happened.
"""
import copy
import math
from types import SimpleNamespace
import tkinter as tk
from tkinter import messagebox, ttk

import numpy as np

from game.mission import ACTIVE, PENDING, Mission, PlanError
from physics import BODIES, Burn, Craft, LONG_BURN, ephemeris, fly, porkchop
from physics import planning
from physics.assist import assist_search
from physics.planning import PlanningError
from system_map import SystemMap

DAY = 86_400.0
TARGETS = ["Earth orbit", "Moon", "Mercury", "Venus", "Mars"]
# Flight-time range (days) searched for each planet
TOF_RANGE = {"Mercury": (60, 200), "Venus": (70, 250), "Mars": (120, 450)}
ASSIST_LEG2_RANGE = (80, 500)      # days from the flyby planet on to the target
DEFAULT_CRAFT = Craft(dry_mass=2000, fuel=12_000, isp=320, thrust=50_000)
PREVIEW_KEY = "__preview__"


def date_str(et, with_time=True):
    s = ephemeris.et2str(et)
    return s[:17] if with_time else s[:11]


def parse_date(text):
    try:
        return ephemeris.str2et(text.strip())
    except Exception:
        raise PlanError(f"Can't read the date {text!r} (try e.g. 2026-11-14 or 2026-11-14 06:00)")


def parse_float(text, what):
    try:
        return float(text)
    except ValueError:
        raise PlanError(f"{what} must be a number")


def _guard(fn):
    """Run a planner handler, showing PlanError/PlanningError in its status line instead of crashing."""
    def wrapper(self, *a, **kw):
        self.status.config(text="", foreground="#a33")
        try:
            return fn(self, *a, **kw)
        except (PlanError, PlanningError) as exc:
            self._error(exc)
    return wrapper


class LabeledEntry(ttk.Frame):
    def __init__(self, parent, label, value, width=12, unit=""):
        super().__init__(parent)
        ttk.Label(self, text=label, width=16).pack(side="left")
        self.var = tk.StringVar(value=str(value))
        self.entry = ttk.Entry(self, textvariable=self.var, width=width)
        self.entry.pack(side="left")
        if unit:
            ttk.Label(self, text=unit).pack(side="left", padx=(4, 0))

    def get(self):
        return self.var.get()

    def set(self, value):
        self.var.set(str(value))

    def lock(self, locked=True):
        self.entry.config(state="disabled" if locked else "normal")


# ---------- Porkchop chart ----------
class PorkchopChart(tk.Canvas):
    """Delta-v over departure date (x) and flight time (y). Click to pick a transfer."""

    W, H = 440, 250            # plot area, pixels
    LEFT, BOTTOM, TOP = 48, 34, 8

    def __init__(self, parent, on_pick):
        super().__init__(parent, width=self.LEFT + self.W + 8, height=self.TOP + self.H + self.BOTTOM,
                         bg="#1e1e1e", highlightthickness=0)
        self.on_pick = on_pick
        self.pc = None
        self.image = None
        self.bind("<Button-1>", self._click)
        self.create_text(self.LEFT + self.W / 2, self.TOP + self.H / 2, fill="#888",
                         text="Find launch windows to see the chart", tags="hint")

    @staticmethod
    def _palette(n=64):
        """Green (cheap) -> yellow -> red (expensive)."""
        out = []
        for k in range(n):
            t = k / (n - 1)
            r = int(255 * min(1, 2 * t))
            g = int(255 * min(1, 2 * (1 - t)))
            out.append(f"#{r:02x}{g:02x}30")
        return out

    def clear(self):
        self.pc = None
        self.delete("all")
        self.create_text(self.LEFT + self.W / 2, self.TOP + self.H / 2, fill="#888",
                         text="Find launch windows to see the chart", tags="hint")

    def show(self, pc, max_dv, x_label="Departure date"):
        self.pc = pc
        self.delete("all")
        dv = pc.dv_total
        finite = np.isfinite(dv)
        if not finite.any():
            self.create_text(self.LEFT + self.W / 2, self.TOP + self.H / 2, fill="#888", text="No transfers found")
            return
        vmin = dv[finite].min()
        vmax = vmin * 2
        pal = self._palette()
        # One colour per grid cell: index into palette, or the extra "unaffordable"/"none" colours
        lut = np.array(pal + [self._dim(c) for c in pal] + ["#202020"])
        level = np.clip(((dv - vmin) / (vmax - vmin) * (len(pal) - 1)), 0, len(pal) - 1)
        level = np.where(finite, level, 0).astype(int)
        level = np.where(finite & (dv > max_dv), level + len(pal), level)
        level = np.where(finite, level, 2 * len(pal))

        n, m = dv.shape            # departures, flight times
        xi = (np.arange(self.W) * n // self.W)
        yj = ((self.H - 1 - np.arange(self.H)) * m // self.H)   # longest flight at the top
        colours = lut[level[xi[None, :], yj[:, None]]]
        self.image = tk.PhotoImage(width=self.W, height=self.H)
        self.image.put(" ".join("{" + " ".join(row) + "}" for row in colours))
        self.create_image(self.LEFT, self.TOP, image=self.image, anchor="nw")

        # Axes
        for k in range(5):
            i = k * (n - 1) // 4
            x = self.LEFT + (i + 0.5) * self.W / n
            self.create_text(x, self.TOP + self.H + 10, text=date_str(pc.depart_et[i], False), fill="#ccc",
                             font=("TkDefaultFont", 7))
        for k in range(5):
            j = k * (m - 1) // 4
            y = self.TOP + self.H - (j + 0.5) * self.H / m
            self.create_text(self.LEFT - 4, y, text=f"{pc.tof[j] / DAY:.0f} d", fill="#ccc", anchor="e",
                             font=("TkDefaultFont", 7))
        self.create_text(self.LEFT + self.W / 2, self.TOP + self.H + 25, fill="#ccc",
                         text=f"{x_label}   (green = cheap, {vmin:.1f} km/s; dim = beyond the craft)",
                         font=("TkDefaultFont", 7))

    @staticmethod
    def _dim(col):
        r, g, b = (int(col[k:k + 2], 16) for k in (1, 3, 5))
        return f"#{r // 4 + 30:02x}{g // 4 + 30:02x}{b // 4 + 40:02x}"

    def mark(self, i, j):
        self.delete("mark")
        n, m = self.pc.dv_total.shape
        x = self.LEFT + (i + 0.5) * self.W / n
        y = self.TOP + self.H - (j + 0.5) * self.H / m
        self.create_oval(x - 5, y - 5, x + 5, y + 5, outline="white", width=2, tags="mark")

    def _click(self, e):
        if self.pc is None:
            return
        n, m = self.pc.dv_total.shape
        x, y = e.x - self.LEFT, e.y - self.TOP
        if 0 <= x < self.W and 0 <= y < self.H:
            self.on_pick(int(x * n / self.W), int((self.H - 1 - y) * m / self.H))


# ---------- Small dialogs ----------
class FormDialog(tk.Toplevel):
    """Modal form: fields = [(key, label, default, unit)] or (key, label, choices-list, unit) for a dropdown."""

    def __init__(self, parent, title, fields, note=""):
        super().__init__(parent)
        self.title(title)
        self.transient(parent)
        self.result = None
        body = ttk.Frame(self, padding=10)
        body.pack(fill="both", expand=True)
        if note:
            ttk.Label(body, text=note, wraplength=360, foreground="#555").pack(anchor="w", pady=(0, 8))
        self.vars = {}
        for key, label, default, unit in fields:
            row = ttk.Frame(body)
            row.pack(fill="x", pady=2)
            ttk.Label(row, text=label, width=24).pack(side="left")
            if isinstance(default, list):
                var = tk.StringVar(value=default[0])
                ttk.Combobox(row, textvariable=var, values=default, state="readonly", width=14).pack(side="left")
            else:
                var = tk.StringVar(value=str(default))
                ttk.Entry(row, textvariable=var, width=16).pack(side="left")
            if unit:
                ttk.Label(row, text=unit).pack(side="left", padx=4)
            self.vars[key] = var
        buttons = ttk.Frame(body)
        buttons.pack(fill="x", pady=(10, 0))
        ttk.Button(buttons, text="OK", command=self._ok).pack(side="right")
        ttk.Button(buttons, text="Cancel", command=self.destroy).pack(side="right", padx=4)
        self.bind("<Return>", lambda e: self._ok())
        self.bind("<Escape>", lambda e: self.destroy())
        self.wait_visibility()
        self.grab_set()

    def _ok(self):
        self.result = {k: v.get() for k, v in self.vars.items()}
        self.destroy()

    def ask(self):
        self.wait_window()
        return self.result


# ---------- The planner ----------
class MissionPlanner(tk.Toplevel):
    def __init__(self, page, mission=None):
        super().__init__(page)
        self.page = page
        self.control = page.control
        self.view = page.view
        self.now = self.view.sim_et
        self.original = mission
        self.draft = copy.deepcopy(mission) if mission else None
        self.pc = None
        self.assist_grid = None        # gravity-assist search behind the chart, if "Via" is set
        self.title(f"Mission: {mission.name}" if mission else "New mission")
        self.geometry("1360x820")
        self.minsize(1330, 760)

        # Pause the clock while planning; restore it on close
        self._was_paused = self.view.paused
        self.view.paused = True
        self.protocol("WM_DELETE_WINDOW", self.close)

        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)
        left = ttk.Frame(self, padding=8)
        left.grid(row=0, column=0, sticky="ns")
        right = ttk.Frame(self, padding=8)
        right.grid(row=0, column=1, sticky="nsew")
        self._build_mission(left)
        self._build_craft(left)
        self._build_launch(left)
        self._build_summary(right)
        tabs = ttk.Notebook(right)
        tabs.pack(fill="both", expand=True, pady=(6, 0))
        plan_tab = ttk.Frame(tabs, padding=4)
        tabs.add(plan_tab, text="Burns & events")
        self._build_burns(plan_tab)
        self._build_projection(plan_tab)
        self.system_map = SystemMap(tabs, on_aim=lambda body, et: self._aim(body, et),
                                    on_assist=lambda body, et: self._gravity_assist(body, et))
        tabs.add(self.system_map, text="System map")
        self._build_buttons()

        self.status = ttk.Label(self, foreground="#a33", wraplength=1100, padding=(8, 0))
        self.status.grid(row=2, column=0, columnspan=2, sticky="w")
        self._apply_locks()
        if self.draft:
            self.horizon.set(f"{(self.draft.horizon_et - self.draft.launch_et) / DAY:.0f}")
        self._refresh()

    # ----- layout -----
    def _build_mission(self, parent):
        box = ttk.LabelFrame(parent, text="Mission", padding=6)
        box.pack(fill="x")
        default_name = self.original.name if self.original else self.control.unique_name("Mission")
        self.name = LabeledEntry(box, "Name", default_name, width=24)
        self.name.pack(anchor="w")
        row = ttk.Frame(box)
        row.pack(anchor="w", pady=2)
        ttk.Label(row, text="Target", width=16).pack(side="left")
        target = "Mars"
        if self.original:
            target = "Earth orbit" if self.original.target == "Earth" else self.original.target
        self.target = tk.StringVar(value=target)
        self.target_box = ttk.Combobox(row, textvariable=self.target, values=TARGETS, state="readonly", width=14)
        self.target_box.pack(side="left")
        self.target_box.bind("<<ComboboxSelected>>", lambda e: self._target_changed())

    def _build_craft(self, parent):
        box = ttk.LabelFrame(parent, text="Craft (fixed after launch)", padding=6)
        box.pack(fill="x", pady=(8, 0))
        c = self.original.craft if self.original else DEFAULT_CRAFT
        self.dry = LabeledEntry(box, "Dry mass", c.dry_mass, unit="kg")
        self.fuel = LabeledEntry(box, "Fuel", c.fuel, unit="kg")
        self.isp = LabeledEntry(box, "Engine Isp", c.isp, unit="s")
        self.thrust = LabeledEntry(box, "Thrust", c.thrust, unit="N")
        for w in (self.dry, self.fuel, self.isp, self.thrust):
            w.pack(anchor="w")
            w.var.trace_add("write", lambda *_: self._craft_changed())
        self.craft_info = ttk.Label(box, foreground="#555")
        self.craft_info.pack(anchor="w", pady=(4, 0))

    def _build_launch(self, parent):
        self.launch_box = ttk.LabelFrame(parent, text="Launch (fixed after launch)", padding=6)
        self.launch_box.pack(fill="both", expand=True, pady=(8, 0))
        self._launch_widgets()

    def _launch_widgets(self):
        for w in self.launch_box.winfo_children():
            w.destroy()
        box = self.launch_box
        target = self.target.get()
        today = date_str(self.now, False)
        tomorrow = date_str(self.now + DAY, False)   # a date-only "today" is already in the past
        if target in TOF_RANGE:
            self.search_from = LabeledEntry(box, "Search from", today, width=14)
            self.search_from.pack(anchor="w")
            self.search_span = LabeledEntry(box, "Search span", 800, unit="days")
            self.search_span.pack(anchor="w")
            self.arrive_alt = LabeledEntry(box, "Pass target at", 300, unit="km altitude")
            self.arrive_alt.pack(anchor="w")
            via_row = ttk.Frame(box)
            via_row.pack(anchor="w")
            ttk.Label(via_row, text="Via (gravity assist)", width=16).pack(side="left")
            self.via = tk.StringVar(value="None")
            via_box = ttk.Combobox(via_row, textvariable=self.via, state="readonly", width=12,
                                   values=["None"] + [p for p in ("Mercury", "Venus", "Mars") if p != target])
            via_box.pack(side="left")
            via_box.bind("<<ComboboxSelected>>", lambda e: self._via_changed())
            row = ttk.Frame(box)
            row.pack(anchor="w", pady=4)
            self.find_btn = ttk.Button(row, text="Find launch windows", command=self._find_windows)
            self.find_btn.pack(side="left")
            self.cheap_btn = ttk.Button(row, text="Cheapest", command=lambda: self._pick_best("fuel"))
            self.cheap_btn.pack(side="left", padx=4)
            self.fast_btn = ttk.Button(row, text="Fastest affordable", command=lambda: self._pick_best("speed"))
            self.fast_btn.pack(side="left")
            self.chart = PorkchopChart(box, self._pick)
            self.chart.pack(pady=4)
            self.pick_info = ttk.Label(box, justify="left")
            self.pick_info.pack(anchor="w")
        else:
            self.launch_date = LabeledEntry(box, "Launch date", tomorrow, width=18)
            self.launch_date.pack(anchor="w")
            if target == "Moon":
                self.moon_tof = LabeledEntry(box, "Flight time", 4, unit="days")
                self.moon_tof.pack(anchor="w")
                self.arrive_alt = LabeledEntry(box, "Pass Moon at", 100, unit="km altitude")
                self.arrive_alt.pack(anchor="w")
                text = "Plan transfer"
            else:
                self.orbit_alt = LabeledEntry(box, "Orbit altitude", 200, unit="km")
                self.orbit_alt.pack(anchor="w")
                text = "Set launch"
            self.set_launch_btn = ttk.Button(box, text=text, command=self._set_simple_launch)
            self.set_launch_btn.pack(anchor="w", pady=6)
            ttk.Label(box, wraplength=420, foreground="#555", text=(
                "Every mission starts in a circular parking orbit around Earth at launch "
                "(getting there is the launch vehicle's job). The Moon can be reached almost "
                "any day, so there's no launch window to search for.")).pack(anchor="w")
        if self.original and self.original.status != PENDING:
            ttk.Label(box, text=f"Launched {date_str(self.original.launch_et)}").pack(anchor="w", pady=4)

    def _build_burns(self, parent):
        box = ttk.LabelFrame(parent, text="Burns", padding=6)
        box.pack(fill="both", expand=True)
        cols = [("when", "Date (UTC)", 130), ("tplus", "T+ days", 70), ("pro", "Prograde m/s", 90),
                ("nrm", "Normal m/s", 80), ("rad", "Radial m/s", 80), ("dv", "Total m/s", 80),
                ("dur", "Duration", 90), ("state", "", 70)]
        self.burn_tree = ttk.Treeview(box, columns=[c[0] for c in cols], show="headings", height=7,
                                      selectmode="browse")
        for key, text, width in cols:
            self.burn_tree.heading(key, text=text)
            self.burn_tree.column(key, width=width, anchor="e" if key not in ("when", "state") else "w")
        self.burn_tree.pack(fill="both", expand=True)
        self.burn_tree.bind("<Double-1>", lambda e: self._edit_burn())
        row = ttk.Frame(box)
        row.pack(fill="x", pady=(4, 0))
        self.burn_buttons = [
            ttk.Button(row, text="Add burn…", command=self._add_burn),
            ttk.Button(row, text="Edit…", command=self._edit_burn),
            ttk.Button(row, text="Delete", command=self._delete_burn),
            ttk.Button(row, text="Aim at a body…", command=self._aim),
            ttk.Button(row, text="Gravity assist…", command=self._gravity_assist),
            ttk.Button(row, text="Capture into orbit…", command=self._capture),
        ]
        for b in self.burn_buttons:
            b.pack(side="left", padx=(0, 4))
        ttk.Label(box, foreground="#555", wraplength=640, text=(
            "Burns are relative to the body the craft is orbiting at the time. Burns that "
            f"take longer than {LONG_BURN / DAY:.0f} day are simulated as they happen; shorter "
            "ones are instant. Burns already carried out are locked.")).pack(anchor="w", pady=(4, 0))

    def _build_summary(self, parent):
        """How the plan ends and the fuel left; above the tabs so both show it."""
        self.outcome = ttk.Label(parent, font=("TkDefaultFont", 11, "bold"), wraplength=740, justify="left")
        self.outcome.pack(anchor="w")
        self.fuel_left = ttk.Label(parent)
        self.fuel_left.pack(anchor="w")

    def _build_projection(self, parent):
        box = ttk.LabelFrame(parent, text="Projected flight", padding=6)
        box.pack(fill="both", expand=True, pady=(8, 0))
        row = ttk.Frame(box)
        row.pack(fill="x")
        self.horizon = LabeledEntry(row, "Show path until T+", 30, unit="days")
        self.horizon.pack(side="left")
        ttk.Button(row, text="Update", command=self._set_horizon).pack(side="left", padx=6)
        self.event_tree = ttk.Treeview(box, columns=("when", "tplus", "event", "body", "note"),
                                       show="headings", height=8, selectmode="none")
        for key, text, width in [("when", "Date (UTC)", 130), ("tplus", "T+ days", 70),
                                 ("event", "Event", 110), ("body", "Body", 80), ("note", "Details", 260)]:
            self.event_tree.heading(key, text=text)
            self.event_tree.column(key, width=width, anchor="e" if key == "tplus" else "w")
        self.event_tree.pack(fill="both", expand=True, pady=(4, 0))

    def _build_buttons(self):
        row = ttk.Frame(self, padding=8)
        row.grid(row=1, column=0, columnspan=2, sticky="ew")
        ttk.Button(row, text="Close", command=self.close).pack(side="right")
        status = self.original.status if self.original else None
        if status is None:
            self.save_btn = ttk.Button(row, text="Schedule launch", command=self._save)
        else:
            self.save_btn = ttk.Button(row, text="Save changes", command=self._save)
        self.save_btn.pack(side="right", padx=4)
        if status == PENDING:
            ttk.Button(row, text="Cancel launch", command=self._cancel_launch).pack(side="left")
        if status == ACTIVE:
            self.declare_btn = ttk.Button(row, text="Declare mission complete", command=self._declare)
            self.declare_btn.pack(side="left")

    # ----- state helpers -----
    @property
    def launched(self):
        return self.original is not None and self.original.status != PENDING

    def _apply_locks(self):
        locked = self.launched
        for w in (self.dry, self.fuel, self.isp, self.thrust):
            w.lock(locked)
        self.target_box.config(state="disabled" if locked else "readonly")
        if locked:
            for w in self.launch_box.winfo_children():
                if isinstance(w, (ttk.Button, ttk.Entry)):
                    w.config(state="disabled")
                for child in w.winfo_children():
                    if isinstance(child, (ttk.Entry, ttk.Button)):
                        child.config(state="disabled")
            if hasattr(self, "chart"):
                self.chart.unbind("<Button-1>")

    def _craft(self):
        c = Craft(parse_float(self.dry.get(), "Dry mass"), parse_float(self.fuel.get(), "Fuel"),
                  parse_float(self.isp.get(), "Isp"), parse_float(self.thrust.get(), "Thrust"))
        if c.dry_mass <= 0 or c.fuel < 0 or c.isp <= 0 or c.thrust <= 0:
            raise PlanError("Masses, Isp and thrust must be positive")
        return c

    def _target_body(self):
        t = self.target.get()
        return "Earth" if t == "Earth orbit" else t

    def _error(self, exc):
        self.status.config(text=str(exc))

    def _info(self, text):
        self.status.config(text=text, foreground="#555")

    # ----- craft and target -----
    def _craft_changed(self):
        try:
            c = self._craft()
        except PlanError as exc:
            self.craft_info.config(text=str(exc))
            return
        self.craft_info.config(text=f"Δv available: {c.dv_remaining:.2f} km/s   "
                                    f"(full-tank burn takes {c.fuel / c.mass_flow / DAY:.2f} days)")
        if self.draft and not self.launched:
            self.draft.set_craft(c)
            self._refresh()

    def _target_changed(self):
        self.draft = None
        self.pc = self.assist_grid = None
        self._launch_widgets()
        self._refresh()

    # ----- launch: planets -----
    @_guard
    def _find_windows(self):
        target = self._target_body()
        start = max(parse_date(self.search_from.get()), self.now)
        span = parse_float(self.search_span.get(), "Search span")
        if not 10 <= span <= 5000:
            raise PlanError("Search span must be between 10 and 5000 days")
        via = self.via.get()
        if via != "None":
            self._find_assist_windows(target, via, start, span)
            return
        self.assist_grid = None
        lo, hi = TOF_RANGE[target]
        dep = start + np.linspace(0, span, 250) * DAY
        dep = dep[dep < ephemeris.COVER_MAX - hi * DAY]
        tof = np.linspace(lo, hi, 90) * DAY
        self.pc = porkchop("Earth", target, dep, tof)
        self.chart.show(self.pc, self._craft().dv_remaining)
        self._pick_best("fuel")

    def _find_assist_windows(self, target, via, start, span):
        """Search Earth -> via -> target routes; chart = best route per (flyby date, leg-1 time)."""
        lo1, hi1 = TOF_RANGE[via]
        lo2, hi2 = ASSIST_LEG2_RANGE
        flyby = start + lo1 * DAY + np.arange(0, span, max(2.0, span / 200)) * DAY
        flyby = flyby[flyby < ephemeris.COVER_MAX - hi2 * DAY]
        grid = assist_search("Earth", via, target, flyby, np.linspace(lo1, hi1, 40) * DAY,
                             np.linspace(lo2, hi2, 40) * DAY)
        self.assist_grid = grid
        total = grid.dv_total
        # The chart shows, for each (flyby date, leg-1 time), the cheapest leg-2 time
        self.pc = SimpleNamespace(dv_total=total.min(axis=2), depart_et=grid.flyby_et, tof=grid.tof1,
                                  best_k=total.argmin(axis=2))
        self.chart.show(self.pc, self._craft().dv_remaining,
                        x_label=f"{via} flyby date (y: days from Earth to {via})")
        self._pick_best("fuel")

    def _via_changed(self):
        self.pc = self.assist_grid = None
        self.chart.clear()
        self.pick_info.config(text="")

    @_guard
    def _pick_best(self, by):
        if self.pc is None:
            raise PlanError("Find launch windows first")
        max_dv = self._craft().dv_remaining if by == "speed" else np.inf
        if self.assist_grid is not None:
            idx = self.assist_grid.best(by, max_dv=max_dv)
            if idx is None:
                raise PlanError("No route on the chart is affordable with this craft")
            self._pick_assist(*idx)
            return
        idx = self.pc.best(by, max_dv=max_dv)
        if idx is None:
            raise PlanError("No transfer on the chart is affordable with this craft")
        self._pick(*idx)

    @_guard
    def _pick_assist(self, f, m, k):
        g = self.assist_grid
        if not np.isfinite(g.dv_total[f, m, k]):
            raise PlanError("No route there (the flyby would have to pass too low)")
        self.chart.mark(f, m)
        dep, fb, arr = g.route(f, m, k)
        via, target = g.via, g.target
        self.pick_info.config(text=(
            f"Leave {date_str(dep)}, pass {via} {date_str(fb)}, arrive {date_str(arr)} "
            f"({(arr - dep) / DAY:.0f} days)\n"
            f"Departure {g.dv_depart[f, m]:.2f} km/s, flyby top-up {g.dv_flyby[f, m, k]:.2f}, "
            f"orbit capture {g.dv_arrive[f, k]:.2f}, total {g.dv_total[f, m, k]:.2f} km/s; "
            f"{via} pass {g.flyby_alt[f, m, k]:,.0f} km up"))
        alt = parse_float(self.arrive_alt.get(), "Pass altitude")
        draft = Mission.via_assist(self.name.get().strip() or "Mission", target, self._craft(),
                                   dep, fb, arr, g.v_inf_depart[f, m], via, alt)
        self._info(f"Gravity assist planned: {via} bends the path {draft.assist.turn_deg:.0f}°. "
                   f"Burns: departure, flyby set-up, a touch-up after {via}. "
                   f"Add a capture burn to stay at {target}.")
        self._set_draft(draft)

    @_guard
    def _pick(self, i, j):
        if self.assist_grid is not None:
            self._pick_assist(i, j, int(self.pc.best_k[i, j]))
            return
        pc = self.pc
        dep, tof = pc.depart_et[i], pc.tof[j]
        if not np.isfinite(pc.dv_total[i, j]):
            raise PlanError("No transfer there")
        self.chart.mark(i, j)
        self.pick_info.config(text=(
            f"Leave {date_str(dep)}, arrive {date_str(dep + tof)} ({tof / DAY:.0f} days)\n"
            f"Departure burn {pc.dv_depart[i, j]:.2f} km/s, orbit capture {pc.dv_arrive[i, j]:.2f} km/s, "
            f"total {pc.dv_total[i, j]:.2f} km/s   (C3 {pc.c3[i, j]:.1f} km²/s²)"))
        craft = self._craft()
        draft = Mission.to_planet(self.name.get().strip() or "Mission", self._target_body(), craft,
                                  dep, dep + tof, pc.v_inf_depart[i, j])
        # Patched conics leave a small miss from Earth's SOI; aim properly a day after leaving it
        alt = parse_float(self.arrive_alt.get(), "Pass altitude")
        exit_et = planning.soi_exit_time(draft.trajectory, "Earth")
        if exit_et is not None and exit_et + DAY < dep + tof:
            before = fly(draft.start, craft, draft.burns, exit_et + DAY)
            aim, got = planning.aim_burn(before.end, before.craft, self._target_body(), dep + tof, alt)
            draft.set_plan(draft.burns + [aim])
            self._info(f"Added a correction burn a day after leaving Earth's SOI: passes "
                       f"{self._target_body()} at {got:.0f} km. Add a capture burn to stay in orbit.")
        self._set_draft(draft)

    # ----- launch: Moon / Earth orbit -----
    @_guard
    def _set_simple_launch(self):
        et = parse_date(self.launch_date.get())
        if et < self.now:
            raise PlanError("Launch can't be in the past")
        craft = self._craft()
        name = self.name.get().strip() or "Mission"
        if self.target.get() == "Moon":
            tof = parse_float(self.moon_tof.get(), "Flight time")
            if not 2 <= tof <= 10:
                raise PlanError("Flight time to the Moon should be 2-10 days")
            draft = Mission.to_moon(name, craft, et, tof * DAY,
                                    parse_float(self.arrive_alt.get(), "Pass altitude"))
        else:
            alt = parse_float(self.orbit_alt.get(), "Orbit altitude")
            if alt < 100:
                raise PlanError("Orbit altitude must be at least 100 km")
            draft = Mission.earth_orbit(name, craft, et, alt)
        self._set_draft(draft)

    def _set_draft(self, draft):
        self.draft = draft
        self.horizon.set(f"{(draft.horizon_et - draft.launch_et) / DAY:.0f}")
        self._refresh()

    # ----- burns -----
    def _need_draft(self):
        if self.draft is None:
            raise PlanError("Set up the launch first")

    def _selected_burn(self):
        sel = self.burn_tree.selection()
        if not sel:
            raise PlanError("Select a burn first")
        return self.draft.burns[int(sel[0])]

    def _burn_form(self, title, burn=None, note=""):
        d = self.draft
        t = (burn.et - d.launch_et) / DAY if burn else max(0.0, (self.now - d.launch_et) / DAY) + 1
        res = FormDialog(self, title, [
            ("t", "Days after launch", f"{t:.4f}", "days"),
            ("pro", "Prograde", f"{(burn.prograde if burn else 0) * 1000:.1f}", "m/s"),
            ("nrm", "Normal", f"{(burn.normal if burn else 0) * 1000:.1f}", "m/s"),
            ("rad", "Radial (out)", f"{(burn.radial if burn else 0) * 1000:.1f}", "m/s"),
        ], note=note).ask()
        if res is None:
            return None
        et = self.draft.launch_et + parse_float(res["t"], "Days after launch") * DAY
        return Burn(et, parse_float(res["pro"], "Prograde") / 1000, parse_float(res["nrm"], "Normal") / 1000,
                    parse_float(res["rad"], "Radial") / 1000)

    def _check_not_past(self, et):
        if self.launched and et <= self.now:
            raise PlanError("That time has already passed")

    @_guard
    def _add_burn(self):
        self._need_draft()
        b = self._burn_form("Add burn", note="Prograde speeds you up along your path, retrograde "
                                              "(negative) slows you; normal tilts the orbit; radial "
                                              "pushes away from (or towards) the body you orbit.")
        if b:
            self._check_not_past(b.et)
            self.draft.set_plan(self.draft.burns + [b])
            self._refresh()

    @_guard
    def _edit_burn(self):
        self._need_draft()
        old = self._selected_burn()
        if self.draft.is_locked(old, self.now):
            raise PlanError("That burn has already happened")
        b = self._burn_form("Edit burn", old)
        if b:
            self._check_not_past(b.et)
            self.draft.set_plan([x for x in self.draft.burns if x is not old] + [b])
            self._refresh()

    @_guard
    def _delete_burn(self):
        self._need_draft()
        old = self._selected_burn()
        if self.draft.is_locked(old, self.now):
            raise PlanError("That burn has already happened")
        self.draft.set_plan([x for x in self.draft.burns if x is not old])
        self._refresh()

    def _body_choices(self):
        target = self._target_body()
        others = [b for b in ["Moon", "Mercury", "Venus", "Mars", "Earth"] if b != target]
        return [target] + others if target in BODIES else others

    @_guard
    def _aim(self, body=None, arrive=None):
        """Aim dialog; the system map pre-fills `body` and `arrive` from a closest approach."""
        self._need_draft()
        d = self.draft
        choices = self._body_choices()
        if body is not None:
            choices = [body] + [b for b in choices if b != body]
            start = self._default_aim_time(arrive)
        else:
            exit_et = planning.soi_exit_time(d.trajectory, "Earth")
            start = max(self.now + 60, (exit_et + DAY) if exit_et else d.launch_et)
            visit = planning.first_visit(d.trajectory, self._target_body())
            arrive = visit.start.et if visit else start + 200 * DAY
        res = FormDialog(self, "Aim at a body", [
            ("body", "Body", choices, ""),
            ("t", "Burn at", f"{(start - d.launch_et) / DAY:.2f}", "days after launch"),
            ("arrive", "Arrive at", f"{(arrive - d.launch_et) / DAY:.2f}", "days after launch"),
            ("alt", "Closest approach", "300", "km altitude"),
            ("side", "Pass on side", ["+", "-"], ""),
        ], note="Works out the burn that reaches the body at the given time and height. "
                "The craft must be orbiting the body's parent then (e.g. the Sun for planets, "
                "Earth for the Moon). Later burns, planned for the old path, are removed.").ask()
        if res is None:
            return
        et = d.launch_et + parse_float(res["t"], "Burn time") * DAY
        arrive_et = d.launch_et + parse_float(res["arrive"], "Arrival time") * DAY
        self._check_not_past(et)
        earlier = [b for b in d.burns if b.et < et]
        before = fly(d.start, d.craft, earlier, et)
        if before.end_reason == "impact":
            raise PlanError("The craft has already landed by then")
        burn, got = planning.aim_burn(before.end, before.craft, res["body"], arrive_et,
                                      parse_float(res["alt"], "Closest approach"),
                                      side=1 if res["side"] == "+" else -1)
        dropped = len(d.burns) - len(earlier)
        d.set_plan(earlier + [burn], horizon_et=max(d.horizon_et, arrive_et + 30 * DAY))
        self.horizon.set(f"{(d.horizon_et - d.launch_et) / DAY:.0f}")
        self._refresh()
        self._info(f"Aimed at {res['body']}: closest approach {got:.0f} km"
                   + (f"; removed {dropped} later burn(s)" if dropped else ""))

    @_guard
    def _gravity_assist(self, via=None, flyby=None):
        """Dialog: fly past `via` so it bends the path on to a target. The map pre-fills via/flyby."""
        self._need_draft()
        d = self.draft
        planets = ["Mercury", "Venus", "Mars"]
        if via is None:
            # Default to the first planet the plan already passes, if any
            passes = [a for a in planning.closest_approaches(d.trajectory) if a.body in planets]
            via, flyby = (passes[0].body, passes[0].et) if passes else ("Venus", None)
        burn_default = self._default_aim_time(flyby) if flyby else self._default_aim_time(math.inf)
        if flyby is None:
            flyby = burn_default + 120 * DAY
        targets = [self._target_body()] + [b for b in planets + ["Earth"] if b != self._target_body()]
        res = FormDialog(self, "Gravity assist", [
            ("via", "Fly past", [via] + [p for p in planets if p != via], ""),
            ("t", "Set-up burn at", f"{(burn_default - d.launch_et) / DAY:.2f}", "days after launch"),
            ("flyby", "Pass it at", f"{(flyby - d.launch_et) / DAY:.2f}", "days after launch"),
            ("target", "Then go to", [t for t in targets if t != via], ""),
            ("arrive", "Arrive at", "auto", "days after launch (auto = cheapest that works)"),
            ("alt", "Pass target at", "300", "km altitude"),
        ], note="Works out the pass that makes the planet bend your path towards the target, "
                "the burn that sets it up, and a small touch-up after the flyby. Later burns "
                "are replaced. The System map's closest-approach table can fill in the first "
                "two times; Find launch windows with 'Via' set plans the whole route.").ask()
        if res is None:
            return
        if res["via"] == res["target"]:
            raise PlanError("Fly past one body to reach another")
        t = d.launch_et + parse_float(res["t"], "Set-up burn time") * DAY
        self._check_not_past(t)
        arrive = res["arrive"].strip().lower()
        arrive_et = None if arrive in ("", "auto") else d.launch_et + parse_float(arrive, "Arrival time") * DAY
        note = d.add_assist(t, res["via"], d.launch_et + parse_float(res["flyby"], "Pass time") * DAY,
                            res["target"], arrive_et, parse_float(res["alt"], "Pass altitude"))
        self.horizon.set(f"{(d.horizon_et - d.launch_et) / DAY:.0f}")
        self._refresh()
        self._info(note)

    def _default_aim_time(self, arrive):
        """A day after the last thing that happens before `arrive` (a burn, or leaving an SOI)."""
        d = self.draft
        before = [e.et for e in d.trajectory.events if e.kind in ("burn", "soi_exit") and e.et < arrive]
        last = max(before + [d.launch_et, self.now if self.launched else d.launch_et])
        return last + DAY if last + DAY < arrive else (last + arrive) / 2

    @_guard
    def _capture(self):
        self._need_draft()
        res = FormDialog(self, "Capture into orbit", [("body", "Body", self._body_choices(), "")],
                         note="Adds a retrograde burn at the craft's closest approach that leaves "
                              "it in a circular orbit at that height.").ask()
        if res is None:
            return
        d = self.draft
        burn = planning.capture_burn(d.trajectory, res["body"])
        self._check_not_past(burn.et)
        d.set_plan([b for b in d.burns if b.et < burn.et] + [burn],
                   horizon_et=max(d.horizon_et, burn.et + 3 * DAY))
        self.horizon.set(f"{(d.horizon_et - d.launch_et) / DAY:.0f}")
        self._refresh()
        self._info(f"Capture burn of {burn.dv * 1000:.0f} m/s added at {date_str(burn.et)}")

    @_guard
    def _set_horizon(self):
        self._need_draft()
        days = parse_float(self.horizon.get(), "Days")
        if days <= 0:
            raise PlanError("Days must be positive")
        self.draft.set_plan(self.draft.burns, horizon_et=self.draft.launch_et + days * DAY)
        self._refresh()

    # ----- refresh -----
    def _refresh(self):
        self._craft_changed_info()
        self.burn_tree.delete(*self.burn_tree.get_children())
        self.event_tree.delete(*self.event_tree.get_children())
        d = self.draft
        if d is None:
            self.outcome.config(text="No launch planned yet")
            self.fuel_left.config(text="")
            self.view.remove_path(PREVIEW_KEY)
            self.system_map.clear()
            return
        for k, b in enumerate(d.burns):
            dur = d.burn_time(b)
            dur_s = "instant" if dur <= LONG_BURN else f"{dur / DAY:.1f} days"
            self.burn_tree.insert("", "end", iid=str(k), values=(
                date_str(b.et), f"{(b.et - d.launch_et) / DAY:.2f}", f"{b.prograde * 1000:.1f}",
                f"{b.normal * 1000:.1f}", f"{b.radial * 1000:.1f}", f"{b.dv * 1000:.1f}", dur_s,
                "done" if d.is_locked(b, self.now) else "planned"))
        traj = d.trajectory
        for e in traj.events:
            self.event_tree.insert("", "end", values=(
                date_str(e.et), f"{(e.et - d.launch_et) / DAY:.2f}", e.kind.replace("_", " "), e.body, e.note))
        out = d.outcome
        text = out.describe()
        if out.ends_mission or out.kind == "flyby":
            text += f" on {date_str(out.et, False)}"
        if out.kind == "flyby":
            text += "\n" + self._capture_hint(d, out.body)
        self.outcome.config(text=text, foreground="#2a7" if out.ends_mission else "#b70")
        c = traj.craft
        self.fuel_left.config(text=f"Fuel left at the end: {c.fuel:,.0f} kg ({c.dv_remaining:.2f} km/s of Δv)")
        if hasattr(self, "declare_btn"):
            self.declare_btn.config(state="normal" if self.original.can_declare_complete(self.now) else "disabled")
        self.view.show_path(PREVIEW_KEY, f"{self.name.get()} (plan)", traj, self.page.PREVIEW_COLOR, thickness=2)
        self.system_map.set_trajectory(traj, [b.et for b in d.burns])

    @staticmethod
    def _capture_hint(mission, body):
        """What it would take to stay at `body` instead of flying past."""
        try:
            burn = planning.capture_burn(mission.trajectory, body)
        except PlanningError:
            return f"To stay, aim for a pass above {body}'s surface, then add a capture burn."
        earlier = [b for b in mission.burns if b.et < burn.et]
        have = fly(mission.start, mission.craft, earlier, burn.et).craft.dv_remaining
        enough = "enough fuel" if have >= burn.dv else "NOT enough fuel"
        return (f"To stay in orbit, add Capture into orbit… — needs {burn.dv:.2f} km/s, "
                f"the craft will have {have:.2f} km/s ({enough}).")

    def _craft_changed_info(self):
        try:
            c = self._craft()
            self.craft_info.config(text=f"Δv available: {c.dv_remaining:.2f} km/s   "
                                        f"(full-tank burn takes {c.fuel / c.mass_flow / DAY:.2f} days)")
        except PlanError as exc:
            self.craft_info.config(text=str(exc))

    # ----- save / close -----
    @_guard
    def _save(self):
        self._need_draft()
        name = self.name.get().strip()
        if not name:
            raise PlanError("Give the mission a name")
        if self.original is None:
            if self.draft.launch_et < self.now:
                raise PlanError("Launch can't be in the past")
            self.draft.name = name
            self.control.add(self.draft)
            self.page.mission_changed(self.draft)
        else:
            if name != self.original.name and any(m.name == name for m in self.control.missions):
                raise PlanError(f"There is already a mission called {name!r}")
            if self.original.status == PENDING:
                self.original.set_start(self.draft.start, self.draft.burns, self.draft.horizon_et)
                self.original.set_craft(self.draft.craft)
                self.original.target = self.draft.target
            else:
                self.original.set_plan(self.draft.burns, self.draft.horizon_et, now=self.now)
            old_name = self.original.name
            self.original.name = name
            self.page.mission_changed(self.original, old_name)
        self.close()

    @_guard
    def _cancel_launch(self):
        if messagebox.askyesno("Cancel launch", f"Cancel {self.original.name}?", parent=self):
            self.control.remove(self.original)
            self.page.mission_changed(self.original, removed=True)
            self.close()

    @_guard
    def _declare(self):
        self.original.declare_complete(self.now)
        self.page.mission_changed(self.original)
        self.close()

    def close(self):
        self.view.remove_path(PREVIEW_KEY)
        self.view.paused = self._was_paused
        self.page.planner_closed(self)
        self.destroy()
