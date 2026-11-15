"""Mission planner window (new missions, and viewing/changing ones already planned or flying).

A mission is one launch of one or more craft. Each craft is built from
catalogue parts (Build craft tab) and has its own route (Route tab); the first
craft given a route sets the launch date for all of them.

The window edits copies ("drafts"). Nothing changes in the campaign until
Schedule/Save, and saving an active mission goes through Flight.set_plan,
which refuses changes to burns that have already happened.
"""
import copy
from dataclasses import dataclass, replace
import math
from types import SimpleNamespace
import tkinter as tk
from tkinter import messagebox, ttk

import numpy as np

from game import catalogue as catalogue_module
from game.design import BAD, INFO, OK, WARN, CraftDesign, assess, starter_design
from game.legs import apply_leg_start, leg_targets, next_leg_name, next_leg_start
from game.mission import ACTIVE, IDLE, PENDING, Flight, Mission, PlanError, park_alt
from physics import BODIES, GAS_GIANTS, PLANETS, Burn, LONG_BURN, ephemeris, fly, porkchop
from physics import planning
from physics.assist import assist_search
from physics.planning import PlanningError
from system_map import SystemMap

DAY = 86_400.0
TARGETS = ["Earth orbit", "Moon"] + [p for p in PLANETS if p != "Earth"]
# Flight-time range (days) searched for each planet
TOF_RANGE = {"Mercury": (60, 200), "Venus": (70, 250), "Mars": (120, 450)}
ASSIST_LEG2_RANGE = (80, 500)      # days from the flyby planet on to the target
PREVIEW_KEY = "__preview__"
LEVEL_COLOURS = {OK: "#2a7", WARN: "#c80", BAD: "#c33", INFO: "#777"}


@dataclass
class CraftDraft:
    """One craft being planned: its design, target and (once it has a route) its flight."""
    name: str
    design: CraftDesign
    target: str = "Mars"             # a route option label (see MissionPlanner._route_options)
    flight: Flight | None = None
    origin: str = "Earth"            # where it leaves from (another body for a next leg)
    leg_of: str | None = None        # the craft this one is the next leg of
    leg_start: object = None         # legs.LegStart: what it takes with it


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
        self.catalogue = catalogue_module.default()
        if mission:
            self.crafts = []
            for f in mission.flights:
                f = copy.deepcopy(f)
                design = f.design or starter_design()
                self.crafts.append(CraftDraft(f.name, design, f.target, f, origin=f.start.body, leg_of=f.leg_of))
        else:
            self.crafts = [CraftDraft("Craft 1", starter_design())]
        self.cur = 0                   # index of the craft being edited
        self.pc = None
        self.assist_grid = None        # gravity-assist search behind the chart, if "Via" is set
        self._preview_keys = set()
        self.title(f"Mission: {mission.name}" if mission else "New mission")
        # Fit on the screen above the taskbar: the action buttons are along the bottom edge
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        w, h = min(1400, sw - 20), min(880, sh - 110)
        self.geometry(f"{w}x{h}+{max(0, (sw - w) // 2)}+{max(0, (sh - 80 - h) // 2)}")
        self.minsize(min(1370, w), min(820, h))
        if sw < 1450 or sh < 960:
            self.state("zoomed")          # small or scaled-up screen (e.g. 1080p at 150%): use all of it

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
        # Packed before the tabs (at the bottom) so it keeps its space on small screens
        self._build_health(left)
        left_tabs = ttk.Notebook(left)
        left_tabs.pack(fill="both", expand=True, pady=(8, 0))
        self.left_tabs = left_tabs          # the Guide switches between Build craft and Route
        build_tab = ttk.Frame(left_tabs, padding=6)
        left_tabs.add(build_tab, text="Build craft")
        route_tab = ttk.Frame(left_tabs, padding=6)
        left_tabs.add(route_tab, text="Route")
        self.build_tab, self.route_tab = build_tab, route_tab
        self._build_craft(build_tab)
        self._build_launch(route_tab)
        self._build_summary(right)
        tabs = ttk.Notebook(right)
        tabs.pack(fill="both", expand=True, pady=(6, 0))
        self.plan_tabs = tabs
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
        self._select_craft(0)

    # ----- layout -----
    def _build_mission(self, parent):
        box = ttk.LabelFrame(parent, text="Mission (one launch)", padding=6)
        box.pack(fill="x")
        default_name = self.original.name if self.original else self.control.unique_name("Mission")
        self.name = LabeledEntry(box, "Name", default_name, width=24)
        self.name.pack(anchor="w")
        row = ttk.Frame(box)
        row.pack(anchor="w", pady=2)
        ttk.Label(row, text="Craft", width=16).pack(side="left")
        self.craft_var = tk.StringVar()
        self.craft_box = ttk.Combobox(row, textvariable=self.craft_var, state="readonly", width=14)
        self.craft_box.pack(side="left")
        self.craft_box.bind("<<ComboboxSelected>>", lambda e: self._select_craft(self.craft_box.current()))
        self.add_craft_btn = ttk.Button(row, text="Add craft", command=self._add_craft)
        self.add_craft_btn.pack(side="left", padx=4)
        self.remove_craft_btn = ttk.Button(row, text="Remove", command=self._remove_craft)
        self.remove_craft_btn.pack(side="left")
        self.next_leg_btn = ttk.Button(row, text="Plan next leg", command=self._plan_next_leg)
        self.next_leg_btn.pack(side="left", padx=4)
        self.craft_name = LabeledEntry(box, "Craft name", "", width=24)
        self.craft_name.pack(anchor="w")
        self.craft_name.entry.bind("<FocusOut>", lambda e: self._rename_craft())
        self.craft_name.entry.bind("<Return>", lambda e: self._rename_craft())
        self.mission_info = ttk.Label(box, foreground="#555", wraplength=500, justify="left")
        self.mission_info.pack(anchor="w", pady=(4, 0))

    def _build_craft(self, parent):
        """The craft builder: launcher, stage, fuel, crew and modules from the catalogue."""
        cat = self.catalogue
        self.launcher_ids = list(cat.launchers)
        self.stage_ids = list(cat.stages)
        self.module_ids = list(cat.modules)

        def picker(label, values):
            row = ttk.Frame(parent)
            row.pack(anchor="w", pady=1)
            ttk.Label(row, text=label, width=12).pack(side="left")
            box = ttk.Combobox(row, values=values, state="readonly", width=46)
            box.pack(side="left")
            return box

        self.launcher_box = picker("Launcher", [
            f"{l.name} — lifts {l.payload_kg / 1000:,.0f} t, ${l.cost_musd:,.0f}M"
            + (f"  (tier {l.tier})" if l.tier else "") for l in cat.launchers.values()])
        self.stage_box = picker("Stage", [
            f"{s.name} — Isp {s.isp_s:,.0f} s" + (", can relaunch" if s.relaunch else "")
            + (f"  (tier {s.tier}, secret)" if s.secret else f"  (tier {s.tier})" if s.tier else "")
            for s in cat.stages.values()])
        for box in (self.launcher_box, self.stage_box):
            box.bind("<<ComboboxSelected>>", lambda e: self._design_changed())

        row = ttk.Frame(parent)
        row.pack(anchor="w", pady=1)
        ttk.Label(row, text="Fuel", width=12).pack(side="left")
        self.fuel_var = tk.StringVar()
        self.fuel_entry = ttk.Entry(row, textvariable=self.fuel_var, width=10)
        self.fuel_entry.pack(side="left")
        self.fuel_max = ttk.Label(row, foreground="#555")
        self.fuel_max.pack(side="left", padx=4)
        self.fill_btn = ttk.Button(row, text="Fill tank", command=self._fill_tank)
        self.fill_btn.pack(side="left")
        ttk.Label(row, text="Crew", width=6).pack(side="left", padx=(12, 0))
        self.crew_var = tk.StringVar()
        self.crew_spin = ttk.Spinbox(row, from_=0, to=20, textvariable=self.crew_var, width=4)
        self.crew_spin.pack(side="left")
        for var in (self.fuel_var, self.crew_var):
            var.trace_add("write", lambda *_: self._design_changed())

        self.module_tree = ttk.Treeview(parent, columns=("name", "count", "mass", "power", "cost"),
                                        show="headings", height=8, selectmode="browse")
        for key, text, w in [("name", "Module", 190), ("count", "On board", 65), ("mass", "Mass", 60),
                             ("power", "Power", 60), ("cost", "Cost", 60)]:
            self.module_tree.heading(key, text=text)
            self.module_tree.column(key, width=w, anchor="w" if key == "name" else "e")
        self.module_tree.pack(fill="x", pady=(6, 0))
        self.module_tree.bind("<<TreeviewSelect>>", lambda e: self._show_module_info())
        self.module_tree.bind("<Double-1>", lambda e: self._change_module(+1))
        row = ttk.Frame(parent)
        row.pack(anchor="w", pady=4)
        self.module_buttons = [ttk.Button(row, text="Add one", command=lambda: self._change_module(+1)),
                               ttk.Button(row, text="Remove one", command=lambda: self._change_module(-1))]
        for b in self.module_buttons:
            b.pack(side="left", padx=(0, 4))
        self.part_info = ttk.Label(parent, foreground="#555", wraplength=500, justify="left")
        self.part_info.pack(anchor="w")
        self._loading_design = False

    def _build_health(self, parent):
        box = ttk.LabelFrame(parent, text="Craft check", padding=6)
        self.health_box = box
        box.pack(side="bottom", fill="x", pady=(8, 0))
        self.health_rows = []
        for _ in range(7):
            row = ttk.Frame(box)
            row.pack(anchor="w")
            dot = tk.Label(row, text="●", font=("TkDefaultFont", 11))
            dot.pack(side="left")
            name = ttk.Label(row, width=12, font=("TkDefaultFont", 9, "bold"))
            name.pack(side="left")
            text = ttk.Label(row, wraplength=420, justify="left")
            text.pack(side="left")
            self.health_rows.append((dot, name, text))

    def _build_launch(self, parent):
        row = ttk.Frame(parent)
        row.pack(anchor="w", pady=(0, 4))
        ttk.Label(row, text="Target", width=16).pack(side="left")
        self.target = tk.StringVar(value="Mars")
        self.target_box = ttk.Combobox(row, textvariable=self.target, values=TARGETS, state="readonly", width=18)
        self.target_box.pack(side="left")
        self.target_box.bind("<<ComboboxSelected>>", lambda e: self._target_changed())
        self.clear_route_btn = ttk.Button(row, text="Clear route", command=self._clear_route)
        self.clear_route_btn.pack(side="left", padx=6)
        self.launch_box = ttk.Frame(parent)
        self.launch_box.pack(fill="both", expand=True)

    def _route_options(self):
        """[(label, kind, body)] the current craft can head for. kind: planet/moon/parent/park."""
        c = self.craft
        if c.leg_of is None:
            return [("Earth orbit", "park", "Earth"), ("Moon", "moon", "Moon"),
                    *[(p, "planet", p) for p in PLANETS if p != "Earth"]]
        return leg_targets(c.origin)

    def _route(self):
        """(kind, body) for the chosen target."""
        for label, kind, body in self._route_options():
            if label == self.target.get():
                return kind, body
        label, kind, body = self._route_options()[0]
        self.target.set(label)
        return kind, body

    def _earliest(self):
        """The first moment this craft can leave: now, or for a next leg, when it arrived."""
        c = self.craft
        if c.leg_of is not None and self.original is not None:
            parent = self.original.flight(c.leg_of)
            return max(self.now, parent.idle_since or self.now)
        return self.now

    def _launch_widgets(self):
        for w in self.launch_box.winfo_children():
            w.destroy()
        box = self.launch_box
        kind, target = self._route()
        origin = self.craft.origin
        earliest = self._earliest()
        is_leg = self.craft.leg_of is not None
        today = date_str(earliest, False)
        # A date-only "today" is already in the past; a next leg can leave an hour after arriving
        first_date = date_str(earliest + 3600) if is_leg else date_str(self.now + DAY, False)
        fixed = self._fixed_launch()
        if is_leg and self.craft.leg_start is not None:
            start = self.craft.leg_start
            how = (f"takes off from the surface ({start.ascent_dv:.1f} km/s of its fuel)"
                   if start.where == "surface" else "leaves from its orbit")
            ttk.Label(box, foreground="#555", wraplength=500, justify="left", text=(
                f"Next leg of {self.craft.leg_of}: starts in a parking orbit around {origin} and "
                f"{how}, with {start.design.crew} crew, {start.food:,.0f} kg food, {start.water:,.0f} kg "
                f"water and its own hardware. Earliest departure {date_str(earliest)}.")).pack(anchor="w", pady=(0, 4))
        if fixed:
            ttk.Label(box, foreground="#555", wraplength=500, justify="left", text=(
                f"Launch date {date_str(fixed[1])}, set by {fixed[0]}: all craft in a mission launch "
                f"together. Clear {fixed[0]}'s route to choose another date.")).pack(anchor="w", pady=(0, 4))
        if kind == "planet":
            self.search_from = LabeledEntry(box, "Search from", today, width=14)
            self.search_span = LabeledEntry(box, "Search span", 800, unit="days")
            if not fixed:
                self.search_from.pack(anchor="w")
                self.search_span.pack(anchor="w")
            self.arrive_alt = LabeledEntry(box, "Pass target at", 300, unit="km altitude")
            self.arrive_alt.pack(anchor="w")
            self._land_checkbox(box, "Land on arrival" + (" (home: crew leave the game)" if target == "Earth" else
                                                          " (instead of passing by)"))
            via_row = ttk.Frame(box)
            via_row.pack(anchor="w")
            ttk.Label(via_row, text="Via (gravity assist)", width=16).pack(side="left")
            self.via = tk.StringVar(value="None")
            via_ok = not fixed
            via_box = ttk.Combobox(via_row, textvariable=self.via, state="readonly" if via_ok else "disabled",
                                   width=12, values=["None"] + [p for p in PLANETS if p not in (target, origin)])
            via_box.pack(side="left")
            via_box.bind("<<ComboboxSelected>>", lambda e: self._via_changed())
            if not via_ok:
                ttk.Label(via_row, text="(use Gravity assist… on the burns)", foreground="#777").pack(side="left", padx=4)
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
            self.launch_date = LabeledEntry(box, "Leave at" if is_leg else "Launch date",
                                            date_str(fixed[1]) if fixed else first_date, width=18)
            self.launch_date.pack(anchor="w")
            if fixed:
                self.launch_date.lock()
            if kind == "moon":
                self.moon_tof = LabeledEntry(box, "Flight time", 4, unit="days")
                self.moon_tof.pack(anchor="w")
                self.arrive_alt = LabeledEntry(box, "Pass Moon at", 100, unit="km altitude")
                self.arrive_alt.pack(anchor="w")
                self._land_checkbox(box)
                text = "Plan transfer"
            elif kind == "parent":
                self.moon_tof = LabeledEntry(box, "Flight time", 4, unit="days")
                self.moon_tof.pack(anchor="w")
                text = f"Plan return to {target}"
                ttk.Label(box, foreground="#555", wraplength=420, text=(
                    f"Heads back down and lands on {target}." + (" Landing on Earth brings the crew home: "
                    "they leave the game." if target == "Earth" else ""))).pack(anchor="w")
            else:
                self.orbit_alt = LabeledEntry(box, "Orbit altitude", f"{park_alt(target):.0f}", unit="km")
                self.orbit_alt.pack(anchor="w")
                text = "Set launch" if not is_leg else f"Stay in {target} orbit"
            self.set_launch_btn = ttk.Button(box, text=text, command=self._set_simple_launch)
            self.set_launch_btn.pack(anchor="w", pady=6)
            if not is_leg:
                ttk.Label(box, wraplength=420, foreground="#555", text=(
                    "Every mission starts in a circular parking orbit around Earth at launch "
                    "(getting there is the launch vehicle's job). The Moon can be reached almost "
                    "any day, so there's no launch window to search for.")).pack(anchor="w")
        live = self._live_flight()
        if live is not None and live.status != PENDING:
            ttk.Label(box, text=f"Launched {date_str(live.launch_et)}").pack(anchor="w", pady=4)

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
            ttk.Button(row, text="Land…", command=self._land),
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
            self.declare_btn = ttk.Button(row, text="End this craft's flight here", command=self._declare)
            self.declare_btn.pack(side="left")
            self.deploy_btn = ttk.Button(row, text="Deploy satellite", command=self._deploy)   # science
            self.deploy_btn.pack(side="left", padx=4)

    # ----- state helpers -----
    @property
    def launched(self):
        return self.original is not None and self.original.status != PENDING

    @property
    def craft(self):
        """The CraftDraft being edited."""
        return self.crafts[self.cur]

    @property
    def draft(self):
        """The flight of the craft being edited (None until it has a route)."""
        return self.craft.flight

    def _fixed_launch(self):
        """(craft name, launch time) if another craft already set the launch date, else None.
        Only the first legs share a date; next legs leave whenever their craft is ready."""
        if self.craft.leg_of is not None:
            return None
        for k, c in enumerate(self.crafts):
            if k != self.cur and c.leg_of is None and c.flight is not None:
                return c.name, c.flight.launch_et
        return None

    def _route_locked(self):
        """The current craft has already set off (its route can't change any more)."""
        live = self._live_flight()
        return live is not None and live.status != PENDING

    def _builder_locked(self):
        """Hardware is fixed once launched, and a next leg flies the craft it already is."""
        return self.craft.leg_of is not None or self._route_locked()

    def _apply_locks(self):
        build = self._builder_locked()
        route = self._route_locked()
        for w in (self.launcher_box, self.stage_box):
            w.config(state="disabled" if build else "readonly")
        self.target_box.config(state="disabled" if route else "readonly")
        for w in (self.fuel_entry, self.crew_spin, self.fill_btn, *self.module_buttons):
            w.config(state="disabled" if build else "normal")
        self.add_craft_btn.config(state="disabled" if self.launched else "normal")
        removable = not route and (not self.launched or self.craft.leg_of is not None)
        self.remove_craft_btn.config(state="normal" if removable else "disabled")
        self.clear_route_btn.config(state="disabled" if route else "normal")
        self.next_leg_btn.config(state="normal" if self._can_fly_on() else "disabled")
        if route:
            for w in self.launch_box.winfo_children():
                if isinstance(w, (ttk.Button, ttk.Entry)):
                    w.config(state="disabled")
                for child in w.winfo_children():
                    if isinstance(child, (ttk.Entry, ttk.Button)):
                        child.config(state="disabled")
            if hasattr(self, "chart"):
                self.chart.unbind("<Button-1>")

    def _can_fly_on(self):
        """The current craft has arrived (idle) and has no next leg yet."""
        live = self._live_flight()
        if live is None or live.status != IDLE:
            return False
        return not any(c.leg_of == live.name for c in self.crafts)

    @_guard
    def _plan_next_leg(self):
        live = self._live_flight()
        if live is None or not self._can_fly_on():
            raise PlanError("Only a craft that has arrived somewhere (idle) can fly on")
        start = next_leg_start(self.control, self.original, live, self.now)
        start.design = self.page.science.without_deployed(live, start.design)   # satellites left behind
        draft = CraftDraft(next_leg_name(self.original, live), start.design, origin=start.body,
                           leg_of=live.name, leg_start=start)
        draft.target = leg_targets(start.body)[0][0]
        self.crafts.append(draft)
        self._select_craft(len(self.crafts) - 1)
        self._info(f"Planning {draft.name}: pick where {live.name} goes next on the Route tab")

    def _craft(self):
        """The physics craft for the current design."""
        return self.craft.design.physics_craft(self.catalogue)

    def _craft_name(self):
        return self.craft.name

    def _target_body(self):
        return self._route()[1]

    def _error(self, exc):
        self.status.config(text=str(exc))

    def _info(self, text):
        self.status.config(text=text, foreground="#555")

    # ----- crafts -----
    def _select_craft(self, index):
        self.cur = max(0, min(index, len(self.crafts) - 1))
        c = self.craft
        self.craft_box.config(values=[x.name for x in self.crafts])
        self.craft_box.current(self.cur)
        self.craft_name.set(c.name)
        options = self._route_options()
        labels = [o[0] for o in options]
        if c.target not in labels:
            # Stored as a body name: pick the option that goes there (preferring a transfer to staying)
            match = [o for o in options if o[2] == c.target]
            c.target = (sorted(match, key=lambda o: o[1] == "park")[0] if match else options[0])[0]
        self.target_box.config(values=labels)
        self.target.set(c.target)
        if c.leg_of is not None and c.leg_start is None and self.original is not None:
            parent = self.original.flight(c.leg_of)
            if parent.status == IDLE:
                c.leg_start = next_leg_start(self.control, self.original, parent,
                                             c.flight.launch_et if c.flight else self.now)
        self._load_design(c.design)
        self.pc = self.assist_grid = None
        self._launch_widgets()
        self._apply_locks()
        if self.draft:
            self.horizon.set(f"{(self.draft.horizon_et - self.draft.launch_et) / DAY:.0f}")
        self._refresh()

    def _add_craft(self):
        names = {c.name for c in self.crafts}
        k = len(self.crafts) + 1
        while f"Craft {k}" in names:
            k += 1
        self.crafts.append(CraftDraft(f"Craft {k}", starter_design()))
        self._select_craft(len(self.crafts) - 1)
        self._info("New craft added: build it, then give it a route. It launches with the others.")

    @_guard
    def _remove_craft(self):
        if len(self.crafts) == 1:
            raise PlanError("A mission needs at least one craft")
        if not messagebox.askyesno("Remove craft", f"Remove {self.craft.name}?", parent=self):
            return
        self.crafts.pop(self.cur)
        self._select_craft(self.cur - 1)

    @_guard
    def _rename_craft(self):
        name = self.craft_name.get().strip()
        if not name or name == self.craft.name:
            return
        if self._route_locked() or any(c.leg_of == self.craft.name for c in self.crafts):
            self.craft_name.set(self.craft.name)
            raise PlanError("A craft can't be renamed once it has launched or has a next leg")
        if any(c.name == name for c in self.crafts):
            self.craft_name.set(self.craft.name)
            raise PlanError(f"There is already a craft called {name!r}")
        self.craft.name = name
        if self.craft.flight is not None:
            self.craft.flight.name = name
        self.craft_box.config(values=[x.name for x in self.crafts])
        self.craft_box.current(self.cur)
        self._refresh()

    # ----- craft design -----
    def _load_design(self, d):
        """Show a design in the builder (without triggering change handlers)."""
        self._loading_design = True
        self.launcher_box.current(self.launcher_ids.index(d.launcher))
        self.stage_box.current(self.stage_ids.index(d.stage))
        self.fuel_var.set(f"{d.fuel_kg:.0f}")
        self.crew_var.set(str(d.crew))
        self._loading_design = False
        self._fill_module_tree()

    def _read_design(self):
        """The design as the builder shows it now."""
        try:
            fuel = float(self.fuel_var.get())
            crew = int(self.crew_var.get())
        except ValueError:
            raise PlanError("Fuel and crew must be numbers")
        return replace(self.craft.design, launcher=self.launcher_ids[self.launcher_box.current()],
                       stage=self.stage_ids[self.stage_box.current()], fuel_kg=fuel, crew=crew)

    def _design_changed(self):
        if self._loading_design or self._builder_locked():
            return
        try:
            d = self._read_design()
        except PlanError as exc:
            self._error(exc)
            return
        locked = self.page.science.newly_locked(self.craft.design, d)     # tech tree
        if locked:
            self._error(f"Locked: {locked[0]}. Unlock it in the Tech tree")
            self._load_design(self.craft.design)
            return
        self._apply_design(d)

    def _apply_design(self, d):
        self.craft.design = d
        cap = self.catalogue.stages[d.stage].fuel_capacity_kg
        self.fuel_max.config(text=f"of {cap:,.0f} kg")
        if self.draft is not None:
            self.draft.set_design(d)
        self._refresh()

    def _fill_tank(self):
        stage = self.catalogue.stages[self.stage_ids[self.stage_box.current()]]
        self.fuel_var.set(f"{stage.fuel_capacity_kg:.0f}")

    def _fill_module_tree(self):
        sel = self.module_tree.selection()
        self.module_tree.delete(*self.module_tree.get_children())
        d = self.craft.design
        for mid in self.module_ids:
            m = self.catalogue.modules[mid]
            n = d.modules.get(mid, 0)
            power = f"{m.power_kw:+.1f} kW" if m.power_kw else ""
            lock = "" if self.page.science.is_unlocked(mid) else "  (locked)"
            self.module_tree.insert("", "end", iid=mid, values=(
                m.name + (f"  (tier {m.tier})" if m.tier else "") + lock, n if n else "",
                f"{m.mass_kg / 1000:.1f} t", power, f"${m.cost_musd:,.0f}M"),
                tags=("onboard",) if n else ())
        self.module_tree.tag_configure("onboard", background="#dff3e4")
        self._mark_locked_stages()
        self.fuel_max.config(text=f"of {self.catalogue.stages[d.stage].fuel_capacity_kg:,.0f} kg")
        if sel and self.module_tree.exists(sel[0]):
            self.module_tree.selection_set(sel)

    def _show_module_info(self):
        sel = self.module_tree.selection()
        if sel:
            m = self.catalogue.modules[sel[0]]
            self.part_info.config(text=f"{m.name}: {m.description}")

    @_guard
    def _change_module(self, delta):
        if self._builder_locked():
            raise PlanError("The craft's hardware can't change after launch (a next leg flies what the craft already is)")
        sel = self.module_tree.selection()
        if not sel:
            raise PlanError("Select a module first")
        d = self.craft.design
        modules = dict(d.modules)
        modules[sel[0]] = max(0, modules.get(sel[0], 0) + delta)
        new = replace(d, modules={k: v for k, v in modules.items() if v})
        locked = self.page.science.newly_locked(d, new)                   # tech tree
        if locked:
            raise PlanError(f"Locked: {locked[0]}. Unlock it in the Tech tree")
        self._apply_design(new)
        self._fill_module_tree()

    # ----- target and route -----
    def _target_changed(self):
        self.craft.target = self.target.get()
        self.craft.flight = None
        self.pc = self.assist_grid = None
        self._launch_widgets()
        self._refresh()

    def _clear_route(self):
        if self.craft.flight is None:
            return
        self.craft.flight = None
        self._launch_widgets()
        self._refresh()
        self._info(f"{self.craft.name}'s route was cleared")

    # ----- launch: planets -----
    @_guard
    def _tof_range(self, origin, target):
        """Flight times (days) to search between two bodies orbiting the same parent."""
        if origin == "Earth" and target in TOF_RANGE:
            return TOF_RANGE[target]
        # Around half an orbit of the transfer ellipse (a Hohmann transfer), give or take
        a = (BODIES[origin].a + BODIES[target].a) / 2
        hohmann = math.pi * math.sqrt(a ** 3 / BODIES[BODIES[origin].parent].gm) / DAY
        return 0.4 * hohmann, 1.6 * hohmann

    def _find_windows(self):
        target = self._target_body()
        origin = self.craft.origin
        fixed = self._fixed_launch()
        if fixed:
            # Launch date set by another craft: only the flight time is free
            self.assist_grid = None
            lo, hi = self._tof_range(origin, target)
            self.pc = porkchop(origin, target, np.array([fixed[1]]), np.linspace(lo, hi, 120) * DAY)
            self.chart.show(self.pc, self._craft().dv_remaining, x_label=f"Launch date fixed by {fixed[0]}")
            self._pick_best("fuel")
            return
        start = max(parse_date(self.search_from.get()), self._earliest())
        span = parse_float(self.search_span.get(), "Search span")
        if not 10 <= span <= 5000:
            raise PlanError("Search span must be between 10 and 5000 days")
        via = self.via.get()
        if via != "None":
            self._find_assist_windows(target, via, start, span)
            return
        self.assist_grid = None
        lo, hi = self._tof_range(origin, target)
        dep = start + np.linspace(0, span, 250) * DAY
        dep = dep[dep < ephemeris.COVER_MAX - hi * DAY]
        tof = np.linspace(lo, hi, 90) * DAY
        self.pc = porkchop(origin, target, dep, tof)
        self.chart.show(self.pc, self._craft().dv_remaining)
        self._pick_best("fuel")

    def _find_assist_windows(self, target, via, start, span):
        """Search origin -> via -> target routes; chart = best route per (flyby date, leg-1 time)."""
        origin = self.craft.origin
        lo1, hi1 = self._tof_range(origin, via)
        lo2, hi2 = ASSIST_LEG2_RANGE
        flyby = start + lo1 * DAY + np.arange(0, span, max(2.0, span / 200)) * DAY
        flyby = flyby[flyby < ephemeris.COVER_MAX - hi2 * DAY]
        grid = assist_search(origin, via, target, flyby, np.linspace(lo1, hi1, 40) * DAY,
                             np.linspace(lo2, hi2, 40) * DAY)
        # The grid runs over flyby dates; rule out routes that would have to leave before `start`
        too_early = grid.flyby_et[:, None] - grid.tof1[None, :] < start
        grid.dv_depart = np.where(too_early, np.inf, grid.dv_depart)
        self.assist_grid = grid
        total = grid.dv_total
        # The chart shows, for each (flyby date, leg-1 time), the cheapest leg-2 time
        self.pc = SimpleNamespace(dv_total=total.min(axis=2), depart_et=grid.flyby_et, tof=grid.tof1,
                                  best_k=total.argmin(axis=2))
        self.chart.show(self.pc, self._craft().dv_remaining,
                        x_label=f"{via} flyby date (y: days from {origin} to {via})")
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
        craft = self._craft()
        origin = g.origin
        if g.dv_depart[f, m] > craft.dv_remaining:
            # Still apply the pick (so the plan matches the chart), but say why it can't work
            draft = Flight.to_planet(self._craft_name(), target, craft, dep, fb, g.v_inf_depart[f, m], origin=origin)
            self._set_draft(draft)
            self._error(self._cant_leave(g.dv_depart[f, m], craft, origin))
            return
        try:
            draft = Flight.via_assist(self._craft_name(), target, craft, dep, fb, arr, g.v_inf_depart[f, m], via,
                                      alt, origin=origin)
        except PlanningError as exc:
            draft = Flight.to_planet(self._craft_name(), target, craft, dep, fb, g.v_inf_depart[f, m], origin=origin)
            self._set_draft(draft)
            self._error(f"Departure planned, but the {via} flyby couldn't be set up: {exc}")
            return
        note = (f"Gravity assist planned: {via} bends the path {draft.assist.turn_deg:.0f}°. "
                f"Burns: departure, flyby set-up, a touch-up after {via}. Add a capture burn to stay at {target}.")
        out_et = planning.soi_exit_time(draft.trajectory, via)
        if self.land_var.get() and out_et is not None:
            try:
                note = f"{via} bends the path {draft.assist.turn_deg:.0f}°; " + draft.add_landing(out_et + DAY, target, arr)
            except PlanningError as exc:
                note += f" (couldn't plan the landing: {exc})"
        self._set_draft(draft)
        self._info(note)

    @staticmethod
    def _cant_leave(dv_depart, craft, origin="Earth"):
        return (f"Not enough fuel for this transfer: leaving {origin} needs {dv_depart:.2f} km/s, the craft "
                f"has {craft.dv_remaining:.2f} km/s in total. Pick a greener cell, or a bigger stage or "
                f"more fuel (Build craft tab).")

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
        target, origin = self._target_body(), self.craft.origin
        draft = Flight.to_planet(self._craft_name(), target, craft, dep, dep + tof, pc.v_inf_depart[i, j],
                                 origin=origin)
        # Always apply the pick, so the plan matches the chart; explain anything that can't be afforded
        if pc.dv_depart[i, j] > craft.dv_remaining:
            self._set_draft(draft)
            self._error(self._cant_leave(pc.dv_depart[i, j], craft, origin))
            return
        # Patched conics leave a small miss from the origin's SOI; aim properly a day after leaving it
        alt = parse_float(self.arrive_alt.get(), "Pass altitude")
        exit_et = planning.soi_exit_time(draft.trajectory, origin)
        note, problem = "", None
        try:
            if self.land_var.get() and exit_et is not None:
                note = draft.add_landing(exit_et + DAY, target, dep + tof)
            elif exit_et is not None and exit_et + DAY < dep + tof:
                before = fly(draft.start, craft, draft.burns, exit_et + DAY)
                aim, got = planning.aim_burn(before.end, before.craft, target, dep + tof, alt)
                draft.set_plan(draft.burns + [aim])
                note = (f"Added a correction burn a day after leaving {origin}'s SOI: passes {target} at "
                        f"{got:.0f} km. Add a capture burn to stay in orbit.")
        except PlanningError as exc:
            problem = f"Departure planned, but no correction burn after it: {exc}"
        self._set_draft(draft)
        self._error(problem) if problem else self._info(note)

    # ----- launch: Moon / Earth orbit -----
    @_guard
    def _set_simple_launch(self):
        fixed = self._fixed_launch()
        et = fixed[1] if fixed else parse_date(self.launch_date.get())
        if et < self._earliest():
            raise PlanError("Can't leave in the past" if self.craft.leg_of is None else
                            f"Can't leave before {date_str(self._earliest())}")
        craft = self._craft()
        name = self._craft_name()
        kind, target = self._route()
        if kind == "moon":
            tof = parse_float(self.moon_tof.get(), "Flight time")
            if not 2 <= tof <= 10:
                raise PlanError("Flight time to the Moon should be 2-10 days")
            draft = Flight.to_moon(name, craft, et, tof * DAY,
                                   parse_float(self.arrive_alt.get(), "Pass altitude"))
            if self.land_var.get():
                self._info(draft.add_landing(et, "Moon", et + tof * DAY))
        elif kind == "parent":
            tof = parse_float(self.moon_tof.get(), "Flight time")
            if not 2 <= tof <= 10:
                raise PlanError("Flight time should be 2-10 days")
            draft = Flight.to_parent(name, craft, self.craft.origin, et, tof * DAY)
        else:
            alt = parse_float(self.orbit_alt.get(), "Orbit altitude")
            if alt < 50:
                raise PlanError("Orbit altitude must be at least 50 km")
            draft = Flight.park(name, craft, target, et, alt)
        self._set_draft(draft)

    def _set_draft(self, flight):
        """Give the current craft this flight (built from its design)."""
        flight.name = self.craft.name
        if self.craft.leg_of is not None:
            # What a next leg carries depends on when it leaves: work it out for this departure
            parent = self.original.flight(self.craft.leg_of)
            start = next_leg_start(self.control, self.original, parent, flight.launch_et)
            self.craft.leg_start = start
            self.craft.design = start.design
            flight.set_design(start.design)
            apply_leg_start(flight, start, self.craft.leg_of)
        else:
            flight.design = self.craft.design
        self.craft.flight = flight
        self.horizon.set(f"{(flight.horizon_et - flight.launch_et) / DAY:.0f}")
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
        others = [b for b in ["Moon"] + PLANETS if b != target]
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
        planets = list(PLANETS)
        if via is None:
            # Default to the first planet the plan already passes, if any
            passes = [a for a in planning.closest_approaches(d.trajectory) if a.body in planets]
            via, flyby = (passes[0].body, passes[0].et) if passes else ("Venus", None)
        burn_default = self._default_aim_time(flyby) if flyby else self._default_aim_time(math.inf)
        if flyby is None:
            flyby = burn_default + 120 * DAY
        targets = [self._target_body()] + [b for b in planets if b != self._target_body()]
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

    def _land_checkbox(self, box, text="Land on arrival (instead of passing by)"):
        self.land_var = tk.BooleanVar(value=False)
        if self._target_body() in GAS_GIANTS:
            ttk.Label(box, foreground="#777", text=f"{self._target_body()} is a gas giant: no surface to land on, "
                                                    "but you can orbit it").pack(anchor="w")
            return
        self.land_check = ttk.Checkbutton(box, text=text, variable=self.land_var)
        self.land_check.pack(anchor="w")

    @_guard
    def _land(self):
        """Dialog: touch down on a body, from orbit around it or by aiming the arrival at it."""
        self._need_draft()
        d = self.draft
        body = self._target_body()
        traj = d.trajectory
        orbiting = traj.segments[-1].body == body and traj.end_reason != "impact"
        if orbiting:
            burn_et, arrive = self._default_aim_time(math.inf), None
        else:
            visit = planning.first_visit(traj, body)
            passes = [a for a in planning.closest_approaches(traj) if a.body == body]
            arrive = visit.start.et if visit else (passes[0].et if passes else self._default_aim_time(math.inf) + 200 * DAY)
            exits = [e.et for e in traj.events if e.kind == "soi_exit" and e.et < arrive]
            burn_et = (max(exits) + DAY) if exits else d.launch_et
            burn_et = max(burn_et, self.now + 60) if self.launched else burn_et
        res = FormDialog(self, "Land", [
            ("body", "Land on", self._body_choices(), ""),
            ("t", "Burn at", f"{(burn_et - d.launch_et) / DAY:.2f}", "days after launch"),
            ("arrive", "Touch down around", "" if arrive is None else f"{(arrive - d.launch_et) / DAY:.2f}",
             "days after launch (not needed when already orbiting it)"),
        ], note="Orbiting the body at the burn time: a retrograde de-orbit burn. Orbiting its parent "
                "(the Sun for a planet, Earth for the Moon): the arrival is aimed at the surface; from "
                "a short parking orbit the craft waits for the cheapest point of its lap. Later burns "
                "are replaced. Landing is free in this game, taking off again isn't.").ask()
        if res is None:
            return
        if res["body"] in GAS_GIANTS:
            raise PlanError(f"{res['body']} is a gas giant: there's no surface to land on (you can orbit it)")
        t = d.launch_et + parse_float(res["t"], "Burn time") * DAY
        self._check_not_past(t)
        arrive_text = res["arrive"].strip()
        arrive_et = d.launch_et + parse_float(arrive_text, "Touchdown time") * DAY if arrive_text else None
        note = d.add_landing(t, res["body"], arrive_et)
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
        self._refresh_deploy()
        self.burn_tree.delete(*self.burn_tree.get_children())
        self.event_tree.delete(*self.event_tree.get_children())
        self._refresh_mission_info()
        self._refresh_health()
        self._refresh_previews()
        d = self.draft
        if d is None:
            self.outcome.config(text=f"{self.craft.name}: no route yet (Route tab)", foreground="#b70")
            self.fuel_left.config(text="")
            self.system_map.clear()
            if hasattr(self, "declare_btn"):
                self.declare_btn.config(state="disabled")
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
        text = f"{self.craft.name}: {out.describe()}"
        if out.ends_mission or out.kind == "flyby":
            text += f" on {date_str(out.et, False)}"
        if out.kind == "flyby":
            text += "\n" + self._capture_hint(d, out.body)
        self.outcome.config(text=text, foreground="#2a7" if out.ends_mission else "#b70")
        c = traj.craft
        fuel = max(0.0, c.fuel)        # rounding can leave a hair below zero after an out-of-fuel burn
        self.fuel_left.config(text=f"Fuel left at the end: {fuel:,.0f} kg ({max(0.0, c.dv_remaining):.2f} km/s of Δv)")
        if hasattr(self, "declare_btn"):
            live = self._live_flight()
            self.declare_btn.config(state="normal" if live and live.can_declare_complete(self.now) else "disabled")
        self.system_map.set_trajectory(traj, [b.et for b in d.burns])

    def _refresh_mission_info(self):
        planned = [c for c in self.crafts if c.flight is not None]
        cost = sum(c.design.cost(self.catalogue) for c in self.crafts)
        when = f"launches {date_str(planned[0].flight.launch_et)}" if planned else "no launch date yet"
        self.mission_info.config(text=(
            f"{len(self.crafts)} craft, {when}; {len(planned)} of {len(self.crafts)} have a route.  "
            f"Total cost ${cost:,.0f}M"))

    def _refresh_health(self):
        for (dot, name, text), ind in zip(self.health_rows,
                                         self._assess() + [None] * 7):
            if ind is None:
                dot.config(text=""), name.config(text=""), text.config(text="")
                continue
            dot.config(text="●", fg=LEVEL_COLOURS[ind.level])
            name.config(text=ind.name)
            text.config(text=ind.text)

    def _assess(self):
        """Health indicators, using what a next leg actually carries (not full supply pods)."""
        c, d = self.craft, self.draft
        food = water = None
        if d is not None and d.food0 is not None:
            food, water = d.food0, d.water0
        elif c.leg_start is not None:
            food, water = c.leg_start.food, c.leg_start.water
        return assess(c.design, d, self.catalogue, food=food, water=water)

    def _refresh_previews(self):
        """Every craft's planned path in the 3D view; the one being edited drawn thicker."""
        keys = set()
        for k, c in enumerate(self.crafts):
            if c.flight is None:
                continue
            key = f"{PREVIEW_KEY}/{k}"
            keys.add(key)
            self.view.show_path(key, f"{c.name} (plan)", c.flight.trajectory, self.page.PREVIEW_COLOR,
                                thickness=3 if k == self.cur else 1.5)
        for key in self._preview_keys - keys:
            self.view.remove_path(key)
        self._preview_keys = keys

    def _live_flight(self):
        """The real (saved) flight of the craft being edited, for active missions."""
        if self.original is None:
            return None
        return next((f for f in self.original.flights if f.name == self.craft.name), None)

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

    # ----- save / close -----
    @_guard
    def _save(self):
        name = self.name.get().strip()
        if not name:
            raise PlanError("Give the mission a name")
        if name != (self.original.name if self.original else None) and \
                any(m.name == name for m in self.control.missions):
            raise PlanError(f"There is already a mission called {name!r}")
        for c in self.crafts:
            if c.flight is None:
                raise PlanError(f"{c.name} has no route yet (Route tab)")
            problems = c.design.problems(self.catalogue)
            if problems and not self.launched:
                raise PlanError(f"{c.name}: {problems[0]}")
            locked = self.page.science.locked_parts(c.design)            # tech tree
            if locked and not self.launched:
                raise PlanError(f"{c.name}: {locked[0]} is locked. Unlock it in the Tech tree")
        if not self._confirm_endings():
            return
        flights = [c.flight for c in self.crafts]
        if self.original is None:
            if flights[0].launch_et < self.now:
                raise PlanError("Launch can't be in the past")
            mission = Mission(name, flights)
            self.control.add(mission)
            self.page.mission_changed(mission)
        elif self.original.status == PENDING:
            old_names = [f.name for f in self.original.flights]
            self.original.set_flights(flights)
            old_name = self.original.name
            self.original.name = name
            self.page.mission_changed(self.original, old_name, old_flights=old_names)
        else:
            live_names = {f.name for f in self.original.flights}
            for c in self.crafts:
                if c.name not in live_names:
                    self.original.add_leg(c.flight)                  # a new next leg
                    continue
                live = self.original.flight(c.name)
                if live.status == PENDING:
                    # A next leg that hasn't left yet: replace it wholesale
                    self.original.flights[self.original.flights.index(live)] = c.flight
                elif live.status == ACTIVE:
                    live.set_plan(c.flight.burns, c.flight.horizon_et, now=self.now)
                elif [b.et for b in live.burns] != [b.et for b in c.flight.burns]:
                    raise PlanError(f"{c.name} has already arrived: use Plan next leg to fly it on")
            # Next legs removed in the planner (only ones that haven't left)
            keep = {c.name for c in self.crafts}
            self.original.flights = [f for f in self.original.flights if f.name in keep or f.status != PENDING]
            old_name = self.original.name
            self.original.name = name
            self.page.mission_changed(self.original, old_name)
        self.close()

    def _confirm_endings(self):
        """Warn if any craft still to fly doesn't end landed, in a stable orbit or back on Earth.

        Such a craft keeps drifting (a flyby, or still travelling when the projection ends):
        it never becomes idle, so it can't fly on, and its crew eat into their supplies until
        they run out. Returns True to go ahead and save.
        """
        loose = []
        for c in self.crafts:
            live = self._live_flight_named(c.name)
            if live is not None and live.status not in (PENDING, ACTIVE):
                continue                                  # already arrived or over: nothing to check
            out = c.flight.outcome
            if not out.ends_mission:
                text = out.describe()
                loose.append(f"• {c.name}: {text[0].lower()}{text[1:]}")
        if not loose:
            return True
        return messagebox.askyesno(
            "Craft won't arrive anywhere",
            "These craft don't end their plan landed or in a stable orbit:\n\n" + "\n".join(loose) +
            "\n\nThey'll keep drifting: they won't arrive (go idle), so they can't fly on later, and "
            "their crew use up their supplies until they run out.\n\nTip: add a Capture into orbit… or "
            "Land… burn, or extend \"Show path until\" if the plan arrives later.\n\nSave anyway?",
            icon="warning", parent=self)

    def _live_flight_named(self, name):
        if self.original is None:
            return None
        return next((f for f in self.original.flights if f.name == name), None)

    @_guard
    def _cancel_launch(self):
        if messagebox.askyesno("Cancel launch", f"Cancel {self.original.name}?", parent=self):
            self.control.remove(self.original)
            self.page.mission_changed(self.original, removed=True)
            self.close()

    @_guard
    def _declare(self):
        live = self._live_flight()
        if live is None:
            raise PlanError("Save the mission first")
        live.declare_complete(self.now)
        self.page.mission_changed(self.original)
        self.close()

    # ----- science (game/science.py) -----
    def tech_changed(self):
        """Called by MainPage after an unlock: refresh the (locked) labels."""
        self._fill_module_tree()
        self._refresh_deploy()

    def _mark_locked_stages(self):
        sci, cat = self.page.science, self.catalogue
        values = [f"{s.name} — Isp {s.isp_s:,.0f} s" + (", can relaunch" if s.relaunch else "")
                  + (f"  (tier {s.tier}, secret)" if s.secret else f"  (tier {s.tier})" if s.tier else "")
                  + ("" if sci.is_unlocked(sid) else "  (locked)") for sid, s in cat.stages.items()]
        if list(self.stage_box.cget("values")) != values:
            k = self.stage_box.current()
            self.stage_box.config(values=values)
            if k >= 0:
                self.stage_box.current(k)

    def _refresh_deploy(self):
        if not hasattr(self, "deploy_btn"):
            return
        live = self._live_flight()
        sci = self.page.science
        n = sci.satellites_on_board(live) if live is not None else 0
        ok = live is not None and sci.deploy_problem(live) is None
        self.deploy_btn.config(text=f"Deploy satellite ({n} on board)" if n > 0 else "Deploy satellite",
                               state="normal" if ok else "disabled")

    @_guard
    def _deploy(self):
        live = self._live_flight()
        if live is None:
            raise PlanError("This craft hasn't launched yet")
        self.page.science.deploy(live, self.now)
        self._info(f"{live.name} left a satellite in orbit around {live.location()[0]}")
        self._refresh_deploy()
        self.page.tech_changed()

    def close(self):
        for key in self._preview_keys:
            self.view.remove_path(key)
        self.view.paused = self._was_paused
        self.page.planner_closed(self)
        self.destroy()
