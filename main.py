"""
Tkinter front end: a start page, then a main page with the 3D solar system view
in the middle, info tables and the missions list on either side and control
buttons underneath. Missions are planned in planner.py's window.

Setup:
    pip install ursina spiceypy numpy
    Put naif0012.tls and de440.bsp (or the smaller de440s.bsp) in the same folder as this script.

Mouse controls inside the 3D view:
    Right-drag      orbit camera      Scroll       zoom
    Middle-drag     pan (free camera only)
"""
import tkinter as tk
from tkinter import messagebox, ttk

from game import catalogue
from game.base import sites_at
from game.mission import ACTIVE, COMPLETE, IDLE, LOST, PENDING, MissionControl
from physics import ephemeris

FRAME_MS = 1          # delay between rendered frames (vsync sets the real rate)
TABLE_MS = 250        # how often the side tables refresh
MIN_SPEED = 0.5       # days/s; the Slower button is disabled below this
BODIES = ["Sun", "Mercury", "Venus", "Earth", "Mars", "Moon"]


class StartPage(ttk.Frame):
    def __init__(self, parent, on_start):
        super().__init__(parent, padding=40)
        ttk.Label(self, text="Inner Solar System", font=("TkDefaultFont", 28, "bold")).pack(pady=(80, 10))
        ttk.Label(self, text="Real planet positions from JPL's DE440 ephemeris").pack(pady=(0, 40))
        ttk.Button(self, text="Start", command=on_start, width=20).pack(ipady=8)


class InfoTable(ttk.LabelFrame):
    """A titled Treeview with one row per body."""

    def __init__(self, parent, title, columns):
        super().__init__(parent, text=title, padding=6)
        self.tree = ttk.Treeview(self, columns=[c for c, _ in columns], show="headings",
                                 height=len(BODIES), selectmode="none")
        for col, width in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=width, anchor="e" if width < 100 else "w", stretch=True)
        self.tree.pack(fill="both", expand=True)

    def set_rows(self, rows):
        self.tree.delete(*self.tree.get_children())
        for row in rows:
            self.tree.insert("", "end", values=row)


def short_date(et):
    return ephemeris.et2str(et)[:11]


class MissionsPanel(ttk.LabelFrame):
    """Pending launches, ongoing missions and previous missions."""

    TABS = [
        (PENDING, "Pending", [("Name", 110), ("Target", 70), ("Launch", 90)]),
        (ACTIVE, "Ongoing", [("Name", 110), ("Orbiting", 70), ("Plan ends with", 150)]),
        ("done", "Previous", [("Name", 110), ("Ended", 90), ("Result", 150)]),
        ("bases", "Bases", [("What", 120), ("Details", 230)]),
    ]

    def __init__(self, parent, page):
        super().__init__(parent, text="Missions", padding=6)
        self.page = page
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True)
        self.trees = {}
        for key, title, cols in self.TABS:
            tree = ttk.Treeview(self.notebook, columns=[c for c, _ in cols], show="tree headings",
                                height=8, selectmode="browse")
            tree.column("#0", width=18, stretch=False)          # expand/collapse arrows for craft rows
            for col, width in cols:
                tree.heading(col, text=col)
                tree.column(col, width=width, anchor="w")
            if key != "bases":
                tree.bind("<Double-1>", lambda e: self._open())
            self.notebook.add(tree, text=title)
            self.trees[key] = tree
        self.trees["bases"].tag_configure("danger", foreground="#c33")
        row = ttk.Frame(self)
        row.pack(fill="x", pady=(6, 0))
        ttk.Button(row, text="New mission…", command=lambda: page.open_planner()).pack(side="left")
        ttk.Button(row, text="Open…", command=self._open).pack(side="left", padx=4)
        ttk.Button(row, text="Follow craft", command=self._follow).pack(side="left")

    def _selected(self):
        """(mission, craft name or None) for the selected row. Craft rows are 'mission/craft'."""
        key = self.TABS[self.notebook.index("current")][0]
        if key == "bases":
            return None, None
        tree = self.trees[key]
        sel = tree.selection()
        if not sel:
            return None, None
        mission_name, _, craft = sel[0].partition("/")
        return self.page.control_by_name(mission_name), craft or None

    def _open(self):
        m, _ = self._selected()
        if m is None:
            messagebox.showinfo("Missions", "Select a mission first", parent=self)
        elif m.status in (PENDING, ACTIVE):
            self.page.open_planner(m)
        else:
            messagebox.showinfo(m.name, f"{m.end_note}\nEnded {short_date(m.ended_et)}", parent=self)

    def _follow(self):
        m, craft = self._selected()
        flights = [] if m is None else [f for f in m.flights if craft is None or f.name == craft]
        if not any(self.page.view.focus_craft(self.page.path_key(m, f)) for f in flights):
            messagebox.showinfo("Follow craft", "Select a mission (or craft) that is flying", parent=self)

    def refresh(self, control, now):
        def flight_row(m, f, key):
            if key == PENDING:
                return f.name, f.target, ""
            if key == ACTIVE:
                if f.status == IDLE:
                    return f.name, f.location()[0], f.end_note
                if f.status == PENDING:
                    return f.name, "", f"Next leg, leaves {short_date(f.launch_et)}"
                if f.status != ACTIVE:
                    return f.name, "", f.end_note
                return f.name, f.state_at(now).body, f.outcome.describe()
            return f.name, short_date(f.ended_et) if f.ended_et else "", f.end_note

        groups = {PENDING: control.by_status(PENDING), ACTIVE: control.by_status(ACTIVE),
                  "done": control.by_status(COMPLETE, LOST)}
        self._refresh_bases(sites_at(control, now), now)
        for key, tree in self.trees.items():
            if key == "bases":
                continue
            selected = tree.selection()
            open_rows = {iid for iid in tree.get_children() if tree.item(iid, "open")}
            tree.delete(*tree.get_children())
            for m in groups[key]:
                if key == PENDING:
                    values = (m.name, m.target, short_date(m.launch_et))
                elif key == ACTIVE:
                    values = (m.name, f"{len(m.flights)} craft", "")
                else:
                    values = (m.name, short_date(m.ended_et) if m.ended_et else "", "")
                tree.insert("", "end", iid=m.name, values=values, open=m.name in open_rows or key == ACTIVE)
                for f in m.flights:
                    tree.insert(m.name, "end", iid=f"{m.name}/{f.name}", values=flight_row(m, f, key))
            keep = [s for s in selected if tree.exists(s)]
            if keep:
                tree.selection_set(keep)


    def _refresh_bases(self, sites, now):
        tree = self.trees["bases"]
        opened = {iid for iid in self._all_rows(tree) if tree.item(iid, "open")}
        selected = tree.selection()
        tree.delete(*tree.get_children())
        cat = catalogue.default()
        if not sites:
            tree.insert("", "end", values=("No bases yet", "Land or put a craft in orbit to start one"))
        for s in sites:
            fd, wd = s.days_left()
            soonest = min(fd, wd)
            summary = f"{s.crew} crew"
            if s.crew:
                summary += f" · supplies {soonest:,.0f} d" if soonest != float("inf") else " · supplies never run out"
            summary += f" · {s.power_kw:+.1f} kW"
            danger = s.deaths > 0 or s.power_kw < 0 or (s.crew and soonest < 10)
            site = f"site:{s.name}"
            tree.insert("", "end", iid=site, values=(s.name, summary), open=site in opened,
                        tags=("danger",) if danger else ())
            crew = f"{s.crew} alive, {s.habitat_places} habitat place(s)" + (f", {s.deaths} died" if s.deaths else "")
            tree.insert(site, "end", values=("Crew", crew), tags=("danger",) if s.deaths else ())
            for label, kg, days in (("Food", s.food, fd), ("Water", s.water, wd)):
                text = f"{kg:,.0f} kg"
                if s.crew and days != float("inf"):
                    text += f" · {days:,.0f} days left, runs out {short_date(now + days * 86_400)}"
                elif s.crew:
                    text += " · not running down"
                tree.insert(site, "end", values=(label, text),
                            tags=("danger",) if s.crew and days < 10 else ())
            tree.insert(site, "end", values=("Power", f"{s.power_kw:+.1f} kW"),
                        tags=("danger",) if s.power_kw < 0 else ())
            stages = sum(s.stages.values())
            tree.insert(site, "end", values=("Fuel", f"{s.fuel:,.0f} kg in {stages} stage(s)"))
            mods = f"{site}/modules"
            tree.insert(site, "end", iid=mods, values=("Hardware", f"{sum(s.modules.values())} modules, "
                                                                   f"{stages} stage(s)"),
                        open=mods in opened)
            for mid, n in s.modules.items():
                tree.insert(mods, "end", values=(cat.modules[mid].name, f"x{n}"))
            for sid, n in s.stages.items():
                tree.insert(mods, "end", values=(cat.stages[sid].name, f"x{n} (landed)"))
            log = f"{site}/log"
            tree.insert(site, "end", iid=log, values=("Log", f"{len(s.log)} entries"), open=log in opened)
            for e in s.log:
                tree.insert(log, "end", values=(short_date(e.et), e.text), tags=("danger",) if e.deaths else ())
        keep = [x for x in selected if tree.exists(x)]
        if keep:
            tree.selection_set(keep)

    @staticmethod
    def _all_rows(tree, parent=""):
        for iid in tree.get_children(parent):
            yield iid
            yield from MissionsPanel._all_rows(tree, iid)


class MainPage(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent, padding=6)
        self.view = None
        self.control = MissionControl()
        self.planner = None
        self._drawn = {}           # path key -> (trajectory, status) currently drawn
        self._deaths_seen = {}     # site name -> deaths already reported in the news
        self._flight_status = {}   # path key -> status last seen (for arrival news)

        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        self.helio = InfoTable(self, "From the Sun",
                               [("Body", 70), ("Distance (AU)", 95), ("Speed (km/s)", 90)])
        self.helio.grid(row=0, column=0, sticky="ns", padx=(0, 6))

        # Panda3D renders straight into this frame's native window
        # Small requested size: the grid gives it all the space the side panels don't need
        self.view_frame = tk.Frame(self, bg="black", width=500, height=400)
        self.view_frame.grid(row=0, column=1, sticky="nsew")

        side = ttk.Frame(self)
        side.grid(row=0, column=2, sticky="ns", padx=(6, 0))
        self.geo = InfoTable(side, "From Earth",
                             [("Body", 70), ("Distance (km)", 110), ("Light time (min)", 100)])
        self.geo.pack(fill="x")
        self.missions = MissionsPanel(side, self)
        self.missions.pack(fill="both", expand=True, pady=(6, 0))

        self._build_controls().grid(row=1, column=0, columnspan=3, sticky="ew", pady=(6, 0))

    def _build_controls(self):
        bar = ttk.Frame(self)

        focus = ttk.LabelFrame(bar, text="Camera", padding=4)
        focus.pack(side="left")
        for name in BODIES:
            ttk.Button(focus, text=name, width=8,
                       command=lambda n=name: self.view and self.view.set_focus(n)).pack(side="left", padx=2)
        ttk.Button(focus, text="Free", width=8,
                   command=lambda: self.view and self.view.set_focus(None)).pack(side="left", padx=2)

        tm = ttk.LabelFrame(bar, text="Time", padding=4)
        tm.pack(side="right")
        # tk.Button supports auto-repeat while held, like holding the arrow keys did
        self.slower_btn = tk.Button(tm, text="Slower", width=7, repeatdelay=300, repeatinterval=50,
                                    command=lambda: self._change_speed("slower"))
        self.slower_btn.pack(side="left", padx=2)
        self.pause_btn = ttk.Button(tm, text="Pause", width=8, command=self._toggle_pause)
        self.pause_btn.pack(side="left", padx=2)
        tk.Button(tm, text="Faster", width=7, repeatdelay=300, repeatinterval=50,
                  command=lambda: self._change_speed("faster")).pack(side="left", padx=2)
        self.time_label = ttk.Label(tm, width=28, anchor="e")
        self.time_label.pack(side="left", padx=(8, 2))

        # Latest mission news (launches, landings...)
        self.news = ttk.Label(bar, foreground="#2a7")
        self.news.pack(side="left", padx=12)
        return bar

    def start_view(self):
        """Create the 3D view once this page is on screen (its window must exist)."""
        self.update()
        from solar_view import SolarSystemView   # heavy import; only needed from here on
        from ursina import color
        self.PREVIEW_COLOR = color.cyan
        self.PATH_COLORS = {PENDING: color.violet, ACTIVE: color.lime, IDLE: color.orange}
        self.view = SolarSystemView(self.view_frame.winfo_id(),
                                    self.view_frame.winfo_width(), self.view_frame.winfo_height())
        self.view_frame.bind("<Configure>", lambda e: self.view.resize(e.width, e.height))
        self._tick()
        self._refresh_tables()

    def _change_speed(self, which):
        if self.view:
            getattr(self.view, which)()
            # Below MIN_SPEED the clock barely moves, so stop slowing down there
            self.slower_btn.config(state="disabled" if self.view.days_per_second < MIN_SPEED else "normal")

    def _toggle_pause(self):
        if self.view:
            self.view.toggle_pause()
            self.pause_btn.config(text="Resume" if self.view.paused else "Pause")

    def _tick(self):
        self.view.step()
        self.after(FRAME_MS, self._tick)

    def _refresh_tables(self):
        v = self.view
        self.helio.set_rows([(n, f"{d:.4f}", f"{s:.2f}") for n, d, s in v.heliocentric_rows()])
        self.geo.set_rows([(n, f"{d:,.0f}", f"{t:.2f}") for n, d, t in v.geocentric_rows()])
        self.time_label.config(text=f"{v.date_str()}   {v.days_per_second:+.1f} days/s")
        self.pause_btn.config(text="Resume" if v.paused else "Pause")   # the planner pauses too

        for m in self.control.update(v.sim_et):
            lost = m.status == LOST or any(f.status == LOST for f in m.flights)
            self.news.config(text=f"{m.name}: launched" if m.status == ACTIVE and not lost else f"{m.name}: {m.end_note}",
                             foreground="#c33" if lost else "#2a7")
        self._report_arrivals()
        self._report_deaths(v.sim_et)
        self._sync_paths()
        self.missions.refresh(self.control, v.sim_et)
        self.after(TABLE_MS, self._refresh_tables)

    def _report_arrivals(self):
        """News when a craft arrives somewhere (goes idle) or a next leg sets off."""
        for m in self.control.missions:
            for f in m.flights:
                key = self.path_key(m, f)
                old = self._flight_status.get(key)
                if old is not None and old != f.status:
                    if f.status == IDLE:
                        self.news.config(text=f"{m.name}: {f.name} arrived: {f.end_note}", foreground="#2a7")
                    elif f.status == ACTIVE and f.leg_of is not None:
                        self.news.config(text=f"{m.name}: {f.name} set off", foreground="#2a7")
                self._flight_status[key] = f.status

    def _report_deaths(self, now):
        """Post news when crew die at a base (deaths are replayed, so compare with what we've shown)."""
        for s in sites_at(self.control, now):
            seen = self._deaths_seen.get(s.name, 0)
            if s.deaths > seen:
                last = [e for e in s.log if e.deaths][-1]
                self.news.config(text=f"{s.name}: {last.text}", foreground="#c33")
                self._deaths_seen[s.name] = s.deaths

    # ---------- Missions ----------
    def control_by_name(self, name):
        return next((m for m in self.control.missions if m.name == name), None)

    def open_planner(self, mission=None):
        from planner import MissionPlanner
        if self.planner is not None:
            self.planner.lift()
            return
        self.planner = MissionPlanner(self, mission)

    def planner_closed(self, planner):
        self.planner = None

    def mission_changed(self, mission, old_name=None, removed=False, old_flights=None):
        """Called by the planner after it schedules, edits, cancels or ends a mission."""
        for name in {old_name, mission.name} - {None}:
            for key in [k for k in self._drawn if k.startswith(name + "/")]:
                self.view.remove_path(key)
                del self._drawn[key]
        if not removed:
            self.news.config(text=f"{mission.name}: plan saved")
        self._sync_paths()
        self.missions.refresh(self.control, self.view.sim_et)

    @staticmethod
    def path_key(mission, flight):
        return f"{mission.name}/{flight.name}"

    def _sync_paths(self):
        """Draw each flying or planned craft's path, redrawing any whose plan or status changed."""
        live = {}
        for m in self.control.by_status(PENDING, ACTIVE):
            for f in m.flights:
                if f.status in (PENDING, ACTIVE, IDLE):
                    live[self.path_key(m, f)] = (m, f)
        for key in list(self._drawn):
            if key not in live:
                self.view.remove_path(key)
                del self._drawn[key]
        for key, (m, f) in live.items():
            traj = f.trajectory
            drawn = self._drawn.get(key)
            if drawn is None or drawn[0] is not traj or drawn[1] != f.status:
                label = m.name if len(m.flights) == 1 else f"{m.name}: {f.name}"
                self.view.show_path(key, label, traj, self.PATH_COLORS[f.status],
                                    craft_window=(f.launch_et, f.end_et))
                self._drawn[key] = (traj, f.status)

    def destroy(self):
        if self.view:
            self.view.destroy()
        super().destroy()


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Inner Solar System (JPL DE440)")
        self.geometry("1400x800")
        self.minsize(1000, 600)
        self.page = StartPage(self, on_start=self.show_main)
        self.page.pack(fill="both", expand=True)

    def show_main(self):
        self.page.destroy()
        self.page = MainPage(self)
        self.page.pack(fill="both", expand=True)
        self.page.start_view()


if __name__ == "__main__":
    App().mainloop()
