"""Pop-up tutorial: a guided tour like a new game's first-run tutorial.

A Coach shows one pop-up at a time. A pop-up either explains something (with
Back/Next) or asks you to do one thing ("Click New mission…"); then an orange
frame blinks around the exact button, list or tab to use, and the pop-up sits
beside it. As soon as you've done it, the next pop-up appears by itself.

Where you are is worked out fresh every refresh from the game itself: the first
step that isn't done yet is the one shown. So if you close the planner halfway,
the tour goes back to "Click New mission…" on its own, and if you do something
early, the steps for it are skipped. Explanation pages count as done once you
press Next.

Pop-ups never take the keyboard focus (typing in the planner still works), and
they hide while another program is in front.

Each scenario's steps are in STEPS (scenario id -> function returning [Step]).
"""
from dataclasses import dataclass, field
import ctypes
import os
import sys
import tkinter as tk
from tkinter import ttk

from game.mission import ACTIVE, IDLE, PENDING

BG, FG, ACCENT, MUTED = "#1d2733", "#f2f2f2", "#ff9f1a", "#9fb0c0"
BORDER = 4


@dataclass
class Step:
    title: str
    text: str
    target: object = None        # (coach) -> widget, (x, y, w, h) screen rect, or None (centred pop-up)
    done: object = None          # (page) -> bool; None = an explanation page (Next button)
    button: str = "Next"         # the button on an explanation page
    enter: object = None         # (coach) -> None, run when the step is first shown (switch a tab...)
    seen: bool = field(default=False, repr=False)

    @property
    def is_info(self):
        return self.done is None


# ---------- What the steps look at ----------
def planner(page):
    p = page.planner
    try:
        return p if p is not None and p.winfo_exists() else None
    except tk.TclError:
        return None


def techtree(page):
    t = page.techtree
    try:
        return t if t is not None and t.winfo_exists() else None
    except tk.TclError:
        return None


def design(page):
    p = planner(page)
    return p.craft.design if p else None


def draft(page):
    p = planner(page)
    return p.draft if p else None


def flights(page):
    return [f for m in page.control.missions for f in m.flights]


def scheduled(page):
    return bool(page.control.missions)


def launched(page):
    return any(m.status != PENDING for m in page.control.missions)


def first_flight(page):
    return page.control.missions[0].flights[0] if page.control.missions else None


def has(d, mid):
    return d is not None and d.modules.get(mid, 0) > 0


def selected(tree, iid):
    try:
        return tuple(tree.selection()) == (iid,)
    except tk.TclError:
        return False


def tab_is(notebook, index):
    try:
        return notebook.index("current") == index
    except tk.TclError:
        return False


def safe(fn, *a):
    try:
        return bool(fn(*a))
    except Exception:
        return False


# ---------- Where to point ----------
def tab_rect(notebook, index):
    """Screen rectangle of one tab's label in a ttk.Notebook."""
    try:
        xs = [x for x in range(2, notebook.winfo_width(), 3) if _tab_at(notebook, x) == index]
    except tk.TclError:
        return None
    if not xs:
        return None
    return notebook.winfo_rootx() + xs[0] - 4, notebook.winfo_rooty(), xs[-1] - xs[0] + 8, 24


def _tab_at(nb, x):
    try:
        return nb.index(f"@{x},10")
    except tk.TclError:
        return None


def row_rect(tree, iid):
    try:
        bb = tree.bbox(iid)
    except tk.TclError:
        return None
    if not bb:
        return None
    x, y, w, h = bb
    return tree.winfo_rootx() + x, tree.winfo_rooty() + y, w, h


def panel_tab(key):
    """Point at a tab of the Missions panel (pending, active, done, bases, goals, terraform)."""
    def target(coach):
        panel = coach.page.missions
        return tab_rect(panel.notebook, [t[0] for t in panel.TABS].index(key))
    return target


def show_panel_tab(key):
    def enter(coach):
        panel = coach.page.missions
        panel.notebook.select([t[0] for t in panel.TABS].index(key))
    return enter


def in_planner(attr):
    """Point at a widget of the planner (e.g. "save_btn"), if the planner is open."""
    def target(coach):
        p = planner(coach.page)
        return getattr(p, attr, None) if p else None
    return target


def planner_tab(index):
    def target(coach):
        p = planner(coach.page)
        return tab_rect(p.left_tabs, index) if p else None
    return target


def module_row(mid):
    def target(coach):
        p = planner(coach.page)
        return row_rect(p.module_tree, mid) if p else None
    return target


def select_module(mid):
    """Scroll the module list so the row is visible (the step then points at it)."""
    def enter(coach):
        p = planner(coach.page)
        if p:
            p.module_tree.see(mid)
    return enter


def main_widget(attr):
    def target(coach):
        return getattr(coach.page, attr, None)
    return target


def panel_widget(attr):
    def target(coach):
        return getattr(coach.page.missions, attr, None)
    return target


# ---------- The tour engine ----------
class Coach:
    def __init__(self, page, steps):
        self.page = page
        self.steps = steps
        self.active = True
        self._shown = None             # index of the step whose pop-up is up
        self._geom = None
        self._blink = 0
        root = page.winfo_toplevel()
        self.pop = self._make_window(root)
        self.frame = [self._make_window(root, bg=ACCENT) for _ in range(4)]
        self._build_popup()
        self._hidden = True

    # ----- windows -----
    def _make_window(self, root, bg=BG):
        w = tk.Toplevel(root, bg=bg)
        w.overrideredirect(True)
        w.attributes("-topmost", True)
        w.withdraw()
        w.update_idletasks()
        if sys.platform == "win32":
            # Never take the keyboard focus or show in the taskbar
            hwnd = ctypes.windll.user32.GetParent(w.winfo_id()) or w.winfo_id()
            GWL_EXSTYLE, WS_EX_NOACTIVATE, WS_EX_TOOLWINDOW = -20, 0x08000000, 0x00000080
            style = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
            ctypes.windll.user32.SetWindowLongW(hwnd, GWL_EXSTYLE, style | WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW)
        return w

    def _build_popup(self):
        outer = tk.Frame(self.pop, bg=ACCENT, padx=2, pady=2)
        outer.pack(fill="both", expand=True)
        box = tk.Frame(outer, bg=BG, padx=14, pady=10)
        box.pack(fill="both", expand=True)
        self.arrow = tk.Label(box, bg=BG, fg=ACCENT, font=("TkDefaultFont", 9, "bold"), anchor="w")
        self.arrow.pack(fill="x")
        self.counter = tk.Label(box, bg=BG, fg=MUTED, font=("TkDefaultFont", 8), anchor="w")
        self.counter.pack(fill="x")
        self.title = tk.Label(box, bg=BG, fg=FG, font=("TkDefaultFont", 12, "bold"), anchor="w",
                              justify="left", wraplength=330)
        self.title.pack(fill="x", pady=(2, 4))
        self.text = tk.Label(box, bg=BG, fg=FG, font=("TkDefaultFont", 10), anchor="w", justify="left",
                             wraplength=330)
        self.text.pack(fill="x")
        self.todo = tk.Label(box, bg=BG, fg=ACCENT, font=("TkDefaultFont", 9, "italic"), anchor="w")
        self.todo.pack(fill="x", pady=(6, 0))
        row = tk.Frame(box, bg=BG)
        row.pack(fill="x", pady=(10, 0))
        self.skip_btn = tk.Button(row, text="Skip tutorial", command=self.skip, bg=BG, fg=MUTED, bd=0,
                                  activebackground=BG, activeforeground=FG, cursor="hand2")
        self.skip_btn.pack(side="left")
        self.next_btn = tk.Button(row, text="Next", command=self.next, bg=ACCENT, fg="#111", bd=0, padx=14,
                                  pady=3, activebackground="#ffbf5a", cursor="hand2",
                                  font=("TkDefaultFont", 10, "bold"))
        self.next_btn.pack(side="right")
        self.back_btn = tk.Button(row, text="Back", command=self.back, bg="#33414f", fg=FG, bd=0, padx=10, pady=3,
                                  activebackground="#46586a", cursor="hand2")
        self.back_btn.pack(side="right", padx=6)

    def _hide(self):
        if not self._hidden:
            self.pop.withdraw()
            for w in self.frame:
                w.withdraw()
            self._hidden = True
            self._geom = None

    # ----- progress -----
    def current(self):
        """The first step not done yet (explanation pages are done once Next was pressed)."""
        for k, s in enumerate(self.steps):
            if s.is_info:
                if not s.seen:
                    return k
            elif not safe(s.done, self.page):
                return k
        return None

    def next(self):
        k = self.current()
        if k is not None and self.steps[k].is_info:
            self.steps[k].seen = True
            if k == len(self.steps) - 1:
                self.finish()
            self.refresh()

    def back(self):
        """Back to the previous explanation page (only shown when the step before is one)."""
        k = self.current()
        if k and self.steps[k - 1].is_info:
            self.steps[k - 1].seen = False
            self.refresh()

    def skip(self):
        self.active = False
        self._hide()

    def finish(self):
        self.active = False
        self._hide()

    def resume(self):
        """The Guide button: show the tour again where it left off (or from the start if finished)."""
        if self.current() is None:
            for s in self.steps:
                s.seen = False
        self.active = True
        self._shown = None
        self.refresh()

    # ----- drawing -----
    def refresh(self):
        if not self.active:
            return
        k = self.current()
        if k is None or not _game_in_front(self.page):
            self._hide()
            return
        step = self.steps[k]
        if k != self._shown:
            self._shown = k
            self._geom = None
            if step.enter:
                safe(step.enter, self)
            n_steps = len(self.steps)
            self.counter.config(text=f"Step {k + 1} of {n_steps}")
            self.title.config(text=step.title)
            self.text.config(text=step.text)
            self.todo.config(text="" if step.is_info else "Do this to carry on: the next step appears by itself.")
            if step.is_info:
                self.next_btn.config(text=step.button)
                self.next_btn.pack(side="right")
            else:
                self.next_btn.pack_forget()
            if k > 0 and self.steps[k - 1].is_info and step.is_info:
                self.back_btn.pack(side="right", padx=6)
            else:
                self.back_btn.pack_forget()
            self.skip_btn.config(text="Close" if k == len(self.steps) - 1 else "Skip tutorial")
        rect = self._rect(step)
        self._place(rect)

    def _rect(self, step):
        if step.target is None:
            return None
        try:
            t = step.target(self)
        except Exception:
            return None
        if t is None:
            return None
        if isinstance(t, tuple):
            return t
        try:
            if not t.winfo_ismapped():
                return None
            return t.winfo_rootx(), t.winfo_rooty(), t.winfo_width(), t.winfo_height()
        except tk.TclError:
            return None

    def _place(self, rect):
        self.pop.update_idletasks()
        pw, ph = self.pop.winfo_reqwidth(), self.pop.winfo_reqheight()
        sw, sh = _screen_size(self.pop)
        if rect is None:
            root = self.page.winfo_toplevel()
            x = root.winfo_rootx() + (root.winfo_width() - pw) // 2
            y = root.winfo_rooty() + (root.winfo_height() - ph) // 3
            arrow = ""
        else:
            rx, ry, rw, rh = rect
            gap = 18
            if rx + rw + gap + pw < sw:                      # to the right
                x, y, arrow = rx + rw + gap, ry + rh // 2 - 30, "◀  over here"
            elif rx - gap - pw > 0:                          # to the left
                x, y, arrow = rx - gap - pw, ry + rh // 2 - 30, "over here  ▶"
            elif ry + rh + gap + ph < sh:                    # below
                x, y, arrow = rx, ry + rh + gap, "▲  over here"
            else:                                            # above
                x, y, arrow = rx, ry - gap - ph, "▼  over here"
        x = max(4, min(int(x), sw - pw - 4))
        y = max(4, min(int(y), sh - ph - 40))
        self.arrow.config(text=arrow)
        geom = (x, y, pw, ph, rect)
        if geom != self._geom:
            self._geom = geom
            _move(self.pop, x, y, pw, ph)
            if rect is not None:
                rx, ry, rw, rh = rect
                b = BORDER
                for w, (fx, fy, fw, fh) in zip(self.frame, [(rx - b - 2, ry - b - 2, rw + 2 * b + 4, b),
                                                            (rx - b - 2, ry + rh + 2, rw + 2 * b + 4, b),
                                                            (rx - b - 2, ry - 2, b, rh + 4),
                                                            (rx + rw + 2, ry - 2, b, rh + 4)]):
                    _move(w, fx, fy, max(1, fw), max(1, fh))
        if self._hidden:
            self.pop.deiconify()
            self._hidden = False
        for w in self.frame:
            if rect is None:
                w.withdraw()
            elif w.state() == "withdrawn":
                w.deiconify()
        # Blink the frame so it catches the eye
        self._blink += 1
        colour = ACCENT if self._blink % 4 < 2 else "#ffe08a"
        for w in self.frame:
            w.config(bg=colour)
        # Keep the pop-ups above windows that opened since (the planner, the tech tree)
        self.pop.lift()
        for w in self.frame:
            w.lift()


def _move(win, x, y, w, h):
    """Move and size a pop-up window. Tk alone sometimes leaves an already-visible
    borderless window where it was, so on Windows the move is also done directly."""
    win.geometry(f"{w}x{h}+{x}+{y}")
    if sys.platform == "win32":
        try:
            hwnd = ctypes.windll.user32.GetParent(win.winfo_id()) or win.winfo_id()
            SWP_NOZORDER, SWP_NOACTIVATE = 0x0004, 0x0010
            ctypes.windll.user32.SetWindowPos(hwnd, 0, int(x), int(y), int(w), int(h), SWP_NOZORDER | SWP_NOACTIVATE)
        except Exception:
            pass


def _screen_size(widget):
    """Screen size in the same pixels as winfo_rootx/y. Once Panda3D starts, the process is
    DPI-aware and Tk positions are real pixels, but winfo_screenwidth can still report the
    scaled-down size (e.g. 1280x720 for 1920x1080 at 150%), so ask Windows directly."""
    if sys.platform == "win32":
        try:
            return ctypes.windll.user32.GetSystemMetrics(0), ctypes.windll.user32.GetSystemMetrics(1)
        except Exception:
            pass
    return widget.winfo_screenwidth(), widget.winfo_screenheight()


def _game_in_front(page):
    """True while this game (any of its windows) is the program in front."""
    if sys.platform != "win32":
        return True
    try:
        user32 = ctypes.windll.user32
        hwnd = user32.GetForegroundWindow()
        pid = ctypes.c_ulong()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        return pid.value == os.getpid()
    except Exception:
        return True


# ---------- The tours ----------
def welcome(goal_text):
    """The opening pages: what the game is, and a tour of the main screen."""
    return [
        Step("Welcome, Mission Director!",
             "You plan real space missions. Every planet in this game is exactly where NASA's JPL says it "
             "is on that date, and your craft fly by real orbital physics.\n\n" + goal_text +
             "\n\nThis tutorial shows you around, then walks you through it click by click.",
             button="Start the tour"),
        Step("The solar system",
             "This is the live 3D view. Click a planet to fly the camera to it, scroll to zoom, and "
             "right-drag to turn the view. Press H in the view for the keyboard keys.",
             target=main_widget("view_frame")),
        Step("Time",
             "The game starts paused. Resume starts the clock, and Slower and Faster change how many days "
             "pass each second. Missions only happen while the clock runs.",
             target=main_widget("time_box")),
        Step("Your missions",
             "Everything you launch is listed here. Pending: scheduled, not launched yet. Ongoing: flying. "
             "Previous: finished, each with a report. Bases: what you've left on other worlds.",
             target=lambda c: c.page.missions.notebook, enter=show_panel_tab(PENDING)),
        Step("Your goal",
             "The Goals tab shows what this level asks for, your budget, and later your score out of 100.",
             target=panel_tab("goals"), enter=show_panel_tab("goals")),
        Step("Science and the tech tree",
             "Doing something for the first time (first orbit, first landing...) earns science points. "
             "Spend them in the Tech tree to unlock better engines and modules.",
             target=main_widget("science_box")),
    ]


def first_satellite():
    def planner_open(p):
        return planner(p) is not None or scheduled(p)

    def sat_on_board(p):
        return has(design(p), "satellite") or scheduled(p)

    def sat_selected(p):
        pl = planner(p)
        return (pl is not None and selected(pl.module_tree, "satellite")) or sat_on_board(p)

    def route_tab(p):
        pl = planner(p)
        return (pl is not None and tab_is(pl.left_tabs, 1)) or draft(p) is not None or scheduled(p)

    def earth_orbit(p):
        pl = planner(p)
        return (pl is not None and pl.target.get() == "Earth orbit") or draft(p) is not None or scheduled(p)

    def route_set(p):
        d = draft(p)
        return (d is not None and d.target == "Earth") or scheduled(p)

    def in_orbit(p):
        f = first_flight(p)
        return f is not None and (f.status == IDLE or bool(p.science.deployments))

    def mission_open(p):
        pl = planner(p)
        return (pl is not None and pl.original is not None and pl.original.status == ACTIVE) or \
            bool(p.science.deployments)

    def select_ongoing(coach):
        panel = coach.page.missions
        panel.notebook.select([t[0] for t in panel.TABS].index(ACTIVE))
        m = coach.page.control.missions[0] if coach.page.control.missions else None
        if m is not None and panel.trees[ACTIVE].exists(m.name):
            panel.trees[ACTIVE].selection_set(m.name)

    return welcome("Your first mission: put a science satellite into orbit around Earth. It takes "
                   "about five minutes.") + [
        Step("Open the mission planner",
             "Every mission is built in the mission planner. Click New mission….",
             target=panel_widget("new_btn"), done=planner_open),
        Step("This is your craft",
             "On the left you build the craft. The launcher lifts it from the ground into orbit. The stage "
             "is the engine and fuel for every burn after that. Then there's the crew and the modules on "
             "board. The starter craft has 2 crew, a capsule, food and water, and a solar array.",
             target=lambda c: getattr(planner(c.page), "build_tab", None)),
        Step("Find the satellite",
             "These are the modules you can carry. Click 'Science satellite' in the list (scroll down if "
             "you can't see it).",
             target=module_row("satellite"), enter=select_module("satellite"), done=sat_selected),
        Step("Put it on board",
             "Click Add one. Its row turns green and the 'On board' column says 1.",
             target=lambda c: planner(c.page).module_buttons[0], done=sat_on_board),
        Step("Craft check",
             "This box checks the craft for you. Green is fine, amber is cutting it close, and red won't work. "
             "Δv is how much the engine can change speed (your fuel). Supplies is food and water. "
             "Everything should be green for this trip.",
             target=in_planner("health_box")),
        Step("Where is it going?",
             "Click the Route tab.",
             target=planner_tab(1), done=route_tab),
        Step("Pick Earth orbit",
             "Open the Target list and choose 'Earth orbit'.",
             target=in_planner("target_box"), done=earth_orbit),
        Step("Set the launch",
             "Leave the date and the 200 km height as they are, and click Set launch. The plan appears on the "
             "right.",
             target=in_planner("set_launch_btn"), done=route_set),
        Step("The plan",
             "The top line on the right says how the flight ends ('Stable orbit around Earth'). Below it are "
             "the burns and events. There's also a System map tab with a top-down view of the route.",
             target=in_planner("outcome")),
        Step("Schedule it",
             "Click Schedule launch. The planner closes and your mission waits in the Pending list.",
             target=in_planner("save_btn"), done=scheduled),
        Step("Start the clock",
             "Time is still paused. Click Resume (or click the 3D view and press Space).",
             target=main_widget("pause_btn"), done=launched),
        Step("Into orbit",
             "Your craft launches and goes once around Earth (about an hour and a half). After a full lap it "
             "counts as settled in orbit. Click the Earth button to watch it.",
             target=lambda c: c.page.camera_box, done=in_orbit),
        Step("Open your mission",
             "It's in orbit. Your mission is selected in the Ongoing tab: click Open….",
             target=panel_widget("open_btn"), enter=select_ongoing, done=mission_open),
        Step("Release the satellite",
             "Click Deploy satellite (bottom left of the planner). The satellite stays in orbit and earns "
             "science every day.",
             target=in_planner("deploy_btn"), done=lambda p: bool(p.science.deployments)),
        Step("Mission complete!",
             "That's the goal done. You'll get your score in a moment, and the Goals tab keeps it.\n\n"
             "Next, try Easy: land two astronauts on the Moon and bring them home. It has its own guide.\n\n"
             "Tip: the Guide button (bottom bar) brings this tutorial back any time.",
             button="Finish"),
    ]


def first_orbit():
    def planner_open(p):
        return planner(p) is not None or scheduled(p)

    def route_tab(p):
        pl = planner(p)
        return (pl is not None and tab_is(pl.left_tabs, 1)) or draft(p) is not None or scheduled(p)

    def earth_orbit(p):
        pl = planner(p)
        return (pl is not None and pl.target.get() == "Earth orbit") or draft(p) is not None or scheduled(p)

    def route_set(p):
        d = draft(p)
        return (d is not None and d.target == "Earth") or scheduled(p)

    return welcome("Your first mission: put a crewed craft into a stable orbit around Earth.") + [
        Step("Open the mission planner", "Click New mission….", target=panel_widget("new_btn"), done=planner_open),
        Step("Your craft",
             "The starter craft (2 crew, a capsule, supplies and a solar array) is all this needs. The Craft "
             "check box below it checks it for you: green is fine, red won't work.",
             target=in_planner("health_box")),
        Step("Where is it going?", "Click the Route tab.", target=planner_tab(1), done=route_tab),
        Step("Pick Earth orbit", "Open the Target list and choose 'Earth orbit'.",
             target=in_planner("target_box"), done=earth_orbit),
        Step("Set the launch", "Click Set launch.", target=in_planner("set_launch_btn"), done=route_set),
        Step("Schedule it", "Click Schedule launch.", target=in_planner("save_btn"), done=scheduled),
        Step("Start the clock", "Click Resume.", target=main_widget("pause_btn"), done=launched),
        Step("One lap", "After one full lap around Earth the craft counts as in orbit and the Goals tab ticks "
                        "it off.",
             target=panel_tab("goals"), enter=show_panel_tab("goals"),
             done=lambda p: p.tracker.status != "playing"),
        Step("Done!", "Next: the satellite tutorial, or land on the Moon (Easy).", button="Finish"),
    ]


def moon_landing():
    def tt_open(p):
        return techtree(p) is not None or p.science.is_unlocked("methalox")

    def methalox_selected(p):
        t = techtree(p)
        return (t is not None and selected(t.tree, "methalox")) or p.science.is_unlocked("methalox")

    def built(p, check):
        d = design(p)
        return (d is not None and check(d)) or scheduled(p)

    def planner_open(p):
        return planner(p) is not None or scheduled(p)

    def route_tab(p):
        pl = planner(p)
        return (pl is not None and tab_is(pl.left_tabs, 1)) or draft(p) is not None or scheduled(p)

    def moon_target(p):
        pl = planner(p)
        return (pl is not None and pl.target.get() == "Moon") or draft(p) is not None or scheduled(p)

    def land_ticked(p):
        pl = planner(p)
        return (pl is not None and getattr(pl, "land_var", None) is not None and pl.land_var.get()) or \
            draft(p) is not None or scheduled(p)

    def lands(p):
        d = draft(p)
        return (d is not None and d.outcome.kind == "impact" and d.outcome.body == "Moon") or scheduled(p)

    def landed(p):
        f = first_flight(p)
        return f is not None and (f.location() == ("Moon", "surface") and f.status != PENDING
                                  and f.idle_since is not None and f.idle_since <= p.view.sim_et)

    def leg_saved(p):
        return any(f.leg_of is not None for f in flights(p))

    def mission_open(p):
        pl = planner(p)
        return (pl is not None and pl.original is not None) or leg_saved(p)

    def leg_planned(p):
        pl = planner(p)
        return (pl is not None and any(c.leg_of is not None for c in pl.crafts)) or leg_saved(p)

    def leg_route_tab(p):
        pl = planner(p)
        return (pl is not None and pl.craft.leg_of is not None and tab_is(pl.left_tabs, 1)) or \
            (pl is not None and pl.craft.leg_of is not None and pl.draft is not None) or leg_saved(p)

    def leg_set(p):
        pl = planner(p)
        return (pl is not None and pl.craft.leg_of is not None and pl.draft is not None) or leg_saved(p)

    def select_ongoing(coach):
        panel = coach.page.missions
        panel.notebook.select([t[0] for t in panel.TABS].index(ACTIVE))
        m = coach.page.control.missions[0] if coach.page.control.missions else None
        if m is not None and panel.trees[ACTIVE].exists(m.name):
            panel.trees[ACTIVE].selection_set(m.name)

    return [
        Step("Moon landing",
             "Land two astronauts on the Moon and bring them home, within the budget. You're scored on how "
             "much money is left and whether everyone survives.\n\nThe tricky part: after landing, the craft "
             "has to take off again. Let's set that up first.",
             button="Let's go"),
        Step("Open the tech tree",
             "Your starting engine can't take off from the Moon. Click Tech tree….",
             target=main_widget("techtree_btn"), done=tt_open),
        Step("Pick Methalox engines",
             "Click 'Methalox engines'. It's a stage that can relaunch from a surface, and it costs 40 of "
             "your 70 science points.",
             target=lambda c: row_rect(techtree(c.page).tree, "methalox"), done=methalox_selected),
        Step("Unlock it", "Click Unlock.",
             target=lambda c: techtree(c.page).unlock_btn, done=lambda p: p.science.is_unlocked("methalox")),
        Step("Open the mission planner",
             "Close the tech tree, then click New mission….",
             target=panel_widget("new_btn"), done=planner_open),
        Step("A bigger rocket",
             "Choose 'Heavy launcher' in the Launcher list. The lander will be heavy with fuel.",
             target=in_planner("launcher_box"), done=lambda p: built(p, lambda d: d.launcher == "heavy")),
        Step("The Methalox stage", "Choose 'Methalox stage' in the Stage list.",
             target=in_planner("stage_box"), done=lambda p: built(p, lambda d: d.stage == "methalox")),
        Step("Fill it up",
             "Click Fill tank. You'll need it all: getting to the Moon, taking off again and coming home.",
             target=in_planner("fill_btn"),
             done=lambda p: built(p, lambda d: d.stage == "methalox" and d.fuel_kg >= 39_999)),
        Step("Somewhere to live",
             "Astronauts who land without a place in a habitat die. Click 'Surface shelter (2)' in the list.",
             target=module_row("shelter2"), enter=select_module("shelter2"),
             done=lambda p: (planner(p) is not None and selected(planner(p).module_tree, "shelter2"))
             or built(p, lambda d: has(d, "shelter2"))),
        Step("Add it", "Click Add one.",
             target=lambda c: planner(c.page).module_buttons[0], done=lambda p: built(p, lambda d: has(d, "shelter2"))),
        Step("Check the craft",
             "Craft check: Mass must fit the launcher, Δv is your fuel, and Crew should say '2 habitat "
             "place(s) on landing' once the route is set. Read it again before you schedule.",
             target=in_planner("health_box")),
        Step("Plan the route", "Click the Route tab.", target=planner_tab(1), done=route_tab),
        Step("To the Moon", "Choose 'Moon' as the Target.", target=in_planner("target_box"), done=moon_target),
        Step("Land on arrival", "Tick 'Land on arrival'. Without it the craft just swings past the Moon.",
             target=in_planner("land_check"), done=land_ticked),
        Step("Plan it", "Click Plan transfer. The top line on the right should say 'Lands on Moon'.",
             target=in_planner("set_launch_btn"), done=lands),
        Step("Schedule it", "Click Schedule launch.", target=in_planner("save_btn"), done=scheduled),
        Step("Fly there", "Click Resume. The trip takes about 4 days of game time. Click the Moon button to "
                          "watch the landing.",
             target=main_widget("pause_btn"), done=launched),
        Step("Touchdown...", "Wait for the landing. The news line under the buttons will say it has arrived.",
             target=lambda c: c.page.camera_box, done=landed),
        Step("Time to come home",
             "They've landed: that's the first goal. Your mission is selected in the Ongoing tab: click Open….",
             target=panel_widget("open_btn"), enter=select_ongoing, done=mission_open),
        Step("Plan the next leg",
             "Click Plan next leg. A craft that has arrived somewhere can fly on to somewhere else.",
             target=in_planner("next_leg_btn"), done=leg_planned),
        Step("The way home", "Click the Route tab.", target=planner_tab(1), done=leg_route_tab),
        Step("Back to Earth",
             "The Target already says 'Earth (back down)'. Click Plan return to Earth. The top line should "
             "say 'Returns to Earth'. If Craft check shows red for Δv, there wasn't enough fuel.",
             target=in_planner("set_launch_btn"), done=leg_set),
        Step("Save it", "Click Save changes.", target=in_planner("save_btn"), done=leg_saved),
        Step("Home", "Resume the clock. When they land on Earth you win and get your score.",
             target=main_widget("pause_btn"), done=lambda p: p.tracker.status != "playing"),
        Step("Well flown!",
             "Every finished mission has a report with a score and tips: Previous tab, then Open….\n\n"
             "Next: Medium, a Mars base.",
             button="Finish"),
    ]


def mars_base():
    return [
        Step("Mars base",
             "Keep four astronauts living on Mars with 12 months of food and water, all at once.\n\n"
             "1. Tech tree: unlock 'Long stays' (Habitat (4), Large supply crate) and 'Big capsule'.\n"
             "2. Planner: Crew 4, a 4-seat capsule, a Habitat (4), Large supply crates (about 800 kg food "
             "and 1,300 kg water a year for 4 crew) and solar arrays (solar gives only 43% at Mars).\n"
             "3. Route: Target Mars, tick Land on arrival, Find launch windows, then Cheapest. Mars windows "
             "come round about every 26 months.\n"
             "4. The Bases tab shows the days of supplies left. Send supply runs before they run out.",
             button="Got it"),
    ]


def terraform_mars():
    return [
        Step("Terraform Mars",
             "Run a crewed Mars base and take Mars through three stages (Terraforming tab):\n\n"
             "1. Atmosphere factories (30 t each) on the surface. They need crew and lots of power: unlock "
             "'Nuclear power' for the 100 kW reactor.\n"
             "2. Orbital mirrors (15 t) settled in Mars orbit. No crew needed.\n"
             "3. Biosphere labs (20 t) on the surface, crewed and powered.\n\n"
             "More equipment means faster progress, and you're scored on speed. The Terraforming tab always "
             "tells you what's holding a stage back.",
             button="Got it"),
    ]


STEPS = {"first_satellite": first_satellite, "first_orbit": first_orbit, "moon_landing": moon_landing,
         "mars_base": mars_base, "terraform_mars": terraform_mars}


def coach_for(page, scenario_id):
    make = STEPS.get(scenario_id)
    return Coach(page, make()) if make else None
