"""
Tkinter front end: a start page, then a main page with the 3D solar system view
in the middle, info tables on either side and control buttons underneath.

Setup:
    pip install ursina spiceypy numpy
    Put naif0012.tls and de440.bsp (or the smaller de440s.bsp) in the same folder as this script.

Mouse controls inside the 3D view:
    Right-drag      orbit camera      Scroll       zoom
    Middle-drag     pan (free camera only)
"""
import tkinter as tk
from tkinter import ttk

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


class MainPage(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent, padding=6)
        self.view = None

        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        self.helio = InfoTable(self, "From the Sun",
                               [("Body", 70), ("Distance (AU)", 95), ("Speed (km/s)", 90)])
        self.helio.grid(row=0, column=0, sticky="ns", padx=(0, 6))

        # Panda3D renders straight into this frame's native window
        self.view_frame = tk.Frame(self, bg="black", width=800, height=600)
        self.view_frame.grid(row=0, column=1, sticky="nsew")

        self.geo = InfoTable(self, "From Earth",
                             [("Body", 70), ("Distance (km)", 110), ("Light time (min)", 100)])
        self.geo.grid(row=0, column=2, sticky="ns", padx=(6, 0))

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
        return bar

    def start_view(self):
        """Create the 3D view once this page is on screen (its window must exist)."""
        self.update()
        from solar_view import SolarSystemView   # heavy import; only needed from here on
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
        self.after(TABLE_MS, self._refresh_tables)

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
