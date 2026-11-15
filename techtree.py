"""Tech tree window: spend science points to unlock stages and modules.

Also lists the firsts earned so far and what is earning science every day.
It only reads what MainPage's Science already worked out, so it costs nothing
to keep open.
"""
import tkinter as tk
from tkinter import ttk

from physics import ephemeris

REFRESH_MS = 500


def short_date(et):
    return ephemeris.et2str(et)[:11]


class TechTreeWindow(tk.Toplevel):
    def __init__(self, page):
        super().__init__(page)
        self.page = page
        self.science = page.science
        self.title("Tech tree")
        self.geometry("820x640")
        self.protocol("WM_DELETE_WINDOW", self.close)

        top = ttk.Frame(self, padding=8)
        top.pack(fill="x")
        self.balance = ttk.Label(top, font=("TkDefaultFont", 12, "bold"))
        self.balance.pack(side="left")
        self.unlock_btn = ttk.Button(top, text="Unlock", command=self._unlock)
        self.unlock_btn.pack(side="right")

        box = ttk.LabelFrame(self, text="Technologies", padding=6)
        box.pack(fill="both", expand=True, padx=8)
        cols = [("Tech", 140), ("Cost", 60), ("Needs", 140), ("Unlocks", 260), ("Status", 160)]
        self.tree = ttk.Treeview(box, columns=[c for c, _ in cols], show="headings", height=10,
                                 selectmode="browse")
        for col, width in cols:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=width, anchor="w")
        self.tree.pack(fill="both", expand=True)
        self.tree.tag_configure("done", foreground="#2a7")
        self.tree.tag_configure("ready", foreground="#000")
        self.tree.tag_configure("locked", foreground="#999")
        self.tree.bind("<<TreeviewSelect>>", lambda e: self._refresh_button())
        self.tree.bind("<Double-1>", lambda e: self._unlock())
        self.note = ttk.Label(self, foreground="#555", padding=(8, 2),
                              text="Tier 0 parts, every launcher and the torch drive are always available.")
        self.note.pack(anchor="w")

        lower = ttk.Frame(self, padding=8)
        lower.pack(fill="both", expand=True)
        lower.columnconfigure(0, weight=1)
        lower.columnconfigure(1, weight=1)
        self.firsts = self._list(lower, "Firsts earned", [("Date", 90), ("What", 230), ("Points", 60)], 0)
        self.sources = self._list(lower, "Earning every day", [("Body", 70), ("What", 200), ("Per day", 70)], 1)

        self.status = ttk.Label(self, foreground="#c33", padding=(8, 0, 8, 8))
        self.status.pack(anchor="w")
        self._shown = None
        self._refresh()

    @staticmethod
    def _list(parent, title, cols, column):
        box = ttk.LabelFrame(parent, text=title, padding=6)
        box.grid(row=0, column=column, sticky="nsew", padx=(0, 6) if column == 0 else 0)
        tree = ttk.Treeview(box, columns=[c for c, _ in cols], show="headings", height=7, selectmode="none")
        for col, width in cols:
            tree.heading(col, text=col)
            tree.column(col, width=width, anchor="e" if col in ("Points", "Per day") else "w")
        tree.pack(fill="both", expand=True)
        return tree

    def _refresh(self):
        s = self.science
        self.balance.config(text=f"Science: {s.balance():,.0f}   ({s.per_day():+.2f} per day)")
        # Only rebuild the lists when something in them changed (keeps the selection steady)
        state = (tuple(s.unlocked), int(s.balance()), len(s.firsts),
                 tuple((x.what, x.body, round(x.per_day, 3)) for x in s.sources))
        if state != self._shown:
            self._shown = state
            self._fill()
        self._refresh_button()
        self.after(REFRESH_MS, self._refresh)

    def _fill(self):
        s, cat = self.science, self.science.catalogue
        sel = self.tree.selection()
        self.tree.delete(*self.tree.get_children())
        for node in s.rules.tech.values():
            parts = [(cat.stages.get(p) or cat.modules.get(p) or cat.launchers.get(p)).name for p in node.parts]
            needs = ", ".join(s.rules.tech[n].name for n in node.needs) or "-"
            if node.id in s.unlocked:
                status, tag = f"Unlocked {short_date(s.unlocked[node.id])}", "done"
            else:
                problem = s.unlock_problem(node.id)
                status, tag = ("Ready to unlock", "ready") if problem is None else (problem, "locked")
            self.tree.insert("", "end", iid=node.id, values=(node.name, f"{node.cost:,.0f}", needs,
                                                             ", ".join(parts), status), tags=(tag,))
        if sel and self.tree.exists(sel[0]):
            self.tree.selection_set(sel)

        self.firsts.delete(*self.firsts.get_children())
        for f in reversed(s.firsts):
            self.firsts.insert("", "end", values=(short_date(f.et), f.text, f"+{f.points:,.0f}"))
        if not s.firsts:
            self.firsts.insert("", "end", values=("", "Nothing yet: go somewhere new!", ""))
        self.sources.delete(*self.sources.get_children())
        for x in s.sources:
            self.sources.insert("", "end", values=(x.body, x.what, f"{x.per_day:.2f}"))
        if not s.sources:
            self.sources.insert("", "end", values=("", "Deploy satellites, or land a science kit", ""))

    def _refresh_button(self):
        sel = self.tree.selection()
        ok = bool(sel) and self.science.unlock_problem(sel[0]) is None
        self.unlock_btn.config(state="normal" if ok else "disabled")

    def _unlock(self):
        sel = self.tree.selection()
        if not sel:
            return
        problem = self.science.unlock_problem(sel[0])
        if problem:
            self.status.config(text=problem, foreground="#c33")
            return
        self.science.unlock(sel[0], self.page.view.sim_et)
        node = self.science.rules.tech[sel[0]]
        self.status.config(text=f"Unlocked {node.name}", foreground="#2a7")
        self.page.tech_changed()
        self._shown = None
        self._fill()

    def close(self):
        self.page.techtree_closed()
        self.destroy()
