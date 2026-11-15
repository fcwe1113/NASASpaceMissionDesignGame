"""Mission report window: the score out of 100, what happened craft by craft, and tips."""
import tkinter as tk
from tkinter import ttk

from game.report import mission_report
from physics import ephemeris


def short_date(et):
    return ephemeris.et2str(et)[:11]


class ReportWindow(tk.Toplevel):
    def __init__(self, page, mission):
        super().__init__(page)
        report = mission_report(mission, page.control, page.science, date=short_date)
        self.title(f"Mission report: {mission.name}")
        self.geometry("640x560")
        colour = {"Success": "#2a7", "Partly lost": "#c80", "Lost": "#c33"}[report.status]

        top = ttk.Frame(self, padding=10)
        top.pack(fill="x")
        ttk.Label(top, text=f"{report.name}: {report.status}", foreground=colour,
                  font=("TkDefaultFont", 14, "bold")).pack(anchor="w")
        ttk.Label(top, text=f"Score {report.score}/100", font=("TkDefaultFont", 12)).pack(anchor="w")
        bar = ttk.Progressbar(top, maximum=100, value=report.score, length=300)
        bar.pack(anchor="w", pady=4)
        ttk.Label(top, text=report.summary, foreground="#555", wraplength=600).pack(anchor="w")

        body = ttk.Frame(self, padding=(10, 0))
        body.pack(fill="both", expand=True)
        text = tk.Text(body, wrap="word", height=20, relief="flat", font=("TkDefaultFont", 10))
        scroll = ttk.Scrollbar(body, command=text.yview)
        text.config(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        text.pack(side="left", fill="both", expand=True)
        text.tag_configure("head", font=("TkDefaultFont", 10, "bold"))
        text.tag_configure("bad", foreground="#c33")
        text.tag_configure("good", foreground="#2a7")
        for c in report.crafts:
            text.insert("end", c.name + "\n", "head")
            for line in c.lines:
                text.insert("end", f"  {line}\n")
            for off, what, _tip in c.problems:
                text.insert("end", f"  ✗ {what} (−{off})\n", "bad")
            text.insert("end", "\n")
        if report.firsts:
            text.insert("end", "Science firsts\n", "head")
            for f in report.firsts:
                text.insert("end", f"  {short_date(f.et)}  {f.text}  +{f.points:,.0f}\n", "good")
            text.insert("end", "\n")
        text.insert("end", "How to do better\n", "head")
        if report.tips:
            for t in report.tips:
                text.insert("end", f"  • {t}\n")
        else:
            text.insert("end", "  Nothing went wrong: a clean mission.\n", "good")
        text.config(state="disabled")
        ttk.Button(self, text="Close", command=self.destroy).pack(pady=8)
