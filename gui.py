#!/usr/bin/env python3
"""
GUI for satellite_traking.py — pick frames, set parameters, run the pipeline,
and view the log, triangulation results, and the annotated image.

Put this file in the SAME folder as satellite_traking.py, then:
    python satellite_gui.py

Needs: tkinter (bundled with Python). Optional: pip install pillow  (image preview)
"""
import os
import sys
import queue
import threading
import traceback
from datetime import datetime

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

try:
    from PIL import Image, ImageTk
    HAVE_PIL = True
except Exception:
    HAVE_PIL = False


def _tuple_time(s):
    """'2025-03-28 14:58:29.795' -> (Y, M, D, h, m, s.sss)."""
    dt = datetime.fromisoformat(s.strip())
    return (dt.year, dt.month, dt.day, dt.hour, dt.minute,
            dt.second + dt.microsecond / 1e6)


class QueueWriter:
    """Redirects print() from the worker thread into the log queue."""
    def __init__(self, q): self.q = q
    def write(self, text):
        if text:
            self.q.put(text)
    def flush(self): pass


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Satellite Tracking")
        self.geometry("1180x760")
        self.log_q = queue.Queue()
        self._img_ref = None
        self._build()
        self.after(80, self._drain_log)

    # ---------- layout ----------
    def _build(self):
        left = ttk.Frame(self, padding=8); left.pack(side="left", fill="y")
        right = ttk.Frame(self, padding=8); right.pack(side="right", fill="both", expand=True)

        # Files
        f = ttk.LabelFrame(left, text="Frames", padding=6); f.pack(fill="x", pady=4)
        self.fileA = self._file_row(f, "Frame A (station A):", 0)
        self.fileB = self._file_row(f, "Frame B (station B):", 1)
        self.pngdir = self._entry_row(f, "Output folder (PNG):", 2,
                                      "/Users/none/internship/png_output")

        # Detection
        d = ttk.LabelFrame(left, text="Detection", padding=6); d.pack(fill="x", pady=4)
        self.length = self._entry_row(d, "LENGTH_PX", 0, "60")
        self.nsigma = self._entry_row(d, "NSIGMA", 1, "4.0")
        self.min_elong = self._entry_row(d, "MIN_ELONG", 2, "3.0")
        self.angle = self._entry_row(d, "ANGLE_DEG (blank=auto)", 3, "")
        self.border = self._entry_row(d, "BORDER_MARGIN", 4, "10")
        self.sat_border = self._entry_row(d, "SAT_BORDER_MARGIN", 5, "60")

        # Stations + time
        s = ttk.LabelFrame(left, text="Stations (lat, lon, elev_m) & time", padding=6)
        s.pack(fill="x", pady=4)
        self.staA = self._entry_row(s, "Station A", 0, "18.5745, 98.4847, 2400")
        self.staB = self._entry_row(s, "Station B", 1, "16.7607, 102.6309, 185")
        self.time = self._entry_row(s, "TIME_UTC", 2, "2025-03-28 14:58:29.795")

        # Options
        o = ttk.LabelFrame(left, text="Options", padding=6); o.pack(fill="x", pady=4)
        self.do_solve = tk.BooleanVar(value=True)
        self.do_tri = tk.BooleanVar(value=True)
        self.do_tle = tk.BooleanVar(value=False)
        ttk.Checkbutton(o, text="Plate solve", variable=self.do_solve).grid(row=0, column=0, sticky="w")
        ttk.Checkbutton(o, text="Triangulate", variable=self.do_tri).grid(row=0, column=1, sticky="w")
        ttk.Checkbutton(o, text="TLE check", variable=self.do_tle).grid(row=0, column=2, sticky="w")
        self.apikey = self._entry_row(o, "astrometry API key", 1, "")
        self.tle1 = self._entry_row(o, "TLE line 1", 2, "")
        self.tle2 = self._entry_row(o, "TLE line 2", 3, "")

        self.run_btn = ttk.Button(left, text="Run", command=self._on_run)
        self.run_btn.pack(fill="x", pady=8)

        # Right: results + log + image
        self.result = ttk.Label(right, text="Results will appear here.",
                                 justify="left", font=("TkDefaultFont", 10, "bold"))
        self.result.pack(fill="x")
        nb = ttk.Notebook(right); nb.pack(fill="both", expand=True, pady=6)

        logframe = ttk.Frame(nb); nb.add(logframe, text="Log")
        self.log = tk.Text(logframe, wrap="word", height=20)
        sb = ttk.Scrollbar(logframe, command=self.log.yview); self.log["yscrollcommand"] = sb.set
        self.log.pack(side="left", fill="both", expand=True); sb.pack(side="right", fill="y")

        imgframe = ttk.Frame(nb); nb.add(imgframe, text="Image")
        btns = ttk.Frame(imgframe); btns.pack(fill="x")
        ttk.Button(btns, text="View A", command=lambda: self._show_png(self.fileA.get())).pack(side="left")
        ttk.Button(btns, text="View B", command=lambda: self._show_png(self.fileB.get())).pack(side="left")
        self.canvas = ttk.Label(imgframe, text=("(image preview — needs `pip install pillow`)"
                                                 if not HAVE_PIL else "Run, then View A/B"))
        self.canvas.pack(fill="both", expand=True)

    def _file_row(self, parent, label, row):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w")
        var = tk.StringVar()
        ttk.Entry(parent, textvariable=var, width=40).grid(row=row, column=1, sticky="we")
        ttk.Button(parent, text="…", width=3,
                   command=lambda: var.set(filedialog.askopenfilename(
                       filetypes=[("FITS", "*.fit *.fits *.fts"), ("All", "*.*")]))
                   ).grid(row=row, column=2)
        parent.columnconfigure(1, weight=1)
        return var

    def _entry_row(self, parent, label, row, default=""):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w")
        var = tk.StringVar(value=default)
        ttk.Entry(parent, textvariable=var, width=40).grid(row=row, column=1, columnspan=2, sticky="we")
        parent.columnconfigure(1, weight=1)
        return var

    # ---------- run ----------
    def _on_run(self):
        self.run_btn.config(state="disabled")
        self.log.delete("1.0", "end")
        self.result.config(text="Running…")
        threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        old = sys.stdout
        sys.stdout = QueueWriter(self.log_q)
        try:
            import satellite_traking as pipe

            # push GUI values into the pipeline module's globals
            pipe.PNG_DIR = self.pngdir.get().strip()
            pipe.LENGTH_PX = int(float(self.length.get()))
            pipe.NSIGMA = float(self.nsigma.get())
            pipe.MIN_ELONG = float(self.min_elong.get())
            pipe.ANGLE_DEG = (float(self.angle.get()) if self.angle.get().strip() else None)
            pipe.BORDER_MARGIN = int(float(self.border.get()))
            pipe.SAT_BORDER_MARGIN = int(float(self.sat_border.get()))
            pipe.DO_PLATE_SOLVE = self.do_solve.get()
            pipe.ASTROMETRY_API_KEY = self.apikey.get().strip()

            staA = tuple(float(v) for v in self.staA.get().split(","))
            staB = tuple(float(v) for v in self.staB.get().split(","))
            tutc = _tuple_time(self.time.get())

            os.makedirs(pipe.PNG_DIR, exist_ok=True)
            frameA, frameB = self.fileA.get().strip(), self.fileB.get().strip()
            if not os.path.isfile(frameA):
                raise SystemExit("Frame A is not a valid file.")

            print("=== Station A ===")
            radec_a = pipe.process_one(frameA)

            summary = ""
            if self.do_tri.get():
                if not os.path.isfile(frameB):
                    raise SystemExit("Triangulation needs a valid Frame B.")
                print("=== Station B ===")
                radec_b = pipe.process_one(frameB)
                if radec_a is None or radec_b is None:
                    raise SystemExit("Need a solved RA/Dec (satellite) in BOTH frames.")
                r = pipe.triangulate(radec_a, staA, radec_b, staB, tutc)
                print("\n=== Triangulation ===")
                print(f"Satellite ECEF XYZ (km) : {r['xyz_ecef_km'].round(2)}")
                print(f"Altitude above ellipsoid: {r['altitude_km']:.2f} km")
                print(f"Sub-satellite point     : lat {r['lat_deg']:.4f}, lon {r['lon_deg']:.4f}")
                print(f"Geocentric distance     : {r['geocentric_km']:.2f} km")
                print(f"Slant range A / B       : {r['slant_km'][0]:.1f} / {r['slant_km'][1]:.1f} km")
                print(f"Ray miss distance       : {r['miss_km']:.3f} km")
                summary = (f"Alt {r['altitude_km']:.1f} km | "
                           f"sub ({r['lat_deg']:.3f}, {r['lon_deg']:.3f}) | "
                           f"miss {r['miss_km']:.2f} km")
                if self.do_tle.get() and self.tle1.get().strip():
                    print("\n=== TLE cross-check ===")
                    pipe.tle_check_frame("A", frameA, staA, radec_a, self.tle1.get(), self.tle2.get())
                    pipe.tle_check_frame("B", frameB, staB, radec_b, self.tle1.get(), self.tle2.get())
            elif self.do_tle.get() and self.tle1.get().strip():
                print("\n=== TLE cross-check ===")
                pipe.tle_check_frame("A", frameA, staA, radec_a, self.tle1.get(), self.tle2.get())

            print("\nDone.")
            self.log_q.put(("__RESULT__", summary or "Done — see log."))
            self.log_q.put(("__IMAGE__", frameA))
        except SystemExit as e:
            print(f"\n[stopped] {e}")
            self.log_q.put(("__RESULT__", f"Stopped: {e}"))
        except Exception:
            print("\n[error]\n" + traceback.format_exc())
            self.log_q.put(("__RESULT__", "Error — see log."))
        finally:
            sys.stdout = old
            self.log_q.put(("__DONE__", None))

    # ---------- log/image pump ----------
    def _drain_log(self):
        try:
            while True:
                item = self.log_q.get_nowait()
                if isinstance(item, tuple):
                    tag, payload = item
                    if tag == "__RESULT__":
                        self.result.config(text=payload)
                    elif tag == "__IMAGE__":
                        self._show_png(payload)
                    elif tag == "__DONE__":
                        self.run_btn.config(state="normal")
                else:
                    self.log.insert("end", item); self.log.see("end")
        except queue.Empty:
            pass
        self.after(80, self._drain_log)

    def _show_png(self, fits_path):
        if not HAVE_PIL or not fits_path:
            return
        stem = os.path.basename(fits_path.rsplit(".", 1)[0])
        png = os.path.join(self.pngdir.get().strip(), stem + "_stars.png")
        if not os.path.isfile(png):
            self.canvas.config(text=f"(no PNG yet: {png})", image="")
            return
        img = Image.open(png)
        img.thumbnail((640, 640))
        self._img_ref = ImageTk.PhotoImage(img)
        self.canvas.config(image=self._img_ref, text="")


if __name__ == "__main__":
    App().mainloop()