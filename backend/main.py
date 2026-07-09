#!/usr/bin/env python3
"""
FastAPI backend that wraps satellite_traking.py.

Copy your satellite_traking.py into this backend/ folder (next to this file),
then run:
    pip install fastapi "uvicorn[standard]" python-multipart
    uvicorn main:app --reload --port 8000

The Next.js frontend posts two FITS files + settings to /run and gets back the
log, triangulation results, and a URL to the annotated PNG.
"""

import os
import io
import sys
import json
import tempfile
import traceback
from contextlib import redirect_stdout

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse

import satellite_tracking as pipe   # <-- your unchanged pipeline

DEFAULT_API_KEY = os.environ.get("ASTROMETRY_API_KEY", "")

app = FastAPI(title="Satellite Tracking API")

# allow the Next.js dev server (localhost:3000) to call us
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # dev: allow any origin (localhost or LAN IP)
    allow_methods=["*"], allow_headers=["*"],
)

OUT_DIR = os.path.join(tempfile.gettempdir(), "sattrack_out")
os.makedirs(OUT_DIR, exist_ok=True)


def _save_upload(upload: UploadFile, folder: str) -> str:
    path = os.path.join(folder, upload.filename)
    with open(path, "wb") as f:
        f.write(upload.file.read())
    return path


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/png/{name}")
def get_png(name: str):
    """Serve an annotated PNG produced by a run."""
    path = os.path.join(OUT_DIR, name)
    if not os.path.isfile(path):
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(path, media_type="image/png")


@app.post("/run")
async def run(
    frameA: UploadFile = File(...),
    frameB: UploadFile = File(None),
    settings: str = Form(...),          # JSON string of the form values
):
    cfg = json.loads(settings)
    workdir = tempfile.mkdtemp(prefix="sattrack_")
    log_buf = io.StringIO()
    result = {"log": "", "triangulation": None, "images": []}

    try:
        # push settings into the pipeline module (no file edits)
        pipe.PNG_DIR = OUT_DIR
        pipe.LENGTH_PX = int(float(cfg.get("length", 60)))
        pipe.NSIGMA = float(cfg.get("nsigma", 4.0))
        pipe.MIN_ELONG = float(cfg.get("min_elong", 3.0))
        pipe.ANGLE_DEG = (float(cfg["angle"]) if str(cfg.get("angle", "")).strip() else None)
        pipe.BORDER_MARGIN = int(float(cfg.get("border", 10)))
        pipe.SAT_BORDER_MARGIN = int(float(cfg.get("sat_border", 60)))
        pipe.DO_PLATE_SOLVE = bool(cfg.get("do_solve", True))
        pipe.ASTROMETRY_API_KEY = cfg.get("api_key", "") or DEFAULT_API_KEY

        pathA = _save_upload(frameA, workdir)
        pathB = _save_upload(frameB, workdir) if frameB is not None else None

        staA = tuple(float(v) for v in str(cfg["stationA"]).split(","))
        staB = tuple(float(v) for v in str(cfg["stationB"]).split(",")) if cfg.get("stationB") else None
        from datetime import datetime
        dt = datetime.fromisoformat(str(cfg["time"]).strip())
        tutc = (dt.year, dt.month, dt.day, dt.hour, dt.minute,
                dt.second + dt.microsecond / 1e6)

        with redirect_stdout(log_buf):
            print("=== Station A ===")
            radec_a = pipe.process_one(pathA)
            result["images"].append(os.path.basename(pathA.rsplit(".", 1)[0]) + "_stars.png")

            if cfg.get("do_triangulate") and pathB is not None:
                print("=== Station B ===")
                radec_b = pipe.process_one(pathB)
                result["images"].append(os.path.basename(pathB.rsplit(".", 1)[0]) + "_stars.png")
                if radec_a and radec_b:
                    r = pipe.triangulate(radec_a, staA, radec_b, staB, tutc)
                    print("\n=== Triangulation ===")
                    print(f"Satellite ECEF XYZ (km) : {r['xyz_ecef_km'].round(2)}")
                    print(f"Altitude above ellipsoid: {r['altitude_km']:.2f} km")
                    print(f"Sub-satellite point     : lat {r['lat_deg']:.4f}, lon {r['lon_deg']:.4f}")
                    print(f"Ray miss distance       : {r['miss_km']:.3f} km")
                    result["triangulation"] = {
                        "ecef_km": [round(float(v), 2) for v in r["xyz_ecef_km"]],
                        "altitude_km": round(r["altitude_km"], 2),
                        "lat": round(r["lat_deg"], 4), "lon": round(r["lon_deg"], 4),
                        "geocentric_km": round(r["geocentric_km"], 2),
                        "slant_km": [round(x, 1) for x in r["slant_km"]],
                        "miss_km": round(r["miss_km"], 3),
                    }
                else:
                    print("Need a solved satellite RA/Dec in BOTH frames.")

            if cfg.get("do_tle") and str(cfg.get("tle1", "")).strip():
                print("\n=== TLE cross-check ===")
                pipe.tle_check_frame("A", pathA, staA, radec_a, cfg["tle1"], cfg["tle2"])

            print("\nDone.")
    except Exception:
        print("\n[error]\n" + traceback.format_exc(), file=log_buf)

    result["log"] = log_buf.getvalue()
    return JSONResponse(result)