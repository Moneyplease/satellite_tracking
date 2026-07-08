"use client";

import { useState } from "react";

// Talk to the backend on the SAME host the page was opened from
// (works whether you use http://localhost:3000 or http://<LAN-IP>:3000)
const API =
  typeof window !== "undefined"
    ? `http://${window.location.hostname}:8000`
    : "http://localhost:8000";

export default function Home() {
  const [frameA, setFrameA] = useState(null);
  const [frameB, setFrameB] = useState(null);
  const [busy, setBusy] = useState(false);
  const [log, setLog] = useState("");
  const [tri, setTri] = useState(null);
  const [images, setImages] = useState([]);

  const [s, setS] = useState({
    length: "60",
    nsigma: "4.0",
    min_elong: "3.0",
    angle: "",
    border: "10",
    sat_border: "60",
    stationA: "18.5745, 98.4847, 2400",
    stationB: "16.7607, 102.6309, 185",
    time: "2025-03-28 14:58:29.795",
    api_key: "",
    tle1: "",
    tle2: "",
    do_solve: true,
    do_triangulate: true,
    do_tle: false,
  });

  const set = (k, v) => setS((p) => ({ ...p, [k]: v }));

  async function run() {
    if (!frameA) {
      alert("Pick Frame A first.");
      return;
    }
    setBusy(true);
    setLog("Running…");
    setTri(null);
    setImages([]);
    const fd = new FormData();
    fd.append("frameA", frameA);
    if (frameB) fd.append("frameB", frameB);
    fd.append("settings", JSON.stringify(s));
    try {
      const res = await fetch(`${API}/run`, { method: "POST", body: fd });
      const data = await res.json();
      setLog(data.log || "(no output)");
      setTri(data.triangulation || null);
      setImages(data.images || []);
    } catch (e) {
      setLog("Request failed — is the FastAPI backend running on :8000?\n" + e);
    } finally {
      setBusy(false);
    }
  }

  const field = (k, label) => (
    <label style={{ display: "block", marginBottom: 8 }}>
      <span style={{ display: "block", fontSize: 12, color: "#555" }}>{label}</span>
      <input
        value={s[k]}
        onChange={(e) => set(k, e.target.value)}
        style={{ width: "100%", padding: 6, boxSizing: "border-box" }}
      />
    </label>
  );

  return (
    <main style={{ display: "flex", gap: 24, padding: 24, fontFamily: "system-ui" }}>
      {/* left: settings */}
      <div style={{ width: 360, flexShrink: 0 }}>
        <h2>Satellite Tracking</h2>

        <fieldset style={{ marginBottom: 12 }}>
          <legend>Frames</legend>
          <div style={{ marginBottom: 8 }}>
            <span style={{ fontSize: 12, color: "#555" }}>Frame A</span>
            <input type="file" accept=".fit,.fits,.fts" onChange={(e) => setFrameA(e.target.files?.[0] ?? null)} />
          </div>
          <div>
            <span style={{ fontSize: 12, color: "#555" }}>Frame B</span>
            <input type="file" accept=".fit,.fits,.fts" onChange={(e) => setFrameB(e.target.files?.[0] ?? null)} />
          </div>
        </fieldset>

        <fieldset style={{ marginBottom: 12 }}>
          <legend>Detection</legend>
          {field("length", "LENGTH_PX")}
          {field("nsigma", "NSIGMA")}
          {field("min_elong", "MIN_ELONG")}
          {field("angle", "ANGLE_DEG (blank = auto)")}
          {field("border", "BORDER_MARGIN")}
          {field("sat_border", "SAT_BORDER_MARGIN")}
        </fieldset>

        <fieldset style={{ marginBottom: 12 }}>
          <legend>Stations & time</legend>
          {field("stationA", "Station A (lat, lon, elev_m)")}
          {field("stationB", "Station B (lat, lon, elev_m)")}
          {field("time", "TIME_UTC (YYYY-MM-DD HH:MM:SS)")}
        </fieldset>

        <fieldset style={{ marginBottom: 12 }}>
          <legend>Options</legend>
          <label style={{ marginRight: 12 }}>
            <input type="checkbox" checked={s.do_solve} onChange={(e) => set("do_solve", e.target.checked)} /> Plate solve
          </label>
          <label style={{ marginRight: 12 }}>
            <input type="checkbox" checked={s.do_triangulate} onChange={(e) => set("do_triangulate", e.target.checked)} /> Triangulate
          </label>
          <label>
            <input type="checkbox" checked={s.do_tle} onChange={(e) => set("do_tle", e.target.checked)} /> TLE check
          </label>
          {field("api_key", "astrometry API key")}
          {field("tle1", "TLE line 1")}
          {field("tle2", "TLE line 2")}
        </fieldset>

        <button type="button" onClick={run} disabled={busy} style={{ width: "100%", padding: 10, fontSize: 16 }}>
          {busy ? "Running…" : "Run"}
        </button>
      </div>

      {/* right: results */}
      <div style={{ flexGrow: 1 }}>
        {tri && (
          <div style={{ background: "#f2fff5", border: "1px solid #bde5c8", padding: 12, marginBottom: 12 }}>
            <b>Triangulation</b>
            <div>Altitude: {tri.altitude_km} km</div>
            <div>Sub-satellite: lat {tri.lat}, lon {tri.lon}</div>
            <div>ECEF XYZ (km): [{tri.ecef_km.join(", ")}]</div>
            <div>Geocentric distance: {tri.geocentric_km} km</div>
            <div>Slant A / B: {tri.slant_km.join(" / ")} km</div>
            <div>Ray miss distance: {tri.miss_km} km</div>
          </div>
        )}

        <div style={{ display: "flex", gap: 16, flexWrap: "wrap", marginBottom: 12 }}>
          {images.map((name) => (
            // eslint-disable-next-line @next/next/no-img-element
            <img key={name} src={`${API}/png/${name}`} alt={name} style={{ width: 420, border: "1px solid #ccc" }} />
          ))}
        </div>

        <pre style={{ background: "#111", color: "#0f0", padding: 12, height: 320, overflow: "auto", whiteSpace: "pre-wrap" }}>
          {log}
        </pre>
      </div>
    </main>
  );
}