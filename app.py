"""Football Analysis — Streamlit app.

Upload a match clip (or use the bundled demo), run the full YOLO + tracking
pipeline in one click, and get:
  1. Annotated output video (players, referees, ball, team colors, speed/distance)
  2. BEFORE / AFTER side-by-side comparison clip (written during the same pass)
  3. Match stats (ball control %, per-player speed & distance) + downloads

The pipeline streams frames (never holds the whole clip in RAM) and caches
results on disk, so re-runs with the same settings are instant.

Run locally:
    pip install -r requirements.txt
    streamlit run app.py

Deploy to Streamlit Community Cloud:
    1. Push this folder to GitHub (keep `model/best.pt` OUT of git — see README).
    2. Go to share.streamlit.io → New app → select repo/branch → main file `app.py`.
    3. In the app sidebar, upload your `best.pt` OR set `MODEL_URL` secret to
       auto-download it at runtime. Then upload a clip and click Run.
"""
import hashlib
import os
import pickle
import shutil
import tempfile
import time
import urllib.request
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import pipeline

st.set_page_config(
    page_title="Football Analysis — YOLO Tracking + Stats",
    page_icon="⚽",
    layout="wide",
)

DEMO_CANDIDATES = ["08fd33_4.mp4", "input_videos/08fd33_3.mp4", "assets/demo.mp4"]
DEFAULT_MODEL = "model/best.pt"
APP_DIR = Path(__file__).parent
CACHE_DIR = Path(tempfile.gettempdir()) / "football_analysis_cache"
RUNS_DIR = CACHE_DIR / "runs"        # one folder of results per (video, model, settings)
UPLOADS_DIR = CACHE_DIR / "uploads"  # uploaded videos / weights, named by content hash
RUNS_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
KEEP_RUNS, KEEP_UPLOADS = 8, 6


def find_demo():
    for c in DEMO_CANDIDATES:
        if (APP_DIR / c).exists():
            return str(APP_DIR / c)
    return None


def ensure_model(model_path: str) -> str:
    """Return a usable .pt path — download from MODEL_URL secret/env if needed."""
    if os.path.exists(model_path):
        return model_path
    try:
        url = st.secrets.get("MODEL_URL", "")
    except Exception:  # no secrets file configured
        url = ""
    url = url or os.environ.get("MODEL_URL", "")
    if url:
        os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
        with st.spinner("Downloading detection model (one-time)…"):
            # download beside the target so an interrupted transfer never
            # leaves a truncated file that looks like valid weights
            part = model_path + ".part"
            urllib.request.urlretrieve(url, part)
            os.replace(part, model_path)
    return model_path


def file_key(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


def prune(directory: Path, keep: int):
    """Delete all but the `keep` most recently used entries of a cache folder."""
    entries = sorted(directory.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    for p in entries[keep:]:
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        else:
            p.unlink(missing_ok=True)


def save_upload(up, slot: str):
    """Persist an uploaded file once per upload, not once per rerun.

    Returns (path, content hash). `slot` names the session-state entry that
    remembers which upload is already on disk.
    """
    ident = (up.name, up.size, getattr(up, "file_id", None))
    saved = st.session_state.get(slot)
    if saved and saved["ident"] == ident and os.path.exists(saved["path"]):
        return saved["path"], saved["sig"]
    data = up.getvalue()
    sig = hashlib.sha1(data).hexdigest()[:16]
    path = UPLOADS_DIR / f"{sig}{Path(up.name).suffix}"
    if path.exists():
        os.utime(path)  # mark as recently used
    else:
        path.write_bytes(data)
        prune(UPLOADS_DIR, KEEP_UPLOADS)
    st.session_state[slot] = {"ident": ident, "path": str(path), "sig": sig}
    return str(path), sig


def run_key(video_sig: str, model_sig: str, conf, max_seconds, cam, spd, device) -> str:
    raw = f"{video_sig}|{model_sig}|{conf}|{max_seconds}|{cam}|{spd}|{device}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def summarise(tracks, team_ball_control):
    ctrl = {}
    valid = team_ball_control[team_ball_control > 0]
    for t in (1, 2):
        ctrl[f"Team {t}"] = float((valid == t).mean() * 100) if len(valid) else 0.0
    rows = []
    for pid in sorted({pid for f in tracks["players"] for pid in f}):
        speeds, dists, team = [], [], None
        for f in tracks["players"]:
            if pid in f:
                if "speed" in f[pid]:
                    speeds.append(f[pid]["speed"])
                if "distance" in f[pid]:
                    dists.append(f[pid]["distance"])
                team = f[pid].get("team", team)
        if speeds or dists:
            rows.append({
                "player_id": pid,
                "team": team,
                "max_speed_kmh": round(max(speeds), 2) if speeds else 0.0,
                "avg_speed_kmh": round(float(np.mean(speeds)), 2) if speeds else 0.0,
                "distance_m": round(max(dists), 2) if dists else 0.0,
            })
    return ctrl, pd.DataFrame(rows).sort_values(["team", "player_id"]) if rows else pd.DataFrame()


def show_results(cdir: Path, elapsed: float):
    with open(cdir / "summary.pkl", "rb") as f:
        ctrl, df, n_frames = pickle.load(f)
    out_mp4, cmp_mp4, poster_path = cdir / "analyzed.mp4", cdir / "before_after.mp4", cdir / "poster.jpg"

    if elapsed:
        st.success(f"Done — {n_frames} annotated frames in {elapsed:.0f}s "
                   f"({n_frames / elapsed:.1f} frames/s). Re-runs are cached.")
    else:
        st.success(f"Done — {n_frames} annotated frames (from cache).")
    c1, c2 = st.columns(2)
    c1.metric("Team 1 ball control", f"{ctrl.get('Team 1', 0):.1f}%")
    c2.metric("Team 2 ball control", f"{ctrl.get('Team 2', 0):.1f}%")

    st.subheader("🎬 Analyzed video")
    st.video(str(out_mp4))
    st.subheader("↔️ BEFORE / AFTER")
    st.video(str(cmp_mp4))
    st.caption("Left = original, right = full analysis overlay. Use the ⤓ buttons to download either clip.")

    if not df.empty:
        st.subheader("📊 Per-player speed & distance")
        st.dataframe(df, use_container_width=True, hide_index=True,
                     column_config={
                         "player_id": st.column_config.NumberColumn("Player ID"),
                         "team": st.column_config.NumberColumn("Team"),
                         "max_speed_kmh": st.column_config.NumberColumn("Max speed (km/h)"),
                         "avg_speed_kmh": st.column_config.NumberColumn("Avg speed (km/h)"),
                         "distance_m": st.column_config.NumberColumn("Distance (m)"),
                     })
        dist_chart = alt.Chart(df).mark_bar().encode(
            x=alt.X("player_id:O", title="Player ID"),
            y=alt.Y("distance_m:Q", title="Distance covered (m)"),
            color=alt.Color("team:N", title="Team"),
            tooltip=["player_id", "team", "max_speed_kmh",
                     "avg_speed_kmh", "distance_m"],
        ).properties(title="Distance covered per player", height=350)
        st.altair_chart(dist_chart, use_container_width=True)
        st.download_button("⬇️ Download stats (CSV)", df.to_csv(index=False), "player_stats.csv", "text/csv")

    with open(out_mp4, "rb") as f:
        st.download_button("⬇️ Download analyzed video (mp4)", f, "analyzed.mp4", "video/mp4")
    with open(cmp_mp4, "rb") as f:
        st.download_button("⬇️ Download BEFORE/AFTER clip (mp4)", f, "before_after.mp4", "video/mp4")
    with open(poster_path, "rb") as f:
        st.download_button("⬇️ Download comparison poster (jpg)", f,
                           "before_after_poster.jpg", "image/jpeg")


# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("⚙️ Settings")
    model_path = st.text_input("Model weights (.pt)", value=DEFAULT_MODEL,
                               help="Custom football YOLO model (detects player / referee / ball / goalkeeper).")
    uploaded_model = st.file_uploader("…or upload best.pt", type=["pt"])
    conf = st.slider("Detection confidence", 0.05, 0.5, 0.1, 0.05)
    max_seconds = st.slider("Max clip length (s)", 5, 60, 20, 5,
                            help="Trimmed from the start. RAM stays flat (streaming) — longer just takes longer.")
    try:
        import torch
        has_cuda = torch.cuda.is_available()
    except Exception:
        has_cuda = False
    device = st.selectbox("Compute device", ["auto"] + (["cuda", "cpu"] if has_cuda else ["cpu"]),
                          help="Local GPU (if any) or CPU. Streamlit Cloud is CPU-only.")
    enable_camera = st.toggle("Camera-movement compensation", value=True)
    enable_speed = st.toggle("Speed / distance overlay", value=True)
    st.divider()
    st.caption("☁️ Deploying? Set a `MODEL_URL` secret (link to best.pt) so Streamlit Cloud can download the weights at runtime — no large files in git.")

st.title("⚽ Football Analysis — detection, tracking & stats")
st.markdown("YOLO tracking → team assignment (K-Means) → ball possession → perspective transform → speed/distance. "
            "Run a clip below and you automatically get the **annotated video + BEFORE/AFTER comparison + stats**.")

demo_path = find_demo()
source = st.radio("Video source", ["Upload a clip", "Use demo clip"] if demo_path else ["Upload a clip"],
                  horizontal=True)
input_path, video_sig = None, None

if source == "Upload a clip":
    up = st.file_uploader("Upload match video (mp4/avi/mov)", type=["mp4", "avi", "mov", "mkv"])
    if up:
        input_path, video_sig = save_upload(up, "video_upload")
        st.caption("ℹ️ Speed and distance are calibrated to the demo clip's camera angle. On other "
                   "footage, tracking, teams and possession still work, but treat km/h and metres "
                   "as rough until `view_transformer` is recalibrated (see README).")
else:
    input_path = demo_path
    if st.session_state.get("demo_sig", (None,))[0] != demo_path:
        st.session_state["demo_sig"] = (demo_path, file_key(demo_path))
    video_sig = st.session_state["demo_sig"][1]
    st.info(f"Using bundled demo: `{demo_path}` ({pipeline.probe_video(demo_path)[1]} frames).")

if input_path:
    st.video(input_path)
    run = st.button("🚀 Run analysis", type="primary", use_container_width=True)

    if run:
        st.session_state.pop("result", None)
        if uploaded_model:
            model_path, model_sig = save_upload(uploaded_model, "model_upload")
        else:
            model_path = ensure_model(model_path)
            if not os.path.exists(model_path):
                st.error(f"Model not found at `{model_path}`. Upload `best.pt` in the sidebar, train one via "
                         f"`development_and_analysis/training/`, or set a `MODEL_URL` secret. See README → *Model weights*.")
                st.stop()
            stt = os.stat(model_path)
            model_sig = f"{os.path.abspath(model_path)}|{stt.st_size}|{stt.st_mtime}"

        dev = None if device == "auto" else device
        key = run_key(video_sig, model_sig, conf, max_seconds, enable_camera, enable_speed, device)
        cdir = RUNS_DIR / key

        try:
            elapsed = 0.0
            if (cdir / "DONE").exists():
                os.utime(cdir)  # mark as recently used
            else:
                shutil.rmtree(cdir, ignore_errors=True)
                cdir.mkdir(parents=True, exist_ok=True)
                bar = st.progress(0, text="Starting…")

                def cb(frac, msg):
                    bar.progress(min(1.0, frac), text=msg)

                t0 = time.perf_counter()
                # stub_dir=None: this run folder already is the cache
                r = pipeline.run(input_path, model_path, str(cdir / "analyzed.mp4"), conf=conf,
                                 max_seconds=max_seconds, enable_camera=enable_camera,
                                 enable_speed=enable_speed, device=dev,
                                 stub_dir=None, progress_cb=cb,
                                 compare_path=str(cdir / "before_after.mp4"),
                                 poster_path=str(cdir / "poster.jpg"))
                ctrl, df = summarise(r["tracks"], r["control"])
                with open(cdir / "summary.pkl", "wb") as f:
                    pickle.dump((ctrl, df, r["n_frames"]), f)
                (cdir / "DONE").touch()
                elapsed = time.perf_counter() - t0
                bar.empty()
                prune(RUNS_DIR, KEEP_RUNS)
            # Kept in session state so results survive the rerun that every
            # widget interaction (including the download buttons) triggers.
            st.session_state["result"] = {"dir": str(cdir), "video_sig": video_sig, "elapsed": elapsed}
        except Exception as e:
            st.exception(e)
            st.error("Analysis failed — most common cause is a COCO-pretrained model (e.g. yolov8n.pt) which lacks "
                     "the football classes (player/referee/ball/goalkeeper). Use your trained `best.pt`.")

    result = st.session_state.get("result")
    if result and result["video_sig"] == video_sig and (Path(result["dir"]) / "DONE").exists():
        show_results(Path(result["dir"]), result["elapsed"])
else:
    st.warning("Upload a clip or add the demo video to get started.")
