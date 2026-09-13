"""Football Analysis — Streamlit app.

Upload a match clip (or use the bundled demo), run the full YOLO + tracking
pipeline in one click, and get:
  1. Annotated output video (players, referees, ball, team colors, speed/distance)
  2. BEFORE / AFTER side-by-side comparison clip (auto-generated after analysis)
  3. Match stats (ball control %, per-player speed & distance) + downloads

Run locally:
    pip install -r requirements.txt
    streamlit run app.py

Deploy to Streamlit Community Cloud:
    1. Push this folder to GitHub (keep `model/best.pt` OUT of git — see README).
    2. Go to share.streamlit.io → New app → select repo/branch → main file `app.py`.
    3. In the app sidebar, upload your `best.pt` OR set `MODEL_URL` secret to
       auto-download it at runtime. Then upload a clip and click Run.
"""
import os
import tempfile
import time
import urllib.request
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import pandas as pd
import streamlit as st

from camera_movement_estimator import CameraMovementEstimator
from make_before_after import make_comparison_frames, write_mp4_bgr
from player_ball_assigner import PlayerBallAssigner
from speed_and_distance_estimator import SpeedAndDistance_Estimator
from team_assigner import TeamAssigner
from trackers import Tracker
from view_transformer import ViewTransformer

st.set_page_config(
    page_title="Football Analysis — YOLO Tracking + Stats",
    page_icon="⚽",
    layout="wide",
)

DEMO_CANDIDATES = ["08fd33_4.mp4", "input_videos/08fd33_3.mp4", "assets/demo.mp4"]
DEFAULT_MODEL = "model/best.pt"
APP_DIR = Path(__file__).parent


def find_demo():
    for c in DEMO_CANDIDATES:
        if (APP_DIR / c).exists():
            return str(APP_DIR / c)
    return None


@st.cache_resource(show_spinner=False)
def load_tracker(model_path: str):
    return Tracker(model_path)


def ensure_model(model_path: str) -> str:
    """Return a usable .pt path — download from MODEL_URL secret/env if needed."""
    if os.path.exists(model_path):
        return model_path
    url = st.secrets.get("MODEL_URL", "") if hasattr(st, "secrets") else ""
    url = url or os.environ.get("MODEL_URL", "")
    if url:
        os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
        with st.spinner("Downloading detection model (one-time)…"):
            urllib.request.urlretrieve(url, model_path)
        return model_path
    return model_path


def read_frames_capped(path, max_frames=None):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        if max_frames and len(frames) >= max_frames:
            break
    cap.release()
    return frames, float(fps)


def save_bgr_mp4(frames_bgr, path, fps=24.0):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with imageio.get_writer(path, fps=fps, codec="libx264", quality=8,
                            macro_block_size=2) as w:
        for f in frames_bgr:
            w.append_data(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))


def run_pipeline(frames, model_path, progress_cb=None, enable_camera=True, enable_speed=True):
    """Mirror main.py but stub-free (Streamlit-safe) and robust to empty frames."""
    def cb(frac, msg):
        if progress_cb:
            progress_cb(frac, msg)

    cb(0.05, "Loading YOLO model…")
    tracker = load_tracker(model_path)

    cb(0.15, "Detecting + tracking players / referees / ball…")
    tracks = tracker.get_object_tracks(frames, read_from_stub=False, stub_path=None)
    tracker.add_position_to_tracks(tracks)

    if enable_camera:
        cb(0.45, "Estimating camera movement (optical flow)…")
        cme = CameraMovementEstimator(frames[0])
        cam_move = cme.get_camera_movement(frames, read_from_stub=False, stub_path=None)
        cme.add_adjust_positions_to_tracks(tracks, cam_move)
    else:
        cme, cam_move = None, [[0, 0]] * len(frames)
        # still need position_adjusted for the view transformer
        for obj, obj_tracks in tracks.items():
            for fn, tr in enumerate(obj_tracks):
                for tid, info in tr.items():
                    info["position_adjusted"] = info.get("position")

    cb(0.60, "Perspective transform → meters…")
    vt = ViewTransformer()
    vt.add_transformed_position_to_tracks(tracks)

    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    if enable_speed:
        cb(0.68, "Computing speed & distance…")
        sde = SpeedAndDistance_Estimator()
        sde.add_speed_and_distance_to_tracks(tracks)
    else:
        sde = None

    cb(0.75, "Assigning teams by shirt colour (K-Means)…")
    ta = TeamAssigner()
    if tracks["players"] and tracks["players"][0]:
        ta.assign_team_color(frames[0], tracks["players"][0])
        for fn, ptrack in enumerate(tracks["players"]):
            for pid, tr in ptrack.items():
                team = ta.get_player_team(frames[fn], tr["bbox"], pid)
                tracks["players"][fn][pid]["team"] = team
                tracks["players"][fn][pid]["team_color"] = ta.team_colors[team]
    else:
        st.warning("No players detected in frame 0 — team colours skipped. Try a lower confidence or a different clip.")

    cb(0.85, "Assigning ball possession…")
    pa = PlayerBallAssigner()
    team_ball_control = []
    for fn, ptrack in enumerate(tracks["players"]):
        ball = tracks["ball"][fn].get(1, {})
        if not ball or "bbox" not in ball:
            team_ball_control.append(team_ball_control[-1] if team_ball_control else 0)
            continue
        assigned = pa.assign_ball_to_player(ptrack, ball["bbox"])
        if assigned != -1:
            tracks["players"][fn][assigned]["has_ball"] = True
            team_ball_control.append(tracks["players"][fn][assigned].get("team", 0))
        else:
            team_ball_control.append(team_ball_control[-1] if team_ball_control else 0)
    team_ball_control = np.array(team_ball_control)

    cb(0.92, "Drawing annotations…")
    out = tracker.draw_annotations(frames, tracks, team_ball_control)
    if cme is not None:
        out = cme.draw_camera_movement(out, cam_move)
    if sde is not None:
        out = sde.draw_speed_and_distance(out, tracks)

    cb(1.0, "Done.")
    return out, tracks, np.array(team_ball_control)


def summarise(tracks, team_ball_control):
    ctrl = {}
    valid = team_ball_control[team_ball_control > 0]
    for t in (1, 2):
        ctrl[f"Team {t}"] = float((valid == t).mean() * 100) if len(valid) else 0.0
    rows = []
    for pid in sorted({pid for f in tracks["players"] for pid in f}):
        speeds, dists, team = [], 0.0, None
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


# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("⚙️ Settings")
    model_path = st.text_input("Model weights (.pt)", value=DEFAULT_MODEL,
                               help="Custom football YOLO model (detects player / referee / ball / goalkeeper).")
    uploaded_model = st.file_uploader("…or upload best.pt", type=["pt"])
    conf = st.slider("Detection confidence", 0.05, 0.5, 0.1, 0.05)
    max_seconds = st.slider("Max clip length (s)", 5, 60, 20, 5,
                            help="Longer clips = slower + more RAM. Trimmed from the start.")
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
input_path, fps_hint = None, 24.0

if source == "Upload a clip":
    up = st.file_uploader("Upload match video (mp4/avi/mov)", type=["mp4", "avi", "mov", "mkv"])
    if up:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(up.name).suffix) as t:
            t.write(up.read())
            input_path = t.name
else:
    input_path = demo_path
    st.info(f"Using bundled demo: `{demo_path}` ({cv2.VideoCapture(demo_path).get(7):.0f} frames).")

if input_path:
    st.video(input_path)
    run = st.button("🚀 Run analysis", type="primary", use_container_width=True)

    if run:
        # Resolve model
        if uploaded_model:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as t:
                t.write(uploaded_model.read())
                model_path = t.name
        else:
            model_path = ensure_model(model_path)

        if not os.path.exists(model_path):
            st.error(f"Model not found at `{model_path}`. Upload `best.pt` in the sidebar, train one via "
                     f"`development_and_analysis/training/`, or set a `MODEL_URL` secret. See README → *Model weights*.")
            st.stop()

        try:
            frames, fps = read_frames_capped(input_path, max_frames=int(max_seconds * 25))
            st.caption(f"Loaded {len(frames)} frames @ {fps:.1f} fps (trimmed to first {max_seconds}s).")
            if len(frames) < 5:
                st.error("Could not read enough frames from this video.")
                st.stop()

            bar = st.progress(0, text="Starting…")
            def cb(frac, msg):
                bar.progress(min(1.0, frac), text=msg)

            # apply confidence: Tracker hardcodes 0.1, so patch predict kwargs via model attribute
            tracker_probe = load_tracker(model_path)
            orig_detect = tracker_probe.detect_frames
            def detect_with_conf(frames_in, _conf=conf):
                batch, out = 20, []
                for i in range(0, len(frames_in), batch):
                    out += tracker_probe.model.predict(frames_in[i:i + batch], conf=_conf)
                return out
            tracker_probe.detect_frames = detect_with_conf

            try:
                out_frames, tracks, control = run_pipeline(
                    frames, model_path, progress_cb=cb,
                    enable_camera=enable_camera, enable_speed=enable_speed)
            finally:
                tracker_probe.detect_frames = orig_detect

            bar.empty()
            tmp = Path(tempfile.mkdtemp())
            out_mp4 = str(tmp / "analyzed.mp4")
            cmp_mp4 = str(tmp / "before_after.mp4")
            save_bgr_mp4(out_frames, out_mp4, fps=24.0)
            n = min(len(frames), len(out_frames))
            save_bgr_mp4(make_comparison_frames(frames[:n], out_frames[:n]), cmp_mp4, fps=24.0)
            poster = np.hstack([frames[n // 2], out_frames[n // 2]])
            _, poster_jpg = cv2.imencode(".jpg", poster)
            ctrl, df = summarise(tracks, control)

            st.success(f"Done in one pass — {len(out_frames)} annotated frames.")
            c1, c2 = st.columns(2)
            c1.metric("Team 1 ball control", f"{ctrl.get('Team 1', 0):.1f}%")
            c2.metric("Team 2 ball control", f"{ctrl.get('Team 2', 0):.1f}%")

            st.subheader("🎬 Analyzed video")
            st.video(out_mp4)
            st.subheader("↔️ BEFORE / AFTER (auto-generated after analysis)")
            st.video(cmp_mp4)
            st.caption("Left = original, right = full analysis overlay. Use the ⤓ buttons to download either clip.")

            if not df.empty:
                st.subheader("📊 Per-player speed & distance")
                st.dataframe(df, use_container_width=True)
                st.bar_chart(df.set_index("player_id")["distance_m"])
                st.download_button("⬇️ Download stats (CSV)", df.to_csv(index=False), "player_stats.csv", "text/csv")

            with open(out_mp4, "rb") as f:
                st.download_button("⬇️ Download analyzed video (mp4)", f, "analyzed.mp4", "video/mp4")
            with open(cmp_mp4, "rb") as f:
                st.download_button("⬇️ Download BEFORE/AFTER clip (mp4)", f, "before_after.mp4", "video/mp4")
            st.download_button("⬇️ Download comparison poster (jpg)", poster_jpg.tobytes(),
                               "before_after_poster.jpg", "image/jpeg")
        except Exception as e:
            st.exception(e)
            st.error("Analysis failed — most common cause is a COCO-pretrained model (e.g. yolov8n.pt) which lacks "
                     "the football classes (player/referee/ball/goalkeeper). Use your trained `best.pt`.")
else:
    st.warning("Upload a clip or add the demo video to get started.")
