"""Streaming, low-RAM football analysis pipeline.

Same stages and same math as main.py — detection + ByteTrack, camera-motion
compensation, perspective transform, ball interpolation, speed/distance, team
assignment, possession — but frames are never all held in RAM:

* detection streams in batches; only per-frame track dicts accumulate (KBs)
* camera movement streams frame-to-frame; only motion vectors accumulate
* drawing + H.264 encode happen in ONE pass, one frame at a time (single copy)
* BEFORE/AFTER comparison streams pairwise off disk

Peak RAM ~= one batch of frames + tracks + model, regardless of clip length.
With identical inputs this produces pixel-identical output to main.py.
"""
import os
import pickle

import cv2
import imageio.v2 as imageio
import numpy as np
import supervision as sv

from camera_movement_estimator import CameraMovementEstimator
from player_ball_assigner import PlayerBallAssigner
from speed_and_distance_estimator import SpeedAndDistance_Estimator
from team_assigner import TeamAssigner
from trackers import Tracker
from view_transformer import ViewTransformer


# ---------------- video helpers ----------------

def probe_video(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    return float(fps), n, w, h


def read_first_frame(path):
    cap = cv2.VideoCapture(path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise ValueError(f"No readable frames: {path}")
    return frame


def iter_batches(path, batch_size, max_frames=None):
    """Yield lists of ≤batch_size BGR frames, in order."""
    cap = cv2.VideoCapture(path)
    batch = []
    count = 0
    while True:
        if max_frames and count >= max_frames:
            break
        ret, frame = cap.read()
        if not ret:
            break
        batch.append(frame)
        count += 1
        if len(batch) >= batch_size:
            yield batch
            batch = []
    cap.release()
    if batch:
        yield batch


# ---------------- stages ----------------

def stream_tracks(video_path, model_path, conf=0.1, batch_size=20, verbose=False,
                  device=None, max_frames=None, stub_path=None, progress_cb=None):
    """Detect + ByteTrack without ever holding all frames.

    Same conversion (ultralytics -> supervision), same goalkeeper->player
    remap, same ByteTrack update order as Tracker.get_object_tracks.
    Returns (tracks, first_frame). Stub format matches get_object_tracks.
    """
    if stub_path and os.path.exists(stub_path):
        with open(stub_path, 'rb') as f:
            tracks = pickle.load(f)
        if progress_cb:
            progress_cb(0.30, "Loaded tracks from stub cache…")
        return tracks, read_first_frame(video_path)

    tracker = Tracker(model_path, conf=conf, batch_size=batch_size, verbose=verbose,
                      device=device)
    names_inv = None
    tracks = {"players": [], "referees": [], "ball": []}
    first_frame = None
    done = 0

    for batch in iter_batches(video_path, batch_size, max_frames):
        if first_frame is None:
            first_frame = batch[0]
        detections = tracker.model.predict(batch, conf=conf, verbose=verbose, device=device)
        for frame in batch:  # keep ref alive while its Results are converted
            det = detections.pop(0)
            if names_inv is None:
                names_inv = {v: k for k, v in det.names.items()}
            dsv = sv.Detections.from_ultralytics(det)
            for oi, cid in enumerate(dsv.class_id):
                if det.names[cid] == "goalkeeper":
                    dsv.class_id[oi] = names_inv["player"]
            dwt = tracker.tracker.update_with_detections(dsv)
            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})
            fn = len(tracks["players"]) - 1
            for fd in dwt:
                bbox = fd[0].tolist()
                cls_id = fd[3]
                tid = fd[4]
                if cls_id == names_inv['player']:
                    tracks["players"][fn][tid] = {"bbox": bbox}
                if cls_id == names_inv['referee']:
                    tracks["referees"][fn][tid] = {"bbox": bbox}
            for fd in dsv:
                bbox = fd[0].tolist()
                cls_id = fd[3]
                if cls_id == names_inv['ball']:
                    tracks["ball"][fn][1] = {"bbox": bbox}
            done += 1
            if progress_cb and done % 25 == 0:
                progress_cb(0.05 + 0.25 * done / (max_frames or max(done, 1)),
                            f"Detecting + tracking… {done} frames")
        del batch, detections

    if stub_path:
        os.makedirs(os.path.dirname(os.path.abspath(stub_path)), exist_ok=True)
        with open(stub_path, 'wb') as f:
            pickle.dump(tracks, f)
    tracker.add_position_to_tracks(tracks)
    return tracks, first_frame


def stream_camera_movement(video_path, first_frame, n_frames,
                           stub_path=None, progress_cb=None):
    """Optical-flow camera vectors, streamed frame-to-frame.

    Same estimator, same per-frame math as get_camera_movement.
    """
    if stub_path and os.path.exists(stub_path):
        with open(stub_path, 'rb') as f:
            return pickle.load(f)
    est = CameraMovementEstimator(first_frame)
    movement = [[0, 0]] * n_frames
    cap = cv2.VideoCapture(video_path)
    ret, _ = cap.read()  # frame 0 already consumed for init
    old_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    old_features = cv2.goodFeaturesToTrack(old_gray, **est.features)
    from utils import measure_distance, measure_xy_distance
    for fn in range(1, n_frames):
        ret, frame = cap.read()
        if not ret:
            break
        fg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        new_f, _, _ = cv2.calcOpticalFlowPyrLK(old_gray, fg, old_features, None, **est.lk_params)
        best, mx, my = 0, 0, 0
        for new, old in zip(new_f, old_features):
            d = measure_distance(new.ravel(), old.ravel())
            if d > best:
                best = d
                mx, my = measure_xy_distance(old.ravel(), new.ravel())
        if best > est.minimum_distance:
            movement[fn] = [mx, my]
            old_features = cv2.goodFeaturesToTrack(fg, **est.features)
        old_gray = fg.copy()
        if progress_cb and fn % 50 == 0:
            progress_cb(0.35 + 0.15 * fn / n_frames, f"Camera motion… {fn}/{n_frames}")
    cap.release()
    if stub_path:
        os.makedirs(os.path.dirname(os.path.abspath(stub_path)), exist_ok=True)
        with open(stub_path, 'wb') as f:
            pickle.dump(movement, f)
    return movement


def assign_possession(tracks):
    pa = PlayerBallAssigner()
    control = []
    for fn, ptrack in enumerate(tracks['players']):
        ball = tracks['ball'][fn].get(1, {})
        if "bbox" not in ball:
            control.append(control[-1] if control else 0)
            continue
        a = pa.assign_ball_to_player(ptrack, ball['bbox'])
        if a != -1:
            tracks['players'][fn][a]['has_ball'] = True
            control.append(tracks['players'][fn][a].get('team', 0))
        else:
            control.append(control[-1] if control else 0)
    return np.array(control)


def open_writer(path, fps):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    return imageio.get_writer(path, fps=fps, codec="libx264", quality=8,
                              macro_block_size=2,
                              ffmpeg_params=["-preset", "veryfast", "-tune", "zerolatency"])


def run(input_path, model_path, output_path, conf=0.1, batch_size=20,
        max_seconds=0, enable_camera=True, enable_speed=True, device=None,
        stub_dir="stubs", progress_cb=None, verbose=False):
    """Full pipeline, streaming. Returns dict with tracks/control/stats/paths."""
    def cb(frac, msg):
        if progress_cb:
            progress_cb(frac, msg)

    src_fps, src_n, _, _ = probe_video(input_path)
    max_frames = int(max_seconds * src_fps) if max_seconds and max_seconds > 0 else None
    if src_n and max_frames:
        max_frames = min(max_frames, src_n)
    t_stub = os.path.join(stub_dir, "track_stubs.pkl") if stub_dir else None
    c_stub = os.path.join(stub_dir, "camera_movement_stub.pkl") if stub_dir else None

    cb(0.02, "Detecting + tracking…")
    tracks, frame0 = stream_tracks(input_path, model_path, conf=conf,
                                   batch_size=batch_size, verbose=verbose,
                                   device=device,
                                   max_frames=max_frames, stub_path=t_stub,
                                   progress_cb=progress_cb)
    n = len(tracks["players"])
    # Drawing + interpolation helpers need no model weights: bare instance.
    tracker = Tracker.__new__(Tracker)

    if enable_camera:
        cb(0.35, "Estimating camera movement…")
        cam = stream_camera_movement(input_path, frame0, n, stub_path=c_stub,
                                     progress_cb=progress_cb)
        cme = CameraMovementEstimator(frame0)
        cme.add_adjust_positions_to_tracks(tracks, cam)
    else:
        cme, cam = None, [[0, 0]] * n
        for obj, ot in tracks.items():
            for tr in ot:
                for info in tr.values():
                    info["position_adjusted"] = info.get("position")

    cb(0.55, "Perspective transform…")
    ViewTransformer().add_transformed_position_to_tracks(tracks)
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    if enable_speed:
        cb(0.60, "Speed + distance…")
        sde = SpeedAndDistance_Estimator()
        sde.add_speed_and_distance_to_tracks(tracks)
    else:
        sde = None

    cb(0.65, "Team assignment…")
    ta = TeamAssigner()
    if tracks["players"][0]:
        ta.assign_team_color(frame0, tracks["players"][0])

    cb(0.70, "Possession + drawing + encode (single pass)…")
    # possession needs teams; teams resolve per-frame during the draw pass below.
    # First pass over tracks for possession requires team labels, so assign teams
    # frame-by-frame while streaming, then possession needs full control array
    # BEFORE drawing. Two cheap streaming reads; frames themselves stay on disk.
    if tracks["players"][0]:
        cap = cv2.VideoCapture(input_path)
        fn = 0
        while fn < n:
            ret, frame = cap.read()
            if not ret:
                break
            for pid, tr in tracks["players"][fn].items():
                if "team" not in tr:
                    t = ta.get_player_team(frame, tr["bbox"], pid)
                    tr["team"] = t
                    tr["team_color"] = ta.team_colors[t]
            fn += 1
        cap.release()
    control = assign_possession(tracks)

    writer = open_writer(output_path, fps=24.0)
    cap = cv2.VideoCapture(input_path)
    fn = 0
    try:
        while fn < n:
            ret, frame = cap.read()
            if not ret:
                break
            f = tracker.annotate_frame(frame, tracks, control, fn)
            if cme is not None:
                f = cme.overlay_frame(f, cam[fn])
            if sde is not None:
                f = sde.overlay_frame(f, tracks, fn)
            writer.append_data(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
            fn += 1
            if progress_cb and fn % 25 == 0:
                progress_cb(0.70 + 0.28 * fn / n, f"Drawing… {fn}/{n}")
    finally:
        cap.release()
        writer.close()

    cb(1.0, "Done.")
    valid = control[control > 0]
    return {
        "tracks": tracks, "control": control, "fps": src_fps, "n_frames": n,
        "output_path": output_path,
        "team1_pct": float((valid == 1).mean() * 100) if len(valid) else 0.0,
        "team2_pct": float((valid == 2).mean() * 100) if len(valid) else 0.0,
    }


def add_label(frame, text):
    labelled = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = 1.2, 3
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    pad = 16
    cv2.rectangle(labelled, (pad, pad), (pad + tw + 20, pad + th + 24), (0, 0, 0), cv2.FILLED)
    cv2.putText(labelled, text, (pad + 10, pad + th + 14), font, scale,
                (255, 255, 255), thickness, cv2.LINE_AA)
    return labelled


def make_before_after_stream(input_path, analyzed_path, out_path, poster_path=None,
                             max_seconds=0, width=640, fps=24.0, progress_cb=None):
    """Side-by-side BEFORE/AFTER written frame-by-frame (no frame lists in RAM)."""
    cap_in, cap_an = cv2.VideoCapture(input_path), cv2.VideoCapture(analyzed_path)
    if not cap_in.isOpened():
        raise FileNotFoundError(input_path)
    if not cap_an.isOpened():
        raise FileNotFoundError(analyzed_path)
    max_n = int(max_seconds * fps) if max_seconds and max_seconds > 0 else None
    ret_in, f_in = cap_in.read()
    ret_an, f_an = cap_an.read()
    if not ret_in or not ret_an:
        raise ValueError("One of the videos has no readable frames.")
    h_raw, w_raw = f_an.shape[:2]
    half_h = int(h_raw * width / w_raw)
    half_h -= half_h % 2
    writer = open_writer(out_path, fps=fps)
    mid_poster, count = None, 0
    try:
        while ret_in and ret_an and (max_n is None or count < max_n):
            b = cv2.resize(f_in, (width, half_h))
            a = cv2.resize(f_an, (width, half_h))
            combo = np.hstack([add_label(b, "BEFORE"), add_label(a, "AFTER")])
            writer.append_data(cv2.cvtColor(combo, cv2.COLOR_BGR2RGB))
            count += 1
            ret_in, f_in = cap_in.read()
            ret_an, f_an = cap_an.read()
            if progress_cb and count % 50 == 0:
                progress_cb(count / (max_n or 1), f"Comparison clip… {count} frames")
    finally:
        cap_in.release()
        cap_an.release()
        writer.close()
    if poster_path:
        # re-read middle frame pair for the poster (cheap, no RAM spike)
        mid = count // 2
        cap_in, cap_an = cv2.VideoCapture(input_path), cv2.VideoCapture(analyzed_path)
        cap_in.set(cv2.CAP_PROP_POS_FRAMES, mid)
        cap_an.set(cv2.CAP_PROP_POS_FRAMES, mid)
        _, f_in = cap_in.read()
        _, f_an = cap_an.read()
        cap_in.release()
        cap_an.release()
        if f_in is not None and f_an is not None:
            b = cv2.resize(f_in, (width, half_h))
            a = cv2.resize(f_an, (width, half_h))
            os.makedirs(os.path.dirname(os.path.abspath(poster_path)) or ".", exist_ok=True)
            cv2.imwrite(poster_path, np.hstack([add_label(b, "BEFORE"), add_label(a, "AFTER")]))
            mid_poster = poster_path
    return out_path, mid_poster, count
