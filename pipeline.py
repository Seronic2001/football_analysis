"""Streaming, low-RAM football analysis pipeline.

Stages: detection + ByteTrack, camera-motion compensation, team assignment,
perspective transform, ball interpolation, speed/distance, possession.
Frames are never all held in RAM, and the video is decoded only twice:

* analysis pass — detection in batches, camera motion and team colours on the
  same frames; only per-frame track dicts and motion vectors accumulate (KBs)
* drawing pass — overlays + H.264 encode one frame at a time, and the
  BEFORE/AFTER comparison is written from the same frames

Peak RAM ~= one batch of frames + tracks + model, regardless of clip length.
"""
import contextlib
import hashlib
import os
import pickle

import cv2
import numpy as np

from camera_movement_estimator import CameraMovementEstimator
from player_ball_assigner import PlayerBallAssigner
from speed_and_distance_estimator import SpeedAndDistance_Estimator
from team_assigner import TeamAssigner
from trackers import Tracker, add_position_to_tracks, interpolate_ball_positions, annotate_frame
from utils import (probe_video, iter_frames, iter_batches, open_writer,
                   half_height, labelled_half)
from view_transformer import ViewTransformer


# ---------------- stages ----------------

def analysis_stub_path(stub_dir, video_path, model_path, conf, max_frames, enable_camera):
    """Stub file name tied to everything the analysis pass depends on, so a
    different clip, model or setting never reuses another run's tracks."""
    parts = []
    for p in (video_path, model_path):
        st = os.stat(p)
        parts += [os.path.abspath(p), st.st_size, st.st_mtime]
    parts += [conf, max_frames, enable_camera]
    sig = hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:12]
    return os.path.join(stub_dir, f"analysis_{sig}.pkl")


def analyze(video_path, model_path, conf=0.1, batch_size=20, verbose=False,
            device=None, max_frames=None, total_frames=None, enable_camera=True,
            stub_path=None, progress_cb=None):
    """Single decode pass: detect + ByteTrack, camera motion, team colours.

    Returns (tracks, camera_movement). tracks is {"players"|"referees"|"ball":
    [per-frame {track_id: info}]}; camera_movement is one [x, y] per frame
    (zeros when enable_camera is off).
    """
    if stub_path and os.path.exists(stub_path):
        with open(stub_path, 'rb') as f:
            result = pickle.load(f)
        if progress_cb:
            progress_cb(0.50, "Loaded analysis from stub cache…")
        return result

    tracker = Tracker(model_path, conf=conf, verbose=verbose, device=device)
    team_assigner = TeamAssigner()
    cme = None
    tracks = {"players": [], "referees": [], "ball": []}
    camera_movement = []

    for batch in iter_batches(video_path, batch_size, max_frames):
        for frame, (players, referees, ball) in zip(batch, tracker.track_batch(batch)):
            tracks["players"].append(players)
            tracks["referees"].append(referees)
            tracks["ball"].append(ball)

            if not enable_camera:
                camera_movement.append([0.0, 0.0])
            elif cme is None:
                cme = CameraMovementEstimator(frame)
                camera_movement.append([0.0, 0.0])
            else:
                camera_movement.append(cme.update(frame))

            # Team colours are fitted on the first frame showing two or more
            # players; each player is then classified where they first appear.
            if team_assigner.kmeans is None:
                team_assigner.assign_team_color(frame, players)
            for player_id, track in players.items():
                team = team_assigner.get_player_team(frame, track["bbox"], player_id)
                if team is not None:
                    track["team"] = team
                    track["team_color"] = team_assigner.team_colors[team]

            done = len(camera_movement)
            if progress_cb and done % 25 == 0:
                frac = min(1.0, done / total_frames) if total_frames else 0.0
                progress_cb(0.02 + 0.48 * frac, f"Detecting + tracking… {done} frames")

    add_position_to_tracks(tracks)
    if stub_path:
        os.makedirs(os.path.dirname(os.path.abspath(stub_path)), exist_ok=True)
        with open(stub_path, 'wb') as f:
            pickle.dump((tracks, camera_movement), f)
    return tracks, camera_movement


def assign_possession(tracks):
    """Mark the player on the ball each frame; returns the controlling team per
    frame (0 until any team has had the ball)."""
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


def possession_share(control):
    """Per-frame cumulative (team 1, team 2) control fractions, shape (n, 2).
    Both are 0 until a team has had the ball."""
    team_1 = np.cumsum(control == 1)
    team_2 = np.cumsum(control == 2)
    total = np.maximum(team_1 + team_2, 1)
    return np.stack([team_1 / total, team_2 / total], axis=1)


def run(input_path, model_path, output_path, conf=0.1, batch_size=20,
        max_seconds=0, enable_camera=True, enable_speed=True, device=None,
        stub_dir="stubs", progress_cb=None, verbose=False,
        compare_path=None, poster_path=None, compare_width=640):
    """Full pipeline, streaming. Returns dict with tracks/control/stats/paths.

    compare_path / poster_path optionally write a BEFORE/AFTER side-by-side
    clip and its middle frame during the same drawing pass.
    """
    def cb(frac, msg):
        if progress_cb:
            progress_cb(frac, msg)

    fps, src_n, width, height = probe_video(input_path)
    max_frames = int(max_seconds * fps) if max_seconds and max_seconds > 0 else None
    if src_n and max_frames:
        max_frames = min(max_frames, src_n)
    stub_path = (analysis_stub_path(stub_dir, input_path, model_path, conf,
                                    max_frames, enable_camera) if stub_dir else None)

    cb(0.02, "Detecting + tracking…")
    tracks, cam = analyze(input_path, model_path, conf=conf, batch_size=batch_size,
                          verbose=verbose, device=device, max_frames=max_frames,
                          total_frames=max_frames or src_n,
                          enable_camera=enable_camera, stub_path=stub_path,
                          progress_cb=progress_cb)
    n = len(tracks["players"])
    if n == 0:
        raise ValueError(f"No readable frames: {input_path}")

    cb(0.55, "Perspective transform…")
    CameraMovementEstimator.add_adjust_positions_to_tracks(tracks, cam)
    ViewTransformer(width or 1920, height or 1080).add_transformed_position_to_tracks(tracks)
    tracks["ball"] = interpolate_ball_positions(tracks["ball"])

    if enable_speed:
        cb(0.60, "Speed + distance…")
        sde = SpeedAndDistance_Estimator(frame_rate=fps)
        sde.add_speed_and_distance_to_tracks(tracks)
    else:
        sde = None

    control = assign_possession(tracks)
    share = possession_share(control)

    cb(0.65, "Drawing + encode…")
    comparing = bool(compare_path or poster_path)
    with contextlib.ExitStack() as stack:
        writer = stack.enter_context(open_writer(output_path, fps=fps))
        cmp_writer = (stack.enter_context(open_writer(compare_path, fps=fps))
                      if compare_path else None)
        for fn, frame in enumerate(iter_frames(input_path, n)):
            if comparing:
                half_h = half_height(frame.shape[1], frame.shape[0], compare_width)
                before = labelled_half(frame, "BEFORE", compare_width, half_h)
            annotate_frame(frame, tracks, share[fn], fn)
            if enable_camera:
                CameraMovementEstimator.overlay_frame(frame, cam[fn])
            if sde is not None:
                sde.overlay_frame(frame, tracks, fn)
            writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if comparing:
                combo = np.hstack([before, labelled_half(frame, "AFTER", compare_width, half_h)])
                if cmp_writer is not None:
                    cmp_writer.append_data(cv2.cvtColor(combo, cv2.COLOR_BGR2RGB))
                if poster_path and fn == n // 2:
                    os.makedirs(os.path.dirname(os.path.abspath(poster_path)), exist_ok=True)
                    cv2.imwrite(poster_path, combo)
            if progress_cb and (fn + 1) % 25 == 0:
                progress_cb(0.65 + 0.33 * (fn + 1) / n, f"Drawing… {fn + 1}/{n}")

    cb(1.0, "Done.")
    valid = control[control > 0]
    return {
        "tracks": tracks, "control": control, "fps": fps, "n_frames": n,
        "output_path": output_path, "compare_path": compare_path,
        "poster_path": poster_path,
        "team1_pct": float((valid == 1).mean() * 100) if len(valid) else 0.0,
        "team2_pct": float((valid == 2).mean() * 100) if len(valid) else 0.0,
    }
