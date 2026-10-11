import os

import cv2
import imageio.v2 as imageio
import numpy as np


def probe_video(path):
    """Return (fps, frame_count, width, height). Count/size are 0 if unknown."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    return float(fps), n, w, h


def iter_frames(path, max_frames=None):
    """Yield BGR frames in order, one at a time."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")
    try:
        count = 0
        while max_frames is None or count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            yield frame
            count += 1
    finally:
        cap.release()


def iter_batches(path, batch_size, max_frames=None):
    """Yield lists of ≤batch_size BGR frames, in order."""
    batch = []
    for frame in iter_frames(path, max_frames):
        batch.append(frame)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def open_writer(path, fps):
    """H.264 mp4 writer (plays in browsers / Streamlit / GitHub). Takes RGB frames."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    return imageio.get_writer(path, fps=fps, codec="libx264", quality=8,
                              macro_block_size=2,
                              ffmpeg_params=["-preset", "veryfast", "-tune", "zerolatency"])


def draw_translucent_box(frame, x1, y1, x2, y2, alpha):
    """Blend a white box into frame in place, touching only the box's pixels."""
    roi = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
    if roi.size:
        roi[:] = cv2.addWeighted(np.full_like(roi, 255), alpha, roi, 1 - alpha, 0)


def half_height(frame_width, frame_height, width):
    """Height of one comparison half at the given width, kept even for H.264."""
    h = int(frame_height * width / frame_width)
    return h - h % 2


def labelled_half(frame, text, width, height):
    """Resized copy of frame with a corner label (one half of a BEFORE/AFTER clip)."""
    half = cv2.resize(frame, (width, height))
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = 1.2, 3
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    pad = 16
    cv2.rectangle(half, (pad, pad), (pad + tw + 20, pad + th + 24), (0, 0, 0), cv2.FILLED)
    cv2.putText(half, text, (pad + 10, pad + th + 14), font, scale,
                (255, 255, 255), thickness, cv2.LINE_AA)
    return half


def make_before_after_stream(input_path, analyzed_path, out_path, poster_path=None,
                             max_seconds=0, width=640, fps=None):
    """Side-by-side BEFORE/AFTER from two existing videos, frame-by-frame.

    fps defaults to the input video's. Returns (out_path, poster_path or None, frames).
    """
    src_fps, n_in, _, _ = probe_video(input_path)
    _, n_an, _, _ = probe_video(analyzed_path)
    fps = fps or src_fps
    max_n = int(max_seconds * fps) if max_seconds and max_seconds > 0 else None
    known = [c for c in (n_in, n_an, max_n) if c]
    poster_at = min(known) // 2 if known else 0
    half_h, poster, count = None, None, 0
    writer = open_writer(out_path, fps=fps)
    try:
        for f_in, f_an in zip(iter_frames(input_path, max_n), iter_frames(analyzed_path, max_n)):
            if half_h is None:
                h_raw, w_raw = f_an.shape[:2]
                half_h = half_height(w_raw, h_raw, width)
            combo = np.hstack([labelled_half(f_in, "BEFORE", width, half_h),
                               labelled_half(f_an, "AFTER", width, half_h)])
            writer.append_data(cv2.cvtColor(combo, cv2.COLOR_BGR2RGB))
            if poster_path and count == poster_at:
                os.makedirs(os.path.dirname(os.path.abspath(poster_path)), exist_ok=True)
                cv2.imwrite(poster_path, combo)
                poster = poster_path
            count += 1
    finally:
        writer.close()
    if count == 0:
        raise ValueError("One of the videos has no readable frames.")
    return out_path, poster, count
