"""Build a BEFORE / AFTER comparison clip from an input video and its analyzed output.

Side-by-side layout with labels, H.264 MP4 output (plays in browsers, GitHub
README embeds and Streamlit) plus a poster JPG taken from the middle of the clip.

Usage:
    python make_before_after.py \
        --input 08fd33_4.mp4 \
        --analyzed output_videos/output_video.mp4 \
        --out assets/before_after.mp4 \
        --poster assets/before_after_poster.jpg \
        --max-seconds 20
"""
import argparse
import os

import cv2
import imageio.v2 as imageio
import numpy as np


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


def add_label(frame, text):
    labelled = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = 1.2, 3
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    pad = 16
    cv2.rectangle(labelled, (pad, pad), (pad + tw + 20, pad + th + 24), (0, 0, 0), cv2.FILLED)
    cv2.putText(labelled, text, (pad + 10, pad + th + 14), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return labelled


def make_comparison_frames(before_frames, after_frames, width=960):
    """Stack BEFORE/AFTER halves side-by-side with labels. Returns BGR frames."""
    n = min(len(before_frames), len(after_frames))
    before_frames, after_frames = before_frames[:n], after_frames[:n]
    h_raw, w_raw = after_frames[0].shape[:2]
    half_h = int(h_raw * width / w_raw)
    half_h -= half_h % 2  # keep even for H.264
    combo = []
    for b, a in zip(before_frames, after_frames):
        b_half = cv2.resize(b, (width, half_h))
        a_half = cv2.resize(a, (width, half_h))
        combo.append(np.hstack([add_label(b_half, "BEFORE"), add_label(a_half, "AFTER")]))
    return combo


def write_mp4_bgr(frames_bgr, path, fps=24.0):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with imageio.get_writer(path, fps=fps, codec="libx264", quality=8,
                            macro_block_size=2) as writer:
        for frame in frames_bgr:
            writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def main():
    parser = argparse.ArgumentParser(description="Make a BEFORE/AFTER comparison clip.")
    parser.add_argument("--input", default="08fd33_4.mp4", help="Original match video")
    parser.add_argument("--analyzed", default="output_videos/output_video.mp4", help="Analyzed output video")
    parser.add_argument("--out", default="assets/before_after.mp4", help="Output comparison clip (mp4)")
    parser.add_argument("--poster", default="assets/before_after_poster.jpg", help="Output poster image (jpg)")
    parser.add_argument("--max-seconds", type=float, default=20.0, help="Trim clip to first N seconds (0 = full length)")
    parser.add_argument("--width", type=int, default=960, help="Width of each half of the comparison")
    parser.add_argument("--fps", type=float, default=24.0, help="Output frame rate")
    args = parser.parse_args()

    before_frames, _ = read_frames_capped(args.input)
    after_frames, _ = read_frames_capped(args.analyzed)
    if not before_frames or not after_frames:
        raise ValueError("One of the videos has no readable frames.")

    n = min(len(before_frames), len(after_frames))
    if args.max_seconds and args.max_seconds > 0:
        n = min(n, int(args.max_seconds * args.fps))
    before_frames, after_frames = before_frames[:n], after_frames[:n]
    print(f"Using {n} frames ({n / args.fps:.1f}s @ {args.fps}fps)")

    combo_frames_bgr = make_comparison_frames(before_frames, after_frames, width=args.width)

    write_mp4_bgr(combo_frames_bgr, args.out, fps=args.fps)
    print(f"Wrote {args.out}")

    poster_dir = os.path.dirname(os.path.abspath(args.poster))
    if poster_dir:
        os.makedirs(poster_dir, exist_ok=True)
    cv2.imwrite(args.poster, combo_frames_bgr[len(combo_frames_bgr) // 2])
    print(f"Wrote {args.poster}")


if __name__ == "__main__":
    main()
