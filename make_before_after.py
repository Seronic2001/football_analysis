"""Build a BEFORE / AFTER comparison clip from an input video and its analyzed output.

Side-by-side layout with labels, H.264 MP4 output (plays in browsers, GitHub
README embeds and Streamlit) plus a poster JPG taken from the middle of the clip.
`main.py --compare` produces the same clip during analysis; use this script to
rebuild one from two existing videos.

Usage:
    python make_before_after.py \
        --input 08fd33_4.mp4 \
        --analyzed output_videos/output_video.mp4 \
        --out assets/before_after.mp4 \
        --poster assets/before_after_poster.jpg \
        --max-seconds 20
"""
import argparse

from utils import make_before_after_stream


def main():
    parser = argparse.ArgumentParser(description="Make a BEFORE/AFTER comparison clip.")
    parser.add_argument("--input", default="08fd33_4.mp4", help="Original match video")
    parser.add_argument("--analyzed", default="output_videos/output_video.mp4", help="Analyzed output video")
    parser.add_argument("--out", default="assets/before_after.mp4", help="Output comparison clip (mp4)")
    parser.add_argument("--poster", default="assets/before_after_poster.jpg", help="Output poster image (jpg)")
    parser.add_argument("--max-seconds", type=float, default=20.0, help="Trim clip to first N seconds (0 = full length)")
    parser.add_argument("--width", type=int, default=960, help="Width of each half of the comparison")
    parser.add_argument("--fps", type=float, default=None, help="Output frame rate (default: the input video's)")
    args = parser.parse_args()

    out, poster, n = make_before_after_stream(
        args.input, args.analyzed, args.out, args.poster,
        max_seconds=args.max_seconds, width=args.width, fps=args.fps)
    print(f"Wrote {out} ({n} frames)")
    if poster:
        print(f"Wrote {poster}")


if __name__ == "__main__":
    main()
