"""CLI entry: full football analysis via the streaming pipeline (low RAM).

Example:
    python main.py --input 08fd33_4.mp4 --model model/best.pt \
        --out output_videos/output_video.mp4 --max-seconds 0
    python make_before_after.py --input 08fd33_4.mp4 \
        --analyzed output_videos/output_video.mp4
"""
import argparse
import time

import pipeline


def main():
    ap = argparse.ArgumentParser(description="Football analysis (streaming, low-RAM).")
    ap.add_argument("--input", default="08fd33_4.mp4")
    ap.add_argument("--model", default="model/best.pt")
    ap.add_argument("--out", default="output_videos/output_video.mp4")
    ap.add_argument("--conf", type=float, default=0.1)
    ap.add_argument("--batch-size", type=int, default=20)
    ap.add_argument("--max-seconds", type=float, default=0.0,
                    help="Trim to first N seconds (0 = full video)")
    ap.add_argument("--no-camera", action="store_true")
    ap.add_argument("--no-speed", action="store_true")
    ap.add_argument("--device", default=None,
                    help="e.g. cpu, 0. Default: ultralytics auto (CUDA if present)")
    ap.add_argument("--stub-dir", default="stubs")
    args = ap.parse_args()

    t0 = time.perf_counter()

    def cb(frac, msg):
        print(f"[{frac * 100:5.1f}%] {msg}", flush=True)

    r = pipeline.run(args.input, args.model, args.out, conf=args.conf,
                     batch_size=args.batch_size, max_seconds=args.max_seconds,
                     enable_camera=not args.no_camera, enable_speed=not args.no_speed,
                     device=args.device, stub_dir=args.stub_dir, progress_cb=cb)
    dt = time.perf_counter() - t0
    print(f"Wrote {r['output_path']}  n={r['n_frames']}  "
          f"Team1={r['team1_pct']:.1f}% Team2={r['team2_pct']:.1f}%  in {dt:.1f}s")


if __name__ == '__main__':
    main()
