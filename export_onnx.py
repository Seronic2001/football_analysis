"""Export the finetuned weights to ONNX (EXPERIMENTAL, opt-in).

Measured on the reference machine (8-core CPU): the ONNX/OpenVINO CPU runtime
was ~10x SLOWER than PyTorch CPU for this particular model, so the pipeline
does NOT auto-use .onnx — best.pt remains the default. Keep this script for
hardware where your local benchmark proves a win (e.g. some ARM/edge chips).

Usage:
    python export_onnx.py [--model model/best.pt] [--out model/best.onnx]
    # then pass --model model/best.onnx explicitly to benchmark it
"""
import argparse
import os

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser(description="Export best.pt to ONNX (CPU fast path).")
    ap.add_argument("--model", default="model/best.pt")
    ap.add_argument("--out", default="model/best.onnx")
    ap.add_argument("--imgsz", type=int, default=640,
                    help="Must match training size; changing it alters accuracy.")
    args = ap.parse_args()

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Weights not found: {args.model}")
    if os.path.exists(args.out):
        print(f"Exists, reusing: {args.out}")
        print("Delete it first to re-export.")
        return args.out
    m = YOLO(args.model)
    exported = m.export(format="onnx", imgsz=args.imgsz, dynamic=True, simplify=True)
    # export() writes next to the weights as best.onnx; move if --out differs
    if os.path.abspath(exported) != os.path.abspath(args.out):
        os.replace(exported, args.out)
    print(f"Exported: {args.out}")
    print("pipeline will now prefer it automatically over best.pt")
    return args.out


if __name__ == "__main__":
    main()
