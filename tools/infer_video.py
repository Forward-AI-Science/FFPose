"""Run pose estimation frame-by-frame on a video file.

Usage:
    python tools/infer_video.py \
        --family rtmpose --variant body-coco-m \
        --checkpoint /path/to/rtmpose-m.pth \
        --video input.mp4 --out output.mp4

Reads with cv2.VideoCapture, writes with cv2.VideoWriter (mp4v codec by default).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

import os as _os
_ffpose_home = _os.environ.get("FFPOSE_HOME") or str(
    Path(__file__).resolve().parent.parent
)
sys.path.insert(0, _ffpose_home)

import FFPose


_FAMILY_CHOICES = ["rtmpose", "vitpose", "hrnet", "swin", "hrformer", "litehrnet"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--family", required=True, choices=_FAMILY_CHOICES)
    p.add_argument("--variant", required=True,
                   help="See tools/infer_image.py for the variant naming.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--video", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--score-thr", type=float, default=0.3)
    p.add_argument("--det-thr", type=float, default=0.5)
    p.add_argument("--max-frames", type=int, default=0,
                   help="Stop after this many frames (0 = all).")
    p.add_argument("--fourcc", default="mp4v",
                   help="cv2 fourcc 4-letter code for the output codec.")
    p.add_argument("--no-viz", action="store_true",
                   help="Just count detections; do not write video.")
    args = p.parse_args()

    pipe = FFPose.TopDownPosePipeline.from_pretrained(
        args.variant, args.checkpoint, family=args.family,
        device=args.device, det_score_thr=args.det_thr,
    )

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"could not open {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or -1

    writer = None
    if not args.no_viz:
        fourcc = cv2.VideoWriter_fourcc(*args.fourcc)
        writer = cv2.VideoWriter(args.out, fourcc, fps, (width, height))
        if not writer.isOpened():
            raise SystemExit(f"could not open writer for {args.out}")

    schema = FFPose.skeletons.COCO_17
    if "wholebody" in args.variant:
        schema = FFPose.skeletons.COCO_WHOLEBODY_133

    t0 = time.time()
    frames_processed = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        out = pipe.predict_image(frame)
        if writer is not None and out.boxes.shape[0] > 0:
            frame = FFPose.draw_skeletons(
                frame, out.keypoints, out.keypoint_scores,
                schema=schema, boxes=out.boxes,
                score_thr=args.score_thr,
            )
        if writer is not None:
            writer.write(frame)
        frames_processed += 1
        if frames_processed % 30 == 0:
            elapsed = time.time() - t0
            eta = (total - frames_processed) * elapsed / max(frames_processed, 1) if total > 0 else 0
            print(f"[frame {frames_processed}/{total if total > 0 else '?'}] "
                  f"persons={out.boxes.shape[0]}  fps={frames_processed/elapsed:.1f}  "
                  f"eta={eta:.0f}s")
        if args.max_frames and frames_processed >= args.max_frames:
            break

    cap.release()
    if writer is not None:
        writer.release()
    elapsed = time.time() - t0
    print(f"done. {frames_processed} frames in {elapsed:.1f}s "
          f"({frames_processed/max(elapsed,1):.1f} fps)")


if __name__ == "__main__":
    main()
