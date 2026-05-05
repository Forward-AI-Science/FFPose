"""Run pose estimation on a single image (or a directory of images).

Usage:
    # single image, end-to-end (detector + pose)
    python tools/infer_image.py \
        --family rtmpose --variant body-coco-m \
        --checkpoint /path/to/rtmpose-m.pth \
        --image photo.jpg --out out.jpg

    # using your own bboxes (skip the detector)
    python tools/infer_image.py \
        --family hrnet --variant w32 \
        --checkpoint /path/to/hrnet-w32.pth \
        --image photo.jpg --bbox "x1,y1,x2,y2" --out out.jpg

    # batch process a folder
    python tools/infer_image.py \
        --family vitpose --variant small \
        --checkpoint /path/to/vitpose-s.pth \
        --image input_dir/ --out output_dir/
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

# allow `python tools/infer_image.py` from a fresh clone without pip install -e .
import os as _os
_ffpose_home = _os.environ.get("FFPOSE_HOME") or str(
    Path(__file__).resolve().parent.parent
)
sys.path.insert(0, _ffpose_home)

import FFPose


_FAMILY_CHOICES = ["rtmpose", "vitpose", "hrnet", "swin", "hrformer", "litehrnet"]
_IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def _build_pipeline(args) -> FFPose.TopDownPosePipeline | object:
    """Return either a full pipeline (detector + pose) or a bare pose inferencer."""
    if args.bbox:
        # bbox supplied, skip detector
        bare_inferencers = {
            "rtmpose":   FFPose.RTMPoseInferencer.from_pretrained,
            "vitpose":   FFPose.ViTPoseInferencer.from_pretrained,
            "hrnet":     FFPose.HRNetPoseInferencer.from_pretrained,
            "swin":      FFPose.SwinPoseInferencer.from_pretrained,
            "hrformer":  FFPose.HRFormerPoseInferencer.from_pretrained,
            "litehrnet": FFPose.LiteHRNetPoseInferencer.from_pretrained,
        }
        if args.family == "rtmpose":
            return bare_inferencers[args.family](
                args.variant, args.checkpoint, device=args.device,
            )
        return bare_inferencers[args.family](
            args.variant, args.checkpoint, device=args.device,
            flip_test=args.flip_test,
        )
    return FFPose.TopDownPosePipeline.from_pretrained(
        args.variant, args.checkpoint, family=args.family,
        device=args.device, det_score_thr=args.det_thr,
    )


def _list_images(p: Path) -> Iterable[Path]:
    if p.is_dir():
        for x in sorted(p.iterdir()):
            if x.suffix.lower() in _IMG_EXTS:
                yield x
    else:
        yield p


def _infer_one(pose, image: np.ndarray, bbox: np.ndarray | None) -> dict:
    """Returns a dict with boxes (N,4), keypoints (N,K,2), scores (N,K)."""
    if bbox is not None:
        res = pose.predict(image, bbox)
        return {
            "boxes": np.asarray([bbox], dtype=np.float32),
            "box_scores": np.array([1.0], dtype=np.float32),
            "keypoints": res.keypoints.astype(np.float32),  # (1, K, 2)
            "keypoint_scores": res.scores.astype(np.float32),
        }
    out = pose.predict_image(image)  # FullFrameResult
    return {
        "boxes": out.boxes,
        "box_scores": out.box_scores,
        "keypoints": out.keypoints,
        "keypoint_scores": out.keypoint_scores,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--family", required=True, choices=_FAMILY_CHOICES)
    p.add_argument("--variant", required=True,
                   help="rtmpose: body-coco-{t,s,m,l} | wholebody-coco-{m,l}; "
                        "vitpose: small / base / small-simple / base-simple; "
                        "hrnet: w32 / w48")
    p.add_argument("--checkpoint", required=True, help="Path to mmpose .pth")
    p.add_argument("--image", required=True, help="Image file or directory")
    p.add_argument("--out", required=True, help="Output file (single) or directory (batch)")
    p.add_argument("--bbox", default=None,
                   help="Comma-separated x1,y1,x2,y2 in image pixels. Skips the detector.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--score-thr", type=float, default=0.3,
                   help="Per-keypoint score threshold for visualization.")
    p.add_argument("--det-thr", type=float, default=0.5,
                   help="Detector score threshold (only used when --bbox is not set).")
    p.add_argument("--flip-test", action="store_true",
                   help="Enable test-time horizontal flip (vitpose / hrnet).")
    p.add_argument("--save-json", default=None,
                   help="Optional path to dump keypoints+scores per image as JSON.")
    p.add_argument("--no-viz", action="store_true",
                   help="Skip writing annotated images (just JSON).")
    args = p.parse_args()

    pose = _build_pipeline(args)

    bbox = None
    if args.bbox:
        bbox = np.array([float(v) for v in args.bbox.split(",")], dtype=np.float32)
        if bbox.shape != (4,):
            raise ValueError("--bbox needs 4 comma-separated floats")

    img_paths = list(_list_images(Path(args.image)))
    if not img_paths:
        raise SystemExit(f"no images found at {args.image}")

    # Decide output schema.
    out_path = Path(args.out)
    is_batch = len(img_paths) > 1 or Path(args.image).is_dir()
    if is_batch:
        out_path.mkdir(parents=True, exist_ok=True)

    # Pick the right keypoint schema for the visualizer.
    schema = FFPose.skeletons.COCO_17
    if "wholebody" in args.variant:
        schema = FFPose.skeletons.COCO_WHOLEBODY_133

    json_records = []
    for ip in img_paths:
        img = cv2.imread(str(ip))
        if img is None:
            print(f"[skip] could not read {ip}")
            continue
        out = _infer_one(pose, img, bbox)
        n = out["keypoints"].shape[0]
        print(f"{ip}: {n} person(s)  "
              f"mean_kpt_score={float(out['keypoint_scores'].mean()):.2f}"
              if n > 0 else f"{ip}: 0 person(s)")

        if not args.no_viz and n > 0:
            viz = FFPose.draw_skeletons(
                img, out["keypoints"], out["keypoint_scores"],
                schema=schema, boxes=out["boxes"],
                score_thr=args.score_thr,
            )
            target = out_path / ip.name if is_batch else out_path
            target.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(target), viz)

        if args.save_json is not None:
            json_records.append({
                "image": str(ip),
                "boxes": out["boxes"].tolist(),
                "box_scores": out["box_scores"].tolist(),
                "keypoints": out["keypoints"].tolist(),
                "keypoint_scores": out["keypoint_scores"].tolist(),
            })

    if args.save_json is not None:
        with open(args.save_json, "w") as f:
            json.dump(json_records, f)
        print(f"[json] wrote {len(json_records)} record(s) -> {args.save_json}")


if __name__ == "__main__":
    main()
