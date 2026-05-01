"""Evaluate one or all FFPose backbones on a COCO-format keypoint annotation file.

Loads each requested backbone from its checkpoint, runs the *ground-truth bbox*
through the model (so we measure pose accuracy alone, not detector quality),
then reports PCK@thr and (optionally) COCO-AP.

Quick smoke check (no real annotations needed — uses bundled test JSON):
    python tools/eval.py --family all \
        --checkpoint-dir /tmp/mmpose_weights \
        --ann-file ../mmpose/tests/data/coco/test_coco.json \
        --img-root  ../mmpose/tests/data/coco \
        --max-samples 12

Full COCO val (proper benchmark — needs annotations + images):
    python tools/eval.py --family rtmpose --variant body-coco-m \
        --checkpoint /path/to/rtmpose-m.pth \
        --ann-file /coco/annotations/person_keypoints_val2017.json \
        --img-root  /coco/val2017 \
        --coco-ap

Output: per-keypoint PCK, mean PCK, throughput, and (with --coco-ap)
the standard COCO-AP / AP50 / AP75 / AR row.
"""
from __future__ import annotations

import argparse
import os as _os
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

# Resolve FFPose: env override or sibling-of-tools default.
_ffpose_home = _os.environ.get("FFPOSE_HOME") or str(
    Path(__file__).resolve().parent.parent
)
sys.path.insert(0, _ffpose_home)

import FFPose
from FFPose.training.metrics import CocoKeypointMetric, pck_accuracy


# --- backbone registry ------------------------------------------------------

# (family, default variant, default checkpoint filename) for each known family.
_BACKBONES: Dict[str, Tuple[Callable, str, str]] = {
    "rtmpose":   (FFPose.RTMPoseInferencer.from_pretrained,    "body-coco-m", "rtmpose-m.pth"),
    "vitpose":   (FFPose.ViTPoseInferencer.from_pretrained,    "small",       "vitpose-s.pth"),
    "hrnet":     (FFPose.HRNetPoseInferencer.from_pretrained,  "w32",         "hrnet-w32.pth"),
    "swin":      (FFPose.SwinPoseInferencer.from_pretrained,   "t",           "swin-t.pth"),
    "hrformer":  (FFPose.HRFormerPoseInferencer.from_pretrained, "small",     "hrformer-s.pth"),
    "litehrnet": (FFPose.LiteHRNetPoseInferencer.from_pretrained, "30",       "litehrnet-30.pth"),
}


# --- evaluation core --------------------------------------------------------

def _bbox_xywh_to_xyxy(bbox: List[float]) -> np.ndarray:
    x, y, w, h = bbox
    return np.array([x, y, x + w, y + h], dtype=np.float32)


def _eval_one(
    inferencer,
    coco,
    ann_ids: List[int],
    img_root: Path,
    pck_thr: float,
    coco_ap: bool,
) -> Tuple[Dict[str, float], Optional[CocoKeypointMetric]]:
    """Run inference on each annotation, compute PCK, optionally accumulate AP.

    Per-instance PCK is normalized by bbox diagonal (mmpose's standard
    "object-wise normalize" — close approximation to OKS without the
    per-keypoint sigma).
    """
    K = inferencer.cfg.out_channels
    metric = CocoKeypointMetric(_ann_file_path) if coco_ap else None

    preds, gts, viss, norms = [], [], [], []
    image_ids, scores_per_inst = [], []

    t0 = time.time()
    for aid in ann_ids:
        ann = coco.anns[aid]
        img_info = coco.imgs[ann["image_id"]]
        img_path = img_root / img_info["file_name"]
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"  [skip] cannot read {img_path}")
            continue

        bbox = _bbox_xywh_to_xyxy(ann["bbox"])
        # Skip degenerate bboxes
        if (bbox[2] - bbox[0]) < 4 or (bbox[3] - bbox[1]) < 4:
            continue

        kp_flat = ann.get("keypoints")
        if kp_flat is None or len(kp_flat) // 3 != K:
            continue
        kp_arr = np.array(kp_flat, dtype=np.float32).reshape(K, 3)
        gt_xy = kp_arr[:, :2]
        vis = (kp_arr[:, 2] > 0).astype(np.float32)
        if vis.sum() == 0:
            continue

        result = inferencer.predict(image, bbox)
        pred_xy = result.keypoints[0]
        pred_sc = result.scores[0]

        preds.append(pred_xy)
        gts.append(gt_xy)
        viss.append(vis)
        norms.append(float(np.hypot(bbox[2] - bbox[0], bbox[3] - bbox[1])))

        if metric is not None:
            metric.add(
                image_id=ann["image_id"],
                category_id=ann.get("category_id", 1),
                keypoints=np.concatenate([pred_xy, pred_sc[:, None]], axis=1),
                score=float(pred_sc.mean()),
            )
        image_ids.append(ann["image_id"])
        scores_per_inst.append(float(pred_sc.mean()))

    elapsed = time.time() - t0
    n = len(preds)
    if n == 0:
        raise RuntimeError("no usable annotations found — check --ann-file / --img-root")

    pred_arr = np.stack(preds, axis=0)
    gt_arr = np.stack(gts, axis=0)
    vis_arr = np.stack(viss, axis=0)
    norm_arr = np.array(norms, dtype=np.float32)

    mean_pck, per_kpt = pck_accuracy(pred_arr, gt_arr, vis_arr, norm_arr, thr=pck_thr)

    summary = {
        "n_instances": n,
        "fps": n / max(elapsed, 1e-9),
        "elapsed_s": elapsed,
        f"PCK@{pck_thr:.2f}": float(mean_pck),
        "mean_kpt_score": float(np.mean(scores_per_inst)),
    }
    return summary, metric, per_kpt


_ann_file_path: Optional[str] = None


def _load_coco(ann_file: str | Path):
    """Returns the loaded COCO object plus its ``ann_file`` (so the metric can
    re-load it). Uses pycocotools."""
    global _ann_file_path
    _ann_file_path = str(ann_file)
    from pycocotools.coco import COCO
    return COCO(str(ann_file))


def _select_ann_ids(coco, max_samples: int) -> List[int]:
    """Pick annotations with at least one labeled keypoint."""
    person_cats = coco.getCatIds(catNms=["person"]) or [1]
    ann_ids = coco.getAnnIds(catIds=person_cats, iscrowd=False)
    keep = []
    for aid in ann_ids:
        ann = coco.anns[aid]
        kp = ann.get("keypoints")
        if kp and any(v > 0 for v in kp[2::3]):
            keep.append(aid)
    if max_samples > 0:
        keep = keep[:max_samples]
    return keep


# --- main -------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--family", default="all",
                   help="rtmpose | vitpose | hrnet | swin | hrformer | litehrnet | all")
    p.add_argument("--variant", default=None,
                   help="Variant override (only with single --family).")
    p.add_argument("--checkpoint", default=None,
                   help="Single .pth path (only with single --family).")
    p.add_argument("--checkpoint-dir", default=None,
                   help="Directory holding default-named checkpoints (used with --family all). "
                        "Expected names: rtmpose-m.pth, vitpose-s.pth, hrnet-w32.pth, "
                        "swin-t.pth, hrformer-s.pth, litehrnet-30.pth")
    p.add_argument("--ann-file", required=True,
                   help="COCO-format annotation JSON.")
    p.add_argument("--img-root", required=True,
                   help="Directory containing the images referenced by ann-file.")
    p.add_argument("--max-samples", type=int, default=0,
                   help="Cap evaluation at this many annotations (0 = all).")
    p.add_argument("--pck-thr", type=float, default=0.05,
                   help="PCK threshold as fraction of bbox diagonal.")
    p.add_argument("--coco-ap", action="store_true",
                   help="Also compute COCO AP via pycocotools.COCOeval.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--flip-test", action="store_true",
                   help="Enable test-time horizontal flip on heatmap models.")
    args = p.parse_args()

    # Build the list of (family, variant, checkpoint_path) jobs.
    if args.family == "all":
        if not args.checkpoint_dir:
            raise SystemExit("--family all requires --checkpoint-dir")
        ckpt_dir = Path(args.checkpoint_dir)
        jobs = []
        for fam, (_builder, default_variant, default_name) in _BACKBONES.items():
            ckpt = ckpt_dir / default_name
            if not ckpt.exists():
                print(f"[skip] {fam}: missing checkpoint {ckpt}")
                continue
            jobs.append((fam, default_variant, ckpt))
    else:
        if args.family not in _BACKBONES:
            raise SystemExit(f"unknown family {args.family!r}; "
                             f"choose from {list(_BACKBONES) + ['all']}")
        if not args.checkpoint:
            raise SystemExit(f"--checkpoint is required when --family={args.family}")
        builder, default_variant, _ = _BACKBONES[args.family]
        variant = args.variant or default_variant
        jobs = [(args.family, variant, Path(args.checkpoint))]

    coco = _load_coco(args.ann_file)
    ann_ids = _select_ann_ids(coco, args.max_samples)
    print(f"=== eval on {len(ann_ids)} annotations from {args.ann_file} ===\n")

    rows = []
    for fam, variant, ckpt in jobs:
        builder, _, _ = _BACKBONES[fam]
        print(f"[{fam}] variant={variant}, ckpt={ckpt.name}")
        try:
            kwargs = dict(device=args.device)
            if args.flip_test and fam in ("vitpose", "hrnet", "swin", "hrformer", "litehrnet"):
                kwargs["flip_test"] = True
            inferencer = builder(variant, str(ckpt), **kwargs)
            summary, metric, per_kpt = _eval_one(
                inferencer, coco, ann_ids, Path(args.img_root),
                pck_thr=args.pck_thr, coco_ap=args.coco_ap,
            )
            row = dict(family=fam, variant=variant, **summary, status="ok")
            if args.coco_ap and metric is not None:
                ap = metric.compute()
                row.update({k: ap[k] for k in ("AP", "AP50", "AP75") if k in ap})
            rows.append(row)
        except Exception as e:
            print(f"  [error] {type(e).__name__}: {e}")
            rows.append(dict(family=fam, variant=variant, status=f"FAIL: {e}"))
        print()

    # Pretty-print a comparison table.
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    cols = ["family", "variant", "n_instances",
            f"PCK@{args.pck_thr:.2f}", "mean_kpt_score", "fps"]
    if args.coco_ap:
        cols += ["AP", "AP50", "AP75"]
    cols += ["status"]

    widths = [max(len(c), max((len(_fmt(r.get(c, ""))) for r in rows), default=0)) for c in cols]
    header = "  ".join(c.ljust(w) for c, w in zip(cols, widths))
    print(header)
    print("-" * len(header))
    for r in rows:
        line = "  ".join(_fmt(r.get(c, "")).ljust(w) for c, w in zip(cols, widths))
        print(line)


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


if __name__ == "__main__":
    main()
