"""Evaluation metrics for keypoint estimation.

PCK (cheap, fast): fraction of keypoints whose Euclidean distance from GT
is below ``thr * normalize`` where ``normalize`` is typically the bbox size
or torso length.

OKS / COCO-AP (full): uses ``pycocotools`` for body-17 or ``xtcocotools`` for
wholebody-133. Requires writing per-image predictions to a COCO-format json
and evaluating against the ground-truth annotation file.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


def pck_accuracy(
    pred: np.ndarray,        # (N, K, 2) keypoints in image-space pixels
    gt: np.ndarray,          # (N, K, 2) GT keypoints in image-space pixels
    visible: np.ndarray,     # (N, K) 0/1
    normalize: np.ndarray,   # (N,) per-instance normalizer (e.g., bbox diag)
    thr: float = 0.05,
) -> Tuple[float, np.ndarray]:
    """Per-keypoint and overall PCK@thr.

    Returns:
        mean_pck: scalar mean accuracy over visible keypoints.
        per_kpt:  (K,) per-keypoint accuracy.
    """
    if pred.shape != gt.shape:
        raise ValueError(f"shape mismatch: pred {pred.shape} vs gt {gt.shape}")
    N, K, _ = pred.shape
    dist = np.linalg.norm(pred - gt, axis=-1)
    norm = normalize.reshape(N, 1)
    correct = (dist <= thr * norm) & (visible > 0)
    counts = (visible > 0).sum(axis=0).astype(np.float32)
    counts = np.where(counts > 0, counts, 1.0)
    per_kpt = correct.sum(axis=0).astype(np.float32) / counts
    total_visible = (visible > 0).sum()
    mean_pck = float(correct.sum() / max(total_visible, 1))
    return mean_pck, per_kpt


# ---- COCO-AP via pycocotools (body-17) and xtcocotools (wholebody-133) -----

# IoU types supported by xtcocotools (subset relevant to mmpose):
#   "keypoints"            -> body-17
#   "keypoints_wholebody"  -> 133-pt
#   "keypoints_face"       -> 68-pt face
#   "keypoints_lefthand"/"keypoints_righthand"
#   "keypoints_foot"
_XTCOCOTOOLS_IOUTYPES = {
    "keypoints_wholebody", "keypoints_face",
    "keypoints_lefthand", "keypoints_righthand", "keypoints_foot",
}


def _import_coco(iou_type: str):
    """Pick the right COCO/COCOeval pair for the IoU type.

    body-17 uses pycocotools (always available). Wholebody/face/hand IoU
    types only exist in xtcocotools — we raise a clear error if the caller
    asks for one without xtcocotools installed.
    """
    if iou_type in _XTCOCOTOOLS_IOUTYPES:
        try:
            from xtcocotools.coco import COCO
            from xtcocotools.cocoeval import COCOeval
        except ImportError as e:
            raise ImportError(
                f"iou_type={iou_type!r} requires xtcocotools. "
                "Install with: pip install ffpose[wholebody]  (or pip install xtcocotools)"
            ) from e
        return COCO, COCOeval
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    return COCO, COCOeval


class CocoKeypointMetric:
    """Compute COCO-AP / AR by running pycocotools/xtcocotools' ``COCOeval``.

    Args:
        gt_ann_file: ground-truth annotation file (COCO format).
        iou_type: "keypoints" (body-17, default), "keypoints_wholebody" (133),
                  or any other xtcocotools-supported keypoint IoU type.

    Usage:
        metric = CocoKeypointMetric(gt_ann_file, iou_type="keypoints_wholebody")
        for prediction in predictions:
            metric.add(image_id, category_id, keypoints, score)
        result = metric.compute()
    """

    def __init__(
        self,
        gt_ann_file: str | Path,
        iou_type: str = "keypoints",
    ) -> None:
        self.iou_type = iou_type
        COCO, _ = _import_coco(iou_type)
        self.coco_gt = COCO(str(gt_ann_file))
        self._results: list[dict] = []

    def add(
        self,
        image_id: int,
        category_id: int,
        keypoints: np.ndarray,   # (K, 2) or (K, 3) xy or xyc
        score: float,
    ) -> None:
        if keypoints.shape[-1] == 2:
            kp_with_score = np.concatenate(
                [keypoints, np.ones((keypoints.shape[0], 1), dtype=np.float32)], axis=-1
            )
        else:
            kp_with_score = keypoints
        flat = kp_with_score.reshape(-1).astype(float).tolist()
        self._results.append({
            "image_id": int(image_id),
            "category_id": int(category_id),
            "keypoints": flat,
            "score": float(score),
        })

    def compute(self) -> dict:
        if not self._results:
            return {"AP": 0.0, "num_predictions": 0}
        _, COCOeval = _import_coco(self.iou_type)
        coco_dt = self.coco_gt.loadRes(self._results)
        # xtcocotools' COCOeval signature is the same as pycocotools' for the
        # body-17 case. For wholebody-style IoU types xtcocotools also accepts
        # a ``sigmas`` kwarg, but the defaults match mmpose's CocoMetric.
        ev = COCOeval(self.coco_gt, coco_dt, iouType=self.iou_type)
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
        keys = ["AP", "AP50", "AP75", "APm", "APl", "AR", "AR50", "AR75", "ARm", "ARl"]
        return dict(zip(keys, [float(x) for x in ev.stats]))

    def dump_results(self, path: str | Path) -> None:
        with open(path, "w") as f:
            json.dump(self._results, f)
