"""Person detector wrapper around torchvision's pretrained detectors.

Avoids any custom CUDA / mmdet code: uses ``fasterrcnn_resnet50_fpn_v2`` whose
weights ship with torchvision and run on PyTorch 2.x out of the box.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import torch
import torchvision
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_V2_Weights,
    fasterrcnn_resnet50_fpn_v2,
)

# COCO class index for "person" in torchvision's pretrained head.
COCO_PERSON_CLASS = 1


@dataclass
class Detections:
    boxes: np.ndarray   # (N, 4) xyxy in image pixels
    scores: np.ndarray  # (N,)


class PersonDetector:
    """Detect "person" boxes with a torchvision Faster R-CNN."""

    def __init__(
        self,
        device: torch.device | str = "cuda",
        score_thr: float = 0.5,
        max_detections: int = 100,
    ) -> None:
        weights = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
        self.model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=score_thr)
        self.model.eval().to(device)
        self.device = torch.device(device)
        self.score_thr = score_thr
        self.max_detections = max_detections
        # FasterRCNN handles its own resize/normalize internally; we only need
        # to convert HWC uint8 -> CHW float in [0, 1] RGB.

    @torch.no_grad()
    def detect(self, image_bgr: np.ndarray) -> Detections:
        """Run detection on a single image (BGR HxWx3 uint8)."""
        if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise ValueError(f"expected HxWx3 image, got {image_bgr.shape}")
        rgb = image_bgr[:, :, ::-1].copy()  # BGR -> RGB
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255.0).to(self.device)

        outputs = self.model([tensor])[0]
        boxes = outputs["boxes"].cpu().numpy()
        scores = outputs["scores"].cpu().numpy()
        labels = outputs["labels"].cpu().numpy()

        keep = (labels == COCO_PERSON_CLASS) & (scores >= self.score_thr)
        boxes = boxes[keep][: self.max_detections]
        scores = scores[keep][: self.max_detections]
        return Detections(boxes=boxes.astype(np.float32), scores=scores.astype(np.float32))
