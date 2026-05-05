"""End-to-end image-to-pose pipeline: detect persons, run top-down pose for each."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

import numpy as np
import torch

from .detector import PersonDetector
from .hrformer_pose import HRFormerPoseInferencer
from .hrnet_pose import HRNetPoseInferencer
from .inference import PoseResult, RTMPoseInferencer
from .litehrnet_pose import LiteHRNetPoseInferencer
from .swin_pose import SwinPoseInferencer
from .vitpose import ViTPoseInferencer


@runtime_checkable
class _PoseInferencer(Protocol):
    """Duck-typed interface satisfied by RTMPoseInferencer / ViTPoseInferencer."""
    cfg: object
    def predict(self, image: np.ndarray, bbox_xyxy: np.ndarray) -> PoseResult: ...


@dataclass
class FullFrameResult:
    boxes: np.ndarray         # (N, 4) xyxy
    box_scores: np.ndarray    # (N,) detector confidence
    keypoints: np.ndarray     # (N, K, 2) image-space pixels
    keypoint_scores: np.ndarray  # (N, K)


class TopDownPosePipeline:
    """Detector -> per-box pose estimator. One ``predict_image()`` call.

    Accepts any pose inferencer that exposes ``cfg.out_channels`` and
    ``predict(image, bbox) -> PoseResult`` (RTMPose and ViTPose both qualify).
    """

    def __init__(
        self,
        pose: _PoseInferencer,
        detector: Optional[PersonDetector] = None,
        device: torch.device | str = "cuda",
    ) -> None:
        self.pose = pose
        self.detector = detector or PersonDetector(device=device)
        self.device = torch.device(device)

    @classmethod
    def from_pretrained(
        cls,
        pose_variant: str,
        pose_checkpoint: str | Path,
        device: torch.device | str = "cuda",
        det_score_thr: float = 0.5,
        family: str = "rtmpose",
    ) -> "TopDownPosePipeline":
        """Build a pipeline from a known variant + checkpoint path.

        Args:
            family: "rtmpose" or "vitpose" — selects which inferencer to use.
        """
        builders = {
            "rtmpose":   RTMPoseInferencer.from_pretrained,
            "vitpose":   ViTPoseInferencer.from_pretrained,
            "hrnet":     HRNetPoseInferencer.from_pretrained,
            "swin":      SwinPoseInferencer.from_pretrained,
            "hrformer":  HRFormerPoseInferencer.from_pretrained,
            "litehrnet": LiteHRNetPoseInferencer.from_pretrained,
        }
        if family not in builders:
            raise ValueError(f"unknown family {family!r}; choose from {list(builders)}")
        pose = builders[family](pose_variant, pose_checkpoint, device=device)
        det = PersonDetector(device=device, score_thr=det_score_thr)
        return cls(pose=pose, detector=det, device=device)

    def predict_image(self, image_bgr: np.ndarray) -> FullFrameResult:
        det = self.detector.detect(image_bgr)
        if det.boxes.shape[0] == 0:
            K = self.pose.cfg.out_channels
            return FullFrameResult(
                boxes=np.zeros((0, 4), dtype=np.float32),
                box_scores=np.zeros((0,), dtype=np.float32),
                keypoints=np.zeros((0, K, 2), dtype=np.float32),
                keypoint_scores=np.zeros((0, K), dtype=np.float32),
            )

        kp_list, sc_list = [], []
        for box in det.boxes:
            res: PoseResult = self.pose.predict(image_bgr, box)
            kp_list.append(res.keypoints[0])   # (K, 2)
            sc_list.append(res.scores[0])       # (K,)

        return FullFrameResult(
            boxes=det.boxes,
            box_scores=det.scores,
            keypoints=np.stack(kp_list, axis=0),
            keypoint_scores=np.stack(sc_list, axis=0),
        )
