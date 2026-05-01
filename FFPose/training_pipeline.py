"""Training pipelines: per-family compose() + GenerateTarget + collate.

For each model family (RTMPose / HRNet / ViTPose) there's a recipe that
returns ``(train_pipeline, val_pipeline)`` matching the corresponding mmpose
config. The ``GenerateTarget`` step at the end of the train pipeline runs
the codec encoder so the dataset directly emits training targets.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate

from .augmentations import (
    GetBBoxCenterScale,
    NormalizeAndToTensor,
    Pipeline,
    RandomBBoxTransform,
    RandomFlip,
    RandomHalfBody,
    Sample,
    TopdownAffine,
    YOLOXHSVRandomAug,
)
from .encoders import (
    encode_msra_heatmap,
    encode_simcc_gaussian,
    encode_udp_heatmap,
)


# ---- target generators -----------------------------------------------------

class GenerateSimCCTarget:
    """Encode keypoints into Gaussian SimCC labels (RTMPose)."""

    def __init__(
        self,
        input_size: Tuple[int, int],
        sigma: Tuple[float, float] = (4.9, 5.66),
        simcc_split_ratio: float = 2.0,
        normalize: bool = False,
    ) -> None:
        self.input_size = input_size
        self.sigma = sigma
        self.simcc_split_ratio = simcc_split_ratio
        self.normalize = normalize

    def __call__(self, s: Sample) -> Sample:
        kp = s.keypoints[None]                          # (1, K, 2)
        vis = s.keypoints_visible[None]                  # (1, K)
        tx, ty, w = encode_simcc_gaussian(
            kp, vis, self.input_size, self.sigma, self.simcc_split_ratio,
            normalize=self.normalize,
        )
        s.targets = dict(simcc_x=tx[0], simcc_y=ty[0], weights=w[0])
        return s


class GenerateMSRAHeatmapTarget:
    def __init__(
        self,
        input_size: Tuple[int, int],
        heatmap_size: Tuple[int, int],
        sigma: float = 2.0,
        unbiased: bool = False,
    ) -> None:
        self.input_size = input_size
        self.heatmap_size = heatmap_size
        self.sigma = sigma
        self.unbiased = unbiased

    def __call__(self, s: Sample) -> Sample:
        kp = s.keypoints[None]
        vis = s.keypoints_visible[None]
        hm, w = encode_msra_heatmap(
            kp, vis, self.input_size, self.heatmap_size, self.sigma, self.unbiased
        )
        s.targets = dict(heatmap=hm, weights=w[0])
        return s


class GenerateUDPHeatmapTarget:
    def __init__(
        self,
        input_size: Tuple[int, int],
        heatmap_size: Tuple[int, int],
        sigma: float = 2.0,
    ) -> None:
        self.input_size = input_size
        self.heatmap_size = heatmap_size
        self.sigma = sigma

    def __call__(self, s: Sample) -> Sample:
        kp = s.keypoints[None]
        vis = s.keypoints_visible[None]
        hm, w = encode_udp_heatmap(
            kp, vis, self.input_size, self.heatmap_size, self.sigma
        )
        s.targets = dict(heatmap=hm, weights=w[0])
        return s


# ---- recipes (one builder per family) --------------------------------------

@dataclass
class TrainRecipe:
    train_pipeline: Pipeline
    val_pipeline: Pipeline
    family: str   # "rtmpose" | "hrnet" | "vitpose"


def rtmpose_recipe(
    input_size: Tuple[int, int] = (192, 256),
    sigma: Tuple[float, float] = (4.9, 5.66),
    simcc_split_ratio: float = 2.0,
    rotate_factor: float = 80.0,
) -> TrainRecipe:
    """RTMPose top-down recipe (matches mmpose configs/body_2d_keypoint/rtmpose/coco)."""
    train = Pipeline([
        GetBBoxCenterScale(padding=1.25),
        RandomFlip(prob=0.5),
        RandomHalfBody(),
        RandomBBoxTransform(scale_factor=(0.6, 1.4), rotate_factor=rotate_factor),
        TopdownAffine(input_size=input_size, use_udp=False),
        YOLOXHSVRandomAug(),
        # NOTE: mmpose RTMPose configs also stack Albumentations
        # (Blur/MedianBlur/CoarseDropout). Skipped here; can be wired in by
        # passing an Albumentations callable into this list.
        NormalizeAndToTensor(),
        GenerateSimCCTarget(input_size=input_size, sigma=sigma,
                            simcc_split_ratio=simcc_split_ratio, normalize=False),
    ])
    val = Pipeline([
        GetBBoxCenterScale(padding=1.25),
        TopdownAffine(input_size=input_size, use_udp=False),
        NormalizeAndToTensor(),
    ])
    return TrainRecipe(train, val, family="rtmpose")


def hrnet_recipe(
    input_size: Tuple[int, int] = (192, 256),
    heatmap_size: Tuple[int, int] = (48, 64),
    sigma: float = 2.0,
    rotate_factor: float = 40.0,
) -> TrainRecipe:
    """HRNet/SimpleBaselines top-down recipe."""
    train = Pipeline([
        GetBBoxCenterScale(padding=1.25),
        RandomFlip(prob=0.5),
        RandomHalfBody(),
        RandomBBoxTransform(rotate_factor=rotate_factor),
        TopdownAffine(input_size=input_size, use_udp=False),
        NormalizeAndToTensor(),
        GenerateMSRAHeatmapTarget(input_size=input_size, heatmap_size=heatmap_size,
                                   sigma=sigma, unbiased=False),
    ])
    val = Pipeline([
        GetBBoxCenterScale(padding=1.25),
        TopdownAffine(input_size=input_size, use_udp=False),
        NormalizeAndToTensor(),
    ])
    return TrainRecipe(train, val, family="hrnet")


def vitpose_recipe(
    input_size: Tuple[int, int] = (192, 256),
    heatmap_size: Tuple[int, int] = (48, 64),
    sigma: float = 2.0,
    rotate_factor: float = 40.0,
) -> TrainRecipe:
    """ViTPose top-down recipe (UDP-aware)."""
    train = Pipeline([
        GetBBoxCenterScale(padding=1.25),
        RandomFlip(prob=0.5),
        RandomHalfBody(),
        RandomBBoxTransform(rotate_factor=rotate_factor),
        TopdownAffine(input_size=input_size, use_udp=True),
        NormalizeAndToTensor(),
        GenerateUDPHeatmapTarget(input_size=input_size, heatmap_size=heatmap_size,
                                  sigma=sigma),
    ])
    val = Pipeline([
        GetBBoxCenterScale(padding=1.25),
        TopdownAffine(input_size=input_size, use_udp=True),
        NormalizeAndToTensor(),
    ])
    return TrainRecipe(train, val, family="vitpose")


# ---- collate function ------------------------------------------------------

def _to_tensor(x: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(x))


def collate_train(batch: List[Sample]) -> Dict[str, torch.Tensor]:
    """Stack training Samples into batched tensors.

    Returns a dict containing the fields the trainer needs: input image, all
    keys from sample.targets stacked, and a few metadata fields kept for
    validation/eval (warp_mat, bbox_center, bbox_scale).
    """
    images = torch.stack([_to_tensor(s.input_image).float() for s in batch])
    out: Dict[str, torch.Tensor] = {"image": images}

    # Targets: collect by key and stack. None for val-pipeline samples.
    keys = set()
    for s in batch:
        t = getattr(s, "targets", None) or {}
        keys.update(t.keys())
    for k in keys:
        vals = [_to_tensor(s.targets[k]) for s in batch]
        out[k] = torch.stack(vals)

    # Useful metadata (still numpy or python).
    out["bbox_center"] = torch.from_numpy(np.stack([s.bbox_center for s in batch])).float()
    out["bbox_scale"] = torch.from_numpy(np.stack([s.bbox_scale for s in batch])).float()
    out["warp_mat"] = torch.from_numpy(np.stack([s.warp_mat for s in batch])).float()
    out["keypoints_input_space"] = torch.from_numpy(
        np.stack([s.keypoints for s in batch])
    ).float()
    out["keypoints_visible"] = torch.from_numpy(
        np.stack([s.keypoints_visible for s in batch])
    ).float()
    return out
