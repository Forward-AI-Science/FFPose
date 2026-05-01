"""COCO keypoints dataset (top-down) using pycocotools.

For each annotation with at least one labeled keypoint, returns one Sample
with the image, the bbox, and the (K, 2) keypoint coords + visibility.

Designed for top-down inference and training. The encoder + augmentation
pipeline is applied externally — this dataset only emits the raw Sample.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional

import cv2
import numpy as np
from pycocotools.coco import COCO
from torch.utils.data import Dataset

from .augmentations import Sample
from .skeletons import COCO_17, COCO_WHOLEBODY_133, KeypointSchema, SCHEMAS


# COCO body 17: hip/leg/etc are "lower"; rest "upper". Matches mmpose's split.
_COCO17_UPPER = list(range(0, 11))   # nose..wrists
_COCO17_LOWER = list(range(11, 17))  # hips, knees, ankles


class CocoKeypoints(Dataset):
    """Top-down COCO keypoints. One sample per annotated instance."""

    def __init__(
        self,
        ann_file: str | Path,
        img_root: str | Path,
        schema: KeypointSchema = COCO_17,
        keypoint_field: str = "keypoints",        # "keypoints" or "keypoints_wholebody"
        require_min_visible: int = 1,             # drop instances with fewer visible kps
        bbox_padding: float = 1.25,
        upper_body_ids: Optional[List[int]] = None,
        lower_body_ids: Optional[List[int]] = None,
        transform: Optional[Callable[[Sample], Sample]] = None,
    ) -> None:
        self.ann_file = str(ann_file)
        self.img_root = Path(img_root)
        self.schema = schema
        self.keypoint_field = keypoint_field
        self.bbox_padding = bbox_padding
        self.transform = transform

        if upper_body_ids is None and schema is COCO_17:
            upper_body_ids = _COCO17_UPPER
        if lower_body_ids is None and schema is COCO_17:
            lower_body_ids = _COCO17_LOWER
        self.upper_body_ids = upper_body_ids
        self.lower_body_ids = lower_body_ids

        self.coco = COCO(self.ann_file)
        # Filter to person annotations with usable keypoints.
        person_cats = self.coco.getCatIds(catNms=["person"]) or [1]
        ann_ids = self.coco.getAnnIds(catIds=person_cats, iscrowd=False)
        self._ann_ids: list[int] = []
        for aid in ann_ids:
            ann = self.coco.anns[aid]
            kp = ann.get(self.keypoint_field, ann.get("keypoints"))
            if kp is None:
                continue
            num_visible = sum(1 for v in kp[2::3] if v > 0)
            if num_visible < require_min_visible:
                continue
            self._ann_ids.append(aid)

    def __len__(self) -> int:
        return len(self._ann_ids)

    def _load_keypoints(self, ann: dict) -> tuple[np.ndarray, np.ndarray]:
        K = self.schema.num_keypoints
        kp_flat = ann.get(self.keypoint_field, ann.get("keypoints"))
        kp_arr = np.array(kp_flat, dtype=np.float32).reshape(-1, 3)
        if kp_arr.shape[0] != K:
            raise ValueError(
                f"annotation {ann['id']} has {kp_arr.shape[0]} keypoints; "
                f"expected {K} for schema {self.schema.name}"
            )
        keypoints = kp_arr[:, :2]
        visible = (kp_arr[:, 2] > 0).astype(np.float32)
        return keypoints, visible

    def __getitem__(self, idx: int) -> Sample:
        ann = self.coco.anns[self._ann_ids[idx]]
        img_info = self.coco.imgs[ann["image_id"]]
        img_path = self.img_root / img_info["file_name"]
        image = cv2.imread(str(img_path))
        if image is None:
            raise FileNotFoundError(f"could not read {img_path}")

        # COCO bbox is xywh; convert to xyxy.
        x, y, w, h = ann["bbox"]
        bbox_xyxy = np.array([x, y, x + w, y + h], dtype=np.float32)

        kpts, vis = self._load_keypoints(ann)

        sample = Sample(
            image=image,
            bbox_xyxy=bbox_xyxy,
            keypoints=kpts,
            keypoints_visible=vis,
            flip_indices=list(self.schema.flip_indices),
            upper_body_ids=self.upper_body_ids,
            lower_body_ids=self.lower_body_ids,
        )

        if self.transform is not None:
            sample = self.transform(sample)
        return sample
