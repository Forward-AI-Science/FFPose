"""Pipeline-switch callback (mmpose's PipelineSwitchHook equivalent).

mmpose's RTMPose configs train with two augmentation stages:
  - stage 1 (epochs 0 .. stage_switch_epoch): heavy augs incl. CoarseDropout p=1.0
  - stage 2 (after switch_epoch):              same set with CoarseDropout p=0.5

In FFPose this is wired through ``TrainConfig.on_epoch_start``: the trainer
calls the callback at the start of each epoch and, if the callback returns a
new ``DataLoader``, it swaps it in for the rest of training.
"""
from __future__ import annotations

from typing import Callable, Optional

from torch.utils.data import DataLoader, Dataset


class PipelineSwitchHook:
    """Swap the train DataLoader's transform at ``switch_epoch``.

    Example:
        switch_hook = PipelineSwitchHook(
            switch_epoch=210,
            dataset=train_dataset,
            new_pipeline=rtmpose_recipe(...).train_pipeline,  # stage-2 pipeline
            build_loader=lambda ds: DataLoader(ds, ...),
        )
        cfg = TrainConfig(..., on_epoch_start=switch_hook)
        PoseTrainer(..., config=cfg)

    Args:
        switch_epoch: 1-indexed epoch at which to switch (matches mmpose's
            convention: ``trainer._epoch`` is incremented before this hook fires).
        dataset: the training Dataset whose ``transform`` will be replaced.
        new_pipeline: the new pipeline (Pipeline or callable Sample->Sample).
        build_loader: function ``(dataset) -> DataLoader`` used to rebuild the
            DataLoader after the dataset is updated. Required because workers
            are forked once at iter time and won't pick up in-place changes.
    """

    def __init__(
        self,
        switch_epoch: int,
        dataset: Dataset,
        new_pipeline: Callable,
        build_loader: Callable[[Dataset], DataLoader],
    ) -> None:
        self.switch_epoch = int(switch_epoch)
        self.dataset = dataset
        self.new_pipeline = new_pipeline
        self.build_loader = build_loader
        self._fired = False

    def __call__(self, trainer, epoch: int) -> Optional[DataLoader]:
        if self._fired or epoch < self.switch_epoch:
            return None
        self.dataset.transform = self.new_pipeline
        self._fired = True
        return self.build_loader(self.dataset)
