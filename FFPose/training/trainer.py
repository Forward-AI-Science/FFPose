"""Single-GPU trainer for top-down pose estimation.

Plain PyTorch loop with AMP, EMA, gradient clip, LR schedule, periodic eval,
and best-checkpoint saving. Family-agnostic: pass a ``loss_fn`` that consumes
``(model_output, batch)`` and returns a scalar loss.

Usage:
    trainer = PoseTrainer(model, train_loader, val_loader, optimizer,
                          scheduler, loss_fn, family="rtmpose")
    trainer.train(num_epochs=20, save_dir="ckpts/")
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

import torch.distributed as torch_dist

from .dist import DistEnv, all_reduce_mean, is_main_process
from .ema import ExpMomentumEMA
from .metrics import pck_accuracy


# ---- per-family loss closures ----------------------------------------------

def make_simcc_loss_fn(loss_module: nn.Module) -> Callable[[tuple, dict], torch.Tensor]:
    """Wraps KLDiscretLoss for a (pred_x, pred_y) RTMPose head."""
    def f(output: tuple, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        pred_x, pred_y = output
        return loss_module(
            (pred_x, pred_y),
            (batch["simcc_x"], batch["simcc_y"]),
            target_weight=batch["weights"],
        )
    return f


def make_heatmap_loss_fn(loss_module: nn.Module) -> Callable[[torch.Tensor, dict], torch.Tensor]:
    """Wraps KeypointMSELoss for HRNet/ViTPose heatmap heads."""
    def f(output: torch.Tensor, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        return loss_module(output, batch["heatmap"], target_weights=batch["weights"])
    return f


# ---- trainer ---------------------------------------------------------------

@dataclass
class TrainConfig:
    num_epochs: int = 10
    log_interval: int = 50            # iters between log lines
    eval_interval_epochs: int = 1     # epochs between validation runs
    grad_clip_norm: float = 1.0
    use_amp: bool = True
    use_ema: bool = True
    ema_momentum: float = 0.0002
    ema_gamma: int = 2000
    ema_update_buffers: bool = True
    save_dir: Path = field(default_factory=lambda: Path("ckpts"))
    save_best_metric: str = "val_pck"   # higher is better
    keep_last: int = 1                  # keep most recent N checkpoints
    # Optional: invoked at the start of each epoch with ``(trainer, epoch)``.
    # If the callback returns a DataLoader, the trainer swaps it in as the
    # new train loader. Used by :class:`FFPose.training.PipelineSwitchHook`
    # to recreate the dataloader with stage-2 augmentations partway through
    # training (matching mmpose's RTMPose recipe).
    on_epoch_start: Optional[Callable] = None


class PoseTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        optimizer: Optimizer,
        scheduler: LRScheduler,
        loss_fn: Callable[[any, dict], torch.Tensor],
        *,
        device: str | torch.device = "cuda",
        config: Optional[TrainConfig] = None,
        eval_fn: Optional[Callable[[nn.Module, DataLoader], dict]] = None,
        dist_env: Optional[DistEnv] = None,
    ) -> None:
        self.cfg = config or TrainConfig()
        self.dist_env = dist_env
        self.device = (dist_env.device if dist_env is not None and dist_env.enabled
                       else torch.device(device))
        self.model = model.to(self.device)
        # When wrapped in DDP, .module is the underlying module — needed by EMA.
        self._raw_model = self.model.module if hasattr(self.model, "module") else self.model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.eval_fn = eval_fn or default_pck_eval

        amp_device = "cuda" if self.device.type == "cuda" else "cpu"
        self.scaler = GradScaler(amp_device, enabled=self.cfg.use_amp)
        self._amp_device = amp_device
        self.ema: Optional[ExpMomentumEMA] = None
        if self.cfg.use_ema:
            # EMA tracks the raw module (unwrapped from DDP).
            self.ema = ExpMomentumEMA(
                self._raw_model,
                momentum=self.cfg.ema_momentum,
                gamma=self.cfg.ema_gamma,
                update_buffers=self.cfg.ema_update_buffers,
                device=self.device,
            )

        self.cfg.save_dir = Path(self.cfg.save_dir)
        if is_main_process(self.dist_env):
            self.cfg.save_dir.mkdir(parents=True, exist_ok=True)
        self._best_metric: Optional[float] = None
        self._epoch = 0
        self._global_step = 0
        self._history: List[dict] = []

    # ---- train loop --------------------------------------------------------

    def train(self, num_epochs: Optional[int] = None) -> List[dict]:
        epochs = num_epochs if num_epochs is not None else self.cfg.num_epochs
        for _ in range(epochs):
            self._epoch += 1
            # Per-epoch callback (e.g., PipelineSwitchHook). If it returns a
            # DataLoader we replace the current train loader.
            if self.cfg.on_epoch_start is not None:
                new_loader = self.cfg.on_epoch_start(self, self._epoch)
                if new_loader is not None:
                    self.train_loader = new_loader
            # DistributedSampler needs the epoch to vary the shuffle seed.
            sampler = getattr(self.train_loader, "sampler", None)
            if sampler is not None and hasattr(sampler, "set_epoch"):
                sampler.set_epoch(self._epoch)

            self._train_one_epoch()
            if (self.val_loader is not None
                    and self._epoch % self.cfg.eval_interval_epochs == 0):
                metrics = self._validate()
                self._maybe_save_best(metrics)
            else:
                metrics = {}
            self._history.append({"epoch": self._epoch, **metrics})
            self._save_checkpoint(self.cfg.save_dir / f"epoch_{self._epoch}.pth", metrics)
            # Other ranks waited at no collective inside _validate; rejoin here.
            if torch_dist.is_initialized():
                torch_dist.barrier()
        return self._history

    def _train_one_epoch(self) -> None:
        self.model.train()
        t0 = time.time()
        loss_sum = 0.0
        n_iters = len(self.train_loader)

        for it, batch in enumerate(self.train_loader):
            batch = {k: (v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
            self.optimizer.zero_grad(set_to_none=True)
            with autocast(self._amp_device, enabled=self.cfg.use_amp, dtype=torch.float16):
                output = self.model(batch["image"])
                loss = self.loss_fn(output, batch)

            self.scaler.scale(loss).backward()
            if self.cfg.grad_clip_norm and self.cfg.grad_clip_norm > 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(),
                                          max_norm=self.cfg.grad_clip_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            self._global_step += 1
            if self.ema is not None:
                self.ema.update(self._raw_model)

            loss_sum += float(loss.detach().item())
            if (it + 1) % self.cfg.log_interval == 0 or (it + 1) == n_iters:
                avg = loss_sum / (it + 1)
                # all_reduce is a collective: every rank must call it.
                if self.dist_env is not None and self.dist_env.enabled:
                    avg = all_reduce_mean(avg)
                if is_main_process(self.dist_env):
                    lr = self.optimizer.param_groups[0]["lr"]
                    elapsed = time.time() - t0
                    print(f"[train] ep{self._epoch} it{it+1}/{n_iters} "
                          f"loss={avg:.4f} lr={lr:.2e} t={elapsed:.0f}s")

    @torch.no_grad()
    def _validate(self) -> dict:
        eval_model = self.ema.module if self.ema is not None else self._raw_model
        eval_model.eval()
        # Validate only on rank 0 — eval is single-process for simplicity.
        if not is_main_process(self.dist_env):
            return {}
        return self.eval_fn(eval_model, self.val_loader)

    # ---- checkpointing -----------------------------------------------------

    def _save_checkpoint(self, path: Path, metrics: dict) -> None:
        if not is_main_process(self.dist_env):
            return
        state = {
            "model": self._raw_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "epoch": self._epoch,
            "global_step": self._global_step,
            "metrics": metrics,
        }
        if self.ema is not None:
            state["ema"] = self.ema.module.state_dict()
            state["ema_steps"] = int(self.ema.steps.item())
        torch.save(state, str(path))
        # rotate last-N
        ckpts = sorted(self.cfg.save_dir.glob("epoch_*.pth"))
        for old in ckpts[: -self.cfg.keep_last]:
            old.unlink(missing_ok=True)

    def _maybe_save_best(self, metrics: dict) -> None:
        if not is_main_process(self.dist_env):
            return
        key = self.cfg.save_best_metric
        if key not in metrics:
            return
        v = metrics[key]
        if self._best_metric is None or v > self._best_metric:
            self._best_metric = v
            best_path = self.cfg.save_dir / "best.pth"
            torch.save({"model": self._raw_model.state_dict(),
                        "ema": self.ema.module.state_dict() if self.ema else None,
                        "epoch": self._epoch,
                        "metrics": metrics},
                       str(best_path))
            print(f"[ckpt] new best {key}={v:.4f} -> {best_path}")


# ---- default eval (PCK on warped-back keypoints) ----------------------------

@torch.no_grad()
def default_pck_eval(model: nn.Module, loader: DataLoader, thr: float = 0.05) -> dict:
    """Cheap PCK evaluation that decodes predictions and compares to GT.

    Decodes per-family by inspecting the model output shape:
      - tuple of two tensors -> SimCC (RTMPose)
      - 4D tensor (B,K,H,W)  -> heatmap (HRNet/ViTPose)

    For PCK normalization we use bbox diagonal (close approximation to mmpose's
    default OKS-style normalization for top-down).
    """
    device = next(model.parameters()).device
    all_pred, all_gt, all_vis, all_norm = [], [], [], []

    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        out = model(x)

        # Decode per family
        if isinstance(out, tuple) and len(out) == 2 and out[0].ndim == 3:
            # SimCC
            simcc_x = out[0].float().cpu().numpy()  # (B, K, Wx)
            simcc_y = out[1].float().cpu().numpy()  # (B, K, Hy)
            x_locs = np.argmax(simcc_x, axis=-1).astype(np.float32)
            y_locs = np.argmax(simcc_y, axis=-1).astype(np.float32)
            # Default split ratio matches RTMPose body configs (2.0).
            split = simcc_x.shape[-1] / batch["image"].shape[-1]
            kp_input = np.stack([x_locs, y_locs], axis=-1) / split
        elif isinstance(out, torch.Tensor) and out.ndim == 4:
            hm = out.float().cpu().numpy()           # (B, K, H, W)
            B, K, H, W = hm.shape
            flat = hm.reshape(B * K, -1)
            argmax = np.argmax(flat, axis=1)
            y_locs = (argmax // W).astype(np.float32)
            x_locs = (argmax % W).astype(np.float32)
            kp_input = np.stack([x_locs, y_locs], axis=-1).reshape(B, K, 2)
            in_h, in_w = batch["image"].shape[-2:]
            kp_input[..., 0] *= in_w / W
            kp_input[..., 1] *= in_h / H
        else:
            raise RuntimeError("default_pck_eval: unrecognized model output shape")

        # Compare to keypoints in input space (already stored by collate_train).
        gt = batch["keypoints_input_space"].numpy()       # (B, K, 2)
        vis = batch["keypoints_visible"].numpy()
        # bbox diagonal in *image* space mapped to input space ~ input_size diag.
        in_h, in_w = batch["image"].shape[-2:]
        norm = np.full((kp_input.shape[0],), float(np.hypot(in_h, in_w)), dtype=np.float32)
        all_pred.append(kp_input)
        all_gt.append(gt)
        all_vis.append(vis)
        all_norm.append(norm)

    pred = np.concatenate(all_pred, axis=0)
    gt = np.concatenate(all_gt, axis=0)
    vis = np.concatenate(all_vis, axis=0)
    norm = np.concatenate(all_norm, axis=0)
    pck, _ = pck_accuracy(pred, gt, vis, norm, thr=thr)
    return {"val_pck": pck, "val_pck_thr": thr}
