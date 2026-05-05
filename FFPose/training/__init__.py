"""Training infrastructure: EMA, schedulers, metrics, trainer, DDP helpers."""
from .dist import (
    DistEnv,
    cleanup,
    init_distributed,
    is_main_process,
    make_train_sampler,
    maybe_wrap_ddp,
)
from .ema import ExpMomentumEMA
from .metrics import CocoKeypointMetric, pck_accuracy
from .pipeline_switch import PipelineSwitchHook
from .scheduler import build_lr_scheduler, layer_decay_param_groups
from .trainer import (
    PoseTrainer,
    TrainConfig,
    default_pck_eval,
    make_heatmap_loss_fn,
    make_simcc_loss_fn,
)

__all__ = [
    "ExpMomentumEMA",
    "build_lr_scheduler",
    "layer_decay_param_groups",
    "pck_accuracy",
    "CocoKeypointMetric",
    "PoseTrainer",
    "TrainConfig",
    "default_pck_eval",
    "make_simcc_loss_fn",
    "make_heatmap_loss_fn",
    "DistEnv",
    "init_distributed",
    "is_main_process",
    "maybe_wrap_ddp",
    "make_train_sampler",
    "cleanup",
    "PipelineSwitchHook",
]
