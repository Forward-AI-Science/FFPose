"""Fine-tune ViTPose-small on COCO body-17 (UDP heatmap, MSE loss).

Uses layer-wise LR decay (rate 0.75) on the ViT backbone — matches mmpose's
LayerDecayOptimWrapperConstructor.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import os as _os
_ffpose_home = _os.environ.get("FFPOSE_HOME") or str(
    Path(__file__).resolve().parent.parent
)
sys.path.insert(0, _ffpose_home)

from FFPose import VITPOSE_COCO_256x192, ViTPose
from FFPose.coco_dataset import CocoKeypoints
from FFPose.inference import _strip_state_dict, safe_torch_load
from FFPose.losses import KeypointMSELoss
from FFPose.skeletons import COCO_17
from FFPose.training import (
    PoseTrainer,
    TrainConfig,
    build_lr_scheduler,
    cleanup,
    init_distributed,
    is_main_process,
    layer_decay_param_groups,
    make_heatmap_loss_fn,
    make_train_sampler,
    maybe_wrap_ddp,
)
from FFPose.training_pipeline import collate_train, vitpose_recipe


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--variant", default="small",
                   choices=["small", "base", "small-simple", "base-simple"])
    p.add_argument("--train-ann", required=True)
    p.add_argument("--train-imgs", required=True)
    p.add_argument("--val-ann")
    p.add_argument("--val-imgs")
    p.add_argument("--pretrained", default=None)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--layer-decay", type=float, default=0.75)
    p.add_argument("--warmup-iters", type=int, default=500)
    p.add_argument("--save-dir", default="runs/vitpose")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    env = init_distributed()

    cfg = VITPOSE_COCO_256x192[args.variant]
    model = ViTPose(cfg)
    if args.pretrained:
        sd = _strip_state_dict(safe_torch_load(args.pretrained))
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if is_main_process(env):
            print(f"[init] loaded {args.pretrained}; "
                  f"missing={len(missing)} unexpected={len(unexpected)}")

    recipe = vitpose_recipe(input_size=cfg.input_size,
                            heatmap_size=cfg.heatmap_size)
    train_ds = CocoKeypoints(
        ann_file=args.train_ann, img_root=args.train_imgs,
        schema=COCO_17, transform=recipe.train_pipeline,
    )
    val_ds = None
    if args.val_ann and args.val_imgs:
        val_ds = CocoKeypoints(
            ann_file=args.val_ann, img_root=args.val_imgs,
            schema=COCO_17, transform=recipe.val_pipeline,
        )

    train_sampler = make_train_sampler(train_ds, env)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=(train_sampler is None), sampler=train_sampler,
        num_workers=args.num_workers, persistent_workers=args.num_workers > 0,
        collate_fn=collate_train, pin_memory=True, drop_last=True,
    )
    val_loader = None
    if val_ds is not None and is_main_process(env):
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, collate_fn=collate_train,
        )

    model = maybe_wrap_ddp(model.to(env.device), env)

    raw_model = model.module if hasattr(model, "module") else model
    num_layers = raw_model.backbone.num_layers
    param_groups = layer_decay_param_groups(
        raw_model, base_lr=args.lr, weight_decay=args.weight_decay,
        num_layers=num_layers, layer_decay_rate=args.layer_decay,
    )
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr,
                                   weight_decay=args.weight_decay,
                                   betas=(0.9, 0.999))
    total_iters = max(1, args.epochs * len(train_loader))
    scheduler = build_lr_scheduler(
        optimizer, total_iters=total_iters, warmup_iters=args.warmup_iters,
        schedule="cosine", cosine_min_factor=0.05,
    )
    loss_fn = make_heatmap_loss_fn(KeypointMSELoss(use_target_weight=True))

    trainer = PoseTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        device=args.device,
        dist_env=env,
        config=TrainConfig(
            num_epochs=args.epochs,
            save_dir=Path(args.save_dir),
            use_amp=True,
            use_ema=False,
            grad_clip_norm=1.0,
        ),
    )
    try:
        history = trainer.train()
        if is_main_process(env):
            print("=== training done ===")
            for h in history:
                print(h)
    finally:
        cleanup()


if __name__ == "__main__":
    main()
