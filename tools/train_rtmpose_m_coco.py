"""Fine-tune RTMPose-m on COCO body-17 keypoints.

Single-GPU training script. For multi-GPU, use ``torchrun`` with the
``--ddp`` flag (see tools/train_ddp_launcher.py).

Example:
    python tools/train_rtmpose_m_coco.py \
        --train-ann data/coco/annotations/person_keypoints_train2017.json \
        --train-imgs data/coco/train2017 \
        --val-ann data/coco/annotations/person_keypoints_val2017.json \
        --val-imgs data/coco/val2017 \
        --pretrained /tmp/mmpose_weights/rtmpose-m.pth \
        --batch-size 32 --epochs 10 --lr 1e-3 \
        --save-dir runs/rtmpose-m_coco_finetune
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Resolve FFPose. Default: assume it sits next to the directory that holds
# this script (e.g. ``<root>/FFPose`` and ``<root>/tools/...``). Override with
# ``FFPOSE_HOME`` if you've moved it elsewhere.
import os as _os
_ffpose_home = _os.environ.get("FFPOSE_HOME") or str(
    Path(__file__).resolve().parent.parent
)
sys.path.insert(0, _ffpose_home)

from FFPose import (
    RTMPOSE_COCO_256x192,
    RTMPose,
)
from FFPose.coco_dataset import CocoKeypoints
from FFPose.inference import _strip_state_dict, safe_torch_load
from FFPose.losses import KLDiscretLoss
from FFPose.skeletons import COCO_17
from FFPose.training import (
    PoseTrainer,
    TrainConfig,
    build_lr_scheduler,
    cleanup,
    init_distributed,
    is_main_process,
    make_simcc_loss_fn,
    make_train_sampler,
    maybe_wrap_ddp,
)
from FFPose.training_pipeline import collate_train, rtmpose_recipe


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--variant", default="m", choices=["t", "s", "m", "l"])
    p.add_argument("--train-ann", required=True)
    p.add_argument("--train-imgs", required=True)
    p.add_argument("--val-ann")
    p.add_argument("--val-imgs")
    p.add_argument("--pretrained", default=None,
                   help="Path to mmpose .pth checkpoint to warm-start from.")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--warmup-iters", type=int, default=500)
    p.add_argument("--save-dir", default="runs/rtmpose")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    env = init_distributed()

    cfg = RTMPOSE_COCO_256x192[args.variant]
    model = RTMPose(cfg)
    if args.pretrained:
        sd = _strip_state_dict(safe_torch_load(args.pretrained))
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if is_main_process(env):
            print(f"[init] loaded {args.pretrained}; "
                  f"missing={len(missing)} unexpected={len(unexpected)}")

    recipe = rtmpose_recipe(input_size=cfg.input_size)
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
        shuffle=(train_sampler is None),
        sampler=train_sampler,
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

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    total_iters = max(1, args.epochs * len(train_loader))
    scheduler = build_lr_scheduler(
        optimizer, total_iters=total_iters, warmup_iters=args.warmup_iters,
        schedule="cosine", cosine_min_factor=0.05,
    )
    loss_fn = make_simcc_loss_fn(KLDiscretLoss(beta=10.0, label_softmax=True,
                                                use_target_weight=True))

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
            use_ema=True,
            ema_momentum=0.0002,
            ema_gamma=2000,
            ema_update_buffers=True,
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
