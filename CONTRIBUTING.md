# Contributing to FFPose

FFPose is a pure-PyTorch 2.x re-implementation of the inference and training
stack of [OpenMMLab mmpose](https://github.com/open-mmlab/mmpose). The goal is
to provide modern, easy-to-read pose-estimation code that loads existing
mmpose checkpoints unchanged, with no `mmengine`/`mmcv`/`mmdet`/`mmpretrain`
dependency.

This document is for contributors. It covers:

1. [What's been ported, with a per-component diff against mmpose](#1-porting-log--what-changed-from-mmpose)
2. [Architecture map of the FFPose codebase](#2-codebase-map)
3. [The contributor roadmap (what's left to do)](#3-roadmap--what-still-needs-doing)
4. [How to add a new backbone (worked example)](#4-how-to-add-a-new-backbone)
5. [How to add a new head, codec, dataset, recipe](#5-other-extension-points)
6. [Testing & validation guide](#6-testing--validation)
7. [Style and conventions](#7-style--conventions)

If you're brand new to the repo, read sections 1-3 to understand scope, then
jump to whichever how-to section matches what you want to add.

---

## 1. Porting log — what changed from mmpose

Every change is listed concretely so contributors can map mmpose code onto
FFPose code mechanically.

### 1a. Dependencies

| | mmpose | FFPose |
|---|---|---|
| `mmengine` | required (Runner, Hook, Config, Registry, Dist, FileIO, EMA, …) | **removed** |
| `mmcv` | required (CNN bricks, transforms, ops) | **removed** |
| `mmdet` | required (RTMPose backbone, RTMO, detectors) | **removed** |
| `mmpretrain` | required (ViTPose backbone) | **removed** |
| `xtcocotools` | required for COCO eval | optional (`pip install ffpose[wholebody]`) |
| `albumentations` | required by some configs | optional (`pip install ffpose[albumentations]`) |
| `pycocotools` | indirect | direct (dataset + body-17 AP) |
| `torchvision` | optional | **used** (built-in detector) |
| PyTorch | 1.8+ | **2.0+** (tested on 2.10) |
| Python | 3.7+ | 3.10+ |

### 1b. Framework abstractions: replaced or dropped

| mmpose / mmengine concept | FFPose replacement |
|---|---|
| `Registry` (string→class lookup) | Direct imports + `@dataclass` configs. Submodule names match mmpose so checkpoints load. |
| `Config` (Python-file DSL with `_base_`, `_scope_`, `**.py` includes) | `@dataclass(frozen=True)` config objects (e.g., `RTMPoseConfig`, `ViTPoseConfig`). |
| `Runner` + `Hook` lifecycle | `FFPose.training.PoseTrainer` — plain torch loop with one `on_epoch_start` callback hook. |
| `BaseModule.init_cfg`-driven init | Plain `nn.Module.__init__` + checkpoint loading; no init-config indirection. |
| `MessageHub` / `Visualizer` (mmengine) | Out of scope; `print()` for now. |
| `mmengine.fileio` (S3, petrel backends) | `cv2.imread` / `pathlib`. |
| `mmengine.runner.load_checkpoint` | `FFPose.inference.safe_torch_load` — custom `pickle.Unpickler` that swaps unresolvable mm-pickled metadata for placeholders, so checkpoints load without any mm-* package installed. |
| `mmengine.dataset.{DefaultSampler, pseudo_collate}` | `torch.utils.data.DataLoader.shuffle` + `FFPose.training_pipeline.collate_train`. |
| `mmengine.optim.OptimWrapper` | `torch.optim.AdamW` / `Adam` + `torch.amp.GradScaler` directly. |
| `mmengine.optim.scheduler.*` (LinearLR + Cosine/MultiStep) | `FFPose.training.scheduler.build_lr_scheduler` (composes `torch.optim.lr_scheduler.SequentialLR`). |
| `mmpose.engine.optim_wrappers.LayerDecayOptimWrapperConstructor` | `FFPose.training.scheduler.layer_decay_param_groups`. |
| `mmengine.hooks.EMAHook` + `mmpose.engine.hooks.ExpMomentumEMA` | `FFPose.training.ema.ExpMomentumEMA` — standalone, no mmengine base class. |
| `mmengine.hooks.CheckpointHook` | Manual `torch.save` inside `PoseTrainer._save_checkpoint`. |
| `mmengine.hooks.LoggerHook` | `print()` per `log_interval`. |
| `mmdet.engine.hooks.PipelineSwitchHook` | `FFPose.training.PipelineSwitchHook` (rebuilds DataLoader at switch_epoch). |
| `mmpose.evaluation.metrics.CocoMetric` | `FFPose.training.metrics.CocoKeypointMetric` — pycocotools/xtcocotools wrapper. |
| `mmpose.evaluation.metrics.PCKAccuracy` | `FFPose.training.metrics.pck_accuracy`. |
| `mmengine.dist.*` | `FFPose.training.dist` — thin wrapper over `torch.distributed`, NCCL/gloo selectable. |

### 1c. Building blocks (mmcv → FFPose/layers.py)

| `mmcv.cnn` | `FFPose/layers.py` |
|---|---|
| `ConvModule` | `ConvModule` (param names match: `.conv`, `.bn`, `.activate`) |
| `DepthwiseSeparableConvModule` | `DepthwiseSeparableConvModule` |
| `DropPath` | `DropPath` |
| `build_conv_layer`, `build_norm_layer`, `build_activation_layer` | `_build_norm`, `_build_act` (private helpers in `layers.py`) |

Norm types supported: `BN` / `BN1d/2d/3d` / `SyncBN` (→ `BatchNorm`),
`GN`, `LN`, `IN`. Activations: `SiLU`/`Swish`, `ReLU`, `ReLU6`, `LeakyReLU`,
`GELU`, `HSigmoid`, `HSwish`, `Sigmoid`, `Tanh`.

### 1d. Backbones ported (6)

For each, submodule names match the mmpose checkpoint exactly so weights load
with `load_state_dict(..., strict=False)` returning **0 missing / 0 unexpected**.

| Backbone | mmpose source | FFPose path | Validated checkpoint |
|---|---|---|---|
| **CSPNeXt** | `mmpose/models/backbones/cspnext.py` (originally mmdet) | `FFPose/backbone.py` + `blocks.py` | `rtmpose-m_simcc-coco_*.pth` |
| **ViT** | `mmpretrain.VisionTransformer` | `FFPose/vit.py` | `td-hm_ViTPose-small_*.pth` |
| **HRNet** | `mmpose/models/backbones/{hrnet,resnet}.py` | `FFPose/hrnet.py` | `td-hm_hrnet-w32_*.pth` |
| **Swin** | `mmpose/models/backbones/swin.py` (V1, windowed) | `FFPose/swin.py` | `swin_t_p4_w7_coco_256x192-*.pth` |
| **HRFormer** | `mmpose/models/backbones/hrformer.py` | `FFPose/hrformer.py` | `hrformer_small_coco_256x192-*.pth` |
| **LiteHRNet** | `mmpose/models/backbones/litehrnet.py` | `FFPose/litehrnet.py` | `litehrnet30_coco_256x192-*.pth` |

**Subtle bits worth knowing:**
- `mmpose/models/utils/csp_layer.py:ChannelAttention` calls
  `with torch.cuda.amp.autocast(enabled=False)` around an avgpool. Modern
  PyTorch handles this automatically; the wrapper is unnecessary in our port.
- `LiteHRNet`'s fuse loop relies on `+=` in-place mutation of `out[0]` mid-loop
  to feed the mutated tensor into subsequent iterations. **Bit-exact replication
  is required** — see `_LiteHRModule.forward` in `FFPose/litehrnet.py`. The
  mmpose comment "y = 0 will lead to decreased accuracy (0.5~1 mAP)" hints at
  this.
- The 2022-era Swin pose checkpoint uses the older `keypoint_head.` state-dict
  prefix (mmpose 0.x) rather than `head.`. `_strip_state_dict` in
  `FFPose/inference.py` handles the rename.
- `HRFormer` reuses our `Bottleneck` from `hrnet.py` and the same stem
  layout, so the port is much smaller than the source's 758 LOC.

### 1e. Heads ported (2)

| | mmpose | FFPose |
|---|---|---|
| **RTMCCHead** (SimCC + GAU) | `mmpose/models/heads/coord_cls_heads/rtmcc_head.py` + `models/utils/rtmcc_block.py` + `models/utils/transformer.py:ScaleNorm` | `FFPose/head.py` |
| **HeatmapHead** | `mmpose/models/heads/heatmap_heads/heatmap_head.py` | `FFPose/heatmap_head.py` |

The remaining heads in mmpose (RegressionHead, RLEHead, MSPNHead, CIDHead,
DEKRHead, EDPoseHead, RTMOHead, RTMWHead, VisHead, …) are **not yet ported**;
see the [roadmap](#3-roadmap--what-still-needs-doing).

### 1f. Codecs

Both encoder (training-target generation) and decoder paths ported:

| Codec | mmpose source | FFPose decoder | FFPose encoder |
|---|---|---|---|
| **SimCC** | `mmpose/codecs/simcc_label.py` | `FFPose/codec.py:SimCCDecoder` | `FFPose/encoders.py:encode_simcc_gaussian` |
| **MSRAHeatmap** | `mmpose/codecs/msra_heatmap.py` | `FFPose/heatmap_codec.py:MSRAHeatmapDecoder` | `FFPose/encoders.py:encode_msra_heatmap` |
| **UDPHeatmap** | `mmpose/codecs/udp_heatmap.py` | `FFPose/heatmap_codec.py:UDPHeatmapDecoder` | `FFPose/encoders.py:encode_udp_heatmap` |

DARK / Dark-UDP refinement (`mmpose/codecs/utils/refinement.py`) is in
`FFPose/heatmap_codec.py`. Gaussian-blur and `get_heatmap_maximum` helpers are
inlined.

UDP requires a different bbox warp matrix (`get_udp_warp_matrix` vs the
standard `get_warp_matrix`). Both are in `FFPose/heatmap_codec.py` /
`FFPose/preprocess.py`.

### 1g. Augmentations

| mmpose | FFPose |
|---|---|
| `GetBBoxCenterScale` | `FFPose/augmentations.py:GetBBoxCenterScale` |
| `RandomFlip` (paired keypoint swap) | `FFPose/augmentations.py:RandomFlip` |
| `RandomHalfBody` | `FFPose/augmentations.py:RandomHalfBody` |
| `RandomBBoxTransform` (scale/rotate/shift jitter) | `FFPose/augmentations.py:RandomBBoxTransform` |
| `TopdownAffine` (training mode w/ rotation) | `FFPose/augmentations.py:TopdownAffine` |
| `mmdet.YOLOXHSVRandomAug` | `FFPose/augmentations.py:YOLOXHSVRandomAug` |
| `Albumentation` wrapper (Blur/MedianBlur/CoarseDropout) | `FFPose/augmentations.py:AlbumentationsWrap` + `build_rtmpose_albumentations_stage1` |
| `Compose` (dict-passing pipeline) | `FFPose/augmentations.py:Pipeline` (callable list, `Sample` dataclass) |

The `Sample` dataclass replaces mmpose's dict-passing convention. Each
augmentation is a `Sample → Sample` callable, composed with `Pipeline`.

### 1h. Losses

| | mmpose | FFPose |
|---|---|---|
| **KLDiscretLoss** (RTMPose SimCC) | `mmpose/models/losses/classification_loss.py` | `FFPose/losses.py` |
| **KeypointMSELoss** (HRNet/ViTPose) | `mmpose/models/losses/heatmap_loss.py` | `FFPose/losses.py` |

mmpose has many other losses (`KeypointL1Loss`, `RLELoss`, `BoneLoss`,
`SoftWingLoss`, `JSDiscretLoss`, `InfoNCELoss`, …) — **not yet ported**.

### 1i. Inference

| | mmpose | FFPose |
|---|---|---|
| Per-family inferencers | `mmpose.apis.inferencers.*` | `RTMPoseInferencer` / `ViTPoseInferencer` / `HRNetPoseInferencer` / `SwinPoseInferencer` / `HRFormerPoseInferencer` / `LiteHRNetPoseInferencer` |
| End-to-end pipeline | n/a (multi-step demo) | `FFPose.TopDownPosePipeline` (one call: image → poses) |
| Person detector | `mmdet` | `FFPose/detector.py` (torchvision Faster R-CNN v2) |
| Test-time flip | `mmpose/models/utils/tta.py` | `FFPose/tta.py:flip_heatmaps`, `flip_simcc_vectors` |
| Visualizer | `mmpose/visualization/local_visualizer.py` | `FFPose/visualization.py:draw_skeleton(s)` |

### 1j. Training infrastructure

| Component | mmpose location | FFPose path |
|---|---|---|
| Trainer | `mmengine.runner.Runner` + many Hooks | `FFPose/training/trainer.py:PoseTrainer` |
| EMA | `mmengine.hooks.EMAHook` + `ExpMomentumEMA` | `FFPose/training/ema.py` |
| LR schedulers | `mmengine.optim.scheduler.*` | `FFPose/training/scheduler.py:build_lr_scheduler` |
| Layer-wise LR decay | `LayerDecayOptimWrapperConstructor` | `scheduler.py:layer_decay_param_groups` |
| AMP / mixed precision | mmengine OptimWrapper | `torch.amp.autocast` + `GradScaler` |
| Gradient clipping | mmengine OptimWrapper | `nn.utils.clip_grad_norm_` |
| DDP | `mmengine.dist.*` + Runner | `FFPose/training/dist.py` (`init_distributed`, `maybe_wrap_ddp`, `make_train_sampler`, `all_reduce_mean`) |
| Pipeline switch | `mmdet.engine.hooks.PipelineSwitchHook` | `FFPose/training/pipeline_switch.py` |
| Per-recipe scripts | one `tools/train.py` + many configs | `tools/train_{rtmpose_m,hrnet_w32,vitpose_s}_coco.py` |

### 1k. Datasets

| | mmpose | FFPose |
|---|---|---|
| `CocoDataset` (body-17, wholebody-133) | `mmpose/datasets/datasets/body/coco_dataset.py` | `FFPose/coco_dataset.py` (pycocotools) |
| Other dataset readers | many (AIC, MPII, JHMDB, PoseTrack, OCHuman, CrowdPose, hand, face, animal, …) | **not ported** |

### 1l. New tooling that didn't exist in mmpose

These are FFPose additions, not ports:

- `tools/infer_image.py` — single image / directory inference, optional bbox.
- `tools/infer_video.py` — frame-by-frame video inference with skeleton overlay.
- `tools/eval.py` — multi-backbone PCK + COCO-AP evaluation in one table.
- `FFPose.TopDownPosePipeline` — one-call detector-plus-pose API.

---

## 2. Codebase map

```
FFPose/                              # repo root
├── README.md                        # user-facing docs
├── CONTRIBUTING.md                  # this file
├── LICENSE                          # Apache 2.0
├── pyproject.toml                   # package metadata
├── requirements.txt                 # runtime deps
├── .gitignore
│
├── FFPose/                          # the importable package
│   │
│   │  ── building blocks ──
│   ├── layers.py                    # ConvModule, DepthwiseSeparableConvModule, DropPath
│   ├── blocks.py                    # CSPNeXt blocks (CSPLayer, CSPNeXtBlock, ChannelAttention, SPP)
│   │
│   │  ── 6 backbones ──
│   ├── backbone.py                  # CSPNeXt
│   ├── vit.py                       # ViT (used by ViTPose)
│   ├── hrnet.py                     # HRNet (BasicBlock, Bottleneck, HRModule, HRNet)
│   ├── swin.py                      # Swin Transformer V1
│   ├── hrformer.py                  # HRFormer (HRNet + local-window transformer)
│   ├── litehrnet.py                 # LiteHRNet (lightweight HRNet)
│   │
│   │  ── 2 heads ──
│   ├── head.py                      # RTMCCHead (SimCC + GAU) + ScaleNorm + RoPE
│   ├── heatmap_head.py              # HeatmapHead (deconv + final conv)
│   │
│   │  ── codecs (encode + decode) ──
│   ├── codec.py                     # SimCCDecoder
│   ├── heatmap_codec.py             # MSRA / UDP decoders + refinement helpers
│   ├── encoders.py                  # encode_simcc / encode_msra / encode_udp
│   │
│   │  ── data ──
│   ├── coco_dataset.py              # CocoKeypoints (pycocotools)
│   ├── augmentations.py             # Sample, Pipeline, RandomFlip, RandomHalfBody, ...
│   ├── training_pipeline.py         # rtmpose_recipe / hrnet_recipe / vitpose_recipe + collate
│   ├── skeletons.py                 # COCO_17 / COCO_WHOLEBODY_133 metainfo
│   │
│   │  ── losses ──
│   ├── losses.py                    # KLDiscretLoss, KeypointMSELoss
│   │
│   │  ── inference ──
│   ├── inference.py                 # safe_torch_load, RTMPoseInferencer, _strip_state_dict
│   ├── model.py                     # RTMPose top-level
│   ├── vitpose.py                   # ViTPose model + inferencer
│   ├── hrnet_pose.py                # HRNetPose model + inferencer
│   ├── swin_pose.py                 # SwinPose model + inferencer
│   ├── hrformer_pose.py             # HRFormerPose model + inferencer
│   ├── litehrnet_pose.py            # LiteHRNetPose model + inferencer
│   ├── pipeline.py                  # TopDownPosePipeline (detector + pose)
│   ├── detector.py                  # PersonDetector (torchvision Faster R-CNN)
│   ├── preprocess.py                # bbox→affine→normalize utilities
│   ├── tta.py                       # flip_heatmaps, flip_simcc_vectors
│   ├── visualization.py             # draw_skeleton(s)
│   │
│   └── training/
│       ├── trainer.py               # PoseTrainer (AMP, EMA, grad-clip, eval)
│       ├── ema.py                   # ExpMomentumEMA
│       ├── scheduler.py             # build_lr_scheduler, layer_decay_param_groups
│       ├── metrics.py               # pck_accuracy, CocoKeypointMetric
│       ├── dist.py                  # DDP wrapper (NCCL/gloo)
│       └── pipeline_switch.py       # PipelineSwitchHook
│
└── tools/
    ├── infer_image.py               # single-image / batch inference
    ├── infer_video.py               # video inference
    ├── eval.py                      # PCK + COCO-AP eval
    ├── train_rtmpose_m_coco.py
    ├── train_hrnet_w32_coco.py
    └── train_vitpose_s_coco.py
```

**Naming convention**: each top-down-pose family gets a `<family>_pose.py`
that defines `<Family>PoseConfig`, `<Family>Pose` (the `nn.Module`), and
`<Family>PoseInferencer`. The bare backbone lives in a separate file
(`vit.py`, `hrnet.py`, `swin.py`, etc.) and is imported from there.

---

## 3. Roadmap — what still needs doing

Items are ordered roughly by user demand × tractability. Each is a
self-contained PR-sized chunk. Tag your PR with the rough size estimate so
reviewers can plan.

### 3a. High value, tractable (recommended first PRs)

- [ ] **RTMW** (RTMPose-wholebody multi-stage). Two-stage RTMPose head with
      separate body/face/hand classifiers. ~1 session. Source:
      `mmpose/models/heads/coord_cls_heads/rtmw_head.py`.
- [ ] **RTMO** (one-stage RTMPose, no detector needed). ~1-2 sessions. Source:
      `projects/rtmo/`. Significant: detector-free, regressing keypoints
      directly.
- [ ] **SimpleBaselines / ResNet** backbone. ~2-4 hours. Source:
      `mmpose/models/backbones/resnet.py` (we already have BasicBlock and
      Bottleneck from the HRNet port; just need the top-level `ResNet`).
- [ ] **MobileNetV2 / V3** backbones. ~2-4 hours each. Source:
      `mmpose/models/backbones/mobilenet_v{2,3}.py`. Useful for mobile/edge
      deployment.
- [ ] **`PipelineSwitchHook` end-to-end smoke test**. We have the hook but no
      integration test. Add a fixture that trains 5 epochs with a switch at
      epoch 3.
- [ ] **Visualizer enhancements**: per-instance color cycling for multi-person
      images, confidence-as-line-thickness option, draw bbox label.
- [ ] **TorchScript / ONNX export**. ~1 session per family. Some attention
      modules (e.g., LiteHRNet's in-place tensor mutation) need rewriting for
      tracing.
- [ ] **`Resume from checkpoint`**: load optimizer + scheduler + EMA + epoch
      from a saved `epoch_N.pth` and continue training. ~30 min in
      `PoseTrainer`.
- [ ] **WandB / TensorBoard logger**. Plug a callback into `PoseTrainer`.
      ~1 hour.
- [ ] **Unit tests** with `pytest`: at minimum, smoke-test each backbone's
      forward pass and key-equivalence to mmpose checkpoints. ~1 session.
- [ ] **GitHub Actions CI**: lint (ruff), mypy, the unit tests above, smoke
      training run on the bundled 12-instance test data. ~1-2 hours.

### 3b. Medium effort

- [ ] **Other heads**: `RegressionHead`, `RLEHead`, `IntegralRegressionHead`,
      `MSPNHead`. Each is ~0.5 session. Source:
      `mmpose/models/heads/regression_heads/`.
- [ ] **HRFormer config: 384x288 variant**. Add to `HRFORMER_POSE_COCO_*`
      with the right window-size/heatmap-size. 30 min.
- [ ] **MPII / AIC / CrowdPose datasets**. Each is a `CocoKeypoints` subclass
      that handles the schema differences. ~30 min - 2 hours each.
- [ ] **DDP NCCL fix**: investigate the host-shim warning ("NCCL/NET (shim)
      mismatch") that forced gloo fallback on the dev host. May be resolvable
      with `TORCH_NCCL_ASYNC_ERROR_HANDLING=1` propagation.
- [ ] **Gradient accumulation** in `PoseTrainer` (effective batch size > GPU
      capacity).
- [ ] **DeepSpeed / FSDP** support — for ViTPose-large/huge.
- [ ] **Loss zoo**: `KeypointL1Loss`, `RLELoss`, `BoneLoss`, `SoftWingLoss`,
      `JSDiscretLoss`. Several are tiny (~30 lines each).

### 3c. Larger ports / new architecture families

- [ ] **3D pose** (`PoseLifter`, MotionBERT, RTMPose3D). New model family —
      different I/O, different loss, different evaluation. Multi-session.
- [ ] **Bottom-up pose** (`AssociativeEmbedding`, `DEKR`, `CID`). The
      paradigm is fundamentally different from top-down: no per-instance
      crop, joint embedding for grouping. ~2-4 sessions per method.
- [ ] **DETR-style detection-pose** (EDPose). Needs `MultiScaleDeformableAttention`
      — currently the only mmcv custom CUDA op anywhere in the supported set.
      Pure-torch fallback exists upstream (in `torchvision.ops`).
- [ ] **More backbones**: HRFormer-base (validate the ckpt), Swin V2,
      ConvNeXt, EVA-02, Sapiens (Meta 2024 — not in mmpose, separate codebase).
- [ ] **Quantization** (PTQ/QAT) and pruning paths.

### 3d. Documentation

- [ ] **API reference** (sphinx or mkdocs).
- [ ] **Tutorial: fine-tune a custom keypoint set**. End-to-end example with a
      small dataset.
- [ ] **Tutorial: deploy a model to ONNX + ONNXRuntime**.
- [ ] **Benchmark page**: full COCO-val PCK + AP for every supported model
      vs. mmpose's published numbers, to confirm bit-exact equivalence.

### 3e. Known limitations / cleanup

- [ ] `xtcocotools` doesn't build under numpy 2.x (their issue, not ours);
      our error message points users at `pip install ffpose[wholebody]`. If
      that ever goes through, write a smoke test that runs wholebody-133 AP
      on the bundled `test_coco_wholebody.json`.
- [ ] **Convergence verification**: we've never actually fine-tuned a model
      from a released checkpoint long enough to confirm the trainer converges
      to mmpose's published numbers. Burn 1-2 GPU-days on COCO val to verify.
- [ ] **Remove `MMPOSE_LITE_BACKEND` env var** name (legacy) and switch to
      `FFPOSE_BACKEND`. Backward-compat alias either way.

---

## 4. How to add a new backbone

Walk-through using a hypothetical `Foo` backbone. The same recipe was used
for Swin, HRFormer, and LiteHRNet — see those PRs / files for reference.

### Step 1: Locate the source

In the upstream mmpose checkout: `mmpose/models/backbones/foo.py`. Read
through it. Identify:

- All `from mmcv.cnn import …` and `from mmengine.* import …` — these are
  the dependencies you'll need to replace.
- All submodules (especially nested ones inside `nn.ModuleList` /
  `nn.Sequential`) and their attribute names. **The submodule path of every
  parameter must match the checkpoint exactly.**
- Whether the backbone subclasses other mmpose backbones (e.g., HRFormer
  subclasses HRNet).

### Step 2: Find the checkpoint URL and download

Look in `configs/.../foo_coco.yml` for the `Weights:` URL. Download to
`/tmp/mmpose_weights/foo.pth` (or wherever).

### Step 3: Skeleton port

Create `FFPose/foo.py`. Use existing files as templates:

- Pure-CNN backbone → start from `FFPose/backbone.py` (CSPNeXt) or
  `FFPose/hrnet.py`.
- Transformer / windowed → start from `FFPose/swin.py` or `FFPose/vit.py`.
- HRNet variant → start from `FFPose/hrformer.py`.

Reuse helpers liberally:

- `FFPose/layers.py:ConvModule`, `DepthwiseSeparableConvModule`, `DropPath`,
  `_build_norm`, `_build_act`.
- `FFPose/hrnet.py:Bottleneck`, `BasicBlock`, `_make_layer` if you need
  HRNet-style components.

### Step 4: Diff the keys

The fastest way to verify the port is right is to load the checkpoint and
compare keys:

```python
import torch
from FFPose.inference import safe_torch_load, _strip_state_dict
from FFPose.foo import Foo

ckpt = safe_torch_load('/tmp/mmpose_weights/foo.pth')
sd = _strip_state_dict(ckpt)
backbone_keys = {k[len('backbone.'):] for k in sd if k.startswith('backbone.')}

m = Foo(...)
my_keys = set(dict(m.state_dict()).keys())
print('mine - ckpt:', sorted(my_keys - backbone_keys)[:5])
print('ckpt - mine:', sorted(backbone_keys - my_keys)[:5])
```

Iterate until both diffs are empty (or only contain known-benign extras like
`num_batches_tracked` from older checkpoints — see LiteHRNet's loader for the
filter pattern).

### Step 5: Wrap as a top-down pose model

Create `FFPose/foo_pose.py` mirroring `swin_pose.py` or
`hrformer_pose.py`. You need:

- `FooPoseConfig` (`@dataclass(frozen=True)`)
- `FOO_POSE_COCO_256x192` dict mapping variant strings → configs
- `FooPose(nn.Module)` combining the backbone with `HeatmapHead`
- `FooPoseInferencer` with `from_pretrained` and `predict` (copy-paste from
  an existing inferencer; the only differences are typically `flip_test`
  defaults and head channel count).

### Step 6: Plug into the public API

Add to `FFPose/__init__.py`:

```python
from .foo_pose import FOO_POSE_COCO_256x192, FooPose, FooPoseConfig, FooPoseInferencer
__all__.extend(["FooPose", "FooPoseConfig", "FOO_POSE_COCO_256x192", "FooPoseInferencer"])
```

And add `"foo": FooPoseInferencer.from_pretrained` to the `builders` dict in
`FFPose/pipeline.py` and to `_FAMILY_CHOICES` in `tools/infer_image.py`,
`tools/infer_video.py`, and `_BACKBONES` in `tools/eval.py`.

### Step 7: Validate

Run inference on the bundled test image and confirm reasonable accuracy:

```bash
python tools/eval.py --family foo --variant <variant> \
    --checkpoint /tmp/mmpose_weights/foo.pth \
    --ann-file ../mmpose/tests/data/coco/test_coco.json \
    --img-root  ../mmpose/tests/data/coco
```

Mean PCK@0.05 should be roughly comparable to the other backbones (~0.9 on
the 12-instance test set; on full COCO val it should match mmpose's
published numbers).

### Step 8: Update README + roadmap

- Add a row to the "supported families" table in `README.md`.
- Strike the corresponding bullet from this file's roadmap.
- Add a brief "subtle bits worth knowing" note if you discovered any
  in-place semantics, weird name remappings, or other gotchas.

---

## 5. Other extension points

### Adding a new head

1. Create a class in `FFPose/heatmap_head.py` (or a new file) following the
   `HeatmapHead` pattern: pure `nn.Module`, submodule names match mmpose's
   checkpoint format.
2. Wrap a top-down model that uses it (`FooPose` already does this for
   HeatmapHead). For a head-only contribution, just expose the class.
3. If it needs a new codec, add encoder + decoder to `encoders.py` /
   `heatmap_codec.py` (or `codec.py` for SimCC-style).
4. If it needs a new loss, add to `losses.py`.

### Adding a new dataset

1. Subclass `torch.utils.data.Dataset`, returning a `Sample` (from
   `FFPose.augmentations`).
2. Look at `FFPose/coco_dataset.py:CocoKeypoints` as a template. Most
   keypoint datasets in mmpose share the COCO JSON schema; you probably just
   need to override `_load_keypoints` and the upper/lower body partitions.
3. Add a `KeypointSchema` for the new keypoint set in `FFPose/skeletons.py`.

### Adding a new training recipe

1. Add a `<name>_recipe()` function to `FFPose/training_pipeline.py`. It
   should return a `TrainRecipe(train_pipeline, val_pipeline, family)`.
2. Add a per-recipe entry script in `tools/train_<name>_<dataset>.py`. Copy
   from `tools/train_rtmpose_m_coco.py` or `tools/train_hrnet_w32_coco.py`.

### Adding a new metric

1. Add a function to `FFPose/training/metrics.py`.
2. Hook it into `PoseTrainer.eval_fn` if you want it to drive
   "save best" — `PoseTrainer.cfg.save_best_metric` is a string key into the
   eval dict.

---

## 6. Testing & validation

### Quick smoke (every PR should pass this)

```bash
# 1. Inference on one image, all 6 backbones:
python tools/eval.py --family all \
    --checkpoint-dir /tmp/mmpose_weights \
    --ann-file ../mmpose/tests/data/coco/test_coco.json \
    --img-root  ../mmpose/tests/data/coco

# 2. Single-GPU training loop, all 3 recipes, 1 epoch each:
python tools/train_rtmpose_m_coco.py \
    --train-ann ../mmpose/tests/data/coco/test_coco.json \
    --train-imgs ../mmpose/tests/data/coco \
    --val-ann ../mmpose/tests/data/coco/test_coco.json \
    --val-imgs ../mmpose/tests/data/coco \
    --pretrained /tmp/mmpose_weights/rtmpose-m.pth \
    --batch-size 2 --epochs 1 --warmup-iters 1 --num-workers 0 \
    --save-dir /tmp/runs/rtmpose-smoke
# (same for HRNet, ViTPose)

# 3. DDP smoke (only if you touched training/dist.py):
CUDA_VISIBLE_DEVICES=0,1 MMPOSE_LITE_BACKEND=gloo \
    torchrun --nproc_per_node=2 tools/train_rtmpose_m_coco.py [...same args, --batch-size 1]
```

### Full benchmark (claim-to-fame validation)

If you're claiming bit-exact match with mmpose's published accuracy:

```bash
python tools/eval.py --family <yours> --variant <yours> \
    --checkpoint <yours> \
    --ann-file /coco/annotations/person_keypoints_val2017.json \
    --img-root  /coco/val2017 \
    --coco-ap
```

Should match mmpose's published COCO-val AP within 0.1 absolute (small
rounding-mode / decode-edge differences are normal).

---

## 7. Style & conventions

- **Python**: 3.10+. Use `from __future__ import annotations` for forward
  references.
- **Type hints**: required on public APIs (functions/classes used outside the
  defining module). Internal helpers can be light on hints.
- **Comments**: explain the *why* (subtle invariants, mmpose-quirk
  reproductions, in-place semantics) — not the *what*. Reading code
  shouldn't require knowing mmpose, so explain when behavior is
  non-obvious-from-pytorch.
- **Submodule naming for ports**: must match the upstream mmpose checkpoint
  attribute path *exactly*. Use docstring tables (see `FFPose/swin.py` top
  comment) to make this explicit.
- **Line length**: 100 cols. (`ruff` default.)
- **Imports**: standard, third-party, local — separated by blank lines.
- **`black` / `ruff`**: not enforced yet, but PRs that follow them will be
  easier to review.
- **No mm-* imports**: violating this is the one hard rule. If you find you
  need an mm-* helper, port it instead. The `_strip_state_dict` /
  `safe_torch_load` pattern is enough to load any mmpose checkpoint without
  mmengine present.

---

## License

Apache 2.0. Files derived from mmpose (which is also Apache 2.0) preserve
attribution in their docstrings: every ported file says "Direct port of
mmpose's …" near the top.

When you contribute new ported code, add a similar one-line attribution to
the source path. When you contribute original code, no attribution needed.

---

Thanks for contributing! Open issues for design questions before sending a
big PR. For trivial fixes (typos, missing docstrings, a small feature like
a new variant config) just send the PR.
