# FFPose

Pure-PyTorch 2.x re-implementation of the inference and training stack of
[OpenMMLab mmpose](https://github.com/open-mmlab/mmpose). **No `mmengine`,
`mmcv`, `mmdet`, or `mmpretrain` dependencies.** Loads existing mmpose
checkpoints unchanged.

Six top-down pose families are ported and verified end-to-end:

| Family       | Variants                                          | Loss          | Codec        |
|--------------|---------------------------------------------------|---------------|--------------|
| RTMPose      | t / s / m / l (body-17), m / l (wholebody-133)    | KLDiscretLoss | SimCC        |
| ViTPose      | small / base / small-simple / base-simple         | KeypointMSE   | UDP heatmap  |
| HRNet        | W32 / W48                                         | KeypointMSE   | MSRA heatmap |
| Swin         | t / b / l (windowed transformer)                  | KeypointMSE   | MSRA heatmap |
| HRFormer     | small / base (multi-resolution transformer)       | KeypointMSE   | MSRA heatmap |
| LiteHRNet    | 18 / 30 (mobile-friendly HRNet)                   | KeypointMSE   | MSRA heatmap |

Plus an end-to-end pipeline that uses `torchvision`'s Faster R-CNN to detect
people, then runs the top-down pose model on each detection.

## Install

```bash
git clone https://github.com/<your-username>/FFPose.git
cd FFPose
pip install -r requirements.txt
pip install -e .
```

Python 3.10+ and PyTorch 2.0+ are required (tested on PyTorch 2.10).

## Quick start — inference

```python
import cv2, numpy as np, FFPose

img = cv2.imread("photo.jpg")

# 1. Top-down with a pre-supplied bbox.
inf = FFPose.RTMPoseInferencer.from_pretrained(
    "body-coco-m", "/path/to/rtmpose-m.pth", device="cuda",
)
res = inf.predict(img, np.array([x1, y1, x2, y2], dtype=np.float32))
# res.keypoints (1, 17, 2) — image-space pixels
# res.scores    (1, 17)

# 2. End-to-end (detector + pose).
pipe = FFPose.TopDownPosePipeline.from_pretrained(
    "body-coco-m", "/path/to/rtmpose-m.pth",
    family="rtmpose", device="cuda",
)
out = pipe.predict_image(img)
# out.boxes (N,4)  out.keypoints (N,17,2)  out.keypoint_scores (N,17)

# Drop-in for any family:
pipe = FFPose.TopDownPosePipeline.from_pretrained("small", "ckpt.pth", family="vitpose")
pipe = FFPose.TopDownPosePipeline.from_pretrained("w32",   "ckpt.pth", family="hrnet")
pipe = FFPose.TopDownPosePipeline.from_pretrained("t",     "ckpt.pth", family="swin")
pipe = FFPose.TopDownPosePipeline.from_pretrained("small", "ckpt.pth", family="hrformer")
pipe = FFPose.TopDownPosePipeline.from_pretrained("30",    "ckpt.pth", family="litehrnet")

# Visualize.
viz = FFPose.draw_skeletons(img, out.keypoints, out.keypoint_scores,
                              boxes=out.boxes, score_thr=0.3)
cv2.imwrite("annotated.jpg", viz)
```

### Command-line inference

```bash
# Single image (or directory) end-to-end (detector + pose):
python tools/infer_image.py \
    --family rtmpose --variant body-coco-m \
    --checkpoint /path/to/rtmpose-m.pth \
    --image photo.jpg --out annotated.jpg

# Skip the detector by passing a bbox directly:
python tools/infer_image.py --family hrformer --variant small \
    --checkpoint hrformer-s.pth --image photo.jpg --out out.jpg \
    --bbox "x1,y1,x2,y2"

# Video, frame-by-frame:
python tools/infer_video.py --family swin --variant t \
    --checkpoint swin-t.pth --video input.mp4 --out output.mp4
```

### Test-time horizontal flip

ViTPose, HRNet, Swin, HRFormer, and LiteHRNet inferencers expose
`flip_test=True`. RTMPose's flip-test is always on (built into `predict()`).

```python
inf = FFPose.HRNetPoseInferencer.from_pretrained(
    "w32", "ckpt.pth", flip_test=True, shift_heatmap=True,
)
```

## Evaluation

`tools/eval.py` runs PCK and (optional) COCO-AP over a COCO-format annotation
file. Use `--family all` to compare every available backbone in one shot:

```bash
python tools/eval.py --family all \
    --checkpoint-dir /path/to/checkpoints \
    --ann-file /coco/annotations/person_keypoints_val2017.json \
    --img-root  /coco/val2017 \
    --pck-thr 0.05 \
    --coco-ap   # also computes pycocotools COCOeval AP/AP50/AP75/AR
```

The `--checkpoint-dir` mode looks for default-named files
(`rtmpose-m.pth`, `vitpose-s.pth`, `hrnet-w32.pth`, `swin-t.pth`,
`hrformer-s.pth`, `litehrnet-30.pth`) and skips any that aren't present.

Single-backbone:

```bash
python tools/eval.py --family hrformer --variant small \
    --checkpoint hrformer-s.pth \
    --ann-file ... --img-root ... \
    --max-samples 200    # quick smoke; 0 = full set
```

Output is a per-backbone summary table:

```
family     variant      n  PCK@0.05  mean_kpt_score  fps    AP      AP50    AP75    status
rtmpose    body-coco-m  12 0.9006    0.7188          32.95  -       -       -       ok
vitpose    small        12 0.9503    0.8298          92.52  -       -       -       ok
hrnet      w32          12 0.9116    0.7969          58.64  -       -       -       ok
swin       t            12 0.9448    0.7950          90.42  -       -       -       ok
hrformer   small        12 0.9006    0.7983          50.55  -       -       -       ok
litehrnet  30           12 0.9116    0.7381          44.40  -       -       -       ok
```

### Loading checkpoints

`FFPose` reads any mmpose `.pth` for the supported architectures. The custom
`safe_torch_load()` swaps unresolvable pickled metadata (e.g. mmengine's
`MessageHub`) for placeholders, so you don't need any mm-* package installed
at load time.

Get the pre-trained weights from the OpenMMLab CDN — see
`docs/checkpoints.md` (or the model-index files in the upstream mmpose repo).

## Training

Three entry scripts live in `tools/`. Each fine-tunes (or trains from scratch)
on COCO body-17 keypoints.

```bash
# RTMPose-m — SimCC head + KLDiscretLoss + EMA
python tools/train_rtmpose_m_coco.py \
    --variant m \
    --train-ann /path/to/coco/annotations/person_keypoints_train2017.json \
    --train-imgs /path/to/coco/train2017 \
    --val-ann   /path/to/coco/annotations/person_keypoints_val2017.json \
    --val-imgs   /path/to/coco/val2017 \
    --pretrained /path/to/rtmpose-m.pth \
    --batch-size 32 --epochs 10 --lr 1e-3 \
    --save-dir runs/rtmpose-m_coco_finetune

# HRNet-W32 — heatmap + MSE
python tools/train_hrnet_w32_coco.py --variant w32 ...

# ViTPose-small — UDP heatmap + MSE + ViT layer-wise LR decay
python tools/train_vitpose_s_coco.py --variant small ...
```

The training loop supports AMP, gradient clipping, EMA (RTMPose), cosine LR
with warmup, and best-checkpoint saving by validation PCK.

### Multi-GPU (DDP)

Use `torchrun`:

```bash
torchrun --nproc_per_node=4 tools/train_rtmpose_m_coco.py [args...]
```

If your host's NCCL is misconfigured, fall back to gloo:

```bash
MMPOSE_LITE_BACKEND=gloo torchrun --nproc_per_node=4 tools/train_rtmpose_m_coco.py [args...]
```

## Contributing

For an in-depth guide — what's been ported from mmpose, how the codebase is
organized, the contributor roadmap, and a worked walkthrough for adding a
new backbone — see [CONTRIBUTING.md](CONTRIBUTING.md).

## What's NOT included (yet)

- Architectures beyond the six above (RTMW, RTMO, DEKR, EDPose, ResNet,
  MobileNet, ConvNeXt, etc.). See [CONTRIBUTING.md §3](CONTRIBUTING.md#3-roadmap--what-still-needs-doing).
- 3D pose lifting (PoseLifter, MotionBERT).
- Bottom-up methods (AssociativeEmbedding, DEKR, CID).
- ONNX / TensorRT export.
- COCO-Wholebody-133 AP evaluation requires `xtcocotools`; install with
  `pip install ffpose[wholebody]`.
- `Albumentations` augmentations for the full RTMPose stage-1/2 recipe are
  optional; install with `pip install ffpose[albumentations]`.

## Module layout

```
FFPose/
├── layers.py, blocks.py             # ConvModule + CSPNeXt building blocks
├── backbone.py, hrnet.py, vit.py    # backbones
├── head.py, heatmap_head.py         # heads
├── codec.py, heatmap_codec.py       # SimCC / MSRA / UDP decoders
├── encoders.py                      # training-target generators
├── augmentations.py                 # RandomFlip / HalfBody / BBoxTransform / TopdownAffine
├── losses.py                        # KLDiscretLoss, KeypointMSELoss
├── coco_dataset.py                  # CocoKeypoints (pycocotools)
├── training_pipeline.py             # per-family pipeline composers
├── tta.py                           # test-time flip helpers
├── skeletons.py                     # COCO-17 / wholebody-133 metainfo
├── preprocess.py                    # bbox->affine warp utilities
├── inference.py, vitpose.py, hrnet_pose.py, model.py, pipeline.py, detector.py
└── training/
    ├── trainer.py    # PoseTrainer (AMP, EMA, grad-clip, eval)
    ├── ema.py        # ExpMomentumEMA (no mmengine)
    ├── scheduler.py  # warmup+cosine, ViT layer-wise LR
    ├── metrics.py    # PCK, COCO-AP via pycocotools
    └── dist.py       # torchrun-driven DDP
```

## License

Apache License 2.0. Portions derived from
[OpenMMLab mmpose](https://github.com/open-mmlab/mmpose), also Apache 2.0.

## Acknowledgements

The model architectures, training recipes, and submodule names are direct
ports of the corresponding files in mmpose. This repo's contribution is
removing the mm-* framework dependencies and modernizing the trainer for
PyTorch 2.x.
