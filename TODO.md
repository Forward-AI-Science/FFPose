# FFPose — Roadmap & TODO

> Estimates assume **1 developer using an AI-assisted coding IDE** (e.g. Cursor, Windsurf, Copilot).
> Times reflect focused working hours, not calendar days.

---

## 🗺️ Roadmap

```
Phase 1 ──► Phase 2 ──► Phase 3 ──► Phase 4 ──► Phase 5 ──► Phase 6
Foundation   Core         Training     Model Zoo   Export &     Docs &
& Packaging  Framework    Pipeline     & Weights   Production   Community
  ~1 week     ~2 weeks     ~2 weeks    ~3 weeks     ~1 week      ongoing
```

---

### Phase 1 — Foundation & Packaging `~1 week`

Get the repo healthy so contributors can land without friction.

| # | Task | Category |
|---|---|---|
| 1.1 | Set up `pyproject.toml` with correct metadata and optional deps | Packaging |
| 1.2 | Create `ffpose/` package skeleton (all sub-packages with `__init__.py`) | Packaging |
| 1.3 | Add `.gitignore` (Python, venv, wandb, checkpoints) | Repo |
| 1.4 | Add `CONTRIBUTING.md` with code style, PR, and branch conventions | Repo |
| 1.5 | Configure `ruff` for linting + formatting | Tooling |
| 1.6 | Configure `pre-commit` hooks (ruff, trailing whitespace, end-of-file) | Tooling |
| 1.7 | Set up GitHub Actions CI: lint + unit tests on push | CI/CD |
| 1.8 | Add pytest skeleton with one smoke test | Testing |
| 1.9 | Add `CHANGELOG.md` | Repo |

---

### Phase 2 — Core Framework `~2 weeks`

The building blocks every model and dataset will depend on.

| # | Task | Category |
|---|---|---|
| 2.1 | `ffpose/structures.py` — `Keypoints` and `PoseResult` dataclasses | Core |
| 2.2 | `ffpose/codecs/heatmap.py` — Gaussian heatmap encode/decode | Codec |
| 2.3 | `ffpose/codecs/simcc.py` — SimCC 1-D classification codec | Codec |
| 2.4 | `ffpose/codecs/rle.py` — RLE (Residual Log-likelihood) codec | Codec |
| 2.5 | `ffpose/codecs/__init__.py` — codec registry (plain dict, no magic) | Codec |
| 2.6 | `ffpose/datasets/base.py` — abstract `PoseDataset` with common transforms | Dataset |
| 2.7 | `ffpose/datasets/coco.py` — COCO keypoint dataset (17 kp) | Dataset |
| 2.8 | `ffpose/datasets/coco_wholebody.py` — COCO-WholeBody (133 kp) | Dataset |
| 2.9 | `ffpose/datasets/mpii.py` — MPII Human Pose dataset | Dataset |
| 2.10 | `ffpose/datasets/transforms.py` — augmentation pipeline (albumentations) | Dataset |
| 2.11 | `ffpose/models/backbones/timm_backbone.py` — thin `timm` wrapper | Backbone |
| 2.12 | `ffpose/models/backbones/cspnext.py` — CSPNeXt (RTMPose backbone) | Backbone |
| 2.13 | `ffpose/models/necks/fpn.py` — simple FPN neck | Neck |
| 2.14 | `ffpose/models/heads/heatmap_head.py` — classic heatmap head + loss | Head |
| 2.15 | `ffpose/models/heads/simcc_head.py` — SimCC head + loss | Head |
| 2.16 | `ffpose/models/heads/rtmpose_head.py` — RTMPose head (SimCC + DCC) | Head |
| 2.17 | `ffpose/models/pose_estimator.py` — top-level model assembler | Model |
| 2.18 | `ffpose/losses/` — KpLoss, OKSLoss, RLELoss | Losses |

---

### Phase 3 — Training & Evaluation Pipeline `~2 weeks`

A clean, native-PyTorch training loop — no hidden runners.

| # | Task | Category |
|---|---|---|
| 3.1 | `ffpose/trainer.py` — training loop (AMP, grad clip, EMA, cosine LR) | Training |
| 3.2 | Config system — plain dataclasses + YAML loader | Config |
| 3.3 | `configs/rtmpose_m_coco.yaml` — first reference training config | Config |
| 3.4 | `tools/train.py` — CLI entry point | Tools |
| 3.5 | `tools/test.py` — evaluation CLI | Tools |
| 3.6 | `ffpose/metrics/oks.py` — OKS / AP / AR computation | Metrics |
| 3.7 | `ffpose/metrics/pck.py` — PCK metric | Metrics |
| 3.8 | Weights & Biases / TensorBoard logging integration (optional flag) | Training |
| 3.9 | Checkpoint save / resume logic | Training |
| 3.10 | Unit tests: codec encode↔decode round-trip | Testing |
| 3.11 | Unit tests: dataset `__getitem__` shape checks | Testing |
| 3.12 | Unit tests: model forward pass (random input → correct output shape) | Testing |
| 3.13 | Integration test: single training step with dummy data | Testing |

---

### Phase 4 — Model Zoo & Pretrained Weights `~3 weeks`

Train, benchmark, and release the first set of models.

| # | Task | Category |
|---|---|---|
| 4.1 | Train RTMPose-t on COCO (256×192) — establish baseline AP | Training |
| 4.2 | Train RTMPose-s on COCO (256×192) | Training |
| 4.3 | Train RTMPose-m on COCO (256×192) | Training |
| 4.4 | Train RTMPose-l on COCO (256×192) | Training |
| 4.5 | Upload weights to Hugging Face Hub (`forward-ai-science/ffpose-models`) | Release |
| 4.6 | `ffpose/api.py` — `from_pretrained()` with auto weight download | API |
| 4.7 | `ffpose/visualization.py` — `draw_pose()`, `draw_heatmap()` helpers | Visualization |
| 4.8 | `tools/demo.py` — webcam / video / image inference demo | Tools |
| 4.9 | Benchmark vs. MMPose equivalents (AP, speed, memory) | Benchmarking |
| 4.10 | Add H36M dataset + 3D lifting baseline (video Transformer) | 3D Pose |
| 4.11 | Train MPII model and release weights | Training |

---

### Phase 5 — Export & Production `~1 week`

Make inference deployable outside a research environment.

| # | Task | Category |
|---|---|---|
| 5.1 | `ffpose/exporter.py` — ONNX export with dynamic batch/spatial axes | Export |
| 5.2 | `tools/export.py` — CLI entry point for export | Tools |
| 5.3 | ONNX inference example + validation against PyTorch output | Export |
| 5.4 | TorchScript (`torch.jit.script`) export path | Export |
| 5.5 | `torch.compile` smoke test + performance report | Optimization |
| 5.6 | TensorRT INT8 quantization guide (documentation) | Optimization |
| 5.7 | Docker image with minimal runtime dependencies | Deployment |

---

### Phase 6 — Documentation & Community `ongoing`

| # | Task | Category |
|---|---|---|
| 6.1 | Set up MkDocs / GitHub Pages site | Docs |
| 6.2 | Write: Installation guide | Docs |
| 6.3 | Write: Prepare Datasets guide | Docs |
| 6.4 | Write: Train a model end-to-end | Docs |
| 6.5 | Write: Add a custom dataset | Docs |
| 6.6 | Write: Add a custom model / head | Docs |
| 6.7 | Write: Export to ONNX | Docs |
| 6.8 | Write: Codec design internals | Docs |
| 6.9 | Gradio / Hugging Face Spaces web demo | Demo |
| 6.10 | Issue templates (bug report, feature request, model request) | Community |
| 6.11 | Discord community onboarding pinned message | Community |
| 6.12 | First public release: tag `v0.1.0` and write release notes | Release |

---

## 📋 All Tasks by Category

<details>
<summary><strong>Packaging & Repo Setup</strong></summary>

- [ ] 1.1 `pyproject.toml`
- [ ] 1.2 Package skeleton
- [ ] 1.3 `.gitignore`
- [ ] 1.4 `CONTRIBUTING.md`
- [ ] 1.9 `CHANGELOG.md`

</details>

<details>
<summary><strong>Tooling & CI</strong></summary>

- [ ] 1.5 Ruff configuration
- [ ] 1.6 pre-commit hooks
- [ ] 1.7 GitHub Actions (lint + test)

</details>

<details>
<summary><strong>Codecs</strong></summary>

- [ ] 2.2 Heatmap codec
- [ ] 2.3 SimCC codec
- [ ] 2.4 RLE codec
- [ ] 2.5 Codec registry

</details>

<details>
<summary><strong>Datasets</strong></summary>

- [ ] 2.6 Base dataset class
- [ ] 2.7 COCO (17 kp)
- [ ] 2.8 COCO-WholeBody (133 kp)
- [ ] 2.9 MPII
- [ ] 2.10 Augmentation transforms
- [ ] 4.10 Human3.6M (3D)

</details>

<details>
<summary><strong>Models (Backbone / Neck / Head)</strong></summary>

- [ ] 2.11 timm backbone wrapper
- [ ] 2.12 CSPNeXt backbone
- [ ] 2.13 FPN neck
- [ ] 2.14 Heatmap head
- [ ] 2.15 SimCC head
- [ ] 2.16 RTMPose head
- [ ] 2.17 Top-level pose estimator
- [ ] 2.18 Loss functions

</details>

<details>
<summary><strong>Training</strong></summary>

- [ ] 3.1 Trainer (AMP, EMA, LR schedule)
- [ ] 3.2 Config system
- [ ] 3.3 RTMPose-m reference config
- [ ] 3.8 Logger integration (W&B / TensorBoard)
- [ ] 3.9 Checkpoint / resume
- [ ] 4.1–4.4 Train RTMPose-t/s/m/l on COCO
- [ ] 4.11 Train MPII model

</details>

<details>
<summary><strong>Testing</strong></summary>

- [ ] 1.8 Pytest skeleton
- [ ] 3.10 Codec round-trip tests
- [ ] 3.11 Dataset shape tests
- [ ] 3.12 Model forward pass tests
- [ ] 3.13 Integration: single training step

</details>

<details>
<summary><strong>Metrics & Evaluation</strong></summary>

- [ ] 3.6 OKS / AP / AR
- [ ] 3.7 PCK
- [ ] 4.9 Benchmark vs MMPose

</details>

<details>
<summary><strong>API & Visualization</strong></summary>

- [ ] 2.1 Structures (Keypoints, PoseResult)
- [ ] 4.6 `from_pretrained()` API
- [ ] 4.7 `draw_pose()` / `draw_heatmap()`

</details>

<details>
<summary><strong>Tools & CLI</strong></summary>

- [ ] 3.4 `tools/train.py`
- [ ] 3.5 `tools/test.py`
- [ ] 4.8 `tools/demo.py`
- [ ] 5.2 `tools/export.py`

</details>

<details>
<summary><strong>Export & Deployment</strong></summary>

- [ ] 5.1 ONNX exporter
- [ ] 5.3 ONNX validation
- [ ] 5.4 TorchScript export
- [ ] 5.5 `torch.compile` test
- [ ] 5.6 TensorRT guide
- [ ] 5.7 Docker image

</details>

<details>
<summary><strong>Documentation</strong></summary>

- [ ] 6.1 MkDocs / GitHub Pages setup
- [ ] 6.2 Installation guide
- [ ] 6.3 Datasets guide
- [ ] 6.4 Training guide
- [ ] 6.5 Custom dataset guide
- [ ] 6.6 Custom model guide
- [ ] 6.7 ONNX export guide
- [ ] 6.8 Codec internals

</details>

<details>
<summary><strong>Community & Release</strong></summary>

- [ ] 4.5 Upload weights to Hugging Face Hub
- [ ] 6.9 Gradio / HF Spaces web demo
- [ ] 6.10 GitHub issue templates
- [ ] 6.11 Discord onboarding
- [ ] 6.12 Tag `v0.1.0` + release notes

</details>
