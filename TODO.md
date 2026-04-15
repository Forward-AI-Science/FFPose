# FFPose — Roadmap & TODO

> **Strategy:** Port and adapt directly from MMPose source using an agentic AI coding IDE.
> **Target:** ~1 month, 1 developer.

---

## 🗺️ Roadmap

| Phase | Focus | Estimate |
|---|---|---|
| **Phase 1** | Repo setup & package skeleton | Week 1 |
| **Phase 2** | Core framework — codecs, datasets, models | Weeks 1–2 |
| **Phase 3** | Training pipeline, evaluation & first models | Weeks 2–3 |
| **Phase 4** | Model zoo, export, docs & first release | Weeks 3–4 |

---

## Phase 1 — Repo Setup `Week 1`

- [ ] Package structure & `pyproject.toml`
- [ ] Linting, formatting, pre-commit hooks
- [ ] GitHub Actions CI (lint + tests)
- [ ] `CONTRIBUTING.md` and `CHANGELOG.md`

---

## Phase 2 — Core Framework `Weeks 1–2`

### Codecs
- [ ] Heatmap codec
- [ ] SimCC codec
- [ ] RLE codec

### Datasets
- [ ] Base dataset class & augmentation pipeline
- [ ] COCO (17 kp)
- [ ] COCO-WholeBody (133 kp)
- [ ] MPII
- [ ] Human3.6M (3D)

### Model Architecture
- [ ] Backbone wrapper (timm + CSPNeXt)
- [ ] Neck (FPN)
- [ ] Heatmap head
- [ ] SimCC / RTMPose head
- [ ] Top-level pose estimator

### Losses
- [ ] Keypoint loss (MSE / L1)
- [ ] OKS loss
- [ ] RLE loss

---

## Phase 3 — Training & Evaluation `Weeks 2–3`

- [ ] Core training loop (AMP, gradient clipping, EMA)
- [ ] Learning rate scheduler
- [ ] Config system (YAML + dataclasses)
- [ ] Checkpoint save & resume
- [ ] Experiment logging (W&B / TensorBoard)
- [ ] OKS / AP / AR metrics
- [ ] PCK metric
- [ ] Evaluation CLI (`tools/test.py`)
- [ ] Training CLI (`tools/train.py`)
- [ ] Unit & integration tests

---

## Phase 4 — Model Zoo, Export & Release `Weeks 3–4`

- [ ] Train & release RTMPose-t/s/m/l on COCO
- [ ] `from_pretrained()` API + Hugging Face Hub upload
- [ ] Inference demo (`tools/demo.py`)
- [ ] Pose visualization utilities
- [ ] ONNX export
- [ ] TorchScript export
- [ ] Docker image
- [ ] Documentation site (MkDocs / GitHub Pages)
- [ ] Gradio / HF Spaces web demo
- [ ] Tag `v0.1.0` + release notes
