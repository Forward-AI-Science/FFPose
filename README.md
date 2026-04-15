<div align="center">

# 🦾 FFPose — Forward Pose

**A modern pose estimation framework — clean, dependency-light, and always up-to-date.**

[![License](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen)](CONTRIBUTING.md)
[![Discord](https://img.shields.io/badge/Discord-Join%20us-5865F2?logo=discord&logoColor=white)](https://discord.gg/22gbzUad)

*Maintained by the [Forward AI Science](https://github.com/Forward-AI-Science) community.*

</div>

---

## What is FFPose?

**FFPose (Forward Pose)** is an open-source pose estimation framework inspired by [MMPose](https://github.com/open-mmlab/mmpose), but designed to stay current and easy to use.

MMPose is an excellent research framework, but its tight coupling to `mmcv` and `mmengine` introduces significant friction: frequent compatibility breaks with new PyTorch and CUDA releases, complex installation steps, and an ever-growing dependency tree. For many practitioners, keeping a working MMPose environment is itself a maintenance burden.

FFPose is our answer to that problem. It takes the best ideas from MMPose — a clean separation of codecs, datasets, backbones, and heads — and rebuilds them on **vanilla PyTorch** with a minimal, stable set of dependencies. The goal is a framework that a researcher can install in seconds, understand in an afternoon, and extend without fighting the tooling.

---

## Motivation

| Pain point in MMPose | FFPose approach |
|---|---|
| Hard dependency on `mmcv` & `mmengine` | Pure PyTorch — no custom ops required |
| Frequent version incompatibilities | Tracks latest stable PyTorch and CUDA |
| Complex source builds | Standard `pip install` |
| Custom registry & config DSL | Plain Python / YAML |
| Heavy abstraction layers | Thin, readable, hackable code |

---

## Status

> 🚧 **FFPose is in early development.** The repository is being actively built out.
> Star or watch to follow progress.

We are working on:

- Core package structure and coding conventions
- Codec implementations (Heatmap, SimCC, RLE)
- Dataset loaders (COCO, MPII, Human3.6M)
- Model zoo with pretrained weights
- Training recipes and documentation

---

## Community

FFPose is maintained by the **[Forward AI Science](https://github.com/Forward-AI-Science)** community — an open group of researchers and engineers who believe that good tools should be simple, transparent, and accessible to everyone.

💬 Join the conversation on **[Discord](https://discord.gg/22gbzUad)** — share ideas, ask questions, and follow development in real time.

We welcome contributions of all kinds: code, documentation, benchmarks, bug reports, and ideas. A `CONTRIBUTING.md` guide will be published alongside the first stable release.

---

## License

FFPose is released under the [Apache 2.0 License](LICENSE).

---

<div align="center">
Made with ❤️ by the <a href="https://github.com/Forward-AI-Science">Forward AI Science</a> community
</div>
