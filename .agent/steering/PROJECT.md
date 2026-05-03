# Project

## Identity

This repository is the bootstrap workspace for an MLX-oriented library covering 3D and spatial model capabilities. The root library does not exist yet; the current repository content is steering state plus vendored reference projects.

## What This Repo Owns

Observed:

- `.agent/` owns project steering and workflow state for the bootstrap process.
- `vendors/` owns a reference corpus of upstream 3D/spatial projects: `HunyuanWorld-Mirror`, `HunyuanWorld-Voyager`, `sam-3d-objects`, `trellis-mac`, and `TRELLIS.2`.
- `vendors/trellis-mac` demonstrates a Python Apple Silicon adaptation of TRELLIS.2 image-to-3D generation with PyTorch MPS and Metal/fallback backends.
- `vendors/sam-3d-objects` demonstrates object mesh/texture/layout reconstruction from images and masks.
- `vendors/HunyuanWorld-Mirror` demonstrates feed-forward geometric prediction across depth, camera, normals, point clouds, and 3D Gaussians.
- `vendors/TRELLIS.2` provides an upstream O-Voxel/image-to-3D reference with CUDA/Linux assumptions.

Inferred:

- The root project should become a first-party MLX package rather than a collection of runnable vendor projects.
- Apple Silicon viability is likely central because the target is MLX and `vendors/trellis-mac` is an Apple Silicon port reference.

Needs confirmation:

- The initial public API surface and package name.
- Which model family should be ported or abstracted first.
- Whether vendored projects should remain long-term references or be pruned after extraction.

## Runtime Surfaces

Observed:

- No root runtime surface exists yet.
- Vendor CLI/script surfaces exist: `trellis-generate`, `python generate.py`, and `python demo.py`.

Not observed:

- No root CLI, API server, UI, worker, test runner, or build system was observed.

## Stack

Observed:

- Current code references are Python projects.
- `trellis-mac` declares Python `>=3.11` and dependencies including `torch`, `torchvision`, `transformers`, `accelerate`, `huggingface_hub`, `safetensors`, `pillow`, `numpy`, `trimesh`, `scipy`, and `tqdm`.
- `sam-3d-objects` uses Hatch/Hatchling packaging with dynamic requirements files.
- Upstream TRELLIS.2 documents Linux, NVIDIA GPU, CUDA Toolkit, Conda, and Python `>=3.8` assumptions.

## Decision Principles

- Preserve a clear boundary between first-party MLX library code and vendored reference code.
- Treat vendor commands as references, not root project commands, until a root manifest exists.
- Promote only file-backed or user-stated facts into planning; keep unproven direction marked as unknown.

## Evidence Anchors

- Root shape: `.`
- Vendor boundary: `vendors/`
- Apple Silicon TRELLIS reference: `vendors/trellis-mac/README.md`, `vendors/trellis-mac/pyproject.toml`
- SAM 3D Objects reference: `vendors/sam-3d-objects/README.md`, `vendors/sam-3d-objects/pyproject.toml`
- HunyuanWorld-Mirror reference: `vendors/HunyuanWorld-Mirror/README.md`
- Upstream TRELLIS.2 reference: `vendors/TRELLIS.2/README.md`
