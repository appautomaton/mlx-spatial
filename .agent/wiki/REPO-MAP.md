# Repo Map

## Conclusion

This repository is at bootstrap stage for an MLX-oriented 3D/spatial model library. The root has no application or package manifest yet; current concrete content is `.agent/` steering state and a `vendors/` reference corpus of upstream 3D/spatial projects.

## Sources Read

| Path | Why it matters |
|---|---|
| `.` | Shows the root currently contains `.agent/`, `.claude/`, `.opencode/`, `.git/`, and `vendors/`, with no root README or package manifest observed. |
| `.agent/steering/STATUS.md` | Existing steering file was scaffold text only. |
| `.agent/steering/PROJECT.md` | Existing steering file was scaffold text only. |
| `vendors/` | Shows five vendored reference projects at one boundary. |
| `vendors/trellis-mac/README.md` | Apple Silicon image-to-3D reference and concrete Mac/MPS adaptation details. |
| `vendors/trellis-mac/pyproject.toml` | Python package metadata and CLI script evidence for the Apple Silicon TRELLIS port. |
| `vendors/sam-3d-objects/README.md` | Single/multi-object 3D reconstruction reference surface. |
| `vendors/sam-3d-objects/pyproject.toml` | Python package metadata for the SAM 3D object reconstruction reference. |
| `vendors/HunyuanWorld-Mirror/README.md` | Geometric prediction reference across depth, camera, normals, point clouds, and 3D Gaussians. |
| `vendors/TRELLIS.2/README.md` | Upstream TRELLIS.2 sparse voxel/image-to-3D reference and CUDA/Linux constraints. |

## Observed Topology

| Area | Observed shape | Evidence |
|---|---|---|
| Root project | Bootstrap workspace with no root README or root package manifest observed. | `.` |
| Agent state | Automaton steering exists but was scaffold-only before this refresh. | `.agent/steering/STATUS.md`, `.agent/steering/PROJECT.md` |
| Reference corpus | Five sibling vendor projects: `HunyuanWorld-Mirror`, `HunyuanWorld-Voyager`, `sam-3d-objects`, `trellis-mac`, `TRELLIS.2`. | `vendors/` |
| Apple Silicon reference | `trellis-mac` ports TRELLIS.2 image-to-3D generation from CUDA-only to Apple Silicon using PyTorch MPS and Metal/fallback backends. | `vendors/trellis-mac/README.md`, `vendors/trellis-mac/pyproject.toml` |
| Upstream image-to-3D reference | `TRELLIS.2` is a 4B image-to-3D model using O-Voxel structured latents, with Linux/CUDA installation assumptions. | `vendors/TRELLIS.2/README.md` |
| Object reconstruction reference | `sam-3d-objects` reconstructs object shape, texture, pose, and layout from single images and masks. | `vendors/sam-3d-objects/README.md`, `vendors/sam-3d-objects/pyproject.toml` |
| Geometric prediction reference | `HunyuanWorld-Mirror` predicts camera poses, intrinsics, depth maps, point clouds, multi-view depths, surface normals, camera parameters, and 3D Gaussians. | `vendors/HunyuanWorld-Mirror/README.md` |

## Runtime Surfaces

| Surface | Status | Evidence |
|---|---|---|
| Root library | Not present yet. No root source/package surface observed in the bounded scan. | `.` |
| CLI | Present only in vendor references. `trellis-mac` exposes `trellis-generate = "generate:main"` and documents `python generate.py`. | `vendors/trellis-mac/pyproject.toml`, `vendors/trellis-mac/README.md` |
| Demo scripts | Present only in vendor references. SAM 3D Objects documents `python demo.py`; TRELLIS.2 documents setup/inference flows. | `vendors/sam-3d-objects/README.md`, `vendors/TRELLIS.2/README.md` |
| UI/API/worker | No root UI, API, or worker surface observed. | `.` |

## Stack

| Layer | Observed facts | Evidence |
|---|---|---|
| Intended direction | MLX-oriented 3D/spatial model library is user-stated during onboarding, not yet encoded in root files. | user prompt, `.` |
| Current executable code | Python-based vendor projects. | `vendors/trellis-mac/pyproject.toml`, `vendors/sam-3d-objects/pyproject.toml` |
| Apple target reference | macOS Apple Silicon, Python 3.11+, PyTorch/MPS, Metal toolchain optional/fallback. | `vendors/trellis-mac/README.md`, `vendors/trellis-mac/pyproject.toml` |
| CUDA/Linux references | TRELLIS.2 and HunyuanWorld-Mirror currently document CUDA/Linux-oriented assumptions. | `vendors/TRELLIS.2/README.md`, `vendors/HunyuanWorld-Mirror/README.md` |
| Packaging | Hatch-based package in `sam-3d-objects`; basic PEP 621 metadata and script entry point in `trellis-mac`. | `vendors/sam-3d-objects/pyproject.toml`, `vendors/trellis-mac/pyproject.toml` |

## Commands Observed

| Command | Scope | Status |
|---|---|---|
| `bash setup.sh` | `vendors/trellis-mac` | Documented vendor setup command. |
| `python generate.py path/to/image.png` | `vendors/trellis-mac` | Documented vendor generation command. |
| `trellis-generate` | `vendors/trellis-mac` | Script entry point declared in `pyproject.toml`. |
| `python demo.py` | `vendors/sam-3d-objects` | Documented vendor demo command after setup. |
| `. ./setup.sh --new-env --basic --flash-attn --nvdiffrast --nvdiffrec --cumesh --o-voxel --flexgemm` | `vendors/TRELLIS.2` | Documented upstream setup command. |

No root install, build, test, lint, or package command was observed in the bounded scan.

## Inferred Direction

The repository is likely meant to become a clean MLX library that ports or reimplements selected 3D/spatial model components from the vendored references, prioritizing Apple Silicon viability. This is inferred from the user prompt plus the presence of `trellis-mac` as an Apple Silicon adaptation reference.

## Unknowns

| Unknown | Why it matters | Evidence boundary |
|---|---|---|
| Root package name, layout, and public API | Needed before implementation planning. | No root package files observed in `.`. |
| First supported model family | The vendor corpus spans image-to-3D generation, object reconstruction, and geometric prediction. | `vendors/` |
| MLX dependency/version policy | Determines kernels, tensor APIs, tests, and examples. | No root manifest observed in `.`. |
| Whether vendors are source-of-truth, references only, or submodules to preserve | Affects cleanup and licensing strategy. | `vendors/` |
| Licensing constraints across vendored projects | Must be resolved before copying or adapting code. | Vendor README/license details were not fully scanned. |
