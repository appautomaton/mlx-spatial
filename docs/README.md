# Documentation Map

Start with the page that matches the job:

| Need | Page |
| --- | --- |
| Run SAM 3D Objects from an image and mask | [sam3d.md](sam3d.md) |
| Run TRELLIS.2 object image-to-3D | [trellis2.md](trellis2.md) |
| Run HY-WorldMirror scene reconstruction | [hyworld2.md](hyworld2.md) |
| Run Apple LiTo image-to-3DGS research inference | [lito.md](lito.md) |
| Start from recommended runnable scripts | [scripts/README.md](../scripts/README.md) |
| Understand module boundaries | [architecture.md](architecture.md) |
| Contribute or verify changes locally | [development.md](development.md) |
| Publish converted model bundles | [model-publishing.md](model-publishing.md) |
| Prepare a PyPI release | [release.md](release.md) |

## Reader Contract

The docs are written for both humans and coding agents:

- Commands should be copyable from a clean checkout after `uv sync`, or from an installed package when explicitly stated.
- Model weights, inputs, and outputs stay under ignored local directories: `weights/`, `inputs/`, and `outputs/`.
- Runtime pages should say what the pipeline is for, what assets it needs, how to run it, what it writes, and what common blockers mean.
- Avoid one-off run logs in stable docs. Put dated evidence in release notes, model cards, or local project notes.
- Keep upstream license and access terms linked, not pasted.

## Package Shape

`mlx-spatial` ships runtime code and CLIs. It does not ship model weights.

Installed CLIs:

```bash
mlx-spatial-sam3d --help
mlx-spatial-trellis2 --help
mlx-spatial-hyworld2 --help
mlx-spatial-lito --help
```

From a cloned repo, use `uv run <command>` for the same CLIs.
