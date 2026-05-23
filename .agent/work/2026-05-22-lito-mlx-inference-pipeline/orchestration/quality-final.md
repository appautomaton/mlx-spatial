# Slice 5Q-3 Real-Input Quality Handoff

Date: 2026-05-23

## Scope

Slice 5Q-3 regenerates checkpoint-backed real-object LiTo outputs after the Slice 5Q-2 decode/init-coverage fix. This is a human-verify checkpoint: command checks can prove the PLYs are checkpoint-backed, finite, and non-degenerate, but AC-07 remains open until visual inspection accepts the surface quality.

`inputs/lito/smoke.png` was not used. It is a color-blob framework probe and is not qualitative evidence.

## Teacup

Command:

```bash
uv run mlx-spatial-lito generate inputs/trellis2/teacup.png --weights-root weights/lito-mlx --output outputs/lito/teacup-quality-fix.ply --memory-profile safe --max-init-coords-per-batch 1024 --render-size 12 --num-steps 20 --seed 42 --print-metrics
```

Inspector:

```bash
uv run python scripts/lito/inspect_quality.py outputs/lito/teacup-quality-fix.ply --compare /tmp/lito-teacup-quality-baseline.json --json /tmp/lito-teacup-quality-fix.json
```

Evidence:

- Output: `outputs/lito/teacup-quality-fix.ply`
- Checkpoint-backed header: yes
- Source-contract smoke header: no
- Vertex count: `65536`
- Property count: `62`
- Inspector flags: none
- Failure classification: `stats_sane_visual_review_required`
- Versus baseline: vertex count increased by `32768`; bbox span delta was `[0.0265, 0.0349, 0.0146]`

## Beer Mug

Command:

```bash
uv run mlx-spatial-lito generate inputs/trellis2/beer-mug.png --weights-root weights/lito-mlx --output outputs/lito/beer-mug-quality-final.ply --memory-profile safe --max-init-coords-per-batch 1024 --render-size 12 --num-steps 20 --seed 42 --print-metrics
```

Metrics:

- Output path: `outputs/lito/beer-mug-quality-final.ply`
- Decoded cells: `1024`
- Exported Gaussians: `65536`
- Peak active MLX memory: `15.2769 GB` in DiT, `1.9974 GB` in decode
- Wall time: `377.9413 s` DiT, `0.4199 s` decode, `0.8303 s` export

Inspector:

```bash
uv run python scripts/lito/inspect_quality.py outputs/lito/beer-mug-quality-final.ply --json /tmp/lito-beer-mug-quality-final.json
```

Evidence:

- Output: `outputs/lito/beer-mug-quality-final.ply`
- Checkpoint-backed header: yes
- Source-contract smoke header: no
- Vertex count: `65536`
- Property count: `62`
- Inspector flags: none
- Failure classification: `stats_sane_visual_review_required`
- Bbox span: `[1.4123, 1.6536, 1.8430]`
- Opacity probability median: `0.005801`
- Scale exp median: `0.001562`
- Quaternion norm median: `1.000000`

## Gate State

Slice 5Q-3 command evidence is complete for two real-object inputs. The capped teacup outputs were visually rejected or only partially accepted; the uncapped teacup output was inspected in a proper Gaussian-splat viewer and judged fair-looking with a known caveat around the teacup handle void/ring. The current final quality handoff artifacts are:

- `outputs/lito/teacup-quality-crop-uncapped.ply`
- `outputs/lito/beer-mug-quality-uncapped.ply`

The remaining final verification step is to run Slice 5Q-4 and keep the accepted quality note honest: 3DGS quality is usable in a splat-aware viewer, the handle-hole topology is imperfect, and mesh/GLB extraction is out of scope.
