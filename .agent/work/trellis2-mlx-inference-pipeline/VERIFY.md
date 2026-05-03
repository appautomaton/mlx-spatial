# VERIFY: TRELLIS.2 MLX Inference Pipeline

## Result

The slice chain is complete through the planned export boundary. The deepest real alpha trace stage is `image-conditioning`; the current live blocker remains:

```text
sparse-structure-sampling / MLX sparse structure ModulatedTransformerCrossBlock forward
```

The real sparse input projection executes to `(1, 4096, 1536)` before that blocker. Downstream boundaries are mapped with fake upstream tensors and real local checkpoints where available, but the main trace does not skip past the sparse-flow blocker.

## Verification Commands

```bash
uv run pytest tests/test_trellis2_export.py
# 6 passed

uv run pytest
# 174 passed, 5 skipped

uv run python -m mlx_spatial.trellis2 attempt-forward-trace weights/trellis2 inputs/trellis2/demo-alpha.webp --dino-root weights/dinov3-vitl16-pretrain-lvd1689m
# completed=('input-image', 'asset-config-validation', 'checkpoint-probe-readiness', 'image-preprocessing-background', 'image-conditioning')
# outputs=('cond',)
# blocker_stage=sparse-structure-sampling
# operation=MLX sparse structure ModulatedTransformerCrossBlock forward
```

## Boundary Evidence

- DINOv3 real local conditioning completes with output `cond` shape `(1, 1029, 1024)`.
- Sparse structure flow loads real `input_layer` tensors, projects to `(1, 4096, 1536)`, then blocks at `ModulatedTransformerCrossBlock`.
- Sparse decoder boundary is mapped; standalone real path currently reports missing `weights/trellis2/microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16.json`.
- Shape SLat boundary maps `1024_cascade`; fake sparse coordinates with real checkpoint tensors project to `(2, 1536)`.
- Texture SLat boundary maps `1024_cascade`; fake `shape_slat` with real checkpoint tensors projects concat features to `(2, 1536)`.
- Combined decode boundary with fake `shape_slat` and `tex_slat` loads real shape/texture decoder tensors and projects both `from_latent` paths to `(2, 1024)`.
- Export boundary validates `.glb`/`.obj` paths under ignored `outputs/` and blocks on upstream mesh/texture payload absence for the real trace.

## Current Risk

Full image-to-3D remains blocked by sparse transformer execution, not export. The next implementation target should be the sparse `ModulatedTransformerCrossBlock` path, including sparse 3D RoPE, shared modulation, self-attention, cross-attention, and MLP residuals.

## Auto-Verify: Slice 8

- Criterion: If a real mesh/texture artifact exists, it is written only under ignored `outputs/` and reported with structured metadata.
  - Result: PASS
  - Evidence: `uv run pytest` reported `174 passed, 5 skipped`, including `tests/test_trellis2_export.py`. Direct inspection shows `outputs/` is ignored by `.gitignore`, `write_trellis2_export_artifact(...)` validates the output path before writing, and it returns `Trellis2ExportArtifact(path, format, bytes_written, detail)`.
  - Gap: none

- Criterion: If export is blocked, the blocker names the exact mesh/export dependency or format issue.
  - Result: PASS
  - Evidence: `tests/test_trellis2_export.py` covers invalid path blocking as `mesh-export / TRELLIS.2 export path validation` and upstream trace blocking as `mesh-export / upstream inference completion before export`. Fresh real trace output remains blocked upstream at `sparse-structure-sampling / MLX sparse structure ModulatedTransformerCrossBlock forward`.
  - Gap: none

- Criterion: README and `VERIFY.md` report the deepest real completed stage and current blocker.
  - Result: PASS
  - Evidence: Direct inspection shows README reports the real alpha trace completes MLX DINOv3 `image-conditioning` and stops at `sparse-structure-sampling`; this `VERIFY.md` reports the same completed stage and blocker. Fresh trace output confirms `completed=('input-image', 'asset-config-validation', 'checkpoint-probe-readiness', 'image-preprocessing-background', 'image-conditioning')` and `operation=MLX sparse structure ModulatedTransformerCrossBlock forward`.
  - Gap: none

Content checks: PASS
- Audience: Repo maintainers continuing TRELLIS.2 MLX parity work; README and VERIFY name the runnable commands, current stage, blocker, and next implementation target.
- Thesis: The change is complete through the export boundary, while real image-to-3D remains blocked upstream by sparse transformer execution.
- Source policy: No external claims or citations were introduced; verification uses local command output and local file inspection.
- Format: README remains project documentation; VERIFY uses a concise verification report with command evidence.
- Anti-slop scan: no promotional claims, vague attribution, or generic completion language found in the verified sections.

Overall: PASS
Remaining gaps: none for Slice 8. Full TRELLIS.2 image-to-3D remains a follow-up beyond this slice.
Recommended next skill: `auto-frame` for the next sparse `ModulatedTransformerCrossBlock` implementation slice.
