# Slice 6 Docs + Roadmap Orchestration

## Implementer

- Agent: Boole (`019e5562-bdf4-7720-98d7-f7cd7c0bb248`)
- Status: completed
- Files changed: `docs/lito.md`, `docs/architecture.md`, `.agent/steering/ROADMAP.md`

## Coordinator Fixes

- Clarified that the current LiTo runtime is a source-contract bring-up and does not claim full Apple checkpoint numerical parity.
- Tightened license wording to research-only and non-commercial.
- Replaced stale future-tense Slice 5 wording with current-state CLI/API wording.
- Corrected the programmatic API example to use `output_path=`.
- Aligned the recommended-defaults table with live `LITO_RECOMMENDED_*` constants by using `LITO_RECOMMENDED_MLX_COMPUTE_DTYPE` and removing the docs-only output-format row.

## Reviews

- Spec review approved the Slice 6 acceptance criteria.
- Quality review requested changes for API example, license wording, stale Slice 5 phrasing, and constants-table drift. Final quality re-review approved.

## Verification

- `test -f docs/lito.md` -> passed
- `test -f docs/architecture.md` -> passed
- `grep -E "^## Phase 3" .agent/steering/ROADMAP.md` -> found Phase 3 LiTo entry
- `grep -E "lito-mlx-inference-pipeline" .agent/steering/ROADMAP.md` -> found active change
- `rg -n "LITO_RECOMMENDED_DTYPE|LITO_RECOMMENDED_OUTPUT_FORMAT|Slice 5 embeds|After Slice 5|after Slice 5|after the pipeline CLI lands" docs/lito.md docs/architecture.md` -> no matches

## Notes

- Docs keep CUDA as static source reference only and optional Torch/MPS parity as non-blocking.
- ROADMAP Phase 3 remains `in-progress` until final verification marks the change done.
