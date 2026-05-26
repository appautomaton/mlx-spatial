# Pixal3D Sparse Flow Boundary Spec

## Bounded Goal

Wire Pixal3D inference from completed sparse projection conditioning into the existing MLX sparse-structure FlowEuler probe, then return the next concrete sparse-decoder or shape-SLat blocker.

## Broader Intent

The current Pixal3D runtime can validate assets, run sparse-stage DINOv3 image conditioning, and write `sparse_projection.npz`, but still stops before using the Pixal3D sparse flow checkpoint. This cycle should consume the projection conditioning that already exists and advance one model boundary without claiming full GLB support.

## Selected Lenses

- **engineering:** Reuse the shared TRELLIS.2 sparse-structure flow probe, including Pixal3D projection-attention support, rather than adding a separate sampler.
- **runtime:** Preserve memory-conscious staged execution and structured blockers for later decoder/SLat/export work.

## Required Outcome

1. Pixal3D runtime resolves `sparse_structure_flow_model` and `sparse_structure_decoder` from `pipeline.json`.
2. Runtime calls `read_sparse_structure_flow_config` and `probe_sparse_structure_forward_boundary` with projection conditioning `{"global": ..., "proj": ...}` and Pixal3D sampler settings.
3. Successful sparse flow execution records sampled latent shape and completed stage metadata.
4. Runtime probes sparse decoder config/checkpoint next and either advances to sparse coordinates or returns a precise sparse-decoding blocker.
5. Existing projection-only behavior remains recoverable when sparse config/checkpoint assets are missing or invalid.

## Constraints

- Do not implement shape SLat cascade, NAF features, texture cascade, MoGe auto-camera, or GLB export in this cycle.
- Do not load PyTorch, vendor code, or CUDA-only dependencies.
- Keep fake tests small; real Pixal3D weights are not present in this checkout.

## Acceptance Criteria

| ID | Requirement | Check |
|---|---|---|
| PXSPARSE-01 | Pixal3D pipeline attempts sparse flow after projection conditioning | Fake valid sparse-flow assets reach completed stage `sparse-structure-flow`. |
| PXSPARSE-02 | Sparse decoder blocker is concrete | Runtime returns `sparse-structure-decoding` when sparse decoder config/checkpoint is not yet valid or mapped. |
| PXSPARSE-03 | Invalid sparse-flow assets still block clearly | Existing minimal fake roots return a `sparse-structure-flow` config/checkpoint blocker, not an exception. |
| PXSPARSE-04 | Existing Pixal3D and shared flow tests pass | Targeted Pixal3D tests plus `tests/test_pixal3d_flow.py` pass. |
| PXSPARSE-05 | Package hygiene remains clean | Full suite, import scan, build, and artifact checks pass. |

## Scope Coverage Decisions

- **Included:** sparse flow config resolution, projection-conditioning handoff, FlowEuler probe execution, sparse decoder probe/blocker, tests, docs, and Automaton evidence.
- **Deferred:** sparse decoder coordinate success with real Pixal3D weights, shape SLat sampling, texture SLat sampling, NAF high-resolution features, and GLB export.
