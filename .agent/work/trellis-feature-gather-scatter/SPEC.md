# SPEC: TRELLIS Feature Gather/Scatter Primitives

## Bounded Goal

Implement MLX sparse feature gather/scatter primitives that consume `sparse_conv_map` rows and deterministically move or accumulate feature vectors between source and target sparse coordinate slots.

## Selected Lenses

- product
- engineering

## Objective

Provide the next reusable sparse tensor building block above sparse convolution maps: feature movement over `(target_index, source_index, kernel_index)` rows without introducing convolution weights, model layers, checkpoints, or vendor dependencies.

## Broader Intent

This change advances TRELLIS support from sparse topology construction toward executable sparse neural operations while preserving the package's MLX-first, model-neutral primitive boundary.

## Target User

Developers building or porting sparse 3D model components, starting with TRELLIS.2-adjacent operations and later reusable for SAM3D or Hunyuan-family geometry workflows.

## Desired Outcome

Given source features and sparse convolution map rows, callers can:

- gather source feature rows in map-row order;
- scatter or accumulate map-row features into target feature rows deterministically;
- verify behavior with MLX-only tests by default;
- optionally compare against a local PyTorch reference when explicitly enabled.

## Constraints

- Must use PyPI `mlx` as the default backend dependency and avoid adding PyTorch to base dependencies.
- Must consume the existing `sparse_conv_map` row contract: `(target_index, source_index, kernel_index)`.
- Must preserve deterministic ordering and accumulation semantics independent of Python hash ordering.
- Must reject invalid map shape, feature rank, index bounds, and incompatible target/source counts with `ValueError`.
- Must keep default tests independent of Torch, Transformers, Hugging Face credentials, checkpoints, vendors, and local absolute paths.
- Optional PyTorch parity must remain gated by `MLX_SPATIAL_RUN_TORCH_PARITY=1` and skipped by default.
- Documentation must make the feature movement semantics explicit enough to support later sparse convolution compute.

## Blocking Questions Or Assumptions

- Assumption: this slice should expose weight-free primitives only; weighted sparse convolution belongs to a later spec.
- Assumption: map rows continue to use `source = target + offset` semantics from `sparse_conv_map`, but gather/scatter should not reinterpret coordinate offsets directly.
- Assumption: scatter accumulation should sum all row features targeting the same target index, because that is the behavior future sparse convolution reduction needs.
- Assumption: MLX-only tests should assert exact small-array behavior rather than approximate model parity.

## Scope Boundary

In scope:

- Public MLX helper to gather source feature vectors according to map rows.
- Public MLX helper to accumulate row feature vectors into target feature slots according to map rows.
- Validation for map rows, feature shapes, index bounds, target count, and feature dimensionality.
- README documentation for gather/scatter semantics and default/optional parity tests.
- MLX-only tests covering ordering, repeated target accumulation, empty maps, invalid inputs, and deterministic results.
- Optional local PyTorch parity scaffolding skipped by default.

## Anti-Goals

- Do not implement weighted sparse convolution.
- Do not implement stride, dilation, transposed maps, or target coordinate generation.
- Do not import, modify, or depend on `vendors/` code.
- Do not load TRELLIS, SAM3D, Hunyuan, Transformers, checkpoints, or Hugging Face assets.
- Do not add PyTorch to base dependencies.
- Do not optimize for GPU kernel fusion or high-performance scatter kernels in this slice.
- Do not introduce persistence, serialization, or model-layer abstractions.

## Acceptance Criteria

- `mlx_spatial.sparse_conv` or another clearly documented public module exposes gather/scatter helpers.
- Gather returns source feature rows ordered exactly as input map rows.
- Scatter accumulation sums row feature vectors into target slots and leaves untouched targets as zero.
- Empty maps return correctly shaped empty gather output and zero scatter output.
- Duplicate target indices accumulate deterministically.
- Invalid map row shape, non-integer map rows, out-of-bounds source/target indices, invalid feature rank, and feature-channel mismatches raise `ValueError`.
- `uv run pytest tests/test_sparse_feature.py` passes with MLX only.
- `uv run pytest` passes without local PyTorch or model assets.
- Optional PyTorch parity test is present, gated by `MLX_SPATIAL_RUN_TORCH_PARITY=1`, and skipped by default.
- README documents row contract, gather ordering, scatter accumulation, and anti-goals.

## Risks

- MLX scatter/update APIs may not support the exact accumulation pattern directly; implementation may need a simple deterministic Python loop for correctness first.
- Validation across MLX dtypes may require converting small index arrays to host lists for reliable error messages.
- Naming may become awkward if future weighted sparse convolution needs different map/value conventions; keep helper names literal and minimal.

## Recommended Next Skill

`auto-plan`
