# SPEC: TRELLIS.2 Real-Weight Tooling

## Bounded Goal

Add reusable TRELLIS.2 real-weight tooling that validates local assets, inspects configured safetensors checkpoints, and load-probes selected tensor groups into MLX arrays without requiring real weights, network access, Hugging Face tooling, PyTorch, Transformers, or vendor imports in default tests.

## Selected Lenses

- product
- engineering
- runtime

## Constraints

- Primary reference for operational expectations is `vendors/trellis-mac`; `vendors/TRELLIS.2` is the original implementation reference to stay aware of, not code to import.
- Hugging Face CLI should be available as dev tooling via `huggingface_hub`, but runtime code and default tests must not depend on it or invoke downloads.
- Real weights must remain local-only under ignored `weights/trellis2/` and must not be committed or required by default tests.
- Default tests must use fake/minimal safetensors fixtures and must not require network access, Hugging Face credentials, PyTorch, Transformers, vendor imports, or local absolute paths.
- The tooling should be code-first: reusable API and/or command entrypoints, not a static committed tensor inventory report.
- The implementation should reuse existing `TRELLIS2_ASSETS`, `validate_model_assets`, `inspect_checkpoint`, and `load_checkpoint_tensors` behavior where possible.
- Probe groups may be conservative placeholders if exact real tensor prefixes require manual local-weight verification, but their assumptions must be explicit and testable against fake fixtures.

## Required Behavior

- Provide a TRELLIS.2-specific workflow helper or command entrypoint that validates the local asset root and reports readiness without downloading anything.
- Provide reusable configured checkpoint inspection across TRELLIS.2 safetensors files from the manifest or a supplied local root.
- Provide reusable load-probe behavior for selected tensor names or named tensor-prefix groups, returning MLX arrays or deterministic summaries suitable for terminal use.
- Include named probe groups that are informed by `vendors/trellis-mac` as the primary reference and compatible with the original TRELLIS.2 architecture reference where practical.
- Surface clear failures for missing assets, unsupported checkpoint formats, empty probe selections, no matching tensors, and invalid local roots.
- Add Hugging Face CLI as a dev-only tool and document/use the manual `huggingface-cli` usage pattern as an operator step.
- Provide an explicit real-weight download attempt workflow that targets the needed TRELLIS.2 assets, remains outside default tests, and fails clearly when network access, credentials, repo ID, or disk space are unavailable.
- Preserve default-test independence from real weights and external services.

## Acceptance Criteria

- A TRELLIS.2 real-weight tooling API and/or command entrypoint exists and is public enough to run from a local checkout.
- The tooling validates a fake TRELLIS.2 asset root using temporary files and reports deterministic readiness details.
- The tooling inspects fake TRELLIS.2 safetensors checkpoints and returns deterministic metadata grouped by checkpoint or configured probe group.
- The tooling load-probes selected fake tensors into MLX arrays or MLX-derived summaries with expected shapes, dtypes, and deterministic ordering.
- Named probe groups are covered by tests and documented with their reference basis and limitations.
- Invalid roots, missing checkpoint files, unsupported formats, empty selections, and no-match probes are tested.
- README documents the manual HF CLI installation/download expectation, local root convention, real-weight validation command, inspection command, load-probe command, and unsupported boundaries.
- A dev-only Hugging Face CLI dependency is declared without adding Hugging Face tooling to base runtime dependencies.
- The plan includes an explicit operator download attempt for the needed TRELLIS.2 model files, with verification separated from default automated tests.
- Default `uv run pytest` passes without real weights, network access, Hugging Face credentials, PyTorch, Transformers, vendor imports, or local absolute paths.
- No real checkpoint artifacts or generated real-weight outputs are committed.

## Blocking Questions Or Assumptions

- Assumption: this slice may read vendor documentation or reference code during planning/execution to define conservative probe groups, but implementation must not import vendor modules.
- Assumption: the first tool should prioritize safetensors files already listed in `TRELLIS2_ASSETS`; `.pt`/`.pth` support remains deferred.
- Assumption: real local verification includes a best-effort HF CLI download attempt during execution when the repo ID, network, credentials, and disk space permit; default automated verification uses fake fixtures only.
- Assumption: a CLI entrypoint is useful if it remains thin over the public API and does not broaden dependency requirements.

## Anti-Goals

- Do not download weights automatically during import, runtime helper calls, or default tests.
- Do not add `huggingface_hub`, PyTorch, Transformers, or vendor code as runtime dependencies.
- Do not require real weights in tests.
- Do not commit a static tensor inventory report from real checkpoints.
- Do not implement TRELLIS.2 model construction, architecture mapping, sparse/transformer block execution, decoder execution, mesh extraction, or GLB export.
- Do not add `.pt`/`.pth` checkpoint loading in this slice.
