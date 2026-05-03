# VERIFY: TRELLIS.2 Real-Weight Tooling

## Verification: Slice 1 Reference-Aware Probe Group Design

- Criterion: Probe groups identify checkpoint-relative paths and exact names or prefixes.
  - Result: PASS
  - Evidence: `TRELLIS2_PROBE_GROUPS` at `src/mlx_spatial/trellis2.py:47-78` defines five groups with checkpoint-relative `.safetensors` paths and exact tensor names.
  - Gap: none

- Criterion: Each probe group has a reference note naming whether it is based on `trellis-mac`, original TRELLIS.2 awareness, or conservative placeholder behavior.
  - Result: PASS
  - Evidence: reference notes are present for each group at `src/mlx_spatial/trellis2.py:52`, `src/mlx_spatial/trellis2.py:58`, `src/mlx_spatial/trellis2.py:64`, `src/mlx_spatial/trellis2.py:70`, and `src/mlx_spatial/trellis2.py:76`; `tests/test_trellis2_tools.py:53-64` asserts groups are named and reference-backed.
  - Gap: none

- Criterion: No implementation imports vendor modules or reads vendor paths at runtime.
  - Result: PASS
  - Evidence: `src/mlx_spatial/trellis2.py:1-244` imports only stdlib, MLX, and `mlx_spatial` modules; vendor references are recorded only as static notes and design evidence at `.agent/work/trellis-real-weight-tools/DESIGN.md:17-24`.
  - Gap: none

## Verification: Slice 2 Dev-Only HF CLI Boundary

- Criterion: `huggingface_hub` is present only in the dev dependency group.
  - Result: PASS
  - Evidence: `pyproject.toml:11-15` runtime dependencies are `mlx`, `numpy`, and `safetensors`; `pyproject.toml:17-21` dev dependencies include `huggingface-hub>=0.36`; `tests/test_trellis2_tools.py:42-50` asserts this boundary; fresh `uv run pytest tests/test_trellis2_tools.py` passed with `10 passed`.
  - Gap: none

- Criterion: Base runtime dependencies still exclude Hugging Face Hub, PyTorch, and Transformers.
  - Result: PASS
  - Evidence: `pyproject.toml:11-15`; `tests/test_trellis2_tools.py:42-50`; fresh targeted tests passed.
  - Gap: none

- Criterion: Tests assert the dependency boundary.
  - Result: PASS
  - Evidence: `tests/test_trellis2_tools.py:42-50`; fresh `uv run pytest tests/test_trellis2_tools.py` passed with `10 passed`.
  - Gap: none

## Verification: Slice 3 TRELLIS.2 Tooling API

- Criterion: API validates a fake TRELLIS.2 asset root using `TRELLIS2_ASSETS` semantics and reports deterministic readiness.
  - Result: PASS
  - Evidence: `validate_trellis2_assets` at `src/mlx_spatial/trellis2.py:81-84`; fake-root test at `tests/test_trellis2_tools.py:67-74`; fresh targeted tests passed.
  - Gap: none

- Criterion: API inspects fake safetensors checkpoints by configured checkpoint path and by named probe group.
  - Result: PASS
  - Evidence: `inspect_trellis2_checkpoints` at `src/mlx_spatial/trellis2.py:96-105`; `inspect_trellis2_probe` at `src/mlx_spatial/trellis2.py:108-120`; tests at `tests/test_trellis2_tools.py:77-99`.
  - Gap: none

- Criterion: API surfaces missing roots, missing checkpoint files, unsupported formats, empty selections, and no-match probes with clear errors.
  - Result: PASS
  - Evidence: validation paths at `src/mlx_spatial/trellis2.py:214-240`; error-path tests at `tests/test_trellis2_tools.py:115-154`; fresh targeted tests passed.
  - Gap: none

- Criterion: Public exports are covered by tests.
  - Result: PASS
  - Evidence: exports in `src/mlx_spatial/__init__.py`; test coverage at `tests/test_trellis2_tools.py:184-191`; fresh targeted tests passed.
  - Gap: none

## Verification: Slice 4 Load-Probe API and Optional CLI

- Criterion: Load-probe returns MLX arrays or deterministic MLX-derived summaries for selected fake tensors.
  - Result: PASS
  - Evidence: `load_trellis2_probe` at `src/mlx_spatial/trellis2.py:123-147`; fake tensor shape, dtype, and value assertions at `tests/test_trellis2_tools.py:102-112`.
  - Gap: none

- Criterion: Output ordering is deterministic by group, checkpoint, and tensor name.
  - Result: PASS
  - Evidence: sorted tensor return at `src/mlx_spatial/trellis2.py:139-147`; deterministic assertions at `tests/test_trellis2_tools.py:83-91`, `tests/test_trellis2_tools.py:97-99`, and `tests/test_trellis2_tools.py:107-112`.
  - Gap: none

- Criterion: Empty probe selections and no-match groups raise clear errors.
  - Result: PASS
  - Evidence: empty group rejection at `src/mlx_spatial/trellis2.py:129-132`; no-match path is covered through `load_checkpoint_tensors`; tests at `tests/test_trellis2_tools.py:115-154`.
  - Gap: none

- Criterion: If CLI is added, it runs against fake fixtures in tests and does not require real weights or downloads.
  - Result: PASS
  - Evidence: CLI entrypoint at `src/mlx_spatial/trellis2.py:164-211`; console script at `pyproject.toml:23-24`; fake-fixture CLI tests at `tests/test_trellis2_tools.py:165-181`; fresh targeted tests passed.
  - Gap: none

## Verification: Slice 5 Operator Documentation

- Criterion: README states HF CLI is installed separately and is not a runtime dependency.
  - Result: PASS
  - Evidence: `README.md:138-143` states Hugging Face CLI is dev tooling and shows dev-environment commands; runtime dependency boundary is verified by `pyproject.toml:11-21`.
  - Gap: none

- Criterion: README documents `weights/trellis2/` as the local root and keeps automatic downloads out of code and tests.
  - Result: PASS
  - Evidence: `README.md:98-102`, `README.md:138-153`; automatic download exclusion is explicit at `README.md:153`.
  - Gap: none

- Criterion: README includes validation, inspection, and load-probe examples using the implemented API or CLI.
  - Result: PASS
  - Evidence: `README.md:145-151` shows `validate`, `inspect`, and `probe --load` CLI examples.
  - Gap: none

- Criterion: README states no full inference, model construction, block execution, decoder, mesh/GLB, `.pt`/`.pth`, or real-weight report is included.
  - Result: PASS
  - Evidence: `README.md:153` states these unsupported boundaries.
  - Gap: none

## Verification: Slice 6 Real-Weight Download Attempt

- Criterion: Download command uses `uv run huggingface-cli` or equivalent dev-environment invocation.
  - Result: PASS
  - Evidence: `huggingface-cli` was attempted during execution and refused due to deprecation; equivalent supported command is implemented at `src/mlx_spatial/trellis2.py:150-161` as `uv run hf download microsoft/TRELLIS.2-4B --local-dir weights/trellis2`; `uv run mlx-spatial-trellis2 download-command` returned that command during execution.
  - Gap: none

- Criterion: Download target is `weights/trellis2/` and no downloaded artifacts are tracked.
  - Result: PASS
  - Evidence: fresh `git status --short --ignored` shows `!! weights/`; `.gitignore:1` ignores `weights/`.
  - Gap: none

- Criterion: If the download succeeds, `validate_model_assets("weights/trellis2")` reports deterministic readiness details.
  - Result: PASS
  - Evidence: fresh command `uv run python -c "from mlx_spatial import validate_model_assets; r = validate_model_assets('weights/trellis2'); print('ready=', r.ready); print('present=', list(r.present)); print('missing=', list(r.missing))"` returned `ready= True`, all 12 manifest paths present, and `missing= []`.
  - Gap: none

- Criterion: If the download cannot run, the blocker is explicit and default tests remain unaffected.
  - Result: PASS
  - Evidence: download did run via equivalent `uv run hf download`; default full suite passed with `54 passed, 5 skipped`.
  - Gap: none

## Verification: Real MLX Load Probes

- Criterion: Real selected tensors load into MLX arrays for all named probe groups.
  - Result: PASS
  - Evidence: fresh real probe commands returned:
    - `sparse-structure-flow`: `tensor blocks.0.norm2.weight shape=(1536,) dtype=bfloat16 group=sparse-structure-flow`
    - `shape-slat-flow`: `tensor blocks.0.norm2.weight shape=(1536,) dtype=bfloat16 group=shape-slat-flow`
    - `texture-slat-flow`: `tensor blocks.0.norm2.weight shape=(1536,) dtype=bfloat16 group=texture-slat-flow`
    - `shape-decoder`: `tensor blocks.0.0.norm.weight shape=(1024,) dtype=float16 group=shape-decoder`
    - `texture-decoder`: `tensor blocks.0.0.norm.weight shape=(1024,) dtype=float16 group=texture-decoder`
  - Gap: none

- Criterion: BF16 real checkpoints are loadable into MLX arrays.
  - Result: PASS
  - Evidence: BF16 fallback via `mlx.core.load` is implemented at `src/mlx_spatial/checkpoint.py:71-108`; fresh real flow probes returned `dtype=bfloat16`.
  - Gap: none

## Verification: Slice 7 Full Verification

- Criterion: Full test suite passes.
  - Result: PASS
  - Evidence: fresh `uv run pytest` passed with `54 passed, 5 skipped`.
  - Gap: none

- Criterion: No real weights or generated real-weight outputs are committed.
  - Result: PASS
  - Evidence: fresh `git status --short --ignored` shows `weights/` only as ignored (`!! weights/`); no weight files appear as tracked or untracked source entries.
  - Gap: none

- Criterion: Base dependencies still exclude Hugging Face Hub, PyTorch, Transformers, and vendor imports; Hugging Face Hub is dev-only.
  - Result: PASS
  - Evidence: `pyproject.toml:11-21`; `tests/test_trellis2_tools.py:42-50`; fresh full suite passed.
  - Gap: none

## Commands Run

- `uv run pytest tests/test_trellis2_tools.py`: PASS, `10 passed`
- `uv run pytest tests/test_checkpoint.py tests/test_trellis2_tools.py`: PASS, `18 passed`
- `uv run python -c "from mlx_spatial import validate_model_assets; r = validate_model_assets('weights/trellis2'); print('ready=', r.ready); print('present=', list(r.present)); print('missing=', list(r.missing))"`: PASS, `ready= True`, 12 present, none missing
- `uv run mlx-spatial-trellis2 probe --root weights/trellis2 sparse-structure-flow --load`: PASS, BF16 tensor loaded
- `uv run mlx-spatial-trellis2 probe --root weights/trellis2 shape-slat-flow --load`: PASS, BF16 tensor loaded
- `uv run mlx-spatial-trellis2 probe --root weights/trellis2 texture-slat-flow --load`: PASS, BF16 tensor loaded
- `uv run mlx-spatial-trellis2 probe --root weights/trellis2 shape-decoder --load`: PASS, F16 tensor loaded
- `uv run mlx-spatial-trellis2 probe --root weights/trellis2 texture-decoder --load`: PASS, F16 tensor loaded
- `uv run pytest`: PASS, `54 passed, 5 skipped`
- `git status --short --ignored`: PASS, `weights/` is ignored

## Content Checks

- Audience: PASS. README targets engineers/operators working with local TRELLIS.2 assets and gives concrete validation, download, inspection, and load-probe commands at `README.md:118-153`.
- Thesis: PASS. The README claims real-weight tooling can validate, inspect, and load-probe selected safetensors into MLX arrays; code and fresh commands support that claim.
- Voice: PASS. The section is direct technical reference prose with explicit limits and commands.
- Content anti-goals: PASS. It does not claim full inference, model construction, block execution, decoder execution, mesh/GLB export, `.pt`/`.pth` support, automatic downloads, or a committed real-weight report.
- Channel: PASS. The artifact is README documentation.
- Source policy: PASS. Claims are grounded in implemented APIs, local vendor reference observations, and fresh command output.
- Factual risk: PASS. Current technical claims are backed by code and fresh verification commands.
- Format: PASS. Markdown headings, bullets, and fenced commands match the existing README format.
- Anti-slop scan: PASS. No significance inflation, promotional framing, vague attribution, sycophantic artifact, or generic conclusion found in the new section.

## Overall

PASS

## Remaining Gaps

none

## Preserved Risks

- Probe groups are conservative selected tensors, not a full TRELLIS.2 architecture mapping.
- Loading weights into MLX RAM does not run model layers or validate numerical parity.
- `hf download` is the supported command in this environment; old `huggingface-cli` is deprecated and refused to run.

## Recommended Next Skill

`auto-office-hours` or `auto-frame` for the next slice: TRELLIS block mapping/parity or decoder/mesh path.
