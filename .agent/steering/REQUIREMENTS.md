# Requirements

## Product Commitments

### Observed

- MLX-first inference: all model pipelines must run on Apple Silicon via `mlx` without PyTorch or CUDA at runtime (`README.md:1-4`).
- Maintain speed and quality of original reference implementations (user-stated goal).
- Exact-mode staged pipelines: incomplete stages produce structured blockers rather than silent approximations (`README.md:194-195`).
- Local-only weight management: no automatic downloads during import, tests, or inference (`README.md:177`).
- Deterministic, reproducible outputs: explicit ordering contracts on sparse conv, topology, and mesh operations.
- Three model pipelines must produce outputs comparable to their PyTorch originals: TRELLIS.2 (shape OBJ + textured GLB), SAM 3D Objects (Gaussian PLY + textured GLB), HY-World 2.0 (multi-view heads).

### Inferred

- Performance should approach or match PyTorch-on-CUDA baselines on equivalent Apple Silicon hardware.
- The library should remain usable as both a CLI tool and a Python library import.

### Needs Confirmation

- Specific performance targets or benchmarks (e.g., "within X% of reference timing on M-series chip Y").
- Whether the HY-World 2.0 parity flag (`numeric_parity_verified=false`) should become `true` for release.

## Technical Constraints and Invariants

### Observed

- Python >=3.11, MLX as sole ML runtime (`pyproject.toml:10-11`).
- safetensors-only checkpoint loading at runtime (`README.md:116`).
- No PyTorch, CUDA, flash-attn, or gsplat as runtime dependencies (`README.md:363`).
- xatlas and fast-simplification are runtime dependencies for mesh export (`pyproject.toml:14,19`).
- Checkpoint formats: safetensors for MLX, with a dev-only conversion path from PyTorch `.ckpt`/`.pt` via `pt-safe-loader`.
- Weights, inputs, outputs, and vendors are gitignored (`.gitignore:8-11`).
- Parity testing requires env var `MLX_SPATIAL_RUN_TORCH_PARITY=1` and local PyTorch checkout (`README.md:401-407`).

### Inferred

- Memory management is important; `mlx_memory.py` suggests memory profiling or constraints are a concern.
- The `--memory-profile` flag on SAM3D (`balanced`, `safe`, `large`) and `--decoder-token-limit` on TRELLIS.2 indicate resource-bounded execution is a requirement.

### Needs Confirmation

- Minimum Apple Silicon target (M1? M2 Pro?).
- Maximum supported memory or resolution limits per pipeline.

## Quality and Operational Expectations

- Testing bar: pytest suite with ~49 test files covering all modules. Parity tests are opt-in.
- No CI automation observed (inferred: not yet configured).
- No lint or typecheck commands observed in `pyproject.toml`.
- Release constraint: no network access at runtime; all weights must be locally pre-positioned.

## Integration Boundaries

- Upstream: Hugging Face model weights (TRELLIS.2-4B, SAM 3D Objects, RMBG 2.0, DINOv3, MoGe, HY-World 2.0) downloaded via CLI and stored locally.
- Downstream: CLI users consuming OBJ, GLB, and PLY outputs; Python library consumers importing `mlx_spatial`.
- External tooling: `huggingface-cli` for weight download (dev-only), `xatlas` for UV unwrapping (runtime).

## Non-Goals

- Training or fine-tuning any model.
- Replacing MLX with another ML framework.
- Supporting non-Apple-Silicon hardware.
- Automatic model weight downloading at runtime.
- Importing or modifying vendored reference code at runtime.
- Running PyTorch in the production inference path.

## Open Risks and Unknowns

- HY-World 2.0 numeric parity is not yet verified (`numeric_parity_verified=false`).
- Performance parity against PyTorch-on-CUDA has no established benchmark suite yet (inferred).
- No CI, lint, or typecheck pipeline exists (observed: absent from `pyproject.toml` and no `.github/`).
- Memory profiles for large inputs (high-res textures, cascade routes) may need more explicit bounds.
- Textured GLB export quality depends on xatlas behavior, which may vary across macOS versions.

## Evidence Anchors

- `pyproject.toml:10-20` — runtime dependencies
- `pyproject.toml:23-27` — dev dependencies
- `README.md:194-195` — structured blocker design
- `README.md:116` — safetensors-only loading
- `README.md:177` — no runtime downloads
- `README.md:363` — no PyTorch/CUDA runtime
- `.gitignore` — excluded paths