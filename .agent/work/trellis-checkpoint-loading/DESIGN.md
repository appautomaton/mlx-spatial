# DESIGN: TRELLIS.2 Checkpoint Inspection and MLX Loading

## API Shape

- Add model-neutral checkpoint helpers under `mlx_spatial.checkpoint`.
- Export public helpers from `mlx_spatial.__init__` only after tests define their contracts.
- Keep TRELLIS.2-specific file names and local placement guidance in `model_assets.py` and README.

## Data Model

- `CheckpointTensorInfo` describes one tensor with `name`, `shape`, `dtype`, and `source`.
- Inspection returns a deterministic sequence sorted by tensor name, then source path when multiple files are inspected.
- Loading returns a dictionary keyed by tensor name with `mlx.core.array` values.

## File Format Boundary

- Implement safetensors support first.
- Add `safetensors` only if needed for parsing `.safetensors` files and avoid any dependency that imports PyTorch or Transformers.
- Reject unsupported file suffixes with a clear `ValueError`.
- Defer `.pt`/`.pth` loading until a later optional-dependency decision.

## Filtering

- Accept exact tensor names and name prefixes.
- At least one filter may be required for loading to avoid accidental full-checkpoint memory use.
- Inspection may support no filter to report all metadata from a file.
- Raise `ValueError` for empty filters where they are not allowed, invalid filter types, or filters that match no tensors when the caller explicitly requested them.

## Verification Strategy

- Use generated tiny safetensors fixtures in tests.
- Assert numeric values after MLX loading with `mx.eval` or conversion to NumPy-compatible values where existing tests already use that pattern.
- Keep default test commands independent of real weights, network, Hugging Face credentials, PyTorch, Transformers, vendor imports, and local absolute paths.

## Runtime Boundary

- This slice does not execute model layers.
- This slice does not infer TRELLIS architecture correctness from tensor names.
- This slice only proves checkpoint metadata can be read and selected tensors can become MLX arrays.
