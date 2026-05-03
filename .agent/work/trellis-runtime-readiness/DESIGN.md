# DESIGN: TRELLIS.2 MLX Runtime Readiness

## Runtime Contracts

### Sparse Map Rows

Use the existing sparse map row contract:

- shape: `(m, 3)`
- dtype: integer
- columns: `(target_index, source_index, kernel_index)`
- row order: deterministic, inherited from `sparse_conv_map`

Weighted compute must not reinterpret coordinates or offsets. It consumes rows only as indices.

### Source Features

Source features use shape `(source_count, in_channels)`.

`source_index` must be in `[0, source_count)`.

### Kernel Weights

Use a simple reference layout:

- shape: `(kernel_count, in_channels, out_channels)`
- row-selected weight: `kernel_weights[kernel_index]`
- per-row contribution: `source_features[source_index] @ kernel_weights[kernel_index]`

`kernel_index` must be in `[0, kernel_count)`.

### Target Features

Weighted sparse convolution returns shape `(target_count, out_channels)`.

For each map row, add the per-row contribution into `target_index`. Duplicate target rows sum deterministically. Targets with no incoming rows remain zero.

Empty maps return a zero tensor with shape `(target_count, out_channels)`. The implementation still needs `kernel_weights` to determine `out_channels`.

## Asset Readiness Contract

Use a local asset convention that is explicit but does not require downloads by default:

- recommended local path: `weights/trellis2/`
- repository protection: ignore `weights/`
- manifest/config location: `src/mlx_spatial/model_assets.py`
- validation API: `validate_model_assets(root, manifest=TRELLIS2_ASSETS)`

The manifest should include stable metadata and expected relative file paths. Exact filenames may be refined later when a concrete TRELLIS.2 checkpoint slice inspects a chosen distribution.

Validation returns structured information rather than raising on missing files by default:

- expected file path
- present/missing status
- absolute resolved path

Tests should use temporary directories and tiny fake files only.

## Dependency Boundary

Default code must not import PyTorch, Transformers, Hugging Face packages, or vendor modules. Hugging Face CLI appears only as documentation for a later manual download step.

## Documentation Boundary

README should say this is runtime readiness, not full TRELLIS.2 inference. It should name the next concrete slices: checkpoint inspection/loading, TRELLIS sparse block parity, decoder/mesh path.
