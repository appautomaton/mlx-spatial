# Pixal3D Shape Upsample Token Guard Spec

## Bounded Goal

Decouple Pixal3D's HR coordinate selection limit from the shape decoder
upsample compute guard so the MLX path matches upstream ordering more closely:
upsample first, then quantize/reduce HR coordinates.

## Source Evidence

Real downloaded-weight smoke now reaches `shape-slat-cascade` and blocks with:

```text
shape decoder upsample stopped before level 2: token_count=61951 exceeds decoder_token_limit=49152
```

When `--max-num-tokens 100000` is used, it advances one more level and blocks
with:

```text
shape decoder upsample stopped before level 3: token_count=259425 exceeds decoder_token_limit=100000
```

The upstream Pixal3D code calls `shape_slat_decoder.upsample(...,
upsample_times=4)` before applying `max_num_tokens` to quantized HR
coordinates.

## Requirements

| ID | Requirement |
| --- | --- |
| PXUP-01 | Keep `max_num_tokens` as the HR coordinate selection/sampling guard. |
| PXUP-02 | Add a separate shape-upsample compute token guard with a safer explicit default. |
| PXUP-03 | Expose the guard through package CLI and script help. |
| PXUP-04 | Real smoke should no longer block at the old `49152`/`100000` shape-upsample guard unless the new guard is reached. |

## Constraints

- Runtime remains Torch-free and CUDA-free.
- Do not remove the upsample guard entirely; Apple GPU memory safety still needs
  a bounded compute limit.
- Do not change package versioning in this cycle.

## Acceptance Criteria

- CLI/script expose `--shape-upsample-token-limit`.
- Tests cover validation and metadata for the new option.
- Focused Pixal3D tests pass.
- Real Pixal3D smoke records the new next stage or blocker.
