# Pixal3D Decoder Token Guards Spec

## Bounded Goal

Decouple Pixal3D final shape/texture decoder compute limits from
`max_num_tokens`, preserving `max_num_tokens` as the HR coordinate selection
guard while keeping explicit Apple GPU memory guards for decoder execution.

## Source Evidence

After the shape-upsample guard was decoupled, real downloaded-weight smoke
advanced through HR shape SLat and texture SLat, wrote six Pixal3D artifacts,
then blocked at final shape decode:

```text
shape decoder stopped before 7-channel output ... level 1 for 60822 tokens above limit 49152
```

Upstream Pixal3D decodes the selected shape SLat after coordinate selection; it
does not use `max_num_tokens` as the decoder compute guard.

## Requirements

| ID | Requirement |
| --- | --- |
| PXDEC-01 | Keep `max_num_tokens` as HR coordinate selection/sampling guard. |
| PXDEC-02 | Add separate shape and texture decoder compute token guards. |
| PXDEC-03 | Expose the guards through package CLI and script help. |
| PXDEC-04 | Real smoke should no longer block on the old coupled shape-decoder `49152` limit. |

## Constraints

- Runtime remains Torch-free and CUDA-free.
- Do not remove decoder guards entirely.
- Do not change package versioning in this cycle.

## Acceptance Criteria

- CLI/script expose `--shape-decoder-token-limit` and
  `--texture-decoder-token-limit`.
- Tests cover validation and trace metadata for the new options.
- Real downloaded-weight smoke records the new next stage/blocker or writes GLB.
- Focused and full tests plus release hygiene pass.
