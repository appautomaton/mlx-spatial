# DESIGN: TRELLIS.2 Forward Trace Conditioning

## Stage Boundary

The current attempt path reaches:

```text
input-image -> asset-config-validation -> checkpoint-probe-readiness -> image-preprocessing-background -> image-conditioning blocker
```

This change adds a forward-trace path after preprocessing. It should not make `dry_run(...)` pretend later compute is ready. The new behavior belongs in attempt mode, where a real or fake image can be traced until a concrete blocker appears.

## Reference Facts

Local `weights/trellis2/pipeline.json` identifies:

- image conditioning model: `DinoV3FeatureExtractor`
- image conditioning model name: `facebook/dinov3-vitl16-pretrain-lvd1689m`
- default pipeline type: `1024_cascade`
- first downstream generation model: sparse structure flow, `ckpts/ss_flow_img_dit_1_3B_64_bf16`

The vendor image-conditioning call sequence is:

```text
preprocess_image(...)
get_cond(image, resolution, include_neg_cond=True)
image_cond_model.image_size = resolution
cond = image_cond_model(image)
neg_cond = zeros_like(cond)
sample_sparse_structure(cond, resolution, ...)
```

The sparse structure flow config expects `cond_channels=1024`. Its checkpoint contains cross-attention keys such as `blocks.0.cross_attn.to_kv.weight`, which should become the first downstream inspection target once conditioning metadata exists.

## Proposed Runtime Shape

Add a small forward-trace module, likely `src/mlx_spatial/trellis2_forward.py`, with dependency-free dataclasses:

- `Trellis2StageOutput`: stage, names, shapes, dtypes, details
- `Trellis2ForwardTraceResult`: completed stages, outputs, blocker
- `Trellis2ConditioningConfig`: model name, model family, resolution, expected feature width

The public pipeline can expose either:

- `Trellis2InferencePipeline.attempt_forward_trace(image_path, ...)`, or
- `Trellis2InferencePipeline.attempt(image_path, forward_trace=True, ...)`

The sibling method is cleaner because it avoids changing the semantics of the existing blocker-oriented `attempt(...)` unexpectedly.

## Conditioning Strategy

The implementation should split three concerns:

1. Config discovery: parse `pipeline.json` and generation checkpoint configs.
2. Image tensor preparation: convert the preprocessed PIL image to MLX tensor layout and normalization expected by DINOv3.
3. DINOv3 model availability/port assessment: identify whether local model weights/config are available and whether an MLX path can start.

If DINOv3 assets are not local, return a precise `image-conditioning` blocker naming `facebook/dinov3-vitl16-pretrain-lvd1689m`; do not download it automatically.

If DINOv3 assets are local but the MLX implementation cannot run, return the first exact unsupported DINOv3 operation, config field, module, or checkpoint key.

If conditioning output is produced, record tensor metadata and dispatch to the sparse-structure boundary.

## Downstream Boundary

The first downstream stage is `sparse-structure-sampling`. This change should not implement the full sampler. It should:

- read the sparse structure flow config;
- inspect the first required checkpoint keys;
- validate that conditioning feature width matches the expected `cond_channels`;
- return a precise blocker for the first missing sparse-flow module/op/sampler step.

Fake-fixture tests may inject a conditioning tensor to verify the downstream dispatch and blocker contract without requiring DINOv3.

## Verification Strategy

Default tests use fake pipeline/config/checkpoint fixtures and tiny generated images. Real local evidence uses `inputs/trellis2/demo-alpha.webp` and `weights/trellis2`, writing attempt artifacts under ignored `outputs/`.
