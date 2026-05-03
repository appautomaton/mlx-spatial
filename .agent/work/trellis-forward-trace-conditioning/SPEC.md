# SPEC: TRELLIS.2 Forward Trace Conditioning

## Bounded Goal

Run the verified alpha-input TRELLIS.2 attempt forward in MLX using local `weights/trellis2`, replacing the current `image-conditioning` blanket blocker with either real conditioning output and the next downstream attempt or the first exact unsupported operation, module, config, or checkpoint-key blocker.

## Selected Lenses

- product
- engineering
- runtime

## Constraints

- Use the existing `mlx-spatial` Python package, `mlx_spatial.trellis2_inference` attempt surface, and current structured blocker shape.
- Use `weights/trellis2` as the local TRELLIS.2 asset root and `inputs/trellis2/demo-alpha.webp` as the primary real local attempt input.
- Stay MLX-first: runtime code must not import PyTorch, TorchVision, Transformers, Hugging Face Hub, ONNX Runtime, or vendor modules.
- Reference vendor code and configs may be inspected, but the implementation must not depend on executing `vendors/TRELLIS.2`.
- Do not solve RMBG/BiRefNet deformable convolution in this change; alpha inputs are the path through this slice.
- Stop at the first real unsupported image-conditioning or downstream TRELLIS operation/key/module rather than faking tensors or claiming full inference.
- Default tests must not require real TRELLIS.2 weights, network access, Hugging Face credentials, PyTorch, TorchVision, Transformers, ONNX Runtime, or vendor imports.
- Any real-weight attempt artifacts must live under ignored `outputs/` and must not commit weights or large generated outputs.

## Required Behavior

- Add a forward-trace attempt path that starts after verified alpha preprocessing and tries to enter TRELLIS.2 `image-conditioning` with MLX arrays.
- Inspect the local TRELLIS.2 config/checkpoint structure needed to identify the image-conditioning asset path, expected inputs, and expected output tensor contract.
- Load only the selected local checkpoint/config tensors needed for the attempted stage; avoid accidental full-checkpoint loads unless deliberately bounded and documented.
- If image-conditioning can be executed far enough to produce a real MLX tensor output, record its shape, dtype, and stage name, then attempt to dispatch into the first downstream TRELLIS.2 stage in pipeline order.
- If image-conditioning cannot execute, return a structured blocker at `image-conditioning` naming the exact missing operation, module, config, asset, or checkpoint key.
- If image-conditioning succeeds but the first downstream stage cannot execute, return a structured blocker at that downstream stage naming the exact missing operation, module, config, asset, or checkpoint key.
- Preserve existing behavior for readiness checks, alpha preprocessing, RGB/RMBG blocker propagation, and missing/invalid image blockers.
- Add fake-fixture tests that validate forward-trace routing, stage-output metadata, and exact blocker propagation without real weights.
- Add real local attempt evidence showing how far `inputs/trellis2/demo-alpha.webp` advances with current `weights/trellis2`.

## Acceptance Criteria

- `Trellis2InferencePipeline.attempt(...)` or an explicitly named sibling method can run the alpha demo through preprocessing and then invoke the forward-trace conditioning path.
- For fake fixtures, a successful conditioning stub or minimal MLX path advances the completed stages past `image-conditioning` and records stage output metadata.
- For fake fixtures, missing config/checkpoint keys or unsupported operations return blockers with `stage`, `operation`, `reference`, `reason`, and `next_slice`.
- The real local alpha attempt no longer returns the old generic `image-conditioning` blocker; it either records real conditioning output metadata or reports a more precise `image-conditioning` or downstream-stage blocker.
- The real local attempt evidence names the input path, TRELLIS.2 root, completed stages, tensor metadata if produced, and the final blocker if any.
- Default `uv run pytest` passes without real weights, network access, Hugging Face credentials, PyTorch, TorchVision, Transformers, ONNX Runtime, or vendor imports.
- Runtime dependency metadata still excludes PyTorch, TorchVision, Transformers, Hugging Face Hub, and ONNX Runtime.
- No real weights, large outputs, or generated attempt artifacts are tracked by git.
- README or TRELLIS docs state the new forward-trace boundary and the next known blocker.

## Blocking Questions Or Assumptions

- Assumption: `weights/trellis2` is already downloaded and validates for real local attempt evidence, but default tests must remain fake-fixture based.
- Assumption: the alpha demo image is enough to bypass RMBG and exercise the TRELLIS.2 core path.
- Assumption: vendor source can be used for static reference paths and stage order, but not runtime imports.
- Assumption: a precise blocker is an acceptable result if a real MLX implementation gap is reached before conditioning output is produced.
- Assumption: "aggressive" means follow the real pipeline call chain into the first downstream boundary, not implement unrelated later stages speculatively.

## Anti-Goals

- Do not implement or replace RMBG/BiRefNet `deform_conv2d`.
- Do not add PyTorch, TorchVision, Transformers, Hugging Face Hub, ONNX Runtime, or vendor runtime dependencies.
- Do not silently download or mutate model weights.
- Do not implement full sparse-structure sampling, SLat sampling, decoders, texture baking, mesh extraction, or GLB/OBJ export unless reached naturally as a blocker target.
- Do not fake image-conditioning tensors or placeholder downstream tensors to advance the pipeline.
- Do not claim full TRELLIS.2 image-to-3D inference works.
