# MapAnything MLX Parity Backbone Spec

## Bounded Goal

Implement the first parity-first MapAnything MLX slice in `mlx-spatial`: create the MapAnything module boundary, load/map the local config and weights far enough to verify the core backbone/preprocessing path, and establish numeric parity checks against the vendored PyTorch reference without adding Torch to the runtime package.

## Broader Intent

This is the first implementation step toward a full MLX-native MapAnything image-only reconstruction pipeline using the local best-performance `facebook/map-anything` CC-BY-NC 4.0 weights and the official Desk example input.

## Work Scale and Shape

- Scale: capability
- Shape: mixed parity + coverage + feature foundation

## Selected Lenses

- **product**: Adds MapAnything as the next scene-level reconstruction path in `mlx-spatial`, with a credible route from official demo input to MLX inference.
- **engineering**: Establishes the module layout, config/weight mapping, preprocessing parity, and first numeric gates needed before deeper model porting.

## Target User or Stakeholder

Developers running scene-level 3D reconstruction on Apple Silicon who want MapAnything behavior through `mlx-spatial` without requiring Torch at runtime.

## Reference Sources

- Vendored reference implementation: `vendors/map-anything`
- Local best-performance model weights: `weights/map-anything`
- Official image-only smoke input: `inputs/map-anything/desk`
- Reference/dev environment: `torch-ref` dependency group only

## Required Outcome

The repo gains a MapAnything MLX foundation that is executable and verifiable, but does not overclaim full end-to-end reconstruction. The implementation must:

1. Add `mapanything_*` source modules following the existing flat `src/mlx_spatial` package style.
2. Represent the local MapAnything config and required model substructure in MLX-facing dataclasses/helpers.
3. Inspect and map relevant `weights/map-anything/model.safetensors` keys into the first ported MLX model boundary.
4. Match vendored `load_images(...)` preprocessing behavior for the official Desk images and small synthetic inputs.
5. Port the first stable model-stage/backbone boundary selected during implementation, with numeric parity checks against vendored Torch reference outputs.
6. Provide a first Desk image-only smoke surface that proves the new MapAnything path loads assets, preprocesses images, and runs the implemented MLX stage without requiring `vendors/` at runtime.

The first stable model-stage boundary should be the deepest boundary that is practical without turning this change into the full MapAnything port. If DINOv2/UniCeption internals make the full encoder too large for this slice, the accepted boundary is the config/weight/preprocessing layer plus the first independently testable MLX block or adapter.

## Constraints

- Do not add Torch, TorchVision, UniCeption, or OpenCV to runtime `dependencies`.
- Torch is allowed only through the existing `torch-ref` group for parity capture/checks.
- `vendors/map-anything` is a dev/reference source and must not be imported by package runtime code.
- `weights/map-anything` and `inputs/map-anything/desk` stay local assets; the package must not bundle or redistribute them.
- The `facebook/map-anything` weights are CC-BY-NC 4.0 and are used as local research/reference assets only.
- Training, DA3 wrappers, external model wrappers, Apache model support, Gradio UI parity, and full all-task MapAnything support are out of scope for this change.
- Scratch parity captures and debug dumps use `/tmp`, except committed small fixtures under `tests/fixtures/` when they are intentionally needed for regression tests.

## Risks

- **DINOv2/UniCeption mapping risk:** the encoder stack may not map cleanly into existing MLX primitives in one change. Mitigation: choose the deepest stable boundary that can be verified now and record the next blocked boundary in the plan.
- **Weight-key mismatch risk:** safetensors keys may require non-trivial remapping. Mitigation: add an inspect/report path and tests for expected key groups before numeric model tests.
- **Smoke-vs-parity risk:** Desk images are useful for demo parity but too coarse for numeric debugging. Mitigation: add targeted synthetic or captured fixtures for parity, and use Desk only for smoke.
- **License/package risk:** local CC-BY-NC weights must not leak into wheel/sdist. Mitigation: package checks include artifact inspection and existing sdist exclude rules.

## Acceptance Criteria

| AC | Requirement | Check |
|---|---|---|
| MA-01 | MapAnything source surface exists | New `src/mlx_spatial/mapanything_*.py` modules define config/assets/preprocess/model-stage boundaries consistent with existing package style. |
| MA-02 | Runtime deps remain MLX-first | `pyproject.toml` runtime dependencies do not gain Torch, TorchVision, UniCeption, OpenCV, or vendor-only packages. |
| MA-03 | Local asset inspection works | A MapAnything helper validates or inspects `weights/map-anything/{config.json,model.safetensors}` and reports model/config facts needed by this slice. |
| MA-04 | Weight mapping is explicit | Tests cover the expected checkpoint key groups for the implemented MLX boundary and fail clearly when required keys are missing. |
| MA-05 | Preprocessing parity passes | MLX preprocessing matches the vendored image loader contract for shape, resize bucket, dtype/range, DINOv2 normalization, and Desk image handling. |
| MA-06 | First model-stage parity passes | The implemented MLX stage matches captured Torch reference tensors within a documented tolerance for at least one deterministic fixture. |
| MA-07 | Desk smoke path runs | A local smoke command or test loads `inputs/map-anything/desk`, preprocesses both images, runs the implemented MLX stage, and emits inspectable metrics or tensor summaries without requiring Torch. |
| MA-08 | Torch reference is opt-in | Torch-based parity capture/checks are marked or isolated so default runtime/package usage does not require Torch. |
| MA-09 | Existing repo checks remain healthy | Targeted new tests pass, existing affected tests pass, `uv lock --check` passes, and package artifact checks confirm weights/inputs/vendors are not included. |

## Scope Coverage Decisions

- **Included:** MLX MapAnything module skeleton, config/weight inspection and mapping, image preprocessing parity, first model-stage/backbone parity, official Desk smoke input, and opt-in Torch reference checks.
- **Deferred:** complete image-to-3D reconstruction output, GLB/export parity, all-task multimodal input support, training, DA3/external wrappers, Apache model variant support, and commercial/default model policy.
- **Anti-goals:** standalone gap-report deliverable, runtime Torch dependency, bundling weights, Gradio UI parity, and claiming full MapAnything parity before model-stage checks prove it.

## Anti-Goals

- Full MapAnything training or fine-tuning.
- DA3 or external model implementation.
- Apache model download/support in this slice.
- Gradio app reproduction.
- Weight redistribution or package bundling of `weights/map-anything`.
- End-to-end quality claims based only on visual smoke output.

## Blocking Questions or Assumptions

No blocking questions remain. The load-bearing assumption is that a meaningful first MLX model-stage boundary can be selected after reading the vendored module/key structure; if the full encoder is too large, the slice must stop at the deepest independently verified boundary and record the next concrete blocker for planning.
