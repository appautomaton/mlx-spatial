"""Inference-only TRELLIS.2 pipeline attempt surface."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import mlx.core as mx
import numpy as np

from .mlx_memory import clear_mlx_cache
from .ovoxel import fill_flexible_dual_grid_mesh_holes, flexi_dual_grid_fields_to_mesh
from .trellis2 import (
    TRELLIS2_PROBE_GROUPS,
    inspect_trellis2_probe,
    load_trellis2_probe,
    validate_trellis2_assets,
)
from .trellis2_decode import (
    read_structured_latent_decoder_config,
    run_shape_decoder_to_fields,
    run_texture_decoder_to_representation,
    run_shape_decoder_upsample_coordinates,
)
from .trellis2_export import (
    TRELLIS2_GLB_DEFAULT_FACE_TARGET,
    TRELLIS2_TEXTURE_BAKE_BACKENDS,
    TRELLIS2_XATLAS_AUTO_FACE_GUARD,
    Trellis2ExportArtifact,
    bake_trellis2_texture_fields_mac_native,
    missing_trellis2_mac_export_dependencies,
    postprocess_trellis2_mesh_for_glb,
    resolve_trellis2_xatlas_face_guard,
    validate_trellis2_export_path,
    write_flexible_dual_grid_obj,
    write_trellis2_textured_glb,
)
from .trellis2_forward import (
    Trellis2ForwardBlocker,
    Trellis2StageOutput,
    Trellis2ForwardTraceResult,
    assess_dinov3_conditioning,
    discover_trellis2_conditioning_config,
    dispatch_sparse_structure_sampling,
    prepare_dinov3_image_tensor,
    sparse_structure_target_resolution,
    _apply_slat_normalization,
)
from .trellis2_preprocess import Trellis2PreprocessBlocker, preprocess_trellis2_image
from .trellis2_sparse_structure import (
    fake_sparse_structure_sampling_metadata,
    probe_sparse_structure_decoder_boundary,
    probe_sparse_structure_forward_boundary,
    read_sparse_structure_decoder_config,
    read_sparse_structure_flow_config,
)
from .trellis2_slat import (
    probe_shape_slat_forward_boundary,
    probe_texture_slat_forward_boundary,
    read_slat_flow_config,
    select_shape_slat_route,
    select_texture_slat_route,
)

TRELLIS2_SHAPE_DECODER_TOKEN_LIMIT = 1_000_000
TRELLIS2_SHAPE_MAX_NUM_TOKENS = 49_152
TRELLIS2_SHAPE_SEED = 42
TRELLIS2_TEXTURE_SIZE = 1024

Trellis2StageStatus = Literal["ready", "blocked", "unimplemented"]

TRELLIS2_INFERENCE_STAGES = (
    "input-image",
    "asset-config-validation",
    "checkpoint-probe-readiness",
    "image-preprocessing-background",
    "image-conditioning",
    "sparse-structure-sampling",
    "shape-slat-sampling",
    "texture-slat-sampling",
    "shape-decoder",
    "texture-decoder",
    "mesh-export",
)


@dataclass(frozen=True)
class Trellis2InferenceBlocker:
    stage: str
    operation: str
    reference: str
    reason: str
    next_slice: str


@dataclass(frozen=True)
class Trellis2StageReport:
    stage: str
    status: Trellis2StageStatus
    detail: str
    blocker: Trellis2InferenceBlocker | None = None


@dataclass(frozen=True)
class Trellis2ReadinessReport:
    root: Path
    ready: bool
    stages: tuple[Trellis2StageReport, ...]
    blocker: Trellis2InferenceBlocker | None = None


@dataclass(frozen=True)
class Trellis2AttemptReport:
    root: Path
    image_path: Path
    completed_stages: tuple[str, ...]
    blocker: Trellis2InferenceBlocker | None

    @property
    def completed(self) -> bool:
        return self.blocker is None


@dataclass(frozen=True)
class Trellis2ShapeGenerationResult:
    trace: Trellis2ForwardTraceResult
    artifact: Trellis2ExportArtifact | None = None

    @property
    def ready(self) -> bool:
        return self.artifact is not None and self.trace.blocker is None


@dataclass(frozen=True)
class Trellis2TexturedGenerationResult:
    trace: Trellis2ForwardTraceResult
    artifact: Trellis2ExportArtifact | None = None

    @property
    def ready(self) -> bool:
        return self.artifact is not None and self.trace.blocker is None


class Trellis2InferencePipeline:
    """Inference-only TRELLIS.2 pipeline attempt that stops at missing MLX compute."""

    stages = TRELLIS2_INFERENCE_STAGES

    def __init__(self, root: str | Path = "weights/trellis2", *, rmbg_root: str | Path | None = None):
        self.root = Path(root)
        self.rmbg_root = Path(rmbg_root) if rmbg_root is not None else None

    def dry_run(self, *, load_probes: bool = False) -> Trellis2ReadinessReport:
        validation = validate_trellis2_assets(self.root)
        reports: list[Trellis2StageReport] = [
            Trellis2StageReport(
                stage="asset-config-validation",
                status="ready" if validation.ready else "blocked",
                detail=f"present={len(validation.present)} missing={len(validation.missing)}",
                blocker=None if validation.ready else _missing_assets_blocker(tuple(validation.missing)),
            )
        ]
        if not validation.ready:
            return Trellis2ReadinessReport(
                root=self.root,
                ready=False,
                stages=tuple(reports),
                blocker=reports[0].blocker,
            )

        probe_count = 0
        for group in TRELLIS2_PROBE_GROUPS:
            tensors = load_trellis2_probe(self.root, group) if load_probes else inspect_trellis2_probe(self.root, group.name)
            probe_count += len(tensors)
        reports.append(
            Trellis2StageReport(
                stage="checkpoint-probe-readiness",
                status="ready",
                detail=f"groups={len(TRELLIS2_PROBE_GROUPS)} tensors={probe_count} loaded={load_probes}",
            )
        )

        blocker = _unimplemented_blocker("image-preprocessing-background")
        for stage in TRELLIS2_INFERENCE_STAGES[3:]:
            reports.append(
                Trellis2StageReport(
                    stage=stage,
                    status="unimplemented",
                    detail="MLX compute stage is not implemented",
                    blocker=_unimplemented_blocker(stage),
                )
            )
        return Trellis2ReadinessReport(root=self.root, ready=False, stages=tuple(reports), blocker=blocker)

    def attempt(self, image_path: str | Path, *, load_probes: bool = False) -> Trellis2AttemptReport:
        image = Path(image_path)
        if not image.is_file():
            return Trellis2AttemptReport(
                root=self.root,
                image_path=image,
                completed_stages=(),
                blocker=Trellis2InferenceBlocker(
                    stage="input-image",
                    operation="image file validation",
                    reference="vendors/trellis-mac/generate.py:68-70",
                    reason=f"image file not found: {image}",
                    next_slice="provide or generate a local sample image before compute-stage work",
                ),
            )

        readiness = self.dry_run(load_probes=load_probes)
        completed = ["input-image"]
        for report in readiness.stages:
            if report.status == "ready":
                completed.append(report.stage)
                continue
            if report.stage == "image-preprocessing-background":
                preprocessed = preprocess_trellis2_image(image, rmbg_root=self.rmbg_root)
                if preprocessed.ready:
                    completed.append("image-preprocessing-background")
                    return Trellis2AttemptReport(
                        root=self.root,
                        image_path=image,
                        completed_stages=tuple(completed),
                        blocker=_unimplemented_blocker("image-conditioning"),
                    )
                return Trellis2AttemptReport(
                    root=self.root,
                    image_path=image,
                    completed_stages=tuple(completed),
                    blocker=_preprocess_blocker(preprocessed.blocker),
                )
            return Trellis2AttemptReport(
                root=self.root,
                image_path=image,
                completed_stages=tuple(completed),
                blocker=report.blocker,
            )
        return Trellis2AttemptReport(root=self.root, image_path=image, completed_stages=tuple(completed), blocker=None)

    def attempt_forward_trace(
        self,
        image_path: str | Path,
        *,
        load_probes: bool = False,
        dino_root: str | Path | None = None,
        conditioning: mx.array | None = None,
        slat_steps: int | None = None,
        decoder_token_limit: int | None = None,
    ) -> Trellis2ForwardTraceResult:
        image = Path(image_path)
        if not image.is_file():
            return Trellis2ForwardTraceResult(
                root=self.root,
                image_path=image,
                completed_stages=(),
                blocker=Trellis2ForwardBlocker(
                    stage="input-image",
                    operation="image file validation",
                    reference="vendors/trellis-mac/generate.py:68-70",
                    reason=f"image file not found: {image}",
                    next_slice="provide or generate a local sample image before compute-stage work",
                ),
            )

        readiness = self.dry_run(load_probes=load_probes)
        completed = ["input-image"]
        for report in readiness.stages:
            if report.status == "ready":
                completed.append(report.stage)
                continue
            if report.stage != "image-preprocessing-background":
                return Trellis2ForwardTraceResult(
                    root=self.root,
                    image_path=image,
                    completed_stages=tuple(completed),
                    blocker=_forward_blocker(report.blocker),
                )

            preprocessed = preprocess_trellis2_image(image, rmbg_root=self.rmbg_root)
            if not preprocessed.ready or preprocessed.image is None:
                return Trellis2ForwardTraceResult(
                    root=self.root,
                    image_path=image,
                    completed_stages=tuple(completed),
                    blocker=_forward_blocker(_preprocess_blocker(preprocessed.blocker)),
                )
            completed.append("image-preprocessing-background")

            discovery = discover_trellis2_conditioning_config(self.root)
            if not discovery.ready or discovery.config is None:
                return Trellis2ForwardTraceResult(
                    root=self.root,
                    image_path=image,
                    completed_stages=tuple(completed),
                    blocker=discovery.blocker,
                )
            config = discovery.config
            if slat_steps is not None:
                if slat_steps <= 0:
                    return Trellis2ForwardTraceResult(
                        root=self.root,
                        image_path=image,
                        completed_stages=tuple(completed),
                        blocker=Trellis2ForwardBlocker(
                            stage="shape-slat-sampling",
                            operation="trace SLat step override validation",
                            reference="src/mlx_spatial/trellis2_inference.py:Trellis2InferencePipeline.attempt_forward_trace",
                            reason=f"slat_steps must be positive, got {slat_steps}",
                            next_slice="use a positive SLat step override for development traces",
                        ),
                    )
                config = replace(
                    config,
                    shape_slat_sampler=replace(config.shape_slat_sampler, steps=slat_steps),
                    texture_slat_sampler=replace(config.texture_slat_sampler, steps=slat_steps),
                )

            image_tensor = prepare_dinov3_image_tensor(
                preprocessed.image.image,
                image_size=config.conditioning_resolution,
            )
            output, blocker = assess_dinov3_conditioning(
                self.root,
                config,
                dino_root=dino_root,
                image_tensor=image_tensor,
                conditioning=conditioning,
            )
            if blocker is not None:
                return Trellis2ForwardTraceResult(
                    root=self.root,
                    image_path=image,
                    completed_stages=tuple(completed),
                    blocker=blocker,
                )
            outputs = (output,) if output is not None else ()
            completed.append("image-conditioning")
            sparse_output, sparse_blocker = dispatch_sparse_structure_sampling(
                self.root,
                config,
                output,
                decoder_token_limit=decoder_token_limit,
            )
            if sparse_output is not None:
                outputs = (*outputs, sparse_output)
                completed.append("sparse-structure-sampling")
            return Trellis2ForwardTraceResult(
                root=self.root,
                image_path=image,
                completed_stages=tuple(completed),
                outputs=outputs,
                blocker=sparse_blocker,
            )
        return Trellis2ForwardTraceResult(root=self.root, image_path=image, completed_stages=tuple(completed))

    def generate_shape_obj(
        self,
        image_path: str | Path,
        *,
        output_path: str | Path,
        dino_root: str | Path | None = None,
        conditioning: mx.array | None = None,
        slat_steps: int | None = None,
        pipeline_type: str | None = None,
        seed: int = TRELLIS2_SHAPE_SEED,
        max_num_tokens: int = TRELLIS2_SHAPE_MAX_NUM_TOKENS,
        decoder_token_limit: int = TRELLIS2_SHAPE_DECODER_TOKEN_LIMIT,
        retain_trace_payloads: bool = True,
    ) -> Trellis2ShapeGenerationResult:
        image = Path(image_path)
        output = Path(output_path)
        if output.suffix.lower() == ".glb":
            return Trellis2ShapeGenerationResult(
                trace=Trellis2ForwardTraceResult(
                    root=self.root,
                    image_path=image,
                    completed_stages=(),
                    blocker=Trellis2ForwardBlocker(
                        stage="mesh-export",
                        operation="TRELLIS.2 textured GLB export",
                        reference=str(output_path),
                        reason=(
                            "generate-shape is an exact shape-only path and writes OBJ; texture SLat, "
                            "texture decoder, MeshWithVoxel, UV baking, and GLB export are not implemented yet"
                        ),
                        next_slice="finish exact shape OBJ first, then implement texture and GLB export",
                    ),
                )
            )
        if output.suffix.lower() != ".obj":
            return Trellis2ShapeGenerationResult(
                trace=Trellis2ForwardTraceResult(
                    root=self.root,
                    image_path=image,
                    completed_stages=(),
                    blocker=Trellis2ForwardBlocker(
                        stage="mesh-export",
                        operation="shape OBJ output format validation",
                        reference=str(output_path),
                        reason=f"generate-shape only writes .obj outputs, got {output.suffix or '<none>'}",
                        next_slice="choose a .obj output path under outputs/ for shape generation",
                    ),
                )
            )
        try:
            validate_trellis2_export_path(output_path)
        except ValueError as error:
            return Trellis2ShapeGenerationResult(
                trace=Trellis2ForwardTraceResult(
                    root=self.root,
                    image_path=image,
                    completed_stages=(),
                    blocker=Trellis2ForwardBlocker(
                        stage="mesh-export",
                        operation="TRELLIS.2 export path validation",
                        reference=str(output_path),
                        reason=str(error),
                        next_slice="choose a .obj output path under outputs/ for shape generation",
                    ),
                )
            )
        if max_num_tokens <= 0:
            return Trellis2ShapeGenerationResult(
                trace=Trellis2ForwardTraceResult(
                    root=self.root,
                    image_path=image,
                    completed_stages=(),
                    blocker=Trellis2ForwardBlocker(
                        stage="shape-slat-sampling",
                        operation="cascade token cap validation",
                        reference="--max-num-tokens",
                        reason=f"max-num-tokens must be positive, got {max_num_tokens}",
                        next_slice="use a positive max-num-tokens cap for cascade shape generation",
                    ),
                )
            )
        if decoder_token_limit <= 0:
            return Trellis2ShapeGenerationResult(
                trace=Trellis2ForwardTraceResult(
                    root=self.root,
                    image_path=image,
                    completed_stages=(),
                    blocker=Trellis2ForwardBlocker(
                        stage="shape-decoder",
                        operation="decoder token limit validation",
                        reference="--decoder-token-limit",
                        reason=f"decoder-token-limit must be positive, got {decoder_token_limit}",
                        next_slice="use a positive decoder-token-limit for shape generation",
                    ),
                )
            )
        if not image.is_file():
            return Trellis2ShapeGenerationResult(
                trace=Trellis2ForwardTraceResult(
                    root=self.root,
                    image_path=image,
                    completed_stages=(),
                    blocker=Trellis2ForwardBlocker(
                        stage="input-image",
                        operation="image file validation",
                        reference="vendors/trellis-mac/generate.py:68-70",
                        reason=f"image file not found: {image}",
                        next_slice="provide or generate a local sample image before shape generation",
                    ),
                )
            )

        readiness = self.dry_run(load_probes=False)
        completed = ["input-image"]
        for report in readiness.stages:
            if report.status == "ready":
                completed.append(report.stage)
                continue
            if report.stage != "image-preprocessing-background":
                return Trellis2ShapeGenerationResult(
                    trace=Trellis2ForwardTraceResult(
                        root=self.root,
                        image_path=image,
                        completed_stages=tuple(completed),
                        blocker=_forward_blocker(report.blocker),
                    )
                )
            break

        preprocessed = preprocess_trellis2_image(image, rmbg_root=self.rmbg_root)
        if not preprocessed.ready or preprocessed.image is None:
            return Trellis2ShapeGenerationResult(
                trace=Trellis2ForwardTraceResult(
                    root=self.root,
                    image_path=image,
                    completed_stages=tuple(completed),
                    blocker=_forward_blocker(_preprocess_blocker(preprocessed.blocker)),
                )
            )
        completed.append("image-preprocessing-background")

        discovery = discover_trellis2_conditioning_config(self.root)
        if not discovery.ready or discovery.config is None:
            return Trellis2ShapeGenerationResult(
                trace=Trellis2ForwardTraceResult(
                    root=self.root,
                    image_path=image,
                    completed_stages=tuple(completed),
                    blocker=discovery.blocker,
                )
            )
        config = discovery.config
        selected_pipeline_type = pipeline_type or config.default_pipeline_type
        try:
            select_shape_slat_route(selected_pipeline_type)
            sparse_resolution = sparse_structure_target_resolution(selected_pipeline_type)
            decode_resolution = _decode_resolution(selected_pipeline_type)
        except ValueError as error:
            return Trellis2ShapeGenerationResult(
                trace=Trellis2ForwardTraceResult(
                    root=self.root,
                    image_path=image,
                    completed_stages=tuple(completed),
                    blocker=Trellis2ForwardBlocker(
                        stage="shape-slat-sampling",
                        operation="shape generation pipeline type validation",
                        reference="--pipeline-type",
                        reason=str(error),
                        next_slice="choose one of 512, 1024, 1024_cascade, or 1536_cascade",
                    ),
                )
            )
        config = replace(config, default_pipeline_type=selected_pipeline_type)
        if slat_steps is not None:
            if slat_steps <= 0:
                return Trellis2ShapeGenerationResult(
                    trace=Trellis2ForwardTraceResult(
                        root=self.root,
                        image_path=image,
                        completed_stages=tuple(completed),
                        blocker=Trellis2ForwardBlocker(
                            stage="shape-slat-sampling",
                            operation="shape generation SLat step override validation",
                            reference="--slat-steps",
                            reason=f"slat_steps must be positive, got {slat_steps}",
                            next_slice="use a positive SLat step override for shape generation",
                        ),
                    )
                )
            config = replace(config, shape_slat_sampler=replace(config.shape_slat_sampler, steps=slat_steps))

        mx.random.seed(int(seed))
        cond_outputs: dict[int, Trellis2StageOutput] = {}
        outputs = []
        for resolution in _conditioning_resolutions(selected_pipeline_type):
            cond_config = replace(config, conditioning_resolution=resolution)
            image_tensor = prepare_dinov3_image_tensor(preprocessed.image.image, image_size=resolution)
            cond_output, blocker = assess_dinov3_conditioning(
                self.root,
                cond_config,
                dino_root=dino_root,
                image_tensor=image_tensor,
                conditioning=conditioning,
            )
            if blocker is not None or cond_output is None:
                return Trellis2ShapeGenerationResult(
                    trace=Trellis2ForwardTraceResult(
                        root=self.root,
                        image_path=image,
                        completed_stages=tuple(completed),
                        outputs=tuple(outputs),
                        blocker=blocker,
                    )
                )
            cond_output = Trellis2StageOutput(
                stage=cond_output.stage,
                name=f"cond_{resolution}",
                shape=cond_output.shape,
                dtype=cond_output.dtype,
                detail=f"{cond_output.detail}; pipeline_type={selected_pipeline_type} conditioning_resolution={resolution}",
                payload=cond_output.payload,
            )
            cond_outputs[resolution] = cond_output
            outputs.append(cond_output if retain_trace_payloads else _without_stage_payload(cond_output))
        cond_512 = cond_outputs[512]
        cond_1024 = cond_outputs.get(1024)
        completed.append("image-conditioning")

        if cond_512.payload is None:
            return Trellis2ShapeGenerationResult(
                trace=Trellis2ForwardTraceResult(
                    root=self.root,
                    image_path=image,
                    completed_stages=tuple(completed),
                    outputs=tuple(outputs),
                    blocker=Trellis2ForwardBlocker(
                        stage="image-conditioning",
                        operation="DINOv3 cond_512 payload availability",
                        reference="vendors/TRELLIS.2/trellis2/pipelines/trellis2_image_to_3d.py:489-533",
                        reason="sparse structure sampling requires cond_512, but conditioning returned no payload",
                        next_slice="produce DINOv3 cond_512 before sparse structure sampling",
                    ),
                )
            )

        try:
            sparse_config = read_sparse_structure_flow_config(self.root, config.sparse_flow_config_path)
            metadata = fake_sparse_structure_sampling_metadata(
                sparse_config,
                steps=config.sparse_structure_sampler.steps,
                rescale_t=config.sparse_structure_sampler.rescale_t,
                guidance_interval=config.sparse_structure_sampler.guidance_interval,
            )
            sparse_probe = probe_sparse_structure_forward_boundary(
                Path(self.root) / config.sparse_flow_checkpoint_path,
                sparse_config,
                conditioning=cond_512.payload,
                steps=config.sparse_structure_sampler.steps,
                rescale_t=config.sparse_structure_sampler.rescale_t,
                guidance_strength=config.sparse_structure_sampler.guidance_strength,
                guidance_rescale=config.sparse_structure_sampler.guidance_rescale,
                guidance_interval=config.sparse_structure_sampler.guidance_interval,
                sigma_min=config.sparse_structure_sampler.sigma_min,
            )
        except (FileNotFoundError, ValueError) as error:
            return Trellis2ShapeGenerationResult(
                trace=Trellis2ForwardTraceResult(
                    root=self.root,
                    image_path=image,
                    completed_stages=tuple(completed),
                    outputs=tuple(outputs),
                    blocker=Trellis2ForwardBlocker(
                        stage="sparse-structure-sampling",
                        operation="sparse flow config/checkpoint forward probe",
                        reference=config.sparse_flow_checkpoint_path,
                        reason=str(error),
                        next_slice="map the sparse structure flow before shape generation",
                    ),
                )
            )
        if sparse_probe.sampled_latent is None or sparse_probe.sampled_latent_shape is None:
            return Trellis2ShapeGenerationResult(
                trace=Trellis2ForwardTraceResult(
                    root=self.root,
                    image_path=image,
                    completed_stages=tuple(completed),
                    outputs=tuple(outputs),
                    blocker=Trellis2ForwardBlocker(
                        stage="sparse-structure-sampling",
                        operation=sparse_probe.blocker_operation,
                        reference=sparse_probe.checkpoint_path,
                        reason=(
                            f"sparse flow config validates noise shape {metadata.noise_shape}, "
                            f"FlowEuler steps={metadata.steps}, loads tensors {sparse_probe.loaded_tensor_names}, "
                            f"and {sparse_probe.blocker_detail}"
                        ),
                        next_slice="produce sampled sparse latent before shape generation",
                    ),
                )
            )
        outputs.append(
            _stage_output(
                "sparse-structure-sampling",
                "sparse_latent",
                sparse_probe.sampled_latent,
                f"MLX sparse structure FlowEuler sampler output after {metadata.steps} steps",
                retain_payload=retain_trace_payloads,
            )
        )
        completed.append("sparse-structure-sampling")

        try:
            sparse_decoder_config = read_sparse_structure_decoder_config(self.root, config.sparse_decoder_config_path)
            sparse_decoder_probe = probe_sparse_structure_decoder_boundary(
                Path(self.root) / config.sparse_decoder_checkpoint_path,
                sparse_decoder_config,
                sparse_latent=sparse_probe.sampled_latent,
                target_resolution=sparse_resolution,
            )
        except (FileNotFoundError, ValueError) as error:
            return Trellis2ShapeGenerationResult(
                trace=Trellis2ForwardTraceResult(
                    root=self.root,
                    image_path=image,
                    completed_stages=tuple(completed),
                    outputs=tuple(outputs),
                    blocker=Trellis2ForwardBlocker(
                        stage="sparse-structure-decoding",
                        operation="sparse structure decoder config/checkpoint probe",
                        reference=config.sparse_decoder_checkpoint_path,
                        reason=str(error),
                        next_slice="decode sparse structure coordinates before shape generation",
                    ),
                )
            )
        if sparse_decoder_probe.coordinates is None or sparse_decoder_probe.coordinates_shape is None:
            return Trellis2ShapeGenerationResult(
                trace=Trellis2ForwardTraceResult(
                    root=self.root,
                    image_path=image,
                    completed_stages=tuple(completed),
                    outputs=tuple(outputs),
                    blocker=Trellis2ForwardBlocker(
                        stage="sparse-structure-decoding",
                        operation=sparse_decoder_probe.blocker_operation,
                        reference=sparse_decoder_probe.checkpoint_path,
                        reason=sparse_decoder_probe.blocker_detail,
                        next_slice="produce sparse structure coordinates before shape SLat sampling",
                    ),
                )
            )

        try:
            shape_coordinates, shape_features, shape_detail, final_resolution = _sample_shape_slat_for_pipeline(
                self.root,
                config,
                sparse_decoder_probe.coordinates,
                cond_512.payload,
                cond_1024.payload if cond_1024 is not None else None,
                max_num_tokens=max_num_tokens,
                decoder_token_limit=decoder_token_limit,
            )
            shape_output = _stage_output(
                "shape-slat-sampling",
                "shape_slat",
                shape_features,
                shape_detail,
                retain_payload=retain_trace_payloads,
            )
        except (FileNotFoundError, ValueError) as error:
            return Trellis2ShapeGenerationResult(
                trace=Trellis2ForwardTraceResult(
                    root=self.root,
                    image_path=image,
                    completed_stages=tuple(completed),
                    outputs=tuple(outputs),
                    blocker=Trellis2ForwardBlocker(
                        stage="shape-slat-sampling",
                        operation="exact TRELLIS.2 shape SLat route execution",
                        reference=config.default_pipeline_type,
                        reason=str(error),
                        next_slice="complete exact non-approximate shape SLat sampling for the selected pipeline type",
                    ),
                )
            )
        outputs.append(shape_output)
        completed.append("shape-slat-sampling")

        try:
            shape_decoder_config = read_structured_latent_decoder_config(self.root, config.shape_decoder_config_path)
            shape_result = run_shape_decoder_to_fields(
                Path(self.root) / config.shape_decoder_checkpoint_path,
                shape_decoder_config,
                shape_coordinates,
                shape_features,
                decoder_token_limit=decoder_token_limit,
            )
            outputs.append(
                _stage_output(
                    "shape-decoder",
                    "shape_flexidualgrid_fields",
                    shape_result.fields,
                    (
                        f"full shape decoder completed {shape_result.probe.completed_levels} levels; "
                        f"coordinates {shape_result.probe.decoder_output_coordinate_shape}; "
                        f"subdivisions {shape_result.probe.subdivision_shapes}"
                    ),
                    retain_payload=retain_trace_payloads,
                )
            )
            completed.append("shape-decoder")
            mesh = flexi_dual_grid_fields_to_mesh(shape_result.coordinates, shape_result.fields, grid_size=final_resolution)
            mesh, hole_stats = fill_flexible_dual_grid_mesh_holes(mesh)
            outputs.append(
                _stage_output(
                    "mesh-export",
                    "shape_mesh_hole_fill",
                    mx.array(
                        [
                            hole_stats.boundary_edges_before,
                            hole_stats.clean_boundary_loops,
                            hole_stats.filled_loops,
                            hole_stats.skipped_large_loops,
                            hole_stats.skipped_complex_components,
                            hole_stats.vertices_added,
                            hole_stats.faces_added,
                        ],
                        dtype=mx.int32,
                    ),
                    (
                        "bounded FlexiDualGrid hole fill matching upstream max_hole_perimeter=3e-2; "
                        f"filled_loops={hole_stats.filled_loops}; faces_added={hole_stats.faces_added}; "
                        f"skipped_complex_components={hole_stats.skipped_complex_components}"
                    ),
                    retain_payload=retain_trace_payloads,
                )
            )
            artifact = write_flexible_dual_grid_obj(mesh, output_path)
            if not retain_trace_payloads:
                del mesh, shape_result
            clear_mlx_cache()
        except (FileNotFoundError, ValueError) as error:
            return Trellis2ShapeGenerationResult(
                trace=Trellis2ForwardTraceResult(
                    root=self.root,
                    image_path=image,
                    completed_stages=tuple(completed),
                    outputs=tuple(outputs),
                    blocker=Trellis2ForwardBlocker(
                        stage="shape-decoder",
                        operation="MLX shape decoder FlexiDualGrid mesh extraction",
                        reference=config.shape_decoder_checkpoint_path,
                        reason=str(error),
                        next_slice="complete exact shape decoder execution and FlexiDualGrid mesh extraction",
                    ),
                )
            )

        completed.append("mesh-export")
        return Trellis2ShapeGenerationResult(
            trace=Trellis2ForwardTraceResult(
                root=self.root,
                image_path=image,
                completed_stages=tuple(completed),
                outputs=tuple(outputs),
            ),
            artifact=artifact,
        )

    def generate_textured_glb(
        self,
        image_path: str | Path,
        *,
        output_path: str | Path,
        dino_root: str | Path | None = None,
        slat_steps: int | None = None,
        pipeline_type: str | None = None,
        seed: int = TRELLIS2_SHAPE_SEED,
        max_num_tokens: int | None = TRELLIS2_SHAPE_MAX_NUM_TOKENS,
        decoder_token_limit: int | None = TRELLIS2_SHAPE_DECODER_TOKEN_LIMIT,
        texture_size: int = TRELLIS2_TEXTURE_SIZE,
        glb_target_faces: int = TRELLIS2_GLB_DEFAULT_FACE_TARGET,
        xatlas_face_guard: int | str = TRELLIS2_XATLAS_AUTO_FACE_GUARD,
        xatlas_parallel_chunks: int = 0,
        texture_bake_backend: str = "trilinear",
        retain_trace_payloads: bool = True,
    ) -> Trellis2TexturedGenerationResult:
        image = Path(image_path)
        output = Path(output_path)
        if output.suffix.lower() != ".glb":
            return Trellis2TexturedGenerationResult(
                trace=Trellis2ForwardTraceResult(
                    root=self.root,
                    image_path=image,
                    completed_stages=(),
                    blocker=Trellis2ForwardBlocker(
                        stage="mesh-export",
                        operation="textured GLB output format validation",
                        reference=str(output_path),
                        reason=f"generate-textured only writes .glb outputs, got {output.suffix or '<none>'}",
                        next_slice="choose a .glb output path under outputs/ for textured generation",
                    ),
                )
            )
        try:
            validate_trellis2_export_path(output_path, suffixes=(".glb",))
        except (OSError, ValueError) as error:
            return Trellis2TexturedGenerationResult(
                trace=Trellis2ForwardTraceResult(
                    root=self.root,
                    image_path=image,
                    completed_stages=(),
                    blocker=Trellis2ForwardBlocker(
                        stage="mesh-export",
                        operation="TRELLIS.2 textured GLB export path validation",
                        reference=str(output_path),
                        reason=str(error),
                        next_slice="choose a .glb output path under outputs/ for textured generation",
                    ),
                )
            )
        if slat_steps is not None and slat_steps <= 0:
            return Trellis2TexturedGenerationResult(
                trace=Trellis2ForwardTraceResult(
                    root=self.root,
                    image_path=image,
                    completed_stages=(),
                    blocker=Trellis2ForwardBlocker(
                        stage="texture-slat-sampling",
                        operation="texture SLat sampler step validation",
                        reference="--slat-steps",
                        reason=f"slat_steps must be positive, got {slat_steps}",
                        next_slice="use a positive slat-steps value for textured generation",
                    ),
                )
            )
        if max_num_tokens is not None and max_num_tokens <= 0:
            return Trellis2TexturedGenerationResult(
                trace=Trellis2ForwardTraceResult(
                    root=self.root,
                    image_path=image,
                    completed_stages=(),
                    blocker=Trellis2ForwardBlocker(
                        stage="texture-slat-sampling",
                        operation="texture cascade token cap validation",
                        reference="--max-num-tokens",
                        reason=f"max-num-tokens must be positive, got {max_num_tokens}",
                        next_slice="use a positive max-num-tokens cap for textured generation",
                    ),
                )
            )
        if decoder_token_limit is not None and decoder_token_limit <= 0:
            return Trellis2TexturedGenerationResult(
                trace=Trellis2ForwardTraceResult(
                    root=self.root,
                    image_path=image,
                    completed_stages=(),
                    blocker=Trellis2ForwardBlocker(
                        stage="texture-decoder",
                        operation="texture decoder token limit validation",
                        reference="--decoder-token-limit",
                        reason=f"decoder-token-limit must be positive, got {decoder_token_limit}",
                        next_slice="use a positive decoder-token-limit for textured generation",
                    ),
                )
            )
        if texture_size <= 0:
            return Trellis2TexturedGenerationResult(
                trace=Trellis2ForwardTraceResult(
                    root=self.root,
                    image_path=image,
                    completed_stages=(),
                    blocker=Trellis2ForwardBlocker(
                        stage="mesh-export",
                        operation="texture size validation",
                        reference="--texture-size",
                        reason=f"texture-size must be positive, got {texture_size}",
                        next_slice="use a positive texture-size for textured generation",
                    ),
                )
            )
        if glb_target_faces <= 0:
            return Trellis2TexturedGenerationResult(
                trace=Trellis2ForwardTraceResult(
                    root=self.root,
                    image_path=image,
                    completed_stages=(),
                    blocker=Trellis2ForwardBlocker(
                        stage="mesh-export",
                        operation="GLB simplification target validation",
                        reference="--glb-target-faces",
                        reason=f"glb-target-faces must be positive, got {glb_target_faces}",
                        next_slice="use a positive GLB target face count",
                    ),
                )
            )
        try:
            resolve_trellis2_xatlas_face_guard(1, xatlas_face_guard)
        except ValueError as error:
            return Trellis2TexturedGenerationResult(
                trace=Trellis2ForwardTraceResult(
                    root=self.root,
                    image_path=image,
                    completed_stages=(),
                    blocker=Trellis2ForwardBlocker(
                        stage="mesh-export",
                        operation="xatlas face guard validation",
                        reference="--xatlas-face-guard",
                        reason=str(error),
                        next_slice="use 'auto' or a positive xatlas face guard",
                    ),
                )
            )
        if xatlas_parallel_chunks < 0:
            return Trellis2TexturedGenerationResult(
                trace=Trellis2ForwardTraceResult(
                    root=self.root,
                    image_path=image,
                    completed_stages=(),
                    blocker=Trellis2ForwardBlocker(
                        stage="mesh-export",
                        operation="xatlas parallel chunk validation",
                        reference="--xatlas-parallel-chunks",
                        reason=f"xatlas-parallel-chunks must be non-negative, got {xatlas_parallel_chunks}",
                        next_slice="use 0 for automatic xatlas chunking or a positive chunk count",
                    ),
                )
            )
        if texture_bake_backend not in TRELLIS2_TEXTURE_BAKE_BACKENDS:
            return Trellis2TexturedGenerationResult(
                trace=Trellis2ForwardTraceResult(
                    root=self.root,
                    image_path=image,
                    completed_stages=(),
                    blocker=Trellis2ForwardBlocker(
                        stage="mesh-export",
                        operation="texture bake backend validation",
                        reference="--texture-bake-backend",
                        reason=(
                            f"texture-bake-backend must be one of {TRELLIS2_TEXTURE_BAKE_BACKENDS}, "
                            f"got {texture_bake_backend}"
                        ),
                        next_slice="use trilinear for parity mode or kdtree for debug comparison",
                    ),
                )
            )
        missing_export_dependencies = missing_trellis2_mac_export_dependencies()
        if missing_export_dependencies:
            return Trellis2TexturedGenerationResult(
                trace=Trellis2ForwardTraceResult(
                    root=self.root,
                    image_path=image,
                    completed_stages=(),
                    blocker=Trellis2ForwardBlocker(
                        stage="mesh-export",
                        operation="Mac-native GLB export dependency validation",
                        reference="xatlas, scipy, fast_simplification",
                        reason=(
                            "generate-textured requires Mac-native GLB export dependencies; "
                            f"missing {', '.join(missing_export_dependencies)}"
                        ),
                        next_slice="install xatlas, scipy, and fast-simplification before textured GLB export",
                    ),
                )
            )

        discovery = discover_trellis2_conditioning_config(self.root)
        if not discovery.ready or discovery.config is None:
            return Trellis2TexturedGenerationResult(
                trace=Trellis2ForwardTraceResult(
                    root=self.root,
                    image_path=image,
                    completed_stages=(),
                    blocker=discovery.blocker,
                )
            )
        completed = ["asset-config-validation"]
        config = discovery.config
        selected_pipeline_type = pipeline_type or config.default_pipeline_type
        try:
            route = select_texture_slat_route(selected_pipeline_type)
        except ValueError as error:
            return Trellis2TexturedGenerationResult(
                trace=Trellis2ForwardTraceResult(
                    root=self.root,
                    image_path=image,
                    completed_stages=tuple(completed),
                    blocker=Trellis2ForwardBlocker(
                        stage="texture-slat-sampling",
                        operation="texture SLat pipeline route selection",
                        reference="--pipeline-type",
                        reason=str(error),
                        next_slice="choose one of 512, 1024, 1024_cascade, or 1536_cascade",
                    ),
                )
            )
        config = replace(config, default_pipeline_type=selected_pipeline_type)
        if slat_steps is not None:
            config = replace(
                config,
                shape_slat_sampler=replace(config.shape_slat_sampler, steps=slat_steps),
                texture_slat_sampler=replace(config.texture_slat_sampler, steps=slat_steps),
            )

        route_blocker = _validate_texture_route_assets(self.root, config, route.model_key)
        if route_blocker is not None:
            return Trellis2TexturedGenerationResult(
                trace=Trellis2ForwardTraceResult(
                    root=self.root,
                    image_path=image,
                    completed_stages=tuple(completed),
                    outputs=(
                        Trellis2StageOutput(
                            stage="texture-slat-sampling",
                            name="texture_route",
                            shape=(),
                            dtype="metadata",
                            detail=(
                                f"pipeline_type={selected_pipeline_type}; texture_model={route.model_key}; "
                                f"seed={seed}; slat_steps={slat_steps}; max_num_tokens={max_num_tokens}; "
                                f"decoder_token_limit={decoder_token_limit}; texture_size={texture_size}; "
                                f"glb_target_faces={glb_target_faces}; xatlas_face_guard={xatlas_face_guard}; "
                                f"xatlas_parallel_chunks={xatlas_parallel_chunks}; "
                                f"texture_bake_backend={texture_bake_backend}; dino_root={dino_root}"
                            ),
                        ),
                    ),
                    blocker=route_blocker,
                )
            )
        completed.append("checkpoint-probe-readiness")

        if not image.is_file():
            return Trellis2TexturedGenerationResult(
                trace=Trellis2ForwardTraceResult(
                    root=self.root,
                    image_path=image,
                    completed_stages=tuple(completed),
                    outputs=(
                        Trellis2StageOutput(
                            stage="texture-slat-sampling",
                            name="texture_route",
                            shape=(),
                            dtype="metadata",
                            detail=(
                                f"pipeline_type={selected_pipeline_type}; texture_model={route.model_key}; "
                                f"seed={seed}; slat_steps={slat_steps}; max_num_tokens={max_num_tokens}; "
                                f"decoder_token_limit={decoder_token_limit}; texture_size={texture_size}; "
                                f"glb_target_faces={glb_target_faces}; xatlas_face_guard={xatlas_face_guard}; "
                                f"xatlas_parallel_chunks={xatlas_parallel_chunks}; "
                                f"texture_bake_backend={texture_bake_backend}; dino_root={dino_root}"
                            ),
                        ),
                    ),
                    blocker=Trellis2ForwardBlocker(
                        stage="input-image",
                        operation="image file validation",
                        reference="vendors/trellis-mac/generate.py:68-70",
                        reason=f"image file not found: {image}",
                        next_slice="provide or generate a local sample image before textured generation",
                    ),
                )
            )

        mx.random.seed(int(seed))
        outputs = [
            Trellis2StageOutput(
                stage="texture-slat-sampling",
                name="texture_route",
                shape=(),
                dtype="metadata",
                detail=(
                    f"pipeline_type={selected_pipeline_type}; texture_model={route.model_key}; "
                    f"conditioning_resolution={route.output_resolution}; final_decode_resolution={_decode_resolution(selected_pipeline_type)}; "
                    f"seed={seed}; slat_steps={config.texture_slat_sampler.steps}; max_num_tokens={max_num_tokens}; "
                    f"decoder_token_limit={decoder_token_limit}; texture_size={texture_size}; "
                    f"glb_target_faces={glb_target_faces}; xatlas_face_guard={xatlas_face_guard}; "
                    f"xatlas_parallel_chunks={xatlas_parallel_chunks}; texture_bake_backend={texture_bake_backend}; "
                    f"dino_root={dino_root}; output={output}"
                ),
            )
        ]

        preprocessed = preprocess_trellis2_image(image, rmbg_root=self.rmbg_root)
        if not preprocessed.ready or preprocessed.image is None:
            return Trellis2TexturedGenerationResult(
                trace=Trellis2ForwardTraceResult(
                    root=self.root,
                    image_path=image,
                    completed_stages=tuple((*completed, "input-image")),
                    outputs=tuple(outputs),
                    blocker=_forward_blocker(_preprocess_blocker(preprocessed.blocker)),
                )
            )
        completed.extend(["input-image", "image-preprocessing-background"])

        cond_outputs: dict[int, Trellis2StageOutput] = {}
        for resolution in _conditioning_resolutions(selected_pipeline_type):
            cond_config = replace(config, conditioning_resolution=resolution)
            image_tensor = prepare_dinov3_image_tensor(preprocessed.image.image, image_size=resolution)
            cond_output, blocker = assess_dinov3_conditioning(
                self.root,
                cond_config,
                dino_root=dino_root,
                image_tensor=image_tensor,
            )
            if blocker is not None or cond_output is None:
                return Trellis2TexturedGenerationResult(
                    trace=Trellis2ForwardTraceResult(
                        root=self.root,
                        image_path=image,
                        completed_stages=tuple(completed),
                        outputs=tuple(outputs),
                        blocker=blocker,
                    )
                )
            cond_output = Trellis2StageOutput(
                stage=cond_output.stage,
                name=f"cond_{resolution}",
                shape=cond_output.shape,
                dtype=cond_output.dtype,
                detail=f"{cond_output.detail}; pipeline_type={selected_pipeline_type} conditioning_resolution={resolution}",
                payload=cond_output.payload,
            )
            cond_outputs[resolution] = cond_output
            outputs.append(cond_output if retain_trace_payloads else _without_stage_payload(cond_output))
        cond_512 = cond_outputs[512]
        cond_1024 = cond_outputs.get(1024)
        completed.append("image-conditioning")

        if cond_512.payload is None:
            return Trellis2TexturedGenerationResult(
                trace=Trellis2ForwardTraceResult(
                    root=self.root,
                    image_path=image,
                    completed_stages=tuple(completed),
                    outputs=tuple(outputs),
                    blocker=Trellis2ForwardBlocker(
                        stage="image-conditioning",
                        operation="DINOv3 cond_512 payload availability",
                        reference="vendors/TRELLIS.2/trellis2/pipelines/trellis2_image_to_3d.py:489-533",
                        reason="sparse structure sampling requires cond_512, but conditioning returned no payload",
                        next_slice="produce DINOv3 cond_512 before sparse structure sampling",
                    ),
                )
            )

        try:
            sparse_config = read_sparse_structure_flow_config(self.root, config.sparse_flow_config_path)
            metadata = fake_sparse_structure_sampling_metadata(
                sparse_config,
                steps=config.sparse_structure_sampler.steps,
                rescale_t=config.sparse_structure_sampler.rescale_t,
                guidance_interval=config.sparse_structure_sampler.guidance_interval,
            )
            sparse_probe = probe_sparse_structure_forward_boundary(
                Path(self.root) / config.sparse_flow_checkpoint_path,
                sparse_config,
                conditioning=cond_512.payload,
                steps=config.sparse_structure_sampler.steps,
                rescale_t=config.sparse_structure_sampler.rescale_t,
                guidance_strength=config.sparse_structure_sampler.guidance_strength,
                guidance_rescale=config.sparse_structure_sampler.guidance_rescale,
                guidance_interval=config.sparse_structure_sampler.guidance_interval,
                sigma_min=config.sparse_structure_sampler.sigma_min,
            )
        except (FileNotFoundError, ValueError) as error:
            return Trellis2TexturedGenerationResult(
                trace=Trellis2ForwardTraceResult(
                    root=self.root,
                    image_path=image,
                    completed_stages=tuple(completed),
                    outputs=tuple(outputs),
                    blocker=Trellis2ForwardBlocker(
                        stage="sparse-structure-sampling",
                        operation="sparse flow config/checkpoint forward probe",
                        reference=config.sparse_flow_checkpoint_path,
                        reason=str(error),
                        next_slice="map the sparse structure flow before textured generation",
                    ),
                )
            )
        if sparse_probe.sampled_latent is None or sparse_probe.sampled_latent_shape is None:
            return Trellis2TexturedGenerationResult(
                trace=Trellis2ForwardTraceResult(
                    root=self.root,
                    image_path=image,
                    completed_stages=tuple(completed),
                    outputs=tuple(outputs),
                    blocker=Trellis2ForwardBlocker(
                        stage="sparse-structure-sampling",
                        operation=sparse_probe.blocker_operation,
                        reference=sparse_probe.checkpoint_path,
                        reason=(
                            f"sparse flow config validates noise shape {metadata.noise_shape}, "
                            f"FlowEuler steps={metadata.steps}, loads tensors {sparse_probe.loaded_tensor_names}, "
                            f"and {sparse_probe.blocker_detail}"
                        ),
                        next_slice="produce sampled sparse latent before textured generation",
                    ),
                )
            )
        outputs.append(
            _stage_output(
                "sparse-structure-sampling",
                "sparse_latent",
                sparse_probe.sampled_latent,
                f"MLX sparse structure FlowEuler sampler output after {metadata.steps} steps",
                retain_payload=retain_trace_payloads,
            )
        )
        completed.append("sparse-structure-sampling")

        try:
            sparse_decoder_config = read_sparse_structure_decoder_config(self.root, config.sparse_decoder_config_path)
            sparse_decoder_probe = probe_sparse_structure_decoder_boundary(
                Path(self.root) / config.sparse_decoder_checkpoint_path,
                sparse_decoder_config,
                sparse_latent=sparse_probe.sampled_latent,
                target_resolution=sparse_structure_target_resolution(selected_pipeline_type),
            )
        except (FileNotFoundError, ValueError) as error:
            return Trellis2TexturedGenerationResult(
                trace=Trellis2ForwardTraceResult(
                    root=self.root,
                    image_path=image,
                    completed_stages=tuple(completed),
                    outputs=tuple(outputs),
                    blocker=Trellis2ForwardBlocker(
                        stage="sparse-structure-decoding",
                        operation="sparse structure decoder config/checkpoint probe",
                        reference=config.sparse_decoder_checkpoint_path,
                        reason=str(error),
                        next_slice="decode sparse structure coordinates before texture SLat sampling",
                    ),
                )
            )
        if sparse_decoder_probe.coordinates is None or sparse_decoder_probe.coordinates_shape is None:
            return Trellis2TexturedGenerationResult(
                trace=Trellis2ForwardTraceResult(
                    root=self.root,
                    image_path=image,
                    completed_stages=tuple(completed),
                    outputs=tuple(outputs),
                    blocker=Trellis2ForwardBlocker(
                        stage="sparse-structure-decoding",
                        operation=sparse_decoder_probe.blocker_operation,
                        reference=sparse_decoder_probe.checkpoint_path,
                        reason=sparse_decoder_probe.blocker_detail,
                        next_slice="produce sparse structure coordinates before texture SLat sampling",
                    ),
                )
            )

        try:
            shape_coordinates, shape_features, shape_detail, final_resolution = _sample_shape_slat_for_pipeline(
                self.root,
                config,
                sparse_decoder_probe.coordinates,
                cond_512.payload,
                cond_1024.payload if cond_1024 is not None else None,
                max_num_tokens=max_num_tokens or TRELLIS2_SHAPE_MAX_NUM_TOKENS,
                decoder_token_limit=decoder_token_limit or TRELLIS2_SHAPE_DECODER_TOKEN_LIMIT,
            )
            outputs.append(
                _stage_output(
                    "shape-slat-sampling",
                    "shape_slat",
                    shape_features,
                    shape_detail,
                    retain_payload=retain_trace_payloads,
                )
            )
        except (FileNotFoundError, ValueError) as error:
            return Trellis2TexturedGenerationResult(
                trace=Trellis2ForwardTraceResult(
                    root=self.root,
                    image_path=image,
                    completed_stages=tuple(completed),
                    outputs=tuple(outputs),
                    blocker=Trellis2ForwardBlocker(
                        stage="shape-slat-sampling",
                        operation="exact TRELLIS.2 shape SLat route execution",
                        reference=config.default_pipeline_type,
                        reason=str(error),
                        next_slice="complete exact non-approximate shape SLat sampling before texture SLat sampling",
                    ),
                )
            )
        completed.append("shape-slat-sampling")

        texture_conditioning = cond_512.payload if route.output_resolution == 512 else (cond_1024.payload if cond_1024 is not None else None)
        if texture_conditioning is None:
            return Trellis2TexturedGenerationResult(
                trace=Trellis2ForwardTraceResult(
                    root=self.root,
                    image_path=image,
                    completed_stages=tuple(completed),
                    outputs=tuple(outputs),
                    blocker=Trellis2ForwardBlocker(
                        stage="texture-slat-sampling",
                        operation="texture SLat conditioning payload availability",
                        reference="vendors/TRELLIS.2/trellis2/pipelines/trellis2_image_to_3d.py:521-532",
                        reason=f"pipeline_type={selected_pipeline_type} requires cond_{route.output_resolution} for texture SLat",
                        next_slice="produce the selected texture conditioning before texture SLat sampling",
                    ),
                )
            )
        try:
            texture_coordinates, texture_features, texture_detail = _sample_texture_slat_model(
                self.root,
                config,
                route.model_key,
                shape_coordinates,
                shape_features,
                texture_conditioning,
            )
        except (FileNotFoundError, ValueError) as error:
            return Trellis2TexturedGenerationResult(
                trace=Trellis2ForwardTraceResult(
                    root=self.root,
                    image_path=image,
                    completed_stages=tuple(completed),
                    outputs=tuple(outputs),
                    blocker=Trellis2ForwardBlocker(
                        stage="texture-slat-sampling",
                        operation="exact TRELLIS.2 texture SLat route execution",
                        reference=route.model_key,
                        reason=str(error),
                        next_slice="complete exact non-approximate texture SLat sampling before texture decoding",
                    ),
                )
            )
        outputs.append(
            _stage_output(
                "texture-slat-sampling",
                "texture_slat",
                texture_features,
                (
                    f"pipeline_type={selected_pipeline_type}; texture_model={route.model_key}; "
                    f"texture_tokens={int(texture_coordinates.shape[0])}; conditioning_resolution={route.output_resolution}; "
                    f"shape_tokens={int(shape_coordinates.shape[0])}; shape_feature_width={int(shape_features.shape[1])}; "
                    f"final_decode_resolution={final_resolution}; {texture_detail}"
                ),
                retain_payload=retain_trace_payloads,
            )
        )
        completed.append("texture-slat-sampling")

        try:
            shape_decoder_config = read_structured_latent_decoder_config(self.root, config.shape_decoder_config_path)
            shape_result = run_shape_decoder_to_fields(
                Path(self.root) / config.shape_decoder_checkpoint_path,
                shape_decoder_config,
                shape_coordinates,
                shape_features,
                decoder_token_limit=decoder_token_limit or TRELLIS2_SHAPE_DECODER_TOKEN_LIMIT,
            )
        except (FileNotFoundError, ValueError) as error:
            return Trellis2TexturedGenerationResult(
                trace=Trellis2ForwardTraceResult(
                    root=self.root,
                    image_path=image,
                    completed_stages=tuple(completed),
                    outputs=tuple(outputs),
                    blocker=Trellis2ForwardBlocker(
                        stage="shape-decoder",
                        operation="MLX shape decoder FlexiDualGrid field execution",
                        reference=config.shape_decoder_checkpoint_path,
                        reason=str(error),
                        next_slice="complete shape decoder fields before guided texture decoding",
                    ),
                )
            )
        outputs.append(
            _stage_output(
                "shape-decoder",
                "shape_flexidualgrid_fields",
                shape_result.fields,
                (
                    f"full shape decoder completed {shape_result.probe.completed_levels} levels; "
                    f"coordinates {shape_result.probe.decoder_output_coordinate_shape}; "
                    f"subdivisions {shape_result.probe.subdivision_shapes}; "
                    f"final_decode_resolution={final_resolution}"
                ),
                retain_payload=retain_trace_payloads,
            )
        )
        completed.append("shape-decoder")

        try:
            texture_decoder_config = read_structured_latent_decoder_config(self.root, config.texture_decoder_config_path)
            texture_result = run_texture_decoder_to_representation(
                Path(self.root) / config.texture_decoder_checkpoint_path,
                texture_decoder_config,
                texture_coordinates,
                texture_features,
                guide_subdivisions=shape_result.subdivisions,
                decoder_token_limit=decoder_token_limit or TRELLIS2_SHAPE_DECODER_TOKEN_LIMIT,
                decode_resolution=final_resolution,
                shape_decoder_coordinates=shape_result.coordinates,
            )
        except (FileNotFoundError, ValueError) as error:
            return Trellis2TexturedGenerationResult(
                trace=Trellis2ForwardTraceResult(
                    root=self.root,
                    image_path=image,
                    completed_stages=tuple(completed),
                    outputs=tuple(outputs),
                    blocker=Trellis2ForwardBlocker(
                        stage="texture-decoder",
                        operation="MLX texture decoder guided SparseUnetVaeDecoder execution",
                        reference=(
                            f"{config.shape_decoder_checkpoint_path}; "
                            f"{config.texture_decoder_checkpoint_path}"
                        ),
                        reason=str(error),
                        next_slice="complete guided texture decoder output before mesh/voxel baking",
                    ),
                )
            )
        outputs.append(
            _stage_output(
                "texture-decoder",
                "texture_voxel_coordinates",
                texture_result.coordinates,
                (
                    f"texture decoder completed {texture_result.probe.completed_levels} levels; "
                    f"decoder_output_coordinates={texture_result.probe.decoder_output_coordinate_shape}; "
                    f"decoder_output_shape={texture_result.probe.decoder_output_shape}; "
                    f"guide_subdivisions={texture_result.guide_subdivision_shapes}; "
                    f"spatial_shape={texture_result.spatial_shape}; batch_size={texture_result.batch_size}; "
                    f"voxel_size={texture_result.voxel_size}"
                ),
                retain_payload=retain_trace_payloads,
            )
        )
        outputs.append(
            _stage_output(
                "texture-decoder",
                "texture_voxel_attrs",
                texture_result.attributes,
                (
                    "6-channel decoded texture attributes postprocessed with raw * 0.5 + 0.5; "
                    f"shape_decoder_coordinates={texture_result.shape_decoder_coordinate_shape}; "
                    f"owned_subdivisions={texture_result.probe.subdivision_shapes}"
                ),
                retain_payload=retain_trace_payloads,
            )
        )
        completed.append("texture-decoder")
        if not retain_trace_payloads:
            clear_mlx_cache()

        try:
            mesh = flexi_dual_grid_fields_to_mesh(shape_result.coordinates, shape_result.fields, grid_size=final_resolution)
            postprocess_result = postprocess_trellis2_mesh_for_glb(mesh, target_faces=glb_target_faces)
            baked_texture = bake_trellis2_texture_fields_mac_native(
                postprocess_result.mesh,
                texture_result.coordinates,
                texture_result.attributes,
                decode_resolution=final_resolution,
                texture_size=texture_size,
                xatlas_face_guard=xatlas_face_guard,
                xatlas_parallel_chunks=xatlas_parallel_chunks,
                texture_bake_backend=texture_bake_backend,
                projection_source_mesh=getattr(postprocess_result, "source_mesh", None),
            )
        except (ImportError, ValueError) as error:
            return Trellis2TexturedGenerationResult(
                trace=Trellis2ForwardTraceResult(
                    root=self.root,
                    image_path=image,
                    completed_stages=tuple(completed),
                    outputs=tuple(outputs),
                    blocker=Trellis2ForwardBlocker(
                        stage="mesh-export",
                        operation="mesh/voxel texture baking",
                        reference="vendors/TRELLIS.2/o-voxel/o_voxel/postprocess.py:232-323",
                        reason=str(error),
                        next_slice="Slice 5: Mesh/Voxel Coupling And Baking Fixtures",
                    ),
                )
            )
        outputs.append(
            _stage_output(
                "mesh-export",
                "texture_mesh_postprocess",
                mx.array(
                    [
                        postprocess_result.stats.original_vertices,
                        postprocess_result.stats.original_faces,
                        postprocess_result.stats.cleaned_vertices,
                        postprocess_result.stats.cleaned_faces,
                        postprocess_result.stats.final_vertices,
                        postprocess_result.stats.final_faces,
                        postprocess_result.stats.duplicate_faces_removed,
                        postprocess_result.stats.degenerate_faces_removed,
                        postprocess_result.stats.unreferenced_vertices_removed,
                        postprocess_result.stats.components_removed,
                        postprocess_result.stats.component_faces_removed,
                        postprocess_result.stats.hole_fill.filled_loops,
                        postprocess_result.stats.hole_fill.faces_added,
                        int(postprocess_result.stats.simplified),
                        postprocess_result.stats.simplification_target_faces,
                        postprocess_result.stats.boundary_edges,
                        postprocess_result.stats.nonmanifold_edges,
                    ],
                    dtype=mx.int32,
                ),
                (
                    "Mac-native GLB mesh cleanup before texture baking; "
                    f"faces {postprocess_result.stats.original_faces}->{postprocess_result.stats.final_faces}; "
                    f"vertices {postprocess_result.stats.original_vertices}->{postprocess_result.stats.final_vertices}; "
                    f"simplified={postprocess_result.stats.simplified}; "
                    f"duplicate_faces_removed={postprocess_result.stats.duplicate_faces_removed}; "
                    f"degenerate_faces_removed={postprocess_result.stats.degenerate_faces_removed}; "
                    f"components_removed={postprocess_result.stats.components_removed}; "
                    f"hole_filled_loops={postprocess_result.stats.hole_fill.filled_loops}; "
                    f"cleaned_source_faces={getattr(getattr(postprocess_result, 'source_mesh', None), 'faces', ()).shape[0] if getattr(postprocess_result, 'source_mesh', None) is not None else postprocess_result.stats.cleaned_faces}; "
                    f"boundary_edges={postprocess_result.stats.boundary_edges}; "
                    f"nonmanifold_edges={postprocess_result.stats.nonmanifold_edges}"
                ),
                retain_payload=retain_trace_payloads,
            )
        )
        outputs.append(
            _stage_output(
                "mesh-export",
                "texture_bake_uvs",
                mx.array(baked_texture.uvs),
                (
                    f"{baked_texture.backend} produced {baked_texture.uvs.shape[0]} UVs for "
                    f"{baked_texture.faces.shape[0]} faces; coverage={baked_texture.coverage_ratio:.4f}; "
                    f"raw_coverage={baked_texture.raw_coverage_ratio:.4f}; "
                    f"texture_size={baked_texture.texture_size}; voxel_count={baked_texture.voxel_count}; "
                    f"k_neighbors={baked_texture.k_neighbors}; unwrap_backend={baked_texture.unwrap_backend}; "
                    f"unwrap_chunks={baked_texture.unwrap_chunks}; "
                    f"xatlas_face_guard={getattr(baked_texture, 'xatlas_face_guard', xatlas_face_guard)}; "
                    f"xatlas_face_guard_mode={getattr(baked_texture, 'xatlas_face_guard_mode', 'manual')}; "
                    f"unwrap_seconds={baked_texture.unwrap_seconds:.3f}; "
                    f"unwrap_chart_count={baked_texture.unwrap_chart_count}; "
                    f"unwrap_utilization={baked_texture.unwrap_utilization}; "
                    f"sampled_texels={getattr(baked_texture, 'sampled_texel_count', 0)}; "
                    f"missing_texels={getattr(baked_texture, 'missing_texel_count', 0)}; "
                    f"out_of_grid_texels={getattr(baked_texture, 'out_of_grid_texel_count', 0)}; "
                    f"source_projection_used={getattr(baked_texture, 'source_projection_used', False)}; "
                    f"source_projection_detail={getattr(baked_texture, 'source_projection_detail', 'unknown')}"
                ),
                retain_payload=retain_trace_payloads,
            )
        )
        outputs.append(
            _stage_output(
                "mesh-export",
                "texture_bake_base_color_rgba",
                mx.array(baked_texture.base_color_rgba),
                (
                    "baked baseColorTexture payload from 6-channel texture voxels; "
                    f"shape={baked_texture.base_color_rgba.shape}; coverage={baked_texture.coverage_ratio:.4f}"
                ),
                retain_payload=retain_trace_payloads,
            )
        )
        outputs.append(
            _stage_output(
                "mesh-export",
                "texture_bake_metallic_roughness",
                mx.array(baked_texture.metallic_roughness),
                (
                    "baked metallicRoughnessTexture payload with G=roughness and B=metallic; "
                    f"shape={baked_texture.metallic_roughness.shape}"
                ),
                retain_payload=retain_trace_payloads,
            )
        )

        try:
            artifact = write_trellis2_textured_glb(baked_texture, output_path)
            if not retain_trace_payloads:
                del baked_texture, postprocess_result, mesh, shape_result, texture_result
                clear_mlx_cache()
        except (OSError, ValueError) as error:
            return Trellis2TexturedGenerationResult(
                trace=Trellis2ForwardTraceResult(
                    root=self.root,
                    image_path=image,
                    completed_stages=tuple(completed),
                    outputs=tuple(outputs),
                    blocker=Trellis2ForwardBlocker(
                        stage="mesh-export",
                        operation="textured GLB writer",
                        reference="src/mlx_spatial/trellis2_export.py:write_trellis2_textured_glb",
                        reason=str(error),
                        next_slice="complete GLB artifact writing for textured generation",
                    ),
                )
            )
        outputs.append(
            Trellis2StageOutput(
                stage="mesh-export",
                name="textured_glb",
                shape=(),
                dtype="metadata",
                detail=(
                    f"wrote textured GLB artifact {artifact.path}; bytes={artifact.bytes_written}; "
                    f"format={artifact.format}"
                ),
            )
        )
        completed.append("mesh-export")
        return Trellis2TexturedGenerationResult(
            trace=Trellis2ForwardTraceResult(
                root=self.root,
                image_path=image,
                completed_stages=tuple(completed),
                outputs=tuple(outputs),
            ),
            artifact=artifact,
        )


def _conditioning_resolutions(pipeline_type: str) -> tuple[int, ...]:
    if pipeline_type == "512":
        return (512,)
    if pipeline_type in {"1024", "1024_cascade", "1536_cascade"}:
        return (512, 1024)
    raise ValueError(f"unsupported TRELLIS.2 pipeline type: {pipeline_type}")


def _sample_shape_slat_for_pipeline(
    root: Path,
    config,
    sparse_coordinates: mx.array,
    cond_512: mx.array,
    cond_1024: mx.array | None,
    *,
    max_num_tokens: int,
    decoder_token_limit: int,
) -> tuple[mx.array, mx.array, str, int]:
    route = select_shape_slat_route(config.default_pipeline_type)
    if not route.cascade:
        condition = cond_512 if route.pipeline_type == "512" else cond_1024
        if condition is None:
            raise ValueError(f"pipeline_type={route.pipeline_type} requires cond_1024")
        coordinates, features, probe_detail = _sample_shape_slat_model(
            root,
            config,
            route.model_keys[0],
            sparse_coordinates,
            condition,
        )
        detail = (
            f"pipeline_type={route.pipeline_type}; sparse_resolution={sparse_structure_target_resolution(route.pipeline_type)}; "
            f"shape_model={route.model_keys[0]}; shape_tokens={int(coordinates.shape[0])}; "
            f"decode_resolution={route.output_resolution}; {probe_detail}"
        )
        return coordinates, features, detail, route.output_resolution

    if cond_1024 is None:
        raise ValueError(f"pipeline_type={route.pipeline_type} requires cond_1024 for high-resolution cascade")
    lr_coordinates, lr_features, lr_detail = _sample_shape_slat_model(
        root,
        config,
        route.model_keys[0],
        sparse_coordinates,
        cond_512,
    )
    shape_decoder_config = read_structured_latent_decoder_config(root, config.shape_decoder_config_path)
    upsample = run_shape_decoder_upsample_coordinates(
        root / config.shape_decoder_checkpoint_path,
        shape_decoder_config,
        lr_coordinates,
        lr_features,
        upsample_times=4,
        decoder_token_limit=decoder_token_limit,
    )
    hr_coordinates, final_resolution = _quantize_cascade_coordinates(
        upsample.coordinates,
        lr_resolution=512,
        target_resolution=route.output_resolution,
        max_num_tokens=max_num_tokens,
    )
    _, hr_features, hr_detail = _sample_shape_slat_model(
        root,
        config,
        route.model_keys[1],
        hr_coordinates,
        cond_1024,
    )
    detail = (
        f"pipeline_type={route.pipeline_type}; sparse_resolution={sparse_structure_target_resolution(route.pipeline_type)}; "
        f"lr_model={route.model_keys[0]}; lr_tokens={int(lr_coordinates.shape[0])}; "
        f"upsample_coordinates={upsample.output_coordinate_shape}; max_num_tokens={max_num_tokens}; "
        f"hr_model={route.model_keys[1]}; hr_tokens={int(hr_coordinates.shape[0])}; "
        f"decode_resolution={final_resolution}; lr=({lr_detail}); hr=({hr_detail})"
    )
    return hr_coordinates, hr_features, detail, final_resolution


def _sample_shape_slat_model(
    root: Path,
    config,
    model_key: str,
    sparse_coordinates: mx.array,
    conditioning: mx.array,
) -> tuple[mx.array, mx.array, str]:
    config_path, checkpoint_path = _shape_slat_model_paths(config, model_key)
    slat_config = read_slat_flow_config(root, config_path)
    probe = probe_shape_slat_forward_boundary(
        root / checkpoint_path,
        slat_config,
        sparse_coordinates,
        conditioning=conditioning,
        steps=config.shape_slat_sampler.steps,
        rescale_t=config.shape_slat_sampler.rescale_t,
        guidance_strength=config.shape_slat_sampler.guidance_strength,
        guidance_rescale=config.shape_slat_sampler.guidance_rescale,
        guidance_interval=config.shape_slat_sampler.guidance_interval,
        sigma_min=config.shape_slat_sampler.sigma_min,
    )
    if probe.sampled_features is None or probe.sampled_feature_shape is None:
        raise ValueError(f"{model_key} did not produce sampled shape_slat features: {probe.blocker_detail}")
    sampled = _apply_slat_normalization(probe.sampled_features, config.shape_slat_normalization, name="shape_slat")
    return sparse_coordinates, sampled, probe.blocker_detail


def _sample_texture_slat_model(
    root: Path,
    config,
    model_key: str,
    shape_coordinates: mx.array,
    shape_features: mx.array,
    conditioning: mx.array,
) -> tuple[mx.array, mx.array, str]:
    config_path, checkpoint_path = _texture_slat_model_paths(config, model_key)
    slat_config = read_slat_flow_config(root, config_path)
    normalized_shape_features = _undo_slat_normalization(
        shape_features,
        config.shape_slat_normalization,
        name="shape_slat",
    )
    probe = probe_texture_slat_forward_boundary(
        root / checkpoint_path,
        slat_config,
        shape_coordinates,
        normalized_shape_features,
        conditioning=conditioning,
        steps=config.texture_slat_sampler.steps,
        rescale_t=config.texture_slat_sampler.rescale_t,
        guidance_strength=config.texture_slat_sampler.guidance_strength,
        guidance_rescale=config.texture_slat_sampler.guidance_rescale,
        guidance_interval=config.texture_slat_sampler.guidance_interval,
        sigma_min=config.texture_slat_sampler.sigma_min,
    )
    if probe.sampled_features is None or probe.sampled_feature_shape is None:
        raise ValueError(f"{model_key} did not produce sampled texture_slat features: {probe.blocker_detail}")
    sampled = _apply_slat_normalization(probe.sampled_features, config.texture_slat_normalization, name="texture_slat")
    detail = (
        f"shape_slat coordinate shape {probe.coordinate_shape}; "
        f"normalized shape feature shape {probe.shape_feature_shape}; "
        f"texture noise feature shape {probe.noise_feature_shape}; "
        f"concat feature shape {probe.concat_feature_shape}; "
        f"{probe.blocker_detail}"
    )
    return shape_coordinates, sampled, detail


def _shape_slat_model_paths(config, model_key: str) -> tuple[str, str]:
    if model_key == "shape_slat_flow_model_512":
        return config.shape_slat_512_config_path, config.shape_slat_512_checkpoint_path
    if model_key == "shape_slat_flow_model_1024":
        return config.shape_slat_1024_config_path, config.shape_slat_1024_checkpoint_path
    raise ValueError(f"unsupported shape SLat model key: {model_key}")


def _texture_slat_model_paths(config, model_key: str) -> tuple[str, str]:
    if model_key == "tex_slat_flow_model_512":
        return config.texture_slat_512_config_path, config.texture_slat_512_checkpoint_path
    if model_key == "tex_slat_flow_model_1024":
        return config.texture_slat_1024_config_path, config.texture_slat_1024_checkpoint_path
    raise ValueError(f"unsupported texture SLat model key: {model_key}")


def _undo_slat_normalization(features: mx.array, normalization, *, name: str) -> mx.array:
    feature_width = int(features.shape[-1])
    if len(normalization.mean) != feature_width or len(normalization.std) != feature_width:
        raise ValueError(
            f"{name} normalization width mismatch: expected {feature_width}, "
            f"got mean={len(normalization.mean)} std={len(normalization.std)}"
        )
    mean = mx.array(normalization.mean, dtype=mx.float32)[None, :]
    std = mx.array(normalization.std, dtype=mx.float32)[None, :]
    return (features.astype(mx.float32) - mean) / std


def _validate_texture_route_assets(root: Path, config, model_key: str) -> Trellis2ForwardBlocker | None:
    if model_key == "tex_slat_flow_model_512":
        slat_config_path = config.texture_slat_512_config_path
        slat_checkpoint_path = config.texture_slat_512_checkpoint_path
    elif model_key == "tex_slat_flow_model_1024":
        slat_config_path = config.texture_slat_1024_config_path
        slat_checkpoint_path = config.texture_slat_1024_checkpoint_path
    else:
        return Trellis2ForwardBlocker(
            stage="texture-slat-sampling",
            operation="texture SLat model path selection",
            reference=model_key,
            reason=f"unsupported texture SLat model key: {model_key}",
            next_slice="select a supported texture SLat model key before texture execution",
        )

    try:
        read_slat_flow_config(root, slat_config_path)
    except (FileNotFoundError, ValueError) as error:
        return Trellis2ForwardBlocker(
            stage="texture-slat-sampling",
            operation="texture SLat config validation",
            reference=slat_config_path,
            reason=str(error),
            next_slice="place and validate the selected texture SLat config before texture execution",
        )
    if not (root / slat_checkpoint_path).is_file():
        return Trellis2ForwardBlocker(
            stage="texture-slat-sampling",
            operation="texture SLat checkpoint validation",
            reference=slat_checkpoint_path,
            reason=f"texture SLat checkpoint file not found: {root / slat_checkpoint_path}",
            next_slice="place the selected texture SLat checkpoint before texture execution",
        )

    try:
        read_structured_latent_decoder_config(root, config.texture_decoder_config_path)
    except (FileNotFoundError, ValueError) as error:
        return Trellis2ForwardBlocker(
            stage="texture-decoder",
            operation="texture decoder config validation",
            reference=config.texture_decoder_config_path,
            reason=str(error),
            next_slice="place and validate the texture decoder config before texture decoding",
        )
    if not (root / config.texture_decoder_checkpoint_path).is_file():
        return Trellis2ForwardBlocker(
            stage="texture-decoder",
            operation="texture decoder checkpoint validation",
            reference=config.texture_decoder_checkpoint_path,
            reason=f"texture decoder checkpoint file not found: {root / config.texture_decoder_checkpoint_path}",
            next_slice="place the texture decoder checkpoint before texture decoding",
        )
    return None


def _quantize_cascade_coordinates(
    hr_coordinates: mx.array,
    *,
    lr_resolution: int,
    target_resolution: int,
    max_num_tokens: int,
) -> tuple[mx.array, int]:
    if max_num_tokens <= 0:
        raise ValueError("max_num_tokens must be positive")
    if lr_resolution <= 0 or target_resolution <= 0:
        raise ValueError("cascade resolutions must be positive")
    coords = np.array(hr_coordinates, dtype=np.int32)
    if coords.ndim != 2 or coords.shape[1] != 4:
        raise ValueError(f"cascade coordinates must have shape (num_tokens, 4), got {coords.shape}")
    final_resolution = int(target_resolution)
    while True:
        spatial_resolution = final_resolution // 16
        quantized = np.concatenate(
            (
                coords[:, :1],
                (((coords[:, 1:] + 0.5) / float(lr_resolution)) * spatial_resolution).astype(np.int32),
            ),
            axis=1,
        )
        quantized = np.unique(quantized, axis=0).astype(np.int32)
        if quantized.shape[0] < max_num_tokens or final_resolution == 1024:
            return mx.array(quantized, dtype=mx.int32), final_resolution
        final_resolution -= 128


def _missing_assets_blocker(missing: tuple[str, ...]) -> Trellis2InferenceBlocker:
    return Trellis2InferenceBlocker(
        stage="asset-config-validation",
        operation="TRELLIS.2 asset validation",
        reference="src/mlx_spatial/model_assets.py:TRELLIS2_ASSETS",
        reason=f"missing local assets: {list(missing)}",
        next_slice="download or place TRELLIS.2 assets under weights/trellis2",
    )


def _preprocess_blocker(blocker: Trellis2PreprocessBlocker | None) -> Trellis2InferenceBlocker:
    if blocker is None:
        return _unimplemented_blocker("image-preprocessing-background")
    return Trellis2InferenceBlocker(
        stage=blocker.stage,
        operation=blocker.operation,
        reference=blocker.reference,
        reason=blocker.reason,
        next_slice=blocker.next_slice,
    )


def _forward_blocker(blocker: Trellis2InferenceBlocker | None) -> Trellis2ForwardBlocker:
    if blocker is None:
        blocker = _unimplemented_blocker("image-conditioning")
    return Trellis2ForwardBlocker(
        stage=blocker.stage,
        operation=blocker.operation,
        reference=blocker.reference,
        reason=blocker.reason,
        next_slice=blocker.next_slice,
    )


def _stage_output(
    stage: str,
    name: str,
    payload: mx.array,
    detail: str,
    *,
    retain_payload: bool = True,
) -> "Trellis2StageOutput":
    from .trellis2_forward import Trellis2StageOutput

    return Trellis2StageOutput(
        stage=stage,
        name=name,
        shape=tuple(int(dim) for dim in payload.shape),
        dtype=str(payload.dtype).removeprefix("mlx.core."),
        detail=detail,
        payload=payload if retain_payload else None,
    )


def _without_stage_payload(output: "Trellis2StageOutput") -> "Trellis2StageOutput":
    from .trellis2_forward import Trellis2StageOutput

    return Trellis2StageOutput(
        stage=output.stage,
        name=output.name,
        shape=output.shape,
        dtype=output.dtype,
        detail=output.detail,
        payload=None,
    )


def _decode_resolution(pipeline_type: str) -> int:
    if pipeline_type == "512":
        return 512
    if pipeline_type in {"1024", "1024_cascade"}:
        return 1024
    if pipeline_type == "1536_cascade":
        return 1536
    raise ValueError(f"unsupported decode pipeline type: {pipeline_type}")


def _unimplemented_blocker(stage: str) -> Trellis2InferenceBlocker:
    references = {
        "image-preprocessing-background": "vendors/TRELLIS.2/trellis2/pipelines/trellis2_image_to_3d.py:127-162",
        "image-conditioning": "vendors/TRELLIS.2/trellis2/pipelines/trellis2_image_to_3d.py:164-186",
        "sparse-structure-sampling": "vendors/TRELLIS.2/trellis2/pipelines/trellis2_image_to_3d.py:188-235",
        "shape-slat-sampling": "vendors/TRELLIS.2/trellis2/pipelines/trellis2_image_to_3d.py:237-364",
        "texture-slat-sampling": "vendors/TRELLIS.2/trellis2/pipelines/trellis2_image_to_3d.py:391-432",
        "shape-decoder": "vendors/TRELLIS.2/trellis2/pipelines/trellis2_image_to_3d.py:366-389",
        "texture-decoder": "vendors/TRELLIS.2/trellis2/pipelines/trellis2_image_to_3d.py:434-453",
        "mesh-export": "vendors/trellis-mac/generate.py:146-299",
    }
    operations = {
        "image-preprocessing-background": "MLX/Python image preprocessing and background removal boundary",
        "image-conditioning": "MLX image feature extraction / conditioning",
        "sparse-structure-sampling": "MLX sparse structure flow sampling and decoding",
        "shape-slat-sampling": "MLX shape SLat flow sampling",
        "texture-slat-sampling": "MLX texture SLat flow sampling",
        "shape-decoder": "MLX shape decoder",
        "texture-decoder": "MLX texture decoder",
        "mesh-export": "MLX mesh extraction and GLB/OBJ export",
    }
    return Trellis2InferenceBlocker(
        stage=stage,
        operation=operations.get(stage, "MLX compute stage"),
        reference=references.get(stage, ".agent/work/trellis-e2e-inference-attempt/FLOW.md"),
        reason="stage is traced but not implemented in MLX",
        next_slice=f"implement {stage} for TRELLIS.2 inference",
    )
