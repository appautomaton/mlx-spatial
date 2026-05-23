"""Standalone texturing-only pipeline for TRELLIS.2 mesh texturing.

Takes an image and an existing mesh, runs the FDG encoder for shape SLat,
samples texture SLat conditioned on DINOv3 + shape SLat, decodes PBR voxels,
then bakes and exports a textured GLB file.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import mlx.core as mx
import numpy as np

from .mesh_to_fdg import mesh_to_flexible_dual_grid
from .mlx_memory import clear_mlx_cache
from .ovoxel import FlexibleDualGridMesh, flexi_dual_grid_fields_to_mesh
from .trellis2_decode import (
    read_structured_latent_decoder_config,
    read_structured_latent_encoder_config,
    run_flexi_dual_grid_vae_encoder,
    run_shape_decoder_to_fields,
    run_texture_decoder_to_representation,
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
    write_trellis2_textured_glb,
)
from .trellis2_forward import (
    Trellis2ForwardBlocker,
    Trellis2ForwardTraceResult,
    Trellis2StageOutput,
    assess_dinov3_conditioning,
    discover_trellis2_conditioning_config,
    prepare_dinov3_image_tensor,
)
from .trellis2_preprocess import Trellis2PreprocessBlocker, preprocess_trellis2_image
from .trellis2_slat import (
    probe_texture_slat_forward_boundary,
    read_slat_flow_config,
    select_texture_slat_route,
)

TRELLIS2_TEXTURING_DEFAULT_SEED = 42
TRELLIS2_TEXTURING_DEFAULT_TEXTURE_SIZE = 1024
TRELLIS2_TEXTURING_DEFAULT_DECODER_TOKEN_LIMIT = 12_000_000

_SHAPE_ENCODER_CONFIG_CONVENTION = "shape_encoder.json"


@dataclass(frozen=True)
class Trellis2TexturingBlocker:
    stage: str
    operation: str
    reference: str
    reason: str
    next_slice: str


@dataclass(frozen=True)
class Trellis2TexturingResult:
    """Result from a TRELLIS.2 texturing-only pipeline run."""

    image_path: Path
    mesh_path: Path
    artifact: Trellis2ExportArtifact | None = None
    blocker: Trellis2TexturingBlocker | None = None
    trace: Trellis2ForwardTraceResult | None = None

    @property
    def ready(self) -> bool:
        return self.artifact is not None and self.blocker is None


class Trellis2TexturingPipeline:
    """Standalone texturing-only pipeline that textures an existing mesh.

    Reuses the FlexiDualGrid VAE encoder for shape SLat extraction,
    DINOv3 + shape SLat conditioning for texture SLat FlowEuler sampling,
    SparseUnetVaeDecoder for PBR voxel decoding, and Mac-native UV baking
    with xatlas for textured GLB export.
    """

    def __init__(
        self,
        root: str | Path = "weights/trellis2",
        *,
        rmbg_root: str | Path | None = None,
        dino_root: str | Path | None = None,
        encoder_config_path: str | None = None,
        encoder_checkpoint_path: str | None = None,
    ):
        self.root = Path(root)
        self.rmbg_root = Path(rmbg_root) if rmbg_root is not None else None
        self.dino_root = Path(dino_root) if dino_root is not None else None
        self.encoder_config_path = encoder_config_path
        self.encoder_checkpoint_path = encoder_checkpoint_path

    def run(
        self,
        image_path: str | Path,
        mesh_path: str | Path,
        *,
        output_path: str | Path,
        pipeline_type: str | None = None,
        seed: int = TRELLIS2_TEXTURING_DEFAULT_SEED,
        grid_size: int = 64,
        slat_steps: int | None = None,
        decoder_token_limit: int = TRELLIS2_TEXTURING_DEFAULT_DECODER_TOKEN_LIMIT,
        texture_size: int = TRELLIS2_TEXTURING_DEFAULT_TEXTURE_SIZE,
        glb_target_faces: int = TRELLIS2_GLB_DEFAULT_FACE_TARGET,
        xatlas_face_guard: int | str = TRELLIS2_XATLAS_AUTO_FACE_GUARD,
        xatlas_parallel_chunks: int = 0,
        texture_bake_backend: str = "trilinear",
    ) -> Trellis2TexturingResult:
        image = Path(image_path)
        mesh_file = Path(mesh_path)
        output = Path(output_path)

        if output.suffix.lower() != ".glb":
            return Trellis2TexturingResult(
                image_path=image,
                mesh_path=mesh_file,
                blocker=Trellis2TexturingBlocker(
                    stage="mesh-export",
                    operation="textured GLB output format validation",
                    reference=str(output_path),
                    reason=f"texturing pipeline only writes .glb outputs, got {output.suffix or '<none>'}",
                    next_slice="choose a .glb output path under outputs/ for texturing",
                ),
            )

        if not image.is_file():
            return Trellis2TexturingResult(
                image_path=image,
                mesh_path=mesh_file,
                blocker=Trellis2TexturingBlocker(
                    stage="input-image",
                    operation="image file validation",
                    reference=str(image_path),
                    reason=f"image file not found: {image}",
                    next_slice="provide a valid image file",
                ),
            )

        if not mesh_file.is_file():
            return Trellis2TexturingResult(
                image_path=image,
                mesh_path=mesh_file,
                blocker=Trellis2TexturingBlocker(
                    stage="mesh-load",
                    operation="mesh file validation",
                    reference=str(mesh_path),
                    reason=f"mesh file not found: {mesh_file}",
                    next_slice="provide a valid OBJ mesh file",
                ),
            )

        if grid_size <= 0:
            return Trellis2TexturingResult(
                image_path=image,
                mesh_path=mesh_file,
                blocker=Trellis2TexturingBlocker(
                    stage="mesh-preprocess",
                    operation="grid size validation",
                    reference="--grid-size",
                    reason=f"grid_size must be positive, got {grid_size}",
                    next_slice="use a positive grid size",
                ),
            )

        try:
            validate_trellis2_export_path(output_path, suffixes=(".glb",))
        except (OSError, ValueError) as error:
            return Trellis2TexturingResult(
                image_path=image,
                mesh_path=mesh_file,
                blocker=Trellis2TexturingBlocker(
                    stage="mesh-export",
                    operation="export path validation",
                    reference=str(output_path),
                    reason=str(error),
                    next_slice="choose a .glb output path under outputs/",
                ),
            )

        missing_deps = missing_trellis2_mac_export_dependencies()
        if missing_deps:
            return Trellis2TexturingResult(
                image_path=image,
                mesh_path=mesh_file,
                blocker=Trellis2TexturingBlocker(
                    stage="mesh-export",
                    operation="Mac-native GLB export dependency validation",
                    reference="xatlas, scipy, fast_simplification",
                    reason=f"missing {', '.join(missing_deps)}",
                    next_slice="install xatlas, scipy, and fast-simplification",
                ),
            )

        preprocessed = preprocess_trellis2_image(image, rmbg_root=self.rmbg_root)
        if not preprocessed.ready or preprocessed.image is None:
            return Trellis2TexturingResult(
                image_path=image,
                mesh_path=mesh_file,
                blocker=_preprocess_texturing_blocker(preprocessed.blocker),
            )

        try:
            mesh_vertices, mesh_faces = _load_obj_mesh(mesh_file)
        except (OSError, ValueError) as error:
            return Trellis2TexturingResult(
                image_path=image,
                mesh_path=mesh_file,
                blocker=Trellis2TexturingBlocker(
                    stage="mesh-load",
                    operation="OBJ mesh parsing",
                    reference=str(mesh_path),
                    reason=str(error),
                    next_slice="provide a valid OBJ triangle mesh",
                ),
            )

        fdg_coords, fdg_dual, fdg_intersected = mesh_to_flexible_dual_grid(
            mesh_vertices, mesh_faces, grid_size=grid_size
        )

        if fdg_coords.shape[0] == 0:
            return Trellis2TexturingResult(
                image_path=image,
                mesh_path=mesh_file,
                blocker=Trellis2TexturingBlocker(
                    stage="mesh-preprocess",
                    operation="FlexiDualGrid voxelization",
                    reference=str(mesh_path),
                    reason="mesh_to_flexible_dual_grid produced no occupied voxels",
                    next_slice="increase grid_size or provide a mesh within the AABB",
                ),
            )

        encoder_coords = np.column_stack(
            [np.zeros(fdg_coords.shape[0], dtype=np.int32), fdg_coords]
        )
        encoder_coords_mx = mx.array(encoder_coords, dtype=mx.int32)
        dual_mx = mx.array(fdg_dual, dtype=mx.float32)
        intersected_mx = mx.array(fdg_intersected.astype(np.float32), dtype=mx.float32)

        discovery = discover_trellis2_conditioning_config(self.root)
        if not discovery.ready or discovery.config is None:
            return Trellis2TexturingResult(
                image_path=image,
                mesh_path=mesh_file,
                blocker=Trellis2TexturingBlocker(
                    stage="asset-config",
                    operation="TRELLIS.2 conditioning config discovery",
                    reference=str(self.root / "pipeline.json"),
                    reason=(
                        discovery.blocker.reason
                        if discovery.blocker
                        else "pipeline config not found"
                    ),
                    next_slice="place valid TRELLIS.2 pipeline.json under weights/trellis2",
                ),
            )
        config = discovery.config
        selected_pipeline_type = pipeline_type or config.default_pipeline_type

        try:
            route = select_texture_slat_route(selected_pipeline_type)
        except ValueError as error:
            return Trellis2TexturingResult(
                image_path=image,
                mesh_path=mesh_file,
                blocker=Trellis2TexturingBlocker(
                    stage="texture-slat",
                    operation="texture SLat route selection",
                    reference=str(selected_pipeline_type),
                    reason=str(error),
                    next_slice="choose one of 512, 1024, 1024_cascade, or 1536_cascade",
                ),
            )
        config = replace(config, default_pipeline_type=selected_pipeline_type)
        if slat_steps is not None:
            if slat_steps <= 0:
                return Trellis2TexturingResult(
                    image_path=image,
                    mesh_path=mesh_file,
                    blocker=Trellis2TexturingBlocker(
                        stage="texture-slat",
                        operation="SLat step override validation",
                        reference="--slat-steps",
                        reason=f"slat_steps must be positive, got {slat_steps}",
                        next_slice="use a positive slat_steps value",
                    ),
                )
            config = replace(
                config,
                texture_slat_sampler=replace(config.texture_slat_sampler, steps=slat_steps),
            )

        resolved_encoder_config_path = self.encoder_config_path or _SHAPE_ENCODER_CONFIG_CONVENTION
        try:
            encoder_config = read_structured_latent_encoder_config(
                self.root, resolved_encoder_config_path
            )
        except (FileNotFoundError, ValueError) as error:
            return Trellis2TexturingResult(
                image_path=image,
                mesh_path=mesh_file,
                blocker=Trellis2TexturingBlocker(
                    stage="fdg-encoder",
                    operation="FDG encoder config validation",
                    reference=str(self.root / resolved_encoder_config_path),
                    reason=str(error),
                    next_slice="place a valid FlexiDualGridVaeEncoder config under weights/trellis2",
                ),
            )

        resolved_encoder_ckpt = self.encoder_checkpoint_path or config.shape_decoder_checkpoint_path
        encoder_ckpt_path = Path(self.root) / resolved_encoder_ckpt
        if not encoder_ckpt_path.is_file():
            return Trellis2TexturingResult(
                image_path=image,
                mesh_path=mesh_file,
                blocker=Trellis2TexturingBlocker(
                    stage="fdg-encoder",
                    operation="FDG encoder checkpoint validation",
                    reference=str(encoder_ckpt_path),
                    reason=f"encoder checkpoint not found: {encoder_ckpt_path}",
                    next_slice="place a valid FDG encoder/decoder checkpoint under weights/trellis2",
                ),
            )

        try:
            encoder_result = run_flexi_dual_grid_vae_encoder(
                encoder_ckpt_path,
                encoder_config,
                encoder_coords_mx,
                dual_mx,
                intersected_mx,
                sample_posterior=False,
            )
        except (FileNotFoundError, ValueError) as error:
            return Trellis2TexturingResult(
                image_path=image,
                mesh_path=mesh_file,
                blocker=Trellis2TexturingBlocker(
                    stage="fdg-encoder",
                    operation="FlexiDualGrid VAE encoder forward",
                    reference=str(encoder_ckpt_path),
                    reason=str(error),
                    next_slice="ensure encoder checkpoint and config are compatible",
                ),
            )

        cond_output, cond_blocker = _resolve_dinov3_conditioning(
            self.root,
            config,
            preprocessed.image.image,
            self.dino_root,
            route.output_resolution,
        )
        if cond_blocker is not None:
            return Trellis2TexturingResult(
                image_path=image,
                mesh_path=mesh_file,
                blocker=cond_blocker,
            )
        if cond_output is None:
            return Trellis2TexturingResult(
                image_path=image,
                mesh_path=mesh_file,
                blocker=Trellis2TexturingBlocker(
                    stage="image-conditioning",
                    operation="DINOv3 conditioning",
                    reference=str(self.dino_root or "weights/dinov3-vitl16-pretrain-lvd1689m"),
                    reason="DINOv3 conditioning returned no output",
                    next_slice="ensure DINOv3 assets are available and configured",
                ),
            )

        try:
            shape_decoder_config = read_structured_latent_decoder_config(
                self.root, config.shape_decoder_config_path
            )
            shape_result = run_shape_decoder_to_fields(
                Path(self.root) / config.shape_decoder_checkpoint_path,
                shape_decoder_config,
                encoder_result.coordinates,
                encoder_result.features,
                decoder_token_limit=decoder_token_limit,
            )
        except (FileNotFoundError, ValueError) as error:
            return Trellis2TexturingResult(
                image_path=image,
                mesh_path=mesh_file,
                blocker=Trellis2TexturingBlocker(
                    stage="shape-decoder",
                    operation="shape decoder FlexiDualGrid field extraction",
                    reference=config.shape_decoder_checkpoint_path,
                    reason=str(error),
                    next_slice="ensure shape decoder checkpoint and config are available",
                ),
            )

        texture_conditioning = cond_output.payload
        if texture_conditioning is None:
            return Trellis2TexturingResult(
                image_path=image,
                mesh_path=mesh_file,
                blocker=Trellis2TexturingBlocker(
                    stage="texture-slat",
                    operation="texture SLat conditioning payload availability",
                    reference=str(route.output_resolution),
                    reason=f"no conditioning payload for resolution {route.output_resolution}",
                    next_slice="produce DINOv3 conditioning before texture SLat sampling",
                ),
            )

        try:
            texture_coords, texture_features, texture_detail = _sample_texture_slat_for_texturing(
                self.root,
                config,
                route.model_key,
                encoder_result.coordinates,
                encoder_result.features,
                texture_conditioning,
            )
        except (FileNotFoundError, ValueError) as error:
            return Trellis2TexturingResult(
                image_path=image,
                mesh_path=mesh_file,
                blocker=Trellis2TexturingBlocker(
                    stage="texture-slat",
                    operation="texture SLat FlowEuler sampling",
                    reference=route.model_key,
                    reason=str(error),
                    next_slice="ensure texture SLat flow config and checkpoint are available",
                ),
            )

        try:
            texture_decoder_config = read_structured_latent_decoder_config(
                self.root, config.texture_decoder_config_path
            )
            texture_result = run_texture_decoder_to_representation(
                Path(self.root) / config.texture_decoder_checkpoint_path,
                texture_decoder_config,
                texture_coords,
                texture_features,
                guide_subdivisions=shape_result.subdivisions,
                decoder_token_limit=decoder_token_limit,
                decode_resolution=route.output_resolution,
                shape_decoder_coordinates=shape_result.coordinates,
            )
        except (FileNotFoundError, ValueError) as error:
            return Trellis2TexturingResult(
                image_path=image,
                mesh_path=mesh_file,
                blocker=Trellis2TexturingBlocker(
                    stage="texture-decoder",
                    operation="texture decoder PBR voxel decoding",
                    reference=config.texture_decoder_checkpoint_path,
                    reason=str(error),
                    next_slice="ensure texture decoder checkpoint and config are available",
                ),
            )

        try:
            mesh = flexi_dual_grid_fields_to_mesh(
                shape_result.coordinates,
                shape_result.fields,
                grid_size=route.output_resolution,
            )
            postprocess_result = postprocess_trellis2_mesh_for_glb(
                mesh, target_faces=glb_target_faces
            )
            baked = bake_trellis2_texture_fields_mac_native(
                postprocess_result.mesh,
                texture_result.coordinates,
                texture_result.attributes,
                decode_resolution=route.output_resolution,
                texture_size=texture_size,
                xatlas_face_guard=xatlas_face_guard,
                xatlas_parallel_chunks=xatlas_parallel_chunks,
                texture_bake_backend=texture_bake_backend,
                projection_source_mesh=getattr(postprocess_result, "source_mesh", None),
            )
        except (ImportError, ValueError) as error:
            return Trellis2TexturingResult(
                image_path=image,
                mesh_path=mesh_file,
                blocker=Trellis2TexturingBlocker(
                    stage="mesh-export",
                    operation="mesh extraction and texture baking",
                    reference="src/mlx_spatial/trellis2_export.py:bake_trellis2_texture_fields_mac_native",
                    reason=str(error),
                    next_slice="ensure mesh extraction, postprocessing, and UV baking are functional",
                ),
            )

        try:
            artifact = write_trellis2_textured_glb(baked, output_path)
            clear_mlx_cache()
        except (OSError, ValueError) as error:
            return Trellis2TexturingResult(
                image_path=image,
                mesh_path=mesh_file,
                blocker=Trellis2TexturingBlocker(
                    stage="mesh-export",
                    operation="textured GLB writer",
                    reference=str(output_path),
                    reason=str(error),
                    next_slice="ensure output directory is writable",
                ),
            )

        return Trellis2TexturingResult(
            image_path=image,
            mesh_path=mesh_file,
            artifact=artifact,
        )


def _load_obj_mesh(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load vertices and faces from a simple triangle OBJ file."""
    verts: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    with open(path, "r") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if parts[0] == "v":
                verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif parts[0] == "f":
                indices = []
                for token in parts[1:]:
                    idx = int(token.split("/")[0])
                    indices.append(idx - 1 if idx > 0 else idx)
                if len(indices) == 3:
                    faces.append((indices[0], indices[1], indices[2]))
                elif len(indices) == 4:
                    faces.append((indices[0], indices[1], indices[2]))
                    faces.append((indices[0], indices[2], indices[3]))
    if not verts:
        raise ValueError("OBJ file contains no vertices")
    if not faces:
        raise ValueError("OBJ file contains no triangular faces")
    return np.array(verts, dtype=np.float32), np.array(faces, dtype=np.int64)


def _resolve_dinov3_conditioning(
    root: Path,
    config,
    image,
    dino_root: Path | None,
    output_resolution: int,
) -> tuple[Trellis2StageOutput | None, Trellis2TexturingBlocker | None]:
    resolution = 512 if output_resolution == 512 else 1024
    cond_config = replace(config, conditioning_resolution=resolution)
    image_tensor = prepare_dinov3_image_tensor(image, image_size=resolution)
    cond_output, blocker = assess_dinov3_conditioning(
        root, cond_config, dino_root=dino_root, image_tensor=image_tensor
    )
    if blocker is not None:
        return None, Trellis2TexturingBlocker(
            stage="image-conditioning",
            operation="DINOv3 conditioning",
            reference=str(dino_root or "weights/dinov3-vitl16-pretrain-lvd1689m"),
            reason=blocker.reason,
            next_slice=blocker.next_slice,
        )
    if cond_output is None:
        return None, None
    if cond_output.payload is None:
        return None, Trellis2TexturingBlocker(
            stage="image-conditioning",
            operation="DINOv3 conditioning payload",
            reference=str(resolution),
            reason=f"no DINOv3 conditioning payload for resolution {resolution}",
            next_slice="produce DINOv3 conditioning before texture SLat sampling",
        )
    return cond_output, None


def _sample_texture_slat_for_texturing(
    root: Path,
    config,
    model_key: str,
    shape_coordinates: mx.array,
    shape_features: mx.array,
    conditioning: mx.array,
) -> tuple[mx.array, mx.array, str]:
    from .trellis2_inference import _shape_slat_model_paths, _undo_slat_normalization

    model_paths = _texture_slat_model_paths_texturing(config, model_key)
    slat_config = read_slat_flow_config(root, model_paths[0])
    normalized_shape = _undo_slat_normalization(
        shape_features, config.shape_slat_normalization, name="shape_slat"
    )
    probe = probe_texture_slat_forward_boundary(
        root / model_paths[1],
        slat_config,
        shape_coordinates,
        normalized_shape,
        conditioning=conditioning,
        steps=config.texture_slat_sampler.steps,
        rescale_t=config.texture_slat_sampler.rescale_t,
        guidance_strength=config.texture_slat_sampler.guidance_strength,
        guidance_rescale=config.texture_slat_sampler.guidance_rescale,
        guidance_interval=config.texture_slat_sampler.guidance_interval,
        sigma_min=config.texture_slat_sampler.sigma_min,
    )
    if probe.sampled_features is None or probe.sampled_feature_shape is None:
        raise ValueError(
            f"{model_key} did not produce sampled texture_slat features: {probe.blocker_detail}"
        )
    from .trellis2_forward import _apply_slat_normalization

    sampled = _apply_slat_normalization(
        probe.sampled_features, config.texture_slat_normalization, name="texture_slat"
    )
    detail = (
        f"shape_slat coordinate shape {probe.coordinate_shape}; "
        f"normalized shape feature shape {probe.shape_feature_shape}; "
        f"texture noise feature shape {probe.noise_feature_shape}; "
        f"concat feature shape {probe.concat_feature_shape}; "
        f"{probe.blocker_detail}"
    )
    return shape_coordinates, sampled, detail


def _texture_slat_model_paths_texturing(config, model_key: str) -> tuple[str, str]:
    if model_key == "tex_slat_flow_model_512":
        return config.texture_slat_512_config_path, config.texture_slat_512_checkpoint_path
    if model_key == "tex_slat_flow_model_1024":
        return config.texture_slat_1024_config_path, config.texture_slat_1024_checkpoint_path
    raise ValueError(f"unsupported texture SLat model key: {model_key}")


def _preprocess_texturing_blocker(
    blocker: Trellis2PreprocessBlocker | None,
) -> Trellis2TexturingBlocker:
    if blocker is not None:
        return Trellis2TexturingBlocker(
            stage=blocker.stage,
            operation=blocker.operation,
            reference=blocker.reference,
            reason=blocker.reason,
            next_slice=blocker.next_slice,
        )
    return Trellis2TexturingBlocker(
        stage="image-preprocessing-background",
        operation="image preprocessing",
        reference="src/mlx_spatial/trellis2_preprocess.py:preprocess_trellis2_image",
        reason="image preprocessing failed without a specific blocker",
        next_slice="provide a valid image with background removal",
    )
