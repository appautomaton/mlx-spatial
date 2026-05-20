"""Staged SAM 3D Objects MLX inference surface."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .mlx_memory import clear_mlx_cache, mlx_memory_snapshot, reset_mlx_peak_memory
from .sam3d_assets import Sam3dAssetBlocker, inspect_sam3d_model_assets
from .sam3d_condition import (
    Sam3dDinoConfig,
    load_sam3d_condition_tensors,
    run_sam3d_slat_condition_stack,
    run_sam3d_ss_condition_stack,
)
from .sam3d_moge import SAM3D_MOGE_DEFAULT_ROOT, run_sam3d_moge_pointmap
from .sam3d_pose import decode_sam3d_scale_shift_invariant_pose
from .sam3d_preprocess import (
    Sam3dOfficialPreprocessOutput,
    Sam3dPreprocessedInput,
    preprocess_sam3d_image_mask,
    preprocess_sam3d_official_tensors,
)
from .sam3d_ss import (
    load_sam3d_ss_decoder_tensors,
    read_sam3d_ss_decoder_config,
    run_sam3d_ss_decoder,
)
from .sam3d_ss_flow import (
    infer_sam3d_ss_flow_config,
    load_sam3d_ss_generator_tensors,
    run_sam3d_ss_shortcut_flow,
)
from .sam3d_slat import (
    infer_sam3d_slat_flow_config,
    load_sam3d_slat_generator_tensors,
    run_sam3d_slat_flow,
)
from .sam3d_decoder import (
    load_sam3d_mesh_decoder_tensors,
    load_sam3d_slat_decoder_tensors,
    read_sam3d_mesh_decoder_config,
    read_sam3d_slat_decoder_config,
    run_sam3d_slat_decoder_network,
)
from .sam3d_export import (
    SAM3D_GLB_DEFAULT_MIN_COMPONENT_FACE_FRACTION,
    SAM3D_GLB_DEFAULT_MIN_COMPONENT_FACES,
    SAM3D_GLB_DEFAULT_TARGET_FACES,
    SAM3D_XATLAS_FACE_GUARD,
    bake_sam3d_gaussian_texture_for_glb,
    compute_sam3d_vertex_normals,
    postprocess_sam3d_mesh_for_glb,
    write_sam3d_basic_glb,
    write_sam3d_gaussians_ply,
    write_sam3d_textured_glb,
)
from .sam3d_gaussian import Sam3dGaussianDecoderConfig, decode_sam3d_gaussian_fields
from .sam3d_mesh import extract_sam3d_mesh_from_features, run_sam3d_mesh_decoder_features


SAM3D_INFERENCE_STAGES = (
    "asset-validation",
    "pipeline-config",
    "image-mask-preprocessing",
    "moge-pointmap",
    "external-pointmap",
    "official-preprocessing",
    "sparse-structure",
    "structured-latent",
    "gaussian-decoder",
    "ply-export",
    "mesh-decoder",
    "glb-export",
)


@dataclass(frozen=True)
class Sam3dOutputArtifact:
    """User-visible artifact produced by the SAM3D pipeline."""

    name: str
    path: Path
    kind: str


@dataclass(frozen=True)
class Sam3dInferenceTrace:
    """Trace metadata for exact-mode SAM3D inference attempts."""

    root: Path
    image_path: Path
    mask_path: Path
    output_path: Path
    glb_output_path: Path | None
    completed_stages: tuple[str, ...]
    outputs: tuple[Sam3dOutputArtifact, ...]
    blocker: Sam3dAssetBlocker | None
    metadata: dict[str, object]

    @property
    def ready(self) -> bool:
        return self.blocker is None


@dataclass(frozen=True)
class Sam3dGenerationResult:
    """Result for `image + mask -> gaussians.ply` plus optional preview GLB."""

    trace: Sam3dInferenceTrace
    artifact: Sam3dOutputArtifact | None


@dataclass(frozen=True)
class Sam3dExternalPointmap:
    """Validated external pointmap input and trace metadata."""

    pointmap: np.ndarray
    metadata: dict[str, object]


class Sam3dInferencePipeline:
    """Exact-mode SAM3D Objects pipeline harness for the MLX port."""

    def __init__(self, root: str | Path = "weights/sam-3d-objects"):
        self.root = Path(root)

    def generate_gaussians_ply(
        self,
        image_path: str | Path,
        *,
        mask_path: str | Path,
        output_path: str | Path,
        glb_output_path: str | Path | None = None,
        moge_root: str | Path = SAM3D_MOGE_DEFAULT_ROOT,
        pointmap_path: str | Path | None = None,
        seed: int = 42,
        stage1_steps: int = 2,
        stage2_steps: int = 12,
        memory_profile: str = "balanced",
        glb_postprocess: str = "cleaned",
        glb_target_faces: int = SAM3D_GLB_DEFAULT_TARGET_FACES,
        glb_min_component_faces: int = SAM3D_GLB_DEFAULT_MIN_COMPONENT_FACES,
        glb_min_component_face_fraction: float = SAM3D_GLB_DEFAULT_MIN_COMPONENT_FACE_FRACTION,
        glb_smooth_iterations: int = 0,
        glb_texture: str = "gaussian",
        glb_texture_size: int = 1024,
        glb_gaussian_k: int = 8,
        glb_texel_chunk_size: int = 262_144,
        glb_xatlas_face_guard: int = SAM3D_XATLAS_FACE_GUARD,
    ) -> Sam3dGenerationResult:
        """Run the staged SAM3D path, blocking rather than faking unported model stages."""

        output = _validate_sam3d_output_path(output_path)
        glb_output = _validate_sam3d_glb_output_path(glb_output_path) if glb_output_path else None
        completed: list[str] = []
        outputs: list[Sam3dOutputArtifact] = []
        if stage1_steps <= 0 or stage2_steps <= 0:
            raise ValueError("stage1_steps and stage2_steps must be positive")
        if memory_profile not in {"safe", "balanced", "large"}:
            raise ValueError(f"unsupported SAM3D memory profile: {memory_profile}")
        if glb_postprocess not in {"cleaned", "basic"}:
            raise ValueError(f"unsupported SAM3D GLB postprocess mode: {glb_postprocess}")
        if glb_texture not in {"gaussian", "none"}:
            raise ValueError(f"unsupported SAM3D GLB texture mode: {glb_texture}")
        if glb_target_faces < 0:
            raise ValueError(f"glb_target_faces must be non-negative, got {glb_target_faces}")
        if glb_min_component_faces <= 0:
            raise ValueError(f"glb_min_component_faces must be positive, got {glb_min_component_faces}")
        if glb_min_component_face_fraction < 0:
            raise ValueError(
                f"glb_min_component_face_fraction must be non-negative, got {glb_min_component_face_fraction}"
            )
        if glb_smooth_iterations < 0:
            raise ValueError(f"glb_smooth_iterations must be non-negative, got {glb_smooth_iterations}")
        if glb_texture_size <= 0:
            raise ValueError(f"glb_texture_size must be positive, got {glb_texture_size}")
        if glb_gaussian_k <= 0:
            raise ValueError(f"glb_gaussian_k must be positive, got {glb_gaussian_k}")
        if glb_texel_chunk_size <= 0:
            raise ValueError(f"glb_texel_chunk_size must be positive, got {glb_texel_chunk_size}")
        if glb_xatlas_face_guard <= 0:
            raise ValueError(f"glb_xatlas_face_guard must be positive, got {glb_xatlas_face_guard}")
        if glb_output is not None and glb_texture == "gaussian" and glb_postprocess != "cleaned":
            raise ValueError("SAM3D Gaussian textured GLB requires --glb-postprocess cleaned")
        reset_mlx_peak_memory()
        metadata: dict[str, object] = {
            "seed": int(seed),
            "exact_mode": True,
            "moge_root": str(moge_root),
            "pointmap_source": "external" if pointmap_path is not None else "moge",
            "stage1_steps": int(stage1_steps),
            "stage2_steps": int(stage2_steps),
            "memory_profile": memory_profile,
            "requested_outputs": {
                "gaussians_ply": str(output),
                "mesh_glb": str(glb_output) if glb_output is not None else None,
            },
            "glb_postprocess": {
                "mode": glb_postprocess,
                "target_faces": int(glb_target_faces),
                "min_component_faces": int(glb_min_component_faces),
                "min_component_face_fraction": float(glb_min_component_face_fraction),
                "simplify": bool(glb_target_faces > 0),
                "smooth_iterations": int(glb_smooth_iterations),
                "texture": glb_texture,
                "texture_size": int(glb_texture_size),
                "gaussian_k": int(glb_gaussian_k),
                "texel_chunk_size": int(glb_texel_chunk_size),
                "xatlas_face_guard": int(glb_xatlas_face_guard),
            },
        }
        _record_mlx_memory(metadata, "start")

        inspection = inspect_sam3d_model_assets(self.root, required_roles=_required_reconstruct_roles(glb_output is not None))
        metadata["asset_validation"] = {
            "ready": inspection.validation.ready,
            "root": str(inspection.validation.root),
            "model_dir": str(inspection.validation.model_dir),
            "pipeline_path": str(inspection.validation.pipeline_path)
            if inspection.validation.pipeline_path is not None
            else None,
            "present": inspection.validation.present,
            "missing": inspection.validation.missing,
        }
        if inspection.blocker is not None:
            return self._blocked(
                image_path,
                mask_path,
                output,
                glb_output,
                completed,
                outputs,
                inspection.blocker,
                metadata,
            )
        completed.extend(("asset-validation", "pipeline-config"))
        metadata["pipeline"] = {
            "target": inspection.config.target if inspection.config else None,
            "dtype": inspection.config.dtype if inspection.config else None,
            "rendering_engine": inspection.config.rendering_engine if inspection.config else None,
            "decode_formats": inspection.config.decode_formats if inspection.config else (),
            "checkpoints": tuple(
                {
                    "role": item.role,
                    "relative_path": item.relative_path,
                    "tensor_count": item.tensor_count,
                    "prefixes": item.prefixes,
                }
                for item in inspection.checkpoints
            ),
        }

        try:
            preprocessed = preprocess_sam3d_image_mask(image_path, mask_path)
        except (FileNotFoundError, ValueError) as error:
            blocker = Sam3dAssetBlocker(
                stage="image-mask-preprocessing",
                operation="load SAM3D RGB image and binary object mask",
                reason=str(error),
                metadata={"image": str(image_path), "mask": str(mask_path)},
            )
            return self._blocked(image_path, mask_path, output, glb_output, completed, outputs, blocker, metadata)

        completed.append("image-mask-preprocessing")
        metadata["input"] = _preprocess_metadata(preprocessed)

        if pointmap_path is not None:
            try:
                external_pointmap = load_sam3d_external_pointmap(
                    pointmap_path,
                    expected_image_shape=preprocessed.rgba.shape[:2],
                )
            except (FileNotFoundError, OSError, ValueError) as error:
                blocker = Sam3dAssetBlocker(
                    stage="external-pointmap",
                    operation="load SAM3D external pointmap input",
                    reason=str(error),
                    metadata={"pointmap_path": str(pointmap_path)},
                )
                return self._blocked(image_path, mask_path, output, glb_output, completed, outputs, blocker, metadata)
            completed.append("external-pointmap")
            metadata["external_pointmap"] = external_pointmap.metadata
            metadata["moge"] = {
                "skipped": True,
                "reason": "external pointmap input supplied",
            }
            pointmap = external_pointmap.pointmap
        else:
            moge_result = run_sam3d_moge_pointmap(
                preprocessed.rgba[..., :3],
                root=moge_root,
                memory_profile=memory_profile,
            )
            metadata["moge"] = {
                "ready": moge_result.ready,
                "inspection": _moge_inspection_metadata(moge_result.inspection),
            }
            if moge_result.blocker is not None:
                return self._blocked(
                    image_path,
                    mask_path,
                    output,
                    glb_output,
                    completed,
                    outputs,
                    moge_result.blocker,
                    metadata,
                )

            completed.append("moge-pointmap")
            assert moge_result.pointmap is not None
            metadata["moge"]["pointmap"] = {
                "shape": tuple(int(value) for value in moge_result.pointmap.pointmap.shape),
                "intrinsics_shape": tuple(int(value) for value in moge_result.pointmap.intrinsics.shape),
                "valid_pixels": int(moge_result.pointmap.mask.sum()),
                "depth_min": float(moge_result.pointmap.depth.min()),
                "depth_max": float(moge_result.pointmap.depth.max()),
                "metadata": moge_result.pointmap.metadata,
            }
            pointmap = moge_result.pointmap.pointmap

        try:
            official = preprocess_sam3d_official_tensors(
                preprocessed.rgba,
                pointmap=pointmap,
            )
        except ValueError as error:
            blocker = Sam3dAssetBlocker(
                stage="official-preprocessing",
                operation="run official SAM3D crop/pad/resize pointmap preprocessing",
                reason=str(error),
                metadata={"image": str(image_path), "mask": str(mask_path)},
            )
            return self._blocked(image_path, mask_path, output, glb_output, completed, outputs, blocker, metadata)
        completed.append("official-preprocessing")
        metadata["official_preprocessing"] = _official_preprocess_metadata(official)
        if pointmap_path is None:
            del moge_result
        del pointmap, preprocessed
        _release_mlx_stage_memory(metadata, "after-official-preprocessing")

        try:
            ss_generator_path = _checkpoint_path_for_role(inspection, "ss_generator")
            ss_condition_tensors = load_sam3d_condition_tensors(ss_generator_path)
            dino_memory = _condition_memory_config(memory_profile)
            ss_condition = run_sam3d_ss_condition_stack(
                official,
                ss_condition_tensors,
                dino_config=Sam3dDinoConfig(
                    attention_chunk_size=dino_memory["attention_chunk"],
                    max_attention_bytes=dino_memory["max_attention_bytes"],
                ),
            )
            metadata["ss_condition"] = dict(ss_condition.metadata)
            metadata["ss_condition"]["materialized_token_shape"] = tuple(int(value) for value in ss_condition.tokens.shape)
        except (KeyError, ValueError, OSError, TypeError) as error:
            blocker = Sam3dAssetBlocker(
                stage="sparse-structure",
                operation="run SAM3D SS condition embedders in MLX",
                reason=str(error),
                metadata={
                    "required_checkpoint_role": "ss_generator",
                    "stage1_steps": int(stage1_steps),
                },
            )
            return self._blocked(image_path, mask_path, output, glb_output, completed, outputs, blocker, metadata)

        try:
            ss_generator_tensors = load_sam3d_ss_generator_tensors(ss_generator_path)
            ss_memory = _stage1_memory_config(memory_profile)
            ss_flow_config = infer_sam3d_ss_flow_config(
                ss_generator_tensors,
                cfg_strength=7.0,
                cfg_interval=(0.0, 500.0),
                rescale_t=3.0,
                attention_chunk_size=ss_memory["attention_chunk"],
            )
            ss_flow = run_sam3d_ss_shortcut_flow(
                ss_condition.tokens,
                ss_generator_tensors,
                seed=seed,
                steps=stage1_steps,
                config=ss_flow_config,
            )
            ss_decoder_config = read_sam3d_ss_decoder_config(_config_path_for_role(inspection, "ss_decoder"))
            ss_decoder_tensors = load_sam3d_ss_decoder_tensors(_checkpoint_path_for_role(inspection, "ss_decoder"))
            downsample_dist = int(inspection.config.raw.get("downsample_ss_dist", 1)) if inspection.config else 1
            ss_decoded = run_sam3d_ss_decoder(
                ss_flow.latents["shape"],
                ss_decoder_tensors,
                ss_decoder_config,
                prune_neighbor_axes_dist=downsample_dist,
            )
            if ss_decoded.coords.shape[0] == 0:
                raise ValueError("SS decoder produced zero occupied coordinates")
            pose = decode_sam3d_scale_shift_invariant_pose(
                {name: np.array(ss_flow.latents[name], dtype=np.float32) for name in ss_flow.latents},
                scene_scale=np.asarray(official.pointmap_scale, dtype=np.float32)
                if official.pointmap_scale is not None
                else None,
                scene_shift=np.asarray(official.pointmap_shift, dtype=np.float32)
                if official.pointmap_shift is not None
                else None,
            )
            completed.append("sparse-structure")
            metadata["sparse_structure"] = {
                "flow": ss_flow.metadata,
                "decoder": ss_decoded.metadata,
                "coords_shape": tuple(int(value) for value in ss_decoded.coords.shape),
                "coords_original_shape": tuple(int(value) for value in ss_decoded.coords_original.shape),
                "pose": pose.metadata,
            }
            del ss_condition, ss_condition_tensors, ss_generator_tensors, ss_decoder_tensors, ss_flow
            _release_mlx_stage_memory(metadata, "after-sparse-structure")
        except (KeyError, ValueError, OSError, TypeError) as error:
            blocker = Sam3dAssetBlocker(
                stage="sparse-structure",
                operation="run SAM3D SS ShortCut flow and SS decoder in MLX",
                reason=str(error),
                metadata={
                    "required_checkpoint_roles": ("ss_generator", "ss_decoder"),
                    "stage1_steps": int(stage1_steps),
                    "memory_profile": memory_profile,
                },
            )
            return self._blocked(image_path, mask_path, output, glb_output, completed, outputs, blocker, metadata)

        try:
            slat_generator_path = _checkpoint_path_for_role(inspection, "slat_generator")
            slat_condition_tensors = load_sam3d_condition_tensors(slat_generator_path)
            dino_memory = _condition_memory_config(memory_profile)
            slat_condition = run_sam3d_slat_condition_stack(
                official,
                slat_condition_tensors,
                dino_config=Sam3dDinoConfig(
                    prenorm_features=True,
                    attention_chunk_size=dino_memory["attention_chunk"],
                    max_attention_bytes=dino_memory["max_attention_bytes"],
                ),
            )
            metadata["slat_condition"] = dict(slat_condition.metadata)
            metadata["slat_condition"]["materialized_token_shape"] = tuple(int(value) for value in slat_condition.tokens.shape)
            slat_generator_tensors = load_sam3d_slat_generator_tensors(slat_generator_path)
            slat_memory = _stage2_memory_config(memory_profile)
            slat_config = infer_sam3d_slat_flow_config(
                slat_generator_tensors,
                cfg_strength=float(inspection.config.raw.get("slat_cfg_strength", 1.0)) if inspection.config else 1.0,
                cfg_interval=(0.0, 500.0),
                rescale_t=float(inspection.config.raw.get("slat_rescale_t", 1.0)) if inspection.config else 1.0,
                attention_chunk_size=slat_memory["attention_chunk"],
            )
            slat_mean = _pipeline_float_tuple(inspection.config.raw.get("slat_mean")) if inspection.config else None
            slat_std = _pipeline_float_tuple(inspection.config.raw.get("slat_std")) if inspection.config else None
            slat = run_sam3d_slat_flow(
                ss_decoded.coords,
                slat_condition.tokens,
                slat_generator_tensors,
                seed=seed,
                steps=stage2_steps,
                config=slat_config,
                slat_mean=slat_mean,
                slat_std=slat_std,
            )
            completed.append("structured-latent")
            metadata["structured_latent"] = {
                **slat.metadata,
                "coords_shape": tuple(int(value) for value in slat.coords.shape),
                "feature_shape": tuple(int(value) for value in slat.feats.shape),
                "token_count": int(slat.coords.shape[0]),
            }
            del slat_condition, slat_condition_tensors, slat_generator_tensors, ss_decoded, official
            _release_mlx_stage_memory(metadata, "after-structured-latent")
        except (KeyError, ValueError, OSError, TypeError) as error:
            blocker = Sam3dAssetBlocker(
                stage="structured-latent",
                operation="run SAM3D SLat FlowMatching in MLX",
                reason=str(error),
                metadata={
                    "required_checkpoint_role": "slat_generator",
                    "stage2_steps": int(stage2_steps),
                    "memory_profile": memory_profile,
                    "coords_shape": tuple(int(value) for value in ss_decoded.coords.shape),
                },
            )
            return self._blocked(image_path, mask_path, output, glb_output, completed, outputs, blocker, metadata)

        try:
            gs_decoder_config = read_sam3d_slat_decoder_config(_config_path_for_role(inspection, "slat_decoder_gs"))
            gs_decoder_tensors = load_sam3d_slat_decoder_tensors(_checkpoint_path_for_role(inspection, "slat_decoder_gs"))
            raw_gaussian = run_sam3d_slat_decoder_network(
                slat.coords,
                slat.feats,
                gs_decoder_tensors,
                gs_decoder_config,
            )
            gaussian_fields = decode_sam3d_gaussian_fields(
                slat.coords,
                np.array(raw_gaussian, dtype=np.float32),
                config=Sam3dGaussianDecoderConfig(resolution=gs_decoder_config.resolution),
            )
            ply_stats = write_sam3d_gaussians_ply(
                output,
                xyz=gaussian_fields.xyz,
                features_dc=gaussian_fields.features_dc,
                opacity=gaussian_fields.opacity,
                scale=gaussian_fields.scale,
                rotation=gaussian_fields.rotation,
                binary=True,
            )
            completed.extend(("gaussian-decoder", "ply-export"))
            outputs.append(Sam3dOutputArtifact(name="gaussians.ply", path=output, kind="gaussian-ply"))
            metadata["gaussian_decoder"] = {
                "network_output_shape": tuple(int(value) for value in raw_gaussian.shape),
                "fields": gaussian_fields.metadata,
            }
            metadata["ply_export"] = {
                "path": str(ply_stats.path),
                "vertex_count": int(ply_stats.vertex_count),
                "bytes_written": int(ply_stats.bytes_written),
                "format": ply_stats.format,
                "fields": ply_stats.fields,
            }
            del gs_decoder_tensors, raw_gaussian
            _release_mlx_stage_memory(metadata, "after-ply-export")
        except (KeyError, ValueError, OSError, TypeError) as error:
            blocker = Sam3dAssetBlocker(
                stage="gaussian-decoder",
                operation="run SAM3D SLat gaussian decoder and write official-field PLY",
                reason=str(error),
                metadata={
                    "required_checkpoint_role": "slat_decoder_gs",
                    "slat_feature_shape": tuple(int(value) for value in slat.feats.shape),
                },
            )
            return self._blocked(image_path, mask_path, output, glb_output, completed, outputs, blocker, metadata)

        if glb_output is None:
            del slat, gaussian_fields
            _release_mlx_stage_memory(metadata, "final")
            trace = Sam3dInferenceTrace(
                root=self.root,
                image_path=Path(image_path),
                mask_path=Path(mask_path),
                output_path=output,
                glb_output_path=None,
                completed_stages=tuple(completed),
                outputs=tuple(outputs),
                blocker=None,
                metadata=metadata,
            )
            return Sam3dGenerationResult(trace=trace, artifact=outputs[-1])

        try:
            mesh_decoder_config = read_sam3d_mesh_decoder_config(_config_path_for_role(inspection, "slat_decoder_mesh"))
            mesh_decoder_tensors = load_sam3d_mesh_decoder_tensors(_checkpoint_path_for_role(inspection, "slat_decoder_mesh"))
            mesh_features = run_sam3d_mesh_decoder_features(
                slat.coords,
                slat.feats,
                mesh_decoder_tensors,
                mesh_decoder_config,
            )
            mesh_memory = _mesh_memory_config(memory_profile)
            mesh_extraction_resolution = _mesh_extraction_resolution(mesh_decoder_config.resolution)
            mesh = extract_sam3d_mesh_from_features(
                mesh_features.coords,
                mesh_features.feats,
                extraction_resolution=mesh_extraction_resolution,
                use_color=mesh_decoder_config.use_color,
                max_dense_bytes=mesh_memory["max_dense_bytes"],
                max_flexicubes_bytes=mesh_memory["max_flexicubes_bytes"],
            )
            metadata["mesh_decoder"] = {
                "config": {
                    "resolution": int(mesh_decoder_config.resolution),
                    "model_channels": int(mesh_decoder_config.model_channels),
                    "latent_channels": int(mesh_decoder_config.latent_channels),
                    "num_blocks": int(mesh_decoder_config.num_blocks),
                    "num_heads": int(mesh_decoder_config.num_heads),
                    "window_size": int(mesh_decoder_config.window_size),
                    "use_color": bool(mesh_decoder_config.use_color),
                    "extraction_resolution": int(mesh_extraction_resolution),
                },
                "feature_decoder": mesh_features.metadata,
                "extraction": mesh.metadata,
                "raw_mesh": {
                    "vertex_count": int(mesh.vertices.shape[0]) if mesh.vertices is not None else 0,
                    "face_count": int(mesh.faces.shape[0]) if mesh.faces is not None else 0,
                    "has_vertex_color": bool(mesh.colors is not None),
                    "artifact_role": "raw SAM3D mesh-decoder preview surface before GLB postprocess",
                },
                "memory_profile": mesh_memory,
            }
            if mesh.blocker is not None:
                return self._blocked(image_path, mask_path, output, glb_output, completed, outputs, mesh.blocker, metadata)
            if not mesh.ready or mesh.vertices is None or mesh.faces is None:
                blocker = Sam3dAssetBlocker(
                    stage="mesh-decoder",
                    operation="run SAM3D SLat mesh decoder and extract FlexiCubes mesh",
                    reason="mesh decoder completed without a ready mesh result",
                    metadata=metadata["mesh_decoder"],
                )
                return self._blocked(image_path, mask_path, output, glb_output, completed, outputs, blocker, metadata)
            completed.append("mesh-decoder")
            del mesh_decoder_tensors, mesh_features, slat
            _release_mlx_stage_memory(metadata, "after-mesh-decoder")
        except (KeyError, ValueError, OSError, TypeError) as error:
            blocker = Sam3dAssetBlocker(
                stage="mesh-decoder",
                operation="run SAM3D SLat mesh decoder and extract FlexiCubes mesh",
                reason=str(error),
                metadata={
                    "required_checkpoint_role": "slat_decoder_mesh",
                    "slat_coords_shape": tuple(int(value) for value in slat.coords.shape),
                    "slat_feature_shape": tuple(int(value) for value in slat.feats.shape),
                    "memory_profile": memory_profile,
                },
            )
            return self._blocked(image_path, mask_path, output, glb_output, completed, outputs, blocker, metadata)

        try:
            if glb_postprocess == "cleaned":
                postprocessed = postprocess_sam3d_mesh_for_glb(
                    mesh.vertices,
                    mesh.faces,
                    colors=mesh.colors,
                    target_faces=int(glb_target_faces),
                    simplify=glb_target_faces > 0,
                    smooth_iterations=int(glb_smooth_iterations),
                    min_component_faces=int(glb_min_component_faces),
                    min_component_face_fraction=float(glb_min_component_face_fraction),
                )
                export_vertices = postprocessed.vertices
                export_faces = postprocessed.faces
                export_colors = postprocessed.colors
                export_normals = postprocessed.normals
                postprocess_metadata = _dataclass_to_plain_dict(postprocessed.stats)
            else:
                export_vertices = mesh.vertices
                export_faces = mesh.faces
                export_colors = None if mesh.colors is None else np.clip(mesh.colors, 0.0, 1.0).astype(np.float32, copy=False)
                export_normals = compute_sam3d_vertex_normals(export_vertices, export_faces)
                postprocess_metadata = {
                    "mode": "basic",
                    "applied": False,
                    "raw_vertices": int(export_vertices.shape[0]),
                    "raw_faces": int(export_faces.shape[0]),
                    "has_vertex_color": export_colors is not None,
                    "has_normals": True,
                }
            texture_metadata: dict[str, object]
            if glb_texture == "gaussian":
                baked_texture = bake_sam3d_gaussian_texture_for_glb(
                    postprocessed,
                    gaussian_xyz=gaussian_fields.xyz,
                    gaussian_features_dc=gaussian_fields.features_dc,
                    gaussian_opacity=gaussian_fields.opacity,
                    gaussian_scale=gaussian_fields.scale,
                    texture_size=int(glb_texture_size),
                    k_neighbors=int(glb_gaussian_k),
                    texel_chunk_size=int(glb_texel_chunk_size),
                    xatlas_face_guard=int(glb_xatlas_face_guard),
                )
                glb_stats = write_sam3d_textured_glb(glb_output, baked_texture)
                texture_metadata = _dataclass_to_plain_dict(baked_texture.stats)
                texture_metadata["base_color_shape"] = tuple(int(value) for value in baked_texture.base_color_rgba.shape)
            else:
                glb_stats = write_sam3d_basic_glb(
                    glb_output,
                    vertices=export_vertices,
                    faces=export_faces,
                    colors=export_colors,
                    normals=export_normals,
                )
                texture_metadata = {"backend": "none", "has_texture": False}
            completed.append("glb-export")
            outputs.append(Sam3dOutputArtifact(name="mesh.glb", path=glb_output, kind="mesh-glb"))
            metadata["glb_export"] = {
                "path": str(glb_stats.path),
                "vertex_count": int(glb_stats.vertex_count),
                "face_count": int(glb_stats.face_count),
                "bytes_written": int(glb_stats.bytes_written),
                "has_vertex_color": bool(glb_stats.has_vertex_color),
                "has_normals": bool(glb_stats.has_normals),
                "has_texture": bool(glb_stats.has_texture),
                "format": glb_stats.format,
                "postprocess": postprocess_metadata,
                "texture": texture_metadata,
                "artifact_role": (
                    "textured preview mesh GLB baked from SAM3D Gaussian colors"
                    if glb_stats.has_texture
                    else "preview mesh GLB derived from SAM3D mesh decoder; not a textured gaussian-splat conversion"
                ),
            }
            del mesh, gaussian_fields
            _release_mlx_stage_memory(metadata, "final")
        except (ImportError, ValueError, OSError) as error:
            blocker = Sam3dAssetBlocker(
                stage="glb-export",
                operation="write SAM3D textured mesh GLB" if glb_texture == "gaussian" else "write SAM3D basic mesh GLB",
                reason=str(error),
                metadata={
                    "glb_output": str(glb_output),
                    "vertex_count": int(mesh.vertices.shape[0]),
                    "face_count": int(mesh.faces.shape[0]),
                    "has_vertex_color": mesh.colors is not None,
                    "glb_postprocess": metadata.get("glb_postprocess"),
                },
            )
            return self._blocked(image_path, mask_path, output, glb_output, completed, outputs, blocker, metadata)

        trace = Sam3dInferenceTrace(
            root=self.root,
            image_path=Path(image_path),
            mask_path=Path(mask_path),
            output_path=output,
            glb_output_path=glb_output,
            completed_stages=tuple(completed),
            outputs=tuple(outputs),
            blocker=None,
            metadata=metadata,
        )
        return Sam3dGenerationResult(trace=trace, artifact=outputs[-1])

    def _blocked(
        self,
        image_path: str | Path,
        mask_path: str | Path,
        output_path: Path,
        glb_output_path: Path | None,
        completed_stages: list[str],
        outputs: list[Sam3dOutputArtifact],
        blocker: Sam3dAssetBlocker,
        metadata: dict[str, object],
    ) -> Sam3dGenerationResult:
        trace = Sam3dInferenceTrace(
            root=self.root,
            image_path=Path(image_path),
            mask_path=Path(mask_path),
            output_path=output_path,
            glb_output_path=glb_output_path,
            completed_stages=tuple(completed_stages),
            outputs=tuple(outputs),
            blocker=blocker,
            metadata=metadata,
        )
        return Sam3dGenerationResult(trace=trace, artifact=None)


def _validate_sam3d_output_path(path: str | Path, outputs_root: str | Path = "outputs") -> Path:
    output = Path(path)
    if output.suffix.lower() != ".ply":
        raise ValueError("SAM3D native parity output must be a .ply gaussian splat file")
    resolved_output = output.resolve()
    resolved_root = Path(outputs_root).resolve()
    try:
        resolved_output.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError(f"SAM3D output path must stay under {outputs_root}") from error
    return output


def load_sam3d_external_pointmap(
    path: str | Path,
    *,
    expected_image_shape: tuple[int, int] | None = None,
) -> Sam3dExternalPointmap:
    """Load and validate an external SAM3D pointmap file."""

    pointmap_path = Path(path)
    suffix = pointmap_path.suffix.lower()
    if suffix == ".npy":
        array = np.load(pointmap_path, allow_pickle=False)
        file_format = "npy"
    elif suffix == ".npz":
        with np.load(pointmap_path, allow_pickle=False) as archive:
            if "pointmap" not in archive.files:
                raise ValueError("SAM3D external pointmap .npz must contain a 'pointmap' array")
            array = archive["pointmap"]
        file_format = "npz"
    else:
        raise ValueError("SAM3D external pointmap must be a .npy or .npz file")

    if array.dtype.hasobject:
        raise ValueError("SAM3D external pointmap must not use an object dtype")
    if array.ndim != 3:
        raise ValueError(f"SAM3D external pointmap must have shape (H, W, 3), got {array.shape}")
    if int(array.shape[2]) != 3:
        raise ValueError(f"SAM3D external pointmap must have exactly 3 channels, got {array.shape[2]}")
    if expected_image_shape is not None and tuple(int(value) for value in array.shape[:2]) != tuple(
        int(value) for value in expected_image_shape
    ):
        raise ValueError(
            f"SAM3D external pointmap spatial shape must match image shape "
            f"{tuple(int(value) for value in expected_image_shape)}, got {tuple(int(value) for value in array.shape[:2])}"
        )

    try:
        pointmap = array.astype(np.float32, copy=False)
    except (TypeError, ValueError) as error:
        raise ValueError("SAM3D external pointmap must be convertible to float32") from error

    metadata = {
        "source_path": str(pointmap_path),
        "file_format": file_format,
        "original_shape": tuple(int(value) for value in array.shape),
        "original_dtype": str(array.dtype),
        "shape": tuple(int(value) for value in pointmap.shape),
        "dtype": str(pointmap.dtype),
        "finite_count": int(np.isfinite(pointmap).sum()),
        "nan_count": int(np.isnan(pointmap).sum()),
        "source_numeric_parity": "not_claimed",
        "source_clipping_parity": "deferred",
        "source_intrinsics_parity": "deferred",
    }
    return Sam3dExternalPointmap(pointmap=pointmap, metadata=metadata)


def _validate_sam3d_glb_output_path(path: str | Path, outputs_root: str | Path = "outputs") -> Path:
    output = Path(path)
    if output.suffix.lower() != ".glb":
        raise ValueError("SAM3D mesh output must be a .glb file")
    resolved_output = output.resolve()
    resolved_root = Path(outputs_root).resolve()
    try:
        resolved_output.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError(f"SAM3D GLB output path must stay under {outputs_root}") from error
    return output


def _required_reconstruct_roles(include_mesh: bool) -> tuple[str, ...]:
    roles = (
        "ss_generator",
        "slat_generator",
        "ss_decoder",
        "slat_decoder_gs",
    )
    return (*roles, "slat_decoder_mesh") if include_mesh else roles


def _preprocess_metadata(preprocessed: Sam3dPreprocessedInput) -> dict[str, object]:
    return {
        "image_path": str(preprocessed.image_path),
        "mask_path": str(preprocessed.mask_path),
        "size": preprocessed.size,
        "foreground_pixels": preprocessed.foreground_pixels,
        "rgba_shape": tuple(int(value) for value in preprocessed.rgba.shape),
    }


def _checkpoint_path_for_role(inspection, role: str) -> Path:
    for item in inspection.checkpoints:
        if item.role == role:
            return item.path
    raise KeyError(f"SAM3D checkpoint role not found in pipeline inspection: {role}")


def _config_path_for_role(inspection, role: str) -> Path:
    for item in inspection.paths:
        if item.role == role and item.kind == "config":
            return item.path
    raise KeyError(f"SAM3D config role not found in pipeline inspection: {role}")


def _condition_memory_config(memory_profile: str) -> dict[str, int]:
    if memory_profile == "safe":
        return {"attention_chunk": 96, "max_attention_bytes": 800_000_000}
    if memory_profile == "large":
        return {"attention_chunk": 256, "max_attention_bytes": 3_000_000_000}
    return {"attention_chunk": 192, "max_attention_bytes": 1_600_000_000}


def _stage1_memory_config(memory_profile: str) -> dict[str, int]:
    if memory_profile == "safe":
        return {"attention_chunk": 64}
    if memory_profile == "large":
        return {"attention_chunk": 256}
    return {"attention_chunk": 128}


def _stage2_memory_config(memory_profile: str) -> dict[str, int]:
    if memory_profile == "safe":
        return {"attention_chunk": 64}
    if memory_profile == "large":
        return {"attention_chunk": 192}
    return {"attention_chunk": 96}


def _mesh_memory_config(memory_profile: str) -> dict[str, int]:
    if memory_profile == "safe":
        return {
            "max_dense_bytes": 1_500_000_000,
            "max_flexicubes_bytes": 1_500_000_000,
        }
    if memory_profile == "large":
        return {
            "max_dense_bytes": 24_000_000_000,
            "max_flexicubes_bytes": 80_000_000_000,
        }
    return {
        "max_dense_bytes": 6_000_000_000,
        "max_flexicubes_bytes": 6_000_000_000,
    }


def _mesh_extraction_resolution(mesh_decoder_resolution: int) -> int:
    return int(mesh_decoder_resolution) * 4


def _record_mlx_memory(metadata: dict[str, object], label: str) -> None:
    memory = metadata.setdefault("mlx_memory", {})
    if isinstance(memory, dict):
        memory[label] = mlx_memory_snapshot().as_dict()


def _release_mlx_stage_memory(metadata: dict[str, object], label: str) -> None:
    clear_mlx_cache()
    _record_mlx_memory(metadata, label)


def _pipeline_float_tuple(value) -> tuple[float, ...] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)):
        raise ValueError("SAM3D pipeline slat mean/std must be a list")
    return tuple(float(item) for item in value)


def _dataclass_to_plain_dict(value) -> dict[str, object]:
    return asdict(value) if hasattr(value, "__dataclass_fields__") else dict(value)


def _moge_inspection_metadata(inspection) -> dict[str, object] | None:
    if inspection is None:
        return None
    return {
        "root": str(inspection.root),
        "checkpoint_path": str(inspection.checkpoint_path),
        "ready": inspection.ready,
        "tensor_count": inspection.tensor_count,
        "sample_keys": inspection.sample_keys,
    }


def _official_preprocess_metadata(output: Sam3dOfficialPreprocessOutput) -> dict[str, object]:
    return {
        "image_shape": tuple(int(value) for value in output.image.shape),
        "mask_shape": tuple(int(value) for value in output.mask.shape),
        "rgb_image_shape": tuple(int(value) for value in output.rgb_image.shape),
        "rgb_image_mask_shape": tuple(int(value) for value in output.rgb_image_mask.shape),
        "pointmap_shape": tuple(int(value) for value in output.pointmap.shape)
        if output.pointmap is not None
        else None,
        "rgb_pointmap_shape": tuple(int(value) for value in output.rgb_pointmap.shape)
        if output.rgb_pointmap is not None
        else None,
        "pointmap_scale": tuple(float(value) for value in output.pointmap_scale)
        if output.pointmap_scale is not None
        else None,
        "pointmap_shift": tuple(float(value) for value in output.pointmap_shift)
        if output.pointmap_shift is not None
        else None,
        "crop_box": output.crop_box,
        "output_size": output.output_size,
    }
