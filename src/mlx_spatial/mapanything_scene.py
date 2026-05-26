"""MapAnything MLX scene-generation pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import mlx.core as mx
import numpy as np

from .mapanything_assets import MAPANYTHING_DEFAULT_ROOT, inspect_mapanything_model_assets
from .mapanything_geometry import MapAnythingPostprocessConfig, postprocess_mapanything_heads_output
from .mapanything_heads import (
    apply_mapanything_fusion_norm,
    load_mapanything_heads_weights,
    mapanything_heads_config_from_model_config,
    run_mapanything_heads,
)
from .mapanything_model import (
    load_mapanything_full_encoder_weights,
    load_mapanything_info_sharing_weights,
    mapanything_encoder_prefix_config_from_model_config,
    mapanything_info_sharing_config_from_model_config,
    run_mapanything_full_encoder,
    run_mapanything_info_sharing,
)
from .mapanything_preprocess import MapAnythingPreprocessedInput, preprocess_mapanything_images


MAPANYTHING_SCENE_REQUIRED_GROUPS = (
    "encoder",
    "info_sharing",
    "dense_head",
    "pose_head",
    "scale_head",
    "fusion_norm_layer",
    "scale_token",
)
MAPANYTHING_SCENE_OUTPUT_KEYS = (
    "images",
    "depth",
    "confidence",
    "masks",
    "intrinsics",
    "camera_poses",
    "extrinsics",
    "world_points",
)


@dataclass(frozen=True)
class MapAnythingSceneBlocker:
    """Structured setup or implementation blocker for scene generation."""

    stage: str
    operation: str
    reason: str
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class MapAnythingScenePredictions:
    """Scene-level MapAnything prediction tensors.

    Arrays use view-first layout. For Desk-style image-only inference this means
    ``[V, H, W, C]`` for images/world points and ``[V, H, W]`` for dense scalar
    maps.
    """

    images: np.ndarray
    depth: np.ndarray
    confidence: np.ndarray
    masks: np.ndarray
    intrinsics: np.ndarray
    camera_poses: np.ndarray
    extrinsics: np.ndarray
    world_points: np.ndarray
    metadata: dict[str, object] = field(default_factory=dict)

    def as_npz_payload(self) -> dict[str, np.ndarray]:
        """Return the stable `.npz` scene payload schema."""

        return {
            "images": self.images,
            "depth": self.depth,
            "confidence": self.confidence,
            "masks": self.masks,
            "intrinsics": self.intrinsics,
            "camera_poses": self.camera_poses,
            "extrinsics": self.extrinsics,
            "world_points": self.world_points,
            "__metadata_json__": np.array(_metadata_json(self.metadata)),
        }


@dataclass(frozen=True)
class MapAnythingSceneTrace:
    """Trace for the MapAnything scene pipeline."""

    completed_stages: tuple[str, ...]
    model_root: Path
    input_path: Path
    frame_count: int
    target_size: tuple[int, int] | None
    output_keys: tuple[str, ...] = MAPANYTHING_SCENE_OUTPUT_KEYS
    metadata: dict[str, object] = field(default_factory=dict)
    blocker: MapAnythingSceneBlocker | None = None

    @property
    def ready(self) -> bool:
        return self.blocker is None


@dataclass(frozen=True)
class MapAnythingSceneResult:
    """Result wrapper for MapAnything scene generation."""

    trace: MapAnythingSceneTrace
    preprocessed: MapAnythingPreprocessedInput | None = None
    predictions: MapAnythingScenePredictions | None = None

    @property
    def ready(self) -> bool:
        return self.trace.ready and self.predictions is not None


class MapAnythingScenePipeline:
    """Validate assets and inputs for MLX-native MapAnything scene generation."""

    def __init__(self, root: str | Path = MAPANYTHING_DEFAULT_ROOT):
        self.root = Path(root)

    def generate(
        self,
        input_path: str | Path,
        *,
        resize_mode: str = "fixed_mapping",
        size: int | tuple[int, int] | None = None,
        stride: int = 1,
    ) -> MapAnythingSceneResult:
        completed: list[str] = []
        source = Path(input_path)

        inspection = inspect_mapanything_model_assets(
            self.root,
            required_groups=MAPANYTHING_SCENE_REQUIRED_GROUPS,
        )
        if inspection.blocker is not None or inspection.config is None:
            return MapAnythingSceneResult(
                trace=_trace(
                    completed,
                    root=self.root,
                    input_path=source,
                    blocker=_scene_blocker_from_asset_inspection(inspection.blocker),
                )
            )
        completed.append("asset-config-validation")

        try:
            preprocessed = preprocess_mapanything_images(
                source,
                resize_mode=resize_mode,
                size=size,
                patch_size=inspection.config.patch_size,
                stride=stride,
            )
        except (FileNotFoundError, ValueError, OSError) as error:
            return MapAnythingSceneResult(
                trace=_trace(
                    completed,
                    root=self.root,
                    input_path=source,
                    blocker=MapAnythingSceneBlocker(
                        stage="image-preprocessing",
                        operation="prepare MapAnything image-only scene views",
                        reason=str(error),
                    ),
                )
            )
        completed.append("image-preprocessing")

        try:
            encoder_config = mapanything_encoder_prefix_config_from_model_config(inspection.config)
            info_config = mapanything_info_sharing_config_from_model_config(inspection.config)
            heads_config = mapanything_heads_config_from_model_config(inspection.config)
        except ValueError as error:
            return MapAnythingSceneResult(
                trace=_trace(
                    completed,
                    root=self.root,
                    input_path=source,
                    frame_count=preprocessed.frame_count,
                    target_size=preprocessed.target_size,
                    blocker=MapAnythingSceneBlocker(
                        stage="model-config",
                        operation="resolve MapAnything MLX scene-generation configs",
                        reason=str(error),
                    ),
                ),
                preprocessed=preprocessed,
            )
        completed.append("model-config")

        try:
            encoder_weights = load_mapanything_full_encoder_weights(self.root, config=encoder_config)
        except (FileNotFoundError, ValueError, OSError) as error:
            return MapAnythingSceneResult(
                trace=_trace(
                    completed,
                    root=self.root,
                    input_path=source,
                    frame_count=preprocessed.frame_count,
                    target_size=preprocessed.target_size,
                    blocker=MapAnythingSceneBlocker(
                        stage="checkpoint-loading",
                        operation="load MapAnything full encoder tensors",
                        reason=str(error),
                    ),
                ),
                preprocessed=preprocessed,
            )
        completed.append("checkpoint-loading:encoder")

        images = mx.concatenate([view.img for view in preprocessed.views], axis=0)
        try:
            encoder_output = run_mapanything_full_encoder(images, encoder_weights, config=encoder_config)
            mx.eval(encoder_output.features, encoder_output.registers)
        except (RuntimeError, ValueError) as error:
            return MapAnythingSceneResult(
                trace=_trace(
                    completed,
                    root=self.root,
                    input_path=source,
                    frame_count=preprocessed.frame_count,
                    target_size=preprocessed.target_size,
                    blocker=MapAnythingSceneBlocker(
                        stage="full-encoder",
                        operation="run MLX MapAnything full encoder",
                        reason=str(error),
                    ),
                ),
                preprocessed=preprocessed,
            )
        completed.append("full-encoder")
        patch_grid = encoder_output.patch_grid
        encoder_features = tuple(
            encoder_output.features[index : index + 1] for index in range(preprocessed.frame_count)
        )
        encoder_registers = tuple(
            encoder_output.registers[index : index + 1] for index in range(preprocessed.frame_count)
        )
        del encoder_output, encoder_weights, images

        try:
            heads_weights = load_mapanything_heads_weights(self.root, config=heads_config)
        except (FileNotFoundError, ValueError, OSError) as error:
            return MapAnythingSceneResult(
                trace=_trace(
                    completed,
                    root=self.root,
                    input_path=source,
                    frame_count=preprocessed.frame_count,
                    target_size=preprocessed.target_size,
                    blocker=MapAnythingSceneBlocker(
                        stage="checkpoint-loading",
                        operation="load MapAnything prediction-head tensors",
                        reason=str(error),
                    ),
                ),
                preprocessed=preprocessed,
            )
        completed.append("checkpoint-loading:heads")

        try:
            fused_features = apply_mapanything_fusion_norm(encoder_features, heads_weights, config=heads_config)
            mx.eval(*fused_features)
        except (ValueError, RuntimeError) as error:
            return MapAnythingSceneResult(
                trace=_trace(
                    completed,
                    root=self.root,
                    input_path=source,
                    frame_count=preprocessed.frame_count,
                    target_size=preprocessed.target_size,
                    blocker=MapAnythingSceneBlocker(
                        stage="fusion-norm",
                        operation="load heads and apply MapAnything fusion norm",
                        reason=str(error),
                    ),
                ),
                preprocessed=preprocessed,
            )
        completed.append("fusion-norm")
        del encoder_features

        try:
            info_weights = load_mapanything_info_sharing_weights(self.root, config=info_config)
        except (FileNotFoundError, ValueError, OSError) as error:
            return MapAnythingSceneResult(
                trace=_trace(
                    completed,
                    root=self.root,
                    input_path=source,
                    frame_count=preprocessed.frame_count,
                    target_size=preprocessed.target_size,
                    blocker=MapAnythingSceneBlocker(
                        stage="checkpoint-loading",
                        operation="load MapAnything info-sharing tensors",
                        reason=str(error),
                    ),
                ),
                preprocessed=preprocessed,
            )
        completed.append("checkpoint-loading:info-sharing")

        try:
            info_output = run_mapanything_info_sharing(
                fused_features,
                info_weights,
                additional_tokens_per_view=encoder_registers,
                config=info_config,
            )
            _eval_info_output(info_output)
        except (ValueError, RuntimeError) as error:
            return MapAnythingSceneResult(
                trace=_trace(
                    completed,
                    root=self.root,
                    input_path=source,
                    frame_count=preprocessed.frame_count,
                    target_size=preprocessed.target_size,
                    blocker=MapAnythingSceneBlocker(
                        stage="info-sharing",
                        operation="run MLX MapAnything multi-view info sharing",
                        reason=str(error),
                    ),
                ),
                preprocessed=preprocessed,
            )
        completed.append("info-sharing")
        del info_weights, encoder_registers

        try:
            dense_features = _dense_head_inputs(fused_features, info_output)
            if info_output.final.additional_token_features is None:
                raise ValueError("info-sharing did not return the global scale token")
            image_shape = preprocessed.views[0].true_shape
            heads_output = run_mapanything_heads(
                dense_features,
                info_output.final.additional_token_features,
                heads_weights,
                image_shape=image_shape,
                config=heads_config,
            )
            mx.eval(
                heads_output.dense.value,
                heads_output.dense.confidence,
                heads_output.dense.mask,
                heads_output.dense.logits,
                heads_output.pose_value,
                heads_output.scale_value,
            )
        except (ValueError, RuntimeError) as error:
            return MapAnythingSceneResult(
                trace=_trace(
                    completed,
                    root=self.root,
                    input_path=source,
                    frame_count=preprocessed.frame_count,
                    target_size=preprocessed.target_size,
                    blocker=MapAnythingSceneBlocker(
                        stage="prediction-heads",
                        operation="run MLX MapAnything dense pose scale heads",
                        reason=str(error),
                    ),
                ),
                preprocessed=preprocessed,
            )
        completed.append("prediction-heads")
        del fused_features, heads_weights, info_output

        try:
            postprocess = postprocess_mapanything_heads_output(
                heads_output,
                preprocessed,
                config=MapAnythingPostprocessConfig(apply_mask=True, mask_edges=True),
            )
            scene_payload = postprocess.scene_payload
        except (ValueError, RuntimeError) as error:
            return MapAnythingSceneResult(
                trace=_trace(
                    completed,
                    root=self.root,
                    input_path=source,
                    frame_count=preprocessed.frame_count,
                    target_size=preprocessed.target_size,
                    blocker=MapAnythingSceneBlocker(
                        stage="scene-postprocess",
                        operation="postprocess MapAnything scene predictions",
                        reason=str(error),
                    ),
                ),
                preprocessed=preprocessed,
            )
        completed.append("scene-postprocess")

        predictions = MapAnythingScenePredictions(
            images=scene_payload["images"],
            depth=scene_payload["depth"],
            confidence=scene_payload["confidence"],
            masks=scene_payload["masks"],
            intrinsics=scene_payload["intrinsics"],
            camera_poses=scene_payload["camera_poses"],
            extrinsics=scene_payload["extrinsics"],
            world_points=scene_payload["world_points"],
            metadata={
                "runtime_depends_on_torch": False,
                "model_root": str(self.root),
                "input_path": str(source),
                "frame_count": preprocessed.frame_count,
                "target_size": list(preprocessed.target_size),
                "patch_grid": list(patch_grid),
                "implemented_boundary": "scene-generation",
                "postprocess": postprocess.trace,
                "optional_mesh_export": "unsupported; npz scene bundle is the supported runtime artifact",
            },
        )
        return MapAnythingSceneResult(
            trace=_trace(
                completed,
                root=self.root,
                input_path=source,
                frame_count=preprocessed.frame_count,
                target_size=preprocessed.target_size,
                metadata={
                    "runtime_depends_on_torch": False,
                    "implemented_boundary": "scene-generation",
                    "output_schema": list(MAPANYTHING_SCENE_OUTPUT_KEYS),
                    "required_model_components": list(MAPANYTHING_SCENE_REQUIRED_GROUPS),
                    "patch_grid": list(patch_grid),
                    "postprocess": postprocess.trace,
                    "optional_mesh_export": "unsupported; npz scene bundle is the supported runtime artifact",
                },
            ),
            preprocessed=preprocessed,
            predictions=predictions,
        )


def main(argv: list[str] | None = None) -> int:
    """CLI for MLX-native MapAnything scene generation."""

    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Generate a MapAnything MLX scene bundle.")
    parser.add_argument("model_root", help="Local MapAnything model root containing config.json and model.safetensors")
    parser.add_argument("input_path", help="Input image file or folder")
    parser.add_argument(
        "--output",
        help="Output .npz scene bundle path; default: outputs/mapanything/<input-stem>/scene.npz",
    )
    parser.add_argument("--resize-mode", default="fixed_mapping")
    parser.add_argument("--stride", type=int, default=1)
    args = parser.parse_args(argv)

    output_path = Path(args.output) if args.output else _default_scene_output_path(args.input_path)
    if output_path.suffix != ".npz":
        print("MapAnything scene generation currently supports .npz output only", file=sys.stderr)
        return 2

    result = MapAnythingScenePipeline(args.model_root).generate(
        args.input_path,
        resize_mode=args.resize_mode,
        stride=args.stride,
    )
    if not result.ready or result.predictions is None:
        blocker = result.trace.blocker
        if blocker is None:
            print("MapAnything scene generation failed without a structured blocker", file=sys.stderr)
        else:
            print(f"{blocker.stage}: {blocker.reason}", file=sys.stderr)
        return 1

    written = write_mapanything_scene_npz(
        output_path,
        result.predictions,
        metadata={
            **result.predictions.metadata,
            "completed_stages": list(result.trace.completed_stages),
        },
    )
    print(f"Wrote MapAnything scene bundle: {written}")
    return 0


def _default_scene_output_path(input_path: str | Path) -> Path:
    source = Path(input_path)
    slug = source.stem or source.name or "mapanything-scene"
    return Path("outputs/mapanything") / slug / "scene.npz"


def _eval_info_output(info_output: object) -> None:
    arrays = []
    final = getattr(info_output, "final")
    arrays.extend(final.features)
    if final.additional_token_features is not None:
        arrays.append(final.additional_token_features)
    for intermediate in getattr(info_output, "intermediates"):
        arrays.extend(intermediate.features)
    mx.eval(*arrays)


def _dense_head_inputs(
    fused_features: tuple[mx.array, ...],
    info_output: object,
) -> tuple[mx.array, ...]:
    dense_inputs = [mx.concatenate(fused_features, axis=0)]
    for intermediate in info_output.intermediates:
        dense_inputs.append(mx.concatenate(intermediate.features, axis=0))
    dense_inputs.append(mx.concatenate(info_output.final.features, axis=0))
    if len(dense_inputs) != 4:
        raise ValueError(
            "MapAnything DPT heads expect fused features, two intermediate info-sharing features, "
            f"and final info-sharing features; got {len(dense_inputs)} tensors"
        )
    return tuple(dense_inputs)


def write_mapanything_scene_npz(
    path: str | Path,
    predictions: MapAnythingScenePredictions,
    *,
    metadata: Mapping[str, object] | None = None,
) -> Path:
    """Write a MapAnything scene prediction bundle."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = predictions.as_npz_payload()
    if metadata:
        merged = {**predictions.metadata, **dict(metadata)}
        payload["__metadata_json__"] = np.array(_metadata_json(merged))
    np.savez_compressed(output_path, **payload)
    return output_path


def _trace(
    completed: list[str],
    *,
    root: Path,
    input_path: Path,
    frame_count: int = 0,
    target_size: tuple[int, int] | None = None,
    metadata: dict[str, object] | None = None,
    blocker: MapAnythingSceneBlocker | None = None,
) -> MapAnythingSceneTrace:
    return MapAnythingSceneTrace(
        completed_stages=tuple(completed),
        model_root=root,
        input_path=input_path,
        frame_count=frame_count,
        target_size=target_size,
        metadata=dict(metadata or {"runtime_depends_on_torch": False}),
        blocker=blocker,
    )


def _scene_blocker_from_asset_inspection(blocker: object | None) -> MapAnythingSceneBlocker:
    if blocker is None:
        return MapAnythingSceneBlocker(
            stage="asset-validation",
            operation="validate MapAnything scene-generation assets",
            reason="MapAnything asset inspection returned no config",
        )
    return MapAnythingSceneBlocker(
        stage=str(getattr(blocker, "stage", "asset-validation")),
        operation=str(getattr(blocker, "operation", "validate MapAnything scene-generation assets")),
        reason=str(getattr(blocker, "reason", "unknown MapAnything asset blocker")),
        metadata=dict(getattr(blocker, "metadata", {})),
    )


def _metadata_json(metadata: Mapping[str, object]) -> str:
    import json

    return json.dumps(dict(metadata), sort_keys=True)


if __name__ == "__main__":
    raise SystemExit(main())
