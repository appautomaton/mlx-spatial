"""Forward-trace helpers for TRELLIS.2 attempt mode."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image

from .trellis2_decode import (
    STRUCTURED_LATENT_DECODER_REFERENCE_TOKEN_LIMIT,
    probe_decode_latents_boundary,
    read_structured_latent_decoder_config,
)
from .trellis2_dinov3 import assess_dinov3_mlx_conditioning
from .trellis2_sparse_structure import (
    fake_sparse_structure_sampling_metadata,
    probe_sparse_structure_forward_boundary,
    probe_sparse_structure_decoder_boundary,
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


@dataclass(frozen=True)
class Trellis2ForwardBlocker:
    stage: str
    operation: str
    reference: str
    reason: str
    next_slice: str


@dataclass(frozen=True)
class Trellis2StageOutput:
    stage: str
    name: str
    shape: tuple[int, ...]
    dtype: str
    detail: str
    payload: mx.array | None = field(default=None, compare=False, repr=False)


@dataclass(frozen=True)
class Trellis2ModelAsset:
    key: str
    base_path: str
    config_path: str
    checkpoint_path: str


@dataclass(frozen=True)
class Trellis2SamplerConfig:
    name: str
    sigma_min: float
    steps: int
    guidance_strength: float
    guidance_rescale: float
    guidance_interval: tuple[float, float]
    rescale_t: float


@dataclass(frozen=True)
class Trellis2NormalizationConfig:
    mean: tuple[float, ...]
    std: tuple[float, ...]


@dataclass(frozen=True)
class Trellis2ConditioningConfig:
    image_model_name: str
    image_model_family: str
    conditioning_resolution: int
    expected_feature_width: int
    sparse_flow_config_path: str
    sparse_flow_checkpoint_path: str
    sparse_decoder_config_path: str
    sparse_decoder_checkpoint_path: str
    shape_slat_512_config_path: str
    shape_slat_512_checkpoint_path: str
    shape_slat_1024_config_path: str
    shape_slat_1024_checkpoint_path: str
    texture_slat_512_config_path: str
    texture_slat_512_checkpoint_path: str
    texture_slat_1024_config_path: str
    texture_slat_1024_checkpoint_path: str
    shape_decoder_config_path: str
    shape_decoder_checkpoint_path: str
    texture_decoder_config_path: str
    texture_decoder_checkpoint_path: str
    default_pipeline_type: str
    model_assets: tuple[Trellis2ModelAsset, ...]
    sparse_structure_sampler: Trellis2SamplerConfig
    shape_slat_sampler: Trellis2SamplerConfig
    texture_slat_sampler: Trellis2SamplerConfig
    shape_slat_normalization: Trellis2NormalizationConfig
    texture_slat_normalization: Trellis2NormalizationConfig


@dataclass(frozen=True)
class Trellis2ForwardConfigResult:
    root: Path
    config: Trellis2ConditioningConfig | None = None
    blocker: Trellis2ForwardBlocker | None = None

    @property
    def ready(self) -> bool:
        return self.config is not None and self.blocker is None


@dataclass(frozen=True)
class Trellis2ForwardTraceResult:
    root: Path
    image_path: Path
    completed_stages: tuple[str, ...]
    outputs: tuple[Trellis2StageOutput, ...] = ()
    blocker: Trellis2ForwardBlocker | None = None

    @property
    def completed(self) -> bool:
        return self.blocker is None


_REQUIRED_MODEL_STAGES = {
    "sparse_structure_flow_model": "sparse-structure-sampling",
    "sparse_structure_decoder": "sparse-structure-sampling",
    "shape_slat_flow_model_512": "shape-slat-sampling",
    "shape_slat_flow_model_1024": "shape-slat-sampling",
    "shape_slat_decoder": "shape-decoder",
    "tex_slat_flow_model_512": "texture-slat-sampling",
    "tex_slat_flow_model_1024": "texture-slat-sampling",
    "tex_slat_decoder": "texture-decoder",
}


def discover_trellis2_conditioning_config(root: str | Path) -> Trellis2ForwardConfigResult:
    """Read local TRELLIS.2 config needed for image-conditioning forward tracing."""

    root_path = Path(root)
    pipeline_path = root_path / "pipeline.json"
    try:
        pipeline = json.loads(pipeline_path.read_text(encoding="utf-8"))
        args = pipeline["args"]
    except FileNotFoundError:
        return Trellis2ForwardConfigResult(
            root=root_path,
            blocker=_blocker(
                "image-conditioning",
                "TRELLIS.2 pipeline config discovery",
                str(pipeline_path),
                f"pipeline config file not found: {pipeline_path}",
                "place TRELLIS.2 pipeline.json under weights/trellis2",
            ),
        )
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        return Trellis2ForwardConfigResult(
            root=root_path,
            blocker=_blocker(
                "image-conditioning",
                "TRELLIS.2 pipeline config discovery",
                str(pipeline_path),
                f"pipeline config is missing required conditioning fields: {error}",
                "map the local TRELLIS.2 pipeline config before image-conditioning execution",
            ),
        )

    try:
        image_cond = args["image_cond_model"]
        image_model_family = image_cond["name"]
        image_model_name = image_cond["args"]["model_name"]
    except (KeyError, TypeError) as error:
        return Trellis2ForwardConfigResult(
            root=root_path,
            blocker=_blocker(
                "image-conditioning",
                "TRELLIS.2 pipeline config discovery",
                str(pipeline_path),
                f"pipeline config is missing required conditioning fields: {error}",
                "map the local TRELLIS.2 pipeline config before image-conditioning execution",
            ),
        )

    models_result = _discover_model_assets(args, pipeline_path)
    if models_result.blocker is not None:
        return Trellis2ForwardConfigResult(root=root_path, blocker=models_result.blocker)
    model_assets = models_result.model_assets
    sparse_flow_model = _model_asset(model_assets, "sparse_structure_flow_model")
    sparse_decoder_model = _model_asset(model_assets, "sparse_structure_decoder")
    shape_slat_512_model = _model_asset(model_assets, "shape_slat_flow_model_512")
    shape_slat_1024_model = _model_asset(model_assets, "shape_slat_flow_model_1024")
    texture_slat_512_model = _model_asset(model_assets, "tex_slat_flow_model_512")
    texture_slat_1024_model = _model_asset(model_assets, "tex_slat_flow_model_1024")
    shape_decoder_model = _model_asset(model_assets, "shape_slat_decoder")
    texture_decoder_model = _model_asset(model_assets, "tex_slat_decoder")

    sparse_sampler_result = _discover_sampler_config(args, "sparse_structure_sampler", "sparse-structure-sampling", pipeline_path)
    if isinstance(sparse_sampler_result, Trellis2ForwardBlocker):
        return Trellis2ForwardConfigResult(root=root_path, blocker=sparse_sampler_result)
    shape_sampler_result = _discover_sampler_config(args, "shape_slat_sampler", "shape-slat-sampling", pipeline_path)
    if isinstance(shape_sampler_result, Trellis2ForwardBlocker):
        return Trellis2ForwardConfigResult(root=root_path, blocker=shape_sampler_result)
    texture_sampler_result = _discover_sampler_config(args, "tex_slat_sampler", "texture-slat-sampling", pipeline_path)
    if isinstance(texture_sampler_result, Trellis2ForwardBlocker):
        return Trellis2ForwardConfigResult(root=root_path, blocker=texture_sampler_result)

    shape_norm_result = _discover_normalization_config(args, "shape_slat_normalization", "shape-slat-sampling", pipeline_path)
    if isinstance(shape_norm_result, Trellis2ForwardBlocker):
        return Trellis2ForwardConfigResult(root=root_path, blocker=shape_norm_result)
    texture_norm_result = _discover_normalization_config(args, "tex_slat_normalization", "texture-slat-sampling", pipeline_path)
    if isinstance(texture_norm_result, Trellis2ForwardBlocker):
        return Trellis2ForwardConfigResult(root=root_path, blocker=texture_norm_result)

    sparse_flow_config_path = sparse_flow_model.config_path
    sparse_config_path = root_path / sparse_flow_config_path
    try:
        sparse_config = json.loads(sparse_config_path.read_text(encoding="utf-8"))
        expected_feature_width = int(sparse_config["args"]["cond_channels"])
    except FileNotFoundError:
        return Trellis2ForwardConfigResult(
            root=root_path,
            blocker=_blocker(
                "sparse-structure-sampling",
                "sparse flow config discovery",
                str(sparse_config_path),
                f"sparse flow config file not found: {sparse_config_path}",
                "place sparse structure flow config under weights/trellis2",
            ),
        )
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        return Trellis2ForwardConfigResult(
            root=root_path,
            blocker=_blocker(
                "sparse-structure-sampling",
                "sparse flow config discovery",
                str(sparse_config_path),
                f"sparse flow config is missing cond_channels: {error}",
                "map the sparse structure flow config before downstream dispatch",
            ),
        )

    return Trellis2ForwardConfigResult(
        root=root_path,
            config=Trellis2ConditioningConfig(
                image_model_name=image_model_name,
                image_model_family=image_model_family,
                conditioning_resolution=int(image_cond["args"].get("image_size", 512)),
                expected_feature_width=expected_feature_width,
                sparse_flow_config_path=sparse_flow_config_path,
                sparse_flow_checkpoint_path=sparse_flow_model.checkpoint_path,
                sparse_decoder_config_path=sparse_decoder_model.config_path,
                sparse_decoder_checkpoint_path=sparse_decoder_model.checkpoint_path,
                shape_slat_512_config_path=shape_slat_512_model.config_path,
                shape_slat_512_checkpoint_path=shape_slat_512_model.checkpoint_path,
                shape_slat_1024_config_path=shape_slat_1024_model.config_path,
                shape_slat_1024_checkpoint_path=shape_slat_1024_model.checkpoint_path,
                texture_slat_512_config_path=texture_slat_512_model.config_path,
                texture_slat_512_checkpoint_path=texture_slat_512_model.checkpoint_path,
                texture_slat_1024_config_path=texture_slat_1024_model.config_path,
                texture_slat_1024_checkpoint_path=texture_slat_1024_model.checkpoint_path,
                shape_decoder_config_path=shape_decoder_model.config_path,
                shape_decoder_checkpoint_path=shape_decoder_model.checkpoint_path,
                texture_decoder_config_path=texture_decoder_model.config_path,
                texture_decoder_checkpoint_path=texture_decoder_model.checkpoint_path,
                default_pipeline_type=str(args.get("default_pipeline_type", "1024_cascade")),
                model_assets=model_assets,
                sparse_structure_sampler=sparse_sampler_result,
                shape_slat_sampler=shape_sampler_result,
                texture_slat_sampler=texture_sampler_result,
                shape_slat_normalization=shape_norm_result,
                texture_slat_normalization=texture_norm_result,
            ),
    )


@dataclass(frozen=True)
class _ModelDiscoveryResult:
    model_assets: tuple[Trellis2ModelAsset, ...] = ()
    blocker: Trellis2ForwardBlocker | None = None


def _discover_model_assets(args: dict, pipeline_path: Path) -> _ModelDiscoveryResult:
    try:
        models = args["models"]
    except (KeyError, TypeError) as error:
        return _ModelDiscoveryResult(
            blocker=_blocker(
                "sparse-structure-sampling",
                "TRELLIS.2 model contract discovery",
                str(pipeline_path),
                f"pipeline config is missing model map: {error}",
                "map the TRELLIS.2 model checkpoint contract before downstream dispatch",
            )
        )

    assets: list[Trellis2ModelAsset] = []
    for key, stage in _REQUIRED_MODEL_STAGES.items():
        try:
            base_path = str(models[key])
        except (KeyError, TypeError) as error:
            return _ModelDiscoveryResult(
                blocker=_blocker(
                    stage,
                    "TRELLIS.2 model contract discovery",
                    str(pipeline_path),
                    f"pipeline config is missing required model key {key!r}: {error}",
                    f"map the {stage} model checkpoint contract before downstream dispatch",
                )
            )
        assets.append(
            Trellis2ModelAsset(
                key=key,
                base_path=base_path,
                config_path=f"{base_path}.json",
                checkpoint_path=f"{base_path}.safetensors",
            )
        )
    return _ModelDiscoveryResult(model_assets=tuple(assets))


def _model_asset(model_assets: tuple[Trellis2ModelAsset, ...], key: str) -> Trellis2ModelAsset:
    for asset in model_assets:
        if asset.key == key:
            return asset
    raise ValueError(f"missing TRELLIS.2 model asset: {key}")


def _discover_sampler_config(
    args: dict,
    key: str,
    stage: str,
    pipeline_path: Path,
) -> Trellis2SamplerConfig | Trellis2ForwardBlocker:
    try:
        sampler = args[key]
        sampler_args = sampler["args"]
        sampler_params = sampler["params"]
        interval = tuple(float(value) for value in sampler_params["guidance_interval"])
        if len(interval) != 2:
            raise ValueError("guidance_interval must contain exactly two values")
        return Trellis2SamplerConfig(
            name=str(sampler["name"]),
            sigma_min=float(sampler_args["sigma_min"]),
            steps=int(sampler_params["steps"]),
            guidance_strength=float(sampler_params["guidance_strength"]),
            guidance_rescale=float(sampler_params["guidance_rescale"]),
            guidance_interval=(interval[0], interval[1]),
            rescale_t=float(sampler_params["rescale_t"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        return _blocker(
            stage,
            "TRELLIS.2 sampler config discovery",
            str(pipeline_path),
            f"pipeline config is missing or has invalid sampler {key!r}: {error}",
            f"map the {stage} sampler config before downstream dispatch",
        )


def _discover_normalization_config(
    args: dict,
    key: str,
    stage: str,
    pipeline_path: Path,
) -> Trellis2NormalizationConfig | Trellis2ForwardBlocker:
    try:
        normalization = args[key]
        mean = tuple(float(value) for value in normalization["mean"])
        std = tuple(float(value) for value in normalization["std"])
        if not mean or len(mean) != len(std):
            raise ValueError("mean and std must be non-empty and equal length")
        return Trellis2NormalizationConfig(mean=mean, std=std)
    except (KeyError, TypeError, ValueError) as error:
        return _blocker(
            stage,
            "TRELLIS.2 normalization config discovery",
            str(pipeline_path),
            f"pipeline config is missing or has invalid normalization {key!r}: {error}",
            f"map the {stage} normalization config before downstream dispatch",
        )


def prepare_dinov3_image_tensor(image: Image.Image, *, image_size: int = 512) -> mx.array:
    """Convert a PIL image to the normalized BCHW MLX tensor used by DINO conditioning."""

    resized = image.convert("RGB").resize((image_size, image_size), Image.Resampling.LANCZOS)
    array = np.array(resized).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    normalized = (array - mean) / std
    return mx.array(np.transpose(normalized, (2, 0, 1))[None, :, :, :])


def assess_dinov3_conditioning(
    root: str | Path,
    config: Trellis2ConditioningConfig,
    *,
    dino_root: str | Path | None = None,
    image_tensor: mx.array | None = None,
    conditioning: mx.array | None = None,
) -> tuple[Trellis2StageOutput | None, Trellis2ForwardBlocker | None]:
    """Attempt the local DINOv3 conditioning boundary or return the first blocker."""

    if conditioning is not None:
        return (
            Trellis2StageOutput(
                stage="image-conditioning",
                name="cond",
                shape=tuple(conditioning.shape),
                dtype=str(conditioning.dtype).removeprefix("mlx.core."),
                detail=f"simulated conditioning output for {config.image_model_name}",
                payload=conditioning,
            ),
            None,
        )

    model_root = Path(dino_root) if dino_root is not None else default_dinov3_root(root, config.image_model_name)
    result = assess_dinov3_mlx_conditioning(
        model_root,
        expected_feature_width=config.expected_feature_width,
        image_tensor=image_tensor,
    )
    if result.blocker is not None:
        return (
            None,
            Trellis2ForwardBlocker(
                stage=result.blocker.stage,
                operation=result.blocker.operation,
                reference=result.blocker.reference,
                reason=_dino_blocker_reason(result.blocker.reason, config.image_model_name, model_root),
                next_slice=result.blocker.next_slice,
            ),
        )

    return (
        Trellis2StageOutput(
            stage="image-conditioning",
            name="cond",
            shape=result.shape or (),
            dtype=result.dtype or "unknown",
            detail=result.detail or f"DINOv3 conditioning output for {config.image_model_name}",
            payload=result.hidden_states,
        ),
        None,
    )


def dispatch_sparse_structure_boundary(
    root: str | Path,
    config: Trellis2ConditioningConfig,
    conditioning: Trellis2StageOutput,
) -> Trellis2ForwardBlocker:
    """Validate conditioning metadata and return the first sparse-structure or decoder blocker."""

    _, blocker = dispatch_sparse_structure_sampling(root, config, conditioning)
    if blocker is None:
        return _blocker(
            "sparse-structure-decoding",
            "sparse structure decoder handoff",
            config.sparse_decoder_config_path,
            "sparse structure sampling completed but no downstream decoder blocker was produced",
            "map sparse structure decoder execution after sparse sampling",
        )
    return blocker


def dispatch_sparse_structure_sampling(
    root: str | Path,
    config: Trellis2ConditioningConfig,
    conditioning: Trellis2StageOutput,
    *,
    decoder_token_limit: int | None = None,
) -> tuple[Trellis2StageOutput | None, Trellis2ForwardBlocker | None]:
    """Run the sparse-structure sampling probe and return its output plus next blocker."""

    actual_width = conditioning.shape[-1] if conditioning.shape else None
    if actual_width != config.expected_feature_width:
        return (
            None,
            _blocker(
                "sparse-structure-sampling",
                "conditioning feature width validation",
                config.sparse_flow_config_path,
                f"conditioning width mismatch: expected {config.expected_feature_width}, got {actual_width}",
                "produce DINOv3 conditioning features with the sparse flow cond_channels width",
            ),
        )

    checkpoint_path = Path(root) / config.sparse_flow_checkpoint_path
    try:
        sparse_config = read_sparse_structure_flow_config(root, config.sparse_flow_config_path)
        metadata = fake_sparse_structure_sampling_metadata(
            sparse_config,
            steps=config.sparse_structure_sampler.steps,
            rescale_t=config.sparse_structure_sampler.rescale_t,
            guidance_interval=config.sparse_structure_sampler.guidance_interval,
        )
        probe = probe_sparse_structure_forward_boundary(
            checkpoint_path,
            sparse_config,
            conditioning=conditioning.payload,
            steps=config.sparse_structure_sampler.steps,
            rescale_t=config.sparse_structure_sampler.rescale_t,
            guidance_strength=config.sparse_structure_sampler.guidance_strength,
            guidance_rescale=config.sparse_structure_sampler.guidance_rescale,
            guidance_interval=config.sparse_structure_sampler.guidance_interval,
            sigma_min=config.sparse_structure_sampler.sigma_min,
        )
    except (FileNotFoundError, ValueError) as error:
        return (
            None,
            _blocker(
                "sparse-structure-sampling",
                "sparse flow config/checkpoint forward probe",
                str(checkpoint_path),
                str(error),
                "map the sparse structure flow config and checkpoint tensors before sampler execution",
            ),
        )

    if probe.sampled_latent is None or probe.sampled_latent_shape is None:
        return (
            None,
            _blocker(
                "sparse-structure-sampling",
                probe.blocker_operation,
                str(checkpoint_path),
                (
                    f"sparse flow config validates noise shape {metadata.noise_shape}, "
                    f"FlowEuler steps={metadata.steps}, active_guidance_steps={metadata.guidance_active_steps}, "
                    f"loads tensors {probe.loaded_tensor_names}, inspects block tensors, and {probe.blocker_detail}"
                ),
                "produce sampled sparse latent before sparse structure decoder handoff",
            ),
        )

    output = Trellis2StageOutput(
        stage="sparse-structure-sampling",
        name="sparse_latent",
        shape=probe.sampled_latent_shape,
        dtype=str(probe.sampled_latent.dtype).removeprefix("mlx.core."),
        detail=(
            f"MLX sparse structure FlowEuler sampler output after {metadata.steps} steps; "
            f"{probe.blocker_detail}"
        ),
        payload=probe.sampled_latent,
    )
    blocker = dispatch_sparse_structure_decoder_boundary(
        root,
        config,
        sparse_latent=probe.sampled_latent,
        conditioning=conditioning.payload,
        decoder_token_limit=decoder_token_limit,
    )
    return output, blocker


def dispatch_sparse_structure_decoder_boundary(
    root: str | Path,
    config: Trellis2ConditioningConfig,
    *,
    sparse_latent: mx.array | None = None,
    conditioning: mx.array | None = None,
    decoder_token_limit: int | None = None,
) -> Trellis2ForwardBlocker:
    """Probe the sparse structure decoder boundary when sparse-flow samples are available."""

    checkpoint_path = Path(root) / config.sparse_decoder_checkpoint_path
    try:
        decoder_config = read_sparse_structure_decoder_config(root, config.sparse_decoder_config_path)
        probe = probe_sparse_structure_decoder_boundary(
            checkpoint_path,
            decoder_config,
            sparse_latent=sparse_latent,
            target_resolution=sparse_structure_target_resolution(config.default_pipeline_type),
        )
    except (FileNotFoundError, ValueError) as error:
        return _blocker(
            "sparse-structure-decoding",
            "sparse structure decoder config/checkpoint probe",
            str(checkpoint_path),
            str(error),
            "place and map the sparse structure decoder config/checkpoint before sparse coordinate decoding",
        )

    if probe.coordinates is not None and probe.coordinates_shape is not None and probe.coordinates_shape[0] > 0:
        shape_output, shape_blocker = dispatch_shape_slat_sampling(
            root,
            config,
            sparse_coordinates=probe.coordinates,
            conditioning=conditioning,
        )
        if shape_output is not None:
            return dispatch_texture_slat_boundary(
                root,
                config,
                shape_slat_coordinates=probe.coordinates,
                shape_slat_features=shape_output.payload,
                conditioning=conditioning,
                decoder_token_limit=decoder_token_limit,
            )
        if shape_blocker is not None:
            return shape_blocker
        return _blocker(
            "shape-slat-sampling",
            "shape SLat sampler handoff",
            config.shape_slat_512_config_path,
            "sparse decoder produced coordinates but shape SLat sampling returned no output or blocker",
            "map shape SLat sampling output before texture SLat sampling",
        )

    return _blocker(
        "sparse-structure-decoding",
        probe.blocker_operation,
        probe.checkpoint_path,
        (
            f"decoder tensors {probe.loaded_tensor_names} validate against {config.sparse_decoder_config_path}; "
            f"{probe.blocker_detail}"
        ),
        "produce non-empty sparse structure coordinates before shape SLat sampling",
    )


def dispatch_shape_slat_boundary(
    root: str | Path,
    config: Trellis2ConditioningConfig,
    *,
    sparse_coordinates: mx.array | None = None,
) -> Trellis2ForwardBlocker:
    """Probe shape SLat sampling route and first MLX boundary."""

    _, blocker = dispatch_shape_slat_sampling(root, config, sparse_coordinates=sparse_coordinates)
    if blocker is None:
        return _blocker(
            "texture-slat-sampling",
            "shape SLat texture handoff",
            config.texture_slat_512_config_path,
            "shape SLat sampling completed but no downstream texture blocker was produced",
            "map texture SLat sampling after shape SLat sampling",
        )
    return blocker


def dispatch_shape_slat_sampling(
    root: str | Path,
    config: Trellis2ConditioningConfig,
    *,
    sparse_coordinates: mx.array | None = None,
    conditioning: mx.array | None = None,
) -> tuple[Trellis2StageOutput | None, Trellis2ForwardBlocker | None]:
    """Run the shape-SLat sampling probe and return its output plus the next blocker."""

    try:
        route = select_shape_slat_route(config.default_pipeline_type)
    except ValueError as error:
        return (
            None,
            _blocker(
                "shape-slat-sampling",
                "shape SLat pipeline route selection",
                config.default_pipeline_type,
                str(error),
                "select a supported TRELLIS.2 pipeline type before shape SLat sampling",
            ),
        )

    model_paths = _shape_slat_model_paths(config, route.model_keys[0])
    if sparse_coordinates is None:
        return (
            None,
            _blocker(
                "shape-slat-sampling",
                "shape SLat upstream sparse coordinate availability",
                model_paths[1],
                (
                    f"pipeline_type={route.pipeline_type} selects shape SLat model route {route.model_keys}, "
                    "but no sparse coordinates are available from sparse structure decoding"
                ),
                "produce sparse structure coordinates before shape SLat sampling",
            ),
        )
    checkpoint_path = Path(root) / model_paths[1]
    try:
        slat_config = read_slat_flow_config(root, model_paths[0])
        probe = probe_shape_slat_forward_boundary(
            checkpoint_path,
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
    except (FileNotFoundError, ValueError) as error:
        return (
            None,
            _blocker(
                "shape-slat-sampling",
                "shape SLat config/checkpoint forward probe",
                str(checkpoint_path),
                str(error),
                "map the shape SLat flow config and checkpoint tensors before SLat sampler execution",
            ),
        )

    cascade_detail = (
        f"; cascade route will also require {route.model_keys[1]} and shape decoder upsample coordinates"
        if route.cascade and len(route.model_keys) > 1
        else ""
    )
    if probe.sampled_features is None or probe.sampled_feature_shape is None:
        return (
            None,
            _blocker(
                "shape-slat-sampling",
                probe.blocker_operation,
                probe.checkpoint_path,
                (
                    f"pipeline_type={route.pipeline_type} selects route {route.model_keys}, "
                    f"sparse coordinate shape {probe.coordinate_shape}, feature shape {probe.feature_shape}, "
                    f"loads tensors {probe.loaded_tensor_names}, and {probe.blocker_detail}{cascade_detail}"
                ),
                "produce shape_slat features before texture SLat sampling",
            ),
        )

    sampled = _apply_slat_normalization(probe.sampled_features, config.shape_slat_normalization, name="shape_slat")
    output = Trellis2StageOutput(
        stage="shape-slat-sampling",
        name="shape_slat",
        shape=tuple(int(dim) for dim in sampled.shape),
        dtype=str(sampled.dtype).removeprefix("mlx.core."),
        detail=(
            f"MLX shape SLat FlowEuler sampler output after {config.shape_slat_sampler.steps} steps; "
            f"{probe.blocker_detail}{cascade_detail}"
        ),
        payload=sampled,
    )
    return output, None


def _apply_slat_normalization(features: mx.array, normalization: Trellis2NormalizationConfig, *, name: str) -> mx.array:
    feature_width = int(features.shape[-1])
    if len(normalization.mean) != feature_width or len(normalization.std) != feature_width:
        raise ValueError(
            f"{name} normalization width mismatch: expected {feature_width}, "
            f"got mean={len(normalization.mean)} std={len(normalization.std)}"
        )
    mean = mx.array(normalization.mean, dtype=mx.float32)[None, :]
    std = mx.array(normalization.std, dtype=mx.float32)[None, :]
    return features.astype(mx.float32) * std + mean


def _shape_slat_model_paths(config: Trellis2ConditioningConfig, model_key: str) -> tuple[str, str]:
    if model_key == "shape_slat_flow_model_512":
        return config.shape_slat_512_config_path, config.shape_slat_512_checkpoint_path
    if model_key == "shape_slat_flow_model_1024":
        return config.shape_slat_1024_config_path, config.shape_slat_1024_checkpoint_path
    raise ValueError(f"unsupported shape SLat model key: {model_key}")


def dispatch_texture_slat_boundary(
    root: str | Path,
    config: Trellis2ConditioningConfig,
    *,
    shape_slat_coordinates: mx.array | None = None,
    shape_slat_features: mx.array | None = None,
    conditioning: mx.array | None = None,
    decoder_token_limit: int | None = None,
) -> Trellis2ForwardBlocker:
    """Probe texture SLat sampling route and first MLX boundary."""

    texture_output, blocker = dispatch_texture_slat_sampling(
        root,
        config,
        shape_slat_coordinates=shape_slat_coordinates,
        shape_slat_features=shape_slat_features,
        conditioning=conditioning,
    )
    if texture_output is None:
        return blocker or _blocker(
            "texture-slat-sampling",
            "texture SLat sampler handoff",
            config.texture_slat_1024_config_path,
            "texture SLat sampling returned no output or blocker",
            "map texture SLat sampling output before latent decoding",
        )
    return dispatch_decode_latents_boundary(
        root,
        config,
        shape_slat_coordinates=shape_slat_coordinates,
        shape_slat_features=shape_slat_features,
        texture_slat_coordinates=shape_slat_coordinates,
        texture_slat_features=texture_output.payload,
        decoder_token_limit=decoder_token_limit,
    )


def dispatch_texture_slat_sampling(
    root: str | Path,
    config: Trellis2ConditioningConfig,
    *,
    shape_slat_coordinates: mx.array | None = None,
    shape_slat_features: mx.array | None = None,
    conditioning: mx.array | None = None,
) -> tuple[Trellis2StageOutput | None, Trellis2ForwardBlocker | None]:
    """Run the texture-SLat sampling probe and return its output plus the next blocker."""

    try:
        route = select_texture_slat_route(config.default_pipeline_type)
    except ValueError as error:
        return (
            None,
            _blocker(
                "texture-slat-sampling",
                "texture SLat pipeline route selection",
                config.default_pipeline_type,
                str(error),
                "select a supported TRELLIS.2 pipeline type before texture SLat sampling",
            ),
        )

    model_paths = _texture_slat_model_paths(config, route.model_key)
    if shape_slat_coordinates is None or shape_slat_features is None:
        return (
            None,
            _blocker(
                "texture-slat-sampling",
                "texture SLat upstream shape_slat availability",
                model_paths[1],
                (
                    f"pipeline_type={route.pipeline_type} selects texture SLat model {route.model_key}, "
                    "but no shape_slat coordinates/features are available from shape SLat sampling"
                ),
                "produce shape_slat before texture SLat sampling",
            ),
        )

    checkpoint_path = Path(root) / model_paths[1]
    try:
        slat_config = read_slat_flow_config(root, model_paths[0])
        probe = probe_texture_slat_forward_boundary(
            checkpoint_path,
            slat_config,
            shape_slat_coordinates,
            shape_slat_features,
            conditioning=conditioning,
            steps=config.texture_slat_sampler.steps,
            rescale_t=config.texture_slat_sampler.rescale_t,
            guidance_strength=config.texture_slat_sampler.guidance_strength,
            guidance_rescale=config.texture_slat_sampler.guidance_rescale,
            guidance_interval=config.texture_slat_sampler.guidance_interval,
            sigma_min=config.texture_slat_sampler.sigma_min,
        )
    except (FileNotFoundError, ValueError) as error:
        return (
            None,
            _blocker(
                "texture-slat-sampling",
                "texture SLat config/checkpoint forward probe",
                str(checkpoint_path),
                str(error),
                "map the texture SLat flow config and checkpoint tensors before texture sampler execution",
            ),
        )

    if probe.sampled_features is None or probe.sampled_feature_shape is None:
        return (
            None,
            _blocker(
                "texture-slat-sampling",
                probe.blocker_operation,
                probe.checkpoint_path,
                (
                    f"pipeline_type={route.pipeline_type} selects model {route.model_key}, "
                    f"shape_slat coordinate shape {probe.coordinate_shape}, shape feature shape {probe.shape_feature_shape}, "
                    f"noise feature shape {probe.noise_feature_shape}, concat feature shape {probe.concat_feature_shape}, "
                    f"loads tensors {probe.loaded_tensor_names}, and {probe.blocker_detail}"
                ),
                "produce texture_slat features before latent decoding",
            ),
        )

    sampled = _apply_slat_normalization(probe.sampled_features, config.texture_slat_normalization, name="texture_slat")
    output = Trellis2StageOutput(
        stage="texture-slat-sampling",
        name="texture_slat",
        shape=tuple(int(dim) for dim in sampled.shape),
        dtype=str(sampled.dtype).removeprefix("mlx.core."),
        detail=(
            f"MLX texture SLat FlowEuler sampler output after {config.texture_slat_sampler.steps} steps; "
            f"{probe.blocker_detail}"
        ),
        payload=sampled,
    )
    return output, None


def _texture_slat_model_paths(config: Trellis2ConditioningConfig, model_key: str) -> tuple[str, str]:
    if model_key == "tex_slat_flow_model_512":
        return config.texture_slat_512_config_path, config.texture_slat_512_checkpoint_path
    if model_key == "tex_slat_flow_model_1024":
        return config.texture_slat_1024_config_path, config.texture_slat_1024_checkpoint_path
    raise ValueError(f"unsupported texture SLat model key: {model_key}")


def dispatch_decode_latents_boundary(
    root: str | Path,
    config: Trellis2ConditioningConfig,
    *,
    shape_slat_coordinates: mx.array | None = None,
    shape_slat_features: mx.array | None = None,
    texture_slat_coordinates: mx.array | None = None,
    texture_slat_features: mx.array | None = None,
    resolution: int | None = None,
    decoder_token_limit: int | None = None,
) -> Trellis2ForwardBlocker:
    """Probe combined TRELLIS.2 shape/texture latent decode boundaries."""

    if shape_slat_coordinates is None or shape_slat_features is None:
        return _blocker(
            "latent-decoding",
            "decode_latent upstream shape_slat availability",
            config.shape_decoder_checkpoint_path,
            "decode_latent requires shape_slat coordinates/features from shape SLat sampling before shape decoder execution",
            "produce shape_slat before latent decoding",
        )
    if texture_slat_coordinates is None or texture_slat_features is None:
        return _blocker(
            "latent-decoding",
            "decode_latent upstream texture_slat availability",
            config.texture_decoder_checkpoint_path,
            "decode_latent requires texture_slat coordinates/features from texture SLat sampling before texture decoder execution",
            "produce texture_slat before latent decoding",
        )

    shape_checkpoint_path = Path(root) / config.shape_decoder_checkpoint_path
    texture_checkpoint_path = Path(root) / config.texture_decoder_checkpoint_path
    try:
        shape_decoder_config = read_structured_latent_decoder_config(root, config.shape_decoder_config_path)
        texture_decoder_config = read_structured_latent_decoder_config(root, config.texture_decoder_config_path)
        probe = probe_decode_latents_boundary(
            shape_checkpoint_path,
            shape_decoder_config,
            texture_checkpoint_path,
            texture_decoder_config,
            shape_slat_coordinates=shape_slat_coordinates,
            shape_slat_features=shape_slat_features,
            texture_slat_coordinates=texture_slat_coordinates,
            texture_slat_features=texture_slat_features,
            resolution=resolution or _decode_resolution(config.default_pipeline_type),
            reference_token_limit=decoder_token_limit or STRUCTURED_LATENT_DECODER_REFERENCE_TOKEN_LIMIT,
        )
    except (FileNotFoundError, ValueError) as error:
        return _blocker(
            "latent-decoding",
            "shape/texture latent decoder config/checkpoint probe",
            f"{shape_checkpoint_path}; {texture_checkpoint_path}",
            str(error),
            "map the shape and texture decoder configs/checkpoints before latent decode execution",
        )

    return _blocker(
        "latent-decoding",
        probe.blocker_operation,
        f"{probe.shape_probe.checkpoint_path}; {probe.texture_probe.checkpoint_path}",
        (
            f"decode resolution={probe.resolution}, shape_slat coordinate shape {probe.shape_probe.coordinate_shape}, "
            f"texture_slat coordinate shape {probe.texture_probe.coordinate_shape}, "
            f"shape decoder tensors {probe.shape_probe.loaded_tensor_names}, "
            f"texture decoder tensors {probe.texture_probe.loaded_tensor_names}, and {probe.blocker_detail}"
        ),
        "optimize large-token MLX sparse latent decoders, then implement FlexiDualGrid mesh extraction and export",
    )


def _decode_resolution(pipeline_type: str) -> int:
    if pipeline_type == "512":
        return 512
    if pipeline_type in {"1024", "1024_cascade"}:
        return 1024
    if pipeline_type == "1536_cascade":
        return 1536
    raise ValueError(f"unsupported decode pipeline type: {pipeline_type}")


def sparse_structure_target_resolution(pipeline_type: str) -> int:
    if pipeline_type == "512":
        return 32
    if pipeline_type == "1024":
        return 64
    if pipeline_type in {"1024_cascade", "1536_cascade"}:
        return 32
    raise ValueError(f"unsupported TRELLIS.2 pipeline type: {pipeline_type}")


def default_dinov3_root(trellis_root: str | Path, model_name: str) -> Path:
    """Return the local convention for explicitly downloaded DINOv3 assets."""

    return Path(trellis_root).parent / model_name.rsplit("/", 1)[-1]


def _dino_blocker_reason(reason: str, model_name: str, model_root: Path) -> str:
    if str(model_root) in reason and model_name in reason:
        return reason
    return f"{model_name} assets at {model_root}: {reason}"


def _blocker(
    stage: str,
    operation: str,
    reference: str,
    reason: str,
    next_slice: str,
) -> Trellis2ForwardBlocker:
    return Trellis2ForwardBlocker(
        stage=stage,
        operation=operation,
        reference=reference,
        reason=reason,
        next_slice=next_slice,
    )
