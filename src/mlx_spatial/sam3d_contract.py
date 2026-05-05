"""SAM 3D Objects source and weight contract audit for the MLX port."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from .checkpoint import inspect_checkpoint
from .sam3d_assets import SAM3D_OBJECTS_MLX_DEFAULT_ROOT, inspect_sam3d_model_assets


@dataclass(frozen=True)
class Sam3dContractIssue:
    code: str
    role: str
    reference: str
    reason: str
    metadata: Mapping[str, object]


@dataclass(frozen=True)
class Sam3dTargetMapping:
    role: str
    config_path: Path
    target_path: str
    target: str
    planned_mlx_module: str


@dataclass(frozen=True)
class Sam3dPrefixMapping:
    role: str
    checkpoint_path: Path
    prefix: str
    tensor_count: int
    planned_mlx_module: str


@dataclass(frozen=True)
class Sam3dComponentMapping:
    component: str
    role: str
    config_path: Path
    checkpoint_path: Path
    target_path: str
    target: str
    required_prefixes: tuple[str, ...]
    planned_mlx_module: str


@dataclass(frozen=True)
class Sam3dContractAudit:
    root: Path
    target_mappings: tuple[Sam3dTargetMapping, ...]
    prefix_mappings: tuple[Sam3dPrefixMapping, ...]
    component_mappings: tuple[Sam3dComponentMapping, ...]
    issues: tuple[Sam3dContractIssue, ...]

    @property
    def ready(self) -> bool:
        return not self.issues


@dataclass(frozen=True)
class _ComponentContract:
    component: str
    role: str
    target_path: tuple[str, ...]
    targets: tuple[str, ...]
    prefixes: tuple[str, ...]
    planned_mlx_module: str


SAM3D_REQUIRED_CONTRACT_COMPONENTS = (
    "ss_condition",
    "ss_generator",
    "ss_decoder",
    "slat_condition",
    "slat_generator",
    "gs_decoder",
    "mesh_decoder",
)

_PREPROCESS_MODULE = "mlx_spatial.sam3d_preprocess"
_MOGE_MODULE = "mlx_spatial.sam3d_moge"
_SS_CONDITION_MODULE = "mlx_spatial.sam3d_condition.Sam3dSparseStructureCondition"
_SS_GENERATOR_MODULE = "mlx_spatial.sam3d_sparse_structure.Sam3dSparseStructureGenerator"
_SS_DECODER_MODULE = "mlx_spatial.sam3d_ss.Sam3dSparseStructureDecoder"
_SLAT_CONDITION_MODULE = "mlx_spatial.sam3d_condition.Sam3dStructuredLatentCondition"
_SLAT_GENERATOR_MODULE = "mlx_spatial.sam3d_slat.Sam3dStructuredLatentGenerator"
_GS_DECODER_MODULE = "mlx_spatial.sam3d_slat.Sam3dGaussianDecoder"
_MESH_DECODER_MODULE = "mlx_spatial.sam3d_slat.Sam3dMeshDecoder"

SAM3D_TARGET_TO_MLX_MODULE: Mapping[str, str] = {
    "sam3d_objects.pipeline.inference_pipeline_pointmap.InferencePipelinePointMap": "mlx_spatial.sam3d_inference.Sam3dInferencePipeline",
    "sam3d_objects.pipeline.depth_models.moge.MoGe": _MOGE_MODULE,
    "moge.model.v1.MoGeModel.from_pretrained": _MOGE_MODULE,
    "sam3d_objects.data.dataset.tdfy.preprocessor.PreProcessor": _PREPROCESS_MODULE,
    "sam3d_objects.data.dataset.tdfy.img_and_mask_transforms.resize_all_to_same_size": _PREPROCESS_MODULE,
    "sam3d_objects.data.dataset.tdfy.img_and_mask_transforms.crop_around_mask_with_padding": _PREPROCESS_MODULE,
    "sam3d_objects.data.dataset.tdfy.img_and_mask_transforms.ObjectCentricSSI": _PREPROCESS_MODULE,
    "sam3d_objects.data.dataset.tdfy.img_processing.pad_to_square_centered": _PREPROCESS_MODULE,
    "torchvision.transforms.Compose": _PREPROCESS_MODULE,
    "torchvision.transforms.Resize": _PREPROCESS_MODULE,
    "sam3d_objects.model.backbone.dit.embedder.embedder_fuser.EmbedderFuser": _SS_CONDITION_MODULE,
    "sam3d_objects.model.backbone.dit.embedder.dino.Dino": _SS_CONDITION_MODULE,
    "sam3d_objects.model.backbone.dit.embedder.pointmap.PointPatchEmbed": _SS_CONDITION_MODULE,
    "sam3d_objects.model.backbone.generator.shortcut.model.ShortCut": _SS_GENERATOR_MODULE,
    "sam3d_objects.model.backbone.generator.classifier_free_guidance.ClassifierFreeGuidanceWithExternalUnconditionalProbability": _SS_GENERATOR_MODULE,
    "sam3d_objects.model.backbone.tdfy_dit.models.mot_sparse_structure_flow.SparseStructureFlowTdfyWrapper": _SS_GENERATOR_MODULE,
    "sam3d_objects.model.backbone.tdfy_dit.models.sparse_structure_vae.SparseStructureDecoderTdfyWrapper": _SS_DECODER_MODULE,
    "sam3d_objects.model.backbone.generator.flow_matching.model.FlowMatching": _SLAT_GENERATOR_MODULE,
    "sam3d_objects.model.backbone.generator.flow_matching.model.lognorm_sampler": _SLAT_GENERATOR_MODULE,
    "sam3d_objects.model.backbone.generator.classifier_free_guidance.ClassifierFreeGuidance": _SLAT_GENERATOR_MODULE,
    "sam3d_objects.model.backbone.tdfy_dit.models.structured_latent_flow.SLatFlowModelTdfyWrapper": _SLAT_GENERATOR_MODULE,
    "sam3d_objects.model.backbone.tdfy_dit.models.mm_latent.Latent": _SS_GENERATOR_MODULE,
    "sam3d_objects.model.backbone.tdfy_dit.models.mm_latent.LearntPositionEmbedder": _SS_GENERATOR_MODULE,
    "sam3d_objects.model.backbone.tdfy_dit.models.mm_latent.ShapePositionEmbedder": _SS_GENERATOR_MODULE,
    "sam3d_objects.config.utils.make_dict": _SS_GENERATOR_MODULE,
    "sam3d_objects.model.backbone.tdfy_dit.models.structured_latent_vae.decoder_gs.SLatGaussianDecoderTdfyWrapper": _GS_DECODER_MODULE,
    "sam3d_objects.model.backbone.tdfy_dit.models.structured_latent_vae.decoder_mesh.SLatMeshDecoderTdfyWrapper": _MESH_DECODER_MODULE,
}

_COMPONENT_CONTRACTS = (
    _ComponentContract(
        component="ss_condition",
        role="ss_generator",
        target_path=("module", "condition_embedder", "backbone"),
        targets=("sam3d_objects.model.backbone.dit.embedder.embedder_fuser.EmbedderFuser",),
        prefixes=("_base_models.condition_embedder.",),
        planned_mlx_module=_SS_CONDITION_MODULE,
    ),
    _ComponentContract(
        component="ss_generator",
        role="ss_generator",
        target_path=("module", "generator", "backbone", "reverse_fn", "backbone"),
        targets=("sam3d_objects.model.backbone.tdfy_dit.models.mot_sparse_structure_flow.SparseStructureFlowTdfyWrapper",),
        prefixes=("_base_models.generator.",),
        planned_mlx_module=_SS_GENERATOR_MODULE,
    ),
    _ComponentContract(
        component="ss_decoder",
        role="ss_decoder",
        target_path=(),
        targets=("sam3d_objects.model.backbone.tdfy_dit.models.sparse_structure_vae.SparseStructureDecoderTdfyWrapper",),
        prefixes=("input_layer.", "blocks.", "middle_block.", "out_layer."),
        planned_mlx_module=_SS_DECODER_MODULE,
    ),
    _ComponentContract(
        component="slat_condition",
        role="slat_generator",
        target_path=("module", "condition_embedder", "backbone"),
        targets=("sam3d_objects.model.backbone.dit.embedder.embedder_fuser.EmbedderFuser",),
        prefixes=("_base_models.condition_embedder.",),
        planned_mlx_module=_SLAT_CONDITION_MODULE,
    ),
    _ComponentContract(
        component="slat_generator",
        role="slat_generator",
        target_path=("module", "generator", "backbone", "reverse_fn", "backbone"),
        targets=("sam3d_objects.model.backbone.tdfy_dit.models.structured_latent_flow.SLatFlowModelTdfyWrapper",),
        prefixes=("_base_models.generator.",),
        planned_mlx_module=_SLAT_GENERATOR_MODULE,
    ),
    _ComponentContract(
        component="gs_decoder",
        role="slat_decoder_gs",
        target_path=(),
        targets=("sam3d_objects.model.backbone.tdfy_dit.models.structured_latent_vae.decoder_gs.SLatGaussianDecoderTdfyWrapper",),
        prefixes=("input_layer.", "blocks.", "out_layer.", "offset_perturbation"),
        planned_mlx_module=_GS_DECODER_MODULE,
    ),
    _ComponentContract(
        component="mesh_decoder",
        role="slat_decoder_mesh",
        target_path=(),
        targets=("sam3d_objects.model.backbone.tdfy_dit.models.structured_latent_vae.decoder_mesh.SLatMeshDecoderTdfyWrapper",),
        prefixes=("input_layer.", "blocks.", "out_layer.", "upsample."),
        planned_mlx_module=_MESH_DECODER_MODULE,
    ),
)


def audit_sam3d_source_weight_contract(
    root: str | Path = SAM3D_OBJECTS_MLX_DEFAULT_ROOT,
) -> Sam3dContractAudit:
    """Audit active SAM3D YAML targets and checkpoint prefixes without model runtime."""

    root_path = Path(root)
    inspection = inspect_sam3d_model_assets(root_path)
    issues: list[Sam3dContractIssue] = []
    target_mappings: list[Sam3dTargetMapping] = []
    prefix_mappings: list[Sam3dPrefixMapping] = []
    component_mappings: list[Sam3dComponentMapping] = []

    if inspection.blocker is not None:
        issue = Sam3dContractIssue(
            code="asset-inspection-blocker",
            role=inspection.blocker.stage,
            reference=str(root_path),
            reason=inspection.blocker.reason,
            metadata=inspection.blocker.metadata,
        )
        return Sam3dContractAudit(root_path, (), (), (), (issue,))

    config_paths = _active_config_paths(inspection)
    loaded_configs: dict[str, Mapping[str, object]] = {}
    for role, config_path in config_paths.items():
        try:
            config = _read_yaml_mapping(config_path)
        except ValueError as error:
            issues.append(
                Sam3dContractIssue(
                    code="invalid-active-config",
                    role=role,
                    reference=str(config_path),
                    reason=str(error),
                    metadata={},
                )
            )
            continue
        loaded_configs[role] = config
        for target_path, target in _iter_targets(config):
            planned_module = SAM3D_TARGET_TO_MLX_MODULE.get(target)
            if planned_module is None:
                issues.append(
                    Sam3dContractIssue(
                        code="unmapped-active-target",
                        role=role,
                        reference=f"{config_path}:{target_path}",
                        reason="active SAM3D config target is not mapped to a planned MLX module",
                        metadata={"target": target, "target_path": target_path},
                    )
                )
                continue
            target_mappings.append(
                Sam3dTargetMapping(
                    role=role,
                    config_path=config_path,
                    target_path=target_path,
                    target=target,
                    planned_mlx_module=planned_module,
                )
            )

    checkpoint_paths = {
        item.role: item.path
        for item in inspection.paths
        if item.kind == "checkpoint" and item.exists
    }
    for contract in _COMPONENT_CONTRACTS:
        config_path = config_paths.get(contract.role)
        checkpoint_path = checkpoint_paths.get(contract.role)
        config = loaded_configs.get(contract.role)
        if config_path is None or config is None:
            issues.append(_missing_component_issue(contract, "config"))
            continue
        if checkpoint_path is None:
            issues.append(_missing_component_issue(contract, "checkpoint"))
            continue

        target = _target_at(config, contract.target_path)
        target_ref = _format_target_path(contract.target_path)
        if target not in contract.targets:
            issues.append(
                Sam3dContractIssue(
                    code="unexpected-component-target",
                    role=contract.role,
                    reference=f"{config_path}:{target_ref}",
                    reason=f"{contract.component} target is not mapped to the expected SAM3D source module",
                    metadata={
                        "component": contract.component,
                        "target": target,
                        "expected_targets": contract.targets,
                    },
                )
            )
            continue

        component_mappings.append(
            Sam3dComponentMapping(
                component=contract.component,
                role=contract.role,
                config_path=config_path,
                checkpoint_path=checkpoint_path,
                target_path=target_ref,
                target=target,
                required_prefixes=contract.prefixes,
                planned_mlx_module=contract.planned_mlx_module,
            )
        )
        for prefix in contract.prefixes:
            try:
                infos = inspect_checkpoint(checkpoint_path, prefixes=(prefix,))
            except ValueError:
                issues.append(
                    Sam3dContractIssue(
                        code="missing-required-prefix",
                        role=contract.role,
                        reference=f"{checkpoint_path}:{prefix}",
                        reason="checkpoint is missing a required active prefix for a planned MLX module",
                        metadata={
                            "component": contract.component,
                            "prefix": prefix,
                            "planned_mlx_module": contract.planned_mlx_module,
                        },
                    )
                )
                continue
            prefix_mappings.append(
                Sam3dPrefixMapping(
                    role=contract.role,
                    checkpoint_path=checkpoint_path,
                    prefix=prefix,
                    tensor_count=len(infos),
                    planned_mlx_module=contract.planned_mlx_module,
                )
            )

    mapped_components = {item.component for item in component_mappings}
    for component in SAM3D_REQUIRED_CONTRACT_COMPONENTS:
        if component not in mapped_components:
            issues.append(
                Sam3dContractIssue(
                    code="missing-component-contract",
                    role=component,
                    reference=str(root_path),
                    reason="required SAM3D source/weight component is not fully mapped",
                    metadata={"component": component},
                )
            )

    return Sam3dContractAudit(
        root=root_path,
        target_mappings=tuple(target_mappings),
        prefix_mappings=tuple(prefix_mappings),
        component_mappings=tuple(component_mappings),
        issues=tuple(issues),
    )


def _active_config_paths(inspection) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    if inspection.validation.pipeline_path is not None:
        paths["pipeline"] = inspection.validation.pipeline_path
    for item in inspection.paths:
        if item.kind == "config" and item.exists and (item.required or item.role == "slat_decoder_gs_4"):
            paths[item.role] = item.path
    return paths


def _read_yaml_mapping(path: Path) -> Mapping[str, object]:
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError as error:  # pragma: no cover - pyyaml is a runtime dependency.
        raise ValueError("pyyaml is required to audit SAM3D configs") from error

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError(f"SAM3D config must be a YAML mapping: {path}")
    return raw


def _iter_targets(value: object, path: tuple[str, ...] = ()) -> tuple[tuple[str, str], ...]:
    targets: list[tuple[str, str]] = []
    if isinstance(value, Mapping):
        target = value.get("_target_")
        if isinstance(target, str):
            targets.append((_format_target_path(path), target))
        for key, item in value.items():
            targets.extend(_iter_targets(item, path + (str(key),)))
    elif isinstance(value, Sequence) and not isinstance(value, str | bytes):
        for index, item in enumerate(value):
            targets.extend(_iter_targets(item, path + (str(index),)))
    return tuple(targets)


def _target_at(config: Mapping[str, object], path: tuple[str, ...]) -> str | None:
    value: object = config
    for part in path:
        if not isinstance(value, Mapping):
            return None
        value = value.get(part)
    if isinstance(value, Mapping) and isinstance(value.get("_target_"), str):
        return value["_target_"]
    return None


def _format_target_path(path: tuple[str, ...]) -> str:
    return "/".join(path) if path else "<root>"


def _missing_component_issue(contract: _ComponentContract, kind: str) -> Sam3dContractIssue:
    return Sam3dContractIssue(
        code=f"missing-component-{kind}",
        role=contract.role,
        reference=contract.component,
        reason=f"{contract.component} requires an active {kind} path in pipeline.yaml",
        metadata={"component": contract.component, "kind": kind},
    )
