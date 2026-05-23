import json
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from PIL import Image
from safetensors.mlx import save_file

from mlx_spatial.sam3d import main
from mlx_spatial.sam3d_inference import (
    SAM3D_DEFAULT_SLAT_CFG_STRENGTH,
    SAM3D_DEFAULT_SLAT_RESCALE_T,
    _preprocess_metadata,
    load_sam3d_external_pointmap,
    sam3d_quality_issue_records,
)
from mlx_spatial.sam3d_preprocess import preprocess_sam3d_image_mask
from mlx_spatial.sam3d_condition import Sam3dConditionStackOutput
from mlx_spatial.sam3d_decoder import Sam3dMeshDecoderConfig, Sam3dSLatDecoderConfig
from mlx_spatial.sam3d_mesh import Sam3dMeshDecoderFeatureResult
from mlx_spatial.sam3d_moge import SAM3D_MOGE_REQUIRED_KEYS, Sam3dMogePointmap, Sam3dMogeResult
from mlx_spatial.sam3d_slat import Sam3dSLatFlowOutput
from mlx_spatial.sam3d_ss import Sam3dSSDecoderOutput
from mlx_spatial.sam3d_ss_flow import Sam3dSSFlowOutput


def _write_sam3d_fixture(root):
    root.mkdir(parents=True, exist_ok=True)
    (root / "configs").mkdir()
    (root / "ckpts").mkdir()
    (root / "pipeline.yaml").write_text(
        """
_target_: sam3d_objects.pipeline.inference_pipeline_pointmap.InferencePipelinePointMap
dtype: bfloat16
decode_formats: [gaussian, mesh]
ss_generator_config_path: configs/ss_generator.yaml
ss_generator_ckpt_path: ckpts/ss_generator.safetensors
slat_generator_config_path: configs/slat_generator.yaml
slat_generator_ckpt_path: ckpts/slat_generator.safetensors
ss_decoder_config_path: configs/ss_decoder.yaml
ss_decoder_ckpt_path: ckpts/ss_decoder.safetensors
slat_decoder_gs_config_path: configs/slat_decoder_gs.yaml
slat_decoder_gs_ckpt_path: ckpts/slat_decoder_gs.safetensors
slat_decoder_mesh_config_path: configs/slat_decoder_mesh.yaml
slat_decoder_mesh_ckpt_path: ckpts/slat_decoder_mesh.safetensors
depth_model:
  _target_: sam3d_objects.pipeline.depth_models.moge.MoGe
""".strip(),
        encoding="utf-8",
    )
    for name in (
        "ss_generator",
        "slat_generator",
        "ss_decoder",
        "slat_decoder_gs",
        "slat_decoder_mesh",
    ):
        (root / "configs" / f"{name}.yaml").write_text("_target_: fixture\n", encoding="utf-8")
        save_file({f"{name}.weight": mx.array([1.0], dtype=mx.float32)}, root / "ckpts" / f"{name}.safetensors")


def _write_image_and_mask(root):
    image = root / "image.png"
    mask = root / "mask.png"
    Image.fromarray(np.full((2, 2, 3), 127, dtype=np.uint8), mode="RGB").save(image)
    Image.fromarray(np.array([[0, 255], [255, 0]], dtype=np.uint8), mode="L").save(mask)
    return image, mask


def _write_moge_fixture(root):
    root.mkdir(parents=True, exist_ok=True)
    save_file(
        {name: mx.array([1.0], dtype=mx.float32) for name in SAM3D_MOGE_REQUIRED_KEYS},
        root / "model.safetensors",
    )


def _fixture_pointmap():
    return np.array(
        [
            [[-0.1, -0.1, 1.0], [0.1, -0.1, 1.0]],
            [[-0.1, 0.1, 1.2], [0.1, 0.1, 1.2]],
        ],
        dtype=np.float32,
    )


def test_sam3d_pointmap_loader_accepts_npy_and_npz(tmp_path):
    pointmap = _fixture_pointmap()
    npy_path = tmp_path / "pointmap.npy"
    npz_path = tmp_path / "pointmap.npz"
    np.save(npy_path, pointmap)
    np.savez(npz_path, pointmap=pointmap, ignored=np.zeros((1,), dtype=np.float32))

    loaded_npy = load_sam3d_external_pointmap(npy_path, expected_image_shape=(2, 2))
    loaded_npz = load_sam3d_external_pointmap(npz_path, expected_image_shape=(2, 2))

    np.testing.assert_allclose(loaded_npy.pointmap, pointmap)
    np.testing.assert_allclose(loaded_npz.pointmap, pointmap)
    assert loaded_npy.metadata["file_format"] == "npy"
    assert loaded_npz.metadata["file_format"] == "npz"
    assert loaded_npy.metadata["original_shape"] == (2, 2, 3)
    assert loaded_npy.metadata["original_dtype"] == "float32"
    assert loaded_npy.metadata["finite_count"] == 12
    assert loaded_npy.metadata["nan_count"] == 0
    assert loaded_npy.metadata["source_numeric_match"] == "not_claimed"
    assert loaded_npy.metadata["source_clipping_match"] == "deferred"


def test_sam3d_preprocess_metadata_ignores_sample_sidecar(tmp_path):
    image, mask = _write_image_and_mask(tmp_path)
    (tmp_path / "sample.json").write_text('{"masks": []}', encoding="utf-8")

    metadata = _preprocess_metadata(preprocess_sam3d_image_mask(image, mask))

    assert metadata["mask_path"] == str(mask)
    assert "sample_mask" not in metadata


def test_sam3d_external_pointmap_validation_rejects_invalid_inputs(tmp_path):
    missing_key = tmp_path / "missing-key.npz"
    wrong_ndim = tmp_path / "wrong-ndim.npy"
    wrong_channels = tmp_path / "wrong-channels.npy"
    wrong_size = tmp_path / "wrong-size.npy"
    unsupported = tmp_path / "pointmap.txt"
    object_array = tmp_path / "object.npy"

    np.savez(missing_key, other=_fixture_pointmap())
    np.save(wrong_ndim, np.zeros((2, 2), dtype=np.float32))
    np.save(wrong_channels, np.zeros((2, 2, 2), dtype=np.float32))
    np.save(wrong_size, np.zeros((3, 2, 3), dtype=np.float32))
    unsupported.write_text("not a pointmap", encoding="utf-8")
    np.save(object_array, np.array([[[object(), object(), object()]]], dtype=object))

    cases = (
        (missing_key, "must contain a 'pointmap' array"),
        (wrong_ndim, "must have shape"),
        (wrong_channels, "exactly 3 channels"),
        (wrong_size, "spatial shape must match image shape"),
        (unsupported, ".npy or .npz"),
        (object_array, "Object arrays cannot be loaded"),
    )

    for path, message in cases:
        with pytest.raises(ValueError, match=message):
            load_sam3d_external_pointmap(path, expected_image_shape=(2, 2))


def _interior_mesh_features(*, origin: int = 0):
    coords = np.array(
        [
            [0, origin + 0, origin + 0, origin + 0],
            [0, origin + 0, origin + 0, origin + 1],
            [0, origin + 0, origin + 1, origin + 0],
            [0, origin + 0, origin + 1, origin + 1],
            [0, origin + 1, origin + 0, origin + 0],
            [0, origin + 1, origin + 0, origin + 1],
            [0, origin + 1, origin + 1, origin + 0],
            [0, origin + 1, origin + 1, origin + 1],
        ],
        dtype=np.int32,
    )
    feats = np.zeros((8, 101), dtype=np.float32)
    feats[:, :8] = 2.0
    for row, coord in enumerate(coords[:, 1:]):
        local_corner = tuple(1 - (coord - origin))
        corner_index = int(local_corner[0] + local_corner[1] * 2 + local_corner[2] * 4)
        feats[row, corner_index] = -2.0
    feats[:, 53:101] = 0.25
    return coords, feats


def test_sam3d_cli_validate_inspect_and_download_command(tmp_path, capsys):
    _write_sam3d_fixture(tmp_path)

    assert main(["validate", str(tmp_path)]) == 0
    validate_output = capsys.readouterr().out
    assert "ready=True" in validate_output
    assert "pipeline=" in validate_output

    assert main(["inspect", str(tmp_path)]) == 0
    inspect_output = capsys.readouterr().out
    assert "decode_formats=('gaussian', 'mesh')" in inspect_output
    assert "checkpoint slat_decoder_gs" in inspect_output

    assert main(["download-command", "weights/sam-3d-objects"]) == 0
    download_output = capsys.readouterr().out
    assert "SAM 3D Objects checkpoints are gated" in download_output
    assert "hf download" in download_output
    assert "facebook/sam-3d-objects" in download_output

    converted = tmp_path / "converted"
    assert main(["convert", str(tmp_path), "--output-root", str(converted)]) == 0
    convert_output = capsys.readouterr().out
    assert "output_root=" in convert_output
    assert "rewritten config pipeline" in convert_output
    assert "ready=True" in convert_output


def test_sam3d_cli_reconstruct_blocks_at_moge_without_fake_output(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    weights = tmp_path / "weights"
    moge = tmp_path / "moge"
    _write_sam3d_fixture(weights)
    _write_moge_fixture(moge)
    image, mask = _write_image_and_mask(tmp_path)
    output = "outputs/sam3d/gaussians.ply"
    glb_output = "outputs/sam3d/mesh.glb"
    trace_output = "outputs/sam3d/trace.json"

    assert main(
        [
            "reconstruct",
            str(weights),
            str(image),
            "--mask",
            str(mask),
            "--output",
            output,
            "--glb-output",
            glb_output,
            "--moge-root",
            str(moge),
            "--stage1-steps",
            "2",
            "--stage2-steps",
            "12",
            "--memory-profile",
            "balanced",
            "--trace-output",
            trace_output,
        ]
    ) == 2

    cli_output = capsys.readouterr().out
    assert "blocker_stage=moge-pointmap" in cli_output
    assert "validate full MoGe v1 tensor shapes" in cli_output
    assert not (tmp_path / output).exists()
    assert not (tmp_path / glb_output).exists()
    trace = json.loads((tmp_path / trace_output).read_text(encoding="utf-8"))
    assert trace["completed_stages"] == ["asset-validation", "pipeline-config", "image-mask-preprocessing"]
    assert trace["blocker"]["stage"] == "moge-pointmap"
    assert trace["glb_output_path"] == glb_output
    assert trace["metadata"]["memory_profile"] == "balanced"
    assert trace["metadata"]["stage1_steps"] == 2
    assert trace["metadata"]["stage2_steps"] == 12
    assert trace["metadata"]["moge"]["inspection"]["ready"] is True


def test_sam3d_cli_gaussian_only_does_not_require_mesh_decoder_assets(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    weights = tmp_path / "weights"
    moge = tmp_path / "moge"
    _write_sam3d_fixture(weights)
    for path in (
        weights / "configs" / "slat_decoder_mesh.yaml",
        weights / "ckpts" / "slat_decoder_mesh.safetensors",
    ):
        path.unlink()
    _write_moge_fixture(moge)
    image, mask = _write_image_and_mask(tmp_path)

    assert main(
        [
            "reconstruct",
            str(weights),
            str(image),
            "--mask",
            str(mask),
            "--output",
            "outputs/sam3d/gaussians.ply",
            "--moge-root",
            str(moge),
        ]
    ) == 2

    cli_output = capsys.readouterr().out
    assert "blocker_stage=moge-pointmap" in cli_output
    assert "pipeline.yaml references files" not in cli_output


def test_sam3d_cli_reconstruct_advances_past_moge_with_pointmap(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    weights = tmp_path / "weights"
    moge = tmp_path / "moge"
    _write_sam3d_fixture(weights)
    _write_moge_fixture(moge)
    image, mask = _write_image_and_mask(tmp_path)
    output = "outputs/sam3d/gaussians.ply"
    trace_output = "outputs/sam3d/trace.json"

    pointmap = _fixture_pointmap()

    def fake_moge(_image_rgb, *, root, memory_profile):
        return Sam3dMogeResult(
            pointmap=Sam3dMogePointmap(
                pointmap=pointmap,
                intrinsics=np.eye(3, dtype=np.float32),
                mask=np.ones((2, 2), dtype=bool),
                depth=pointmap[..., 2],
                metadata={"fixture": True, "memory_profile": memory_profile, "root": str(root)},
            )
        )

    import mlx_spatial.sam3d_inference as sam3d_inference

    monkeypatch.setattr(sam3d_inference, "run_sam3d_moge_pointmap", fake_moge)

    assert main(
        [
            "reconstruct",
            str(weights),
            str(image),
            "--mask",
            str(mask),
            "--output",
            output,
            "--moge-root",
            str(moge),
            "--trace-output",
            trace_output,
        ]
    ) == 2

    cli_output = capsys.readouterr().out
    assert "blocker_stage=sparse-structure" in cli_output
    trace = json.loads((tmp_path / trace_output).read_text(encoding="utf-8"))
    assert trace["completed_stages"] == [
        "asset-validation",
        "pipeline-config",
        "image-mask-preprocessing",
        "moge-pointmap",
        "official-preprocessing",
    ]
    assert trace["metadata"]["moge"]["pointmap"]["shape"] == [2, 2, 3]
    assert trace["metadata"]["moge"]["pointmap"]["metadata"]["fixture"] is True
    assert trace["metadata"]["official_preprocessing"]["pointmap_shape"] == [3, 518, 518]
    assert trace["metadata"]["stage1_steps"] == 25
    assert trace["metadata"]["stage2_steps"] == 25
    assert trace["metadata"]["ss_cfg_strength"] == 7.0
    assert trace["metadata"]["ss_rescale_t"] == 3.0
    assert trace["metadata"]["ss_cfg_interval"] == [0.0, 500.0]
    assert trace["metadata"]["slat_cfg_strength_requested"] is None
    assert trace["metadata"]["slat_rescale_t_requested"] is None
    assert trace["metadata"]["slat_cfg_interval"] == [0.0, 500.0]
    assert not (tmp_path / output).exists()


def test_sam3d_external_pointmap_cli_bypasses_moge_and_records_trace(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    weights = tmp_path / "weights"
    _write_sam3d_fixture(weights)
    image, mask = _write_image_and_mask(tmp_path)
    pointmap_path = tmp_path / "pointmap.npz"
    np.savez(pointmap_path, pointmap=_fixture_pointmap())
    output = "outputs/sam3d/gaussians.ply"
    trace_output = "outputs/sam3d/trace.json"

    import mlx_spatial.sam3d_inference as sam3d_inference

    def fail_moge(*_args, **_kwargs):
        raise AssertionError("external pointmap route should bypass MoGe")

    monkeypatch.setattr(sam3d_inference, "run_sam3d_moge_pointmap", fail_moge)

    assert main(
        [
            "reconstruct",
            str(weights),
            str(image),
            "--mask",
            str(mask),
            "--pointmap",
            str(pointmap_path),
            "--output",
            output,
            "--trace-output",
            trace_output,
        ]
    ) == 2

    cli_output = capsys.readouterr().out
    assert "blocker_stage=sparse-structure" in cli_output
    trace = json.loads((tmp_path / trace_output).read_text(encoding="utf-8"))
    assert trace["completed_stages"] == [
        "asset-validation",
        "pipeline-config",
        "image-mask-preprocessing",
        "external-pointmap",
        "official-preprocessing",
    ]
    assert trace["metadata"]["pointmap_source"] == "external"
    assert trace["metadata"]["moge"]["skipped"] is True
    assert trace["metadata"]["external_pointmap"]["file_format"] == "npz"
    assert trace["metadata"]["external_pointmap"]["original_shape"] == [2, 2, 3]
    assert trace["metadata"]["external_pointmap"]["finite_count"] == 12
    assert trace["metadata"]["external_pointmap"]["source_numeric_match"] == "not_claimed"
    assert trace["metadata"]["external_pointmap"]["source_clipping_match"] == "deferred"
    assert trace["metadata"]["official_preprocessing"]["pointmap_shape"] == [3, 518, 518]
    assert not (tmp_path / output).exists()


def test_sam3d_external_pointmap_cli_reports_validation_blocker(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    weights = tmp_path / "weights"
    _write_sam3d_fixture(weights)
    image, mask = _write_image_and_mask(tmp_path)
    bad_pointmap = tmp_path / "bad-pointmap.npy"
    np.save(bad_pointmap, np.zeros((3, 2, 3), dtype=np.float32))
    output = "outputs/sam3d/gaussians.ply"
    trace_output = "outputs/sam3d/trace.json"

    assert main(
        [
            "reconstruct",
            str(weights),
            str(image),
            "--mask",
            str(mask),
            "--pointmap",
            str(bad_pointmap),
            "--output",
            output,
            "--trace-output",
            trace_output,
        ]
    ) == 2

    cli_output = capsys.readouterr().out
    assert "blocker_stage=external-pointmap" in cli_output
    assert "spatial shape must match image shape" in cli_output
    trace = json.loads((tmp_path / trace_output).read_text(encoding="utf-8"))
    assert trace["completed_stages"] == ["asset-validation", "pipeline-config", "image-mask-preprocessing"]
    assert trace["blocker"]["stage"] == "external-pointmap"
    assert trace["metadata"]["pointmap_source"] == "external"
    assert not (tmp_path / output).exists()


def test_sam3d_cli_reconstruct_writes_gaussian_ply_and_textured_glb_with_fixture_pipeline(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    weights = tmp_path / "weights"
    moge = tmp_path / "moge"
    _write_sam3d_fixture(weights)
    _write_moge_fixture(moge)
    image, mask = _write_image_and_mask(tmp_path)
    output = "outputs/sam3d/gaussians.ply"
    glb_output = "outputs/sam3d/mesh.glb"
    trace_output = "outputs/sam3d/trace.json"
    pointmap = _fixture_pointmap()
    slat_coords = np.array([[0, 0, 0, 0], [0, 4, 4, 4]], dtype=np.int32)
    slat_feats = mx.ones((2, 1), dtype=mx.float32)
    mesh_coords, mesh_feats = _interior_mesh_features(origin=3)
    calls: dict[str, object] = {}

    import mlx_spatial.sam3d_inference as sam3d_inference
    from mlx_spatial.sam3d_mesh import extract_sam3d_mesh_from_features as real_extract_mesh
    from mlx_spatial.sam3d_export import write_sam3d_basic_glb as real_write_glb

    monkeypatch.setattr(
        sam3d_inference,
        "run_sam3d_moge_pointmap",
        lambda _image_rgb, *, root, memory_profile: Sam3dMogeResult(
            pointmap=Sam3dMogePointmap(
                pointmap=pointmap,
                intrinsics=np.eye(3, dtype=np.float32),
                mask=np.ones((2, 2), dtype=bool),
                depth=pointmap[..., 2],
                metadata={"fixture": True, "memory_profile": memory_profile, "root": str(root)},
            )
        ),
    )
    monkeypatch.setattr(
        sam3d_inference,
        "run_sam3d_ss_condition_stack",
        lambda *_args, **_kwargs: Sam3dConditionStackOutput(tokens=mx.zeros((1, 2, 1), dtype=mx.float32), metadata={"fixture": True}),
    )
    monkeypatch.setattr(
        sam3d_inference,
        "run_sam3d_slat_condition_stack",
        lambda *_args, **_kwargs: Sam3dConditionStackOutput(tokens=mx.zeros((1, 2, 1), dtype=mx.float32), metadata={"fixture": True}),
    )
    monkeypatch.setattr(sam3d_inference, "load_sam3d_condition_tensors", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(sam3d_inference, "load_sam3d_ss_generator_tensors", lambda *_args, **_kwargs: {})
    def fake_infer_ss_flow_config(*_args, **kwargs):
        calls["ss_flow_cfg_strength"] = kwargs["cfg_strength"]
        calls["ss_flow_rescale_t"] = kwargs["rescale_t"]
        calls["ss_flow_cfg_interval"] = kwargs["cfg_interval"]
        return SimpleNamespace()

    monkeypatch.setattr(sam3d_inference, "infer_sam3d_ss_flow_config", fake_infer_ss_flow_config)
    monkeypatch.setattr(
        sam3d_inference,
        "run_sam3d_ss_shortcut_flow",
        lambda *_args, **_kwargs: Sam3dSSFlowOutput(
            latents={
                "shape": mx.zeros((1, 1, 1), dtype=mx.float32),
                "6drotation_normalized": mx.zeros((1, 1, 6), dtype=mx.float32),
                "translation": mx.zeros((1, 1, 3), dtype=mx.float32),
                "scale": mx.ones((1, 1, 3), dtype=mx.float32),
                "translation_scale": mx.ones((1, 1, 1), dtype=mx.float32),
            },
            metadata={"fixture": True},
        ),
    )
    monkeypatch.setattr(sam3d_inference, "read_sam3d_ss_decoder_config", lambda *_args, **_kwargs: SimpleNamespace())
    monkeypatch.setattr(sam3d_inference, "load_sam3d_ss_decoder_tensors", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        sam3d_inference,
        "run_sam3d_ss_decoder",
        lambda *_args, **_kwargs: Sam3dSSDecoderOutput(
            occupancy=np.ones((1, 1, 1, 1, 1), dtype=np.float32),
            coords_original=slat_coords,
            coords=slat_coords,
            downsample_factor=1,
            metadata={"fixture": True, "coords_count": int(slat_coords.shape[0])},
        ),
    )
    monkeypatch.setattr(sam3d_inference, "decode_sam3d_scale_shift_invariant_pose", lambda *_args, **_kwargs: SimpleNamespace(metadata={"fixture": True}))
    monkeypatch.setattr(sam3d_inference, "load_sam3d_slat_generator_tensors", lambda *_args, **_kwargs: {})
    def fake_infer_slat_flow_config(*_args, **kwargs):
        calls["slat_flow_cfg_strength"] = kwargs["cfg_strength"]
        calls["slat_flow_rescale_t"] = kwargs["rescale_t"]
        calls["slat_flow_cfg_interval"] = kwargs["cfg_interval"]
        return SimpleNamespace()

    monkeypatch.setattr(sam3d_inference, "infer_sam3d_slat_flow_config", fake_infer_slat_flow_config)
    monkeypatch.setattr(
        sam3d_inference,
        "run_sam3d_slat_flow",
        lambda *_args, **_kwargs: Sam3dSLatFlowOutput(coords=slat_coords, feats=slat_feats, metadata={"fixture": True}),
    )
    monkeypatch.setattr(
        sam3d_inference,
        "read_sam3d_slat_decoder_config",
        lambda *_args, **_kwargs: Sam3dSLatDecoderConfig(resolution=2, model_channels=1, latent_channels=1, num_blocks=0, num_heads=1, window_size=1),
    )
    monkeypatch.setattr(sam3d_inference, "load_sam3d_slat_decoder_tensors", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(sam3d_inference, "run_sam3d_slat_decoder_network", lambda *_args, **_kwargs: mx.zeros((slat_coords.shape[0], 448), dtype=mx.float32))
    monkeypatch.setattr(
        sam3d_inference,
        "read_sam3d_mesh_decoder_config",
        lambda *_args, **_kwargs: Sam3dMeshDecoderConfig(resolution=2, model_channels=1, latent_channels=1, num_blocks=0, num_heads=1, window_size=1, use_color=True),
    )
    def fake_load_mesh_decoder_tensors(*_args, **_kwargs):
        calls["loaded_mesh_decoder_tensors"] = True
        return {}

    monkeypatch.setattr(sam3d_inference, "load_sam3d_mesh_decoder_tensors", fake_load_mesh_decoder_tensors)

    def fake_mesh_decoder_features(coords, feats, *_args, **_kwargs):
        calls["mesh_decoder_input_coords"] = np.array(coords, copy=True)
        calls["mesh_decoder_input_feat_shape"] = tuple(int(value) for value in feats.shape)
        return Sam3dMeshDecoderFeatureResult(coords=mesh_coords, feats=mx.array(mesh_feats), metadata={"fixture": True})

    monkeypatch.setattr(
        sam3d_inference,
        "run_sam3d_mesh_decoder_features",
        fake_mesh_decoder_features,
    )

    def spy_extract_mesh(coords, feats, **kwargs):
        calls["extract_mesh_coords"] = np.array(coords, copy=True)
        calls["extract_mesh_feat_shape"] = tuple(int(value) for value in feats.shape)
        result = real_extract_mesh(coords, feats, **kwargs)
        calls["expected_vertices"] = np.array(result.vertices, copy=True)
        calls["expected_faces"] = np.array(result.faces, copy=True)
        return result

    monkeypatch.setattr(sam3d_inference, "extract_sam3d_mesh_from_features", spy_extract_mesh)

    def spy_write_glb(path, *, vertices, faces, colors=None, normals=None):
        calls["glb_vertices"] = np.array(vertices, copy=True)
        calls["glb_faces"] = np.array(faces, copy=True)
        calls["glb_normals_shape"] = None if normals is None else tuple(int(value) for value in normals.shape)
        return real_write_glb(path, vertices=vertices, faces=faces, colors=colors, normals=normals)

    monkeypatch.setattr(sam3d_inference, "write_sam3d_basic_glb", spy_write_glb)

    def fake_bake_texture(
        mesh,
        *,
        gaussian_xyz,
        gaussian_features_dc,
        gaussian_opacity,
        gaussian_scale,
        texture_size,
        k_neighbors,
        texel_chunk_size,
        xatlas_face_guard,
    ):
        calls["texture_mesh_vertices"] = np.array(mesh.vertices, copy=True)
        calls["texture_mesh_faces"] = np.array(mesh.faces, copy=True)
        calls["texture_gaussian_count"] = int(np.asarray(gaussian_xyz).shape[0])
        calls["texture_size"] = int(texture_size)
        calls["texture_k_neighbors"] = int(k_neighbors)
        calls["texture_chunk_size"] = int(texel_chunk_size)
        calls["texture_xatlas_face_guard"] = int(xatlas_face_guard)
        return SimpleNamespace(
            vertices=np.asarray(mesh.vertices, dtype=np.float32),
            faces=np.asarray(mesh.faces, dtype=np.int32),
            normals=np.asarray(mesh.normals, dtype=np.float32),
            uvs=np.zeros((mesh.vertices.shape[0], 2), dtype=np.float32),
            base_color_rgba=np.full((16, 16, 4), 255, dtype=np.uint8),
            stats={
                "backend": "gaussian-kdtree",
                "texture_size": int(texture_size),
                "gaussian_count": int(np.asarray(gaussian_xyz).shape[0]),
                "k_neighbors": int(k_neighbors),
                "texel_chunk_size": int(texel_chunk_size),
                "sampled_texel_count": 4,
                "raster_texel_count": 4,
                "raw_coverage_ratio": 1.0,
                "final_coverage_ratio": 1.0,
                "unwrap_backend": "xatlas",
                "xatlas_face_guard": int(xatlas_face_guard),
                "unwrap_seconds": 0.0,
                "unwrap_chunks": 1,
                "unwrap_chart_count": 1,
                "unwrap_utilization": 1.0,
                "elapsed_seconds": 0.0,
            },
        )

    def fake_write_textured_glb(path, baked_texture):
        calls["textured_glb_path"] = str(path)
        calls["textured_glb_uv_shape"] = tuple(int(value) for value in baked_texture.uvs.shape)
        output_path = tmp_path / path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"glTFfixture")
        return SimpleNamespace(
            path=path,
            vertex_count=int(baked_texture.vertices.shape[0]),
            face_count=int(baked_texture.faces.shape[0]),
            bytes_written=output_path.stat().st_size,
            has_vertex_color=False,
            has_normals=True,
            has_texture=True,
            format="glb",
        )

    monkeypatch.setattr(sam3d_inference, "bake_sam3d_gaussian_texture_for_glb", fake_bake_texture)
    monkeypatch.setattr(sam3d_inference, "write_sam3d_textured_glb", fake_write_textured_glb)

    assert main(
        [
            "reconstruct",
            str(weights),
            str(image),
            "--mask",
            str(mask),
            "--output",
            output,
            "--glb-output",
            glb_output,
            "--moge-root",
            str(moge),
            "--glb-texture-size",
            "16",
            "--glb-gaussian-k",
            "4",
            "--glb-texel-chunk-size",
            "8",
            "--ss-cfg-strength",
            "2.0",
            "--ss-rescale-t",
            "1.0",
            "--ss-cfg-interval",
            "0",
            "250",
            "--slat-cfg-strength",
            "3.0",
            "--slat-rescale-t",
            "1.5",
            "--slat-cfg-interval",
            "0",
            "300",
            "--trace-output",
            trace_output,
        ]
    ) == 0

    cli_output = capsys.readouterr().out
    assert "artifact=outputs/sam3d/mesh.glb" in cli_output
    assert (tmp_path / output).stat().st_size > 0
    assert (tmp_path / glb_output).read_bytes()[:4] == b"glTF"
    trace = json.loads((tmp_path / trace_output).read_text(encoding="utf-8"))
    assert trace["blocker"] is None
    assert trace["completed_stages"][-2:] == ["mesh-decoder", "glb-export"]
    assert [artifact["name"] for artifact in trace["outputs"]] == ["gaussians.ply", "mesh.glb"]
    assert trace["metadata"]["structured_latent"]["coords_shape"] == [2, 4]
    assert trace["metadata"]["structured_latent"]["feature_shape"] == [2, 1]
    assert trace["metadata"]["structured_latent"]["feature_stats"]["shape"] == [2, 1]
    assert trace["metadata"]["structured_latent"]["feature_stats"]["finite_count"] == 2
    assert trace["metadata"]["ss_cfg_strength"] == 2.0
    assert trace["metadata"]["ss_rescale_t"] == 1.0
    assert trace["metadata"]["ss_cfg_interval"] == [0.0, 250.0]
    assert trace["metadata"]["slat_cfg_strength"] == 3.0
    assert trace["metadata"]["slat_rescale_t"] == 1.5
    assert trace["metadata"]["slat_cfg_interval"] == [0.0, 300.0]
    assert trace["metadata"]["mesh_decoder"]["extraction"]["vertex_count"] > 0
    assert trace["metadata"]["mesh_decoder"]["raw_mesh"]["face_count"] > 0
    assert trace["metadata"]["glb_postprocess"]["mode"] == "cleaned"
    assert trace["metadata"]["glb_postprocess"]["target_faces"] == 300000
    assert trace["metadata"]["glb_postprocess"]["min_component_faces"] == 256
    assert trace["metadata"]["glb_postprocess"]["min_component_face_fraction"] == 0.0005
    assert trace["metadata"]["glb_postprocess"]["xatlas_face_guard"] == 400000
    assert trace["metadata"]["glb_export"]["face_count"] > 0
    assert trace["metadata"]["glb_export"]["has_normals"] is True
    assert trace["metadata"]["glb_export"]["has_texture"] is True
    assert trace["metadata"]["glb_export"]["texture"]["backend"] == "gaussian-kdtree"
    assert trace["metadata"]["glb_export"]["texture"]["base_color_shape"] == [16, 16, 4]
    assert trace["metadata"]["gaussian_decoder"]["opacity_quality"]["status"] == "nominal"
    assert trace["metadata"]["gaussian_decoder"]["network_output_stats"]["shape"] == [2, 448]
    assert trace["metadata"]["gaussian_decoder"]["network_channel_stats"]["_opacity"]["shape"] == [2, 32]
    assert trace["metadata"]["glb_export"]["postprocess"]["mode"] == "cleaned"
    assert trace["metadata"]["glb_export"]["postprocess"]["components_after"] >= 1
    assert trace["metadata"]["glb_export"]["artifact_role"].startswith("textured preview mesh GLB")
    assert calls["loaded_mesh_decoder_tensors"] is True
    np.testing.assert_array_equal(calls["mesh_decoder_input_coords"], slat_coords)
    assert calls["mesh_decoder_input_feat_shape"] == (2, 1)
    np.testing.assert_array_equal(calls["extract_mesh_coords"], mesh_coords)
    assert calls["extract_mesh_feat_shape"] == mesh_feats.shape
    assert calls["texture_mesh_vertices"].shape[1] == 3
    assert calls["texture_mesh_faces"].shape[1] == 3
    assert calls["texture_gaussian_count"] > 0
    assert calls["texture_size"] == 16
    assert calls["texture_k_neighbors"] == 4
    assert calls["texture_chunk_size"] == 8
    assert calls["texture_xatlas_face_guard"] == 400000
    assert calls["ss_flow_cfg_strength"] == 2.0
    assert calls["ss_flow_rescale_t"] == 1.0
    assert calls["ss_flow_cfg_interval"] == (0.0, 250.0)
    assert calls["slat_flow_cfg_strength"] == 3.0
    assert calls["slat_flow_rescale_t"] == 1.5
    assert calls["slat_flow_cfg_interval"] == (0.0, 300.0)
    assert calls["textured_glb_uv_shape"][1] == 2

    calls.clear()
    basic_glb_output = "outputs/sam3d/mesh-basic.glb"
    basic_trace_output = "outputs/sam3d/trace-basic.json"
    assert main(
        [
            "reconstruct",
            str(weights),
            str(image),
            "--mask",
            str(mask),
            "--output",
            output,
            "--glb-output",
            basic_glb_output,
            "--glb-texture",
            "none",
            "--glb-postprocess",
            "basic",
            "--moge-root",
            str(moge),
            "--trace-output",
            basic_trace_output,
        ]
    ) == 0
    basic_trace = json.loads((tmp_path / basic_trace_output).read_text(encoding="utf-8"))
    assert basic_trace["metadata"]["glb_export"]["postprocess"]["mode"] == "basic"
    assert basic_trace["metadata"]["glb_export"]["postprocess"]["applied"] is False
    assert basic_trace["metadata"]["slat_cfg_strength"] == SAM3D_DEFAULT_SLAT_CFG_STRENGTH
    assert basic_trace["metadata"]["slat_rescale_t"] == SAM3D_DEFAULT_SLAT_RESCALE_T
    assert calls["slat_flow_cfg_strength"] == SAM3D_DEFAULT_SLAT_CFG_STRENGTH
    assert calls["slat_flow_rescale_t"] == SAM3D_DEFAULT_SLAT_RESCALE_T
    np.testing.assert_allclose(calls["glb_vertices"], calls["expected_vertices"])
    np.testing.assert_array_equal(calls["glb_faces"], calls["expected_faces"])


def test_sam3d_cli_reconstruct_rejects_non_outputs_path(tmp_path, capsys):
    weights = tmp_path / "weights"
    _write_sam3d_fixture(weights)
    image, mask = _write_image_and_mask(tmp_path)

    assert main(["reconstruct", str(weights), str(image), "--mask", str(mask), "--output", str(tmp_path / "bad.ply")]) == 2
    output = capsys.readouterr().out
    assert "blocker_stage=argument-validation" in output
    assert "reason=SAM3D output path must stay under outputs" in output


def test_sam3d_cli_reconstruct_ignores_sample_json_sidecar(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    weights = tmp_path / "weights"
    _write_sam3d_fixture(weights)
    image, mask = _write_image_and_mask(tmp_path)
    (tmp_path / "sample.json").write_text(
        json.dumps({"masks": [{"id": "other", "path": "other.png", "primary": True}]}),
        encoding="utf-8",
    )

    assert (
        main(
            [
                "reconstruct",
                str(weights),
                str(image),
                "--mask",
                str(mask),
                "--output",
                "outputs/sam3d/gaussians.ply",
                "--trace-output",
                "outputs/sam3d/trace.json",
            ]
        )
        == 2
    )
    output = capsys.readouterr().out
    assert "blocker_stage=moge-pointmap" in output

    trace = json.loads((tmp_path / "outputs/sam3d/trace.json").read_text(encoding="utf-8"))
    assert trace["blocker"]["stage"] == "moge-pointmap"
    assert trace["mask_path"] == str(mask)
    assert trace["metadata"]["input"]["mask_path"] == str(mask)
    assert "sample_mask" not in trace["metadata"]["input"]
    assert "quality_issues" not in trace["metadata"]


def test_sam3d_quality_issues_detect_flat_sparse_and_gaussian_geometry():
    issues = sam3d_quality_issue_records(
        {
            "sparse_structure": {
                "occupancy_quality": {"status": "nominal", "positive_fraction": 0.02},
                "geometry_quality": {
                    "status": "flat-geometry",
                    "axis_range": (63, 63, 0),
                    "min_axis_range": 0,
                    "min_axis_range_threshold": 3,
                },
            },
            "gaussian_decoder": {
                "geometry_quality": {
                    "status": "flat-geometry",
                    "axis_range": (1.0, 1.0, 0.01),
                    "min_axis_range": 0.01,
                    "min_axis_range_threshold": 0.05,
                },
                "opacity_quality": {"status": "nominal"},
            },
        }
    )

    assert [issue["kind"] for issue in issues] == ["sparse_structure", "gaussian_geometry"]
    assert issues[0]["status"] == "flat-geometry"
    assert issues[0]["axis_range"] == (63, 63, 0)
    assert issues[1]["status"] == "flat-geometry"
    assert issues[1]["axis_range"] == (1.0, 1.0, 0.01)


def test_sam3d_cli_reconstruct_rejects_gaussian_texture_without_cleaned_postprocess(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    weights = tmp_path / "weights"
    _write_sam3d_fixture(weights)
    image, mask = _write_image_and_mask(tmp_path)

    assert (
        main(
            [
                "reconstruct",
                str(weights),
                str(image),
                "--mask",
                str(mask),
                "--output",
                "outputs/sam3d/gaussians.ply",
                "--glb-output",
                "outputs/sam3d/mesh.glb",
                "--glb-postprocess",
                "basic",
            ]
        )
        == 2
    )
    output = capsys.readouterr().out
    assert "blocker_stage=argument-validation" in output
    assert "requires --glb-postprocess cleaned" in output
