import json
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
from PIL import Image
from safetensors.mlx import save_file

from mlx_spatial.sam3d import main
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
    assert not (tmp_path / output).exists()


def test_sam3d_cli_reconstruct_writes_gaussian_ply_and_basic_glb_with_fixture_pipeline(tmp_path, capsys, monkeypatch):
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
    slat_coords = np.array([[0, 0, 0, 0], [0, 1, 1, 1]], dtype=np.int32)
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
    monkeypatch.setattr(sam3d_inference, "infer_sam3d_ss_flow_config", lambda *_args, **_kwargs: SimpleNamespace())
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
    monkeypatch.setattr(sam3d_inference, "infer_sam3d_slat_flow_config", lambda *_args, **_kwargs: SimpleNamespace())
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

    def spy_write_glb(path, *, vertices, faces, colors=None):
        calls["glb_vertices"] = np.array(vertices, copy=True)
        calls["glb_faces"] = np.array(faces, copy=True)
        return real_write_glb(path, vertices=vertices, faces=faces, colors=colors)

    monkeypatch.setattr(sam3d_inference, "write_sam3d_basic_glb", spy_write_glb)

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
    assert trace["metadata"]["mesh_decoder"]["extraction"]["vertex_count"] > 0
    assert trace["metadata"]["glb_export"]["face_count"] > 0
    assert calls["loaded_mesh_decoder_tensors"] is True
    np.testing.assert_array_equal(calls["mesh_decoder_input_coords"], slat_coords)
    assert calls["mesh_decoder_input_feat_shape"] == (2, 1)
    np.testing.assert_array_equal(calls["extract_mesh_coords"], mesh_coords)
    assert calls["extract_mesh_feat_shape"] == mesh_feats.shape
    np.testing.assert_allclose(calls["glb_vertices"], calls["expected_vertices"])
    np.testing.assert_array_equal(calls["glb_faces"], calls["expected_faces"])


def test_sam3d_cli_reconstruct_rejects_non_outputs_path(tmp_path):
    weights = tmp_path / "weights"
    _write_sam3d_fixture(weights)
    image, mask = _write_image_and_mask(tmp_path)

    try:
        main(["reconstruct", str(weights), str(image), "--mask", str(mask), "--output", str(tmp_path / "bad.ply")])
    except ValueError as error:
        assert "must stay under outputs" in str(error)
    else:
        raise AssertionError("expected reconstruct to reject output outside outputs")
