import mlx.core as mx
from safetensors.mlx import save_file

import mlx_spatial
from mlx_spatial.trellis2_dinov3 import (
    DinoV3CheckpointInventory,
    DinoV3ConditioningResult,
    DinoV3InspectionResult,
    DinoV3ModelConfig,
    DinoV3PortBlocker,
    assess_dinov3_mlx_conditioning,
    inspect_dinov3_assets,
    inspect_dinov3_checkpoint,
    read_dinov3_config,
)


def _write_config(root, *, hidden_size=1024, fake=False):
    root.mkdir(parents=True, exist_ok=True)
    model_type = "mlx_spatial_fake_dinov3" if fake else "dinov3_vit"
    root.joinpath("config.json").write_text(
        f"""{{
            "model_type": "{model_type}",
            "image_size": 2,
            "patch_size": 1,
            "hidden_size": {hidden_size},
            "num_hidden_layers": 1,
            "num_attention_heads": 16,
            "intermediate_size": 4096,
            "layer_norm_eps": 0.000001,
            "use_swiglu_ffn": true,
            "num_register_tokens": 0,
            "rope_theta": 100.0,
            "pos_embed_rescale": 2.0,
            "mlx_spatial_fake_conditioning": {str(fake).lower()}
        }}"""
    )


def _write_checkpoint(root, *, hidden_size=1024, patch_shape=None):
    shape = patch_shape or (hidden_size, 3, 1, 1)
    save_file(
        {
            "embeddings.patch_embeddings.weight": mx.ones(shape, dtype=mx.float32),
            "layer.0.norm1.weight": mx.ones((hidden_size,), dtype=mx.float32),
            "layer.0.attention.query.weight": mx.ones((hidden_size, hidden_size), dtype=mx.float32),
        },
        root / "model.safetensors",
    )


def _write_forward_checkpoint(root, *, hidden_size=1024):
    save_file(
        {
            "embeddings.cls_token": mx.zeros((1, 1, hidden_size), dtype=mx.float32),
            "embeddings.patch_embeddings.bias": mx.zeros((hidden_size,), dtype=mx.float32),
            "embeddings.patch_embeddings.weight": mx.ones((hidden_size, 3, 1, 1), dtype=mx.float32),
            "layer.0.attention.k_proj.weight": mx.ones((hidden_size, hidden_size), dtype=mx.float32),
            "layer.0.attention.o_proj.bias": mx.zeros((hidden_size,), dtype=mx.float32),
            "layer.0.attention.o_proj.weight": mx.ones((hidden_size, hidden_size), dtype=mx.float32),
            "layer.0.attention.q_proj.bias": mx.zeros((hidden_size,), dtype=mx.float32),
            "layer.0.attention.q_proj.weight": mx.ones((hidden_size, hidden_size), dtype=mx.float32),
            "layer.0.attention.v_proj.bias": mx.zeros((hidden_size,), dtype=mx.float32),
            "layer.0.attention.v_proj.weight": mx.ones((hidden_size, hidden_size), dtype=mx.float32),
            "layer.0.layer_scale1.lambda1": mx.ones((hidden_size,), dtype=mx.float32),
            "layer.0.layer_scale2.lambda1": mx.ones((hidden_size,), dtype=mx.float32),
            "layer.0.mlp.down_proj.bias": mx.zeros((hidden_size,), dtype=mx.float32),
            "layer.0.mlp.down_proj.weight": mx.ones((hidden_size, 4096), dtype=mx.float32),
            "layer.0.mlp.up_proj.bias": mx.zeros((4096,), dtype=mx.float32),
            "layer.0.mlp.up_proj.weight": mx.ones((4096, hidden_size), dtype=mx.float32),
            "layer.0.norm1.bias": mx.zeros((hidden_size,), dtype=mx.float32),
            "layer.0.norm1.weight": mx.ones((hidden_size,), dtype=mx.float32),
            "layer.0.norm2.bias": mx.zeros((hidden_size,), dtype=mx.float32),
            "layer.0.norm2.weight": mx.ones((hidden_size,), dtype=mx.float32),
            "norm.bias": mx.zeros((hidden_size,), dtype=mx.float32),
            "norm.weight": mx.ones((hidden_size,), dtype=mx.float32),
        },
        root / "model.safetensors",
    )


def _write_dinov3_root(root, *, fake=False):
    _write_config(root, fake=fake)
    _write_checkpoint(root)


def test_read_dinov3_config_reports_required_fields(tmp_path):
    _write_config(tmp_path)

    result = read_dinov3_config(tmp_path / "config.json")

    assert result.ready is False
    assert result.blocker is None
    assert result.config == DinoV3ModelConfig(
        model_type="dinov3_vit",
        image_size=2,
        patch_size=1,
        hidden_size=1024,
        num_hidden_layers=1,
        num_attention_heads=16,
        intermediate_size=4096,
        layer_norm_eps=0.000001,
        use_swiglu_ffn=True,
        num_register_tokens=0,
        expected_feature_width=1024,
        fake_conditioning=False,
        rope_theta=100.0,
        pos_embed_rescale=2.0,
    )


def test_read_dinov3_config_blocks_on_missing_required_field(tmp_path):
    tmp_path.joinpath("config.json").write_text("{}")

    result = read_dinov3_config(tmp_path / "config.json")

    assert result.config is None
    assert result.blocker is not None
    assert result.blocker.operation == "DINOv3 config field validation"
    assert "model_type" in result.blocker.reason


def test_inspect_dinov3_checkpoint_reports_key_inventory(tmp_path):
    _write_config(tmp_path)
    _write_checkpoint(tmp_path)
    config = read_dinov3_config(tmp_path / "config.json").config

    result = inspect_dinov3_checkpoint(tmp_path / "model.safetensors", config)

    assert result.ready
    assert result.inventory == DinoV3CheckpointInventory(
        checkpoint_path=tmp_path / "model.safetensors",
        tensor_count=3,
        patch_embedding_key="embeddings.patch_embeddings.weight",
        patch_embedding_shape=(1024, 3, 1, 1),
        layer_prefix="layer.",
        observed_layer_count=1,
        norm_keys=("layer.0.norm1.weight",),
        sample_keys=(
            "embeddings.patch_embeddings.weight",
            "layer.0.attention.query.weight",
            "layer.0.norm1.weight",
        ),
    )


def test_inspect_dinov3_checkpoint_blocks_on_bad_patch_shape(tmp_path):
    _write_config(tmp_path)
    _write_checkpoint(tmp_path, patch_shape=(8, 3, 1, 1))
    config = read_dinov3_config(tmp_path / "config.json").config

    result = inspect_dinov3_checkpoint(tmp_path / "model.safetensors", config)

    assert result.inventory is None
    assert result.blocker is not None
    assert result.blocker.operation == "DINOv3 checkpoint shape validation"
    assert "hidden_size=1024" in result.blocker.reason


def test_inspect_dinov3_assets_reports_missing_assets(tmp_path):
    result = inspect_dinov3_assets(tmp_path)

    assert result.blocker is not None
    assert result.blocker.operation == "local DINOv3 asset validation"
    assert "config.json" in result.blocker.reason
    assert "model.safetensors" in result.blocker.reason


def test_assess_dinov3_mlx_conditioning_returns_fake_fixture_output(tmp_path):
    _write_dinov3_root(tmp_path, fake=True)
    image_tensor = mx.zeros((1, 3, 512, 512), dtype=mx.float32)

    result = assess_dinov3_mlx_conditioning(
        tmp_path,
        expected_feature_width=1024,
        image_tensor=image_tensor,
    )

    assert result == DinoV3ConditioningResult(
        shape=(1, 4, 1024),
        dtype="float32",
        detail=f"fake DINOv3 conditioning from {tmp_path} using embeddings.patch_embeddings.weight",
        blocker=None,
    )


def test_assess_dinov3_mlx_conditioning_blocks_on_real_forward_probe(tmp_path):
    _write_config(tmp_path)
    _write_forward_checkpoint(tmp_path)

    result = assess_dinov3_mlx_conditioning(tmp_path, expected_feature_width=1024)

    assert result.shape is None
    assert result.blocker is not None
    assert result.blocker.operation == "MLX DINOv3 attention block forward"
    assert "RoPE geometry" in result.blocker.reason


def test_dinov3_helpers_are_public_exports():
    assert mlx_spatial.DinoV3CheckpointInventory is DinoV3CheckpointInventory
    assert mlx_spatial.DinoV3ConditioningResult is DinoV3ConditioningResult
    assert mlx_spatial.DinoV3InspectionResult is DinoV3InspectionResult
    assert mlx_spatial.DinoV3ModelConfig is DinoV3ModelConfig
    assert mlx_spatial.DinoV3PortBlocker is DinoV3PortBlocker
    assert mlx_spatial.assess_dinov3_mlx_conditioning is assess_dinov3_mlx_conditioning
    assert mlx_spatial.inspect_dinov3_assets is inspect_dinov3_assets
    assert mlx_spatial.inspect_dinov3_checkpoint is inspect_dinov3_checkpoint
    assert mlx_spatial.read_dinov3_config is read_dinov3_config
