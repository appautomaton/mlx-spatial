import ast
from pathlib import Path

import mlx.core as mx


ROOT = Path(__file__).resolve().parents[1]
ROUTINE_PIPELINE_GUARDS = {
    "hyworld2": (
        "tests/test_hyworld2_inference.py",
        "test_fixture_reconstruct_writes_staged_outputs_under_outputs",
    ),
    "lito": (
        "tests/test_lito_inference.py",
        "test_full_pipeline_runs_on_sample_input",
    ),
    "mapanything": (
        "tests/test_mapanything_scene_pipeline.py",
        "test_mapanything_miniature_scene_pipeline_matches_golden",
    ),
    "pixal3d": (
        "tests/test_pixal3d_pipeline.py",
        "test_pixal3d_pipeline_writes_textured_glb_with_fake_export_route",
    ),
    "sam3d": (
        "tests/test_sam3d_tools.py",
        "test_sam3d_cli_reconstruct_writes_gaussian_ply_and_textured_glb_with_fixture_pipeline",
    ),
    "trellis2": (
        "tests/test_trellis2_golden_fixture.py",
        "test_miniature_int8_pipeline_emits_golden_trace_and_glb",
    ),
}
ROUTINE_EXCLUSION_MARKERS = {"heavy", "real_assets", "torch_parity"}


def test_pytest_sets_mlx_cpu_default():
    assert mx.default_device() == mx.cpu


def test_pytest_defaults_select_bounded_self_contained_tests(pytestconfig):
    addopts = pytestconfig.getini("addopts")
    joined = " ".join(addopts if isinstance(addopts, list) else [addopts])
    markers = pytestconfig.getini("markers")

    assert "-m" in addopts
    assert "not (heavy or real_assets or torch_parity)" in joined
    for name in (
        "integration",
        "real_assets",
        "metal",
        "heavy",
        "benchmark",
        "torch_parity",
    ):
        assert any(marker.startswith(f"{name}:") for marker in markers)


def test_supported_pipelines_keep_a_routine_integration_guard():
    errors = []
    for model, (relative_path, function_name) in ROUTINE_PIPELINE_GUARDS.items():
        path = ROOT / relative_path
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        function = next(
            (
                node
                for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == function_name
            ),
            None,
        )
        if function is None:
            errors.append(f"{model}: missing {relative_path}::{function_name}")
            continue

        markers = {
            marker
            for decorator in function.decorator_list
            if (marker := _pytest_marker_name(decorator)) is not None
        }
        if "integration" not in markers:
            errors.append(f"{model}: representative guard is not marked integration")
        excluded = markers & ROUTINE_EXCLUSION_MARKERS
        if excluded:
            errors.append(
                f"{model}: representative guard is excluded from routine CI by {sorted(excluded)}"
            )

    assert not errors, "\n".join(errors)


def _pytest_marker_name(decorator: ast.expr) -> str | None:
    target = decorator.func if isinstance(decorator, ast.Call) else decorator
    if not isinstance(target, ast.Attribute):
        return None
    mark = target.value
    if not isinstance(mark, ast.Attribute) or mark.attr != "mark":
        return None
    if not isinstance(mark.value, ast.Name) or mark.value.id != "pytest":
        return None
    return target.attr
