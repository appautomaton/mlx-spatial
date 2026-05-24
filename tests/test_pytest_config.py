import mlx.core as mx


def test_pytest_sets_mlx_cpu_default():
    assert mx.default_device() == mx.cpu


def test_pytest_defaults_skip_heavy_tests(pytestconfig):
    addopts = pytestconfig.getini("addopts")
    joined = " ".join(addopts if isinstance(addopts, list) else [addopts])
    markers = pytestconfig.getini("markers")

    assert "-m" in addopts
    assert "not heavy" in joined
    assert any(marker.startswith("heavy:") for marker in markers)
