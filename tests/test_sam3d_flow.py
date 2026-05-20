import mlx.core as mx
import numpy as np

from mlx_spatial.sam3d_flow import (
    compare_sam3d_shortcut_outputs,
    denormalize_sam3d_slat,
    sam3d_classifier_free_guidance,
    sam3d_euler_solve,
    sam3d_flow_time_sequence,
    sam3d_seeded_normal,
    sam3d_shortcut_schedule,
)


def test_sam3d_flow_time_sequence_matches_rescaled_inference_schedule():
    t_seq = sam3d_flow_time_sequence(2, rescale_t=3.0)

    assert np.allclose(t_seq, np.array([0.0, 0.25, 1.0], dtype=np.float32))


def test_sam3d_shortcut_schedule_uses_no_shortcut_during_exact_inference():
    schedule = sam3d_shortcut_schedule(4, no_shortcut=True)

    assert schedule.shortcut_d == 0.0
    assert schedule.time_scale == 1000.0
    assert schedule.t_seq.shape == (5,)


def test_sam3d_shortcut_reference_report_accepts_fewer_step_fixture():
    reference = {
        "shape": mx.array([[1.0, 2.0, 3.0]], dtype=mx.float32),
        "pose": mx.array([[0.25, -0.5]], dtype=mx.float32),
    }
    fewer_step = {
        "shape": mx.array([[1.0, 2.0 + 1e-6, 3.0]], dtype=mx.float32),
        "pose": mx.array([[0.25, -0.5]], dtype=mx.float32),
    }

    report = compare_sam3d_shortcut_outputs(reference, fewer_step, atol=1e-5, rtol=1e-4)

    assert report.passed
    assert report.tensor_count == 2
    assert report.max_abs_error < 1e-5
    assert report.tensor_errors["shape"]["passed"] is True


def test_sam3d_shortcut_reference_report_rejects_mismatched_output():
    reference = {"shape": mx.array([[1.0]], dtype=mx.float32)}
    fewer_step = {"shape": mx.array([[1.1]], dtype=mx.float32)}

    report = compare_sam3d_shortcut_outputs(reference, fewer_step, atol=1e-5, rtol=1e-4)

    assert not report.passed
    assert report.tensor_errors["shape"]["passed"] is False


def test_sam3d_seeded_normal_is_reproducible():
    first = sam3d_seeded_normal((2, 3), seed=42)
    second = sam3d_seeded_normal((2, 3), seed=42)

    assert np.allclose(np.array(first), np.array(second))


def test_sam3d_classifier_free_guidance_respects_interval():
    cond = mx.ones((2,), dtype=mx.float32)
    uncond = mx.zeros((2,), dtype=mx.float32)

    active = sam3d_classifier_free_guidance(cond, uncond, strength=2.0, interval=(0.0, 500.0), t_scaled=250.0)
    inactive = sam3d_classifier_free_guidance(cond, uncond, strength=2.0, interval=(0.0, 500.0), t_scaled=750.0)

    assert np.allclose(np.array(active), np.array([3.0, 3.0], dtype=np.float32))
    assert np.allclose(np.array(inactive), np.array([1.0, 1.0], dtype=np.float32))


def test_sam3d_euler_solve_uses_forward_difference_updates():
    x0 = mx.array([0.0], dtype=mx.float32)

    solved = sam3d_euler_solve(x0, lambda _x, _t: mx.array([2.0], dtype=mx.float32), t_seq=np.array([0.0, 0.25, 1.0]))

    assert np.allclose(np.array(solved), np.array([2.0], dtype=np.float32))


def test_denormalize_sam3d_slat_applies_channel_stats():
    features = mx.zeros((1, 2, 8), dtype=mx.float32)

    denorm = denormalize_sam3d_slat(features)

    assert tuple(denorm.shape) == (1, 2, 8)
    assert np.allclose(np.array(denorm[0, 0, 0]), -2.1687546)
