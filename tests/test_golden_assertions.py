"""Focused tests for shared golden-fixture comparison helpers."""

from __future__ import annotations

import numpy as np
import pytest

from tests.golden_assertions import assert_golden_close, summarize_array


def test_summarize_array_uses_exact_integer_digest_and_float_statistics():
    integer_summary = summarize_array(np.asarray([[0, 1], [2, 3]], dtype=np.int32))
    float_summary = summarize_array(np.asarray([-1.0, 1.0], dtype=np.float32))

    assert integer_summary["shape"] == [2, 2]
    assert integer_summary["dtype"] == "int32"
    assert len(integer_summary["sha256"]) == 64
    assert float_summary == {
        "shape": [2],
        "dtype": "float32",
        "statistics": {
            "min": -1.0,
            "max": 1.0,
            "mean": 0.0,
            "std": 1.0,
            "l2": 2**0.5,
        },
    }


def test_assert_golden_close_tolerates_small_float_drift_but_not_contract_changes():
    assert_golden_close(
        {"shape": [1, 2], "mean": 1.00001},
        {"shape": [1, 2], "mean": 1.0},
    )

    with pytest.raises(AssertionError, match=r"golden\.shape\[1\]"):
        assert_golden_close({"shape": [1, 3], "mean": 1.0}, {"shape": [1, 2], "mean": 1.0})
