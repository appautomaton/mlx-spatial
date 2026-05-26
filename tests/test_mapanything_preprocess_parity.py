import os
import sys
from pathlib import Path

import numpy as np
import pytest

from mlx_spatial.mapanything_preprocess import preprocess_mapanything_images


MAPANYTHING_TORCH_PARITY_ENV = "MAPANYTHING_TORCH_REF"


pytestmark = pytest.mark.torch_parity


def test_mapanything_preprocess_matches_vendored_load_images_on_desk_pair():
    if os.environ.get(MAPANYTHING_TORCH_PARITY_ENV) != "1":
        pytest.skip(f"{MAPANYTHING_TORCH_PARITY_ENV}=1 is required for vendored Torch parity")
    desk = Path("inputs/map-anything/desk")
    if not desk.is_dir():
        pytest.skip(f"Desk inputs not present: {desk}")

    sys.path.insert(0, str(Path("vendors/map-anything").resolve()))
    try:
        from mapanything.utils.image import load_images
    finally:
        try:
            sys.path.remove(str(Path("vendors/map-anything").resolve()))
        except ValueError:
            pass

    actual = preprocess_mapanything_images(desk)
    reference = load_images(str(desk))

    assert len(reference) == actual.frame_count == 2
    for view, ref in zip(actual.views, reference, strict=True):
        assert tuple(ref["true_shape"][0]) == view.true_shape
        assert ref["idx"] == view.idx
        assert ref["instance"] == view.instance
        assert ref["data_norm_type"] == [view.data_norm_type]
        np.testing.assert_allclose(
            np.array(view.img),
            ref["img"].detach().cpu().numpy(),
            rtol=1e-6,
            atol=1e-6,
        )
