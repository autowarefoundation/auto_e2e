from types import SimpleNamespace

import numpy as np
import pytest

from accuracy_benchmark.drive_jepa.drive_jepa_perception_adapter import (
    projection_metadata,
)


def test_camera_to_reference_is_inverted_before_projection():
    camera_to_reference = np.eye(4)
    camera_to_reference[0, 3] = 1.0
    intrinsic = np.array([[100.0, 0.0, 200.0], [0.0, 100.0, 150.0], [0.0, 0.0, 1.0]])
    projection, shape = projection_metadata(
        SimpleNamespace(extrinsic=camera_to_reference, intrinsic=intrinsic),
        (300, 400, 3),
    )
    projected = projection @ np.array([3.0, 1.0, 10.0, 1.0])
    # Reference point corresponds to camera point [2,1,10], pixel [220,160].
    np.testing.assert_allclose(projected[:2] / projected[2], [88.0, 64.0])
    assert shape == [128, 160, 3]


def test_real_front_resolution_sets_projection_normalization_shape():
    _, shape = projection_metadata(
        SimpleNamespace(extrinsic=np.eye(4), intrinsic=np.eye(3)), (3104, 5856, 3)
    )
    assert shape == [1248, 2368, 3]


def test_nonfinite_camera_calibration_rejected():
    with pytest.raises(ValueError):
        projection_metadata(
            SimpleNamespace(extrinsic=np.eye(4) * np.nan, intrinsic=np.eye(3)),
            (300, 400, 3),
        )
