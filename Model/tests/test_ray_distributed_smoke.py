from __future__ import annotations

import pytest

from distributed_training.ray_smoke import validate_smoke_config


@pytest.mark.parametrize("num_workers", [2, 4, 8])
def test_validate_smoke_config_accepts_reviewed_topologies(num_workers):
    validate_smoke_config(
        num_workers=num_workers,
        steps=1,
        learning_rate=1e-4,
    )


@pytest.mark.parametrize("num_workers", [0, 1, 3, 9])
def test_validate_smoke_config_rejects_unreviewed_topologies(num_workers):
    with pytest.raises(ValueError, match="num_workers"):
        validate_smoke_config(
            num_workers=num_workers,
            steps=1,
            learning_rate=1e-4,
        )


def test_validate_smoke_config_rejects_invalid_step_or_rate():
    with pytest.raises(ValueError, match="steps"):
        validate_smoke_config(
            num_workers=4,
            steps=0,
            learning_rate=1e-4,
        )
    with pytest.raises(ValueError, match="learning_rate"):
        validate_smoke_config(
            num_workers=4,
            steps=1,
            learning_rate=0.0,
        )
