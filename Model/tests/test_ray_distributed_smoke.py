from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from distributed_training.ray_smoke import (
    expected_hostname_count,
    validate_capacity_block_window,
    validate_smoke_config,
)


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


@pytest.mark.parametrize(
    ("num_workers", "hostname_count"),
    [(2, 2), (4, 1), (8, 1)],
)
def test_expected_hostname_count_matches_worker_pod_topology(
    num_workers,
    hostname_count,
):
    assert expected_hostname_count(num_workers) == hostname_count


def test_capacity_block_window_requires_one_remaining_hour():
    valid_end = datetime.now(timezone.utc) + timedelta(hours=2)
    validate_capacity_block_window(valid_end.isoformat())

    with pytest.raises(ValueError, match="required"):
        validate_capacity_block_window("")
    with pytest.raises(ValueError, match="UTC offset"):
        validate_capacity_block_window("2099-01-01T00:00:00")
    with pytest.raises(ValueError, match="one hour"):
        validate_capacity_block_window("2000-01-01T00:00:00Z")
