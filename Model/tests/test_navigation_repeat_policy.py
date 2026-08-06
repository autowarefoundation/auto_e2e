"""Tests for deterministic navigation-aware training exposure."""

import hashlib
import json

import pytest

pytest.importorskip("webdataset")
import webdataset as wds

from data_parsing.pre_extracted import (
    NavigationRepeatPolicy,
    discover_navigation_exposure,
    make_pre_extracted_loader,
)


def _sample(
    sample_uid,
    *,
    group_uid="train",
    route_valid=True,
    maneuver="straight",
    intersection=False,
):
    return {
        "__key__": sample_uid,
        "meta.json": json.dumps({
            "sample_uid": sample_uid,
            "split_group_uid": group_uid,
        }).encode(),
        "navigation_meta.json": json.dumps({
            "route_valid": route_valid,
            "route_maneuver": maneuver,
            "route_intersection": intersection,
        }).encode(),
    }


def _write_shard(path, samples):
    with wds.TarWriter(str(path / "shard-000000.tar")) as sink:
        for sample in samples:
            sink.write(sample)


def test_repeat_policy_uses_max_and_requires_valid_route():
    policy = NavigationRepeatPolicy()

    assert policy.repeat_count({
        "route_valid": True,
        "route_maneuver": "left",
        "route_intersection": True,
    }) == 4
    assert policy.repeat_count({
        "route_valid": True,
        "route_maneuver": "straight",
        "route_intersection": True,
    }) == 2
    assert policy.repeat_count({
        "route_valid": False,
        "route_maneuver": "right",
        "route_intersection": True,
    }) == 1


def test_raw_repeat_stage_preserves_deterministic_exposure():
    policy = NavigationRepeatPolicy()
    samples = [
        _sample("left", maneuver="left", intersection=True),
        _sample("junction", intersection=True),
        _sample("invalid", route_valid=False, maneuver="right"),
    ]

    repeated = [sample["__key__"] for sample in policy(iter(samples))]

    assert repeated == (
        ["left"] * 4 + ["junction"] * 2 + ["invalid"]
    )


def test_repeat_policy_composes_with_webdataset_tar_reader(tmp_path):
    policy = NavigationRepeatPolicy()
    _write_shard(tmp_path, [
        _sample("left", maneuver="left"),
        _sample("straight"),
    ])
    dataset = wds.WebDataset(
        [str(tmp_path / "shard-000000.tar")],
        shardshuffle=False,
        empty_check=False,
        nodesplitter=wds.single_node_only,
    ).compose(policy)

    assert [sample["__key__"] for sample in dataset] == (
        ["left"] * 4 + ["straight"]
    )


def test_exposure_audit_excludes_validation_and_has_stable_digest(tmp_path):
    policy = NavigationRepeatPolicy()
    _write_shard(tmp_path, [
        _sample("z-invalid", route_valid=False, maneuver="right"),
        _sample("a-left", maneuver="left", intersection=True),
        _sample(
            "m-validation",
            group_uid="validation",
            intersection=True,
        ),
    ])

    audit = discover_navigation_exposure(
        [tmp_path],
        policy=policy,
        validation_group_uids=["validation"],
    )

    expected_payload = b"a-left\t4\nz-invalid\t1\n"
    assert audit.unique_sample_count == 2
    assert audit.effective_exposure_count == 5
    assert audit.route_valid_sample_count == 1
    assert audit.maneuver_unique_counts == {
        "left": 1,
        "route_invalid": 1,
    }
    assert audit.maneuver_exposure_counts == {
        "left": 4,
        "route_invalid": 1,
    }
    assert audit.junction_unique_counts == {
        "junction": 1,
        "non_junction": 1,
    }
    assert audit.exposure_digest == hashlib.sha256(
        expected_payload
    ).hexdigest()
    assert discover_navigation_exposure(
        [tmp_path],
        policy=policy,
        validation_group_uids=["validation"],
    ).exposure_digest == audit.exposure_digest


def test_repeat_policy_cannot_modify_validation_distribution(tmp_path):
    _write_shard(tmp_path, [_sample("sample")])

    with pytest.raises(ValueError, match="only for the train split"):
        make_pre_extracted_loader(
            str(tmp_path),
            split="val",
            val_fraction=0.1,
            navigation_repeat_policy=NavigationRepeatPolicy(),
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"version": "unknown"},
        {"turn_repeat": 5},
        {"junction_repeat": 0},
    ],
)
def test_invalid_repeat_policy_is_rejected(kwargs):
    with pytest.raises(ValueError):
        NavigationRepeatPolicy(**kwargs)
