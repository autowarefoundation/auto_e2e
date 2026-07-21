"""Tests for narrow overlay runtime identities."""

from pathlib import Path

import pytest

from Platform.pipelines.reproducibility import (
    container_digest_from_image,
    inference_contract_files,
    model_inference_code_digest,
    preprocessing_contract_digest,
    sha256_files,
    validate_runtime_contract,
)


ROOT = Path(__file__).resolve().parents[2]


def test_contract_digests_are_stable_sha256_values():
    preprocessing = preprocessing_contract_digest(ROOT)
    inference = model_inference_code_digest(ROOT)
    assert len(preprocessing) == len(inference) == 64
    assert preprocessing == preprocessing_contract_digest(ROOT)
    assert inference == model_inference_code_digest(ROOT)
    assert ROOT / "Platform/pipelines/inference.py" in (
        inference_contract_files(ROOT)
    )


def test_file_identity_covers_path_and_content(tmp_path):
    first = tmp_path / "first.py"
    first.write_text("value = 1\n")
    original = sha256_files([first], root=tmp_path)

    first.write_text("value = 2\n")
    assert sha256_files([first], root=tmp_path) != original

    renamed = tmp_path / "renamed.py"
    first.rename(renamed)
    assert sha256_files([renamed], root=tmp_path) != original


def test_runtime_contract_requires_the_running_image_digest():
    preprocessing = preprocessing_contract_digest(ROOT)
    inference = model_inference_code_digest(ROOT)
    image = "example.test/auto-e2e/eval@sha256:" + "a" * 64
    assert validate_runtime_contract(
        preprocessing_digest=preprocessing,
        inference_code_digest=inference,
        container_image_digest="sha256:" + "a" * 64,
        task_image=image,
        root=ROOT,
    ) == "sha256:" + "a" * 64

    with pytest.raises(ValueError, match="container_image_digest"):
        validate_runtime_contract(
            preprocessing_digest=preprocessing,
            inference_code_digest=inference,
            container_image_digest="sha256:" + "b" * 64,
            task_image=image,
            root=ROOT,
        )


def test_mutable_image_tags_are_rejected():
    with pytest.raises(ValueError, match="pinned"):
        container_digest_from_image("example.test/auto-e2e/eval:latest")
