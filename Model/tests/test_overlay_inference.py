import hashlib
import inspect

import numpy as np
import pytest
import torch

from Platform.pipelines.inference import (
    load_policy,
    noise_from,
    predict_control,
    sha256_file,
    stable_seed64,
)
from Platform.pipelines.overlay_precompute import infer_loader_controls
from training.dataset_policy import KITSCENES_TRAINING_POLICY


def test_sha256_file_streams_expected_digest(tmp_path):
    payload = (b"auto-e2e-overlay" * 4096) + b"tail"
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(payload)

    assert sha256_file(checkpoint, chunk_size=97) == hashlib.sha256(payload).hexdigest()


def test_stable_seed_and_noise_are_identity_scoped():
    identity = ("model-sha", "manifest-sha", "l2d-v1-e000001-f000064", 0)
    assert stable_seed64(*identity) == stable_seed64(*identity)
    assert stable_seed64(*identity) != stable_seed64(
        "model-sha", "manifest-sha", "l2d-v1-e000001-f000065", 0
    )

    first = noise_from(*identity, shape=(128,), device="cpu")
    second = noise_from(*identity, shape=(128,), device="cpu")
    other = noise_from(
        "model-sha",
        "manifest-sha",
        "l2d-v1-e000001-f000065",
        0,
        shape=(128,),
        device="cpu",
    )
    assert torch.equal(first, second)
    assert not torch.equal(first, other)


class _FakePlanner:
    num_timesteps = 64
    num_signals = 2


class _FakeReactive:
    TrajectoryPlanner = _FakePlanner()


class _NoiseEchoPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.Reactive_E2E = _FakeReactive()
        self.reset_count = 0
        self.last_egomotion_history = None

    def reset_visual_history(self):
        self.reset_count += 1

    def forward(
        self,
        camera_tiles,
        map_input,
        visual_history,
        egomotion_history,
        *,
        initial_noise,
        **kwargs,
    ):
        self.last_egomotion_history = egomotion_history.detach().clone()
        return initial_noise + self.anchor


def _batch(size):
    return {
        "visual_tiles": torch.zeros(size, 2, 3, 4, 4),
        "map_input": torch.zeros(size, 3, 4, 4),
        "visual_history": torch.zeros(size, 896),
        "egomotion_history": torch.zeros(size, 256),
    }


def test_predict_control_is_uid_stable_across_batch_order():
    model = _NoiseEchoPolicy().eval()
    uids = ["l2d-v1-e000001-f000064", "l2d-v1-e000002-f000064"]

    original = predict_control(
        model,
        _batch(2),
        sample_uids=uids,
        model_artifact_id="model-sha",
        dataset_manifest_digest="manifest-sha",
    )
    reordered = predict_control(
        model,
        _batch(2),
        sample_uids=list(reversed(uids)),
        model_artifact_id="model-sha",
        dataset_manifest_digest="manifest-sha",
    )

    assert original.shape == (2, 64, 2)
    np.testing.assert_array_equal(original[0], reordered[1])
    np.testing.assert_array_equal(original[1], reordered[0])
    assert model.reset_count == 2


def test_predict_control_rejects_uid_count_mismatch():
    with pytest.raises(ValueError, match="sample_uids length"):
        predict_control(
            _NoiseEchoPolicy(),
            _batch(2),
            sample_uids=["only-one"],
            model_artifact_id="model-sha",
            dataset_manifest_digest="manifest-sha",
        )


def test_infer_loader_controls_emits_seed_fan_and_v0():
    model = _NoiseEchoPolicy().eval()
    first = _batch(2)
    first["sample_uid"] = [
        "l2d-v1-e000001-f000064",
        "l2d-v1-e000001-f000065",
    ]
    first["egomotion_history"][:, -4] = torch.tensor([3.0, 4.0])
    second = _batch(1)
    second["sample_uid"] = ["l2d-v1-e000001-f000066"]
    second["egomotion_history"][:, -4] = 5.0

    class Loader(list):
        projection = None
        geometry_type = "pseudo"

    uids, controls, v0, seeds = infer_loader_controls(
        model,
        Loader([first, second]),
        model_artifact_id="model-sha",
        dataset_manifest_digest="manifest-sha",
        base_seeds=(0, 1),
        device="cpu",
    )

    assert uids == first["sample_uid"] + second["sample_uid"]
    assert controls.shape == (3, 2, 64, 2)
    np.testing.assert_array_equal(v0, [3.0, 4.0, 5.0])
    assert seeds == (0, 1)
    assert not np.array_equal(controls[:, 0], controls[:, 1])


def test_overlay_inference_applies_checkpoint_data_sanitization():
    model = _NoiseEchoPolicy().eval()
    batch = _batch(1)
    batch["sample_uid"] = ["kitscenes-v1-scene-a-f000064"]
    history = batch["egomotion_history"].reshape(1, 64, 4)
    history[:] = 1.0
    history[:, -1, 0] = 3.0

    class Loader(list):
        projection = None
        geometry_type = "pinhole"

    _, _, v0, _ = infer_loader_controls(
        model,
        Loader([batch]),
        model_artifact_id="model-sha",
        dataset_manifest_digest="manifest-sha",
        device="cpu",
        training_policy=KITSCENES_TRAINING_POLICY,
    )

    adapted = model.last_egomotion_history.reshape(1, 64, 4)
    assert torch.count_nonzero(adapted[:, :24]) == 24 * 4
    assert adapted[0, -1, 0].item() == 3.0
    assert adapted[0, -1, 1].item() == 0.0
    np.testing.assert_array_equal(v0, [3.0])


def test_overlay_task_never_decodes_future_images():
    pytest.importorskip("flytekit")
    from Platform.pipelines.overlay_tasks import precompute_overlay_partition

    source = inspect.getsource(
        precompute_overlay_partition.task_function
    )
    assert "decode_future_frames=False" in source


def test_load_policy_filters_config_and_returns_checkpoint_identity(
    tmp_path, monkeypatch
):
    from model_components import auto_e2e as auto_e2e_module

    class TinyPolicy(torch.nn.Module):
        def __init__(self, width=2):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(width))

    source = TinyPolicy(width=3)
    source.weight.data.copy_(torch.tensor([1.0, 2.0, 3.0]))
    checkpoint = tmp_path / "best.pt"
    torch.save(
        {
            "model_state_dict": source.state_dict(),
            "config": {"width": 3, "removed_argument": "ignored"},
            "epoch": 1,
        },
        checkpoint,
    )
    monkeypatch.setattr(auto_e2e_module, "AutoE2E", TinyPolicy)

    loaded, config, artifact_id = load_policy(checkpoint, "cpu")

    assert isinstance(loaded, TinyPolicy)
    assert loaded.training is False
    assert torch.equal(loaded.weight, source.weight)
    assert config["removed_argument"] == "ignored"
    assert artifact_id == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
