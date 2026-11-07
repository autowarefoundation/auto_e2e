import math

import torch

from training import localization_noise as noise


def _history(batch: int, speed: float = 10.0, yaw_rate: float = 0.0) -> torch.Tensor:
    signals = torch.zeros(batch, noise.HISTORY_STEPS, noise.SIGNALS)
    signals[..., 0] = speed
    signals[..., 2] = yaw_rate
    signals[..., 3] = yaw_rate / speed
    return signals.reshape(batch, -1)


def test_shapes_and_shares():
    estimate = noise.simulate_localization(_history(4000), torch.Generator().manual_seed(0))
    assert estimate.pose.shape == (4000, 3)
    assert estimate.covariance.shape == (4000, 3, 3)
    assert abs(float(estimate.map_matching.float().mean()) - noise.MAP_MATCHING_SHARE) < 0.03
    assert torch.isfinite(estimate.pose).all() and torch.isfinite(estimate.covariance).all()
    # Covariances stay symmetric positive definite.
    assert torch.allclose(estimate.covariance, estimate.covariance.transpose(1, 2), atol=1e-4)
    assert (torch.linalg.eigvalsh(estimate.covariance) > 0).all()


def test_errors_have_the_designed_scale():
    estimate = noise.simulate_localization(_history(4000), torch.Generator().manual_seed(1))
    distance = estimate.pose[:, :2].norm(dim=-1)
    # Nominal GNSS keeps most estimates within a couple of metres, while
    # jumps and outages leave a tail of metre-scale errors.
    assert 0.2 < float(distance.median()) < 2.0
    assert float((distance > 2.0).float().mean()) > 0.05
    assert float(estimate.pose[:, 2].abs().max()) < math.radians(20.0)


def test_same_generator_state_is_reproducible():
    history = _history(8, yaw_rate=0.1)
    first = noise.simulate_localization(history, torch.Generator().manual_seed(3))
    second = noise.simulate_localization(history, torch.Generator().manual_seed(3))
    assert torch.equal(first.pose, second.pose) and torch.equal(first.map_matching, second.map_matching)
