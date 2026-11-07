"""Shared test fixtures.

Full-model integration tests that need pretrained weights carry the
'integration' marker.
"""

import pytest
import torch


@pytest.fixture(scope="session")
def device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
