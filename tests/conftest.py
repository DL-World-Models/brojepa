"""Shared test fixtures and configuration."""

import tempfile
from pathlib import Path

import pytest
import torch
from torchvision import datasets, transforms


@pytest.fixture
def dummy_mnist_dataset():
    """Provides actual MNIST training dataset for testing.
    
    Uses 60,000 real MNIST training samples (auto-downloads on first run).
    Includes standard normalization (mean=0.1307, std=0.3081).
    """
    t = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    # Use actual MNIST data, train split
    mnist = datasets.MNIST(root="./data", train=True, download=True, transform=t)
    return mnist


@pytest.fixture
def temp_data_dir() -> Path:
    """Provides a temporary directory for test data."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def device() -> torch.device:
    """Provides the appropriate device for testing (CPU for consistency)."""
    return torch.device("cpu")


@pytest.fixture
def batch_data(device: torch.device) -> dict[str, torch.Tensor]:
    """Provides a sample batch for model testing."""
    batch_size = 4
    return {
        "ctx_img": torch.randn(batch_size, 1, 28, 28, device=device),
        "ctx_label": torch.randint(0, 10, (batch_size,), device=device),
        "op": torch.tensor([1.0, 2.0, -1.0, -2.0], device=device),
        "tgt_label": torch.randint(0, 10, (batch_size,), device=device),
    }
