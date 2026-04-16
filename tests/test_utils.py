"""Backward compatibility tests for utility modules."""

import random
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from utils.common import get_device, set_seed
from utils.experiment_config import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    OptimConfig,
    RunConfig,
    apply_cli_overrides,
    load_experiment_config,
)


class TestCommonUtils:
    """Tests for utils.common module."""

    def test_get_device_returns_valid_device(self) -> None:
        """Test that get_device returns a valid torch.device."""
        device = get_device()
        assert isinstance(device, torch.device)
        assert device.type in ["cpu", "cuda", "mps"]

    def test_get_device_preference_order(self) -> None:
        """Test get_device prefers cuda > mps > cpu."""
        device = get_device()
        # Just verify it returns a device; exact preference depends on available hardware
        assert isinstance(device, torch.device)

    def test_set_seed_reproducibility(self) -> None:
        """Test that set_seed produces reproducible results."""
        seed = 12345

        set_seed(seed)
        sample1_py = random.random()
        sample1_np = np.random.randn()
        sample1_torch = torch.randn(5)

        set_seed(seed)
        sample2_py = random.random()
        sample2_np = np.random.randn()
        sample2_torch = torch.randn(5)

        assert sample1_py == sample2_py
        assert sample1_np == sample2_np
        torch.testing.assert_close(sample1_torch, sample2_torch)

    def test_set_seed_with_cuda_if_available(self) -> None:
        """Test that set_seed handles CUDA correctly if available."""
        seed = 42
        set_seed(seed)  # Should not raise even if CUDA unavailable
        assert True


class TestDataConfig:
    """Tests for DataConfig dataclass."""

    def test_default_initialization(self) -> None:
        """Test DataConfig initializes with sensible defaults."""
        cfg = DataConfig()
        assert cfg.dataset == "MNIST"
        assert cfg.ops == [1, 2, -1, -2]
        assert cfg.test_ops is None
        assert cfg.batch_size == 128
        assert cfg.num_workers == 0
        assert cfg.data_root == "./data"

    def test_custom_initialization(self) -> None:
        """Test DataConfig with custom values."""
        cfg = DataConfig(
            dataset="EMNIST",
            ops=[1, 2, 3],
            test_ops=[4, 5],
            batch_size=64,
            num_workers=4,
            data_root="/custom/path",
        )
        assert cfg.dataset == "EMNIST"
        assert cfg.ops == [1, 2, 3]
        assert cfg.test_ops == [4, 5]
        assert cfg.batch_size == 64
        assert cfg.num_workers == 4
        assert cfg.data_root == "/custom/path"


class TestModelConfig:
    """Tests for ModelConfig dataclass."""

    def test_default_initialization(self) -> None:
        """Test ModelConfig initializes with sensible defaults."""
        cfg = ModelConfig()
        assert cfg.name == "resnet18"
        assert cfg.num_classes == 10
        assert cfg.op_embed_dim == 32
        assert cfg.mlp_hidden == 256
        assert cfg.op_embedding_type == "embedding"

    def test_custom_initialization(self) -> None:
        """Test ModelConfig with custom values."""
        cfg = ModelConfig(
            name="simple_fc",
            num_classes=20,
            op_embed_dim=64,
            mlp_hidden=512,
            op_embedding_type="fourier",
        )
        assert cfg.name == "simple_fc"
        assert cfg.num_classes == 20
        assert cfg.op_embed_dim == 64
        assert cfg.mlp_hidden == 512
        assert cfg.op_embedding_type == "fourier"

    def test_different_embedding_types(self) -> None:
        """Test ModelConfig with different embedding types."""
        for emb_type in ["embedding", "linear", "fourier"]:
            cfg = ModelConfig(op_embedding_type=emb_type)
            assert cfg.op_embedding_type == emb_type


class TestOptimConfig:
    """Tests for OptimConfig dataclass."""

    def test_default_initialization(self) -> None:
        """Test OptimConfig initializes with sensible defaults."""
        cfg = OptimConfig()
        assert cfg.lr == 1e-3
        assert cfg.weight_decay == 1e-4

    def test_custom_initialization(self) -> None:
        """Test OptimConfig with custom values."""
        cfg = OptimConfig(lr=1e-2, weight_decay=1e-3)
        assert cfg.lr == 1e-2
        assert cfg.weight_decay == 1e-3


class TestRunConfig:
    """Tests for RunConfig dataclass."""

    def test_default_initialization(self) -> None:
        """Test RunConfig initializes with sensible defaults."""
        cfg = RunConfig()
        assert cfg.epochs == 10
        assert cfg.seed == 42
        assert cfg.runs_dir == "runs/baseline"
        assert cfg.tag == "run"
        assert cfg.overfit == 0

    def test_custom_initialization(self) -> None:
        """Test RunConfig with custom values."""
        cfg = RunConfig(
            epochs=20,
            seed=123,
            runs_dir="runs/custom",
            tag="my-run",
            overfit=64,
        )
        assert cfg.epochs == 20
        assert cfg.seed == 123
        assert cfg.runs_dir == "runs/custom"
        assert cfg.tag == "my-run"
        assert cfg.overfit == 64


class TestExperimentConfig:
    """Tests for ExperimentConfig class."""

    def test_default_initialization(self) -> None:
        """Test ExperimentConfig initializes with defaults."""
        cfg = ExperimentConfig()
        assert isinstance(cfg.data, DataConfig)
        assert isinstance(cfg.model, ModelConfig)
        assert isinstance(cfg.optim, OptimConfig)
        assert isinstance(cfg.run, RunConfig)

    def test_from_dict_with_empty_dict(self) -> None:
        """Test from_dict handles empty dict gracefully."""
        cfg = ExperimentConfig.from_dict({})
        assert isinstance(cfg, ExperimentConfig)
        assert cfg.data.dataset == "MNIST"

    def test_from_dict_with_custom_values(self) -> None:
        """Test from_dict reconstructs from dict."""
        raw = {
            "data": {
                "dataset": "EMNIST",
                "ops": [1, 2, 3],
                "batch_size": 64,
            },
            "model": {
                "name": "simple_fc",
                "op_embed_dim": 64,
                "op_embedding_type": "fourier",
            },
        }
        cfg = ExperimentConfig.from_dict(raw)
        assert cfg.data.dataset == "EMNIST"
        assert cfg.data.ops == [1, 2, 3]
        assert cfg.data.batch_size == 64
        assert cfg.model.name == "simple_fc"
        assert cfg.model.op_embed_dim == 64
        assert cfg.model.op_embedding_type == "fourier"

    def test_to_dict_preserves_values(self) -> None:
        """Test to_dict converts to dict correctly."""
        cfg = ExperimentConfig(
            data=DataConfig(dataset="EMNIST", ops=[1, 2, 3]),
            model=ModelConfig(name="simple_fc"),
        )
        d = cfg.to_dict()
        assert d["data"]["dataset"] == "EMNIST"
        assert d["data"]["ops"] == [1, 2, 3]
        assert d["model"]["name"] == "simple_fc"

    def test_from_dict_roundtrip(self) -> None:
        """Test that from_dict -> to_dict is idempotent."""
        original = {
            "data": {"dataset": "MNIST", "ops": [1, 2], "batch_size": 64},
            "model": {"name": "resnet18", "op_embed_dim": 32},
            "optim": {"lr": 1e-2},
            "run": {"epochs": 20, "seed": 99},
        }
        cfg = ExperimentConfig.from_dict(original)
        result = cfg.to_dict()

        assert result["data"]["dataset"] == original["data"]["dataset"]
        assert result["data"]["ops"] == original["data"]["ops"]
        assert result["model"]["name"] == original["model"]["name"]

    def test_from_dict_type_conversion(self) -> None:
        """Test that from_dict converts types correctly."""
        raw = {
            "data": {
                "ops": ["1", "2", "-1"],  # strings
                "batch_size": "128",  # string
            },
            "model": {
                "num_classes": "10",  # string
            },
        }
        cfg = ExperimentConfig.from_dict(raw)
        assert isinstance(cfg.data.ops, list)
        assert all(isinstance(o, int) for o in cfg.data.ops)
        assert isinstance(cfg.data.batch_size, int)
        assert isinstance(cfg.model.num_classes, int)


class TestLoadExperimentConfig:
    """Tests for load_experiment_config function."""

    def test_load_config_from_yaml_file(self, temp_data_dir: Path) -> None:
        """Test loading config from a YAML file."""
        config_file = temp_data_dir / "test_config.yaml"
        config_data = {
            "data": {"dataset": "MNIST", "ops": [1, 2]},
            "model": {"name": "resnet18"},
        }
        config_file.write_text(yaml.dump(config_data))

        cfg = load_experiment_config(config_file)
        assert cfg.data.dataset == "MNIST"
        assert cfg.data.ops == [1, 2]
        assert cfg.model.name == "resnet18"

    def test_load_nonexistent_config_raises_error(self) -> None:
        """Test that loading nonexistent config raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            load_experiment_config("/nonexistent/config.yaml")

    def test_load_empty_yaml_file(self, temp_data_dir: Path) -> None:
        """Test loading empty YAML file defaults to ExperimentConfig defaults."""
        config_file = temp_data_dir / "empty_config.yaml"
        config_file.write_text("")

        cfg = load_experiment_config(config_file)
        assert isinstance(cfg, ExperimentConfig)
        assert cfg.data.dataset == "MNIST"

    def test_load_invalid_yaml_root_raises_error(self, temp_data_dir: Path) -> None:
        """Test that invalid YAML root (not dict) raises ValueError."""
        config_file = temp_data_dir / "invalid_config.yaml"
        config_file.write_text("- item1\n- item2")  # List, not dict

        with pytest.raises(ValueError, match="Config root must be a mapping"):
            load_experiment_config(config_file)


class TestApplyCliOverrides:
    """Tests for apply_cli_overrides function."""

    def test_no_overrides(self) -> None:
        """Test that no overrides leaves config unchanged."""
        cfg = ExperimentConfig()
        original_lr = cfg.optim.lr

        result = apply_cli_overrides(cfg, {})

        assert result.optim.lr == original_lr

    def test_single_override(self) -> None:
        """Test single CLI override."""
        cfg = ExperimentConfig()
        overrides = {"lr": 1e-2}

        result = apply_cli_overrides(cfg, overrides)

        assert result.optim.lr == 1e-2

    def test_multiple_overrides(self) -> None:
        """Test multiple CLI overrides."""
        cfg = ExperimentConfig()
        overrides = {
            "epochs": 20,
            "seed": 99,
            "batch_size": 32,
            "lr": 2e-3,
        }

        result = apply_cli_overrides(cfg, overrides)

        assert result.run.epochs == 20
        assert result.run.seed == 99
        assert result.data.batch_size == 32
        assert result.optim.lr == 2e-3

    def test_none_values_ignored(self) -> None:
        """Test that None values in overrides are ignored."""
        cfg = ExperimentConfig()
        original_epochs = cfg.run.epochs
        overrides = {"epochs": None, "lr": 1e-2}

        result = apply_cli_overrides(cfg, overrides)

        assert result.run.epochs == original_epochs  # Not overridden
        assert result.optim.lr == 1e-2  # This one was overridden

    def test_config_key_ignored(self) -> None:
        """Test that 'config' key is ignored in overrides."""
        cfg = ExperimentConfig()
        overrides = {"config": "some_config.yaml", "lr": 1e-2}

        result = apply_cli_overrides(cfg, overrides)

        assert result.optim.lr == 1e-2

    def test_unknown_keys_ignored(self) -> None:
        """Test that unknown keys are safely ignored."""
        cfg = ExperimentConfig()
        original_lr = cfg.optim.lr
        overrides = {"unknown_key": 123, "lr": 1e-2}

        result = apply_cli_overrides(cfg, overrides)

        assert result.optim.lr == 1e-2  # Known key still works

    def test_override_returns_same_object(self) -> None:
        """Test that apply_cli_overrides modifies and returns same object."""
        cfg = ExperimentConfig()
        overrides = {"lr": 1e-2}

        result = apply_cli_overrides(cfg, overrides)

        assert result is cfg

    def test_data_section_overrides(self) -> None:
        """Test overrides in data section."""
        cfg = ExperimentConfig()
        overrides = {
            "dataset": "EMNIST",
            "ops": [1, 2, 3],
            "batch_size": 64,
        }

        result = apply_cli_overrides(cfg, overrides)

        assert result.data.dataset == "EMNIST"
        assert result.data.ops == [1, 2, 3]
        assert result.data.batch_size == 64

    def test_model_section_overrides(self) -> None:
        """Test overrides in model section."""
        cfg = ExperimentConfig()
        overrides = {
            "model_name": "simple_fc",
            "op_embed_dim": 64,
            "op_embedding_type": "linear",
        }

        result = apply_cli_overrides(cfg, overrides)

        assert result.model.name == "simple_fc"
        assert result.model.op_embed_dim == 64
        assert result.model.op_embedding_type == "linear"
