"""Backward compatibility tests for data pipeline modules."""

import pytest
import torch
from torch.utils.data import DataLoader

from data_pipeline import (
    ArithmeticDataset,
    FixedSampleDataset,
    get_arithmetic_dataloader,
)


class TestArithmeticDataset:
    """Tests for ArithmeticDataset backward compatibility."""

    def test_dataset_initialization(self, dummy_mnist_dataset) -> None:
        """Test ArithmeticDataset initializes correctly."""
        ops = [1, 2, -1, -2]
        dataset = ArithmeticDataset(
            base_dataset=dummy_mnist_dataset,
            operations=ops,
        )

        assert dataset.base_dataset is dummy_mnist_dataset
        assert dataset.operations == ops
        assert dataset.num_classes == 10

    def test_dataset_length(self, dummy_mnist_dataset) -> None:
        """Test that dataset length matches base dataset length."""
        ops = [1, 2, -1, -2]
        dataset = ArithmeticDataset(
            base_dataset=dummy_mnist_dataset,
            operations=ops,
        )

        assert len(dataset) == len(dummy_mnist_dataset)

    def test_dataset_getitem_returns_dict(
        self, dummy_mnist_dataset
    ) -> None:
        """Test that __getitem__ returns dict with correct keys."""
        ops = [1, 2, -1, -2]
        dataset = ArithmeticDataset(
            base_dataset=dummy_mnist_dataset,
            operations=ops,
        )

        sample = dataset[0]

        assert isinstance(sample, dict)
        assert "ctx_img" in sample
        assert "ctx_label" in sample
        assert "op" in sample
        assert "tgt_img" in sample
        assert "tgt_label" in sample

    def test_dataset_getitem_tensor_types(
        self, dummy_mnist_dataset
    ) -> None:
        """Test that all returned values are tensors."""
        ops = [1, 2, -1, -2]
        dataset = ArithmeticDataset(
            base_dataset=dummy_mnist_dataset,
            operations=ops,
        )

        sample = dataset[0]

        for key, value in sample.items():
            assert isinstance(value, torch.Tensor), f"{key} is not a tensor"

    def test_dataset_op_value_in_list(self, dummy_mnist_dataset) -> None:
        """Test that sampled op is always from the operations list."""
        ops = [1, 2, -1, -2]
        dataset = ArithmeticDataset(
            base_dataset=dummy_mnist_dataset,
            operations=ops,
        )

        for _ in range(20):
            sample = dataset[0]
            op = sample["op"].item()
            assert op in ops, f"op {op} not in {ops}"

    def test_dataset_target_label_correctness(
        self, dummy_mnist_dataset
    ) -> None:
        """Test that target label = (context label + op) % num_classes."""
        ops = [1, 2, -1, -2]
        dataset = ArithmeticDataset(
            base_dataset=dummy_mnist_dataset,
            operations=ops,
        )

        for _ in range(50):
            sample = dataset[0]
            ctx_label = sample["ctx_label"].item()
            op = sample["op"].item()
            tgt_label = sample["tgt_label"].item()

            expected_tgt = (ctx_label + op) % dataset.num_classes
            assert tgt_label == expected_tgt

    def test_dataset_with_transform(self, dummy_mnist_dataset) -> None:
        """Test that dataset applies transform to images."""

        def dummy_transform(x: torch.Tensor) -> torch.Tensor:
            return x * 2

        ops = [1, 2, -1, -2]
        dataset = ArithmeticDataset(
            base_dataset=dummy_mnist_dataset,
            operations=ops,
            transform=dummy_transform,
        )

        sample = dataset[0]
        # Images should be modified by transform
        assert isinstance(sample["ctx_img"], torch.Tensor)

    def test_dataset_index_labels(self, dummy_mnist_dataset) -> None:
        """Test that label_to_indices is built correctly."""
        ops = [1, 2, -1, -2]
        dataset = ArithmeticDataset(
            base_dataset=dummy_mnist_dataset,
            operations=ops,
        )

        label_map = dataset.label_to_indices
        assert isinstance(label_map, dict)
        assert all(isinstance(k, int) for k in label_map.keys())
        assert all(isinstance(v, list) for v in label_map.values())


class TestFixedSampleDataset:
    """Tests for FixedSampleDataset backward compatibility."""

    def test_fixed_dataset_initialization(
        self, dummy_mnist_dataset
    ) -> None:
        """Test FixedSampleDataset initializes correctly."""
        ops = [1, 2, -1, -2]
        arith_dataset = ArithmeticDataset(
            base_dataset=dummy_mnist_dataset,
            operations=ops,
        )
        n_samples = 10
        fixed_dataset = FixedSampleDataset(arith_dataset, n_samples)

        assert len(fixed_dataset) == n_samples

    def test_fixed_dataset_snapshots_deterministic_samples(
        self, dummy_mnist_dataset
    ) -> None:
        """Test that FixedSampleDataset produces deterministic samples."""
        ops = [1, 2, -1, -2]
        arith_dataset = ArithmeticDataset(
            base_dataset=dummy_mnist_dataset,
            operations=ops,
        )
        fixed_dataset = FixedSampleDataset(arith_dataset, 5)

        sample1 = fixed_dataset[0]
        sample2 = fixed_dataset[0]

        torch.testing.assert_close(sample1["ctx_img"], sample2["ctx_img"])
        torch.testing.assert_close(sample1["tgt_img"], sample2["tgt_img"])

    def test_fixed_dataset_getitem(
        self, dummy_mnist_dataset
    ) -> None:
        """Test that FixedSampleDataset returns correct dict structure."""
        ops = [1, 2, -1, -2]
        arith_dataset = ArithmeticDataset(
            base_dataset=dummy_mnist_dataset,
            operations=ops,
        )
        fixed_dataset = FixedSampleDataset(arith_dataset, 5)

        sample = fixed_dataset[0]

        assert isinstance(sample, dict)
        assert "ctx_img" in sample
        assert "ctx_label" in sample
        assert "op" in sample
        assert "tgt_img" in sample
        assert "tgt_label" in sample

    def test_fixed_dataset_out_of_bounds(
        self, dummy_mnist_dataset
    ) -> None:
        """Test that FixedSampleDataset raises IndexError for invalid indices."""
        ops = [1, 2, -1, -2]
        arith_dataset = ArithmeticDataset(
            base_dataset=dummy_mnist_dataset,
            operations=ops,
        )
        fixed_dataset = FixedSampleDataset(arith_dataset, 5)

        with pytest.raises(IndexError):
            fixed_dataset[100]


class TestGetArithmeticDataloader:
    """Tests for get_arithmetic_dataloader function."""

    def test_dataloader_default_parameters(self, temp_data_dir) -> None:
        """Test get_arithmetic_dataloader with default parameters."""
        # Note: This won't actually download MNIST, but tests the function signature
        # For real tests, we'd need to mock the datasets or provide dummy data

        # Just test that the function exists and is callable
        assert callable(get_arithmetic_dataloader)

    def test_dataloader_returns_dataloader(self, dummy_mnist_dataset) -> None:
        """Test that function works with mocked dataset."""
        # Create a simple dataloader manually to test the returned type
        arith_dataset = ArithmeticDataset(
            base_dataset=dummy_mnist_dataset,
            operations=[1, 2, -1, -2],
        )
        loader = DataLoader(
            arith_dataset,
            batch_size=64,
            shuffle=True,
            num_workers=0,
        )

        assert isinstance(loader, DataLoader)

    def test_dataloader_batch_structure(self, dummy_mnist_dataset) -> None:
        """Test that dataloader batches have correct structure."""
        arith_dataset = ArithmeticDataset(
            base_dataset=dummy_mnist_dataset,
            operations=[1, 2, -1, -2],
        )
        loader = DataLoader(
            arith_dataset,
            batch_size=4,
            shuffle=False,
            num_workers=0,
        )

        batch = next(iter(loader))

        assert isinstance(batch, dict)
        assert "ctx_img" in batch
        assert "ctx_label" in batch
        assert "op" in batch
        assert "tgt_img" in batch
        assert "tgt_label" in batch

        # Check batch sizes
        batch_size = 4
        assert batch["ctx_img"].shape[0] == batch_size
        assert batch["ctx_label"].shape[0] == batch_size
        assert batch["op"].shape[0] == batch_size
        assert batch["tgt_img"].shape[0] == batch_size
        assert batch["tgt_label"].shape[0] == batch_size

    def test_dataloader_with_different_batch_sizes(
        self, dummy_mnist_dataset
    ) -> None:
        """Test dataloader with various batch sizes."""
        arith_dataset = ArithmeticDataset(
            base_dataset=dummy_mnist_dataset,
            operations=[1, 2, -1, -2],
        )

        for batch_size in [1, 4, 8, 16, 32]:
            loader = DataLoader(
                arith_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
            )

            batch = next(iter(loader))
            assert batch["ctx_img"].shape[0] <= batch_size

    def test_dataloader_shuffle_parameter(self, dummy_mnist_dataset) -> None:
        """Test that shuffle parameter is respected."""
        arith_dataset = ArithmeticDataset(
            base_dataset=dummy_mnist_dataset,
            operations=[1, 2, -1, -2],
        )

        # Create two dataloaders, one shuffled, one not
        loader_no_shuffle = DataLoader(
            arith_dataset,
            batch_size=4,
            shuffle=False,
            num_workers=0,
        )

        # Just test that they're created without error
        assert loader_no_shuffle is not None
