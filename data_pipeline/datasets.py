import random
from collections.abc import Callable, Sequence
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


class ArithmeticDataset(Dataset[dict[str, torch.Tensor]]):
    """Returns arithmetic transition samples for MNIST-like datasets."""

    def __init__(
        self,
        base_dataset: Any,
        operations: Sequence[int],
        transform: Callable[[Any], torch.Tensor] | None = None,
    ) -> None:
        self.base_dataset = base_dataset
        self.operations = list(operations)
        self.transform = transform
        self.label_to_indices = self._index_labels()
        self.num_classes = len(self.label_to_indices.keys())

    def _index_labels(self) -> dict[int, list[int]]:
        idx_map: dict[int, list[int]] = {}
        for i in range(len(self.base_dataset)):
            _, label = self.base_dataset[i]
            if label not in idx_map:
                idx_map[label] = []
            idx_map[label].append(i)
        return idx_map

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ctx_img, ctx_label = self.base_dataset[idx]

        op_val = random.choice(self.operations)
        tgt_label = (ctx_label + op_val) % self.num_classes

        tgt_idx = random.choice(self.label_to_indices[tgt_label])
        tgt_img, _ = self.base_dataset[tgt_idx]

        if self.transform:
            ctx_img = self.transform(ctx_img)
            tgt_img = self.transform(tgt_img)

        return {
            "ctx_img": ctx_img,
            "ctx_label": torch.tensor(ctx_label),
            "op": torch.tensor(op_val, dtype=torch.float32),
            "tgt_img": tgt_img,
            "tgt_label": torch.tensor(tgt_label),
        }


class FixedSampleDataset(Dataset[dict[str, torch.Tensor]]):
    """Snapshots a deterministic subset for overfit sanity checks."""

    def __init__(self, base: Dataset[dict[str, torch.Tensor]], n: int) -> None:
        self.samples: list[dict[str, torch.Tensor]] = [base[i] for i in range(n)]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.samples[idx]


def get_arithmetic_dataloader(
    dataset_name: str = "MNIST",
    batch_size: int = 64,
    ops: Sequence[int] = (1, 2, -1, -2),
    is_train: bool = True,
    data_root: str = "./data",
    num_workers: int = 0,
) -> DataLoader[dict[str, torch.Tensor]]:
    t = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    if dataset_name.upper() == "MNIST":
        base = datasets.MNIST(root=data_root, train=is_train, download=True)
    elif dataset_name.upper() == "EMNIST":
        base = datasets.EMNIST(
            root=data_root, split="digits", train=is_train, download=True
        )
    else:
        raise ValueError("Only MNIST and EMNIST are currently supported.")

    arith_ds = ArithmeticDataset(base, operations=ops, transform=t)

    return DataLoader(
        arith_ds,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
    )
