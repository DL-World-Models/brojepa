"""Late-fusion ResNet-18 baseline for the MNIST-arithmetic task.

Given a context digit image and an arithmetic operation (e.g. +1, -2), predict
the resulting digit class directly. This is the non-JEPA baseline we pit
against the JEPA world model: same encoder, different training paradigm.
"""

from collections.abc import Sequence

import torch
import torch.nn as nn
from torchvision.models import resnet18


def _build_mnist_resnet18() -> nn.Module:
    """ResNet-18 adapted for 1-channel 28x28 inputs.

    Standard ImageNet ResNet-18 starts with a 7x7 stride-2 conv + 3x3 stride-2
    maxpool, which collapses a 28x28 input to 7x7 before any residual block
    ever runs. We follow the common CIFAR/MNIST adaptation: replace conv1 with
    a 3x3 stride-1 1-channel conv, and drop the initial maxpool. Everything
    downstream is untouched, so parameter count stays ~11M and the feature
    pipeline is still a legitimate "ResNet-18".
    """
    m = resnet18(weights=None)
    m.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    m.fc = nn.Identity()  # expose the 512-d penultimate features directly
    return m


class ResNet18LateFusion(nn.Module):
    """Late-fusion baseline: ResNet-18 image features + op embedding -> MLP.

    - image branch: grayscale-adapted ResNet-18 -> 512-d features
    - op branch:    nn.Embedding over the set of allowed integer operations
    - head:         concat -> 2-layer MLP -> num_classes logits

    The set of allowed operations is fixed at construction time and stored as
    a buffer, so the model is a self-contained artifact: at forward time we
    accept raw integer op values (as produced by data_pipeline.ArithmeticDataset)
    and look up the embedding index internally.
    """

    FEAT_DIM = 512

    def __init__(
        self,
        ops: Sequence[int],
        num_classes: int = 10,
        op_embed_dim: int = 32,
        mlp_hidden: int = 256,
    ) -> None:
        super().__init__()

        op_list = [int(o) for o in ops]
        if len(set(op_list)) != len(op_list):
            raise ValueError(f"ops must be unique, got {ops}")
        sorted_ops = sorted(op_list)

        # Buffer (not parameter) so it moves with .to(device) but isn't trained.
        self.register_buffer("op_values", torch.tensor(sorted_ops, dtype=torch.long))

        self.encoder = _build_mnist_resnet18()
        self.op_embed = nn.Embedding(len(sorted_ops), op_embed_dim)
        self.head = nn.Sequential(
            nn.Linear(self.FEAT_DIM + op_embed_dim, mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden, num_classes),
        )

    def _op_vals_to_idx(self, op_vals: torch.Tensor) -> torch.Tensor:
        op_vals = op_vals.long().view(-1, 1)
        matches = op_vals == self.op_values.view(1, -1)  # [B, K]
        if not torch.all(matches.any(dim=1)):
            missing = op_vals[~matches.any(dim=1)].flatten().tolist()
            raise ValueError(f"unknown op values: {missing}")
        return matches.long().argmax(dim=1)

    def forward(self, img: torch.Tensor, op_vals: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(img)  # [B, 512]
        op_idx = self._op_vals_to_idx(op_vals)  # [B]
        op_vec = self.op_embed(op_idx)  # [B, op_embed_dim]
        fused = torch.cat([feats, op_vec], dim=1)  # [B, 512 + op_embed_dim]
        return self.head(fused)  # [B, num_classes]
