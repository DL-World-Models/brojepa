"""Late-fusion ResNet-18 baseline for the MNIST-arithmetic task.

Given a context digit image and an arithmetic operation (e.g. +1, -2), predict
the resulting digit class directly. This is the non-JEPA baseline we pit
against the JEPA world model: same encoder, different training paradigm.

Supports multiple operation embedding strategies via configuration.
"""

from collections.abc import Sequence

import torch
import torch.nn as nn
from torchvision.models import resnet18

from models.op_embedding import create_operation_embedding, OpEmbeddingType
from models.rotation import ActionRotation


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
    """Late-fusion ResNet-18: image features + configurable op embedding -> MLP.

    - image branch: grayscale-adapted ResNet-18 -> 512-d features
    - op branch:    configurable embedding (embedding/linear/fourier)
    - head:         concat -> 2-layer MLP -> num_classes logits

    Args:
        ops: Sequence of allowed integer operations
        num_classes: Number of output classes (default 10)
        op_embed_dim: Dimension of operation embedding (default 32)
        mlp_hidden: Hidden layer size of MLP head (default 256)
        op_embedding_type: Type of operation embedding ("embedding", "linear", "fourier")
    """

    FEAT_DIM = 512

    def __init__(
        self,
        ops: Sequence[int],
        num_classes: int = 10,
        op_embed_dim: int = 32,
        mlp_hidden: int = 256,
        op_embedding_type: OpEmbeddingType = "embedding",
        predictor_type: str = "mlp",
        predictor_normalize: bool = False,
        rotation_angle_init: float = 0.6283185307179586,
        rotation_learnable: bool = True,
        rotation_init_mode: str = "constant",
        rotation_period: float = 10.0,
        use_action_rotation: bool = False,
        include_op_embedding_in_head: bool = True,
    ) -> None:
        super().__init__()

        self.op_embedding_type = op_embedding_type
        self.predictor_type = predictor_type
        self.use_action_rotation = use_action_rotation
        self.include_op_embedding_in_head = include_op_embedding_in_head
        self.encoder = _build_mnist_resnet18()
        self.op_embed = create_operation_embedding(
            embedding_type=op_embedding_type,
            op_embed_dim=op_embed_dim,
            ops=list(ops) if op_embedding_type == "embedding" else None,
        )
        rotation_type = (
            predictor_type
            if use_action_rotation and predictor_type in {"block_rotation", "simple_rotation"}
            else "none"
        )
        self.action_rotation = ActionRotation(
            dim=self.FEAT_DIM,
            rotation_type=rotation_type,
            angle_init=rotation_angle_init,
            learnable=rotation_learnable,
            init_mode=rotation_init_mode,
            period=rotation_period,
            normalize=predictor_normalize,
        )
        head_in_dim = self.FEAT_DIM + op_embed_dim if include_op_embedding_in_head else self.FEAT_DIM
        self.head = nn.Sequential(
            nn.Linear(head_in_dim, mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden, num_classes),
        )

    def forward(
        self,
        img: torch.Tensor,
        op_vals: torch.Tensor,
        return_features: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        feats = self.encoder(img)  # [B, 512]
        feats = self.action_rotation(feats, op_vals)
        rotation_features = feats
        if self.include_op_embedding_in_head:
            op_vec = self.op_embed(op_vals)  # [B, op_embed_dim]
            feats = torch.cat([feats, op_vec], dim=1)  # [B, 512 + op_embed_dim]
        logits = self.head(feats)  # [B, num_classes]
        return (logits, rotation_features) if return_features else logits
