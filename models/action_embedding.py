"""Operation embedding strategies for MNIST arithmetic models.

Provides different ways to embed integer operation values:
- embedding: Fixed lookup table for discrete operations
- linear: Linear projection of raw scalar values
- fourier: Fourier features + projection for smooth generalization
"""

from typing import Literal

import torch
import torch.nn as nn

OpEmbeddingType = Literal["embedding", "linear", "fourier"]


class BaseOperationEmbedding(nn.Module):
    """Base class for operation embedding strategies."""

    def __init__(self, op_embed_dim: int) -> None:
        super().__init__()
        self.op_embed_dim = op_embed_dim

    def forward(self, op_vals: torch.Tensor) -> torch.Tensor:
        """Embed operation values to fixed dimension.

        Args:
            op_vals: Tensor of operation values, shape [B]

        Returns:
            Embedded operations, shape [B, op_embed_dim]
        """
        raise NotImplementedError


class OperationEmbeddingLookup(BaseOperationEmbedding):
    """Fixed embedding lookup for discrete operations.

    Requires a finite set of known operations. Operations must be unique.
    Good for fixed-operation settings where you want discrete embeddings.
    """

    def __init__(self, op_embed_dim: int, ops: list[int] = [1, 2]) -> None:
        super().__init__(op_embed_dim)

        if len(set(ops)) != len(ops):
            raise ValueError(f"ops must be unique for embedding mode, got {ops}")

        sorted_ops = sorted(ops)
        self.register_buffer("op_values", torch.tensor(sorted_ops, dtype=torch.long))
        self.op_embed = nn.Embedding(len(sorted_ops), op_embed_dim)

    def forward(self, op_vals: torch.Tensor) -> torch.Tensor:
        op_vals = op_vals.long().view(-1, 1)
        matches = op_vals == self.op_values.view(1, -1)  # [B, K]
        if not torch.all(matches.any(dim=1)):
            missing = op_vals[~matches.any(dim=1)].flatten().tolist()
            raise ValueError(f"unknown op values: {missing}")
        idx = matches.long().argmax(dim=1)  # [B]
        return self.op_embed(idx)  # [B, op_embed_dim]


class OperationEmbeddingLinear(BaseOperationEmbedding):
    """Linear projection of scalar operation values.

    Maps any real-valued operation to a fixed embedding dimension.
    Simple and fast. Works with any operation values (not limited to discrete set).
    """

    def __init__(self, op_embed_dim: int) -> None:
        super().__init__(op_embed_dim)
        self.proj = nn.Sequential(
            nn.Linear(1, op_embed_dim),
            nn.ReLU(),
            nn.Linear(op_embed_dim, op_embed_dim),
        )

    def forward(self, op_vals: torch.Tensor) -> torch.Tensor:
        op_vals = op_vals.unsqueeze(-1).float()  # (B, 1)
        return self.proj(op_vals)  # [B, op_embed_dim]


class OperationEmbeddingFourier(BaseOperationEmbedding):
    """Fourier feature encoding of operation values.

    Encodes ops as sin/cos features at multiple frequencies, then projects.
    Provides smooth, continuous representation good for generalization.
    Features: sin(x), cos(x), sin(2x), cos(2x) -> 4 features -> projection
    """

    def __init__(self, op_embed_dim: int) -> None:
        super().__init__(op_embed_dim)
        # 4 Fourier features: sin(x), cos(x), sin(2x), cos(2x)
        self.proj = nn.Sequential(
            nn.Linear(4, op_embed_dim),
            nn.ReLU(),
            nn.Linear(op_embed_dim, op_embed_dim),
        )

    def forward(self, op_vals: torch.Tensor) -> torch.Tensor:
        op_vals = op_vals.unsqueeze(-1).float()  # (B, 1)
        fourier_feat = torch.cat([
            torch.sin(op_vals),
            torch.cos(op_vals),
            torch.sin(2 * op_vals),
            torch.cos(2 * op_vals),
        ], dim=-1)  # [B, 4]
        return self.proj(fourier_feat)  # [B, op_embed_dim]

class OperationEmbedding(nn.Module):
    def __init__(self, embedding_type: OpEmbeddingType, op_embed_dim: int, **kwargs) -> torch.Tensor:
        super().__init__()
        embedding_type = embedding_type.lower()
        embedding_map = {"embedding": OperationEmbeddingLookup,
                         "linear": OperationEmbeddingLinear,
                         "fourier": OperationEmbeddingFourier}
        if embedding_type not in embedding_map.keys():
            raise ValueError( f"embedding_type must be 'embedding', 'linear', or 'fourier', "
            f"got {embedding_type}")
        self.embedding = embedding_map[embedding_type](op_embed_dim, **kwargs)

    def forward(self, op_vals: torch.Tensor) -> torch.Tensor:
        return self.embedding(op_vals)


