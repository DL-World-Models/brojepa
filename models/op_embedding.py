"""Operation embedding strategies for MNIST arithmetic models.

Provides different ways to embed integer operation values:
- embedding: Fixed lookup table for discrete operations
- linear: Linear projection of raw scalar values
- fourier: Fourier features + projection for smooth generalization
"""

from typing import Literal

import torch
import torch.nn as nn

OpEmbeddingType = Literal[
    "embedding",
    "linear",
    "fourier",
    "hybrid",
    "additive",
    "additive_embedding",
    "additive_embedding_normalized",
]


class OperationEmbedding(nn.Module):
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


class OperationEmbeddingLookup(OperationEmbedding):
    """Fixed embedding lookup for discrete operations.

    Requires a finite set of known operations. Operations must be unique.
    Good for fixed-operation settings where you want discrete embeddings.
    """

    def __init__(self, ops: list[int], op_embed_dim: int) -> None:
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


class OperationEmbeddingLinear(OperationEmbedding):
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


class OperationEmbeddingFourier(OperationEmbedding):
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
        fourier_feat = torch.cat(
            [
                torch.sin(op_vals),
                torch.cos(op_vals),
                torch.sin(2 * op_vals),
                torch.cos(2 * op_vals),
            ],
            dim=-1,
        )  # [B, 4]
        return self.proj(fourier_feat)  # [B, op_embed_dim]


class OperationEmbeddingHybrid(OperationEmbedding):
    """Hybrid encoding separating operator (sign) and operand (magnitude).

    Categorically embeds the sign (+/-) and uses Fourier features for the magnitude,
    concatenates them, and projects to the final op_embed_dim.
    """

    def __init__(self, op_embed_dim: int) -> None:
        super().__init__(op_embed_dim)

        # 1. Sign Embedding (2 classes: Negative=0, Positive/Zero=1)
        # We allocate half of the target dimension to the sign for balance
        self.sign_dim = op_embed_dim // 2
        self.sign_embed = nn.Embedding(2, self.sign_dim)

        # 2. Magnitude Embedding (Fourier)
        # 4 Fourier features: sin(x), cos(x), sin(2x), cos(2x)
        self.fourier_dim = 4

        # 3. Projection layer
        # Takes the concatenated [sign_dim + fourier_dim] and projects to target op_embed_dim
        self.proj = nn.Sequential(
            nn.Linear(self.sign_dim + self.fourier_dim, op_embed_dim),
            nn.ReLU(),
            nn.Linear(op_embed_dim, op_embed_dim),
        )

    def forward(self, op_vals: torch.Tensor) -> torch.Tensor:
        # Extract sign: 0 for negative, 1 for positive
        signs = (op_vals >= 0).long()

        # Extract magnitude (the actual number)
        mags = torch.abs(op_vals).unsqueeze(-1).float()  # [B, 1]

        # Embed the sign
        sign_vecs = self.sign_embed(signs)  # [B, sign_dim]

        # Embed the magnitude using Fourier features
        fourier_feat = torch.cat(
            [
                torch.sin(mags),
                torch.cos(mags),
                torch.sin(2 * mags),
                torch.cos(2 * mags),
            ],
            dim=-1,
        )  # [B, 4]

        # Concatenate and project
        hybrid_vec = torch.cat([sign_vecs, fourier_feat], dim=-1)  # [B, sign_dim + 4]
        return self.proj(hybrid_vec)  # [B, op_embed_dim]


class OperationEmbeddingAdditive(OperationEmbedding):
    """Additive operator embedding with separate base vectors for positive and negative.

    op_embed(k) = (k / max_mag) * base_plus
    op_embed(-k) = (k / max_mag) * base_minus

    By default `max_mag=9.0` (MNIST digits 0-9). Normalizing keeps embedding
    magnitudes in a consistent range and stabilizes training.
    """

    def __init__(self, op_embed_dim: int, max_mag: float = 9.0) -> None:
        super().__init__(op_embed_dim)

        # Learnable base vectors for positive and negative operands
        self.base_plus = nn.Parameter(torch.randn(op_embed_dim) * 0.01)
        self.base_minus = nn.Parameter(torch.randn(op_embed_dim) * 0.01)
        self.max_mag = float(max_mag) if max_mag != 0.0 else 1.0

    def forward(self, op_vals: torch.Tensor) -> torch.Tensor:
        # op_vals: [B]
        vals = op_vals.unsqueeze(-1).float()  # [B, 1]
        mags = torch.abs(vals) / float(self.max_mag)  # normalized magnitude [B, 1]
        signs = (vals >= 0).float()  # [B, 1], 1 for non-negative, 0 for negative

        # Broadcast base vectors to [B, op_embed_dim]
        base = signs * self.base_plus + (1.0 - signs) * self.base_minus  # [B, op_embed_dim]

        return mags * base  # [B, op_embed_dim]


def create_operation_embedding(
    embedding_type: OpEmbeddingType,
    op_embed_dim: int,
    ops: list[int] | None = None,
) -> OperationEmbedding:
    """Factory function to create operation embedding.

    Args:
        embedding_type: Type of embedding ("embedding", "linear", "fourier")
        op_embed_dim: Output dimension
        ops: List of operations (required for "embedding" type)

    Returns:
        OperationEmbedding instance

    Raises:
        ValueError: If embedding_type is invalid or ops missing for embedding type
    """
    if embedding_type == "embedding":
        if ops is None:
            raise ValueError("ops required for embedding type")
        return OperationEmbeddingLookup(ops, op_embed_dim)
    elif embedding_type == "linear":
        return OperationEmbeddingLinear(op_embed_dim)
    elif embedding_type == "fourier":
        return OperationEmbeddingFourier(op_embed_dim)
    elif embedding_type == "hybrid":
        return OperationEmbeddingHybrid(op_embed_dim)
    elif embedding_type in ("additive", "additive_embedding"):
        # Backwards-compatible non-normalized additive (scale by raw k)
        return OperationEmbeddingAdditive(op_embed_dim, max_mag=1.0)
    elif embedding_type == "additive_embedding_normalized":
        # Normalized for MNIST digits 0-9
        return OperationEmbeddingAdditive(op_embed_dim, max_mag=9.0)
    else:
        raise ValueError(
            f"embedding_type must be one of 'embedding', 'linear', 'fourier', 'hybrid', "
            f"'additive', 'additive_embedding', or 'additive_embedding_normalized', got {embedding_type}"
        )
