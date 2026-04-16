"""Simple 2-layer fully-connected late-fusion baseline.

This provides a lightweight non-ResNet baseline for MNIST arithmetic:
- flatten image to 784 features
- embed operation value (configurable strategy)
- concatenate [img_flat, op_embed]
- 2-layer MLP -> logits

Supports multiple operation embedding strategies via configuration.
"""

from collections.abc import Sequence

import torch
import torch.nn as nn

from models.op_embedding import create_operation_embedding, OpEmbeddingType


class SimpleFCLateFusion(nn.Module):
	"""2-layer FC baseline: [flatten(img), op_embed] -> MLP -> logits.
	
	Args:
		ops: Sequence of allowed integer operations
		num_classes: Number of output classes (default 10)
		op_embed_dim: Dimension of operation embedding (default 32)
		mlp_hidden: Hidden layer size of MLP head (default 256)
		op_embedding_type: Type of operation embedding ("embedding", "linear", "fourier")
	"""

	def __init__(
		self,
		ops: Sequence[int],
		num_classes: int = 10,
		op_embed_dim: int = 32,
		mlp_hidden: int = 256,
		op_embedding_type: OpEmbeddingType = "embedding",
	) -> None:
		super().__init__()

		self.op_embedding_type = op_embedding_type
		self.op_embed = create_operation_embedding(
			embedding_type=op_embedding_type,
			op_embed_dim=op_embed_dim,
			ops=list(ops) if op_embedding_type == "embedding" else None,
		)
		self.net = nn.Sequential(
			nn.Linear(28 * 28 + op_embed_dim, mlp_hidden),
			nn.ReLU(inplace=True),
			nn.Linear(mlp_hidden, num_classes),
		)

	def forward(self, img: torch.Tensor, op_vals: torch.Tensor) -> torch.Tensor:
		img_flat = img.flatten(start_dim=1)
		op_vec = self.op_embed(op_vals)
		fused = torch.cat([img_flat, op_vec], dim=1)
		return self.net(fused)
