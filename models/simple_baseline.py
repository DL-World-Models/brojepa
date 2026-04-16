"""Simple 2-layer fully-connected late-fusion baseline.

This provides a lightweight non-ResNet baseline for MNIST arithmetic:
- flatten image to 784 features
- embed operation value
- concatenate [img_flat, op_embed]
- 2-layer MLP -> logits
"""

from collections.abc import Sequence

import torch
import torch.nn as nn


class SimpleFCLateFusion(nn.Module):
	"""2-layer FC baseline: [flatten(img), op_embed] -> MLP -> logits."""

	def __init__(
		self,
		ops: Sequence[int],
		num_classes: int = 10,
		op_embed_dim: int = 32,
		mlp_hidden: int = 256,
	) -> None:
		super().__init__()

		self.op_proj = nn.Sequential(
			nn.Linear(1, op_embed_dim),
			nn.ReLU(),
			nn.Linear(op_embed_dim, op_embed_dim),
		)
		self.net = nn.Sequential(
			nn.Linear(28 * 28 + op_embed_dim, mlp_hidden),
			nn.ReLU(inplace=True),
			nn.Linear(mlp_hidden, num_classes),
		)

	def forward(self, img: torch.Tensor, op_vals: torch.Tensor) -> torch.Tensor:
		img_flat = img.flatten(start_dim=1)
		op_vals = op_vals.unsqueeze(-1).float()  # (B, 1)
		op_vec = self.op_proj(op_vals)
		fused = torch.cat([img_flat, op_vec], dim=1)
		return self.net(fused)
