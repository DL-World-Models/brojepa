"""Simple 2-layer FC baseline with Fourier-encoded operations.

This variant uses Fourier/positional encoding for the operation value:
- flatten image to 784 features
- encode operation as sin/cos Fourier features
- project Fourier features to embedding
- concatenate [img_flat, op_embed]
- 2-layer MLP -> logits

Fourier features should improve generalization to unseen operation values
by providing a smoother, continuous representation.
"""

from collections.abc import Sequence

import torch
import torch.nn as nn


class SimpleFCFourier(nn.Module):
	"""2-layer FC baseline with Fourier-encoded operations."""

	def __init__(
		self,
		ops: Sequence[int],
		num_classes: int = 10,
		op_embed_dim: int = 32,
		mlp_hidden: int = 256,
	) -> None:
		super().__init__()

		# Fourier features: sin(x), cos(x), sin(2x), cos(2x) -> 4 features
		self.op_proj = nn.Sequential(
			nn.Linear(4, op_embed_dim),
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

		op_vals = op_vals.unsqueeze(-1)  # (B, 1)

		# Fourier features: sin and cos at different frequencies
		op_feat = torch.cat([
			torch.sin(op_vals),
			torch.cos(op_vals),
			torch.sin(2 * op_vals),
			torch.cos(2 * op_vals),
		], dim=-1)  # (B, 4)

		op_vec = self.op_proj(op_feat)
		fused = torch.cat([img_flat, op_vec], dim=1)
		return self.net(fused)
