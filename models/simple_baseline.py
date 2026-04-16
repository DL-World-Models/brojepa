"""Simple 2-layer fully-connected late-fusion baseline.

This provides a lightweight non-ResNet baseline for MNIST arithmetic:
- flatten image to 784 features
- embed operation value
- concatenate [img_flat, op_embed]
- 2-layer MLP -> logits
"""

from collections.abc import Sequence
from typing import cast

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

		op_list = [int(o) for o in ops]
		if len(set(op_list)) != len(op_list):
			raise ValueError(f"ops must be unique, got {ops}")
		sorted_ops = sorted(op_list)
		self.op_to_idx = {op: idx for idx, op in enumerate(sorted_ops)}

		self.register_buffer("op_values", torch.tensor(sorted_ops, dtype=torch.long))
		self.op_embed = nn.Embedding(len(sorted_ops), op_embed_dim)
		self.net = nn.Sequential(
			nn.Linear(28 * 28 + op_embed_dim, mlp_hidden),
			nn.ReLU(inplace=True),
			nn.Linear(mlp_hidden, num_classes),
		)

	def _op_vals_to_idx(self, op_vals: torch.Tensor) -> torch.Tensor:
		op_list = cast(list[int], op_vals.long().flatten().tolist())
		try:
			idx_list = [self.op_to_idx[val] for val in op_list]
		except KeyError as exc:
			raise ValueError(f"unknown op value: {exc.args[0]}") from exc
		return torch.tensor(idx_list, device=op_vals.device, dtype=torch.long)

	def forward(self, img: torch.Tensor, op_vals: torch.Tensor) -> torch.Tensor:
		img_flat = img.flatten(start_dim=1)
		op_idx = self._op_vals_to_idx(op_vals)
		op_vec = self.op_embed(op_idx)
		fused = torch.cat([img_flat, op_vec], dim=1)
		return self.net(fused)
