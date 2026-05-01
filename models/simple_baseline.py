"""Simple 2-layer fully-connected baseline for MNIST arithmetic.

This provides a lightweight non-ResNet baseline for MNIST arithmetic:
- flatten image to 784 features
- encode image with an MLP layer
- optionally rotate encoded image features by the raw action value
- optionally concatenate an operation embedding
- classification head -> logits

Supports multiple operation embedding strategies via configuration.
"""

from collections.abc import Sequence

import torch
import torch.nn as nn

from models.op_embedding import create_operation_embedding, OpEmbeddingType
from models.rotation import ActionRotation


class SimpleFCLateFusion(nn.Module):
	"""2-layer FC baseline with an explicit image encoder and classifier.
	
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
		simple_encoder_depth: int = 1,
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
		self.simple_encoder_depth = max(1, int(simple_encoder_depth))
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
		self.uses_split_path = use_action_rotation or not include_op_embedding_in_head
		if self.uses_split_path:
			encoder_layers: list[nn.Module] = []
			in_dim = 28 * 28
			for _ in range(self.simple_encoder_depth):
				encoder_layers.append(nn.Linear(in_dim, mlp_hidden))
				encoder_layers.append(nn.ReLU(inplace=True))
				in_dim = mlp_hidden
			self.encoder = nn.Sequential(*encoder_layers)
			self.action_rotation = ActionRotation(
				dim=mlp_hidden,
				rotation_type=rotation_type,
				angle_init=rotation_angle_init,
				learnable=rotation_learnable,
				init_mode=rotation_init_mode,
				period=rotation_period,
				normalize=predictor_normalize,
			)
			head_in_dim = mlp_hidden + op_embed_dim if include_op_embedding_in_head else mlp_hidden
			self.classifier = nn.Linear(head_in_dim, num_classes)
		else:
			self.net = nn.Sequential(
				nn.Linear(28 * 28 + op_embed_dim, mlp_hidden),
				nn.ReLU(inplace=True),
				nn.Linear(mlp_hidden, num_classes),
			)

	def forward(
		self,
		img: torch.Tensor,
		op_vals: torch.Tensor,
		return_features: bool = False,
	) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
		img_flat = img.flatten(start_dim=1)
		if not self.uses_split_path:
			op_vec = self.op_embed(op_vals)
			fused = torch.cat([img_flat, op_vec], dim=1)
			features = self.net[:2](fused)
			logits = self.net[2](features)
			return (logits, features) if return_features else logits

		features = self.encoder(img_flat)
		features = self.action_rotation(features, op_vals)
		head_features = features
		if self.include_op_embedding_in_head:
			op_vec = self.op_embed(op_vals)
			head_features = torch.cat([features, op_vec], dim=1)
		logits = self.classifier(head_features)
		return (logits, features) if return_features else logits
