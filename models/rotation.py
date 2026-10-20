"""Latent/vector rotation helpers shared by JEPA and baseline models."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ActionRotation(nn.Module):
    """Rotate encoded image feature pairs by an action-scaled angle."""

    def __init__(
        self,
        dim: int,
        rotation_type: str = "none",
        angle_init: float = 2.0 * math.pi / 10.0,
        learnable: bool = True,
        init_mode: str = "constant",
        period: float = 10.0,
        normalize: bool = False,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.rotation_type = rotation_type
        self.normalize = normalize
        self.num_pairs = dim // 2

        valid_types = {"none", "block_rotation", "simple_rotation"}
        if rotation_type not in valid_types:
            raise ValueError(f"rotation_type must be one of {sorted(valid_types)}, got {rotation_type}")

        if rotation_type == "block_rotation":
            if init_mode == "constant":
                init_angles = torch.full((self.num_pairs,), float(angle_init))
            elif init_mode == "cyclic_harmonics":
                max_freq = max(1, int(float(period) // 2))
                freqs = torch.arange(self.num_pairs, dtype=torch.float32) % max_freq
                freqs = freqs + 1.0
                init_angles = 2.0 * math.pi * freqs / float(period)
            else:
                raise ValueError(
                    "init_mode must be 'constant' or 'cyclic_harmonics', "
                    f"got {init_mode}"
                )
            if learnable:
                init_angles = init_angles + 0.01 * torch.randn_like(init_angles)
                self.rotation_angles = nn.Parameter(init_angles)
            else:
                self.register_buffer("rotation_angles", init_angles)
        elif rotation_type == "simple_rotation":
            init_angle = torch.tensor(float(angle_init))
            if learnable:
                init_angle = init_angle + 0.01 * torch.randn_like(init_angle)
                self.rotation_angles = nn.Parameter(init_angle)
            else:
                self.register_buffer("rotation_angles", init_angle)
        else:
            self.rotation_angles = None

    def forward(self, x: torch.Tensor, action_values: torch.Tensor | None = None) -> torch.Tensor:
        if self.rotation_type == "none":
            return x
        if self.rotation_angles is None:
            raise RuntimeError("rotation_angles are only available when rotation_type is enabled")

        if action_values is None:
            action_values = torch.ones(x.shape[0], dtype=x.dtype, device=x.device)
        action_values = action_values.view(-1).float()

        if self.rotation_type == "block_rotation":
            angles = action_values.unsqueeze(-1) * self.rotation_angles.unsqueeze(0)
            cos = torch.cos(angles)
            sin = torch.sin(angles)
        else:
            angles = action_values * self.rotation_angles
            cos = torch.cos(angles).unsqueeze(-1)
            sin = torch.sin(angles).unsqueeze(-1)

        paired = x[..., : 2 * self.num_pairs].view(x.shape[0], self.num_pairs, 2)
        first = paired[..., 0]
        second = paired[..., 1]
        rotated = torch.stack(
            [
                cos * first - sin * second,
                sin * first + cos * second,
            ],
            dim=-1,
        ).flatten(start_dim=1)

        if self.dim % 2 == 1:
            rotated = torch.cat([rotated, x[..., -1:]], dim=-1)
        return F.normalize(rotated, dim=-1) if self.normalize else rotated
