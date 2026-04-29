"""Regression tests for JEPA rollout evaluation."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset

from evaluation.metrics import rollout_evaluation


class TinyArithmeticDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, ops: list[int], num_classes: int = 10) -> None:
        self.samples: list[dict[str, torch.Tensor]] = []
        for ctx_label in range(num_classes):
            for op in ops:
                tgt_label = (ctx_label + op) % num_classes
                self.samples.append(
                    {
                        "ctx_img": torch.tensor(ctx_label),
                        "ctx_label": torch.tensor(ctx_label),
                        "op": torch.tensor(float(op)),
                        "tgt_img": torch.tensor(tgt_label),
                        "tgt_label": torch.tensor(tgt_label),
                    }
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.samples[idx]


class OneHotEncoder(torch.nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.num_classes = num_classes

    def forward(self, labels: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.one_hot(labels.long(), self.num_classes).float()


class UnitStepPredictor(torch.nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.num_classes = num_classes

    def forward(
        self,
        context_latent: torch.Tensor,
        actions_embedded: torch.Tensor,
        action_values: torch.Tensor | None = None,
        op_embed: torch.nn.Module | None = None,
    ) -> torch.Tensor:
        del actions_embedded, op_embed
        if action_values is None:
            raise AssertionError("test predictor requires action_values")
        labels = context_latent.argmax(dim=-1)
        next_labels = (labels + action_values.long()) % self.num_classes
        return torch.nn.functional.one_hot(next_labels, self.num_classes).float()


class TinyModel(torch.nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.context_encoder = OneHotEncoder(num_classes)
        self.target_encoder = OneHotEncoder(num_classes)
        self.jepa_predictor = UnitStepPredictor(num_classes)
        self.class_head = torch.nn.Identity()

    def op_embed(self, actions: torch.Tensor) -> torch.Tensor:
        return torch.zeros(actions.shape[0], 1)

    def forward(
        self,
        context_images: torch.Tensor,
        target_images: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        context_latent = self.context_encoder(context_images)
        target_latent = self.target_encoder(target_images)
        actions_embedded = self.op_embed(actions)
        predicted_latent = self.jepa_predictor(
            context_latent,
            actions_embedded,
            action_values=actions,
            op_embed=self.op_embed,
        )
        return predicted_latent, target_latent, predicted_latent


def test_rollout_evaluation_handles_mixed_operation_magnitudes() -> None:
    ops = [2, 3, -4]
    loader = DataLoader(TinyArithmeticDataset(ops), batch_size=7)

    stats = rollout_evaluation(
        model=TinyModel(),
        loader=loader,
        device=torch.device("cpu"),
        ops=[1, -1],
        test_ops=ops,
        num_classes=10,
    )

    assert stats["knn_accuracy"] == 1.0
    assert stats["cls_accuracy"] == 1.0
