"""Evaluation module for JEPA models."""
from .metrics import (
    evaluate,
    rollout_evaluation,
    evaluate_classification,
    direct_knn_evaluation,
    collect_latent_diagnostics,
    latent_metrics,
)

__all__ = [
    "evaluate",
    "rollout_evaluation",
    "evaluate_classification",
    "direct_knn_evaluation",
    "collect_latent_diagnostics",
    "latent_metrics",
]
