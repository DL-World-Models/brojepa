"""Evaluation functions for JEPA models."""
from typing import Any
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.jepa_model import IJepaModel, compute_loss


@torch.no_grad()
def collect_latent_diagnostics(
        model: IJepaModel,
        dataloader: DataLoader,
        device: torch.device,
        num_classes: int,
) -> dict[str, torch.tensor]:
    """Collect latent representations for all samples by class."""
    pred_latents = {i: [] for i in range(num_classes)}
    target_latents = {i: [] for i in range(num_classes)}

    for batch in dataloader:
        context_img = batch['ctx_img'].to(device, non_blocking=True)
        target_img = batch['tgt_img'].to(device, non_blocking=True)
        actions = batch['op'].to(device, non_blocking=True)
        target_labels = batch['tgt_label'].to(device, non_blocking=True)
        pred_latent, target_latent, class_logits = model(context_img, target_img, actions)
        for label in target_labels.unique():
            label_int = int(label.item())
            mask = (target_labels == label_int)
            pred_latents[label_int].append(pred_latent[mask])
            target_latents[label_int].append(target_latent[mask])
    
    pred_latents = {k: torch.cat(v, dim=0) for k, v in pred_latents.items() if len(v) > 0}
    target_latents = {k: torch.cat(v, dim=0) for k, v in target_latents.items() if len(v) > 0}

    pred_latents_mean = torch.stack(
        [pred_latents[k].mean(dim=0) for k in range(num_classes)]
    )
    pred_latents_var = torch.stack(
        [pred_latents[k].var(dim=0, unbiased=False) for k in range(num_classes)]
    )
    target_latents_mean = torch.stack(
        [target_latents[k].mean(dim=0) for k in range(num_classes)]
    )
    target_latents_var = torch.stack(
        [target_latents[k].var(dim=0, unbiased=False) for k in range(num_classes)]
    )

    return {
        'pred_latents_mean': pred_latents_mean,
        'pred_latents_var': pred_latents_var,
        'target_latents_mean': target_latents_mean,
        'target_latents_var': target_latents_var
    }


def latent_metrics(
        pred_latents_mean: torch.tensor,
        pred_latents_var: torch.tensor,
        target_latents_mean: torch.tensor,
        target_latents_var: torch.tensor
) -> dict[str, Any]:
    """Compute metrics on latent representations."""
    pred_norm = F.normalize(pred_latents_mean, dim=1)
    cls_similarity_pred = pred_norm @ pred_norm.T
    mask = ~ torch.eye(pred_norm.shape[0], dtype=torch.bool).to(pred_norm.device)
    cls_similarity_pred_mean = cls_similarity_pred[mask].mean()

    target_norm = F.normalize(target_latents_mean, dim=1)
    cls_similarity_target = target_norm @ target_norm.T
    cls_similarity_target_mean = cls_similarity_target[mask].mean()

    pair_similarity = F.cosine_similarity(pred_norm, target_norm, dim=1)

    mean_feature_var_pred = pred_latents_var.mean(dim=1)
    min_feature_var_pred = pred_latents_var.min(dim=1)

    mean_feature_var_target = target_latents_var.mean(dim=1)
    min_feature_var_target = target_latents_var.min(dim=1)

    return {
        'cls_similarity_pred': cls_similarity_pred,
        'cls_similarity_target': cls_similarity_target,
        'cls_similarity_pred_mean': cls_similarity_pred_mean,
        'cls_similarity_target_mean': cls_similarity_target_mean,
        'pair_similarity': pair_similarity,
        'mean_feature_var_pred': mean_feature_var_pred,
        'min_feature_var_pred': min_feature_var_pred[0],
        'mean_feature_var_target': mean_feature_var_target,
        'min_feature_var_target': min_feature_var_target[0]
    }


def evaluate(
        model: IJepaModel,
        loader: DataLoader,
        device: torch.device,
        **kwargs
) -> dict[str, float]:
    """Evaluate model on a dataset with standard metrics."""
    model.eval()
    total_loss = 0.0
    jepa_loss, cls_loss, comp_loss = 0.0, 0.0, 0.0
    accuracy = 0.0
    num_batches = len(loader)
    with torch.no_grad():
        for batch in loader:
            context_img = batch["ctx_img"].to(device, non_blocking=True)
            target_img = batch["tgt_img"].to(device, non_blocking=True)
            target_labels = batch["tgt_label"].to(device, non_blocking=True)
            actions = batch["op"].to(device, non_blocking=True)
            
            predicted_latent, target_latent, class_logits = model(context_img, target_img, actions)
            
            # Extract context_latent only if composition loss is enabled
            context_latent = None
            use_comp = kwargs.get('use_composition_loss', False) and kwargs.get('composition_lambda', 0.0) > 0.0
            if use_comp:
                context_latent = model.context_encoder(context_img)
            
            # Merge composition kwargs with model and context info for this batch
            batch_kwargs = {**kwargs, 'model': model, 'context_latent': context_latent, 'device': device}
            losses = compute_loss(predicted_latent, target_latent,
                                  target_labels=target_labels,
                                  class_logits=class_logits,
                                  **batch_kwargs
                                  )

            total_loss += losses["total_loss"].item()
            jepa_loss += losses["jepa_loss"].item()
            accuracy += losses["accuracy"].item()
            cls_loss += losses["cls_loss"].item()
            if 'comp_loss' in losses:
                comp_loss += losses['comp_loss']

    result = {
        "total_loss": total_loss / num_batches,
        "jepa_loss": jepa_loss / num_batches,
        "accuracy": accuracy / num_batches,
        "cls_loss": cls_loss / num_batches
    }
    # Add composition loss if enabled
    if kwargs.get('use_composition_loss', False) and kwargs.get('composition_lambda', 0.0) > 0.0:
        result['comp_loss'] = comp_loss / num_batches
    
    return result


@torch.no_grad()
def rollout_evaluation(
        model: IJepaModel,
        loader: DataLoader,
        device: torch.device,
        test_ops: Sequence[int],
        num_classes: int,
) -> dict[str, float]:
    """
    Evaluate model rollout ability on unseen operations.
    
    Tests if the model can recursively learn compositional relationships in addition:
    E.g., +3 = f(f(f(+1)+1)+1), -3 = f(f(f(-1)-1)-1)
    
    To use this evaluation, the training ops must be unit addition [+1, -1]
    and test_ops must be a scale*[+1, -1], e.g., [+3, -3].
    
    Args:
        model: The JEPA model to evaluate
        loader: DataLoader with test data
        device: Device to run on
        test_ops: Unseen operations to evaluate on
        num_classes: Number of classes
        
    Returns:
        Dictionary with 'knn_accuracy', 'cls_accuracy', 'cls_loss'
    """
    model.eval()
    num_batches = len(loader)
    knn_accuracy = 0.0
    scale = abs(test_ops[0])
    cls_accuracy = 0.0

    # Compute prototypes once for efficiency
    outputs = collect_latent_diagnostics(model, loader, device, num_classes)
    prototypes = outputs["target_latents_mean"]

    for batch in loader:
        context_img = batch["ctx_img"].to(device, non_blocking=True)
        target_labels = batch["tgt_label"].to(device, non_blocking=True)
        actions = batch["op"].to(device, non_blocking=True)

        unit_actions = actions // scale
        unit_actions_embedded = model.op_embed(unit_actions)
        context_latent = model.context_encoder(context_img)
        for _ in range(scale):
            predicted_latent = model.jepa_predictor(context_latent, unit_actions_embedded)
            context_latent = predicted_latent
        
        # Accuracy calculated using ClassHead
        class_logits = model.class_head(predicted_latent)
        cls_loss = F.cross_entropy(class_logits, target_labels, reduction="mean")
        pred_class = torch.argmax(class_logits, dim=-1)
        cls_accuracy += (pred_class == target_labels).float().mean()

        # Accuracy calculated using 1-nearest neighbor
        dists = torch.cdist(predicted_latent, prototypes)
        pred_class = dists.argmin(dim=1)
        knn_accuracy += (pred_class == target_labels).float().mean()

    cls_accuracy /= num_batches
    knn_accuracy /= num_batches
    return {
        "knn_accuracy": float(knn_accuracy),
        "cls_accuracy": float(cls_accuracy),
        "cls_loss": float(cls_loss),
    }


def evaluate_classification(
        model: IJepaModel,
        loader: DataLoader,
        device: torch.device,
) -> dict[str, float]:
    """Evaluate classification accuracy on context images."""
    model.eval()
    total_cls_loss = 0.0
    accuracy = 0.0
    num_batches = len(loader)
    with torch.no_grad():
        for batch in loader:
            context_img = batch["ctx_img"].to(device, non_blocking=True)
            ctx_labels = batch["ctx_label"].to(device, non_blocking=True)
            context_latent = model.context_encoder(context_img)
            class_logits = model.class_head(context_latent)
            cls_loss = F.cross_entropy(class_logits, ctx_labels, reduction="mean")
            pred = torch.argmax(class_logits, dim=-1)
            batch_acc = (pred == ctx_labels).float().mean()

            total_cls_loss += cls_loss.item()
            accuracy += batch_acc.item()

    return {
        "cls_loss": total_cls_loss / num_batches,
        "accuracy": accuracy / num_batches,
    }
