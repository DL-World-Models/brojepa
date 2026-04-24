"""Train configurable jepa models for the MNIST-arithmetic task.

Typical usage
-------------
Train jepa models from YAML (MPS / CUDA picked automatically):

    python train_jepa.py --config configs/jepa_test.yaml


Only override config values via CLI when you want to quickly test a single change without editing YAML, or to sweep over a few values in a script.
Override selected values inline (CLI wins over YAML):

    python train_jepa.py --epochs 10 --overfit 64 --lr 3e-4

The script writes all artifacts (config, history, best/last checkpoints) to
`runs/jepa/<timestamp>-<tag>/` so every run is self-describing and
reproducible given the saved config + git hash.
"""
from typing import Any
from tqdm import tqdm
import json

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from data_pipeline import FixedSampleDataset, get_arithmetic_dataloader
from utils.experiment_config import apply_cli_overrides, load_experiment_config, parse_args
from models.jepa_model import IJepaModel, compute_loss
from utils.common import get_device, set_seed, build_run_dir, git_hash

@torch.no_grad()
def collect_latent_diagnostics(
        model: IJepaModel,
        dataloader: DataLoader,
        device: torch.device,
        num_classes: int,
) -> dict[str, torch.tensor]:
    pred_latents = {i:[] for i in range(num_classes)}
    target_latents = {i:[] for i in range(num_classes)}

    for batch in dataloader:
        contex_img = batch['ctx_img'].to(device,  non_blocking=True)
        target_img = batch['tgt_img'].to(device, non_blocking=True)
        actions = batch['op'].to(device, non_blocking=True)
        target_labels = batch['tgt_label'].to(device, non_blocking=True)
        pred_latent, target_latent, class_logits = model(contex_img, target_img, actions)
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
) -> dict[str, float]:
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
        'min_feature_var_pred': min_feature_var_pred,
        'mean_feature_var_target': mean_feature_var_target,
        'min_feature_var_target': min_feature_var_target
    }

def evaluate(
        model: IJepaModel,
        loader: DataLoader,
        device: torch.device,
    cls_loss_weight: float,
):
    model.eval()
    total_loss = 0.0
    jepa_loss, cls_loss = 0.0, 0.0
    accuracy = 0.0
    num_batches = len(loader)
    with torch.no_grad():
        for batch in loader:
            context_img = batch["ctx_img"].to(device, non_blocking=True)
            target_img = batch["tgt_img"].to(device, non_blocking=True)
            target_labels = batch["tgt_label"].to(device, non_blocking=True)
            actions = batch["op"].to(device, non_blocking=True)
            predicted_latent, target_latent, class_logits = model(context_img, target_img, actions)
            losses = compute_loss(predicted_latent, target_latent,
                                  target_labels=target_labels,
                                  class_logits=class_logits,
                                  cls_loss_weight=cls_loss_weight)

            total_loss += losses["total_loss"].item()
            jepa_loss += losses["jepa_loss"].item()
            accuracy += losses["accuracy"].item()
            cls_loss += losses["cls_loss"].item()

    return {
        "total_loss": total_loss / num_batches,
        "jepa_loss": jepa_loss / num_batches,
        "accuracy": accuracy / num_batches,
        "cls_loss": cls_loss / num_batches}

def main():
    # ---- config------ #
    # Get the hyperparameters' values from the configure file #
    args = parse_args()
    config = load_experiment_config(args.config)
    config = apply_cli_overrides(config, vars(args))
    set_seed(config.run.seed)
    device = get_device()
    run_dir = build_run_dir(config.run.runs_dir, config.run.tag + "_" + config.model.encoder_type)

    print(f"[run] device={device}  run_dir={run_dir}")
    print(f"[run] config={config.to_dict()}")

    config_blob: dict[str, Any] = {
        **config.to_dict(),
        "git_hash": git_hash(),
        "device": str(device),
    }
    (run_dir / "config.json").write_text(json.dumps(config_blob, indent=2))

    # ---- data ---- #
    # Use test_ops if specified, otherwise default to training ops
    test_ops = config.data.test_ops if config.data.test_ops else config.data.ops

    train_loader = get_arithmetic_dataloader(
        dataset_name=config.data.dataset,
        batch_size=config.data.batch_size,
        ops=config.data.ops,
        is_train=True,
        data_root=config.data.data_root,
        num_workers=config.data.num_workers,
    )
    seen_test_loader = None
    if test_ops != config.data.ops:
        seen_test_loader = get_arithmetic_dataloader(
            dataset_name=config.data.dataset,
            batch_size=config.data.batch_size,
            ops=config.data.ops,
            is_train=False,
            data_root=config.data.data_root,
            num_workers=config.data.num_workers,
        )

    test_loader = get_arithmetic_dataloader(
        dataset_name=config.data.dataset,
        batch_size=config.data.batch_size,
        ops=test_ops,
        is_train=False,
        data_root=config.data.data_root,
        num_workers=config.data.num_workers,
    )

    if config.run.overfit > 0:
        # this section is not needed.
        snap = FixedSampleDataset(train_loader.dataset, config.run.overfit)
        bs = min(config.data.batch_size, config.run.overfit)
        train_loader = DataLoader(
            snap,
            batch_size=bs,
            shuffle=True,
            num_workers=config.data.num_workers,
        )
        test_loader = DataLoader(
            snap,
            batch_size=bs,
            shuffle=False,
            num_workers=config.data.num_workers,
        )
        seen_test_loader = test_loader
        print(f"[overfit] snapshotted {config.run.overfit} samples (train == eval set)")

    # ---- model / optim ---- #
    # For embedding mode, include all ops the model might see (train + test)
    # For linear/fourier modes, only training ops needed (they generalize to any value)
    if config.model.op_embedding_type == "embedding":
        model_ops = sorted(list(set(config.data.ops) | set(test_ops)))
    else:
        model_ops = config.data.ops

    encoder_kwargs: dict[str, Any] = {
        'normalize': config.model.encoder_normalize,
        'activation': config.model.activation
    }
    if config.model.encoder_type.lower() in ['mlp', 'cnn']:
        encoder_kwargs['hidden_dim'] = config.model.encoder_hidden_dim

    elif config.model.encoder_type.lower() == 'vit':
        encoder_kwargs.update({
            "patch_size": config.model.patch_size,
            "d_model": config.model.d_model,
            "depth": config.model.depth,
            "nhead": config.model.nhead,
            "mlp_dim": config.model.mlp_dim,
            "use_cls_token": config.model.use_cls_token
        })
    classifier_kwargs = {'num_classes': config.model.num_classes,
                         'hidden_dim': config.model.classifier_hidden_dim}
    kwargs: dict[str, Any] = {'encoder_kwargs': encoder_kwargs, 'classifier_kwargs': classifier_kwargs}

    model = IJepaModel(
        ops=model_ops,
        latent_dim=config.model.latent_dim,
        op_embed_dim=config.model.op_embed_dim,
        op_embedding_type=config.model.op_embedding_type,
        encoder_type=config.model.encoder_type,
        predictor_hidden_dim=tuple(config.model.predictor_hidden_dim),
        predictor_dropout=config.model.predictor_dropout,
        predictor_normalize=config.model.predictor_normalize,
        joint_cls_training=config.optim.cls_weight > 0,
        momentum =config.model.momentum,
        **kwargs
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=config.optim.lr, weight_decay=config.optim.weight_decay)
    cls_probe_optimizer = optim.AdamW(model.class_head.parameters(),
                                      lr=config.optim.lr, weight_decay=config.optim.weight_decay)

    # ----- train ----- #
    epochs = config.run.epochs
    num_batches = len(train_loader)
    history: list[dict[str, Any]] = []
    best_test_acc = 0.0  # why -1.0
    check_point = None
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        jepa_loss, cls_loss = 0.0, 0.0
        accuracy = 0
        pbar: Any = tqdm(train_loader, desc=f'training epoch {epoch}', leave=False)
        for batch in pbar:
            context_img = batch["ctx_img"].to(device, non_blocking=True)
            target_img = batch["tgt_img"].to(device, non_blocking=True)
            target_labels = batch["tgt_label"].to(device, non_blocking=True)
            actions = batch["op"].to(device, non_blocking=True)
            predicted_latent, target_latent, class_logits = model(context_img, target_img, actions)
            losses = compute_loss(predicted_latent, target_latent,
                                target_labels=target_labels, class_logits=class_logits,
                                cls_loss_weight=config.optim.cls_weight)

            optimizer.zero_grad(set_to_none=True)
            if  config.optim.cls_weight > 0.0:
                losses['total_loss'].backward()
                optimizer.step()
                model.update_target_encoder()
            else:
                losses['jepa_loss'].backward()
                optimizer.step()
                model.update_target_encoder()

                cls_probe_optimizer.zero_grad(set_to_none=True)
                losses['cls_loss'].backward()
                cls_probe_optimizer.step()

            total_loss += losses['total_loss'].item()
            jepa_loss += losses['jepa_loss'].item()
            cls_loss += losses['cls_loss'].item()
            accuracy += losses['accuracy'].item()

        train_stats = {
            'total_loss': total_loss / num_batches,
            'jepa_loss': jepa_loss / num_batches,
            'cls_loss': cls_loss / num_batches,
            'accuracy': accuracy / num_batches
        }
        test_stats = evaluate(model, test_loader, device, config.optim.cls_weight)
        seen_stats = None
        if seen_test_loader is not None:
            seen_stats = evaluate(model, seen_test_loader, device, config.optim.cls_weight)

        log_lines = [
            f"[epoch {epoch:3d}]",
            (
                f"  {'train':<11}"
                f"total={train_stats['total_loss']:.4f} "
                f"jepa={train_stats['jepa_loss']:.4f} "
                f"cls={train_stats['cls_loss']:.4f} "
                f"acc={train_stats['accuracy']:.4f}"
            ),
        ]
        if seen_stats is not None:
            log_lines.append(
                (
                    f"  {'test seen':<11}"
                    f"total={seen_stats['total_loss']:.4f} "
                    f"jepa={seen_stats['jepa_loss']:.4f} "
                    f"cls={seen_stats['cls_loss']:.4f} "
                    f"acc={seen_stats['accuracy']:.4f}"
                )
            )
        log_lines.append(
            (
                f"  {'test unseen':<11}"
                f"total={test_stats['total_loss']:.4f} "
                f"jepa={test_stats['jepa_loss']:.4f} "
                f"cls={test_stats['cls_loss']:.4f} "
                f"acc={test_stats['accuracy']:.4f}"
            )
        )
        print("\n".join(log_lines))

        epoch_record = {"epoch": epoch, "train": train_stats, "test": test_stats}
        if seen_stats is not None:
            epoch_record["test_seen_ops"] = seen_stats
        history.append(epoch_record)
        (run_dir / "history.json").write_text(json.dumps(history, indent=2))

        if test_stats["accuracy"] > best_test_acc:
            best_test_acc = test_stats["accuracy"]
            check_point ={
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "test_acc": best_test_acc,
                    "config": config_blob,
                }
    if check_point is None:
        check_point = {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "test_acc": test_stats["accuracy"],
                    "config": config_blob,
                }
    torch.save(check_point, run_dir / "checkpoint.pt")
    ops_note = (
        f" (unseen ops: {test_ops})"
        if test_ops != config.data.ops
        else ""
    )
    print(f"[done] best test acc={best_test_acc:.4f}{ops_note}  artifacts in {run_dir}")

    # check the latent at the end of training
    outputs = collect_latent_diagnostics(model, train_loader, device, config.model.num_classes)
    metrics = latent_metrics(
        outputs['pred_latents_mean'],
        outputs['pred_latents_var'],
        outputs['target_latents_mean'],
        outputs['target_latents_var']
    )
    print(f"================Checking the latent metrics=============")
    for k, v in metrics.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()




















