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
from evaluation.metrics import (
    evaluate,
    rollout_evaluation,
    evaluate_classification,
    collect_latent_diagnostics,
    latent_metrics,
)


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
    kwargs_model: dict[str, Any] = {'encoder_kwargs': encoder_kwargs, 'classifier_kwargs': classifier_kwargs}

    kwargs_vicreg: dict[str, Any] = {'cls_loss_weight': config.optim.cls_weight,
                                    'use_vic_reg': config.model.use_vic_reg,
                                    'lamda': config.model.lamda,
                                    'gamma': config.model.gamma,
                                    'miu': config.model.miu,
                                    'nu': config.model.nu,}

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
        **kwargs_model
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=config.optim.lr, weight_decay=config.optim.weight_decay)
    cls_probe_optimizer = optim.AdamW(model.class_head.parameters(),
                                      lr=config.optim.lr, weight_decay=config.optim.weight_decay)
    num_batches = len(train_loader)

    # ---- optional warmup: classification pretraining before JEPA ---- #
    warmup_epochs = max(0, int(config.run.warmup_epochs))
    if warmup_epochs > 0:
        warmup_lr = config.run.warmup_lr if config.run.warmup_lr is not None else config.optim.lr
        freeze_context_post_warmup = config.run.freeze_context_post_warmup
        
        # During warmup, both encoders and class head are trainable
        warmup_params = list(model.context_encoder.parameters()) + list(model.class_head.parameters())
        warmup_optimizer = optim.AdamW(warmup_params, lr=warmup_lr, weight_decay=config.optim.weight_decay)
        print(
            f"[warmup] enabled epochs={warmup_epochs} "
            f"lr={warmup_lr:.2e} freeze_context_post_warmup={freeze_context_post_warmup} "
            f"(encoder->class_head pretraining before JEPA)"
        )

        for warmup_epoch in range(1, warmup_epochs + 1):
            model.train()
            total_cls_loss = 0.0
            total_acc = 0.0
            pbar: Any = tqdm(
                train_loader,
                desc=f"warmup epoch {warmup_epoch}/{warmup_epochs}",
                leave=False,
            )
            for batch in pbar:
                context_img = batch["ctx_img"].to(device, non_blocking=True)
                ctx_labels = batch["ctx_label"].to(device, non_blocking=True)

                # Both encoders and class head are trainable during warmup
                context_latent = model.context_encoder(context_img)

                class_logits = model.class_head(context_latent)
                cls_loss = F.cross_entropy(class_logits, ctx_labels, reduction="mean")
                batch_acc = (torch.argmax(class_logits, dim=-1) == ctx_labels).float().mean()

                warmup_optimizer.zero_grad(set_to_none=True)
                cls_loss.backward()
                warmup_optimizer.step()

                total_cls_loss += cls_loss.item()
                total_acc += batch_acc.item()

            warmup_train_stats = {
                "cls_loss": total_cls_loss / num_batches,
                "accuracy": total_acc / num_batches,
            }
            warmup_test_stats = evaluate_classification(model, test_loader, device)

            log_lines = [
                f"[warmup {warmup_epoch:3d}/{warmup_epochs}]",
                (
                    f"  {'train':<15}"
                    f"cls={warmup_train_stats['cls_loss']:.4f} "
                    f"acc={warmup_train_stats['accuracy']:.4f}"
                ),
                (
                    f"  {'test':<15}"
                    f"cls={warmup_test_stats['cls_loss']:.4f} "
                    f"acc={warmup_test_stats['accuracy']:.4f}"
                ),
            ]
            print("\n".join(log_lines))

        # Start JEPA phase from warmed online encoder/predictor weights.
        model.sync_target_encoder()
        # Freeze context encoder after warmup only if flag is set
        if freeze_context_post_warmup:
            for p in model.context_encoder.parameters():
                p.requires_grad = False
            print("[warmup->JEPA] context_encoder frozen after warmup")

    # ----- train ----- #
    epochs = config.run.epochs
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
                                  target_labels=target_labels,
                                  class_logits=class_logits,
                                  **kwargs_vicreg
                                  )

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
        test_stats = evaluate(model, test_loader, device, **kwargs_vicreg)
        seen_stats = None
        if seen_test_loader is not None:
            seen_stats = evaluate(model, seen_test_loader, device, **kwargs_vicreg)

        # Compute rollout accuracy for unseen operations
        rollout_stats = None
        if test_ops != config.data.ops:
            rollout_stats = rollout_evaluation(model, test_loader, device, test_ops, config.model.num_classes)

        log_lines = [
            f"[epoch {epoch:3d}]",
            (
                f"  {'train':<15}"
                f"total={train_stats['total_loss']:.4f} "
                f"jepa={train_stats['jepa_loss']:.4f} "
                f"cls={train_stats['cls_loss']:.4f} "
                f"acc={train_stats['accuracy']:.4f}"
            ),
        ]
        if seen_stats is not None:
            log_lines.append(
                (
                    f"  {'test seen op':<15}"
                    f"total={seen_stats['total_loss']:.4f} "
                    f"jepa={seen_stats['jepa_loss']:.4f} "
                    f"cls={seen_stats['cls_loss']:.4f} "
                    f"acc={seen_stats['accuracy']:.4f}"
                )
            )
        log_lines.append(
            (
                f"  {'test unseen op':<15}"
                f"total={test_stats['total_loss']:.4f} "
                f"jepa={test_stats['jepa_loss']:.4f} "
                f"cls={test_stats['cls_loss']:.4f} "
                f"acc={test_stats['accuracy']:.4f}"
                + (
                    f" rollout_cls={rollout_stats['cls_accuracy']:.4f} "
                    f"rollout_knn={rollout_stats['knn_accuracy']:.4f}"
                    if rollout_stats is not None
                    else ""
                )
            )
        )
        print("\n".join(log_lines))

        epoch_record = {"epoch": epoch, "train": train_stats, "test": test_stats}
        if seen_stats is not None:
            epoch_record["test_seen_ops"] = seen_stats
        if rollout_stats is not None:
            epoch_record["rollout"] = rollout_stats
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

    # Check the latent representations at the end of training
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

    # Compute final rollout evaluation for unseen operations
    if test_ops != config.data.ops:
        rollout_accuracy = rollout_evaluation(model, test_loader, device, test_ops, config.model.num_classes)
        print(f"=======Final Rollout test accuracy=============")
        for k, v in rollout_accuracy.items():
            print(f"{k}: {v}")

if __name__ == "__main__":
    main()




















