"""Train configurable baselines for the MNIST-arithmetic task.

Typical usage
-------------
Train ResNet baseline from YAML (MPS / CUDA picked automatically):

    python train_baseline.py --config configs/resnet_baseline.yaml

Train simple 2-layer FC baseline:

    python train_baseline.py --config configs/simple_baseline.yaml


Only override config values via CLI when you want to quickly test a single change without editing YAML, or to sweep over a few values in a script.
Override selected values inline (CLI wins over YAML):

    python train_baseline.py --epochs 10 --overfit 64 --lr 3e-4

The script writes all artifacts (config, history, best/last checkpoints) to
`runs/baseline/<timestamp>-<tag>/` so every run is self-describing and
reproducible given the saved config + git hash.
"""

import json
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassAccuracy
from tqdm import tqdm

from data_pipeline import FixedSampleDataset, get_arithmetic_dataloader
from utils.experiment_config import apply_cli_overrides, load_experiment_config, parse_args
from models import ResNet18LateFusion, SimpleFCLateFusion
from utils.common import get_device, set_seed, build_run_dir, git_hash


# --------------------------------------------------------------------------- #
# One train / eval epoch
# --------------------------------------------------------------------------- #

def _run_epoch(
    model: nn.Module,
    loader: DataLoader[dict[str, torch.Tensor]],
    criterion: nn.Module,
    device: torch.device,
    acc_metric: MulticlassAccuracy,
    optimizer: torch.optim.Optimizer | None,
    desc: str,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    acc_metric.reset()

    total_loss, n = 0.0, 0
    pbar: Any = tqdm(loader, desc=desc, leave=False)
    for batch in pbar:
        img = batch["ctx_img"].to(device, non_blocking=True)
        op = batch["op"].to(device, non_blocking=True)
        tgt = batch["tgt_label"].to(device, non_blocking=True)

        with torch.set_grad_enabled(is_train):
            logits = model(img, op)
            loss = criterion(logits, tgt)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        bs = tgt.size(0)
        total_loss += loss.item() * bs
        n += bs
        acc_metric.update(logits.detach(), tgt)
        pbar.set_postfix({"loss": f"{total_loss / n:.4f}"})

    return {"loss": total_loss / max(n, 1), "acc": acc_metric.compute().item()}


def _evaluate_by_op(
    model: nn.Module,
    ops: list[int],
    config: Any,
    criterion: nn.Module,
    device: torch.device,
) -> list[dict[str, float | int]]:
    model.eval()
    rows: list[dict[str, float | int]] = []
    for op_val in ops:
        loader = get_arithmetic_dataloader(
            dataset_name=config.data.dataset,
            batch_size=config.data.batch_size,
            ops=[op_val],
            is_train=False,
            data_root=config.data.data_root,
            num_workers=config.data.num_workers,
        )
        correct, total, total_loss = 0, 0, 0.0
        acc_metric = MulticlassAccuracy(num_classes=config.model.num_classes).to(device)
        acc_metric.reset()
        with torch.no_grad():
            for batch in loader:
                img = batch["ctx_img"].to(device, non_blocking=True)
                op = batch["op"].to(device, non_blocking=True)
                tgt = batch["tgt_label"].to(device, non_blocking=True)
                logits = model(img, op)
                loss = criterion(logits, tgt)
                bs = tgt.size(0)
                total_loss += loss.item() * bs
                total += bs
                correct += (logits.argmax(dim=1) == tgt).sum().item()
                acc_metric.update(logits, tgt)
        rows.append(
            {
                "op": int(op_val),
                "micro_acc": correct / max(total, 1),
                "macro_acc": acc_metric.compute().item(),
                "loss": total_loss / max(total, 1),
                "n": total,
            }
        )
    return rows


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)
    config = apply_cli_overrides(config, vars(args))
    set_seed(config.run.seed)
    device = get_device()
    run_dir = build_run_dir(config.run.runs_dir, config.run.tag)

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
    test_loader = get_arithmetic_dataloader(
        dataset_name=config.data.dataset,
        batch_size=config.data.batch_size,
        ops=test_ops,
        is_train=False,
        data_root=config.data.data_root,
        num_workers=config.data.num_workers,
    )

    if config.run.overfit > 0:
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
        print(f"[overfit] snapshotted {config.run.overfit} samples (train == eval set)")

    # ---- model / optim ---- #
    # For embedding mode, include all ops the model might see (train + test)
    # For linear/fourier modes, only training ops needed (they generalize to any value)
    if config.model.op_embedding_type == "embedding":
        model_ops = sorted(list(set(config.data.ops) | set(test_ops)))
    else:
        model_ops = config.data.ops
    
    if config.model.name == "resnet18":
        model = ResNet18LateFusion(
            ops=model_ops,
            num_classes=config.model.num_classes,
            op_embed_dim=config.model.op_embed_dim,
            mlp_hidden=config.model.mlp_hidden,
            op_embedding_type=config.model.op_embedding_type,
            predictor_type=config.model.predictor_type,
            predictor_normalize=config.model.predictor_normalize,
            rotation_angle_init=config.model.rotation_angle_init,
            rotation_learnable=config.model.rotation_learnable,
            rotation_init_mode=config.model.rotation_init_mode,
            rotation_period=config.model.rotation_period,
            use_action_rotation=config.model.use_action_rotation,
            include_op_embedding_in_head=config.model.include_op_embedding_in_head,
        ).to(device)
    elif config.model.name == "simple_fc2":
        model = SimpleFCLateFusion(
            ops=model_ops,
            num_classes=config.model.num_classes,
            op_embed_dim=config.model.op_embed_dim,
            mlp_hidden=config.model.mlp_hidden,
            simple_encoder_depth=config.model.simple_encoder_depth,
            op_embedding_type=config.model.op_embedding_type,
            predictor_type=config.model.predictor_type,
            predictor_normalize=config.model.predictor_normalize,
            rotation_angle_init=config.model.rotation_angle_init,
            rotation_learnable=config.model.rotation_learnable,
            rotation_init_mode=config.model.rotation_init_mode,
            rotation_period=config.model.rotation_period,
            use_action_rotation=config.model.use_action_rotation,
            include_op_embedding_in_head=config.model.include_op_embedding_in_head,
        ).to(device)
    else:
        raise ValueError(
            f"Unknown model.name='{config.model.name}'. "
            "Supported: ['resnet18', 'simple_fc2']"
        )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] parameters={n_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.optim.lr,
        weight_decay=config.optim.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()
    acc_metric = MulticlassAccuracy(num_classes=config.model.num_classes).to(device)

    # ---- train ---- #
    history: list[dict[str, Any]] = []
    best_test_acc = -1.0
    for epoch in range(1, config.run.epochs + 1):
        train_stats = _run_epoch(
            model,
            train_loader,
            criterion,
            device,
            acc_metric,
            optimizer=optimizer,
            desc=f"epoch {epoch}/{config.run.epochs} train",
        )
        test_stats = _run_epoch(
            model,
            test_loader,
            criterion,
            device,
            acc_metric,
            optimizer=None,
            desc=f"epoch {epoch}/{config.run.epochs}  eval",
        )
        print(
            f"[epoch {epoch:3d}] "
            f"train loss={train_stats['loss']:.4f} acc={train_stats['acc']:.4f}  "
            f"test loss={test_stats['loss']:.4f} acc={test_stats['acc']:.4f}"
        )
        history.append({"epoch": epoch, "train": train_stats, "test": test_stats})
        (run_dir / "history.json").write_text(json.dumps(history, indent=2))

        if test_stats["acc"] > best_test_acc:
            best_test_acc = test_stats["acc"]
            save_fn: Any = getattr(torch, "save")
            save_fn(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "test_acc": best_test_acc,
                    "config": config_blob,
                },
                str(run_dir / "best.pt"),
            )

    save_fn: Any = getattr(torch, "save")
    save_fn(
        {
            "model": model.state_dict(),
            "epoch": config.run.epochs,
            "config": config_blob,
        },
        str(run_dir / "last.pt"),
    )
    if config.run.eval_by_op and test_ops != config.data.ops:
        load_fn: Any = getattr(torch, "load")
        best_checkpoint = load_fn(str(run_dir / "best.pt"), map_location=device)
        model.load_state_dict(best_checkpoint["model"])
        per_op_stats = _evaluate_by_op(model, test_ops, config, criterion, device)
        (run_dir / "per_op_metrics.json").write_text(json.dumps(per_op_stats, indent=2))
        print("[per-op] op  micro_acc  macro_acc  loss")
        for row in per_op_stats:
            print(
                "[per-op] "
                f"{int(row['op']):>3}  "
                f"{float(row['micro_acc']):.4f}     "
                f"{float(row['macro_acc']):.4f}     "
                f"{float(row['loss']):.4f}"
            )
    ops_note = (
        f" (unseen ops: {test_ops})"
        if test_ops != config.data.ops
        else ""
    )
    print(f"[done] best test acc={best_test_acc:.4f}{ops_note}  artifacts in {run_dir}")


if __name__ == "__main__":
    main()
