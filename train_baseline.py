"""Train the ResNet-18 late-fusion baseline for the MNIST-arithmetic task.

Typical usage
-------------
Full training run (MPS / CUDA picked automatically):

    python train_baseline.py --epochs 10

Sanity-check by overfitting a deterministic 64-sample subset:

    python train_baseline.py --overfit 64 --epochs 50 --lr 3e-4

The script writes all artifacts (config, history, best/last checkpoints) to
`runs/baseline/<timestamp>-<tag>/` so every run is self-describing and
reproducible given the saved config + git hash.
"""

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchmetrics.classification import MulticlassAccuracy
from tqdm import tqdm

from data_utils import get_arithmetic_dataloader
from models import ResNet18LateFusion
from utils import get_device, set_seed

# --------------------------------------------------------------------------- #
# Run-directory helpers
# --------------------------------------------------------------------------- #


def _git_hash() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _build_run_dir(root: str, tag: str) -> Path:
    ts = time.strftime("%Y%m%d-%H%M%S")
    run_dir = Path(root) / f"{ts}-{tag}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


# --------------------------------------------------------------------------- #
# Overfit sanity-check support
# --------------------------------------------------------------------------- #


class _FixedSampleDataset(Dataset):
    """Materializes a small, deterministic snapshot of an ArithmeticDataset.

    The underlying dataset samples operations and target images randomly at
    every __getitem__, so a naive Subset would return a different (img, op,
    tgt) triple every epoch — useless for overfit sanity-checks. We snapshot
    the first N samples once and serve them forever after.
    """

    def __init__(self, base: Dataset, n: int) -> None:
        self.samples = [base[i] for i in range(n)]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.samples[idx]


# --------------------------------------------------------------------------- #
# One train / eval epoch
# --------------------------------------------------------------------------- #


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
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
    pbar = tqdm(loader, desc=desc, leave=False)
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
        pbar.set_postfix(loss=f"{total_loss / n:.4f}")

    return {"loss": total_loss / max(n, 1), "acc": acc_metric.compute().item()}


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="MNIST", choices=["MNIST", "EMNIST"])
    p.add_argument("--ops", type=int, nargs="+", default=[1, 2, -1, -2])
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--op-embed-dim", type=int, default=32)
    p.add_argument("--mlp-hidden", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--runs-dir", type=str, default="runs/baseline")
    p.add_argument("--tag", type=str, default="run")
    p.add_argument(
        "--overfit",
        type=int,
        default=0,
        help="If > 0, train on a deterministic snapshot of N samples "
        "(sanity check that the model + loop can drive loss to ~0).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    run_dir = _build_run_dir(args.runs_dir, args.tag)

    print(f"[run] device={device}  run_dir={run_dir}")
    print(f"[run] config={vars(args)}")

    # ---- data ---- #
    train_loader = get_arithmetic_dataloader(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        ops=args.ops,
        is_train=True,
    )
    test_loader = get_arithmetic_dataloader(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        ops=args.ops,
        is_train=False,
    )

    if args.overfit > 0:
        snap = _FixedSampleDataset(train_loader.dataset, args.overfit)
        bs = min(args.batch_size, args.overfit)
        train_loader = DataLoader(
            snap, batch_size=bs, shuffle=True, num_workers=args.num_workers
        )
        test_loader = DataLoader(
            snap, batch_size=bs, shuffle=False, num_workers=args.num_workers
        )
        print(f"[overfit] snapshotted {args.overfit} samples (train == eval set)")

    # ---- model / optim ---- #
    model = ResNet18LateFusion(
        ops=args.ops,
        num_classes=10,
        op_embed_dim=args.op_embed_dim,
        mlp_hidden=args.mlp_hidden,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] parameters={n_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    criterion = nn.CrossEntropyLoss()
    acc_metric = MulticlassAccuracy(num_classes=10).to(device)

    # Persist config + git hash up-front so a mid-run crash is still traceable.
    config_blob: dict[str, Any] = {
        **vars(args),
        "git_hash": _git_hash(),
        "device": str(device),
        "num_params": n_params,
    }
    (run_dir / "config.json").write_text(json.dumps(config_blob, indent=2))

    # ---- train ---- #
    history = []
    best_test_acc = -1.0
    for epoch in range(1, args.epochs + 1):
        train_stats = _run_epoch(
            model,
            train_loader,
            criterion,
            device,
            acc_metric,
            optimizer=optimizer,
            desc=f"epoch {epoch}/{args.epochs} train",
        )
        test_stats = _run_epoch(
            model,
            test_loader,
            criterion,
            device,
            acc_metric,
            optimizer=None,
            desc=f"epoch {epoch}/{args.epochs}  eval",
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
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "test_acc": best_test_acc,
                    "config": config_blob,
                },
                run_dir / "best.pt",
            )

    torch.save(
        {"model": model.state_dict(), "epoch": args.epochs, "config": config_blob},
        run_dir / "last.pt",
    )
    print(f"[done] best test acc={best_test_acc:.4f}  artifacts in {run_dir}")


if __name__ == "__main__":
    main()
