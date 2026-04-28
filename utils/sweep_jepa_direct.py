from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path


def parse_float_list(raw: str) -> list[float]:
    return [float(item) for item in raw.split(",") if item]


def parse_str_list(raw: str) -> list[str]:
    return [item for item in raw.split(",") if item]


def latest_run_dir(runs_dir: Path, tag: str) -> Path:
    matches = sorted(runs_dir.glob(f"*-{tag}_*"), key=lambda path: path.stat().st_mtime)
    if not matches:
        raise FileNotFoundError(f"No run directory found for tag={tag!r} in {runs_dir}")
    return matches[-1]


def best_metrics(history_path: Path) -> dict[str, float | int]:
    history = json.loads(history_path.read_text())
    best_epoch = max(history, key=lambda record: record["test"]["accuracy"])
    best_rollout_epoch = max(
        history,
        key=lambda record: record.get("rollout", {}).get("cls_accuracy", 0.0),
    )
    return {
        "best_direct_epoch": int(best_epoch["epoch"]),
        "best_direct_acc": float(best_epoch["test"]["accuracy"]),
        "best_direct_cls_loss": float(best_epoch["test"]["cls_loss"]),
        "best_rollout_epoch": int(best_rollout_epoch["epoch"]),
        "best_rollout_cls": float(best_rollout_epoch.get("rollout", {}).get("cls_accuracy", 0.0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/jepa_compositionality.yaml")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--runs-dir", default="runs/jepa")
    parser.add_argument("--tag-prefix", default="direct_sweep")
    parser.add_argument("--op-embedding-types", default="additive_embedding,additive_embedding_normalized")
    parser.add_argument("--predictor-types", default="mlp,scaled_residual")
    parser.add_argument("--composition-lambdas", default="0,0.1,0.3,1.0")
    parser.add_argument("--cls-weights", default="0")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    results: list[dict[str, object]] = []
    grid = itertools.product(
        parse_str_list(args.op_embedding_types),
        parse_str_list(args.predictor_types),
        parse_float_list(args.composition_lambdas),
        parse_float_list(args.cls_weights),
    )

    for idx, (op_embedding_type, predictor_type, composition_lambda, cls_weight) in enumerate(grid, start=1):
        use_composition = composition_lambda > 0.0
        tag = (
            f"{args.tag_prefix}_e{idx:02d}_"
            f"{op_embedding_type}_{predictor_type}_cl{composition_lambda:g}_cw{cls_weight:g}"
        )
        cmd = [
            sys.executable,
            "train_jepa.py",
            "--config",
            args.config,
            "--epochs",
            str(args.epochs),
            "--tag",
            tag,
            "--op-embedding-type",
            op_embedding_type,
            "--predictor-type",
            predictor_type,
            "--composition-lambda",
            str(composition_lambda),
            "--cls-weight",
            str(cls_weight),
            "--composition-detach-teacher",
            "--use-composition-loss" if use_composition else "--no-use-composition-loss",
        ]
        if args.lr is not None:
            cmd.extend(["--lr", str(args.lr)])

        print("[sweep]", " ".join(cmd), flush=True)
        if args.dry_run:
            continue

        subprocess.run(cmd, check=True)
        run_dir = latest_run_dir(runs_dir, tag)
        metrics = best_metrics(run_dir / "history.json")
        result = {
            "tag": tag,
            "run_dir": str(run_dir),
            "op_embedding_type": op_embedding_type,
            "predictor_type": predictor_type,
            "composition_lambda": composition_lambda,
            "cls_weight": cls_weight,
            **metrics,
        }
        results.append(result)
        print("[result]", json.dumps(result, sort_keys=True), flush=True)

    if results:
        out_path = runs_dir / f"{args.tag_prefix}_summary.json"
        out_path.write_text(json.dumps(results, indent=2))
        best = max(results, key=lambda row: row["best_direct_acc"])
        print(f"[summary] wrote {out_path}")
        print("[summary] best direct:", json.dumps(best, sort_keys=True))


if __name__ == "__main__":
    main()
