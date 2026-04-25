from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, cast

import yaml
import argparse


@dataclass(slots=True)
class DataConfig:
    dataset: str = "MNIST"
    ops: list[int] = field(default_factory=lambda: [1, 2, -1, -2])
    test_ops: list[int] | None = None  # If None, defaults to ops
    batch_size: int = 128
    num_workers: int = 0
    data_root: str = "./data"


@dataclass(slots=True)
class ModelConfig:
    name: str = "resnet18"
    num_classes: int = 10
    op_embed_dim: int = 32
    mlp_hidden: int = 256
    op_embedding_type: str = "embedding"  # "embedding", "linear", or "fourier"

    encoder_type: str = "cnn"  # "cnn", "mlp", or "vit"
    latent_dim: int = 64
    encoder_normalize: bool = True
    activation: str = "relu"
    encoder_hidden_dim: int = 256

    patch_size: int = 7
    d_model: int = 128
    depth: int = 4
    nhead: int = 4
    mlp_dim: int = 256
    use_cls_token: bool = False

    predictor_dropout: float = 0.2
    predictor_hidden_dim: list[int] = field(default_factory=lambda: [256, 256])
    predictor_normalize: bool = True

    classifier_hidden_dim: int = 64

    momentum: float = 0.99


@dataclass(slots=True)
class OptimConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-4
    cls_weight: float = 0.0


@dataclass(slots=True)
class RunConfig:
    epochs: int = 10
    warmup_epochs: int = 0
    warmup_lr: float | None = None
    freeze_context_post_warmup: bool = False
    seed: int = 42
    runs_dir: str = "runs/baseline"
    tag: str = "run"
    overfit: int = 0


@dataclass(slots=True)
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    run: RunConfig = field(default_factory=RunConfig)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ExperimentConfig":
        data_raw = raw.get("data", {})
        model_raw = raw.get("model", {})
        optim_raw = raw.get("optim", {})
        run_raw = raw.get("run", {})

        data = DataConfig(
            dataset=data_raw.get("dataset", "MNIST"),
            ops=[int(v) for v in data_raw.get("ops", [1, 2, -1, -2])],
            test_ops=(
                [int(v) for v in data_raw.get("test_ops")]
                if data_raw.get("test_ops")
                else None
            ),
            batch_size=int(data_raw.get("batch_size", 128)),
            num_workers=int(data_raw.get("num_workers", 0)),
            data_root=data_raw.get("data_root", "./data"),
        )
        model = ModelConfig(
            name=str(model_raw.get("name", "resnet18")),
            num_classes=int(model_raw.get("num_classes", 10)),
            op_embed_dim=int(model_raw.get("op_embed_dim", 32)),
            mlp_hidden=int(model_raw.get("mlp_hidden", 256)),
            op_embedding_type=str(model_raw.get("op_embedding_type", "embedding")),

            encoder_type=str(model_raw.get("encoder_type", "cnn")),
            latent_dim=int(model_raw.get("latent_dim", 64)),
            encoder_normalize=bool(model_raw.get("encoder_normalize", True)),
            activation=str(model_raw.get("activation", "relu")),
            patch_size=int(model_raw.get("patch_size", 7)),
            d_model=int(model_raw.get("d_model", 128)),
            depth=int(model_raw.get("depth", 4)),
            nhead=int(model_raw.get("nhead", 4)),
            mlp_dim=int(model_raw.get("mlp_dim", 256)),
            use_cls_token=bool(model_raw.get("use_cls_token", False)),

            predictor_dropout=float(model_raw.get("predictor_dropout", 0.2)),
            predictor_hidden_dim=model_raw.get("predictor_hidden_dim",[256, 256]),
            predictor_normalize=bool(model_raw.get("predictor_normalize", True)),

            classifier_hidden_dim=int(model_raw.get("classifier_hidden_dim", 64)),
            momentum=float(model_raw.get("momentum", 0.99)),
        )
        optim = OptimConfig(
            lr=float(optim_raw.get("lr", 1e-3)),
            weight_decay=float(optim_raw.get("weight_decay", 1e-4)),
            cls_weight=float(optim_raw.get("cls_weight", 0.0))
        )
        run = RunConfig(
            epochs=int(run_raw.get("epochs", 10)),
            warmup_epochs=int(run_raw.get("warmup_epochs", 0)),
            warmup_lr=(
                float(run_raw["warmup_lr"])
                if run_raw.get("warmup_lr") is not None
                else None
            ),
            freeze_context_post_warmup=bool(run_raw.get("freeze_context_post_warmup", False)),
            seed=int(run_raw.get("seed", 42)),
            runs_dir=run_raw.get("runs_dir", "runs/baseline"),
            tag=run_raw.get("tag", "run"),
            overfit=int(run_raw.get("overfit", 0)),
        )

        # Backward-compatibility for earlier typoed top-level key.
        if run.warmup_epochs == 0 and bool(raw.get("warnmup_encoders", False)):
            run.warmup_epochs = 1
        return cls(data=data, model=model, optim=optim, run=run)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_experiment_config(config_path: str | Path) -> ExperimentConfig:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {path}. Create configs/baseline.yaml or pass --config."
        )

    parsed = yaml.safe_load(path.read_text())
    if parsed is None:
        parsed = {}
    if not isinstance(parsed, dict):
        raise ValueError("Config root must be a mapping/object")

    return ExperimentConfig.from_dict(cast(dict[str, Any], parsed))


def apply_cli_overrides(
    config: ExperimentConfig,
    overrides: dict[str, Any],
) -> ExperimentConfig:
    # Only explicitly provided CLI flags (non-None) override YAML values.
    mapping: dict[str, tuple[str, str]] = {
        "dataset": ("data", "dataset"),
        "ops": ("data", "ops"),
        "test_ops": ("data", "test_ops"),
        "batch_size": ("data", "batch_size"),
        "num_workers": ("data", "num_workers"),
        "data_root": ("data", "data_root"),
        "model_name": ("model", "name"),
        "num_classes": ("model", "num_classes"),
        "op_embed_dim": ("model", "op_embed_dim"),
        "mlp_hidden": ("model", "mlp_hidden"),
        "op_embedding_type": ("model", "op_embedding_type"),
        "lr": ("optim", "lr"),
        "weight_decay": ("optim", "weight_decay"),
        "epochs": ("run", "epochs"),
        "warmup_epochs": ("run", "warmup_epochs"),
        "warmup_lr": ("run", "warmup_lr"),
        "freeze_context_post_warmup": ("run", "freeze_context_post_warmup"),
        "seed": ("run", "seed"),
        "runs_dir": ("run", "runs_dir"),
        "tag": ("run", "tag"),
        "overfit": ("run", "overfit"),
    }

    for key, value in overrides.items():
        if key == "config" or value is None or key not in mapping:
            continue
        section_name, field_name = mapping[key]
        section = getattr(config, section_name)
        setattr(section, field_name, value)

    return config

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/resnet_baseline.yaml")
    p.add_argument("--dataset", type=str, default=None, choices=["MNIST", "EMNIST"])
    p.add_argument("--ops", type=int, nargs="+", default=None)
    p.add_argument("--test-ops", type=int, nargs="+", default=None)
    p.add_argument("--model-name", type=str, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--data-root", type=str, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--warmup-epochs", type=int, default=None)
    p.add_argument("--warmup-lr", type=float, default=None)
    p.add_argument("--freeze-context-post-warmup", action="store_true", default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--num-classes", type=int, default=None)
    p.add_argument("--op-embed-dim", type=int, default=None)
    p.add_argument("--mlp-hidden", type=int, default=None)
    p.add_argument("--op-embedding-type", type=str, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--runs-dir", type=str, default=None)
    p.add_argument("--tag", type=str, default=None)
    p.add_argument(
        "--overfit",
        type=int,
        default=None,
        help="If > 0, train on a deterministic snapshot of N samples "
        "(sanity check that the model + loop can drive loss to ~0).",
    )
    return p.parse_args()
