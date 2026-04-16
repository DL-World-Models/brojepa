from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, cast

import yaml


@dataclass(slots=True)
class DataConfig:
    dataset: str = "MNIST"
    ops: list[int] = field(default_factory=lambda: [1, 2, -1, -2])
    batch_size: int = 128
    num_workers: int = 0
    data_root: str = "./data"


@dataclass(slots=True)
class ModelConfig:
    name: str = "resnet18"
    num_classes: int = 10
    op_embed_dim: int = 32
    mlp_hidden: int = 256


@dataclass(slots=True)
class OptimConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-4


@dataclass(slots=True)
class RunConfig:
    epochs: int = 10
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
            batch_size=int(data_raw.get("batch_size", 128)),
            num_workers=int(data_raw.get("num_workers", 0)),
            data_root=data_raw.get("data_root", "./data"),
        )
        model = ModelConfig(
            name=str(model_raw.get("name", "resnet18")),
            num_classes=int(model_raw.get("num_classes", 10)),
            op_embed_dim=int(model_raw.get("op_embed_dim", 32)),
            mlp_hidden=int(model_raw.get("mlp_hidden", 256)),
        )
        optim = OptimConfig(
            lr=float(optim_raw.get("lr", 1e-3)),
            weight_decay=float(optim_raw.get("weight_decay", 1e-4)),
        )
        run = RunConfig(
            epochs=int(run_raw.get("epochs", 10)),
            seed=int(run_raw.get("seed", 42)),
            runs_dir=run_raw.get("runs_dir", "runs/baseline"),
            tag=run_raw.get("tag", "run"),
            overfit=int(run_raw.get("overfit", 0)),
        )
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
        "batch_size": ("data", "batch_size"),
        "num_workers": ("data", "num_workers"),
        "data_root": ("data", "data_root"),
        "model_name": ("model", "name"),
        "num_classes": ("model", "num_classes"),
        "op_embed_dim": ("model", "op_embed_dim"),
        "mlp_hidden": ("model", "mlp_hidden"),
        "lr": ("optim", "lr"),
        "weight_decay": ("optim", "weight_decay"),
        "epochs": ("run", "epochs"),
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
