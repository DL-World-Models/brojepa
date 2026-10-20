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
    simple_encoder_depth: int = 1
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
    predictor_type: str = "mlp"
    residual_scale_init: float = 0.1
    rotation_angle_init: float = 0.6283185307179586
    rotation_learnable: bool = True
    rotation_init_mode: str = "constant"
    rotation_period: float = 10.0
    use_action_rotation: bool = False
    include_op_embedding_in_head: bool = True

    classifier_hidden_dim: int = 64

    momentum: float = 0.99

    # VICReg parameters
    use_vic_reg: bool = False
    lamda: float = 25  # weight for invariance reg
    miu: float = 25    # weight for variance reg
    nu: float = 1.0    # weight for covariance reg
    gamma: float = 1.0 # hinge loss threshold for variance reg

    # Composition loss parameters
    use_composition_loss: bool = False
    composition_lambda: float = 0.0
    composition_ops: list[int] = field(default_factory=list)
    composition_detach_teacher: bool = True


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
    eval_by_op: bool = False


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
            simple_encoder_depth=int(model_raw.get("simple_encoder_depth", 1)),
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
            predictor_type=str(model_raw.get("predictor_type", "mlp")),
            residual_scale_init=float(model_raw.get("residual_scale_init", 0.1)),
            rotation_angle_init=float(model_raw.get("rotation_angle_init", 0.6283185307179586)),
            rotation_learnable=bool(model_raw.get("rotation_learnable", True)),
            rotation_init_mode=str(model_raw.get("rotation_init_mode", "constant")),
            rotation_period=float(model_raw.get("rotation_period", 10.0)),
            use_action_rotation=bool(model_raw.get("use_action_rotation", False)),
            include_op_embedding_in_head=bool(model_raw.get("include_op_embedding_in_head", True)),

            classifier_hidden_dim=int(model_raw.get("classifier_hidden_dim", 64)),
            momentum=float(model_raw.get("momentum", 0.99)),
            use_vic_reg=bool(model_raw.get("use_vic_reg", False)),
            lamda=float(model_raw.get("lamda", 25)),
            miu=float(model_raw.get("miu", 25)),
            nu=float(model_raw.get("nu", 1.0)),
            gamma=float(model_raw.get("gamma", 1.0)),
            use_composition_loss=bool(model_raw.get("use_composition_loss", False)),
            composition_lambda=float(model_raw.get("composition_lambda", 0.0)),
            composition_ops=[int(v) for v in model_raw.get("composition_ops", [])],
            composition_detach_teacher=bool(model_raw.get("composition_detach_teacher", True)),
        )
        optim = OptimConfig(
            lr=float(optim_raw.get("lr", 1e-3)),
            weight_decay=float(optim_raw.get("weight_decay", 1e-4)),
            cls_weight=float(optim_raw.get("cls_weight", 0.0)),
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
            eval_by_op=bool(run_raw.get("eval_by_op", False)),
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
        "simple_encoder_depth": ("model", "simple_encoder_depth"),
        "op_embedding_type": ("model", "op_embedding_type"),
        "encoder_type": ("model", "encoder_type"),
        "latent_dim": ("model", "latent_dim"),
        "encoder_hidden_dim": ("model", "encoder_hidden_dim"),
        "predictor_type": ("model", "predictor_type"),
        "residual_scale_init": ("model", "residual_scale_init"),
        "rotation_angle_init": ("model", "rotation_angle_init"),
        "rotation_learnable": ("model", "rotation_learnable"),
        "rotation_init_mode": ("model", "rotation_init_mode"),
        "rotation_period": ("model", "rotation_period"),
        "use_action_rotation": ("model", "use_action_rotation"),
        "include_op_embedding_in_head": ("model", "include_op_embedding_in_head"),
        "predictor_dropout": ("model", "predictor_dropout"),
        "predictor_normalize": ("model", "predictor_normalize"),
        "encoder_normalize": ("model", "encoder_normalize"),
        "use_vic_reg": ("model", "use_vic_reg"),
        "lamda": ("model", "lamda"),
        "miu": ("model", "miu"),
        "nu": ("model", "nu"),
        "gamma": ("model", "gamma"),
        "composition_lambda": ("model", "composition_lambda"),
        "composition_ops": ("model", "composition_ops"),
        "use_composition_loss": ("model", "use_composition_loss"),
        "composition_detach_teacher": ("model", "composition_detach_teacher"),
        "cls_weight": ("optim", "cls_weight"),
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
        "eval_by_op": ("run", "eval_by_op"),
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
    p.add_argument("--simple-encoder-depth", type=int, default=None)
    p.add_argument("--op-embedding-type", type=str, default=None)
    p.add_argument("--encoder-type", type=str, default=None, choices=["mlp", "cnn", "vit", "resnet", "resnet18"])
    p.add_argument("--latent-dim", type=int, default=None)
    p.add_argument("--encoder-hidden-dim", type=int, default=None)
    p.add_argument("--encoder-normalize", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument(
        "--predictor-type",
        type=str,
        default=None,
        choices=[
            "mlp",
            "residual",
            "scaled_residual",
            "linear_magnitude",
            "block_rotation",
            "simple_rotation",
        ],
    )
    p.add_argument("--residual-scale-init", type=float, default=None)
    p.add_argument("--rotation-angle-init", type=float, default=None)
    p.add_argument("--rotation-learnable", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--rotation-init-mode", type=str, default=None, choices=["constant", "cyclic_harmonics"])
    p.add_argument("--rotation-period", type=float, default=None)
    p.add_argument("--use-action-rotation", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--include-op-embedding-in-head", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--predictor-dropout", type=float, default=None)
    p.add_argument("--predictor-normalize", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--use-vic-reg", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--lamda", type=float, default=None)
    p.add_argument("--miu", type=float, default=None)
    p.add_argument("--nu", type=float, default=None)
    p.add_argument("--gamma", type=float, default=None)
    p.add_argument("--use-composition-loss", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--composition-lambda", type=float, default=None)
    p.add_argument("--composition-ops", type=int, nargs="+", default=None)
    p.add_argument("--composition-detach-teacher", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--cls-weight", type=float, default=None)
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
    p.add_argument("--eval-by-op", action=argparse.BooleanOptionalAction, default=None)
    return p.parse_args()
