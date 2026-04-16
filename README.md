# mnist-math

MNIST arithmetic prediction baselines with a config-driven training pipeline.

The task is: given a context digit image and an operation (for example, +1 or -2), predict the resulting target digit class.

## Repository structure

```text
mnist-math/
	configs/
		resnet_baseline.yaml      # ResNet-18 late-fusion baseline config
		simple_baseline.yaml      # Simple 2-layer FC late-fusion baseline config
		overfit.yaml              # Deterministic overfit sanity-check config

	data_pipeline/
		datasets.py               # ArithmeticDataset, FixedSampleDataset, dataloader factory
		__init__.py               # Public exports for data pipeline

	models/
		resnet_baseline.py        # ResNet18LateFusion model
		simple_baseline.py        # SimpleFCLateFusion model
		__init__.py               # Model exports used by training

	utils/
		experiment_config.py      # Dataclasses + YAML load + CLI override merge
		common.py                 # Device selection + seed helpers

	notebooks/
		dataloader_test.ipynb     # Data loader sanity/visualization notebook

	train_baseline.py           # Main training entrypoint
	requirements.txt            # Runtime dependencies
	requirements-dev.txt        # Dev tools (ruff, basedpyright, pre-commit)
	CONTRIBUTING.md             # Collaboration and dev workflow notes
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
```

## Training workflow

All runs go through train_baseline.py and follow this order:

1. Load YAML config (for example, configs/resnet_baseline.yaml).
2. Apply any non-empty CLI flags as overrides.
3. Build run directory at runs/<tag family>/<timestamp>-<tag>/.
4. Create train and eval dataloaders.
5. Optionally replace with FixedSampleDataset if run.overfit > 0.
6. Build selected model from model.name.
7. Train, evaluate each epoch, and save artifacts.

## Config system

Config schema is split into four sections:

- data: dataset, ops, batch_size, num_workers, data_root
- model: name, num_classes, op_embed_dim, mlp_hidden
- optim: lr, weight_decay
- run: epochs, seed, runs_dir, tag, overfit

Model names currently supported:

- resnet18
- simple_fc2

## Common commands

Run ResNet baseline:

```bash
python train_baseline.py --config configs/resnet_baseline.yaml
```

Run simple 2-layer FC baseline:

```bash
python train_baseline.py --config configs/simple_baseline.yaml
```

Run deterministic overfit sanity-check:

```bash
python train_baseline.py --config configs/overfit.yaml
```

Run with inline overrides (CLI wins over YAML):

```bash
python train_baseline.py --config configs/simple_baseline.yaml --epochs 5 --tag quick-check
```

## Run artifacts

Each run writes to a unique directory:

- config.json: resolved config plus runtime metadata (device and git hash)
- history.json: per-epoch train/test loss and accuracy
- best.pt: checkpoint from best test accuracy epoch
- last.pt: checkpoint from final epoch



