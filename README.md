# mnist-math

MNIST arithmetic prediction baselines with a config-driven training pipeline.

The task is: given a context digit image and an operation (for example, +1 or -2), predict the resulting target digit class.

## Repository structure

```text
mnist-math/
	configs/
		resnet_baseline.yaml      # ResNet-18 baseline (embedding mode for ops)
		resnet_linear.yaml        # ResNet-18 with linear op projection
		resnet_fourier.yaml       # ResNet-18 with Fourier op encoding
		simple_baseline.yaml      # Simple 2-layer FC late-fusion baseline
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

## Testing

All code has backward compatibility tests. Before committing or merging changes, run:

```bash
pytest tests/ -v
```

See [TESTING.md](TESTING.md) for detailed testing guide including how to:
- Run tests for specific modules
- Run with coverage reports
- Run tests in watch mode
- Add new tests

Quick examples:

```bash
# Run all tests
pytest tests/ -v

# Run only model tests
pytest tests/test_models.py -v

# Run with coverage
pytest tests/ --cov=models --cov=utils --cov=data_pipeline --cov-report=html
```

## Config system

Config schema is split into four sections:

- data: dataset, ops, batch_size, num_workers, data_root
- model: name, num_classes, op_embed_dim, mlp_hidden, op_embedding_type
- optim: lr, weight_decay
- run: epochs, seed, runs_dir, tag, overfit

Model names currently supported:

- resnet18
- simple_fc2
- simple_fc_fourier

### Operation Embedding Modes

For ResNet18, the `op_embedding_type` field controls how integer operations are encoded:

- **embedding** (default): Fixed embedding lookup for discrete operations (requires `ops` to be unique)
- **linear**: Linear projection of raw operation values (works with any values)
- **fourier**: Fourier features (sin/cos) + linear projection for smooth generalization to unseen operations

Example configs:
- `configs/resnet_baseline.yaml` (embedding mode)
- `configs/resnet_linear.yaml` (linear mode)
- `configs/resnet_fourier.yaml` (Fourier mode)

## Common commands

Run ResNet baseline (embedding mode):

```bash
python train_baseline.py --config configs/resnet_baseline.yaml
```

Run ResNet with linear operation projection:

```bash
python train_baseline.py --config configs/resnet_linear.yaml
```

Run ResNet with Fourier operation encoding (for generalization to unseen ops):

```bash
python train_baseline.py --config configs/resnet_fourier.yaml
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
python train_baseline.py --config configs/resnet_baseline.yaml --epochs 20 --op-embedding-type fourier --tag exp1
```

## Run artifacts

Each run writes to a unique directory:

- config.json: resolved config plus runtime metadata (device and git hash)
- history.json: per-epoch train/test loss and accuracy
- best.pt: checkpoint from best test accuracy epoch
- last.pt: checkpoint from final epoch



