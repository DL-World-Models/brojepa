# mnist-math

## Training

Run baseline training directly:

- `python train_baseline.py`

Use the overfit sanity config directly (no launcher script):

- `python train_baseline.py --config configs/overfit.yaml`

Override any value inline when needed:

- `python train_baseline.py --config configs/baseline.yaml --epochs 5 --tag quick-check`
