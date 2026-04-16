# mnist-math

## Training

Run ResNet baseline:

- `python train_baseline.py --config configs/resnet_baseline.yaml`

Run simple 2-layer FC baseline:

- `python train_baseline.py --config configs/simple_baseline.yaml`

Use the overfit sanity config directly (no launcher script):

- `python train_baseline.py --config configs/overfit.yaml`

Override any value inline when needed:

- `python train_baseline.py --config configs/simple_baseline.yaml --epochs 5 --tag quick-check`
