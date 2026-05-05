# mnist-math

MNIST arithmetic prediction with YAML configs.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

Use a config from [configs/](configs/) with either training entrypoint:

```bash
python train_baseline.py --config configs/resnet_baseline.yaml
python train_baseline.py --config configs/simple_baseline.yaml
python train_baseline.py --config configs/overfit.yaml
python train_jepa.py --config configs/jepa_test.yaml
```

CLI flags override YAML values. For example:

```bash
python train_baseline.py --config configs/resnet_baseline.yaml --epochs 20 --tag exp1
```



