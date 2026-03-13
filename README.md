# Vand3D

Vand3D is a research prototype for 3D anomaly detection and segmentation. It adapts the
[M3D-CLIP](https://github.com/) model to process medical volumes such as MRI scans.
The repository contains training and evaluation scripts as well as configuration files
for experiments on the BraTS brain tumour dataset.

## Setup

```bash
pip install -r requirements.txt
```

The code requires the BraTS dataset preprocessed as NumPy volumes. The dataset directory
must include a `meta.json` file describing the train/validation/test splits and the paths
to each volume and mask. Update the dataset paths in the YAML configs to match your
local setup.

## Training

Use the provided configuration to launch training:

```bash
python train.py --config configs/train_3d.yaml
```

## Validation and Testing

To evaluate a trained model run:

```bash
python validation.py --config configs/eval_3d.yaml
```

For slice‑wise inference use `test.py` with the same configuration file. The config
files under `configs/` control all hyper‑parameters, dataset locations and prompt
settings.
