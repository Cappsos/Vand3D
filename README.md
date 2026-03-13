# VAND3D

Vision-Language Anomaly Detection for 3D Medical Volumes.

VAND3D adapts [M3D-CLIP](https://huggingface.co/GoodBaiBai88/M3D-CLIP) for zero- and few-shot **3D anomaly detection and segmentation** in brain MRI scans (BraTS dataset). The approach is inspired by CLIP-based anomaly detection methods proposed in the VAND challenge report (https://arxiv.org/abs/2305.17382), which align visual features with text embeddings through additional linear layers to generate anomaly maps. 

In our implementation, a lightweight trainable linear adapter aligns frozen vision transformer features with text prompt embeddings, producing voxel-level anomaly maps via cosine similarity.
```
                                    ┌──────────────────┐
  3D MRI Sub-volume ───► Frozen ViT ──► Patch Tokens ──► Linear Adapter ──┐
   (32×256×256)          (M3D-CLIP)     (layers 3,4,11)                   │
                                                                    cosine similarity
                                                                          │
  Text Prompts ─────► Frozen BERT ──► Text Embeddings ──► Prompt Centroid ┘
  ("healthy brain",    (M3D-CLIP)     (healthy/tumour)          │
   "tumorous brain")                                            ▼
                                                         Anomaly Map
                                                        (32×240×240)
```

## Repository Structure

```
├── train.py                 # Training script (adapter + focal/dice loss)
├── validation.py            # Validation with threshold sweep
├── test.py                  # Test-time inference + volume reconstruction
├── score.py                 # Metrics: Dice, Hausdorff, ROC-AUC, AP, F1
├── score_final.py           # Extended metrics: lesion recall, per-size analysis
│
├── models/
│   ├── m3dclip.py           # M3D-CLIP loading, text embedding, prompt centroids
│   ├── adapters.py          # LinearLayer: trainable adapter module
│   └── decoder.py           # FiLMUNet3D: alternative decoder head
│
├── datasets/
│   └── dataset3d.py         # BraTS3DSubVolumeDataset (PyTorch Dataset)
│
├── utils/
│   ├── transforms.py        # 3D volume/mask transforms for M3D-CLIP
│   ├── metrics.py           # Dice, Hausdorff, ROC-AUC, AP, F1, IoU
│   ├── loss.py              # FocalLoss, BinaryDiceLoss
│   ├── io.py                # Volume/mask I/O, evaluation result persistence
│   ├── volume_utils.py      # Volume reconstruction from slices/subvolumes
│   ├── fusion.py            # Laplacian fusion, multi-scale crop fusion
│   └── postproc.py          # Post-processing: z-score, Gaussian, region growing
│
├── modified_model/          # Modified M3D-CLIP with intermediate feature extraction
├── configs/                 # YAML configs for train/eval/infer/score
├── patients_seeds/          # Patient ID lists for k-shot experiments
├── bash_experiments/        # Shell scripts to reproduce experiment sweeps
├── scripts/                 # Data exploration, analysis, and plotting helpers
└── requirements.txt         # Pinned Python dependencies
```

## Requirements

- Python 3.10+
- CUDA-capable GPU (tested on NVIDIA A100 / RTX 3090)
- ~16 GB GPU memory for batch size 1 at depth 32

## Installation

```bash
git clone https://github.com/<your-username>/VAND3D.git
cd VAND3D
pip install -r requirements.txt
```

The M3D-CLIP model weights are downloaded automatically from HuggingFace on first run (`GoodBaiBai88/M3D-CLIP`).

## Data Preparation

VAND3D expects the [BraTS 2023](https://www.synapse.org/Synapse:syn51156910) dataset preprocessed as NumPy volumes.

### Expected directory layout

```
data/BraTS_3D/
├── meta.json
├── BraTS-GLI-00000-000/
│   ├── BraTS-GLI-00000-000-t2w.npy    # T2-weighted volume (240×240×155)
│   └── BraTS-GLI-00000-000-seg.npy    # Segmentation mask  (240×240×155)
├── BraTS-GLI-00001-000/
│   ├── ...
```

### `meta.json` format

```json
{
  "train": {
    "BraTS-GLI-00000-000": {
      "img_path": "BraTS-GLI-00000-000/BraTS-GLI-00000-000-t2w.npy",
      "mask_path": "BraTS-GLI-00000-000/BraTS-GLI-00000-000-seg.npy"
    }
  },
  "val": { ... },
  "test": { ... }
}
```

Update the `train_data_path` / `test_data_path` / `gt_root` fields in the YAML configs under `configs/` to point to your data directory.

## Quick Start

The pipeline runs in four stages. Each stage uses a YAML config file.

### 1. Train

```bash
python train.py --config configs/train_3d.yaml
```

Trains the linear adapter on top of frozen M3D-CLIP features. Checkpoints are saved to `save_path` specified in the config.

### 2. Validate

```bash
python validation.py --config configs/eval_3d.yaml
```

Runs inference on the validation set and sweeps binarization thresholds to find the optimal one. Saves `best_threshold.json`.

### 3. Test

```bash
python test.py --config configs/infer_3d.yaml
```

Runs inference on the test set using the best threshold from validation. Produces per-patient anomaly maps.

### 4. Score

```bash
python score.py --config configs/score_3d.yaml
```

Reconstructs full-resolution 3D volumes from sub-volume predictions and computes metrics (Dice, Hausdorff-95, ROC-AUC, AP, F1-max, IoU). For extended metrics including lesion-level analysis, use `score_final.py`.

## K-Shot Experiments

To run few-shot experiments with a fixed subset of training patients:

```bash
python train.py --config configs/train_3d_ensemble_k=5.yaml
```

Patient subsets are defined in `patients_seeds/patients_k=N.txt`. Available values: k = 1, 5, 10, 15, 20, 30, 40, 50.

To reproduce the full sweep of k-shot experiments, see `bash_experiments/run_experiments_train_rationale.sh`.

## Configuration

All hyperparameters are controlled through YAML config files. See [`configs/README.md`](configs/README.md) for details on the config file naming convention and key parameters.

Key parameters in training configs:
| Parameter | Default | Description |
|-----------|---------|-------------|
| `features_list` | `[3, 4, 11]` | ViT layers to extract patch tokens from |
| `depth` | `32` | Sub-volume depth in slices |
| `image_size` | `256` | Spatial resolution (H, W) |
| `temperature` | `0.2` | Softmax temperature for cosine similarity |
| `prompt_ensemble` | `True` | Average multiple text prompts per class |
| `epoch` | `20` | Number of training epochs |
| `learning_rate` | `1e-4` | Adam learning rate |

```

## License
This project is licensed under the [MIT License](LICENSE).\

TODO - Add license.
