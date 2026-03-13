# Configuration Files

YAML config files control all hyperparameters, data paths, and experiment settings. Each pipeline stage has its own configs.

## Naming Convention

| Prefix | Stage | Script |
|--------|-------|--------|
| `train_*.yaml` | Training | `train.py` |
| `eval_*.yaml` | Validation (threshold sweep) | `validation.py` |
| `infer_*.yaml` | Test inference | `test.py` |
| `score_*.yaml` | Final metric computation | `score.py` / `score_final.py` |

## Suffixes

| Suffix | Meaning |
|--------|---------|
| `_2d` | Slice-wise (depth=1) mode |
| `_3d` | Volumetric (depth=32) mode |
| `_ensemble` | Prompt ensemble enabled |
| `_no_ensemble` | Single prompt (no ensemble) |
| `_fuse` | Multi-scale fusion |
| `_ensemble_k=N` | K-shot experiment with N training patients |
| `_multi_scale` | Multi-scale crop training |

## Other Files

- `prompts_brats.yaml` — Healthy and tumour text prompts for BraTS

## Data Paths

All data paths in the configs use relative placeholders (e.g., `./data/BraTS_3D`). Update these to match your local data directory before running.
