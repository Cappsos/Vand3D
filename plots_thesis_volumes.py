#!/usr/bin/env python3
import os, re, argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from utils.volume_utils import reconstruct_volume_from_slices, combine_subvolumes_from_folder
from utils.io import load_gt_mask, load_brain_volume  # expects <PID>/<PID>-seg.npy and <PID>-t2w.npy

def numeric_key(name: str):
    m = re.search(r"(\d+)", name)
    return int(m.group(1)) if m else float("inf")

def ensure_dhw(vol: np.ndarray) -> np.ndarray:
    """Accept (D,H,W) or (H,W,D); return (D,H,W)."""
    if vol.ndim != 3:
        raise ValueError(f"Expected 3D volume, got {vol.shape}")
    if vol.shape[1] == 240 and vol.shape[2] == 240:  # (D,H,W)
        return vol
    if vol.shape[0] == 240 and vol.shape[1] == 240:  # (H,W,D)
        return np.transpose(vol, (2, 0, 1))
    raise ValueError(f"Unexpected shape {vol.shape}")

def load_pred_volume(patient_dir, mode, prefix, full_depth, target_h, target_w, block_depth):
    """mode='subvolume' → fuse overlapping blocks; mode='slice' → stack per-slice .npy."""
    if mode == "subvolume":
        vol = combine_subvolumes_from_folder(
            patient_dir,
            prefix=prefix,
            full_depth=full_depth,
            block_depth=block_depth
        )  # (D,H,W)
    else:
        vol = reconstruct_volume_from_slices(
            patient_dir,
            prefix=prefix,
            full_depth=full_depth,
            target_h=target_h,
            target_w=target_w
        )  # (D,H,W)
    return vol.astype(np.float32)

def robust_01(x: np.ndarray) -> np.ndarray:
    """Percentile [1,99] normalization to [0,1] for display."""
    p1, p99 = np.percentile(x, [1, 99])
    d = max(1e-6, p99 - p1)
    return np.clip((x - p1) / d, 0, 1)

def overlay_tp_fp_fn(bg01, pred_bin, gt_bin, a_tp=0.70, a_fp=0.55, a_fn=0.60):
    """RGB overlay with TP=green, FP=red, FN=blue atop grayscale bg01 in [0,1]."""
    bg01 = np.clip(bg01, 0, 1).astype(np.float32)
    rgb = np.stack([bg01, bg01, bg01], axis=-1)

    TP = np.logical_and(pred_bin, gt_bin)
    FP = np.logical_and(pred_bin, np.logical_not(gt_bin))
    FN = np.logical_and(np.logical_not(pred_bin), gt_bin)

    rgb[..., 0] = np.where(FP, (1 - a_fp)*rgb[..., 0] + a_fp*1.0, rgb[..., 0])  # red
    rgb[..., 1] = np.where(TP, (1 - a_tp)*rgb[..., 1] + a_tp*1.0, rgb[..., 1])  # green
    rgb[..., 2] = np.where(FN, (1 - a_fn)*rgb[..., 2] + a_fn*1.0, rgb[..., 2])  # blue
    return np.clip(rgb, 0, 1)

def main():
    ap = argparse.ArgumentParser("Fuse predictions and save TP/FP/FN overlays per slice.")
    ap.add_argument("--pred_root",  default="/mnt/external/results_nicoc/3D_final_final/full_shot_fused",
                    help="Root with prediction results; expects <pred_root>/test/<PID>/anomaly_map_depth_*.npy")
    ap.add_argument("--gt_root",    default="/mnt/external/data/BraTS_3D",
                    help="GT/T2w root; expects <PID>/<PID>-seg.npy and <PID>-t2w.npy")
    ap.add_argument("--out_root",   default="/mnt/external/results_nicoc/plots_volumes/full_shot_fused",
                    help="Where to save overlays")
    ap.add_argument("--reconstruct_mode", choices=["subvolume","slice"], default="subvolume")
    ap.add_argument("--prefix",     default="anomaly_map_depth_")
    ap.add_argument("--depth",      type=int, default=32, help="Block depth used at inference (for subvolume mode)")
    ap.add_argument("--full_depth", type=int, default=155)
    ap.add_argument("--target_h",   type=int, default=240)
    ap.add_argument("--target_w",   type=int, default=240)
    ap.add_argument("--threshold",  type=float, default=1.38, help="Threshold in the same scale as saved maps")
    ap.add_argument("--patients",   type=str,   default="", help="Comma-separated PIDs; default: all in test/")
    ap.add_argument("--zrev", action="store_true", help="Reverse Z for predictions (debug)")
    ap.add_argument("--zshift", type=int, default=0, help="Shift Z for predictions (debug)")
    args = ap.parse_args()

    os.makedirs(args.out_root, exist_ok=True)

    # patients live under <pred_root>/test/<PID>
    test_root = os.path.join(args.pred_root, "test")
    if not os.path.isdir(test_root):
        raise FileNotFoundError(f"Expected test dir at {test_root}")

    if args.patients.strip():
        pids = [p.strip() for p in args.patients.split(",") if p.strip()]
    else:
        pids = sorted([d for d in os.listdir(test_root)
                       if os.path.isdir(os.path.join(test_root, d))])

    print(f"Found {len(pids)} patients in {test_root}")

    for pid in pids:
        pdir = os.path.join(test_root, pid)
        print(f"\n[{pid}] fusing predictions from {pdir}")

        pred_vol = load_pred_volume(
            patient_dir=pdir,
            mode=args.reconstruct_mode,
            prefix=args.prefix,
            full_depth=args.full_depth,
            target_h=args.target_h,
            target_w=args.target_w,
            block_depth=args.depth
        )
        pred_vol = ensure_dhw(pred_vol)

        # depth debug options
        if args.zrev:
            pred_vol = pred_vol[::-1]
        if args.zshift != 0:
            D = pred_vol.shape[0]
            s = args.zshift
            if s > 0:
                pred_vol = np.pad(pred_vol, ((s,0),(0,0),(0,0)), mode="constant")[:D]
            else:
                s = -s
                pred_vol = np.pad(pred_vol, ((0,s),(0,0),(0,0)), mode="constant")[s:]

        # GT + brain (T2w)
        gt = load_gt_mask(pid, args.gt_root)         # (D,H,W) or (H,W,D)
        brain = load_brain_volume(pid, args.gt_root) # (D,H,W) or (H,W,D)
        #gt = np.rot90(gt, k=1)
        #brain = np.rot90(brain, k=1)
        gt = ensure_dhw(gt).astype(np.uint8)
        brain = ensure_dhw(brain).astype(np.float32)

        pred_bin = (pred_vol > args.threshold).astype(np.uint8)
        #pred_bin = np.rot90(pred_bin, k=1)

        out_dir = os.path.join(args.out_root, pid)
        os.makedirs(out_dir, exist_ok=True)

        D, H, W = pred_vol.shape
        legend = [
            Patch(facecolor='green', label='True Positive'),
            Patch(facecolor='red',   label='False Positive'),
            Patch(facecolor='blue',  label='False Negative')
        ]

        for z in range(D):
            bg01 = robust_01(brain[z])
            ov = overlay_tp_fp_fn(bg01, pred_bin[z].astype(bool), (gt[z] > 0))
            ov = np.rot90(ov, k=1)

            fig, ax = plt.subplots(1, 1, figsize=(4.2, 4.2))
            ax.imshow(ov, origin='lower')
            ax.set_title(f"Sub: {pid.split('-')[-2]}  Slice: {z:03d}", fontsize=10)
            ax.axis('off')
            #ax.legend(handles=legend, loc='lower left', fontsize=8, framealpha=0.7)

            out_png = os.path.join(out_dir, f"slice_{z:03d}.png")
            plt.savefig(out_png, dpi=200, bbox_inches='tight')
            plt.close(fig)

        print(f"Saved overlays → {out_dir}")

if __name__ == "__main__":
    main()
