#!/usr/bin/env python3
import os, re, glob, argparse
import numpy as np
from PIL import Image

# --- use the SAME GT loader as your big score script ---
from utils.io import load_gt_mask  # returns (D,H,W) binary/0-1 (per your repo)

def numeric_key(path: str):
    """Sort by the first integer in filename (fallback to name)."""
    name = os.path.basename(path)
    m = re.search(r"(\d+)", name)
    return (int(m.group(1)) if m else float('inf'), name)

def load_pred_volume_from_images(patient_dir: str,
                                 depth: int = 155,
                                 H: int = 240,
                                 W: int = 240,
                                 exts=("jpg", "jpeg", "png")) -> np.ndarray:
    """Load per-slice prediction images → (D,H,W), float32 in 0..255 (no rescale)."""
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(patient_dir, f"*.{e}")))
    if len(files) == 0:
        raise FileNotFoundError(f"No image slices found in {patient_dir} (exts={exts})")

    files = sorted(files, key=numeric_key)

    slices = []
    for fp in files:
        with Image.open(fp) as im:
            im = im.convert("L")  # 8-bit gray
            if im.size != (W, H):
                im = im.resize((W, H), resample=Image.BILINEAR)
            arr = np.array(im, dtype=np.uint8).astype(np.float32)  # keep 0..255
        slices.append(arr)

    # pad/truncate to depth
    if len(slices) < depth:
        pad = [np.zeros((H, W), dtype=np.float32) for _ in range(depth - len(slices))]
        slices.extend(pad)
    slices = slices[:depth]

    vol = np.stack(slices, axis=0)  # (D,H,W)
    return vol

def dice3d(pred_bin: np.ndarray, gt_bin: np.ndarray, eps=1e-8) -> float:
    inter = np.logical_and(pred_bin, gt_bin).sum()
    denom = pred_bin.sum() + gt_bin.sum()
    return float((2.0 * inter) / (denom + eps)) if denom > 0 else 1.0

def get_patients(pred_root: str) -> list:
    """Infer patients from prediction root (folders)."""
    pids = [d for d in os.listdir(pred_root) if os.path.isdir(os.path.join(pred_root, d))]
    return sorted(pids)

def main():
    ap = argparse.ArgumentParser(description="Stack JPEG predictions, load GT via utils.io.load_gt_mask, sweep 0..255, compute 3D Dice.")
    ap.add_argument("--pred_root", default= "/mnt/external/results_thesis_final/zero_shot/test/heatmaps", help="Root with per-patient prediction JPEG folders.")
    ap.add_argument("--gt_root",   default= "/mnt/external/data/BraTS_3D", help="GT root (same you pass to the big score script).")
    ap.add_argument("--patients",  type=str, default="", help="Comma-separated patient IDs to evaluate; default: all in pred_root.")
    ap.add_argument("--depth",     type=int, default=155)
    ap.add_argument("--height",    type=int, default=240)
    ap.add_argument("--width",     type=int, default=240)
    ap.add_argument("--zrev", action="store_true", help="Reverse depth for predictions.")
    ap.add_argument("--zshift", type=int, default=0, help="Shift predictions along depth (+/-).")
    args = ap.parse_args()

    D, H, W = args.depth, args.height, args.width
    if args.patients.strip():
        patients = [p.strip() for p in args.patients.split(",") if p.strip()]
    else:
        patients = get_patients(args.pred_root)

    if len(patients) == 0:
        raise RuntimeError("No patients found to evaluate.")

    print(f"Patients: {len(patients)}")
    all_pred = []
    all_gt   = []

    for pid in patients:
        pdir = os.path.join(args.pred_root, pid)
        print(f"\nLoading {pid}")
        pred_vol = load_pred_volume_from_images(pdir, D, H, W)            # (D,H,W) float32 in 0..255
        gt_vol   = load_gt_mask(pid, args.gt_root)                         # (D,H,W)

        # sanity shapes
        if gt_vol.shape != (D, H, W) and gt_vol.shape == (H, W, D):
            gt_vol = np.transpose(gt_vol, (2,0,1))
        if gt_vol.shape != (D, H, W):
            raise ValueError(f"GT shape {gt_vol.shape} != expected {(D,H,W)} for {pid}")

        # optional depth reverse / shift on predictions (debug toggles)
        if args.zrev:
            pred_vol = pred_vol[::-1, :, :]
        if args.zshift != 0:
            s = args.zshift
            if s > 0:
                pred_vol = np.pad(pred_vol, ((s,0),(0,0),(0,0)), mode="constant")[:D]
            else:
                s = -s
                pred_vol = np.pad(pred_vol, ((0,s),(0,0),(0,0)), mode="constant")[s:]

        # binarize GT (any >0)
        gt_bin = (gt_vol > 0).astype(np.uint8)

        all_pred.append(pred_vol)
        all_gt.append(gt_bin)

        print(f"  pred range: [{float(pred_vol.min()):.1f}, {float(pred_vol.max()):.1f}]  "
              f"GT voxels: {int(gt_bin.sum())}")

    # Sweep thresholds 0..255 (global best across patients)
    best_thr, best_dice = 0, -1.0
    all_means = []
    print("\nSweeping thresholds 0..255 ...")
    for t in range(256):
        dices_t = []
        for pred_vol, gt_bin in zip(all_pred, all_gt):
            pred_bin = (pred_vol >= t).astype(np.uint8)
            d = dice3d(pred_bin, gt_bin)
            dices_t.append(d)
        mean_d = float(np.mean(dices_t))
        all_means.append(mean_d)
        if mean_d > best_dice:
            best_dice, best_thr = mean_d, t
        if t % 32 == 0:
            print(f"  t={t:3d}  mean Dice={mean_d:.4f}")

    print("\n=== GLOBAL ===")
    print(f"Best threshold: {best_thr}   Mean 3D Dice: {best_dice:.4f}")

    # Per-patient Dice at best threshold
    print("\nPer-patient Dice @ best threshold")
    print("--------------------------------")
    per_dice = []
    for pid, pred_vol, gt_bin in zip(patients, all_pred, all_gt):
        pred_bin = (pred_vol >= best_thr).astype(np.uint8)
        d = dice3d(pred_bin, gt_bin)
        per_dice.append(d)
        print(f"{pid}: {d:.4f}")

    per_dice = np.array(per_dice, dtype=np.float32)
    print("\n=== STATS ===")
    print(f"N={len(per_dice)}  mean={per_dice.mean():.4f}  std={per_dice.std():.4f}  "
          f"min={per_dice.min():.4f}  max={per_dice.max():.4f}")

if __name__ == "__main__":
    main()
