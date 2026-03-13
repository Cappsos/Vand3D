# score.py
import os
import argparse
import yaml
import numpy as np
import pandas as pd
import nibabel as nib
from scipy import ndimage
from scipy.ndimage import label

from utils.volume_utils import (
    reconstruct_volume_from_slices,
    combine_subvolumes_from_folder,
    resize_volume
)
from utils.metrics import (
    dice_coefficient_3d,
    hausdorff_distance_3d,
    load_threshold_from_json,
    iou3d
)
from utils.io import load_gt_mask


# ---------------------------
# Utilities
# ---------------------------
def setup_seed(seed):
    np.random.seed(seed)


def parse_args_with_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",           type=str,   help="YAML config file")
    parser.add_argument("--results_root",     type=str,   default="./results")
    parser.add_argument("--volume_output_dir", type=str,  default="results_nibabel_k5_fused/",
                        help="Subdirectory for saving results")
    parser.add_argument("--gt_root",          type=str,   default="./data/gt_masks")
    parser.add_argument("--methods",          nargs="+",  default=["2D","3D"])
    parser.add_argument("--prefix",           type=str,   default="anomaly_map_depth_")
    parser.add_argument("--full_depth",       type=int,   default=155)
    parser.add_argument("--target_h",         type=int,   default=240)
    parser.add_argument("--target_w",         type=int,   default=240)
    parser.add_argument("--depth",            type=int,   default=32,
                        help="Block depth used at inference")
    parser.add_argument("--reconstruct_mode", choices=["slice","subvolume"],
                        default="subvolume",
                        help="How to rebuild full volume: 'slice' for 1×H×W masks, 'subvolume' for overlapping blocks")
    parser.add_argument("--threshold",        type=float, default=None,
                        help="Override binarization threshold")
    parser.add_argument("--threshold_json",   type=str,   default=None,
                        help="Path to JSON file containing best_threshold")

    # cleanup
    parser.add_argument("--cleanup", action="store_true", default=False,
                        help="Remove .npy after evaluation")

    args = parser.parse_args()

    # 1) load YAML if given
    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        parser.set_defaults(**cfg)
        args = parser.parse_args()

    # 2) load threshold from JSON if needed
    if args.threshold is None and args.threshold_json:
        args.threshold = load_threshold_from_json(args.threshold_json)
    if args.threshold is None:
        raise ValueError("You must specify --threshold or --threshold_json")

    return args


# ---------------------------
# Metrics helpers
# ---------------------------
def confusion_metrics_3d(pred_bin, gt_bin, eps=1e-8):
    """Voxel-wise confusion metrics per patient."""
    p = pred_bin.astype(bool)
    g = gt_bin.astype(bool)
    TP = np.logical_and(p, g).sum()
    FP = np.logical_and(p, ~g).sum()
    FN = np.logical_and(~p, g).sum()
    TN = np.logical_and(~p, ~g).sum()

    sens = TP / (TP + FN + eps)         # recall
    ppv  = TP / (TP + FP + eps)         # precision
    spec = TN / (TN + FP + eps)         # specificity
    dice = (2*TP) / (2*TP + FP + FN + eps)
    iou  = dice / (2 - dice + eps)      # exact transform

    return dict(TP=int(TP), FP=int(FP), FN=int(FN), TN=int(TN),
                sensitivity=float(sens), ppv=float(ppv), specificity=float(spec),
                dice3d=float(dice), iou3d=float(iou))


def lesion_recall_at_iou(gt_mask, pred_mask, thr=0.5):
    """
    Lesion-level recall: fraction of GT components with IoU >= thr
    against at least one predicted component. 26-connectivity.
    Returns NaN if there are no GT lesions (skip in macro-avg).
    """
    gt_lab, n_gt = label(gt_mask.astype(bool), structure=np.ones((3,3,3)))
    pred_lab, _  = label(pred_mask.astype(bool), structure=np.ones((3,3,3)))
    if n_gt == 0:
        return np.nan

    hits = 0
    for lid in range(1, n_gt+1):
        g = (gt_lab == lid)
        # restrict to GT bbox for speed
        gz, gy, gx = np.where(g)
        z0,z1 = gz.min(), gz.max()+1
        y0,y1 = gy.min(), gy.max()+1
        x0,x1 = gx.min(), gx.max()+1
        sub_pred = pred_lab[z0:z1, y0:y1, x0:x1]
        sub_gt   = g[z0:z1, y0:y1, x0:x1]

        best_iou = 0.0
        for pid in np.unique(sub_pred):
            if pid == 0: continue
            p = (sub_pred == pid)
            inter = np.logical_and(p, sub_gt).sum()
            uni   = p.sum() + sub_gt.sum() - inter
            if uni > 0:
                best_iou = max(best_iou, inter/uni)
        hits += (best_iou >= thr)
    return hits / n_gt


def median_slice_dice(pred_bin, gt_bin):
    """Median per-slice Dice (iterate last axis as depth)."""
    dices = []
    for k in range(pred_bin.shape[-1]):
        p = pred_bin[..., k].astype(bool)
        g = gt_bin[..., k].astype(bool)
        inter = np.logical_and(p, g).sum()
        denom = p.sum() + g.sum()
        dices.append(0.0 if denom == 0 else (2.0*inter)/denom)
    return float(np.median(dices))


def analyze_lesions_by_size(gt_mask, pred_mask, bin_thresholds=[10, 50, 200, 1000, 5000], iou_thresholds=(0.2, 0.5)):
    """
    Per-lesion best-match analysis with size bins (voxels).
    Adds detection rates per bin at IoU >= 0.2 and >= 0.5.
    Bins: [0-10), [10-50), [50-200), [200-1000), [1000-5000), [5000+)
    """
    labeled_gt, num_gt = ndimage.label(gt_mask > 0)
    labeled_pred, num_pred = ndimage.label(pred_mask > 0)

    # build bin names
    bin_names = []
    for i, thresh in enumerate(bin_thresholds):
        bin_names.append(f"tiny_<{thresh}" if i == 0 else f"size_{bin_thresholds[i-1]}-{thresh}")
    bin_names.append(f"large_>{bin_thresholds[-1]}")

    # init per-bin accumulators
    results = {}
    for b in bin_names:
        results[b] = {
            'lesion_count': 0,
            'detected_count': 0,              # any overlap (>0 voxels)
            'detected_iou_0p2': 0,            # IoU >= 0.2
            'detected_iou_0p5': 0,            # IoU >= 0.5
            'total_dice': 0.0,
            'total_iou': 0.0,
            'lesion_sizes': [],
            'individual_dice': [],
            'individual_iou': []
        }

    # per-lesion best match
    for lid in range(1, num_gt+1):
        lesion = (labeled_gt == lid).astype(np.uint8)
        size = int(lesion.sum())

        # choose bin
        idx = next((i for i,t in enumerate(bin_thresholds) if size < t), len(bin_thresholds))
        bname = bin_names[idx]

        # best overlap component in prediction
        best_ov = 0
        best_pred = None
        for pid in range(1, num_pred+1):
            comp = (labeled_pred == pid).astype(np.uint8)
            ov = int(np.logical_and(lesion, comp).sum())
            if ov > best_ov:
                best_ov = ov
                best_pred = comp

        if best_pred is not None and best_ov > 0:
            ldice = dice_coefficient_3d(best_pred, lesion)
            liou  = iou3d(best_pred, lesion)     # voxel IoU between this GT lesion and its best-matching pred
            detected_any  = True
            detected_02   = (liou >= 0.02)
            detected_05   = (liou >= 0.5)
        else:
            ldice, liou = 0.0, 0.0
            detected_any = detected_02 = detected_05 = False

        r = results[bname]
        r['lesion_count']     += 1
        r['detected_count']   += int(detected_any)
        r['detected_iou_0p2'] += int(detected_02)
        r['detected_iou_0p5'] += int(detected_05)
        r['total_dice']       += ldice
        r['total_iou']        += liou
        r['lesion_sizes'].append(size)
        r['individual_dice'].append(ldice)
        r['individual_iou'].append(liou)

    # finalize per bin
    for bname, r in results.items():
        n = r['lesion_count']
        if n > 0:
            r['avg_dice'] = r['total_dice']/n
            r['std_dice'] = float(np.std(r['individual_dice']))
            r['avg_iou']  = r['total_iou']/n
            r['std_iou']  = float(np.std(r['individual_iou']))

            # detection rates per IoU threshold
            r['detection_rate_any']    = r['detected_count']/n
            r['detection_rate_iou_0p2'] = r['detected_iou_0p2']/n
            r['detection_rate_iou_0p5'] = r['detected_iou_0p5']/n
        else:
            r.update({
                'avg_dice': 0.0, 'std_dice': 0.0, 'avg_iou': 0.0, 'std_iou': 0.0,
                'detection_rate_any': 0.0, 'detection_rate_iou_0p2': 0.0, 'detection_rate_iou_0p5': 0.0
            })
    return results


def lesion_iou_and_fp(gt_mask, pred_mask):
    """
    Component-level proxy (kept for continuity, not used in summaries).
    """
    gt_cc, n_gt   = label(gt_mask > 0)
    pred_cc, n_pred = label(pred_mask > 0)

    gt_hit  = np.zeros(n_gt+1,  bool)
    pred_fp = np.ones (n_pred+1, bool)

    # Single-pass voxel scan
    for g, p in zip(gt_cc.flat, pred_cc.flat):
        if g > 0 and p > 0:
            gt_hit[g]  = True
            pred_fp[p] = False

    tp = int(gt_hit.sum()   - 1)  # ignore background
    fn = int(n_gt           - tp)
    fp = int(pred_fp.sum()  - 1)

    iou = tp / (tp + fn + fp) if (tp+fn+fp)>0 else 1.0
    return {'TP':tp, 'FN':fn, 'FP':fp, 'lesion_iou':iou}


# ---------------------------
# Evaluation loop
# ---------------------------
def evaluate_method(root, gt_root, method, cfg, volume_output_dir):
    df_rows = []
    lesion_bin_rows = []

    # Allow either results_root/test/* or results_root/method/test/*
    maybe = os.path.join(root, method, "test")
    patients_dir = maybe if os.path.isdir(maybe) else os.path.join(root, "test")

    for pid in sorted(os.listdir(patients_dir)):
        pd_dir = os.path.join(patients_dir, pid)
        if not os.path.isdir(pd_dir):
            continue

        # 1) reconstruction → (D,H,W)
        if cfg.reconstruct_mode == "slice":
            vol240 = reconstruct_volume_from_slices(
                pd_dir,
                prefix=cfg.prefix,
                full_depth=cfg.full_depth,
                target_h=cfg.target_h,
                target_w=cfg.target_w
            )
        else:
            vol240 = combine_subvolumes_from_folder(
                pd_dir,
                prefix=cfg.prefix,
                full_depth=cfg.full_depth,
                block_depth=cfg.depth
            )

        # 2) resize to (D, target_h, target_w)
        if vol240.shape[1] != cfg.target_h or vol240.shape[2] != cfg.target_w:
            print(f"Resizing volume {pid} from {vol240.shape} to ({cfg.full_depth}, {cfg.target_h}, {cfg.target_w})")
            vol240 = resize_volume(vol240, cfg.target_h, cfg.target_w)

        # 3) load GT (D,H,W)
        gt240 = load_gt_mask(pid, gt_root)
       

        print(f"Evaluating patient {pid} with method {method}:")
        print(f"  Volume shape: {vol240.shape}, GT shape: {gt240.shape}")
        print(f"  min vol240: {np.min(vol240):.6f}, max vol240: {np.max(vol240):.6f}")

        # 4) to (H,W,D)
        
        vol240 = np.transpose(vol240, (1, 2, 0))
        gt240  = np.transpose(gt240,  (1, 2, 0))
        print(f"  Transformed to (H,W,D): vol {vol240.shape}, gt {gt240.shape}")

        # 5) threshold→binary
        bin_pred = (vol240 > cfg.threshold).astype(np.uint8)
        print(f"  Threshold {cfg.threshold:.4f}: bin_pred shape {bin_pred.shape}")

        # 6) metrics
        # voxel-wise confusion + Dice/IoU
        cm = confusion_metrics_3d(bin_pred, gt240)

        # lesion-level recall at IoU thresholds
        r02 = lesion_recall_at_iou(gt240, bin_pred, thr=0.2)
        r05 = lesion_recall_at_iou(gt240, bin_pred, thr=0.5)

        # 3D consistency
        med2D = median_slice_dice(bin_pred, gt240)
        cons_gap = med2D - cm['dice3d']

        # optional legacy prob-based curves (comment out if not needed)
        vol_norm = np.clip(vol240, 0.0, 1.0)

        # lesion size analysis (per patient)
        lesion_results = analyze_lesions_by_size(gt240, bin_pred, bin_thresholds=[10, 50, 200, 1000, 5000])
        for bin_name, bin_data in lesion_results.items():
            if bin_data['lesion_count'] == 0:
                continue
            lesion_bin_rows.append({
            "method": method,
            "patient": pid,
            "size_bin": bin_name,
            "lesion_count":    bin_data['lesion_count'],
            "detected_count":  bin_data['detected_count'],        # any-overlap count
            "detected_iou_0p2": bin_data['detected_iou_0p2'],     # IoU>=0.2 count
            "detected_iou_0p5": bin_data['detected_iou_0p5'],     # IoU>=0.5 count
            "detection_rate_any":     bin_data['detection_rate_any'],
            "detection_rate_iou_0p2": bin_data['detection_rate_iou_0p2'],
            "detection_rate_iou_0p5": bin_data['detection_rate_iou_0p5'],
            "avg_dice":        bin_data['avg_dice'],
            "std_dice":        bin_data['std_dice'],
            "avg_iou":         bin_data['avg_iou'],
            "std_iou":         bin_data['std_iou'],
            "mean_lesion_size": int(np.mean(bin_data['lesion_sizes'])) if bin_data['lesion_sizes'] else 0,
            "total_volume":     int(sum(bin_data['lesion_sizes'])) if bin_data['lesion_sizes'] else 0,
        })

        # add per-patient row
        df_rows.append({
            "method":         method,
            "patient":        pid,
            "reconstruct":    cfg.reconstruct_mode,

            # overlap
            "dice3d":         cm['dice3d'],
            "iou3d":          cm['iou3d'],

            # voxel-wise confusion-derived
            "sensitivity":    cm['sensitivity'],
            "ppv":            cm['ppv'],
            "specificity":    cm['specificity'],

            # lesion-level detection
            "recall_iou_0p2": r02,
            "recall_iou_0p5": r05,

            # volumetric coherence
            "median_slice_dice": med2D,
            "consistency_gap":   cons_gap,
        })

    return pd.DataFrame(df_rows), pd.DataFrame(lesion_bin_rows)


# ---------------------------
# Main
# ---------------------------
def main():
    args = parse_args_with_config()
    setup_seed(42)

    all_dfs = []
    all_lesion_dfs = []

    for m in args.methods:
        df, lesion_df = evaluate_method(
            args.results_root, args.gt_root, m, args, args.volume_output_dir
        )
        all_dfs.append(df)
        all_lesion_dfs.append(lesion_df)

    # Per-patient table
    full_df = pd.concat(all_dfs, ignore_index=True)

    # GLOBAL summary (macro-avg over patients; NaN-safe)
    summary_rows = []
    for method, grp in full_df.groupby("method"):
        stats = {"method": method, "patient": "GLOBAL"}

        # lesion FP (from old proxy, if you still compute it elsewhere; else skip)
        if "lesion_FP_count" in grp.columns:
            total_fp = int(grp["lesion_FP_count"].sum())
            stats["lesion_FP_total"] = total_fp
            stats["lesion_FP_mean"]  = f"{grp['lesion_FP_count'].mean():.2f}±{grp['lesion_FP_count'].std():.2f}"

        # main metrics
        for col in [
            "dice3d","iou3d","sensitivity","ppv","specificity",
            "recall_iou_0p2","recall_iou_0p5",
            "median_slice_dice","consistency_gap",
        ]:
            if col in grp.columns:
                mean = np.nanmean(grp[col].values.astype(float))
                std  = np.nanstd (grp[col].values.astype(float))
                stats[col] = f"{mean:.4f} ± {std:.4f}"
        summary_rows.append(stats)

    summary_df = pd.DataFrame(summary_rows)

    # Save per-patient metrics
    out_csv = os.path.join(args.results_root, "all_methods_metrics.csv")
    full_df.to_csv(out_csv, index=False)
    print(f"Saved per-patient metrics to {out_csv}")

    # Save GLOBAL summary
    summary_csv = os.path.join(args.results_root, "global_summary_metrics.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"Saved GLOBAL summary to {summary_csv}")

    # Lesion size analysis (detailed + summary)
    lesion_full_df = pd.concat(all_lesion_dfs, ignore_index=True) if len(all_lesion_dfs) > 0 else pd.DataFrame()

    if len(lesion_full_df) > 0:
        lesion_csv = os.path.join(args.results_root, "lesion_size_analysis.csv")
        lesion_full_df.to_csv(lesion_csv, index=False)
        print(f"Saved lesion-by-size breakdown to {lesion_csv}")

        # Aggregate per (method, size_bin)
        lesion_summary_rows = []
        for (method, size_bin), grp in lesion_full_df.groupby(["method", "size_bin"]):
            total_lesions = int(grp['lesion_count'].sum())
            if total_lesions > 0:
                weighted_dice = np.average(grp['avg_dice'], weights=grp['lesion_count'])
                weighted_iou  = np.average(grp['avg_iou'],  weights=grp['lesion_count'])

                # aggregate IoU-thresholded detection
                det_any  = int(grp['detected_count'].sum())
                det_02   = int(grp['detected_iou_0p2'].sum()) if 'detected_iou_0p2' in grp else 0
                det_05   = int(grp['detected_iou_0p5'].sum()) if 'detected_iou_0p5' in grp else 0

                lesion_summary_rows.append({
                    "method": method,
                    "patient": "SUMMARY",
                    "size_bin": size_bin,
                    "total_lesions": total_lesions,
                    "avg_dice": weighted_dice,
                    "avg_iou":  weighted_iou,

                    # detection rates by size (this is what you asked for)
                    "detection_rate_any":     det_any/total_lesions,
                    "detection_rate_iou_0p2": det_02/total_lesions,
                    "detection_rate_iou_0p5": det_05/total_lesions,

                    "std_dice": grp['avg_dice'].std(),
                    "std_iou":  grp['avg_iou'].std(),
                    "num_patients": int((grp['lesion_count'] > 0).sum()),
                    "mean_lesion_size": float(grp['mean_lesion_size'].mean()),
                    "total_volume": int(grp['total_volume'].sum())
                })


        lesion_summary_df = pd.DataFrame(lesion_summary_rows)
        lesion_summary_csv = os.path.join(args.results_root, "lesion_size_summary.csv")
        lesion_summary_df.to_csv(lesion_summary_csv, index=False)
        print(f"Saved lesion-size summary to {lesion_summary_csv}")

    # Optional cleanup of .npy
    if args.cleanup:
        npy_count = 0
        for r, _, files in os.walk(args.results_root):
            for file in files:
                if file.endswith('.npy'):
                    os.remove(os.path.join(r, file))
                    npy_count += 1
        print(f"Removed {npy_count} .npy files from {args.results_root} and subdirectories")


if __name__ == "__main__":
    main()
