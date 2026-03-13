# score.py
import os
import argparse
import yaml
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.ndimage import label
import numpy as np

from utils.volume_utils import (
    reconstruct_volume_from_slices,
    combine_subvolumes_from_folder,
    resize_volume
)
from utils.metrics import (
    dice_coefficient_3d,
    hausdorff_distance_3d,
    roc_auc_3d,
    average_precision_3d,
    f1_max_3d,
    load_threshold_from_json,
    iou3d
)
from utils.io import load_gt_mask   

def setup_seed(seed):
    np.random.seed(seed)

def parse_args_with_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",           type=str,   help="YAML config file")
    parser.add_argument("--results_root",     type=str,   default="./results")
    parser.add_argument("--volume_output_dir",      type=str,   default="results_nibabel_k5_fused/",
                        help="Subdirectory for saving results")
    parser.add_argument("--gt_root",          type=str,   )
    parser.add_argument("--methods",          nargs="+", default=["2D","3D"])
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
    parser.add_argument("--cleanup", default=False, help = "remove .npy after evaluation", action="store_true")

    
    args = parser.parse_args()
    # 1) load YAML if given
    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        parser.set_defaults(**cfg)
    

    # 2) final parse
    args = parser.parse_args()

    # 3) load threshold from JSON if needed
    if args.threshold is None and args.threshold_json:
        args.threshold = load_threshold_from_json(args.threshold_json)
    if args.threshold is None:
        raise ValueError("You must specify --threshold or --threshold_json")

    return args
from scipy import ndimage

def analyze_lesions_by_size(gt_mask, pred_mask, bin_thresholds=[10, 50, 200, 1000, 5000]):

    """
    Analyze lesions by size bins using best-matching predicted components
    """
    # Find connected components
    labeled_gt, num_gt_lesions = ndimage.label(gt_mask > 0)
    labeled_pred, num_pred_lesions = ndimage.label(pred_mask > 0)
    
    # Define bins: [0-10), [10-50), [50-200), [200-1000), [1000-5000), [5000+)
    bin_names = []
    for i, thresh in enumerate(bin_thresholds):
        if i == 0:
            bin_names.append(f"tiny_<{thresh}")
        else:
            bin_names.append(f"size_{bin_thresholds[i-1]}-{thresh}")
    bin_names.append(f"large_>{bin_thresholds[-1]}")
    
    # Initialize results
    results = {}
    for bin_name in bin_names:
        results[bin_name] = {
            'lesion_count': 0,
            'detected_count':  0,
            'total_dice': 0.0,
            'total_iou': 0.0,
            'lesion_sizes': [],
            'individual_dice': [],
            'individual_iou': []
        }
    
    # Analyze each GT lesion
    for lesion_id in range(1, num_gt_lesions + 1):
        lesion_mask = (labeled_gt == lesion_id).astype(np.uint8)
        lesion_size = np.sum(lesion_mask)
        
        # Find best matching predicted component
        best_pred_mask = None
        best_overlap = 0
        
        for pred_id in range(1, num_pred_lesions + 1):
            pred_component = (labeled_pred == pred_id).astype(np.uint8)
            overlap = np.sum(lesion_mask & pred_component)
            
            if overlap > best_overlap:
                best_overlap = overlap
                best_pred_mask = pred_component
        
        # Determine bin
        bin_idx = len(bin_thresholds)
        for i, thresh in enumerate(bin_thresholds):
            if lesion_size < thresh:
                bin_idx = i
                break
        bin_name = bin_names[bin_idx]
        
        # Compute metrics using best matching component (or zeros if no match)
        if best_pred_mask is not None and best_overlap > 0:
            lesion_dice = dice_coefficient_3d(best_pred_mask, lesion_mask)
            lesion_iou = iou3d(best_pred_mask, lesion_mask)
            detected = True
        else:
            lesion_dice = 0.0
            lesion_iou = 0.0
            detected = False
            
        # Store results
        results[bin_name]['lesion_count'] += 1
        results[bin_name]['total_dice'] += lesion_dice
        results[bin_name]['total_iou'] += lesion_iou
        results[bin_name]['lesion_sizes'].append(lesion_size)
        results[bin_name]['individual_dice'].append(lesion_dice)
        results[bin_name]['individual_iou'].append(lesion_iou)

    # Calculate average dice per bin
    for bin_name in results:
        n = results[bin_name]['lesion_count']
        if n > 0:
            results[bin_name]['avg_dice'] = results[bin_name]['total_dice'] / n
            results[bin_name]['std_dice'] = np.std(results[bin_name]['individual_dice'])
            results[bin_name]['avg_iou'] = results[bin_name]['total_iou'] / n
            results[bin_name]['std_iou'] = np.std(results[bin_name]['individual_iou'])
        else:
            results[bin_name]['avg_dice'] = 0.0
            results[bin_name]['std_dice'] = 0.0
            results[bin_name]['avg_iou'] = 0.0
            results[bin_name]['std_iou'] = 0.0
        if results[bin_name]['lesion_count'] > 0:
            results[bin_name]['detection_rate'] = (
                results[bin_name]['detected_count'] 
                 / results[bin_name]['lesion_count']
            )
        else:
            results[bin_name]['detection_rate'] = 0.0
    
    return results

def lesion_iou_and_fp(gt_mask, pred_mask):


    gt_cc, n_gt   = label(gt_mask > 0)
    pred_cc, n_pred = label(pred_mask > 0)

    gt_hit  = np.zeros(n_gt+1,  bool)
    pred_fp = np.ones (n_pred+1,  bool)

    # Single-pass voxel scan
    for g, p in zip(gt_cc.flat, pred_cc.flat):
        if g>0 and p>0:
            gt_hit[g]     = True
            pred_fp[p]    = False

    tp = int(gt_hit.sum()   - 1)  # ignore background
    fn = int(n_gt        - tp)
    fp = int(pred_fp.sum() - 1)

    iou = tp / (tp + fn + fp) if (tp+fn+fp)>0 else 1.0
    return {'TP':tp, 'FN':fn, 'FP':fp, 'lesion_iou':iou}


def evaluate_method(root, gt_root, method, cfg, volume_output_dir):
    df_rows = []
    lesion_bin_rows = []  # New: for lesion size analysis
    
    patients_dir = os.path.join(root, "test")

    for pid in sorted(os.listdir(patients_dir)):
        pd_dir = os.path.join(patients_dir, pid)

        # 1) reconstruction
        if cfg.reconstruct_mode == "slice":
            vol240 = reconstruct_volume_from_slices(
                pd_dir,
                prefix=cfg.prefix,
                full_depth=cfg.full_depth,
                target_h=cfg.target_h,
                target_w=cfg.target_w
            )
        else:  # subvolume fusion
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

        # 3) load GT
        gt240 = load_gt_mask(pid, gt_root)  # return (full_depth, target_h, target_w) binary

        #debug vol240 values
        print(f"Evaluating patient {pid} with method {method}:")
        print(f"  Volume shape: {vol240.shape}, GT shape: {gt240.shape}")
        print(f"min vol240: {np.min(vol240)}, max vol240: {np.max(vol240)}")
        # save gt and volume as nibabel files for debugging

        
        vol240 = np.transpose(vol240, (1, 2, 0))  # (D, H, W) -> (H, W, D)
        print(f"Transformed volume shape: {vol240.shape}")
        #make dir 
        if not os.path.exists(volume_output_dir):
            os.makedirs(volume_output_dir)
            
        # Save vol240 as NIfTI for debugging
        img = nib.Nifti1Image(vol240, np.eye(4))  # Save axis for data (just identity)
        img.header.get_xyzt_units()
        #img.to_filename(os.path.join(volume_output_dir, f"{pid}_map.nii.gz"))
        
        # transform gt to 155x240x155 to 240x240x155
        gt240 = np.transpose(gt240, (1, 2, 0))  # (D, H, W) -> (H, W, D)
        print(f"Transformed GT shape: {gt240.shape}")
        gt_img = nib.Nifti1Image(gt240.astype(np.uint8), np.eye(4))
        gt_img.header.get_xyzt_units()
        #gt_img.to_filename(os.path.join(volume_output_dir, f"{pid}_gt.nii.gz"))
        

        # 4) threshold→binary
        bin_pred = (vol240 > cfg.threshold).astype(np.uint8)
        
        
        
        print(f"Thresholding with {cfg.threshold}, bin_pred shape: {bin_pred.shape}")
        # Save bin_pred as NIfTI for debugging
        img = nib.Nifti1Image(bin_pred, np.eye(4))  # Save axis for data (just identity)
        img.header.get_xyzt_units()
        #img.to_filename(os.path.join(volume_output_dir, f"{pid}_bin_map.nii.gz"))

        # 5) compute metrics
        liou_res = lesion_iou_and_fp(gt240, bin_pred)


        fp_count   = liou_res['FP']
        lesion_iou = liou_res['lesion_iou']
        
        d3  = dice_coefficient_3d(bin_pred, gt240)
        #hd  = hausdorff_distance_3d(bin_pred, gt240)
        hd = 0  # Hausdorff is slow for 3D, so we skip it here
        auc = roc_auc_3d(vol240, gt240)
        ap  = average_precision_3d(vol240, gt240)
        f1m = f1_max_3d(vol240, gt240)

        # Lesion size analysis
        lesion_results = analyze_lesions_by_size(gt240, bin_pred)
        
        
        for bin_name, bin_data in lesion_results.items():
            if bin_data['lesion_count'] == 0:
                continue
            lesion_bin_rows.append({
                "method": method,
                "patient": pid,
                "size_bin": bin_name,
                "lesion_count": bin_data['lesion_count'],
                "detected_count":  bin_data['detected_count'],     
                "detection_rate":  bin_data['detection_rate'], 
                "avg_dice": bin_data['avg_dice'],
                "std_dice": bin_data['std_dice'],
                "avg_iou": bin_data['avg_iou'],
                "std_iou": bin_data['std_iou'],
                "mean_lesion_size": int(np.mean(bin_data['lesion_sizes'])),
                "total_volume": int(sum(bin_data['lesion_sizes'])),
            })

        df_rows.append({
            "method":         method,
            "patient":        pid,
            "reconstruct":    cfg.reconstruct_mode,
            "dice3d":         d3,
            "hausdorff95":    hd,
            "roc_auc":        auc,
            "avg_precision":  ap,
            "f1_max":         f1m,
            "lesion_iou":       lesion_iou,                   # from liou_res['lesion_iou']
            "lesion_FP_count":  int(fp_count) 
        })

    return pd.DataFrame(df_rows), pd.DataFrame(lesion_bin_rows)

def main():
    args = parse_args_with_config()
    all_dfs = []
    all_lesion_dfs = []  # New: for lesion analysis
    
    for m in args.methods:
        df, lesion_df = evaluate_method(
            args.results_root, args.gt_root, m, args, args.volume_output_dir
        )
        all_dfs.append(df)
        all_lesion_dfs.append(lesion_df)
    
    # Original metrics
    full_df = pd.concat(all_dfs, ignore_index=True)
    summary_rows = []
    for method, grp in full_df.groupby("method"):
        stats = {"method": method, "patient": "GLOBAL"}
        
        total_fp = int(grp["lesion_FP_count"].sum())
        stats["lesion_FP_total"] = total_fp
        stats["lesion_FP_mean"]  = f"{grp['lesion_FP_count'].mean():.2f}±{grp['lesion_FP_count'].std():.2f}"
     
        for col in ["dice3d","hausdorff95","roc_auc","avg_precision","f1_max"]:
            stats[col] = f"{grp[col].mean():.4f} ± {grp[col].std():.4f}"
        summary_rows.append(stats)

    summary_df = pd.DataFrame(summary_rows)
    # 1a) Save just the detailed, one-row-per-patient table:
    out_csv = os.path.join(args.results_root, "all_methods_metrics.csv")
    full_df.to_csv(out_csv, index=False)
    print(f"Saved per-patient metrics to {out_csv}")

    # 1b) Save the GLOBAL summary in its own file:
    summary_csv = os.path.join(args.results_root, "global_summary_metrics.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"Saved GLOBAL summary to {summary_csv}")


    
    print(f"Saved per-patient + GLOBAL summary metrics to {out_csv}")

    # NEW: Lesion size analysis
    lesion_full_df = pd.concat(all_lesion_dfs, ignore_index=True)
    
    # Create summary for each method and size bin
    lesion_summary_rows = []
    for (method, size_bin), grp in lesion_full_df.groupby(["method", "size_bin"]):
        # Only include bins that have lesions
        total_lesions = grp['lesion_count'].sum()
        if total_lesions > 0:
            # Weighted average dice (by number of lesions per patient)
            weighted_dice = np.average(grp['avg_dice'], weights=grp['lesion_count'])
            
            lesion_summary_rows.append({
                "method": method,
                "patient": "SUMMARY",
                "size_bin": size_bin,
                "total_lesions": total_lesions,
                "avg_dice": weighted_dice,
                "std_dice": grp['avg_dice'].std(),
                "num_patients": len(grp[grp['lesion_count'] > 0]),
                "mean_lesion_size": grp['mean_lesion_size'].mean(),
                "total_volume": grp['total_volume'].sum()
            })
    
    lesion_summary_df = pd.DataFrame(lesion_summary_rows)
   
    
    lesion_csv = os.path.join(args.results_root, "lesion_size_analysis.csv")
    # 2a) Detailed breakdown per (patient, size_bin)
    lesion_full_df.to_csv(lesion_csv, index=False)
    print(f"Saved lesion-by-size breakdown to {lesion_csv}")

    # 2b) Aggregate summary by size-bin
    lesion_summary_csv = os.path.join(args.results_root, "lesion_size_summary.csv")
    lesion_summary_df.to_csv(lesion_summary_csv, index=False)
    print(f"Saved lesion-size summary to {lesion_summary_csv}")
        
  
    
    if args.cleanup:
    # Remove all .npy files recursively in the results directory
        npy_count = 0
        for root, dirs, files in os.walk(args.results_root):
            for file in files:
                if file.endswith('.npy'):
                    file_path = os.path.join(root, file)
                    os.remove(file_path)
                    npy_count += 1
        print(f"Removed {npy_count} .npy files from {args.results_root} and subdirectories")


if __name__ == "__main__":
    setup_seed(42)
    main()
