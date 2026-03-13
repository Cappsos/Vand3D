import os
import argparse
import json
import numpy as np
import pandas as pd
from scipy.ndimage import label
import tqdm


def categorize_masks(meta_json_path, small_vol_thresh, large_vol_thresh, multi_foci_thresh):
    """
    Read a metadata JSON describing patients and their mask paths, then categorize each patient
    based on their ground-truth segmentation mask into:
      - 'small' lesion (total volume < small_vol_thresh voxels)
      - 'large' lesion (total volume > large_vol_thresh)
      - 'medium' otherwise
      - 'multi_foci' if number of connected components > multi_foci_thresh

    Returns a DataFrame with columns: patient_id, total_voxels, num_components, largest_component, category
    """
    # Derive the root folder containing masks from the JSON location
    data_root = os.path.dirname(os.path.abspath(meta_json_path))

    with open(meta_json_path, 'r') as f:
        meta = json.load(f)

    # Collect all entries across splits and classes
    entries = []
    for split in meta.values():
        if isinstance(split, dict):
            for cls_list in split.values():
                entries.extend(cls_list)
        elif isinstance(split, list):
            entries.extend(split)

    records = []
    for e in tqdm.tqdm(entries, desc="Processing entries"):
        pid = e['patient_id']
        mask_rel = e['mask_path']  # e.g., 'BraTS-MET-00001-000/BraTS-MET-00001-000-seg.npy'
        mask_path = os.path.join(data_root, mask_rel)
        if not os.path.isfile(mask_path):
            print(f"Warning: mask not found for patient {pid} at {mask_path}")
            continue
        mask = np.load(mask_path)
        # Binarize
        mask_bin = (mask > 0).astype(np.uint8)
        total_vox = int(mask_bin.sum())
        # Connected components in 3D
        labeled, num_comp = label(mask_bin)
        # Largest component size
        if num_comp > 0:
            sizes = np.bincount(labeled.flatten())[1:]
            largest = int(sizes.max())
        else:
            largest = 0
        # Determine category
        if num_comp > multi_foci_thresh:
            category = 'multi_foci'
        elif total_vox < small_vol_thresh:
            category = 'small'
        elif total_vox > large_vol_thresh:
            category = 'large'
        else:
            category = 'medium'
        records.append({
            'patient_id': pid,
            'total_voxels': total_vox,
            'num_components': int(num_comp),
            'largest_component': largest,
            'category': category
        })

    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(description="Categorize patients by mask size/foci using meta JSON")
    parser.add_argument('--meta_json',  default='/home/nicoc/data/BraTS_3D/meta.json', type=str,
                        help='Path to metadata JSON (which includes mask_path entries)')
    parser.add_argument('--small',       type=int, default=1000,
                        help='Small lesion threshold (voxels)')
    parser.add_argument('--large',       type=int, default=50000,
                        help='Large lesion threshold (voxels)')
    parser.add_argument('--multi_foci',  type=int, default=1,
                        help='If number of components > this, category is multi_foci')
    parser.add_argument('--output_csv',  type=str, default='patient_categories.csv',
                        help='Output CSV file path')
    args = parser.parse_args()

    df = categorize_masks(
        meta_json_path=args.meta_json,
        small_vol_thresh=args.small,
        large_vol_thresh=args.large,
        multi_foci_thresh=args.multi_foci
    )
    df.to_csv(args.output_csv, index=False)
    print(f"Categorized {len(df)} patients and saved to {args.output_csv}")


if __name__ == '__main__':
    main()
