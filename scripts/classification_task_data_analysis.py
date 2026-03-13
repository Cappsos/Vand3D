import torch
from torch.utils.data import DataLoader
import numpy as np
import argparse
import yaml
from pathlib import Path
import sys
sys.path.append("..")


from utils.transforms import Transform3DForM3DCLIP, Transform3DMask
from datasets.dataset3d import BraTS3DSubVolumeDataset

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)

def main(args):
    # load any config overrides
    if args.config:
        cfg = load_config(args.config)
        for k, v in cfg.items():
            setattr(args, k, v)

    # set up transforms (match those you actually use!)
    img_transform  = Transform3DForM3DCLIP(target_height=256,
                                           target_width=256,
                                           target_depth=args.sub_volume_depth)
    mask_transform = Transform3DMask(target_height=240,
                                     target_width=240,
                                     target_depth=args.sub_volume_depth)

    # instantiate dataset in ‘test’ or ‘train’ mode — masks exist for all volumes
    dataset = BraTS3DSubVolumeDataset(
        root=args.data_root,
        transform_img=img_transform,
        transform_mask=mask_transform,
        dataset_name='brats',
        mode='test',                  # or 'train' / 'val' as you like
        sub_volume_depth=args.sub_volume_depth,
        stride_depth=args.stride_depth,
        patient_seed_path=None
    )


    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    total = 0
    positives = 0
    negatives = 0

    print(f"Found {len(dataset)} sub-volumes in {args.data_root}")

    for item in loader:
        total += 1
        # your dataset already computes anomaly_label based on mask
        label = item['anomaly'].item()
        if label == 1:
            positives += 1
        else:
            negatives += 1

    print("==== Sub-volume Summary ====")
    print(f"Total sub-volumes: {total}")
    print(f"Positive (contain lesion): {positives} ({positives/total*100:.2f}%)")
    print(f"Negative (no lesion):      {negatives} ({negatives/total*100:.2f}%)")

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data_root',    type=str, default='/home/nicoc/data/BraTS_3D', help='root folder with meta.json & .npy volumes')
    p.add_argument('--config',       type=str, default=None,       help='optional YAML to override args')
    p.add_argument('--sub_volume_depth', type=int, default=32,    help='depth of each sub-volume')
    p.add_argument('--stride_depth',     type=int, default=16,    help='stride between sub-volumes')
    args = p.parse_args()
    main(args)
