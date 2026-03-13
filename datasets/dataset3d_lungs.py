import torch
import torch.utils.data as data
import json
import numpy as np
import os
from pathlib import Path
import logging

def generate_class_info(dataset_name):
    class_name_map_class_id = {}
    if dataset_name.lower() == 'lungs':
        obj_list = ['lung']
    elif dataset_name.lower() == 'brats':
        obj_list = ['brain']
    else:
        print(f"Warning: generate_class_info not configured for dataset: {dataset_name}")
        obj_list = ['object'] # Generic
    for k, index in zip(obj_list, range(len(obj_list))):
        class_name_map_class_id[k] = index
    return obj_list, class_name_map_class_id

def load_patient_ids_from_file(patient_seed_path):
    if not patient_seed_path or not os.path.exists(patient_seed_path):
        return None
    patient_ids = []
    with open(patient_seed_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            first = line.split()[0]
            patient_ids.append(first)
    print(f"Loaded {len(patient_ids)} patient IDs from {patient_seed_path}")
    return set(patient_ids)

class Lung3DSubVolumeDataset(data.Dataset):
    def __init__(
        self,
        root,
        transform_img,
        transform_mask,
        dataset_name,
        mode="test",
        sub_volume_depth=32,
        stride_depth=16,
        patient_seed_path=None,
        scales=[1]
    ):
        self.root = Path(root)
        self.transform_img = transform_img
        self.transform_mask = transform_mask
        self.sub_volume_depth = sub_volume_depth
        self.stride_depth = stride_depth
        self.scales = scales
        self.items = []
        print(self.root)
        meta_file_path = self.root / 'meta.json'
        if not meta_file_path.exists():
            raise FileNotFoundError(f"meta.json not found in {self.root}")

        with open(meta_file_path, 'r') as f:
            meta_info_full = json.load(f)

        if mode not in meta_info_full:
            raise KeyError(f"Mode '{mode}' not found in meta.json. Available modes: {list(meta_info_full.keys())}")

        class_name = 'lung' if dataset_name.lower() == 'lungs' else 'brain'
        patient_volume_infos = meta_info_full[mode].get(class_name, [])
        print(f"Found {len(patient_volume_infos)} patient volume infos.")
        
        allowed_patient_ids = load_patient_ids_from_file(patient_seed_path)
        for patient_info in patient_volume_infos:
            if allowed_patient_ids is not None:
                patient_id = patient_info.get('patient_id', 'unknown')
                if patient_id not in allowed_patient_ids:
                    continue

            volume_path = self.root / patient_info['volume_path']
            try:
                # Attempt to memory-map the file to read metadata without loading the whole file
                vol = np.load(volume_path)
                if vol.ndim == 3:
                    original_depth = vol.shape[2]
                else:
                    logging.error(f"Unexpected volume dimension {vol.ndim} for {volume_path}. Skipping.")
                    continue
            except Exception as e:
                logging.error(f"Could not read shape from {volume_path}: {e}. Skipping patient.")
                continue

            for scale_id, s in enumerate(self.scales):
                block_starts = []
                for d_start in range(0, original_depth, self.stride_depth):
                    if d_start + self.sub_volume_depth <= original_depth:
                        block_starts.append(d_start)
                
                if not block_starts or (block_starts[-1] + self.sub_volume_depth) < original_depth:
                    final_start = original_depth - self.sub_volume_depth
                    if final_start >= 0 and (not block_starts or final_start > block_starts[-1]):
                         block_starts.append(final_start)

                for d_start in sorted(list(set(block_starts))):
                    self.items.append({
                        'patient_info': patient_info,
                        'start_depth_idx': d_start,
                        'scale_id': scale_id,
                        'original_depth': original_depth
                    })

        self.obj_list, self.class_name_map_class_id = generate_class_info(dataset_name)
        print(f"Found {len(self.items)} items in the dataset.")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item_info = self.items[index]
        patient_data = item_info['patient_info']
        start_d = item_info['start_depth_idx']
        scale_id = item_info['scale_id']
        original_depth = item_info['original_depth']

        full_volume_path = self.root / patient_data['volume_path']
        full_mask_path_rel = patient_data.get('mask_path')

        try:
            full_volume_np_hwd = np.load(full_volume_path)
            sub_volume_np_hwd = full_volume_np_hwd[:, :, start_d : start_d + self.sub_volume_depth]
            sub_volume_np_dhw = np.transpose(sub_volume_np_hwd, (2, 0, 1))
        except Exception as e:
            print(f"Error loading or slicing volume {full_volume_path} at depth {start_d}: {e}")
            dummy_img = torch.zeros((1, self.sub_volume_depth, 512, 512), dtype=torch.float32)
            dummy_mask = torch.zeros((1, self.sub_volume_depth, 512, 512), dtype=torch.uint8)
            return {'img': dummy_img, 'img_mask': dummy_mask, 'cls_name': 'lung',
                    'anomaly': 0, 'img_path': str(full_volume_path),
                    "cls_id": self.class_name_map_class_id.get('lung',0),
                    'patient_id': patient_data.get('patient_id', 'unknown'),
                    'start_depth_idx': start_d,
                    'original_depth': original_depth,
                    'scale_id': scale_id,
                    'crop_box': (0,0,512,512)}

        anomaly_label = patient_data.get('anomaly', 0)
        if anomaly_label == 0 or full_mask_path_rel is None:
            sub_mask_np_dhw = np.zeros_like(sub_volume_np_dhw, dtype=np.uint8)
        else:
            full_mask_path = self.root / full_mask_path_rel
            if not full_mask_path.exists():
                sub_mask_np_dhw = np.zeros_like(sub_volume_np_dhw, dtype=np.uint8)
            else:
                try:
                    full_mask_np_hwd = np.load(full_mask_path)
                    sub_mask_np_hwd = full_mask_np_hwd[:, :, start_d : start_d + self.sub_volume_depth]
                    sub_mask_np_dhw = np.transpose(sub_mask_np_hwd, (2, 0, 1))
                except Exception as e:
                    print(f"Error loading or slicing mask {full_mask_path} at depth {start_d}: {e}. Creating zero mask.")
                    sub_mask_np_dhw = np.zeros_like(sub_volume_np_dhw, dtype=np.uint8)

        anomaly_label = 1 if np.any(sub_mask_np_dhw > 0) else 0

        s = self.scales[scale_id]
        D, H, W = sub_volume_np_dhw.shape
        h_crop, w_crop = int(H * s), int(W * s)
        
        y0 = (H - h_crop) // 2
        x0 = (W - w_crop) // 2
        sub_volume_np_dhw = sub_volume_np_dhw[:, y0 : y0 + h_crop, x0 : x0 + w_crop]
        sub_mask_np_dhw   = sub_mask_np_dhw[:,   y0 : y0 + h_crop, x0 : x0 + w_crop]

        crop_box = (y0, x0, h_crop, w_crop)
        
        img_tensor = self.transform_img(sub_volume_np_dhw)
        mask_tensor = self.transform_mask(sub_mask_np_dhw)
        
        cls_name_actual = patient_data.get('cls_name', 'lung')

        return {
            'img': img_tensor,
            'img_mask': mask_tensor,
            'cls_name': cls_name_actual,
            'anomaly': anomaly_label,
            'img_path': str(full_volume_path),
            "cls_id": self.class_name_map_class_id.get(cls_name_actual, 0),
            'patient_id': patient_data.get('patient_id', 'unknown'),
            'start_depth_idx': start_d,
            'original_depth': original_depth,
            'scale_id': scale_id,
            'crop_box': crop_box
        }