import torch
import torch.utils.data as data
import json
import numpy as np
import os
from pathlib import Path
import logging



def generate_class_info(dataset_name):
    class_name_map_class_id = {}
    if dataset_name == 'brats': # Or your specific dataset name
        obj_list = ['brain']
    else:
        # Fallback or error for other datasets not configured for 3D
        print(f"Warning: generate_class_info not fully configured for 3D dataset: {dataset_name}")
        obj_list = ['object'] # Generic
    for k, index in zip(obj_list, range(len(obj_list))):
        class_name_map_class_id[k] = index
    return obj_list, class_name_map_class_id

def load_patient_ids_from_file(patient_seed_path):
    """Load patient IDs from a text file, one ID per line (ignore anything after the first token)."""
    if not patient_seed_path or not os.path.exists(patient_seed_path):
        return None

    patient_ids = []
    with open(patient_seed_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            # take only the first token (splitting on whitespace)
            first = line.split()[0]
            patient_ids.append(first)

    print(f"Loaded {len(patient_ids)} patient IDs from {patient_seed_path}")
    return set(patient_ids)

class BraTS3DSubVolumeDataset(data.Dataset):
    def __init__(
        self,
        root,
        transform_img,
        transform_mask,
        dataset_name,
        mode="test",
        sub_volume_depth=32,
        stride_depth=16,
        min_foreground_percent=0.001,
        patient_seed_path=None,
        scales=[1]
    ):
        self.root = Path(root)
        self.transform_img = transform_img       # Should be an instance of Transform3DForM3DCLIP
        self.transform_mask = transform_mask   # Should be an instance of Transform3DMask
        self.sub_volume_depth = sub_volume_depth
        self.stride_depth = stride_depth
        self.scales = scales
        self.min_foreground_percent = min_foreground_percent # To optionally filter out mostly empty sub-volumes in training

        self.items = [] # This will store tuples of (full_volume_info, start_depth_idx)

        meta_file_path = self.root / 'meta.json'
        if not meta_file_path.exists():
            raise FileNotFoundError(f"meta.json not found in {self.root}")

        with open(meta_file_path, 'r') as f:
            meta_info_full = json.load(f)

        if mode not in meta_info_full:
            raise KeyError(f"Mode '{mode}' not found in meta.json. Available modes: {list(meta_info_full.keys())}")
        
        # Assuming for BraTS, the class is always 'brain'
        patient_volume_infos = meta_info_full[mode].get('brain', [])

        allowed_patient_ids = load_patient_ids_from_file(patient_seed_path)
 
        for patient_info in patient_volume_infos:
            # Filter patients if patient_seed_path is provided
            if allowed_patient_ids is not None:
                patient_id = patient_info.get('patient_id', 'unknown')
                if patient_id not in allowed_patient_ids:
                    continue  # Skip this patient
            
            original_depth = 155
            # Calculate start indices for sub-volumes with complete coverage
            for scale_id, s in enumerate(self.scales):
                # Generate block starts that cover the entire volume
                block_starts = []
                
                # Regular stride-based blocks
                for d_start in range(0, original_depth, self.stride_depth):
                    if d_start + self.sub_volume_depth <= original_depth:
                        block_starts.append(d_start)
                    else:
                        break
                
                # Add final block if needed to cover remaining slices
                if block_starts:
                    last_covered_slice = block_starts[-1] + self.sub_volume_depth - 1
                    if last_covered_slice < original_depth - 1:
                        # Add final block that ends at volume boundary
                        final_start = original_depth - self.sub_volume_depth  # 155 - 32 = 123
                        if final_start not in block_starts:
                            block_starts.append(final_start)
                
                # Create items for all block starts
                for d_start in sorted(block_starts):
                    self.items.append(
                        {
                            'patient_info': patient_info,
                            'start_depth_idx': d_start,
                            'scale_id': scale_id
                        }
                    )
        
        self.obj_list, self.class_name_map_class_id = generate_class_info(dataset_name)
        if not self.obj_list: # safety
             self.obj_list = ['brain'] # Default for brats
             self.class_name_map_class_id = {'brain':0}


    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item_info = self.items[index]
        patient_data = item_info['patient_info']
        start_d = item_info['start_depth_idx']
        scale_id = item_info['scale_id']
        
        
        full_volume_path = self.root / patient_data['volume_path']
        full_mask_path_rel = patient_data.get('mask_path')

        try:
            # Load the full 3D volume (HxWxD or DxHxW)
            # IMPORTANT: Ensure your .npy files are saved in a consistent dimension order.
            # Let's assume they are saved as HxWxD (240, 240, 155)
            full_volume_np_hwd = np.load(full_volume_path)
            
            # Dynamic padding to make (depth - sub_volume_depth) divisible by stride 
            
            """ 
            # Remove since may not be necessary
            orig_d = full_volume_np_hwd.shape[2]    
            pad_back = (self.stride_depth - 
                        ((orig_d - self.sub_volume_depth) % self.stride_depth)) % self.stride_depth
            if pad_back > 0:
                # reflect-pad the volume in depth
                full_volume_np_hwd = np.pad(
                    full_volume_np_hwd,
                    pad_width=((0,0),(0,0),(0,pad_back)),
                    mode='constant', constant_values=0
                )
                
            """
            orig_d = full_volume_np_hwd.shape[2]
            
            h, w, d = full_volume_np_hwd.shape
            if h != 240 or w != 240 or d != 155:  # Remove the divisibility check
                logging.warning(f"Unexpected volume shape {full_volume_np_hwd.shape}")
                
                if (full_volume_np_hwd.shape[0] == 155 or full_volume_np_hwd.shape[0] == 160) and full_volume_np_hwd.shape[1] == 240 and full_volume_np_hwd.shape[2] == 240:
                    full_volume_np_hwd = np.transpose(full_volume_np_hwd, (1,2,0)) # DxHxW to HxWxD

            # Extract the 32-slice sub-volume along depth (assuming depth is the last dimension here)
            # If your data is DxHxW, you'd slice along the first dimension: full_volume_np[start_d : start_d + self.sub_volume_depth, :, :]
            sub_volume_np_hwd = full_volume_np_hwd[:, :, start_d : start_d + self.sub_volume_depth]
            # Transpose to DxHxW for the transform, as M3D-CLIP wants DxHxW after channel
            sub_volume_np_dhw = np.transpose(sub_volume_np_hwd, (2, 0, 1)) # Now (32, H_orig, W_orig)

        except Exception as e:
            print(f"Error loading or slicing volume {full_volume_path} at depth {start_d}: {e}")
            # Return dummy tensors
            dummy_img = torch.zeros((1, self.sub_volume_depth, 256, 256), dtype=torch.float32)
            dummy_mask = torch.zeros((1, self.sub_volume_depth, 256, 256), dtype=torch.uint8)
            return {'img': dummy_img, 'img_mask': dummy_mask, 'cls_name': patient_data.get('cls_name','brain'),
                    'anomaly': patient_data.get('anomaly',0), 'img_path': str(full_volume_path),
                    "cls_id": self.class_name_map_class_id.get(patient_data.get('cls_name','brain'),0),
                    'patient_id': patient_data.get('patient_id', 'unknown'),
                    'start_depth_idx': start_d}


        anomaly_label = patient_data.get('anomaly', 0)
        if anomaly_label == 0 or full_mask_path_rel is None:
            sub_mask_np_dhw = np.zeros_like(sub_volume_np_dhw, dtype=np.uint8)
        else:
            full_mask_path = self.root / full_mask_path_rel
            if not full_mask_path.exists():
                print(f"Warning: Mask path {full_mask_path} does not exist. Creating zero mask for sub-volume.")
                sub_mask_np_dhw = np.zeros_like(sub_volume_np_dhw, dtype=np.uint8)
            else:
                try:
                    full_mask_np_hwd = np.load(full_mask_path) # Assuming HxWxD
                    
                    # Check and fix mask shape (same logic as volume)
                    if full_mask_np_hwd.shape != (240, 240, 155):
                        print(f"Warning: Mask {full_mask_path} has unexpected shape {full_mask_np_hwd.shape}. Adapting...")
                        if full_mask_np_hwd.shape[0] == 155 and full_mask_np_hwd.shape[1] == 240 and full_mask_np_hwd.shape[2] == 240:
                            full_mask_np_hwd = np.transpose(full_mask_np_hwd, (1,2,0))
                    
                    # *** MISSING: Extract the sub-volume slice from the mask ***
                    sub_mask_np_hwd = full_mask_np_hwd[:, :, start_d : start_d + self.sub_volume_depth]
                    # Transpose to DxHxW to match the volume format
                    sub_mask_np_dhw = np.transpose(sub_mask_np_hwd, (2, 0, 1))  # Now (32, H_orig, W_orig)
                    
                except Exception as e:
                    print(f"Error loading or slicing mask {full_mask_path} at depth {start_d}: {e}. Creating zero mask.")
                    sub_mask_np_dhw = np.zeros_like(sub_volume_np_dhw, dtype=np.uint8)

        # Optional: Filter out sub-volumes with too little foreground if training for segmentation
        # if mode == 'train' and anomaly_label == 1 and self.min_foreground_percent > 0:
        #     if np.sum(sub_mask_np_dhw) / sub_mask_np_dhw.size < self.min_foreground_percent:
        #         # This sub-volume has too little foreground, could try to get another one
        #         # Or just mark it as less useful. For simplicity, we process it.
        #         # A more complex __getitem__ could resample if this happens too often.
        #         pass

        anomaly_label = 1 if np.any(sub_mask_np_dhw > 0) else 0 # label is 0 if all zeros, we could check the ratio but for volumes is not immediate

        s = self.scales[scale_id]                       # e.g. 1.0, 0.75, 0.5
        D, H, W = sub_volume_np_dhw.shape               # (32, 240, 240)
        h_crop, w_crop = int(H * s), int(W * s)  # e.g. 240, 180, 120 for 1.0, 0.75, 0.5  
        
        y0 = (H - h_crop) // 2
        x0 = (W - w_crop) // 2
        sub_volume_np_dhw = sub_volume_np_dhw[:, y0 : y0 + h_crop, x0 : x0 + w_crop]
        sub_mask_np_dhw   = sub_mask_np_dhw[:,   y0 : y0 + h_crop, x0 : x0 + w_crop]     

        # keep the window (y, x, h, w) in image coordinates *before* any resize
        crop_box = (y0, x0, h_crop, w_crop)
        
        img_tensor = self.transform_img(sub_volume_np_dhw)
        mask_tensor = self.transform_mask(sub_mask_np_dhw)
        
        
        cls_name_actual = patient_data.get('cls_name', 'brain')
        


        return {
            'img': img_tensor,
            'img_mask': mask_tensor,
            'cls_name': cls_name_actual,
            'anomaly': anomaly_label,
            'img_path': str(full_volume_path), # Path to the original full volume
            "cls_id": self.class_name_map_class_id.get(cls_name_actual, 0),
            'patient_id': patient_data.get('patient_id', 'unknown'), # Good to have for aggregation
            'start_depth_idx': start_d, # Useful for reconstructing full volume prediction
            'scale_id': scale_id, # Useful for multi-scale training
            'crop_box': crop_box  # Keep the crop box for resscale
        }


  
