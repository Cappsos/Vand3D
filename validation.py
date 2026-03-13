import os
import torch
import random
import logging
import json
import argparse,yaml
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
import matplotlib.pyplot as plt
from datetime import datetime


from models.adapters import LinearLayer
from models.m3dclip import load_m3dclip_model, get_text_embedding,prepare_text_embeddings, load_prompt_centroids
from utils.transforms import Transform3DForM3DCLIP, Transform3DMask
from utils.metrics import dice_coefficient_3d, iou3d
from utils.fusion import laplacian_fuse
from utils.io import load_gt_mask, load_brain_volume
from utils.volume_utils import reconstruct_volume_from_slices,combine_subvolumes_from_folder
from datasets.dataset3d import BraTS3DSubVolumeDataset




#prova scipy
from sklearn.metrics import precision_recall_curve
from scipy.ndimage import label
from glob import glob

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)

def plot_full_size_inspection(args):
    """
    Plot 16 slices of full-size anomaly maps with GT overlays and underlying brain tissue for visual inspection
    """
    from utils.volume_utils import combine_subvolumes_from_folder, resize_volume
    from glob import glob
    import matplotlib.pyplot as plt
    
    # Find patient directories
    patient_dirs = sorted(glob(os.path.join(f'{args.save_path}/patients/validation/*')))
    
    # Select first few patients for inspection
    patients_to_inspect = patient_dirs[:3]  # Inspect first 3 patients
    
    for patient_dir in patients_to_inspect:
        patient_id = os.path.basename(patient_dir)
        print(f"Inspecting patient: {patient_id}")
        
        # Load and process anomaly map
        full256 = combine_subvolumes_from_folder(
            patient_dir,
            prefix="anomaly_map_brain_depth_",
            full_depth=155,
            block_depth=1 # only 1 slice per file
        )
        full240 = resize_volume(full256, target_h=240, target_w=240)
        
        # Load ground truth
        gt = load_gt_mask(patient_id, args.gt_root)
        
        # Load underlying brain volume
        brain_volume = load_brain_volume(patient_id, args.gt_root)
        
        # Normalize anomaly map for visualization
        anomaly_normalized = normalize(full240)
        
        # Normalize brain volume for visualization
        brain_normalized = normalize(brain_volume)
        
        # Print statistics
        print(f"  Anomaly map: shape={full240.shape}, min={full240.min():.4f}, max={full240.max():.4f}")
        print(f"  GT: shape={gt.shape}, positive_voxels={gt.sum()}/{gt.size} ({100*gt.mean():.2f}%)")
        print(f"  Brain volume: shape={brain_volume.shape}, min={brain_volume.min():.4f}, max={brain_volume.max():.4f}")
        
        # Select 16 slices evenly distributed through volume
        depth = full240.shape[0]
        slice_indices = np.linspace(5, depth-5, 16, dtype=int)  # Avoid first/last few slices
        
        # Create 4x4 subplot
        fig, axes = plt.subplots(4, 4, figsize=(20, 20))
        fig.suptitle(f'Patient {patient_id} - Brain + Anomaly Maps + GT Overlay', fontsize=18)
        
        for i, slice_idx in enumerate(slice_indices):
            row = i // 4
            col = i % 4
            ax = axes[row, col]
            
            # Get slice data
            brain_slice = brain_normalized[slice_idx]      # [H, W] - underlying brain
            anomaly_slice = anomaly_normalized[slice_idx]  # [H, W] - anomaly scores
            gt_slice = gt[slice_idx]                       # [H, W] - ground truth
            
            # Show brain volume as grayscale background
            ax.imshow(brain_slice, cmap='gray', vmin=0, vmax=1, origin='lower')
            
            # Overlay anomaly map (hot colors) - only show significant anomalies
            anomaly_threshold = 0.1  # Only show anomaly scores > 0.1
            anomaly_masked = np.ma.masked_where(anomaly_slice < anomaly_threshold, anomaly_slice)
            im1 = ax.imshow(anomaly_masked, cmap='hot', alpha=0.6, vmin=0, vmax=1, origin='lower')
            
            # Overlay ground truth (green contours) - make it more visible
            gt_contours = np.ma.masked_where(gt_slice == 0, gt_slice)
            im2 = ax.imshow(gt_contours, cmap='Greens', alpha=0.7, vmin=0, vmax=1, origin='lower')
            
            # Add slice info with more details
            gt_pixels = gt_slice.sum()
            anomaly_mean = anomaly_slice.mean()
            anomaly_max = anomaly_slice.max()
            brain_mean = brain_slice.mean()
            
            ax.set_title(f'Slice {slice_idx}\n'
                        f'GT: {gt_pixels} pixels\n'
                        f'Anomaly: μ={anomaly_mean:.3f}, max={anomaly_max:.3f}\n'
                        f'Brain: μ={brain_mean:.3f}', 
                        fontsize=9)
            ax.axis('off')
        
        # Add colorbars with better positioning
        # Anomaly colorbar (hot)
        #cbar1 = fig.colorbar(im1, ax=axes[:, -1], location='right', shrink=0.6, pad=0.02)
        #cbar1.set_label('Anomaly Score (Hot)', rotation=270, labelpad=20, fontsize=12)
        
        # GT colorbar (green) 
        #cbar2 = fig.colorbar(im2, ax=axes[:, -1], location='right', shrink=0.6, pad=0.08)
        #cbar2.set_label('Ground Truth (Green)', rotation=270, labelpad=20, fontsize=12)
        
        # Save plot
        plots_dir = os.path.join(args.save_path, "inspection_plots")
        os.makedirs(plots_dir, exist_ok=True)
        save_path_plot = os.path.join(plots_dir, f'inspection_with_brain_{patient_id}.png')
        plt.savefig(save_path_plot, dpi=200, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved brain+anomaly inspection plot: {save_path_plot}")
        
        # Also create a threshold comparison plot with brain background
        plot_threshold_comparison_with_brain(full240, gt, brain_volume, patient_id, args.save_path)

def load_brain_volume(patient_id, gt_root):
    """
    Load the underlying brain volume (T2W) for visualization
    """
    pat_dir = os.path.join(gt_root, patient_id)
    
    # Look for T2W file
    t2w_files = [f for f in os.listdir(pat_dir) if f.endswith('-t2w.npy')]
    if len(t2w_files) != 1:
        raise FileNotFoundError(f"Expected exactly one '*-t2w.npy' in {pat_dir}, found {t2w_files}")
    
    brain_vol = np.load(os.path.join(pat_dir, t2w_files[0]))
    
    # Fix shape if needed (same logic as GT)
    if brain_vol.shape != (155, 240, 240):
        print(f"Warning: Brain volume shape {brain_vol.shape} for {patient_id}, attempting to fix...")
        if brain_vol.shape == (240, 240, 155):
            brain_vol = np.transpose(brain_vol, (2, 0, 1))  # (H,W,D) -> (D,H,W)
        elif brain_vol.shape == (240, 155, 240):
            brain_vol = np.transpose(brain_vol, (1, 0, 2))  # (H,D,W) -> (D,H,W)
        else:
            raise ValueError(f"Unexpected brain volume shape {brain_vol.shape} for {patient_id}")
    
    print(f"Brain volume loaded: {patient_id}, shape={brain_vol.shape}")
    return brain_vol.astype(np.float32)

def plot_threshold_comparison_with_brain(anomaly_map, gt, brain_volume, patient_id, save_path):
    """
    Plot the same slice with different threshold values with brain background
    """
    # Select a slice with reasonable GT content
    gt_counts_per_slice = [gt[i].sum() for i in range(gt.shape[0])]
    best_slice_idx = np.argmax(gt_counts_per_slice)
    
    if gt_counts_per_slice[best_slice_idx] == 0:
        # If no GT found, pick middle slice
        best_slice_idx = gt.shape[0] // 2
    
    brain_slice = normalize(brain_volume[best_slice_idx])
    anomaly_slice = anomaly_map[best_slice_idx]
    gt_slice = gt[best_slice_idx]
    
    # Test different thresholds
    thresholds = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Patient {patient_id} - Slice {best_slice_idx} - Threshold Comparison with Brain Background', fontsize=16)
    
    for i, threshold in enumerate(thresholds):
        row = i // 3
        col = i % 3
        ax = axes[row, col]
        
        # Apply threshold
        binary_pred = (anomaly_slice > threshold).astype(np.uint8)
        
        # Compute Dice for this slice
        dice_score = dice_coefficient_3d(binary_pred, gt_slice)
        
        # Show brain volume as grayscale background
        ax.imshow(brain_slice, cmap='gray', vmin=0, vmax=1, origin='lower')
        
        # Overlay anomaly scores (faded hot colormap)
        anomaly_overlay = np.ma.masked_where(anomaly_slice < 0.1, anomaly_slice)
        ax.imshow(anomaly_overlay, cmap='hot', alpha=0.3, vmin=0, vmax=3, origin='lower')
        
        # Overlay binary prediction (blue contours)
        pred_overlay = np.ma.masked_where(binary_pred == 0, binary_pred)
        ax.imshow(pred_overlay, cmap='Blues', alpha=0.6, origin='lower')
        
        # Overlay GT (green contours)
        gt_overlay = np.ma.masked_where(gt_slice == 0, gt_slice)
        ax.imshow(gt_overlay, cmap='Greens', alpha=0.8, origin='lower')
        
        # Count predicted pixels
        pred_pixels = binary_pred.sum()
        gt_pixels = gt_slice.sum()
        
        ax.set_title(f'Threshold: {threshold:.1f}\n'
                    f'Dice: {dice_score:.3f}\n'
                    f'Pred: {pred_pixels}, GT: {gt_pixels}', 
                    fontsize=11)
        ax.axis('off')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='gray', label='Brain Tissue'),
        Patch(facecolor='red', alpha=0.3, label='Anomaly Scores'),
        Patch(facecolor='blue', alpha=0.6, label='Binary Prediction'),
        Patch(facecolor='green', alpha=0.8, label='Ground Truth')
    ]
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=4, fontsize=12)
    
    # Save threshold comparison
    plots_dir = os.path.join(save_path, "threshold_plots")
    os.makedirs(plots_dir, exist_ok=True)
    save_file = os.path.join(plots_dir, f'thresholds_with_brain_{patient_id}_slice_{best_slice_idx}.png')
    plt.savefig(save_file, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved brain+threshold comparison: {save_file}")
    
def get_text_embedding(text_prompt, model, tokenizer, device):
    text_tensor = tokenizer(text_prompt, max_length=512, truncation=True, padding="max_length", return_tensors="pt")
    input_id = text_tensor["input_ids"].to(device=device)
    attention_mask = text_tensor["attention_mask"].to(device=device)
    with torch.inference_mode(), torch.cuda.amp.autocast():
        text_features_all_tokens = model.encode_text(input_id, attention_mask)
        text_embedding = text_features_all_tokens[:, 0, :]  # CLS token
    return text_embedding

def normalize(pred, max_value=None, min_value=None):
    if max_value is None or min_value is None:
        return (pred - pred.min()) / (pred.max() - pred.min())
    else:
        return (pred - min_value) / (max_value - min_value)

def plot_anomaly_overlay(original_volume, anomaly_map, gt_mask, save_path, sample_name, temperature):
    """
    Plot 5 slices with anomaly overlay and ground truth
    """
    # Normalize original volume for display
    original_normalized = normalize(original_volume)
    
    # Create plots directory
    plots_dir = os.path.join(save_path, "plots_3D")
    os.makedirs(plots_dir, exist_ok=True)
    
    print(f"Final Anomaly Map min: {anomaly_map.min():.4f}, max: {anomaly_map.max():.4f}, mean: {anomaly_map.mean():.4f}")
    
    slices_to_show = [5, 10, 15, 25, 30]
    # Adjust slice indices if volume is smaller
    max_slice = original_volume.shape[0] - 1
    slices_to_show = [min(s, max_slice) for s in slices_to_show]
    
    fig, axes = plt.subplots(1, len(slices_to_show), figsize=(5 * len(slices_to_show), 5))
    if len(slices_to_show) == 1:
        axes = [axes]
    
    for i, slice_idx in enumerate(slices_to_show):
        img = original_normalized[slice_idx]        # (H, W)
        anomaly = anomaly_map[slice_idx]           # (H, W)
        gt = gt_mask[slice_idx]                    # (H, W), 0/1 mask
        
        ax = axes[i]
        ax.imshow(img, cmap='gray', origin='lower')
        # anomaly overlay
        im = ax.imshow(anomaly, cmap='jet', alpha=0.4, vmin=0, vmax=1, origin='lower')
        # ground-truth overlay in red
        ax.imshow(gt, cmap='Reds', alpha=0.3, vmin=0, vmax=1, origin='lower')
        
        ax.set_title(f"Slice {slice_idx}")
        ax.axis('off')
    
    # single colorbar (for anomaly)
    cbar = fig.colorbar(im, ax=axes, fraction=0.046, pad=0.04)
    cbar.set_label('Anomaly score')
    
    filename = f"vand_overlay_{sample_name}_temp_{str(temperature).replace('.', '')}.png"
    save_file = os.path.join(plots_dir, filename)
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved plot: {save_file}")
    

def validation_dice3d(args):
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    save_path = args.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # Simple logging
    print(f"Testing 3D Dice Score")
    print(f"Dataset: {args.dataset}")
    print(f"Test data path: {args.test_data_path}")
    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"Temperature: {args.temperature_scale}")
    print(f"Prompt ensemble: {args.prompt_ensemble}")
    print("-" * 50)
    
    model, tokenizer = load_m3dclip_model(device)
    
    if args.prompt_ensemble:
        stacked_text_features = load_prompt_centroids(prompt_yaml_path=args.prompt_path, model=model, tokenizer=tokenizer, device=device)
    else:
        stacked_text_features = prepare_text_embeddings(model, tokenizer, device)
    
    model.eval()  # Set model to evaluation mode
    
    # Load adapter
    feature_dim = model.config.hidden_size
    output_adapter_dim = model.config.hidden_size
    adapter_k = len(args.features_list)
    
    linearlayer = LinearLayer(
        dim_in=feature_dim,
        dim_out=output_adapter_dim,
        k=adapter_k,
        model_type_str='vit_3d'
    ).to(device)
    
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    linearlayer.load_state_dict(checkpoint["trainable_linearlayer"])
    linearlayer.eval()
    
    # Dataset
    img_transform = Transform3DForM3DCLIP(target_height=256, target_width=256, target_depth=32)
    mask_transform = Transform3DMask(target_height=240, target_width=240, target_depth=args.depth)
    
    if args.depth == 1:
        stride = 1
    else:
        stride = 16
        
    test_data = BraTS3DSubVolumeDataset(
        root=args.test_data_path, 
        transform_img=img_transform, 
        transform_mask=mask_transform, 
        dataset_name=args.dataset, 
        mode='val',
        sub_volume_depth=args.depth,  # e.g. 32
        stride_depth=stride,  # Depth of each sub-volume
        scales=args.fusion_scales
    )
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)
    
    # Results storage
    dice_scores = []
    all_maps = []
    all_gts = []

    thresholds = np.linspace(0.0, adapter_k, 51)         # e.g. 51 candidates from 0.00 to 1.00
    nT = len(thresholds)

    TP = np.zeros(nT, dtype=np.float64)            # true‐positives at each thr
    PP = np.zeros(nT, dtype=np.float64)            # predicted‐positives at each thr
    GTpos = 0.0                                    # total ground‐truth positives

    eps = 1e-8
    
    # Create dict to store views for each patient
    acc = {}   # key = (patient_id, start_depth_idx) → list_of_maps
    S = len(test_data.scales)
    
    print("Starting 3D inference...")
    
    # Inference loop
    for idx, items in enumerate(tqdm(test_dataloader)):
        
        if idx == 0:
            pass
   
        volume = items['img'].to(device)  # [B, C, D, H, W]
        cls_name = items['cls_name'][0]
        gt_mask = items['img_mask']       # [B, D, H, W] or [B, C, D, H, W]
        
        start_depth_idx = items['start_depth_idx'][0].item()  # Get start depth index
        patient_id = items['patient_id'][0]
        
        # create a save dir for each patient to save the different sub-volumes
        patient_save_path = os.path.join(f'{save_path}/validation', patient_id)
        os.makedirs(patient_save_path, exist_ok=True)
        
        original_volume = volume.squeeze(0).squeeze(0).cpu().numpy()
        
        # Process ground truth
        if gt_mask.dim() == 5:
            gt_mask = gt_mask.squeeze(1)  # Remove channel dim if present
        gt_mask = gt_mask.squeeze(0).numpy()  # [D, H, W]
        gt_mask = (gt_mask > 0.5).astype(np.float32)  # Binarize

        with torch.no_grad(), torch.cuda.amp.autocast():
            # Extract features
            
            all_image_token_features, attentions_all_layers, layer_list = model.encode_image(volume, args.features_list)
            
            # Prepare patch tokens
            selected_vit_layer_patch_tokens = []
            for layer in layer_list:
                layer = layer[:, 1:, :]  # Exclude CLS token
                selected_vit_layer_patch_tokens.append(layer)
            
            # Adapt features
            adapted_multi_layer_patch_tokens_list = linearlayer(selected_vit_layer_patch_tokens)
            
            # Compute anomaly maps for each layer
            layer_anomaly_maps_full_res = []
            for i in range(adapter_k):
                adapted_patch_features = adapted_multi_layer_patch_tokens_list[i]
                adapted_patch_norm = F.normalize(adapted_patch_features, dim=-1)
                
                # Compute logits
                anomaly_map = ((1/args.temperature_scale) * adapted_patch_norm @ stacked_text_features)
                B, L, C = anomaly_map.shape
                
                # Reshape to 3D structure
                num_patches_d = model.config.img_size[0] // model.config.patch_size[0] 
                num_patches_h = model.config.img_size[1] // model.config.patch_size[1]
                num_patches_w = model.config.img_size[2] // model.config.patch_size[2]
                
                reshaped_logits = anomaly_map.permute(0, 2, 1).view(
                    B, C, num_patches_d, num_patches_h, num_patches_w
                )
                
                # Interpolate to full resolution
                upsampled_logits = F.interpolate(
                    reshaped_logits,
                    size=(32,240,240),
                    mode='trilinear',
                    align_corners=False
                )
                # take middle slice along depth for a 2D anomaly map
                if args.depth == 1:
                    upsampled_logits = upsampled_logits[:, :, 16, :, :]  # [B, C, H, W]
                # Softmax and extract anomaly channel
                probs = torch.softmax(upsampled_logits, dim=1)  # [B,2,D,H,W]
                anomaly_map_layer = probs[:, 1]  # [B,D,H,W] - anomaly channel only
                layer_anomaly_maps_full_res.append(anomaly_map_layer)
            
            # Combine layers (sum as in VAND)
            final_anomaly_map = torch.stack(layer_anomaly_maps_full_res, dim=0).sum(dim=0)
            final_anomaly_map = final_anomaly_map.squeeze(0).cpu().numpy()  # [D,H,W]
            
            # save the anomaly map for this patient in RAM for later
            key = (patient_id, start_depth_idx)
            acc.setdefault(key, []).append(final_anomaly_map)
            
            # If we have enough maps for fusion, fuse them
            if len(acc[key]) == S:
                if args.fusion:
                    fused = laplacian_fuse(acc[key], orig_size=acc[key][0].shape)
                    np.save(os.path.join(patient_save_path,
                            f"anomaly_map_depth_{start_depth_idx}.npy"), fused)
                else:
                    np.save(os.path.join(patient_save_path,
                            f"anomaly_map_depth_{start_depth_idx}.npy"), final_anomaly_map)
                del acc[key]      # free RAM
                
            all_maps.append(final_anomaly_map)
            all_gts.append(gt_mask)
            m_flat = final_anomaly_map.ravel()
            g_flat = gt_mask.ravel().astype(np.uint8)

            # accumulate total positives
            GTpos += g_flat.sum()
            
            preds = (m_flat[None, :] > thresholds[:, None])    # shape [nT, M]
            PP   += preds.sum(axis=1)                          # predicted positives
            TP   += (preds & (g_flat[None, :] == 1)).sum(axis=1)
            # Normalize anomaly map for plotting (0-1 range)
            
            
    precision = TP / (PP + eps)
    recall    = TP / (GTpos + eps)
    f1_scores = 2 * precision * recall / (precision + recall + eps)
    
    # Find the best threshold on the validation set
    plot_full_size_inspection(args)
    patient_dirs = sorted(glob(os.path.join(f'{save_path}/validation/*')))
    n_patients = len(patient_dirs)

    all_preds = []
    all_gts = []

    
   
    orig_depth = 155
    
    for patient_dir in patient_dirs:
        patient_id = os.path.basename(patient_dir)

        if args.fusion == True:
            reconstructed_volume = combine_subvolumes_from_folder(
                patient_dir,
                prefix="anomaly_map_depth_",
                full_depth=155,
                block_depth=args.depth
            )
            
            print(f"Reconstructed volume shape: {reconstructed_volume.shape}")
            
        elif args.depth == 1:
            reconstructed_volume = reconstruct_volume_from_slices(
                patient_dir,
                prefix="anomaly_map_depth_",
                full_depth=155,
                target_h=240,
                target_w=240
            )
        else:
            reconstructed_volume = combine_subvolumes_from_folder(
                patient_dir,
                prefix="anomaly_map_depth_",
                full_depth=155,
                block_depth=args.depth
            )
            
        all_preds.append(reconstructed_volume)
            
        # Load GT
        gt_root = getattr(args, 'gt_root', args.test_data_path)
        gt = load_gt_mask(patient_id, gt_root)
        all_gts.append(gt)


    # 3) threshold sweep in [0, k] (raw range)
    k = 3
    thresholds = np.linspace(0, k, 51)

    # accumulate
    dice = np.zeros(len(thresholds), dtype=np.float64)  # Dice per threshold
    for pred, gt in zip(all_preds, all_gts):
        
        print(f'  Anomaly map shape: {pred.shape}, min={pred.min():.4f}, max={pred.max():.4f}')
        for i, thr in enumerate(thresholds):
            mask = (pred > thr)
            dice_score = dice_coefficient_3d(mask, gt)
            dice[i] += dice_score  # accumulate Dice scores
    dice /= n_patients  # average over all patients

    # compute dice3d per threshold

    best_idx = np.argmax(dice)
    best_thr = thresholds[best_idx]
    best_dice = dice[best_idx]

    print(f"Best threshold = {best_thr:.3f}  →  mean Dice = {best_dice:.4f}")
    
    # Save threshold with metadata
    threshold_data = {
        'best_threshold': float(best_thr),
        'best_dice': float(best_dice),
        'evaluation_date': datetime.now().isoformat(),
        'model_checkpoint': args.checkpoint_path,
        'dataset': args.dataset,
        'depth': args.depth,
        'temperature_scale': args.temperature_scale,
        'features_list': args.features_list,
        'total_patients': n_patients
    }

    threshold_file = os.path.join(save_path, 'best_threshold.json')
    with open(threshold_file, 'w') as f:
        json.dump(threshold_data, f, indent=2)

    print(f"Threshold saved to: {threshold_file}")

    # optional: print the full curve
    for t, d in zip(thresholds, dice):
        print(f"{t:.3f}  {d:.4f}")


    # Compute statistics
    if dice_scores:
        mean_dice = np.mean(dice_scores)
        std_dice = np.std(dice_scores)
        median_dice = np.median(dice_scores)
        
        print("\n" + "="*50)
        print("3D DICE SCORE RESULTS")
        print("="*50)
        print(f"Number of test samples: {len(dice_scores)}")
        print(f"Mean Dice3D: {mean_dice:.4f} ± {std_dice:.4f}")
        print(f"Median Dice3D: {median_dice:.4f}")
        print(f"Min Dice3D: {np.min(dice_scores):.4f}")
        print(f"Max Dice3D: {np.max(dice_scores):.4f}")
        
        # Save results
        results_file = os.path.join(save_path, 'dice3d_results.txt')
        with open(results_file, 'w') as f:
            f.write(f"3D Dice Score Results\n")
            f.write(f"Number of test samples: {len(dice_scores)}\n")
            f.write(f"Mean Dice3D: {mean_dice:.4f} ± {std_dice:.4f}\n")
            f.write(f"Median Dice3D: {median_dice:.4f}\n")
            f.write(f"Min Dice3D: {np.min(dice_scores):.4f}\n")
            f.write(f"Max Dice3D: {np.max(dice_scores):.4f}\n")
            f.write(f"\nIndividual scores:\n")
            for i, score in enumerate(dice_scores):
                f.write(f"Sample {i+1}: {score:.4f}\n")
        
        print(f"\nResults saved to {results_file}")
    else:
        print("No test samples processed!")
    
    if args.cleanup:
        print("Cleaning up temporary .npy files...")
        for patient_dir in patient_dirs:
            npy_files = glob(os.path.join(patient_dir, '*.npy'))
            for npy_file in npy_files:
                os.remove(npy_file)
        print("Cleanup complete.")


def arg_parser_with_config():
    parser = argparse.ArgumentParser("VAND3D Eval", add_help=True)
    
    parser.add_argument("--config", type=str, help="YAML config file")
    # Paths
    parser.add_argument("--test_data_path", type=str, default="/home/nicoc/data/BraTS_3D", help="test dataset path")
    parser.add_argument("--save_path", type=str, default='./test_results_dice3d/3D', help='path to save results')
    parser.add_argument("--checkpoint_path", type=str, default='./experiments_3D/epoch_12.pth', help='path to trained model')
    parser.add_argument("--gt_root", type=str, default="/home/nicoc/data/BraTS_3D/", help="ground truth masks root path")
    
    # prompt ensemble
    parser.add_argument("--prompt_ensemble", type=bool, default=True, help="use prompt ensemble or not")
    parser.add_argument("--prompt_path", type=str, default='./configs/prompts_brats.yaml', help="prompt ensemble path")
    
    # fusion
    parser.add_argument("--fusion", type=bool, default=False, help="use fusion of sub-volumes or not")
    parser.add_argument("--fusion_scales", type=list, nargs="+", default=[1], help="scales for fusion (e.g. [1, 2, 4])")
    
    # Model
    parser.add_argument("--dataset", type=str, default='brats', help="test dataset name")
    parser.add_argument("--features_list", type=int, nargs="+", default=[3, 4, 11], help="features used")
    parser.add_argument("--temperature_scale", type=float, default=0.2, help="temperature scaling for logits")
    parser.add_argument("--depth", type=int, default=32, help="depth of the sub-volumes (default 32)")
    
    #cleanup
    parser.add_argument("--cleanup", type=bool, default=True, help="remove saved .npy files after evaluation")
    # Testing
    parser.add_argument("--seed", type=int, default=111, help="random seed")
    
    args, _ = parser.parse_known_args()
    # 2) if config given, override defaults
    if args.config:
        cfg = load_config(args.config)
        parser.set_defaults(**cfg)
    # 3) full parse
    
    return parser.parse_args()

if __name__ == '__main__':
    print("VAND3D Eval Script")
    args = arg_parser_with_config()
    
    setup_seed(args.seed)
    validation_dice3d(args)
