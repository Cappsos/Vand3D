import os
import torch
import random
import logging
import argparse, yaml
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
import matplotlib.pyplot as plt




from models.adapters import LinearLayer
from utils.transforms import Transform3DForM3DCLIP, Transform3DMask
from utils.metrics import dice_coefficient_3d
from utils.io_lung import load_gt_mask, load_brain_volume, get_best_threshold
from utils.volume_utils import reconstruct_volume_from_slices,combine_subvolumes_from_folder
from utils.fusion import  paste_and_fuse, ensure_key
from models.m3dclip import load_m3dclip_model, prepare_text_embeddings, get_text_embedding, load_prompt_centroids
from datasets.dataset3d_lungs import Lung3DSubVolumeDataset


#prova scipy
from sklearn.metrics import precision_recall_curve
from scipy.ndimage import label
from glob import glob
from skimage.transform import resize



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
    


def normalize(pred, max_value=None, min_value=None):
    if max_value is None or min_value is None:
        return (pred - pred.min()) / (pred.max() - pred.min())
    else:
        return (pred - min_value) / (max_value - min_value)



def plot_anomaly_overlay(original_volume, anomaly_map, gt_mask, save_path, sample_name, temperature):
    """
    Plot single 2D slice with anomaly overlay and ground truth
    """
    # Handle the volume dimensions properly
    original_volume = np.squeeze(original_volume)
    
    # Check if we have a 3D volume and need to extract a slice
    if original_volume.ndim == 3:
        # For 3D volume, take the middle slice along the depth dimension
        if args.depth == 1:
            # If depth=1, we're working with 2D slices, take the middle slice
            middle_slice = original_volume.shape[2] // 2
            original_slice = original_volume[:, :, middle_slice]
        else:
            # For 3D blocks, take middle slice along first dimension
            middle_slice = original_volume.shape[0] // 2
            original_slice = original_volume[middle_slice, :, :]
    elif original_volume.ndim == 2:
        # Already 2D
        original_slice = original_volume
    else:
        raise ValueError(f"Unexpected original_volume shape: {original_volume.shape}")
    
    # Resize to match anomaly map dimensions if needed
    if original_slice.shape != (512, 512):
        original_slice = resize(
            original_slice,
            (512, 512),
            order=1,              # bilinear
            mode='reflect',       # boundary handling
            anti_aliasing=True,   # reduce aliasing artifacts
            preserve_range=True   # keep original data range
        )
    
    # Ensure gt_mask is 2D
    gt_mask = np.squeeze(gt_mask)
    if gt_mask.ndim == 3:
        # Take the same slice as the original volume
        if args.depth == 1:
            middle_slice = gt_mask.shape[2] // 2 if gt_mask.shape[2] > 1 else 0
            gt_mask = gt_mask[:, :, middle_slice]
        else:
            middle_slice = gt_mask.shape[0] // 2 if gt_mask.shape[0] > 1 else 0
            gt_mask = gt_mask[middle_slice, :, :]
    
    # Ensure anomaly_map is 2D
    anomaly_map = np.squeeze(anomaly_map)
    if anomaly_map.ndim == 3:
        # Take the same slice as the original volume
        if args.depth == 1:
            middle_slice = anomaly_map.shape[2] // 2 if anomaly_map.shape[2] > 1 else 0
            anomaly_map = anomaly_map[:, :, middle_slice]
        else:
            middle_slice = anomaly_map.shape[0] // 2 if anomaly_map.shape[0] > 1 else 0
            anomaly_map = anomaly_map[middle_slice, :, :]
    
    # Normalize original slice for display
    original_normalized = normalize(original_slice)
    
    # Create plots directory
    plots_dir = os.path.join(save_path, "plots_2D")
    os.makedirs(plots_dir, exist_ok=True)
    
    print(f"2D Anomaly Map min: {anomaly_map.min():.4f}, max: {anomaly_map.max():.4f}, mean: {anomaly_map.mean():.4f}")
    print(f"Original slice shape: {original_slice.shape}, GT mask shape: {gt_mask.shape}, Anomaly map shape: {anomaly_map.shape}")
    
    # Create single subplot
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    # Show original image as grayscale background
    ax.imshow(original_normalized, cmap='gray', vmin=0, vmax=1, origin='lower')
    
    # Overlay anomaly map (hot colors) - mask low values to show only significant anomalies
    anomaly_threshold = 0.1  # Only show anomaly scores > 0.1
    anomaly_masked = np.ma.masked_where(anomaly_map < anomaly_threshold, anomaly_map)
    im1 = ax.imshow(anomaly_masked, cmap='hot', alpha=0.6, vmin=0, vmax=1, origin='lower')
    
    # Overlay ground truth (green contours)
    gt_overlay = np.ma.masked_where(gt_mask == 0, gt_mask)
    im2 = ax.imshow(gt_overlay, cmap='Greens', alpha=0.8, vmin=0, vmax=1, origin='lower')
    
    # Add title with statistics
    gt_pixels = gt_mask.sum()
    anomaly_mean = anomaly_map.mean()
    anomaly_max = anomaly_map.max()
    
    ax.set_title(f'2D VAND-style with 3D CLIP - {sample_name}\n'
                f'GT: {gt_pixels} pixels | Anomaly: μ={anomaly_mean:.3f}, max={anomaly_max:.3f}', 
                fontsize=12)
    ax.axis('off')
    
    # Add colorbar for anomaly scores
    cbar = fig.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Anomaly Score', fontsize=12)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='gray', label='Brain Tissue'),
        Patch(facecolor='red', alpha=0.6, label='Anomaly Scores'),
        Patch(facecolor='green', alpha=0.8, label='Ground Truth')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    # Save with descriptive filename
    filename = f"2d_vand_overlay_{sample_name}_temp_{str(temperature).replace('.', '')}.png"
    save_file = os.path.join(plots_dir, filename)
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved 2D plot: {save_file}")
    
def test_dice3d(args):
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
    print("-" * 50)
    
    model, tokenizer = load_m3dclip_model(device)
    if args.prompt_ensemble:
        stacked_text_features = load_prompt_centroids(prompt_yaml_path=args.prompt_path, model=model, tokenizer=tokenizer, device=device)
    else:
        stacked_text_features = prepare_text_embeddings(model, tokenizer, device)
    
    model.eval()
    # check cosine similarity between the text features
    print(f"Text features shape: {stacked_text_features.shape}")
    if stacked_text_features.shape[1] == 2:
        feature_1 = stacked_text_features[:, 0]  # [D]
        feature_2 = stacked_text_features[:, 1]  # [D]
        
        # Calculate cosine similarity
        cosine_sim = F.cosine_similarity(feature_1.unsqueeze(0), feature_2.unsqueeze(0), dim=1)
        print(f"Cosine similarity between text features: {cosine_sim.item():.4f}")
    
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
    mask_transform = Transform3DMask(target_height=512, target_width=512, target_depth=args.depth)
    
    if args.depth == 1:
        stride=1
    else:
        stride = 16
        
        
    test_data = Lung3DSubVolumeDataset(
        root=args.test_data_path, 
        transform_img=img_transform, 
        transform_mask=mask_transform, 
        dataset_name=args.dataset, 
        mode='test',
        sub_volume_depth=args.depth,
        stride_depth=stride, # Depth of each sub-volume
        scales = args.fusion_scales
    
    )
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)
    
    # Results storage
    dice_scores = []
    all_maps = []
    all_gts = []
    
         # total ground‐truth positives

    eps = 1e-8
    
    
    # Create dict to store views for each patient
    acc = {}   # key = (patient_id, start_depth_idx) → list_of_maps
    S = len(test_data.scales)

    print("Starting 3D inference...")
    
    # Inference loop
    for idx, items in enumerate(tqdm(test_dataloader)):
        if idx == 24*5:
            pass
            
        volume = items['img'].to(device)  # [B, C, D, H, W]
        cls_name = items['cls_name'][0]
        gt_mask = items['img_mask']       # [B, D, H, W] or [B, C, D, H, W]
        
        start_depth_idx = items['start_depth_idx'][0].item()  # Get start depth index
        patient_id = items['patient_id'][0]
        scale_id = items['scale_id'].item()
        original_depth = items['original_depth'][0]
      
        # create a save dir for each patient to save the different sub-volumes
        patient_save_path = os.path.join(f'{save_path}/validation', f'{patient_id}_{original_depth}')
        os.makedirs(patient_save_path, exist_ok=True)
        
        original_volume = volume.squeeze(0).squeeze(0).cpu().numpy()
        
        full_shape = (32, 512, 512)               # D, H, W  (your upsampled size)
        key = (patient_id, start_depth_idx)
        ensure_key(acc, key, k=adapter_k, shape=full_shape)
        buf = acc[key]
        
        # save the reference volume and mask for visualization
        if buf["ref_vol"] is None and scale_id == 0:
            buf["ref_vol"]  = volume.squeeze(0).squeeze(0).cpu().numpy()
            buf["ref_mask"] = gt_mask.squeeze(0).numpy()

        
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
                    size=(32,512,512),
                    mode='trilinear',
                    align_corners=False
                )
                if args.depth ==1:
                    # take middle slice along depth for a 2D anomaly map
                    upsampled_logits = upsampled_logits[:, :, 16, :, :]  # [B, C, H, W]
                    
                # Softmax and extract anomaly channel
                probs = torch.softmax(upsampled_logits, dim=1)  # [B,2,D,H,W]
                anomaly_map_layer = probs[:, 1]  # [B,D,H,W] - anomaly channel only
                layer_anomaly_maps_full_res.append(anomaly_map_layer)
                
            # store layer by layer for later fusion
            y0, x0, h_crop, w_crop = items['crop_box']
            for i, layer_map in enumerate(layer_anomaly_maps_full_res):
                layer_np = layer_map.squeeze(0).cpu().numpy()          # (D,240,240)

                # resize to the crop size of this view
                resized = torch.nn.functional.interpolate(
                    torch.from_numpy(layer_np)[None, None],
                    size=(layer_np.shape[0], h_crop, w_crop),
                    mode="trilinear", align_corners=False
                ).squeeze().numpy().astype(np.float32)

                buf["layer_views"][i].append(resized)
                buf["layer_boxes"][i].append((int(y0), int(x0), int(h_crop), int(w_crop)))
    
            
            # Combine layers (sum as in VAND)
            final_anomaly_map = torch.stack(layer_anomaly_maps_full_res, dim=0).sum(dim=0)
            final_anomaly_map = final_anomaly_map.squeeze(0).cpu().numpy()  # [D,H,W]
                       
           
            
            # If we have enough maps for fusion, fuse them
            ready = all(len(v) == S for v in buf["layer_views"])
            if ready:
                fused_layers = []
                for i in range(adapter_k):
                    fused_i = paste_and_fuse(
                        buf["layer_views"][i],
                        buf["layer_boxes"][i],
                        full_shape=full_shape,          # (32,240,240)
                        mode="mean"                     # or "max"
                    )
                    fused_layers.append(fused_i)

                fused = np.sum(np.stack(fused_layers, axis=0), axis=0)   # sum over layers

                out_path = os.path.join(patient_save_path,
                                        f"anomaly_map_depth_{start_depth_idx}.npy")
                np.save(out_path, fused.astype(np.float32))

               
                plot_every_k = 1
                if (idx + 1) % plot_every_k == 0 or idx == 0:
                    sample_name = f"sample_{start_depth_idx}_{patient_id}"
                    plot_anomaly_overlay(
                        buf["ref_vol"], normalize(fused), buf["ref_mask"],
                        save_path, sample_name, args.temperature_scale
                    )

                del acc[key]      # free RAM
               
             

            anomaly_map_normalized = normalize(final_anomaly_map)

            
            



    patient_dirs = sorted(glob(os.path.join(f'{save_path}/test/*')))
    n_patients = len(patient_dirs)

    all_preds = []
    all_gts = []
    

    
    
    

        
    for patient_dir in patient_dirs:
        patient_folder_name = os.path.basename(patient_dir)
        
        # Extract patient_id and original_depth from folder name
        # Format: "lung_047_636" where 636 is the depth
        try:
            parts = patient_folder_name.split('_')
            if len(parts) >= 3:
                # Take the last part as depth, join the rest as patient_id
                original_depth = int(parts[-1])
                patient_id = '_'.join(parts[:-1])  # Join all parts except the last one
            else:
                # Fallback if unexpected format
                patient_id = patient_folder_name
                original_depth = 155
                print(f"Warning: Unexpected folder name format {patient_folder_name}, using default depth 155")
        except (ValueError, IndexError) as e:
            patient_id = patient_folder_name
            original_depth = 155
            print(f"Warning: Error parsing folder name {patient_folder_name}: {e}, using default depth 155")

        try:
            if args.depth == 1:
                reconstructed_volume = reconstruct_volume_from_slices(
                    patient_dir,
                    prefix="anomaly_map_depth_",
                    full_depth=original_depth,  # Use extracted depth
                    target_h=512,  # Match your upsampled size
                    target_w=512   # Match your upsampled size
                )
            else:
                reconstructed_volume = combine_subvolumes_from_folder(
                    patient_dir,
                    prefix="anomaly_map_depth_",
                    full_depth=original_depth,  # Use extracted depth
                    block_depth=args.depth
                )
                
            all_preds.append(reconstructed_volume)
            
            # Load GT
            gt_root = getattr(args, 'gt_root', args.gt_root)
            gt = np.load(os.path.join(gt_root,f'{patient_id}.npy'))
            gt = np.transpose(gt, (2, 0, 1))
            all_gts.append(gt)
            
        except Exception as e:
                print(f"Error processing patient {patient_id} with depth {original_depth}: {e}")
                continue
    
    threshold = get_best_threshold(args.dice_threshold)
    
    if threshold is not None:
        print(f"Using threshold from {args.dice_threshold}: {threshold}")
    else:
        print("No threshold found, using default 0.5")
        threshold = 0.5  # Default threshold if none provided
    dice_scores = []
    for m, g in zip(all_preds, all_gts):
        # threshold
        binary = (m > threshold).astype(np.uint8)
        gt = (gt > 0.5).astype(np.uint8)

        dice_scores.append(dice_coefficient_3d(binary, g))

    # report
    mean_d   = np.mean(dice_scores)
    std_d    = np.std(dice_scores)
    median_d = np.median(dice_scores)
    print(f"Per‐case Dice: {mean_d:.3f} ± {std_d:.3f}, median {median_d:.3f}")
    
    
                
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

def arg_parser_with_config():
    parser = argparse.ArgumentParser("VAND3D Testing", add_help=True)
    
    parser.add_argument("--config", type=str, help="YAML config file")
    
    # Paths
    parser.add_argument("--test_data_path", type=str, default="/home/nicoc/data/BraTS_3D", help="test dataset path")
    parser.add_argument("--save_path", type=str, default='./test_results_dice3d/2D', help='path to save results')
    parser.add_argument("--checkpoint_path", type=str, default='./experimentds_3D/epoch_12.pth', help='path to trained model')
    # prompt ensemble
    parser.add_argument("--prompt_ensemble", type=bool, default=True, help="use prompt ensemble or not")
    parser.add_argument("--prompt_path", type=str, default='./configs/prompts_brats.yaml', help="prompt ensemble path")
    
    # fusion
    parser.add_argument("--fusion", type=bool, default=False, help="Use fusion, default False")
    parser.add_argument("--fusion_scales", type=list, default=[1], help="Scales to use for fusion, default [1]")
    # Model
    parser.add_argument("--dataset", type=str, default='brats', help="test dataset name")
    parser.add_argument("--features_list", type=int, nargs="+", default=[3, 4, 11], help="features used")
    parser.add_argument("--temperature_scale", type=float, default=0.2, help="temperature scaling for logits")
    parser.add_argument("--dice_threshold", type=str, default="./test_results_dice3d/2D/", help="path of the json with the threshold for dice score")
    parser.add_argument("--depth", type=int, default=1, help="depth of the sub-volume")
    parser.add_argument("--gt_root", type=str, default="/home/nicoc/data/BraTS_3D", help="path to ground truth masks")
    
    
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
    print("VAND3D Training Script")
    args = arg_parser_with_config()
    

    
    setup_seed(args.seed)
    test_dice3d(args)
