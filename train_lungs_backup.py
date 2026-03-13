
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
import os
import matplotlib.pyplot as plt
import transformers
import torch.nn.functional as F
from models.adapters import LinearLayer
from utils.transforms import Transform3DForM3DCLIP, Transform3DMask
from models.m3dclip import load_m3dclip_model, prepare_text_embeddings, get_text_embedding, load_prompt_centroids
import random
import logging
from datasets.dataset3d_lungs import BraTS3DSubVolumeDataset
from utils.loss import FocalLoss, BinaryDiceLoss, FocalLoss_logits
import argparse, yaml
from tqdm import tqdm
from torch.utils.data import DataLoader
import json
from glob import glob

from utils.metrics import dice_coefficient_3d
from Vand.Vand3D.utils.io_lung import load_gt_mask
from utils.volume_utils import reconstruct_volume_from_slices,combine_subvolumes_from_folder
from utils.fusion import  paste_and_fuse, ensure_key

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
    

def train(args):
    # configs
    epochs = args.epoch
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    image_size = args.image_size
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    save_path = args.save_path
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    txt_path = os.path.join(save_path, 'log.txt')  # log
    
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    root_logger.setLevel(logging.WARNING)
    logger = logging.getLogger('train')
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s',
                                  datefmt='%y-%m-%d %H:%M:%S')
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(txt_path, mode='w')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    

    # model configs
    features_list = args.features_list
    
    model, tokenizer = load_m3dclip_model(device)
    
    if args.prompt_ensemble:
        stacked_text_features = load_prompt_centroids(prompt_yaml_path=args.prompt_path,model=model, tokenizer=tokenizer, device=device) 
    else:
        stacked_text_features = prepare_text_embeddings(model, tokenizer, device)
    
    # debug cosine similarity text features
    print(f"Text features shape: {stacked_text_features.shape}")
    if stacked_text_features.shape[1] == 2:
        feature_1 = stacked_text_features[:, 0]  # [D]
        feature_2 = stacked_text_features[:, 1]  # [D]
        
        # Calculate cosine similarity
        cosine_sim = F.cosine_similarity(feature_1.unsqueeze(0), feature_2.unsqueeze(0), dim=1)
        print(f"Cosine similarity between text features: {cosine_sim.item():.4f}")

    img_transform = Transform3DForM3DCLIP(target_height=256, target_width=256, target_depth=32)
    mask_transform = Transform3DMask(target_height=512, target_width=512, target_depth=args.depth)

    if args.depth == 1:
        stride = 1
    else:
        stride = 16
        
    if args.patient_seed_path is not None and os.path.exists(args.patient_seed_path):
        train_data = BraTS3DSubVolumeDataset(
            root=args.train_data_path,
            transform_img=img_transform, 
            transform_mask=mask_transform, 
            dataset_name=args.dataset, 
            mode ='train',
            stride_depth=stride, 
            sub_volume_depth=args.depth, 
            patient_seed_path=args.patient_seed_path,
            scales=args.fusion_scales
        )
    else:
        train_data = BraTS3DSubVolumeDataset(
            root=args.train_data_path, 
            transform_img=img_transform, 
            transform_mask=mask_transform, 
            dataset_name=args.dataset, 
            mode='train', 
            stride_depth=stride, 
            sub_volume_depth=args.depth,
            scales=args.fusion_scales
        )

    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

    val_data = BraTS3DSubVolumeDataset(
            root=args.val_data_path, 
            transform_img=img_transform, 
            transform_mask=mask_transform, 
            dataset_name=args.dataset, 
            mode='val',
            sub_volume_depth=args.depth,  # e.g. 32
            stride_depth=stride,  # Depth of each sub-volume
            scales=args.fusion_scales
        )
    val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=1, shuffle=False)


    feature_dim = model.config.hidden_size
    output_adapter_dim = model.config.hidden_size
    adapter_k = len(features_list)
    temperature_local = args.temperature
    
    
    trainable_layer =LinearLayer(
    dim_in=feature_dim,         # Each patch token has 'feature_dim' features
    dim_out=output_adapter_dim, # The adapter transforms it to this dimension
    k=adapter_k,
    model_type_str='vit_3d' # Just a string to trigger the ViT logic in LinearLayer
    ).to(device)
    
    if args.checkpoint_path is not None and os.path.exists(args.checkpoint_path):
        logger.info(f"Loading checkpoint from {args.checkpoint_path}")
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        trainable_layer.load_state_dict(checkpoint["trainable_layer"])

    # Add parameter logging right after logger setup
    logger.info("="*50)
    logger.info("TRAINING CONFIGURATION")
    logger.info("="*50)
    for arg in vars(args):
        logger.info(f'{arg}: {getattr(args, arg)}')
    
    logger.info("="*50)
    logger.info("MODEL CONFIGURATION")
    logger.info("="*50)
    logger.info(f'Device: {device}')
    logger.info(f'Feature dimensions: {feature_dim}')
    logger.info(f'Output adapter dimensions: {output_adapter_dim}')
    logger.info(f'Number of adapters: {adapter_k}')
    logger.info(f'Temperature: {temperature_local}')
    logger.info(f'Dataset size: {len(train_data)}')
    logger.info(f'Batches per epoch: {len(train_dataloader)}')
    
    # Log model parameters
    total_params = sum(p.numel() for p in trainable_layer.parameters())
    trainable_params = sum(p.numel() for p in trainable_layer.parameters() if p.requires_grad)
    logger.info(f'Total parameters: {total_params:,}')
    logger.info(f'Trainable parameters: {trainable_params:,}')
    
    logger.info("="*50)
    logger.info("STARTING TRAINING")
    logger.info("="*50)
    

    optimizer = torch.optim.Adam(list(trainable_layer.parameters()), lr=learning_rate, betas=(0.5, 0.999))

    loss_focal = FocalLoss()
    loss_dice = BinaryDiceLoss()
 
    for epoch in range(epochs):
        print(epoch)
        loss_list = []
        focal_loss_list = []
        dice_loss_list = []
        idx = 0
        for items in tqdm(train_dataloader):
            idx += 1
           
            volume = items['img'].to(device)
            
            
            #print("volume shape:", volume.shape) # (B, C, D, H, W)
            cls_name = items['cls_name']
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    all_image_token_features, attentions_all_layers, layer_list = model.encode_image(volume,features_list )
                    
                selected_vit_layer_patch_tokens = []
                for layer in layer_list:
                    layer = layer[:, 1:, :] # Exclude CLS token
                    selected_vit_layer_patch_tokens.append(layer)
                    
                adapted_multi_layer_patch_tokens_list = trainable_layer(selected_vit_layer_patch_tokens)
                layer_anomaly_maps_full_res = []
                upsampled_logits_list = []
                for i in range(adapter_k):
                    adapted_patch_features_this_layer = adapted_multi_layer_patch_tokens_list[i]
                    
                    # Normalize features from this adapted layer
                    adapted_patch_norm = F.normalize(adapted_patch_features_this_layer, dim=-1)
                    anomaly_map = ((1/temperature_local) * adapted_patch_norm @ stacked_text_features)
                    # Logits for each patch for this layer: (B, NumPatches, 2)
                    # Get dimensions for reshaping
                    B, L, C = anomaly_map.shape
                    #print("Anomaly map shape:", anomaly_map.shape) # (B, NumPatches, 2)
                    #print("Anomaly map min:", anomaly_map.min().item())
                    #print("Anomaly map max:", anomaly_map.max().item())
                    #print("Anomaly map mean:", anomaly_map.mean().item())
                    #print("Anomaly map std:", anomaly_map.std().item())
                    
                    # Reshape to 3D structure
                    num_patches_d = model.config.img_size[0] // model.config.patch_size[0] 
                    num_patches_h = model.config.img_size[1] // model.config.patch_size[1]
                    num_patches_w = model.config.img_size[2] // model.config.patch_size[2]
                    
                    # First permute to put text classes in front, then reshape
                    reshaped_logits = anomaly_map.permute(0, 2, 1).view(
                        B, C, num_patches_d, num_patches_h, num_patches_w
                    )
                    
                    # Interpolate to full resolution BEFORE softmax (like in test.py)
                    upsampled_logits = F.interpolate(
                        reshaped_logits,
                        size=(32,512,512),  # (32, 240, 240)
                        mode='trilinear',
                        align_corners=False
                    )
                    if args.depth == 1:
                        # If depth is 1, pick only central slice
                        upsampled_logits = upsampled_logits[:, :, 16, :, :]
                    probs = torch.softmax(upsampled_logits, dim=1)  # [B,2,D,H,W]
                    layer_anomaly_maps_full_res.append(probs)
                    upsampled_logits_list.append(upsampled_logits)

            gt = items['img_mask'].to(device) #(B,D, H, W)
            
            
            gt[gt > 0.5], gt[gt <= 0.5] = 1, 0
            loss = 0
            #print("gt shape:", gt.shape) # (B, D, H, W)
            #print("layer_anomaly_maps_full_res shape:", layer_anomaly_maps_full_res[0].shape) # (B, D, H, W)
            
            for probs, logits in zip(layer_anomaly_maps_full_res, upsampled_logits_list):

                pred_fg = probs[:, 1:2, ...]  # [B,1,D,H,W]
                
                loss += loss_focal(probs, gt)
                loss += loss_dice(pred_fg, gt)
                
    
            # print avg loss
            #print(f"Batch {idx}=> Pt_mean: {probs.mean().item():.4f}, Pt_max: {probs.max().item():.4f}, Pt_min: {probs.min().item():.4f}, Loss: {loss.item():.4f}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
            focal_loss_list.append(loss_focal(probs, gt).item())
            dice_loss_list.append(loss_dice(pred_fg, gt).item())
            
     

        avg_loss = np.mean(loss_list)
        avg_focal = np.mean(focal_loss_list)
        avg_dice = np.mean(dice_loss_list)
        
        print(f"Epoch {epoch+1}: avg loss: {avg_loss:.4f}")
        
        val_metrics = validation(model, trainable_layer, val_dataloader, stacked_text_features, device, args, save_path, adapter_k, scales=args.fusion_scales)
        
        #log metrics
        logger.info(f"EPOCH {epoch + 1} SUMMARY:")
        logger.info(f"  Training Losses:")
        logger.info(f"    Average Total Loss: {avg_loss:.4f}")
        logger.info(f"    Average Focal Loss: {avg_focal:.4f}")
        logger.info(f"    Average Dice Loss: {avg_dice:.4f}")
        logger.info(f"  Validation Metrics:")
        logger.info(f"    Dice Score: {val_metrics['val_dice_mean']:.4f} ± {val_metrics['val_dice_std']:.4f}")
        logger.info(f"    F1 Score: {val_metrics['val_f1_mean']:.4f} ± {val_metrics['val_f1_std']:.4f}")
        logger.info(f"    Precision: {val_metrics['val_precision_mean']:.4f} ± {val_metrics['val_precision_std']:.4f}")
        logger.info(f"    Recall: {val_metrics['val_recall_mean']:.4f} ± {val_metrics['val_recall_std']:.4f}")
        logger.info(f"    Best Threshold: {val_metrics['val_best_threshold']:.4f}")
        logger.info(f"    Patients processed: {val_metrics['val_samples']}")
        
        print(f"Epoch {epoch+1}: Train Loss: {avg_loss:.4f}, Val Dice: {val_metrics['val_dice_mean']:.4f}")
        
        # Save metrics to file for plotting later
        metrics_file = os.path.join(save_path, 'training_metrics.json')
        epoch_data = {
            'epoch': epoch + 1,
            'train_loss': avg_loss,
            'train_focal_loss': avg_focal,
            'train_dice_loss': avg_dice,
            **val_metrics
        }
        
        # Append to metrics file
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                all_metrics = json.load(f)
        else:
            all_metrics = []
        
        all_metrics.append(epoch_data)
        
        with open(metrics_file, 'w') as f:
            json.dump(all_metrics, f, indent=2)
    
        # save model
        if (epoch + 1) % args.save_freq == 0:
            ckp_path = os.path.join(save_path, 'epoch_' + str(epoch + 1) + '.pth')
            torch.save({'trainable_linearlayer': trainable_layer.state_dict()}, ckp_path)
            
    




def validation(model, linearlayer, val_dataloader, stacked_text_features, device, args, save_path, adapter_k=3, scales=[1]):
    """
    Validation function that processes full patients and returns metrics for logging
    """
    model.eval()
    linearlayer.eval()
    
    print("Starting validation inference...")
    
    acc = {}   # key = (patient_id, start_depth_idx) → list_of_maps
    S = len(scales)
    
    # Inference loop - save all sub-volumes first
    for idx, items in enumerate(tqdm(val_dataloader, desc="Processing sub-volumes")):
        volume = items['img'].to(device)  # [B, C, D, H, W]
        cls_name = items['cls_name'][0]
        gt_mask = items['img_mask']       # [B, D, H, W] or [B, C, D, H, W]
        
        start_depth_idx = items['start_depth_idx'][0].item()  # Get start depth index
        patient_id = items['patient_id'][0]
        scale_id = items['scale_id'].item()
        
        # Create save dir for each patient to save the different sub-volumes
        patient_save_path = os.path.join(f'{save_path}/validation', patient_id)
        os.makedirs(patient_save_path, exist_ok=True)
        
        full_shape = (32, 512, 512)               # D, H, W  (your upsampled size)
        key = (patient_id, start_depth_idx)
        ensure_key(acc, key, k=adapter_k, shape=full_shape)
        buf = acc[key]
        
        # save the reference volume and mask for visualization
        if buf["ref_vol"] is None and scale_id == 0:
            buf["ref_vol"]  = volume.squeeze(0).squeeze(0).cpu().numpy()
            buf["ref_mask"] = gt_mask.squeeze(0).numpy()
            
        
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
                temperature = getattr(args, 'temperature_scale', args.temperature)
                anomaly_map = ((1/temperature) * adapted_patch_norm @ stacked_text_features)
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
                
                # Take middle slice along depth for a 2D anomaly map
                if args.depth == 1:
                    upsampled_logits = upsampled_logits[:, :, 16, :, :]  # [B, C, H, W]
                    
                # Softmax and extract anomaly channel
                probs = torch.softmax(upsampled_logits, dim=1)  # [B,2,D,H,W] or [B,2,H,W]
                anomaly_map_layer = probs[:, 1]  # [B,D,H,W] or [B,H,W] - anomaly channel only
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
            final_anomaly_map = final_anomaly_map.squeeze(0).cpu().numpy()  # [D,H,W] or [H,W]
            
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


                del acc[key]      # free RAM
               
               
            
            # Save the anomaly map for this patient
            #final_anomaly_map_path = os.path.join(patient_save_path, f'anomaly_map_depth_{start_depth_idx}.npy')
            #np.save(final_anomaly_map_path, final_anomaly_map)

    # Now reconstruct full volumes and compute metrics
    print("Reconstructing full volumes and computing metrics...")
    patient_dirs = sorted(glob(os.path.join(f'{save_path}/validation/*')))
    n_patients = len(patient_dirs)
    
    if n_patients == 0:
        print("No patients found for validation!")
        return {
            'val_dice_mean': 0.0,
            'val_dice_std': 0.0,
            'val_f1_mean': 0.0,
            'val_f1_std': 0.0,
            'val_precision_mean': 0.0,
            'val_precision_std': 0.0,
            'val_recall_mean': 0.0,
            'val_recall_std': 0.0,
            'val_samples': 0,
            'val_best_threshold': 0.0
        }

    all_preds = []
    all_gts = []

    # Reconstruct volumes for each patient
    for patient_dir in patient_dirs:
        patient_id = os.path.basename(patient_dir)

        try:
            if args.depth == 1:
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
            
            # Load full GT mask
            gt_root = getattr(args, 'gt_root', args.val_data_path)
            gt = load_gt_mask(patient_id, gt_root)
            all_gts.append(gt)
            
        except Exception as e:
            print(f"Error processing patient {patient_id}: {e}")
            continue

    if len(all_preds) == 0:
        print("No valid predictions reconstructed!")
        return {
            'val_dice_mean': 0.0,
            'val_dice_std': 0.0,
            'val_f1_mean': 0.0,
            'val_f1_std': 0.0,
            'val_precision_mean': 0.0,
            'val_precision_std': 0.0,
            'val_recall_mean': 0.0,
            'val_recall_std': 0.0,
            'val_samples': 0,
            'val_best_threshold': 0.0
        }

    # Threshold sweep to find best threshold
    k = adapter_k  # Use the actual adapter_k value
    thresholds = np.linspace(0, k, 21)  # Reduced from 51 to 21 for faster validation

    # Accumulate Dice scores per threshold
    dice_per_threshold = np.zeros(len(thresholds), dtype=np.float64)
    
    # Also compute F1, precision, recall per threshold
    f1_per_threshold = np.zeros(len(thresholds), dtype=np.float64)
    precision_per_threshold = np.zeros(len(thresholds), dtype=np.float64)
    recall_per_threshold = np.zeros(len(thresholds), dtype=np.float64)
    
    eps = 1e-8
    
    for pred, gt in zip(all_preds, all_gts):
        for i, thr in enumerate(thresholds):
            # Create binary mask
            binary_pred = (pred > thr).astype(np.uint8)
            
            # Compute Dice
            dice_score = dice_coefficient_3d(binary_pred, gt)
            dice_per_threshold[i] += dice_score
            
            # Compute precision, recall, F1
            pred_flat = binary_pred.flatten()
            gt_flat = gt.flatten().astype(np.uint8)
            
            tp = np.sum((pred_flat == 1) & (gt_flat == 1))
            fp = np.sum((pred_flat == 1) & (gt_flat == 0))
            fn = np.sum((pred_flat == 0) & (gt_flat == 1))
            
            precision = tp / (tp + fp + eps)
            recall = tp / (tp + fn + eps)
            f1 = 2 * precision * recall / (precision + recall + eps)
            
            precision_per_threshold[i] += precision
            recall_per_threshold[i] += recall
            f1_per_threshold[i] += f1

    # Average over all patients
    n_valid_patients = len(all_preds)
    dice_per_threshold /= n_valid_patients
    f1_per_threshold /= n_valid_patients
    precision_per_threshold /= n_valid_patients
    recall_per_threshold /= n_valid_patients

    # Find best threshold based on Dice score
    best_idx = np.argmax(dice_per_threshold)
    best_thr = thresholds[best_idx]
    best_dice = dice_per_threshold[best_idx]
    best_f1 = f1_per_threshold[best_idx]
    best_precision = precision_per_threshold[best_idx]
    best_recall = recall_per_threshold[best_idx]

    print(f"Validation - Best threshold: {best_thr:.3f} → Dice: {best_dice:.4f}, F1: {best_f1:.4f}")

    # Compute per-patient metrics at best threshold for std calculation
    patient_dice_scores = []
    patient_f1_scores = []
    patient_precision_scores = []
    patient_recall_scores = []
    
    for pred, gt in zip(all_preds, all_gts):
        binary_pred = (pred > best_thr).astype(np.uint8)
        
        # Dice
        dice_score = dice_coefficient_3d(binary_pred, gt)
        patient_dice_scores.append(dice_score)
        
        # F1, precision, recall
        pred_flat = binary_pred.flatten()
        gt_flat = gt.flatten().astype(np.uint8)
        
        tp = np.sum((pred_flat == 1) & (gt_flat == 1))
        fp = np.sum((pred_flat == 1) & (gt_flat == 0))
        fn = np.sum((pred_flat == 0) & (gt_flat == 1))
        
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        
        patient_precision_scores.append(precision)
        patient_recall_scores.append(recall)
        patient_f1_scores.append(f1)

    # Prepare metrics dictionary
    metrics = {
        'val_dice_mean': float(np.mean(patient_dice_scores)),
        'val_dice_std': float(np.std(patient_dice_scores)),
        'val_f1_mean': float(np.mean(patient_f1_scores)),
        'val_f1_std': float(np.std(patient_f1_scores)),
        'val_precision_mean': float(np.mean(patient_precision_scores)),
        'val_precision_std': float(np.std(patient_precision_scores)),
        'val_recall_mean': float(np.mean(patient_recall_scores)),
        'val_recall_std': float(np.std(patient_recall_scores)),
        'val_samples': n_valid_patients,
        'val_best_threshold': float(best_thr)
    }
    
    # Set models back to train mode
    model.train()
    linearlayer.train()
    
    return metrics
    
def arg_parser_with_config():
    """
    Parse command line arguments and load configuration from a YAML file.
    Returns:
        argparse.Namespace: Parsed arguments with configuration values.
    """
    parser = argparse.ArgumentParser("VAND3D Training", add_help=True)
    # config file
    parser.add_argument("--config", type=str, help="YAML config file")
    parser.add_argument("--train_data_path", type=str, default="/home/nicoc/data/BraTS_3D", help="train dataset path")
    parser.add_argument("--val_data_path", type=str, default="/home/nicoc/data/BraTS_3D", help="validation dataset path")
    parser.add_argument("--save_path", type=str, default='./experiments_3D/prompt_ensemble', help='path to save results')
    parser.add_argument("--config_path", type=str, default='./open_clip/model_configs/ViT-B-16.json', help="model configs")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="pretrained weight path")
    parser.add_argument("--patient_seed_path", type=str, default=None, help="seed for k patient in train set, if None, use all patients")
    # prompt ensemble
    parser.add_argument("--prompt_ensemble", type=bool, default=True, help="use prompt ensemble or not")
    parser.add_argument("--prompt_path", type=str, default='./configs/prompts_brats.yaml', help="prompt ensemble path")
    
     # fusion
    parser.add_argument("--fusion", type=bool, default=False, help="Use fusion, default False")
    parser.add_argument("--fusion_scales", type=list, default=[1], help="Scales to use for fusion, default [1]")
    
    # model
    parser.add_argument("--dataset", type=str, default='brats', help="train dataset name")
    parser.add_argument("--model", type=str, default="ViT-B-16", help="model used")
    parser.add_argument("--pretrained", type=str, default="laion400m_e32", help="pretrained weight used")
    parser.add_argument("--features_list", type=int, nargs="+", default=[3, 4, 11], help="features used")
    parser.add_argument("--finetuning", action='store_true', help="finetune industrial model")
    # hyper-parameter
    parser.add_argument("--epoch", type=int, default=20, help="epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--depth", type=int, default=32, help="depth of 3D volume")
    parser.add_argument("--image_size", type=int, default=256, help="image size")
    parser.add_argument("--aug_rate", type=float, default=0.2, help="image size")
    parser.add_argument("--print_freq", type=int, default=30, help="print frequency")
    parser.add_argument("--save_freq", type=int, default=1, help="save frequency")
    parser.add_argument("--temperature", type=float, default=0.2, help="temperature for softmax")
    
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

    setup_seed(111)
    train(args)

