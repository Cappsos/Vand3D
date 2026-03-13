
import numpy as np
import os
import json
from datetime import datetime
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)



def load_gt_mask(patient_id, gt_root):
    """
    Given a patient_id, find and load the corresponding '*.npy' 
    ground truth mask, binarize it, and return a (D, H, W) array.
    This version is more flexible and doesn't hardcode shapes.
    """
    pat_dir = os.path.join(gt_root, patient_id)
    try:
        seg_files = [f for f in os.listdir(pat_dir) if f.endswith('.npy')]
        if len(seg_files) != 1:
            raise FileNotFoundError(
                f"Expected one '*.npy' in {pat_dir}, found {len(seg_files)}"
            )
        seg_path = os.path.join(pat_dir, seg_files[0])
        seg = np.load(seg_path)
    except FileNotFoundError as e:
        logger.error(e)
        return None
    except Exception as e:
        logger.error(f"Could not load GT for {patient_id}: {e}")
        return None

    # Ensure depth-first (D, H, W)
    if seg.shape[1] != seg.shape[2]: # Simple check for HxWxD vs DxHxW
        if seg.shape[0] == seg.shape[1]:
             # This is likely HxWxD, so transpose
            seg = np.transpose(seg, (2, 0, 1))

    logger.info(f"GT loaded: {patient_id}, shape={seg.shape}")
    return (seg > 0).astype(np.uint8)


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
        logger.warning(
            f"Brain volume shape {brain_vol.shape} for {patient_id}, attempting to fix..."
        )
        if brain_vol.shape == (240, 240, 155):
            brain_vol = np.transpose(brain_vol, (2, 0, 1))  # (H,W,D) -> (D,H,W)
        elif brain_vol.shape == (240, 155, 240):
            brain_vol = np.transpose(brain_vol, (1, 0, 2))  # (H,D,W) -> (D,H,W)
        else:
            raise ValueError(f"Unexpected brain volume shape {brain_vol.shape} for {patient_id}")
    
    logger.info(f"Brain volume loaded: {patient_id}, shape={brain_vol.shape}")
    return brain_vol.astype(np.float32)

def load_evaluation_results(results_dir: str, filename: str = 'best_threshold.json') -> Dict[str, Any]:
    """
    Load evaluation results from JSON file.
    
    Args:
        results_dir: Directory containing the JSON file
        filename: Name of the JSON file (default: 'best_threshold.json')
        
    Returns:
        Dictionary with all evaluation results and metadata
        
    Raises:
        FileNotFoundError: If the JSON file doesn't exist
        json.JSONDecodeError: If the JSON file is corrupted
    """
    filepath = os.path.join(results_dir, filename)
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Evaluation results not found: {filepath}")
    
    try:
        with open(filepath, 'r') as f:
            results = json.load(f)
        
        logger.info(f"Loaded evaluation results from: {filepath}")
        return results
    
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Corrupted JSON file {filepath}: {e}")


def get_best_threshold(results_dir: str) -> float:
    """
    Quick function to get just the best threshold value.
    
    Args:
        results_dir: Directory containing evaluation results
        
    Returns:
        Best threshold value as float
    """
    results = load_evaluation_results(results_dir)
    return results['best_threshold']


def save_evaluation_results(
    save_path: str, 
    best_threshold: float, 
    best_dice: float, 
    all_thresholds: np.ndarray, 
    all_dice_scores: np.ndarray,
    args: Any,
    n_patients: int,
    additional_metadata: Optional[Dict] = None
) -> str:
    """
    Save evaluation results to JSON files with comprehensive metadata.
    
    Args:
        save_path: Directory to save results
        best_threshold: Optimal threshold value
        best_dice: Best Dice score achieved
        all_thresholds: Array of all tested thresholds
        all_dice_scores: Array of corresponding Dice scores
        args: Arguments object from training/evaluation
        n_patients: Number of patients evaluated
        additional_metadata: Optional extra metadata to include
        
    Returns:
        Path to the saved threshold file
    """
    os.makedirs(save_path, exist_ok=True)
    
    # Main results with metadata
    results = {
        'evaluation_results': {
            'best_threshold': float(best_threshold),
            'best_dice': float(best_dice),
            'total_patients': n_patients,
            'threshold_range': [float(all_thresholds[0]), float(all_thresholds[-1])],
            'num_thresholds_tested': len(all_thresholds)
        },
        'model_config': {
            'checkpoint_path': getattr(args, 'checkpoint_path', None),
            'dataset': getattr(args, 'dataset', None),
            'depth': getattr(args, 'depth', None),
            'temperature_scale': getattr(args, 'temperature_scale', None),
            'features_list': getattr(args, 'features_list', None),
            'batch_size': getattr(args, 'batch_size', None)
        },
        'metadata': {
            'evaluation_date': datetime.now().isoformat(),
            'script_version': '1.0',
            'save_path': save_path,
            'git_commit': None  # You can add this later
        }
    }
    
    # Add any additional metadata
    if additional_metadata:
        results['metadata'].update(additional_metadata)
    
    # Save main results
    threshold_file = os.path.join(save_path, 'best_threshold.json')
    with open(threshold_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save detailed curve data
    detailed_results = {
        'thresholds': all_thresholds.tolist(),
        'dice_scores': all_dice_scores.tolist(),
        'threshold_dice_pairs': [
            {'threshold': float(t), 'dice': float(d)} 
            for t, d in zip(all_thresholds, all_dice_scores)
        ],
        'statistics': {
            'max_dice': float(all_dice_scores.max()),
            'min_dice': float(all_dice_scores.min()),
            'mean_dice': float(all_dice_scores.mean()),
            'std_dice': float(all_dice_scores.std())
        }
    }
    
    curve_file = os.path.join(save_path, 'threshold_curve.json')
    with open(curve_file, 'w') as f:
        json.dump(detailed_results, f, indent=2)
    
    logger.info(f"Evaluation results saved to: {threshold_file}")
    logger.info(f"Detailed curve saved to: {curve_file}")
    
    return threshold_file