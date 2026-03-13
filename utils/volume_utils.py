import numpy as np
import os
import re
from skimage.transform import resize  # or cv2.resize
import logging

logger = logging.getLogger(__name__)


def reconstruct_volume_from_slices(
    patient_dir: str,
    prefix: str = "anomaly_map_depth_",
    full_depth: int = 155,
    target_h: int = 240,
    target_w: int = 240,
    fill_value: float = 0.0
) -> np.ndarray:
    """
    Reconstruct a full 3D volume from individual 2D slice files.
    
    Args:
        patient_dir: Directory containing the slice files for one patient
        prefix: Filename prefix before the depth index (e.g., "anomaly_map_depth_")
        full_depth: Total depth of the reconstructed volume (e.g., 155)
        target_h: Target height of the volume (e.g., 240)
        target_w: Target width of the volume (e.g., 240)
        fill_value: Value to fill missing slices (default: 0.0)
    
    Returns:
        volume: np.ndarray of shape (full_depth, target_h, target_w)
        
    Example:
        volume = reconstruct_volume_from_slices(
            "/path/to/patient/BraTS-MET-00003-000",
            prefix="anomaly_map_depth_",
            full_depth=155
        )
    """
    from glob import glob
    
    # Find all slice files matching the pattern
    slice_files = sorted(glob(os.path.join(patient_dir, f'{prefix}*.npy')))
    
    if not slice_files:
        raise FileNotFoundError(f"No slice files found in {patient_dir} with prefix '{prefix}'")
    
    # Initialize output volume
    volume = np.full((full_depth, target_h, target_w), fill_value, dtype=np.float32)
    
    # Track which slices were loaded
    loaded_indices = []
    
    # Load and place each slice
    for slice_file in slice_files:
        try:
            # Extract depth index from filename
            filename = os.path.basename(slice_file)
            depth_idx = int(filename.replace(prefix, '').replace('.npy', ''))
            
            # Validate depth index
            if depth_idx < 0 or depth_idx >= full_depth:
                logger.warning(
                    f"Depth index {depth_idx} out of bounds [0, {full_depth}), skipping {filename}"
                )
                continue
            
            # Load slice data
            slice_data = np.load(slice_file)
            
            # Handle different slice shapes
            if slice_data.ndim == 2:
                # Already 2D: (H, W)
                h, w = slice_data.shape
            elif slice_data.ndim == 3 and slice_data.shape[0] == 1:
                # 3D with depth=1: (1, H, W)
                slice_data = slice_data.squeeze(0)
                h, w = slice_data.shape
            else:
                logger.warning(
                    f"Unexpected slice shape {slice_data.shape} in {filename}, skipping"
                )
                continue
            
            # Resize slice if needed
            if (h, w) != (target_h, target_w):
                slice_data = resize(
                    slice_data,
                    (target_h, target_w),
                    order=1,              # bilinear interpolation
                    mode='reflect',       # boundary handling
                    anti_aliasing=True,   # reduce aliasing
                    preserve_range=True   # keep original data range
                )
            
            # Place slice in volume
            volume[depth_idx] = slice_data.astype(np.float32)
            loaded_indices.append(depth_idx)
            
        except (ValueError, IOError) as e:
            logger.error(f"Error processing {slice_file}: {e}")
            continue
    
    # Report reconstruction summary
    loaded_indices = sorted(loaded_indices)
    missing_slices = set(range(full_depth)) - set(loaded_indices)
    
    logger.info(f"Reconstructed volume: shape={volume.shape}")
    logger.info(f"  Loaded {len(loaded_indices)}/{full_depth} slices")
    logger.info(f"  Range: [{volume.min():.4f}, {volume.max():.4f}]")
    if missing_slices:
        missing_ranges = _get_consecutive_ranges(sorted(missing_slices))
        logger.info(f"  Missing slices: {missing_ranges}")
    
    return volume


def _get_consecutive_ranges(indices):
    """Helper function to format consecutive missing indices as ranges."""
    if not indices:
        return "None"
    
    ranges = []
    start = indices[0]
    end = start
    
    for i in indices[1:]:
        if i == end + 1:
            end = i
        else:
            if start == end:
                ranges.append(str(start))
            else:
                ranges.append(f"{start}-{end}")
            start = end = i
    
    # Add the last range
    if start == end:
        ranges.append(str(start))
    else:
        ranges.append(f"{start}-{end}")
    
    return ", ".join(ranges)


def batch_reconstruct_volumes(
    patients_root_dir: str,
    prefix: str = "anomaly_map_depth_",
    full_depth: int = 155,
    target_h: int = 240,
    target_w: int = 240
) -> dict:
    """
    Reconstruct volumes for multiple patients in batch.
    
    Args:
        patients_root_dir: Root directory containing patient subdirectories
        prefix: Filename prefix for slice files
        full_depth: Target depth for reconstructed volumes
        target_h: Target height
        target_w: Target width
    
    Returns:
        dict: {patient_id: reconstructed_volume}
    """
    from glob import glob
    
    patient_dirs = sorted(glob(os.path.join(patients_root_dir, '*')))
    reconstructed_volumes = {}
    
    logger.info(f"Batch reconstructing volumes for {len(patient_dirs)} patients...")
    
    for patient_dir in patient_dirs:
        if not os.path.isdir(patient_dir):
            continue
            
        patient_id = os.path.basename(patient_dir)
        
        try:
            volume = reconstruct_volume_from_slices(
                patient_dir,
                prefix=prefix,
                full_depth=full_depth,
                target_h=target_h,
                target_w=target_w
            )
            reconstructed_volumes[patient_id] = volume
            logger.info(f"✓ {patient_id}: Successfully reconstructed")
            
        except Exception as e:
            logger.error(f"✗ {patient_id}: Failed to reconstruct - {e}")
            continue
    
    logger.info(
        f"Batch reconstruction complete: {len(reconstructed_volumes)}/{len(patient_dirs)} successful"
    )
    return reconstructed_volumes
  
def combine_subvolumes_from_folder(
    patient_dir: str,
    prefix: str = "anomaly_map_brain_depth_",
    full_depth: int = 155,
    block_depth: int = 32
) -> np.ndarray:
    """
    Combine overlapping subvolume anomaly maps saved as:
      anomaly_map_brain_depth_{start}.npy
    into a full-volume map of shape (full_depth, H, W) by averaging overlaps.

    Args:
      patient_dir: path containing .npy files for one patient
      prefix: filename prefix before the depth index
      full_depth: total number of slices in the original volume (e.g. 155)
      block_depth: depth of each subvolume (e.g. 32)

    Returns:
      combined: np.ndarray of shape (full_depth, H, W)
    """
    # discover subvolume files and parse start indices
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)\.npy$")
    entries = []
    for fname in os.listdir(patient_dir):
        m = pattern.match(fname)
        if m:
            start = int(m.group(1))
            entries.append((start, os.path.join(patient_dir, fname)))
    if not entries:
        raise ValueError(f"No subvolume files found in {patient_dir} with prefix {prefix}")

    # load the first block to get dimensions
    entries = sorted(entries, key=lambda x: x[0])
    first_start, first_path = entries[0]
    block = np.load(first_path)
    if len(block.shape) == 2:
      D, H, W = 1, block.shape[0], block.shape[1]
    elif len(block.shape) == 3:
      D, H, W = block.shape
      
      
    else:
        raise ValueError(f"Unexpected shape {block.shape} for subvolume file {first_path}")

    # prepare accumulators
    sum_map = np.zeros((full_depth, H, W), dtype=np.float32)
    count_map = np.zeros((full_depth, H, W), dtype=np.uint16)

    # accumulate values and counts
    for start, path in entries:
        block = np.load(path)
        end = min(start + block_depth, full_depth)
        length = end - start
        sum_map[start:end] += block[:length]
        count_map[start:end] += 1

    # average where count > 0
    mask = count_map > 0
    combined = np.zeros_like(sum_map)
    combined[mask] = sum_map[mask] / count_map[mask]
    return combined


def resize_volume(
    volume: np.ndarray,
    target_h: int = 240,
    target_w: int = 240
) -> np.ndarray:
    """
    Resize each slice of a (D,H,W) volume to (D,target_h,target_w).

    Args:
      volume: np.ndarray with shape (D, H, W)
      target_h: desired slice height
      target_w: desired slice width

    Returns:
      resized: np.ndarray with shape (D, target_h, target_w)
    """
    D, H, W = volume.shape
    resized = np.zeros((D, target_h, target_w), dtype=volume.dtype)
    for i in range(D):
        resized[i] = resize(
            volume[i],
            (target_h, target_w),
            order=1,
            mode='reflect',
            anti_aliasing=True
        )
    return resized
