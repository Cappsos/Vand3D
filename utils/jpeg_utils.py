# utils/jpeg_utils.py
import os
import numpy as np
from skimage.io import imread
from skimage.transform import resize
import logging

logger = logging.getLogger(__name__)

def reconstruct_volume_from_jpegs(
    patient_dir: str,
    full_depth: int = 155,
    target_h: int = 240,
    target_w: int = 240,
    fill_value: float = 0.0
) -> np.ndarray:
    """
    Reads per-slice JPEG anomaly maps from patient_dir, stacks them into
    a (full_depth, H, W) float32 array in [0,1], then resizes to (full_depth, target_h, target_w).

    Assumes file names: "{prefix}{z}.jpeg" where z in [0, full_depth).

    Returns:
        volume: np.ndarray shape (full_depth, target_h, target_w), dtype float32
    """
    # 1) Initialize an empty float volume at the slice resolution
    example_jpeg = os.path.join(patient_dir, f"000.jpeg")
    if not os.path.isfile(example_jpeg):
        raise FileNotFoundError(f"No JPEG with prefix '000' found in {patient_dir}")

    # Read one slice just to get its H, W
    sample = imread(example_jpeg)    # shape (H1, W1) or (H1, W1, 3)
    

    H1, W1 = sample.shape
    vol_raw = np.full((full_depth, H1, W1), fill_value, dtype=np.float32)
    loaded = []

    # 2) Loop over slices 0..(full_depth-1)
    for z in range(full_depth):
        
        jp = os.path.join(patient_dir, f"{z:03d}.jpeg")
        if not os.path.isfile(jp):
            # leave fill_value
            continue
        im = imread(jp)
        logger.debug("im max: %s, min: %s", im.max(), im.min())
        if im.ndim == 3:
            # convert to grayscale [0,255] → [0,1]
            im = np.mean(im, axis=2)
        im = im.astype(np.float32) / 255.0
        # If original JPEG resolution (H1, W1) ≠ (target_h, target_w), we will resize later
        vol_raw[z] = im
        loaded.append(z)

    if not loaded:
        raise RuntimeError(f"No JPEG slices loaded for patient {os.path.basename(patient_dir)}")

    # 3) Resize entire volume from (full_depth, H1, W1) → (full_depth, target_h, target_w)
    if (H1, W1) != (target_h, target_w):
        vol_resized = np.zeros((full_depth, target_h, target_w), dtype=np.float32)
        for z in range(full_depth):
            vol_resized[z] = resize(
                vol_raw[z],
                (target_h, target_w),
                order=1,            # bilinear
                mode='reflect',
                anti_aliasing=True,
                preserve_range=True
            )
        return vol_resized.astype(np.float32)
    else:
        return vol_raw
