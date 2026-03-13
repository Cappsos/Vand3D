from scipy.ndimage import zoom
import numpy as np

def laplacian_fuse(gaussians, orig_size):
    """gaussians: list of np.ndarray D×H_s×W_s, fine→coarse."""
    laps = []
    for i in range(len(gaussians)-1):
        up = zoom(gaussians[i+1], zoom=[1,
              gaussians[i].shape[1]/gaussians[i+1].shape[1],
              gaussians[i].shape[2]/gaussians[i+1].shape[2]],
              order=1)                         # trilinear
        laps.append(gaussians[i] - up)
    laps.append(gaussians[-1])
    fused = np.zeros(orig_size, np.float32)
    for l in laps:
        fused += zoom(l, zoom=[1,
                 orig_size[1]/l.shape[1],
                 orig_size[2]/l.shape[2]], order=1)
    return fused



def paste_and_fuse(views, boxes, full_shape, mode="mean"):
    """
    Paste a list of anomaly maps back into a full canvas.

    Args
    ----
    views   : list[np.ndarray]   each (D, h_i, w_i)
    boxes   : list[tuple]        (y0, x0, h_i, w_i) in *full* coord-frame
    full_shape : tuple           (D_full, H_full, W_full) of target volume
    mode    : "mean" | "max"     fusion rule inside overlapping regions
    """
    canvas = np.zeros(full_shape, np.float32)
    counter = np.zeros(full_shape, np.uint8)

    for m, (y0, x0, h, w) in zip(views, boxes):
        canvas[:, y0:y0+h, x0:x0+w] += m
        counter[:, y0:y0+h, x0:x0+w] += 1

    if mode == "mean":
        fused = canvas / np.maximum(counter, 1)
    elif mode == "max":
        fused = np.where(counter, canvas, 0)      # remove never-covered voxels
        # where multiple maps overlap keep the max
        fused = np.maximum.reduce([
            fused, *(np.where(counter == k, canvas/k, 0)
                     for k in range(2, counter.max()+1))
        ])
    else:
        raise ValueError("mode must be 'mean' or 'max'")
    return fused

def ensure_key(acc, key, *, k, shape):
    """
    k         number of ViT layers you keep (len(args.features_list))
    shape     full canvas size, e.g. (32,240,240)
    """
    if key in acc:
        return

    acc[key] = {
        "layer_views":  [ [] for _ in range(k) ],   # views per layer
        "layer_boxes":  [ [] for _ in range(k) ],   # (y0,x0,h,w) per view
        "ref_vol":      None,                       # for plotting
        "ref_mask":     None
    }