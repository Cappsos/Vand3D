# metrics.py
import numpy as np
import json
from scipy.ndimage import label as cc_label
from scipy.spatial.distance import directed_hausdorff
from sklearn.metrics import roc_auc_score, precision_recall_curve

def dice_coefficient_3d(pred: np.ndarray, gt: np.ndarray, smooth: float = 1e-6) -> float:
    """
    3D Dice coefficient between binary pred and gt volumes.
    """
    p, g = pred.flatten(), gt.flatten()
    intersection = (p & g).sum()
    return (2.0 * intersection + smooth) / (p.sum() + g.sum() + smooth)

def hausdorff_distance_3d(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Approximates the Hausdorff distance between binary pred and gt surfaces.
    """
    coords_p = np.argwhere(pred)
    coords_g = np.argwhere(gt)
    if coords_p.size == 0 or coords_g.size == 0:
        return np.nan
    d1 = directed_hausdorff(coords_p, coords_g)[0]
    d2 = directed_hausdorff(coords_g, coords_p)[0]
    return max(d1, d2)

def roc_auc_3d(score_map: np.ndarray, gt: np.ndarray) -> float:
    """
    Volume‐wise ROC AUC on raw anomaly scores.
    """
    flat_s = score_map.ravel()
    flat_g = gt.ravel()
    try:
        return roc_auc_score(flat_g, flat_s)
    except ValueError:
        return np.nan

def average_precision_3d(score_map: np.ndarray, gt: np.ndarray) -> float:
    """
    Volume‐wise average precision (area under PR curve).
    """
    flat_s = score_map.ravel()
    flat_g = gt.ravel()
    prec, rec, _ = precision_recall_curve(flat_g, flat_s)
    return np.trapz(rec, prec)

def f1_max_3d(score_map: np.ndarray, gt: np.ndarray, thresholds=None) -> float:
    """
    Max F1 score over a grid of thresholds in [0,1].
    """
    if thresholds is None:
        thresholds = np.linspace(0, 1, 51)
    flat_s = score_map.ravel()
    flat_g = gt.ravel().astype(bool)
    best_f1 = 0.0
    for thr in thresholds:
        p = flat_s > thr
        tp = np.logical_and(p, flat_g).sum()
        fp = np.logical_and(p, ~flat_g).sum()
        fn = np.logical_and(~p, flat_g).sum()
        f1 = 2 * tp / (2*tp + fp + fn + 1e-6)
        if f1 > best_f1:
            best_f1 = f1
    return best_f1

def load_threshold_from_json(json_path: str) -> float:
    """
    Reads a JSON file and returns the first recognized threshold key.
    """
    with open(json_path) as f:
        data = json.load(f)
    for key in ("best_threshold", "threshold", "best_thr"):
        if key in data:
            return float(data[key])
    # if JSON is just a number:
    if isinstance(data, (int, float)):
        return float(data)
    raise KeyError(f"No threshold key found in {json_path}")


def iou3d(pred, gt):
    inter = (pred & gt).sum()
    union = (pred | gt).sum()
    return inter / (union + 1e-8)