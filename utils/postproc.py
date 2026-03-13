import numpy as np
from scipy.ndimage import gaussian_filter, binary_propagation, label, binary_dilation
from skimage.morphology import remove_small_objects
from typing import Tuple, Dict, Sequence

__all__ = [
    "zscore",
    "fuse_layers",
    "PostProcessor",
]

# ----------------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------------

def zscore(score: np.ndarray, brain_mask: np.ndarray | None = None) -> np.ndarray:
    """Robust z‑normalisation of a 3‑D anomaly map.

    Parameters
    ----------
    score : np.ndarray
        Raw anomaly map (D × H × W).
    brain_mask : np.ndarray | None, optional
        Boolean mask of voxels that belong to brain tissue.  If *None* a quick
        intensity threshold on the *score* itself is used.

    Returns
    -------
    np.ndarray
        Z‑normalised map with mean 0 and std 1 inside *brain_mask*.
    """
    if brain_mask is None:
        # crude surrogate – treat anything > median intensity inside score as brain
        thresh = np.median(score)
        brain_mask = score > thresh
        if brain_mask.sum() < 100:  # fallback in degenerate cases
            brain_mask = np.ones_like(score, dtype=bool)

    mu = score[brain_mask].mean()
    sigma = score[brain_mask].std() + 1e-8  # avoid /0
    score_z = (score - mu) / sigma

    # clamp extreme outliers (optional, stabilises grid search)
    return np.clip(score_z, -3.0, 8.0)


def fuse_layers(layer_maps: Sequence[np.ndarray], weights: Sequence[float] | None = None) -> np.ndarray:
    """Fuse anomaly maps from different ViT layers by a weighted sum.

    Parameters
    ----------
    layer_maps : list of np.ndarray
        Each element shape = D × H × W.
    weights : list[float] | None
        Same length as *layer_maps*.  If *None*, all maps are averaged.
    """
    layer_maps = [np.asarray(m, dtype=np.float32) for m in layer_maps]
    if weights is None:
        weights = np.ones(len(layer_maps), dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float32)
    weights /= (weights.sum() + 1e-8)

    fused = np.zeros_like(layer_maps[0], dtype=np.float32)
    for w, m in zip(weights, layer_maps):
        fused += w * m
    return fused

# ----------------------------------------------------------------------------
# Main post‑processing class
# ----------------------------------------------------------------------------

class PostProcessor:
    """Inference‑time clean‑up for 3‑D anomaly maps.

    The pipeline implements the steps described in our ChatGPT discussion:

    1. z‑normalisation inside a brain mask
    2. Gaussian smoothing (σ ≈ 1 voxel)
    3. Dual‑threshold region growing (high → seeds, low → mask)
    4. Small‑object removal + (optional) keep largest component

    All parameters are exposed so you can grid‑search them on a validation set.
    """

    def __init__(
        self,
        high: float = 3.5,
        low: float = 2.0,
        gaussian_sigma: float = 1.0,
        min_component: int = 200,
        keep_largest: bool = False,
    ) -> None:
        assert high > low, "'high' threshold must be > 'low' threshold"
        self.high = high
        self.low = low
        self.gaussian_sigma = gaussian_sigma
        self.min_component = min_component
        self.keep_largest = keep_largest

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def __call__(
        self,
        raw_map: np.ndarray,
        volume_for_mask: np.ndarray | None = None,
        *,
        return_intermediate: bool = False,
    ) -> np.ndarray | Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Run the post‑processing pipeline.

        Parameters
        ----------
        raw_map : np.ndarray
            Raw anomaly map (D × H × W) – *not* normalised.
        volume_for_mask : np.ndarray | None, optional
            Original T1/FLAIR/T2 volume used to derive a brain mask.  If *None*
            the brain mask will be estimated from *raw_map*.
        return_intermediate : bool, default False
            If *True* returns a tuple *(pred_mask, cache)* where *cache* holds
            intermediate arrays useful for debugging/visualisation.
        """

        # 1) Brain mask ( quick surrogate )
        brain_mask = None
        if volume_for_mask is not None:
            # use Otsu‑like threshold on the original anatomy
            thresh = np.median(volume_for_mask) + 0.1 * volume_for_mask.std()
            brain_mask = volume_for_mask > thresh

        # 2) Z‑normalise
        score_z = zscore(raw_map, brain_mask)

        # 3) Gaussian smoothing (optional)
        if self.gaussian_sigma is not None and self.gaussian_sigma > 0:
            score_z = gaussian_filter(score_z, sigma=self.gaussian_sigma)

        # 4) Dual‑threshold region growing
        seed_core = score_z > self.high
        seed_lo = score_z > self.low
        grown = binary_propagation(seed_core, mask=seed_lo)
        pred_mask = grown.astype(bool)

        # 5) Small object removal
        if self.min_component and self.min_component > 0:
            pred_mask = remove_small_objects(pred_mask, min_size=self.min_component)

        # 6) Keep only the largest component if requested
        if self.keep_largest and pred_mask.any():
            lbl, num = label(pred_mask)
            if num > 1:
                counts = np.bincount(lbl.flatten())
                counts[0] = 0  # background
                largest = counts.argmax()
                pred_mask = lbl == largest

        if return_intermediate:
            cache = {
                "score_z": score_z,
                "seed_core": seed_core,
                "seed_low": seed_lo,
            }
            return pred_mask.astype(np.uint8), cache
        else:
            return pred_mask.astype(np.uint8)

    # ------------------------------------------------------------------
    # Static utilities for quick grid‑search over a validation set
    # ------------------------------------------------------------------

    @staticmethod
    def grid_search(
        maps: Sequence[np.ndarray],
        gts: Sequence[np.ndarray],
        highs: Sequence[float] = (3.0, 3.5, 4.0),
        lows: Sequence[float] = (1.5, 2.0, 2.5),
        sigmas: Sequence[float] = (0.5, 1.0, 1.5),
        min_sizes: Sequence[int] = (50, 100, 200),
        keep_largest: bool = False,
    ) -> Tuple["PostProcessor", float]:
        """Brute‑force search for parameters that maximise mean Dice.

        Parameters
        ----------
        maps, gts : lists of np.ndarray
            Validation anomaly maps and corresponding ground truth masks.
        highs, lows, sigmas, min_sizes : sequences
            Candidate values for each hyper‑parameter.
        keep_largest : bool
            Whether to keep only the largest component during search.

        Returns
        -------
        best_processor : PostProcessor
            Instance with the best hyper‑parameters.
        best_dice : float
            Mean Dice on *maps/gts* achieved by *best_processor*.
        """
        from itertools import product

        def dice_coefficient(pred: np.ndarray, tgt: np.ndarray, eps: float = 1e-5) -> float:
            pred = pred.astype(bool).ravel()
            tgt = tgt.astype(bool).ravel()
            inter = np.logical_and(pred, tgt).sum()
            return (2 * inter + eps) / (pred.sum() + tgt.sum() + eps)

        best = None
        best_dice = -1.0

        for h, l, s, m in product(highs, lows, sigmas, min_sizes):
            if h <= l:
                continue  # invalid combo
            proc = PostProcessor(h, l, s, m, keep_largest)
            dices = []
            for m_, g_ in zip(maps, gts):
                pred = proc(m_)
                dices.append(dice_coefficient(pred, g_))
            mean_dice = float(np.mean(dices))
            if mean_dice > best_dice:
                best_dice = mean_dice
                best = proc

        return best, best_dice
