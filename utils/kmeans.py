# utils/kmeans.py
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import torch
from threadpoolctl import threadpool_limits
from config import Config
from utils.spectral_features import compute_spectral_features_batch

def _unit_norm(x: np.ndarray):
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / n

def _entropy_norm(alpha: torch.Tensor, eps=1e-8):
    H = -(alpha * (alpha + eps).log()).sum(dim=1) / np.log(alpha.shape[1])
    return torch.clamp(H, 0.0, 1.0)

def _sanitize_weights(weights, n):
    if weights is None:
        return None
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if w.shape[0] != n:
        if w.shape[0] > n:
            w = w[:n]
        else:
            w = np.pad(w, (0, n - w.shape[0]), mode='constant')
    w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    w = np.clip(w, 0.0, None)
    s = w.sum()
    if not np.isfinite(s) or s <= 0:
        return None
    return w

def _mean_aggregate(patch_estimates: np.ndarray):
    X = patch_estimates
    if getattr(Config, "kmeans_unit_normalize", True):
        X = _unit_norm(X)
    mu = X.mean(axis=0)
    mu = mu / (np.linalg.norm(mu) + 1e-12)
    return mu

def select_illuminant_kmeans(patch_estimates: np.ndarray, weights: np.ndarray = None):
    X = patch_estimates
    if getattr(Config, "kmeans_unit_normalize", True):
        X = _unit_norm(X)

    if len(X) == 1:
        return X[0]

    best_k, best_score = 1, -1
    threads = int(getattr(Config, "kmeans_threads", 1))
    max_k = min(int(getattr(Config, "kmeans_max_k", 6)), len(X))
    for k in range(2, max_k + 1):
        try:
            # 如果你 sklearn 老，n_init="auto" 会报错 -> 改成 n_init=10
            km = KMeans(n_clusters=k, random_state=42, n_init="auto")
            with threadpool_limits(limits=threads):
                labels = km.fit_predict(X)
            if len(np.unique(labels)) < 2:
                continue
            score = silhouette_score(X, labels)
            if score > best_score:
                best_score, best_k = score, k
        except Exception:
            continue

    w = _sanitize_weights(weights, len(X)) if weights is not None else None
    km = KMeans(n_clusters=best_k, random_state=42, n_init="auto")
    with threadpool_limits(limits=threads):
        if w is not None:
            km.fit(X, sample_weight=w)
        else:
            km.fit(X)
    labels = km.labels_

    if w is not None:
        wsum = np.zeros(best_k, dtype=np.float64)
        for c in range(best_k):
            wsum[c] = w[labels == c].sum()
        largest = int(wsum.argmax())
    else:
        unique, counts = np.unique(labels, return_counts=True)
        largest = int(unique[counts.argmax()])
    return km.cluster_centers_[largest]

@torch.no_grad()
def process_image(model, rgb_img, spectral_cube, patch_size=8, device='cuda',
                  wp_quantile=0.95, bright_topk_frac=0.10, return_debug=False,
                  spec_mode="gate", agg_mode="wkmeans"):

    model.eval()
    H, W = rgb_img.shape[1], rgb_img.shape[2]
    preds, alphas, coords = [], [], []
    bright_weights = []

    rgb_img = rgb_img.to(device, non_blocking=True)
    spectral_cube = spectral_cube.to(device, non_blocking=True)

    spec_mode = (spec_mode or "gate").lower()
    agg_mode = (agg_mode or "wkmeans").lower()

    for i in range(0, H - patch_size + 1, patch_size):
        for j in range(0, W - patch_size + 1, patch_size):
            rgb_patch = rgb_img[:, i:i + patch_size, j:j + patch_size].unsqueeze(0)
            spec_patch = spectral_cube[:, i:i + patch_size, j:j + patch_size].unsqueeze(0)

            if spec_mode == "none":
                feats = None
            else:
                feats = compute_spectral_features_batch(
                    spec_patch,
                    wp_quantile=wp_quantile,
                    bright_topk_frac=bright_topk_frac,
                    use_pca_bright=getattr(Config, 'use_pca_bright_eval', True),
                    pb_mode=getattr(Config, 'pb_mode', 'bright_dark')
                ).to(device)

            pred, alpha = model(rgb_patch, feats)
            preds.append(pred.squeeze(0).detach().cpu().numpy())
            coords.append((i, j))

            if agg_mode == "wkmeans":
                I_patch = spec_patch.sum(dim=1).mean().item()
                bright_weights.append(I_patch)
                if alpha is not None:
                    alphas.append(alpha.squeeze(0).detach())

    preds = np.stack(preds, axis=0)

    if agg_mode == "mean":
        est = _mean_aggregate(preds)
        est = torch.from_numpy(est).float()
        return (est, dict(preds=preds, coords=np.array(coords))) if return_debug else est

    if agg_mode == "kmeans":
        est = select_illuminant_kmeans(preds, weights=None)
        est = torch.from_numpy(est).float()
        return (est, dict(preds=preds, coords=np.array(coords))) if return_debug else est

    if agg_mode == "wkmeans":
        bright = np.asarray(bright_weights, dtype=np.float64)
        bright_norm = bright / (bright.mean() + 1e-12)
        p = float(getattr(Config, 'conf_brightness_power', 0.5))
        conf = np.clip(np.maximum(bright_norm, 0.0) ** p, 0.0, None)

        alphas_np = None
        if len(alphas) > 0:
            alphas_t = torch.stack(alphas, dim=0)
            Hnorm = _entropy_norm(alphas_t).cpu().numpy()
            conf = conf * (1.0 - np.clip(Hnorm, 0.0, 1.0))
            alphas_np = alphas_t.cpu().numpy()

        est = select_illuminant_kmeans(preds, weights=conf)
        est = torch.from_numpy(est).float()

        if return_debug:
            return est, dict(preds=preds, coords=np.array(coords), weights=conf, alphas=alphas_np)
        return est

    raise ValueError(f"Unknown agg_mode={agg_mode}")
