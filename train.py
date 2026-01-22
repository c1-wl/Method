import os, time, math, argparse, contextlib
import numpy as np
import torch
from sklearn.model_selection import KFold

try:
    from torch.cuda.amp import GradScaler, autocast
    _HAS_AMP = True
except Exception:
    GradScaler = None
    @contextlib.contextmanager
    def autocast(*args, **kwargs):
        yield
    _HAS_AMP = False

from config import Config
from models.ca_model import CA8Model
from models.losses import AngularCosineHybridLoss
from utils.dataset import NUSDataset
from utils.patchify import extract_nonoverlap_patches
from utils.spectral_features import compute_spectral_features_batch
from utils.kmeans import process_image

torch.backends.cudnn.benchmark = True

def set_seed(seed: int):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def save_rng_to(ckpt: dict):
    ckpt["torch_rng"] = torch.get_rng_state()
    ckpt["numpy_rng"] = np.random.get_state()
    if torch.cuda.is_available():
        ckpt["cuda_rng"] = torch.cuda.get_rng_state_all()

def load_rng_from(ckpt: dict):
    if "torch_rng" in ckpt: torch.set_rng_state(ckpt["torch_rng"])
    if "numpy_rng" in ckpt: np.random.set_state(ckpt["numpy_rng"])
    if torch.cuda.is_available() and ("cuda_rng" in ckpt):
        torch.cuda.set_rng_state_all(ckpt["cuda_rng"])

def angular_error_deg(pred: torch.Tensor, gt: torch.Tensor) -> float:
    pn = torch.nn.functional.normalize(pred, p=2, dim=0)
    gn = torch.nn.functional.normalize(gt, p=2, dim=0)
    cos = torch.clamp((pn * gn).sum(), -1.0 + 1e-6, 1.0 - 1e-6)
    return float(torch.arccos(cos).detach().cpu().numpy() * 180.0 / math.pi)

@torch.no_grad()
def evaluate_image_level(model, dataset, indices, device, spec_mode, agg_mode):
    model.eval()
    errs = []
    ps = getattr(Config, "patch_size", 8)
    for idx in indices:
        item = dataset[idx]
        if isinstance(item, dict):
            rgb = item['rgb_img']; spec = item['spectral_cube']; gt = item['rgb_illuminant']
        else:
            rgb, spec, gt = item
        if isinstance(gt, np.ndarray):
            gt = torch.from_numpy(gt).float()

        est = process_image(
            model, rgb.float(), spec.float(),
            patch_size=ps, device=device,
            wp_quantile=getattr(Config, "wp_quantile", 0.95),
            bright_topk_frac=getattr(Config, "bright_topk_frac", 0.10),
            return_debug=False,
            spec_mode=spec_mode,
            agg_mode=agg_mode,
        )
        errs.append(angular_error_deg(est, gt))
    a = np.array(errs, dtype=np.float32)
    return float(a.mean()), float(np.median(a))

def train_one_epoch(model, optimizer, dataset, indices, device, spec_mode):
    model.train()
    total_loss, total_patches = 0.0, 0
    ps = getattr(Config, "patch_size", 8)
    bs = getattr(Config, "batch_size", 256)
    criterion = AngularCosineHybridLoss(alpha=getattr(Config, "loss_alpha", 0.2))
    scaler = GradScaler(enabled=_HAS_AMP)

    for idx in indices:
        item = dataset[idx]
        if isinstance(item, dict):
            rgb = item['rgb_img']; spec = item['spectral_cube']; gt = item['rgb_illuminant']
        else:
            rgb, spec, gt = item
        if isinstance(gt, np.ndarray):
            gt = torch.from_numpy(gt).float()

        rgb_patches, L = extract_nonoverlap_patches(rgb.float().unsqueeze(0), ps)
        spec_patches, _ = extract_nonoverlap_patches(spec.float().unsqueeze(0), ps)

        for s in range(0, L, bs):
            rgb_b = rgb_patches[s:s+bs].to(device, non_blocking=True)
            spec_b = spec_patches[s:s+bs].to(device, non_blocking=True)

            if spec_mode == "none":
                feats = None
            else:
                feats = compute_spectral_features_batch(
                    spec_b,
                    wp_quantile=getattr(Config, "wp_quantile", 0.95),
                    bright_topk_frac=getattr(Config, "bright_topk_frac", 0.10),
                    use_pca_bright=getattr(Config, "use_pca_bright_train", True),
                    pb_mode=getattr(Config, "pb_mode", "bright_dark")
                ).to(device)

            gt_b = gt.to(device).unsqueeze(0).expand(rgb_b.size(0), -1)

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=_HAS_AMP):
                pred, _ = model(rgb_b, feats)
                loss = criterion(pred, gt_b)

            if _HAS_AMP:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            total_loss += float(loss.detach().cpu()) * rgb_b.size(0)
            total_patches += rgb_b.size(0)

    return total_loss / max(1, total_patches)

def main():
    ap = argparse.ArgumentParser("Train ONLY the 3rd fold (split identical to train_kfold.py)")
    ap.add_argument("--save_split", type=str, default="./checkpoints/fold3_fixed_from_kfold.npz")
    ap.add_argument("--resume", action="store_true")

    ap.add_argument("--spec_mode", type=str, default=getattr(Config, "spec_mode_default", "gate"),
                    choices=["none", "gw", "wp", "br", "pb", "avg4", "gate"])
    ap.add_argument("--agg_mode", type=str, default=getattr(Config, "agg_mode_default", "wkmeans"),
                    choices=["mean", "kmeans", "wkmeans"])
    ap.add_argument("--tag", type=str, default="", help="用于 ckpt 命名，方便做消融表格")
    args = ap.parse_args()

    spec_mode = (args.spec_mode or "gate").lower()
    agg_mode  = (args.agg_mode or "wkmeans").lower()
    run_tag = args.tag.strip() or f"spec-{spec_mode}_agg-{agg_mode}"

    set_seed(getattr(Config, "seed", 666))
    device = getattr(Config, "device", "cuda" if torch.cuda.is_available() else "cpu")

    ds = NUSDataset(Config.data_root, mode='training', crop_size=512)
    n = len(ds)
    print(f"Total training images: {n} | folds={getattr(Config,'num_folds',5)} | seed={getattr(Config,'seed',666)}")
    print(f"[Ablation] spec_mode={spec_mode} | agg_mode={agg_mode} | tag={run_tag}")

    idx_all = np.arange(n)
    k = int(getattr(Config, "num_folds", 5))
    kf = KFold(n_splits=k, shuffle=True, random_state=getattr(Config, "seed", 666))

    fold = 0
    train_idx = val_idx = None
    for tr, va in kf.split(idx_all):
        fold += 1
        if fold == 3:
            train_idx, val_idx = np.asarray(tr, int), np.asarray(va, int)
            break
    assert train_idx is not None

    os.makedirs(os.path.dirname(args.save_split), exist_ok=True)
    names = getattr(ds, "filenames", None)
    if names is not None:
        train_files = np.array([names[i] for i in train_idx], dtype=object)
        val_files   = np.array([names[i] for i in val_idx], dtype=object)
    else:
        train_files = np.array([], dtype=object)
        val_files   = np.array([], dtype=object)

    np.savez_compressed(args.save_split,
                        train_idx=train_idx, val_idx=val_idx,
                        train_files=train_files, val_files=val_files)

    model = CA8Model(
        spectral_bands=getattr(Config, "spectral_bands", 31),
        spec_feat_k=getattr(Config, "spec_feat_k", 4),
        spec_mode=spec_mode
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=getattr(Config, "lr", 1e-4),
        weight_decay=getattr(Config, "weight_decay", 0.0)
    )

    os.makedirs(Config.save_dir, exist_ok=True)
    metric_key = getattr(Config, "select_metric", "mean")
    best_path  = os.path.join(Config.save_dir, f"best_fold3_fixed_{metric_key}_{run_tag}.pth")
    last_path  = os.path.join(Config.save_dir, f"last_fold3_fixed_{run_tag}.pt")

    start_epoch = 0
    best_score  = float("inf")
    best_pair   = (np.inf, np.inf)

    if args.resume and os.path.isfile(last_path):
        ckpt = torch.load(last_path, map_location=device)
        model.load_state_dict(ckpt["model"], strict=False)
        optimizer.load_state_dict(ckpt["optimizer"])
        load_rng_from(ckpt)
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        best_score  = float(ckpt.get("best_score", best_score))
        best_pair   = tuple(ckpt.get("best_pair", best_pair))
        print(f"[Resume] {last_path} | start_epoch={start_epoch}, best={best_pair}")

    max_epochs = getattr(Config, "max_epochs", 1000)
    for epoch in range(start_epoch, max_epochs):
        t0 = time.time()
        train_loss = train_one_epoch(model, optimizer, ds, train_idx, device, spec_mode)
        val_mean, val_median = evaluate_image_level(model, ds, val_idx, device, spec_mode, agg_mode)

        score = val_mean if metric_key == "mean" else val_median
        if score < best_score:
            best_score = score
            best_pair  = (val_mean, val_median)
            torch.save(model.state_dict(), best_path)
            print(f">> Saved BEST @ epoch {epoch+1}: mean={val_mean:.3f}°, median={val_median:.3f}° -> {os.path.basename(best_path)}")

        payload = dict(
            epoch=epoch,
            model=model.state_dict(),
            optimizer=optimizer.state_dict(),
            best_score=best_score,
            best_pair=best_pair,
            train_idx=train_idx,
            val_idx=val_idx,
            spec_mode=spec_mode,
            agg_mode=agg_mode,
            tag=run_tag
        )
        save_rng_to(payload)
        torch.save(payload, last_path)

        dt = time.time() - t0
        print(f"Epoch {epoch+1}/{max_epochs} | TrainLoss {train_loss:.4f} | Val mean {val_mean:.3f}°, median {val_median:.3f}° | {dt:.1f}s")

    print(f"[Done] best mean={best_pair[0]:.3f}°, median={best_pair[1]:.3f}° | {best_path}")

if __name__ == "__main__":
    main()
