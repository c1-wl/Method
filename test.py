import os, csv, argparse
import numpy as np
import torch


from config import Config
from models.ca_model import CA8Model
from utils.dataset import NUSDataset
from utils.kmeans import process_image

def unit_vec(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(x) + 1e-8)
    return x / n

def angular_error_deg(pred: np.ndarray, gt: np.ndarray) -> float:
    p = unit_vec(pred); g = unit_vec(gt)
    cosv = float(np.clip(np.dot(p, g), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosv)))

def summary_stats(errors):
    arr = np.asarray(errors, dtype=np.float64)
    return float(arr.mean()), float(np.median(arr)), float(np.percentile(arr, 95)), float(arr.max())

def load_state(model: torch.nn.Module, ckpt_path: str, device: torch.device):

    try:
        state = torch.load(ckpt_path, map_location=device, weights_only=True)  # 新版
    except TypeError:
        state = torch.load(ckpt_path, map_location=device)                     # 旧版
    if isinstance(state, dict) and not any(k in state for k in ("model","state_dict","model_state_dict")):
        model.load_state_dict(state, strict=False)
    else:
        for k in ("model", "state_dict", "model_state_dict"):
            if k in state and isinstance(state[k], dict):
                model.load_state_dict(state[k], strict=False)
                break

def main():
    ap = argparse.ArgumentParser("Evaluate single fold checkpoint on testing set")
    ap.add_argument("--ckpt", type=str, default="checkpoints/best_model.pth")
    ap.add_argument("--data_root", type=str, default=None)
    ap.add_argument("--crop_size", type=int, default=512)
    ap.add_argument("--out", type=str, default="./outputs/test_single_fold1.csv")
    ap.add_argument("--device", type=str, default=None, help="cuda / cpu; 默认取 Config.device")
    ap.add_argument("--patch_size", type=int, default=None, help="覆盖 Config.patch_size 做滑窗")
    args = ap.parse_args()

    device = torch.device(args.device or getattr(Config, "device", "cuda" if torch.cuda.is_available() else "cpu"))
    torch.set_grad_enabled(False)


    model = CA8Model(
        spectral_bands=getattr(Config, "spectral_bands", 31),
        spec_feat_k=getattr(Config, "spec_feat_k", 4)
    ).to(device).eval()


    if not os.path.isfile(args.ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")
    load_state(model, args.ckpt, device)


    data_root = args.data_root or getattr(Config, "data_root")
    if not data_root:
        raise ValueError("Please set --data_root or Config.data_root")
    test_set = NUSDataset(data_root, mode="testing", crop_size=args.crop_size)


    ps = args.patch_size or getattr(Config, "patch_size", 8)
    wp_q = getattr(Config, "wp_quantile", 0.95)
    bright_frac = getattr(Config, "bright_topk_frac", 0.10)


    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    errs = []
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filename","gt_R","gt_G","gt_B","pred_R","pred_G","pred_B","err_deg"])

        print(f"\n[Single-Fold Eval] ckpt={args.ckpt}")
        print(f"Testing images: {len(test_set)} | patch_size={ps} | crop={args.crop_size}\n")

        for i in range(len(test_set)):
            item = test_set[i]
            if isinstance(item, dict):
                rgb = item['rgb_img']; spec = item['spectral_cube']; gt = item['rgb_illuminant']
                name = getattr(item, 'filename', f"img_{i:04d}")
            else:

                rgb, spec, gt, *rest = item
                name = rest[0] if rest else f"img_{i:04d}"


            est = process_image(
                model, rgb.float(), spec.float(),
                patch_size=ps, device=device.type,
                wp_quantile=wp_q, bright_topk_frac=bright_frac, return_debug=False
            )
            if isinstance(est, torch.Tensor):
                pred = est.detach().cpu().numpy().reshape(-1)
            else:
                pred = np.asarray(est, dtype=np.float32).reshape(-1)

            gt_np = gt.detach().cpu().numpy().reshape(-1) if torch.is_tensor(gt) else np.asarray(gt, np.float32).reshape(-1)
            err = angular_error_deg(pred, gt_np)
            errs.append(err)

            w.writerow([name, f"{gt_np[0]:.6f}", f"{gt_np[1]:.6f}", f"{gt_np[2]:.6f}",
                              f"{pred[0]:.6f}", f"{pred[1]:.6f}", f"{pred[2]:.6f}", f"{err:.6f}"])

    mean, median, p95, mx = summary_stats(errs)
    print("[Result]")
    print(f"  Mean   : {mean:.3f}°")
    print(f"  Median : {median:.3f}°")
    print(f"  95th   : {p95:.3f}°")
    print(f"  Max    : {mx:.3f}°")
    print(f"\nSaved per-image predictions to: {args.out}\n")

if __name__ == "__main__":
    main()
