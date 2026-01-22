import torch
import torch.nn.functional as F  # Add this import
import numpy as np


def angular_error(pred, target):


    pred_norm = F.normalize(pred, p=2, dim=-1)
    target_norm = F.normalize(target, p=2, dim=-1)

    dot_product = (pred_norm * target_norm).sum(dim=-1)

    dot_product = torch.clamp(dot_product, -1.0, 1.0)

    angle = torch.acos(dot_product)

    # 转换为度
    angle_deg = angle * (180.0 / torch.pi)

    return angle_deg


def recovery_angular_error(pred, target):
    return angular_error(pred, target)


def compute_metrics(pred, target):
    metrics = {}


    if isinstance(pred, np.ndarray):
        pred = torch.from_numpy(pred)
    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target)


    ae = angular_error(pred, target)

    metrics['mean_angular_error'] = ae.mean().item()
    metrics['median_angular_error'] = ae.median().item()
    metrics['max_angular_error'] = ae.max().item()
    metrics['min_angular_error'] = ae.min().item()
    metrics['percentile_95'] = np.percentile(ae.numpy(), 95)

    return metrics


def print_metrics(metrics):

    print("\nEvaluation Metrics:")
    print(f"Mean Angular Error: {metrics['mean_angular_error']:.4f} degrees")
    print(f"Median Angular Error: {metrics['median_angular_error']:.4f} degrees")
    print(f"Max Angular Error: {metrics['max_angular_error']:.4f} degrees")
    print(f"Min Angular Error: {metrics['min_angular_error']:.4f} degrees")
    print(f"95th Percentile Error: {metrics['percentile_95']:.4f} degrees")