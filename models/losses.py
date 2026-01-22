import torch
import torch.nn as nn
import torch.nn.functional as F


class CosineLoss(nn.Module):

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pn = F.normalize(pred, p=2, dim=1)
        tn = F.normalize(target, p=2, dim=1)
        dot = torch.sum(pn * tn, dim=1).clamp(-1.0 + self.eps, 1.0 - self.eps)
        loss = 1.0 - dot
        return loss.mean()


class AngularCosineHybridLoss(nn.Module):

    def __init__(self, alpha: float = 0.2, eps: float = 1e-6):
        super().__init__()
        assert 0.0 <= alpha <= 1.0
        self.alpha = alpha
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pn = F.normalize(pred, p=2, dim=1)
        tn = F.normalize(target, p=2, dim=1)
        dot = torch.sum(pn * tn, dim=1).clamp(-1.0 + self.eps, 1.0 - self.eps)
        ang = torch.acos(dot)               # radians
        cos = 1.0 - dot
        loss = self.alpha * ang + (1.0 - self.alpha) * cos
        return loss.mean()
