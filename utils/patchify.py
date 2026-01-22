import torch
import torch.nn.functional as F

def extract_nonoverlap_patches(x: torch.Tensor, ps: int):

    B, C, H, W = x.shape
    unfold = F.unfold(x, kernel_size=ps, stride=ps)  # (B, C*ps*ps, L)
    L = unfold.shape[-1]
    patches = unfold.transpose(1, 2).reshape(B * L, C, ps, ps)
    return patches, L
