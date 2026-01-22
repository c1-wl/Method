# spectral_features.py
import torch

@torch.no_grad()
def compute_spectral_features_batch(
    spec_batch: torch.Tensor,              # (B, N, ps, ps)
    wp_quantile: float = 0.95,
    bright_topk_frac: float = 0.10,
    use_pca_bright: bool = True,
    pb_mode: str = 'bright_dark'
) -> torch.Tensor:

    assert spec_batch.dim() == 4
    B, N, h, w = spec_batch.shape
    P = h * w

    X = spec_batch.reshape(B, N, P).permute(0, 2, 1)
    I = X.sum(dim=2)

    m = max(3, int(bright_topk_frac * P))
    if pb_mode == 'bright':
        m = min(m, P)
    else:
        m = min(m, P // 2)

    idx_b = torch.topk(I, k=m, dim=1, largest=True,  sorted=False).indices
    Bset  = torch.gather(X, 1, idx_b.unsqueeze(-1).expand(-1, -1, N))  # (B,m,N)

    if pb_mode == 'bright_dark':
        idx_d = torch.topk(I, k=m, dim=1, largest=False, sorted=False).indices
        Dset  = torch.gather(X, 1, idx_d.unsqueeze(-1).expand(-1, -1, N))  # (B,m,N)
        Sset  = torch.cat([Bset, Dset], dim=1)  # (B,2m,N)
    else:
        Sset  = Bset


    gw = X.mean(dim=1)                               # (B,N)
    br = Bset.mean(dim=1)                            # (B,N)
    wp = torch.quantile(Bset, q=wp_quantile, dim=1)  # (B,N)

    if use_pca_bright and Sset.shape[1] >= 3:
        Sc = Sset - Sset.mean(dim=1, keepdim=True)
        try:
            U, Svals, Vh = torch.linalg.svd(Sc, full_matrices=False)  # Vh:(B,k,N)
            pc1 = Vh[:, 0, :]                                        # (B,N)
        except RuntimeError:
            C = torch.matmul(Sc.transpose(1, 2), Sc) / max(Sset.shape[1] - 1, 1)
            eigvals, eigvecs = torch.linalg.eigh(C)
            pc1 = eigvecs[:, :, -1]

        sign = torch.sign((pc1 * br).sum(dim=1, keepdim=True))
        sign[sign == 0] = 1
        pb = pc1 * sign
    else:
        pb = br.clone()

    # 谱形归一（去亮度，只保形状）
    def norm_shape(v):  # (B,N)
        return v / (v.sum(dim=1, keepdim=True) + 1e-8)

    feats = torch.cat([norm_shape(gw), norm_shape(wp),
                       norm_shape(br), norm_shape(pb)], dim=1).to(spec_batch.dtype)  # (B,4N)
    return feats


@torch.no_grad()
def compute_spectral_features(
    spectral_patch: torch.Tensor,   # (N,h,w)
    wp_quantile: float = 0.95,
    bright_topk_frac: float = 0.10,
    use_pca_bright: bool = True,
    pb_mode: str = 'bright_dark'
) -> torch.Tensor:
    """
    单 patch 版四候选 [GW | WP | BR | PB] → (4N,)
    """
    assert spectral_patch.dim() == 3
    N, h, w = spectral_patch.shape
    X = spectral_patch.reshape(N, -1).transpose(0, 1)  # (P,N)
    P = X.shape[0]
    I = X.sum(dim=1)

    m = max(3, int(bright_topk_frac * P))
    if pb_mode == 'bright':
        m = min(m, P)
    else:
        m = min(m, P // 2)

    # 亮/暗子集
    topv, idx_b = torch.topk(I, k=m, largest=True,  sorted=False)
    B = X[idx_b]  # (m,N)
    if pb_mode == 'bright_dark':
        botv, idx_d = torch.topk(I, k=m, largest=False, sorted=False)
        D = X[idx_d]
        S = torch.cat([B, D], dim=0)
    else:
        S = B

    # GW / BR / WP
    gw = X.mean(dim=0)
    br = B.mean(dim=0)
    wp = torch.quantile(B, q=wp_quantile, dim=0)

    # PCA 主轴
    if use_pca_bright and S.shape[0] >= 3:
        Sc = S - S.mean(dim=0, keepdim=True)
        try:
            U, Svals, Vh = torch.linalg.svd(Sc, full_matrices=False)
            pc1 = Vh[0]  # (N,)
        except RuntimeError:
            C = (Sc.T @ Sc) / max(Sc.shape[0] - 1, 1)
            eigvals, eigvecs = torch.linalg.eigh(C)
            pc1 = eigvecs[:, -1]
        if torch.dot(pc1, br) < 0:
            pc1 = -pc1
        pb = pc1
    else:
        pb = br.clone()

    # 形状归一
    def norm_shape_1d(v):
        return v / (v.sum() + 1e-8)

    feats = torch.cat([norm_shape_1d(gw), norm_shape_1d(wp),
                       norm_shape_1d(br), norm_shape_1d(pb)], dim=0).to(spectral_patch.dtype)  # (4N,)
    return feats
