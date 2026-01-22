# models/ca_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------
# RGB Branch: One-Net → 31
# -------------------------
class OneNetRGB(nn.Module):
    def __init__(self, output_dim=31):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),  nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(32, 64, 3, padding=1), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, 128, 3, padding=1),nn.LeakyReLU(0.1, inplace=True),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Linear(128, 128), nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x).flatten(1)
        x = self.head(x)               # (B,31)
        return x

# -------------------------
# Spectral Branch (gate-then-map, NO τ)
# -------------------------
class SpectralGate(nn.Module):
    def __init__(self, bands=31, num_candidates=4, hidden=32):
        super().__init__()
        self.K = num_candidates
        self.N = bands
        self.scorer = nn.Sequential(
            nn.Linear(self.N, hidden), nn.LeakyReLU(0.1),
            nn.Linear(hidden, 1)
        )

    def forward(self, cand: torch.Tensor):
        B, K, N = cand.shape
        logits = torch.cat([self.scorer(cand[:, k, :]) for k in range(K)], dim=1)  # (B,K)
        alpha  = F.softmax(logits, dim=1)  # ★ 没有除以 τ
        return alpha, logits

class SpectralEncoderShared(nn.Module):
    def __init__(self, bands=31):
        super().__init__()
        self.N = bands
        self.norm = nn.LayerNorm(self.N)
        self.mlp = nn.Sequential(
            nn.Linear(self.N, 60), nn.LeakyReLU(0.1),
            nn.Linear(60, 60),    nn.LeakyReLU(0.1),
            nn.Linear(60, self.N)
        )

    def forward(self, g: torch.Tensor):
        # g: (B,N) or (B,K,N)
        if g.dim() == 3:
            B, K, N = g.shape
            x = g.reshape(B * K, N)
            x = self.mlp(self.norm(x))
            return x.reshape(B, K, N)
        return self.mlp(self.norm(g))

class SpectralBranch(nn.Module):
    def __init__(self, bands=31, num_candidates=4, gate_hidden=32):
        super().__init__()
        self.N = bands
        self.K = num_candidates
        self.encoder = SpectralEncoderShared(bands=self.N)
        self.gate = SpectralGate(bands=self.N, num_candidates=self.K, hidden=gate_hidden)
        self.post_norm = nn.LayerNorm(self.N)

    def forward(self, spec_feats_flat):
        """
        spec_feats_flat: (B, K*N)
        """
        B = spec_feats_flat.size(0)
        x = spec_feats_flat.view(B, self.K, self.N)          # (B,K,N)
        alpha, _ = self.gate(x)                              # (B,K)
        g_mix = torch.sum(alpha.unsqueeze(-1) * x, dim=1)    # (B,N)
        out = self.encoder(g_mix)                            # (B,N)
        out = self.post_norm(out)
        return out, alpha

# -------------------------
# Fusion & Regressor
# -------------------------
class Encoder(nn.Module):
    def __init__(self, input_dim=62):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 15)
        self.fc2 = nn.Linear(15, 5)
        self.fc3 = nn.Linear(5, 3)
        self.act = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.act(self.fc3(x))
        return x

class CA8Model(nn.Module):
    """
    spec_mode:
      - none : 去光谱分支（RGB-only）
      - gw/wp/br/pb : 单条先验
      - avg4 : 四条先验平均（无门控）
      - gate : 四条先验 + 门控（完整）
    """
    def __init__(self, spectral_bands=31, spec_feat_k=4, spec_mode="gate"):
        super().__init__()
        self.N = spectral_bands
        self.K = spec_feat_k
        self.spec_mode = (spec_mode or "gate").lower()

        self.rgb_branch = OneNetRGB(output_dim=self.N)

        self._single_map = {"gw": 0, "wp": 1, "br": 2, "pb": 3}

        if self.spec_mode == "gate":
            self.spectral_branch = SpectralBranch(bands=self.N, num_candidates=self.K)
            self.spec_encoder = None
            self.spec_post_norm = None
            enc_in = 2 * self.N
        elif self.spec_mode in ("avg4", "gw", "wp", "br", "pb"):
            self.spectral_branch = None
            self.spec_encoder = SpectralEncoderShared(bands=self.N)
            self.spec_post_norm = nn.LayerNorm(self.N)
            enc_in = 2 * self.N
        elif self.spec_mode == "none":
            self.spectral_branch = None
            self.spec_encoder = None
            self.spec_post_norm = None
            enc_in = self.N
        else:
            raise ValueError(f"Unknown spec_mode={self.spec_mode}")

        self.encoder = Encoder(input_dim=enc_in)

    def forward(self, rgb_patch, spectral_feats=None):
        rgb_features = self.rgb_branch(rgb_patch)  # (B,31)
        alpha = None

        if self.spec_mode == "none":
            combined = rgb_features
        else:
            if spectral_feats is None:
                # 防御：如果忘传，就当 0
                spec_features = torch.zeros_like(rgb_features)
            else:
                B = spectral_feats.size(0)
                x = spectral_feats.view(B, self.K, self.N)  # (B,4,31)

                if self.spec_mode == "gate":
                    spec_features, alpha = self.spectral_branch(spectral_feats)  # (B,31),(B,4)
                else:
                    if self.spec_mode == "avg4":
                        g = x.mean(dim=1)  # (B,31)
                    else:
                        g = x[:, self._single_map[self.spec_mode], :]  # (B,31)

                    spec_features = self.spec_encoder(g)
                    spec_features = self.spec_post_norm(spec_features)

            combined = torch.cat([rgb_features, spec_features], dim=1)  # (B,62)

        pred = self.encoder(combined)  # (B,3)
        return pred, alpha
