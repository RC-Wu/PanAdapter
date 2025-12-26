import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


def _apply_linear(linear: nn.Linear, x: torch.Tensor) -> torch.Tensor:
    orig = x.shape
    x2 = x.reshape(-1, orig[-1])
    y = linear(x2)
    return y.reshape(*orig[:-1], y.shape[-1])

class LPEBlock(nn.Module):
    def __init__(self, in_ch: int, d_prime: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, d_prime, 1, 1, 0)
        self.act = nn.GELU()
        self.dw = nn.Conv2d(d_prime, d_prime, 3, 1, 1, groups=d_prime)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act(x)
        x = self.dw(x)
        return x


class SSPEN(nn.Module):
    def __init__(self, spe_edsr, spa_edsr, n_layers: int = 16, d_prime: int = 16, m: int = 64):
        super().__init__()
        self.spe_edsr = spe_edsr
        self.spa_edsr = spa_edsr

        # freeze pretrained CNNs
        for p in self.spe_edsr.parameters():
            p.requires_grad = False
        for p in self.spa_edsr.parameters():
            p.requires_grad = False

        self.n_layers = n_layers
        self.m = m

        self.spe_lpes = nn.ModuleList([LPEBlock(64, d_prime) for _ in range(n_layers)])
        self.spa_lpes = nn.ModuleList([LPEBlock(64, d_prime) for _ in range(n_layers)])

        self.spe_proj = nn.Conv2d(n_layers * d_prime, m, 1, 1, 0)
        self.spa_proj = nn.Conv2d(n_layers * d_prime, m, 1, 1, 0)

    def forward(self, M: torch.Tensor, Q: torch.Tensor):

        _, spe_tokens = self.spe_edsr(M)   # (n,B,64,16,16)
        _, spa_tokens = self.spa_edsr(Q)   # (n,B,64,64,64)

        spe_feats = []
        spa_feats = []

        for i in range(self.n_layers):
            spe_feats.append(self.spe_lpes[i](spe_tokens[i]))
            spa_feats.append(self.spa_lpes[i](spa_tokens[i]))

        spe_cat = torch.cat(spe_feats, dim=1)
        spa_cat = torch.cat(spa_feats, dim=1)

        A = self.spe_proj(spe_cat)  # (B,m,16,16)
        B = self.spa_proj(spa_cat)  # (B,m,64,64)
        return A, B

class WeightingNet(nn.Module):

    def __init__(self, k: int, hidden: Optional[int] = None):
        super().__init__()
        hidden = hidden if hidden is not None else max(1, k // 4)
        self.fc1 = nn.Linear(k, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, k)

    def forward(self, x):
        x = _apply_linear(self.fc1, x)
        x = self.act(x)
        x = _apply_linear(self.fc2, x)
        return torch.sigmoid(x)


class TokenFFN(nn.Module):
    def __init__(self, k: int, hidden: Optional[int] = None):
        super().__init__()
        hidden = hidden if hidden is not None else k
        self.fc1 = nn.Linear(k, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, k)

    def forward(self, x):
        y = _apply_linear(self.fc1, x)
        y = self.act(y)
        y = _apply_linear(self.fc2, y)
        return x + y


class CTF(nn.Module):

    def __init__(self, k: int, num_heads: int = 4):
        super().__init__()
        self.w = WeightingNet(k)
        self.attn = nn.MultiheadAttention(k, num_heads, batch_first=False)
        self.ffn = TokenFFN(k, hidden=k)

    def forward(self, prior, other, vit_k):
        fused = prior * self.w(other) + prior
        attn_out, _ = self.attn(query=vit_k, key=fused, value=fused, need_weights=False)
        fused = attn_out + fused
        fused = self.ffn(fused)
        return fused


class CTI(nn.Module):

    def __init__(self, k: int, num_heads: int = 4):
        super().__init__()
        self.fus = nn.Linear(2 * k, k)
        self.attn = nn.MultiheadAttention(k, num_heads, batch_first=False)
        self.scale = nn.Parameter(torch.zeros(1))

    def forward(self, spe, spa, vit_k):
        fus = torch.cat([spa, spe], dim=-1)
        fus = _apply_linear(self.fus, fus)
        attn_out, _ = self.attn(query=fus, key=vit_k, value=vit_k, need_weights=False)
        return self.scale * attn_out


class PanAdapterLayer(nn.Module):

    def __init__(self, vit_dim: int, k: int = 96, num_heads: int = 4):
        super().__init__()
        self.vit_to_k = nn.Linear(vit_dim, k)
        self.k_to_vit = nn.Linear(k, vit_dim)
        self.ctf_spa = CTF(k, num_heads=num_heads)
        self.ctf_spe = CTF(k, num_heads=num_heads)
        self.cti = CTI(k, num_heads=num_heads)

    def forward(self, vit_tokens, spe_tokens, spa_tokens):
        vit_k = _apply_linear(self.vit_to_k, vit_tokens)
        spa_next = self.ctf_spa(spa_tokens, spe_tokens, vit_k)
        spe_next = self.ctf_spe(spe_tokens, spa_tokens, vit_k)
        delta_k = self.cti(spe_next, spa_next, vit_k)
        vit_tokens = vit_tokens + _apply_linear(self.k_to_vit, delta_k)
        return vit_tokens, spe_next, spa_next

class Sine(nn.Module):
    def __init__(self, w0=1.0):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


class SirenMLP(nn.Module):

    def __init__(self, in_ch: int, hidden: int, out_ch: int, num_hidden_layers: int = 1, w0=1.0):
        super().__init__()
        layers = [nn.Conv2d(in_ch, hidden, 1), Sine(w0)]
        for _ in range(num_hidden_layers):
            layers += [nn.Conv2d(hidden, hidden, 1), Sine(w0)]
        layers += [nn.Conv2d(hidden, out_ch, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
