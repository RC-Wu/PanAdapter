import torch
import torch.nn as nn
import torch.nn.functional as F

import models
from models.edsr import make_edsr_baseline
from models.basic import SELayer, default_conv
from models.adapter import SSPEN, PanAdapterLayer, SirenMLP
from models.ipt import VisionTransformer


class Config:
    def __init__(self):
        self.n_out_colors = 8
        self.img_dim = 16
        self.patch_dim = 1
        self.n_feats = 576
        self.num_heads = 12
        self.num_layers = 12
        self.hidden_dim = 576 * 4
        self.num_queries = 6
        self.dropout_rate = 0
        self.k = 96


class ipt_token(nn.Module):

    def __init__(self, args: Config, conv=default_conv, m=64):
        super().__init__()
        self.stage = 2
        self.k = args.k
        s = args.n_out_colors

        self.head = nn.Sequential(
            SELayer(s + 1),
            conv(s + 1, args.n_feats, 3),
            models.basic.ResBlock(conv, args.n_feats, 5, act=nn.ReLU(True)),
            models.basic.ResBlock(conv, args.n_feats, 5, act=nn.ReLU(True)),
        )
        self.head_down = nn.Conv2d(args.n_feats, args.n_feats, 4, 4)

        self.vit = VisionTransformer(
            img_dim=args.img_dim, patch_dim=args.patch_dim,
            num_channels=args.n_feats,
            embedding_dim=args.n_feats,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            hidden_dim=args.hidden_dim,
            num_queries=args.num_queries,
            dropout_rate=args.dropout_rate,
        )

        spe_edsr = make_edsr_baseline(n_colors=s, no_upsampling=True)
        spa_edsr = make_edsr_baseline(n_colors=s + 1, no_upsampling=True)
        self.sspen = SSPEN(spe_edsr, spa_edsr, n_layers=16, d_prime=16, m=m)

        self.a_proj = nn.Linear(m, self.k)
        self.b_proj = nn.Linear(16 * m, self.k)

        self.adapters = nn.ModuleList([PanAdapterLayer(vit_dim=args.n_feats, k=self.k, num_heads=4) for _ in range(6)])

        in_ch = 2 * self.k + 2
        self.tail2 = SirenMLP(in_ch=in_ch, hidden=576, out_ch=s, num_hidden_layers=1, w0=1.0)

        self.set_stage(2)

    def set_stage(self, stage: int):
        self.stage = stage
        # freeze all
        for p in self.parameters():
            p.requires_grad = False

        if stage == 1:
            for n, p in self.sspen.named_parameters():
                p.requires_grad = True
        else:
            for p in self.adapters.parameters():
                p.requires_grad = True
            for p in self.a_proj.parameters():
                p.requires_grad = True
            for p in self.b_proj.parameters():
                p.requires_grad = True
            for p in self.tail2.parameters():
                p.requires_grad = True

    def _tokenize_A(self, A):
        # A: (B,m,16,16) -> tokens (L,B,m)
        B, m, h, w = A.shape
        tok = A.flatten(2).transpose(1, 2)   # B,256,m
        tok = self.a_proj(tok)               # B,256,k
        return tok.transpose(0, 1).contiguous()

    def _tokenize_B(self, Bfeat):
        # B: (B,m,64,64) patch=4 -> tokens B,256,16m -> proj->k
        unfold = nn.Unfold(kernel_size=4, stride=4)
        patches = unfold(Bfeat)  # B, m*16, 256
        patches = patches.transpose(1, 2)   # B,256,16m
        patches = self.b_proj(patches)      # B,256,k
        return patches.transpose(0, 1).contiguous()

    def _tokens_to_feat(self, tok):
        # tok: (L,B,k) -> (B,k,16,16)
        tok = tok.transpose(0, 1)  # B,L,k
        B, L, k = tok.shape
        return tok.transpose(1, 2).reshape(B, k, 16, 16)

    def forward(self, lms, pan):
        # lms: (B,8,64,64), pan: (B,1,64,64)
        if self.stage == 1:
            raise NotImplementedError("Stage1 training not required here, but SSPEN is implemented.")

        # prepare M and Q
        M = F.interpolate(lms, scale_factor=0.25, mode="bicubic", align_corners=False)
        Q = torch.cat([lms, pan], dim=1)

        with torch.no_grad():
            A, B = self.sspen(M, Q)

        spe_tok = self._tokenize_A(A)      # (256,B,k)
        spa_tok = self._tokenize_B(B)      # (256,B,k)

        # vit input
        x = self.head(Q)
        x = self.head_down(x)              # (B,576,16,16)

        adapter_idx = 0

        def inject_fn(stage, layer_i, tokens):
            nonlocal spe_tok, spa_tok, adapter_idx
            tokens, spe_tok, spa_tok = self.adapters[adapter_idx](tokens, spe_tok, spa_tok)
            adapter_idx += 1
            return tokens

        _ = self.vit.forward_with_injections(x, inject_fn=inject_fn, t=4)

        # final priors
        A_hat = self._tokens_to_feat(spe_tok)
        B_hat = self._tokens_to_feat(spa_tok)

        # INR tail: sample features -> residual
        coord = self._make_coord(64, 64, device=lms.device)
        coord_map = coord.permute(2, 0, 1).unsqueeze(0).repeat(lms.shape[0], 1, 1, 1)

        A_up = F.grid_sample(A_hat, coord.unsqueeze(0), mode="bilinear", align_corners=False)
        B_up = F.grid_sample(B_hat, coord.unsqueeze(0), mode="bilinear", align_corners=False)

        inp = torch.cat([A_up, B_up, coord_map], dim=1)
        res = self.tail2(inp)
        return res + lms

    def _make_coord(self, H, W, device):
        xs = torch.linspace(-1, 1, W, device=device)
        ys = torch.linspace(-1, 1, H, device=device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        return torch.stack([grid_x, grid_y], dim=-1)