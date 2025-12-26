import math
from argparse import Namespace

import torch
import torch.nn as nn
from models.basic import ResBlock
from models.basic import default_conv


class MS_encoder(nn.Module):
    def __init__(self, ms_colors, ms_feats, mid_feats=64, conv=default_conv):
        super().__init__()

        act = nn.ReLU(True)
        kernel_size = 3
        self.head = nn.Sequential(
                conv(ms_colors, mid_feats, kernel_size),
                ResBlock(conv, mid_feats, 5, act=act),
                ResBlock(conv, mid_feats, 5, act=act),
                conv(mid_feats, ms_feats, 1),
            )

    def forward(self, x):
        return self.head(x)


if __name__ == '__main__':
    model = MS_encoder(8, 64)

    x = torch.rand(1, 8, 16, 16)
    out = model(x)
    print(out.shape)