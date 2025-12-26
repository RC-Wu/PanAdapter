import models
from models.basic import _get_clones, SELayer
from models.Adapter import Injector

import torch
from torch import nn
import torch.nn.functional as F
import math


class Config:
    def __init__(self):
        self.rgb_range = 255
        self.n_colors = 9
        self.n_out_colors = 8
        self.scale = [1]

        self.img_dim = 64
        self.patch_dim = 1
        self.n_feats = 576

        self.num_heads = 12
        self.num_layers = 12

        self.num_queries = 6
        self.dropout_rate = 0


class ipt(nn.Module):
    def __init__(self, args, conv=models.basic.default_conv):
        super(ipt, self).__init__()

        self.scale_idx = 0

        n_feats = args.n_feats
        kernel_size = 3
        act = nn.ReLU(True)

        self._head = nn.Sequential(
                SELayer(9),
                conv(args.n_colors, n_feats, kernel_size),
                models.basic.ResBlock(conv, n_feats, 5, act=act),
                models.basic.ResBlock(conv, n_feats, 5, act=act)
            )

        self._head1 = nn.Conv2d(576, 576, kernel_size=4, stride=4)

        self.body = VisionTransformer(img_dim=args.img_dim, patch_dim=args.patch_dim, num_channels=n_feats,
                                      embedding_dim=n_feats * args.patch_dim * args.patch_dim, num_heads=args.num_heads,
                                      num_layers=args.num_layers,
                                      hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
                                      num_queries=args.num_queries, dropout_rate=args.dropout_rate)

        self._tail = nn.Sequential(
                models.basic.Upsampler(conv, 4, n_feats, act=False),
                conv(n_feats, args.n_out_colors, kernel_size),
                SELayer(8)
            )

    def forward(self, x):
        b = self._head(x)
        b = self._head1(b)

        res = self.body(b, self.scale_idx)
        res += b

        x = self._tail(res)

        return x

    def set_scale(self, scale_idx):
        self.scale_idx = scale_idx


class VisionTransformer(nn.Module):
    def __init__(
            self,
            img_dim,
            patch_dim,
            num_channels,
            embedding_dim,
            num_heads,
            num_layers,
            hidden_dim,
            num_queries,
            positional_encoding_type="learned",
            dropout_rate=0,
            no_norm=False,
            mlp=False,
            pos_every=True,
            no_pos=False,
    ):
        super(VisionTransformer, self).__init__()

        assert embedding_dim % num_heads == 0
        assert img_dim % patch_dim == 0
        self.no_norm = no_norm
        self.mlp = mlp
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.patch_dim = patch_dim
        self.num_channels = num_channels

        self.img_dim = img_dim
        self.pos_every = pos_every
        self.num_patches = int((img_dim // patch_dim) ** 2)
        self.seq_length = self.num_patches
        self.flatten_dim = patch_dim * patch_dim * num_channels

        self.out_dim = patch_dim * patch_dim * num_channels

        self.no_pos = no_pos

        if not self.mlp:
            self.linear_encoding = nn.Linear(self.flatten_dim, embedding_dim)
            self.mlp_head = nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.Dropout(dropout_rate),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.out_dim),
                nn.Dropout(dropout_rate)
            )

            self._query_embed = nn.Embedding(num_queries, embedding_dim * self.seq_length)    # 不加载预训练权重

        encoder_layer = TransformerEncoderLayer(embedding_dim, num_heads, hidden_dim, dropout_rate, self.no_norm)
        self.encoder = TransformerEncoder(encoder_layer, num_layers)

        decoder_layer = TransformerDecoderLayer(embedding_dim, num_heads, hidden_dim, dropout_rate, self.no_norm)
        self.decoder = TransformerDecoder(decoder_layer, num_layers)

        if not self.no_pos:     # 不加载预训练权重
            self._position_encoding = LearnedPositionalEncoding(
                self.seq_length, self.embedding_dim, self.seq_length
            )

        self.dropout_layer1 = nn.Dropout(dropout_rate)

        if no_norm:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=1 / m.weight.size(1))

        self.injector = Injector()
        self.emb_upsampler = nn.Upsample(scale_factor=img_dim**2/self.seq_length, mode='nearest', )

    def forward(self, x, cnn_token=0, query_idx=0, con=False):
        if self.img_dim != x.shape[-1]:
            self.emb_upsampler = nn.Upsample(scale_factor=x.shape[-1] ** 2 / self.seq_length, mode='nearest')
            self.img_dim = x.shape[-1]

        x = torch.nn.functional.unfold(x, self.patch_dim, stride=self.patch_dim).transpose(1, 2).transpose(0,
                                                                                                           1).contiguous()
        with torch.no_grad():
            if not self.mlp:
                x = self.dropout_layer1(self.linear_encoding(x)) + x
                query_embed = self._query_embed.weight[query_idx].view(-1, 1, self.embedding_dim).repeat(1, x.size(1), 1)
                query_embed = self.emb_upsampler(query_embed.permute(1, 2, 0)).permute(2, 0, 1)

            else:
                query_embed = None

            if not self.no_pos:
                pos = self._position_encoding(x).transpose(0, 1)
                pos = self.emb_upsampler(pos.permute(1, 2, 0)).permute(2, 0, 1)

            if self.pos_every:
                x = self.encoder(x, pos=pos)
            elif self.no_pos:
                x = self.encoder(x)
            else:
                x = self.encoder(x + pos)

        x = self.injector(x, cnn_token)

        with torch.no_grad():
            if self.pos_every:
                x = self.decoder(x, x, pos=pos, query_pos=query_embed)      # pos和query_embed尺度有问题
            elif self.no_pos:
                x = self.decoder(x, x, query_pos=query_embed)
            else:
                x = self.decoder(x, x, query_pos=query_embed)

            if not self.mlp:
                x = self.mlp_head(x) + x

        x = x.transpose(0, 1).contiguous().view(x.size(1), -1, self.flatten_dim)

        if con:
            con_x = x
            x = torch.nn.functional.fold(x.transpose(1, 2).contiguous(), int(self.img_dim), self.patch_dim,
                                         stride=self.patch_dim)
            return x, con_x

        x = torch.nn.functional.fold(x.transpose(1, 2).contiguous(), int(self.img_dim), self.patch_dim,
                                     stride=self.patch_dim)

        return x


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, max_position_embeddings, embedding_dim, seq_length):
        super(LearnedPositionalEncoding, self).__init__()
        self.pe = nn.Embedding(max_position_embeddings, embedding_dim)
        self.seq_length = seq_length

        self.register_buffer(
            "position_ids", torch.arange(self.seq_length).expand((1, -1))
        )

    def forward(self, x, position_ids=None):
        if position_ids is None:
            position_ids = self.position_ids[:, : self.seq_length]

        position_embeddings = self.pe(position_ids)
        return position_embeddings


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, src, pos=None):
        output = src

        for layer in self.layers:
            output = layer(output, pos=pos)

        return output


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, no_norm=False,
                 activation="relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, bias=False)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model) if not no_norm else nn.Identity()
        self.norm2 = nn.LayerNorm(d_model) if not no_norm else nn.Identity()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

        nn.init.kaiming_uniform_(self.self_attn.in_proj_weight, a=math.sqrt(5))

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, src, pos=None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, src2)
        src = src + self.dropout1(src2[0])
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)

        self.num_layers = num_layers

    def forward(self, tgt, memory, pos=None, query_pos=None):
        output = tgt

        for layer in self.layers:
            output = layer(output, memory, pos=pos, query_pos=query_pos)

        return output


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, no_norm=False,
                 activation="relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, bias=False)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, bias=False)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model) if not no_norm else nn.Identity()
        self.norm2 = nn.LayerNorm(d_model) if not no_norm else nn.Identity()
        self.norm3 = nn.LayerNorm(d_model) if not no_norm else nn.Identity()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos):      # 耗费很长时间：8s
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory, pos=None, query_pos=None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


if __name__ == '__main__':
    args = Config()
    model = ipt(args).cuda()

    # ############ 通道测试 ##############
    # HW = 64
    # pan = torch.rand(1, 9, HW, HW)
    #
    # out = model(pan)
    #
    # print(out.shape)

    from torchsummary import summary

    summary(model, (9, 64, 64))
    # ############## 加载权重 #################

    # pretrain_ipt = torch.load('..\PreWeight\IPT\IPT_pretrain.pt')
    # missing_keys, unexpected_keys = model.load_state_dict(pretrain_ipt, strict=False)
    # print("[missing_keys]:", *missing_keys, sep="\n")
    # print("[unexpected_keys]:", *unexpected_keys, sep="\n")
