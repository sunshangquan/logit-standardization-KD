# Copyright (c) [2012]-[2021] Shanghai Yitu Technology Co., Ltd.
#
# This source code is licensed under the Clear BSD License
# LICENSE file in the root directory of this file
# All rights reserved.

"""
Modified from the official implementation of T2T-ViT.
https://github.com/yitu-opensource/T2T-ViT/blob/main/models/t2t_vit.py
"""

import math
import torch
import numpy as np
import torch.nn as nn

from ..build import MODEL
from pycls.core.config import cfg
from .base import BaseTransformerModel
from .common import MLP, TransformerLayer, layernorm


class PerformerAttention(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_heads,
                 drop_rate=0.1,
                 kernel_ratio=0.5):
        super(PerformerAttention, self).__init__()
        assert out_channels % num_heads == 0
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.head_channels = out_channels // num_heads

        self.qkv_transform = nn.Linear(in_channels, out_channels * 3)
        self.projection = nn.Linear(out_channels, out_channels)
        self.dropout = nn.Dropout(drop_rate)

        self.m = int(out_channels * kernel_ratio)
        self.w = torch.randn(self.m, out_channels)
        self.w = nn.Parameter(nn.init.orthogonal_(self.w) * math.sqrt(self.m), requires_grad=False)

    def prm_exp(self, x):
        xd = ((x * x).sum(dim=-1, keepdim=True)).repeat(1, 1, self.m) / 2
        wtx = torch.einsum('bti,mi->btm', x.float(), self.w)
        return torch.exp(wtx - xd) / math.sqrt(self.m)

    def forward(self, x):
        k, q, v = torch.split(self.qkv_transform(x), self.head_channels, dim=-1)
        kp, qp = self.prm_exp(k), self.prm_exp(q)
        D = torch.einsum('bti,bi->bt', qp, kp.sum(dim=1)).unsqueeze(dim=2)
        kptv = torch.einsum('bin,bim->bnm', v.float(), kp)
        y = torch.einsum('bti,bni->btn', qp, kptv) / (D.repeat(1, 1, self.head_channels) + 1e-8)
        y = self.dropout(self.projection(y))
        if self.in_channels != self.out_channels:
            y = v + y
        return y


class PerformerLayer(nn.Module):

    def __init__(self,
                 in_channels,
                 num_heads,
                 out_channels=None,
                 hidden_ratio=1.,
                 drop_rate=0.,
                 kernel_ratio=0.5):
        super(PerformerLayer, self).__init__()
        if out_channels is None:
            out_channels = in_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.norm1 = layernorm(in_channels)
        self.attn = PerformerAttention(
            in_channels=in_channels,
            out_channels=out_channels,
            num_heads=num_heads,
            drop_rate=drop_rate,
            kernel_ratio=kernel_ratio)
        self.norm2 = layernorm(out_channels)
        self.mlp = MLP(
            in_channels=out_channels,
            out_channels=out_channels,
            drop_rate=drop_rate,
            hidden_ratio=hidden_ratio)

    def forward(self, x):
        if self.in_channels == self.out_channels:
            x = x + self.attn(self.norm1(x))
        else:
            x = self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class Token2TokenModule(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 img_size):
        super(Token2TokenModule, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.img_size = (img_size, img_size)
        self.token_channels = cfg.T2T.TOKEN_DIM
        self.kernel_size = cfg.T2T.KERNEL_SIZE
        self.stride = cfg.T2T.STRIDE
        self.padding = cfg.T2T.PADDING
        assert len(self.kernel_size) == len(self.stride)

        self.soft_split0 = nn.Unfold(
            kernel_size=self.kernel_size[0],
            stride=self.stride[0],
            padding=self.padding[0])

        self.soft_split = nn.ModuleList()
        self.attention = nn.ModuleList()
        cur_channels = in_channels * self.kernel_size[0] ** 2
        for i in range(1, len(self.kernel_size)):
            soft_split, attention = self._make_layer(
                in_channels=cur_channels,
                out_channels=self.token_channels,
                kernel_size=self.kernel_size[i],
                stride=self.stride[i],
                padding=self.padding[i])
            self.soft_split.append(soft_split)
            self.attention.append(attention)
            cur_channels = self.token_channels * self.kernel_size[i] ** 2
        self.projection = nn.Linear(cur_channels, out_channels)

    def _make_layer(self, in_channels, out_channels, kernel_size, stride, padding):
        soft_split = nn.Unfold(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding)
        attention = PerformerLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            num_heads=1,
            hidden_ratio=1,
            kernel_ratio=0.5)
        return soft_split, attention

    def forward(self, x):
        H, W = x.shape[-2:]
        ratio = H / W
        x = self.soft_split0(x).transpose(-1, -2)
        for attention, soft_split in zip(self.attention, self.soft_split):
            x = attention(x).transpose(-1, -2)
            N, C, L = x.shape
            W = int(L ** 0.5 / ratio)
            H = L // W
            x = x.view(N, C, H, W)
            x = soft_split(x).transpose(-1, -2)
        x = self.projection(x)
        return x


@MODEL.register()
class T2TViT(BaseTransformerModel):

    def __init__(self):
        super(T2TViT, self).__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))

        self.t2t_module = Token2TokenModule(
            in_channels=self.in_channels,
            out_channels=self.hidden_dim,
            img_size=self.img_size)

        feat_size = self.img_size
        for stride in cfg.T2T.STRIDE:
            feat_size = feat_size // stride
        self.num_patches = feat_size ** 2
        pe = self._get_position_embedding(self.num_patches + 1, self.hidden_dim)
        self.pos_embed = nn.Parameter(pe, requires_grad=False)
        self.pe_dropout = nn.Dropout(self.drop_rate)

        self.layers = nn.ModuleList()
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.layers.extend([TransformerLayer(
            in_channels=self.hidden_dim,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            drop_rate=self.drop_rate,
            attn_drop_rate=self.attn_drop_rate,
            drop_path_rate=dpr[i]) for i in range(self.depth)])
        self.initialize_hooks(self.layers)

        self.norm = layernorm(self.hidden_dim)
        self.head = nn.Linear(self.hidden_dim, self.num_classes)

        nn.init.normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _feature_hook(self, module, inputs, outputs):
        feat_size = int(self.num_patches ** 0.5)
        x = outputs[:, 1:].view(outputs.size(0), feat_size, feat_size, self.hidden_dim)
        x = x.permute(0, 3, 1, 2).contiguous()
        self.features.append(x)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _get_position_embedding(self, n_position, d_hid):

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        x = self.t2t_module(x)
        x = torch.cat([self.cls_token.repeat(x.size(0), 1, 1), x], dim=1)
        x = self.pe_dropout(x + self.pos_embed)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        x = self.head(x[:, 0])

        return x
