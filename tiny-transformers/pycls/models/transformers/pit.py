# Copyright 2021-present NAVER Corp.
# Apache License v2.0

"""
Modified from the official implementation of PiT.
https://github.com/naver-ai/pit/blob/master/pit.py
"""

import math

import torch
import torch.nn as nn
from einops import rearrange
from timm.models.layers import trunc_normal_

from ..build import MODEL
from pycls.core.config import cfg
from .base import BaseTransformerModel
from .common import TransformerLayer, layernorm


class Transformer(nn.Module):

    def __init__(self, embed_dim, depth, heads, mlp_ratio,
                 drop_rate=.0, attn_drop_rate=.0, drop_path_prob=None):
        super(Transformer, self).__init__()
        self.layers = nn.ModuleList([])

        if drop_path_prob is None:
            drop_path_prob = [0.0 for _ in range(depth)]

        self.blocks = nn.ModuleList([
            TransformerLayer(
                in_channels=embed_dim,
                num_heads=heads,
                qkv_bias=True,
                mlp_ratio=mlp_ratio,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=drop_path_prob[i])
            for i in range(depth)])

    def forward(self, x, cls_tokens):
        h, w = x.shape[2:4]
        x = rearrange(x, 'b c h w -> b (h w) c')

        token_length = cls_tokens.shape[1]
        x = torch.cat((cls_tokens, x), dim=1)
        for blk in self.blocks:
            blk.shape_info = (token_length, h, w)
            x = blk(x)

        cls_tokens = x[:, :token_length]
        x = x[:, token_length:]
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

        return x, cls_tokens


class conv_head_pooling(nn.Module):

    def __init__(self, in_feature, out_feature, stride,
                 padding_mode='zeros'):
        super(conv_head_pooling, self).__init__()

        self.conv = nn.Conv2d(in_feature, out_feature, kernel_size=stride + 1,
                              padding=stride // 2, stride=stride,
                              padding_mode=padding_mode, groups=in_feature)
        self.fc = nn.Linear(in_feature, out_feature)

    def forward(self, x, cls_token):

        x = self.conv(x)
        cls_token = self.fc(cls_token)

        return x, cls_token


class conv_embedding(nn.Module):

    def __init__(self, in_channels, out_channels, patch_size,
                 stride, padding):
        super(conv_embedding, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=patch_size,
                              stride=stride, padding=padding, bias=True)

    def forward(self, x):
        x = self.conv(x)
        return x


@MODEL.register()
class PiT(BaseTransformerModel):

    def __init__(self):
        super(PiT, self).__init__()
        self.stride = cfg.PIT.STRIDE
        total_block = sum(self.depth)
        block_idx = 0
        embed_size = math.floor((self.img_size - self.patch_size) / self.stride + 1)

        self.pos_embed = nn.Parameter(
            torch.randn(1, self.hidden_dim[0], embed_size, embed_size),
            requires_grad=True
        )
        self.patch_embed = conv_embedding(self.in_channels, self.hidden_dim[0],
                                          self.patch_size, self.stride, 0)

        self.cls_token = nn.Parameter(torch.randn(1, 1, self.hidden_dim[0]))
        self.pos_drop = nn.Dropout(p=self.drop_rate)

        self.transformers = nn.ModuleList([])
        self.pools = nn.ModuleList([])

        for stage in range(len(self.depth)):
            drop_path_prob = [self.drop_path_rate * i / total_block
                              for i in range(block_idx, block_idx + self.depth[stage])]
            block_idx += self.depth[stage]

            self.transformers.append(
                Transformer(self.hidden_dim[stage], self.depth[stage], self.num_heads[stage],
                            self.mlp_ratio,
                            self.drop_rate, self.attn_drop_rate, drop_path_prob)
            )
            if stage < len(self.depth) - 1:
                self.pools.append(
                    conv_head_pooling(self.hidden_dim[stage], self.hidden_dim[stage + 1], stride=2))

        layers = [[m for m in t.blocks] for t in self.transformers]
        layers = sum(layers, [])
        self.initialize_hooks(layers)

        self.norm = layernorm(self.hidden_dim[-1])
        self.embed_dim = self.hidden_dim[-1]

        self.head = nn.Linear(self.hidden_dim[-1], self.num_classes)

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _feature_hook(self, module, inputs, outputs):
        token_length, h, w = module.shape_info
        x = outputs[:, token_length:]
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        self.features.append(x)

    def _init_weights(self, m):
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        x = self.patch_embed(x)

        pos_embed = self.pos_embed
        x = self.pos_drop(x + pos_embed)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)

        for stage in range(len(self.pools)):
            x, cls_tokens = self.transformers[stage](x, cls_tokens)
            x, cls_tokens = self.pools[stage](x, cls_tokens)
        x, cls_tokens = self.transformers[-1](x, cls_tokens)

        cls_tokens = self.norm(cls_tokens)

        return cls_tokens

    def forward(self, x):
        cls_token = self.forward_features(x)
        cls_token = self.head(cls_token[:, 0])
        return cls_token
