# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.

"""
Modified from the official implementation of DeiT.
https://github.com/facebookresearch/deit/blob/main/models.py
"""

import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_

from ..build import MODEL
from pycls.core.config import cfg
from .base import BaseTransformerModel
from .common import PatchEmbedding, TransformerLayer, layernorm


@MODEL.register()
class DeiT(BaseTransformerModel):

    def __init__(self):
        super(DeiT, self).__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))

        self.patch_embed = PatchEmbedding(img_size=self.img_size, patch_size=self.patch_size, in_channels=self.in_channels, out_channels=self.hidden_dim)
        self.num_patches = self.patch_embed.num_patches
        self.num_tokens = 1 + cfg.DISTILLATION.ENABLE_LOGIT
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + self.num_tokens, self.hidden_dim))
        self.pe_dropout = nn.Dropout(p=self.drop_rate)

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]  # stochastic depth decay rule
        self.layers = nn.ModuleList([TransformerLayer(
            in_channels=self.hidden_dim,
            num_heads=self.num_heads,
            qkv_bias=True,
            mlp_ratio=self.mlp_ratio,
            drop_rate=self.drop_rate,
            attn_drop_rate=self.attn_drop_rate,
            drop_path_rate=dpr[i]) for i in range(self.depth)])
        self.initialize_hooks(self.layers)

        self.norm = layernorm(self.hidden_dim)
        self.apply(self._init_weights)

        self.head = nn.Linear(self.hidden_dim, self.num_classes)
        nn.init.zeros_(self.head.weight)
        nn.init.constant_(self.head.bias, 0)

        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        self.distill_logits = None

        self.distill_token = None
        self.distill_head = None
        if cfg.DISTILLATION.ENABLE_LOGIT:
            self.distill_token = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
            self.distill_head = nn.Linear(self.hidden_dim, self.num_classes)
            nn.init.zeros_(self.distill_head.weight)
            nn.init.constant_(self.distill_head.bias, 0)
            trunc_normal_(self.distill_token, std=.02)

    def _feature_hook(self, module, inputs, outputs):
        feat_size = int(self.num_patches ** 0.5)
        x = outputs[:, self.num_tokens:].view(outputs.size(0), feat_size, feat_size, self.hidden_dim)
        x = x.permute(0, 3, 1, 2).contiguous()
        self.features.append(x)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def forward(self, x):
        x = self.patch_embed(x)
        if self.num_tokens == 1:
            x = torch.cat([self.cls_token.repeat(x.size(0), 1, 1), x], dim=1)
        else:
            x = torch.cat([self.cls_token.repeat(x.size(0), 1, 1), self.distill_token.repeat(x.size(0), 1, 1), x], dim=1)
        x = self.pe_dropout(x + self.pos_embed)
        
        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        logits = self.head(x[:, 0])

        if self.num_tokens == 1:
            return logits

        self.distill_logits = None
        self.distill_logits = self.distill_head(x[:, 1])

        if self.training:
            return logits
        else:
            return (logits + self.distill_logits) / 2
