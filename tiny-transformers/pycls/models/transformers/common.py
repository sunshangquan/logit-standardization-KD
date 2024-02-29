import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath

from pycls.core.config import cfg


def layernorm(w_in):
    return nn.LayerNorm(w_in, eps=cfg.TRANSFORMER.LN_EPS)


class MultiheadAttention(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_heads,
                 qkv_bias=False,
                 attn_drop_rate=0.,
                 proj_drop_rate=0.,
                 qk_scale=None):
        super(MultiheadAttention, self).__init__()
        assert out_channels % num_heads == 0
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads

        self.norm_factor = qk_scale if qk_scale else (out_channels // num_heads) ** -0.5
        self.qkv_transform = nn.Linear(in_channels, out_channels * 3, bias=qkv_bias)
        self.projection = nn.Linear(out_channels, out_channels)
        self.attention_dropout = nn.Dropout(attn_drop_rate)
        self.projection_dropout = nn.Dropout(proj_drop_rate)

    def forward(self, x):
        N, L, _ = x.shape
        x = self.qkv_transform(x).view(N, L, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        query, key, value = x[0], x[1], x[2]

        qk = query @ key.transpose(-1, -2) * self.norm_factor
        qk = F.softmax(qk, dim=-1)
        qk = self.attention_dropout(qk)

        out = qk @ value
        out = out.transpose(1, 2).contiguous().view(N, L, self.out_channels)
        out = self.projection(out)
        out = self.projection_dropout(out)
        
        if self.in_channels != self.out_channels:
            out = out + value.squeeze(1)

        return out


class MLP(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 drop_rate=0.,
                 hidden_ratio=1.):
        super(MLP, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = int(in_channels * hidden_ratio)
        self.fc1 = nn.Linear(in_channels, self.hidden_channels)
        self.fc2 = nn.Linear(self.hidden_channels, out_channels)
        self.drop = nn.Dropout(drop_rate)

    def forward(self, x):
        x = F.gelu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerLayer(nn.Module):

    def __init__(self,
                 in_channels,
                 num_heads,
                 qkv_bias=False,
                 out_channels=None,
                 mlp_ratio=1.,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 qk_scale=None):
        super(TransformerLayer, self).__init__()
        if out_channels is None:
            out_channels = in_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.norm1 = layernorm(in_channels)
        self.attn = MultiheadAttention(
            in_channels=in_channels,
            out_channels=out_channels,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=drop_rate,
            qk_scale=qk_scale)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()
        self.norm2 = layernorm(out_channels)
        self.mlp = MLP(
            in_channels=out_channels,
            out_channels=out_channels,
            drop_rate=drop_rate,
            hidden_ratio=mlp_ratio)

    def forward(self, x):
        if self.in_channels == self.out_channels:
            x = x + self.drop_path(self.attn(self.norm1(x)))
        else:
            x = self.attn(self.norm1(x))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbedding(nn.Module):

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_channels=3,
                 out_channels=768):
        super(PatchEmbedding, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.projection = nn.Conv2d(in_channels, out_channels, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        _, _, H, W = x.shape
        assert H == self.img_size and W == self.img_size
        x = self.projection(x)
        x = x.flatten(2).transpose(1, 2)
        return x
