from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from einops import pack, rearrange, repeat, unpack
from timm.layers import DropPath, to_3tuple, trunc_normal_


def stem(inp, oup, image_size, downsample=False):
    stride = 1 if downsample == False else 2
    return nn.Sequential(
        nn.Conv3d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm3d(oup),
        nn.GELU(),
        nn.Conv3d(oup, oup, 3, 1, 1, bias=False),
        nn.BatchNorm3d(oup),
        nn.GELU(),
    )


def DecomposedStem(inp, oup, image_size, kernel_size, downsample=False):
    return nn.Sequential(
        DecompConv3D(inp, oup, 7, 4, 1, nn.GELU()),
        DecompConv3D(oup, oup, 3, 1, 1, nn.GELU()),
        DecompConv3D(oup, oup, 3, 1, 1, nn.GELU()),
        DecompConv3D(oup, oup, 3, 1, 1, nn.GELU()),
    )


class DecompConv3D(nn.Module):
    def __init__(
        self, in_dim, out_dim, kernel_size, stride=1, groups=1, norm=True, act=None
    ) -> None:
        super().__init__()
        self.act = act

        self.c1 = nn.Sequential(
            nn.Conv3d(
                in_dim,
                out_dim,
                kernel_size=(kernel_size, 1, 1),
                padding=(kernel_size // 2, 0, 0),
                stride=stride,
                groups=groups,
            ),
            nn.BatchNorm3d(out_dim) if norm else nn.Identity(),
        )
        self.c2 = nn.Sequential(
            nn.Conv3d(
                in_dim,
                out_dim,
                kernel_size=(1, kernel_size, 1),
                padding=(0, kernel_size // 2, 0),
                stride=stride,
                groups=groups,
            ),
            nn.BatchNorm3d(out_dim) if norm else nn.Identity(),
        )
        self.c3 = nn.Sequential(
            nn.Conv3d(
                in_dim,
                out_dim,
                kernel_size=(1, 1, kernel_size),
                padding=(0, 0, kernel_size // 2),
                stride=stride,
                groups=groups,
            ),
            nn.BatchNorm3d(out_dim) if norm else nn.Identity(),
        )

    def forward(self, x):
        x = self.c1(x) + self.c2(x) + self.c3(x)
        if self.act is not None:
            x = self.act(x)
        return x


class ConvPosEnc(nn.Module):

    def __init__(self, dim, k=3, decompose=False):
        super().__init__()
        if decompose:
            self.proj = DecompConv3D(dim, dim, k, groups=dim, norm=None)
        else:
            self.proj = nn.Conv3d(
                dim, dim, to_3tuple(k), to_3tuple(1), to_3tuple(k // 2), groups=dim
            )

    def forward(self, x, size):
        B, N, C = x.shape
        H, W, T = size
        assert N == H * W * T
        feat = rearrange(x, "b (h w t) c -> b c h w t", h=H, w=W, t=T)
        feat = self.proj(feat)
        feat = rearrange(feat, "b c h w t -> b (h w t) c ")
        x = x + feat
        return x


class MLP(nn.Module):
    def __init__(self, oup, mlp_dim, dp=0.0):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(oup, mlp_dim),
            nn.GELU(),
            nn.Dropout(dp),
            nn.Linear(mlp_dim, oup),
            nn.Dropout(dp),
        )

    def forward(self, x):
        x = self.mlp(x)
        return x


class ScaleDotProduct(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, qkv):
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = self.softmax(attn)

        x = (attn @ v).transpose(1, 2)
        return x


class DecomposedAttention(nn.Module):
    def __init__(self, oup, head_num):
        super().__init__()

        self.head_num = head_num
        scale = (oup // head_num) ** (1 / 2)
        self.sdp = ScaleDotProduct(scale)
        self.qkv = nn.Linear(oup, oup * 3, bias=False)
        self.proj = nn.Linear(oup, oup, bias=False)

    def forward(self, x, size):
        b, n, c = x.shape
        h, w, t = size
        assert n == h * w * t
        B_, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B_, N, 3, self.head_num, C // self.head_num)
            .permute(2, 0, 3, 1, 4)
        )

        x = rearrange(qkv, "k b nh (h w t) c -> k b c nh h w t", h=h, w=w, t=t)

        x1 = rearrange(x, "k b c nh h w t -> k (b t) nh (h w) c")
        x2 = rearrange(x, "k b c nh h w t -> k (b w) nh (h t) c")
        x3 = rearrange(x, "k b c nh h w t -> k (b h) nh (w t) c")

        x1 = self.sdp(x1)
        x2 = self.sdp(x2)
        x3 = self.sdp(x3)

        x1 = rearrange(x1, "(b t) (h w) nh c -> b (h w t) (nh c)", h=h, w=w, t=t)
        x2 = rearrange(x2, "(b w) (h t) nh c -> b (h w t) (nh c)", h=h, w=w, t=t)
        x3 = rearrange(x3, "(b h) (w t) nh c -> b (h w t) (nh c)", h=h, w=w, t=t)
        x = self.proj(x1 + x2 + x3)

        return x


class SelfAttention(nn.Module):
    def __init__(self, oup, head_num):
        super().__init__()

        self.head_num = head_num
        scale = (oup // head_num) ** (1 / 2)
        self.sdp = ScaleDotProduct(scale)
        self.qkv = nn.Linear(oup, oup * 3, bias=False)
        self.proj = nn.Linear(oup, oup, bias=False)

    def forward(self, x, size=None):
        B_, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B_, N, 3, self.head_num, C // self.head_num)
            .permute(2, 0, 3, 1, 4)
        )

        x = self.sdp(qkv).reshape(B_, N, C)
        x = self.proj(x)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, oup, head_num):
        super().__init__()

        self.head_num = head_num
        self.scale = (oup // head_num) ** (1 / 2)
        self.softmax = nn.Softmax(dim=-1)

        self.qkv = nn.Linear(oup, oup * 3, bias=False)
        self.proj = nn.Linear(oup, oup, bias=False)

    def forward(self, x):
        B, N, C = x.shape

        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.head_num, C // self.head_num)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        k = k * self.scale
        attention = k.transpose(-1, -2) @ v
        attention = self.softmax(attention)
        x = (attention @ q.transpose(-1, -2)).transpose(-1, -2)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class ChannelBlock(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        hidden_dim = int(dim * 4)

        self.cpe = nn.ModuleList(
            [
                ConvPosEnc(dim=dim, k=3, decompose=True),
                ConvPosEnc(dim=dim, k=3, decompose=True),
            ]
        )

        self.attn = ChannelAttention(dim, heads)

        self.layer_norm1 = nn.LayerNorm(dim)

        self.mlp1 = MLP(dim, hidden_dim)
        self.layer_norm2 = nn.LayerNorm(dim)

    def forward(self, x, size):
        x = self.cpe[0](x, size)
        _x = self.layer_norm1(x)

        _x = self.attn(_x)
        x = x + _x

        x = self.cpe[1](x, size)
        _x = self.layer_norm2(x)
        _x = self.mlp1(_x)
        x = x + _x
        return x


class SpatialBlock(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        hidden_dim = int(dim * 4)

        self.cpe = nn.ModuleList(
            [
                ConvPosEnc(dim=dim, k=3, decompose=True),
                ConvPosEnc(dim=dim, k=3, decompose=True),
            ]
        )

        # self.attn = DecomposedAttention(dim, heads)
        self.attn = SelfAttention(dim, heads)

        self.layer_norm1 = nn.LayerNorm(dim)

        self.mlp1 = MLP(dim, hidden_dim)
        self.layer_norm2 = nn.LayerNorm(dim)

    def forward(self, x, size):
        x = self.cpe[0](x, size)
        _x = self.layer_norm1(x)

        _x = self.attn(_x, size)
        x = x + _x

        x = self.cpe[1](x, size)
        _x = self.layer_norm2(x)
        _x = self.mlp1(_x)
        x = x + _x
        return x


class TransformerBlock(nn.Module):
    def __init__(
        self,
        inp,
        oup,
        image_size,
        kernel_size,
        heads=8,
        dim_head=32,
        downsample=False,
        dropout=0.0,
    ):
        super().__init__()
        hidden_dim = int(inp * 4)

        self.ih, self.iw, self.it = image_size
        self.downsample = downsample

        if self.downsample:
            self.pool1 = nn.MaxPool3d(3, 2, 1)
            # self.pool2 = nn.MaxPool3d(3, 2, 1)
            self.proj = nn.Conv3d(inp, oup, 1, 1, 0, bias=False)

        self.spatial_attention = SpatialBlock(oup, heads)
        # self.channel_attention = ChannelBlock(oup, heads)

    def forward(self, x):
        if self.downsample:
            x = self.proj(self.pool1(x))
        # if self.downsample:
        #     x = self.proj(self.pool1(x)) + self.attn(self.pool2(x))

        # x = x.permute(0, 2, 3, 4, 1)
        h, w, t = x.shape[2], x.shape[3], x.shape[4]
        size = (h, w, t)
        x = rearrange(x, "b c h w t -> b (h w t) c ")

        x = self.spatial_attention(x, size)
        # x = self.channel_attention(x, size)

        x = rearrange(x, "b (h w t) c -> b c h w t", h=h, w=w, t=t)

        return x


class ConvBlock(nn.Module):
    def __init__(
        self, inp, oup, image_size, kernel_size=7, downsample=False, expansion=4
    ):
        super().__init__()
        self.downsample = downsample
        stride = 1 if self.downsample == False else 2
        hidden_dim = int(oup * expansion)
        drop_path = 0.0
        layer_scale_init_value = 1e-6
        if self.downsample:
            self.pool = nn.MaxPool3d(3, 2, 1)
            self.proj = nn.Conv3d(inp, oup, 1, 1, 0, bias=False)

        # self.dwconv = nn.Sequential(nn.Conv3d(oup, oup, kernel_size=7, padding=3, groups=oup), nn.BatchNorm3d(oup)) # depthwise conv
        self.dwconv = DecompConv3D(oup, oup, kernel_size, groups=oup)
        self.mlp = MLP(oup, hidden_dim)

        self.scale = (
            nn.Parameter(layer_scale_init_value * torch.ones((oup)), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        if self.downsample:
            x = self.proj(self.pool(x))
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 4, 1)  # (N, C, H, W, T) -> (N, H, W, T, C)

        x = self.mlp(x)

        if self.scale is not None:
            x = self.scale * x
        x = x.permute(0, 4, 1, 2, 3)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        input_size,
        in_channels,
        num_blocks,
        channels,
        kernel_sizes=[7, 7, 7, 7],
        block_types=["C", "C", "C", "C"],
    ):
        super().__init__()
        self.dims = channels
        ih, iw, it = input_size
        block = {"C": ConvBlock, "T": TransformerBlock}
        i = 4
        self.s0 = self._make_layer(
            DecomposedStem,
            in_channels,
            channels[0],
            num_blocks[0],
            kernel_sizes[0],
            (ih // i, iw // i, it // i),
        )
        self.s1 = self._make_layer(
            block[block_types[0]],
            channels[0],
            channels[1],
            num_blocks[1],
            kernel_sizes[0],
            (ih // (i * 2**1), iw // (i * 2**1), it // (i * 2**1)),
        )
        self.s2 = self._make_layer(
            block[block_types[1]],
            channels[1],
            channels[2],
            num_blocks[2],
            kernel_sizes[1],
            (ih // (i * 2**2), iw // (i * 2**2), it // (i * 2**2)),
        )
        self.s3 = self._make_layer(
            block[block_types[2]],
            channels[2],
            channels[3],
            num_blocks[3],
            kernel_sizes[2],
            (ih // (i * 2**3), iw // (i * 2**3), it // (i * 2**3)),
        )
        self.s4 = self._make_layer(
            block[block_types[3]],
            channels[3],
            channels[4],
            num_blocks[4],
            kernel_sizes[3],
            (ih // (i * 2**4), iw // (i * 2**4), it // (i * 2**4)),
        )

    def forward(self, x):
        hidden_states = []

        x = x.permute(0, 1, 3, 4, 2)

        for i in range(5):
            if hasattr(self, "s" + str(i)):
                x = getattr(self, "s" + str(i))(x)
                hidden_states.append(x)

        return hidden_states

    def _make_layer(self, block, inp, oup, depth, kernel_size, image_size):
        layers = nn.ModuleList([])
        for i in range(depth):
            if i == 0:
                layers.append(block(inp, oup, image_size, kernel_size, downsample=True))
            else:
                layers.append(block(oup, oup, image_size, kernel_size))
        return nn.Sequential(*layers)


class DecompModel(nn.Module):
    def __init__(
        self,
        input_size=(512, 512, 256),
        in_channels=1,
        num_blocks=[2, 2, 3, 5, 2],
        channels=[64, 96, 192, 384, 768],
        # kernel_sizes=[7, 7, 7, 7],
        kernel_sizes=[13, 11, 9, 7],
        block_types=["C", "C", "C", "C"],
        codebook_size=8192,
    ):
        super().__init__()
        self.channels = channels
        self.encoder = Encoder(
            input_size, in_channels, num_blocks, channels, kernel_sizes, block_types
        )
        # self.vq = VectorQuantize(dim = channels[-1], codebook_size = codebook_size, use_cosine_sim = True)

    def forward(self, video, mask=None, device="cuda"):
        hidden_states = self.encoder(video)
        # tokens = rearrange(tokens, "b d h w t -> b t h w d")
        # shape = tokens.shape
        # *_, h, w, _ = shape
        # quantize

        # tokens, _ = pack([tokens], "b * d")

        for i in range(len(hidden_states)):
            hidden_states[i] = rearrange(hidden_states[i], "b d h w t -> b t h w d")
            hidden_states[i], _ = pack([hidden_states[i]], "b * d")

        # vq_mask = None

        # tokens, _, _ = self.vq(tokens, mask = vq_mask)

        # tokens = rearrange(tokens, 'b (t h w) d -> b t h w d', h = h, w = w)

        return hidden_states


def decomp_nano(
    input_size=(512, 512, 256),
    # input_size=(256, 256, 128),
):

    model = DecompModel(
        input_size=input_size,
        num_blocks=[1, 1, 1, 1, 1],
        channels=[32, 32, 64, 128, 256],
    )
    return model


def decomp_naive(
    input_size=(512, 512, 256),
    # input_size=(256, 256, 128),
):

    model = DecompModel(
        input_size=input_size,
        num_blocks=[1, 2, 2, 2, 2],
        # channels = [64, 64, 128, 256, 512]
        channels=[32, 64, 128, 256, 512],
    )
    return model


def decomp_tiny(
    input_size=(512, 512, 256),
):

    model = DecompModel(
        input_size=input_size,
        num_blocks=[1, 2, 3, 3, 2],
        # channels = [64, 64, 128, 256, 512]
        channels=[64, 96, 192, 384, 768],
    )
    return model


def decomp_small(
    input_size=(512, 512, 256),
):

    model = DecompModel(
        input_size=input_size,
        num_blocks=[1, 2, 3, 6, 2],
        channels=[64, 96, 192, 384, 768],
    )
    return model


def decomp_base(
    input_size=(512, 512, 256),
):

    model = DecompModel(
        input_size=input_size,
        num_blocks=[1, 2, 6, 6, 2],
        channels=[64, 128, 256, 512, 1024],
    )
    return model


def decomp_large(
    input_size=(512, 512, 256),
):

    model = DecompModel(
        input_size=input_size,
        num_blocks=[1, 2, 6, 12, 2],
        # channels=[64, 192, 384, 768, 1536],
        channels=[64, 256, 512, 1024, 2048],
    )
    return model
