import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def readable_params(num):
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return "%.2f%s" % (num, ["", "K", "M", "G", "T", "P"][magnitude])


class MLPLayer(nn.Module):
    def __init__(self, embed_dim, scale=4, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.linear1 = nn.Linear(embed_dim, embed_dim * scale)
        self.linear2 = nn.Linear(embed_dim * scale, embed_dim)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.act(x)
        x = self.linear2(x)
        return x


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        output_dim,
        num_heads=8,
        proj_out_num=32,
    ):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.proj_out_num = proj_out_num
        self.mlp = MLPLayer(embed_dim)

        assert (
            self.head_dim * num_heads == embed_dim
        ), "embed_dim must be divisible by num_heads"

        self.norm1 = nn.LayerNorm(embed_dim)

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.norm2 = nn.LayerNorm(embed_dim)
        self.act = nn.GELU()
        self.out_layer = nn.Linear(embed_dim, output_dim)

        self.scale = math.sqrt(self.head_dim)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        x = self.norm1(x)

        q = (
            self.q_proj(x)
            .reshape(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(x)
            .reshape(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(x)
            .reshape(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        attn_weights = F.softmax(attn_weights, dim=-1)

        attn_output = torch.matmul(attn_weights, v)

        attn_output = attn_output.transpose(1, 2).reshape(
            batch_size, seq_len, self.embed_dim
        )
        output = self.out_proj(attn_output)

        output = self.norm2(output)

        output = self.mlp(output)

        output = self.act(output)
        output = self.out_layer(output)

        return output
