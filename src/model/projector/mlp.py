import torch
import torch.nn as nn
from einops import rearrange


class MultiLayerPerceptron(nn.Module):
    def __init__(self, hidden_size, depth):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            *[
                nn.Sequential(nn.GELU(), nn.Linear(hidden_size, hidden_size))
                for _ in range(depth - 1)
            ]
        )

    def forward(self, x):
        return self.mlp(x)


class MultiModalProjector(nn.Module):
    def __init__(self, input_size, output_size, mlp_depth, proj_out_num=256):
        super().__init__()
        self.proj_out_num = proj_out_num
        self.mm_projector = nn.Sequential(
            nn.Linear(input_size, output_size),
            *[
                nn.Sequential(
                    nn.GELU(),
                    nn.Linear(output_size, output_size),
                )
                for _ in range(mlp_depth - 1)
            ]
        )

    def forward(self, x):
        return self.mm_projector(x)


class LowHighHybridMLP(nn.Module):
    def __init__(
        self, low_input_size, high_input_size, output_size, mlp_depth, proj_out_num=288
    ):
        super().__init__()
        self.proj_out_num = proj_out_num
        self.low_up_mlp = nn.Linear(low_input_size, output_size)
        self.high_up_mlp = nn.Linear(high_input_size, output_size)
        modules = []
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(output_size, output_size))
        self.mm_projector = nn.Sequential(*modules)

    def forward(self, x):
        low_x, high_x = x

        low_x = self.low_up_mlp(low_x)
        high_x = self.high_up_mlp(high_x)
        x = torch.cat([low_x, high_x], dim=1)

        x = self.mm_projector(x)

        return x


class MixerLayer(nn.Module):
    def __init__(self, input_size, output_size, mlp_depth=2):
        super().__init__()
        self.ln1 = nn.LayerNorm(input_size[1])
        self.ln2 = nn.LayerNorm(input_size[1])

        self.mlp1 = MultiModalProjector(
            input_size=input_size[0], output_size=output_size[0], mlp_depth=mlp_depth
        )
        self.mlp2 = MultiModalProjector(
            input_size=input_size[1], output_size=output_size[1], mlp_depth=mlp_depth
        )

    def forward(self, x):
        x = self.ln1(x)
        x = rearrange(x, "b n d -> b d n")
        x = self.mlp1(x)
        x = rearrange(x, "b d n -> b n d")
        x = self.ln2(x)
        x = self.mlp2(x)

        return x


class MixerLowHighHybridMLP(nn.Module):
    def __init__(
        self,
        low_input_size: tuple = (256, 384),
        low_output_size: list = [192, 128],
        high_input_size: tuple = (32, 768),
        high_output_size: list = [64, 128],
        output_dim=3584,
        depth=2,
        mlp_depth=2,
        proj_out_num=256,
    ):
        assert (
            len(low_output_size) == len(high_output_size) == depth
        ), "Output size must be same for both low and high input"
        assert output_dim % (2**depth) == 0, "Output dim must be divisible by 2**depth"

        super().__init__()

        self.proj_out_num = proj_out_num

        self.low_mixer = nn.ModuleList(
            [
                MixerLayer(
                    input_size=(
                        (low_output_size[i - 1], output_dim // (2 ** (depth - i)))
                        if i > 0
                        else low_input_size
                    ),
                    output_size=(
                        low_output_size[i],
                        output_dim // (2 ** (depth - i - 1)),
                    ),
                    mlp_depth=mlp_depth,
                )
                for i in range(depth)
            ]
        )
        self.high_mixer = nn.ModuleList(
            [
                MixerLayer(
                    input_size=(
                        (high_output_size[i - 1], output_dim // (2 ** (depth - i)))
                        if i > 0
                        else high_input_size
                    ),
                    output_size=(
                        high_output_size[i],
                        output_dim // (2 ** (depth - i - 1)),
                    ),
                    mlp_depth=mlp_depth,
                )
                for i in range(depth)
            ]
        )

    def forward(self, x):
        low_x, high_x = x
        for low_layer, high_layer in zip(self.low_mixer, self.high_mixer):
            low_x = low_layer(low_x)
            high_x = high_layer(high_x)
        x = torch.cat([low_x, high_x], dim=1)

        return x

