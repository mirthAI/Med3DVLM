import torch.nn as nn

from .mhsa import MultiHeadSelfAttention
from .mlp import LowHighHybridMLP, MixerLowHighHybridMLP, MultiModalProjector


class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": "identity"}


def build_mm_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, "mm_projector_type", "linear")

    if projector_type == "linear":
        return nn.Linear(config.mm_hidden_size, config.hidden_size)
    elif projector_type == "mlp":
        return MultiModalProjector(
            input_size=config.mm_hidden_size,
            output_size=config.hidden_size,
            mlp_depth=config.mm_mlp_depth,
            proj_out_num=config.proj_out_num,
        )
    elif projector_type == "low_high_mlp":
        return LowHighHybridMLP(
            low_input_size=config.low_input_size,
            high_input_size=config.high_input_size,
            output_size=config.hidden_size,
            mlp_depth=config.mm_mlp_depth,
            proj_out_num=config.proj_out_num,
        )
    elif projector_type == "mixer":
        return MixerLowHighHybridMLP(
            low_input_size=config.low_input_size,
            low_output_size=config.low_output_size,
            high_input_size=config.high_input_size,
            high_output_size=config.high_output_size,
            output_dim=config.hidden_size,
            depth=len(config.low_output_size),
            mlp_depth=config.mm_mlp_depth,
            proj_out_num=config.proj_out_num,
        )
    elif projector_type == "mhsa":
        return MultiHeadSelfAttention(
            embed_dim=config.mm_hidden_size,
            output_dim=config.hidden_size,
            num_heads=hasattr(config, "num_heads") and config.num_heads or 8,
            proj_out_num=config.proj_out_num,
        )
    elif projector_type == "identity":
        return IdentityMap()
    else:
        raise ValueError(f"Unknown projector type: {projector_type}")
