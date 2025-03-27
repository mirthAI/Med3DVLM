import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, PretrainedConfig, PreTrainedModel

from src.model.encoder.dcformer import (
    decomp_base,
    decomp_naive,
    decomp_nano,
    decomp_small,
    decomp_tiny,
)
from src.model.encoder.vit import Vit3D
from src.model.projector.mlp import MultiLayerPerceptron

try:
    import torch.distributed.nn
    from torch import distributed as dist

    has_distributed = True
except ImportError:
    has_distributed = False


class DEC_CLIPConfig(PretrainedConfig):
    model_type = "dec_clip"

    def __init__(
        self,
        language_model_name_or_path: str = "",
        local_loss: bool = False,
        gather_loss: bool = True,
        input_size: tuple = (256, 256, 128),
        dim: int = 768,
        depth: int = 12,
        hidden_size: int = 512,
        mlp_depth: int = 2,
        loss_type: str = "nce",
        t_prime: float = np.log(1 / 0.07),
        bias: float = 0.0,
        efficient_loss: bool = False,
        **kwargs,
    ):
        self.language_model_name_or_path = language_model_name_or_path
        self.input_size = input_size
        self.dim = dim
        self.depth = depth
        self.hidden_size = hidden_size
        self.mlp_depth = mlp_depth
        self.local_loss = local_loss
        self.gather_loss = gather_loss
        self.loss_type = loss_type
        self.t_prime = t_prime
        self.bias = bias
        self.efficient_loss = efficient_loss
        super().__init__(**kwargs)


class DEC_CLIP(PreTrainedModel):
    config_class = DEC_CLIPConfig

    def __init__(self, config):
        super().__init__(config)

        self.config = config

        if config.vision_encoder == "vit3d":
            self.vision_encoder = Vit3D(
                input_size=config.input_size,
                dim=config.dim,
                depth=config.depth,
            )
        elif config.vision_encoder == "dcformer":
            self.vision_encoder = decomp_small(input_size=config.input_size)
        else:
            raise ValueError(f"Unexpected vision encoder: {config.vision_encoder}")

        self.language_encoder = AutoModel.from_pretrained(
            config.language_model_name_or_path
        )

        self.mm_vision_proj = nn.Linear(
            self.vision_encoder.channels[-1], config.hidden_size
        )
        self.mm_language_proj = nn.Linear(
            self.language_encoder.config.dim, config.hidden_size
        )

        self.efficient_loss = config.efficient_loss
        self.local_loss = config.local_loss
        self.gather_loss = config.gather_loss
        self.loss_type = config.loss_type

        if self.loss_type == "sigmoid":
            self.t_prime = nn.Parameter(torch.tensor(config.t_prime))
            self.bias = nn.Parameter(torch.tensor(config.bias))
        else:
            self.logit_scale = nn.Parameter(torch.ones([]) * config.t_prime)

    def encode_image(self, image):
        image_feats = self.vision_encoder(image)
        if isinstance(image_feats, list):
            image_feats = image_feats[-1]
        image_feats = image_feats.mean(dim=1)
        image_feats = self.mm_vision_proj(image_feats)
        image_feats = F.normalize(image_feats, dim=-1)

        return image_feats

    def encode_text(self, input_id, attention_mask):
        text_feats = self.language_encoder(input_id, attention_mask=attention_mask)[
            "last_hidden_state"
        ]
        text_feats = text_feats[:, 0]
        text_feats = self.mm_language_proj(text_feats)
        text_feats = F.normalize(text_feats, dim=-1)

        return text_feats

    def forward(self, images, input_ids, attention_mask, labels, **kwargs):
        image_features = self.encode_image(images)
        text_features = self.encode_text(input_ids, attention_mask)

        rank = 0
        world_size = 1
        if has_distributed and dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()

        batch_size = image_features.size(0)
        device = image_features.device
        if self.loss_type == "sigmoid":
            if has_distributed and dist.is_initialized():
                if self.efficient_loss:
                    t = torch.exp(self.t_prime)
                    loss = 0.0

                    for target_rank in range(world_size):
                        if rank == target_rank:
                            target_text_features = text_features
                        else:
                            target_text_features = torch.distributed.nn.broadcast(
                                text_features.requires_grad_(), target_rank
                            )

                        local_logits_per_image = (
                            image_features @ target_text_features.T
                        ) * t + self.bias
                        local_logits_per_text = local_logits_per_image.T

                        if rank == target_rank:
                            local_labels = 2 * torch.eye(
                                batch_size, device=device
                            ) - torch.ones(batch_size, batch_size, device=device)
                        else:
                            local_labels = -torch.ones(
                                batch_size, batch_size, device=device
                            )

                        local_logits = (
                            local_logits_per_image + local_logits_per_text
                        ) / 2.0
                        local_loss = -torch.sum(
                            F.logsigmoid(local_labels * local_logits)
                        ) / (batch_size * world_size)

                        loss += local_loss

                    torch.distributed.nn.all_reduce(loss)
                    torch.cuda.synchronize()

                    if self.training:
                        logits = 0
                else:
                    t = torch.exp(self.t_prime)

                    all_image_features, all_text_features = gather_features(
                        image_features,
                        text_features,
                        gather_with_grad=True,
                        rank=rank,
                        world_size=world_size,
                    )

                    logits_per_image = (
                        all_image_features @ all_text_features.T
                    ) * t + self.bias
                    logits_per_text = logits_per_image.T
                    batch_size = all_image_features.size(0)

                    labels = 2 * torch.eye(
                        batch_size, device=image_features.device
                    ) - torch.ones(batch_size, device=image_features.device)

                    logits = (logits_per_image + logits_per_text) / 2.0
                    loss = -torch.sum(F.logsigmoid(labels * logits)) / batch_size

            else:
                logits_per_image = (
                    image_features @ text_features.T
                ) * self.t_prime + self.bias
                logits_per_text = logits_per_image.T

                labels = 2 * torch.eye(batch_size, device=device) - torch.ones(
                    batch_size, batch_size, device=device
                )

                logits = (logits_per_image + logits_per_text) / 2.0
                loss = -torch.sum(F.logsigmoid(logits * labels))
        else:
            all_image_features, all_text_features = gather_features(
                image_features,
                text_features,
                local_loss=self.local_loss,
                gather_with_grad=True,
                rank=rank,
                world_size=world_size,
            )

            if self.gather_loss:
                if self.local_loss:
                    logits_per_image = (
                        self.logit_scale * image_features @ all_text_features.T
                    )
                    logits_per_text = (
                        self.logit_scale * text_features @ all_image_features.T
                    )
                else:
                    logits_per_image = (
                        self.logit_scale * all_image_features @ all_text_features.T
                    )
                    logits_per_text = logits_per_image.T
            else:
                logits_per_image = self.logit_scale * image_features @ text_features.T
                logits_per_text = self.logit_scale * text_features @ image_features.T

            image_loss = F.cross_entropy(logits_per_image, labels)
            text_loss = F.cross_entropy(logits_per_text, labels)

            loss = (image_loss + text_loss) / 2.0
            logits = ((logits_per_image + logits_per_text) / 2.0,)

        ret = {
            "loss": loss,
            "logits": logits,
        }

        return ret


def gather_features(
    image_features,
    text_features,
    local_loss=False,
    gather_with_grad=True,
    rank=0,
    world_size=1,
):
    assert (
        has_distributed
    ), "torch.distributed did not import correctly, please use a PyTorch version with support."

    if not (has_distributed and dist.is_initialized()):
        return image_features, text_features

    if gather_with_grad:
        all_image_features = torch.cat(
            torch.distributed.nn.all_gather(image_features), dim=0
        )
        all_text_features = torch.cat(
            torch.distributed.nn.all_gather(text_features), dim=0
        )
    else:
        gathered_image_features = [
            torch.zeros_like(image_features) for _ in range(world_size)
        ]
        gathered_text_features = [
            torch.zeros_like(text_features) for _ in range(world_size)
        ]
        dist.all_gather(gathered_image_features, image_features)
        dist.all_gather(gathered_text_features, text_features)
        if not local_loss:
            gathered_image_features[rank] = image_features
            gathered_text_features[rank] = text_features
        all_image_features = torch.cat(gathered_image_features, dim=0)
        all_text_features = torch.cat(gathered_text_features, dim=0)

    return all_image_features, all_text_features


AutoConfig.register("dec_clip", DEC_CLIPConfig)
AutoModel.register(DEC_CLIPConfig, DEC_CLIP)
