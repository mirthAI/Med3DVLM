from abc import ABC, abstractmethod

import torch
from transformers import AutoModel

from .CLIP import *
from .encoder.builder import build_vision_tower
from .projector.builder import build_mm_projector


class VLMMetaModel:

    def __init__(self, config):
        super(VLMMetaModel, self).__init__(config)

        if hasattr(config, "vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=True)
            self.mm_projector = build_mm_projector(config)

    def get_vision_tower(self):
        vision_tower = getattr(self, "vision_tower", None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args):
        self.config.input_size = model_args.input_size
        self.config.patch_size = model_args.patch_size
        self.config.dim = model_args.dim
        self.config.depth = model_args.depth

        self.config.vision_tower = model_args.vision_tower
        self.config.vision_select_layer = model_args.vision_select_layer
        self.config.vision_select_feature = model_args.vision_select_feature

        self.config.mm_projector_type = model_args.mm_projector_type
        self.config.mm_mlp_depth = model_args.mm_mlp_depth
        self.config.proj_out_num = model_args.proj_out_num

        # vision tower
        if self.get_vision_tower() is None:
            self.vision_tower = build_vision_tower(self.config)
            self.vision_tower.requires_grad_(not model_args.freeze_vision_tower)

            if self.config.vision_tower == "hybrid":
                self.config.low_input_size = self.vision_tower.low_input_size
                self.config.high_input_size = self.vision_tower.high_input_size
            elif self.config.mm_projector_type == "mixer":
                self.config.low_output_size = model_args.low_output_size
                self.config.high_output_size = model_args.high_output_size
                self.config.low_input_size = (256, 384)
                self.config.high_input_size = (32, 768)

        if model_args.pretrain_vision_model is not None:
            vision_model_weights = torch.load(
                model_args.pretrain_vision_model, map_location="cpu"
            )
            self.vision_tower.vision_tower.load_state_dict(
                vision_model_weights, strict=True
            )

        if model_args.pretrain_clip_model is not None:
            clip_model = AutoModel.from_pretrained(model_args.pretrain_clip_model)
            self.vision_tower.vision_tower = clip_model.vision_encoder

        self.config.mm_hidden_size = self.vision_tower.hidden_size

        # mm_projector
        if getattr(self, "mm_projector", None) is None:
            self.mm_projector = build_mm_projector(self.config)

        if model_args.pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(
                model_args.pretrain_mm_mlp_adapter, map_location="cpu"
            )

            if self.config.mm_projector_type == "mlp":

                def get_w(weights, keyword):
                    return {
                        f"{keyword}.{k.split(keyword + ".")[2]}": v
                        for k, v in weights.items()
                        if keyword in k
                    }

            elif self.config.mm_projector_type == "low_high_mlp":

                def get_w(weights, keyword):
                    result = {}
                    for k, v in weights.items():
                        if keyword in k:
                            if f"{keyword}.{keyword}" in k:
                                part = k.split(f"{keyword}.{keyword}.")[1]
                                result[f"mm_projector.{part}"] = v
                            elif f"{keyword}." in k:
                                part = k.split(f"{keyword}.")[1]
                                result[part] = v
                    return result

            elif self.config.mm_projector_type == "mixer":

                def get_w(weights, keyword):
                    result = {}
                    for k, v in weights.items():
                        if keyword in k:
                            new_key = k.split(".")
                            if len(new_key) > 2:
                                new_key = ".".join(new_key[2:])
                                result[new_key] = v
                    return result

            else:

                def get_w(weights, keyword):
                    result = {}
                    for k, v in weights.items():
                        if keyword in k:
                            new_key = k.split(".")
                            if len(new_key) > 2:
                                new_key = ".".join(new_key[2:])
                                result[new_key] = v
                    return result

            self.mm_projector.load_state_dict(
                get_w(mm_projector_weights, "mm_projector"), strict=True
            )


class VLMMetaForCausalLM(ABC):
    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def encode_images(self, images):
        image_features = self.get_model().get_vision_tower()(images)
        image_features = self.get_model().mm_projector(image_features)
        return image_features

    def prepare_inputs_for_multimodal(
        self,
        input_ids,
        position_ids,
        attention_mask,
        past_key_values,
        labels,
        images,
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                None,
                labels,
            )
        else:
            image_features = self.encode_images(images)
            inputs_embeds = self.get_model().embed_tokens(input_ids)
            inputs_embeds = torch.cat(
                (
                    inputs_embeds[:, :1, :],
                    image_features,
                    inputs_embeds[:, (image_features.shape[1] + 1) :, :],
                ),
                dim=1,
            )
        return (
            None,
            position_ids,
            attention_mask,
            past_key_values,
            inputs_embeds,
            labels,
        )

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        num_new_tokens = model_args.num_new_tokens

        self.resize_token_embeddings(len(tokenizer))

        if num_new_tokens > 0:
            input_embeddings = self.get_input_embeddings().weight.data
            output_embeddings = self.get_output_embeddings().weight.data

            input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                dim=0, keepdim=True
            )
            output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                dim=0, keepdim=True
            )

            input_embeddings[-num_new_tokens:] = input_embeddings_avg
            output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
            else:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = True

        if model_args.pretrain_mm_mlp_adapter:
            mm_projector_weights = torch.load(
                model_args.pretrain_mm_mlp_adapter, map_location="cpu"
            )

            embed_tokens_weight = mm_projector_weights["model.embed_tokens.weight"]

            if input_embeddings.shape == embed_tokens_weight.shape:
                input_embeddings = embed_tokens_weight
            elif embed_tokens_weight.shape[0] == num_new_tokens:
                input_embeddings[-num_new_tokens:] = embed_tokens_weight
            else:
                raise ValueError(
                    f"Unexpected embed_tokens_weight shape. "
                    f"Pretrained: {embed_tokens_weight.shape}. "
                    f"Current: {input_embeddings.shape}. "
                    f"Number of new tokens: {num_new_tokens}."
                )
