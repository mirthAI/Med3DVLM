from typing import Any, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    Qwen2Config,
    Qwen2ForCausalLM,
    Qwen2Model,
)
from transformers.generation.utils import GenerateOutput
from transformers.modeling_outputs import CausalLMOutputWithPast

from ..vlm_arch import VLMMetaForCausalLM, VLMMetaModel


class VLMQwenConfig(Qwen2Config):
    model_type = "vlm_qwen"


class VLMQwenModel(VLMMetaModel, Qwen2Model):
    config_class = VLMQwenConfig

    def __init__(self, config: Qwen2Config):
        super(VLMQwenModel, self).__init__(config)


class VLMQwenForCausalLM(Qwen2ForCausalLM, VLMMetaForCausalLM):
    config_class = VLMQwenConfig

    def __init__(self, config):
        super(Qwen2ForCausalLM, self).__init__(config)
        self.model = VLMQwenModel(config)
        self.pretraining_tp = getattr(config, "pretraining_tp", None)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        num_logits_to_keep: Optional[int] = None,
        **kwargs: Any,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
            ) = self.prepare_inputs_for_multimodal(
                input_ids, position_ids, attention_mask, past_key_values, labels, images
            )

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    @torch.no_grad()
    def generate(
        self,
        images: Optional[torch.Tensor] = None,
        inputs: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor, Any]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (inputs, position_ids, attention_mask, _, inputs_embeds, _) = (
                self.prepare_inputs_for_multimodal(
                    inputs,
                    position_ids,
                    attention_mask,
                    None,
                    None,
                    images,
                )
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        output_ids = super().generate(inputs_embeds=inputs_embeds, **kwargs)
        return output_ids

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs
    ):
        images = kwargs.pop("images", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        if images is not None:
            inputs["images"] = images
        return inputs


AutoConfig.register("vlm_qwen", VLMQwenConfig)
AutoModelForCausalLM.register(VLMQwenConfig, VLMQwenForCausalLM)
