import os
import re
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import transformers
from transformers import AutoTokenizer

from src.model.llm import VLMQwenForCausalLM


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default="Qwen/Qwen2.5-7B-Instruct",
        metadata={"help": "Path to the LLM or MLLM."},
    )
    model_type: Optional[str] = field(default="vlm_qwen")
    model_with_lora: Optional[str] = field(
        default="./output/Med3DVLM-Qwen-2.5-7B-finetune/model_with_lora.bin"
    )

    freeze_backbone: bool = field(default=False)
    pretrain_mllm: Optional[str] = field(default=None)

    tune_mm_mlp_adapter: bool = field(
        default=False,
        metadata={"help": "Used in pretrain: tune mm_projector and embed_tokens"},
    )
    pretrain_mm_mlp_adapter: Optional[str] = field(
        default=None,
        metadata={"help": "Path to pretrained mm_projector and embed_tokens."},
    )

    # image
    input_size: tuple = field(default=(256, 256, 128))
    patch_size: int = field(default=(16, 16, 16))
    dim: int = field(default=768)
    depth: int = field(default=12)

    # vision
    vision_tower: Optional[str] = field(default="dcformer")
    vision_select_layer: Optional[int] = field(default=-2)
    vision_select_feature: Optional[str] = field(default="cls_patch")
    pretrain_vision_model: str = field(
        default=None, metadata={"help": "Path to pretrained model for ViT."}
    )
    pretrain_clip_model: str = field(
        default=None, metadata={"help": "Path to pretrained model for CLIP."}
    )
    freeze_vision_tower: bool = field(default=False)

    # projector
    mm_projector_type: Optional[str] = field(default="mlp", metadata={"help": "spp"})
    mm_mlp_depth: int = field(
        default=2, metadata={"help": "Depth of MLP in projector."}
    )
    proj_layer_type: str = field(
        default="mlp",
        metadata={"help": "Type of layer in projector. options: [linear, mlp]."},
    )
    proj_layer_num: int = field(
        default=2, metadata={"help": "Number of layers in projector."}
    )
    proj_pooling_type: str = field(
        default="spatial",
        metadata={
            "help": "Type of pooling in projector. options: [spatial, sequence]."
        },
    )
    proj_pooling_size: int = field(
        default=2, metadata={"help": "Size of pooling in projector."}
    )
    proj_residual: bool = field(
        default=False, metadata={"help": "Residual in projector."}
    )

    low_output_size: List[int] = field(
        default_factory=lambda: [192, 128],
        metadata={"help": "Output size of low feature."},
    )
    high_output_size: List[int] = field(
        default_factory=lambda: [64, 128],
        metadata={"help": "Output size of high feature."},
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    lora_enable: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"

    cache_dir: Optional[str] = field(default=None)
    output_dir: str = "./output/Med3DVLM-pretrain-test"


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    # Process of elimination: LoRA only targets on LLM backbone
    ignore_keywords = [
        "vision_tower",
        "mm_projector",
        "embed_tokens",
        "lm_head",
    ]
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in ignore_keywords):
            continue
        if isinstance(module, cls):
            lora_module_names.add(name)
    return list(lora_module_names)


def main():
    parser = transformers.HfArgumentParser((ModelArguments, TrainingArguments))
    model_args, training_args = parser.parse_args_into_dataclasses()

    print("Tokenizer preparation")
    # Load tokenizer from the given path with specified configurations
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        padding_side="right",
        use_fast=False,
    )

    # Define and add special tokens
    special_token = {"additional_special_tokens": ["<im_patch>"]}
    tokenizer.add_special_tokens(special_token)

    if tokenizer.unk_token is not None and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token

    # Convert special tokens to token IDs and set related arguments
    model_args.img_token_id = tokenizer.convert_tokens_to_ids("<im_patch>")
    model_args.vocab_size = len(tokenizer)
    print("vocab_size: ", model_args.vocab_size)

    print("Model preparation")

    if model_args.mm_projector_type is not None:
        if model_args.mm_projector_type == "low_high_mlp":
            model_args.proj_out_num = 288
        elif model_args.mm_projector_type == "mlp":
            model_args.proj_out_num = 32
        else:
            model_args.proj_out_num = 256
    else:
        raise ValueError(f"Unknown Projector Type {model_args.mm_projector_type}")

    if "qwen" in model_args.model_type:
        model = VLMQwenForCausalLM.from_pretrained(
            model_args.model_name_or_path, cache_dir=training_args.cache_dir
        )
    else:
        raise ValueError(f"Unknown Model Type {model_args.model_type}")

    if model_args.vision_tower is not None:
        model.get_model().initialize_vision_modules(model_args=model_args)

    model_args.num_new_tokens = 1
    model.initialize_vision_tokenizer(model_args, tokenizer)

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model

        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        print("Adding LoRA adapters only on LLM.")
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    print("Load weights with LoRA")
    state_dict = torch.load(model_args.model_with_lora, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)

    print("Merge weights with LoRA")
    model = model.merge_and_unload()
    state_dict = model.state_dict()

    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)

    model.model.config.architectures = model.__class__.__name__
    model._name_or_path = training_args.output_dir

    print("Save pretrained")
    model.config.save_pretrained(training_args.output_dir)
    model.save_pretrained(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)

    print("Save vision tower")
    vision_tower = model.get_model().vision_tower.state_dict()
    torch.save(vision_tower, os.path.join(training_args.output_dir, "vision_tower.bin"))

    print("Finish")


if __name__ == "__main__":
    main()
