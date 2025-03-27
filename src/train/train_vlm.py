import logging
import os
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
import torch.distributed as dist
import transformers
from transformers import AutoTokenizer, LlamaForCausalLM

import wandb
from src.dataset.mllm_dataset import CapDataset, TextDatasets, TextYNDatasets
from src.model.llm.qwen import VLMQwenForCausalLM
from src.train.trainer import MLLMTrainer


def is_rank_zero():
    if "RANK" in os.environ:
        if int(os.environ["RANK"]) != 0:
            return False
    if dist.is_available() and dist.is_initialized():
        if dist.get_rank() != 0:
            return False
    return True


def rank0_print(*args):
    if is_rank_zero():
        print(*args)


@dataclass
class ModelArguments:
    wb_name: Optional[str] = field(default="MLLM")
    model_name_or_path: Optional[str] = field(
        default="Qwen/Qwen2.5-7B-Instruct",
        metadata={"help": "Path to the LLM or MLLM."},
    )
    model_type: Optional[str] = field(default="vlm_qwen")

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
    mm_projector_type: Optional[str] = field(default="mlp")
    mm_mlp_depth: int = field(
        default=2, metadata={"help": "Depth of MLP in projector."}
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
class DataArguments:
    data_root: str = field(
        default="./data/", metadata={"help": "Root directory for all data."}
    )

    # caption data
    cap_data_path: str = field(
        default="./data/M3D_Cap_npy/M3D_Cap.json",
        metadata={"help": "Path to caption data."},
    )

    # VQA data
    vqa_data_train_path: str = field(
        default="./data/M3D-VQA/M3D_VQA_train.csv",
        metadata={"help": "Path to training VQA data."},
    )
    vqa_data_val_path: str = field(
        default="./data/M3D-VQA/M3D_VQA_val.csv",
        metadata={"help": "Path to validation VQA data."},
    )
    vqa_data_test_path: str = field(
        default="./data/M3D-VQA/M3D_VQA_test.csv",
        metadata={"help": "Path to testing VQA data."},
    )

    vqa_yn_data_train_path: str = field(
        default="./data/M3D-VQA/M3D_VQA_yn_train.csv",
        metadata={"help": "Path to training VQA Yes or No data."},
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    # lora
    lora_enable: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"

    cache_dir: Optional[str] = field(default=None)
    remove_unused_columns: bool = field(default=False)
    model_max_length: int = field(
        default=512,  # 512
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    seed: int = 42
    ddp_backend: str = "nccl"
    ddp_timeout: int = 128000
    ddp_find_unused_parameters: bool = False
    optim: str = field(default="adamw_torch")

    # This is set up to facilitate debugging, pls config these in bash file in training.
    bf16: bool = True
    output_dir: str = "./output/Med3DVLM-pretrain-test"
    num_train_epochs: float = 1
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    eval_strategy: str = "steps"
    eval_accumulation_steps: int = 1
    eval_steps: float = 0.04
    save_strategy: str = "steps"
    save_steps: int = 2000
    save_total_limit: int = 2
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    logging_steps: float = 10  # 0.001
    gradient_checkpointing: bool = False  # train fast
    dataloader_pin_memory: bool = True  # fast
    dataloader_num_workers: int = 0
    report_to: str = "tensorboard"


def compute_metrics(eval_preds):
    labels_ids = eval_preds.label_ids
    pred_ids = eval_preds.predictions

    labels = labels_ids[:, 1:]
    preds = pred_ids[:, :-1]

    labels_flatten = labels.reshape(-1)
    preds_flatten = preds.reshape(-1)
    valid_indices = np.where(labels_flatten != -100)
    filtered_preds = preds_flatten[valid_indices]
    filtered_labels = labels_flatten[valid_indices]
    acc_score = sum(filtered_preds == filtered_labels) / len(filtered_labels)

    return {"accuracy": acc_score}


def preprocess_logits_for_metrics(logits, labels):
    pred_ids = torch.argmax(logits, dim=-1)
    return pred_ids


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(
                    f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}"
                )
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_mm_projector_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {
        k: t
        for k, t in named_params
        if any(key_match in k for key_match in keys_to_match)
    }
    to_return = {
        k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()
    }
    return to_return


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""

    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        # Only save projector and embed_tokens in pretrain
        keys_to_match = ["mm_projector", "embed_tokens", "embeddings"]

        weight_to_save = get_mm_projector_state_maybe_zero_3(
            trainer.model.named_parameters(), keys_to_match
        )
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split("/")[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith("checkpoint-"):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(
                    weight_to_save,
                    os.path.join(mm_projector_folder, f"{current_folder}.bin"),
                )
            else:
                torch.save(
                    weight_to_save, os.path.join(output_dir, f"mm_projector.bin")
                )
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


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


@dataclass
class DataCollator:
    def __call__(self, batch: list) -> dict:
        images, input_ids, labels, attention_mask = tuple(
            [b[key] for b in batch]
            for key in ("image", "input_id", "label", "attention_mask")
        )

        images = torch.cat([_.unsqueeze(0) for _ in images], dim=0)
        input_ids = torch.cat([_.unsqueeze(0) for _ in input_ids], dim=0)
        labels = torch.cat([_.unsqueeze(0) for _ in labels], dim=0)
        attention_mask = torch.cat([_.unsqueeze(0) for _ in attention_mask], dim=0)

        return_dict = dict(
            images=images,
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )

        return return_dict


def main():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    rank0_print("=" * 20 + " Tokenizer preparation " + "=" * 20)
    # Load tokenizer from the given path with specified configurations
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    # Define and add special tokens
    special_token = {"additional_special_tokens": ["<im_patch>"]}
    tokenizer.add_special_tokens(special_token)

    if tokenizer.unk_token is not None and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
    if "llama3" in model_args.model_type:
        tokenizer.eos_token_id = 128001
        tokenizer.pad_token = tokenizer.eos_token

    # Convert special tokens to token IDs and set related arguments
    model_args.img_token_id = tokenizer.convert_tokens_to_ids("<im_patch>")
    model_args.vocab_size = len(tokenizer)
    rank0_print("vocab_size: ", model_args.vocab_size)

    if model_args.mm_projector_type is not None:
        if model_args.mm_projector_type == "low_high_mlp":
            model_args.proj_out_num = 288
        elif (
            model_args.mm_projector_type == "mlp"
            or model_args.mm_projector_type == "mhsa"
        ):
            model_args.proj_out_num = 32
        else:
            model_args.proj_out_num = 256
    else:
        raise ValueError(f"Unknown Projector Type {model_args.mm_projector_type}")

    rank0_print("=" * 20 + " Model preparation " + "=" * 20)
    if model_args.vision_tower is not None:
        if "qwen" in model_args.model_type:
            model = VLMQwenForCausalLM.from_pretrained(
                model_args.model_name_or_path, cache_dir=training_args.cache_dir
            )
        else:
            raise ValueError(f"Unknown Model Type {model_args.model_type}")
    else:
        model = LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path, cache_dir=training_args.cache_dir
        )

    model.config.use_cache = False

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    model.enable_input_require_grads()
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    if model_args.vision_tower is not None:
        model.get_model().initialize_vision_modules(model_args=model_args)

    model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = (
        model_args.tune_mm_mlp_adapter
    )
    if model_args.tune_mm_mlp_adapter:
        model.requires_grad_(False)
        for p in model.get_model().mm_projector.parameters():
            p.requires_grad = True

    model_args.num_new_tokens = 1
    model.initialize_vision_tokenizer(model_args, tokenizer)

    if model_args.pretrain_mllm:
        ckpt = torch.load(model_args.pretrain_mllm, map_location="cpu")
        model.load_state_dict(ckpt, strict=True)
        rank0_print("load pretrained MLLM weights.")

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
        rank0_print("Adding LoRA adapters only on LLM.")
        model = get_peft_model(model, lora_config)

        for n, p in model.named_parameters():
            if any(
                [
                    x in n
                    for x in [
                        "vision_tower",
                        "mm_projector",
                        "embed_tokens",
                        "lm_head",
                    ]
                ]
            ):
                p.requires_grad = True

        model.print_trainable_parameters()

    rank0_print("=" * 20 + " Dataset preparation " + "=" * 20)
    data_args.max_length = training_args.model_max_length
    data_args.proj_out_num = model.get_model().mm_projector.proj_out_num
    rank0_print("vision tokens output from projector: ", data_args.proj_out_num)

    if model_args.tune_mm_mlp_adapter:
        train_dataset = TextDatasets(data_args, tokenizer, mode="train")
    else:
        train_dataset = TextYNDatasets(data_args, tokenizer, mode="train")

    eval_dataset = CapDataset(data_args, tokenizer, mode="validation")
    data_collator = DataCollator()

    rank0_print("=" * 20 + " Training " + "=" * 20)
    trainer = MLLMTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )

    if is_rank_zero():
        wandb.login()
        wandb.init(project="MLLM", name=model_args.wb_name)

    if os.path.exists(training_args.output_dir):
        checkpoints = sorted(
            [
                d
                for d in os.listdir(training_args.output_dir)
                if d.startswith("checkpoint-")
                and os.path.isdir(os.path.join(training_args.output_dir, d))
            ],
            key=lambda x: int(x.split("-")[-1]) if "-" in x else 0,
        )
        if checkpoints:
            last_checkpoint = checkpoints[-1]
            resume_ckpt = os.path.join(training_args.output_dir, last_checkpoint)
            rank0_print(f"Resuming from checkpoint: {resume_ckpt}")
            trainer.train(resume_from_checkpoint=resume_ckpt)
        else:
            trainer.train()
    else:
        trainer.train()

    trainer.save_state()
    model.config.use_cache = True

    rank0_print("=" * 20 + " Save model " + "=" * 20)
    if training_args.lora_enable:
        state_dict_with_lora = model.state_dict()
        torch.save(
            state_dict_with_lora,
            os.path.join(training_args.output_dir, "model_with_lora.bin"),
        )
    else:
        safe_save_model_for_hf_trainer(
            trainer=trainer, output_dir=training_args.output_dir
        )

    if is_rank_zero():
        wandb.finish()

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
