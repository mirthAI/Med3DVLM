import os
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.distributed as dist
import transformers
from safetensors.torch import load_file
from trainer import CLIPTrainer
from transformers import AutoTokenizer

import wandb
from src.dataset.clip_dataset import CLIPDataset
from src.model.CLIP import DEC_CLIP, DEC_CLIPConfig


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
    wb_name: Optional[str] = field(default="CLIP")
    language_model_name_or_path: str = field(
        default="./LaMed/pretrained_model/bert_base_uncased/"
    )

    efficient_loss: bool = field(
        default=False,
        metadata={
            "help": "Use efficient loss for training. If False, use the original loss."
        },
    )

    gather_loss: bool = field(
        default=True,
        metadata={
            "help": "Gather all distributed batch data of multiple GPUs and calculate contrastive loss together."
        },
    )
    local_loss: bool = field(default=False)

    pretrained_model: str = field(default=None)

    input_size: tuple = field(default=(256, 256, 128))
    dim: int = field(default=768)
    depth: int = field(default=12)
    hidden_size: int = field(default=768)
    mlp_depth: int = field(default=2)

    vision_encoder: Optional[str] = field(default="dcformer")
    loss_type: str = field(default="nce")
    siglip_margin: float = field(default=0.1)


@dataclass
class DataArguments:
    data_root: str = field(
        default="./data", metadata={"help": "Root directory for all data."}
    )
    # caption data
    cap_data_path: str = field(
        default="./data/M3D_Cap_npy/M3D_Cap.json",
        metadata={"help": "Path to caption data."},
    )
    max_length: int = field(default=512)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)

    ddp_backend: str = "nccl"
    ddp_find_unused_parameters: bool = False

    # config in bash file
    bf16: bool = True
    output_dir: str = "./output/CLIP"
    num_train_epochs: int = 100
    per_device_train_batch_size: int = 32  # 32
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    eval_strategy: str = "steps"
    eval_accumulation_steps: int = 1
    eval_steps: float = 0.04  # 0.04
    save_strategy: str = "steps"
    save_steps: int = 1000
    save_total_limit: int = 1
    learning_rate: float = 1e-4  # 1e-4
    weight_decay: float = 0.1
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    logging_steps: float = 0.001  # 0.001
    gradient_checkpointing: bool = False  # train fast
    dataloader_pin_memory: bool = True  # fast
    dataloader_num_workers: int = 4
    report_to: str = "wandb"


def compute_metrics(eval_pred):
    preds = eval_pred.predictions
    labels = eval_pred.label_ids
    correct = (preds == labels).sum()
    total = labels.size
    acc = correct / total
    return {"accuracy": acc}


def preprocess_logits_for_metrics(logits, labels):
    preds = torch.argmax(logits, dim=-1)
    return preds


@dataclass
class DataCollator:
    def __init__(self, gather_all):
        self.gather_all = gather_all

    def __call__(self, batch: list) -> dict:
        images, texts, input_ids, attention_mask = tuple(
            [b[key] for b in batch]
            for key in ("image", "text", "input_id", "attention_mask")
        )

        images = torch.cat([_.unsqueeze(0) for _ in images], dim=0)
        input_ids = torch.cat([_.unsqueeze(0) for _ in input_ids], dim=0)
        attention_mask = torch.cat([_.unsqueeze(0) for _ in attention_mask], dim=0)

        batch_size = images.shape[0]
        if self.gather_all:
            world_size = torch.distributed.get_world_size()
            batch_size *= world_size

        labels = torch.arange(batch_size, device=images.device, dtype=torch.long)

        return_dict = dict(
            images=images,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        return return_dict


def main():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    tokenizer = AutoTokenizer.from_pretrained(model_args.language_model_name_or_path)

    config = DEC_CLIPConfig.from_dict(vars(model_args))
    model = DEC_CLIP(config)

    if model_args.pretrained_model:
        # ckpt = torch.load(model_args.pretrained_model)
        ckpt = load_file(model_args.pretrained_model)
        model.load_state_dict(ckpt, strict=True)
        print("load pretrained model.")

    train_dataset = CLIPDataset(data_args, tokenizer, mode="train")
    eval_dataset = CLIPDataset(data_args, tokenizer, mode="validation")

    if model_args.gather_loss and not model_args.local_loss:
        gather_all = True
    else:
        gather_all = False
    data_collator = DataCollator(gather_all)

    trainer = CLIPTrainer(
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
        wandb.init(project="Med3DVLM", name=model_args.wb_name)

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
    model.config.save_pretrained(training_args.output_dir)
    model.save_pretrained(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)

    vit = model.vision_encoder.state_dict()
    torch.save(vit, os.path.join(training_args.output_dir, "pretrained_ViT.bin"))

    if is_rank_zero():
        wandb.finish()

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
