import os

import torch
from torch import distributed as dist
from tqdm import tqdm
from transformers import Trainer

import wandb


def is_rank_zero():
    if "RANK" in os.environ:
        if int(os.environ["RANK"]) != 0:
            return False
    if dist.is_available() and dist.is_initialized():
        if dist.get_rank() != 0:
            return False
    return True


class CLIPTrainer(Trainer):
    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        outputs = model(**inputs)
        loss = outputs["loss"]

        if is_rank_zero():
            wandb.log(
                {
                    "train/loss": loss.item(),
                    "train/learning_rate": self.lr_scheduler.get_last_lr()[0],
                    "train/step": self.state.global_step,
                },
                step=self.state.global_step,
            )

        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        with torch.no_grad():
            outputs = model(**inputs)
            loss = outputs["loss"]
            logits = outputs["logits"]

            if prediction_loss_only:
                return (loss, None, None)

            labels = inputs["labels"]
            return (loss, logits, labels)


class MLLMTrainer(Trainer):
    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        outputs = model(**inputs)
        loss = outputs["loss"]

        if is_rank_zero():
            wandb.log(
                {
                    "train/loss": loss.item(),
                    "train/learning_rate": self.lr_scheduler.get_last_lr()[0],
                    "train/step": self.state.global_step,
                },
                step=self.state.global_step,
            )

        return (loss, outputs) if return_outputs else loss
