#!/bin/bash

export PYTHONPATH=$(pwd):$PYTHONPATH
# export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

deepspeed --master_port 29550 src/train/finetune_flare.py \
    --deepspeed ./scripts/zero2.json \
    --run_name Finetune_FLARE_B4 \
    --vision_tower "dcformer" \
    --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
    --model_type vlm_qwen \
    --mm_projector_type "mixer" \
    --lora_enable True \
    --vision_select_layer -2 \
    --pretrain_vision_model output/pretrained/vision_encoder.safetensors \
    --pretrain_mm_mlp_adapter output/pretrained/mm_projector.safetensors \
    --lora_enable True \
    --bf16 True \
    --output_dir output/FLARE_VLM \
    --num_train_epochs 100 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --eval_strategy "no" \
    --eval_accumulation_steps 1 \
    --eval_steps 0.04 \
    --save_strategy "steps" \
    --save_steps 0.1 \
    --save_total_limit 1 \
    --learning_rate 5e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 0.001 \
    --gradient_checkpointing False \
    --dataloader_pin_memory True \
    --dataloader_num_workers 8