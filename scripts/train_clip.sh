#!/bin/bash

deepspeed src/train/train_clip.py \
    --deepspeed ./scripts/zero2.json \
    --language_model_name_or_path medicalai/ClinicalBERT \
    --wb_name DCFormer_SigLIP \
    --vision_encoder "dcformer" \
    --loss_type "sigmoid" \
    --data_root ./data \
    --max_length 512 \
    --bf16 True \
    --output_dir ./output/DCFormer_SigLIP \
    --num_train_epochs 100 \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --eval_strategy "no" \
    --eval_accumulation_steps 1 \
    --eval_steps 0.04 \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 1e-4 \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 0.001 \
    --gradient_checkpointing False \
    --dataloader_pin_memory False \
    --dataloader_num_workers 4