#!/bin/bash

python src/utils/merge_lora_weights_and_save_hf_model.py \
    --model_name_or_path Qwen/Qwen2.5-14B-Instruct \
    --model_type vlm_qwen \
    --mm_projector_type "mixer" \
    --pretrain_vision_model ./output/DCFormer_SigLIP/pretrained_ViT.bin \
    --vision_tower "dcformer" \
    --model_with_lora ./output/Med3DVLM-Qwen-2.5-7B-finetune/model_with_lora.bin \
    --output_dir ./models/Med3DVLM-Qwen-2.5-7B