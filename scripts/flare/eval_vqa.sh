#!/bin/bash

export PYTHONPATH=$(pwd):$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

python src/flare_infer/vqa.py \
    --model_name_or_path output/FLARE_VLM \
    --json_path FLARE_npy/val_processed.json \
    --data_root FLARE_npy/validation

python src/flare_infer/eval_vqa.py \
    --pred_csv output/FLARE_VLM/predictions.csv \
    --val_json FLARE_npy/val_processed.json