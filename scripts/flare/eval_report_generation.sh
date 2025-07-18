#!/bin/bash

export PYTHONPATH=$(pwd):$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

python src/flare_infer/report_generation.py \
    --model_name_or_path output/FLARE_VLM \
    --json_path FLARE_npy/val_processed.json \
    --data_root FLARE_npy/validation

python src/flare_infer/eval_report.py \
    --csv_path output/FLARE_VLM/val_processed.csv