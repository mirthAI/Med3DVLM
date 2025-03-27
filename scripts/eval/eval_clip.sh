#!/bin/bash

python src/eval/eval_clip.py \
    --model_name_or_path output/DCFormer_SigLIP \
    --data_root ./data \
    --save_output True \
    --output_dir ./output/eval \
    --max_length 512