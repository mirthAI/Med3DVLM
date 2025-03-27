#!/bin/bash

python src/eval/eval_caption.py \
    --model_name_or_path ./models/Med3DVLM-Qwen-2.5-7B \
    --data_root ./data \
    --max_length 512 \
    --proj_out_num 256 \
    --do_sample \
    --output_dir ./output/eval