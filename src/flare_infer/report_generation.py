import argparse
import concurrent.futures
import json
import os
import subprocess
import warnings
from argparse import Namespace

import numpy as np
import pandas as pd
import torch
from generate_green_score import GenerateGreenScore
from tqdm import tqdm
from transformers import AutoTokenizer

from src.dataset.flare_dataset import FLARECapDataset
from src.model.llm.qwen import VLMQwenForCausalLM

warnings.filterwarnings("ignore")


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def caption_and_score(args):
    seed_everything(42)
    device = torch.device("cuda")
    dtype = torch.bfloat16

    model = VLMQwenForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        model_max_length=args.model_max_length,
        padding_side="left",
        use_fast=False,
        trust_remote_code=True,
    )

    try:
        model.set_token_id(tokenizer)
    except AttributeError:
        print("Model does not have set_token_id method, skipping.")

    model.eval()
    resize_size = model.config.input_size
    proj_out_num = args.proj_out_num

    tag = os.path.splitext(os.path.basename(args.json_path))[0]
    output_csv = os.path.join(args.model_name_or_path, f"{tag}.csv")

    data_args = Namespace()
    data_args.proj_out_num = proj_out_num
    data_args.json_path = [args.json_path]
    data_args.data_root = [args.data_root]
    data_args.max_length = args.model_max_length
    data_args.prompt = args.prompt
    data_args.data_img_size = resize_size

    dataset = FLARECapDataset(data_args, tokenizer, mode="validation")

    results = {"generated": [], "gt": [], "name": []}

    for item in tqdm(dataset):
        image_name = item["image_name"]
        image = item["image"].unsqueeze(0).to(device, dtype=dtype)
        input_id = item["input_id"].to(device)
        gt_text = item["answer"]

        generation = model.generate(
            image,
            input_id,
            max_new_tokens=512,
            do_sample=False,
            top_p=0.9,
            temperature=0,
        )
        gen_text = tokenizer.decode(generation[0], skip_special_tokens=True)

        results["gt"].append(gt_text)
        results["name"].append(image_name)
        results["generated"].append(gen_text)

        pd.DataFrame(results).to_csv(output_csv, index=False)

    print("Generating GREEN score...")
    g = GenerateGreenScore(output_csv, cache_dir="./GREEN_model")
    g.run()


def run_single_model_on_gpu(args, model_path, gpu_id):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    cmd = [
        "python",
        __file__,
        "--model_name_or_path",
        model_path,
        "--model_max_length",
        str(args.model_max_length),
        "--proj_out_num",
        str(args.proj_out_num),
        "--prompt",
        args.prompt,
        "--zoom",
        str(args.zoom),
        "--json_path",
        args.json_path,
        "--data_root",
        args.data_root,
    ]

    try:
        print(f"[GPU {gpu_id}] Running: {model_path}")
        subprocess.run(cmd, env=env, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[GPU {gpu_id}] Failed: {model_path}")
        print(e)


def batch_mode(args):
    model_paths = []
    for root, dirs, _ in os.walk(args.model_name_or_path):
        for d in dirs:
            if not d.startswith("checkpoint-"):
                continue
            model_path = os.path.join(root, d, "model")
            if os.path.isdir(model_path):
                model_paths.append(model_path)

    model_paths.sort()
    gpus = list(range(torch.cuda.device_count()))
    print(f"Found {len(model_paths)} models and {len(gpus)} GPUs: {gpus}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        futures = []
        for i, model_path in enumerate(model_paths):
            gpu_id = gpus[i % len(gpus)]
            futures.append(
                executor.submit(run_single_model_on_gpu, args, model_path, gpu_id)
            )
        concurrent.futures.wait(futures)

    print("All models finished.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--model_max_length", type=int, default=1024)
    parser.add_argument("--proj_out_num", type=int, default=256)
    parser.add_argument("--prompt", type=str, default="simple")
    parser.add_argument("--zoom", type=bool, default=False)
    parser.add_argument("--json_path", type=str, required=True)
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--batch_mode", action="store_true")

    args = parser.parse_args()

    if args.batch_mode:
        batch_mode(args)
    else:
        caption_and_score(args)


if __name__ == "__main__":
    main()
