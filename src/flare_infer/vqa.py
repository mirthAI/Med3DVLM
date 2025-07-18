import argparse
import concurrent.futures
import json
import os
import random
import re
import subprocess
import warnings
from pathlib import Path

import monai.transforms as mtf
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from src.dataset.flare_dataset import read_image
from src.model.llm.qwen import VLMQwenForCausalLM

warnings.filterwarnings("ignore")
join = os.path.join


def seed_everything(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def format_pred(pred: str) -> str:
    lines = [l.strip() for l in pred.splitlines() if l.strip()]
    numbered = [
        re.sub(r"^\s*\d+\.\s*", "", l).strip()
        for l in lines
        if re.match(r"^\s*\d+\.\s*", l)
    ]
    return "|".join(numbered) if numbered and len(numbered) == len(lines) else pred


def build_prompt(q_txt: str, proj_tokens: int, tokenizer) -> str:
    image_tokens = "<im_patch>" * proj_tokens
    conversation = [
        {
            "role": "system",
            "content": (
                "You are an AI assistant acting as a radiologist tasked with "
                "answering a multiple-choice question based on a CT scan."
            ),
        },
        {"role": "user", "content": image_tokens + " " + q_txt},
    ]
    return tokenizer.apply_chat_template(conversation, tokenize=False)


def evaluation(args):
    device, dtype = torch.device("cuda"), torch.bfloat16

    with open(args.json_path) as f:
        dataset = json.load(f)

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
        pass

    resize_size = model.config.input_size
    transform = mtf.Compose([mtf.Resize(resize_size), mtf.ToTensor(dtype=torch.float)])
    proj_tokens = args.proj_out_num
    rows = []

    for sample in tqdm(dataset):
        case_id = sample["case_id"]
        img_path = join(args.data_root, case_id)
        image = transform(read_image(img_path)).unsqueeze(0).to(device, dtype)

        for g in sample["global_vqa"]:
            q_txt = g["question"].rstrip()
            if "choices" in g and g["choices"]:
                q_txt = f"{q_txt} Choices: {g['choices']}"
            prompt = build_prompt(q_txt, proj_tokens, tokenizer)
            input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
            out = model.generate(
                image,
                input_ids,
                max_new_tokens=128,
                do_sample=True,
                top_p=0.9,
                temperature=1.0,
            )
            pred = (
                tokenizer.decode(out[0], skip_special_tokens=True)
                .replace(prompt, "")
                .strip()
            )
            rows.append(
                dict(
                    case_id=case_id,
                    scope="global",
                    question_id="",
                    question=g["question"],
                    prediction=pred or "None",
                )
            )

        locals_ = sample["local_vqa"]
        roots = [q for q in locals_ if q["follow_up"] == -1]
        id2qs = {}
        for q in locals_:
            id2qs.setdefault(q["follow_up"], []).append(q)

        for root in roots:
            chain = [root] + id2qs.get(root["id"], [])
            q_lines = []
            for idx, q in enumerate(chain, 1):
                q_line = q["question"].rstrip()
                if "choices" in q and q["choices"]:
                    q_line = f"{q_line} Choices: {q['choices']}"
                q_lines.append(f"{idx}. {q_line}")
            q_txt = "\n".join(q_lines)
            prompt = build_prompt(q_txt, proj_tokens, tokenizer)
            input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
            out = model.generate(
                image,
                input_ids,
                max_new_tokens=64,
                do_sample=True,
                top_p=0.9,
                temperature=1.0,
            )
            pred = (
                tokenizer.decode(out[0], skip_special_tokens=True)
                .replace(prompt, "")
                .strip()
            )
            pred = format_pred(pred)
            rows.append(
                dict(
                    case_id=case_id,
                    scope="local",
                    question_id=root["id"],
                    question=q_txt,
                    prediction=pred or "None",
                )
            )

    out_csv = Path(args.model_name_or_path) / "predictions.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Saved → {out_csv}")


def run_evaluation_on_gpu(
    model_path, json_path, data_root, model_max_length, proj_out_num, gpu_id
):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    cmd = [
        "python",
        __file__,
        "--model_name_or_path",
        model_path,
        "--json_path",
        json_path,
        "--data_root",
        data_root,
        "--model_max_length",
        str(model_max_length),
        "--proj_out_num",
        str(proj_out_num),
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
                executor.submit(
                    run_evaluation_on_gpu,
                    model_path,
                    args.json_path,
                    args.data_root,
                    args.model_max_length,
                    args.proj_out_num,
                    gpu_id,
                )
            )
        concurrent.futures.wait(futures)
    print("All evaluations finished.")


def main():
    p = argparse.ArgumentParser("AMOS-VQA inference")
    p.add_argument("--model_name_or_path", required=True)
    p.add_argument("--json_path", required=True)
    p.add_argument("--data_root", required=True)
    p.add_argument("--model_max_length", type=int, default=1024)
    p.add_argument("--proj_out_num", type=int, default=256)
    p.add_argument("--batch_mode", action="store_true")
    args = p.parse_args()

    seed_everything()
    if args.batch_mode:
        batch_mode(args)
    else:
        evaluation(args)


if __name__ == "__main__":
    main()
