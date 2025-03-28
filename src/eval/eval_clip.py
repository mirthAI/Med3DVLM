import argparse
import csv
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from src.dataset.clip_dataset import CLIPDataset
from src.model.CLIP import *


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path", type=str, default="./models/Med3DVLM-DCFormer-SigLIP"
    )
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])

    # data
    parser.add_argument("--data_root", type=str, default="./data/")
    parser.add_argument(
        "--cap_data_path", type=str, default="./data/M3D_Cap_npy/M3D_Cap.json"
    )
    parser.add_argument("--output_dir", type=str, default="./output/eval/")
    parser.add_argument("--save_output", type=bool, default=False)

    parser.add_argument(
        "--test_method",
        type=tuple,
        default=(
            "recall",
            "accuracy",
        ),  # ("recall", "precision", "f1_score", "accuracy")
    )
    parser.add_argument("--test_topk", type=tuple, default=(1, 5, 10))
    parser.add_argument("--test_size", type=tuple, default=(100, 500, 1000, 2000))

    return parser.parse_args(args)


def calculate_recall(similarity_matrix, k):
    _, topk_indices = similarity_matrix.topk(k, dim=1)
    diagonal_indices = torch.arange(similarity_matrix.size(0)).to(
        similarity_matrix.device
    )
    correct_matches = torch.eq(topk_indices, diagonal_indices.view(-1, 1))
    recall_at_k = correct_matches.float().sum(dim=1).mean()
    return recall_at_k


def calculate_precision(similarity_matrix, k):
    _, topk_indices = similarity_matrix.topk(k, dim=1)
    diagonal_indices = torch.arange(similarity_matrix.size(0)).to(
        similarity_matrix.device
    )
    correct_matches = torch.eq(topk_indices, diagonal_indices.view(-1, 1))
    precision_at_k = correct_matches.float().sum() / (similarity_matrix.size(0) * k)
    return precision_at_k


def calculate_f1_score(similarity_matrix, k):
    precision = calculate_precision(similarity_matrix, k)
    recall = calculate_recall(similarity_matrix, k)

    if precision + recall == 0:
        return torch.tensor(0.0).to(similarity_matrix.device)

    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score


def calculate_accuracy(similarity_matrix, k):
    _, topk_indices = similarity_matrix.topk(k, dim=1)
    diagonal_indices = torch.arange(similarity_matrix.size(0)).to(
        similarity_matrix.device
    )
    correct_matches = torch.eq(topk_indices, diagonal_indices.view(-1, 1)).any(dim=1)
    accuracy = correct_matches.float().mean()
    return accuracy


def main():
    seed_everything(42)
    args = parse_args()
    device = torch.device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        model_max_length=args.max_length,
        padding_side="right",
        use_fast=False,
    )
    model = AutoModel.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    model = model.to(device=device)

    results = {}

    for test_size in args.test_size:
        test_dataset = CLIPDataset(
            args, tokenizer=tokenizer, mode="test", test_size=test_size
        )

        test_dataloader = DataLoader(
            test_dataset,
            batch_size=1,
            num_workers=4,
            pin_memory=True,
            shuffle=False,
            drop_last=False,
        )

        txt_feats_all = []
        img_feats_all = []
        for sample in tqdm(test_dataloader):
            input_id = sample["input_id"].to(device=device)
            attention_mask = sample["attention_mask"].to(device=device)
            image = sample["image"].to(device=device)
            with torch.inference_mode():
                image_features = model.encode_image(image)
                text_features = model.encode_text(input_id, attention_mask)
            txt_feats_all.append(text_features.detach().cpu())
            img_feats_all.append(image_features.detach().cpu())

        txt_feats_all = torch.cat(txt_feats_all, dim=0)
        img_feats_all = torch.cat(img_feats_all, dim=0)

        scores_mat = torch.matmul(img_feats_all, txt_feats_all.transpose(0, 1))

        for test_method in args.test_method:
            for test_topk in args.test_topk:
                if test_method == "recall":
                    i_to_t = calculate_recall(scores_mat, test_topk)
                    t_to_i = calculate_recall(scores_mat.transpose(0, 1), test_topk)
                    print(f"IR_{test_topk}@{test_size}: ", i_to_t)
                    print(f"TR_{test_topk}@{test_size}: ", t_to_i)
                    results[f"IR_{test_topk}@{test_size}"] = i_to_t
                    results[f"TR_{test_topk}@{test_size}"] = t_to_i
                elif test_method == "precision":
                    i_to_t = calculate_precision(scores_mat, test_topk)
                    t_to_i = calculate_precision(scores_mat.transpose(0, 1), test_topk)
                    print(f"IP_{test_topk}@{test_size}: ", i_to_t)
                    print(f"TP_{test_topk}@{test_size}: ", t_to_i)
                    results[f"IP_{test_topk}@{test_size}"] = i_to_t
                    results[f"TP_{test_topk}@{test_size}"] = t_to_i
                elif test_method == "f1_score":
                    i_to_t = calculate_f1_score(scores_mat, test_topk)
                    t_to_i = calculate_f1_score(scores_mat.transpose(0, 1), test_topk)
                    print(f"IF1_{test_topk}@{test_size}: ", i_to_t)
                    print(f"TF1_{test_topk}@{test_size}: ", t_to_i)
                    results[f"IF1_{test_topk}@{test_size}"] = i_to_t
                    results[f"TF1_{test_topk}@{test_size}"] = t_to_i
                elif test_method == "accuracy":
                    i_to_t = calculate_accuracy(scores_mat, test_topk)
                    t_to_i = calculate_accuracy(scores_mat.transpose(0, 1), test_topk)
                    print(f"IAcc_{test_topk}@{test_size}: ", i_to_t)
                    print(f"TAcc_{test_topk}@{test_size}: ", t_to_i)
                    results[f"IAcc_{test_topk}@{test_size}"] = i_to_t
                    results[f"TAcc_{test_topk}@{test_size}"] = t_to_i
                else:
                    raise ValueError(f"Invalid test method: {test_method}")

    if args.save_output:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        model_name = args.model_name_or_path.split("/")[-1]
        output_path = os.path.join(args.output_dir, f"{model_name}_eval_retrieval.csv")

        with open(output_path, mode="w") as outfile:
            writer = csv.writer(outfile)
            for key, value in results.items():
                writer.writerow([key, f"{value.item():.4f}"])
        print("Save eval results successfully!")


if __name__ == "__main__":
    main()
