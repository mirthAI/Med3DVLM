import csv
import json
import os
from pathlib import Path

target_categories = [
    "lymphatic system",
    "liver",
    "mediastinum",
    "respiratory tract",
    "abdominal cavity and peritoneum",
    "blood vessels",
    "esophagus",
    "musculoskeletal system",
    "endocrine system",
    "lungs and pleura",
    "heart",
    "gastrointestinal tract",
    "kidneys",
    "pancreas",
    "biliary system",
    "spleen",
    "breast tissue",
    "diaphragm",
]


def evaluate_csv(csv_path):
    category_sums = {cat: 0.0 for cat in target_categories}
    category_counts = {cat: 0 for cat in target_categories}
    green_sum = 0.0
    green_count = 0

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                green_val = float(row["green"])
                green_sum += green_val
                green_count += 1
                for cat in target_categories:
                    val = float(row[cat])
                    category_sums[cat] += val
                    category_counts[cat] += 1
            except (ValueError, KeyError):
                continue

    if green_count == 0:
        return None

    result = {"green_avg": round(green_sum / green_count, 6)}
    for cat in target_categories:
        if category_counts[cat] > 0:
            result[cat] = round(category_sums[cat] / category_counts[cat], 6)
        else:
            result[cat] = None
    return result


def batch_eval_to_json(base_dir, output_json="green_results.json"):
    results = {}

    for root, dirs, _ in os.walk(base_dir):
        for d in dirs:
            if not d.startswith("checkpoint-"):
                continue
            eval_csv = os.path.join(root, d, "model", "val_processed.csv")
            model_path = os.path.join(root, d, "model")
            if os.path.exists(eval_csv):
                res = evaluate_csv(eval_csv)
                if res:
                    results[model_path] = res

    def extract_ckpt_num(path):
        import re

        m = re.search(r"checkpoint-(\d+)", path)
        return int(m.group(1)) if m else float("inf")

    sorted_items = sorted(results.items(), key=lambda x: extract_ckpt_num(x[0]))

    with open(output_json, "w") as f:
        json.dump(sorted_items, f, indent=4)

    print(f"Saved JSON summary → {output_json}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir", type=str, required=True, help="Folder with checkpoints"
    )
    parser.add_argument("--out", type=str, default="report_results.json")
    args = parser.parse_args()

    batch_eval_to_json(args.dir, args.out)
