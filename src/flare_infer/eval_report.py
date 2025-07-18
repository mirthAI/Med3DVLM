import argparse
import csv

parser = argparse.ArgumentParser(description="Evaluate report generation")
parser.add_argument("--csv_path", type=str, required=True)
args = parser.parse_args()

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

category_sums = {cat: 0.0 for cat in target_categories}
category_counts = {cat: 0 for cat in target_categories}
green_sum = 0.0
green_count = 0

with open(args.csv_path, "r", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        green_val = float(row["green"])
        green_sum += green_val
        green_count += 1

        for cat in target_categories:
            val = float(row[cat])
            category_sums[cat] += val
            category_counts[cat] += 1

green_avg = green_sum / green_count
print(f"green average: {green_avg:.4f}\n")

print("Per-category average:")
for cat in target_categories:
    avg = category_sums[cat] / category_counts[cat]
    print(f"{cat}: {avg:.4f}")




