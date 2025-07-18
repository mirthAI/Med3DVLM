import argparse
import json
import math
import os
from pathlib import Path

import pandas as pd


def canonical(cid: str) -> str:
    fname = Path(cid).name
    if fname.endswith(".nii.gz"):
        return fname[:-7]
    return Path(fname).stem


def norm(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    return str(x).strip().lower()


def global_accuracy(df: pd.DataFrame, gt: dict) -> float:
    glob = df[df.scope == "global"]
    tot, n = 0.0, 0
    for _, r in glob.iterrows():
        cid = canonical(r.case_id)
        preds = [norm(p) for p in str(r.prediction).split(",") if norm(p)]
        gts = [norm(a) for a in gt[cid]["global_vqa"][0]["answer"]]

        if not gts and not preds:
            acc = 1.0
        else:
            acc = sum(p in gts for p in preds) / max(len(preds), len(gts))
        tot += acc
        n += 1
    return 0.0 if n == 0 else tot / n


def build_chains(local_gt):
    by_id = {q["id"]: q for q in local_gt}
    chains = {}
    for q in local_gt:
        root = q["id"]
        while by_id[root]["follow_up"] != -1:
            root = by_id[root]["follow_up"]
        chains.setdefault(root, set()).add(q["id"])
    return {r: [r] + sorted(ids - {r}) for r, ids in chains.items()}


def local_accuracy(df: pd.DataFrame, gt: dict) -> float:
    loc = df[df.scope == "local"]
    chain_score_sum, chain_cnt = 0.0, 0

    for cid_csv, rows in loc.groupby("case_id"):
        cid = canonical(cid_csv)
        local_gt = gt[cid]["local_vqa"]
        chains = build_chains(local_gt)
        gt_answers = {q["id"]: norm(q["answer"]) for q in local_gt}

        for _, r in rows.iterrows():
            root_id = int(r.question_id)
            if root_id not in chains:
                continue

            preds_list = [norm(p) for p in str(r.prediction).split("|")]
            qids = chains[root_id]

            chain_cnt += 1
            if not preds_list or preds_list[0] != gt_answers[root_id]:
                continue

            correct = 0
            for idx, qid in enumerate(qids):
                pred = preds_list[idx] if idx < len(preds_list) else ""
                if pred == gt_answers[qid]:
                    correct += 1
            chain_score_sum += correct / len(qids)

    return 0.0 if chain_cnt == 0 else chain_score_sum / chain_cnt


def evaluate_one(pred_csv_path: Path, gt: dict) -> dict:
    preds = pd.read_csv(pred_csv_path)
    return {
        "global_accuracy": round(global_accuracy(preds, gt), 6),
        "local_accuracy": round(local_accuracy(preds, gt), 6),
    }


def batch_mode(preds_root: Path, gt_json_path: Path, out_json: Path):
    with open(gt_json_path) as f:
        gt = {canonical(d["case_id"]): d for d in json.load(f)}

    results = {}
    for root, dirs, _ in os.walk(preds_root):
        for d in dirs:
            if not d.startswith("checkpoint-"):
                continue
            pred_csv = Path(root) / d / "model" / "predictions.csv"
            if pred_csv.exists():
                score = evaluate_one(pred_csv, gt)
                results[str(pred_csv.parent)] = score

    def extract_ckpt_num(path):
        import re

        m = re.search(r"checkpoint-(\d+)", path)
        return int(m.group(1)) if m else float("inf")

    sorted_items = sorted(results.items(), key=lambda x: extract_ckpt_num(x[0]))

    with open(out_json, "w") as f:
        json.dump(dict(sorted_items), f, indent=4)

    print(f"Saved batch results → {out_json}")
    for k, v in sorted_items:
        print(
            f"{k} | global: {v['global_accuracy']:.4f}, local: {v['local_accuracy']:.4f}"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_csv", required=True, help="CSV or folder with predictions")
    ap.add_argument("--val_json", required=True, help="Ground-truth JSON")
    ap.add_argument("--out_json", default="vqa_results.json")
    ap.add_argument("--batch_mode", action="store_true")
    args = ap.parse_args()

    if args.batch_mode:
        batch_mode(Path(args.pred_csv), Path(args.val_json), Path(args.out_json))
    else:
        with open(args.val_json) as f:
            gt = {canonical(d["case_id"]): d for d in json.load(f)}
        score = evaluate_one(Path(args.pred_csv), gt)
        print(f"Global VQA accuracy: {score['global_accuracy']:.4f}")
        print(f"Local  VQA accuracy: {score['local_accuracy']:.4f}")
        with open(args.out_json, "w") as f:
            json.dump(score, f, indent=2)


if __name__ == "__main__":
    main()
