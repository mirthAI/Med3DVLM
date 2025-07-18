import json
import os
import re

import pandas as pd

# from GREEN.green_score import GREEN
from green_score import GREEN
from tqdm import tqdm


def to_dict(text: str):
    if isinstance(text, dict):
        return text
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except (json.JSONDecodeError, TypeError):
        pass

    out = {}
    for line in filter(None, map(str.strip, text.splitlines())):
        m = re.match(r"([^:–\-]+)[:–\-]\s*(.+)", line)
        if m:
            key, val = m.groups()
            out[key.strip().lower()] = val.strip()
    return out


class GenerateGreenScore:
    def __init__(self, csv_path, cache_dir=None, save_every=10):
        self.csv_path = csv_path
        self.save_every = save_every
        self.model = GREEN(
            model_name="StanfordAIMI/GREEN-radllama2-7b",
            output_dir=cache_dir,
            # cache_dir=cache_dir,
        )
        self.df = pd.read_csv(self.csv_path)
        self.df.fillna("", inplace=True)

    def run(self):
        self.generate_scores()
        self.save_df()
        self.save_summary()
        return self.df

    def save_df(self):
        self.df.to_csv(self.csv_path, index=False)

    def save_summary(self):
        summary = {}

        overall = [n for n in self.df["green"].to_list() if n != -1]
        summary["green"] = sum(overall) / len(overall) if overall else 0.0

        region_totals = {}
        region_counts = {}

        for _, row in self.df.iterrows():
            for region in to_dict(row["gt"]).keys():
                if region not in self.df.columns:
                    continue
                region_totals[region] = region_totals.get(region, 0.0) + row[region]
                region_counts[region] = region_counts.get(region, 0) + 1

        summary["region_means"] = {
            r: region_totals[r] / region_counts[r] for r in region_totals
        }

        path = os.sep.join(self.csv_path.split(os.sep)[:-1])
        with open(os.path.join(path, "summary.json"), "w") as fh:
            json.dump(summary, fh, indent=4)

    def _compute_region_scores(self, gt_raw: str, gen_raw: str) -> dict:
        gt_dict = to_dict(gt_raw)
        gen_dict = to_dict(gen_raw)

        region_scores = {}
        matching_regions = [
            r for r in gen_dict if r in gt_dict
        ]  # region in both gt and gen

        if matching_regions:
            refs = [gt_dict[r] for r in matching_regions]
            hyps = [gen_dict[r] for r in matching_regions]

            _, _, green_list, *_ = self.model(refs=refs, hyps=hyps)

            for r, sc in zip(matching_regions, green_list):
                region_scores[r] = sc

        # assume gt is normal (in gen but not in gt)
        for r in gen_dict.keys() - gt_dict.keys():
            _, _, green_list, *_ = self.model(
                refs=[gen_dict[r]], hyps=[f"{r} is normal."]
            )
            region_scores[r] = green_list[0]

        # omissions (in gt but not in gen)
        for r in gt_dict.keys() - gen_dict.keys():
            region_scores[r] = 0.0

        return region_scores

    def generate_scores(self):
        if "green" not in self.df.columns:
            self.df["green"] = -1.0
        if "details" not in self.df.columns:
            self.df["details"] = ""

        self.df["green"] = -1.0
        self.df["details"] = ""

        for idx, row in tqdm(self.df.iterrows(), total=len(self.df)):
            scores = self._compute_region_scores(row["gt"], row["generated"])

            for region in scores:
                if region not in self.df.columns:
                    self.df[region] = 0.0

            for region, sc in scores.items():
                self.df.at[idx, region] = sc

            self.df.at[idx, "green"] = sum(scores.values()) / (len(scores) or 1)
            self.df.at[idx, "details"] = json.dumps(scores, ensure_ascii=False)

            if (idx + 1) % self.save_every == 0:
                self.save_df()

        region_cols = set(self.df.columns) - {"gt", "generated", "green", "details"}
        self.df[list(region_cols)] = self.df[list(region_cols)].fillna(0.0)

        self.save_df()
        self.save_summary()
