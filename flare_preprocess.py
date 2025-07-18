from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from skimage.transform import resize
from tqdm import tqdm


def _load_and_normalise(nifti: Path) -> np.ndarray:
    img_sitk = sitk.ReadImage(str(nifti))
    vol = sitk.GetArrayFromImage(img_sitk)

    if vol.ndim == 4:
        vol = vol[0]

    if "amos" in str(nifti):
        vol = np.clip(vol, -160.0, 240.0)
    else:
        vol = np.clip(vol, -1350, 150)

    vol = (vol - vol.min()) / (vol.max() - vol.min())

    vol = resize(vol, (128, 256, 256), anti_aliasing=True)
    return vol.astype(np.float32)[None]  # add channel dim


def _process_case(case: OrderedDict, nifti_root: Path, npy_root: Path) -> OrderedDict:
    """Worker: convert a single case to .npy and update its JSON record."""
    case_id = case["case_id"]
    nifti_p = nifti_root / case_id
    out_stub = case_id.replace(".nii.gz", "")
    vol = _load_and_normalise(nifti_p)
    np.save(npy_root / f"{out_stub}.npy", vol)

    new_case = OrderedDict(case)
    new_case["case_id"] = f"{out_stub}.npy"
    return new_case


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Pre‑process NIfTI volumes into normalised .npy tensors."
    )
    ap.add_argument("--json_in", type=Path, help="Path to the JSON metadata")
    ap.add_argument(
        "--nifti_dir", type=Path, help="Directory containing the *.nii.gz volumes"
    )
    ap.add_argument(
        "--out_dir",
        type=Path,
        default="./FLARE_npy",
        help="Destination directory for processed volumes "
        "(default: <nifti_dir>Processed)",
    )
    ap.add_argument(
        "--workers", type=int, default=32, help="Parallel workers (default: 32)"
    )
    args = ap.parse_args()

    npy_root = (
        args.out_dir
        if args.out_dir is not None
        else args.nifti_dir.with_name(args.nifti_dir.name + "Processed")
    )
    npy_root.mkdir(parents=True, exist_ok=True)

    with args.json_in.open() as f:
        cases: list[OrderedDict] = json.load(f, object_pairs_hook=OrderedDict)

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        new_cases = list(
            tqdm(
                pool.map(
                    _process_case,
                    cases,
                    [args.nifti_dir] * len(cases),
                    [npy_root] * len(cases),
                ),
                total=len(cases),
                desc="processing volumes",
            )
        )

    out_json = args.json_in.with_name(args.json_in.stem + "_processed.json")
    with out_json.open("w") as f:
        json.dump(new_cases, f, indent=2)

    print(f"Saved processed volumes to: {npy_root}")
    print(f"Updated JSON written to    : {out_json}")


if __name__ == "__main__":
    main()
