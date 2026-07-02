#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from segablation.prepare_b_datasets import prepare_real, prepare_synthetic_roi


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare 64x128 ROI manifests for the final HFUS segmentation ablation.")
    p.add_argument("--data_root", type=Path, default=REPO_ROOT / "data")
    p.add_argument("--out_root", type=Path, default=REPO_ROOT / "outputs")
    p.add_argument("--synthetic_root", type=Path, default=None)
    p.add_argument("--synthetic_metadata", default="metadata.json")
    p.add_argument("--mendeley_image_dir", type=Path, default=None)
    p.add_argument("--mendeley_mask_dir", type=Path, default=None)
    p.add_argument("--real_folds", type=int, default=5)
    p.add_argument("--real_val_frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=66)
    p.add_argument("--num_workers", type=int, default=6)
    p.add_argument("--only_synthetic", action="store_true")
    p.add_argument("--only_real", action="store_true")
    p.add_argument("--check_only", action="store_true")
    p.add_argument("--no_roi_visuals", action="store_true")
    p.add_argument("--max_roi_visuals_per_split", type=int, default=0)
    return p.parse_args()


def _check(path: Path, desc: str, must_exist: bool = True) -> None:
    if must_exist and not path.exists():
        raise FileNotFoundError(f"{desc} not found: {path}")
    print(f"[OK] {desc}: {path}")


def main() -> None:
    args = parse_args()
    args.data_root = args.data_root.resolve()
    args.out_root = args.out_root.resolve()

    if args.synthetic_root is None:
        args.synthetic_root = args.data_root / "synthetic_sample"
    if args.mendeley_image_dir is None:
        args.mendeley_image_dir = args.data_root / "Mendeley" / "images"
    if args.mendeley_mask_dir is None:
        args.mendeley_mask_dir = args.data_root / "Mendeley" / "masks"

    args.synthetic_root = args.synthetic_root.resolve()
    args.mendeley_image_dir = args.mendeley_image_dir.resolve()
    args.mendeley_mask_dir = args.mendeley_mask_dir.resolve()
    args.save_roi_visuals = not args.no_roi_visuals

    print(f"[INFO] data_root   = {args.data_root}")
    print(f"[INFO] out_root    = {args.out_root}")
    print(f"[INFO] synthetic   = {args.synthetic_root}")
    print(f"[INFO] mendeley im = {args.mendeley_image_dir}")
    print(f"[INFO] mendeley ms = {args.mendeley_mask_dir}")

    if args.only_real and args.only_synthetic:
        raise RuntimeError("Use only one of --only_real or --only_synthetic.")

    if not args.only_real:
        _check(args.synthetic_root, "synthetic root")
        _check(args.synthetic_root / args.synthetic_metadata, "synthetic metadata.json")
        _check(args.synthetic_root / "images", "synthetic images directory")
        _check(args.synthetic_root / "masks", "synthetic masks directory")
    if not args.only_synthetic:
        _check(args.mendeley_image_dir, "Mendeley images directory")
        _check(args.mendeley_mask_dir, "Mendeley masks directory")

    if args.check_only:
        print("[DONE] data structure check passed.")
        return

    args.out_root.mkdir(parents=True, exist_ok=True)

    if not args.only_synthetic:
        real_manifest = prepare_real(args)
        print(f"[DONE] real ROI manifest: {real_manifest}")
    if not args.only_real:
        synthetic_manifest = prepare_synthetic_roi(args)
        print(f"[DONE] synthetic ROI manifest: {synthetic_manifest}")


if __name__ == "__main__":
    main()
