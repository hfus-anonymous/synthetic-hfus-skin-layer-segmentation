#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path) -> Any:
    for enc in ("utf-8", "utf-8-sig", "cp949"):
        try:
            return json.loads(path.read_text(encoding=enc))
        except Exception:
            pass
    raise RuntimeError(f"Could not read JSON: {path}")


def count_pngs(path: Path) -> int:
    if not path.exists():
        return 0
    return len([p for p in path.iterdir() if p.is_file() and p.suffix.lower() == ".png"])


def validate_synthetic_sample(root: Path) -> None:
    images_dir = root / "images"
    masks_dir = root / "masks"
    metadata_dir = root / "metadata"
    metadata_json = root / "metadata.json"

    missing = []
    for path, desc in [
        (root, "synthetic_sample root"),
        (images_dir, "images directory"),
        (masks_dir, "masks directory"),
        (metadata_dir, "metadata directory"),
        (metadata_json, "metadata.json"),
    ]:
        if not path.exists():
            missing.append(f"{desc}: {path}")

    if missing:
        raise FileNotFoundError(
            "Downloaded data/synthetic_sample is incomplete.\n"
            + "\n".join(f"  - missing {x}" for x in missing)
            + "\n\nExpected data.zip to provide:\n"
            + "  data/synthetic_sample/images/*.png\n"
            + "  data/synthetic_sample/masks/*.png\n"
            + "  data/synthetic_sample/metadata/*.json\n"
            + "  data/synthetic_sample/metadata.json\n"
        )

    image_count = count_pngs(images_dir)
    mask_count = count_pngs(masks_dir)
    meta_count = len([p for p in metadata_dir.iterdir() if p.is_file() and p.suffix.lower() == ".json"])

    if image_count == 0:
        raise RuntimeError(f"No PNG images found in: {images_dir}")
    if mask_count == 0:
        raise RuntimeError(f"No PNG masks found in: {masks_dir}")
    if image_count != mask_count:
        raise RuntimeError(f"Image/mask count mismatch: images={image_count}, masks={mask_count}")

    meta = read_json(metadata_json)
    if isinstance(meta, dict) and isinstance(meta.get("samples"), list):
        sample_count = len(meta["samples"])
    elif isinstance(meta, list):
        sample_count = len(meta)
    elif isinstance(meta, dict):
        sample_count = len([v for v in meta.values() if isinstance(v, dict)])
    else:
        sample_count = 0

    print("[OK] downloaded synthetic_sample is available.")
    print(f"     root          : {root}")
    print(f"     images        : {image_count}")
    print(f"     masks         : {mask_count}")
    print(f"     metadata files: {meta_count}")
    print(f"     metadata rows : {sample_count}")
    print("[DONE] data/synthetic_sample is ready for run_experiments.py.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the downloaded repository-local data/synthetic_sample folder. "
            "This script does not read any external path."
        )
    )
    parser.add_argument(
        "--sample_root",
        type=Path,
        default=REPO_ROOT / "data" / "synthetic_sample",
        help="Repository-local synthetic sample folder. Default: data/synthetic_sample",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_synthetic_sample(args.sample_root.resolve())


if __name__ == "__main__":
    main()
