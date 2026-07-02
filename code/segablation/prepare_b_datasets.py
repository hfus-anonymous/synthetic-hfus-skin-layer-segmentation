from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from tqdm import tqdm

from .datasets import _as_samples, _path_value, _get_sample_id, infer_source, _resolve

from .common import (
    EPI_SLEB_PALETTE_BGR,
    V66_PALETTE_BGR,
    read_gray,
    read_json,
    read_rgb,
    resize_image_and_mask,
    v66_to_epi_sleb,
    write_json,
    write_png,
    overlay_mask_on_rgb,
    colorize_mask,
)

IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def standardize_mendeley_mask(mask_gray: np.ndarray, min_fg: int = 8) -> np.ndarray:
    """Convert Mendeley mask image to 0 other / 1 epidermis / 2 SLEB."""
    mask = np.zeros(mask_gray.shape, dtype=np.uint8)
    fg = mask_gray > min_fg
    if not np.any(fg):
        return mask
    vals = mask_gray[fg].astype(np.float32)
    p10, p90 = np.percentile(vals, [10, 90])
    if p90 - p10 > 20:
        thr = (p10 + p90) / 2.0
        epi = mask_gray > thr
        sleb = fg & ~epi
        mask[epi] = 1
        mask[sleb] = 2
        y_epi = np.median(np.where(mask == 1)[0]) if np.any(mask == 1) else 1e9
        y_sleb = np.median(np.where(mask == 2)[0]) if np.any(mask == 2) else -1
        if y_epi > y_sleb:
            m2 = mask.copy()
            mask[m2 == 1] = 2
            mask[m2 == 2] = 1
        return mask

    n, labels, stats, centroids = cv2.connectedComponentsWithStats(fg.astype(np.uint8), connectivity=8)
    comps = []
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= 20:
            comps.append((float(centroids[i][1]), i, area))
    comps.sort(key=lambda x: x[0])
    if comps:
        mask[labels == comps[0][1]] = 1
    for _, comp_id, _ in comps[1:]:
        mask[labels == comp_id] = 2
    return mask


def compute_roi(mask3: np.ndarray, min_height: int = 256, max_height: int = 768, top_factor: float = 0.5, bottom_factor: float = 2.0) -> Tuple[int, int, int, int]:
    h, w = mask3.shape
    ys = np.where((mask3 == 1) | (mask3 == 2))[0]
    if len(ys) == 0:
        return 0, h, 0, w
    y0 = int(ys.min())
    y1 = int(ys.max()) + 1
    target_h = max(1, y1 - y0)
    roi0 = int(round(y0 - top_factor * target_h))
    roi1 = int(round(y1 + bottom_factor * target_h))
    center = int(round((y0 + y1) / 2))
    roi0 = max(0, roi0)
    roi1 = min(h, roi1)
    if roi1 - roi0 < min_height:
        half = min_height // 2
        roi0 = max(0, center - half)
        roi1 = min(h, roi0 + min_height)
        roi0 = max(0, roi1 - min_height)
    if roi1 - roi0 > max_height:
        half = max_height // 2
        roi0 = max(0, center - half)
        roi1 = min(h, roi0 + max_height)
        roi0 = max(0, roi1 - max_height)
    return int(roi0), int(roi1), 0, int(w)


def find_image_files(root: Path) -> List[Path]:
    return [p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS and not p.name.startswith("._")]


def normalize_mendeley_stem(path: Path, is_mask: bool = False) -> str:
    stem = path.stem.lower().strip()
    for suffix in ("_mask", "-mask", "_label", "-label", "_annotation", "-annotation", "_gt", "-gt"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    if not is_mask:
        for suffix in ("_image", "-image", "_img", "-img"):
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
    return stem


def pair_real_images(image_dir: Path, mask_dir: Path) -> List[Tuple[Path, Path, str]]:
    images = sorted(find_image_files(image_dir))
    masks = sorted(find_image_files(mask_dir))
    if not images:
        raise RuntimeError(f"No image files found in Mendeley image dir: {image_dir}")
    if not masks:
        raise RuntimeError(f"No mask files found in Mendeley mask dir: {mask_dir}")

    mask_by_stem: Dict[str, Path] = {}
    duplicates: Dict[str, List[Path]] = {}
    for m in masks:
        key = normalize_mendeley_stem(m, is_mask=True)
        if key in mask_by_stem:
            duplicates.setdefault(key, [mask_by_stem[key]]).append(m)
        else:
            mask_by_stem[key] = m
    if duplicates:
        msg = "; ".join([f"{k}: {[x.name for x in v]}" for k, v in list(duplicates.items())[:5]])
        raise RuntimeError(f"Duplicate Mendeley mask stems after normalization. Remove duplicates. Examples: {msg}")

    pairs: List[Tuple[Path, Path, str]] = []
    missing: List[str] = []
    for img in images:
        key = normalize_mendeley_stem(img, is_mask=False)
        if key in mask_by_stem:
            pairs.append((img, mask_by_stem[key], img.stem))
        else:
            missing.append(img.name)
    if not pairs:
        raise RuntimeError(
            "No real image/mask pairs found. Masks may have the same name as images, "
            "e.g. images/1AD.png and masks/1AD.png, or suffix form 1AD_mask.png."
        )
    if missing:
        print(f"[pair_real_images] warning: {len(missing)} image(s) have no matching mask. First examples: {missing[:5]}")
    print(f"[pair_real_images] paired {len(pairs)} real image/mask files.")
    return sorted(pairs, key=lambda x: x[2])


def draw_crop_box(image_rgb: np.ndarray, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
    out = image_rgb.copy()
    # cv2 expects RGB array here if color is RGB tuple because write_png converts RGB->BGR.
    cv2.rectangle(out, (int(x0), int(y0)), (int(x1) - 1, int(y1) - 1), (255, 0, 0), 4)
    return out


def _safe_read_rgb(path: str | Path) -> np.ndarray:
    return read_rgb(Path(path))


def _safe_read_gray(path: str | Path) -> np.ndarray:
    return read_gray(Path(path))


def save_roi_visualization_cache(out: Path, df: pd.DataFrame, palette=EPI_SLEB_PALETTE_BGR, alpha: float = 0.35, max_per_split: int = 0) -> None:
    """Save ROI QC visualization for train/val/test.

    This is for checking whether ROI crop/resize is correct before interpreting B-2/B-3/B-4 metrics.
    If max_per_split <= 0, all samples are saved.
    """
    required = {"sample_id", "image", "mask", "split"}
    missing = required - set(df.columns)
    if missing:
        print(f"[save_roi_visualization_cache] skip; missing columns: {sorted(missing)}")
        return
    all_rows = []
    for split, sdf in df.groupby("split"):
        split = str(split)
        if max_per_split and max_per_split > 0:
            sdf = sdf.head(int(max_per_split))
        vis_root = out / "visualization_cache" / split
        for _, row in tqdm(sdf.iterrows(), total=len(sdf), desc=f"ROI visualization {out.name}/{split}"):
            sid = str(row["sample_id"])
            img = _safe_read_rgb(row["image"])
            mask = _safe_read_gray(row["mask"])
            paths = {}
            img_path = vis_root / "images" / f"{sid}_roi.png"
            mask_path = vis_root / "masks" / f"{sid}_roi_mask.png"
            color_path = vis_root / "color_masks" / f"{sid}_roi_mask_color.png"
            overlay_path = vis_root / "overlays" / f"{sid}_roi_overlay.png"
            write_png(img_path, img)
            write_png(mask_path, mask)
            write_png(color_path, colorize_mask(mask, palette, include_zero=False))
            write_png(overlay_path, overlay_mask_on_rgb(img, mask, palette, alpha=alpha, include_zero=False))
            paths.update({"image": str(img_path), "mask": str(mask_path), "color_mask": str(color_path), "overlay": str(overlay_path)})

            if {"original_image", "roi_y0", "roi_y1", "roi_x0", "roi_x1"}.issubset(df.columns):
                try:
                    full = _safe_read_rgb(row["original_image"])
                    box = draw_crop_box(full, int(row["roi_y0"]), int(row["roi_y1"]), int(row["roi_x0"]), int(row["roi_x1"]))
                    box_path = vis_root / "crop_boxes" / f"{sid}_crop_box.png"
                    write_png(box_path, box)
                    paths["crop_box"] = str(box_path)
                except Exception as e:
                    paths["crop_box_error"] = str(e)
            all_rows.append({"split": split, "sample_id": sid, **paths})
    pd.DataFrame(all_rows).to_csv(out / "visualization_cache" / "visualization_manifest.csv", index=False)
    print(f"[save_roi_visualization_cache] saved visualization cache under: {out / 'visualization_cache'}")


def prepare_real(args) -> Path:
    out = Path(args.out_root) / "prepared" / "mendeley_real_roi_64x128"
    out.mkdir(parents=True, exist_ok=True)
    image_dir = Path(args.mendeley_image_dir)
    mask_dir = Path(args.mendeley_mask_dir)
    pairs = pair_real_images(image_dir, mask_dir)
    ids = [sid for _, _, sid in pairs]
    kf = KFold(n_splits=args.real_folds, shuffle=True, random_state=args.seed)
    fold_map = {}
    for fold, (_trainval_idx, test_idx) in enumerate(kf.split(ids)):
        for i in test_idx:
            fold_map[ids[i]] = fold

    def process_pair(item):
        img_path, mask_path, sid = item
        img = read_rgb(img_path)
        raw = read_gray(mask_path)
        mask3 = standardize_mendeley_mask(raw)
        y0, y1, x0, x1 = compute_roi(mask3)
        roi_img = img[y0:y1, x0:x1]
        roi_mask = mask3[y0:y1, x0:x1]
        roi_img_64, roi_mask_64 = resize_image_and_mask(roi_img, roi_mask, (64, 128))
        out_img = out / "images" / f"{sid}.png"
        out_mask = out / "masks" / f"{sid}_mask.png"
        write_png(out_img, roi_img_64)
        write_png(out_mask, roi_mask_64)
        full_mask = out / "full_masks" / f"{sid}_mask3.png"
        write_png(full_mask, mask3)
        write_png(out / "full_mask_color" / f"{sid}_mask3_color.png", colorize_mask(mask3, EPI_SLEB_PALETTE_BGR))
        write_png(out / "full_gt_overlays" / f"{sid}_gt_overlay.png", overlay_mask_on_rgb(img, mask3, EPI_SLEB_PALETTE_BGR, alpha=0.35))
        return {
            "sample_id": sid,
            "id": sid,
            "image": str(out_img),
            "mask": str(out_mask),
            "original_image": str(img_path),
            "original_mask": str(mask_path),
            "full_standard_mask": str(full_mask),
            "fold": int(fold_map[sid]),
            "roi_y0": y0, "roi_y1": y1, "roi_x0": x0, "roi_x1": x1,
            "original_h": img.shape[0], "original_w": img.shape[1],
        }

    rows = []
    workers = max(1, int(getattr(args, "num_workers", 1)))
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            rows = list(tqdm(ex.map(process_pair, pairs), total=len(pairs), desc=f"prepare real ROI ({workers} threads)"))
    else:
        for item in tqdm(pairs, desc="prepare real ROI"):
            rows.append(process_pair(item))
    df = pd.DataFrame(rows)
    df["split"] = "train"
    df.loc[df["fold"] == 0, "split"] = "test"
    train_idx = df.index[df["split"] == "train"].to_numpy()
    _tr_idx, val_idx = train_test_split(train_idx, test_size=args.real_val_frac, random_state=args.seed)
    df.loc[val_idx, "split"] = "val"
    df.to_csv(out / "manifest.csv", index=False)
    write_json(out / "dataset_info.json", {"num_samples": len(df), "folds": args.real_folds, "class_names": {"0": "other", "1": "epidermis", "2": "SLEB"}})
    if bool(getattr(args, "save_roi_visuals", True)):
        save_roi_visualization_cache(out, df, EPI_SLEB_PALETTE_BGR, max_per_split=int(getattr(args, "max_roi_visuals_per_split", 0)))
    print(f"[prepare_real] saved: {out / 'manifest.csv'}")
    return out / "manifest.csv"


def prepare_synthetic_roi(args) -> Path:
    out = Path(args.out_root) / "prepared" / "v66_synthetic_roi_64x128"
    out.mkdir(parents=True, exist_ok=True)
    root = Path(args.synthetic_root)
    meta = read_json(root / args.synthetic_metadata)
    samples0 = _as_samples(meta)
    samples = []
    for s0 in samples0:
        if not isinstance(s0, dict):
            continue
        s = dict(s0)
        s.setdefault("id", _get_sample_id(s))
        s["image"] = _path_value(s, "image")
        s["mask"] = _path_value(s, "mask")
        s["source"] = infer_source(s)
        samples.append(s)
    if not samples:
        raise RuntimeError(f"No synthetic samples found in metadata: {root / args.synthetic_metadata}")

    def process_sample(s: dict):
        img_path = _resolve(root, s["image"])
        mask_path = _resolve(root, s["mask"])
        img = read_rgb(img_path)
        mask8 = read_gray(mask_path)
        mask3 = v66_to_epi_sleb(mask8)
        y0, y1, x0, x1 = compute_roi(mask3)
        roi_img = img[y0:y1, x0:x1]
        roi_mask = mask3[y0:y1, x0:x1]
        roi_img_64, roi_mask_64 = resize_image_and_mask(roi_img, roi_mask, (64, 128))
        sid = str(s["id"])
        out_img = out / "images" / f"{sid}.png"
        out_mask = out / "masks" / f"{sid}_mask.png"
        full_mask = out / "full_masks" / f"{sid}_mask3.png"
        write_png(out_img, roi_img_64)
        write_png(out_mask, roi_mask_64)
        write_png(full_mask, mask3)
        write_png(out / "full_mask_color" / f"{sid}_mask3_color.png", colorize_mask(mask3, EPI_SLEB_PALETTE_BGR))
        write_png(out / "full_gt_overlays" / f"{sid}_gt_overlay.png", overlay_mask_on_rgb(img, mask3, EPI_SLEB_PALETTE_BGR, alpha=0.35))
        return {
            "sample_id": sid,
            "id": sid,
            "image": str(out_img),
            "mask": str(out_mask),
            "original_image": str(img_path),
            "original_mask": str(mask_path),
            "full_standard_mask": str(full_mask),
            "split": s.get("split"),
            "source": s.get("source"),
            "parent_job_id": s.get("parent_job_id") or s.get("split_group_id") or s.get("parent_id"),
            "original_sample_id": s.get("original_sample_id") or s.get("parent_id"),
            "category": s.get("category"),
            "severity": s.get("severity"),
            "roi_y0": y0, "roi_y1": y1, "roi_x0": x0, "roi_x1": x1,
            "original_h": img.shape[0], "original_w": img.shape[1],
        }

    workers = max(1, int(getattr(args, "num_workers", 1)))
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            rows = list(tqdm(ex.map(process_sample, samples), total=len(samples), desc=f"prepare synthetic ROI ({workers} threads)"))
    else:
        rows = []
        for s in tqdm(samples, desc="prepare synthetic ROI"):
            rows.append(process_sample(s))
    df = pd.DataFrame(rows)

    # Public sample datasets are often re-numbered as 1.png..100.png.
    # In that case the original train/val split may be absent.  V4 training
    # requires synthetic train rows and an original-only synthetic validation set,
    # so create a deterministic split if needed.
    if "source" not in df.columns:
        df["source"] = "augmented"
    df["source"] = df["source"].fillna("augmented").astype(str).str.lower()
    if "split" not in df.columns:
        df["split"] = ""
    df["split"] = df["split"].fillna("").astype(str).str.lower()

    has_train = (df["split"] == "train").any()
    has_original_val = ((df["split"] == "val") & (df["source"] == "original")).any()
    if (not has_train) or (not has_original_val):
        df["split"] = "train"
        original_idx = df.index[df["source"] == "original"].tolist()
        if original_idx:
            rng = np.random.default_rng(int(getattr(args, "seed", 66)))
            original_idx = list(rng.permutation(original_idx))
            n_val = max(1, int(round(0.20 * len(original_idx))))
            n_val = min(n_val, len(original_idx))
            df.loc[original_idx[:n_val], "split"] = "val"
        else:
            # Absolute fallback for malformed metadata: keep the pipeline runnable.
            rng = np.random.default_rng(int(getattr(args, "seed", 66)))
            all_idx = list(rng.permutation(df.index.to_list()))
            n_val = max(1, int(round(0.20 * len(all_idx))))
            df.loc[all_idx[:n_val], "split"] = "val"
            df.loc[all_idx[:n_val], "source"] = "original"
        print("[prepare_synthetic_roi] assigned deterministic public-sample split/source fallback")

    df.to_csv(out / "manifest.csv", index=False)
    write_json(out / "dataset_info.json", {"num_samples": len(df), "class_names": {"0": "other", "1": "epidermis", "2": "SLEB"}})
    if bool(getattr(args, "save_roi_visuals", True)):
        save_roi_visualization_cache(out, df, EPI_SLEB_PALETTE_BGR, max_per_split=int(getattr(args, "max_roi_visuals_per_split", 0)))
    print(f"[prepare_synthetic_roi] saved: {out / 'manifest.csv'}")
    return out / "manifest.csv"


def main():
    p = argparse.ArgumentParser(description="Prepare ROI 64x128 datasets for Experiment B.")
    p.add_argument("--synthetic_root", default="./data/augmented")
    p.add_argument("--synthetic_metadata", default="metadata.json")
    p.add_argument("--mendeley_image_dir", default="")
    p.add_argument("--mendeley_mask_dir", default="")
    p.add_argument("--out_root", default="./outputs/segexp")
    p.add_argument("--real_folds", type=int, default=5)
    p.add_argument("--real_val_frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=66)
    p.add_argument("--num_workers", type=int, default=6, help="Thread count for image/mask preprocessing")
    p.add_argument("--only_real", action="store_true", help="Prepare only the Mendeley real ROI dataset.")
    p.add_argument("--only_synthetic", action="store_true", help="Prepare only the V66 synthetic ROI dataset.")
    p.add_argument("--no_roi_visuals", action="store_true", help="Disable train/val/test ROI visualization_cache creation.")
    p.add_argument("--max_roi_visuals_per_split", type=int, default=0, help="0 means save all ROI QC images per split.")
    args = p.parse_args()
    args.save_roi_visuals = not args.no_roi_visuals
    if args.only_real and args.only_synthetic:
        raise RuntimeError("Use only one of --only_real or --only_synthetic, not both.")
    if args.only_synthetic:
        prepare_synthetic_roi(args)
        return
    if args.only_real:
        if not args.mendeley_image_dir or not args.mendeley_mask_dir:
            raise RuntimeError("--mendeley_image_dir and --mendeley_mask_dir are required for --only_real.")
        prepare_real(args)
        return
    if not args.mendeley_image_dir or not args.mendeley_mask_dir:
        raise RuntimeError("--mendeley_image_dir and --mendeley_mask_dir are required unless --only_synthetic is used.")
    prepare_real(args)
    prepare_synthetic_roi(args)


if __name__ == "__main__":
    main()
