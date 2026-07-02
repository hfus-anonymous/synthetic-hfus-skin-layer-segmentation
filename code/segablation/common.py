from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
import torch

IGNORE_INDEX = 255

V66_CLASS_NAMES = {
    0: "air",
    1: "coupling",
    2: "epidermis",
    3: "SLEB",
    4: "dermis",
    5: "subcutis",
    6: "fascia",
    7: "muscle",
}

EPI_SLEB_CLASS_NAMES = {
    0: "other",
    1: "epidermis",
    2: "SLEB",
}

# BGR palettes for OpenCV visual outputs.
V66_PALETTE_BGR = {
    0: (0, 0, 0),
    1: (120, 120, 120),
    2: (0, 255, 255),
    3: (0, 160, 255),
    4: (0, 180, 0),
    5: (200, 80, 0),
    6: (255, 0, 255),
    7: (180, 80, 30),
}

EPI_SLEB_PALETTE_BGR = {
    0: (0, 0, 0),
    1: (0, 255, 255),
    2: (0, 160, 255),
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path | str) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: Path | str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path | str, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=lambda o: o.item() if hasattr(o, 'item') else str(o))


def read_rgb(path: Path | str) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def read_gray(path: Path | str) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read grayscale image: {path}")
    return img


def write_png(path: Path | str, array: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = array
    if out.ndim == 3 and out.shape[2] == 3:
        out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    ok = cv2.imwrite(str(path), out)
    if not ok:
        raise IOError(f"Failed to write PNG: {path}")


def resize_image_and_mask(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    size_hw: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    h, w = size_hw
    img_r = cv2.resize(image_rgb, (w, h), interpolation=cv2.INTER_LINEAR)
    mask_r = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return img_r, mask_r


def v66_to_epi_sleb(mask8: np.ndarray) -> np.ndarray:
    out = np.zeros(mask8.shape, dtype=np.uint8)
    out[mask8 == 2] = 1
    out[mask8 == 3] = 2
    out[mask8 == IGNORE_INDEX] = IGNORE_INDEX
    return out


def colorize_mask(mask: np.ndarray, palette: Dict[int, Tuple[int, int, int]], include_zero: bool = False) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)  # RGB output
    for cls_id, bgr in palette.items():
        if cls_id == 0 and not include_zero:
            continue
        rgb = (bgr[2], bgr[1], bgr[0])
        out[mask == cls_id] = rgb
    return out


def overlay_mask_on_rgb(image_rgb: np.ndarray, mask: np.ndarray, palette: Dict[int, Tuple[int, int, int]], alpha: float = 0.35, include_zero: bool = False) -> np.ndarray:
    base = image_rgb.copy().astype(np.float32)
    color = colorize_mask(mask, palette, include_zero=include_zero).astype(np.float32)
    valid = mask != IGNORE_INDEX
    if not include_zero:
        valid &= mask != 0
    valid3 = valid[..., None]
    out = base.copy()
    out[valid3.repeat(3, axis=2)] = ((1 - alpha) * base + alpha * color)[valid3.repeat(3, axis=2)]
    return np.clip(out, 0, 255).astype(np.uint8)


def confusion_matrix_np(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int, ignore_index: int = IGNORE_INDEX) -> np.ndarray:
    y_true = y_true.reshape(-1).astype(np.int64)
    y_pred = y_pred.reshape(-1).astype(np.int64)
    valid = (y_true != ignore_index) & (y_true >= 0) & (y_true < num_classes) & (y_pred >= 0) & (y_pred < num_classes)
    enc = num_classes * y_true[valid] + y_pred[valid]
    return np.bincount(enc, minlength=num_classes * num_classes).reshape(num_classes, num_classes).astype(np.int64)


def metrics_from_cm(cm: np.ndarray, class_names: Dict[int, str], report_class_ids: Optional[Sequence[int]] = None) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    if report_class_ids is None:
        report_class_ids = list(class_names.keys())
    total = cm.sum()
    pixel_acc = float(np.trace(cm) / total) if total > 0 else float("nan")
    rows = []
    ious, dices = [], []
    for cls_id in report_class_ids:
        tp = float(cm[cls_id, cls_id])
        fp = float(cm[:, cls_id].sum() - tp)
        fn = float(cm[cls_id, :].sum() - tp)
        support = float(cm[cls_id, :].sum())
        union = tp + fp + fn
        iou = tp / union if union > 0 else float("nan")
        dice = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else float("nan")
        precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        rows.append({
            "class_id": int(cls_id),
            "class_name": class_names[int(cls_id)],
            "iou": iou,
            "dice": dice,
            "precision": precision,
            "recall": recall,
            "support_pixels": support,
        })
        if not math.isnan(iou):
            ious.append(iou)
        if not math.isnan(dice):
            dices.append(dice)
    summary = {
        "pixel_accuracy_all_classes": pixel_acc,
        "mean_iou_reported_classes": float(np.mean(ious)) if ious else float("nan"),
        "mean_dice_reported_classes": float(np.mean(dices)) if dices else float("nan"),
    }
    return summary, rows


def save_prediction_bundle(
    out_dir: Path,
    sample_id: str,
    image_rgb: np.ndarray,
    gt_mask: Optional[np.ndarray],
    pred_mask: np.ndarray,
    palette: Dict[int, Tuple[int, int, int]],
    alpha: float = 0.35,
    include_zero: bool = False,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, str] = {}
    pred_path = out_dir / "pred_masks" / f"{sample_id}_pred.png"
    write_png(pred_path, pred_mask.astype(np.uint8))
    paths["pred_mask"] = str(pred_path)

    color_path = out_dir / "pred_color_masks" / f"{sample_id}_pred_color.png"
    write_png(color_path, colorize_mask(pred_mask, palette, include_zero=include_zero))
    paths["pred_color_mask"] = str(color_path)

    overlay = overlay_mask_on_rgb(image_rgb, pred_mask, palette, alpha=alpha, include_zero=include_zero)
    overlay_path = out_dir / "pred_overlays" / f"{sample_id}_pred_overlay.png"
    write_png(overlay_path, overlay)
    paths["pred_overlay"] = str(overlay_path)

    img_path = out_dir / "images" / f"{sample_id}_image.png"
    write_png(img_path, image_rgb)
    paths["image_copy"] = str(img_path)

    if gt_mask is not None:
        gt_path = out_dir / "gt_masks" / f"{sample_id}_gt.png"
        write_png(gt_path, gt_mask.astype(np.uint8))
        paths["gt_mask"] = str(gt_path)
        gt_color_path = out_dir / "gt_color_masks" / f"{sample_id}_gt_color.png"
        write_png(gt_color_path, colorize_mask(gt_mask, palette, include_zero=include_zero))
        paths["gt_color_mask"] = str(gt_color_path)
        gt_overlay = overlay_mask_on_rgb(image_rgb, gt_mask, palette, alpha=alpha, include_zero=include_zero)
        gt_overlay_path = out_dir / "gt_overlays" / f"{sample_id}_gt_overlay.png"
        write_png(gt_overlay_path, gt_overlay)
        paths["gt_overlay"] = str(gt_overlay_path)

    meta = {"sample_id": sample_id, "paths": paths}
    if extra:
        meta.update(extra)
    write_json(out_dir / "metadata" / f"{sample_id}.json", meta)
    return paths


def save_manifest_csv(path: Path | str, rows: List[Dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
