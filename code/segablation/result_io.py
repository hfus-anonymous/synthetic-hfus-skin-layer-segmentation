from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


def _clean_float(x: Any) -> Any:
    try:
        if x is None:
            return None
        if isinstance(x, (np.floating, float)):
            if np.isnan(float(x)):
                return None
            return float(x)
        if isinstance(x, (np.integer, int)):
            return int(x)
    except Exception:
        pass
    return x


def load_json(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=_clean_float)


def _read_classwise_metrics(run_dir: Path, split: str = "test") -> Dict[str, Any]:
    csv_path = run_dir / f"{split}_classwise_metrics.csv"
    out: Dict[str, Any] = {}
    if not csv_path.exists():
        return out
    df = pd.read_csv(csv_path)
    for _, r in df.iterrows():
        name = str(r.get("class_name", r.get("class", ""))).strip()
        if not name:
            continue
        key = name.lower().replace(" ", "_")
        for metric in ["dice", "iou", "precision", "recall", "support_pixels"]:
            if metric in r:
                out[f"{key}_{metric}"] = _clean_float(r[metric])
    dice_vals = [out.get("epidermis_dice"), out.get("sleb_dice")]
    iou_vals = [out.get("epidermis_iou"), out.get("sleb_iou")]
    out["mean_dice"] = float(np.nanmean([v for v in dice_vals if v is not None])) if any(v is not None for v in dice_vals) else None
    out["mean_iou"] = float(np.nanmean([v for v in iou_vals if v is not None])) if any(v is not None for v in iou_vals) else None
    return out


def _read_summary_metrics(run_dir: Path, split: str = "test") -> Dict[str, Any]:
    js = load_json(run_dir / "metrics_summary.json")
    if not js:
        return {}
    # fit_model: {"best_val_metric": ..., "train": ..., "val": ..., "test": ...}
    # eval_external_preds: {"test": ..., "num_evaluated": ...}
    d = js.get(split, {}) if isinstance(js, dict) else {}
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out[k] = _clean_float(v)
    if isinstance(js, dict) and "best_val_metric" in js:
        out["best_val_metric"] = _clean_float(js.get("best_val_metric"))
    if isinstance(js, dict) and "num_evaluated" in js:
        out["num_evaluated"] = _clean_float(js.get("num_evaluated"))
    return out


def extract_result_row(
    experiment_dir: str | Path,
    experiment_id: str,
    experiment_name: str,
    model: str,
    init: str,
    synthetic_stage: str,
    real_stage: str,
    final_run_subdir: str | None = None,
    split: str = "test",
) -> Dict[str, Any]:
    exp_dir = Path(experiment_dir)
    run_dir = exp_dir / final_run_subdir if final_run_subdir else exp_dir
    class_m = _read_classwise_metrics(run_dir, split=split)
    summ_m = _read_summary_metrics(run_dir, split=split)
    row = {
        "experiment_id": experiment_id,
        "experiment_name": experiment_name,
        "model": model,
        "initialization": init,
        "synthetic_stage": synthetic_stage,
        "real_stage": real_stage,
        "split": split,
        "experiment_dir": str(exp_dir),
        "final_run_dir": str(run_dir),
        "epidermis_dice": class_m.get("epidermis_dice"),
        "sleb_dice": class_m.get("sleb_dice"),
        "mean_dice": class_m.get("mean_dice", summ_m.get("mean_dice_reported_classes")),
        "epidermis_iou": class_m.get("epidermis_iou"),
        "sleb_iou": class_m.get("sleb_iou"),
        "mean_iou": class_m.get("mean_iou", summ_m.get("mean_iou_reported_classes")),
        "pixel_accuracy_all_classes": summ_m.get("pixel_accuracy_all_classes"),
        "loss": summ_m.get("loss"),
        "best_val_metric": summ_m.get("best_val_metric"),
        "num_evaluated": summ_m.get("num_evaluated"),
    }
    return row


def _fmt(v: Any) -> str:
    if v is None:
        return "NA"
    try:
        if isinstance(v, float) and np.isnan(v):
            return "NA"
        if isinstance(v, (float, np.floating)):
            return f"{float(v):.6f}"
    except Exception:
        pass
    return str(v)


def save_experiment_result_txt(row: Dict[str, Any], out_path: str | Path, config: Optional[Dict[str, Any]] = None, notes: Optional[List[str]] = None) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    config = config or {}
    notes = notes or []
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("============================================================\n")
        f.write("Experiment Result\n")
        f.write("============================================================\n\n")
        f.write(f"Experiment ID: {row.get('experiment_id')}\n")
        f.write(f"Experiment Name: {row.get('experiment_name')}\n")
        f.write(f"Model: {row.get('model')}\n")
        f.write(f"Initialization: {row.get('initialization')}\n")
        f.write(f"Synthetic stage: {row.get('synthetic_stage')}\n")
        f.write(f"Real stage: {row.get('real_stage')}\n")
        f.write(f"Evaluation split: {row.get('split', 'test')}\n")
        f.write(f"Final run dir: {row.get('final_run_dir')}\n\n")
        f.write("Protocol:\n")
        f.write("  Input: 64 x 128 x 3 ROI patches\n")
        f.write("  Output classes: 0 other / 1 epidermis / 2 SLEB\n")
        f.write("  Ignore index: 255\n")
        f.write("  Primary metrics: Dice and IoU for epidermis and SLEB\n")
        f.write("  Evaluation data: fixed Mendeley real HFUS test split only\n\n")
        f.write("Test Results:\n")
        f.write(f"  Epidermis Dice: {_fmt(row.get('epidermis_dice'))}\n")
        f.write(f"  SLEB Dice: {_fmt(row.get('sleb_dice'))}\n")
        f.write(f"  Mean Dice: {_fmt(row.get('mean_dice'))}\n")
        f.write(f"  Epidermis IoU: {_fmt(row.get('epidermis_iou'))}\n")
        f.write(f"  SLEB IoU: {_fmt(row.get('sleb_iou'))}\n")
        f.write(f"  Mean IoU: {_fmt(row.get('mean_iou'))}\n")
        f.write(f"  Pixel accuracy, all classes: {_fmt(row.get('pixel_accuracy_all_classes'))}\n")
        f.write(f"  Loss: {_fmt(row.get('loss'))}\n")
        f.write(f"  Best validation metric: {_fmt(row.get('best_val_metric'))}\n")
        f.write(f"  Number evaluated: {_fmt(row.get('num_evaluated'))}\n\n")
        if notes:
            f.write("Notes:\n")
            for n in notes:
                f.write(f"  - {n}\n")
            f.write("\n")
        if config:
            f.write("Configuration snapshot:\n")
            for k in sorted(config):
                f.write(f"  {k}: {config[k]}\n")


def save_all_results(results: List[Dict[str, Any]], out_root: str | Path) -> None:
    out_root = Path(out_root)
    summary_dir = out_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    ordered = [
        "experiment_id", "experiment_name", "model", "initialization", "synthetic_stage", "real_stage", "split",
        "epidermis_dice", "sleb_dice", "mean_dice", "epidermis_iou", "sleb_iou", "mean_iou",
        "pixel_accuracy_all_classes", "loss", "best_val_metric", "num_evaluated", "experiment_dir", "final_run_dir",
    ]
    cols = [c for c in ordered if c in df.columns] + [c for c in df.columns if c not in ordered]
    df = df[cols] if not df.empty else df
    df.to_csv(summary_dir / "all_results.csv", index=False)
    write_json(summary_dir / "all_results.json", results)

    with open(summary_dir / "all_results.txt", "w", encoding="utf-8") as f:
        f.write("============================================================\n")
        f.write("Final Experiment Summary\n")
        f.write("============================================================\n\n")
        f.write("Evaluation protocol:\n")
        f.write("  Input: 64 x 128 x 3 ROI patches\n")
        f.write("  Output classes: other / epidermis / SLEB\n")
        f.write("  Evaluation split: fixed real HFUS test split\n")
        f.write("  Metrics: Dice and IoU for epidermis and SLEB\n")
        f.write("  Ignore index: 255\n\n")
        for r in results:
            f.write("------------------------------------------------------------\n")
            f.write(f"{r.get('experiment_id')} | {r.get('experiment_name')}\n")
            f.write("------------------------------------------------------------\n")
            f.write(f"Model: {r.get('model')}\n")
            f.write(f"Init: {r.get('initialization')}\n")
            f.write(f"Synthetic: {r.get('synthetic_stage')}\n")
            f.write(f"Real: {r.get('real_stage')}\n")
            f.write(f"Epidermis Dice: {_fmt(r.get('epidermis_dice'))}\n")
            f.write(f"SLEB Dice: {_fmt(r.get('sleb_dice'))}\n")
            f.write(f"Mean Dice: {_fmt(r.get('mean_dice'))}\n")
            f.write(f"Epidermis IoU: {_fmt(r.get('epidermis_iou'))}\n")
            f.write(f"SLEB IoU: {_fmt(r.get('sleb_iou'))}\n")
            f.write(f"Mean IoU: {_fmt(r.get('mean_iou'))}\n\n")

    paper_cols = [
        "experiment_name", "initialization", "synthetic_stage", "real_stage",
        "epidermis_dice", "sleb_dice", "mean_dice", "epidermis_iou", "sleb_iou", "mean_iou",
    ]
    df_paper = df[[c for c in paper_cols if c in df.columns]].copy() if not df.empty else df
    df_paper.to_csv(summary_dir / "paper_table.csv", index=False)
    with open(summary_dir / "paper_table.txt", "w", encoding="utf-8") as f:
        if df_paper.empty:
            f.write("No result rows were collected.\n")
        else:
            f.write(df_paper.to_string(index=False))
            f.write("\n")

    # Graph-ready long-form CSV. This makes plotting Dice/IoU bars immediate.
    long_rows = []
    for r in results:
        for cls in ["epidermis", "sleb", "mean"]:
            for metric in ["dice", "iou"]:
                key = f"{cls}_{metric}" if cls != "mean" else f"mean_{metric}"
                long_rows.append({
                    "experiment_id": r.get("experiment_id"),
                    "experiment_name": r.get("experiment_name"),
                    "model": r.get("model"),
                    "initialization": r.get("initialization"),
                    "training_protocol": f"synthetic={r.get('synthetic_stage')} | real={r.get('real_stage')}",
                    "class": cls,
                    "metric": metric,
                    "value": r.get(key),
                })
    pd.DataFrame(long_rows).to_csv(summary_dir / "plot_ready_metrics_long.csv", index=False)


def collect_existing_results(out_root: str | Path) -> List[Dict[str, Any]]:
    out_root = Path(out_root)
    experiments_dir = out_root / "experiments"
    rows = []
    if not experiments_dir.exists():
        return rows
    for exp_dir in sorted([p for p in experiments_dir.iterdir() if p.is_dir()]):
        meta = load_json(exp_dir / "experiment_meta.json")
        if not meta:
            continue
        final_subdir = meta.get("final_run_subdir")
        row = extract_result_row(
            exp_dir,
            experiment_id=meta.get("experiment_id", exp_dir.name),
            experiment_name=meta.get("experiment_name", exp_dir.name),
            model=meta.get("model", "unknown"),
            init=meta.get("initialization", "unknown"),
            synthetic_stage=meta.get("synthetic_stage", "unknown"),
            real_stage=meta.get("real_stage", "unknown"),
            final_run_subdir=final_subdir,
            split="test",
        )
        rows.append(row)
    return rows
