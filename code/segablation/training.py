from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .common import (
    IGNORE_INDEX,
    confusion_matrix_np,
    metrics_from_cm,
    save_prediction_bundle,
    save_manifest_csv,
    write_json,
    read_rgb,
    read_gray,
    resize_image_and_mask,
)
from .datasets import collate_with_meta


class CrossEntropyDiceLoss(nn.Module):
    def __init__(self, num_classes: int, ce_weight: float = 1.0, dice_weight: float = 1.0, ignore_index: int = IGNORE_INDEX):
        super().__init__()
        self.num_classes = num_classes
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = nn.functional.cross_entropy(logits, target, ignore_index=self.ignore_index)
        valid = target != self.ignore_index
        if not torch.any(valid):
            return self.ce_weight * ce
        probs = torch.softmax(logits, dim=1)
        target_safe = target.clone()
        target_safe[~valid] = 0
        onehot = nn.functional.one_hot(target_safe, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        valid_f = valid.unsqueeze(1).float()
        probs = probs * valid_f
        onehot = onehot * valid_f
        dims = (0, 2, 3)
        inter = torch.sum(probs * onehot, dims)
        card = torch.sum(probs + onehot, dims)
        dice = (2 * inter + 1e-6) / (card + 1e-6)
        return self.ce_weight * ce + self.dice_weight * (1.0 - dice.mean())


def make_loader(dataset, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
        collate_fn=collate_with_meta,
    )


def build_optimizer(model: nn.Module, optimizer_name: str, lr: float, weight_decay: float, momentum: float = 0.9):
    optimizer_name = optimizer_name.lower()
    if optimizer_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if optimizer_name in {"sgdm", "sgd"}:
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=False)
    raise ValueError("optimizer must be adamw or sgdm")


SEGUNET_ENCODER_MODULES = ("e1", "e2", "e3", "e4", "bridge")
SEGUNET_DECODER_MODULES = ("up1", "d1", "up2", "d2", "up3", "d3", "up4", "d4", "out")


def _named_existing_modules(model: nn.Module, names: Sequence[str]) -> List[nn.Module]:
    return [getattr(model, name) for name in names if hasattr(model, name)]


def set_segunet_encoder_trainable(model: nn.Module, trainable: bool) -> None:
    """Freeze/unfreeze the SegUNet encoder/bridge modules.

    The current V8 experiments are SegUNet-only. For non-SegUNet models this
    function falls back to doing nothing, which keeps backward compatibility.
    """
    for module in _named_existing_modules(model, SEGUNET_ENCODER_MODULES):
        for p in module.parameters():
            p.requires_grad = trainable


def set_all_trainable(model: nn.Module, trainable: bool = True) -> None:
    for p in model.parameters():
        p.requires_grad = trainable


def _unique_params(params):
    seen = set()
    out = []
    for p in params:
        if id(p) not in seen:
            seen.add(id(p))
            out.append(p)
    return out


def _segunet_param_groups(model: nn.Module, encoder_lr: float, decoder_lr: float):
    encoder_params = []
    for module in _named_existing_modules(model, SEGUNET_ENCODER_MODULES):
        encoder_params.extend([p for p in module.parameters() if p.requires_grad])
    decoder_params = []
    for module in _named_existing_modules(model, SEGUNET_DECODER_MODULES):
        decoder_params.extend([p for p in module.parameters() if p.requires_grad])

    encoder_params = _unique_params(encoder_params)
    decoder_params = _unique_params(decoder_params)

    # Fallback for models that do not expose SegUNet-style module names.
    if not encoder_params and not decoder_params:
        return [{"params": [p for p in model.parameters() if p.requires_grad], "lr": decoder_lr}]

    groups = []
    if encoder_params:
        groups.append({"params": encoder_params, "lr": encoder_lr, "name": "encoder"})
    if decoder_params:
        groups.append({"params": decoder_params, "lr": decoder_lr, "name": "decoder"})
    return groups


def build_optimizer_with_optional_groups(
    model: nn.Module,
    optimizer_name: str,
    lr: float,
    weight_decay: float,
    momentum: float = 0.9,
    encoder_lr: Optional[float] = None,
    decoder_lr: Optional[float] = None,
):
    """Build optimizer with either a single LR or SegUNet encoder/decoder LRs."""
    optimizer_name = optimizer_name.lower()
    if encoder_lr is None and decoder_lr is None:
        params = [p for p in model.parameters() if p.requires_grad]
        if not params:
            raise ValueError("No trainable parameters found.")
        param_groups = params
    else:
        enc_lr = float(encoder_lr if encoder_lr is not None else lr)
        dec_lr = float(decoder_lr if decoder_lr is not None else lr)
        param_groups = _segunet_param_groups(model, enc_lr, dec_lr)
        if not any(len(g.get("params", [])) for g in param_groups):
            raise ValueError("No trainable parameters found for grouped optimizer.")

    if optimizer_name == "adamw":
        return torch.optim.AdamW(param_groups, lr=lr, weight_decay=weight_decay)
    if optimizer_name in {"sgdm", "sgd"}:
        return torch.optim.SGD(param_groups, lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=False)
    raise ValueError("optimizer must be adamw or sgdm")


def train_one_epoch(model, loader, optimizer, criterion, device, amp: bool) -> float:
    model.train()
    losses = []
    scaler = torch.cuda.amp.GradScaler(enabled=amp and device.type == "cuda")
    for images, masks, _ in tqdm(loader, desc="train", leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp and device.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, masks)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else float("nan")


@torch.no_grad()
def evaluate(model, loader, criterion, device, num_classes: int, class_names: Dict[int, str], report_class_ids: Sequence[int]):
    model.eval()
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    losses = []
    per_sample_rows = []
    pred_cache: Dict[str, np.ndarray] = {}
    for images, masks, meta in tqdm(loader, desc="eval", leave=False):
        images = images.to(device, non_blocking=True)
        masks_dev = masks.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, masks_dev)
        losses.append(float(loss.detach().cpu()))
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy().astype(np.uint8)
        gts = masks.numpy().astype(np.uint8)
        for i, m in enumerate(meta):
            sample_id = str(m.get("id") or m.get("sample_id") or m.get("name") or len(pred_cache))
            pred_cache[sample_id] = preds[i]
            cm_i = confusion_matrix_np(gts[i], preds[i], num_classes=num_classes)
            cm += cm_i
            ssum, _ = metrics_from_cm(cm_i, class_names, report_class_ids=report_class_ids)
            row = {"sample_id": sample_id, **ssum}
            for c in report_class_ids:
                sub_sum, rows = metrics_from_cm(cm_i, class_names, report_class_ids=[c])
                row[f"dice_{class_names[int(c)]}"] = rows[0]["dice"]
                row[f"iou_{class_names[int(c)]}"] = rows[0]["iou"]
            per_sample_rows.append(row)
    summary, class_rows = metrics_from_cm(cm, class_names, report_class_ids=report_class_ids)
    summary["loss"] = float(np.mean(losses)) if losses else float("nan")
    return summary, class_rows, per_sample_rows, pred_cache


def fit_model(
    model: nn.Module,
    train_ds,
    val_ds,
    test_ds,
    out_dir: Path,
    num_classes: int,
    class_names: Dict[int, str],
    report_class_ids: Sequence[int],
    epochs: int,
    batch_size: int,
    num_workers: int,
    optimizer_name: str,
    lr: float,
    weight_decay: float,
    momentum: float,
    amp: bool,
    device: torch.device,
    save_visuals: bool,
    palette,
    overlay_alpha: float,
    select_metric: str = "mean_iou_reported_classes",
    early_stop_patience: int = 0,
    early_stop_min_delta: float = 0.0,
    freeze_encoder: bool = False,
    encoder_lr: Optional[float] = None,
    decoder_lr: Optional[float] = None,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    train_loader = make_loader(train_ds, batch_size, True, num_workers)
    val_loader = make_loader(val_ds, batch_size, False, num_workers)
    test_loader = make_loader(test_ds, batch_size, False, num_workers) if test_ds is not None else None

    model = model.to(device)
    set_all_trainable(model, True)
    if freeze_encoder:
        set_segunet_encoder_trainable(model, False)
    criterion = CrossEntropyDiceLoss(num_classes=num_classes).to(device)
    optimizer = build_optimizer_with_optional_groups(
        model,
        optimizer_name=optimizer_name,
        lr=lr,
        weight_decay=weight_decay,
        momentum=momentum,
        encoder_lr=encoder_lr,
        decoder_lr=decoder_lr,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    log_rows = []
    best_metric = -1e9
    best_epoch = 0
    epochs_without_improvement = 0
    best_path = ckpt_dir / "best_model.pth"

    for epoch in range(1, epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, amp=amp)
        val_summary, val_class_rows, _, _ = evaluate(model, val_loader, criterion, device, num_classes, class_names, report_class_ids)
        scheduler.step()
        lr_values = {f"lr_group_{i}": g.get("lr", float("nan")) for i, g in enumerate(optimizer.param_groups)}
        row = {"epoch": epoch, "train_loss": tr_loss, "lr": optimizer.param_groups[0]["lr"], **lr_values, **{f"val_{k}": v for k, v in val_summary.items()}}
        log_rows.append(row)
        pd.DataFrame(log_rows).to_csv(out_dir / "training_log.csv", index=False)
        print(f"epoch {epoch}/{epochs} train_loss={tr_loss:.4f} val_{select_metric}={val_summary.get(select_metric, float('nan')):.4f}")
        metric = float(val_summary.get(select_metric, -1e9))
        improved = metric > (best_metric + float(early_stop_min_delta))
        if improved:
            best_metric = metric
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "val_summary": val_summary, "class_names": class_names}, best_path)
            pd.DataFrame(val_class_rows).to_csv(out_dir / "best_val_classwise_metrics.csv", index=False)
        else:
            epochs_without_improvement += 1

        if early_stop_patience and early_stop_patience > 0 and epochs_without_improvement >= early_stop_patience:
            print(
                f"early stopping at epoch {epoch}/{epochs}: "
                f"best_val_{select_metric}={best_metric:.4f} at epoch {best_epoch}, "
                f"patience={early_stop_patience}, min_delta={early_stop_min_delta}"
            )
            break

    last_epoch = log_rows[-1]["epoch"] if log_rows else 0
    torch.save({"model_state_dict": model.state_dict(), "epoch": last_epoch, "class_names": class_names}, ckpt_dir / "last_model.pth")
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])

    final = {
        "best_val_metric": best_metric,
        "best_val_epoch": best_epoch,
        "epochs_ran": log_rows[-1]["epoch"] if log_rows else 0,
        "early_stop_patience": early_stop_patience,
        "early_stop_min_delta": early_stop_min_delta,
        "early_stopped": bool(early_stop_patience and early_stop_patience > 0 and log_rows and log_rows[-1]["epoch"] < epochs),
        "freeze_encoder": freeze_encoder,
        "encoder_lr": encoder_lr,
        "decoder_lr": decoder_lr,
    }

    # Final evaluation/visualization pass.  We include train as well as val/test
    # so Experiment A can save prediction masks, color masks, overlays, and
    # per-sample metrics for every train image.  Train is evaluated with
    # shuffle=False to make saved prediction bundles deterministic.
    for split_name, ds in [("train", train_ds), ("val", val_ds), ("test", test_ds)]:
        if ds is None:
            continue
        loader = make_loader(ds, batch_size, False, num_workers)
        summary, class_rows, sample_rows, pred_cache = evaluate(model, loader, criterion, device, num_classes, class_names, report_class_ids)
        final[split_name] = summary
        pd.DataFrame(class_rows).to_csv(out_dir / f"{split_name}_classwise_metrics.csv", index=False)
        pd.DataFrame(sample_rows).to_csv(out_dir / f"{split_name}_per_sample_metrics.csv", index=False)
        if save_visuals:
            vis_rows = []
            split_vis = out_dir / "visualization_cache" / split_name
            for i in range(len(ds)):
                image_t, mask_t, meta = ds[i]
                sample_id = str(meta.get("id") or meta.get("sample_id") or i)
                pred = pred_cache.get(sample_id)
                if pred is None:
                    continue
                img_np = (image_t.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
                gt_np = mask_t.numpy().astype(np.uint8)
                paths = save_prediction_bundle(split_vis, sample_id, img_np, gt_np, pred, palette=palette, alpha=overlay_alpha, include_zero=False, extra={"meta": meta})
                vis_rows.append({"sample_id": sample_id, **paths})
            save_manifest_csv(split_vis / "visualization_manifest.csv", vis_rows)
    write_json(out_dir / "metrics_summary.json", final)
    return final
