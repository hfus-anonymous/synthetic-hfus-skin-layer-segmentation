#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import torch
import pandas as pd

from .common import EPI_SLEB_CLASS_NAMES, EPI_SLEB_PALETTE_BGR, set_seed, write_json
from .datasets import ImageMaskFolderDataset
from .models import build_model, get_model_source
from .training import fit_model
from .result_io import extract_result_row, save_experiment_result_txt, write_json as write_json2



def filter_manifest(in_csv: Path, out_csv: Path, split: str | None = None, sources: list[str] | None = None) -> Path:
    """Filter a ROI manifest by split/source and write a temporary CSV for training."""
    df = pd.read_csv(in_csv)
    if split is not None and "split" in df.columns:
        df = df[df["split"].astype(str).str.lower() == str(split).lower()].copy()
    if sources is not None and "source" in df.columns:
        src_set = {str(x).lower() for x in sources}
        df = df[df["source"].astype(str).str.lower().isin(src_set)].copy()
    if df.empty:
        raise ValueError(f"No rows after filtering manifest={in_csv}, split={split}, sources={sources}")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return out_csv

def _make_model(args):
    return build_model(
        args.model,
        in_channels=3,
        num_classes=3,
        encoder=args.encoder,
        encoder_weights=args.encoder_weights,
        segformer_checkpoint=args.segformer_checkpoint,
        segformer_local_files_only=args.segformer_local_files_only,
        id2label=EPI_SLEB_CLASS_NAMES,
    )


def _optimizer_for_model(model_name: str, requested: str) -> str:
    if requested != 'auto':
        return requested
    return 'sgdm' if model_name.lower() == 'segunet' else 'adamw'


def _default_lr_for_model(model_name: str, mode: str, requested: float | None) -> float:
    if requested is not None:
        return requested
    if model_name.lower() == 'segunet':
        return 1e-3 if mode != 'finetune' else 1e-4
    return 1e-4 if mode != 'finetune' else 1e-5


def _exp_dir(args) -> Path:
    # This is the core difference from train_protocol.py:
    # v2 writes directly under outputs/segmentation_ablation/experiments_v2/<ID>
    return Path(args.experiments_dir) / args.experiment_id


def _make_meta(args, final_run_subdir: str | None) -> Dict:
    return {
        'experiment_id': args.experiment_id,
        'experiment_name': args.experiment_name,
        'model': args.model,
        'initialization': args.initialization,
        'synthetic_stage': args.synthetic_stage,
        'real_stage': args.real_stage,
        'final_run_subdir': final_run_subdir,
        'v2_budget_matched': True,
        'v2_experiments_dir': str(args.experiments_dir),
    }


def _write_final_text(args, exp_dir: Path, final_run_subdir: str | None, notes: List[str]) -> Dict:
    row = extract_result_row(
        exp_dir,
        experiment_id=args.experiment_id,
        experiment_name=args.experiment_name,
        model=args.model,
        init=args.initialization,
        synthetic_stage=args.synthetic_stage,
        real_stage=args.real_stage,
        final_run_subdir=final_run_subdir,
    )
    row['v2_budget_matched'] = True
    row['total_epochs_config'] = args.total_epochs_config
    row['synthetic_epochs_config'] = args.pretrain_epochs if args.mode == 'synthetic_pretrain_real_finetune' else (args.epochs if args.mode == 'synthetic_only' else 0)
    row['real_epochs_config'] = args.finetune_epochs if args.mode == 'synthetic_pretrain_real_finetune' else (args.epochs if args.mode == 'real_only' else 0)
    save_experiment_result_txt(row, exp_dir / 'result.txt', config=vars(args), notes=notes)
    write_json2(exp_dir / 'result.json', row)
    return row


def _write_run_config(args, exp_dir: Path, model) -> None:
    cfg = vars(args).copy()
    cfg['model_source'] = get_model_source(model)
    cfg['output_policy'] = 'v2 results are saved under outputs/segmentation_ablation/experiments_v2, not experiments.'
    write_json(exp_dir / 'run_config.json', cfg)


def _reset_module_parameters(module: torch.nn.Module) -> None:
    for child in module.modules():
        if hasattr(child, "reset_parameters"):
            try:
                child.reset_parameters()
            except Exception:
                pass


def _reset_decoder_for_real_finetune(model: torch.nn.Module, model_name: str) -> None:
    """Reduce synthetic-to-real negative transfer before stage-2 real fine-tuning.

    The previous V4 protocol loaded the full synthetic-pretrained network into
    real fine-tuning. Because synthetic-only direct transfer is weak in this
    dataset, the decoder/classifier can encode synthetic boundary statistics that
    conflict with real HFUS. This function keeps lower-level representation
    transfer where possible, while resetting the segmentation decoder/head before
    real fine-tuning.
    """
    name = model_name.lower()

    # PyTorch released-style SegUNet: keep encoder blocks, reset decoder/head.
    if name == "segunet":
        for attr in ["up1", "d1", "up2", "d2", "up3", "d3", "up4", "d4", "out"]:
            if hasattr(model, attr):
                _reset_module_parameters(getattr(model, attr))
        return

    # SMP models are wrapped as NormalizedModel(model=<smp model>).
    inner = getattr(model, "model", model)
    for attr in ["decoder", "segmentation_head", "classification_head"]:
        if hasattr(inner, attr) and getattr(inner, attr) is not None:
            _reset_module_parameters(getattr(inner, attr))

    # SegFormer wrapper: NormalizedModel -> HFSegFormerWrapper -> SegformerForSemanticSegmentation.
    hf_wrapper = getattr(model, "model", None)
    hf_model = getattr(hf_wrapper, "model", None)
    if hf_model is not None and hasattr(hf_model, "decode_head"):
        _reset_module_parameters(hf_model.decode_head)


def _apply_transfer_strategy(model: torch.nn.Module, args) -> str:
    strategy = getattr(args, "transfer_strategy", "full")
    if strategy == "full":
        return "full synthetic-pretrained weights loaded into real fine-tuning"
    if strategy == "reset_decoder":
        _reset_decoder_for_real_finetune(model, args.model)
        return "synthetic-pretrained encoder/backbone retained; decoder/head reset before real fine-tuning"
    raise ValueError(f"Unknown transfer_strategy: {strategy}")


def run_synthetic_only(args) -> Dict:
    exp_dir = _exp_dir(args)
    exp_dir.mkdir(parents=True, exist_ok=True)
    tmp = exp_dir / 'manifests'
    train_csv = filter_manifest(Path(args.synthetic_roi_manifest), tmp / 'synthetic_train_original_augmented.csv', split='train', sources=['original', 'augmented'])
    val_csv = filter_manifest(Path(args.synthetic_roi_manifest), tmp / 'synthetic_val_original_only.csv', split='val', sources=['original'])
    test_csv = filter_manifest(Path(args.real_roi_manifest), tmp / 'real_test.csv', split='test')

    model = _make_model(args)
    args.initialization = 'fresh' if args.model == 'segunet' else 'fixed_pretrained_or_configured'
    args.synthetic_stage = 'synthetic_only_train'
    args.real_stage = 'none_train_real_test_only'
    _write_run_config(args, exp_dir, model)
    write_json2(exp_dir / 'experiment_meta.json', _make_meta(args, None))

    optimizer = _optimizer_for_model(args.model, args.optimizer)
    lr = _default_lr_for_model(args.model, 'train', args.lr)
    fit_model(
        model,
        ImageMaskFolderDataset(train_csv),
        ImageMaskFolderDataset(val_csv),
        ImageMaskFolderDataset(test_csv),
        exp_dir,
        3,
        EPI_SLEB_CLASS_NAMES,
        [1, 2],
        args.epochs,
        args.batch_size,
        args.num_workers,
        optimizer,
        lr,
        args.weight_decay,
        args.momentum,
        args.amp,
        torch.device(args.device),
        args.save_visuals,
        EPI_SLEB_PALETTE_BGR,
        0.45,
        select_metric='mean_dice_reported_classes',
        early_stop_patience=args.early_stop_patience,
        early_stop_min_delta=args.early_stop_min_delta,
    )
    return _write_final_text(args, exp_dir, None, ['V2 synthetic-only run. Synthetic train uses original+augmented; synthetic val uses original-only; final evaluation uses fixed real test.'])


def run_real_only(args) -> Dict:
    exp_dir = _exp_dir(args)
    exp_dir.mkdir(parents=True, exist_ok=True)
    tmp = exp_dir / 'manifests'
    train_csv = filter_manifest(Path(args.real_roi_manifest), tmp / 'real_train.csv', split='train')
    val_csv = filter_manifest(Path(args.real_roi_manifest), tmp / 'real_val.csv', split='val')
    test_csv = filter_manifest(Path(args.real_roi_manifest), tmp / 'real_test.csv', split='test')

    model = _make_model(args)
    args.initialization = 'fresh' if args.model == 'segunet' else 'fixed_pretrained_or_configured'
    args.synthetic_stage = 'none'
    args.real_stage = 'real_only_train'
    _write_run_config(args, exp_dir, model)
    write_json2(exp_dir / 'experiment_meta.json', _make_meta(args, None))

    optimizer = _optimizer_for_model(args.model, args.optimizer)
    lr = _default_lr_for_model(args.model, 'finetune', args.lr if args.lr is not None else args.finetune_lr)
    fit_model(
        model,
        ImageMaskFolderDataset(train_csv),
        ImageMaskFolderDataset(val_csv),
        ImageMaskFolderDataset(test_csv),
        exp_dir,
        3,
        EPI_SLEB_CLASS_NAMES,
        [1, 2],
        args.epochs,
        args.batch_size,
        args.num_workers,
        optimizer,
        lr,
        args.weight_decay,
        args.momentum,
        args.amp,
        torch.device(args.device),
        args.save_visuals,
        EPI_SLEB_PALETTE_BGR,
        0.45,
        select_metric='mean_dice_reported_classes',
        early_stop_patience=args.early_stop_patience,
        early_stop_min_delta=args.early_stop_min_delta,
    )
    return _write_final_text(args, exp_dir, None, ['V2 real-only control. Used the same fixed real train/val/test split as synthetic+real fine-tuning arms.'])


def run_synthetic_pretrain_real_finetune(args) -> Dict:
    exp_dir = _exp_dir(args)
    exp_dir.mkdir(parents=True, exist_ok=True)
    tmp = exp_dir / 'manifests'
    synth_train_csv = filter_manifest(Path(args.synthetic_roi_manifest), tmp / 'synthetic_train_original_augmented.csv', split='train', sources=['original', 'augmented'])
    synth_val_csv = filter_manifest(Path(args.synthetic_roi_manifest), tmp / 'synthetic_val_original_only.csv', split='val', sources=['original'])
    real_train_csv = filter_manifest(Path(args.real_roi_manifest), tmp / 'real_train.csv', split='train')
    real_val_csv = filter_manifest(Path(args.real_roi_manifest), tmp / 'real_val.csv', split='val')
    real_test_csv = filter_manifest(Path(args.real_roi_manifest), tmp / 'real_test.csv', split='test')

    model = _make_model(args)
    args.initialization = 'fresh' if args.model == 'segunet' else 'fixed_pretrained_or_configured'
    args.synthetic_stage = 'synthetic_pretrain'
    args.real_stage = 'real_finetune'
    _write_run_config(args, exp_dir, model)
    write_json2(exp_dir / 'experiment_meta.json', _make_meta(args, 'stage2_real_finetune'))

    optimizer = _optimizer_for_model(args.model, args.optimizer)
    pre_lr = _default_lr_for_model(args.model, 'train', args.pretrain_lr)
    ft_lr = _default_lr_for_model(args.model, 'finetune', args.finetune_lr)
    device = torch.device(args.device)

    stage1 = exp_dir / 'stage1_synthetic_pretrain'
    fit_model(
        model,
        ImageMaskFolderDataset(synth_train_csv),
        ImageMaskFolderDataset(synth_val_csv),
        None,
        stage1,
        3,
        EPI_SLEB_CLASS_NAMES,
        [1, 2],
        args.pretrain_epochs,
        args.batch_size,
        args.num_workers,
        optimizer,
        pre_lr,
        args.weight_decay,
        args.momentum,
        args.amp,
        device,
        args.save_visuals,
        EPI_SLEB_PALETTE_BGR,
        0.45,
        select_metric='mean_dice_reported_classes',
        early_stop_patience=args.early_stop_patience,
        early_stop_min_delta=args.early_stop_min_delta,
    )
    ckpt = torch.load(stage1 / 'checkpoints' / 'best_model.pth', map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    transfer_note = _apply_transfer_strategy(model, args)
    write_json2(exp_dir / 'transfer_strategy.json', {'transfer_strategy': args.transfer_strategy, 'note': transfer_note})

    real_train_ds = ImageMaskFolderDataset(real_train_csv)
    real_val_ds = ImageMaskFolderDataset(real_val_csv)
    real_test_ds = ImageMaskFolderDataset(real_test_csv)

    real_stage_notes = []
    if args.real_stage_strategy in {'staged', 'staged_difflr'} and args.decoder_warmup_epochs > 0:
        warmup_epochs = min(int(args.decoder_warmup_epochs), int(args.finetune_epochs))
        stage2a = exp_dir / 'stage2a_real_decoder_warmup'
        fit_model(
            model,
            real_train_ds,
            real_val_ds,
            None,
            stage2a,
            3,
            EPI_SLEB_CLASS_NAMES,
            [1, 2],
            warmup_epochs,
            args.batch_size,
            args.num_workers,
            optimizer,
            ft_lr,
            args.weight_decay,
            args.momentum,
            args.amp,
            device,
            args.save_visuals,
            EPI_SLEB_PALETTE_BGR,
            0.45,
            select_metric='mean_dice_reported_classes',
            early_stop_patience=0,
            early_stop_min_delta=args.early_stop_min_delta,
            freeze_encoder=True,
            encoder_lr=None,
            decoder_lr=None,
        )
        real_stage_notes.append(f'Stage 2a decoder/head-only warm-up for {warmup_epochs} epochs with encoder frozen.')
        remaining_epochs = int(args.finetune_epochs) - warmup_epochs
    else:
        remaining_epochs = int(args.finetune_epochs)

    if remaining_epochs <= 0:
        final_subdir = 'stage2a_real_decoder_warmup'
        real_stage_notes.append('No remaining full fine-tuning epochs after decoder warm-up.')
    else:
        final_subdir = 'stage2b_real_full_finetune' if args.real_stage_strategy in {'staged', 'staged_difflr'} else 'stage2_real_finetune'
        stage2 = exp_dir / final_subdir

        use_difflr = args.real_stage_strategy in {'difflr', 'staged_difflr'} or args.encoder_finetune_lr is not None or args.decoder_finetune_lr is not None
        enc_lr = args.encoder_finetune_lr if use_difflr else None
        dec_lr = args.decoder_finetune_lr if use_difflr else None

        fit_model(
            model,
            real_train_ds,
            real_val_ds,
            real_test_ds,
            stage2,
            3,
            EPI_SLEB_CLASS_NAMES,
            [1, 2],
            remaining_epochs,
            args.batch_size,
            args.num_workers,
            optimizer,
            ft_lr,
            args.weight_decay,
            args.momentum,
            args.amp,
            device,
            args.save_visuals,
            EPI_SLEB_PALETTE_BGR,
            0.45,
            select_metric='mean_dice_reported_classes',
            early_stop_patience=args.early_stop_patience,
            early_stop_min_delta=args.early_stop_min_delta,
            freeze_encoder=False,
            encoder_lr=enc_lr,
            decoder_lr=dec_lr,
        )
        if use_difflr:
            real_stage_notes.append(f'Stage 2 full fine-tuning used differential LR: encoder_lr={enc_lr}, decoder_lr={dec_lr}.')
        else:
            real_stage_notes.append('Stage 2 used standard full-network real fine-tuning.')

    write_json2(exp_dir / 'real_stage_strategy.json', {
        'real_stage_strategy': args.real_stage_strategy,
        'decoder_warmup_epochs': args.decoder_warmup_epochs,
        'encoder_finetune_lr': args.encoder_finetune_lr,
        'decoder_finetune_lr': args.decoder_finetune_lr,
        'final_run_subdir': final_subdir,
        'notes': real_stage_notes,
    })

    return _write_final_text(args, exp_dir, final_subdir, [
        'V2 two-stage run. Stage 1: synthetic pretraining. Stage 2: real fine-tuning. Final evaluation uses fixed real test.',
        f'Transfer strategy: {args.transfer_strategy}',
        f'Real-stage strategy: {args.real_stage_strategy}',
        *real_stage_notes,
    ])


def main():
    p = argparse.ArgumentParser(description='Run one v2 budget-matched segmentation ablation experiment.')
    p.add_argument('--experiment_id', required=True)
    p.add_argument('--experiment_name', required=True)
    p.add_argument('--mode', required=True, choices=['synthetic_only', 'real_only', 'synthetic_pretrain_real_finetune'])
    p.add_argument('--model', required=True, choices=['segunet', 'unet', 'deeplabv3plus', 'segformer'])
    p.add_argument('--synthetic_roi_manifest', default='./outputs/segmentation_ablation/prepared/v66_synthetic_roi_64x128/manifest.csv')
    p.add_argument('--real_roi_manifest', default='./outputs/segmentation_ablation/prepared/mendeley_real_roi_64x128/manifest.csv')
    p.add_argument('--out_root', default='./outputs/segmentation_ablation')
    p.add_argument('--experiments_dir', default='./outputs/segmentation_ablation/experiments_v2')
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--pretrain_epochs', type=int, default=160)
    p.add_argument('--finetune_epochs', type=int, default=60)
    p.add_argument('--total_epochs_config', type=int, default=0)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--lr', type=float, default=None)
    p.add_argument('--pretrain_lr', type=float, default=None)
    p.add_argument('--finetune_lr', type=float, default=None)
    p.add_argument('--weight_decay', type=float, default=5e-4)
    p.add_argument('--momentum', type=float, default=0.9)
    p.add_argument('--optimizer', choices=['auto', 'sgdm', 'adamw'], default='auto')
    p.add_argument('--encoder', default='resnet34')
    p.add_argument('--encoder_weights', default='imagenet')
    p.add_argument('--segformer_checkpoint', default='nvidia/segformer-b0-finetuned-ade-512-512')
    p.add_argument('--segformer_local_files_only', action='store_true')
    p.add_argument('--num_workers', type=int, default=6)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--amp', action='store_true')
    p.add_argument('--save_visuals', action='store_true')
    p.add_argument('--seed', type=int, default=66)
    p.add_argument('--early_stop_patience', type=int, default=0,
                   help='Stop training if validation metric does not improve for this many epochs. 0 disables early stopping.')
    p.add_argument('--early_stop_min_delta', type=float, default=0.0,
                   help='Minimum validation metric improvement required to reset early-stopping patience.')
    p.add_argument('--transfer_strategy', choices=['full', 'reset_decoder'], default='full',
                   help='For synthetic_pretrain_real_finetune: full keeps all synthetic weights; reset_decoder keeps the synthetic encoder/backbone but resets decoder/head before real fine-tuning.')
    p.add_argument('--real_stage_strategy', choices=['standard', 'staged', 'difflr', 'staged_difflr'], default='standard',
                   help='Real fine-tuning protocol. staged uses decoder-only warm-up before full fine-tuning; difflr uses lower encoder LR and higher decoder LR.')
    p.add_argument('--decoder_warmup_epochs', type=int, default=0,
                   help='For staged protocols: number of real fine-tuning epochs with encoder frozen.')
    p.add_argument('--encoder_finetune_lr', type=float, default=None,
                   help='Optional encoder LR for differential-LR real fine-tuning.')
    p.add_argument('--decoder_finetune_lr', type=float, default=None,
                   help='Optional decoder/head LR for differential-LR real fine-tuning.')
    args = p.parse_args()

    set_seed(args.seed)
    args.initialization = ''
    args.synthetic_stage = ''
    args.real_stage = ''

    if args.mode == 'synthetic_only':
        run_synthetic_only(args)
    elif args.mode == 'real_only':
        run_real_only(args)
    elif args.mode == 'synthetic_pretrain_real_finetune':
        run_synthetic_pretrain_real_finetune(args)


if __name__ == '__main__':
    main()
