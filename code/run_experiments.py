#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List

CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

MODEL_SPECS = [
    ("SegUNet", "segunet"),
    ("UNet", "unet"),
    ("DeepLabV3Plus", "deeplabv3plus"),
    ("SegFormer", "segformer"),
]


@dataclass
class ExperimentSpec:
    experiment_id: str
    experiment_name: str
    model_label: str
    model_arg: str
    mode_arg: str
    training_regime: str
    synthetic_epochs: int
    real_epochs: int
    total_epochs: int
    transfer_strategy: str
    real_stage_strategy: str
    batch_size: int
    lr: float
    pretrain_lr: float
    finetune_lr: float
    optimizer: str
    weight_decay: float
    momentum: float
    encoder: str
    encoder_weights: str
    seed: int
    notes: str


def model_specific_lrs(model_arg: str) -> tuple[float, float, float]:
    """Return real-only LR, synthetic pretrain LR, and real fine-tune LR.

    Final paper protocol:
      SegUNet: SGDM-style LR scale found stable in the released-SegUNet setting.
      UNet/DeepLabV3+/SegFormer: lower AdamW LR scale for pretrained encoders/backbones.
    """
    if model_arg == "segunet":
        return 1e-4, 1e-3, 1e-4
    return 1e-5, 1e-4, 1e-5


def build_experiments(models: Iterable[str], batch_size: int, seed: int) -> List[ExperimentSpec]:
    wanted = {m.lower() for m in models}
    out: List[ExperimentSpec] = []
    for model_label, model_arg in MODEL_SPECS:
        if model_label.lower() not in wanted and model_arg.lower() not in wanted:
            continue
        real_lr, pretrain_lr, finetune_lr = model_specific_lrs(model_arg)

        out.append(ExperimentSpec(
            experiment_id=f"{model_label}_R300",
            experiment_name=f"{model_label} real-only 300",
            model_label=model_label,
            model_arg=model_arg,
            mode_arg="real_only",
            training_regime="Real only 300",
            synthetic_epochs=0,
            real_epochs=300,
            total_epochs=300,
            transfer_strategy="full",
            real_stage_strategy="standard",
            batch_size=batch_size,
            lr=real_lr,
            pretrain_lr=pretrain_lr,
            finetune_lr=finetune_lr,
            optimizer="auto",
            weight_decay=5e-4,
            momentum=0.9,
            encoder="resnet34",
            encoder_weights="imagenet",
            seed=seed,
            notes="Model-specific-LR real-only baseline.",
        ))

        out.append(ExperimentSpec(
            experiment_id=f"{model_label}_S100R300_ENC",
            experiment_name=f"{model_label} synthetic100 encoder warm-up plus real fine-tuning 300",
            model_label=model_label,
            model_arg=model_arg,
            mode_arg="synthetic_pretrain_real_finetune",
            training_regime="Synthetic100 encoder warm-up + RealFT300",
            synthetic_epochs=100,
            real_epochs=300,
            total_epochs=400,
            transfer_strategy="reset_decoder",
            real_stage_strategy="standard",
            batch_size=batch_size,
            lr=real_lr,
            pretrain_lr=pretrain_lr,
            finetune_lr=finetune_lr,
            optimizer="auto",
            weight_decay=5e-4,
            momentum=0.9,
            encoder="resnet34",
            encoder_weights="imagenet",
            seed=seed,
            notes="Synthetic-pretrained encoder/backbone retained; decoder/head reset before real-domain fine-tuning.",
        ))
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the final model-specific-LR HFUS segmentation experiments.")
    p.add_argument("--synthetic_roi_manifest", type=Path, default=None)
    p.add_argument("--real_roi_manifest", type=Path, default=None)
    p.add_argument("--out_root", type=Path, default=REPO_ROOT / "outputs")
    p.add_argument("--data_root", type=Path, default=REPO_ROOT / "data")
    p.add_argument("--skip_prepare", action="store_true", help="Do not run prepare_data.py automatically when ROI manifests are missing.")
    p.add_argument("--prepare_no_roi_visuals", action="store_true", help="Pass --no_roi_visuals to prepare_data.py when auto-preparing manifests.")
    p.add_argument("--experiments_dir", type=Path, default=None)
    p.add_argument("--models", default="SegUNet,UNet,DeepLabV3Plus,SegFormer",
                   help="Comma-separated model labels to run. Example: SegUNet,UNet")
    p.add_argument("--only", default="", help="Comma-separated experiment IDs to run.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--num_workers", type=int, default=6)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--seed", type=int, default=66)
    p.add_argument("--early_stop_patience", type=int, default=40)
    p.add_argument("--early_stop_min_delta", type=float, default=0.0001)
    p.add_argument("--segformer_checkpoint", default="nvidia/segformer-b0-finetuned-ade-512-512")
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--save_visuals", action="store_true", default=True)
    p.add_argument("--no_save_visuals", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--clean_existing", action="store_true", help="Remove the model_lr experiment directory before running.")
    p.add_argument("--skip_existing", action="store_true", help="Skip experiments with existing result.json.")
    p.add_argument("--continue_on_error", action="store_true")
    # Smoke-test overrides.
    p.add_argument("--override_real_epochs", type=int, default=None)
    p.add_argument("--override_synthetic_epochs", type=int, default=None)
    return p.parse_args()


def setup(args: argparse.Namespace) -> argparse.Namespace:
    args.out_root = args.out_root.resolve()
    args.data_root = args.data_root.resolve()
    if args.synthetic_roi_manifest is None:
        args.synthetic_roi_manifest = args.out_root / "prepared" / "v66_synthetic_roi_64x128" / "manifest.csv"
    if args.real_roi_manifest is None:
        args.real_roi_manifest = args.out_root / "prepared" / "mendeley_real_roi_64x128" / "manifest.csv"
    args.synthetic_roi_manifest = args.synthetic_roi_manifest.resolve()
    args.real_roi_manifest = args.real_roi_manifest.resolve()
    args.experiments_dir = (args.experiments_dir or args.out_root / "model_lr_experiments").resolve()
    if args.no_amp:
        args.amp = False
    if args.no_save_visuals:
        args.save_visuals = False
    return args


def select_experiments(exps: List[ExperimentSpec], only: str) -> List[ExperimentSpec]:
    if not only.strip():
        return exps
    wanted = {x.strip() for x in only.split(",") if x.strip()}
    existing = {e.experiment_id for e in exps}
    missing = sorted(wanted - existing)
    if missing:
        raise ValueError(f"Unknown experiment IDs: {missing}. Available={sorted(existing)}")
    return [e for e in exps if e.experiment_id in wanted]


def write_plan(exps: List[ExperimentSpec], experiments_dir: Path) -> None:
    experiments_dir.mkdir(parents=True, exist_ok=True)
    rows = [asdict(e) for e in exps]
    (experiments_dir / "model_lr_experiment_plan.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with (experiments_dir / "model_lr_experiment_plan.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for row in rows:
            w.writerow(row)


def write_run_config(exp: ExperimentSpec, exp_dir: Path, args: argparse.Namespace) -> None:
    exp_dir.mkdir(parents=True, exist_ok=True)
    cfg = asdict(exp)
    cfg.update({
        "synthetic_roi_manifest": str(args.synthetic_roi_manifest),
        "real_roi_manifest": str(args.real_roi_manifest),
        "out_root": str(args.out_root),
        "experiments_dir": str(args.experiments_dir),
        "device": args.device,
        "num_workers": args.num_workers,
        "amp": args.amp,
        "save_visuals": args.save_visuals,
        "protocol": "final_model_specific_lr_architecture_comparison",
    })
    (exp_dir / "run_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def write_meta(exp: ExperimentSpec, exp_dir: Path, status: str, message: str = "") -> None:
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "experiment_meta.json").write_text(json.dumps({
        "experiment_id": exp.experiment_id,
        "experiment_name": exp.experiment_name,
        "model_label": exp.model_label,
        "model": exp.model_arg,
        "mode": exp.mode_arg,
        "training_regime": exp.training_regime,
        "status": status,
        "message": message,
    }, indent=2), encoding="utf-8")


def command_for(exp: ExperimentSpec, args: argparse.Namespace) -> List[str]:
    synthetic_epochs = args.override_synthetic_epochs if args.override_synthetic_epochs is not None else exp.synthetic_epochs
    real_epochs = args.override_real_epochs if args.override_real_epochs is not None else exp.real_epochs
    if exp.mode_arg == "real_only":
        epochs = real_epochs
        total_epochs = real_epochs
    else:
        epochs = synthetic_epochs
        total_epochs = synthetic_epochs + real_epochs

    cmd = [
        sys.executable, "-m", "segablation.train_protocol_v2",
        "--experiment_id", exp.experiment_id,
        "--experiment_name", exp.experiment_name,
        "--model", exp.model_arg,
        "--mode", exp.mode_arg,
        "--synthetic_roi_manifest", str(args.synthetic_roi_manifest),
        "--real_roi_manifest", str(args.real_roi_manifest),
        "--out_root", str(args.out_root),
        "--experiments_dir", str(args.experiments_dir),
        "--epochs", str(epochs),
        "--pretrain_epochs", str(synthetic_epochs),
        "--finetune_epochs", str(real_epochs),
        "--total_epochs_config", str(total_epochs),
        "--batch_size", str(exp.batch_size),
        "--lr", str(exp.lr),
        "--pretrain_lr", str(exp.pretrain_lr),
        "--finetune_lr", str(exp.finetune_lr),
        "--weight_decay", str(exp.weight_decay),
        "--momentum", str(exp.momentum),
        "--optimizer", exp.optimizer,
        "--encoder", exp.encoder,
        "--encoder_weights", exp.encoder_weights,
        "--segformer_checkpoint", args.segformer_checkpoint,
        "--num_workers", str(args.num_workers),
        "--device", args.device,
        "--seed", str(exp.seed),
        "--early_stop_patience", str(args.early_stop_patience),
        "--early_stop_min_delta", str(args.early_stop_min_delta),
    ]
    if exp.mode_arg == "synthetic_pretrain_real_finetune":
        cmd += ["--transfer_strategy", exp.transfer_strategy, "--real_stage_strategy", exp.real_stage_strategy]
    if args.amp:
        cmd.append("--amp")
    if args.save_visuals:
        cmd.append("--save_visuals")
    return cmd


def shjoin(cmd: List[str]) -> str:
    return " ".join([("'" + str(x) + "'") if any(c.isspace() for c in str(x)) else str(x) for x in cmd])


def run_command(cmd: List[str], log_path: Path, dry_run: bool) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.open("a", encoding="utf-8").write(shjoin(cmd) + "\n")
    print("[RUN]", shjoin(cmd), flush=True)
    if dry_run:
        return
    env = os.environ.copy()
    env["PYTHONPATH"] = str(CODE_DIR) + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(cmd, check=True, cwd=CODE_DIR, env=env)


def clean_outputs(args: argparse.Namespace) -> None:
    d = args.experiments_dir
    if d.exists():
        print(f"[CLEAN] removing {d}", flush=True)
        shutil.rmtree(d)



def ensure_prepared_manifests(args: argparse.Namespace) -> None:
    """Run prepare_data.py automatically if ROI manifests are missing."""
    if args.synthetic_roi_manifest.exists() and args.real_roi_manifest.exists():
        print("[OK] prepared ROI manifests already exist.")
        print(f"     synthetic: {args.synthetic_roi_manifest}")
        print(f"     real     : {args.real_roi_manifest}")
        return
    if args.skip_prepare:
        missing = [str(p) for p in [args.synthetic_roi_manifest, args.real_roi_manifest] if not p.exists()]
        raise FileNotFoundError("Prepared manifest(s) missing and --skip_prepare was used: " + ", ".join(missing))

    cmd = [
        sys.executable, str(CODE_DIR / "prepare_data.py"),
        "--data_root", str(args.data_root),
        "--out_root", str(args.out_root),
    ]
    if getattr(args, "prepare_no_roi_visuals", False):
        cmd.append("--no_roi_visuals")
    print("[PREPARE]", shjoin(cmd), flush=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(CODE_DIR) + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(cmd, check=True, cwd=CODE_DIR, env=env)

    if not args.synthetic_roi_manifest.exists():
        raise FileNotFoundError(f"synthetic_roi_manifest still missing after prepare_data.py: {args.synthetic_roi_manifest}")
    if not args.real_roi_manifest.exists():
        raise FileNotFoundError(f"real_roi_manifest still missing after prepare_data.py: {args.real_roi_manifest}")


def main() -> None:
    args = setup(parse_args())
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    experiments = build_experiments(models=models, batch_size=args.batch_size, seed=args.seed)
    selected = select_experiments(experiments, args.only)

    if args.clean_existing and not args.skip_existing and not args.dry_run:
        clean_outputs(args)

    if not args.dry_run:
        ensure_prepared_manifests(args)

    args.experiments_dir.mkdir(parents=True, exist_ok=True)
    write_plan(experiments, args.experiments_dir)

    print(f"[INFO] selected experiments: {[e.experiment_id for e in selected]}")
    log_path = args.experiments_dir / "commands_model_lr.log"
    if log_path.exists() and not args.skip_existing and not args.dry_run:
        log_path.unlink()

    failures: List[str] = []
    for exp in selected:
        exp_dir = args.experiments_dir / exp.experiment_id
        if args.skip_existing and (exp_dir / "result.json").exists():
            print(f"[SKIP] {exp.experiment_id}: result.json exists.", flush=True)
            continue
        write_run_config(exp, exp_dir, args)
        write_meta(exp, exp_dir, "started", "Training command launched or prepared.")
        try:
            run_command(command_for(exp, args), log_path, dry_run=args.dry_run)
            write_meta(exp, exp_dir, "dry_run" if args.dry_run else "completed")
        except subprocess.CalledProcessError as e:
            failures.append(exp.experiment_id)
            write_meta(exp, exp_dir, "failed", f"return code={e.returncode}")
            if not args.continue_on_error:
                raise

    if failures:
        print("[WARN] failed experiments:", failures)
    print("[DONE] model-specific-LR experiment run finished.")


if __name__ == "__main__":
    main()
