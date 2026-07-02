# synthetic-hfus-skinlayer-segmentation

Synthetic-to-real HFUS epidermis/SLEB segmentation ablation code.

This repository contains the final model-specific learning-rate experiment protocol used for the segmentation ablation. The code is designed to run with repository-relative paths only.

## 1. Clone the repository

```bash
git clone https://github.com/hfus-anonymous/synthetic-hfus-skin-layer-segmentation.git
cd synthetic-hfus-skin-layer-segmentation
```

## 2. Create the Python environment

```bash
conda create -n synthetic-hfus python=3.10 -y
conda activate synthetic-hfus
pip install -r requirements.txt
```

CUDA is recommended for the full experiment.

## 3. Download the data

Download `data.zip` from Google Drive:

```text
https://drive.google.com/uc?export=download&id=1fIulaT48gtsr8skzWfKmdfj_V6nARAaj
```

The zip should contain the repository `data/` folder. After extraction, the repository should look like this:

```text
synthetic-hfus-skinlayer-segmentation/
├── code/
├── data/
│   ├── Mendeley/
│   │   ├── images/
│   │   ├── masks/
│   ├── synthetic_sample/
│   │   ├── images/
│   │   ├── masks/
│   │   ├── metadata/
│   │   ├── metadata.json
│   │   ├── manifest.csv
│   │   └── sample_build_summary.json
│   └── README_data.md
├── outputs/
├── requirements.txt
└── README.md
```

If `data.zip` contains a top-level `data/` directory, extract it into the repository root:

```bash
unzip -o /path/to/data.zip -d .
```

If `data.zip` contains the contents of the data folder directly, extract it into `data/`:

```bash
mkdir -p data
unzip -o /path/to/data.zip -d data
```

## 4. Check the downloaded data

Run this from the repository root:

```bash
python code/make_sample.py
```

This script only validates the downloaded repository-local folder:

```text
data/synthetic_sample/
```

It does not access any external or developer-specific path. Expected result:

```text
[OK] downloaded synthetic_sample is available.
[DONE] data/synthetic_sample is ready for run_experiments.py.
```

## 5. Run the final segmentation ablation

Run all 8 experiments:

```bash
python code/run_experiments.py --device cuda --amp --prepare_no_roi_visuals
```

`run_experiments.py` automatically runs ROI preparation first if the prepared manifests are missing.

The prepared ROI manifests are written to:

```text
outputs/prepared/v66_synthetic_roi_64x128/manifest.csv
outputs/prepared/mendeley_real_roi_64x128/manifest.csv
```

The experiment outputs are written to:

```text
outputs/model_lr_experiments/
├── SegUNet_R300/
├── SegUNet_S100R300_ENC/
├── UNet_R300/
├── UNet_S100R300_ENC/
├── DeepLabV3Plus_R300/
├── DeepLabV3Plus_S100R300_ENC/
├── SegFormer_R300/
└── SegFormer_S100R300_ENC/
```

Each experiment directory contains the run configuration, training logs, validation/test metrics, checkpoints, and optional prediction visualizations.

## Final experiment protocol

The maintained experiment is:

```text
4 models × 2 regimes = 8 experiments
```

Models:

```text
SegUNet
U-Net
DeepLabV3+
SegFormer
```

Regimes:

```text
R300
  Real-only training for 300 epochs.

S100R300-ENC
  Synthetic pretraining for 100 epochs
  → keep encoder/backbone
  → reset decoder/head
  → real fine-tuning for 300 epochs.
```

Model-specific learning rates:

| Model | Real-only LR | Synthetic pretrain LR | Real fine-tune LR | Optimizer |
|---|---:|---:|---:|---|
| SegUNet | `1e-4` | `1e-3` | `1e-4` | `auto` → SGDM |
| U-Net | `1e-5` | `1e-4` | `1e-5` | `auto` → AdamW |
| DeepLabV3+ | `1e-5` | `1e-4` | `1e-5` | `auto` → AdamW |
| SegFormer | `1e-5` | `1e-4` | `1e-5` | `auto` → AdamW |

## Useful run options

Run selected models only:

```bash
python code/run_experiments.py --models SegUNet,UNet --device cuda --amp --prepare_no_roi_visuals
```

Run one experiment only:

```bash
python code/run_experiments.py \
  --only SegUNet_S100R300_ENC \
  --device cuda \
  --amp \
  --prepare_no_roi_visuals
```

Resume after interruption:

```bash
python code/run_experiments.py --device cuda --amp --prepare_no_roi_visuals --skip_existing
```

Generate commands without training:

```bash
python code/run_experiments.py --dry_run --models SegUNet
```

Clean previous experiment outputs and rerun:

```bash
python code/run_experiments.py --device cuda --amp --prepare_no_roi_visuals --clean_existing
```

Save ROI/prediction visualizations:

```bash
python code/run_experiments.py --device cuda --amp --save_visuals
```

## Data Availability

The original full dataset is not included in this anonymized repository due to data usage and privacy restrictions. Instead, this repository provides the source code and a compact sample dataset of 100 images to support code inspection and pipeline verification.

The sample dataset is provided only for demonstrating the repository-local data structure, preprocessing flow, and execution pipeline. Because it is a reduced sample rather than the complete original dataset, the reproduced training results may differ from the full-dataset results reported in the paper.

## Anonymity Note

This repository has been anonymized for double-blind review. Author names, affiliations, institutional identifiers, and personal account information are intentionally omitted during the review period.

## Notes

- All default paths are repository-relative.
- The public workflow assumes `data.zip` already contains `data/synthetic_sample/` and `data/Mendeley/`.
- No developer-specific Windows or WSL absolute path is required.
- `same_lr` ablation code is intentionally excluded.
- Smoke-test, one-epoch-test, table-generation, and plot-generation scripts are intentionally excluded.
- Metrics are saved inside each experiment folder by the training protocol.
