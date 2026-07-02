# Data folder

The repository does not track actual PNG/JSON/CSV data files.

For public use, download `data.zip` from the Google Drive link in the main `README.md`, then extract it so that this directory has the following structure:

```text
data/
├── Mendeley/
│   ├── images/
│   ├── masks/
├── synthetic_sample/
│   ├── images/
│   ├── masks/
│   ├── metadata/
│   ├── metadata.json
│   ├── manifest.csv
│   └── sample_build_summary.json
└── README_data.md
```

The training pipeline uses only repository-relative paths by default:

```text
data/synthetic_sample/
data/Mendeley/images/
data/Mendeley/masks/
```
