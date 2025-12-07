# Notebook Suite

This folder holds the end-to-end clickbait detection workflow, from raw data checks to final evaluation.

## Contents

- `01_eda.ipynb`: sanity checks on raw sources and label balance.
- `02_clean_merge.ipynb`: cleaning, normalization, and merging across datasets.
- `03_feature_engineering.ipynb`: text preprocessing and classic feature builds.
- `04_modeling_baselines.ipynb`: traditional ML baselines with validation.
- `05_modeling_embeddings.ipynb`: embedding-based models (e.g., Gemma-derived features).
- `06_evaluation_report.ipynb`: consolidated metrics and error analysis.

## How to use

- Run in order for a reproducible narrative.
- Point notebooks to data under `data/raw` (and any processed outputs under `data/processed` if you create them).
- Avoid committing large artifacts or verbose cell outputs; keep checkpoints lightweight.
