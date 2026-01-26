# Notebook Suite

This folder holds the end-to-end clickbait detection workflow, from raw data checks to final evaluation.

## Main pipeline

- `01_data_overview.ipynb`: sanity checks on raw sources and label balance.
- `02_clean_merge.ipynb`: cleaning, normalization, and merging across datasets.
- `03_feature_engineering.ipynb`: text preprocessing and classic feature builds.
- `04_modeling_embeddings.ipynb`: embedding-based models (Gemma-derived features).
- `05_evaluation_report.ipynb`: consolidated metrics and error analysis.

## Side experiments

- `Gradient_Booster.ipynb`: tree-based baseline experiments.
- `SGD.ipynb`: linear classifier experiments.

## How to use

- Run the main pipeline in order for a reproducible narrative.
- Point notebooks to data under `data/raw`; outputs land in `data/merged` and `data/clean`.
- Avoid committing large artifacts or verbose cell outputs; keep checkpoints lightweight.
