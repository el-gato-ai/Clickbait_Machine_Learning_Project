# Data Directory

This folder groups raw sources and derived artifacts used by the notebooks and scripts.

## Layout

- `raw/`: original datasets with provenance notes in `data/raw/README.md`.
- `merged/`: merged datasets (for example `data_merged.csv`).
- `clean/`: cleaned splits and feature artifacts (parquet outputs, PCA/UMAP projections).
- `embedded/`: created by `src/vectorization` when generating title embeddings.
