# Clickbait Detection Project

Experiments, data pipelines, and notebooks for detecting clickbait across multiple news sources and languages.

## Repository map

- `data/`: raw sources plus merged/cleaned/embedded outputs (see `data/README.md`).
- `data_collection/`: LangChain + Tavily agent that gathers headlines into `data/raw/custom/custom_news.xlsx` (see `data_collection/README.md`).
- `notebooks/`: end-to-end analysis pipeline and modeling experiments (see `notebooks/README.md`).
- `src/`: reusable Python modules (vectorization, EDA, feature engineering, MLflow helpers).
- `mlruns/`: MLflow tracking artifacts created by local runs.

## Key directories

### data/

Raw sources live in `data/raw` with provenance notes, while `data/merged` and `data/clean` hold intermediate outputs from the notebooks. Embeddings are written to `data/embedded` by the vectorization pipeline.

### data_collection/

The search agent builds topic queries with a rolling date window and writes URL-deduplicated results into `data/raw/custom/custom_news.xlsx`. Use this when you want fresh, country/language-specific headlines.

### notebooks/

The main pipeline starts at `01_data_overview.ipynb` and flows through cleaning, feature engineering, modeling, and evaluation. Side experiments (e.g., SGD, Gradient Boosting) live alongside the core notebooks.

### src/

Reusable code for embeddings (`src/vectorization`), EDA helpers (`src/eda`), feature engineering (`src/feature_eng`), and MLflow logging (`src/Models`).

### mlruns/

Local MLflow run artifacts produced by training notebooks and helpers. Safe to delete and regenerate.

## Setup

- Python 3.10+. Create a virtualenv and install `requirements.txt`.
- If you plan to run the collection agent, also install `data_collection/requirements.txt`.
- Copy `.env.example` to `.env` and set:
  - `OPENAI_API_KEY` and `TAVILY_API_KEY` for data collection.
  - `HUGGINGFACE_TOKEN` for Gemma downloads in vectorization.

## Common workflows

### Collect fresh headlines (optional)

```bash
python -m data_collection --country "France" --language "French" --mode news
```

The agent defaults to `--mode clickbait` and a ~30-day UTC date window when omitted.

### Generate embeddings

Run the batch embedding pipeline over everything in `data/raw`:

```bash
python -m src.vectorization
```

This writes flattened embedding files under `data/embedded/<model_name>/`.

### Notebook pipeline

Start with `notebooks/01_data_overview.ipynb` and follow the sequence in `notebooks/README.md`.
