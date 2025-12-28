# Clickbait Detection Project

Experiments, data pipelines, and notebooks for detecting clickbait across multiple news sources and languages.

## What’s here

- Data sources documented under `data/raw/` (Kaggle, Webis, GitHub, and custom pulls).
- Collection scripts in `data_collection/`:
  - LangChain + Tavily + GPT-4.1 agent (run via `python -m data_collection`) that logs country/language-specific news to `data/raw/custom/custom_news.xlsx`.
- Modeling utilities in `src/`:
  - `vectorization.py`: Gemma 3 (google/gemma-3-4b-it) mean-pooled hidden-state embeddings.
  - `__init__.py`: helpers to load a dataset, embed titles, and save embeddings next to the code.
- Notebook pipeline in `notebooks/` from data overview through evaluation.

## Quick start

- Python 3.10+. Create a venv and install project requirements (plus `data_collection/requirements.txt` if using the agent).
- Copy `.env.example` to `.env` and set:
  - `OPENAI_API_KEY`, `TAVILY_API_KEY` for `data_collection` tools.
  - `HUGGINGFACE_TOKEN` for Gemma downloads.
- Run the collection agent to refresh `data/raw/custom/custom_news.xlsx` (examples below).
- Use the notebooks in order (`01_` to `06_`) for the full modeling narrative.

### Collection agent usage

From the repo root:

```bash
# Balanced news, all topics, rolling ~30 days
python -m data_collection --country "France" --language "French" --mode news

# Clickbait focus, single topic
python -m data_collection --country "Germany" --language "German" --mode clickbait --topics Sports

# Multiple topics
python -m data_collection --country "United States" --language "English" --mode news --topics Politics Economy

# Another language, all topics
python -m data_collection --country "Spain" --language "Spanish" --mode news

# Custom date range (YYYY-MM-DD)
python -m data_collection --country "Italy" --language "Italian" --mode news --start-date 2025-01-01 --end-date 2025-01-07
```

Flags:

- `--country` (required): Target country/region to bias the search.
- `--language` (required): Language for search/output.
- `--mode`: `news` (default) or `clickbait`.
- `--topics`: optional space-separated subset of topics; defaults to all.
- `--start-date` / `--end-date`: optional date window; defaults to last ~30 days ending today (UTC).

## Embeddings shortcut

- Call `embed_titles` in `src/__init__.py` to read a CSV/Excel (e.g., `data/raw/kaggle/clickbait_data.csv`), keep the title column, generate embeddings, and save a Parquet file alongside `vectorization.py`.
