# Clickbait Detection Project

Experiments, data pipelines, and notebooks for detecting clickbait across English and Greek news sources.

## What’s here

- Data sources documented under `data/raw/` (Kaggle, Webis, GitHub, and custom Greek pulls).
- Collection scripts in `data_collection/`:
  - LangChain + Tavily + GPT-5 agent (run via `python -m data_collection`) that logs Greek news to `data/raw/custom/greek_news.xlsx`.
  - `scrape_lifo.py`: BeautifulSoup scraper for LIFO “Most Popular” headlines tagged as positives.
- Modeling utilities in `src/`:
  - `vectorization.py`: Gemma 3 (google/gemma-3-4b-it) mean-pooled hidden-state embeddings.
  - `__init__.py`: helpers to load a dataset, embed titles, and save embeddings next to the code.
- Notebook pipeline in `notebooks/` from data overview through evaluation.

## Quick start

- Python 3.10+. Create a venv and install project requirements (plus `data_collection/requirements.txt` if using the scrapers/agent).
- Copy `.env.example` to `.env` and set:
  - `OPENAI_API_KEY`, `TAVILY_API_KEY` for `data_collection` tools.
  - `HUGGINGFACE_TOKEN` for Gemma downloads.
- Run collection scripts from `data_collection/` as needed to refresh `data/raw/custom`.
- Use the notebooks in order (`01_` to `06_`) for the full modeling narrative.

## Embeddings shortcut

- Call `embed_titles` in `src/__init__.py` to read a CSV/Excel (e.g., `data/raw/kaggle/clickbait_data.csv`), keep the title column, generate embeddings, and save a Parquet file alongside `vectorization.py`.
