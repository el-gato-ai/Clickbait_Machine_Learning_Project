# Data Collection Toolkit

This folder contains the LangChain + Tavily agent used to gather news articles (clickbait or balanced) across countries and languages. Results are URL-deduplicated and appended to `data/raw/custom/custom_news.xlsx`.

## Prerequisites

- Python 3.10+
- Install dependencies from `data_collection/requirements.txt` inside a virtualenv:

  ```bash
  cd data_collection
  python -m venv .venv
  source .venv/bin/activate  # or .\\.venv\\Scripts\\activate on Windows
  pip install -r requirements.txt
  ```

- Environment variables in the project root `.env`:
  - `OPENAI_API_KEY` (for ChatOpenAI)
  - `TAVILY_API_KEY` (for TavilySearch)

## Running the search agent

From the repo root:

```bash
# Balanced news, all topics (rolling ~30 days)
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

- `--country` (required): Target country/region to bias results and domains (where available).
- `--language` (required): Language to search/respond in.
- `--mode`: `clickbait` (default) for sensational headlines, or `news` for balanced coverage.
- `--topics`: optional space-separated subset (e.g., `Politics Economy`). Defaults to all predefined topics in `data_collection/prompt_eng.py`.
- `--start-date` / `--end-date`: optional date window; defaults to last ~30 days ending today (UTC).

## Output schema

Each appended row includes:

- `timestamp_utc`, `query`, `url`, `title`, `description`, `topic`, `country`, `language`

## Notes

- Adjust the rolling date window in `data_collection/__init__.py` if you need a different range.
- Keep an eye on API quotas for both OpenAI and Tavily on multi-topic runs.
- If you add new countries frequently, extend the domain hints in `data_collection/llms.py` for better targeting.
