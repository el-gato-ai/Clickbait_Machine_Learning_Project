# Data Collection Toolkit

This directory packages the scripts that keep the `data/raw/custom` folder fresh with newly scraped Greek clickbait material. Two separate pipelines ship with the repo:

1. `search_agent.py` - a LangChain/Tavily/ChatGPT agent that hunts for recent Greek articles, summarizes them, and logs them into `data/raw/custom/greek_news.xlsx`.
2. `scrape_lifo.py` - a focused BeautifulSoup scraper that captures the headlines from LIFO's "Most Popular" feeds and saves the result as CSVs under `data/raw/custom/`.

Use them together when you need a mix of curated positives (LIFO) and broader, labeled Greek articles.

## Prerequisites

- Python 3.10+.
- The dependencies listed in `requirements.txt`. Install them once you are inside this folder:
  
  ```bash
  cd data_collection
  python -m venv .venv
  .\\.venv\\Scripts\\activate  # or source .venv/bin/activate on macOS/Linux
  pip install -r requirements.txt
  ```

- API keys stored in the project root `.env` file:
  - `OPENAI_API_KEY` for `ChatOpenAI`.
  - `TAVILY_API_KEY` for the Tavily news search.

Rename the provided `.env.example` at the project root to `.env`, replace the placeholder values with your API keys, and the scripts will pick it up automatically via `find_dotenv()`.

## 1. LangChain Search Agent (`search_agent.py`)

What it does:

- Builds topic-specific queries (Politics, Economy, Sports, Society, Technology, Culture, Fashion).
- Uses TavilySearch to gather recent Greek articles per topic.
- Invokes GPT-5 via LangChain to summarize each article (in English) and assign the correct topic label.
- Appends the deduplicated rows (URL, title, description, topic, timestamps) to `../data/raw/custom/greek_news.xlsx`.

How to run:

```bash
cd data_collection
python search_agent.py
```

The default main function requests articles between the dates set near the bottom of the script. Adjust `start_date`, `end_date`, or `TOPICS` before running if you want a different window or subset. Each execution prints the streaming agent output and reports how many rows were appended to the Excel file.

## 2. LIFO Most-Popular Scraper (`scrape_lifo.py`)

What it does:

- Fetches `https://www.lifo.gr/mostpopular`, `/7days`, and `/today` with polite headers.
- Parses the sectioned lists, normalizes the titles, and removes duplicates.
- Marks every captured headline as `clickbait=1` and persists the merged results to `../data/raw/custom/lifo_mostpopular_7days_today.csv` (plus any suffixed backups you create).

How to run:

```bash
cd data_collection
python scrape_lifo.py
```

After a successful run you will find a CSV in `data/raw/custom/` containing `section`, `title`, and `clickbait` columns - ideal as a high-precision positive set.

## Tips for Fresh Runs

- Keep an eye on rate limits for both OpenAI and Tavily when looping over long date ranges.
- If you rotate API keys or environments, rerun `pip install -r requirements.txt` and double-check that `.env` is discoverable (the scripts call `dotenv.load_dotenv()` at import time).
- Commit the refreshed artifacts in `data/raw/custom/` only if you want them tracked; otherwise, add paths to `.gitignore` before experimenting.
