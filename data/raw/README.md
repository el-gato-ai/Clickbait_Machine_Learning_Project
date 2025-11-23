# Data Sources

Everything inside this `data/raw` folder supports experimentation on clickbait detection. Each section below describes what lives in the corresponding subdirectory, how it was collected, and why it matters.

## Kaggle Collections (`data/raw/kaggle`)

This directory mirrors three separate Kaggle contributions:

- `clickbait_data.csv` (AmAnand Rai) mixes short headlines with binary clickbait labels, perfect for quick model spikes or benchmarking lightweight classifiers.
- `train2.csv` (News Clickbait Dataset by Vikas Singh) contains longer article teasers, giving models more context to reason about style, tone, and clickbait cues in extended text.
- `clickbait-news-detection` is the full export from the Kaggle competition of the same name, including the organizers' official splits so you can reproduce leaderboard-ready experiments.

Using all three together lets you probe how models transfer between short and long-form clickbait writing.

## GitHub Snapshot (`data/raw/github`)

`clickbait.csv` is a compact CSV that Amit Chaudhary shared through a public Gist. The file offers a quick sanity-check dataset: small enough for notebooks and demos, but still labeled so you can validate preprocessing pipelines or run smoke tests before scaling up.

## Zenodo Corpus (`data/raw/zenodo`)

`webis-clickbait-22` is the Webis Clickbait 22 release hosted on Zenodo. It bundles teaser texts, article metadata, and relevance judgments from the Webis shared task, making it the most comprehensive drop in this folder. The dataset is ideal for advanced experiments such as multi-modal modeling or domain adaptation.

## Custom Harvest (`data/raw/custom`)

This directory holds the Greek-language samples we assembled manually:

- `greek_news.xlsx` is maintained by `data_collection/search_agent.py`, a LangChain + Tavily workflow that queries major Greek outlets per topic, summarizes each article in English, and appends deduplicated rows (URL, title, summary, topic) to the workbook.
- `lifo_mostpopular_7days_today*.csv` files come from `data_collection/scrape_lifo.py`, a BeautifulSoup scraper that walks the LIFO "Most Popular" listings (today, 7 days, and blended) and tags every captured headline as clickbait-positive. These act as a curated set of catchy headlines straight from a local publisher.

Adjust the scripts inside `data_collection/` if you expand these pipelines, and drop the resulting exports back into `data/raw/custom` so the provenance stays easy to audit.

## Source Reference

| Dataset | Local Path | URL |
| --- | --- | --- |
| Clickbait Dataset (AmAnand Rai) | `raw/kaggle/clickbait_data.csv` | https://www.kaggle.com/datasets/amananandrai/clickbait-dataset |
| News Clickbait Dataset (Vikas Singh) | `raw/kaggle/train2.csv` | https://www.kaggle.com/datasets/vikassingh1996/news-clickbait-dataset?select=train2.csv |
| Clickbait News Detection Competition | `raw/kaggle/clickbait-news-detection` | https://www.kaggle.com/competitions/clickbait-news-detection/data |
| Clickbait CSV (Amit Chaudhary Gist) | `raw/github/clickbait.csv` | https://gist.github.com/amitness/0a2ddbcb61c34eab04bad5a17fd8c86b |
| Webis Clickbait 22 | `raw/zenodo/webis-clickbait-22` | https://zenodo.org/records/6362726#.YsbdSTVBzrk |
| Custom Greek News Agent Pull | `raw/custom/greek_news.xlsx` | Collected via `data_collection/search_agent.py` |
| LIFO Most Popular Scrape | `raw/custom/lifo_mostpopular_7days_today*.csv` | Collected via `data_collection/scrape_lifo.py` |
