# Data Sources

Everything inside `data/raw` supports experimentation on clickbait detection. Each section below describes what lives in the corresponding subdirectory, how it was collected, and why it matters.

## Kaggle collections (`data/raw/kaggle`)

This directory mirrors three separate Kaggle contributions:

- `clickbait_data.csv` (AmAnand Rai): short headlines with binary clickbait labels.
- `train2.csv` (News Clickbait Dataset by Vikas Singh): longer teasers with labeled clickbait.
- `clickbait-news-detection/`: competition export with train/valid/test splits and unlabeled data.

Using all three together lets you probe how models transfer between short and long-form clickbait writing.

## GitHub snapshot (`data/raw/github`)

`clickbait.csv` is a compact CSV from a public Gist by Amit Chaudhary. It is small enough for notebook smoke tests while still providing labels.

## Zenodo corpus (`data/raw/zenodo`)

`webis-clickbait-22` is the Webis Clickbait 22 release hosted on Zenodo. This repo includes the train and validation JSONL files (see the README inside the folder for full context).

## Custom harvest (`data/raw/custom`)

- `custom_news.xlsx` is maintained by the `data_collection` LangChain + Tavily workflow (run via `python -m data_collection`). Each appended row includes `timestamp_utc`, `query`, `url`, `title`, `description`, `topic`, `country`, and `language`.
- `greek_news.xlsx` is a manually assembled Greek-language sample used for early exploration.

Adjust the scripts inside `data_collection/` if you expand these pipelines, and drop the resulting exports back into `data/raw/custom` so the provenance stays easy to audit.

## Source reference

| Dataset | Local Path | URL |
| --- | --- | --- |
| Clickbait Dataset (AmAnand Rai) | `raw/kaggle/clickbait_data.csv` | https://www.kaggle.com/datasets/amananandrai/clickbait-dataset |
| News Clickbait Dataset (Vikas Singh) | `raw/kaggle/train2.csv` | https://www.kaggle.com/datasets/vikassingh1996/news-clickbait-dataset?select=train2.csv |
| Clickbait News Detection Competition | `raw/kaggle/clickbait-news-detection` | https://www.kaggle.com/competitions/clickbait-news-detection/data |
| Clickbait CSV (Amit Chaudhary Gist) | `raw/github/clickbait.csv` | https://gist.github.com/amitness/0a2ddbcb61c34eab04bad5a17fd8c86b |
| Webis Clickbait 22 | `raw/zenodo/webis-clickbait-22` | https://zenodo.org/records/6362726 |
| Custom News Agent Pull | `raw/custom/custom_news.xlsx` | Collected via `python -m data_collection` |
