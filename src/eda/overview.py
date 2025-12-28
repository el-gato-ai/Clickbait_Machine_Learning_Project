from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"



# ---------------------------------------------------------------------------
# Loading helpers for individual raw datasets
# ---------------------------------------------------------------------------
def load_greek_news(raw_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    Load the custom Greek news Excel file produced by running `python -m data_collection`.

    Expected columns (as written by the agent script):
    - timestamp_utc, query, url, title, description, topic
    """
    base = raw_dir or RAW_DIR
    path = base / "custom" / "greek_news.xlsx"
    return pd.read_excel(path)


def load_lifo_mostpopular(raw_dir: Optional[Path] = None) -> Dict[str, pd.DataFrame]:
    """
    Load the LIFO 'Most Popular' clickbait CSVs.

    Returns a dictionary:
        {
            "lifo_mostpopular_7days_today": DataFrame,
            "lifo_mostpopular_7days_today_2": DataFrame,
        }
    """
    base = raw_dir or RAW_DIR
    custom_dir = base / "custom"

    data: Dict[str, pd.DataFrame] = {}
    lifo_files = [
        ("lifo_mostpopular_7days_today", "lifo_mostpopular_7days_today.csv"),
        ("lifo_mostpopular_7days_today_2", "lifo_mostpopular_7days_today_2.csv"),
    ]

    for name, filename in lifo_files:
        path = custom_dir / filename
        if path.exists():
            data[name] = pd.read_csv(path)

    return data


def load_github_clickbait(raw_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    Load the small GitHub/Gist clickbait CSV (`title`, `label`).
    """
    base = raw_dir or RAW_DIR
    path = base / "github" / "clickbait.csv"
    return pd.read_csv(path)


def load_kaggle_clickbait_data(raw_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    Load `clickbait_data.csv` from Kaggle (headline + binary clickbait flag).
    """
    base = raw_dir or RAW_DIR
    path = base / "kaggle" / "clickbait_data.csv"
    return pd.read_csv(path)


def load_kaggle_train2(raw_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    Load `train2.csv` from Kaggle's News Clickbait Dataset.
    """
    base = raw_dir or RAW_DIR
    path = base / "kaggle" / "train2.csv"
    return pd.read_csv(path)


def load_kaggle_clickbait_news_detection(
    raw_dir: Optional[Path] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Load the 'clickbait-news-detection' competition files.

    Returns a dict with keys:
        'train', 'valid', 'test', 'unlabeled', 'sample_submission'
    (only those that actually exist on disk).
    """
    base = raw_dir or RAW_DIR
    root = base / "kaggle" / "clickbait-news-detection"

    datasets: Dict[str, pd.DataFrame] = {}
    mapping = {
        "train": "train.csv",
        "valid": "valid.csv",
        "test": "test.csv",
        "unlabeled": "unlabeled.csv",
        "sample_submission": "sample_submission.csv",
    }

    for name, filename in mapping.items():
        path = root / filename
        if path.exists():
            datasets[name] = pd.read_csv(path)

    return datasets


def _flatten_webis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flatten key fields from the Webis Clickbait 22 JSONL format into a simpler table.

    - postText is a list; we take the first element as 'post_text'.
    """
    out = pd.DataFrame()
    out["uuid"] = df.get("uuid")
    out["post_platform"] = df.get("postPlatform")
    out["post_text"] = df.get("postText").apply(
        lambda items: items[0] if isinstance(items, list) and items else None
    )
    out["target_title"] = df.get("targetTitle")
    out["target_description"] = df.get("targetDescription")
    out["tags"] = df.get("tags")
    return out


def load_webis_clickbait22(
    raw_dir: Optional[Path] = None,
    flatten: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Load the Webis Clickbait 22 JSONL files from Zenodo.

    If `flatten` is True, returns simplified tables with:
        uuid, post_platform, post_text, target_title, target_description, tags
    Otherwise, returns the raw JSONL DataFrames.
    """
    base = raw_dir or RAW_DIR
    root = base / "zenodo" / "webis-clickbait-22"

    datasets: Dict[str, pd.DataFrame] = {}
    mapping = {
        "train": "train.jsonl",
        "validation": "validation.jsonl",
    }

    for name, filename in mapping.items():
        path = root / filename
        if not path.exists():
            continue
        df = pd.read_json(path, lines=True)
        datasets[name] = _flatten_webis(df) if flatten else df

    return datasets


def load_all_raw_datasets(raw_dir: Optional[Path] = None) -> Dict[str, pd.DataFrame]:
    """
    Convenience loader that returns all known raw datasets in a single dictionary.

    Keys roughly follow the pattern:
        custom_greek_news, lifo_mostpopular_*, github_clickbait,
        kaggle_clickbait_data, kaggle_train2,
        cbd_train/valid/test/unlabeled/sample_submission,
        webis_train/validation
    """
    base = raw_dir or RAW_DIR

    datasets: Dict[str, pd.DataFrame] = {}

    # Custom Greek
    try:
        datasets["custom_greek_news"] = load_greek_news(base)
    except FileNotFoundError:
        pass

    for name, df in load_lifo_mostpopular(base).items():
        datasets[name] = df

    # GitHub
    try:
        datasets["github_clickbait"] = load_github_clickbait(base)
    except FileNotFoundError:
        pass

    # Kaggle: standalone CSVs
    try:
        datasets["kaggle_clickbait_data"] = load_kaggle_clickbait_data(base)
    except FileNotFoundError:
        pass

    try:
        datasets["kaggle_train2"] = load_kaggle_train2(base)
    except FileNotFoundError:
        pass

    # Kaggle: clickbait-news-detection competition
    for name, df in load_kaggle_clickbait_news_detection(base).items():
        datasets[f"cbd_{name}"] = df

    # Webis Clickbait 22
    for name, df in load_webis_clickbait22(base, flatten=True).items():
        datasets[f"webis_{name}"] = df

    return datasets



# ---------------------------------------------------------------------------
# EDA utilities (statistics, overviews) for use inside notebooks
# ---------------------------------------------------------------------------
def compute_label_stats(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    """
    Compute basic counts and ratios for a label column.
    """
    if label_col not in df.columns:
        raise ValueError(f"Column '{label_col}' not found in DataFrame.")

    counts = df[label_col].value_counts(dropna=False)
    ratios = counts / len(df) if len(df) else 0.0

    return pd.DataFrame({"count": counts, "ratio": ratios})


def compute_text_lengths(df: pd.DataFrame, text_col: str) -> pd.Series:
    """
    Return a Series with character lengths for a given text column.
    """
    if text_col not in df.columns:
        raise ValueError(f"Column '{text_col}' not found in DataFrame.")
    return df[text_col].astype(str).str.len()


def summarize_text_lengths(df: pd.DataFrame, text_col: str) -> Dict[str, float]:
    """
    Summarize text length statistics for a given column.

    Returns a dictionary with min, max, mean, median.
    """
    lengths = compute_text_lengths(df, text_col)
    if lengths.empty:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "median": 0.0}

    return {
        "min": float(lengths.min()),
        "max": float(lengths.max()),
        "mean": float(lengths.mean()),
        "median": float(lengths.median()),
    }


def build_raw_overview_table(
    datasets: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Build a high-level overview table for all loaded raw datasets.

    The table includes:
        - name
        - n_rows
        - n_columns
        - example_text_cols (best guess)
        - example_label_col (best guess)
    """
    rows: List[Dict[str, object]] = []

    for name, df in datasets.items():
        columns = list(df.columns)

        # Heuristic guesses for text and label columns
        text_candidates: Iterable[str] = [
            col
            for col in columns
            if any(
                key in col.lower()
                for key in ("title", "headline", "text", "description", "post_text")
            )
        ]
        label_candidates: Iterable[str] = [
            col
            for col in columns
            if any(key == col.lower() for key in ("label", "clickbait", "topic"))
        ]

        rows.append(
            {
                "name": name,
                "n_rows": len(df),
                "n_columns": len(columns),
                "example_text_cols": list(text_candidates),
                "example_label_cols": list(label_candidates),
            }
        )

    return pd.DataFrame(rows)
