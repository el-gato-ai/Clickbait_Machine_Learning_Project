from pathlib import Path
from typing import Iterable, Union
import pandas as pd


SUPPORTED_INPUT_EXTS = {".csv", ".xlsx", ".xls", ".jsonl"}
SUPPORTED_OUTPUT_FORMATS = {"parquet", "csv", "xlsx"}


def _iter_raw_files(raw_root: Union[str, Path]) -> Iterable[Path]:
    """
    Recursively yield all supported files under `raw_root`.
    """
    root = Path(raw_root)
    if not root.exists():
        raise FileNotFoundError(f"{raw_root} does not exist")
    return (
        p
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_INPUT_EXTS
    )


def _build_output_path(
    raw_root: Path,
    embedded_root: Path,
    raw_file: Path,
    output_format: str,
    model_name: str,
) -> Path:
    """
    Construct output path by flattening relative path from raw_root to raw_file,
    joining parts with '__', and suffixing with `_embed.<format>`, nested under
    a model-name directory.
    """
    rel = raw_file.relative_to(raw_root).with_suffix("")
    flat_name = "__".join(rel.parts) + f"_embed.{output_format}"
    model_dir = model_name.replace("/", "__")
    return embedded_root / model_dir / flat_name


def _select_column(df: pd.DataFrame, column: Union[str, int, Iterable[str]]) -> str:
    """
    Resolve which column to use. Accepts:
    - int index
    - single column name
    - iterable of candidate names (first found is used)
    """
    if isinstance(column, int):
        if column < 0 or column >= len(df.columns):
            raise ValueError(f"Column index {column} out of range")
        return df.columns[column]

    if isinstance(column, str):
        if column not in df.columns:
            raise ValueError(f"Column {column!r} not found in dataframe")
        return column

    # Iterable of candidates
    for candidate in column:
        if candidate in df.columns:
            return candidate
    # fallback to first column
    return df.columns[0]


def load_dataframe(
    data_path: Union[str, Path],
    column: Union[str, int, Iterable[str]] = ("title", "headline", "targetTitle"),
) -> pd.DataFrame:
    """
    Load a dataframe from CSV/Excel/JSONL and return a single-column frame with the requested field.

    `column` may be a column name (str), positional index (int), or iterable of candidate names.
    If no candidate is found, the first column is used as a fallback.
    """
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"{data_path} does not exist")

    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    elif suffix == ".jsonl":
        df = pd.read_json(path, lines=True)
    elif suffix == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file type for {data_path}")

    if df.empty:
        raise ValueError(f"No data found in {data_path}")

    col_name = _select_column(df, column)
    if isinstance(column, Iterable) and not isinstance(column, (str, bytes)) and col_name == df.columns[0] and col_name not in column:
        print(f"[WARN] None of {list(column)} found in {data_path}; using first column {col_name!r}")
    elif isinstance(column, str) and col_name == df.columns[0] and col_name != column:
        print(f"[WARN] Column {column!r} not found in {data_path}; using first column {col_name!r}")

    titles = df[[col_name]].dropna()
    return titles
