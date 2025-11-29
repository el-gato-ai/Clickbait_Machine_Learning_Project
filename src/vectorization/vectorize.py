from pathlib import Path
from typing import Iterable, Union
import pandas as pd

from model import HFEmbeddings
from utils import (
    load_dataframe,
    _iter_raw_files,
    _build_output_path,
    SUPPORTED_OUTPUT_FORMATS
)


def create_embeddings(
    data_path: Union[str, Path],
    column: Union[str, int, Iterable[str]] = ("title", "headline", "targetTitle"),
    batch_size: int = 100,
    embedder: HFEmbeddings | None = None,
    model_name: str | None = None,
) -> pd.DataFrame:
    """
    Generate Gemma 3 embeddings for a dataset, returning the embeddings DataFrame.
    """
    titles_df = load_dataframe(data_path, column=column)
    embedder = embedder or HFEmbeddings(model_name=model_name or "google/gemma-3-4b-it")
    return embedder.transform(titles_df, batch_size=batch_size)


def save_embeddings(
    embeddings: pd.DataFrame,
    output_path: str | Path,
    output_format: str = "parquet",
) -> Path:
    """
    Save embeddings to the desired path in one of: parquet, csv, xlsx.
    """
    fmt = output_format.lower()
    if fmt not in {"parquet", "csv", "xlsx"}:
        raise ValueError("output_format must be one of: parquet, csv, xlsx")

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "parquet":
        embeddings.to_parquet(out_path, index=False)
    elif fmt == "csv":
        embeddings.to_csv(out_path, index=False)
    else:  # xlsx
        embeddings.to_excel(out_path, index=False)

    return out_path


def embed_titles(
    data_path: Union[str, Path],
    column: Union[str, int, Iterable[str]] = ("title", "headline", "targetTitle"),
    batch_size: int = 100,
    output_path: str | Path | None = None,
    output_format: str = "parquet",
    embedder: HFEmbeddings | None = None,
    model_name: str | None = None,
) -> Path:
    """
    Convenience wrapper: generate embeddings and persist them.

    If output_path is None, saves next to this file as `title_embeddings.<format>`.
    """
    embeddings = create_embeddings(
        data_path=data_path,
        column=column,
        batch_size=batch_size,
        embedder=embedder,
        model_name=model_name,
    )

    if output_path is None:
        output_path = Path(__file__).resolve().parent / f"title_embeddings.{output_format.lower()}"

    return save_embeddings(embeddings, output_path, output_format=output_format)


def embed_all_raw(
    raw_root: Union[str, Path] = Path(__file__).resolve().parents[2] / "data" / "raw",
    embedded_root: Union[str, Path] = Path(__file__).resolve().parents[2] / "data" / "embedded",
    column: Union[str, int, Iterable[str]] = ("title", "headline", "targetTitle"),
    batch_size: int = 100,
    output_format: str = "parquet",
    embedder: HFEmbeddings | None = None,
    skip_existing: bool = True,
    model_name: str | None = None,
) -> list[Path]:
    """
    Walk `raw_root` for CSV/XLSX/JSONL files, embed the chosen column, and save to `embedded_root`.

    Output files are flattened (path parts joined with '__') and suffixed with `_embed.<format>`.
    When skip_existing=True, files with an existing embedding output are skipped.
    """
    fmt = output_format.lower()
    if fmt not in SUPPORTED_OUTPUT_FORMATS:
        raise ValueError(f"output_format must be one of: {', '.join(sorted(SUPPORTED_OUTPUT_FORMATS))}")

    raw_root = Path(raw_root)
    embedded_root = Path(embedded_root)
    embedded_root.mkdir(parents=True, exist_ok=True)
    embedder = embedder or HFEmbeddings(model_name=model_name or "google/gemma-3-4b-it")

    outputs: list[Path] = []
    for raw_file in _iter_raw_files(raw_root):
        out_path = _build_output_path(raw_root, embedded_root, raw_file, fmt)
        if skip_existing and out_path.exists():
            print(f"[SKIP] Embeddings already exist for {raw_file} -> {out_path}")
            outputs.append(out_path)
            continue
        try:
            saved = embed_titles(
                data_path=raw_file,
                column=column,
                batch_size=batch_size,
                output_path=out_path,
                output_format=fmt,
                embedder=embedder,
            )
            outputs.append(saved)
            print(f"[INFO] Saved embeddings for {raw_file} -> {saved}")
        except Exception as exc:
            print(f"[ERROR] Failed to embed {raw_file}: {exc}")
    return outputs

