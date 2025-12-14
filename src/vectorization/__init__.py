"""Vectorization module for creating and saving text embeddings."""
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

# Embedding generic classes
from src.vectorization.model import HFEmbeddings

# Utility functions
from src.vectorization.utils import (
    load_dataframe,
    _iter_raw_files,
    _build_output_path,
    _select_column,
    SUPPORTED_OUTPUT_FORMATS,
    SUPPORTED_INPUT_EXTS
)

# Core vectorization functions
from src.vectorization.vectorize import (
    create_embeddings,
    save_embeddings,
    embed_titles,
    embed_all,
)

