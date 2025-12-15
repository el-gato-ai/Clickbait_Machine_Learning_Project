from pathlib import Path
from typing import Optional, Tuple
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import plotly.express as px
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split



# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------
def train_valid_test_split(
    df: pd.DataFrame,
    train_frac: float = 0.8,
    valid_frac: float = 0.1,
    test_frac: float = 0.1,
    random_state: int = 42,
    shuffle: bool = True,
    use_valid: bool = True,
) -> Tuple[pd.DataFrame, ...]:
    """
    Split a DataFrame into train/valid/test (or train/test if use_valid=False).
    Fractions must sum to 1.0 when use_valid is True.
    When use_valid is False, valid_frac is ignored and train_frac + test_frac must sum to 1.0.
    """
    if use_valid:
        if not np.isclose(train_frac + valid_frac + test_frac, 1.0):
            raise ValueError("train_frac + valid_frac + test_frac must sum to 1.0 when use_valid=True")
    else:
        if not np.isclose(train_frac + test_frac, 1.0):
            raise ValueError("train_frac + test_frac must sum to 1.0 when use_valid=False")
        valid_frac = 0.0

    if shuffle:
        df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    train_size = train_frac

    if use_valid:
        # First split off train
        train_df, temp_df = train_test_split(
            df,
            train_size=train_size,
            random_state=random_state,
            shuffle=False,
        )
        # Split temp into valid/test
        valid_size = valid_frac / (valid_frac + test_frac)
        valid_df, test_df = train_test_split(
            temp_df,
            train_size=valid_size,
            random_state=random_state,
            shuffle=False,
        )
        return (
            train_df.reset_index(drop=True),
            valid_df.reset_index(drop=True),
            test_df.reset_index(drop=True),
        )
    else:
        # Only train/test split
        train_df, test_df = train_test_split(
            df,
            train_size=train_size,
            random_state=random_state,
            shuffle=False,
        )
        return (
            train_df.reset_index(drop=True),
            test_df.reset_index(drop=True),
        )



# ---------------------------------------------------------------------------
# Dimensionality reduction helpers
# ---------------------------------------------------------------------------
def fit_reduce_embeddings(
    df: pd.DataFrame,
    algorithm: str,
    n_components: int = 500,
    random_state: int = 42,
    **kwargs,
) -> Tuple[pd.DataFrame, object]:
    """
    Fit a reducer (PCA/UMAP/t-SNE) and return (reduced_df, fitted_model).
    """
    algo = algorithm.lower()
    if algo == "pca":
        reducer = PCA(n_components=n_components, random_state=random_state, **kwargs)
    elif algo == "tsne":
        reducer = TSNE(n_components=n_components, random_state=random_state, **kwargs)
    elif algo == "umap":
        if umap is None:
            raise ImportError("umap-learn is not installed. Run `pip install umap-learn`.")
        reducer = umap.UMAP(n_components=n_components, random_state=random_state, **kwargs)
    else:
        raise ValueError("algorithm must be one of: pca, umap, tsne")

    reduced = reducer.fit_transform(df)
    cols = [f"{algo}_{i + 1}" for i in range(reduced.shape[1])]
    return pd.DataFrame(reduced, columns=cols), reducer


def save_reduced_embeddings(
    reduced_df: pd.DataFrame,
    algorithm: str,
    source_path: Path,
    output_dir: Optional[Path] = None,
) -> Path:
    """
    Save reduced embeddings next to the source file, preserving format.
    Output filename: <source_stem>_<algo><suffix>
    """
    algo = algorithm.lower()
    output_dir = output_dir or source_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    suffix = source_path.suffix
    out_name = f"{source_path.stem}_{algo}{suffix}"
    out_path = output_dir / out_name

    if suffix == ".parquet":
        reduced_df.to_parquet(out_path, index=False)
    elif suffix == ".csv":
        reduced_df.to_csv(out_path, index=False)
    elif suffix in {".xlsx", ".xls"}:
        reduced_df.to_excel(out_path, index=False)
    else:
        raise ValueError(f"Unsupported output format: {suffix}")

    print(f"[INFO] Saved {algo} embeddings to {out_path}")
    return out_path


def save_reducer(
    reducer: object,
    source_path: Path,
    algorithm: str,
    output_dir: Optional[Path] = None,
) -> Path:
    """
    Save the fitted reducer as a pickle for reuse.
    Output filename: <source_stem>_<algo>.pkl
    """
    algo = algorithm.lower()
    output_dir = output_dir or source_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    out_path = output_dir / f"{source_path.stem}_{algo}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(reducer, f)

    print(f"[INFO] Saved {algo} reducer to {out_path}")
    return out_path



# ---------------------------------------------------------------------------
# PCA variance exploration
# ---------------------------------------------------------------------------
def plot_pca_explained_variance(
    df: pd.DataFrame,
    max_components: int = 800,
    n_samples: int = 15,
    start_components: int = 25,
    random_state: int = 42,
) -> None:
    """
    Plot cumulative explained variance for PCA across a range of component counts.
    """
    max_components = min(df.shape[1], max_components)
    component_counts = np.linspace(start_components, max_components, num=n_samples, dtype=int)
    component_counts = sorted(set(c for c in component_counts if 1 <= c <= max_components))

    cumulative_variances = []
    for n in component_counts:
        pca = PCA(n_components=n, random_state=random_state)
        pca.fit(df)
        cumulative_variances.append(pca.explained_variance_ratio_.sum())

    plt.figure(figsize=(8, 4))
    plt.plot(component_counts, cumulative_variances, marker="o")
    plt.xlabel("Number of PCA components")
    plt.ylabel("Cumulative explained variance")
    plt.title("PCA cumulative explained variance")
    plt.grid(True)
    plt.show()



# ---------------------------------------------------------------------------
# 2D reduction and plotting
# ---------------------------------------------------------------------------
def reduce_and_plot_2d(
    df: pd.DataFrame,
    algorithm: str,
    random_state: int = 42,
    hue: Optional[pd.Series] = None,
    **kwargs,
) -> Tuple[pd.DataFrame, object]:
    """
    Fit a reducer (PCA/UMAP/t-SNE) to 2D, transform, and plot.
    Optionally color points by `hue` (Series aligned with df).
    Returns (reduced_df, fitted_model).
    """
    reduced_df, reducer = fit_reduce_embeddings(
        df=df,
        algorithm=algorithm,
        n_components=2,
        random_state=random_state,
        **kwargs,
    )

    plt.figure(figsize=(6, 5))
    if hue is not None:
        sns.scatterplot(
            x=reduced_df.iloc[:, 0],
            y=reduced_df.iloc[:, 1],
            hue=hue,
            s=10,
            alpha=0.7,
            palette="viridis",
        )
    else:
        plt.scatter(
            reduced_df.iloc[:, 0],
            reduced_df.iloc[:, 1],
            s=10,
            alpha=0.7,
        )
    plt.title(f"{algorithm.upper()} – 2D projection")
    plt.xlabel(reduced_df.columns[0])
    plt.ylabel(reduced_df.columns[1])
    plt.tight_layout()
    plt.show()

    return reduced_df, reducer


def reduce_and_plot_3d(
    df: pd.DataFrame,
    algorithm: str,
    random_state: int = 42,
    hue: Optional[pd.Series] = None,
    sample_frac: Optional[float] = None,
    sample_n: Optional[int] = None,
    **kwargs,
) -> Tuple[pd.DataFrame, object]:
    """
    Fit a reducer (PCA/UMAP/t-SNE) to 3D, transform, and render an interactive Plotly scatter.
    Optionally color points by `hue` (Series aligned with df).
    You can subsample the data for faster plotting via `sample_n` or `sample_frac`.
    Returns (reduced_df, fitted_model).
    """
    if px is None:
        raise ImportError("plotly is not installed. Run `pip install plotly` to enable 3D plots.")

    work_df = df
    work_hue = hue

    if sample_n is not None:
        work_df = df.sample(n=sample_n, random_state=random_state)
        if hue is not None:
            work_hue = hue.loc[work_df.index]
    elif sample_frac is not None:
        work_df = df.sample(frac=sample_frac, random_state=random_state)
        if hue is not None:
            work_hue = hue.loc[work_df.index]

    reduced_df, reducer = fit_reduce_embeddings(
        df=work_df,
        algorithm=algorithm,
        n_components=3,
        random_state=random_state,
        **kwargs,
    )

    plot_df = reduced_df.copy()
    if work_hue is not None:
        plot_df["hue"] = work_hue

    fig = px.scatter_3d(
        plot_df,
        x=reduced_df.columns[0],
        y=reduced_df.columns[1],
        z=reduced_df.columns[2],
        color="hue" if work_hue is not None else None,
        opacity=0.7,
        title=f"{algorithm.upper()} – 3D projection",
    )
    fig.show()

    return reduced_df, reducer



# ---------------------------------------------------------------------------
# Embedding attachment
# ---------------------------------------------------------------------------
def attach_embeddings(
    data_df: pd.DataFrame,
    embeddings_df: pd.DataFrame,
    embed_col: str = "embedding",
) -> pd.DataFrame:
    """
    Attach embeddings to a data dataframe as a single column containing vectors.

    Assumes `data_df` and `embeddings_df` are aligned row-wise.
    """
    if len(data_df) != len(embeddings_df):
        raise ValueError(
            f"Length mismatch: data_df ({len(data_df)}) vs embeddings_df ({len(embeddings_df)})"
        )

    # Convert each row of embeddings to a list
    embed_list = embeddings_df.to_numpy().tolist()

    out = data_df.copy()
    out[embed_col] = embed_list
    return out
