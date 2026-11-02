import os
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import umap

sns.set(style='white', context='notebook', rc={'figure.figsize':(14,10)})

import sys
from pathlib import Path

    
def plot_umap_target(
    target_key: np.ndarray,
    values: pd.Series,
    is_categorical: bool,
    colorbar_label: str,
    n_values,
    embeddings_2d
):
    print(f"Generating 3x1 grid plot for {target_key}...")
    
    # Create a 3x1 subplot layout (3 rows, 1 column)
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 6))

    mask_nan = values.isna()
    mask_valid = ~mask_nan
    valid_vals = values[mask_valid]

    if is_categorical:
        categories = valid_vals.astype("category")
        cat_codes = categories.cat.codes
        cat_labels = list(categories.cat.categories)

    for idx, n in enumerate(n_values):
        ax = axes[idx]
        emb = embeddings_2d[n]

        # Plot missing/NaN/uncertain values in gray first
        ax.scatter(
            emb[mask_nan, 0],
            emb[mask_nan, 1],
            c="lightgray",
            s=1,
            alpha=0.3,
            label="Uncertain/Missing" if idx == 0 else ""
        )

        if is_categorical:
            scatter = ax.scatter(
                emb[mask_valid, 0],
                emb[mask_valid, 1],
                c=cat_codes,
                cmap="tab10",
                s=5,
                alpha=0.8,
            )
            if idx == 0:
                ax.legend(
                handles=scatter.legend_elements()[0],
                labels=cat_labels,
                bbox_to_anchor=(1.05, 1),
                loc="upper left",
                fontsize="small",
            )
        else:
            scatter = ax.scatter(
                emb[mask_valid, 0],
                emb[mask_valid, 1],
                c=valid_vals,
                cmap="coolwarm",
                s=5,
                alpha=0.8,
            )
            # if idx == 0:
            #     fig.colorbar(scatter, ax=axes, orientation="vertical", label=colorbar_label, shrink=0.6)
        ax.set_title(f"UMAP (n_neighbors={n}) - {target_key.replace('_', ' ').title()}")
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.legend()

    plt.tight_layout()
    filename = f"umap_{target_key}_grid.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {filename}")
    
    
    
def main():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `src` importable

    from src.data import study_table, findings_texts, usable_findings, binary_label
    from src.embeddings import embedding_path, load_embeddings

    studies = study_table()
    X = load_embeddings(embedding_path("findings"), studies)   # shape (10000, 768)
    
    n_values = [5, 15, 200]
    
    # Precompute UMAP embeddings for each n_neighbors to save computation time
    embeddings_2d = {}
    for n in n_values:
        print(f"Running UMAP with n_neighbors={n}...")
        reducer = umap.UMAP(n_neighbors=n, min_dist=0.1, random_state=42)
        embeddings_2d[n] = reducer.fit_transform(X)
        
    # Define the targets you want to visualize in 3x1 grids
    edema = binary_label(studies, "edema")
    health_sys = studies["health_system"]
    
    targets = [
        ("edema", edema, False, "Edema Label (0 = Negative, 1 = Positive)"),
        ("health_system", health_sys, True, "Health System")
    ]
    
    for target_key, values, is_categorical, colorbar_label in targets:
        plot_umap_target(
            target_key=target_key,
            values=values,
            is_categorical=is_categorical,
            colorbar_label=colorbar_label,
            n_values=n_values,
            embeddings_2d=embeddings_2d
        )
    
    

if __name__ == "__main__":
    main()