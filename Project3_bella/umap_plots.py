# Code from ChatGPT
import umap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
from src.data import study_table, findings_texts, usable_findings, binary_label
from src.embeddings import embedding_path, load_embeddings


def plot_umap_comparison(
    X,
    df,
    color_column,
    neighbors=(5, 15, 200),
    fixed_colors=None,
    missing_color="lightgrey",
    faded_categories=("Missing", "blank", "uncertain"),
    faded_alpha=0.1,
    highlight_alpha=0.85
):
    
    if len(X) != len(df):
        raise ValueError(
            f"X has {len(X)} rows but df has {len(df)} rows."
        )

    # Convert labels to strings and normalize missing values
    labels = df[color_column].astype("string").str.strip()

    labels = labels.replace(
        ["nan", "NaN", "None", "none", "null", ""],
        pd.NA
    ).fillna("Missing")

    categories = list(pd.unique(labels))

    # Assign colors
    if fixed_colors is not None:
        color_map = fixed_colors.copy()

        # Automatically assign colors to unmapped categories
        unmapped = [
            c for c in categories
            if c not in color_map and c != "Missing"
        ]

        cmap = plt.get_cmap("tab20", max(len(unmapped), 1))

        for i, category in enumerate(unmapped):
            color_map[category] = cmap(i)

    else:
        valid_categories = [
            c for c in categories if c != "Missing"
        ]

        cmap = plt.get_cmap(
            "tab20" if len(valid_categories) <= 20 else "gist_ncar",
            max(len(valid_categories), 1)
        )

        color_map = {
            category: cmap(i)
            for i, category in enumerate(valid_categories)
        }

    # Always assign missing values a valid color
    color_map["Missing"] = missing_color

    point_colors = labels.map(color_map).tolist()
    # Identify categories that should be less visible
    faded_mask = labels.str.lower().isin(
        [c.lower() for c in faded_categories]
    ).to_numpy()

    # Everything else should stand out
    highlight_mask = ~faded_mask


    fig, axes = plt.subplots(
        1,
        len(neighbors),
        figsize=(6 * len(neighbors), 5),
        squeeze=False,
        constrained_layout=True
    )

    axes = axes.ravel()

    for ax, n in zip(axes, neighbors):
        reducer = umap.UMAP(
            n_neighbors=n,
            n_components=2,
            min_dist=0.05,
            metric="cosine",
            random_state=40,
            init = "random"
        )

        embedding = reducer.fit_transform(X)

        # Plot faded categories first (background)
        ax.scatter(
            embedding[faded_mask, 0],
            embedding[faded_mask, 1],
            c=np.asarray(point_colors, dtype=object)[faded_mask].tolist(),
            s=12,
            alpha=faded_alpha,
            linewidths=0,
            zorder=1
        )

        # Plot important categories on top
        ax.scatter(
            embedding[highlight_mask, 0],
            embedding[highlight_mask, 1],
            c=np.asarray(point_colors, dtype=object)[highlight_mask].tolist(),
            s=20,
            alpha=highlight_alpha,
            linewidths=0,
            zorder=2
        )

        ax.set_title(f"n_neighbors = {n}")
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")

    # Shared legend
    handles = [
        Line2D(
            [0], [0],
            marker="o",
            linestyle="",
            color="w",
            markerfacecolor=color_map[category],
            label=str(category),
            markersize=7
        )
        for category in categories
    ]

    fig.legend(
        handles=handles,
        loc="outside lower center",
        ncol=min(len(handles), 6),
        title=color_column
    )

    # Create directory if necessary
    Path("images").mkdir(exist_ok=True)

    plt.savefig(
        f"images/umap_{color_column}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    return fig

studies = study_table()
# mask = studies["chexbert_edema"].isin(["positive", "negative"])

X = load_embeddings(embedding_path("findings"), studies)   # shape (10000, 768)

columns = ["n_images_in_study", "sex", "ethnic_group", "modality", "body_part", "view_position", "view_position_raw", "health_system",
           "chexbert_enlarged_cardiomediastinum", "chexbert_cardiomegaly", "chexbert_lung_opacity", "chexbert_lung_lesion", "chexbert_edema",
           "chexbert_consolidation", "chexbert_pneumonia", "chexbert_pneumonia", "chexbert_atelectasis", "chexbert_pneumothorax",
           "chexbert_pleural_effusion", "chexbert_pleural_other", "chexbert_fracture", "chexbert_support_devices", "chexbert_no_finding"]


# Sentiment: fixed colours
# Categories: 'chexbert_edema', 'chexbert_pleural_effusion', 'health_system', 'manufacturer', 'age_years', 'study_year', 
# plot_umap_comparison(
#     X,
#     studies,
#     "chexbert_pleural_effusion",
#     fixed_colors={
#         "positive": "red",
#         "negative": "blue",
#         "blank": "yellow",
#         "uncertain": "grey"
#     }
# )

for c in columns:
    plot_umap_comparison(
        X, studies, c)

