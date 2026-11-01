import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, ConfusionMatrixDisplay, confusion_matrix, balanced_accuracy_score
import matplotlib.pyplot as plt
from pathlib import Path

from src.data import (
    study_table,
    findings_texts,
    usable_findings,
    binary_label,
)
from src.embeddings import embedding_path, load_embeddings


# --------------------------------------------------
# Configuration
# --------------------------------------------------

# FINDINGS = [
#     "edema",
#     "pleural_effusion",
# ]

FINDINGS = [   "enlarged_cardiomediastinum", "cardiomegaly", "edema","pneumonia", "atelectasis", "pneumothorax",
           "pleural_effusion", "fracture", "lung_opacity", "lung_lesion", "consolidation"]


N_SPLITS = 5
RANDOM_STATE = 42

# --------------------------------------------------
# Confusion matrix


def plot_confusion_matrices(
    y_true,
    embedding_predictions,
    tfidf_predictions,
    task,
    threshold=0.5,
):
    """
    Plot normalized confusion matrices using out-of-fold
    predicted probabilities.

    AUROC is calculated from probabilities.
    Balanced accuracy is calculated from binary predictions.
    """

    Path("images").mkdir(exist_ok=True)

    fig, axes = plt.subplots(
        1, 2,
        figsize=(13, 5),
        constrained_layout=True
    )

    models = [
        ("Embedding probe", embedding_predictions),
        ("TF-IDF baseline", tfidf_predictions),
    ]

    for ax, (name, probabilities) in zip(axes, models):

        # Convert probabilities into binary predictions
        predictions = (
            np.asarray(probabilities) >= threshold
        ).astype(int)

        # Confusion matrix uses binary predictions
        cm = confusion_matrix(
            y_true,
            predictions,
            labels=[0, 1],
            normalize="true"
        )

        ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["Negative", "Positive"]
        ).plot(
            ax=ax,
            cmap="Blues",
            values_format=".2f",
            colorbar=False
        )

        # AUROC uses probabilities
        auc = roc_auc_score(y_true, probabilities)

        # Balanced accuracy uses class predictions
        balanced_acc = balanced_accuracy_score(
            y_true,
            predictions
        )

        ax.set_title(
            f"{name}\n"
            f"AUROC = {auc:.3f} | "
            f"Balanced accuracy = {balanced_acc:.3f}"
        )

    fig.suptitle(
        f"Confusion matrices: {task}",
        fontsize=15
    )

    fig.savefig(
        f"images/confusion_matrix_{task}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

# --------------------------------------------------


# --------------------------------------------------
# Cross-validation evaluation
# --------------------------------------------------

def evaluate_model(X, y, groups, model, splits):
    """
    Train on 4 folds and predict probabilities on the
    held-out fold.

    Returns one AUROC calculated from all out-of-fold
    probabilities.
    """

    probabilities = np.full(len(y), np.nan)

    for fold, (train_idx, test_idx) in enumerate(splits, start=1):

        # Verify no patient appears in both sets
        train_patients = set(groups[train_idx])
        test_patients = set(groups[test_idx])

        assert train_patients.isdisjoint(test_patients), (
            "Patient leakage detected!"
        )

        model.fit(X[train_idx], y[train_idx])

        probabilities[test_idx] = model.predict_proba(
            X[test_idx]
        )[:, 1]

        print(f"  Fold {fold}/{len(splits)} completed")

    if np.isnan(probabilities).any():
        raise ValueError("Some studies have no predictions.")

    auc = roc_auc_score(y, probabilities)

    return auc, probabilities


# --------------------------------------------------
# Evaluate one finding
# --------------------------------------------------

def evaluate_finding(studies, embeddings, texts, finding):

    print(f"\n{'=' * 50}")
    print(f"Finding: {finding}")
    print(f"{'=' * 50}")

    label_column = f"chexbert_{finding}"

    # Only positive/negative labels are used.
    # Blank, uncertain and missing labels are excluded.
    mask = studies[label_column].isin([
        "positive",
        "negative",
    ])

    # Also require a valid patient ID and report text
    mask &= studies["patient_id"].notna()
    mask &= texts.notna()

    indices = np.flatnonzero(mask.to_numpy())

    X_embeddings = embeddings[indices]

    X_text = texts.iloc[indices].fillna("").astype(str).to_numpy()

    y = (
        studies.iloc[indices][label_column]
        .eq("positive")
        .astype(int)
        .to_numpy()
    )

    groups = (
        studies.iloc[indices]["patient_id"]
        .to_numpy()
    )

    print(f"Studies: {len(y)}")
    print(f"Patients: {len(np.unique(groups))}")
    print(f"Positive: {sum(y == 1)}")
    print(f"Negative: {sum(y == 0)}")

    if len(np.unique(y)) != 2:
        raise ValueError(
            f"{finding}: both classes are required."
        )

    # Same folds for both models
    cv = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    splits = list(
        cv.split(X_embeddings, y, groups=groups)
    )

    # ----------------------------------------------
    # 1. Linear probe on embeddings
    # ----------------------------------------------

    print("\nEvaluating embedding linear probe...")

    embedding_model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
        )),
    ])

    embedding_auc, embedding_probs = evaluate_model(
        X_embeddings,
        y,
        groups,
        embedding_model,
        splits,
    )

    # ----------------------------------------------
    # 2. TF-IDF baseline
    # ----------------------------------------------

    print("\nEvaluating TF-IDF baseline...")

    tfidf_model = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_features=50000,
        )),
        ("classifier", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
        )),
    ])

    tfidf_auc, tfidf_probs = evaluate_model(
        X_text,
        y,
        groups,
        tfidf_model,
        splits,
    )

    print(f"\nResults for {finding}:")
    print(f"Embedding AUROC: {embedding_auc:.4f}")
    print(f"TF-IDF AUROC:    {tfidf_auc:.4f}")

    plot_confusion_matrices(
        y_true=y,
        embedding_predictions=embedding_probs,
        tfidf_predictions=tfidf_probs,
        task=finding,
    )
    
    return {
        "finding": finding,
        "n_studies": len(y),
        "n_patients": len(np.unique(groups)),
        "n_positive": int(sum(y == 1)),
        "n_negative": int(sum(y == 0)),
        "embedding_auc": embedding_auc,
        "tfidf_auc": tfidf_auc,
    }


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    studies = study_table().reset_index(drop=True)

    # Load your existing fixed embeddings
    embeddings = load_embeddings(
        embedding_path("findings"),
        studies,
    )

    # Extract the findings report text.
    # Assumes findings_texts returns a Series aligned
    # with studies; adjust if your function differs.
    texts = findings_texts(studies)

    if not isinstance(texts, pd.Series):
        texts = pd.Series(texts)

    texts = texts.reset_index(drop=True)

    assert len(studies) == len(embeddings)
    assert len(studies) == len(texts)

    results = []

    for finding in FINDINGS:
        result = evaluate_finding(
            studies,
            embeddings,
            texts,
            finding,
        )
        results.append(result)

    results_df = pd.DataFrame(results)

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)

    print(results_df.to_string(index=False))

    results_df.to_csv(
        "linear_probe_results.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
