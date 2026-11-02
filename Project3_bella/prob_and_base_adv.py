
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import StratifiedGroupKFold, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import balanced_accuracy_score, r2_score
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt

from src.data import study_table, findings_texts, top_k_classes
from src.embeddings import embedding_path, load_embeddings


N_SPLITS = 5
RANDOM_STATE = 42

# --------------------------------------------------
# Confusion matrix


def plot_confusion_matrices(
    y_true,
    embedding_predictions,
    tfidf_predictions,
    task,
):
    """
    Plot normalized confusion matrices for the embedding
    probe and TF-IDF baseline using out-of-fold predictions.
    """

    Path("images").mkdir(exist_ok=True)

    # Use consistent class order in both matrices
    classes = np.unique(y_true)

    if task == "support_devices":
        display_labels = ["Negative", "Positive"]
    else:
        display_labels = [str(c) for c in classes]

    fig, axes = plt.subplots(
        1, 2,
        figsize=(15, 6),
        constrained_layout=True
    )

    models = [
        ("Embedding probe", embedding_predictions),
        ("TF-IDF baseline", tfidf_predictions),
    ]

    for ax, (name, predictions) in zip(axes, models):

        cm = confusion_matrix(
            y_true,
            predictions,
            labels=classes,
            normalize="true"
        )

        ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=display_labels
        ).plot(
            ax=ax,
            cmap="Blues",
            values_format=".2f",
            colorbar=False
        )

        score = balanced_accuracy_score(
            y_true,
            predictions
        )

        ax.set_title(
            f"{name}\nBalanced accuracy = {score:.3f}"
        )

        ax.tick_params(axis="x", labelrotation=45)

    fig.suptitle(
        f"Confusion matrices: {task}",
        fontsize=15
    )

    # Create images directory if it doesn't exist
    output_dir = Path(__file__).resolve().parent / "images"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save confusion matrix
    plt.savefig(
        output_dir / f"final_confusion_matrix_{task}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)
# --------------------------------------------------


# --------------------------------------------------
# Scatter plot
def plot_regression_predictions(
    y_true,
    embedding_predictions,
    tfidf_predictions,
    task,
):
    """
    Compare actual and predicted values for regression.
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

    for ax, (name, predictions) in zip(axes, models):

        ax.scatter(
            y_true,
            predictions,
            alpha=0.3,
            s=12
        )

        # Perfect prediction reference line
        lower = min(y_true.min(), predictions.min())
        upper = max(y_true.max(), predictions.max())

        ax.plot(
            [lower, upper],
            [lower, upper],
            "r--",
            label="Perfect prediction"
        )

        score = r2_score(y_true, predictions)

        ax.set_title(f"{name}\nR² = {score:.3f}")
        ax.set_xlabel(f"Actual {task} value")
        ax.set_ylabel(f"Predicted {task} value")
        ax.legend()

    fig.suptitle(
        f"{task} prediction",
        fontsize=15
    )

    plt.savefig(
        f"final_regression_{task}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)
# --------------------------------------------------

# --------------------------------------------------
# Patient-wise cross-validation
# --------------------------------------------------

def evaluate_model(X, y, groups, model, splits, task):
    """
    Generate one out-of-fold prediction per study.

    Classification: predicted class labels
    Regression: predicted continuous values

    Return one overall score.
    """

    predictions = np.empty(len(y), dtype=object if task != "regression" else float)

    if task == "regression":
        predictions[:] = np.nan
    else:
        predictions[:] = None

    for fold, (train_idx, test_idx) in enumerate(splits, start=1):

        # Verify patient-wise separation
        assert set(groups[train_idx]).isdisjoint(
            set(groups[test_idx])
        ), "Patient leakage detected!"

        model.fit(X[train_idx], y[train_idx])

        predictions[test_idx] = model.predict(X[test_idx])

        print(f"  Fold {fold}/{len(splits)} completed")

    if pd.isna(predictions).any():
        raise ValueError("Some studies have no predictions.")

    if task == "regression":
        predictions = predictions.astype(float)
        score = r2_score(y, predictions)
    else:
        predictions = np.asarray(predictions.tolist())
        score = balanced_accuracy_score(y, predictions)

    return score, predictions


# --------------------------------------------------
# Prepare target variables
# --------------------------------------------------

def prepare_target(studies, texts, task):

    if task == "manufacturer":

        labels = studies["manufacturer"].astype("string").str.strip()
        labels = labels.replace("", pd.NA)

        # Keep only the 6 most frequent manufacturers
        # Assumes top_k_classes returns the selected classes
        top_classes = top_k_classes(labels.dropna(), 6)

        mask = labels.isin(top_classes)

        y_all = labels

    elif task == "institution":

        labels = studies["institution"].astype("string").str.strip()
        labels = labels.replace("", pd.NA)

        # Keep only the 6 most frequent manufacturers
        # Assumes top_k_classes returns the selected classes
        top_classes = top_k_classes(labels.dropna(), 10)

        mask = labels.isin(top_classes)

        y_all = labels

    elif task == "health_system":
    
            labels = studies["health_system"].astype("string").str.strip()
            labels = labels.replace("", pd.NA)
    
            # Keep only the 6 most frequent manufacturers
            # Assumes top_k_classes returns the selected classes
            top_classes = top_k_classes(labels.dropna(), 3)
    
            mask = labels.isin(top_classes)
    
            y_all = labels

    elif task == "report_length":

        # Number of words in the findings text
        y_all = studies["findings_words"].astype("int")

        mask = texts.notna() & texts.astype(str).str.strip().ne("")

    elif task == "age_years":
        y_all = pd.to_numeric(studies["age_years"], errors="coerce")
        mask = y_all.notna() & np.isfinite(y_all)

    elif task == "support_devices":

        labels = studies["chexbert_support_devices"]

        mask = labels.isin(["positive", "negative"])

        y_all = labels.map({
            "negative": 0,
            "positive": 1,
        })

    else:
        raise ValueError(f"Unknown task: {task}")

    # Require valid patient IDs and report texts
    mask &= studies["patient_id"].notna()
    mask &= texts.notna()
    mask &= texts.astype(str).str.strip().ne("")

    indices = np.flatnonzero(mask.to_numpy())

    y = y_all.iloc[indices].to_numpy()

    return indices, y


# --------------------------------------------------
# Evaluate one task
# --------------------------------------------------

def evaluate_task(studies, embeddings, texts, task):

    print(f"\n{'=' * 55}")
    print(f"Task: {task}")
    print(f"{'=' * 55}")

    indices, y = prepare_target(studies, texts, task)

    X_embeddings = embeddings[indices]
    X_text = texts.iloc[indices].astype(str).to_numpy()

    groups = studies.iloc[indices]["patient_id"].to_numpy()

    print(f"Studies: {len(y)}")
    print(f"Patients: {len(np.unique(groups))}")

    # ----------------------------------------------
    # Cross-validation strategy
    # ----------------------------------------------

    if task in ("report_length","age_years"):

        cv = GroupKFold(n_splits=N_SPLITS)

        splits = list(
            cv.split(X_embeddings, y, groups=groups)
        )

        task_type = "regression"
        metric_name = "R2"

        embedding_model = Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", Ridge(alpha=1.0)),
        ])

        tfidf_model = Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=10000,
                stop_words="english"
            )),
            ("regressor", Ridge(alpha=1.0)),
        ])

    else:

        if len(np.unique(y)) < 2:
            raise ValueError(
                f"{task} requires at least two classes."
            )

        cv = StratifiedGroupKFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=RANDOM_STATE,
        )

        splits = list(
            cv.split(X_embeddings, y, groups=groups)
        )

        task_type = "classification"
        metric_name = "balanced_accuracy"

        embedding_model = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
            )),
        ])

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

    # ----------------------------------------------
    # Embedding probe
    # ----------------------------------------------

    print("\nEvaluating embedding probe...")

    embedding_score, embedding_predictions = evaluate_model(
        X_embeddings,
        y,
        groups,
        embedding_model,
        splits,
        task_type,
    )

    # ----------------------------------------------
    # TF-IDF baseline
    # ----------------------------------------------

    print("\nEvaluating TF-IDF baseline...")

    tfidf_score, tfidf_predictions = evaluate_model(
        X_text,
        y,
        groups,
        tfidf_model,
        splits,
        task_type,
    )

    print(f"\n{task} results:")
    print(f"Embedding {metric_name}: {embedding_score:.4f}")
    print(f"TF-IDF {metric_name}:    {tfidf_score:.4f}")

    if task == "report_length" or task == "age_years":
        plot_regression_predictions(
            y_true=y,
            embedding_predictions=embedding_predictions,
            tfidf_predictions=tfidf_predictions,
            task=task,
        )
    else:
        plot_confusion_matrices(
            y_true=y,
            embedding_predictions=embedding_predictions,
            tfidf_predictions=tfidf_predictions,
            task=task,
        )

    return {
        "task": task,
        "metric": metric_name,
        "n_studies": len(y),
        "n_patients": len(np.unique(groups)),
        "embedding_score": embedding_score,
        "tfidf_score": tfidf_score,
    }


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    studies = study_table().reset_index(drop=True)

    embeddings = load_embeddings(
        embedding_path("findings"),
        studies,
    )

    # Assumes findings_texts returns an aligned Series
    texts = findings_texts(studies)

    if not isinstance(texts, pd.Series):
        texts = pd.Series(texts)

    texts = texts.reset_index(drop=True)

    assert len(studies) == len(embeddings)
    assert len(studies) == len(texts)

    tasks = [
        "report_length",
        "health_system",
        "age_years"
    ]

    results = []

    for task in tasks:
        result = evaluate_task(
            studies,
            embeddings,
            texts,
            task,
        )
        results.append(result)

    results_df = pd.DataFrame(results)

    print("\n" + "=" * 65)
    print("FINAL RESULTS")
    print("=" * 65)

    print(results_df.to_string(index=False))

    results_df.to_csv(
        "probe_results.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
