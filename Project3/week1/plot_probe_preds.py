import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import os
import json
import sys
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `src` importable

from src.data import findings_texts, study_table, usable_findings
from src.embeddings import embedding_path, load_embeddings
from week1.probes import evaluate_seed


def plot_confusion_matrices(
    y_true,
    models,
    task,
    is_binary,
    output_dir: str = "week1/images",
    filename: str = "confusion_matrix"
):
    """
    Plot normalized confusion matrices for the embedding
    probe and TF-IDF baseline using out-of-fold predictions.
    """
    
    # Font size variables defined at the top of the function
    SUPTITLE_FS = 22
    TITLE_FS = 18
    LABEL_FS = 16
    TICK_FS = 14
    MATRIX_TEXT_FS = 20
    
    

    fig, axes = plt.subplots(
        1, 2,
        figsize=(14, 6),
    )
    # Adjust horizontal space between the two subplots
    fig.subplots_adjust(top=0.75,wspace=0.3) # top=0.75 for Edema, wspace=0.1 for Edema, 

    for ax, (name, predictions, score, metric, n) in zip(axes, models):
        if is_binary:
            y_true_labelled = ["Negative" if x == 0.0 else "Positive" for x in y_true]
            y_preds_labelled = ["Negative" if x == 0.0 else "Positive" for x in predictions]
        else:
            y_true_labelled = y_true
            y_preds_labelled = predictions

        # Use consistent class order in both matrices
        classes = np.unique(y_true_labelled)
        display_labels = [str(c) for c in classes]
        
        cm = confusion_matrix(y_true_labelled, y_preds_labelled, labels=classes, normalize="true")

        ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=display_labels
        ).plot(
            ax=ax,
            cmap="Blues",
            values_format=".2f",
            colorbar=False
        )
        
        # Scale font sizes for the numbers inside the confusion matrix cells
        for text in ax.texts:
            text.set_fontsize(MATRIX_TEXT_FS)

        ax.set_title(
            f"{name}\n{metric} = {score:.3f}",
            fontsize=TITLE_FS
        )

        ax.set_xlabel("Predicted label", fontsize=LABEL_FS)
        ax.set_ylabel("True label", fontsize=LABEL_FS)
        ax.tick_params(axis="x", labelrotation=45, labelsize=TICK_FS)
        ax.tick_params(axis="y", labelsize=TICK_FS)

    fig.suptitle(
        f"Confusion matrices: {task}\nn_samples={n}",
        fontsize=SUPTITLE_FS,
        y=0.96  # Anchors the super-title near the top edge
    )

    # Create images directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save confusion matrix
    plt.savefig(
        f"{output_dir}/{filename}_{task}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)
    
    
def plot_regression_predictions(
    y_true,
    models,
    task,
    output_dir: str = "week1/images",
    filename: str = "regression"
):
    """
    Compare actual and predicted values for regression.
    """
    
    # Font size variables defined at the top of the function
    SUPTITLE_FS = 22
    TITLE_FS = 18
    LABEL_FS = 15
    TICK_FS = 14
    LEGEND_FS = 14

    fig, axes = plt.subplots(
        1, 2,
        figsize=(13, 5),
        constrained_layout=True,
        # gridspec_kw={'wspace': 0.1}  # Reduces space between the two subplots
    )

    for ax, (name, predictions, score, metric, n) in zip(axes, models):

        ax.scatter(
            y_true,
            predictions,
            alpha=0.3,
            s=15
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

        ax.set_title(f"{name}\n{metric} = {score:.3f}", fontsize=TITLE_FS)
        ax.set_xlabel(f"Actual {task}", fontsize=LABEL_FS)
        ax.set_ylabel(f"Predicted {task}", fontsize=LABEL_FS)
        ax.tick_params(axis='x', labelsize=TICK_FS)
        ax.tick_params(axis='y', labelsize=TICK_FS)
        ax.legend(fontsize=LEGEND_FS)

    fig.suptitle(
        f"Regression: {task}\nn_samples={n}",
        fontsize=SUPTITLE_FS
    )
    
    # Create images directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save confusion matrix
    plt.savefig(
        f"{output_dir}/{filename}_{task}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)
    
    
def plot_tasks(
    score_embed,
    score_tfidf,
    n,
    y_preds_embed,
    y_preds_tfidf,
    y_true,
    task_type,
    task_name
):
    if task_type == "regression":
        models = [
            ("Embedding Probe", y_preds_embed, score_embed, "R²", n),
            ("TF-IDF Probe", y_preds_tfidf, score_tfidf, "R²", n)
        ]
        plot_regression_predictions(y_true, models, task_name)
    elif task_type == "multiclass":
        models = [
            ("Embedding Probe", y_preds_embed, score_embed, "Balanced Accuracy", n),
            ("TF-IDF Probe", y_preds_tfidf, score_tfidf, "Balanced Accuracy", n)
        ]
        plot_confusion_matrices(y_true, models, task_name, is_binary=False)
    elif task_type == "binary":
        models = [
            ("Embedding Probe", y_preds_embed, score_embed, "AUROC", n),
            ("TF-IDF Probe", y_preds_tfidf, score_tfidf, "AUROC", n)
        ]
        plot_confusion_matrices(y_true, models, task_name, is_binary=True)
    else:
        raise ValueError(f"Unknown task type: {task_type}")
    
    
def plot_tasks_from_new_eval(
    studies, 
    X_embed,
    texts,
    mask_usable,
    tasks,
    configs,
    random_state=42
):
    scores = evaluate_seed(
        random_state, tasks, configs, studies, X_embed, texts, mask_usable
    )
    for task_key, task_name in tasks.items():
        scores_task = scores[task_name]
        plot_tasks(
            score_embed = scores_task["Embedding Score"],
            score_tfidf = scores_task["TF-IDF Score"],
            n = scores_task["Num Samples"],
            y_preds_embed = scores_task["y_preds_embed"],
            y_preds_tfidf = scores_task["y_preds_tfidf"],
            y_true = scores_task["y_true"],
            task_type = scores_task["Task Type"],
            task_name=task_name
        )
        
        
def plot_tasks_from_json(in_filepath, tasks, random_state):
    
    all_scores = json.load(open(in_filepath, "r"))
    
    for task_key, task_name in tasks.items():
        scores = all_scores["scores_per_seed"][random_state][task_name]
        
        plot_tasks(
            score_embed = scores["Embedding Score"],
            score_tfidf = scores["TF-IDF Score"],
            n = scores["Num Samples"],
            y_preds_embed = scores["y_preds_embed"],
            y_preds_tfidf = scores["y_preds_tfidf"],
            y_true = scores["y_true"],
            task_type = scores["Task Type"],
            task_name=task_name
        )
    
    
    
def main():
    random_state = 42
    
    min_findings_length = 3  # Minimum number of words in Findings text to be considered usable
        
    tasks = {
        "edema": "Edema",
        # "pleural_effusion": "Pleural Effusion",
        "sex": "Sex",
        # "health_system": "Health System",
        # "age_years": "Age (Years)",
        # "study_year": "Study Year",
        "findings_words": "Report Length (Words)",
        # "study_description": "Study Description",
        # "manufacturer": "Scanner Manufacturer",
        # "institution": "Hospital Site",
    }
    # Load hyperparameter configurations for each task from the JSON file
    configs = json.load(open("week1/config_probing.json", "r"))
    
    # Load studies, embeddings, and texts
    studies = study_table()
    X_embed = load_embeddings(embedding_path("findings"), studies)
    texts = findings_texts(studies)
    
    # Create a mask for usable findings
    mask_usable = usable_findings(studies, min_words=min_findings_length)
    
    plot_tasks_from_new_eval(studies, X_embed, texts, mask_usable, tasks, configs, random_state=random_state)
    
    # plot_tasks_from_json(
    #     in_filepath="week1/metadata_probe_results.json",
    #     tasks=tasks,
    #     random_state=random_state
    # )


if __name__ == "__main__":
    main()