# week1/run_metadata_probes.py
from pathlib import Path
import numpy as np
import json
import pandas as pd
import sys
from tqdm import tqdm
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `src` importable
        
from src.data import binary_label, findings_texts, study_table, usable_findings, top_k_classes
from src.embeddings import embedding_path, load_embeddings

def evaluate_class_probe(
        X_or_texts, 
        y, 
        groups, 
        hyperparams={
            "min_df":1, 
            "ngram_range":(1,2), 
            "max_features":50000, 
            "max_iter_tfidf":2000, 
            "max_iter_scalar":2000,
            "n_splits":5,
        }, 
        is_text: bool = False, 
        is_binary: bool = False,
        random_state: int = 42
    ):
    """
    Run StratifiedGroupKFold cross-validation and return out-of-fold AUROC for binary classification or Balanced Accuracy for multi-class classification.
    
    Args:
        X_or_texts: Either a 2D array of embeddings or a list of texts.
        y: Target variable (binary or multi-class).
        groups: Group labels for StratifiedGroupKFold (patient IDs).
        hyperparams: Dictionary of hyperparameters for the probe (From config_probing.json).
        is_text: Whether the input is text (True) or embeddings (False).
        random_state: Random seed for reproducibility.
    Returns:
        AUROC score for binary classification or Balanced Accuracy score for multi-class classification.
    """
    # Use StratifiedGroupKFold to ensure that all samples from the same patient are in the same fold and that class distribution is preserved
    cv = StratifiedGroupKFold(n_splits=hyperparams["n_splits"], shuffle=True, random_state=random_state)

    # Initialize out-of-fold predictions and true labels
    if is_binary:
        oof_preds = np.zeros(len(y))
        oof_true = np.zeros(len(y))
    else:
        oof_preds = []
        oof_true = []

    # Train and evaluate the probe for each fold
    for train_idx, test_idx in tqdm(cv.split(X_or_texts, y, groups), total=hyperparams["n_splits"], desc=f"Cross-Validation of Classification Probe for {"TF-IDF" if is_text else "Embeddings"}"):
        # Define the pipeline based on whether the input is text or embeddings
        if is_text:
            pipeline = Pipeline([
                ("tfidf", TfidfVectorizer(max_features=hyperparams["max_features"], ngram_range=tuple(hyperparams["ngram_range"]), min_df=hyperparams["min_df"], stop_words="english")),
                ("clf", LogisticRegression(max_iter=hyperparams["max_iter_tfidf"], class_weight="balanced", random_state=random_state)),
            ])
            X_train = [X_or_texts[i] for i in train_idx]
            X_test = [X_or_texts[i] for i in test_idx]
        else:
            pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=hyperparams["max_iter_scalar"], class_weight="balanced", random_state=random_state)),
            ])
            X_train = X_or_texts[train_idx]
            X_test = X_or_texts[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        # Fit the pipeline on the training data and predict on the test data
        pipeline.fit(X_train, y_train)

        if is_binary:
            # For binary targets, collect probability of the positive class for AUROC
            preds = pipeline.predict_proba(X_test)[:, 1]
            oof_preds[test_idx] = preds
            oof_true[test_idx] = y_test
        else:
            # For multi-class targets, collect discrete class predictions for Balanced Accuracy
            preds = pipeline.predict(X_test)
            oof_preds.extend(preds)
            oof_true.extend(y_test)
            
    return oof_true, oof_preds


def evaluate_regression_probe(
        X_or_texts, 
        y, 
        groups, 
        hyperparams={
            "min_df":1, 
            "ngram_range":(1,2), 
            "max_features":50000, 
            "max_iter_tfidf":2000, 
            "max_iter_scalar":2000,
            "n_splits":5,
        }, 
        is_text: bool = False, 
        random_state: int = 42
    ):
    """
    Run GroupKFold cross-validation and return out-of-fold R^2 score for linear probe.
    
    Args:
        X_or_texts: Either a 2D array of embeddings or a list of texts.
        y: Target variable.
        groups: Group labels for GroupKFold (patient IDs).
        hyperparams: Dictionary of hyperparameters for the probe (From config_probing.json).
        is_text: Whether the input is text (True) or embeddings (False).
        random_state: Random seed for reproducibility.
    Returns:
        R^2 score for the out-of-fold predictions.
    """
    # Use GroupKFold to ensure that all samples from the same patient are in the same fold
    cv = GroupKFold(n_splits=hyperparams["n_splits"])
    oof_preds = np.zeros(len(y))
    oof_true = np.zeros(len(y))

    # Train and evaluate the probe for each fold
    for train_idx, test_idx in tqdm(cv.split(X_or_texts, y, groups), total=hyperparams["n_splits"], desc=f"Cross-Validation of Regression Probe for {"TF-IDF" if is_text else "Embeddings"}"):
        # Define the pipeline based on whether the input is text or embeddings
        if is_text:
            pipeline = Pipeline([
                ("tfidf", TfidfVectorizer(max_features=hyperparams["max_features"], ngram_range=tuple(hyperparams["ngram_range"]), min_df=hyperparams["min_df"], stop_words="english")),
                ("reg", Ridge(random_state=random_state)),
            ])
            X_train = [X_or_texts[i] for i in train_idx]
            X_test = [X_or_texts[i] for i in test_idx]
        else:
            pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("reg", Ridge(random_state=random_state)),
            ])
            X_train = X_or_texts[train_idx]
            X_test = X_or_texts[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        # Fit the pipeline on the training data and predict on the test data
        pipeline.fit(X_train, y_train)
        oof_preds[test_idx] = pipeline.predict(X_test)
        oof_true[test_idx] = y_test

    return oof_true, oof_preds

def evaluate_probe(
        X_or_texts, 
        y, 
        groups, 
        task_type: str,
        hyperparams={
            "min_df":1, 
            "ngram_range":(1,2), 
            "max_features":50000, 
            "max_iter_tfidf":2000, 
            "max_iter_scalar":2000,
            "n_splits":5,
        }, 
        is_text: bool = False, 
        random_state: int = 42
    ):
    """
    Evaluate a probe (either classification or regression) using cross-validation and return out-of-fold predictions.
    
    Args:
        X_or_texts: Either a 2D array of embeddings or a list of texts.
        y: Target variable (binary, multi-class, or continuous).
        groups: Group labels for cross-validation (patient IDs).
        task_type: Type of the probe ("multiclass", "binary" or "regression").
        hyperparams: Dictionary of hyperparameters for the probe (From config_probing.json).
        is_text: Whether the input is text (True) or embeddings (False).
        random_state: Random seed for reproducibility.
    Returns:
        Tuple of (true labels, predicted labels or values, metric-name).
    """
    if task_type == "multiclass":
        y_true, y_preds = evaluate_class_probe(X_or_texts, y, groups, hyperparams=hyperparams, is_text=is_text, is_binary=False, random_state=random_state)
        metric = "Balanced Accuracy"
    elif task_type == "binary":
        y_true, y_preds = evaluate_class_probe(X_or_texts, y, groups, hyperparams=hyperparams, is_text=is_text, is_binary=True, random_state=random_state)
        metric = "AUROC"
    elif task_type == "regression":
        y_true, y_preds = evaluate_regression_probe(X_or_texts, y, groups, hyperparams=hyperparams, is_text=is_text, random_state=random_state)
        metric = "R^2"
    else:
        raise ValueError(f"Unknown task type: {task_type}")
        
    return y_true, y_preds, metric
    


def score_task(y_true, y_pred, task_type):
    """
    Compute the evaluation metric for a given task type.
    
    Args:
        y_true: True labels or values.
        y_pred: Predicted labels or values.
        task_type: Type of the task ("binary", "multiclass", or "regression").
    Returns:
        Evaluation score (AUROC, Balanced Accuracy, or R^2).
    """
    if task_type == "binary":
        return roc_auc_score(y_true, y_pred)
    elif task_type == "multiclass":
        return balanced_accuracy_score(y_true, y_pred)
    elif task_type == "regression":
        return r2_score(y_true, y_pred)
    else:
        raise ValueError(f"Unknown task type: {task_type}")


def evaluate_task(
        task_key, 
        task_name, 
        config, 
        studies, 
        X_embed, 
        texts, 
        mask_usable, 
        random_state=42
    ):
    """
    Evaluate a specific task (binary, multiclass, or regression) using both embedding-based and TF-IDF-based probes.
    
    Args:
        task_key: Key for the task in the studies DataFrame.
        task_name: Human-readable name for the task.
        config: Hyperparameter configuration for the task.
        studies: DataFrame containing study metadata.
        X_embed: 2D array of embeddings for the studies.
        texts: List of text findings for the studies.
        mask_usable: Boolean mask indicating usable findings (at least 3 words).
        random_state: Random seed for reproducibility.
    Returns:
        Tuple containing embedding score, TF-IDF score, the evaluation metric name, number of samples, preds from embedded data, preds from TF-IDF and ground-truth.
    """
    # Add binary labels for edema and pleural effusion tasks to studies DataFrame
    if task_key in ["edema", "pleural_effusion"]:
        studies[task_key] = binary_label(studies, task_key)
        
    # Determine the target variable based on the task type
    if config["top_k"] != -1:
        task_col = top_k_classes(studies[task_key], config["top_k"]) # For multi-class tasks with many classes, filter to top-k classes
    else:
        task_col = studies[task_key]
    
    # Filter studies for usable findings and non-null target variable
    mask = mask_usable & task_col.notna()
    sub_studies = studies[mask].reset_index(drop=True)
    
    y_task = sub_studies[task_key].values
    groups = sub_studies["patient_id"].values
    
    valid_indices = studies.index[mask].values
    
    X_sub_embed = X_embed[valid_indices]
    sub_texts = [texts[i] for i in valid_indices]
    
    assert len(y_task) == len(groups) == X_sub_embed.shape[0] == len(sub_texts), "Filtered studies, embeddings, and texts must have the same length."
    n = len(y_task)
    
    # Evaluating models
    y_true, y_preds_embed, metric = evaluate_probe(X_sub_embed, y_task, groups, config["type"], hyperparams=config, is_text=False, random_state=random_state)
    y_true, y_preds_tfidf, metric = evaluate_probe(sub_texts, y_task, groups, config["type"], hyperparams=config, is_text=True, random_state=random_state)

    # Score models
    score_embed = score_task(y_true, y_preds_embed, config["type"])
    score_tfidf = score_task(y_true, y_preds_tfidf, config["type"])
    
    # Print results
    print(
        f"{task_name} -> Random {metric}: {config['random_result']} | BioClinicalBERT {metric}: {score_embed:.4f} | TF-IDF {metric}: {score_tfidf:.4f}"
    )
    print("-" * 80)
    
    # Convert binary scores to predictions
    if config["type"] == "binary":
        y_preds_embed = np.round(y_preds_embed)
        y_preds_tfidf = np.round(y_preds_tfidf)
    
    return score_embed, score_tfidf, metric, n, y_preds_embed, y_preds_tfidf, y_true

def evaluate_seed(
        random_state, 
        tasks, 
        configs, 
        studies, 
        X_embed, 
        texts, 
        mask_usable
    ):
    """
    Evaluate all tasks for a specific random seed and return the scores.
    
    Args:
        random_state: Random seed for reproducibility.
        tasks: Dictionary of task keys and names.
        configs: Dictionary of hyperparameter configurations for each task.
        studies: DataFrame containing study metadata.
        X_embed: 2D array of embeddings for the studies.
        texts: List of text findings for the studies.
        mask_usable: Boolean mask indicating usable findings (at least 3 words).
    Returns:
        Dictionary containing scores for each task.
    """
    # Run evaluation for each task
    scores = {}
    for task_key, task_name in tasks.items():
        print(f"--- Evaluating {task_name} ---")
        config = configs[task_key]
        
        # Filter task-specific studies and data
        filtered_studies = studies.copy()
        filtered_texts = texts.copy()
        filtered_X_embed = X_embed.copy()
        filtered_mask_usable = mask_usable.copy()
        if task_key == "sex":
            # Create filtering mask for studies with sex labels M or F
            mask_sex = filtered_studies[task_key].isin(["M", "F"])
            filtered_studies = filtered_studies[mask_sex].reset_index(drop=True)
            filtered_texts = [filtered_texts[i] for i in filtered_studies.index]
            filtered_X_embed = filtered_X_embed[mask_sex]
            filtered_mask_usable = filtered_mask_usable[mask_sex].reset_index(drop=True)
            assert len(filtered_studies) == len(filtered_texts) == filtered_X_embed.shape[0] == len(filtered_mask_usable), "Filtered studies, texts, embeddings, and mask must have the same length."
            
            # Map sex labels to binary values: M -> 1.0, F -> 0.0
            filtered_studies[task_key] = filtered_studies[task_key].map({"M": 1.0, "F": 0.0})
        
        # Evaluate the task and store the results
        score_embed, score_tdidf, metric, n, y_preds_embed, y_preds_tfidf, y_true = evaluate_task(task_key, task_name, config, filtered_studies, filtered_X_embed, filtered_texts, filtered_mask_usable, random_state=random_state)
        scores[task_name] = {
            "Embedding Score": float(score_embed),
            "TF-IDF Score": float(score_tdidf),
            "Random Result": float(config["random_result"]),
            "Metric": str(metric),
            "Task Type": str(config["type"]),
            "Num Samples": int(n),
            "random_state": int(random_state),
            "y_preds_embed": list(y_preds_embed),
            "y_preds_tfidf": list(y_preds_tfidf),
            "y_true": list(y_true)
        }
    return scores


def evaluate_all_seeds(
    n_random_seeds,
    tasks,
    configs,
    studies,
    X_embed,
    texts,
    mask_usable,
):
    """
    Evaluate all tasks across multiple random seeds and return the average scores.
    
    Args:
        n_random_seeds: Number of random seeds to evaluate.
        tasks: Dictionary of task keys and names.
        configs: Dictionary of hyperparameter configurations for each task.
        studies: DataFrame containing study metadata.
        X_embed: 2D array of embeddings for the studies.
        texts: List of text findings for the studies.
        mask_usable: Boolean mask indicating usable findings (at least 3 words).
    Returns:
        Dictionary containing average scores for each task across all random seeds.
    """
    scores = {}
    random_seeds = list(range(n_random_seeds))
    for random_state in random_seeds:
        print("")
        print("=" * 80)
        print(f"=== Running evaluation for random seed {random_state} ===")
        scores[random_state] = {"random_state": random_state}
        
        # Evaluate all tasks for the current random seed
        scores[random_state].update(evaluate_seed(random_state, tasks, configs, studies, X_embed, texts, mask_usable))
    
    # Average results across random seeds
    avg_scores = {}
    for task_name in tasks.values():
        avg_scores[task_name] = {
            "Embedding Score": float(np.mean([scores[seed][task_name]["Embedding Score"] for seed in range(n_random_seeds)])),
            "TF-IDF Score": float(np.mean([scores[seed][task_name]["TF-IDF Score"] for seed in range(n_random_seeds)])),
            "Random Result": float(scores[0][task_name]["Random Result"]),
            "Metric": str(scores[0][task_name]["Metric"]),
            "Num Samples": int(scores[0][task_name]["Num Samples"]),
        }
    avg_scores["n_random_seeds"] = n_random_seeds
    avg_scores["random_seeds"] = random_seeds
    avg_scores["configs"] = configs
    avg_scores["scores_per_seed"] = scores
    
    return avg_scores

def main():    
    time_start = time.time()
    
    n_random_seeds = 5
    
    out_filepath = "week1/metadata_probe_results.json"
    min_findings_length = 3  # Minimum number of words in Findings text to be considered usable
    
    tasks = {
        "edema": "Edema",
        "pleural_effusion": "Pleural Effusion",
        "sex": "Sex",
        "health_system": "Health System",
        "age_years": "Age (Years)",
        "study_year": "Study Year",
        "findings_words": "Report Length (Words)",
        "study_description": "Study Description",
        "manufacturer": "Scanner Manufacturer",
        "institution": "Hospital Site",
    }
    # Load hyperparameter configurations for each task from the JSON file
    configs = json.load(open("week1/config_probing.json", "r"))
    
    # Load studies, embeddings, and texts
    studies = study_table()
    X_embed = load_embeddings(embedding_path("findings"), studies)
    texts = findings_texts(studies)
    
    # Create a mask for usable findings
    mask_usable = usable_findings(studies, min_words=min_findings_length)

    # Evaluate all tasks across multiple random seeds
    avg_scores = evaluate_all_seeds(n_random_seeds, tasks, configs, studies, X_embed, texts, mask_usable)
    # Save the average scores to a JSON file
    json.dump(avg_scores, open(out_filepath, "w"), indent=4)
    
    time_end = time.time()
    print(f"Total time: {int(time_end-time_start)}")
    


if __name__ == "__main__":
    main()