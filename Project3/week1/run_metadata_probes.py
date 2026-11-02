# scripts/run_metadata_probes.py
from pathlib import Path
import numpy as np
import pandas as pd
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def evaluate_class_probe(X_or_texts, y, groups, is_text: bool = False):
    classes = np.unique(y)
    is_binary = len(classes) == 2

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    if is_binary:
        oof_preds = np.zeros(len(y))
        oof_true = np.zeros(len(y))
    else:
        oof_preds = []
        oof_true = []

    for train_idx, test_idx in cv.split(X_or_texts, y, groups):
        if is_text:
            pipeline = Pipeline([
                ("tfidf", TfidfVectorizer(max_features=10000, ngram_range=(1,2),stop_words="english")),
                ("clf", LogisticRegression(max_iter=1000, random_state=42)),
            ])
            X_train = [X_or_texts[i] for i in train_idx]
            X_test = [X_or_texts[i] for i in test_idx]
        else:
            pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, random_state=42)),
            ])
            X_train = X_or_texts[train_idx]
            X_test = X_or_texts[test_idx]

        y_train = y[train_idx]
        y_test = y[test_idx]

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

    if is_binary:
        return roc_auc_score(oof_true, oof_preds)
    else:
        return balanced_accuracy_score(oof_true, oof_preds)


def evaluate_regression_probe(X_or_texts, y, groups, is_text: bool = False):
    cv = GroupKFold(n_splits=5)
    oof_preds = np.zeros(len(y))
    oof_true = np.zeros(len(y))

    for train_idx, test_idx in cv.split(X_or_texts, y, groups):
        if is_text:
            pipeline = Pipeline([
                ("tfidf", TfidfVectorizer(max_features=10000, ngram_range=(1,2), stop_words="english")),
                ("reg", Ridge(random_state=42)),
            ])
            X_train = [X_or_texts[i] for i in train_idx]
            X_test = [X_or_texts[i] for i in test_idx]
        else:
            pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("reg", Ridge(random_state=42)),
            ])
            X_train = X_or_texts[train_idx]
            X_test = X_or_texts[test_idx]

        y_train = y[train_idx]
        y_test = y[test_idx]

        pipeline.fit(X_train, y_train)
        oof_preds[test_idx] = pipeline.predict(X_test)
        oof_true[test_idx] = y_test

    return r2_score(oof_true, oof_preds)


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `src` importable
        
    from src.data import binary_label, findings_texts, study_table, usable_findings,top_k_classes
    from src.embeddings import embedding_path, load_embeddings
    
    studies = study_table()
    mask_usable = usable_findings(studies, min_words=3)

    X_embed = load_embeddings(embedding_path("findings"), studies)
    texts = findings_texts(studies)

    # Target 1: Health System (3 classes) -> Balanced Accuracy
    print("--- Evaluating Health System ---")
    hs = studies["health_system"]
    mask = mask_usable & hs.notna()
    sub_studies = studies[mask].reset_index(drop=True)
    y_hs = sub_studies["health_system"].values
    groups = sub_studies["patient_id"].values
    valid_idx = studies.index[mask].values

    acc_embed = evaluate_class_probe(
        X_embed[valid_idx], y_hs, groups, is_text=False
    )
    acc_tfidf = evaluate_class_probe(
        [texts[i] for i in valid_idx], y_hs, groups, is_text=True
    )
    print(
        f"Health System [3 classes, chance ~0.33] -> BioClinicalBERT Acc:"
        f" {acc_embed:.4f} | TF-IDF Acc: {acc_tfidf:.4f}"
    )

    # Target 2: Manufacturer (Top 6 brands) -> Balanced Accuracy
    print("\n--- Evaluating Manufacturer ---")
    mfg = top_k_classes(studies["manufacturer"], 6)
    mask = mask_usable & mfg.notna()
    sub_studies = studies[mask].reset_index(drop=True)
    y_mfg = sub_studies["manufacturer"].values
    groups = sub_studies["patient_id"].values
    valid_idx = studies.index[mask].values

    acc_embed_m = evaluate_class_probe(
        X_embed[valid_idx], y_mfg, groups, is_text=False
    )
    acc_tfidf_m = evaluate_class_probe(
        [texts[i] for i in valid_idx], y_mfg, groups, is_text=True
    )
    print(
        f"Manufacturer [6 classes, chance ~0.17] -> BioClinicalBERT Acc:"
        f" {acc_embed_m:.4f} | TF-IDF Acc: {acc_tfidf_m:.4f}"
    )
    
    # Target 3: Institution (Top 10 sites) -> Balanced Accuracy
    print("\n--- Evaluating Institution ---")
    instg = top_k_classes(studies["institution"], 10)
    mask = mask_usable & instg.notna()
    sub_studies = studies[mask].reset_index(drop=True)
    y_mfg = sub_studies["institution"].values
    groups = sub_studies["patient_id"].values
    valid_idx = studies.index[mask].values

    acc_embed_m = evaluate_class_probe(
        X_embed[valid_idx], y_mfg, groups, is_text=False
    )
    acc_tfidf_m = evaluate_class_probe(
        [texts[i] for i in valid_idx], y_mfg, groups, is_text=True
    )
    print(
        f"Institution [10 classes, chance ~0.10] -> BioClinicalBERT Acc:"
        f" {acc_embed_m:.4f} | TF-IDF Acc: {acc_tfidf_m:.4f}"
    )

    # Target 4: Report Length (findings_words) -> Ridge Regression R^2
    print("\n--- Evaluating Report Length ---")
    mask = mask_usable & studies["findings_words"].notna()
    sub_studies = studies[mask].reset_index(drop=True)
    y_len = sub_studies["findings_words"].values
    groups = sub_studies["patient_id"].values
    valid_idx = studies.index[mask].values

    r2_embed = evaluate_regression_probe(
        X_embed[valid_idx], y_len, groups, is_text=False
    )
    r2_tfidf = evaluate_regression_probe(
        [texts[i] for i in valid_idx], y_len, groups, is_text=True
    )
    print(
        f"Report Length [Continuous, chance 0.0] -> BioClinicalBERT R^2:"
        f" {r2_embed:.4f} | TF-IDF R^2: {r2_tfidf:.4f}"
    )


if __name__ == "__main__":
    main()