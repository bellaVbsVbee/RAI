# week1/run_probes.py
from pathlib import Path
import numpy as np
import pandas as pd
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler




def evaluate_probe(X_or_texts, y, groups, is_text: bool = False) -> float:
    """Run 5-fold StratifiedGroupKFold cross-validation and return out-of-fold AUROC."""
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    oof_preds = np.zeros(len(y))
    oof_true = np.zeros(len(y))

    for train_idx, test_idx in cv.split(X_or_texts, y, groups):
        if is_text:
            # Pipeline fits TfidfVectorizer strictly on the training fold to prevent data leakage
            pipeline = Pipeline([
                ("tfidf", TfidfVectorizer(max_features=10000, ngram_range=(1,2), stop_words="english")),
                ("clf", LogisticRegression(max_iter=1000, random_state=42)),
            ])
            X_train = [X_or_texts[i] for i in train_idx]
            X_test = [X_or_texts[i] for i in test_idx]
        else:
            # Pipeline standardizes embedding features before logistic regression
            pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, random_state=42)),
            ])
            X_train = X_or_texts[train_idx]
            X_test = X_or_texts[test_idx]

        y_train = y[train_idx]
        y_test = y[test_idx]

        pipeline.fit(X_train, y_train)

        # Collect out-of-fold predicted probabilities for the positive class
        preds = pipeline.predict_proba(X_test)[:, 1]
        oof_preds[test_idx] = preds
        oof_true[test_idx] = y_test

    return roc_auc_score(oof_true, oof_preds)


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `src` importable
    
    from src.data import binary_label, findings_texts, study_table, usable_findings
    from src.embeddings import embedding_path, load_embeddings
    print("Loading studies and metadata...")
    studies = study_table()

    # Filter for usable findings (at least 3 words)
    mask_usable = usable_findings(studies, min_words=3)

    observations = ["edema", "pleural_effusion"]

    print("Loading BioClinicalBERT embeddings...")
    X_embed = load_embeddings(embedding_path("findings"), studies)
    texts = findings_texts(studies)

    for obs in observations:
        print(f"\n==============================")
        print(f" Evaluating Target: {obs}")
        print(f"==============================")

        # Get binary labels (1.0, 0.0, or NaN for uncertain/blank)
        y_raw = binary_label(studies, obs)

        # Combine usable findings mask with non-NaN label mask
        mask = mask_usable & y_raw.notna()

        subset_studies = studies[mask].reset_index(drop=True)
        y = y_raw[mask].values
        groups = subset_studies["patient_id"].values

        # Subset embeddings and texts for valid samples
        valid_indices = studies.index[mask].values
        X_sub_embed = X_embed[valid_indices]
        sub_texts = [texts[i] for i in valid_indices]

        print(
            f"Valid samples: {len(y)} across"
            f" {pd.Series(groups).nunique():,} patients."
        )

        # 1. BioClinicalBERT Representation Probe
        auc_embed = evaluate_probe(X_sub_embed, y, groups, is_text=False)
        print(f"-> BioClinicalBERT Embedding AUROC: {auc_embed:.4f}")

        # 2. TF-IDF Baseline Probe
        auc_tfidf = evaluate_probe(sub_texts, y, groups, is_text=True)
        print(f"-> TF-IDF Baseline AUROC:           {auc_tfidf:.4f}")


if __name__ == "__main__":
    main()