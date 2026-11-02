import os
import sys
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

def inspect_feature(study_feature, n = 10):
    print(study_feature.unique())
    print(f"Number of elements: {len(study_feature)}")
    print(f"Number of unique elements: {len(study_feature.unique())}")
    # Counts of n most popular:
    print(study_feature.value_counts())
    
    
def main():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `src` importable
    
    from src.data import study_table, findings_texts, usable_findings, binary_label
    from src.embeddings import embedding_path, load_embeddings

    studies = study_table()
    X = load_embeddings(embedding_path("findings"), studies)   # shape (10000, 768)
    
    feature = "health_system"
    inspect_feature(studies[feature])

if __name__ == "__main__":
    main()