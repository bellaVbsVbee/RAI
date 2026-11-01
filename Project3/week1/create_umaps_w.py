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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `src` importable

from src.data import study_table, findings_texts, usable_findings, binary_label
from src.embeddings import embedding_path, load_embeddings

studies = study_table()
X = load_embeddings(embedding_path("findings"), studies)   # shape (10000, 768)

def inspect_feature(feature, n = 10):
    print(studies[feature].unique())
    print(f"Number of elements: {len(studies[feature])}")
    print(f"Number of unique elements: {len(studies[feature].unique())}")
    # Counts of n most popular:
    print(studies[feature].value_counts())
    
def create_umaps(data, n_neighbors=15, verbose=False, random_state=42):
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        #random_state=random_state
    )
    
    reducer.fit(data)
    
    embedding = reducer.transform(data)
    assert(np.all(embedding == reducer.embedding_))
    
    if verbose:
        print(f"Shape of input data: {data.shape}")
        print(f"Shape of umap embeddings: {embedding.shape}")
        
    return embedding
    
    
def plot_umap_embeddings(embedding, targets, target_label):
    plt.scatter(embedding[:, 0], embedding[:, 1], c=targets, cmap='Spectral', s=5)
    plt.gca().set_aspect('equal', 'datalim')
    plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
    plt.title(f'UMAP projection of the {target_label} feature', fontsize=24)
    plt.savefig("week1/figures/umap")
    

# inspect_feature("ethnic_group")
# print(studies.columns)
target_label = "health_system"
umap_embedding = create_umaps(X, n_neighbors=15, verbose=True, random_state=42)
plot_umap_embeddings(umap_embedding, studies[target_label], target_label)