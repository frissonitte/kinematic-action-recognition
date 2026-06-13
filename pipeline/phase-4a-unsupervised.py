import os
import time
import tracemalloc

import dask.dataframe as dd
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (classification_report, confusion_matrix,
                              f1_score)
from sklearn.preprocessing import StandardScaler

PLOTS_DIR = 'plots'

N_CLUSTERS  = 15   # 15 distinct action classes in dataset
N_PCA_COMPS = 50   # 528 features → 50 principal components
               # High-dimensional spaces make all distances converge (curse of
               # dimensionality), PCA removes noise and restores meaningful geometry


def hungarian_match(true_labels, cluster_labels, n_clusters):
    """Map cluster IDs to ground-truth class IDs via the Hungarian algorithm.

    K-Means assigns arbitrary cluster numbers (0-14). Hungarian algorithm finds
    the optimal 1-to-1 assignment that maximises overlap with true labels,
    letting us compute a meaningful F1-score on unsupervised output.
    """
    unique_true = sorted(np.unique(true_labels))          # [0,1,3,4,...,15]
    label_to_idx = {l: i for i, l in enumerate(unique_true)}
    true_idx = np.array([label_to_idx[l] for l in true_labels])

    # Build cost matrix: rows = clusters, cols = true class indices
    cost = np.zeros((n_clusters, n_clusters), dtype=np.int64)
    for c in range(n_clusters):
        mask = cluster_labels == c
        for t_idx in range(n_clusters):
            cost[c, t_idx] = np.sum(true_idx[mask] == t_idx)

    # linear_sum_assignment minimises cost — negate to maximise overlap
    row_ind, col_ind = linear_sum_assignment(-cost)

    # Build cluster_id → true_label mapping
    cluster_to_label = {}
    for r, c in zip(row_ind, col_ind):
        cluster_to_label[r] = unique_true[c]

    return cluster_to_label


def main():
    print("Loading feature matrix from Parquet...")
    feat_df = dd.read_parquet('data/main_data_features_parquet', engine='pyarrow').compute()

    y_true = feat_df['LABEL'].values.astype(int)
    X = feat_df.drop(columns=['LABEL']).values.astype(np.float32)
    print(f"Feature matrix: {X.shape[0]} rows x {X.shape[1]} features")

    # K-Means is distance-based — must normalise or std-heavy channels dominate
    print("Normalising features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA before K-Means: reduces 528 → 50 dims, drops noise, fixes distance geometry
    print(f"Applying PCA ({N_PCA_COMPS} components)...")
    pca = PCA(n_components=N_PCA_COMPS, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    explained = pca.explained_variance_ratio_.cumsum()[-1]
    print(f"Explained variance retained: {explained:.1%}")

    # MiniBatchKMeans: same result as KMeans, fraction of the RAM and time
    print(f"Fitting MiniBatchKMeans (k={N_CLUSTERS})...")
    tracemalloc.start()
    t0 = time.perf_counter()

    kmeans = MiniBatchKMeans(
        n_clusters=N_CLUSTERS,
        random_state=42,
        batch_size=4096,
        n_init=10,
        max_iter=300,
        verbose=0
    )
    kmeans.fit(X_pca)

    train_time = time.perf_counter() - t0
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"Training time : {train_time:.1f}s")
    print(f"Peak RAM usage: {peak_mem / 1e6:.1f} MB")

    # --- Inference latency on 1000-sample batch ---
    t_inf = time.perf_counter()
    _ = kmeans.predict(X_pca[:1000])
    inf_latency_ms = (time.perf_counter() - t_inf) * 1000
    print(f"Inference latency (1000 samples): {inf_latency_ms:.2f} ms")

    raw_clusters = kmeans.labels_

    # --- Persist model and scaler ---
    joblib.dump(pca,    'models/pca.pkl')

    # --- Hungarian matching ---
    print("\nRunning Hungarian algorithm for cluster → label assignment...")
    cluster_to_label = hungarian_match(y_true, raw_clusters, N_CLUSTERS)
    print("Cluster → Label mapping:", cluster_to_label)

    y_pred = np.array([cluster_to_label[c] for c in raw_clusters])

    # --- Evaluation ---
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    weighted_f1 = f1_score(y_true, y_pred, average='weighted')

    print(f"\n{'='*50}")
    print(f"Macro F1-Score    : {macro_f1:.4f}")
    print(f"Weighted F1-Score : {weighted_f1:.4f}")
    print(f"{'='*50}")
    print("\nPer-class report:")
    print(classification_report(y_true, y_pred))

    # --- Visualisations ---
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # PCA explained variance curve
    cumvar = np.cumsum(pca.explained_variance_ratio_) * 100
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(cumvar) + 1), cumvar, marker='o', markersize=3, color='steelblue')
    ax.axhline(cumvar[-1], color='crimson', linestyle='--',
               label=f'{N_PCA_COMPS} components → {cumvar[-1]:.1f}% variance')
    ax.set_xlabel('Number of PCA Components')
    ax.set_ylabel('Cumulative Explained Variance (%)')
    ax.set_title('PCA Explained Variance — 528 features → 50 components')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'phase4a_pca_variance.png'), dpi=150)
    plt.close(fig)
    print(f"Saved: {PLOTS_DIR}/phase4a_pca_variance.png")

    # Confusion matrix heatmap
    cm = confusion_matrix(y_true, y_pred)
    unique_labels = sorted(np.unique(y_true))
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=unique_labels, yticklabels=unique_labels, ax=ax)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title(f'Confusion Matrix — MiniBatchKMeans (k={N_CLUSTERS})\n'
                 f'Macro F1={macro_f1:.3f}  Weighted F1={weighted_f1:.3f}')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'phase4a_confusion_matrix.png'), dpi=150)
    plt.close(fig)
    print(f"Saved: {PLOTS_DIR}/phase4a_confusion_matrix.png")

    print("Confusion matrix (raw):")
    print(cm)

    joblib.dump(kmeans, 'models/kmeans_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    np.save('models/cluster_to_label.npy', cluster_to_label)
    print("\nModel saved to models/")

    input("Press Enter to exit...")


if __name__ == '__main__':
    import os
    os.makedirs('models', exist_ok=True)
    main()
