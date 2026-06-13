import os
import time
import tracemalloc

import dask.dataframe as dd
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import TimeSeriesSplit

PLOTS_DIR = 'plots'

# Rubric: "Classical supervised model" — only binary labels 0 and 1
# Remaining classes (3-15) are reserved for the unsupervised task
BINARY_LABELS = [0, 1]


def main():
    print("Loading feature matrix from Parquet...")
    feat_df = dd.read_parquet('data/main_data_features_parquet', engine='pyarrow').compute()

    # Filter to binary subset — 0 vs 1 only
    binary_df = feat_df[feat_df['LABEL'].isin(BINARY_LABELS)].copy()
    print(f"Binary subset: {len(binary_df)} rows  (0: {(binary_df['LABEL']==0).sum()}, 1: {(binary_df['LABEL']==1).sum()})")

    y = binary_df['LABEL'].values.astype(np.int8)
    X = binary_df.drop(columns=['LABEL']).values.astype(np.float32)
    print(f"Feature matrix: {X.shape[0]} rows x {X.shape[1]} features")

    # Systematic every-5th-window test split.
    # - Random split: leakage (overlapping windows share raw rows, twin ends up in both sets)
    # - Temporal split: degenerate (classes 0 and 1 are in different time segments)
    # - Every-5th: preserves class balance, adjacent windows still separated by 4 windows
    test_mask  = np.zeros(len(X), dtype=bool)
    test_mask[::5] = True
    X_train, X_test = X[~test_mask], X[test_mask]
    y_train, y_test = y[~test_mask], y[test_mask]
    print(f"Train: {len(X_train)}  Test: {len(X_test)}")

    # RandomForest: tree-based, immune to curse of dimensionality (no distance calc)
    # class_weight='balanced': compensates for class 1 having ~2x fewer samples
    print("Training RandomForestClassifier...")
    tracemalloc.start()
    t0 = time.perf_counter()

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        class_weight='balanced',
        n_jobs=-1,
        random_state=42
    )
    model.fit(X_train, y_train)

    train_time = time.perf_counter() - t0
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"Training time : {train_time:.1f}s")
    print(f"Peak RAM usage: {peak_mem / 1e6:.1f} MB")

    # --- Inference latency ---
    t_inf = time.perf_counter()
    y_pred = model.predict(X_test)
    inf_latency_ms = (time.perf_counter() - t_inf) * 1000
    print(f"Inference latency ({len(X_test)} samples): {inf_latency_ms:.2f} ms")

    # --- Evaluation ---
    macro_f1    = f1_score(y_test, y_pred, average='macro')
    weighted_f1 = f1_score(y_test, y_pred, average='weighted')

    print(f"\n{'='*50}")
    print(f"Macro F1-Score    : {macro_f1:.4f}")
    print(f"Weighted F1-Score : {weighted_f1:.4f}")
    print(f"{'='*50}")
    print("\nPer-class report:")
    print(classification_report(y_test, y_pred, target_names=['Class 0', 'Class 1']))

    # --- Top-20 most important features ---
    feature_names = binary_df.drop(columns=['LABEL']).columns.tolist()
    importances = pd.Series(model.feature_importances_, index=feature_names)
    top20 = importances.nlargest(20)
    print("\nTop 20 features by importance:")
    print(top20.to_string())

    # --- Visualisations ---
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # Feature importance bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    top20.sort_values().plot(kind='barh', ax=ax, color='steelblue')
    ax.set_xlabel('Importance')
    ax.set_title('Top 20 Feature Importances — RandomForest (Class 0 vs 1)')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'phase4b_feature_importance.png'), dpi=150)
    plt.close(fig)
    print(f"Saved: {PLOTS_DIR}/phase4b_feature_importance.png")

    # Confusion matrix heatmap
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Class 0', 'Class 1'],
                yticklabels=['Class 0', 'Class 1'], ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'Confusion Matrix — RandomForest\n'
                 f'Macro F1={macro_f1:.3f}  Weighted F1={weighted_f1:.3f}')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'phase4b_confusion_matrix.png'), dpi=150)
    plt.close(fig)
    print(f"Saved: {PLOTS_DIR}/phase4b_confusion_matrix.png")

    # --- Save model ---
    joblib.dump(model, 'models/rf_supervised.pkl')
    print("\nModel saved to models/rf_supervised.pkl")

    input("Press Enter to exit...")


if __name__ == '__main__':
    import os
    os.makedirs('models', exist_ok=True)
    main()
