import os
import time
import tracemalloc

import dask.dataframe as dd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from river import drift, forest, metrics, preprocessing

PLOTS_DIR = 'plots'

# Simulate concept drift by injecting label noise after this fraction of stream
DRIFT_INJECT_AT = 0.5   # halfway through the stream
DRIFT_NOISE_RATE = 0.30  # flip 30% of labels after drift point


def simulate_stream(feat_df: pd.DataFrame, inject_drift: bool = True):
    """Yield (x_dict, y) one window at a time, optionally injecting label noise.

    In real Industry 5.0 deployment the stream is live sensor windows.
    Here we replay the feature matrix in row order — temporal order preserved,
    no shuffling — to simulate a production data feed.

    Drift injection: after DRIFT_INJECT_AT fraction, randomly flip DRIFT_NOISE_RATE
    of labels. This simulates a process change (e.g. operator fatigue, line reconfiguration)
    and lets us verify that ADWIN detects the distribution shift.
    """
    n = len(feat_df)
    drift_start = int(n * DRIFT_INJECT_AT)
    rng = np.random.default_rng(42)

    for i, row in enumerate(feat_df.itertuples(index=False)):
        label = int(row.LABEL)
        if inject_drift and i >= drift_start:
            if rng.random() < DRIFT_NOISE_RATE:
                # Flip to a random different class — simulates concept drift
                classes = [c for c in range(16) if c != label and c != 2]
                label = int(rng.choice(classes))

        x = {col: getattr(row, col) for col in feat_df.columns if col != 'LABEL'}
        yield x, label


def main():
    print("Loading feature matrix from Parquet...")
    feat_df = dd.read_parquet('data/main_data_features_parquet', engine='pyarrow').compute()
    print(f"Stream length: {len(feat_df)} windows")

    # --- Model: Adaptive Random Forest ---
    # ARF replaces trees that degrade under drift with freshly trained ones.
    # Chosen over Hoeffding Tree because our 15-class problem needs ensemble power.
    model = preprocessing.StandardScaler() | forest.ARFClassifier(
        n_models=10,
        seed=42
    )

    # --- Drift detector: ADWIN ---
    # ADWIN maintains a sliding window of error rates; shrinks window when
    # a statistically significant change in error rate is detected.
    drift_detector = drift.ADWIN(delta=0.002)

    # --- Metrics (prequential / test-then-train) ---
    accuracy     = metrics.Accuracy()
    macro_f1     = metrics.MacroF1()
    kappa        = metrics.CohenKappa()

    drift_points   = []
    accuracy_log   = []          # (window_index, rolling_accuracy)
    log_every      = 1000        # print progress every N windows

    print("Starting prequential (test-then-train) stream simulation...")
    print(f"Drift injected at window {int(len(feat_df) * DRIFT_INJECT_AT)} "
          f"(noise rate: {DRIFT_NOISE_RATE:.0%})\n")

    tracemalloc.start()
    t0 = time.perf_counter()

    for i, (x, y_true) in enumerate(simulate_stream(feat_df, inject_drift=True)):
        # 1. Predict before training (prequential evaluation — unbiased estimate)
        y_pred = model.predict_one(x)

        # 2. Update metrics
        if y_pred is not None:
            accuracy.update(y_true, y_pred)
            macro_f1.update(y_true, y_pred)
            kappa.update(y_true, y_pred)

            # 3. Feed error signal to drift detector (1 = error, 0 = correct)
            drift_detector.update(int(y_pred != y_true))
            if drift_detector.drift_detected:
                drift_points.append(i)
                print(f"  [DRIFT DETECTED] window={i:,}  rolling_accuracy={accuracy.get():.4f}")

        # 4. Train on this sample
        model.learn_one(x, y_true)

        if i % log_every == 0 and i > 0:
            accuracy_log.append((i, accuracy.get()))
            print(f"  window={i:6,}  acc={accuracy.get():.4f}  "
                  f"macro_f1={macro_f1.get():.4f}  kappa={kappa.get():.4f}")

    elapsed   = time.perf_counter() - t0
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"\n{'='*55}")
    print(f"Final Accuracy    : {accuracy.get():.4f}")
    print(f"Final Macro F1    : {macro_f1.get():.4f}")
    print(f"Final Cohen Kappa : {kappa.get():.4f}")
    print(f"Drift events      : {len(drift_points)} at windows {drift_points[:5]}...")
    print(f"Total stream time : {elapsed:.1f}s  ({len(feat_df)/elapsed:.0f} windows/sec)")
    print(f"Peak RAM usage    : {peak_mem / 1e6:.1f} MB")
    print(f"{'='*55}")

    # --- Accuracy over time log ---
    print("\nAccuracy progression (every 1000 windows):")
    for idx, acc in accuracy_log:
        bar = '#' * int(acc * 30)
        print(f"  {idx:6,}  {acc:.4f}  |{bar}")

    # --- Accuracy over time plot ---
    os.makedirs(PLOTS_DIR, exist_ok=True)
    if accuracy_log:
        xs, ys = zip(*accuracy_log)
        drift_inject_window = int(len(feat_df) * DRIFT_INJECT_AT)

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(xs, ys, color='steelblue', linewidth=1.5, label='Rolling accuracy')
        ax.axvline(drift_inject_window, color='crimson', linestyle='--',
                   label=f'Drift injected (window {drift_inject_window:,})')
        for dp in drift_points:
            ax.axvline(dp, color='orange', linestyle=':', alpha=0.7, linewidth=0.8)
        if drift_points:
            ax.axvline(drift_points[0], color='orange', linestyle=':', alpha=0.7,
                       linewidth=0.8, label='ADWIN drift detected')
        ax.set_xlabel('Window index')
        ax.set_ylabel('Prequential Accuracy')
        ax.set_title('Streaming Accuracy over Time — ARF + ADWIN Drift Detection')
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, 'phase5_accuracy_over_time.png'), dpi=150)
        plt.close(fig)
        print(f"Saved: {PLOTS_DIR}/phase5_accuracy_over_time.png")

    input("\nPress Enter to exit...")


if __name__ == '__main__':
    main()
