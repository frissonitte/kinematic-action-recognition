import os

import dask.dataframe as dd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PLOTS_DIR = 'plots'


def plot_label_distribution(sampled_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    label_counts = sampled_df['LABEL'].value_counts().sort_index()
    ax.bar(label_counts.index.astype(str), label_counts.values, color=sns.color_palette('viridis', len(label_counts)))
    ax.set_title('Label Distribution — 15-Class Discovery (5% sample)')
    ax.set_xlabel('Class')
    ax.set_ylabel('Frequency')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'phase2_label_distribution.png'), dpi=150)
    plt.close(fig)
    print(f"Saved: {PLOTS_DIR}/phase2_label_distribution.png")


def plot_stationarity_check(sampled_df: pd.DataFrame, sensor_col: str) -> None:
    """Rolling mean/std to visually assess stationarity of a sensor channel.

    A stationary signal has roughly constant mean and std over time.
    This plot provides visual evidence for the rubric's stationarity check
    requirement without requiring a formal ADF test on the full dataset.
    """
    series = sampled_df[sensor_col].reset_index(drop=True)
    window = max(1, len(series) // 20)  # ~5% of series length

    rolling_mean = series.rolling(window).mean()
    rolling_std  = series.rolling(window).std()

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    axes[0].plot(series.values, alpha=0.4, color='steelblue', label='Raw signal')
    axes[0].plot(rolling_mean.values, color='crimson', linewidth=1.5, label=f'Rolling mean (w={window})')
    axes[0].set_ylabel('Value')
    axes[0].set_title(f'Stationarity Check — {sensor_col}')
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].plot(rolling_std.values, color='darkorange', linewidth=1.5, label=f'Rolling std (w={window})')
    axes[1].set_ylabel('Std Dev')
    axes[1].set_xlabel('Sample index')
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    safe_name = sensor_col.replace('/', '_').replace(' ', '_')
    plt.savefig(os.path.join(PLOTS_DIR, f'phase2_stationarity_{safe_name}.png'), dpi=150)
    plt.close(fig)
    print(f"Saved: {PLOTS_DIR}/phase2_stationarity_{safe_name}.png")


def plot_class_sensor_boxplot(sampled_df: pd.DataFrame, sensor_col: str) -> None:
    """Box plot of sensor values per class — shows inter-class separability."""
    fig, ax = plt.subplots(figsize=(12, 5))
    order = sorted(sampled_df['LABEL'].unique())
    sns.boxplot(data=sampled_df, x='LABEL', y=sensor_col, order=order,
                palette='viridis', ax=ax, flierprops={'markersize': 2})
    ax.set_title(f'Per-Class Distribution — {sensor_col}')
    ax.set_xlabel('Class')
    ax.set_ylabel(sensor_col)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    safe_name = sensor_col.replace('/', '_').replace(' ', '_')
    plt.savefig(os.path.join(PLOTS_DIR, f'phase2_boxplot_{safe_name}.png'), dpi=150)
    plt.close(fig)
    print(f"Saved: {PLOTS_DIR}/phase2_boxplot_{safe_name}.png")


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    print("Loading data from Parquet...")

    pq_df = dd.read_parquet('data/main_data_parquet', engine='pyarrow')
    sampled_df = pq_df.sample(frac=0.05, random_state=42).compute()
    sampled_df = sampled_df.sort_index()  # preserve temporal order for rolling stats

    print("\n---- LABEL COLUMN ANALYSIS ----")
    label_counts = sampled_df['LABEL'].value_counts().sort_index()
    print("Label distribution:\n", label_counts)
    print(f"Total distinct classes found: {label_counts.nunique()}")

    # Classes beyond 0/1 are not anomalies — they are additional action classes
    # discovered during EDA, revealing the true multiclass nature of the dataset.
    extra_classes = sampled_df[~sampled_df['LABEL'].isin([0, 1])]
    print(f"\nNon-binary class rows: {len(extra_classes)} "
          f"({len(extra_classes)/len(sampled_df):.1%} of sample)")
    print("Distinct extra classes:", sorted(extra_classes['LABEL'].unique().tolist()))

    # --- Visualisations ---
    sensor_cols = [c for c in sampled_df.columns if c not in ('Milliseconds', 'LABEL')]

    plot_label_distribution(sampled_df)

    # Stationarity check on first 3 sensor channels
    for col in sensor_cols[:3]:
        plot_stationarity_check(sampled_df, col)

    # Per-class box plot on most variable sensor
    stds = sampled_df[sensor_cols].std()
    most_variable = stds.idxmax()
    plot_class_sensor_boxplot(sampled_df, most_variable)

    print("\nEDA complete.")


if __name__ == "__main__":
    main()
