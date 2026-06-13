import dask.dataframe as dd
import pandas as pd
import numpy as np
from dask.distributed import Client

WINDOW_SIZE = 200   # 200 rows x 5ms = 1 second per window
STEP_SIZE   = 100   # 50% overlap — doubles sample count, reduces boundary artifacts


def extract_window_features(df: pd.DataFrame) -> pd.DataFrame:
    """Slide a window over one partition and extract statistical features.

    Rolling-window approach: each 1-second window becomes one row with
    mean/std/max/min per sensor channel. std captures movement intensity
    (high std = rapid motion), which is the most discriminative feature
    for action recognition tasks.
    """
    sensor_cols = [c for c in df.columns if c not in ('Milliseconds', 'LABEL')]

    records = []
    n = len(df)

    for start in range(0, n - WINDOW_SIZE + 1, STEP_SIZE):
        window = df.iloc[start : start + WINDOW_SIZE]

        # Majority vote: assign the most frequent label in this window
        label = int(window['LABEL'].mode()[0])

        feats = {'LABEL': label}
        for col in sensor_cols:
            vals = window[col].values
            feats[f'{col}_mean'] = np.mean(vals)
            feats[f'{col}_std']  = np.std(vals)
            feats[f'{col}_max']  = np.max(vals)
            feats[f'{col}_min']  = np.min(vals)

        records.append(feats)

    return pd.DataFrame(records)


def main():
    # 8 workers x 3GB = 24GB ceiling; leaves ~8GB headroom on 32GB system
    client = Client(n_workers=8, threads_per_worker=2, memory_limit='3GB')
    print(f"Dask Dashboard: {client.dashboard_link}")

    try:
        print("Loading Parquet...")
        pq_df = dd.read_parquet('data/main_data_parquet', engine='pyarrow')

        # Build meta so Dask knows output schema before compute
        sample_cols = [c for c in pq_df.columns if c not in ('Milliseconds', 'LABEL')]
        meta_cols = {'LABEL': 'int8'}
        for col in sample_cols:
            meta_cols[f'{col}_mean'] = 'float32'
            meta_cols[f'{col}_std']  = 'float32'
            meta_cols[f'{col}_max']  = 'float32'
            meta_cols[f'{col}_min']  = 'float32'
        meta = pd.DataFrame({k: pd.Series(dtype=v) for k, v in meta_cols.items()})

        print(f"Extracting features (window={WINDOW_SIZE} rows = 1s, step={STEP_SIZE} rows)...")
        features_df = pq_df.map_partitions(extract_window_features, meta=meta)

        print("Saving feature matrix to Parquet...")
        features_df.to_parquet(
            'data/main_data_features_parquet',
            engine='pyarrow',
            write_index=False,
            overwrite=True
        )
        print("Feature extraction complete!")

        feat_pq = dd.read_parquet('data/main_data_features_parquet', engine='pyarrow')
        print(f"Feature matrix shape: {feat_pq.shape[1]} columns")
        print(f"Row count: {len(feat_pq)}")
        print(f"\nLabel distribution:\n{feat_pq['LABEL'].value_counts().compute().sort_index()}")
        print(f"\nSample features:\n{feat_pq.head(3)}")

    finally:
        client.close()

    input("Press Enter to exit...")


if __name__ == '__main__':
    main()
