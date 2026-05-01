import dask.dataframe as dd
from dask.distributed import Client
import pandas as pd


def clean_and_cast_partition(df_part):
    """Applied per 64MB chunk via map_partitions.

    CSV has embedded repeated header rows (sensor logger restarts mid-file).
    Reading as dtype='object' then coercing to numeric turns those string rows
    into NaN rows, which dropna() removes cleanly.
    """
    # Strip whitespace from column names — source CSV has trailing spaces (e.g. 'Label ')
    df_part = df_part.rename(columns=lambda x: str(x).strip())

    target_col = [c for c in df_part.columns if c.lower() == 'label'][0]
    sensor_cols = [c for c in df_part.columns if c != target_col]

    # errors='coerce': non-numeric strings (header rows) become NaN instead of raising
    for col in sensor_cols:
        df_part[col] = pd.to_numeric(df_part[col], errors='coerce').astype('float32')

    df_part[target_col] = pd.to_numeric(df_part[target_col], errors='coerce')

    # Drops any row where coercion failed (i.e. embedded header rows)
    df_part = df_part.dropna()

    df_part[target_col] = df_part[target_col].astype('int8')

    return df_part


def main():
    # 8 workers x 3GB = 24GB ceiling; leaves ~8GB headroom on 32GB system
    client = Client(n_workers=8, threads_per_worker=2, memory_limit='3GB')
    print(f"Dask Dashboard: {client.dashboard_link}")

    try:
        # dtype='object' prevents cast failure at parse time — clean_and_cast_partition handles types
        # on_bad_lines='skip' drops malformed rows (wrong column count) without crashing
        df = dd.read_csv(
            'data/main_data.csv',
            blocksize='64MB',
            dtype='object',
            on_bad_lines='skip',
            engine='c'
        )

        df_cleaned = df.map_partitions(clean_and_cast_partition)

        print("Converting to Parquet...")
        # overwrite=True: safe to re-run without manually deleting the output dir
        df_cleaned.to_parquet(
            'data/main_data_parquet',
            engine='pyarrow',
            write_index=False,
            overwrite=True
        )
        print("Parquet saving completed!")

        pq_df = dd.read_parquet('data/main_data_parquet', engine='pyarrow')

        print("Calculating population statistics...")
        population_stats = pq_df.describe().compute()
        print(population_stats)

        # 5% sample pulled into memory for EDA/visualization — full data stays out-of-core
        print("Taking sample for visualization...")
        sampled_df = pq_df.sample(frac=0.05, random_state=42).compute()
        print(f"Sample shape: {sampled_df.shape}")

    finally:
        client.close()

    input("Press Enter to exit...")


if __name__ == '__main__':
    main()
