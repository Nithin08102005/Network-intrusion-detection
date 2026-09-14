"""
data_loader.py
==============
Data loading and sampling utilities for the CICIDS-2017 dataset.

Supports:
1. Downloading a representative stratified sample (~50k-100k rows) for instant local development.
2. Loading full multi-file datasets locally or from Kaggle environments.
"""

import os
import sys
import pandas as pd
import numpy as np

# Ensure Windows console supports UTF-8 characters without dying
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Verified fast parquet mirror of CICIDS-2017 (MachineLearningCVE standard schema)
BASE_HF_URL = "https://huggingface.co/datasets/bvsam/cic-ids-2017/resolve/main/machine_learning/"

FILES = {
    "PortScan": "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv.parquet",
    "DDoS": "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv.parquet",
    "DoS": "Wednesday-workingHours.pcap_ISCX.csv.parquet",
    "WebAttacks": "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv.parquet"
}


def download_representative_sample(
    output_path: str = "data/sample_cicids.parquet",
    target_samples_per_file: int = 25000,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Downloads a lightweight, diverse subset of traffic from key attack days + benign traffic.
    Creates a ~100k row dataset covering PortScan, DDoS, DoS, Web Attacks, and Benign traffic.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sampled_dfs = []

    print("[*] Building representative CICIDS-2017 dataset...")
    for category, filename in FILES.items():
        url = BASE_HF_URL + filename
        print(f"  -> Fetching sample from {category} ({filename})...")
        try:
            df = pd.read_parquet(url)
            # Clean column names (strip whitespace)
            df.columns = df.columns.str.strip()

            # Stratified sample if possible, otherwise regular sample
            if len(df) > target_samples_per_file:
                # Group by label to keep rare attack classes
                df_sampled = df.groupby('Label', group_keys=False).apply(
                    lambda x: x.sample(
                        n=min(len(x), max(500, int(target_samples_per_file * (len(x) / len(df))))),
                        random_state=random_state
                    ),
                    include_groups=False
                )
                # Restore Label column if dropped by include_groups=False
                if 'Label' not in df_sampled.columns:
                    df_sampled['Label'] = df.loc[df_sampled.index, 'Label']
            else:
                df_sampled = df

            sampled_dfs.append(df_sampled)
            print(f"     Loaded {len(df_sampled):,} rows. Classes: {dict(df_sampled['Label'].value_counts())}")
        except Exception as e:
            print(f"     Warning: Could not fetch {filename}: {e}")

    if not sampled_dfs:
        raise RuntimeError("No data could be loaded. Check your internet connection.")

    combined_df = pd.concat(sampled_dfs, ignore_index=True)
    # Deduplicate in case overlapping benign flows exist
    combined_df.drop_duplicates(inplace=True)
    
    # Save as parquet for speed and small file size
    combined_df.to_parquet(output_path, index=False)
    print(f"\n[+] Saved representative dataset to '{output_path}'")
    print(f"[+] Total Shape: {combined_df.shape}")
    print(f"[+] Class distribution:\n{combined_df['Label'].value_counts()}\n")

    return combined_df


def load_dataset(data_path: str) -> pd.DataFrame:
    """
    Loads dataset from parquet or csv, automatically stripping column whitespace.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file '{data_path}' not found. Run download_representative_sample() first.")

    if data_path.endswith('.parquet'):
        df = pd.read_parquet(data_path)
    elif data_path.endswith('.csv'):
        df = pd.read_csv(data_path, low_memory=False)
    else:
        raise ValueError(f"Unsupported file format: {data_path}")

    df.columns = df.columns.str.strip()
    return df


if __name__ == "__main__":
    download_representative_sample()
