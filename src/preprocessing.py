"""
preprocessing.py
================
Cleaning and preprocessing pipeline for CICIDS-2017 flows.

Handles:
1. Label sanitation & mapping to unified attack families.
2. Binary target generation (0: BENIGN, 1: ATTACK).
3. Infinite (Inf/-Inf) and NaN handling without data leakage.
4. Stratified splitting into Train & Test sets.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List
from sklearn.model_selection import train_test_split


ATTACK_FAMILY_MAP = {
    # Benign
    'BENIGN': 'BENIGN',
    
    # DDoS
    'DDoS': 'DDoS',
    
    # PortScan
    'PortScan': 'PortScan',
    
    # DoS Family
    'DoS Hulk': 'DoS',
    'DoS GoldenEye': 'DoS',
    'DoS slowloris': 'DoS',
    'DoS Slowhttptest': 'DoS',
    'Heartbleed': 'DoS',
    
    # Brute Force
    'FTP-Patator': 'Brute Force',
    'SSH-Patator': 'Brute Force',
    
    # Web Attacks (handling potential encoding quirks like )
    'Web Attack – Brute Force': 'Web Attack',
    'Web Attack – XSS': 'Web Attack',
    'Web Attack – Sql Injection': 'Web Attack',
    'Web Attack  Brute Force': 'Web Attack',
    'Web Attack  XSS': 'Web Attack',
    'Web Attack  Sql Injection': 'Web Attack',
    
    # Botnet & Infiltration
    'Bot': 'Botnet',
    'Infiltration': 'Infiltration'
}


def clean_labels(df: pd.DataFrame, label_col: str = 'Label') -> pd.DataFrame:
    """
    Standardizes label strings, maps to unified attack families,
    and adds a binary target column 'is_attack'.
    """
    df = df.copy()
    # Normalize strings
    labels = df[label_col].astype(str).str.strip()
    
    # Handle known variations
    def map_family(val: str) -> str:
        if val in ATTACK_FAMILY_MAP:
            return ATTACK_FAMILY_MAP[val]
        # Fuzzy match for web attack if encoding differs
        if 'Web Attack' in val:
            return 'Web Attack'
        if 'Patator' in val:
            return 'Brute Force'
        if 'DoS' in val:
            return 'DoS'
        return val

    df['Attack_Family'] = labels.apply(map_family)
    df['is_attack'] = (df['Attack_Family'] != 'BENIGN').astype(int)
    return df


def clean_inf_and_nan(
    df: pd.DataFrame,
    feature_cols: List[str],
    impute_values: Dict[str, float] = None
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Replaces Inf/-Inf with NaN and imputes NaNs using column medians.
    If impute_values is None, calculates medians (training mode).
    Otherwise, applies precomputed medians (test/inference mode).
    """
    df = df.copy()
    
    # Convert features to numeric
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Replace infinite values with NaN
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    
    if impute_values is None:
        impute_values = df[feature_cols].median().to_dict()
    
    # Fill remaining NaNs with medians (or 0 if entire column is NaN)
    for col in feature_cols:
        val = impute_values.get(col, 0.0)
        if pd.isna(val):
            val = 0.0
        df[col] = df[col].fillna(val)
        
    return df, impute_values


def get_stratified_split(
    df: pd.DataFrame,
    target_col: str = 'Attack_Family',
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Stratified split preserving exact attack class distribution.
    Falls back gracefully if rare classes have < 2 samples.
    """
    counts = df[target_col].value_counts()
    rare_classes = counts[counts < 2].index.tolist()
    
    if rare_classes:
        # Keep rare classes in train set, stratify the rest
        common_df = df[~df[target_col].isin(rare_classes)]
        rare_df = df[df[target_col].isin(rare_classes)]
        
        train_df, test_df = train_test_split(
            common_df,
            test_size=test_size,
            random_state=random_state,
            stratify=common_df[target_col]
        )
        train_df = pd.concat([train_df, rare_df], ignore_index=True)
    else:
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=df[target_col]
        )
        
    return train_df, test_df
