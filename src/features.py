"""
features.py
===========
Empirical feature auditing and selection utilities.

Instead of guessing which features to eliminate, this module inspects the data:
1. Zero-Variance Audit: Finds columns with 0 variance (constant values).
2. Collinearity Audit: Identifies pairs of features with correlation |r| > threshold.
3. Redundancy Pruning: In each collinear pair, retains the feature with higher target relevance.
4. Destination Port Leakage Audit: Checks if models are relying on port memorization vs flow behavior.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict


def audit_zero_variance(df: pd.DataFrame, feature_cols: List[str]) -> List[str]:
    """
    Identifies features with zero variance (constant value across all rows).
    These features carry 0 information and waste RAM/compute.
    """
    zero_var_cols = []
    for col in feature_cols:
        # Check if nunique <= 1 or std is 0
        if df[col].nunique(dropna=False) <= 1 or np.isclose(df[col].std(ddof=0), 0.0):
            zero_var_cols.append(col)
            
    return zero_var_cols


def audit_collinear_pairs(
    df: pd.DataFrame,
    feature_cols: List[str],
    threshold: float = 0.95
) -> List[Tuple[str, str, float]]:
    """
    Calculates Pearson correlation matrix and identifies all pairs of features
    with absolute correlation >= threshold (e.g., 0.95).
    """
    corr_matrix = df[feature_cols].corr().abs()
    pairs = []
    
    # Iterate over upper triangle of correlation matrix
    cols = list(feature_cols)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            col1 = cols[i]
            col2 = cols[j]
            r = corr_matrix.loc[col1, col2]
            if r >= threshold and not np.isnan(r):
                pairs.append((col1, col2, float(r)))
                
    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs


def select_features_redundancy_pruned(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = 'is_attack',
    threshold: float = 0.95
) -> Tuple[List[str], List[Dict]]:
    """
    Data-driven pruning:
    1. Removes zero-variance features.
    2. Identifies collinear pairs (|r| >= threshold).
    3. Evaluates target correlation for both features in the pair.
    4. Drops the feature that has weaker target correlation.
    """
    # 1. Zero-variance drop
    zero_var = audit_zero_variance(df, feature_cols)
    active_cols = [c for c in feature_cols if c not in zero_var]
    
    # 2. Target correlation of remaining features
    target_corrs = df[active_cols].apply(lambda s: s.corr(df[target_col])).abs().fillna(0.0)
    
    # 3. Collinear pairs
    corr_matrix = df[active_cols].corr().abs()
    
    cols_to_drop = set(zero_var)
    audit_log = []
    
    for c in zero_var:
        audit_log.append({
            'dropped_feature': c,
            'reason': 'Zero Variance (Constant column)',
            'kept_partner': None,
            'correlation': 0.0
        })

    # Sort pairs by correlation descending
    col_list = [c for c in active_cols if c not in cols_to_drop]
    for i in range(len(col_list)):
        col1 = col_list[i]
        if col1 in cols_to_drop:
            continue
        for j in range(i + 1, len(col_list)):
            col2 = col_list[j]
            if col2 in cols_to_drop:
                continue
                
            r = corr_matrix.loc[col1, col2]
            if r >= threshold and not np.isnan(r):
                # Compare relevance to target
                score1 = target_corrs.get(col1, 0.0)
                score2 = target_corrs.get(col2, 0.0)
                
                if score1 >= score2:
                    drop_candidate = col2
                    keep_candidate = col1
                else:
                    drop_candidate = col1
                    keep_candidate = col2
                    
                cols_to_drop.add(drop_candidate)
                audit_log.append({
                    'dropped_feature': drop_candidate,
                    'reason': f'Collinear redundancy (|r| = {r:.4f})',
                    'kept_partner': keep_candidate,
                    'correlation': float(r)
                })

    selected_features = [c for c in feature_cols if c not in cols_to_drop]
    return selected_features, audit_log


def audit_port_leakage(df: pd.DataFrame, port_col: str = 'Destination Port', label_col: str = 'Attack_Family') -> pd.DataFrame:
    """
    Audits Destination Port distribution across attack types.
    Highlights whether specific attacks exclusively hit single ports
    (e.g., Port 80 for DoS, Port 21 for FTP Brute Force).
    """
    if port_col not in df.columns:
        return pd.DataFrame()
        
    summary = df.groupby(label_col)[port_col].agg(
        total_flows='count',
        unique_ports='nunique',
        top_port=lambda x: x.mode().iloc[0] if not x.empty else np.nan,
        top_port_pct=lambda x: (x == x.mode().iloc[0]).mean() * 100 if not x.empty else 0.0
    ).reset_index()
    
    return summary
