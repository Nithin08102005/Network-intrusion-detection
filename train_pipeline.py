"""
train_pipeline.py
=================
Automated training and benchmarking script.
Trains Baseline and Tree models on pruned features, logs metrics,
and saves the best models to models/.
"""

import os
import sys
import pandas as pd
from src.preprocessing import clean_labels, clean_inf_and_nan, get_stratified_split
from src.features import select_features_redundancy_pruned, audit_port_leakage
from src.models import IntrusionDetectionPipeline

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

def main():
    data_path = "data/sample_cicids.parquet"
    if not os.path.exists(data_path):
        print(f"[!] Dataset '{data_path}' not found. Please run src/data_loader.py first.")
        return

    print(f"[*] Loading dataset from '{data_path}'...")
    df = pd.read_parquet(data_path)
    df = clean_labels(df)
    
    raw_feats = [c for c in df.columns if c not in ['Label', 'Attack_Family', 'is_attack']]
    df, medians = clean_inf_and_nan(df, raw_feats)

    print("\n[*] Performing empirical feature selection...")
    selected_feats, drop_log = select_features_redundancy_pruned(
        df, raw_feats, target_col='is_attack', threshold=0.95
    )
    print(f"    Raw features     : {len(raw_feats)}")
    print(f"    Selected features: {len(selected_feats)}")
    print(f"    Pruned features  : {len(drop_log)}")

    print("\n[*] Splitting into stratified Train (80%) and Test (20%)...")
    train_df, test_df = get_stratified_split(df, target_col='Attack_Family', test_size=0.2, random_state=42)
    print(f"    Train flows: {len(train_df):,}")
    print(f"    Test flows : {len(test_df):,}")

    os.makedirs("models", exist_ok=True)

    # -------------------------------------------------------------
    # 1. Binary Classification: BENIGN vs ATTACK
    # -------------------------------------------------------------
    print("\n" + "="*60)
    print(" TASK 1: BINARY CLASSIFICATION (BENIGN vs ATTACK)")
    print("="*60)

    # Baseline: Logistic Regression
    print("[*] Training Baseline Logistic Regression (balanced weights)...")
    lr_bin = IntrusionDetectionPipeline(model_type="lr", task="binary")
    lr_bin.fit(train_df[selected_feats], train_df["is_attack"])
    lr_metrics = lr_bin.evaluate(test_df[selected_feats], test_df["is_attack"])
    print(f"    Logistic Regression -> Macro-F1: {lr_metrics['macro_f1']:.4f}, "
          f"Precision: {lr_metrics['macro_precision']:.4f}, Recall: {lr_metrics['macro_recall']:.4f}, "
          f"ROC-AUC: {lr_metrics['roc_auc']:.4f}")

    # Reference: Random Forest
    print("[*] Training Random Forest Classifier...")
    rf_bin = IntrusionDetectionPipeline(model_type="rf", task="binary")
    rf_bin.fit(train_df[selected_feats], train_df["is_attack"])
    rf_metrics = rf_bin.evaluate(test_df[selected_feats], test_df["is_attack"])
    print(f"    Random Forest       -> Macro-F1: {rf_metrics['macro_f1']:.4f}, "
          f"Precision: {rf_metrics['macro_precision']:.4f}, Recall: {rf_metrics['macro_recall']:.4f}, "
          f"ROC-AUC: {rf_metrics['roc_auc']:.4f}")

    rf_bin.save("models/best_binary_model.pkl")

    # -------------------------------------------------------------
    # 2. Multiclass Classification: Attack Families
    # -------------------------------------------------------------
    print("\n" + "="*60)
    print(" TASK 2: MULTICLASS CLASSIFICATION (ATTACK FAMILIES)")
    print("="*60)

    print("[*] Training Multiclass Random Forest Classifier...")
    rf_multi = IntrusionDetectionPipeline(model_type="rf", task="multiclass")
    rf_multi.fit(train_df[selected_feats], train_df["Attack_Family"])
    multi_metrics = rf_multi.evaluate(test_df[selected_feats], test_df["Attack_Family"])

    print(f"    Macro-F1    : {multi_metrics['macro_f1']:.4f}")
    print(f"    Weighted-F1 : {multi_metrics['weighted_f1']:.4f}")
    print("\n    Per-Class Performance breakdown:")
    for cls, rep in multi_metrics["classification_report"].items():
        if isinstance(rep, dict) and cls not in ["macro avg", "weighted avg"]:
            print(f"      - {cls:<12}: Precision={rep['precision']:.4f}, Recall={rep['recall']:.4f}, F1={rep['f1-score']:.4f} (support={rep['support']})")

    rf_multi.save("models/best_multiclass_model.pkl")
    print("\n[+] Pipeline training complete. Models saved in models/")

if __name__ == "__main__":
    main()
