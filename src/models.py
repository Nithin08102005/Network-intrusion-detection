"""
models.py
=========
Machine Learning models and evaluation suite for intrusion detection.

Includes:
1. Baseline Logistic Regression (with balanced class weights).
2. High-performance Gradient Boosted Trees (XGBoost / Random Forest).
3. Full evaluation suite: Precision, Recall, Macro-F1, PR-AUC, Confusion Matrix.
4. Model serialization with joblib for production/demo serving.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


class IntrusionDetectionPipeline:
    def __init__(
        self,
        model_type: str = "rf",
        task: str = "binary",
        random_state: int = 42
    ):
        """
        model_type: 'lr' (Logistic Regression), 'rf' (Random Forest), or 'xgb' (XGBoost)
        task: 'binary' or 'multiclass'
        """
        self.model_type = model_type.lower()
        self.task = task.lower()
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.feature_names = []
        self.classes_ = []
        self.model = self._init_model()

    def _init_model(self):
        if self.model_type == "lr":
            return LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=self.random_state,
                solver="saga" if self.task == "multiclass" else "lbfgs",
                n_jobs=-1
            )
        elif self.model_type == "rf":
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                class_weight="balanced",
                random_state=self.random_state,
                n_jobs=-1
            )
        elif self.model_type == "xgb":
            if not HAS_XGBOOST:
                print("[!] XGBoost not installed. Falling back to Random Forest.")
                return self._init_rf()
            if self.task == "binary":
                return xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    eval_metric="logloss",
                    random_state=self.random_state,
                    n_jobs=-1
                )
            else:
                return xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    eval_metric="mlogloss",
                    random_state=self.random_state,
                    n_jobs=-1
                )
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Fits scaler and model."""
        self.feature_names = list(X_train.columns)
        self.classes_ = np.unique(y_train)
        
        # Fit scaler on training data only
        X_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_scaled, y_train)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Transforms features and generates predictions."""
        X_scaled = self.scaler.transform(X[self.feature_names])
        return self.model.predict(X_scaled)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Returns predicted class probabilities."""
        X_scaled = self.scaler.transform(X[self.feature_names])
        return self.model.predict_proba(X_scaled)

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Comprehensive evaluation suite."""
        y_pred = self.predict(X_test)
        
        metrics = {
            "macro_f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
            "macro_precision": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
            "macro_recall": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
            "classification_report": classification_report(y_test, y_pred, output_dict=True, zero_division=0),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
        }

        if self.task == "binary":
            try:
                y_proba = self.predict_proba(X_test)[:, 1]
                metrics["roc_auc"] = float(roc_auc_score(y_test, y_proba))
            except Exception:
                metrics["roc_auc"] = None

        return metrics

    def save(self, filepath: str):
        """Saves entire pipeline (model + scaler + feature list)."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        bundle = {
            "model": self.model,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "classes_": self.classes_,
            "model_type": self.model_type,
            "task": self.task
        }
        joblib.dump(bundle, filepath)
        print(f"[+] Saved pipeline to '{filepath}'")

    @classmethod
    def load(cls, filepath: str):
        """Loads serialized pipeline."""
        bundle = joblib.load(filepath)
        instance = cls(model_type=bundle["model_type"], task=bundle["task"])
        instance.model = bundle["model"]
        instance.scaler = bundle["scaler"]
        instance.feature_names = bundle["feature_names"]
        instance.classes_ = bundle["classes_"]
        return instance
