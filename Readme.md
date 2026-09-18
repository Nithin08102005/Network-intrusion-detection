# 🛡️ End-to-End Network Intrusion Detection System (CICIDS-2017)

[![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nithinbachupally/notebook7f29a88e82)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)

An end-to-end Machine Learning pipeline for real-time Network Intrusion Detection (NIDS) trained and evaluated on the official **CICIDS-2017** benchmark dataset.

Unlike conventional implementations that rely on hardcoded feature drops or suffer from data leakage, this project applies **empirical feature auditing** (zero-variance elimination, correlation redundancy pruning at $|r| \ge 0.95$, and testbed data leakage mitigation) to systematically reduce feature dimensionality while maintaining top-tier classification performance across diverse cyber attack vectors.

👉 **[View Full-Scale Execution on Kaggle (3.1M Records)](https://www.kaggle.com/code/nithinbachupally/notebook7f29a88e82)**

---

## 🌟 Key Highlights for Resume & Interviews

* **Full-Scale Big-Data Training**: Trained on all **8 official daily NetFlow CSVs** (over **3.1 million network flow records**) on Kaggle, evaluated across a holdout test set of **623,869 flows**.
* **Empirical Feature Selection**: Systematically pruned **46% of redundant features** via automated zero-variance filtering (eliminating dead columns with standard deviation = 0) and pairwise collinearity filtering ($|r| \ge 0.95$).
* **Data Leakage & Hygiene**: Addressed division-by-zero infinite values (`Inf`) in flow rates using median imputation; purged synthetic testbed identifiers (decimal IP encodings, timestamps, and pre-computed one-hot label leaks) to ensure models learn genuine traffic behavior.
* **Cost-Sensitive Class Imbalance**: Mitigated severe class imbalance using cost-sensitive balanced loss weighting (`class_weight='balanced'`), successfully detecting rare attacks (Web Attacks, Infiltration, Botnets) without synthetic SMOTE evaluation artifacts.
* **Rigorous Benchmarking**: Established a linear baseline with Logistic Regression (**0.9157 Macro-F1**, **0.9842 ROC-AUC**) and scaled to Random Forest (**0.9772 Macro-F1**, **0.9994 Accuracy**, **1.0000 ROC-AUC**).
* **Interactive Deployment**: Built an interactive **Streamlit Web Application** (`app.py`) for real-time network packet inspection, simulated attack presets, and batch CSV telemetry audits.

---

## 📊 Full-Scale Benchmark Results (Kaggle Cloud Run)

Evaluated on an un-oversampled, real-world holdout test set of **623,869 flows** from the official CICIDS-2017 capture:

### 1. Binary Task (BENIGN vs ATTACK)
| Model | Macro-F1 | ROC-AUC |
|---|---:|---:|
| **Baseline (Logistic Regression)** | 0.9157 | 0.9842 |
| **Random Forest Ensemble** | **0.9992** | **1.0000** |

### 2. Multiclass Task (Attack Families Breakdown)
* **Overall Macro-F1:** **0.9772** (97.7%) | **Accuracy:** **0.9994** (99.94%) | **Weighted-F1:** **0.9995**

| Attack Family | Precision | Recall | F1-Score | Test Support |
|---|---:|---:|---:|---:|
| **BENIGN** | 0.9999 | 0.9994 | **0.9996** | 454,620 |
| **DDoS** | 1.0000 | 0.9996 | **0.9998** | 25,605 |
| **DoS** | 0.9960 | 0.9996 | **0.9978** | 50,534 |
| **PortScan** | 0.9999 | 0.9996 | **0.9997** | 31,786 |
| **Brute Force** | 1.0000 | 0.9989 | **0.9995** | 2,767 |
| **Web Attack** | 0.9793 | 0.9748 | **0.9770** | 436 |
| **Infiltration** | 1.0000 | 0.8571 | **0.9231** | 7 |
| **Botnet** | 0.8180 | 0.9949 | **0.8978** | 393 |

> **Analysis:** High-volume flood attacks (DDoS, DoS, PortScan) are categorized near 100% due to distinct flow rate and window signatures. Stealthy low-volume attacks (Infiltration, Botnets) show realistic, publication-grade variance (89.8% to 92.3% F1).

---

## 📁 Repository Structure

```
CICIDS/
├── app.py                          # Interactive Streamlit Web App for real-time inference
├── train_pipeline.py               # End-to-end automated training & benchmarking script
├── requirements.txt                # Production package dependencies
├── Readme.md                       # Documentation & interview guide
├── data/
│   └── sample_cicids.parquet       # Representative multi-class dataset (97k rows, 79 features)
├── models/
│   ├── best_binary_model.pkl       # Serialized binary classifier & scaler
│   └── best_multiclass_model.pkl   # Serialized multiclass classifier & scaler
├── notebooks/
│   └── CICIDS2017_Empirical_Pipeline.ipynb # Self-contained analysis notebook (Local + Kaggle)
└── src/
    ├── data_loader.py              # Parquet streaming & stratified sampling utility
    ├── preprocessing.py            # Label normalizer, Inf/NaN handler, stratified split
    ├── features.py                 # Zero-variance audit, correlation pruning, leakage audit
    └── models.py                   # Classifier wrappers, evaluation suite & serialization
```

---

## 🔬 Empirical Data Audits

### 1. Zero-Variance Elimination
The raw dataset contains 10 features with standard deviation = 0 (constant 0 across every record):
* `Bwd PSH Flags`, `Fwd URG Flags`, `Bwd URG Flags`, `CWE Flag Count`
* `Fwd/Bwd Avg Bytes/Bulk`, `Fwd/Bwd Avg Packets/Bulk`, `Fwd/Bwd Avg Bulk Rate`

*Resolution*: Automatically detected and pruned to reduce memory consumption and matrix dimensionality.

### 2. Collinear Redundancy Pruning ($|r| \ge 0.95$)
We identified 70 pairs of features with absolute Pearson correlation $\ge 0.95$:
* `Total Fwd Packets` $\leftrightarrow$ `Subflow Fwd Packets` ($r = 1.0000$)
* `Total Length of Fwd Packets` $\leftrightarrow$ `Subflow Fwd Bytes` ($r = 1.0000$)
* `Fwd Packet Length Mean` $\leftrightarrow$ `Avg Fwd Segment Size` ($r = 1.0000$)
* `RST Flag Count` $\leftrightarrow$ `ECE Flag Count` ($r = 1.0000$)

*Resolution*: In every redundant pair, our algorithm computes each feature's correlation with the target label and retains the superior predictor, trimming redundant columns while retaining all unique predictive information.

### 3. Destination Port Leakage Audit
In the CICIDS-2017 laboratory environment:
* **100% of DDoS attacks** targeted Port 80.
* **99.86% of DoS attacks** targeted Port 80.
* **100% of Web attacks** targeted Port 80.

*Engineering Decision*: Relying purely on `Destination Port` causes severe false alarms in enterprise networks (where standard HTTP traffic runs on Port 80). Our model was validated on behavioral flow statistics (Packet Length Variance, TCP Window sizes, Flow Inter-Arrival Times) to ensure genuine anomaly detection.

---

## 🚀 Quickstart Guide

### 1. Setup Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run Training Pipeline Locally
To run the automated feature selection and train models locally on the 97k sample:
```bash
python train_pipeline.py
```

### 3. Launch Interactive Dashboard
To launch the real-time Streamlit dashboard:
```bash
streamlit run app.py
```

---

## ☁️ Running on Full 3.1M Dataset in Kaggle

You can view and execute the full 3.1M run directly in the cloud:
👉 **[Open Kaggle Notebook](https://www.kaggle.com/code/nithinbachupally/notebook7f29a88e82)**

1. Attach the official `CICIDS2017 Official Flow Feature CSV Files` dataset.
2. Select **Accelerator: None** (CPU with 30 GB RAM) or **GPU T4 x2**.
3. Click **Run All** to reproduce the exact metrics reported above.

---

## 💼 Suggested Resume Bullet Points

Feel free to adapt these for your resume:

* **Network Intrusion Detection System (CICIDS-2017) | Python, Scikit-Learn, Streamlit**
  * *Engineered an end-to-end network intrusion detection pipeline processing **3.1 million NetFlow records** across 8 threat families (DDoS, DoS, PortScan, Brute Force, Web Attacks, Infiltration, Botnets).*
  * *Designed an automated feature auditing pipeline that pruned **46% of redundant features** via zero-variance detection and pairwise collinearity filtering ($|r| \ge 0.95$).*
  * *Audited and mitigated **Destination Port testbed data leakage**, ensuring models generalize to genuine flow telemetry (inter-arrival times, TCP window flags) rather than static port heuristics.*
  * *Trained cost-sensitive Random Forest and Gradient Boosted models, achieving **0.9772 Macro-F1** and **0.9994 Accuracy** on 623k holdout test flows with sub-millisecond inference latency.*
  * *Deployed an interactive **Streamlit web application** enabling real-time packet classification, simulated attack scenarios, and batch CSV telemetry audits.*
  * *Published and documented the full-scale cloud execution on **[Kaggle](https://www.kaggle.com/code/nithinbachupally/notebook7f29a88e82)**.*
