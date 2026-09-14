"""
app.py
======
Interactive Intrusion Detection System (CICIDS-2017) Dashboard.
Built with Streamlit for real-time packet inspection, threat prediction, and batch analysis.
"""

import os
import sys
import pandas as pd
import numpy as np
import streamlit as st
import joblib

# Set Page Config
st.set_page_config(
    page_title="AI Network Intrusion Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 8px;
        padding: 16px;
        border: 1px solid #E2E8F0;
    }
    .attack-alert {
        background-color: #FEF2F2;
        border-left: 5px solid #EF4444;
        padding: 14px;
        border-radius: 4px;
        color: #991B1B;
        font-weight: 600;
        font-size: 1.15rem;
    }
    .benign-alert {
        background-color: #F0FDF4;
        border-left: 5px solid #22C55E;
        padding: 14px;
        border-radius: 4px;
        color: #166534;
        font-weight: 600;
        font-size: 1.15rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models():
    binary_path = "models/best_binary_model.pkl"
    multi_path = "models/best_multiclass_model.pkl"

    if not os.path.exists(binary_path) or not os.path.exists(multi_path):
        return None, None

    bin_bundle = joblib.load(binary_path)
    multi_bundle = joblib.load(multi_path)
    return bin_bundle, multi_bundle


def main():
    st.markdown('<div class="main-header">🛡️ Network Intrusion Detection System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Trained on CICIDS-2017 flow telemetry with empirical feature selection & leakage mitigation</div>', unsafe_allow_html=True)

    bin_bundle, multi_bundle = load_models()

    if bin_bundle is None or multi_bundle is None:
        st.error("⚠️ Model artifacts not found in `models/`. Please run `python train_pipeline.py` first.")
        return

    feature_names = bin_bundle["feature_names"]
    bin_model = bin_bundle["model"]
    bin_scaler = bin_bundle["scaler"]
    multi_model = multi_bundle["model"]
    multi_scaler = multi_bundle["scaler"]

    # Sidebar Navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio("Choose Mode:", ["Live Flow Inspector", "Batch CSV Audit", "Model & Feature Architecture"])

    # -------------------------------------------------------------
    # Mode 1: Live Flow Inspector
    # -------------------------------------------------------------
    if app_mode == "Live Flow Inspector":
        st.subheader("🔍 Real-Time Network Flow Inference")
        st.write("Inspect individual network flow characteristics or select pre-configured presets to simulate network events.")

        # Presets
        preset = st.selectbox(
            "Select Traffic Simulation Preset:",
            [
                "Custom Manual Input",
                "Benign Web Session (Normal HTTPS)",
                "DDoS Flooding Attack (High Volume, Short Duration)",
                "PortScan Reconnaissance (Single Packet Probes)",
                "DoS Slowloris (Low Rate, Hanging Connections)"
            ]
        )

        # Baseline sample values
        default_vals = {f: 0.0 for f in feature_names}
        
        # Heuristic presets for demonstration
        if preset == "Benign Web Session (Normal HTTPS)":
            default_vals.update({
                "Flow Duration": 150000.0,
                "Total Fwd Packets": 10.0,
                "Total Backward Packets": 12.0,
                "Flow Bytes/s": 45000.0,
                "Flow Packets/s": 140.0,
                "Init_Win_bytes_forward": 29200.0,
                "Init_Win_bytes_backward": 28960.0,
                "Destination Port": 443.0,
            })
        elif preset == "DDoS Flooding Attack (High Volume, Short Duration)":
            default_vals.update({
                "Flow Duration": 2000.0,
                "Total Fwd Packets": 150.0,
                "Total Backward Packets": 0.0,
                "Flow Bytes/s": 1800000.0,
                "Flow Packets/s": 75000.0,
                "Destination Port": 80.0,
                "SYN Flag Count": 1.0,
                "Init_Win_bytes_forward": 256.0,
                "Init_Win_bytes_backward": -1.0
            })
        elif preset == "PortScan Reconnaissance (Single Packet Probes)":
            default_vals.update({
                "Flow Duration": 45.0,
                "Total Fwd Packets": 1.0,
                "Total Backward Packets": 0.0,
                "Flow Bytes/s": 0.0,
                "Flow Packets/s": 22222.0,
                "Destination Port": 8080.0,
                "SYN Flag Count": 1.0,
                "Init_Win_bytes_forward": 1024.0,
                "Init_Win_bytes_backward": -1.0
            })
        elif preset == "DoS Slowloris (Low Rate, Hanging Connections)":
            default_vals.update({
                "Flow Duration": 12000000.0,
                "Total Fwd Packets": 3.0,
                "Total Backward Packets": 1.0,
                "Flow Bytes/s": 12.0,
                "Flow Packets/s": 0.33,
                "Destination Port": 80.0,
                "Init_Win_bytes_forward": 8192.0,
                "Init_Win_bytes_backward": 256.0
            })

        st.markdown("#### Flow Parameters")
        # Display top prominent features for input
        cols = st.columns(3)
        input_data = {}
        
        # Key intuitive features for user interaction
        prominent_keys = [
            "Destination Port", "Flow Duration", "Total Fwd Packets",
            "Total Backward Packets", "Flow Bytes/s", "Flow Packets/s",
            "Init_Win_bytes_forward", "Init_Win_bytes_backward", "SYN Flag Count"
        ]
        
        for i, feat in enumerate(feature_names):
            val = default_vals.get(feat, 0.0)
            col_idx = i % 3
            with cols[col_idx]:
                if feat in prominent_keys:
                    input_data[feat] = st.number_input(f"**{feat}**", value=float(val), key=feat)
                else:
                    input_data[feat] = st.number_input(feat, value=float(val), key=feat, help="Advanced flow feature")

        if st.button("🚀 Analyze Network Flow", type="primary", use_container_width=True):
            input_df = pd.DataFrame([input_data])[feature_names]
            
            # Predict Binary
            X_bin = bin_scaler.transform(input_df)
            bin_pred = bin_model.predict(X_bin)[0]
            bin_proba = bin_model.predict_proba(X_bin)[0]

            # Predict Multiclass
            X_multi = multi_scaler.transform(input_df)
            multi_pred = multi_model.predict(X_multi)[0]
            multi_proba = multi_model.predict_proba(X_multi)[0]
            multi_classes = multi_bundle["classes_"]

            st.write("---")
            res_col1, res_col2 = st.columns([1, 1])

            with res_col1:
                st.markdown("### Threat Assessment")
                if bin_pred == 1:
                    attack_prob = bin_proba[1] * 100
                    st.markdown(
                        f'<div class="attack-alert">🚨 MALICIOUS ACTIVITY DETECTED<br>'
                        f'<span style="font-size: 0.95rem; font-weight: normal;">Attack Family: <b>{multi_pred}</b> ({attack_prob:.1f}% confidence)</span></div>',
                        unsafe_allow_html=True
                    )
                else:
                    benign_prob = bin_proba[0] * 100
                    st.markdown(
                        f'<div class="benign-alert">✅ BENIGN NETWORK FLOW<br>'
                        f'<span style="font-size: 0.95rem; font-weight: normal;">Normal Traffic Pattern ({benign_prob:.1f}% confidence)</span></div>',
                        unsafe_allow_html=True
                    )

            with res_col2:
                st.markdown("### Class Probabilities")
                proba_df = pd.DataFrame({
                    "Attack Family": multi_classes,
                    "Probability": multi_proba
                }).sort_values("Probability", ascending=False)
                st.bar_chart(proba_df.set_index("Attack Family"))

    # -------------------------------------------------------------
    # Mode 2: Batch CSV Audit
    # -------------------------------------------------------------
    elif app_mode == "Batch CSV Audit":
        st.subheader("📂 Batch Traffic CSV Inspection")
        st.write("Upload a CSV export of NetFlow records to run automated intrusion detection on all rows.")

        uploaded_file = st.file_uploader("Upload Network Flow CSV", type=["csv"])
        if uploaded_file is not None:
            df_batch = pd.read_csv(uploaded_file)
            df_batch.columns = df_batch.columns.str.strip()
            st.write(f"Loaded **{len(df_batch):,} rows** with {df_batch.shape[1]} columns.")

            missing = [f for f in feature_names if f not in df_batch.columns]
            if missing:
                st.warning(f"⚠️ Note: {len(missing)} features were missing in CSV and imputed with 0: {missing[:5]}...")
                for m in missing:
                    df_batch[m] = 0.0

            if st.button("⚡ Run Batch Detection", type="primary"):
                with st.spinner("Classifying flows..."):
                    X_scaled = bin_scaler.transform(df_batch[feature_names])
                    preds = bin_model.predict(X_scaled)
                    
                    X_multi_scaled = multi_scaler.transform(df_batch[feature_names])
                    multi_preds = multi_model.predict(X_multi_scaled)

                    df_batch["Prediction"] = np.where(preds == 1, "MALICIOUS", "BENIGN")
                    df_batch["Attack_Category"] = multi_preds

                    m1, m2, m3 = st.columns(3)
                    m1.metric("Total Flows", f"{len(df_batch):,}")
                    m2.metric("Benign Flows", f"{(preds == 0).sum():,}")
                    m3.metric("Attacks Detected", f"{(preds == 1).sum():,}", delta=f"{(preds == 1).mean()*100:.1f}% attack rate", delta_color="inverse")

                    st.dataframe(df_batch[["Prediction", "Attack_Category"] + [c for c in df_batch.columns if c not in ["Prediction", "Attack_Category"]][:6]].head(50))

    # -------------------------------------------------------------
    # Mode 3: Model & Feature Architecture
    # -------------------------------------------------------------
    elif app_mode == "Model & Feature Architecture":
        st.subheader("🔬 Engineering Methodology & Feature Auditing")
        st.markdown("""
        ### Why this pipeline is robust:
        1. **Empirical Redundancy Pruning:**
           * The raw CICIDS-2017 dataset has 79 features with severe multi-collinearity ($|r| > 0.95$ across 70 pairs).
           * Rather than guessing which features to drop, our pipeline measures target correlation for every redundant pair and retains the superior predictor.
           * Reduced dimensionality from **78 features down to 42 high-signal features**.

        2. **Zero-Variance Filtering:**
           * 10 columns (`Bwd PSH Flags`, `Fwd URG Flags`, `Fwd/Bwd Avg Bulk Rates`, etc.) are constant 0 across all records.
           * Dropped automatically to save compute and eliminate useless matrix dimensions.

        3. **Mitigating Destination Port Leakage:**
           * In synthetic testbeds, attacks often target a single port (e.g. 100% of DDoS on Port 80).
           * Our models are evaluated with and without port numbers to guarantee learning genuine flow telemetry (packet sizes, inter-arrival times, TCP window flags) rather than testbed memorization.
        """)

        st.markdown("#### Currently Active Model Features (42 features):")
        feat_df = pd.DataFrame({
            "Index": range(1, len(feature_names) + 1),
            "Feature Name": feature_names
        })
        st.dataframe(feat_df, use_container_width=True)


if __name__ == "__main__":
    main()
