import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Vital Watch · Patient Deterioration Predictor",
    page_icon="🏥",
    layout="wide",
)

# ─────────────────────────────────────────────
# Load model + scaler
# ─────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    BASE_DIR = os.path.dirname(__file__)

    model  = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
    scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
    return model, scaler

try:
    model, scaler = load_artifacts()
    MODEL_LOADED = True
    MODEL_LOAD_ERROR = ""
except FileNotFoundError:
    MODEL_LOADED = False
    MODEL_LOAD_ERROR = "model.pkl / scaler.pkl not found in the app directory."
except Exception as e:
    MODEL_LOADED = False
    MODEL_LOAD_ERROR = f"Failed to load saved artifacts: {type(e).__name__}: {e}"

# ─────────────────────────────────────────────
# Feature engineering — mirrors the notebook
# ─────────────────────────────────────────────
OPTIMAL_THRESHOLD = 0.35  

def calculate_mews(row: dict) -> int:
    score = 0
    rr = row["respiratory_rate"]
    hr = row["heart_rate"]
    sbp = row["systolic_bp"]
    temp = row["temperature_c"]
    spo2 = row["spo2_pct"]

    if rr < 9 or rr > 24:   score += 2
    elif rr > 20:            score += 1

    if hr < 40 or hr > 130:  score += 2
    elif hr > 110:            score += 1

    if sbp < 90:              score += 2
    elif sbp < 100:           score += 1

    if temp < 35 or temp > 38.5: score += 2
    elif temp > 38:              score += 1

    if spo2 < 90:             score += 2
    elif spo2 < 94:           score += 1

    return score


def build_feature_row(inp: dict) -> pd.DataFrame:
    """
    Accepts raw vital-sign inputs and returns a fully-engineered
    single-row DataFrame aligned to the training feature space.
    """
    r = inp.copy()

    # --- derived flags ---
    r["low_spo2"]      = int(r["spo2_pct"] < 92)
    r["high_hr"]       = int(r["heart_rate"] > 110)
    r["fever"]         = int(r["temperature_c"] > 38)
    r["low_bp"]        = int(r["systolic_bp"] < 90)
    r["high_rr"]       = int(r["respiratory_rate"] > 24)
    r["high_lactate"]  = int(r["lactate"] > 2)
    r["high_wbc"]      = int(r["wbc_count"] > 11000)
    r["high_crp"]      = int(r["crp_level"] > 10)
    r["spo2_hr_risk"]  = int((r["spo2_pct"] < 94) and (r["heart_rate"] > 100))
    r["mews"]          = calculate_mews(r)

    # --- rolling / change features (single observation → set to 0) ---
    for col in ["hr_change", "bp_change", "spo2_change",
                "hr_avg_3", "spo2_avg_3", "rr_avg_3",
                "hr_volatility_3", "spo2_volatility_3"]:
        r[col] = 0.0

    # --- one-hot: oxygen_device ---
    for dev in ["nasal_cannula", "face_mask", "high_flow", "room_air", "ventilator"]:
        r[f"oxygen_device_{dev}"] = int(r.get("oxygen_device", "") == dev)

    # --- one-hot: gender ---
    for g in ["female", "male", "other"]:
        r[f"gender_{g}"] = int(r.get("gender", "") == g)

    # --- one-hot: admission_type ---
    for at in ["elective", "emergency", "transfer"]:
        r[f"admission_type_{at}"] = int(r.get("admission_type", "") == at)

    # drop original categorical columns
    for col in ["oxygen_device", "gender", "admission_type"]:
        r.pop(col, None)

    df = pd.DataFrame([r])
    df.replace([np.inf, -np.inf], 0, inplace=True)
    df.fillna(0, inplace=True)
    return df


def risk_label(prob: float, threshold: float) -> tuple[str, str]:
    """Returns (label, colour)."""
    if prob < threshold * 0.7:
        return "🟢 Low Risk", "green"
    elif prob < threshold * 1.3:
        return "🟡 Medium Risk", "orange"
    else:
        return "🔴 High Risk", "red"


# ─────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────
st.title("🏥 Vital Watch — Patient Deterioration Predictor")
st.caption("AI-based early warning system · predicts physiological deterioration in the next 12 hours")

if not MODEL_LOADED:
    st.error(
        "**Model artifacts could not be loaded.**  \n"
        "Run/re-run training to generate compatible `model.pkl` and `scaler.pkl`, "
        "then place both files in the same directory as `app.py`."
    )
    if MODEL_LOAD_ERROR:
        st.caption(MODEL_LOAD_ERROR)
    st.stop()

# ── Tabs ────────────────────────────────────
tab_single, tab_batch = st.tabs(["🔬 Single Patient", "📋 Batch (CSV Upload)"])

# ═══════════════════════════════════════════
# TAB 1 — Single patient
# ═══════════════════════════════════════════
with tab_single:
    st.subheader("Enter Vital Signs & Lab Values")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Vitals**")
        heart_rate       = st.number_input("Heart Rate (bpm)",        min_value=20,  max_value=250, value=80)
        systolic_bp      = st.number_input("Systolic BP (mmHg)",      min_value=40,  max_value=250, value=120)
        diastolic_bp     = st.number_input("Diastolic BP (mmHg)",     min_value=20,  max_value=150, value=80)
        respiratory_rate = st.number_input("Respiratory Rate (/min)", min_value=4,   max_value=60,  value=16)
        spo2_pct         = st.number_input("SpO₂ (%)",                min_value=50,  max_value=100, value=98)
        temperature_c    = st.number_input("Temperature (°C)",         min_value=30.0, max_value=43.0, value=37.0, step=0.1)

    with col2:
        st.markdown("**Labs**")
        lactate    = st.number_input("Lactate (mmol/L)",  min_value=0.0,  max_value=20.0, value=1.0, step=0.1)
        wbc_count  = st.number_input("WBC Count (/µL)",   min_value=0,    max_value=100000, value=8000, step=100)
        crp_level  = st.number_input("CRP Level (mg/L)",  min_value=0.0,  max_value=300.0, value=5.0, step=0.1)
        creatinine = st.number_input("Creatinine (mg/dL)", min_value=0.0, max_value=20.0,  value=1.0, step=0.01)
        hemoglobin = st.number_input("Hemoglobin (g/dL)", min_value=0.0,  max_value=25.0,  value=13.0, step=0.1)
        platelet_count = st.number_input("Platelets (×10³/µL)", min_value=0, max_value=2000, value=250)

    with col3:
        st.markdown("**Patient Info**")
        age                 = st.number_input("Age (years)",           min_value=0,  max_value=120, value=55)
        hour_from_admission = st.number_input("Hours from Admission",  min_value=0,  max_value=240, value=6)
        gender              = st.selectbox("Gender",        ["male", "female", "other"])
        admission_type      = st.selectbox("Admission Type", ["emergency", "elective", "transfer"])
        oxygen_device       = st.selectbox("Oxygen Device",  ["room_air", "nasal_cannula", "face_mask", "high_flow", "ventilator"])

    st.divider()

    if st.button("⚡ Predict Deterioration Risk", type="primary", use_container_width=True):
        raw = dict(
            heart_rate=heart_rate,
            systolic_bp=systolic_bp,
            diastolic_bp=diastolic_bp,
            respiratory_rate=respiratory_rate,
            spo2_pct=spo2_pct,
            temperature_c=temperature_c,
            lactate=lactate,
            wbc_count=wbc_count,
            crp_level=crp_level,
            creatinine=creatinine,
            hemoglobin=hemoglobin,
            platelet_count=platelet_count,
            age=age,
            hour_from_admission=hour_from_admission,
            gender=gender,
            admission_type=admission_type,
            oxygen_device=oxygen_device,
        )

        feat_df = build_feature_row(raw)

        # align to training columns
        try:
            feat_scaled = scaler.transform(feat_df.reindex(columns=scaler.feature_names_in_, fill_value=0))
        except AttributeError:
            feat_scaled = scaler.transform(feat_df)

        prob = model.predict_proba(feat_scaled)[0, 1]
        label, colour = risk_label(prob, OPTIMAL_THRESHOLD)
        mews_score = calculate_mews(raw)

        # ── Result display ─────────────────────
        r1, r2, r3 = st.columns(3)
        r1.metric("Deterioration Probability", f"{prob:.1%}")
        r2.metric("Risk Level", label)
        r3.metric("MEWS Score", mews_score)

        # colour bar
        bar_pct = int(prob * 100)
        st.markdown(
            f"""
            <div style="background:#e0e0e0;border-radius:8px;height:18px;width:100%">
              <div style="background:{'#d62728' if colour=='red' else '#ff7f0e' if colour=='orange' else '#2ca02c'};
                          width:{bar_pct}%;height:18px;border-radius:8px;
                          transition:width .4s ease"></div>
            </div>
            <p style="text-align:right;font-size:0.8rem;margin-top:2px">{bar_pct}%</p>
            """,
            unsafe_allow_html=True,
        )

        # clinical flags
        st.markdown("#### 🚨 Active Clinical Flags")
        flags = []
        if spo2_pct < 92:                               flags.append("Low SpO₂ (<92%)")
        if heart_rate > 110:                            flags.append("Tachycardia (HR >110)")
        if temperature_c > 38:                          flags.append("Fever (>38 °C)")
        if systolic_bp < 90:                            flags.append("Hypotension (SBP <90)")
        if respiratory_rate > 24:                       flags.append("Tachypnoea (RR >24)")
        if lactate > 2:                                 flags.append("Elevated Lactate (>2)")
        if wbc_count > 11000:                           flags.append("Elevated WBC (>11 000)")
        if crp_level > 10:                              flags.append("Elevated CRP (>10)")
        if (spo2_pct < 94) and (heart_rate > 100):     flags.append("SpO₂–HR Combined Risk")

        if flags:
            for f in flags:
                st.warning(f"⚠️ {f}")
        else:
            st.success("✅ No critical thresholds breached")

        with st.expander("📊 Feature Values Sent to Model"):
            st.dataframe(feat_df.T.rename(columns={0: "value"}), use_container_width=True)

# ═══════════════════════════════════════════
# TAB 2 — Batch prediction
# ═══════════════════════════════════════════
with tab_batch:
    st.subheader("Upload a CSV for Batch Prediction")
    st.info(
        "CSV must contain at minimum: `heart_rate`, `systolic_bp`, `diastolic_bp`, "
        "`respiratory_rate`, `spo2_pct`, `temperature_c`, `lactate`, `wbc_count`, "
        "`crp_level`, `creatinine`, `hemoglobin`, `platelet_count`, `age`, "
        "`hour_from_admission`, `gender`, `admission_type`, `oxygen_device`"
    )

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded is not None:
        df_raw = pd.read_csv(uploaded)
        df_raw.columns = df_raw.columns.str.strip().str.lower()
        st.write(f"**Rows:** {len(df_raw)}  |  **Columns:** {df_raw.shape[1]}")
        st.dataframe(df_raw.head(), use_container_width=True)

        if st.button("⚡ Run Batch Prediction", type="primary"):
            results = []
            for _, row in df_raw.iterrows():
                try:
                    feat = build_feature_row(row.to_dict())
                    try:
                        scaled = scaler.transform(feat.reindex(columns=scaler.feature_names_in_, fill_value=0))
                    except AttributeError:
                        scaled = scaler.transform(feat)
                    prob = model.predict_proba(scaled)[0, 1]
                    label, _ = risk_label(prob, OPTIMAL_THRESHOLD)
                    mews = calculate_mews(row.to_dict())
                    results.append({"probability": round(prob, 4), "risk_level": label, "mews": mews})
                except Exception as e:
                    results.append({"probability": None, "risk_level": f"Error: {e}", "mews": None})

            out_df = pd.concat([df_raw.reset_index(drop=True), pd.DataFrame(results)], axis=1)

            st.success("✅ Batch prediction complete!")
            st.dataframe(out_df[["probability", "risk_level", "mews"]].head(20), use_container_width=True)

            st.markdown("**Risk Distribution**")
            dist = pd.DataFrame(results)["risk_level"].value_counts()
            st.bar_chart(dist)

            csv_out = out_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Results CSV",
                data=csv_out,
                file_name="deterioration_predictions.csv",
                mime="text/csv",
            )

# ─────────────────────────────────────────────
# Sidebar — model info
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ℹ️ Model Info")
    st.markdown(
        """
        **Algorithm:** Random Forest (sklearn)  
        **Training:** 3-Fold Stratified CV + SMOTE  
        **ROC-AUC:** ~0.952  
        **Threshold:** Youden's Index  
        **Features:** Vitals · Labs · MEWS · Rolling stats · Clinical flags  
        """
    )
    st.divider()
    st.markdown("**Threshold used:** `{:.2f}`".format(OPTIMAL_THRESHOLD))
    st.divider()
    st.warning("⚠️ For clinical decision support only. Not a substitute for physician judgment.")