# 🏥 VitalWatch
### AI-Powered Patient Deterioration Predictor

> Early warning system that predicts physiological deterioration **up to 12 hours in advance**

[![Presentation](https://img.shields.io/badge/Canva-View%20Presentation-blueviolet?logo=canva)](https://canva.link/c2fq69aujnse7tm)

---

## ✨ What It Does

VitalWatch takes a patient's current vital signs and lab values, runs them through a trained ML model, and outputs a **deterioration risk score** with clinical flags highlighting which thresholds are breached.

| | |
|---|---|
| 🤖 **Best Model** | LightGBM |
| 📊 **AUROC** | 0.949 (95% CI: 0.946 – 0.952) |
| ⚖️ **Training** | 3-Fold Stratified CV + SMOTE |
| 🎯 **Decision Threshold** | 0.347 (Youden's Index) |
| ⏱️ **Prediction Window** | 12-hour deterioration horizon |
| 🗃️ **Dataset Size** | 293,248 training records |

---

## 🧪 Model Comparison

Four models were trained and evaluated; LightGBM was selected for deployment based on highest AUROC and sensitivity.

| Model | Description | AUROC |
|---|---|---|
| Logistic Regression | Baseline linear model, fast and interpretable | Lowest |
| Random Forest | Bagging ensemble, robust to noise | Moderate |
| XGBoost | Gradient boosting, captures non-linear patterns | High |
| **LightGBM ✅** | **Efficient gradient boosting, best on tabular data** | **~0.949** |

**Evaluation metrics used:** AUROC · AUPRC · F1 Score · Sensitivity · Specificity

---

## 📈 Key Results

| Metric | Value |
|---|---|
| Accuracy | 89.18% |
| Recall | **85.58%** |
| Precision | 31.54% |
| F1 Score | 46.09% |
| AUROC | **0.949** |

> **Why high recall matters here:** Missing a deteriorating patient is far riskier than a false alarm. The model is intentionally tuned to catch as many true cases as possible.

---

## 🖥️ App Features

### 🔬 Single Patient Mode
Enter vitals manually and get an instant prediction:
- Deterioration probability (0–100%)
- Risk level: 🟢 Low / 🟡 Medium / 🔴 High
- MEWS (Modified Early Warning Score)
- Active clinical flag alerts (hypotension, tachycardia, low SpO₂, etc.)

### 📋 Batch Mode (CSV Upload)
Upload a CSV of multiple patients and get:
- Risk scores for every row
- Risk distribution bar chart
- Downloadable results CSV

---

## 🚀 Getting Started

### 1. Clone & install dependencies

```bash
git clone https://github.com/your-username/vitalwatch.git
cd vitalwatch
pip install -r requirements.txt
```

### 2. Add the model artifacts

Make sure `model.pkl` and `scaler.pkl` are in the same directory as `app.py`.

```
vitalwatch/
├── app.py
├── model.pkl       ← required
├── scaler.pkl      ← required
└── requirements.txt
```

### 3. Launch the app

```bash
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 📋 Input Features

### Vitals
| Field | Unit |
|---|---|
| Heart Rate | bpm |
| Systolic / Diastolic BP | mmHg |
| Respiratory Rate | /min |
| SpO₂ | % |
| Temperature | °C |

### Labs
| Field | Unit |
|---|---|
| Lactate | mmol/L |
| WBC Count | /µL |
| CRP Level | mg/L |
| Creatinine | mg/dL |
| Hemoglobin | g/dL |
| Platelets | ×10³/µL |

### Patient Info
- Age, Hours from Admission
- Gender · Admission Type · Oxygen Device

> **Top predictive features:** Lactate · Respiratory Rate · SpO₂ · Systolic BP · Creatinine · Comorbidity Index

---

## 🧠 System Workflow

```
Patient Vitals & Labs
        ↓
  Data Cleaning, Encoding & Scaling
        ↓
  Feature Engineering
  (Trends, Rolling Averages, Risk Flags, MEWS)
        ↓
  LightGBM Model
        ↓
  Risk Score + SHAP Explainability
        ↓
  Clinician Dashboard & Alerts
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| `streamlit` | Dashboard UI |
| `lightgbm` / `scikit-learn` | ML models |
| `pandas` / `numpy` | Data processing |
| `SHAP` | Model explainability |
| `joblib` | Model serialization |

---

## ⚠️ Disclaimer

VitalWatch is intended for **clinical decision support only**. It is not a substitute for physician judgment.

---
