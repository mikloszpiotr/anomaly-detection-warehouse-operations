
# 📦 Anomaly Detection in Warehouse Operations

This project is part of a professional portfolio demonstrating how machine learning can be applied to warehouse operations to detect operational anomalies using historical sensor and task log data.

---

## 📌 Business Objective

**Goal:** Identify abnormal warehouse tasks (e.g., picking, packing, scanning) to improve operational efficiency and prevent potential delays or errors.

**Users:** Operations managers, warehouse supervisors, and supply chain analysts.

**ML Objective:** Unsupervised anomaly detection using Isolation Forest.

**Success Criteria:** High anomaly precision (Precision@N), explainable detection logic, and real-time visibility.

---

## 🧾 Data Collection

Synthetic dataset simulating warehouse activity logs:
- `warehouse_logs.csv` contains:
  - `worker_id`, `task_type`, `zone_id`
  - `duration_sec`, `errors`, `device_lag_ms`, `scanner_failures`
  - `timestamp`, `is_anomaly` (for evaluation)

---

## 🔍 Exploratory Data Analysis

EDA notebook explores:
- Class balance between normal and anomaly
- Device lag vs task duration (highlighting anomalies)
- Distributions of zones, task types, and error frequency

📁 Notebook: `notebooks/01_eda_anomaly_detection.ipynb`

---

## 🧼 Data Preprocessing

- Timestamp features: `hour`, `dayofweek`
- Categorical encoding: OneHot (task type, zone)
- Numerical scaling: StandardScaler
- Missing value imputation

🧱 Logic in: `src/features/build_features.py`

---

## 🤖 Model Training

Model: **Isolation Forest**
- Trained on feature matrix using:
  - `duration`, `lag`, `errors`, etc.
- Auto-labeled anomalies: output flag in app

🏗 Script: `src/models/train_iforest.py`  
🧠 Trained file: `models/iforest_model.pkl`

---

## 💻 Streamlit App

Main dashboard file: `app.py`

**App features:**
- Upload warehouse logs (.csv)
- Detect anomalies (visual + table)
- Download results with anomaly flags

---

## 📁 Project Structure

```
anomaly_detection_warehouse/
├── data/
│   ├── raw/              # warehouse_logs.csv
│   └── processed/        # (optional exports)
├── models/               # Trained Isolation Forest
├── notebooks/            # EDA visualizations
├── src/
│   ├── data/             # (optional loaders)
│   ├── features/         # Feature builder
│   └── models/           # Training script
├── app.py                # Streamlit UI
└── README.md
```

---

## 🚀 How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

---

## 📈 Portfolio Value

This project highlights:
- End-to-end anomaly detection pipeline
- Realistic supply chain sensor simulation
- Production-ready Streamlit dashboard
