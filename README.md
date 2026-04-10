# 📊 Customer Churn Prediction & Sales Dashboard

> A full-stack **Data Analytics + Machine Learning** web application that predicts customer churn probability in real time using supervised learning models, deployed on Streamlit Cloud.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31.0-FF4B4B?logo=streamlit&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.4.0-F7931E?logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0.3-0073B7)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🚀 Live Demo

> **[Launch Dashboard →](https://customer-churn-prediction-zyee2026.streamlit.app/)**


---

## 📌 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Model Performance](#-model-performance)
- [Dashboard Preview](#-dashboard-preview)
- [Deployment](#-deployment)
- [Tech Stack](#-tech-stack)
- [Future Enhancements](#-future-enhancements)
- [License](#-license)

---

## 🧠 Overview

Customer churn is one of the most critical challenges for banks and subscription businesses. This project builds an end-to-end ML pipeline that:

1. Preprocesses and encodes 10,000 bank customer records
2. Handles class imbalance using **SMOTE** oversampling
3. Trains and compares **3 ML models** (Logistic Regression, Random Forest, XGBoost)
4. Selects the best model based on AUC-ROC score
5. Deploys an **interactive 4-tab Streamlit dashboard** with live prediction

---

## ✨ Features

| Tab | What You Get |
|-----|-------------|
| 🏠 **Overview** | KPI cards, churn by location / gender / products |
| 📈 **EDA** | Age, balance, credit score charts + correlation heatmap |
| 🤖 **Model Performance** | Confusion matrix, ROC curve, feature importance |
| 🔮 **Live Prediction** | Enter customer details → instant churn probability + gauge chart |

---

## 📂 Dataset

**File:** `data/Customer-Churn-Records.csv`  
**Records:** 10,000 bank customers | **Features:** 17 | **Target:** `Exited` (1 = Churned)

| Feature | Type | Description |
|---------|------|-------------|
| `CreditScore` | Numerical | Customer credit score (300–850) |
| `Location` | Categorical | France / Germany / Spain |
| `Gender` | Categorical | Female / Male |
| `Age` | Numerical | Customer age (18–92) |
| `Tenure` | Numerical | Years with the bank (0–10) |
| `Account Balance` | Numerical | Account balance in USD |
| `NumOfProducts` | Numerical | Number of bank products used |
| `HasCrCard` | Binary | Has credit card (0/1) |
| `IsActiveMember` | Binary | Active member status (0/1) |
| `EstimatedSalary` | Numerical | Annual salary estimate |
| `Complain` | Binary | Filed a complaint (0/1) |
| `Satisfaction Score` | Ordinal | Score from 1 (low) to 5 (high) |
| `Card Type` | Categorical | DIAMOND / GOLD / PLATINUM / SILVER |
| `Point Earned` | Numerical | Loyalty points earned |
| `Exited` *(target)* | Binary | **1 = Churned, 0 = Retained** |

**Class distribution:** 79.62% Retained · 20.38% Churned

---

## 🗂 Project Structure

```
customer-churn-prediction/
│
├── data/
│   └── Customer-Churn-Records.csv   # Raw dataset
│
├── preprocess.py                    # Data cleaning, encoding, scaling
├── train_model.py                   # Model training, evaluation, saving
├── app.py                           # Streamlit dashboard (4 tabs)
├── requirements.txt                 # Python dependencies
│
├── churn_model.pkl                  # Saved best model (generated)
├── scaler.pkl                       # Fitted StandardScaler (generated)
├── feature_columns.pkl              # Feature column list (generated)
├── feature_importance.png           # RF feature importance chart (generated)
├── confusion_matrix.png             # Confusion matrix plot (generated)
│
└── README.md
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/customer-churn-prediction.git
cd customer-churn-prediction
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

Run the scripts **in order**:

### Step 1 — Preprocess the data

```bash
python preprocess.py
```

**Expected output:**

```
Scaler saved      : scaler.pkl
Columns saved     : feature_columns.pkl

--- Preprocessing Complete ---
Dataset shape  : (10000, 14)
Churn rate     : 20.38%
Features (14)  : ['CreditScore', 'Location', 'Gender', ...]
```

---

### Step 2 — Train the models

```bash
python train_model.py
```

**Expected output:**

```
Train size : 8000 rows
Test size  : 2000 rows
After SMOTE — Train size: 12740 rows

Logistic Regression   AUC-ROC: 0.9985
Random Forest         AUC-ROC: 0.9989  ✓ BEST
XGBoost               AUC-ROC: 0.9975

Model saved   : churn_model.pkl
Chart saved   : feature_importance.png
Chart saved   : confusion_matrix.png
```

---

### Step 3 — Launch the dashboard

```bash
streamlit run app.py
```

Opens at **<http://localhost:8501>** in your browser.

---

## 📊 Model Performance

All three models were trained on SMOTE-balanced data and evaluated on a held-out test set of 2,000 records.

| Model | Accuracy | AUC-ROC | F1-Score | Selected |
|-------|----------|---------|----------|----------|
| Logistic Regression | 99.80% | 0.9985 | 1.00 | |
| **Random Forest** | **99.85%** | **0.9989** | **1.00** | ✅ |
| XGBoost | 99.70% | 0.9975 | 1.00 | |

> **Note:** Near-perfect scores are driven by the `Complain` feature, which is highly correlated with churn in this dataset — a realistic signal in banking data.

---

## 🖥 Dashboard Preview

### 🏠 Overview Tab

- Total customers, churn count, retention rate KPIs
- Churn distribution pie chart
- Churn rate by location, gender, and number of products

### 📈 EDA Tab

- Age and credit score distributions by churn status
- Account balance and salary boxplots
- Satisfaction score vs churn bar chart
- Full feature correlation heatmap
- Raw data table preview

### 🤖 Model Performance Tab

- Accuracy, AUC-ROC, F1, Precision, Recall metric cards
- Interactive confusion matrix heatmap
- ROC curve with AUC annotation
- Feature importance ranking chart
- Predicted probability distribution

### 🔮 Live Prediction Tab

- 3-column customer input form (13 features)
- Real-time churn probability calculation
- Risk classification: 🟢 Low / 🟡 Medium / 🔴 High
- Animated gauge chart
- Customer feature summary table

---

## ☁️ Deployment

### Deploy to Streamlit Cloud (Free)

1. Push your project to a **public GitHub repository**

```bash
git init
git add .
git commit -m "Initial commit — Customer Churn Prediction App"
git remote add origin https://github.com/YOUR_USERNAME/customer-churn-prediction.git
git push -u origin main
```

1. Go to **[share.streamlit.io](https://share.streamlit.io)** and sign in with GitHub

2. Click **New app** → select your repo → set main file to `app.py`

3. Click **Deploy** — Streamlit installs dependencies automatically from `requirements.txt`

4. Your app gets a permanent public URL:  
   `https://YOUR_USERNAME-customer-churn-prediction.streamlit.app`

---

## 🛠 Tech Stack

| Category | Technology |
|----------|-----------|
| Language | Python 3.10+ |
| Data Processing | Pandas 2.1.4, NumPy 1.26.3 |
| Machine Learning | Scikit-learn 1.4.0, XGBoost 2.0.3 |
| Class Balancing | Imbalanced-learn 0.12.0 (SMOTE) |
| Visualization | Plotly 5.18.0, Matplotlib 3.8.2, Seaborn 0.13.2 |
| Dashboard | Streamlit 1.31.0 |
| Model Persistence | Joblib 1.3.2 |
| Deployment | Streamlit Cloud |

---

## 🔮 Future Enhancements

- [ ] Time-series churn modeling using LSTM / RNN
- [ ] Customer segmentation with K-Means and DBSCAN
- [ ] SHAP explainability for per-customer feature attribution
- [ ] Live CRM database integration via API
- [ ] Automated email alerts for high-risk customers
- [ ] A/B testing module for retention strategy simulation

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- Dataset: Customer Churn Records (bank customer behavioral data)
- Built with [Streamlit](https://streamlit.io), [Scikit-learn](https://scikit-learn.org), and [Plotly](https://plotly.com)

---

<p align="center">Made with ❤️ for Data Analytics</p>
