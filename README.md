# 💳 Credit Card Fraud Detection

> A machine learning project that detects fraudulent credit card transactions using both Supervised and Semi-Supervised learning models, following the CRISP-DM methodology.

---

## 📌 Project Overview

Credit card fraud is one of the most prevalent forms of identity theft. From 2017 through 2019, it was the most common type of identity theft reported, and it continues to be a leading threat. Key challenges in fraud detection include:

- No reliable method to detect fraud in a timely manner
- Anyone with card information can access and misuse it
- Fraud can typically only be tracked **after** it occurs

This project builds a robust fraud detection system by experimenting with multiple machine learning models to identify the best-performing approach for timely and accurate fraud categorization.

---

## 🔄 Methodology — CRISP-DM

The project follows the **CRISP-DM (Cross Industry Standard Process for Data Mining)** framework:

```
1. Business Understanding   → Scope definition & requirements
2. Data Understanding       → Dataset selection & EDA
3. Data Preparation         → Wrangling, transformation, feature extraction
4. Modeling                 → Supervised & Semi-Supervised ML models
5. Evaluation               → Performance comparison across models
6. Deployment               → (Future scope)
```

---

## 📂 Dataset

| Property | Details |
|----------|---------|
| **Source** | Sparkov Data Generation by Brandon Harris |
| **Format** | `.csv` |
| **Total Records** | 1,852,394 |
| **Features** | 20+ |
| **Dataset Size** | 501.59 MB |

### Data Wrangling Summary

| Stage | Shape |
|-------|-------|
| Raw file 1 | 555,719 × 22 |
| Raw file 2 | 1,296,675 × 22 |
| After merging | 1,852,394 × 22 |
| After cleaning + random sampling | 300,000 × 22 |
| After feature engineering | 300,000 × 25 |

---

## 🔧 Data Preprocessing Pipeline

```
Raw CSV Files
      ↓
  Data Merging & Cleaning
      ↓
  Exploratory Data Analysis (EDA)
      ↓
  Data Profiling
      ↓
  Feature Transformation & Extraction  (300,000 × 25)
      ↓
  SMOTE (handle class imbalance)
      ↓
  Train / Test Split
      ↓
  Model Building & Evaluation
```

### Key Preprocessing Steps

- **Data Cleaning** — removed nulls and inconsistencies across merged files
- **EDA** — analyzed daily, weekly, and monthly fraud trends
- **Feature Engineering** — expanded feature set from 22 → 25 columns
- **SMOTE** — Synthetic Minority Oversampling to handle imbalanced fraud labels
- **Standardization** — normalized numerical features for model compatibility
- **PCA** — applied for dimensionality reduction in semi-supervised pipeline
- **Clustering** — generated cluster labels added as additional features

---

## 🤖 Models

### Supervised Learning
Traditional labeled classification models trained on the processed dataset with SMOTE-balanced classes.

### Semi-Supervised Learning
Combined unsupervised clustering (PCA + K-Means) with supervised classification — useful when labeled data is limited. Cluster assignments were incorporated as additional features before model training.

---

## 📊 Evaluation

Models were evaluated using standard classification metrics. Both supervised and semi-supervised approaches were compared to identify the best-performing model for fraud detection.

Key evaluation criteria:

| Metric | Purpose |
|--------|---------|
| **Precision** | Minimize false fraud alerts |
| **Recall** | Catch as many real fraud cases as possible |
| **F1-Score** | Balance between precision and recall |
| **ROC-AUC** | Overall discriminative ability of the model |

---

## ⚠️ Limitations

- Dataset is synthetically generated and may not fully capture real-world transaction patterns
- Random sampling to 300,000 records may introduce sampling bias
- Fraud patterns evolve over time — models may require periodic retraining

---

## 🚀 Future Scope

- Real-time fraud detection pipeline using streaming data
- Deep learning approaches (e.g., LSTM for sequential transaction patterns)
- Deployment as an API or microservice for integration with banking systems
- Continuous model retraining with incoming transaction data

---

## 📚 References

1. [IEEE — 9835751](https://ieeexplore.ieee.org/document/9835751)
2. [IEEE — 9057851](https://ieeexplore.ieee.org/document/9057851)
3. [IEEE — 9655848](https://ieeexplore.ieee.org/document/9655848)
4. [IEEE — 9197762](https://ieeexplore.ieee.org/document/9197762)
5. [IEEE — 9432308](https://ieeexplore.ieee.org/document/9432308)
6. [IEEE — 9751922](https://ieeexplore.ieee.org/document/9751922)
7. [IEEE — 9155615](https://ieeexplore.ieee.org/document/9155615)
8. [IEEE — 9676262](https://ieeexplore.ieee.org/document/9676262)
9. [IEEE — 9243602](https://ieeexplore.ieee.org/document/9243602)
