# Oral Cancer Prediction

A machine learning project that predicts oral cancer diagnosis from patient health and lifestyle data using Logistic Regression and Random Forest classifiers.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
- [Results](#results)
- [Limitations & Next Steps](#limitations--next-steps)

---

## Overview

This project walks through a full machine learning pipeline — from data cleaning and exploratory data analysis (EDA) through model training and evaluation — to predict whether a patient has oral cancer based on a set of clinical and lifestyle features.

Two binary classification models are trained and compared:
- **Logistic Regression**
- **Random Forest**

---

## Dataset

**File:** `synthetic_oral_cancer_dataset_noisy.csv`

The dataset is **synthetic and intentionally noisy**, simulating real-world imperfections. It contains demographic, lifestyle, and clinical symptom features along with a binary oral cancer label.

### Features

| Feature | Type | Description |
|---|---|---|
| `age` | Numerical | Patient age |
| `gender` | Categorical | Gender (Male / Female) |
| `region` | Categorical | Urban / Semi-Urban / Rural |
| `education_level` | Categorical | Primary / Secondary / Tertiary |
| `smoking` | Binary | Whether the patient smokes |
| `alcohol_use` | Binary | Whether the patient uses alcohol |
| `betel_nut_use` | Binary | Whether the patient chews betel nut |
| `poor_oral_hygiene` | Binary | Poor oral hygiene indicator |
| `diet_quality` | Categorical | Good / Average / Poor |
| `oral_lesions` | Binary | Presence of oral lesions |
| `bleeding_gums` | Binary | Presence of bleeding gums |
| `persistent_sore_throat` | Binary | Persistent sore throat indicator |
| `numbness_mouth` | Binary | Numbness in the mouth |
| `difficulty_chewing` | Binary | Difficulty chewing |
| `family_history_cancer` | Binary | Family history of cancer |
| `regular_dental_visits` | Binary | Regular dental check-up attendance |
| `oral_cancer` | Binary | **Target variable** (0 = No, 1 = Yes) |

### Missing Values

Several features had missing values prior to cleaning:

| Feature | Missing Count |
|---|---|
| `education_level` | 16,351 |
| `persistent_sore_throat` | 4,254 |
| `bleeding_gums` | 4,326 |
| `diet_quality` | 4,250 |

---

## Project Structure

```
oral-cancer-prediction/
│
├── oral_cancer_prediction.ipynb          # Main Jupyter Notebook
├── synthetic_oral_cancer_dataset_noisy.csv  # Dataset (required)
├── oral_cancer_distribution.png          # EDA plot: target distribution
├── correlation_matrix.png                # EDA plot: feature correlations
└── README.md
```

---

## Installation

### Prerequisites

- Python 3.8+
- Jupyter Notebook or JupyterLab

### Install Dependencies

```bash
pip install pandas matplotlib seaborn scikit-learn
```

---

## Usage

1. Clone or download this repository.
2. Place `synthetic_oral_cancer_dataset_noisy.csv` in the same directory as the notebook.
3. Launch Jupyter Notebook:

```bash
jupyter notebook oral_cancer_prediction.ipynb
```

4. Run all cells from top to bottom.

---

## Methodology

### 1. Data Cleaning & Preprocessing

- **Missing value imputation:**
  - Categorical columns → filled with the **mode**
  - Numerical columns → filled with the **mean**
- **One-hot encoding** applied to all categorical features (`gender`, `region`, `education_level`, `diet_quality`) with `drop_first=True` to avoid multicollinearity.

### 2. Exploratory Data Analysis (EDA)

- Distribution of the target variable (`oral_cancer`)
- Correlation heatmap across all numerical features

### 3. Model Training

The dataset is split **80% training / 20% testing** with stratification on the target variable (`random_state=42`).

Two models are trained:

**Logistic Regression**
```python
LogisticRegression(max_iter=1000, random_state=42)
```

**Random Forest**
```python
RandomForestClassifier(random_state=42)
```

### 4. Evaluation

Both models are evaluated using scikit-learn's `classification_report`, which reports precision, recall, F1-score, and accuracy per class.

---

## Results

### Logistic Regression

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| 0 (No Cancer) | 0.57 | 0.99 | 0.72 | 9,670 |
| 1 (Cancer) | 0.51 | 0.01 | 0.02 | 7,315 |
| **Accuracy** | | | **0.57** | 16,985 |

> Logistic Regression collapses almost entirely to predicting the majority class, resulting in near-zero recall for the positive (cancer) class.

### Random Forest

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| 0 (No Cancer) | 0.58 | 0.67 | 0.62 | 9,670 |
| 1 (Cancer) | 0.45 | 0.35 | 0.39 | 7,315 |
| **Accuracy** | | | **0.53** | 16,985 |

> Random Forest shows more balanced predictions but still achieves modest performance overall.

---

## Limitations & Next Steps

Current model performance is limited due to several factors. Potential improvements include:

1. **Data Quality** — The dataset is intentionally noisy. Improved data cleaning or noise reduction could boost model learning.

2. **Feature Engineering** — Creating interaction terms or derived features may reveal stronger signals in the data.

3. **Hyperparameter Tuning** — Use `GridSearchCV` or `RandomizedSearchCV` to optimise model parameters.

4. **Class Imbalance Handling** — Techniques such as SMOTE (oversampling) or class-weight adjustments may improve recall on the minority class.

5. **Alternative Models** — Gradient boosting frameworks (XGBoost, LightGBM, CatBoost) often outperform basic classifiers on noisy tabular data.

---

## License

This project is for educational purposes and uses a synthetic dataset. No real patient data is involved.
