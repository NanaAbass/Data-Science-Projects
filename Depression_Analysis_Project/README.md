# **Depression Detection using Actigraphy Data and Machine Learning**

This project analyzes **actigraphy** (wearable movement) data to uncover behavioral patterns linked to depression and build predictive models for classifying depression risk. By examining differences in daytime activity (often reduced) and nighttime activity (often increased) in depressed individuals, we aim to validate clinical observations and develop practical, automated screening tools.

## Project Overview

Depression is a global health challenge affecting more than **300 million people** and ranking as a leading cause of disability worldwide. Research consistently shows that depression is associated with distinct motor activity patterns observable through actigraphy — such as lower overall daytime movement and disrupted sleep/wake cycles.

This project uses data science and machine learning to:
- Confirm these behavioral signatures in real-world data
- Extract interpretable features from raw minute-level actigraphy
- Train and compare classification models
- Provide an interactive interface for visualization and prediction

## Objectives

- **Pattern Discovery**: Identify key actigraphy-based behavioral markers that differentiate depressed from non-depressed individuals
- **Exploratory Data Analysis (EDA)**: Examine relationships among demographic, clinical, and activity variables
- **Predictive Modeling**: Develop and evaluate machine learning classifiers (Random Forest, SVM, and potentially others) for depression risk prediction
- **User-Friendly Delivery**: Create an interactive **Streamlit** dashboard for data exploration, model interpretation, and real-time classification on new data

## Dataset

The project is based on a well-known public dataset available on Kaggle: **[The Depression Dataset](https://www.kaggle.com/datasets/arashnic/the-depression-dataset)**.

### Groups
- **Control Group** — 32 non-depressed individuals (23 hospital employees, 5 students, 4 former patients with no current symptoms)
- **Condition Group** — Individuals diagnosed with unipolar depression, bipolar disorder (I/II), or related mood disorders (some hospitalized)

### Data Structure
Raw actigraphy files (one CSV per participant) contain minute-level recordings with:
- `timestamp`: Date and time (one-minute intervals)
- `date`: Calendar date
- `activity`: Raw activity count from the wrist-worn actigraph

Additional metadata (`scores.csv`):
- Participant demographics (age group, gender, education, marital status, employment)
- Clinical variables (diagnosis type, melancholia, inpatient/outpatient status)
- MADRS depression severity scores (at start and end of recording)

## Methodology

### 1. Data Preprocessing & Feature Engineering
- Handle heavily right-skewed activity counts (many zero-activity minutes)
- Apply **log transformation** (or alternatives like Yeo-Johnson) to stabilize variance
- Aggregate minute-level data into daily/nightly behavioral features, e.g.:
  - Total activity, mean activity
  - Daytime vs. nighttime activity ratios
  - Activity onset/offset times
  - Variability metrics (std, entropy)
  - Circadian rhythm indicators

### 2. Machine Learning
Implemented and compared:
- **Random Forest** — Excellent for feature importance ranking and robustness to noise
- **Support Vector Machine (SVM)** — Strong performance in high-dimensional settings

Additional steps:
- Cross-validation
- Hyperparameter tuning
- Evaluation metrics: Accuracy, Precision, Recall, F1-score, ROC-AUC
- Feature importance analysis to highlight the most discriminative actigraphy patterns

### 3. Interactive Dashboard
A **Streamlit** web app (`depression_dashboard.py`) enables users to:
- Upload new actigraphy CSV files
- Visualize distributions, time-series plots, and group comparisons
- View feature importance and model explanations
- Obtain instant depression risk predictions for uploaded data

## Requirements

```text
Python >= 3.8
pandas
numpy
matplotlib
seaborn
scikit-learn
streamlit
# Optional / depending on extensions:
jupyter          # for notebook exploration
imbalanced-learn # if class imbalance handling is needed

Install via:bash

pip install -r requirements.txt

**How to Run**

1. Exploratory Analysis & ModelingOpen the Jupyter notebook:bash

jupyter notebook Depression-analysis.ipynb

(or use VS Code, Google Colab, etc.)This notebook covers the full pipeline:Data loading & merging
EDA visualizations
Feature engineering
Model training & evaluation
Results interpretation

2. Launch Interactive DashboardFrom the project root:bash

streamlit run depression_dashboard.py

Open your browser at the displayed local URL (usually http://localhost:8501).Upload a new participant's activity CSV to see live predictions and visualizations.

**Future Improvements**
Add more advanced models (XGBoost, LightGBM, neural networks)
Incorporate time-series specific approaches (LSTM, Transformer)
Explore additional features (sleep efficiency, fragmentation indices)
Test on external datasets for generalizability
Deploy dashboard to Streamlit Cloud or similar

**License**
MIT License (or specify your preferred license)Acknowledgments

**Dataset:** Originally shared on Kaggle by arashnic (based on earlier psychiatric motor activity research)

**Inspiration:**
Clinical literature on actigraphy in mood disorders

Contributions, issues, and pull requests are welcome!