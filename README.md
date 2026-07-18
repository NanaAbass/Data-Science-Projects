# 🏥 Insurance Risk ML — Streamlit App
**ALX Data Club · End-to-End Insurance Risk Prediction Dashboard**

---

## Setup & Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch the app
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

---

## Usage

1. **Upload** your `insurance.csv` using the sidebar file uploader.
2. The full ML pipeline runs automatically (cached after first run — ~30 seconds).
3. Navigate between sections using the sidebar radio buttons.

---

## App Sections

| Page | What You'll See |
|---|---|
| 🏠 Overview | Dataset stats, risk distribution, pipeline summary |
| 📊 EDA | Numerical/categorical distributions, bivariate analysis, correlation heatmap, outlier plots |
| 🎯 Target Engineering | Risk Score formula, threshold visualisation, category breakdown |
| ⚙️ Preprocessing | Feature engineering details, encoding strategy, scaling explanation, SMOTE before/after |
| 🤖 Model Training & CV | Model overview, 5-fold CV results chart + table, ensemble info |
| 📈 Evaluation | Leaderboard, per-model confusion matrix, ROC curves, classification report |
| 🔍 Feature Importance | Tree-based importance bar chart + ranked table (selectable model) |
| 🧮 Risk Predictor | Live single-customer risk scoring with score breakdown gauge |

---

## Expected CSV Columns

```
Customer_ID, Full_Name, Age, Gender, Smoker, BMI, Grade_Level,
Monthly_Income_GHS, Premium_GHS, Tenure_Months, Claim_Frequency,
Claim_Severity, Payment_Behavior, Region, Product_Applied,
Marital_Status, Payment_Method, Policy_Status, Application_Date,
Agent_ID, Agent_Name, Dependents (optional)
```

---

## Risk Score Formula (Target Engineering)

| Variable | Max Pts | Rule |
|---|---|---|
| Age | 20 | <30→5 · 30–44→10 · 45–59→15 · 60+→20 |
| BMI | 20 | <18.5→5 · 18.5–24.9→10 · 25–29.9→15 · ≥30→20 |
| Smoker | 15 | No→0 · Yes→15 |
| Claim Frequency | 20 | Each claim = 5 pts (max 20) |
| Claim Severity | 15 | None→0 · Low→3 · Med→7 · High→11 · Critical→15 |
| Payment Behavior | 10 | Consistent→0 · Occ. Late→5 · Freq. Late→10 |

**Thresholds:** LOW <35 · MEDIUM 35–51 · HIGH 52–67 · CRITICAL ≥68

---

## ML Pipeline

1. Load & clean (dedup, drop IDs)
2. Engineer target (Risk Score → 4 categories)
3. Feature engineering (10 new features + interactions)
4. Encode (binary, ordinal, one-hot)
5. Cap outliers (IQR winsorisation)
6. Log-transform skewed columns
7. 80/20 stratified train/test split
8. StandardScaler (fit on train only)
9. SMOTE (train only)
10. 5-fold StratifiedKFold CV
11. Train 6 models (LR, DT, RF, GB, XGB, LGBM)
12. Evaluate (accuracy, F1-macro, ROC-AUC)
13. Soft voting ensemble (top 3 CV models)
