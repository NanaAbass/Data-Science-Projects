import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score, f1_score, accuracy_score, roc_curve, auc
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# ── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Insurance Risk Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── THEME COLOURS ────────────────────────────────────────────────────────────
CAT_CLR = {
    "Low":      "#2ecc71",
    "Medium":   "#f39c12",
    "High":     "#008080",
    "Critical": "#e74c3c",
}
PALETTE = ["#2ecc71", "#008080", "#f39c12", "#e74c3c"]

# ── CUSTOM CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* sidebar */
section[data-testid="stSidebar"] { background: #0f1923; }
section[data-testid="stSidebar"] * { color: #e0e0e0 !important; }
section[data-testid="stSidebar"] .stRadio label { font-size: 0.95rem; }

/* metric cards */
div[data-testid="metric-container"] {
    background: #1e2a38;
    border: 1px solid #2e3d50;
    border-radius: 10px;
    padding: 14px 18px;
}
div[data-testid="metric-container"] label { color: #8fa8c8 !important; }
div[data-testid="metric-container"] div   { color: #ffffff !important; }

/* tabs */
button[data-baseweb="tab"] { font-size: 0.92rem; font-weight: 600; }

/* risk badge */
.risk-badge {
    display: inline-block;
    padding: 6px 18px;
    border-radius: 20px;
    font-weight: 700;
    font-size: 1.1rem;
    color: #fff;
    margin-top: 6px;
}
.badge-Low      { background: #2ecc71; }
.badge-Medium   { background: #f39c12; }
.badge-High     { background: #008080; }
.badge-Critical { background: #e74c3c; }

/* divider */
hr { border-color: #2e3d50; }
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# PIPELINE HELPERS  (cached so they run once per session)
# ════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_and_preprocess(uploaded):
    """Load CSV, clean, engineer features, encode, return processed df + raw df."""
    df = pd.read_csv(uploaded)

    # ── basic cleaning ───────────────────────────────────────────────────
    if "Customer_ID" in df.columns:
        df.drop(columns=["Customer_ID"], inplace=True)
    df.drop_duplicates(inplace=True)
    df["Application_Date"] = pd.to_datetime(df["Application_Date"], errors="coerce")

    df1 = df.drop(columns=["Full_Name", "Occupation", "Agent_ID", "Agent_Name"], errors="ignore")

    # ── target engineering ───────────────────────────────────────────────
    df1["Claim_Severity"] = df1["Claim_Severity"].fillna("None")

    age_sc   = np.select([df1["Age"] < 30, df1["Age"] < 45, df1["Age"] < 60], [5, 10, 15], 20)
    bmi_sc   = np.select([df1["BMI"] < 18.5, df1["BMI"] < 25, df1["BMI"] < 30], [5, 10, 15], 20)
    smk_sc   = np.where(df1["Smoker"] == "Yes", 15, 0)
    cf_sc    = (df1["Claim_Frequency"] * 5).clip(0, 20)
    sev_sc   = df1["Claim_Severity"].map({"None":0,"Low":3,"Medium":7,"High":11,"Critical":15}).fillna(0)
    beh_sc   = df1["Payment_Behavior"].map({"Consistent":0,"Occasionally Late":5,"Frequently Late":10})

    df1["Risk_Score"] = age_sc + bmi_sc + smk_sc + cf_sc + sev_sc + beh_sc
    df1["Risk_category"] = pd.cut(
        df1["Risk_Score"], bins=[0,34,51,67,101],
        labels=["Low","Medium","High","Critical"], include_lowest=True
    )

    # ── feature engineering ──────────────────────────────────────────────
    df1["Age_Group"]    = pd.cut(df1["Age"], bins=[0,30,45,60,100],
                                 labels=["Young","Middle Age","Senior","Elder"], include_lowest=True)
    df1["BMI_Category"] = pd.cut(df1["BMI"], bins=[0,18.5,25,30,100],
                                 labels=["Underweight","Normal","Overweight","Obese"], include_lowest=True)
    df1["Premium_Burden"]     = (df1["Premium_GHS"] / df["Monthly_Income_GHS"]).round(4)
    ref = pd.Timestamp("2026-01-01")
    df1["Policy_Age_in_Days"] = (ref - df["Application_Date"]).dt.days
    df1["Late_payment_flag"]  = (df1["Payment_Behavior"] == "Frequently Late").astype(int)
    df1["High_claim_flag"]    = (df1["Claim_Frequency"] >= 3).astype(int)
    df1["Is_smoker"]          = (df1["Smoker"] == "Yes").astype(int)
    df1["BMI_x_Age"]          = df1["BMI"] * df1["Age"]
    df1["Claim_x_Severity"]   = df1["Claim_Frequency"] * df1["Claim_Severity"].map(
        {"None":0,"Low":1,"Medium":2,"High":3,"Critical":4}).fillna(0)
    df1["Income_per_Dependent"] = df["Monthly_Income_GHS"] / (df.get("Dependents", pd.Series(1, index=df.index)) + 1)

    df1.drop(columns=["Age","Risk_Score","Claim_Severity"], inplace=True)

    # ── encoding ─────────────────────────────────────────────────────────
    df1["Gender_enc"] = (df1["Gender"] == "Female").astype(int)
    df1.drop(columns=["Gender","Smoker"], inplace=True)

    df1["Grade_level_enc"]      = df1["Grade_Level"].map({"Junior":0,"Mid":1,"Senior":2})
    df1["Payment_behavior_enc"] = df1["Payment_Behavior"].map({"Consistent":0,"Occasionally Late":1,"Frequently Late":2})
    df1["BMI_category_enc"]     = df1["BMI_Category"].astype(str).map({"Underweight":0,"Normal":1,"Overweight":2,"Obese":3})
    df1["Age_group_enc"]        = df1["Age_Group"].astype(str).map({"Young":0,"Middle Age":1,"Senior":2,"Elder":3})
    df1.drop(columns=["Grade_Level","Payment_Behavior","BMI_Category","Age_Group"], inplace=True)

    ohe_cols = ["Region","Product_Applied","Marital_Status","Payment_Method","Policy_Status"]
    df1 = pd.get_dummies(df1, columns=[c for c in ohe_cols if c in df1.columns], drop_first=True)
    bool_cols = df1.select_dtypes(include=bool).columns
    df1[bool_cols] = df1[bool_cols].astype(int)

    # ── outlier capping ──────────────────────────────────────────────────
    for col in ["BMI","Monthly_Income_GHS","Premium_GHS","Premium_Burden","BMI_x_Age","Income_per_Dependent"]:
        if col in df1.columns:
            Q1, Q3 = df1[col].quantile([0.25, 0.75])
            IQR = Q3 - Q1
            df1[col] = df1[col].clip(Q1-1.5*IQR, Q3+1.5*IQR)

    # ── log transforms ───────────────────────────────────────────────────
    log_tgts = [c for c in ["Monthly_Income_GHS","Premium_GHS","Premium_Burden","Policy_Age_in_Days"] if c in df1.columns]
    for col in log_tgts:
        df1[f"Log_{col}"] = np.log1p(df1[col].clip(lower=0))
    df1.drop(columns=log_tgts, inplace=True)

    # ── label encode target ──────────────────────────────────────────────
    le = LabelEncoder()
    df1["Risk_category_enc"] = le.fit_transform(df1["Risk_category"])

    return df, df1, le


@st.cache_data(show_spinner=False)
def run_pipeline(_df1, _le):
    """Scale, SMOTE, train all models, cross-validate, evaluate."""
    drop_cols = ["Risk_category","Risk_category_enc"]
    feat_cols = [c for c in _df1.columns if c not in drop_cols]
    X = _df1[feat_cols].select_dtypes(include=[np.number]).copy()
    y = _df1["Risk_category_enc"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_tr_sc = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_te_sc = pd.DataFrame(scaler.transform(X_test),      columns=X_test.columns,  index=X_test.index)

    smote = SMOTE(random_state=42, k_neighbors=5)
    X_tr_res, y_tr_res = smote.fit_resample(X_tr_sc, y_train)

    models = {
        "Logistic Regression":  LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs", random_state=42),
        "Decision Tree":        DecisionTreeClassifier(max_depth=8, min_samples_split=20, random_state=42),
        "Random Forest":        RandomForestClassifier(n_estimators=200, max_depth=10, class_weight="balanced", random_state=42, n_jobs=-1),
        "XGBoost":              XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, eval_metric="mlogloss", random_state=42, n_jobs=-1)        
    }

    cv    = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_res = {}
    for name, m in models.items():
        sc = cross_validate(m, X_tr_res, y_tr_res, cv=cv, scoring=["accuracy","f1_macro"], n_jobs=-1)
        cv_res[name] = {"acc_mean": sc["test_accuracy"].mean(), "acc_std": sc["test_accuracy"].std(),
                        "f1_mean": sc["test_f1_macro"].mean()}

    trained = {}
    for name, m in models.items():
        m.fit(X_tr_res, y_tr_res)
        trained[name] = m

    results = {}
    for name, m in trained.items():
        yp = m.predict(X_te_sc)
        ypr = m.predict_proba(X_te_sc)
        results[name] = {
            "name": name,
            "accuracy":  accuracy_score(y_test, yp),
            "f1_macro":  f1_score(y_test, yp, average="macro"),
            "roc_auc":   roc_auc_score(y_test, ypr, multi_class="ovr", average="macro"),
            "y_pred":    yp,
            "y_proba":   ypr,
            "report":    classification_report(y_test, yp, target_names=_le.classes_, output_dict=True),
        }

    cv_df    = pd.DataFrame(cv_res).T.sort_values("f1_mean", ascending=False)
    top3     = cv_df.head(3).index.tolist()
    ensemble = VotingClassifier(estimators=[(n, trained[n]) for n in top3], voting="soft", n_jobs=-1)
    ensemble.fit(X_tr_res, y_tr_res)
    yp_e  = ensemble.predict(X_te_sc)
    ypr_e = ensemble.predict_proba(X_te_sc)
    results["Voting Ensemble"] = {
        "name": "Voting Ensemble",
        "accuracy": accuracy_score(y_test, yp_e),
        "f1_macro": f1_score(y_test, yp_e, average="macro"),
        "roc_auc":  roc_auc_score(y_test, ypr_e, multi_class="ovr", average="macro"),
        "y_pred":   yp_e,
        "y_proba":  ypr_e,
        "report":   classification_report(y_test, yp_e, target_names=_le.classes_, output_dict=True),
    }

    return {
        "X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test,
        "X_tr_sc": X_tr_sc, "X_te_sc": X_te_sc, "X_tr_res": X_tr_res, "y_tr_res": y_tr_res,
        "scaler": scaler, "trained": trained, "cv_res": cv_res, "cv_df": cv_df,
        "results": results, "ensemble": ensemble, "top3": top3,
        "y_tr_before": y_train,
    }


# ════════════════════════════════════════════════════════════════════════════
# PLOT HELPERS
# ════════════════════════════════════════════════════════════════════════════

def fig_num_dist(df):
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cols = [c for c in num_cols if c not in ("Dependents",)][:8]
    n = len(cols)
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    axes = axes.flatten()
    for i, col in enumerate(cols):
        axes[i].hist(df[col].dropna(), bins=25, color="#3498db", edgecolor="white", alpha=0.85)
        axes[i].axvline(df[col].mean(),   color="#e74c3c", ls="--", lw=1.5, label=f"Mean {df[col].mean():.1f}")
        axes[i].axvline(df[col].median(), color="#2ecc71", ls="--", lw=1.5, label=f"Med {df[col].median():.1f}")
        axes[i].set_title(col, fontweight="bold", fontsize=9)
        axes[i].legend(fontsize=7)
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Numerical Feature Distributions", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig

def fig_cat_dist(df):
    cats = ["Gender","Smoker","Grade_Level","Policy_Status","Payment_Behavior","Marital_Status","Claim_Severity"]
    cats = [c for c in cats if c in df.columns]
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes = axes.flatten()
    for i, col in enumerate(cats):
        vc = df[col].value_counts()
        colors = sns.color_palette("pastel", len(vc))
        bars = axes[i].barh(vc.index, vc.values, color=colors)
        axes[i].set_title(col, fontweight="bold", fontsize=9)
        for bar, val in zip(bars, vc.values):
            axes[i].text(val + vc.max()*0.01, bar.get_y()+bar.get_height()/2, f"{val:,}", va="center", fontsize=8)
    axes[-1].set_visible(False)
    fig.suptitle("Categorical Feature Distributions", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig

def fig_risk_dist(df1, le):
    counts = df1["Risk_category"].value_counts().reindex(["Low","Medium","High","Critical"])
    risk_score_col = None
    # recompute for histogram
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    bars = axes[0].bar(counts.index, counts.values,
                       color=[CAT_CLR[c] for c in counts.index], edgecolor="white", lw=1.5)
    for bar, val in zip(bars, counts.values):
        axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+counts.max()*0.01,
                     f"{val:,}\n({val/len(df1)*100:.1f}%)", ha="center", fontsize=9)
    axes[0].set_title("Risk Category Distribution", fontweight="bold")
    axes[0].set_ylabel("Count")

    enc_vc = df1["Risk_category_enc"].value_counts().sort_index()
    axes[1].pie(enc_vc.values, labels=[le.classes_[i] for i in enc_vc.index],
                colors=PALETTE, autopct="%1.1f%%", startangle=140,
                wedgeprops=dict(edgecolor="white", linewidth=1.5))
    axes[1].set_title("Risk Category Pie", fontweight="bold")
    fig.tight_layout()
    return fig

def fig_smote(y_before, y_after, le):
    cats   = [le.classes_[i] for i in sorted(np.unique(y_before))]
    before = pd.Series(y_before).value_counts().sort_index()
    after  = pd.Series(y_after).value_counts().sort_index()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for ax, data, title in zip(axes, [before, after], ["Before SMOTE", "After SMOTE"]):
        bars = ax.bar(cats, data.values, color=PALETTE[:len(cats)], edgecolor="white")
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel("Count")
        for bar, val in zip(bars, data.values):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+data.max()*0.01,
                    f"{val:,}", ha="center", fontsize=9)
    fig.suptitle("Class Imbalance Treatment — SMOTE", fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig

def fig_cv_comparison(cv_res):
    cv_df = pd.DataFrame(cv_res).T.sort_values("f1_mean", ascending=False)
    fig, ax = plt.subplots(figsize=(10, 4))
    x    = np.arange(len(cv_df))
    bars = ax.bar(x, cv_df["f1_mean"], yerr=cv_df["acc_std"], capsize=5,
                  color=sns.color_palette("Blues_d", len(cv_df)), edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(cv_df.index, rotation=20, ha="right")
    ax.set_ylabel("CV F1-Score (Macro)")
    ax.set_title("5-Fold Cross-Validation F1-Score", fontweight="bold")
    ax.set_ylim(0, 1.1)
    for bar, val in zip(bars, cv_df["f1_mean"]):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                f"{val:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    fig.tight_layout()
    return fig

def fig_confusion(res, le):
    cm   = confusion_matrix(res["y_test_ref"], res["y_pred"])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(cm, display_labels=le.classes_)
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix\n{res['name']}", fontweight="bold", fontsize=10)
    fig.tight_layout()
    return fig

def fig_roc(res, le):
    fig, ax = plt.subplots(figsize=(5, 4))
    for j, cls in enumerate(le.classes_):
        y_bin = (res["y_test_ref"] == j).astype(int)
        fpr, tpr, _ = roc_curve(y_bin, res["y_proba"][:, j])
        ax.plot(fpr, tpr, lw=1.5, label=f"{cls} (AUC={auc(fpr,tpr):.3f})")
    ax.plot([0,1],[0,1],"k--", lw=0.8)
    ax.set_title(f"ROC Curves — {res['name']}", fontweight="bold", fontsize=10)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.legend(fontsize=7)
    fig.tight_layout()
    return fig

def fig_feature_importance(model, feature_names, model_name, top_n=20):
    if not hasattr(model, "feature_importances_"):
        return None
    fi = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False).head(top_n)
    fig, ax = plt.subplots(figsize=(10, 6))
    fi.sort_values().plot(kind="barh", ax=ax, color="steelblue", edgecolor="white")
    ax.set_title(f"Top {top_n} Feature Importances — {model_name}", fontweight="bold")
    ax.set_xlabel("Importance Score")
    fig.tight_layout()
    return fig

def fig_final_comparison(results):
    df = pd.DataFrame([
        {"Model": r["name"], "Accuracy": r["accuracy"],
         "F1 Macro": r["f1_macro"], "ROC-AUC": r["roc_auc"]}
        for r in results.values()
    ]).sort_values("F1 Macro", ascending=False)

    x = np.arange(len(df)); w = 0.27
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(x-w, df["Accuracy"],  w, label="Accuracy", color="#3498db", edgecolor="white")
    ax.bar(x,   df["F1 Macro"], w, label="F1 Macro",  color="#2ecc71", edgecolor="white")
    ax.bar(x+w, df["ROC-AUC"],  w, label="ROC-AUC",   color="#e74c3c", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(df["Model"], rotation=20, ha="right")
    ax.set_ylim(0, 1.1); ax.set_ylabel("Score")
    ax.set_title("Final Model Comparison", fontweight="bold", fontsize=13)
    ax.legend(); ax.axhline(0.9, color="grey", ls="--", lw=0.8, alpha=0.5)
    for i, (_, row) in enumerate(df.iterrows()):
        ax.text(i-w, row["Accuracy"]+0.012, f"{row['Accuracy']:.3f}", ha="center", fontsize=8)
        ax.text(i,   row["F1 Macro"]+0.012, f"{row['F1 Macro']:.3f}", ha="center", fontsize=8)
        ax.text(i+w, row["ROC-AUC"]+0.012,  f"{row['ROC-AUC']:.3f}",  ha="center", fontsize=8)
    fig.tight_layout()
    return fig


# ════════════════════════════════════════════════════════════════════════════
# RISK PREDICTOR  (single customer)
# ════════════════════════════════════════════════════════════════════════════

def compute_risk_score(age, bmi, smoker, claim_freq, claim_sev, payment_beh):
    age_sc  = 5 if age<30 else (10 if age<45 else (15 if age<60 else 20))
    bmi_sc  = 5 if bmi<18.5 else (10 if bmi<25 else (15 if bmi<30 else 20))
    smk_sc  = 15 if smoker=="Yes" else 0
    cf_sc   = min(claim_freq*5, 20)
    sev_map = {"None":0,"Low":3,"Medium":7,"High":11,"Critical":15}
    beh_map = {"Consistent":0,"Occasionally Late":5,"Frequently Late":10}
    score   = age_sc + bmi_sc + smk_sc + cf_sc + sev_map[claim_sev] + beh_map[payment_beh]
    cat     = "Low" if score<35 else ("Medium" if score<52 else ("High" if score<68 else "Critical"))
    breakdown = {
        "Age Score": age_sc, "BMI Score": bmi_sc, "Smoker Score": smk_sc,
        "Claim Freq Score": cf_sc, "Claim Severity Score": sev_map[claim_sev],
        "Payment Behavior Score": beh_map[payment_beh],
    }
    return score, cat, breakdown


# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.image("https://img.icons8.com/fluency/80/hospital.png", width=60)
    st.title("🏥 Insurance Risk ML")
    st.markdown("**ALX Data Club**")
    st.divider()

    page = st.radio(
        "Navigate",
        ["Overview",
         "EDA",
         "Target Engineering",
         "Preprocessing",
         "Model Training & CV",
         "Evaluation",
         "Feature Importance",
         "Risk Predictor"],
        label_visibility="collapsed"
    )
    st.divider()
    uploaded = st.file_uploader("Upload insurance.csv", type="csv")
    if uploaded:
        st.success("✅ File uploaded")

# ════════════════════════════════════════════════════════════════════════════
# GATE: require upload
# ════════════════════════════════════════════════════════════════════════════

if not uploaded:
    st.markdown("## 🏥 Insurance Risk ML Dashboard")
    st.info("👈 **Upload your `insurance.csv` file in the sidebar to get started.**")
    st.markdown("""
    ### What this app covers
    | Section | Content |
    |---|---|
    |  EDA | Distributions, correlations, outlier analysis |
    |  Target Engineering | Risk Score formula & category breakdown |
    |  Preprocessing | Outlier capping, log transforms, SMOTE |
    |  Model Training & CV | 6 models + 5-fold cross validation |
    |  Evaluation | Confusion matrices, ROC curves, leaderboard |
    |  Feature Importance | Tree-based importances |
    |  Risk Predictor | Live single-customer risk scoring |
    """)
    st.stop()

# ── Run pipeline (cached after first run) ───────────────────────────────────
with st.spinner("🔄 Running ML pipeline… (first run may take ~30 s)"):
    df_raw, df1, le = load_and_preprocess(uploaded)
    pipe = run_pipeline(df1, le)

# attach y_test reference to results for plotting
for name, res in pipe["results"].items():
    res["y_test_ref"] = pipe["y_test"]

best_name = max(pipe["results"], key=lambda k: pipe["results"][k]["f1_macro"] if k != "Voting Ensemble" else -1)


# ════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ════════════════════════════════════════════════════════════════════════════

if page == "🏠 Overview":
    st.title("🏥 Insurance Risk ML — Overview")
    st.markdown("**ALX Data Club · End-to-End ML Pipeline**")
    st.divider()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Records",   f"{len(df_raw):,}")
    c2.metric("Features",        f"{df_raw.shape[1]}")
    c3.metric("Risk Categories", "4")
    c4.metric("Models Trained",  "6 + Ensemble")
    c5.metric("Best F1 (Test)",  f"{max(r['f1_macro'] for r in pipe['results'].values()):.4f}")

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Dataset Sample")
        st.dataframe(df_raw.head(8), use_container_width=True, height=280)
    with col2:
        st.subheader("Risk Category Distribution")
        st.pyplot(fig_risk_dist(df1, le))

    st.divider()
    st.subheader("📋 Pipeline Summary")
    pipeline_steps = [
        ("1", "Data Loading & Cleaning", "Remove duplicates, handle missing values"),
        ("2", "Exploratory Data Analysis", "Distributions, correlations, outlier detection"),
        ("3", "Target Engineering", "Risk Score (0–100) → 4 risk categories"),
        ("4", "Feature Engineering", "Age groups, BMI categories, interaction terms"),
        ("5", "Encoding", "Binary, ordinal, and one-hot encoding"),
        ("6", "Outlier Capping", "IQR-based winsorisation on skewed columns"),
        ("7", "Log Transforms", "Reduce skewness in income / premium columns"),
        ("8", "Train/Test Split", "80/20 stratified split"),
        ("9", "StandardScaler", "Zero-mean, unit-variance scaling (no leakage)"),
        ("10", "SMOTE", "Synthetic oversampling on training set only"),
        ("11", "Cross Validation", "5-fold StratifiedKFold on resampled train"),
        ("12", "Model Training", "6 models: LR, DT, RF, GB, XGB, LGBM"),
        ("13", "Evaluation", "Accuracy, F1-Macro, ROC-AUC, confusion matrix"),
        ("14", "Ensemble", "Soft voting on top-3 CV models"),
    ]
    for step, name, desc in pipeline_steps:
        st.markdown(f"**Step {step}** · **{name}** — {desc}")


# ════════════════════════════════════════════════════════════════════════════
# PAGE: EDA
# ════════════════════════════════════════════════════════════════════════════

elif page == "📊 EDA":
    st.title("📊 Exploratory Data Analysis")
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Numerical Dist.", "Categorical Dist.", "Bivariate", "Correlation", "Outliers"]
    )

    with tab1:
        st.subheader("Numerical Feature Distributions")
        st.pyplot(fig_num_dist(df_raw))
        st.subheader("Descriptive Statistics")
        st.dataframe(df_raw.describe().round(2), use_container_width=True)

    with tab2:
        st.subheader("Categorical Feature Distributions")
        st.pyplot(fig_cat_dist(df_raw))

    with tab3:
        st.subheader("Bivariate Analysis")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Smoker vs Claim Severity**")
            df_raw_copy = df_raw.copy()
            df_raw_copy["Claim_Severity"] = df_raw_copy["Claim_Severity"].fillna("None")
            cross = pd.crosstab(df_raw_copy["Smoker"], df_raw_copy["Claim_Severity"], normalize="index") * 100
            order = ["None","Low","Medium","High","Critical"]
            cross = cross[[c for c in order if c in cross.columns]]
            fig, ax = plt.subplots(figsize=(6, 4))
            cross.plot(kind="bar", ax=ax, colormap="viridis", stacked=True, width=0.6)
            ax.set_xlabel("Smoker"); ax.set_ylabel("%")
            ax.tick_params(axis="x", rotation=0)
            ax.legend(title="Claim Severity", bbox_to_anchor=(1,1), fontsize=8)
            fig.tight_layout()
            st.pyplot(fig)
        with col2:
            st.markdown("**Avg Claim Frequency by Payment Behavior**")
            order2 = ["Consistent","Occasionally Late","Frequently Late"]
            means  = df_raw.groupby("Payment_Behavior")["Claim_Frequency"].mean().reindex(order2)
            fig, ax = plt.subplots(figsize=(6, 4))
            bars = ax.barh(means.index, means.values, color=["#2ecc71","#f39c12","#e74c3c"])
            for bar, val in zip(bars, means.values):
                ax.text(val+0.005, bar.get_y()+bar.get_height()/2, f"{val:.3f}", va="center", fontsize=9)
            ax.set_xlabel("Avg Claims")
            fig.tight_layout()
            st.pyplot(fig)

        st.markdown("**BMI by Claim Severity**")
        df_cs = df_raw[df_raw["Claim_Severity"].notna()].copy()
        sev_order = ["Low","Medium","High","Critical"]
        df_cs["Claim_Severity"] = pd.Categorical(df_cs["Claim_Severity"], categories=sev_order, ordered=True)
        fig, ax = plt.subplots(figsize=(8, 4))
        df_cs.boxplot(column="BMI", by="Claim_Severity", ax=ax, grid=False,
                      boxprops=dict(color="steelblue"), medianprops=dict(color="red", linewidth=2))
        ax.set_title("BMI by Claim Severity", fontweight="bold")
        ax.set_xlabel("Claim Severity"); ax.set_ylabel("BMI")
        plt.suptitle("")
        fig.tight_layout()
        st.pyplot(fig)

    with tab4:
        st.subheader("Correlation Matrix")
        num_df = df_raw.select_dtypes(include="number")
        fig, ax = plt.subplots(figsize=(10, 7))
        mask = np.triu(np.ones_like(num_df.corr(), dtype=bool))
        sns.heatmap(num_df.corr(), mask=mask, annot=True, fmt=".2f",
                    linewidths=0.5, cmap="crest", ax=ax)
        ax.set_title("Correlation Matrix", fontweight="bold")
        fig.tight_layout()
        st.pyplot(fig)

    with tab5:
        st.subheader("Outlier Detection (IQR)")
        cols_out = ["Age","BMI","Monthly_Income_GHS","Premium_GHS","Tenure_Months","Claim_Frequency"]
        cols_out = [c for c in cols_out if c in df_raw.columns]
        fig, axes = plt.subplots(2, 3, figsize=(16, 8))
        axes = axes.flatten()
        for i, col in enumerate(cols_out):
            axes[i].boxplot(df_raw[col].dropna(), patch_artist=True,
                            boxprops=dict(facecolor="lightblue", alpha=0.7),
                            medianprops=dict(color="red", linewidth=2),
                            flierprops=dict(marker="o", color="orange", alpha=0.5))
            Q1, Q3 = df_raw[col].quantile([0.25,0.75])
            IQR    = Q3 - Q1
            n_out  = ((df_raw[col]<Q1-1.5*IQR)|(df_raw[col]>Q3+1.5*IQR)).sum()
            axes[i].set_title(col, fontweight="bold")
            axes[i].set_xlabel(f"Outliers: {n_out:,} ({n_out/len(df_raw)*100:.2f}%)", fontsize=9)
        fig.suptitle("Outlier Detection", fontsize=13, fontweight="bold")
        fig.tight_layout()
        st.pyplot(fig)


# ════════════════════════════════════════════════════════════════════════════
# PAGE: TARGET ENGINEERING
# ════════════════════════════════════════════════════════════════════════════

elif page == "🎯 Target Engineering":
    st.title("🎯 Target Variable Engineering — Risk Score")

    st.info("""
    The dataset has **no pre-built target**. A domain-informed **Risk Score (0–100)** is computed
    from 6 clinical and behavioural variables, then bucketed into 4 risk categories.
    """)

    st.subheader("Scoring Formula")
    formula_df = pd.DataFrame({
        "Variable": ["Age","BMI","Smoker","Claim Frequency","Claim Severity","Payment Behavior"],
        "Max Points": [20, 20, 15, 20, 15, 10],
        "Rule": [
            "<30→5 | 30–44→10 | 45–59→15 | 60+→20",
            "<18.5→5 | 18.5–24.9→10 | 25–29.9→15 | ≥30→20",
            "No→0 | Yes→15",
            "Each claim = 5 pts (max 4 claims)",
            "None→0 | Low→3 | Med→7 | High→11 | Critical→15",
            "Consistent→0 | Occ. Late→5 | Freq. Late→10",
        ]
    })
    st.dataframe(formula_df, use_container_width=True, hide_index=True)

    st.subheader("Risk Thresholds")
    t1, t2, t3, t4 = st.columns(4)
    t1.markdown('<div class="risk-badge badge-Low">🟢 LOW &lt; 35</div>', unsafe_allow_html=True)
    t2.markdown('<div class="risk-badge badge-Medium">🟡 MEDIUM 35–51</div>', unsafe_allow_html=True)
    t3.markdown('<div class="risk-badge badge-High">🔵 HIGH 52–67</div>', unsafe_allow_html=True)
    t4.markdown('<div class="risk-badge badge-Critical">🔴 CRITICAL ≥ 68</div>', unsafe_allow_html=True)

    st.divider()
    st.subheader("Distribution Results")
    st.pyplot(fig_risk_dist(df1, le))

    st.subheader("Category Counts")
    vc = df1["Risk_category"].value_counts().reindex(["Low","Medium","High","Critical"])
    count_df = pd.DataFrame({
        "Category": vc.index,
        "Count": vc.values,
        "Percentage": (vc.values / len(df1) * 100).round(1)
    })
    st.dataframe(count_df, use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE: PREPROCESSING
# ════════════════════════════════════════════════════════════════════════════

elif page == "⚙️  Preprocessing":
    st.title("⚙️ Preprocessing Pipeline")

    tab1, tab2, tab3, tab4 = st.tabs(["Feature Engineering", "Encoding", "Scaling", "SMOTE"])

    with tab1:
        st.subheader("Engineered Features")
        feats = [
            ("Age_Group",           "Bins age into: Young / Middle Age / Senior / Elder"),
            ("BMI_Category",        "Bins BMI into: Underweight / Normal / Overweight / Obese"),
            ("Premium_Burden",      "Premium_GHS / Monthly_Income_GHS — affordability ratio"),
            ("Policy_Age_in_Days",  "Days since policy application date (ref: 2026-01-01)"),
            ("Late_payment_flag",   "1 if Payment_Behavior = 'Frequently Late'"),
            ("High_claim_flag",     "1 if Claim_Frequency ≥ 3"),
            ("Is_smoker",           "1 if Smoker = 'Yes'"),
            ("BMI_x_Age",           "Interaction: BMI × Age — compound obesity-age risk"),
            ("Claim_x_Severity",    "Interaction: Claim_Frequency × severity encoding"),
            ("Income_per_Dependent","Monthly_Income_GHS / (Dependents + 1)"),
        ]
        st.dataframe(pd.DataFrame(feats, columns=["Feature","Description"]),
                     use_container_width=True, hide_index=True)

    with tab2:
        st.subheader("Encoding Strategy")
        st.markdown("""
        | Type | Columns | Reason |
        |---|---|---|
        | **Binary** | Gender, Smoker | Two values only |
        | **Ordinal** | Grade_Level, Payment_Behavior, BMI_Category, Age_Group | Natural order exists |
        | **One-Hot** | Region, Product_Applied, Marital_Status, Payment_Method, Policy_Status | No ordinal relationship |
        | **Label** | Risk_category (target) | Required for sklearn classifiers |
        """)
        st.info("Label map: Critical → 0 · High → 1 · Low → 2 · Medium → 3")

    with tab3:
        st.subheader("StandardScaler — Scaling")
        st.markdown("""
        **Why scale?** Logistic Regression and gradient-based models are sensitive to feature magnitude.
        StandardScaler transforms each feature to μ=0, σ=1.

        **Data leakage prevention:** The scaler is **fit only on the training set**, then applied to both
        train and test. Fitting on test data would leak test distribution into training.
        """)
        sample_scale = pd.DataFrame({
            "Split": ["Train", "Test"],
            "Action": ["fit_transform()", "transform() only"],
            "Leakage Risk": ["None", "None"],
        })
        st.dataframe(sample_scale, use_container_width=True, hide_index=True)

    with tab4:
        st.subheader("SMOTE — Synthetic Minority Oversampling")
        st.pyplot(fig_smote(pipe["y_tr_before"], pipe["y_tr_res"], le))
        st.markdown("""
        **Why SMOTE?** The target classes are imbalanced. SMOTE generates synthetic samples
        by interpolating between minority-class neighbours in feature space — more robust
        than simply duplicating rows.

        ⚠️ **Applied only to the training set.** The test set is left untouched to reflect
        the real-world class distribution and give honest evaluation metrics.
        """)


# ════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL TRAINING & CV
# ════════════════════════════════════════════════════════════════════════════

elif page == "🤖 Model Training & CV":
    st.title("🤖 Model Training & Cross Validation")

    st.subheader("Models Overview")
    models_info = pd.DataFrame({
        "Model": ["Logistic Regression","Decision Tree","Random Forest",
                  "Gradient Boosting","XGBoost","LightGBM"],
        "Type": ["Linear","Tree","Ensemble (Bagging)","Ensemble (Boosting)","Ensemble (Boosting)","Ensemble (Boosting)"],
        "Key Params": [
            "C=1.0, multinomial, lbfgs",
            "max_depth=8, min_samples_split=20",
            "n_est=200, max_depth=10, balanced weights",
            "n_est=200, lr=0.05, subsample=0.8",
            "n_est=300, lr=0.05, colsample=0.8",
            "n_est=300, lr=0.05, balanced weights",
        ],
        "Notes": [
            "Baseline; fast & interpretable",
            "Non-linear; depth-limited to avoid overfit",
            "Handles noisy features well",
            "Sequential; strong on tabular",
            "Regularised with parallel trees",
            "Leaf-wise; fastest on large data",
        ]
    })
    st.dataframe(models_info, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("5-Fold Stratified Cross Validation Results")
    st.pyplot(fig_cv_comparison(pipe["cv_res"]))

    cv_table = pd.DataFrame(pipe["cv_res"]).T.sort_values("f1_mean", ascending=False)
    cv_table.columns = ["CV Accuracy", "CV Acc Std", "CV F1 Macro"]
    st.dataframe(cv_table.style.format("{:.4f}").background_gradient(cmap="Greens", subset=["CV F1 Macro"]),
                 use_container_width=True)

    st.divider()
    st.subheader("🏆 Voting Ensemble (Soft, Top 3)")
    st.info(f"Ensemble members: **{', '.join(pipe['top3'])}**")
    st.markdown("""
    Soft voting averages the predicted class **probabilities** from each member model,
    producing more calibrated and robust predictions than hard-majority voting.
    """)


# ════════════════════════════════════════════════════════════════════════════
# PAGE: EVALUATION
# ════════════════════════════════════════════════════════════════════════════

elif page == "📈 Evaluation":
    st.title("📈 Model Evaluation — Test Set")

    # Leaderboard
    st.subheader("Leaderboard")
    lb = pd.DataFrame([
        {"Model": r["name"], "Accuracy": r["accuracy"],
         "F1 Macro": r["f1_macro"], "ROC-AUC (OvR)": r["roc_auc"]}
        for r in pipe["results"].values()
    ]).sort_values("F1 Macro", ascending=False).reset_index(drop=True)
    st.dataframe(
        lb.style.format({"Accuracy":"{:.4f}","F1 Macro":"{:.4f}","ROC-AUC (OvR)":"{:.4f}"})
          .background_gradient(cmap="Greens", subset=["F1 Macro","ROC-AUC (OvR)"])
          .bar(subset=["Accuracy"], color="#5fba7d"),
        use_container_width=True
    )

    st.divider()
    st.subheader("Final Comparison Chart")
    st.pyplot(fig_final_comparison(pipe["results"]))

    st.divider()
    st.subheader("Per-Model Deep Dive")
    selected_model = st.selectbox("Select model", list(pipe["results"].keys()))
    res = pipe["results"][selected_model]

    m1, m2, m3 = st.columns(3)
    m1.metric("Accuracy",    f"{res['accuracy']:.4f}")
    m2.metric("F1 Macro",    f"{res['f1_macro']:.4f}")
    m3.metric("ROC-AUC OvR", f"{res['roc_auc']:.4f}")

    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(fig_confusion(res, le))
    with col2:
        st.pyplot(fig_roc(res, le))

    st.subheader("Classification Report")
    report_df = pd.DataFrame(res["report"]).T.round(4)
    st.dataframe(report_df.style.background_gradient(cmap="Blues", subset=["f1-score"]),
                 use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE: FEATURE IMPORTANCE
# ════════════════════════════════════════════════════════════════════════════

elif page == "🔍 Feature Importance":
    st.title("🔍 Feature Importance")

    tree_models = {k: v for k, v in pipe["trained"].items() if hasattr(v, "feature_importances_")}
    selected = st.selectbox("Select tree-based model", list(tree_models.keys()))
    top_n    = st.slider("Top N features", 10, 30, 20)
    model    = tree_models[selected]
    feature_names = pipe["X_train"].columns.tolist()

    fig = fig_feature_importance(model, feature_names, selected, top_n)
    if fig:
        st.pyplot(fig)

    st.divider()
    st.subheader("Feature Importance Table")
    fi_df = (pd.Series(model.feature_importances_, index=feature_names)
               .sort_values(ascending=False)
               .head(top_n)
               .reset_index())
    fi_df.columns = ["Feature","Importance"]
    fi_df["Cumulative"] = fi_df["Importance"].cumsum().round(4)
    st.dataframe(fi_df.style.bar(subset=["Importance"], color="#3498db").format({"Importance":"{:.5f}","Cumulative":"{:.4f}"}),
                 use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE: RISK PREDICTOR
# ════════════════════════════════════════════════════════════════════════════

elif page == "🧮 Risk Predictor":
    st.title("🧮 Live Risk Predictor")
    st.markdown("Enter a customer profile below to compute their **Risk Score** and **Risk Category** in real time.")

    with st.form("risk_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            age         = st.slider("Age", 18, 80, 35)
            bmi         = st.slider("BMI", 12.0, 50.0, 24.0, 0.1)
        with c2:
            smoker      = st.selectbox("Smoker", ["No","Yes"])
            claim_freq  = st.slider("Claim Frequency (last period)", 0, 4, 1)
        with c3:
            claim_sev   = st.selectbox("Claim Severity", ["None","Low","Medium","High","Critical"])
            payment_beh = st.selectbox("Payment Behavior", ["Consistent","Occasionally Late","Frequently Late"])

        submitted = st.form_submit_button("⚡ Compute Risk", use_container_width=True)

    if submitted:
        score, category, breakdown = compute_risk_score(age, bmi, smoker, claim_freq, claim_sev, payment_beh)

        st.divider()
        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric("Risk Score", f"{score} / 100")
            badge_cls = f"badge-{category}"
            cat_emoji = {"Low":"🟢","Medium":"🟡","High":"🔵","Critical":"🔴"}[category]
            st.markdown(
                f'<div class="risk-badge {badge_cls}">{cat_emoji} {category} Risk</div>',
                unsafe_allow_html=True
            )
            st.markdown("")
            threshold_info = {
                "Low": "Score < 35 — standard premium, minimal monitoring",
                "Medium": "Score 35–51 — moderate premium, periodic review",
                "High": "Score 52–67 — elevated premium, frequent review",
                "Critical": "Score ≥ 68 — high premium, immediate case review",
            }
            st.info(threshold_info[category])

        with col2:
            st.subheader("Score Breakdown")
            bd_df = pd.DataFrame(list(breakdown.items()), columns=["Component","Points"])
            fig, ax = plt.subplots(figsize=(7, 4))
            colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in bd_df["Points"]]
            ax.barh(bd_df["Component"], bd_df["Points"], color=colors, edgecolor="white")
            ax.axvline(0, color="grey", lw=0.8)
            ax.set_xlabel("Points Contributed")
            ax.set_title(f"Risk Score Components (Total: {score}/100)", fontweight="bold")
            for i, val in enumerate(bd_df["Points"]):
                ax.text(val + 0.2, i, str(val), va="center", fontsize=10, fontweight="bold")
            fig.tight_layout()
            st.pyplot(fig)

        st.divider()
        st.subheader("Score Gauge")
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.barh(["Risk Score"], [100], color="#f0f0f0", edgecolor="white", height=0.5)
        bar_color = CAT_CLR[category]
        ax.barh(["Risk Score"], [score], color=bar_color, edgecolor="white", height=0.5)
        for thr, lbl in zip([34, 51, 67], ["Low|Med","Med|High","High|Crit"]):
            ax.axvline(thr, color="white", lw=2, ls="--")
            ax.text(thr, 0.32, lbl, ha="center", fontsize=7, color="grey")
        ax.text(score, 0, f"  {score}", va="center", fontsize=12, fontweight="bold", color=bar_color)
        ax.set_xlim(0, 100); ax.set_xlabel("Risk Score")
        ax.set_title("Score Position on 0–100 Scale", fontweight="bold")
        fig.tight_layout()
        st.pyplot(fig)
