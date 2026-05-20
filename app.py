import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
import io
from collections import Counter

warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
    mean_squared_error, mean_absolute_error, r2_score
)

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

RANDOM_STATE = 42
sns.set_style('whitegrid')
plt.rcParams['font.size'] = 10

# ─────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Workforce Burnout Prediction System",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
.metric-card {
    background: #f8f9fa;
    border-left: 4px solid #1565C0;
    border-radius: 6px;
    padding: 12px 16px;
    margin: 6px 0;
}
.critical { border-left-color: #D32F2F !important; }
.high     { border-left-color: #F57C00 !important; }
.moderate { border-left-color: #FBC02D !important; }
.watch    { border-left-color: #388E3C !important; }
.stage-header {
    font-size: 1.4rem;
    font-weight: 700;
    margin-bottom: 0.3rem;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
PALETTE   = {'Low': '#4CAF50', 'Medium': '#FFC107', 'High': '#F44336'}
RISK_ORDER = ['Low', 'Medium', 'High']
PRIORITY_COLORS = {
    'CRITICAL': '#D32F2F', 'HIGH': '#F57C00',
    'MODERATE': '#FBC02D', 'WATCH': '#388E3C'
}

def fig_to_img(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=110)
    buf.seek(0)
    plt.close(fig)
    return buf


# ─────────────────────────────────────────────────────────────
# Pipeline functions
# ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def run_pipeline(csv_bytes: bytes):
    """Run the full 3-stage pipeline and return all artefacts."""
    df = pd.read_csv(io.BytesIO(csv_bytes))

    # ── Stage 0: Cleaning ──────────────────────────────────
    constant_cols = [c for c in df.columns if df[c].nunique() == 1]
    df.drop(columns=constant_cols, inplace=True)
    df.drop_duplicates(inplace=True)

    cat_cols = df.select_dtypes(include='object').columns.tolist()
    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # ── Feature Engineering: BurnoutRisk ──────────────────
    def calc_burnout(row):
        score = 0
        if row['OverTime'] == 1:           score += 2
        if row['WorkLifeBalance'] == 1:    score += 2
        elif row['WorkLifeBalance'] == 2:  score += 1
        if row['JobSatisfaction'] == 1:    score += 2
        elif row['JobSatisfaction'] == 2:  score += 1
        if row['EnvironmentSatisfaction'] == 1:  score += 2
        elif row['EnvironmentSatisfaction'] == 2: score += 1
        if row['MonthlyIncome'] < 3000:    score += 1
        if row['YearsAtCompany'] > 10:     score += 1
        return 'High' if score >= 5 else ('Medium' if score >= 2 else 'Low')

    df['BurnoutRisk'] = df.apply(calc_burnout, axis=1)

    # ── Stage 1: Classification ────────────────────────────
    selected_features = [
        'OverTime', 'WorkLifeBalance', 'JobSatisfaction', 'EnvironmentSatisfaction',
        'MonthlyIncome', 'YearsAtCompany', 'Age', 'JobLevel', 'JobInvolvement',
        'RelationshipSatisfaction', 'PerformanceRating', 'TotalWorkingYears',
        'YearsSinceLastPromotion', 'NumCompaniesWorked', 'DistanceFromHome',
        'BusinessTravel', 'MaritalStatus', 'Department', 'Gender'
    ]
    X = df[selected_features]
    le_target = LabelEncoder()
    y = le_target.fit_transform(df['BurnoutRisk'])
    class_names = le_target.classes_

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    lr_model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    lr_model.fit(X_train_s, y_train)
    lr_preds = lr_model.predict(X_test_s)

    rf_model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
    rf_model.fit(X_train_s, y_train)
    rf_preds = rf_model.predict(X_test_s)

    def metrics(y_true, y_pred):
        return {
            'Accuracy':  round(accuracy_score(y_true, y_pred), 4),
            'Precision': round(precision_score(y_true, y_pred, average='macro', zero_division=0), 4),
            'Recall':    round(recall_score(y_true, y_pred, average='macro', zero_division=0), 4),
            'F1-Score':  round(f1_score(y_true, y_pred, average='macro', zero_division=0), 4),
        }

    s1_results = {
        'Logistic Regression': metrics(y_test, lr_preds),
        'Random Forest':       metrics(y_test, rf_preds),
    }
    s1_cm = {
        'Logistic Regression': confusion_matrix(y_test, lr_preds),
        'Random Forest':       confusion_matrix(y_test, rf_preds),
    }
    s1_feature_importance = pd.Series(
        rf_model.feature_importances_, index=selected_features
    ).sort_values(ascending=True)

    # ── Stage 2: Regression ────────────────────────────────
    df['BurnoutRisk_Encoded'] = le_target.transform(df['BurnoutRisk'])
    df2 = df.copy()

    overtime_flag = (df2['OverTime'] == 1).astype(int)
    low_income    = (df2['MonthlyIncome'] < df2['MonthlyIncome'].median()).astype(int)
    df2['SickDays'] = np.clip((
        8 +
        overtime_flag * 4 +
        (5 - df2['JobSatisfaction']) * 1.5 +
        (5 - df2['WorkLifeBalance']) * 2 +
        (5 - df2['EnvironmentSatisfaction']) * 1.2 +
        low_income * 3
    ).round(), 0, 25)

    reg_features = [c for c in df2.columns
                    if c not in ['EmployeeNumber', 'BurnoutRisk',
                                 'BurnoutRisk_Encoded', 'SickDays']]
    X_reg = df2[reg_features]
    y_reg = df2['SickDays']
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=RANDOM_STATE)
    scaler_reg = StandardScaler()
    X_tr_s = scaler_reg.fit_transform(X_tr)
    X_te_s = scaler_reg.transform(X_te)

    lr_reg  = LinearRegression()
    lr_reg.fit(X_tr_s, y_tr)
    p_lr = lr_reg.predict(X_te_s)

    rf_reg = RandomForestRegressor(n_estimators=100, max_depth=15,
                                   random_state=RANDOM_STATE, n_jobs=-1)
    rf_reg.fit(X_tr_s, y_tr)
    p_rf = rf_reg.predict(X_te_s)

    if XGBOOST_AVAILABLE:
        xgb_reg = XGBRegressor(n_estimators=100, max_depth=6,
                               learning_rate=0.1, random_state=RANDOM_STATE, verbosity=0)
    else:
        xgb_reg = RandomForestRegressor(n_estimators=100, max_depth=6,
                                        random_state=RANDOM_STATE, n_jobs=-1)
    xgb_reg.fit(X_tr_s, y_tr)
    p_xgb = xgb_reg.predict(X_te_s)

    def reg_metrics(yt, yp):
        return {
            'MAE':      round(mean_absolute_error(yt, yp), 4),
            'RMSE':     round(np.sqrt(mean_squared_error(yt, yp)), 4),
            'R² Score': round(r2_score(yt, yp), 4),
        }

    s2_results = {
        'Linear Regression': reg_metrics(y_te, p_lr),
        'Random Forest':     reg_metrics(y_te, p_rf),
        'XGBoost':           reg_metrics(y_te, p_xgb),
    }
    s2_preds = {'Linear Regression': p_lr, 'Random Forest': p_rf, 'XGBoost': p_xgb}
    s2_y_test = y_te

    rf_reg_imp = pd.Series(rf_reg.feature_importances_, index=reg_features).nlargest(12)
    xgb_imp    = pd.Series(xgb_reg.feature_importances_, index=reg_features).nlargest(12)

    # ── Stage 3: HR Interventions ──────────────────────────
    def gen_interventions(row):
        interventions, drivers = [], []
        if row['OverTime'] == 1:
            interventions += ["Reduce or cap overtime hours immediately",
                              "Introduce flexible / compressed work-week scheduling"]
            drivers.append("working overtime")
        if row['WorkLifeBalance'] == 1:
            interventions += ["Enrol in mandatory Employee Wellness Programme",
                              "Assign a dedicated HR wellness check-in (bi-weekly)"]
            drivers.append("very poor work-life balance")
        elif row['WorkLifeBalance'] == 2:
            interventions.append("Offer optional wellness / mindfulness sessions")
            drivers.append("below-average work-life balance")
        if row['JobSatisfaction'] == 1:
            interventions += ["Conduct a one-on-one career development conversation",
                              "Review job role alignment and consider internal transfer"]
            drivers.append("very low job satisfaction")
        elif row['JobSatisfaction'] == 2:
            interventions.append("Schedule a manager-employee feedback session")
            drivers.append("low job satisfaction")
        if row['EnvironmentSatisfaction'] == 1:
            interventions += ["Investigate workplace environment concerns urgently",
                              "Consider remote / hybrid work arrangement"]
            drivers.append("very poor environment satisfaction")
        elif row['EnvironmentSatisfaction'] == 2:
            interventions.append("Gather anonymous feedback on workplace conditions")
            drivers.append("poor environment satisfaction")
        if row['MonthlyIncome'] < 3000:
            interventions += ["Conduct immediate compensation review",
                              "Provide access to Employee Assistance Programme (EAP)"]
            drivers.append("low monthly income (financial stress)")
        elif row['MonthlyIncome'] < 6500 and row['JobLevel'] >= 3:
            interventions.append("Review compensation relative to job level and market rates")
        if row['YearsSinceLastPromotion'] >= 5:
            interventions.append("Fast-track promotion review or provide stretch assignments")
            drivers.append(f"{int(row['YearsSinceLastPromotion'])} years without promotion")
        elif row['YearsSinceLastPromotion'] >= 3:
            interventions.append("Discuss career growth pathway and set a promotion timeline")
        if row['YearsAtCompany'] > 10 and row['JobSatisfaction'] <= 2:
            interventions.append("Offer sabbatical leave or role rotation programme")
            drivers.append("long tenure with low satisfaction")
        if row['BusinessTravel'] == 1:
            interventions.append("Reduce business travel frequency or offer travel allowance")
            drivers.append("frequent business travel")
        if row['BurnoutRisk'] == 'High':
            interventions += ["Refer to counselling / mental health support services",
                              "Place on a structured 30-day HR recovery plan"]
        if row['PerformanceRating'] == 3 and row['JobSatisfaction'] <= 2:
            interventions.append("Implement a peer-recognition and rewards programme")
        if not interventions:
            interventions += ["Continue regular manager check-ins",
                              "Encourage participation in optional team-building activities"]
        seen, unique = set(), []
        for i in interventions:
            if i not in seen:
                seen.add(i); unique.append(i)
        if row['BurnoutRisk'] == 'High':               priority = 'CRITICAL'
        elif row['BurnoutRisk'] == 'Medium' and len(unique) >= 4: priority = 'HIGH'
        elif row['BurnoutRisk'] == 'Medium':           priority = 'MODERATE'
        else:                                          priority = 'WATCH'
        urgency = ("Key drivers: " + ", ".join(drivers) + ".") if drivers else \
                  "No critical stress signals detected. Monitor regularly."
        return {'priority': priority, 'interventions': unique, 'urgency_note': urgency}

    recs = df2.apply(gen_interventions, axis=1)
    df2['Priority']          = recs.apply(lambda x: x['priority'])
    df2['Interventions']     = recs.apply(lambda x: x['interventions'])
    df2['Urgency_Note']      = recs.apply(lambda x: x['urgency_note'])
    df2['Num_Interventions'] = df2['Interventions'].apply(len)

    return dict(
        df=df, df2=df2,
        class_names=class_names,
        s1_results=s1_results, s1_cm=s1_cm,
        s1_feature_importance=s1_feature_importance,
        y_test=y_test,
        s2_results=s2_results, s2_preds=s2_preds, s2_y_test=s2_y_test,
        reg_features=reg_features, rf_reg_imp=rf_reg_imp, xgb_imp=xgb_imp,
    )


# ─────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4228/4228787.png", width=60)
    st.title("Burnout Prediction\nSystem")
    st.caption("IBM HR Analytics Dataset")
    st.divider()

    uploaded = st.file_uploader(
        "Upload IBM HR CSV", type=["csv"],
        help="WA_Fn-UseC_-HR-Employee-Attrition.csv"
    )
    st.divider()
    st.markdown("**Stages**")
    st.markdown("🧹 0 — Data Cleaning & EDA")
    st.markdown("🎯 1 — Burnout Classification")
    st.markdown("📅 2 — Sick-Day Forecasting")
    st.markdown("💡 3 — HR Interventions")
    st.divider()
    if not XGBOOST_AVAILABLE:
        st.warning("XGBoost not installed — using Random Forest as fallback.")
    st.caption("Moringa School • DS Capstone 2025-26")


# ─────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────
st.markdown("## 🔥 Workforce Burnout Prediction System")
st.caption("Burnout Risk Classification · Sick-Day Forecasting · HR Intervention Recommendations")
st.divider()

if uploaded is None:
    st.info("👈 Upload the **IBM HR Employee Attrition CSV** in the sidebar to get started.")
    st.markdown("""
    **What this dashboard does:**
    - **Stage 1** — Classifies each employee as Low / Medium / High burnout risk
    - **Stage 2** — Forecasts expected sick days per employee
    - **Stage 3** — Generates personalised HR intervention recommendations
    
    **Dataset:** IBM HR Analytics Employee Attrition & Performance  
    File: `WA_Fn-UseC_-HR-Employee-Attrition.csv`
    """)
    st.stop()

# ─────────────────────────────────────────────────────────────
# Run pipeline
# ─────────────────────────────────────────────────────────────
csv_bytes = uploaded.read()
with st.spinner("Running full pipeline (Stage 1 → 2 → 3)…"):
    P = run_pipeline(csv_bytes)

df   = P['df']
df2  = P['df2']
cn   = P['class_names']

# ─────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────
tab0, tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Overview & EDA",
    "🎯 Stage 1 — Classification",
    "📅 Stage 2 — Sick-Day Forecast",
    "💡 Stage 3 — HR Interventions",
    "🔍 Employee Lookup",
])


# ══════════════════════════════════════════════════════════════
# TAB 0 — Overview & EDA
# ══════════════════════════════════════════════════════════════
with tab0:
    st.markdown('<p class="stage-header">Dataset Overview & Exploratory Analysis</p>',
                unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Employees", f"{len(df):,}")
    c2.metric("Features", df.shape[1])
    c3.metric("High Risk", f"{(df['BurnoutRisk']=='High').sum():,}",
              f"{(df['BurnoutRisk']=='High').mean()*100:.1f}%")
    c4.metric("Overtime Workers",
              f"{(df['OverTime']==1).sum():,}",
              f"{(df['OverTime']==1).mean()*100:.1f}%")
    st.divider()

    # Burnout distribution
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("BurnoutRisk Distribution")
        counts = df['BurnoutRisk'].value_counts().reindex(RISK_ORDER)
        fig, ax = plt.subplots(figsize=(5, 3))
        bars = ax.bar(RISK_ORDER, counts,
                      color=[PALETTE[k] for k in RISK_ORDER], edgecolor='white', width=0.5)
        for b in bars:
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 5,
                    str(int(b.get_height())), ha='center', fontsize=9, fontweight='bold')
        ax.set_ylabel("Employees"); ax.set_xlabel("Burnout Risk")
        plt.tight_layout()
        st.image(fig_to_img(fig))

    with col_b:
        st.subheader("OverTime vs Burnout Risk")
        ct = pd.crosstab(df['OverTime'], df['BurnoutRisk'])[RISK_ORDER]
        ct.index = ['No Overtime', 'Yes Overtime']
        fig, ax = plt.subplots(figsize=(5, 3))
        ct.plot(kind='bar', ax=ax, color=[PALETTE[k] for k in RISK_ORDER],
                edgecolor='white', width=0.6)
        ax.set_xlabel(""); ax.set_ylabel("Employees")
        ax.legend(title='Burnout Risk')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        plt.tight_layout()
        st.image(fig_to_img(fig))

    col_c, col_d = st.columns(2)
    with col_c:
        st.subheader("Job Satisfaction vs Burnout Risk")
        ct2 = pd.crosstab(df['JobSatisfaction'], df['BurnoutRisk'])[RISK_ORDER]
        ct2.index = ['Low (1)', 'Medium (2)', 'High (3)', 'Very High (4)']
        fig, ax = plt.subplots(figsize=(5, 3))
        ct2.plot(kind='bar', ax=ax, color=[PALETTE[k] for k in RISK_ORDER],
                 edgecolor='white', width=0.65)
        ax.set_xlabel("Job Satisfaction"); ax.set_ylabel("Employees")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        ax.legend(title='Burnout Risk')
        plt.tight_layout()
        st.image(fig_to_img(fig))

    with col_d:
        st.subheader("Monthly Income by Burnout Risk")
        fig, ax = plt.subplots(figsize=(5, 3))
        for risk in RISK_ORDER:
            df[df['BurnoutRisk'] == risk]['MonthlyIncome'].plot(
                kind='kde', ax=ax, label=risk, color=PALETTE[risk], linewidth=2)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))
        ax.set_xlabel("Monthly Income"); ax.set_ylabel("Density")
        ax.legend(title='Burnout Risk')
        plt.tight_layout()
        st.image(fig_to_img(fig))

    st.subheader("Correlation Heatmap — Key Features")
    key_feats = ['OverTime', 'WorkLifeBalance', 'JobSatisfaction',
                 'EnvironmentSatisfaction', 'MonthlyIncome', 'YearsAtCompany',
                 'Age', 'JobLevel', 'TotalWorkingYears', 'Attrition']
    corr = df[key_feats].corr()
    fig, ax = plt.subplots(figsize=(9, 6))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdYlGn',
                center=0, linewidths=0.5, square=True, cbar_kws={'shrink': 0.7}, ax=ax)
    ax.set_title('Correlation Heatmap — Key HR Features', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.image(fig_to_img(fig))


# ══════════════════════════════════════════════════════════════
# TAB 1 — Classification
# ══════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<p class="stage-header">🎯 Stage 1 — Burnout Risk Classification</p>',
                unsafe_allow_html=True)
    st.caption("Logistic Regression vs Random Forest · 80/20 stratified split")

    # Metrics table
    res_df = pd.DataFrame(P['s1_results']).T
    st.subheader("Model Performance")
    c1, c2 = st.columns(2)
    for idx, (model_name, row) in enumerate(res_df.iterrows()):
        col = c1 if idx == 0 else c2
        with col:
            st.markdown(f"**{model_name}**")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Accuracy",  f"{row['Accuracy']*100:.2f}%")
            m2.metric("Precision", f"{row['Precision']:.4f}")
            m3.metric("Recall",    f"{row['Recall']:.4f}")
            m4.metric("F1-Score",  f"{row['F1-Score']:.4f}")

    st.divider()

    # Model comparison bar chart
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Performance Comparison")
        metrics_list = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        x = np.arange(len(metrics_list))
        w = 0.35
        fig, ax = plt.subplots(figsize=(6, 3.5))
        b1 = ax.bar(x - w/2, [P['s1_results']['Logistic Regression'][m] for m in metrics_list],
                    w, label='Logistic Regression', color='#1565C0', alpha=0.85)
        b2 = ax.bar(x + w/2, [P['s1_results']['Random Forest'][m] for m in metrics_list],
                    w, label='Random Forest', color='#2E7D32', alpha=0.85)
        for b in list(b1) + list(b2):
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.005,
                    f'{b.get_height():.3f}', ha='center', fontsize=7.5)
        ax.set_xticks(x); ax.set_xticklabels(metrics_list)
        ax.set_ylim(0, 1.12); ax.legend(fontsize=8)
        ax.axhline(1.0, color='grey', linestyle='--', alpha=0.4)
        plt.tight_layout()
        st.image(fig_to_img(fig))

    with col_b:
        st.subheader("Feature Importance — Random Forest")
        fi = P['s1_feature_importance']
        fig, ax = plt.subplots(figsize=(6, 4.5))
        colors_fi = ['#F44336' if f in
                     ['OverTime', 'WorkLifeBalance', 'JobSatisfaction',
                      'EnvironmentSatisfaction', 'MonthlyIncome', 'YearsAtCompany']
                     else '#90CAF9' for f in fi.index]
        fi.plot(kind='barh', ax=ax, color=colors_fi, edgecolor='white')
        ax.set_xlabel("Importance Score")
        from matplotlib.patches import Patch
        ax.legend(handles=[
            Patch(facecolor='#F44336', label='Core Burnout Features'),
            Patch(facecolor='#90CAF9', label='Supporting Features'),
        ], loc='lower right', fontsize=7)
        plt.tight_layout()
        st.image(fig_to_img(fig))

    # Confusion matrices
    st.subheader("Confusion Matrices")
    col_c, col_d = st.columns(2)
    cms_items = list(P['s1_cm'].items())
    for i, (name, cm) in enumerate(cms_items):
        with (col_c if i == 0 else col_d):
            st.caption(name)
            cmap = 'Blues' if i == 0 else 'Greens'
            fig, ax = plt.subplots(figsize=(4, 3))
            sns.heatmap(cm, annot=True, fmt='d', cmap=cmap,
                        xticklabels=cn, yticklabels=cn,
                        linewidths=0.5, square=True, ax=ax)
            ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
            ax.set_title(f'Confusion Matrix — {name}', fontsize=9, fontweight='bold')
            plt.tight_layout()
            st.image(fig_to_img(fig))


# ══════════════════════════════════════════════════════════════
# TAB 2 — Regression
# ══════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<p class="stage-header">📅 Stage 2 — Sick-Day Forecasting</p>',
                unsafe_allow_html=True)
    st.caption("Linear Regression · Random Forest · XGBoost · 80/20 split")

    # Results table
    reg_df = pd.DataFrame(P['s2_results']).T
    st.subheader("Model Performance")
    c1, c2, c3 = st.columns(3)
    for idx, (mname, row) in enumerate(reg_df.iterrows()):
        col = [c1, c2, c3][idx]
        with col:
            st.markdown(f"**{mname}**")
            m1, m2, m3 = st.columns(3)
            m1.metric("MAE",       f"{row['MAE']:.4f} days")
            m2.metric("RMSE",      f"{row['RMSE']:.4f} days")
            m3.metric("R² Score",  f"{row['R² Score']:.4f}")

    st.divider()

    # Comparison bar
    st.subheader("Regression Model Comparison")
    metrics_r = ['MAE', 'RMSE', 'R² Score']
    x_pos = np.arange(len(metrics_r))
    w = 0.25
    colors_r = plt.cm.tab10.colors
    fig, ax = plt.subplots(figsize=(8, 4))
    for k, (mname, clr) in enumerate(zip(
        ['Linear Regression', 'Random Forest', 'XGBoost'],
        [colors_r[0], colors_r[2], colors_r[1]]
    )):
        vals = [P['s2_results'][mname][m] for m in metrics_r]
        bars = ax.bar(x_pos + (k-1)*w, vals, w, label=mname, color=clr, alpha=0.85)
        for b in bars:
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.005,
                    f'{b.get_height():.3f}', ha='center', fontsize=7.5)
    ax.set_xticks(x_pos); ax.set_xticklabels(metrics_r)
    ax.set_ylabel("Score"); ax.legend(fontsize=8)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    st.image(fig_to_img(fig))

    # Actual vs Predicted
    st.subheader("Actual vs Predicted Sick Days")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    clrs_s = plt.cm.Set2.colors
    for i, mname in enumerate(['Linear Regression', 'Random Forest', 'XGBoost']):
        preds = P['s2_preds'][mname]
        ax = axes[i]
        ax.scatter(P['s2_y_test'], preds, alpha=0.5, color=clrs_s[i],
                   edgecolors='black', linewidth=0.3, s=18)
        mn = min(P['s2_y_test'].min(), preds.min())
        mx = max(P['s2_y_test'].max(), preds.max())
        ax.plot([mn, mx], [mn, mx], 'k--', lw=1.5, label='Perfect')
        ax.set_xlabel('Actual'); ax.set_ylabel('Predicted')
        ax.set_title(f'{mname}\nR²={r2_score(P["s2_y_test"], preds):.3f}',
                     fontsize=9, fontweight='bold')
        ax.grid(linestyle='--', alpha=0.3)
    plt.tight_layout()
    st.image(fig_to_img(fig))

    # Feature importance
    st.subheader("Feature Importance — Tree Models")
    col_e, col_f = st.columns(2)
    for (col, imp_series, title) in [
        (col_e, P['rf_reg_imp'].sort_values(), "Random Forest — Top 12"),
        (col_f, P['xgb_imp'].sort_values(), "XGBoost — Top 12"),
    ]:
        with col:
            fig, ax = plt.subplots(figsize=(5, 4))
            imp_series.plot(kind='barh', ax=ax, color='#1565C0', edgecolor='white')
            ax.set_xlabel("Importance"); ax.set_title(title, fontsize=9, fontweight='bold')
            ax.grid(axis='x', linestyle='--', alpha=0.4)
            plt.tight_layout()
            st.image(fig_to_img(fig))


# ══════════════════════════════════════════════════════════════
# TAB 3 — HR Interventions
# ══════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<p class="stage-header">💡 Stage 3 — HR Intervention Recommendations</p>',
                unsafe_allow_html=True)

    # Summary metrics
    priority_order = ['CRITICAL', 'HIGH', 'MODERATE', 'WATCH']
    pc = df2['Priority'].value_counts()
    c1, c2, c3, c4 = st.columns(4)
    for col, p, cls in zip([c1,c2,c3,c4], priority_order,
                           ['critical','high','moderate','watch']):
        n = int(pc.get(p, 0))
        pct = n / len(df2) * 100
        col.metric(p, f"{n:,}", f"{pct:.1f}% of workforce")

    st.divider()

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Priority Distribution")
        counts_p = df2['Priority'].value_counts().reindex(priority_order)
        fig, ax = plt.subplots(figsize=(5, 3))
        bars = ax.bar(priority_order, counts_p,
                      color=[PRIORITY_COLORS[p] for p in priority_order],
                      edgecolor='white', width=0.5)
        for b in bars:
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 5,
                    str(int(b.get_height())), ha='center', fontweight='bold', fontsize=9)
        ax.set_ylabel("Employees")
        plt.tight_layout()
        st.image(fig_to_img(fig))

    with col_b:
        st.subheader("Intervention Count by Burnout Risk")
        fig, ax = plt.subplots(figsize=(5, 3))
        for risk, clr in [('High','#F44336'),('Medium','#FFC107'),('Low','#4CAF50')]:
            df2[df2['BurnoutRisk'] == risk]['Num_Interventions'].plot(
                kind='kde', ax=ax, label=risk, color=clr, linewidth=2)
        ax.set_xlabel("# Interventions"); ax.set_ylabel("Density")
        ax.legend(title='Burnout Risk')
        plt.tight_layout()
        st.image(fig_to_img(fig))

    # Most common interventions
    st.subheader("Most Frequently Recommended Interventions")
    all_int = [i for sub in df2['Interventions'] for i in sub]
    int_df = (pd.DataFrame(Counter(all_int).items(), columns=['Intervention', 'Count'])
              .sort_values('Count', ascending=True))
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(int_df['Intervention'], int_df['Count'],
                   color='#1565C0', edgecolor='white')
    for b in bars:
        ax.text(b.get_width() + 5, b.get_y() + b.get_height()/2,
                str(int(b.get_width())), va='center', fontsize=8)
    ax.set_xlabel("Number of Employees")
    plt.tight_layout()
    st.image(fig_to_img(fig))

    # Sample recommendations per priority
    st.subheader("Sample Employee Recommendations")
    for priority in priority_order:
        subset = df2[df2['Priority'] == priority]
        if subset.empty:
            continue
        sample = subset.iloc[0]
        css_cls = priority.lower()
        emp_id = int(sample.get('EmployeeNumber', 0))
        with st.expander(f"{'🔴' if priority=='CRITICAL' else '🟠' if priority=='HIGH' else '🟡' if priority=='MODERATE' else '🟢'} {priority} — Employee #{emp_id}  |  Burnout: {sample['BurnoutRisk']}  |  Est. Sick Days: {int(sample['SickDays'])}"):
            st.info(sample['Urgency_Note'])
            st.markdown("**Recommended Interventions:**")
            for k, action in enumerate(sample['Interventions'], 1):
                st.markdown(f"{k}. {action}")

    # Download
    st.divider()
    st.subheader("Export Recommendations")
    export = df2[['EmployeeNumber','BurnoutRisk','Priority','Urgency_Note',
                  'Num_Interventions','SickDays','Interventions']].copy()
    export['Interventions'] = export['Interventions'].apply(lambda x: ' | '.join(x))
    csv_out = export.to_csv(index=False).encode()
    st.download_button("⬇️ Download Full Recommendations CSV",
                       data=csv_out,
                       file_name="hr_intervention_recommendations.csv",
                       mime="text/csv")


# ══════════════════════════════════════════════════════════════
# TAB 4 — Employee Lookup
# ══════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<p class="stage-header">🔍 Employee Lookup</p>',
                unsafe_allow_html=True)
    st.caption("Search an employee by ID and view their full risk profile.")

    if 'EmployeeNumber' in df2.columns:
        emp_ids = sorted(df2['EmployeeNumber'].unique())
        selected_id = st.selectbox("Select Employee Number", emp_ids)
        emp = df2[df2['EmployeeNumber'] == selected_id].iloc[0]

        risk_color = {'High': '🔴', 'Medium': '🟡', 'Low': '🟢'}
        priority_icon = {'CRITICAL': '🔴', 'HIGH': '🟠', 'MODERATE': '🟡', 'WATCH': '🟢'}

        c1, c2, c3 = st.columns(3)
        c1.metric("Burnout Risk",
                  f"{risk_color.get(emp['BurnoutRisk'], '')} {emp['BurnoutRisk']}")
        c2.metric("HR Priority",
                  f"{priority_icon.get(emp['Priority'], '')} {emp['Priority']}")
        c3.metric("Est. Sick Days / Year", f"{int(emp['SickDays'])} days")

        st.divider()

        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Key Stress Indicators")
            indicators = {
                'OverTime':               'Yes' if emp['OverTime'] == 1 else 'No',
                'WorkLifeBalance':        f"{int(emp['WorkLifeBalance'])} / 4",
                'JobSatisfaction':        f"{int(emp['JobSatisfaction'])} / 4",
                'EnvironmentSatisfaction':f"{int(emp['EnvironmentSatisfaction'])} / 4",
                'MonthlyIncome':          f"${int(emp['MonthlyIncome']):,}",
                'YearsAtCompany':         f"{int(emp['YearsAtCompany'])} yrs",
                'YearsSinceLastPromotion':f"{int(emp['YearsSinceLastPromotion'])} yrs",
            }
            for k, v in indicators.items():
                st.markdown(f"**{k}:** {v}")

        with col_b:
            st.subheader("Urgency Note")
            st.info(emp['Urgency_Note'])
            st.subheader("Recommended Interventions")
            for k, action in enumerate(emp['Interventions'], 1):
                st.markdown(f"{k}. {action}")
    else:
        st.warning("EmployeeNumber column not found in dataset.")
