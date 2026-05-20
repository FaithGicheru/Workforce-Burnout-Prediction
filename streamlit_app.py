import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(
    page_title="HR Burnout Prediction",
    layout="wide",
    page_icon="🏢"
)

st.title("🏢 HR Employee Burnout Prediction System")
st.markdown("---")

# -------------------------
# LOAD MODEL
# -------------------------
model = joblib.load("burnout_model.pkl")

# -------------------------
# LOAD DATA
# -------------------------
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/FaithGicheru/Workforce-Burnout-Prediction/main/WA_Fn-UseC_-HR-Employee-Attrition.csv"
    try:
        df = pd.read_csv(url)
    except:
        np.random.seed(42)
        n = 1470
        df = pd.DataFrame({
            'Age': np.random.randint(18, 65, n),
            'Attrition': np.random.choice(['Yes', 'No'], n, p=[0.16, 0.84]),
            'Department': np.random.choice(['Sales', 'HR', 'IT', 'R&D'], n),
            'JobRole': np.random.choice(['Manager', 'Executive', 'Sales Rep', 'Analyst'], n),
            'MonthlyIncome': np.random.randint(1000, 20000, n),
            'JobSatisfaction': np.random.randint(1, 5, n),
            'WorkLifeBalance': np.random.randint(1, 5, n),
            'YearsAtCompany': np.random.randint(0, 40, n),
            'YearsInCurrentRole': np.random.randint(0, 30, n),
            'EnvironmentSatisfaction': np.random.randint(1, 5, n)
        })
    return df

df = load_data()

# -------------------------
# SIDEBAR NAVIGATION
# -------------------------
page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Data Analysis", "Model Info", "Make Prediction"]
)

# -------------------------
# DASHBOARD
# -------------------------
if page == "Dashboard":
    st.header("📊 Executive Dashboard")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Employees", len(df))

    with col2:
        attrition_rate = (df['Attrition'] == 'Yes').mean() * 100
        st.metric("Attrition Rate", f"{attrition_rate:.1f}%")

    with col3:
        st.metric("Avg Age", f"{df['Age'].mean():.0f}")

    with col4:
        st.metric("Avg Income", f"${df['MonthlyIncome'].mean():,.0f}")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Attrition by Department")
        dept = pd.crosstab(df['Department'], df['Attrition'])
        st.bar_chart(dept)

    with col2:
        st.subheader("Age Distribution")
        fig, ax = plt.subplots()
        ax.hist(df['Age'], bins=20)
        st.pyplot(fig)

# -------------------------
# DATA ANALYSIS
# -------------------------
elif page == "Data Analysis":
    st.header("🔍 Data Exploration")

    st.write("Shape:", df.shape)
    st.dataframe(df.head())

    st.subheader("Summary Statistics")
    st.dataframe(df.describe())

# -------------------------
# MODEL INFO (SAFE VERSION)
# -------------------------
elif page == "Model Info":
    st.header("🤖 Model Information")

    st.info("This app uses a pre-trained RandomForest model loaded from burnout_model.pkl")

    st.write("Model type:", type(model).__name__)

    st.warning("Feature importance and evaluation metrics are not available unless saved during training.")

# -------------------------
# PREDICTION
# -------------------------
elif page == "Make Prediction":
    st.header("🔮 Predict Burnout Risk")

    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", 18, 65, 30)
        income = st.slider("Monthly Income", 1000, 20000, 5000, 500)
        years = st.slider("Years at Company", 0, 40, 5)

    with col2:
        department = st.selectbox("Department", df['Department'].unique())
        job_role = st.selectbox("Job Role", df['JobRole'].unique())
        satisfaction = st.slider("Job Satisfaction", 1, 4, 2)

    if st.button("Predict"):
        try:
            input_df = pd.DataFrame([{
                "Age": age,
                "MonthlyIncome": income,
                "YearsAtCompany": years,
                "JobSatisfaction": satisfaction,
                "Department": department,
                "JobRole": job_role
            }])

            # FIX: encode categorical properly for model
            input_df = pd.get_dummies(input_df)

            # align with model training features
            input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)

            prediction = model.predict(input_df)[0]
            proba = model.predict_proba(input_df)[0]

            st.markdown("---")

            if prediction == 1:
                st.error(f"⚠️ HIGH BURNOUT RISK ({proba[1]:.1%})")
            else:
                st.success(f"✅ LOW BURNOUT RISK ({proba[0]:.1%})")

            st.info(f"Model Confidence: {max(proba):.1%}")

        except Exception as e:
            st.error(f"Prediction error: {e}")

# -------------------------
# FOOTER
# -------------------------
st.markdown("---")
st.caption("HR Burnout Prediction System © 2025")