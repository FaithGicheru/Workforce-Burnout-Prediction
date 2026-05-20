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
try:
    model = joblib.load("burnout_model.pkl")
except FileNotFoundError:
    st.error("❌ Model file 'burnout_model.pkl' not found. Please upload the model.")
    st.stop()

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
        ax.hist(df['Age'], bins=20, edgecolor='black')
        ax.set_xlabel("Age")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)

# -------------------------
# DATA ANALYSIS
# -------------------------
elif page == "Data Analysis":
    st.header("🔍 Data Exploration")

    st.write("**Dataset Shape:**", df.shape)
    st.dataframe(df.head(), use_container_width=True)

    st.subheader("Summary Statistics")
    st.dataframe(df.describe(), use_container_width=True)

# -------------------------
# MODEL INFO
# -------------------------
elif page == "Model Info":
    st.header("🤖 Model Information")

    st.info("This app uses a pre-trained RandomForest model loaded from `burnout_model.pkl`")

    st.write("**Model Type:**", type(model).__name__)

    # Try to show model info if available
    if hasattr(model, 'feature_names_in_'):
        st.write("**Expected Features:**", list(model.feature_names_in_))
    
    if hasattr(model, 'n_estimators'):
        st.write("**Number of Trees:**", model.n_estimators)

    st.warning("⚠️ Feature importance and evaluation metrics are not available unless saved during training.")

# -------------------------
# PREDICTION
# -------------------------
elif page == "Make Prediction":
    st.header("🔮 Predict Burnout Risk")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Personal Information")
        age = st.slider("Age", 18, 65, 30)
        income = st.slider("Monthly Income ($)", 1000, 20000, 5000, 500)
        years_company = st.slider("Years at Company", 0, 40, 5)
        years_role = st.slider("Years in Current Role", 0, 30, 2)

    with col2:
        st.subheader("Work Environment")
        department = st.selectbox("Department", sorted(df['Department'].unique()))
        job_role = st.selectbox("Job Role", sorted(df['JobRole'].unique()))
        
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("Satisfaction & Balance")
        job_satisfaction = st.slider("Job Satisfaction (1=Low, 4=High)", 1, 4, 2)
        work_life_balance = st.slider("Work-Life Balance (1=Low, 4=High)", 1, 4, 2)
    
    with col4:
        environment_satisfaction = st.slider("Environment Satisfaction (1=Low, 4=High)", 1, 4, 2)

    if st.button("🎯 Predict Burnout Risk", use_container_width=True):
        try:
            # Create input dataframe with all required features
            input_data = {
                "Age": age,
                "MonthlyIncome": income,
                "YearsAtCompany": years_company,
                "YearsInCurrentRole": years_role,
                "JobSatisfaction": job_satisfaction,
                "WorkLifeBalance": work_life_balance,
                "EnvironmentSatisfaction": environment_satisfaction,
                "Department": department,
                "JobRole": job_role
            }
            
            input_df = pd.DataFrame([input_data])

            # One-hot encode categorical variables
            input_df = pd.get_dummies(input_df, columns=['Department', 'JobRole'], drop_first=False)

            # Get expected features from model
            if hasattr(model, 'feature_names_in_'):
                expected_features = model.feature_names_in_
                # Align with model's training features
                input_df = input_df.reindex(columns=expected_features, fill_value=0)
            
            # Make prediction
            prediction = model.predict(input_df)[0]
            proba = model.predict_proba(input_df)[0]

            st.markdown("---")

            # Display results
            if prediction == 1:
                st.error(f"⚠️ **HIGH BURNOUT RISK** - {proba[1]:.1%} probability")
            else:
                st.success(f"✅ **LOW BURNOUT RISK** - {proba[0]:.1%} probability")

            st.info(f"**Model Confidence:** {max(proba):.1%}")
            
            # Show prediction breakdown
            st.write("**Prediction Details:**")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Low Risk Probability", f"{proba[0]:.1%}")
            with col2:
                st.metric("High Risk Probability", f"{proba[1]:.1%}")

        except Exception as e:
            st.error(f"❌ Prediction error: {str(e)}")
            st.info("Make sure all features match the model's training data.")

# -------------------------
# FOOTER
# -------------------------
st.markdown("---")
st.caption("HR Burnout Prediction System © 2025")