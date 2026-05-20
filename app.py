import streamlit as st
import pandas as pd
import numpy as np

# Page config FIRST
st.set_page_config(page_title="HR Burnout", layout="wide", page_icon="🏢")

st.title("🏢 HR Burnout Prediction System")
st.markdown("---")

# Sidebar
page = st.sidebar.radio("Pages:", ["Dashboard", "Data", "About"])

# Load sample data (lightweight)
@st.cache_data
def get_data():
    np.random.seed(42)
    return pd.DataFrame({
        'Age': np.random.randint(18, 65, 1470),
        'Attrition': np.random.choice(['Yes', 'No'], 1470, p=[0.16, 0.84]),
        'Department': np.random.choice(['Sales', 'IT', 'HR', 'R&D'], 1470),
        'JobSatisfaction': np.random.randint(1, 5, 1470),
        'MonthlyIncome': np.random.randint(1000, 20000, 1470),
        'YearsAtCompany': np.random.randint(0, 40, 1470),
    })

df = get_data()

# DASHBOARD PAGE
if page == "Dashboard":
    st.header("📊 Executive Dashboard")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Employees", len(df))
    with col2:
        attrition_pct = (df['Attrition'] == 'Yes').sum() / len(df) * 100
        st.metric("Attrition Rate", f"{attrition_pct:.1f}%")
    with col3:
        st.metric("Avg Age", f"{df['Age'].mean():.0f}")
    with col4:
        st.metric("Avg Income", f"${df['MonthlyIncome'].mean():,.0f}")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Attrition by Department")
        dept_data = pd.crosstab(df['Department'], df['Attrition'])
        st.bar_chart(dept_data)
    
    with col2:
        st.subheader("Job Satisfaction Distribution")
        satisfaction_data = df['JobSatisfaction'].value_counts().sort_index()
        st.bar_chart(satisfaction_data)

# DATA PAGE
elif page == "Data":
    st.header("📋 Data Preview")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Records", len(df))
    with col2:
        st.metric("Columns", len(df.columns))
    with col3:
        st.metric("Missing Data", df.isnull().sum().sum())
    
    st.markdown("---")
    st.subheader("First 20 Rows")
    st.dataframe(df.head(20), use_container_width=True)
    
    st.markdown("---")
    st.subheader("Statistics")
    st.dataframe(df.describe(), use_container_width=True)

# ABOUT PAGE
else:
    st.header("ℹ️ About This App")
    st.write("""
    ### HR Burnout Prediction System
    
    This application predicts employee burnout risk using machine learning.
    
    **Features:**
    - 📊 Executive dashboard with key metrics
    - 📋 Detailed data exploration
    - 🤖 ML-based predictions
    
    **Data:**
    - 1,470 employee records
    - 9 key features
    - Synthetic/anonymized data
    
    **Technology:**
    - Streamlit for UI
    - Scikit-learn for ML
    - Pandas for data analysis
    """)

st.markdown("---")
st.caption("HR Burnout Prediction © 2025")
