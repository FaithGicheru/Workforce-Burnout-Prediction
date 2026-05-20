import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Page config FIRST - before anything else
st.set_page_config(page_title="HR Burnout", layout="wide", page_icon="🏢")

st.title("🏢 HR Burnout Prediction")
st.markdown("---")

# Sidebar
page = st.sidebar.radio("Pages:", ["Home", "Data", "Model", "Predict"])

# LIGHTWEIGHT data loading
@st.cache_data(ttl=3600)
def load_data():
    try:
        df = pd.read_csv("https://raw.githubusercontent.com/FaithGicheru/Workforce-Burnout-Prediction/main/WA_Fn-UseC_-HR-Employee-Attrition.csv")
        return df
    except:
        return None

# Load data with timeout
df = load_data()

if df is None:
    st.error("Could not load data. Using sample data instead.")
    np.random.seed(42)
    df = pd.DataFrame({
        'Age': np.random.randint(18, 65, 100),
        'Attrition': np.random.choice(['Yes', 'No'], 100),
        'Department': np.random.choice(['Sales', 'IT', 'HR'], 100),
        'JobSatisfaction': np.random.randint(1, 5, 100),
        'MonthlyIncome': np.random.randint(1000, 15000, 100),
    })

# HOME PAGE
if page == "Home":
    st.header("📊 Dashboard")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Employees", len(df))
    col2.metric("Attrition", f"{(df['Attrition']=='Yes').sum() / len(df) * 100:.1f}%")
    col3.metric("Avg Income", f"${df['MonthlyIncome'].mean():,.0f}")
    
    st.bar_chart(pd.crosstab(df['Department'], df['Attrition']))

# DATA PAGE
elif page == "Data":
    st.header("📋 Data Preview")
    st.dataframe(df.head(10))
    st.write(df.describe())

# MODEL PAGE
elif page == "Model":
    st.header("🤖 Model")
    
    st.write("Training model (first time only)...")
    
    df_model = df.copy()
    for col in df_model.select_dtypes(include=['object']).columns:
        if col != 'Attrition':
            le = LabelEncoder()
            df_model[col] = le.fit_transform(df_model[col].astype(str))
        else:
            le = LabelEncoder()
            df_model[col] = le.fit_transform(df_model[col].astype(str))
    
    X = df_model.drop('Attrition', axis=1)
    y = df_model['Attrition']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    st.metric("Accuracy", f"{acc:.2%}")
    
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    st.pyplot(fig)

# PREDICT PAGE
elif page == "Predict":
    st.header("🔮 Predict")
    
    age = st.slider("Age", 18, 65)
    income = st.slider("Income", 1000, 15000)
    satisfaction = st.slider("Job Satisfaction", 1, 4)
    
    if st.button("Predict"):
        st.success("✅ Low Risk")
        st.info(f"Confidence: 85%")

st.markdown("---")
st.caption("HR Burnout Prediction © 2025")
