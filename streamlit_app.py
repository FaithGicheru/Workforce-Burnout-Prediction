import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(page_title="HR Burnout Prediction", layout="wide", page_icon="🏢")

# Title
st.title("🏢 HR Employee Burnout Prediction System")
st.markdown("---")

# Sidebar Navigation
st.sidebar.header("Navigation")
page = st.sidebar.radio(
    "Select Page:",
    ["Dashboard", "Data Analysis", "Model Performance", "Make Prediction"]
)

# Load Data Function
@st.cache_data
def load_data():
    """Load HR dataset"""
    try:
        # Try loading from GitHub
        url = "https://raw.githubusercontent.com/FaithGicheru/Workforce-Burnout-Prediction/main/WA_Fn-UseC_-HR-Employee-Attrition.csv"
        df = pd.read_csv(url)
        return df
    except Exception as e:
        st.warning(f"Could not load from GitHub: {e}")
        # Fallback to sample data
        np.random.seed(42)
        n = 1470
        df = pd.DataFrame({
            'Age': np.random.randint(18, 65, n),
            'Attrition': np.random.choice(['Yes', 'No'], n, p=[0.16, 0.84]),
            'Department': np.random.choice(['Sales', 'HR', 'IT', 'Research & Development'], n),
            'JobRole': np.random.choice(['Manager', 'Executive', 'Sales Rep', 'Analyst'], n),
            'MonthlyIncome': np.random.randint(1000, 20000, n),
            'JobSatisfaction': np.random.randint(1, 5, n),
            'WorkLifeBalance': np.random.randint(1, 5, n),
            'YearsAtCompany': np.random.randint(0, 40, n),
            'YearsInCurrentRole': np.random.randint(0, 30, n),
            'Age': np.random.randint(18, 65, n),
            'EnvironmentSatisfaction': np.random.randint(1, 5, n),
        })
        return df

# Train Model Function
@st.cache_resource
def train_model(df):
    """Train the prediction model"""
    df_model = df.copy()
    
    # Encode categorical variables
    le_dict = {}
    categorical_cols = ['Attrition', 'Department', 'JobRole']
    
    for col in categorical_cols:
        if col in df_model.columns:
            le = LabelEncoder()
            df_model[col] = le.fit_transform(df_model[col].astype(str))
            le_dict[col] = le
    
    # Prepare features and target
    feature_cols = [col for col in df_model.columns if col not in ['Attrition', 'EmployeeNumber']]
    X = df_model[feature_cols]
    y = df_model['Attrition']
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    model.fit(X, y)
    
    return model, X, y, le_dict, feature_cols

# Load data
df = load_data()
model, X, y, le_dict, feature_cols = train_model(df)

# PAGE 1: DASHBOARD
if page == "Dashboard":
    st.header("📊 Executive Dashboard")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Employees", len(df))
    
    with col2:
        attrition_rate = (df['Attrition'] == 'Yes').sum() / len(df) * 100
        st.metric("Attrition Rate", f"{attrition_rate:.1f}%")
    
    with col3:
        avg_age = df['Age'].mean()
        st.metric("Avg Age", f"{avg_age:.0f} years")
    
    with col4:
        avg_income = df['MonthlyIncome'].mean()
        st.metric("Avg Income", f"${avg_income:,.0f}")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Attrition by Department")
        dept_data = pd.crosstab(df['Department'], df['Attrition'])
        st.bar_chart(dept_data)
    
    with col2:
        st.subheader("Age Distribution")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(df['Age'], bins=20, color='steelblue', edgecolor='black')
        ax.set_xlabel('Age')
        ax.set_ylabel('Count')
        st.pyplot(fig)
        plt.close()

# PAGE 2: DATA ANALYSIS
elif page == "Data Analysis":
    st.header("🔍 Data Exploration")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Rows", len(df))
    with col2:
        st.metric("Total Columns", df.shape[1])
    with col3:
        st.metric("Missing Values", df.isnull().sum().sum())
    
    st.markdown("---")
    
    st.subheader("Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)
    
    st.markdown("---")
    
    st.subheader("Statistical Summary")
    st.dataframe(df.describe(), use_container_width=True)

# PAGE 3: MODEL PERFORMANCE
elif page == "Model Performance":
    st.header("🤖 Model Performance")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Accuracy", f"{accuracy:.2%}")
    with col2:
        st.metric("F1 Score", f"{f1:.2%}")
    with col3:
        st.metric("Test Samples", len(y_test))
    
    st.markdown("---")
    
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    st.pyplot(fig)
    plt.close()
    
    st.markdown("---")
    
    st.subheader("Feature Importance")
    importance_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False).head(10)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(importance_df['Feature'], importance_df['Importance'], color='steelblue')
    ax.set_xlabel('Importance')
    st.pyplot(fig)
    plt.close()

# PAGE 4: PREDICTION
elif page == "Make Prediction":
    st.header("🔮 Predict Burnout Risk")
    
    st.markdown("Enter employee details:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.slider("Age", 18, 65, 30)
        monthly_income = st.slider("Monthly Income", 1000, 20000, 5000, 500)
        years_company = st.slider("Years at Company", 0, 40, 5)
    
    with col2:
        department = st.selectbox("Department", sorted(df['Department'].unique()))
        job_role = st.selectbox("Job Role", sorted(df['JobRole'].unique()))
        job_satisfaction = st.slider("Job Satisfaction (1-4)", 1, 4, 2)
    
    if st.button("🚀 Predict", use_container_width=True):
        try:
            # Prepare input
            input_df = pd.DataFrame({
                'Age': [age],
                'MonthlyIncome': [monthly_income],
                'YearsAtCompany': [years_company],
                'JobSatisfaction': [job_satisfaction],
                'Department': [department],
                'JobRole': [job_role],
            })
            
            # Make prediction
            prediction = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0]
            
            st.markdown("---")
            col1, col2 = st.columns(2)
            
            with col1:
                if prediction == 1:
                    st.error(f"⚠️ HIGH BURNOUT RISK: {probability[1]:.1%}")
                else:
                    st.success(f"✅ LOW BURNOUT RISK: {probability[0]:.1%}")
            
            with col2:
                st.info(f"Confidence: {max(probability):.1%}")
        
        except Exception as e:
            st.error(f"Error: {e}")

st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>HR Burnout Prediction System © 2025</p>", unsafe_allow_html=True)
