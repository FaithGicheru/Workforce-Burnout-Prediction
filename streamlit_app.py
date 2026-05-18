import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(page_title="Workforce Burnout Prediction", layout="wide", initial_sidebar_state="expanded")

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 0rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Title
st.title("🏢 Workforce Burnout Prediction System")
st.markdown("---")

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.radio("Select a Page:", 
    ["Dashboard", "Data Exploration", "Model Performance", "Predictions"])

# Load sample data function
@st.cache_data
def load_data():
    """Load and preprocess the HR dataset"""
    # Create sample data since we don't have the original CSV
    np.random.seed(42)
    
    n_samples = 1470
    data = {
        'Age': np.random.randint(18, 65, n_samples),
        'Department': np.random.choice(['Sales', 'HR', 'IT', 'Finance'], n_samples),
        'JobRole': np.random.choice(['Manager', 'Executive', 'Sales Rep', 'Analyst', 'Technician'], n_samples),
        'MonthlyIncome': np.random.randint(1000, 20000, n_samples),
        'YearsAtCompany': np.random.randint(0, 40, n_samples),
        'YearsInCurrentRole': np.random.randint(0, 30, n_samples),
        'OverTime': np.random.choice(['Yes', 'No'], n_samples),
        'JobSatisfaction': np.random.randint(1, 5, n_samples),
        'WorkLifeBalance': np.random.randint(1, 5, n_samples),
        'Attrition': np.random.choice(['Yes', 'No'], n_samples, p=[0.16, 0.84])
    }
    
    df = pd.DataFrame(data)
    return df

@st.cache_resource
def train_model(df):
    """Train the burnout prediction model"""
    # Prepare data
    df_model = df.copy()
    
    # Encode categorical variables
    le_dict = {}
    for col in ['Department', 'JobRole', 'OverTime', 'Attrition']:
        if col in df_model.columns:
            le = LabelEncoder()
            df_model[col] = le.fit_transform(df_model[col].astype(str))
            le_dict[col] = le
    
    # Features and target
    X = df_model.drop('Attrition', axis=1)
    y = df_model['Attrition']
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X, y)
    
    return model, X, y, le_dict

df = load_data()
model, X, y, le_dict = train_model(df)

# PAGE 1: DASHBOARD
if page == "Dashboard":
    st.header("📊 Executive Dashboard")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Employees",
            len(df),
            delta=None
        )
    
    with col2:
        attrition_rate = (df['Attrition'] == 'Yes').sum() / len(df) * 100
        st.metric(
            "Attrition Rate",
            f"{attrition_rate:.1f}%",
            delta=None
        )
    
    with col3:
        avg_age = df['Age'].mean()
        st.metric(
            "Avg. Employee Age",
            f"{avg_age:.0f} years",
            delta=None
        )
    
    with col4:
        avg_income = df['MonthlyIncome'].mean()
        st.metric(
            "Avg. Monthly Income",
            f"${avg_income:,.0f}",
            delta=None
        )
    
    st.markdown("---")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Attrition by Department")
        dept_attrition = pd.crosstab(df['Department'], df['Attrition'])
        st.bar_chart(dept_attrition)
    
    with col2:
        st.subheader("Employee Distribution by Age")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(df['Age'], bins=20, color='steelblue', edgecolor='black', alpha=0.7)
        ax.set_xlabel('Age')
        ax.set_ylabel('Count')
        st.pyplot(fig)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Work-Life Balance vs Attrition")
        balance_attrition = pd.crosstab(df['WorkLifeBalance'], df['Attrition'])
        st.bar_chart(balance_attrition)
    
    with col2:
        st.subheader("Income Distribution")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(df['MonthlyIncome'], bins=30, color='coral', edgecolor='black', alpha=0.7)
        ax.set_xlabel('Monthly Income ($)')
        ax.set_ylabel('Count')
        st.pyplot(fig)

# PAGE 2: DATA EXPLORATION
elif page == "Data Exploration":
    st.header("🔍 Data Exploration")
    
    st.subheader("Dataset Overview")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Rows", len(df))
    with col2:
        st.metric("Total Columns", df.shape[1])
    with col3:
        st.metric("Missing Values", df.isnull().sum().sum())
    
    st.markdown("---")
    
    # Display dataframe
    st.subheader("First 10 Rows")
    st.dataframe(df.head(10), use_container_width=True)
    
    st.markdown("---")
    
    # Statistical summary
    st.subheader("Statistical Summary")
    st.dataframe(df.describe(), use_container_width=True)
    
    st.markdown("---")
    
    # Column selection for analysis
    st.subheader("Detailed Column Analysis")
    selected_col = st.selectbox("Select a column to analyze:", df.columns)
    
    if df[selected_col].dtype == 'object':
        value_counts = df[selected_col].value_counts()
        st.bar_chart(value_counts)
    else:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(df[selected_col], bins=30, color='steelblue', edgecolor='black', alpha=0.7)
        ax.set_title(f"Distribution of {selected_col}")
        st.pyplot(fig)

# PAGE 3: MODEL PERFORMANCE
elif page == "Model Performance":
    st.header("🤖 Model Performance")
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Predictions
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Display metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Model Accuracy", f"{accuracy:.2%}")
    
    with col2:
        st.metric("Test Samples", len(X_test))
    
    st.markdown("---")
    
    # Confusion Matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    st.pyplot(fig)
    
    st.markdown("---")
    
    # Classification Report
    st.subheader("Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df, use_container_width=True)
    
    st.markdown("---")
    
    # Feature Importance
    st.subheader("Feature Importance")
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(feature_importance['Feature'], feature_importance['Importance'], color='steelblue')
    ax.set_xlabel('Importance')
    ax.set_title('Feature Importance in Burnout Prediction')
    st.pyplot(fig)

# PAGE 4: PREDICTIONS
elif page == "Predictions":
    st.header("🔮 Make a Prediction")
    
    st.markdown("Enter employee details to predict burnout risk:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.slider("Age", min_value=18, max_value=65, value=30)
        monthly_income = st.slider("Monthly Income ($)", min_value=1000, max_value=20000, value=5000, step=500)
        years_at_company = st.slider("Years at Company", min_value=0, max_value=40, value=5)
    
    with col2:
        department = st.selectbox("Department", df['Department'].unique())
        job_role = st.selectbox("Job Role", df['JobRole'].unique())
        over_time = st.selectbox("Overtime", ['Yes', 'No'])
    
    col1, col2 = st.columns(2)
    
    with col1:
        job_satisfaction = st.slider("Job Satisfaction (1-4)", min_value=1, max_value=4, value=2)
        work_life_balance = st.slider("Work-Life Balance (1-4)", min_value=1, max_value=4, value=2)
    
    with col2:
        years_in_role = st.slider("Years in Current Role", min_value=0, max_value=30, value=2)
    
    # Make prediction
    if st.button("🔍 Predict Burnout Risk", use_container_width=True):
        # Prepare input
        input_data = pd.DataFrame({
            'Age': [age],
            'Department': [department],
            'JobRole': [job_role],
            'MonthlyIncome': [monthly_income],
            'YearsAtCompany': [years_at_company],
            'YearsInCurrentRole': [years_in_role],
            'OverTime': [over_time],
            'JobSatisfaction': [job_satisfaction],
            'WorkLifeBalance': [work_life_balance]
        })
        
        # Encode categorical variables
        input_encoded = input_data.copy()
        for col in ['Department', 'JobRole', 'OverTime']:
            if col in le_dict:
                input_encoded[col] = le_dict[col].transform(input_data[col])
        
        # Make prediction
        prediction = model.predict(input_encoded)[0]
        probability = model.predict_proba(input_encoded)[0]
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if prediction == 1:
                st.error(f"⚠️ **HIGH BURNOUT RISK** - {probability[1]:.1%}")
            else:
                st.success(f"✅ **LOW BURNOUT RISK** - {probability[0]:.1%}")
        
        with col2:
            st.info(f"Confidence: {max(probability):.1%}")

st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray;'>Workforce Burnout Prediction System © 2025</p>",
    unsafe_allow_html=True
)
