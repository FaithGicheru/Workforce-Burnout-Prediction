import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, roc_auc_score, roc_curve
import plotly.graph_objects as go
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIG & STYLING
# ============================================================================

st.set_page_config(
    page_title="HR Burnout Prediction System",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional look
st.markdown("""
<style>
    :root {
        --primary-color: #2E86AB;
        --secondary-color: #A23B72;
        --success-color: #06A77D;
        --warning-color: #F18F01;
        --danger-color: #C73E1D;
        --light-bg: #F8F9FA;
        --dark-text: #1F2937;
    }
    
    * {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    .stMetric {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 12px;
        color: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .metric-title {
        font-size: 0.9rem;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-weight: 600;
    }
    
    .stMarkdown h1 {
        color: #2E86AB;
        border-bottom: 3px solid #A23B72;
        padding-bottom: 10px;
    }
    
    .stMarkdown h2 {
        color: #2E86AB;
        margin-top: 25px;
    }
    
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #2E86AB 0%, #A23B72 100%);
        color: white;
        border: none;
        padding: 12px;
        border-radius: 8px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(46, 134, 171, 0.4);
    }
    
    .alert-success {
        background: #D1FAE5;
        border-left: 4px solid #06A77D;
        padding: 15px;
        border-radius: 8px;
        color: #065F46;
    }
    
    .alert-danger {
        background: #FEE2E2;
        border-left: 4px solid #C73E1D;
        padding: 15px;
        border-radius: 8px;
        color: #7F1D12;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD AND PREPARE DATA
# ============================================================================

@st.cache_data
def load_data():
    """Load all datasets"""
    try:
        df_main = pd.read_csv('https://raw.githubusercontent.com/FaithGicheru/Workforce-Burnout-Prediction/main/WA_Fn-UseC_-HR-Employee-Attrition.csv')
        df_cleaned = pd.read_csv('https://raw.githubusercontent.com/FaithGicheru/Workforce-Burnout-Prediction/main/hr_cleaned_stage2.csv')
        df_interventions = pd.read_csv('https://raw.githubusercontent.com/FaithGicheru/Workforce-Burnout-Prediction/main/hr_intervention_recommendations.csv')
        return df_main, df_cleaned, df_interventions
    except Exception as e:
        st.warning(f"Loading from GitHub failed: {e}. Using sample data instead.")
        # Fallback to sample data
        np.random.seed(42)
        n = 1471
        df_main = pd.DataFrame({
            'Age': np.random.randint(18, 65, n),
            'Attrition': np.random.choice(['Yes', 'No'], n, p=[0.16, 0.84]),
            'Department': np.random.choice(['Sales', 'HR', 'IT', 'Research & Development'], n),
            'JobRole': np.random.choice(['Manager', 'Executive', 'Sales Rep', 'Analyst', 'Technician'], n),
            'MonthlyIncome': np.random.randint(1000, 20000, n),
            'JobSatisfaction': np.random.randint(1, 5, n),
            'WorkLifeBalance': np.random.randint(1, 5, n),
            'YearsAtCompany': np.random.randint(0, 40, n),
            'YearsInCurrentRole': np.random.randint(0, 30, n),
        })
        df_cleaned = df_main.copy()
        df_interventions = pd.DataFrame({
            'EmployeeNumber': range(n),
            'BurnoutRisk': np.random.choice(['Low', 'Medium', 'High'], n),
            'Priority': np.random.choice(['Low', 'Medium', 'High'], n)
        })
        return df_main, df_cleaned, df_interventions

@st.cache_resource
def train_model(df):
    """Train burnout prediction model"""
    df_model = df.copy()
    
    # Encode categorical variables
    le_dict = {}
    categorical_cols = ['Attrition', 'Department', 'JobRole', 'Gender', 'MaritalStatus', 'OverTime']
    
    for col in categorical_cols:
        if col in df_model.columns:
            le = LabelEncoder()
            df_model[col] = le.fit_transform(df_model[col].astype(str))
            le_dict[col] = le
    
    # Prepare features and target
    X = df_model.drop(['Attrition', 'EmployeeNumber'], axis=1, errors='ignore')
    y = df_model['Attrition']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    model.fit(X_scaled, y)
    
    return model, X, y, le_dict, scaler, X.columns

# Load data
df_main, df_cleaned, df_interventions = load_data()
model, X, y, le_dict, scaler, feature_names = train_model(df_cleaned)

# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================

st.sidebar.markdown("# 🏢 HR Analytics Suite")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate to:",
    ["📊 Dashboard", "🔍 Detailed Analysis", "🤖 Model Performance", "🔮 Predict Burnout", "💡 Interventions"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.info(
    "💡 **About**: Predict employee burnout risk using machine learning. "
    "Identify at-risk employees and recommend targeted interventions."
)

# ============================================================================
# PAGE 1: EXECUTIVE DASHBOARD
# ============================================================================

if page == "📊 Dashboard":
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "👥 Total Employees",
            f"{len(df_main):,}",
            delta=None,
            delta_color="off"
        )
    
    with col2:
        attrition_pct = (df_main['Attrition'] == 'Yes').sum() / len(df_main) * 100
        st.metric(
            "⚠️ Attrition Rate",
            f"{attrition_pct:.1f}%",
            delta=f"{attrition_pct-16:.1f}%",
            delta_color="inverse"
        )
    
    with col3:
        avg_income = df_main['MonthlyIncome'].mean()
        st.metric(
            "💰 Avg Monthly Income",
            f"${avg_income:,.0f}",
            delta=None,
            delta_color="off"
        )
    
    with col4:
        avg_age = df_main['Age'].mean()
        st.metric(
            "📅 Avg Employee Age",
            f"{avg_age:.0f} years",
            delta=None,
            delta_color="off"
        )
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Attrition by Department")
        dept_attrition = pd.crosstab(df_main['Department'], df_main['Attrition'])
        fig = px.bar(
            dept_attrition,
            labels={'value': 'Employee Count', 'index': 'Department'},
            color_discrete_sequence=['#06A77D', '#C73E1D']
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("👥 Distribution by Age")
        fig = px.histogram(
            df_main,
            x='Age',
            nbins=30,
            color_discrete_sequence=['#2E86AB']
        )
        fig.update_layout(
            xaxis_title='Age',
            yaxis_title='Count',
            showlegend=False,
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Job Satisfaction vs Attrition")
        satisfaction_attrition = pd.crosstab(df_main['JobSatisfaction'], df_main['Attrition'])
        fig = px.bar(
            satisfaction_attrition,
            labels={'value': 'Employee Count', 'index': 'Job Satisfaction'},
            color_discrete_sequence=['#06A77D', '#C73E1D']
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("💼 By Job Role")
        role_counts = df_main['JobRole'].value_counts()
        fig = px.pie(
            values=role_counts.values,
            names=role_counts.index,
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# PAGE 2: DETAILED ANALYSIS
# ============================================================================

elif page == "🔍 Detailed Analysis":
    st.header("📊 Deep Dive Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Records", f"{len(df_main):,}")
    with col2:
        st.metric("Total Columns", df_main.shape[1])
    with col3:
        st.metric("Missing Values", df_main.isnull().sum().sum())
    
    st.markdown("---")
    
    tab1, tab2, tab3 = st.tabs(["Data Preview", "Statistics", "Correlations"])
    
    with tab1:
        st.subheader("Dataset Sample")
        st.dataframe(df_main.head(10), use_container_width=True)
    
    with tab2:
        st.subheader("Statistical Summary")
        st.dataframe(df_main.describe(), use_container_width=True)
    
    with tab3:
        st.subheader("Feature Correlations with Attrition")
        df_corr = df_main.copy()
        for col in df_corr.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            df_corr[col] = le.fit_transform(df_corr[col].astype(str))
        
        corr_with_attrition = df_corr.corr()['Attrition'].sort_values(ascending=False)
        
        fig = px.bar(
            x=corr_with_attrition.index,
            y=corr_with_attrition.values,
            labels={'x': 'Feature', 'y': 'Correlation'},
            color=corr_with_attrition.values,
            color_continuous_scale='RdBu'
        )
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# PAGE 3: MODEL PERFORMANCE
# ============================================================================

elif page == "🤖 Model Performance":
    st.header("📈 Model Evaluation")
    
    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🎯 Accuracy", f"{accuracy:.2%}")
    with col2:
        st.metric("📊 F1 Score", f"{f1:.2%}")
    with col3:
        st.metric("📈 ROC-AUC", f"{roc_auc:.2%}")
    with col4:
        st.metric("🧪 Test Samples", len(y_test))
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig = go.Figure(
            data=go.Heatmap(
                z=cm,
                x=['No Attrition', 'Attrition'],
                y=['No Attrition', 'Attrition'],
                text=cm,
                texttemplate='%{text}',
                colorscale='Blues',
                showscale=False
            )
        )
        fig.update_layout(
            xaxis_title='Predicted',
            yaxis_title='Actual',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("ROC Curve")
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        fig = go.Figure(
            data=[
                go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC (AUC={roc_auc:.3f})', 
                          line=dict(color='#2E86AB', width=3)),
                go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random',
                          line=dict(color='gray', dash='dash'))
            ]
        )
        fig.update_layout(
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            height=400,
            hovermode='closest'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    st.subheader("📊 Feature Importance")
    feature_imp = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False).head(15)
    
    fig = px.bar(
        feature_imp,
        x='Importance',
        y='Feature',
        orientation='h',
        color='Importance',
        color_continuous_scale='Viridis'
    )
    fig.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# PAGE 4: INDIVIDUAL PREDICTION
# ============================================================================

elif page == "🔮 Predict Burnout":
    st.header("🔮 Employee Burnout Risk Prediction")
    
    st.markdown("Enter employee details to predict burnout risk:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.slider("Age", min_value=18, max_value=65, value=30)
        department = st.selectbox("Department", sorted(df_main['Department'].unique()))
        job_role = st.selectbox("Job Role", sorted(df_main['JobRole'].unique()))
    
    with col2:
        monthly_income = st.slider("Monthly Income ($)", min_value=1000, max_value=20000, value=5000, step=500)
        years_company = st.slider("Years at Company", min_value=0, max_value=40, value=5)
        years_role = st.slider("Years in Current Role", min_value=0, max_value=30, value=2)
    
    with col3:
        job_satisfaction = st.select_slider("Job Satisfaction", options=[1, 2, 3, 4], value=2)
        work_life_balance = st.select_slider("Work-Life Balance", options=[1, 2, 3, 4], value=2)
        environment_satisfaction = st.select_slider("Environment Satisfaction", options=[1, 2, 3, 4], value=2)
    
    if st.button("🚀 Predict Burnout Risk", use_container_width=True):
        try:
            # Prepare input
            input_data = pd.DataFrame({
                'Age': [age],
                'Department': [department],
                'JobRole': [job_role],
                'MonthlyIncome': [monthly_income],
                'YearsAtCompany': [years_company],
                'YearsInCurrentRole': [years_role],
                'JobSatisfaction': [job_satisfaction],
                'WorkLifeBalance': [work_life_balance],
                'EnvironmentSatisfaction': [environment_satisfaction],
            })
            
            # Scale features
            input_scaled = scaler.transform(input_data)
            
            # Predict
            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0]
            
            st.markdown("---")
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                if prediction == 1:
                    st.markdown(
                        "<div class='alert-danger'>"
                        f"<h3>⚠️ HIGH BURNOUT RISK</h3>"
                        f"<p style='font-size: 1.5rem; margin: 10px 0;'>{probability[1]:.1%}</p>"
                        "</div>",
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        "<div class='alert-success'>"
                        f"<h3>✅ LOW BURNOUT RISK</h3>"
                        f"<p style='font-size: 1.5rem; margin: 10px 0;'>{probability[0]:.1%}</p>"
                        "</div>",
                        unsafe_allow_html=True
                    )
            
            with col2:
                st.metric("Confidence Level", f"{max(probability):.1%}")
                
                # Risk gauge
                fig = go.Figure(
                    data=[go.Indicator(
                        mode="gauge+number+delta",
                        value=probability[1] * 100,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Burnout Risk %"},
                        delta={'reference': 50},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 33], 'color': "#06A77D"},
                                {'range': [33, 66], 'color': "#F18F01"},
                                {'range': [66, 100], 'color': "#C73E1D"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 50
                            }
                        }
                    )]
                )
                st.plotly_chart(fig, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error in prediction: {e}")

# ============================================================================
# PAGE 5: INTERVENTIONS
# ============================================================================

elif page == "💡 Interventions":
    st.header("💡 Burnout Intervention Recommendations")
    
    if 'Interventions' in df_interventions.columns:
        st.markdown("### Recommended Actions by Risk Level")
        
        # Group by risk
        risk_groups = df_interventions.groupby('BurnoutRisk').size()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("🟢 Low Risk", risk_groups.get('Low', 0))
        with col2:
            st.metric("🟡 Medium Risk", risk_groups.get('Medium', 0))
        with col3:
            st.metric("🔴 High Risk", risk_groups.get('High', 0))
        
        st.markdown("---")
        
        # Sample interventions
        st.subheader("Sample Intervention Recommendations")
        
        high_risk = df_interventions[df_interventions['BurnoutRisk'] == 'High'].head(5)
        
        if len(high_risk) > 0:
            st.markdown("#### High-Risk Employees (Priority)")
            for idx, row in high_risk.iterrows():
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.markdown(f"**Employee #{int(row['EmployeeNumber'])}**")
                with col2:
                    interventions_text = str(row.get('Interventions', 'Standard support program'))
                    st.markdown(f"• {interventions_text[:100]}...")
        
        # Display intervention table
        st.markdown("---")
        st.subheader("Full Intervention Database")
        st.dataframe(
            df_interventions.head(20),
            use_container_width=True,
            hide_index=True
        )

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #999; font-size: 0.9rem;'>"
    "🏢 HR Burnout Prediction System | Machine Learning for Employee Wellness"
    "</p>",
    unsafe_allow_html=True
)
