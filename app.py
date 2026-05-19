from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
import pickle
import os
import warnings

warnings.filterwarnings('ignore')

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

# Global variables for models and scalers
models = {}
scalers = {}
encoders = {}

# ==========================================
# Initialize Models and Scalers
# ==========================================

def initialize_models():
    """
    Initialize or load pre-trained models.
    In production, load saved model files here.
    For demo, we'll create simple models.
    """
    global models, scalers, encoders
    
    # Load sample training data to fit models
    try:
        df = pd.read_csv('data/WA_Fn-UseC_-HR-Employee-Attrition.csv')
    except:
        print("Warning: Could not load training data. Using demo mode.")
        return False
    
    # Prepare features
    categorical_cols = ['Department', 'Gender', 'JobRole', 'MaritalStatus', 'EducationField']
    numerical_cols = ['Age', 'MonthlyIncome', 'YearsAtCompany', 'JobSatisfaction', 
                      'WorkLifeBalance', 'EnvironmentSatisfaction', 'OverTime', 'MonthlyRate']
    
    # Handle categorical encoding
    df_processed = df.copy()
    for col in categorical_cols:
        if col in df_processed.columns:
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col].astype(str))
            encoders[col] = le
    
    # Create burnout risk score (simplified for demo)
    df['BurnoutRisk'] = (
        (5 - df.get('JobSatisfaction', 2).fillna(2)) * 0.3 +
        (5 - df.get('WorkLifeBalance', 2).fillna(2)) * 0.3 +
        (df.get('OverTime', 'No').fillna('No') == 'Yes') * 2 +
        (5 - df.get('EnvironmentSatisfaction', 2).fillna(2)) * 0.2
    )
    df['BurnoutRisk'] = (df['BurnoutRisk'] > 5).astype(int)
    df['SickDays'] = np.random.randint(0, 20, len(df))
    
    # Prepare feature set
    available_cols = [col for col in numerical_cols if col in df.columns]
    X = df_processed[available_cols].fillna(0)
    
    # Burnout Classification Model
    y_burnout = df.get('BurnoutRisk', pd.Series([0]*len(df)))
    burnout_model = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=10)
    burnout_model.fit(X, y_burnout)
    models['burnout'] = burnout_model
    
    # Sick Days Prediction Model
    y_sick = df.get('SickDays', pd.Series([5]*len(df)))
    sick_model = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=10)
    sick_model.fit(X, y_sick)
    models['sick_days'] = sick_model
    
    # Scaler for input normalization
    scaler = StandardScaler()
    scaler.fit(X)
    scalers['features'] = scaler
    
    return True


def get_hr_recommendations(burnout_risk, sick_days_forecast):
    """
    Generate HR intervention recommendations based on predictions.
    """
    recommendations = {
        'priority': 'Low',
        'actions': [],
        'urgency_color': 'green'
    }
    
    burnout_prob = burnout_risk if isinstance(burnout_risk, (int, float)) else burnout_risk[1]
    
    if burnout_prob > 0.7 or sick_days_forecast > 15:
        recommendations['priority'] = 'Critical'
        recommendations['urgency_color'] = 'red'
        recommendations['actions'] = [
            "Immediate one-on-one meeting with employee",
            "Consider workload reduction",
            "Offer mental health support/counseling",
            "Review work schedule and flexibility options"
        ]
    elif burnout_prob > 0.5 or sick_days_forecast > 10:
        recommendations['priority'] = 'High'
        recommendations['urgency_color'] = 'orange'
        recommendations['actions'] = [
            "Schedule check-in meeting",
            "Monitor workload and task distribution",
            "Encourage use of wellness programs",
            "Consider professional development opportunities"
        ]
    else:
        recommendations['priority'] = 'Standard'
        recommendations['urgency_color'] = 'green'
        recommendations['actions'] = [
            "Continue regular performance monitoring",
            "Promote workplace wellness initiatives",
            "Maintain open communication channels",
            "Support career development goals"
        ]
    
    return recommendations


# ==========================================
# Flask Routes
# ==========================================

@app.route('/')
def index():
    """Render the home page."""
    return render_template('index.html')


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    API endpoint for burnout prediction.
    Expects JSON with employee data.
    """
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['Age', 'MonthlyIncome', 'YearsAtCompany', 
                          'JobSatisfaction', 'WorkLifeBalance', 'EnvironmentSatisfaction']
        
        for field in required_fields:
            if field not in data or data[field] == '':
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Prepare input features
        input_data = np.array([[
            float(data.get('Age', 30)),
            float(data.get('MonthlyIncome', 5000)),
            float(data.get('YearsAtCompany', 5)),
            float(data.get('JobSatisfaction', 3)),
            float(data.get('WorkLifeBalance', 3)),
            float(data.get('EnvironmentSatisfaction', 3)),
            1.0 if data.get('OverTime', 'No') == 'Yes' else 0.0,
            float(data.get('MonthlyRate', 1000))
        ]])
        
        # Scale input
        if 'features' in scalers:
            input_scaled = scalers['features'].transform(input_data)
        else:
            input_scaled = input_data
        
        # Make predictions
        burnout_pred = models['burnout'].predict_proba(input_scaled)[0]
        sick_days_pred = max(0, models['sick_days'].predict(input_scaled)[0])
        
        # Get recommendations
        recommendations = get_hr_recommendations(burnout_pred[1], sick_days_pred)
        
        return jsonify({
            'success': True,
            'burnout_risk_score': round(burnout_pred[1] * 100, 2),
            'burnout_risk_category': 'High Risk' if burnout_pred[1] > 0.5 else 'Low Risk',
            'predicted_sick_days': round(sick_days_pred, 1),
            'recommendations': recommendations,
            'confidence': round(max(burnout_pred) * 100, 2)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/batch-predict', methods=['POST'])
def batch_predict():
    """
    API endpoint for batch predictions from CSV.
    """
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Read CSV
        df = pd.read_csv(file)
        required_cols = ['Age', 'MonthlyIncome', 'YearsAtCompany', 
                        'JobSatisfaction', 'WorkLifeBalance', 'EnvironmentSatisfaction']
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            return jsonify({'error': f'Missing columns: {", ".join(missing_cols)}'}), 400
        
        # Prepare features
        feature_cols = ['Age', 'MonthlyIncome', 'YearsAtCompany', 
                       'JobSatisfaction', 'WorkLifeBalance', 'EnvironmentSatisfaction',
                       'OverTime', 'MonthlyRate']
        
        # Fill missing columns with defaults
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0 if col == 'OverTime' else 5000
        
        X = df[feature_cols].fillna(0).values
        
        # Scale if available
        if 'features' in scalers:
            X_scaled = scalers['features'].transform(X)
        else:
            X_scaled = X
        
        # Predictions
        burnout_probs = models['burnout'].predict_proba(X_scaled)[:, 1]
        sick_days = models['sick_days'].predict(X_scaled)
        
        # Create results
        results = df.copy()
        results['burnout_risk_score'] = (burnout_probs * 100).round(2)
        results['burnout_category'] = results['burnout_risk_score'].apply(
            lambda x: 'High Risk' if x > 50 else 'Low Risk'
        )
        results['predicted_sick_days'] = sick_days.round(1)
        
        # Save results
        output_file = 'burnout_predictions.csv'
        results.to_csv(output_file, index=False)
        
        return jsonify({
            'success': True,
            'message': f'Processed {len(results)} employees',
            'high_risk_count': int((burnout_probs > 0.5).sum()),
            'average_burnout_score': round(burnout_probs.mean() * 100, 2),
            'average_sick_days': round(sick_days.mean(), 1),
            'file_download': output_file
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/info', methods=['GET'])
def get_info():
    """Return system information."""
    return jsonify({
        'system': 'Workforce Burnout Prediction System',
        'version': '1.0',
        'models': list(models.keys()),
        'status': 'operational' if models else 'models not loaded'
    })


@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def server_error(error):
    return jsonify({'error': 'Internal server error'}), 500


# ==========================================
# Main Entry Point
# ==========================================

if __name__ == '__main__':
    print("Initializing models...")
    if initialize_models():
        print("✓ Models loaded successfully")
    else:
        print("⚠ Running in demo mode - models may not be fully functional")
    
    print("\nStarting Flask server...")
    print("Visit http://localhost:5000 in your browser")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
