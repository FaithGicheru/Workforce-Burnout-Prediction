# Workforce Burnout Prediction, Sick-Day Forecasting & HR Intervention System

**Moringa School Data Science Capstone — 2025/26**  
Team: Faith Ng'endo, Alan Muchiri, William Nyawir, Sarah Owendi, Anthony Njeru  
Dataset: IBM HR Analytics Employee Attrition & Performance (1,470 employees, 35 features)

---

## Overview

This project builds a three-stage machine learning pipeline that identifies at-risk employees, forecasts absenteeism, and recommends targeted HR actions — all from standard HR data.

---

## Project Structure

```
Burnout_Prediction/
├── Data/
│   ├── WA_Fn-UseC_-HR-Employee-Attrition.csv   # Raw IBM HR dataset
│   └── hr_cleaned_stage2.csv                    # Cleaned output from Stage 1
├── main.ipynb                                   # Full modelling notebook (Stages 1–3)
├── app.py                                       # Streamlit dashboard
├── requirements.txt
└── README.md
```

---

## The Three Stages

### Stage 1 — Burnout Risk Classification

A `BurnoutRisk` target variable (Low / Medium / High) is engineered using a scoring function based on overtime, job satisfaction, work-life balance, environment satisfaction, income, and tenure. Two classifiers are trained and compared on an 80/20 stratified split.

| Model | Accuracy | F1-Score (macro) |
|---|---|---|
| Logistic Regression | 86.05% | 0.8525 |
| Random Forest | 94.56% | 0.9033 |

Random Forest is the production model for Stage 1. Feature importance confirms the five engineered scoring features (EnvironmentSatisfaction, JobSatisfaction, OverTime, WorkLifeBalance, MonthlyIncome) are the top predictors.

### Stage 2 — Sick-Day Forecasting

Since the dataset contains no absenteeism records, a synthetic `SickDays` target is constructed from the same stress indicators used in Stage 1. Three regression models are compared.

| Model | MAE (days) | R² Score |
|---|---|---|
| Linear Regression | 0.9599 | 0.8581 |
| Random Forest | 0.2595 | 0.9799 |
| XGBoost | 0.1282 | 0.9920 |

XGBoost is the best performer. The high R² reflects the synthetic target — real sick-day records would produce a harder problem.

### Stage 3 — HR Intervention Recommendations

A rule-based engine assigns each employee a priority level and a personalised list of interventions based on 10 workplace rules. No ML model is used here; the logic mirrors the burnout scoring from Stage 1.

| Priority | Count | Share |
|---|---|---|
| CRITICAL | 170 | 11.6% |
| HIGH | 510 | 34.7% |
| MODERATE | 376 | 25.6% |
| WATCH | 414 | 28.2% |

The five most commonly recommended actions are overtime reduction, flexible scheduling, compensation review, EAP access, and peer-recognition programmes.

---

## Feature Engineering

The original dataset has no burnout column. `BurnoutRisk` is constructed by scoring each employee across six stress indicators (max score = 12): scores >= 5 → High, 2–4 → Medium, 0–1 → Low.

---

## Setup

```bash
git clone <repo-url>
cd Burnout_Prediction
pip install -r requirements.txt
```

To run the notebook:
```bash
jupyter notebook main.ipynb
```

To run the Streamlit dashboard:
```bash
streamlit run app.py
```
Upload `WA_Fn-UseC_-HR-Employee-Attrition.csv` from the sidebar. The pipeline trains live on upload — no pre-saved models required.

## Demo

[https://burnoutprediction.streamlit.app/](https://burnoutprediction.streamlit.app/)

---

## Key Findings

- Overtime and low compensation are not just individual risk factors — they affect over 400 employees each and require organisation-level policy responses.
- 46% of the workforce (CRITICAL + HIGH) needs active HR attention based on current conditions.
- Random Forest and XGBoost both confirm that burnout cannot be captured by linear patterns alone; interaction effects between satisfaction scores and overtime status are significant.

---


## Next Steps

- Collect real sick-day records for a more robust regression model
- Add SHAP explanations so HR teams can audit individual predictions
- Retrain models quarterly as new employee data is collected
- Validate burnout scoring rules with HR domain experts

## This project can be useful for:

HR analytics portfolios
Workforce analytics demonstrations
Employee wellbeing research
Predictive analytics case studies
Organizational health monitoring

## Conclusion

This project demonstrates how workforce analytics and machine learning can be applied to employee wellbeing challenges.

By identifying burnout risk early and supporting proactive intervention strategies, organizations can improve retention, employee health outcomes, and operational performance.
