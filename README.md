# Workforce Burnout Prediction, Sick-Day Forecasting & HR Intervention Recommendation System

**Group 5** — Faith Ngendo · Alan Muchiri · William Nyawir · Sarah Owendi · Anthony Njeru

---

## Overview

This project builds an end-to-end machine learning HR analytics system that proactively identifies employee burnout risk, forecasts absenteeism, and recommends personalised HR interventions — all from a single employee dataset.

The system replaces reactive HR decision-making with an evidence-based, three-stage pipeline that processes 1,470 employee records and produces actionable, prioritised outputs for every individual.

---

## Project Structure

```
├── Data/
│   ├── WA_Fn-UseC_-HR-Employee-Attrition.csv   # Raw IBM HR dataset
│   ├── hr_cleaned_stage2.csv                    # Cleaned dataset output from Stage 1
│   └── hr_intervention_recommendations.csv      # Final recommendations output from Stage 3
├── main.ipynb                                   # Main notebook (all 3 stages)
└── README.md
```

---

## Dataset

**IBM HR Analytics Employee Attrition & Performance Dataset** (via Kaggle)

- 1,470 employee records · 35 attributes
- Key features: `JobSatisfaction`, `WorkLifeBalance`, `OverTime`, `MonthlyIncome`, `YearsAtCompany`, `EnvironmentSatisfaction`, `PerformanceRating`, `Department`, `Age`
- No missing values · No duplicate rows
- 3 constant columns dropped: `EmployeeCount`, `StandardHours`, `Over18`

---

## System Architecture

### Stage 1 — Burnout Risk Classification

A custom scoring function engineers the `BurnoutRisk` target variable (Low / Medium / High) from 9 workplace stress indicators. Two classifiers are trained and compared on an 80/20 stratified split.

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Logistic Regression | 86.05% | 0.8796 | 0.8311 | 0.8525 |
| **Random Forest** ✓ | **94.56%** | **0.9568** | **0.8706** | **0.9033** |

**Random Forest** is the best-performing classifier. Its top features — `EnvironmentSatisfaction` (18.2%), `JobSatisfaction` (14.9%), `OverTime` (13.8%), `WorkLifeBalance` (8.7%), `MonthlyIncome` (8.0%) — independently validate the burnout scoring logic.

Class distribution: **Low 28.2%** · **Medium 60.3%** · **High 11.6%**

---

### Stage 2 — Sick-Day Forecasting (Regression)

Since the dataset contains no sick-day records, a synthetic `SickDays` target is engineered from burnout stress factors (range: 13–25 days, mean: ~21 days). Three regression models are trained and compared.

| Model | MAE (days) | RMSE (days) | R² Score |
|---|---|---|---|
| Linear Regression | 0.9599 | 1.1565 | 0.8581 |
| Random Forest | 0.2595 | 0.4349 | 0.9799 |
| **XGBoost** ✓ | **0.1282** | **0.2739** | **0.9920** |

**XGBoost** achieves the best results, predicting sick days to within 0.13 days on average. Top predictors: `MonthlyIncome` (22.8%), `OverTime` (22.3%), `JobSatisfaction` (21.6%).

> **Note:** High R² scores reflect the synthetic nature of the target variable. Real sick-day records would produce a more challenging benchmark.

---

### Stage 3 — HR Intervention Recommendation Engine

A rule-based engine with 10 workplace rules generates personalised, prioritised HR interventions for every employee based on their burnout risk and stress profile. No additional training data required.

| Priority | Count | % of Workforce |
|---|---|---|
|  CRITICAL | 170 | 11.6% |
|  HIGH | 510 | 34.7% |
|  MODERATE | 376 | 25.6% |
|  WATCH | 414 | 28.2% |

**46.3% of the workforce** (CRITICAL + HIGH) requires active, near-term HR intervention.

Average interventions assigned: High risk **8.85** · Medium risk **3.82** · Low risk **1.79**

Top 5 most recommended interventions across the organisation:

1. Peer-recognition and rewards programme — 478 employees
2. Reduce or cap overtime hours immediately — 416 employees
3. Flexible / compressed work-week scheduling — 416 employees
4. Immediate compensation review — 395 employees
5. Employee Assistance Programme (EAP) access — 395 employees

---

## How to Run

### Requirements

```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn
```

### Steps

1. Clone the repository and place the dataset in the `Data/` folder
2. Open `main.ipynb` in Jupyter Notebook or JupyterLab
3. Run all cells top to bottom — each section builds on the previous
4. Outputs are saved automatically:
   - `Data/hr_cleaned_stage2.csv` — after Section 11
   - `Data/hr_intervention_recommendations.csv` — after Section 15

---

## Key Findings

- **Overtime and low compensation are systemic, organisation-wide problems** — not just individual risk factors. They affect over 400 employees each and demand policy-level responses.
- **Random Forest validated our feature engineering**: its top predictors matched exactly the features used to construct `BurnoutRisk`, confirming the scoring logic captures genuine workplace stress.
- **11.6% of employees are at critical burnout risk** and were automatically flagged for counselling and a structured 30-day HR recovery plan.
- The full pipeline processes 1,470 employees end-to-end and produces ranked, actionable recommendations without any manual HR review.

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python 3 | Core language |
| pandas / NumPy | Data manipulation |
| scikit-learn | Preprocessing, models, evaluation |
| XGBoost | Gradient boosting regression |
| Matplotlib / Seaborn | Visualisation |
| Jupyter Notebook | Development environment |

---

## Limitations & Next Steps

- Replace synthetic `SickDays` with real absenteeism records for a more robust regression benchmark
- Add **SHAP explainability** so HR managers can understand individual predictions
- Deploy as a **Streamlit dashboard** for live upload-and-predict HR use
- Retrain models quarterly as new employee data is collected
- Validate the burnout scoring logic with HR domain experts and real performance reviews
