# Diabetes Risk Prediction App

A Streamlit-powered machine learning application that enables individuals to assess their personal risk for type II diabetes based on self-reported health data. Built with a mission to promote early intervention on the patient side, this tool combines accessible design with predictive analytics to empower users with health insights—supporting preventative care and healthcare equity.

## Overview

Diabetes is one of the most prevalent chronic conditions in the United States, and many individuals remain undiagnosed until symptoms emerge. This project applies classification models—**Logistic Regression**, **Random Forest**, and **XGBoost**—to predict diabetes risk using survey-based data. Users can interact with the deployed model via a web-based interface, input their health metrics, and receive a personalized risk score.

## Features

- ✅ Patient-facing web app built with **Streamlit**
- 🔢 Multiple models: Logistic Regression, XGBoost, Random Forest
- 🧠 Model selection toggle (in local version)
- 📈 Real-time prediction and feedback
- 📊 Feature impact visualization (for Logistic Regression)
- 🌐 Deployed publicly via Streamlit Cloud

## Repository Structure

```
├── app.py                     # Main Streamlit app
├── diabetes_logistic_model.pkl
├── diabetes_xgboost_model.pkl
├── diabetes_random_forest_model.pkl #NOTE: you will need to retrain the RF model and build your own .pkl file (our file was too large)
├── scaler.pkl                 # Scaler used for Random Forest/XGBoost
├── lr_scaler.pkl              # Scaler used for Logistic Regression
├── requirements.txt
└── README.md
```

## Models and Metrics

| Model              | Recall | ROC AUC |
|-------------------|--------|---------|
| Logistic Regression | 0.766  | 0.822   |
| Random Forest       | 0.791  | 0.821   |
| XGBoost             | 0.794  | 0.823   |

> 📌 We prioritized **recall** over accuracy to reduce the number of false negatives in health screening use cases.

## How to Use

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/diabetes-risk-predictor.git
cd diabetes-risk-predictor
```

### 2. Install Requirements

```bash
pip install -r requirements.txt
```

### 3. Run the App

```bash
streamlit run app.py
```

## Force for Good

This project was developed in alignment with the "Force for Good" mission to promote equitable access to health insights and preventive care. By enabling users to self-assess their diabetes risk, we aim to empower proactive decision-making and reduce healthcare disparities.
