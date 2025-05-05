import streamlit as st
import numpy as np
import pickle

# File paths
model_files = {
    "XGBoost": "diabetes_xgboost_model.pkl",
    "Logistic Regression": "diabetes_logistic_model.pkl"
}

# Load scaler
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# --- Streamlit Config ---
st.set_page_config(page_title="Diabetes Risk Predictor", layout="centered")
st.title("Diabetes Risk Predictor")

# --- Model Selector ---
st.markdown("### Choose your model")
model_choice = st.selectbox("Select a prediction model:", list(model_files.keys()))
with open(model_files[model_choice], "rb") as f:
    model = pickle.load(f)


# --- Inputs ---
st.markdown("### Provide your health information")

def yes_no(label):
    return 1 if st.selectbox(label, ["No", "Yes"]) == "Yes" else 0

HighBP = yes_no("Do you have high Blood Pressure (systolic pressure of 130 mm Hg or higher, or a diastolic pressure of 80 mm Hg or higher)?")
HighChol = yes_no("Do you have high Cholesterol (total cholesterol is 200 mg/dL or higher)?")
CholCheck = yes_no("Have you had a cholesterol check in last 5 years?")
use_bmi_calc = st.checkbox("I don't know my BMI")

if use_bmi_calc:
    feet = st.number_input("Height (feet)", 3, 8, 5)
    inches = st.number_input("Height (inches)", 0, 11, 5)
    weight = st.number_input("Weight (lbs)", 50, 500, 150)
    height_m = (feet * 12 + inches) * 0.0254
    weight_kg = weight * 0.453592
    BMI = round(weight_kg / (height_m ** 2), 1)
else:
    BMI = st.number_input("BMI", 10.0, 60.0, 25.0)

Smoker = yes_no("Have you smoked over 100 cigarettes in your life?")
Stroke = yes_no("History of Stroke?")
HeartDiseaseorAttack = yes_no("History of Heart Disease or Heart Attack?")
PhysActivity = yes_no("Physical Activity in last 30 days?")
Fruits = yes_no("Eat fruit daily?")
Veggies = yes_no("Eat vegetables daily?")
HvyAlcoholConsump = yes_no("Heavy alcohol consumption? (adult men having > 14 drinks & adult women having > 7 drinks per week)")
AnyHealthcare = yes_no("Do you have any health coverage (including health insurance, prepaid plans such as HMO, etc.)?")
NoDocbcCost = yes_no("Has there been a time in the last 12 months where you couldn't see doctor due to cost?")
GenHlth = st.slider("General Health (1=Excellent, 5=Poor)", 1, 5, 3)
MentHlth = st.slider("Poor mental health days in the last month (0-30)", 0, 30, 0)
PhysHlth = st.slider("Poor physical health days in the last month (0-30)", 0, 30, 0)
DiffWalk = yes_no("Difficulty walking or climbing stairs?")
Sex = 1 if st.selectbox("Biological Sex", ["Female", "Male"]) == "Male" else 0

age_map = {
    "18–24": 1, "25–29": 2, "30–34": 3, "35–39": 4, "40–44": 5, "45–49": 6,
    "50–54": 7, "55–59": 8, "60–64": 9, "65–69": 10, "70–74": 11, "75–79": 12, "80+": 13
}
Age = age_map[st.selectbox("Age Group", list(age_map.keys()))]

edu_map = {
    "Never attended": 1, "Elementary": 2, "Some high school": 3,
    "High school grad": 4, "Some college": 5, "College grad": 6
}
Education = edu_map[st.selectbox("Education Level", list(edu_map.keys()))]

income_map = {
    "< $10k": 1, "$10k–15k": 2, "$15k–20k": 3, "$20k–25k": 4,
    "$25k–35k": 5, "$35k–50k": 6, "$50k–75k": 7, "> $75k": 8
}
Income = income_map[st.selectbox("Annual Income", list(income_map.keys()))]

# --- Predict ---
input_data = np.array([[HighBP, HighChol, CholCheck, BMI, Smoker, Stroke,
                        HeartDiseaseorAttack, PhysActivity, Fruits, Veggies,
                        HvyAlcoholConsump, AnyHealthcare, NoDocbcCost, GenHlth,
                        MentHlth, PhysHlth, DiffWalk, Sex, Age, Education, Income]])
scaled_input = scaler.transform(input_data)

if st.button("Check My Diabetes Risk"):
    prediction = model.predict(scaled_input)[0]
    prob = model.predict_proba(scaled_input)[0][1]

    if prediction == 1:
        st.error(f"⚠️ You may be at risk for diabetes. Estimated risk: **{prob:.2%}**")
    else:
        st.success(f"You are not currently predicted to be at risk. Estimated risk: **{prob:.2%}**")

