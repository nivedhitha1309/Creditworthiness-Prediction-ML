import streamlit as st
import pandas as pd
import joblib

# -------------------------------------------------------
# Load Trained Model
# -------------------------------------------------------

model = joblib.load("models/best_model_tuned.pkl")

st.set_page_config(
    page_title="Creditworthiness Prediction",
    page_icon="💳",
    layout="centered"
)

st.title("💳 Creditworthiness Prediction System")

st.write(
    """
Predict whether a loan applicant is **Creditworthy**
or **Credit Risk** using Machine Learning.
"""
)

st.markdown("---")

st.header("Applicant Details")

# -----------------------------
# User Inputs
# -----------------------------

person_age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=30
)

person_income = st.number_input(
    "Annual Income",
    min_value=1000,
    value=50000
)

person_home_ownership = st.selectbox(
    "Home Ownership",
    ["RENT", "OWN", "MORTGAGE", "OTHER"]
)

person_emp_length = st.number_input(
    "Employment Length (Years)",
    min_value=0,
    max_value=50,
    value=5
)

loan_intent = st.selectbox(
    "Loan Purpose",
    [
        "PERSONAL",
        "EDUCATION",
        "MEDICAL",
        "VENTURE",
        "HOMEIMPROVEMENT",
        "DEBTCONSOLIDATION"
    ]
)

loan_grade = st.selectbox(
    "Loan Grade",
    ["A", "B", "C", "D", "E", "F", "G"]
)

loan_amnt = st.number_input(
    "Loan Amount",
    min_value=500,
    value=12000
)

loan_int_rate = st.number_input(
    "Interest Rate (%)",
    min_value=1.0,
    max_value=40.0,
    value=11.5
)

loan_percent_income = st.number_input(
    "Loan Percent Income",
    min_value=0.01,
    max_value=1.0,
    value=0.24
)

cb_person_default_on_file = st.selectbox(
    "Previous Loan Default",
    ["N", "Y"]
)

cb_person_cred_hist_length = st.number_input(
    "Credit History Length",
    min_value=1,
    max_value=40,
    value=7
)

# -------------------------------------
# Prediction Button
# -------------------------------------

if st.button("Predict"):

    income_to_loan_ratio = person_income / loan_amnt

    sample = pd.DataFrame([{
        "person_age": person_age,
        "person_income": person_income,
        "person_home_ownership": person_home_ownership,
        "person_emp_length": person_emp_length,
        "loan_intent": loan_intent,
        "loan_grade": loan_grade,
        "loan_amnt": loan_amnt,
        "loan_int_rate": loan_int_rate,
        "loan_percent_income": loan_percent_income,
        "cb_person_default_on_file": cb_person_default_on_file,
        "cb_person_cred_hist_length": cb_person_cred_hist_length,
        "income_to_loan_ratio": income_to_loan_ratio
    }])

    prediction = model.predict(sample)[0]

    probability = model.predict_proba(sample)[0]

    confidence = max(probability) * 100

    st.markdown("---")

    if prediction == 0:
        st.success("Creditworthy")
    else:
        st.error("Credit Risk")

    st.metric(
        label="Confidence",
        value=f"{confidence:.2f}%"
    )

    st.write("Prediction Probabilities")

    st.write(
        pd.DataFrame(
            {
                "Class": ["Creditworthy", "Credit Risk"],
                "Probability": probability
            }
        )
    )