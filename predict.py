import joblib
import pandas as pd

# Load trained model
model = joblib.load("models/best_model_tuned.pkl")

# Sample customer
sample_customer = pd.DataFrame([
    {
        "person_age": 30,
        "person_income": 50000,
        "person_home_ownership": "RENT",
        "person_emp_length": 5,
        "loan_intent": "PERSONAL",
        "loan_grade": "B",
        "loan_amnt": 12000,
        "loan_int_rate": 11.5,
        "loan_percent_income": 0.24,
        "cb_person_default_on_file": "N",
        "cb_person_cred_hist_length": 7,
        "income_to_loan_ratio": 50000 / 12000
    }
])

print(sample_customer)

prediction = model.predict(sample_customer)
probability = model.predict_proba(sample_customer)

if prediction[0] == 0:
    print("\nLoan Prediction")
    print("----------------------------")
    print("Status      : Creditworthy")
    print(f"Confidence  : {probability[0][0] * 100:.2f}%")
else:
    print("\nLoan Prediction")
    print("----------------------------")
    print("Status      : Credit Risk")
    print(f"Confidence  : {probability[0][1] * 100:.2f}%")