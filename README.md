# Creditworthiness-Prediction-ML
Machine Learning project to predict loan creditworthiness using classification algorithms.
Project Overview

Creditworthiness Prediction is a Machine Learning project that predicts whether a loan applicant is Creditworthy or Credit Risk based on financial and personal information.

The project follows a complete Machine Learning pipeline including:

- Exploratory Data Analysis (EDA)
- Data Cleaning
- Feature Engineering
- Data Preprocessing
- Model Training
- Hyperparameter Tuning
- Model Evaluation
- Model Saving
- Loan Prediction

The objective is to assist financial institutions in making informed loan approval decisions while reducing the risk of loan defaults.

Problem Statement

Banks receive thousands of loan applications every day.
Approving loans without evaluating an applicant's financial background may lead to financial losses.
This project predicts the probability of loan default using Machine Learning, helping financial institutions make faster and more reliable lending decisions.

Dataset

Dataset contains customer financial information such as:

- Person Age
- Annual Income
- Employment Length
- Home Ownership
- Loan Intent
- Loan Grade
- Loan Amount
- Interest Rate
- Loan Percentage of Income
- Previous Credit History
- Previous Loan Default Status

Target Variable:

loan_status

- 0 → Creditworthy
- 1 → Credit Risk

---

Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn
- Joblib
- GridSearchCV


Machine Learning Pipeline

1.Data Cleaning

- Removed duplicate records
- Filled missing values
- Removed unrealistic age values
- Feature engineering
  
2. Feature Engineering

Created a new feature:

Income_to_Loan_Ratio = Person Income / Loan Amount

This significantly improved model performance.

3. Data Preprocessing

Categorical Features:

- Home Ownership
- Loan Intent
- Loan Grade
- Previous Loan Default

Numerical Features:
- Age
- Income
- Employment Length
- Loan Amount
- Interest Rate
- Credit History Length
- Income-to-Loan Ratio

Preprocessing techniques:

- Median Imputation
- Standard Scaling
- One-Hot Encoding
  
Models Trained:

- Logistic Regression
- Decision Tree
- Random Forest

Hyperparameter tuning was performed using GridSearchCV.

Model Performance

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |

| Logistic Regression | 87.07% | 78.05% | 56.91% | 65.82% | 86.70% |
| Decision Tree | 88.94% | 73.54% | 77.22% | 75.34% | 84.72% |
| Random Forest (Best) | 93.38% | 96.52% | 72.36% | 82.71% | 92.95% |

Best Model:
Random Forest Classifier
After hyperparameter tuning:

- Accuracy: 93.38%
- ROC-AUC: 92.95%

The trained model is saved using Joblib for future predictions.

Important Features

Top contributing features:

- Income to Loan Ratio
- Loan Percentage of Income
- Annual Income
- Interest Rate
- Home Ownership
- Loan Amount
- Employment Length
- Loan Grade
- Age
- Mortgage Status

How to Run the Project

1 Clone Repository

bash
git clone https://github.com/nivedhitha1309/Creditworthiness-Prediction-ML.git

2 Move into Project

bash
cd Creditworthiness-Prediction-ML

3 Install Dependencies

bash
pip install -r requirements.txt

4 Train the Model

bash
python train_model.py

5 Run Prediction

bash
python predict.py

Sample Prediction

Example Input

Age = 30

Income = 50000

Employment Length = 5

Loan Amount = 12000

Interest Rate = 11.5%

Prediction

Status : Creditworthy

Confidence : 95.50%

Project Outputs

The project generates:

- Confusion Matrix
- ROC Curve
- Feature Importance Plot
- Model Comparison CSV
- Saved Trained Model

Author

NIVEDHITHA S

Machine Learning | Data Science Enthusiast | Artificial Intelligence 

GitHub:

https://github.com/nivedhitha1309

If you found this project useful

Please consider giving it a STAR on GitHub.



