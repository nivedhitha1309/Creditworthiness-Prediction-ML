import os
import warnings

import joblib
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)


dataset_path = "Task1/dataset/credit_risk_dataset.csv"

df = pd.read_csv(dataset_path)

print("=" * 60)
print(" CREDIT RISK DATASET LOADED SUCCESSFULLY ")
print("=" * 60)

print("\nFirst 5 Rows")
print(df.head())

print("\nDataset Shape")
print(df.shape)

print("\nColumn Names")
print(df.columns.tolist())

print("\nData Types")
print(df.dtypes)

print("\nMissing Values")
print(df.isnull().sum())

print("\nDuplicate Rows")
print(df.duplicated().sum())

print("\nStatistical Summary")
print(df.describe())



# EXPLORATORY DATA ANALYSIS (EDA)


print("\n" + "=" * 60)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 60)


# Class Distribution


plt.figure(figsize=(6,4))

sns.countplot(
    x="loan_status",
    data=df,
    palette="viridis"
)

plt.title("Loan Status Distribution")
plt.xlabel("Loan Status")
plt.ylabel("Count")

plt.savefig("outputs/loan_status_distribution.png")

plt.show()


# Histograms


numerical_columns = df.select_dtypes(
    include=["int64","float64"]
).columns

for column in numerical_columns:

    plt.figure(figsize=(7,4))

    sns.histplot(
        df[column],
        kde=True,
        bins=30
    )

    plt.title(f"Distribution of {column}")

    plt.tight_layout()

    plt.savefig(f"outputs/{column}_histogram.png")

    plt.show()



# Boxplots


for column in numerical_columns:

    plt.figure(figsize=(6,3))

    sns.boxplot(
        x=df[column]
    )

    plt.title(f"Boxplot of {column}")

    plt.tight_layout()

    plt.savefig(f"outputs/{column}_boxplot.png")

    plt.show()



plt.figure(figsize=(10,8))

correlation = df.corr(numeric_only=True)

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")

plt.tight_layout()

plt.savefig("outputs/correlation_heatmap.png")

plt.show()




# DATA CLEANING & PREPROCESSING


print("\n" + "=" * 60)
print("DATA CLEANING & PREPROCESSING")
print("=" * 60)

# Remove Duplicate Records


print("\nDuplicate Rows Before Removing :", df.duplicated().sum())

df = df.drop_duplicates()

print("Duplicate Rows After Removing :", df.duplicated().sum())

print("\nNew Dataset Shape :", df.shape)



# Handle Missing Values


print("\nMissing Values Before Cleaning")
print(df.isnull().sum())

# Numerical Columns

df["person_emp_length"].fillna(
    df["person_emp_length"].median(),
    inplace=True
)

df["person_emp_length"] = df["person_emp_length"].fillna(
    df["person_emp_length"].median()

)
df["loan_int_rate"] = df["loan_int_rate"].fillna(
    df["loan_int_rate"].median()
)

print("\nMissing Values After Cleaning")
print(df.isnull().sum())



# Remove Unrealistic Age Values


print("\nMaximum Age Before Cleaning :", df["person_age"].max())

df = df[df["person_age"] <= 100]

print("Maximum Age After Cleaning :", df["person_age"].max())


# Create New Feature


df["income_to_loan_ratio"] = (
    df["person_income"] /
    df["loan_amnt"]
)

print("\nNew Feature Created Successfully")

print(df[[
    "person_income",
    "loan_amnt",
    "income_to_loan_ratio"
]].head())



# Separate Features and Target


X = df.drop("loan_status", axis=1)

y = df["loan_status"]

print("\nFeatures Shape :", X.shape)

print("Target Shape :", y.shape)






# FEATURE ENCODING & DATA PREPARATION


from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

print("\n" + "=" * 60)
print("FEATURE ENCODING & DATA PREPARATION")
print("=" * 60)


# Identify Numerical & Categorical Columns


categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
numerical_features = X.select_dtypes(exclude=["object"]).columns.tolist()

print("\nCategorical Features:")
print(categorical_features)

print("\nNumerical Features:")
print(numerical_features)


# Create Preprocessing Pipelines


numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

print("\nPreprocessing Pipeline Created Successfully!")


# Train-Test Split


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Data Shape :", X_train.shape)
print("Testing Data Shape :", X_test.shape)

print("\nTraining Target Shape :", y_train.shape)
print("Testing Target Shape :", y_test.shape)



# MODEL TRAINING


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

print("\n" + "=" * 60)
print("MODEL TRAINING")
print("=" * 60)


# Create Models


models = {
    "Logistic Regression": LogisticRegression(max_iter=3000),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )
}

trained_models = {}


# Train Models


for model_name, model in models.items():

    print(f"\nTraining {model_name}...")

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    pipeline.fit(X_train, y_train)

    trained_models[model_name] = pipeline

    print(f"{model_name} Trained Successfully!")




# MODEL EVALUATION


from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)

print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

results = []

best_model = None
best_accuracy = 0

for model_name, pipeline in trained_models.items():

    print(f"\n{'='*60}")
    print(f"{model_name}")
    print(f"{'='*60}")

    # Predictions
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC AUC  : {roc_auc:.4f}")

    print("\nClassification Report")
    print(classification_report(y_test, y_pred))

    # Save Results
    results.append([
        model_name,
        accuracy,
        precision,
        recall,
        f1,
        roc_auc
    ])

    # Save Best Model
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = pipeline

   
    # Confusion Matrix
  

    plt.figure(figsize=(6,5))

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        cmap="Blues"
    )

    plt.title(f"{model_name} Confusion Matrix")

    plt.tight_layout()

    plt.savefig(
        f"outputs/{model_name.replace(' ','_')}_confusion_matrix.png"
    )

    plt.show()

  
    # ROC Curve
    

    RocCurveDisplay.from_predictions(
        y_test,
        y_prob
    )

    plt.title(f"{model_name} ROC Curve")

    plt.tight_layout()

    plt.savefig(
        f"outputs/{model_name.replace(' ','_')}_roc_curve.png"
    )

    plt.show()

   
# MODEL COMPARISON


print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

comparison_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC"
    ]
)

print(comparison_df)

comparison_df.to_csv(
    "outputs/model_comparison.csv",
    index=False
)


# Accuracy Comparison Graph


plt.figure(figsize=(8,5))

sns.barplot(
    data=comparison_df,
    x="Model",
    y="Accuracy"
)

plt.title("Model Accuracy Comparison")

plt.xticks(rotation=10)

plt.tight_layout()

plt.savefig(
    "outputs/model_accuracy_comparison.png"
)

plt.show()

print("\nModel comparison saved successfully!")


# SAVE BEST MODEL


joblib.dump(
    best_model,
    "models/best_model.pkl"
)

print("\nBest model saved successfully!")
print("Location : models/best_model.pkl")



# FEATURE IMPORTANCE


rf_model = best_model.named_steps["model"]

feature_names = best_model.named_steps[
    "preprocessor"
].get_feature_names_out()

importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": rf_model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 10 Important Features")

print(importance.head(10))

plt.figure(figsize=(12,7))

sns.barplot(
    data=importance.head(10),
    x="Importance",
    y="Feature"
)

plt.title("Top 10 Important Features")

plt.tight_layout()

plt.savefig(
    "outputs/feature_importance.png"
)

plt.show()


# ==========================================================
# HYPERPARAMETER TUNING - RANDOM FOREST
# ==========================================================

from sklearn.model_selection import GridSearchCV

print("\n" + "=" * 60)
print("HYPERPARAMETER TUNING - RANDOM FOREST")
print("=" * 60)

# Create a fresh Random Forest pipeline
rf_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(random_state=42))
    ]
)

# Parameters to search
param_grid = {
    "model__n_estimators": [100, 200, 300],
    "model__max_depth": [None, 10, 20],
    "model__min_samples_split": [2, 5],
    "model__min_samples_leaf": [1, 2]
}

grid_search = GridSearchCV(
    estimator=rf_pipeline,
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
    verbose=2
)

print("\nTraining GridSearchCV...")
grid_search.fit(X_train, y_train)

print("\nGrid Search Completed!")

print("\nBest Parameters:")
print(grid_search.best_params_)

print("\nBest Cross Validation Accuracy:")
print(round(grid_search.best_score_, 4))

# Best tuned model
best_rf = grid_search.best_estimator_

# Evaluate on test data
y_pred = best_rf.predict(X_test)
y_prob = best_rf.predict_proba(X_test)[:, 1]

print("\n")
print("=" * 60)
print("TUNED RANDOM FOREST RESULTS")
print("=" * 60)

print("Accuracy :", round(accuracy_score(y_test, y_pred), 4))
print("Precision:", round(precision_score(y_test, y_pred), 4))
print("Recall   :", round(recall_score(y_test, y_pred), 4))
print("F1 Score :", round(f1_score(y_test, y_pred), 4))
print("ROC AUC  :", round(roc_auc_score(y_test, y_prob), 4))

print("\nClassification Report")
print(classification_report(y_test, y_pred))

# Save the tuned model
joblib.dump(best_rf, "models/best_model_tuned.pkl")

print("\nTuned model saved successfully!")
print("Location : models/best_model_tuned.pkl")