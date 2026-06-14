"""
setup_test_model.py
-------------------
Test Fixture Generator for AI Assurance Toolkit — Module A

PURPOSE:
    Downloads the UCI German Credit Risk dataset, trains a Random Forest
    classifier, and saves:
        - model.pkl          (the trained model)
        - test_data.csv      (held-out test set with labels + sensitive attribute)

    These files are ready to pass directly into performance_evaluator.py
    and all subsequent Module A components.

USAGE:
    python setup_test_model.py

OUTPUTS:
    model.pkl
    test_data.csv
    train_data.csv   (for reference)
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# -----------------------------------------------------------------------
# 1. LOAD THE GERMAN CREDIT DATASET
# -----------------------------------------------------------------------
# The UCI German Credit dataset is available directly via a public URL.
# It has 1,000 records and classifies applicants as good (1) or bad (2) credit risk.
# We remap to 0 = good, 1 = bad for standard binary classification convention.

print("[INFO] Downloading UCI German Credit dataset...")

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"

column_names = [
    "checking_account_status",   # A1  — Status of existing checking account
    "duration_months",           # A2  — Duration of credit in months
    "credit_history",            # A3  — Credit history
    "purpose",                   # A4  — Purpose of loan
    "credit_amount",             # A5  — Credit amount
    "savings_account",           # A6  — Savings account / bonds
    "employment_since",          # A7  — Present employment since
    "installment_rate",          # A8  — Installment rate (% of disposable income)
    "personal_status_sex",       # A9  — Personal status and sex  ← sensitive attribute
    "other_debtors",             # A10 — Other debtors / guarantors
    "residence_since",           # A11 — Present residence since
    "property",                  # A12 — Property
    "age",                       # A13 — Age in years  ← sensitive attribute
    "other_installment_plans",   # A14 — Other installment plans
    "housing",                   # A15 — Housing
    "existing_credits",          # A16 — Number of existing credits
    "job",                       # A17 — Job
    "dependents",                # A18 — Number of people liable to provide maintenance
    "telephone",                 # A19 — Telephone
    "foreign_worker",            # A20 — Foreign worker
    "credit_risk",               # Target: 1 = Good, 2 = Bad
]

df = pd.read_csv(url, sep=" ", header=None, names=column_names)
print(f"[INFO] Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns.")

# -----------------------------------------------------------------------
# 2. FEATURE ENGINEERING
# -----------------------------------------------------------------------

# Remap target: 1 (Good) → 0, 2 (Bad) → 1
# Convention: 1 = high credit risk (positive class we care about detecting)
df["credit_risk"] = df["credit_risk"].map({1: 0, 2: 1})

# Extract a clean "sex" sensitive attribute from the combined personal_status_sex column.
# Codes: A91=male divorced, A92=female divorced/separated/married,
#        A93=male single, A94=male married/widowed, A95=female single
df["sex"] = df["personal_status_sex"].map({
    "A91": "male",
    "A92": "female",
    "A93": "male",
    "A94": "male",
    "A95": "female",
})

# Create an age group sensitive attribute for Component 2 subgroup analysis
df["age_group"] = pd.cut(
    df["age"],
    bins=[0, 25, 35, 50, 100],
    labels=["18-25", "26-35", "36-50", "51+"]
)

# Encode all remaining categorical columns as integers (label encoding)
# This keeps the model simple and universally loadable without a pipeline object
categorical_cols = [
    "checking_account_status", "credit_history", "purpose",
    "savings_account", "employment_since", "personal_status_sex",
    "other_debtors", "property", "other_installment_plans",
    "housing", "job", "telephone", "foreign_worker",
]

le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col].astype(str))

# -----------------------------------------------------------------------
# 3. TRAIN / TEST SPLIT
# -----------------------------------------------------------------------

# Features: drop target and the derived sensitive attribute columns
# (sensitive attributes are kept in the CSV for subgroup analysis but
#  are NOT fed into the model — important for fairness evaluation)
feature_cols = [c for c in df.columns if c not in ["credit_risk", "sex", "age_group", "age"]]

X = df[feature_cols]
y = df["credit_risk"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

print(f"[INFO] Train set: {len(X_train)} samples | Test set: {len(X_test)} samples")
print(f"[INFO] Class distribution in test set:")
print(y_test.value_counts().rename({0: "Good Credit (0)", 1: "High Risk (1)"}))

# -----------------------------------------------------------------------
# 4. TRAIN THE MODEL
# -----------------------------------------------------------------------

print("\n[INFO] Training Random Forest classifier...")

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=8,
    min_samples_leaf=5,
    random_state=42,
    class_weight="balanced",   # Important for imbalanced credit risk data
)
model.fit(X_train, y_train)
print("[INFO] Training complete.")

# -----------------------------------------------------------------------
# 5. SAVE MODEL
# -----------------------------------------------------------------------

joblib.dump(model, "model.pkl")
print("[INFO] Model saved to: model.pkl")

# -----------------------------------------------------------------------
# 6. SAVE TEST DATASET (with sensitive attributes for Component 2)
# -----------------------------------------------------------------------
# Reconstruct test set with sensitive attribute columns re-attached
# so Component 2 (subgroup analyzer) can use them.

test_df = X_test.copy()
test_df["age_group"] = df.loc[X_test.index, "age_group"].values
test_df["sex"] = df.loc[X_test.index, "sex"].values
test_df["credit_risk"] = y_test.values   # target column last

test_df.to_csv("test_data.csv", index=False)
print("[INFO] Test data saved to: test_data.csv")

# Save training data for reference / audit traceability
train_df = X_train.copy()
train_df["credit_risk"] = y_train.values
train_df.to_csv("train_data.csv", index=False)
print("[INFO] Train data saved to: train_data.csv")

# -----------------------------------------------------------------------
# 7. QUICK SANITY CHECK
# -----------------------------------------------------------------------

from sklearn.metrics import accuracy_score, f1_score
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n" + "="*55)
print("  SETUP COMPLETE — Quick Sanity Check")
print("="*55)
print(f"  Test Accuracy:  {acc:.4f}")
print(f"  Test F1 Score:  {f1:.4f}")
print("="*55)
print("\n  Files ready for Module A evaluation:")
print("    model.pkl        ← trained Random Forest")
print("    test_data.csv    ← 250 test records + labels")
print("    train_data.csv   ← 750 training records (reference)")
print("\n  Run the evaluator with:")
print("    python performance_evaluator.py \\")
print("        --model model.pkl \\")
print("        --dataset test_data.csv \\")
print("        --target credit_risk \\")
print("        --model-name \"German Credit Risk Classifier\"")
print("="*55 + "\n")
