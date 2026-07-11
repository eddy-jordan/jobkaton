"""
train.py
--------
Project 7: End-to-End MLOps System (Capstone)

CI-friendly retraining script. Unlike the Project 2 notebook (which explores
12 hyperparameter combinations interactively), this script trains ONE
specific, already-chosen configuration -- the best one found during that
exploration -- and is built to run non-interactively inside GitHub Actions
on every push.

What it does:
  1. Loads and cleans the raw dataset (same rules as pipeline.py).
  2. Builds the same preprocessing pipeline as Project 1.
  3. Trains a Gradient Boosting model using the best hyperparameters found
     during Project 2's experiment tracking.
  4. Evaluates on a held-out test split.
  5. Logs the run to MLflow (local file-based tracking -- no server needed,
     works fine inside a CI runner).
  6. Compares the new model's accuracy against the previous best
     (stored in metrics_baseline.json, committed to the repo).
  7. If the new model is better: saves the new model + preprocessing
     pipeline as artifacts, updates the baseline file, and signals
     "model_improved=true" to GitHub Actions.
  8. If not: signals "model_improved=false" and leaves the previous
     best model/baseline untouched.

This "only promote if better" check is what prevents a bad commit from
silently pushing a worse model to production.
"""

import json
import os
import sys

import joblib
import mlflow
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = "credit_risk_dataset.csv"
TARGET_COL = "loan_status"
BASELINE_PATH = "metrics_baseline.json"
CURRENT_METRICS_PATH = "metrics_current.json"
MODEL_OUTPUT_PATH = "model.joblib"
PREPROCESSOR_OUTPUT_PATH = "preprocessing_pipeline.joblib"

# Best hyperparameters found in Project 2's experiment tracking
BEST_PARAMS = {"n_estimators": 200, "learning_rate": 0.1, "max_depth": 5}


def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """Same fixed-threshold cleaning rule as pipeline.py (Project 1)."""
    df = df[df["person_age"] <= 100]
    df = df[(df["person_emp_length"].isna()) | (df["person_emp_length"] <= 60)]
    return df.reset_index(drop=True)


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols),
    ])


def load_baseline_accuracy() -> float:
    """Return the previous best accuracy, or -1 if this is the first ever run."""
    if not os.path.exists(BASELINE_PATH):
        return -1.0
    with open(BASELINE_PATH) as f:
        return json.load(f)["accuracy"]


def main():
    print("Loading and cleaning data...")
    df = pd.read_csv(DATA_PATH)
    df = clean_raw_data(df)

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = build_preprocessor(X_train)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    print(f"Training GradientBoostingClassifier with params: {BEST_PARAMS}")
    mlflow.set_experiment("credit-risk-capstone-ci")
    with mlflow.start_run():
        model = GradientBoostingClassifier(random_state=42, **BEST_PARAMS)
        model.fit(X_train_processed, y_train)

        preds = model.predict(X_test_processed)
        proba = model.predict_proba(X_test_processed)[:, 1]

        accuracy = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        auc = roc_auc_score(y_test, proba)

        for key, value in BEST_PARAMS.items():
            mlflow.log_param(key, value)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("auc_roc", auc)

        print(f"New model -> accuracy={accuracy:.4f}  f1={f1:.4f}  auc={auc:.4f}")

    current_metrics = {"accuracy": accuracy, "f1_score": f1, "auc_roc": auc}
    with open(CURRENT_METRICS_PATH, "w") as f:
        json.dump(current_metrics, f, indent=2)

    baseline_accuracy = load_baseline_accuracy()
    print(f"Baseline accuracy: {baseline_accuracy}")
    print(f"New accuracy:      {accuracy}")

    model_improved = accuracy > baseline_accuracy

    if model_improved:
        print("New model IMPROVED over baseline. Saving as the new best model.")
        joblib.dump(model, MODEL_OUTPUT_PATH)
        joblib.dump(preprocessor, PREPROCESSOR_OUTPUT_PATH)
        with open(BASELINE_PATH, "w") as f:
            json.dump(current_metrics, f, indent=2)
    else:
        print("New model did NOT beat the baseline. Keeping the existing best model.")

    # Signal the result to GitHub Actions so the next job can decide whether to deploy
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a") as f:
            f.write(f"model_improved={'true' if model_improved else 'false'}\n")

    print(f"model_improved={'true' if model_improved else 'false'}")


if __name__ == "__main__":
    main()
