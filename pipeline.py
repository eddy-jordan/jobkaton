"""
pipeline.py
-----------
Project 1: The Reproducible Data Pipeline (ETL for ML)

Takes a raw CSV, splits it into train/test, and applies a leakage-safe
preprocessing pipeline (imputing, scaling, encoding). Saves:
  - processed train/test arrays (as .npy)
  - the fitted pipeline object (as .joblib), so you can reuse it later
    on new incoming data without retraining preprocessing from scratch

Usage:
    python pipeline.py --input data/credit_risk.csv --target loan_status --outdir processed/

Why it's built this way:
  - Splitting BEFORE fitting the pipeline prevents data leakage
    (test data must never influence how we impute/scale/encode).
  - Using sklearn's Pipeline + ColumnTransformer means the exact same
    steps run in the exact same order every time -- no manual notebook
    steps to forget or run out of order.
"""

import argparse
import logging
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def load_data(input_path: str) -> pd.DataFrame:
    """Load the raw CSV into a DataFrame."""
    logger.info(f"Loading raw data from {input_path}")
    df = pd.read_csv(input_path)
    logger.info(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns")
    return df


def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows with physically impossible values using fixed, common-sense
    thresholds (not statistics computed from the data itself).

    This is safe to run BEFORE the train/test split: because the thresholds
    are hardcoded domain rules (e.g. nobody is 144 years old), not something
    learned from the dataset's distribution, applying this before splitting
    does not leak any test-set information into training.
    """
    before = len(df)

    if "person_age" in df.columns:
        df = df[df["person_age"] <= 100]

    if "person_emp_length" in df.columns:
        df = df[(df["person_emp_length"].isna()) | (df["person_emp_length"] <= 60)]

    dropped = before - len(df)
    if dropped:
        logger.info(f"Dropped {dropped} row(s) with physically impossible values (age > 100 or emp_length > 60)")

    return df.reset_index(drop=True)


def split_features_target(df: pd.DataFrame, target_col: str):
    """Separate features (X) from the target column (y)."""
    if target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found. Available columns: {list(df.columns)}"
        )
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def identify_column_types(X: pd.DataFrame):
    """Auto-detect numeric vs categorical columns."""
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    logger.info(f"Numeric columns ({len(numeric_cols)}): {numeric_cols}")
    logger.info(f"Categorical columns ({len(categorical_cols)}): {categorical_cols}")
    return numeric_cols, categorical_cols


def build_pipeline(numeric_cols, categorical_cols) -> ColumnTransformer:
    """
    Build the preprocessing pipeline.

    Numeric columns:    median impute -> standard scale
    Categorical columns: most-frequent impute -> one-hot encode
    """
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols),
    ])

    return preprocessor


def run_pipeline(input_path: str, target_col: str, outdir: str, test_size: float = 0.2, random_state: int = 42):
    df = load_data(input_path)
    df = clean_raw_data(df)
    X, y = split_features_target(df, target_col)
    numeric_cols, categorical_cols = identify_column_types(X)

    # Split BEFORE fitting anything -- this is what prevents data leakage.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y if y.nunique() < 20 else None
    )
    logger.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    preprocessor = build_pipeline(numeric_cols, categorical_cols)

    # Fit ONLY on training data.
    logger.info("Fitting preprocessing pipeline on training data only...")
    X_train_processed = preprocessor.fit_transform(X_train)
    # Test data is only ever transformed, never fitted on.
    X_test_processed = preprocessor.transform(X_test)

    # Handle sparse output from OneHotEncoder if present.
    if hasattr(X_train_processed, "toarray"):
        X_train_processed = X_train_processed.toarray()
        X_test_processed = X_test_processed.toarray()

    os.makedirs(outdir, exist_ok=True)

    np.save(os.path.join(outdir, "X_train.npy"), X_train_processed)
    np.save(os.path.join(outdir, "X_test.npy"), X_test_processed)
    np.save(os.path.join(outdir, "y_train.npy"), y_train.to_numpy())
    np.save(os.path.join(outdir, "y_test.npy"), y_test.to_numpy())

    pipeline_path = os.path.join(outdir, "preprocessing_pipeline.joblib")
    joblib.dump(preprocessor, pipeline_path)

    logger.info(f"Saved processed arrays and fitted pipeline to '{outdir}/'")
    logger.info(f"  X_train.npy shape: {X_train_processed.shape}")
    logger.info(f"  X_test.npy shape:  {X_test_processed.shape}")
    logger.info(f"Fitted pipeline saved to: {pipeline_path}")


def main():
    parser = argparse.ArgumentParser(description="Reproducible ML data pipeline (Project 1)")
    parser.add_argument("--input", required=True, help="Path to raw input CSV")
    parser.add_argument("--target", required=True, help="Name of the target column")
    parser.add_argument("--outdir", default="processed", help="Directory to save processed output")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction of data to hold out for testing")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    run_pipeline(
        input_path=args.input,
        target_col=args.target,
        outdir=args.outdir,
        test_size=args.test_size,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()
