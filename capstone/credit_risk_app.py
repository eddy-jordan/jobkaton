"""
credit_risk_app.py
-------------------
Project 7: End-to-End MLOps System (Capstone)

Serves the credit risk model produced by train.py. This is the live
endpoint that the GitHub Actions pipeline redeploys automatically
whenever a retrained model beats the previous best.

Usage (local, without Docker):
    uvicorn credit_risk_app:app --host 0.0.0.0 --port 8000

Example request:
    curl -X POST "http://localhost:8000/predict" \
         -H "Content-Type: application/json" \
         -d '{
               "person_age": 25,
               "person_income": 55000,
               "person_home_ownership": "RENT",
               "person_emp_length": 3.0,
               "loan_intent": "EDUCATION",
               "loan_grade": "B",
               "loan_amnt": 8000,
               "loan_int_rate": 11.5,
               "loan_percent_income": 0.15,
               "cb_person_default_on_file": "N",
               "cb_person_cred_hist_length": 3
             }'
"""

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

MODEL_PATH = "model.joblib"
PREPROCESSOR_PATH = "preprocessing_pipeline.joblib"

app = FastAPI(
    title="Credit Risk Prediction API",
    description="Predicts loan default risk. Automatically redeployed by the CI/CD pipeline whenever a retrained model outperforms the previous best.",
    version="1.0.0",
)

model = joblib.load(MODEL_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)


class PredictRequest(BaseModel):
    person_age: int
    person_income: float
    person_home_ownership: str
    person_emp_length: float
    loan_intent: str
    loan_grade: str
    loan_amnt: float
    loan_int_rate: float
    loan_percent_income: float
    cb_person_default_on_file: str
    cb_person_cred_hist_length: int


class PredictResponse(BaseModel):
    prediction: str
    default_probability: float


@app.get("/", response_class=HTMLResponse)
def root():
    with open("static_ui.html") as f:
        return f.read()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    try:
        input_df = pd.DataFrame([request.model_dump()])
        input_processed = preprocessor.transform(input_df)
        probability = float(model.predict_proba(input_processed)[0][1])
        prediction = "DEFAULT_RISK" if probability >= 0.5 else "LOW_RISK"
        return PredictResponse(prediction=prediction, default_probability=probability)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")
