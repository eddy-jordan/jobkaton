# Pre ASCEND Jobkathon — MLOps Portfolio
**From Jupyter Notebooks to Production AI Systems**

A 7-project portfolio covering the full machine learning lifecycle: reproducible data pipelines, experiment tracking, deep learning, model optimization, containerized deployment, drift monitoring, and a fully automated CI/CD system with a live production endpoint.

## 🔴 Live Demo
**Credit Risk Prediction API:** [https://jobkaton.onrender.com/docs](https://jobkaton.onrender.com/docs)
Automatically deployed by the GitHub Actions pipeline in [`capstone/`](./capstone) — every push retrains the model and redeploys it live, but only if the new model actually beats the previous best.

*(Note: free-tier hosting spins down after inactivity — the first request may take 10-30 seconds to wake up.)*

---

## Project 1 — Reproducible Data Pipeline
**TL;DR:** A single script that turns raw credit risk data into clean, model-ready arrays — no manual notebook preprocessing, no data leakage.

- **Tech stack:** Python, pandas, scikit-learn (`Pipeline`, `ColumnTransformer`)
- **File:** [`pipeline.py`](./pipeline.py)
- **What it handles:** missing value imputation, feature scaling, one-hot encoding, and removal of physically impossible values (e.g. age > 100) — all fit *only* on the training split to prevent leakage.
- **Quickstart:**
  ```bash
  pip install -r requirements-training.txt
  python pipeline.py --input credit_risk_dataset.csv --target loan_status --outdir processed/
  ```

## Project 2 — Experiment Tracking Mastery
**TL;DR:** Trained and compared 12 model configurations (Random Forest + Gradient Boosting), with every hyperparameter and metric logged to MLflow.

- **Tech stack:** scikit-learn, MLflow
- **File:** [`train_experiments.ipynb`](./train_experiments.ipynb)
- **Result:** best run — Gradient Boosting, 200 estimators, depth 5 — reached **93.4% accuracy, 0.826 F1, 0.949 AUC-ROC**.
- **Quickstart:**
  ```bash
  pip install -r requirements-training.txt
  jupyter notebook train_experiments.ipynb   # run all cells
  python -m mlflow ui                        # view dashboard at localhost:5000
  ```

## Project 3 — Deep Learning Application (Computer Vision)
**TL;DR:** Fine-tuned a ResNet50 (transfer learning) to classify chest X-rays as NORMAL or PNEUMONIA, with early stopping and LR scheduling to prevent overfitting.

- **Tech stack:** PyTorch, torchvision
- **File:** [`train_cv_model.ipynb`](./train_cv_model.ipynb)
- **Dataset:** [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) (Kaggle)
- **Result:** **83% test accuracy**, **95% recall on PNEUMONIA** — prioritizing catching true positives, appropriate for a medical screening context. Early stopping correctly halted training at epoch 2 when validation loss stopped improving. See [`training_validation_loss.png`](./training_validation_loss.png) and [`confusion_matrix.png`](./confusion_matrix.png).
- **Quickstart:**
  ```bash
  pip install -r requirements-training.txt
  # download & unzip the dataset into a chest_xray/ folder (train/val/test structure)
  jupyter notebook train_cv_model.ipynb
  ```

## Project 4 — Model Optimization
**TL;DR:** Converted the ResNet50 to ONNX format and benchmarked the latency difference.

- **Tech stack:** PyTorch, ONNX, ONNX Runtime
- **File:** [`optimize_model.ipynb`](./optimize_model.ipynb) → [`optimization_results.md`](./optimization_results.md)
- **Result:** **72.9% latency reduction** (366ms → 99ms per inference on CPU), with output predictions verified to match the original PyTorch model almost exactly.
- **Quickstart:**
  ```bash
  pip install -r requirements-training.txt
  jupyter notebook optimize_model.ipynb
  ```

## Project 5 — The Containerized ML API
**TL;DR:** The optimized pneumonia classifier, served through a FastAPI `/predict` endpoint and packaged in Docker.

- **Tech stack:** FastAPI, ONNX Runtime, Docker
- **Files:** [`app.py`](./app.py), [`Dockerfile`](./Dockerfile), [`requirements.txt`](./requirements.txt)
- **Tested:** real chest X-ray correctly classified as PNEUMONIA with 99.9% confidence (see [`test_predict.py`](./test_predict.py) for the test harness).
- **Quickstart:**
  ```bash
  docker build -t pneumonia-api .
  docker run -p 8000:8000 pneumonia-api
  curl http://localhost:8000/health
  ```

## Project 6 — Model Monitoring & Data Drift
**TL;DR:** Simulated a real-world data drift scenario (incoming applicant incomes suddenly 50% higher) and used Evidently AI to detect it automatically.

- **Tech stack:** Evidently AI, pandas
- **File:** [`detect_drift.ipynb`](./detect_drift.ipynb) → [`data_drift_report.html`](./data_drift_report.html)
- **Result:** `person_income` correctly flagged as drifted (drift score 0.61), while all 11 other columns correctly showed no drift — confirming the monitoring system isolates real shifts without false alarms.
- **Quickstart:**
  ```bash
  pip install -r requirements-training.txt
  jupyter notebook detect_drift.ipynb
  ```

## Project 7 — End-to-End MLOps System (Capstone)
**TL;DR:** A GitHub Actions pipeline that automatically retrains a model on every push, only promotes it if it beats the previous best, and redeploys the live API accordingly.

- **Tech stack:** GitHub Actions, MLflow, Docker, Render
- **Full details:** [`capstone/README.md`](./capstone/README.md) (includes architecture diagram)
- **Live:** [https://jobkaton.onrender.com](https://jobkaton.onrender.com)

---

## Overall Tech Stack
Python · scikit-learn · PyTorch · MLflow · ONNX Runtime · FastAPI · Docker · Evidently AI · GitHub Actions · Render

## Repository Structure
```
.
├── pipeline.py                    # Project 1
├── train_experiments.ipynb        # Project 2
├── train_cv_model.ipynb           # Project 3
├── optimize_model.ipynb           # Project 4 (+ optimization_results.md)
├── app.py, Dockerfile             # Project 5 (pneumonia API)
├── detect_drift.ipynb             # Project 6 (+ data_drift_report.html)
├── capstone/                      # Project 7 (credit risk API, CI/CD pipeline)
├── .github/workflows/             # GitHub Actions CI/CD definition
├── requirements-training.txt      # deps for Projects 1, 2, 3, 4, 6 notebooks
└── requirements.txt               # deps for Project 5's API specifically
```

## Author
**Edward** — [github.com/eddy-jordan](https://github.com/eddy-jordan)
