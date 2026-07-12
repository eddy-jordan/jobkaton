# Use a slim Python base image -- smaller image size, faster Render builds/deploys
FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (separate layer) so Docker caches this step
# and only reinstalls when requirements-api.txt itself actually changes
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Copy the app code and the model artifacts produced by train.py in CI
COPY credit_risk_app.py .
COPY static_ui.html .
COPY model.joblib .
COPY preprocessing_pipeline.joblib .

# Render (and most cloud platforms) inject the port to bind to via $PORT.
# Default to 8000 for local `docker run` testing.
ENV PORT=8000
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://localhost:{os.environ.get(\"PORT\", 8000)}/health')" || exit 1

CMD ["sh", "-c", "uvicorn credit_risk_app:app --host 0.0.0.0 --port ${PORT}"]
