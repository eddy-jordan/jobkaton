# Use a slim Python base image -- smaller image size, faster pulls/deploys
FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (separate layer) so Docker can cache this step
# and skip reinstalling every time only the code changes, not requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the application code and the trained model
COPY app.py .
COPY resnet50_pneumonia.onnx .

EXPOSE 8000

# Basic container-level health check, so orchestrators (Docker/Render/K8s) know if the app is alive
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
