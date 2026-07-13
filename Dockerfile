# Use a slim Python base image -- smaller image size, faster pulls/deploys
FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (separate layer) so Docker can cache this step
# and skip reinstalling every time only the code changes, not requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the application code and the trained model
COPY app.py .
COPY static_ui.html .
COPY resnet50_pneumonia.onnx .

# Render (and most cloud platforms) inject the port to bind to via $PORT.
# Default to 8000 for local `docker run` testing.
ENV PORT=8000
EXPOSE 8000

# Basic container-level health check, so orchestrators (Docker/Render/K8s) know if the app is alive
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://localhost:{os.environ.get(\"PORT\", 8000)}/health')" || exit 1

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT}"]
