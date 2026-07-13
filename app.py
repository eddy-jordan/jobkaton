"""
app.py
------
Project 5: The Containerized ML API

A FastAPI service that wraps the ONNX-optimized pneumonia classifier
(from Project 4) and exposes it via a /predict endpoint.

Why ONNX here instead of the PyTorch .pth file: the ONNX Runtime is a much
lighter dependency than full PyTorch, which means a smaller Docker image,
faster container startup, and lower memory usage in production -- exactly
the kind of deployment decision that's worth explaining in an interview.

Usage (local, without Docker):
    uvicorn app:app --host 0.0.0.0 --port 8000

Then send a request:
    curl -X POST "http://localhost:8000/predict" \
         -H "Content-Type: application/json" \
         -d "{\"image_base64\": \"<base64-encoded JPEG/PNG string>\"}"
"""

import base64
import io

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from PIL import Image
from pydantic import BaseModel

MODEL_PATH = "resnet50_pneumonia.onnx"
IMG_SIZE = 224
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]

# ImageNet normalization stats -- must match what was used during training (Project 3)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

app = FastAPI(
    title="Pneumonia Detection API",
    description="Classifies chest X-ray images as NORMAL or PNEUMONIA using a fine-tuned ResNet50 (ONNX).",
    version="1.0.0",
)

# Load the model once at startup, not on every request -- loading is the expensive part.
session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])


class PredictRequest(BaseModel):
    image_base64: str


class PredictResponse(BaseModel):
    prediction: str
    confidence: float
    probabilities: dict


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Decode raw image bytes and prepare them exactly as the model expects."""
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not decode image: {e}")

    image = image.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(image, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = arr.transpose(2, 0, 1)  # HWC -> CHW
    arr = np.expand_dims(arr, axis=0)  # add batch dimension
    return arr.astype(np.float32)


def softmax(logits: np.ndarray) -> np.ndarray:
    exp = np.exp(logits - np.max(logits))
    return exp / exp.sum()


@app.get("/", response_class=HTMLResponse)
def root():
    with open("static_ui.html") as f:
        return f.read()


@app.get("/health")
def health():
    """Basic health check -- used by Docker/orchestrators to confirm the service is alive."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    try:
        image_bytes = base64.b64decode(request.image_base64)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 string: {e}")

    input_array = preprocess_image(image_bytes)

    outputs = session.run(None, {"input": input_array})[0]
    probs = softmax(outputs[0])

    predicted_idx = int(np.argmax(probs))
    predicted_class = CLASS_NAMES[predicted_idx]
    confidence = float(probs[predicted_idx])

    return PredictResponse(
        prediction=predicted_class,
        confidence=confidence,
        probabilities={CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))},
    )
