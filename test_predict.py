"""
test_predict.py
----------------
Small helper script to test the /predict endpoint with a real image file.

Usage:
    python test_predict.py path/to/xray_image.jpeg
"""

import base64
import sys

import requests

if len(sys.argv) != 2:
    print("Usage: python test_predict.py <path_to_image>")
    sys.exit(1)

image_path = sys.argv[1]

with open(image_path, "rb") as f:
    image_bytes = f.read()

image_b64 = base64.b64encode(image_bytes).decode("utf-8")

response = requests.post(
    "http://localhost:8000/predict",
    json={"image_base64": image_b64},
)

print("Status code:", response.status_code)
print("Response:", response.json())
