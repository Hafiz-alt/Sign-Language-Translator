"""
============================================================
ISL Translator — Flask Backend (app.py)
============================================================
This server exposes:
    1. The frontend website (/)
    2. A REST endpoint (/predict_frame) which accepts base64
       images from the browser, runs ONNX MobileNetV3 inference
       and returns the prediction. 
============================================================
"""

import base64
import mimetypes
import os
import uuid
import numpy as np
import cv2

from flask import Flask, render_template, request, jsonify, send_from_directory
from camera import ISLModel

# ── Fix Windows MIME type for CSS ───────────────────────────
mimetypes.add_type("text/css", ".css")
mimetypes.add_type("application/javascript", ".js")

app = Flask(__name__)

model = ISLModel()
COLLECTION_MODE = os.environ.get("ISL_COLLECTION_MODE", "0") == "1"
COLLECTION_LABELS = set("abcdefghijklmnopqrstuvwxyz") | {"no_sign"}


def decode_image(data):
    """Decode a browser data URL into an OpenCV image."""
    if not data or "image" not in data:
        return None
    image_data = data["image"]
    idx = image_data.find("base64,")
    base64_img = image_data[idx + 7:] if idx != -1 else image_data
    img_bytes = base64.b64decode(base64_img)
    img_arr = np.frombuffer(img_bytes, dtype=np.uint8)
    return cv2.imdecode(img_arr, cv2.IMREAD_COLOR)

@app.route("/static/<path:filename>")
def serve_static(filename):
    response = send_from_directory("static", filename)
    if filename.endswith(".css"):
        response.headers["Content-Type"] = "text/css; charset=utf-8"
    elif filename.endswith(".js"):
        response.headers["Content-Type"] = "application/javascript; charset=utf-8"
    return response

@app.route("/")
def index():
    return render_template("index.html", collection_mode=COLLECTION_MODE)

@app.route("/predict_frame", methods=["POST"])
def predict_frame():
    """
    Receives a JSON payload containing { "image": "data:image/jpeg;base64,..." }
    """
    try:
        img_cv = decode_image(request.get_json())
        if img_cv is None:
            return jsonify({"error": "Invalid image data"}), 400
            
        result = model.predict_details(img_cv)

        return jsonify({
            "label": result["label"],
            "candidate": result["candidate"],
            "confidence": round(result["confidence"] * 100, 1),
            "margin": round(result["margin"] * 100, 1),
            "accepted": result["accepted"],
            "status": result["status"],
        })

    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return jsonify({"error": "Internal Server Error"}), 500


@app.route("/collect_sample", methods=["POST"])
def collect_sample():
    """Store labeled local webcam crops for training in collection mode."""
    if not COLLECTION_MODE:
        return jsonify({"error": "Collection mode is disabled"}), 403

    data = request.get_json()
    label = str((data or {}).get("label", "")).lower()
    if label not in COLLECTION_LABELS:
        return jsonify({"error": "Unsupported collection label"}), 400

    img_cv = decode_image(data)
    if img_cv is None:
        return jsonify({"error": "Invalid image data"}), 400

    class_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset", label)
    os.makedirs(class_dir, exist_ok=True)
    output_path = os.path.join(class_dir, f"webcam_{uuid.uuid4().hex}.jpg")
    if not cv2.imwrite(output_path, img_cv):
        return jsonify({"error": "Could not save sample"}), 500
    return jsonify({"label": label, "saved": os.path.basename(output_path)})


if __name__ == "__main__":
    print("ISL Alphabet Recognizer server starting...")
    port = int(os.environ.get("PORT", 7860))
    print(f"Open http://127.0.0.1:{port} in your browser\n")
    app.run(
        host="127.0.0.1" if COLLECTION_MODE else "0.0.0.0",
        port=port,
        debug=False,
    )
