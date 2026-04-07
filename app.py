"""
============================================================
ISL Translator — Flask Backend (app.py)
============================================================
This server exposes:
    1. The frontend website (/)
    2. A REST endpoint (/predict_frame) which accepts base64
       images from the browser, runs MobileNetV2 inference
       and returns the prediction. 
============================================================
"""

import base64
import mimetypes
import numpy as np
import cv2

from flask import Flask, render_template, request, jsonify, send_from_directory
from camera import ISLModel

# ── Fix Windows MIME type for CSS ───────────────────────────
mimetypes.add_type("text/css", ".css")
mimetypes.add_type("application/javascript", ".js")

app = Flask(__name__)

model = ISLModel()

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
    return render_template("index.html")

@app.route("/predict_frame", methods=["POST"])
def predict_frame():
    """
    Receives a JSON payload containing { "image": "data:image/jpeg;base64,..." }
    """
    try:
        data = request.get_json()
        if not data or "image" not in data:
            return jsonify({"error": "No image data provided"}), 400
        
        # Extract base64 image data (remove the data:image/jpeg;base64, header if present)
        idx = data["image"].find("base64,")
        if idx != -1:
            base64_img = data["image"][idx + 7:]
        else:
            base64_img = data["image"]
            
        # Decode base64 to numpy array
        img_bytes = base64.b64decode(base64_img)
        img_arr = np.frombuffer(img_bytes, dtype=np.uint8)
        img_cv = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        
        if img_cv is None:
            return jsonify({"error": "Invalid image data"}), 400
            
        # Run inference
        label, confidence = model.predict_image(img_cv)
        
        return jsonify({
            "label": label,
            "confidence": round(confidence * 100, 1)
        })

    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return jsonify({"error": "Internal Server Error"}), 500

import os
if __name__ == "__main__":
    print("🚀  ISL Translator Cloud-ready server starting...")
    port = int(os.environ.get("PORT", 7860))
    print(f"🌐  Open  http://127.0.0.1:{port}  in your browser\n")
    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
    )
