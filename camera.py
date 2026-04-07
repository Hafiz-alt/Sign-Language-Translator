"""
============================================================
ISL Translator — Inference Module (camera.py)
============================================================
This module loads the ML model and runs prediction on explicitly
provided image arrays (e.g. uploaded from the frontend).
============================================================
"""

import os
import json
import cv2
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

MODEL_PATH  = "isl_model.h5"
LABELS_PATH = "class_labels.json"
IMG_SIZE    = (224, 224)


class ISLModel:
    """
    Loads the MobileNetV2 model and performs inference on
    a given image (expected as OpenCV BGR numpy array).
    """

    def __init__(self):
        self.model  = None
        self.labels = None
        self._load_model()

    def _load_model(self):
        if os.path.exists(MODEL_PATH) and os.path.exists(LABELS_PATH):
            print("📦  Loading model from", MODEL_PATH)
            self.model = load_model(MODEL_PATH)
            with open(LABELS_PATH, "r") as f:
                self.labels = json.load(f)
            print("✅  Model loaded —", len(self.labels), "classes")
        else:
            print("⚠️   Model or label file not found. Inference will return empty results.")

    def predict_image(self, image_array):
        """
        Pre-process a BGR frame and run it through MobileNetV2.
        
        Returns:
            (label: str, confidence: float)   e.g. ("A", 0.93)
        """
        if self.model is None or image_array is None:
            return "", 0.0

        try:
            # Resize and convert colour space
            img = cv2.resize(image_array, IMG_SIZE)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Preprocess for MobileNetV2
            img = preprocess_input(img.astype(np.float32))

            # Expand dims:  (224,224,3) → (1,224,224,3)
            img = np.expand_dims(img, axis=0)

            # Run inference
            predictions = self.model.predict(img, verbose=0)
            class_idx   = int(np.argmax(predictions[0]))
            confidence  = float(predictions[0][class_idx])

            label = self.labels.get(str(class_idx), "Unknown")
            return label, round(confidence, 3)
        except Exception as e:
            print("Predict error:", e)
            return "", 0.0
