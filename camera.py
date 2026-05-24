"""Model loading and conservative single-frame recognition."""

import json
import os

import cv2
import numpy as np


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ONNX_MODEL_PATH = os.path.join(BASE_DIR, "isl_model.onnx")
LABELS_PATH = os.path.join(BASE_DIR, "class_labels.json")
IMG_SIZE = (224, 224)
NO_SIGN_LABELS = {"no_sign", "none", "unknown", "background"}
MIN_CONFIDENCE = float(os.environ.get("ISL_MIN_CONFIDENCE", "0.70"))
MIN_MARGIN = float(os.environ.get("ISL_MIN_MARGIN", "0.15"))
IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


class ISLModel:
    """Loads the exported ONNX recognizer."""

    def __init__(self):
        self.model = None
        self.labels = None
        self.backend = None
        self.input_name = None
        self._load_model()

    def _load_model(self):
        if not os.path.exists(LABELS_PATH):
            print("Label file not found. Inference is disabled.")
            return
        with open(LABELS_PATH, "r", encoding="utf-8") as labels_file:
            self.labels = json.load(labels_file)

        if not os.path.exists(ONNX_MODEL_PATH):
            print("ONNX model file not found. Inference is disabled.")
            return

        import onnxruntime as ort

        self.model = ort.InferenceSession(
            ONNX_MODEL_PATH, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.model.get_inputs()[0].name
        self.backend = "onnx"
        print("ONNX model loaded:", len(self.labels), "classes")

    def _predict_probabilities(self, image_array):
        img = cv2.cvtColor(cv2.resize(image_array, IMG_SIZE), cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        batch = np.transpose(img, (2, 0, 1))[None, :, :, :]
        logits = self.model.run(None, {self.input_name: batch})[0][0]
        logits = logits - np.max(logits)
        values = np.exp(logits)
        return values / np.sum(values)

    def predict_details(self, image_array):
        """Return candidate information plus an accepted label when reliable."""
        empty_result = {
            "label": "",
            "candidate": "",
            "confidence": 0.0,
            "margin": 0.0,
            "accepted": False,
            "status": "model_unavailable",
        }
        if self.model is None or image_array is None:
            return empty_result

        try:
            predictions = self._predict_probabilities(image_array)
            ranked = np.argsort(predictions)[::-1]
            top_idx = int(ranked[0])
            second_idx = int(ranked[1]) if len(ranked) > 1 else top_idx
            confidence = float(predictions[top_idx])
            margin = confidence - float(predictions[second_idx])
            candidate = self.labels.get(str(top_idx), "unknown")
            result = {
                "label": "",
                "candidate": candidate,
                "confidence": round(confidence, 4),
                "margin": round(margin, 4),
                "accepted": False,
                "status": "uncertain",
            }
            if candidate.lower() in NO_SIGN_LABELS:
                result["status"] = "no_sign"
            elif confidence >= MIN_CONFIDENCE and margin >= MIN_MARGIN:
                result.update(label=candidate, accepted=True, status="accepted")
            return result
        except Exception as exc:
            print("Prediction error:", exc)
            empty_result["status"] = "prediction_error"
            return empty_result

    def predict_image(self, image_array):
        """Compatibility helper returning only accepted output."""
        result = self.predict_details(image_array)
        return result["label"], result["confidence"]
