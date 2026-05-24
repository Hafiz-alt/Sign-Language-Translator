"""Evaluate the exported ONNX model on held-out prepared validation images."""

import argparse
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


IMG_SIZE = (224, 224)
MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(path):
    image = cv2.imread(str(path))
    image = cv2.cvtColor(cv2.resize(image, IMG_SIZE), cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0
    image = (image - MEAN) / STD
    return np.transpose(image, (2, 0, 1))


def evaluate(args):
    with open(args.labels, "r", encoding="utf-8") as label_file:
        label_map = json.load(label_file)
    classes = [label_map[str(index)] for index in range(len(label_map))]
    validation_root = Path(args.dataset) / "validation"
    paths = []
    truth = []
    for index, label in enumerate(classes):
        for path in sorted((validation_root / label).glob("*.jpg")):
            paths.append(path)
            truth.append(index)
    if not paths:
        raise FileNotFoundError("No prepared validation images found.")

    session = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    predictions = []
    for start in range(0, len(paths), args.batch_size):
        batch = np.asarray([preprocess(path) for path in paths[start:start + args.batch_size]])
        logits = session.run(None, {input_name: batch})[0]
        predictions.extend(np.argmax(logits, axis=1).tolist())

    accuracy = accuracy_score(truth, predictions)
    print(f"Validation accuracy: {accuracy:.4f}")
    print(classification_report(truth, predictions, target_names=classes, zero_division=0))
    matrix = confusion_matrix(truth, predictions)
    plt.figure(figsize=(15, 13))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.title(f"ISL Alphabet Recognizer Confusion Matrix (accuracy={accuracy:.3f})")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(args.output, dpi=250)
    print(f"Saved confusion matrix to {args.output}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="isl_model.onnx")
    parser.add_argument("--labels", default="class_labels.json")
    parser.add_argument("--dataset", default="prepared_dataset")
    parser.add_argument("--output", default="confusion_matrix.png")
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
