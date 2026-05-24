"""Train and export the ISL alphabet recognizer with PyTorch.

Run ``prepare_dataset.py`` first. Training uses a pretrained MobileNetV3-small
backbone and exports ``isl_model.onnx`` for lightweight web inference.
"""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


IMAGE_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def data_loaders(dataset_root, batch_size, workers):
    root = Path(dataset_root)
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomAffine(degrees=10, translate=(0.06, 0.06), scale=(0.92, 1.08)),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    validation_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    train_dataset = datasets.ImageFolder(root / "train", transform=train_transform)
    validation_dataset = datasets.ImageFolder(root / "validation", transform=validation_transform)
    if train_dataset.classes != validation_dataset.classes:
        raise ValueError("Training and validation classes do not match.")
    loaders = {
        "train": DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=workers, pin_memory=torch.cuda.is_available(),
        ),
        "validation": DataLoader(
            validation_dataset, batch_size=batch_size, shuffle=False,
            num_workers=workers, pin_memory=torch.cuda.is_available(),
        ),
    }
    return loaders, train_dataset.classes


def build_model(num_classes):
    weights = models.MobileNet_V3_Small_Weights.DEFAULT
    model = models.mobilenet_v3_small(weights=weights)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    return model


def class_weights(loader, device):
    targets = np.asarray(loader.dataset.targets)
    counts = np.bincount(targets)
    weights = targets.size / (len(counts) * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def run_epoch(model, loader, criterion, optimizer, device):
    training = optimizer is not None
    model.train(training)
    running_loss = 0.0
    correct = 0
    total = 0
    predictions = []
    truth = []
    for inputs, labels in loader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(inputs)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
        predicted = logits.argmax(dim=1)
        running_loss += loss.item() * labels.size(0)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        predictions.extend(predicted.detach().cpu().tolist())
        truth.extend(labels.detach().cpu().tolist())
    return running_loss / total, correct / total, truth, predictions


def train(args):
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training device: {device}")
    loaders, classes = data_loaders(args.dataset, args.batch_size, args.workers)
    print(f"Classes ({len(classes)}): {classes}")
    print(f"Images: train={len(loaders['train'].dataset)}, validation={len(loaders['validation'].dataset)}")
    if "no_sign" not in classes:
        raise ValueError("Prepared dataset must contain the no_sign class.")

    model = build_model(len(classes)).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights(loaders["train"], device))
    best_accuracy = -1.0
    best_state = None
    for parameter in model.features.parameters():
        parameter.requires_grad = False
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        weight_decay=1e-4,
    )
    for epoch in range(args.epochs):
        if epoch == args.frozen_epochs:
            for parameter in model.features.parameters():
                parameter.requires_grad = True
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=args.fine_tune_learning_rate, weight_decay=1e-4
            )
        train_loss, train_accuracy, _, _ = run_epoch(
            model, loaders["train"], criterion, optimizer, device
        )
        val_loss, val_accuracy, truth, predictions = run_epoch(
            model, loaders["validation"], criterion, None, device
        )
        print(
            f"Epoch {epoch + 1}/{args.epochs}: "
            f"train loss={train_loss:.4f} acc={train_accuracy:.4f}; "
            f"validation loss={val_loss:.4f} acc={val_accuracy:.4f}"
        )
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, args.checkpoint)

    model.load_state_dict(best_state)
    _, _, truth, predictions = run_epoch(model, loaders["validation"], criterion, None, device)
    print(f"\nBest validation accuracy: {best_accuracy:.4f}")
    print(classification_report(truth, predictions, target_names=classes, zero_division=0))
    np.save(args.confusion_matrix, confusion_matrix(truth, predictions))

    labels = {str(index): label for index, label in enumerate(classes)}
    with open(args.labels, "w", encoding="utf-8") as labels_file:
        json.dump(labels, labels_file, indent=2)

    model.eval().cpu()
    dummy = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    torch.onnx.export(
        model,
        dummy,
        args.output,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    print(f"Exported model to {args.output}")
    print(f"Saved labels to {args.labels}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="prepared_dataset")
    parser.add_argument("--output", default="isl_model.onnx")
    parser.add_argument("--checkpoint", default="isl_model_best.pt")
    parser.add_argument("--labels", default="class_labels.json")
    parser.add_argument("--confusion-matrix", default="validation_confusion_matrix.npy")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--frozen-epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--fine-tune-learning-rate", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
