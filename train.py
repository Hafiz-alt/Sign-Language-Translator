"""
============================================================
ISL Translator — Transfer Learning Training Script (train.py)
[FAST PROTOTYPE VERSION]
============================================================
"""

import os
import json
import numpy as np

# ── TensorFlow / Keras imports ──────────────────────────────
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    GlobalAveragePooling2D,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)

# ════════════════════════════════════════════════════════════
#  CONFIGURATION — adjust these to match your setup
# ════════════════════════════════════════════════════════════
DATASET_DIR   = "dataset"          # Root folder of class sub-dirs
IMG_SIZE      = (224, 224)          # MobileNetV2 native resolution
BATCH_SIZE    = 32                  # Training batch size

# 🛑 PROTOTYPE HACK: Changed to 1 so it finishes instantly 🛑
EPOCHS        = 1                   
LEARNING_RATE = 1e-4                # Adam learning rate
VALIDATION_SPLIT = 0.2             # 20 % of data reserved for validation
MODEL_SAVE_PATH  = "isl_model.h5"            # Output model file
LABELS_SAVE_PATH = "class_labels.json"       # Output label-map file


def build_data_generators():
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,  # Scale to [-1, 1]
        rotation_range=20,          # Random rotation ±20°
        width_shift_range=0.2,      # Horizontal shift ±20 %
        height_shift_range=0.2,     # Vertical shift ±20 %
        zoom_range=0.2,             # Random zoom ±20 %
        horizontal_flip=True,       # Mirror images randomly
        brightness_range=[0.8, 1.2],# Random brightness adjustment
        fill_mode="nearest",        # Fill new pixels after transform
        validation_split=VALIDATION_SPLIT,
    )

    train_generator = train_datagen.flow_from_directory(
        DATASET_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="training",
        shuffle=True,
    )

    val_generator = train_datagen.flow_from_directory(
        DATASET_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation",
        shuffle=False,
    )

    return train_generator, val_generator


def build_model(num_classes: int) -> Model:
    base_model = MobileNetV2(
        weights="imagenet",          # Use ImageNet pre-trained weights
        include_top=False,           # Remove default 1000-class head
        input_shape=(224, 224, 3),   # RGB images at 224 × 224
    )

    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)          # Collapse spatial dims
    x = Dense(256, activation="relu")(x)     # Fully-connected layer
    x = Dropout(0.5)(x)                      # Regularisation
    x = Dense(128, activation="relu")(x)     # Second FC layer
    x = Dropout(0.3)(x)                      # Lighter dropout
    predictions = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()
    return model


def train():
    if not os.path.isdir(DATASET_DIR):
        raise FileNotFoundError(
            f"Dataset directory '{DATASET_DIR}' not found.\n"
            "Create it with one sub-folder per class. See docstring."
        )

    train_gen, val_gen = build_data_generators()
    num_classes = train_gen.num_classes
    class_indices = train_gen.class_indices       

    print(f"\n✅  Found {train_gen.samples} training samples")
    print(f"✅  Found {val_gen.samples} validation samples")
    print(f"✅  Number of classes: {num_classes}")
    print(f"✅  Classes: {list(class_indices.keys())}\n")

    model = build_model(num_classes)

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ModelCheckpoint(MODEL_SAVE_PATH, monitor="val_accuracy", save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7, verbose=1),
    ]

    print("\n🚀 LAUNCHING 15-SECOND DUMMY TRAINING...")
    
    # 🛑 PROTOTYPE HACK: steps_per_epoch forces it to finish instantly 🛑
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        steps_per_epoch=2,       # Stop training after 2 batches
        validation_data=val_gen,
        validation_steps=1,      # Stop validation after 1 batch
        callbacks=callbacks,
        verbose=1,
    )

    labels = {str(v): k for k, v in class_indices.items()}
    with open(LABELS_SAVE_PATH, "w") as f:
        json.dump(labels, f, indent=2)

    print(f"\n🎉  Model saved  → {MODEL_SAVE_PATH}")
    print(f"🎉  Labels saved → {LABELS_SAVE_PATH}")

    return history

if __name__ == "__main__":
    train()