import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

def generate_confusion_matrix():
    if not os.path.exists("isl_model.h5"):
        print("Model file 'isl_model.h5' not found.")
        return
        
    if not os.path.exists("class_labels.json"):
        print("Labels file 'class_labels.json' not found.")
        return
        
    print("Loading model 'isl_model.h5'...")
    model = load_model("isl_model.h5")
    
    print("Loading class labels...")
    with open("class_labels.json", "r") as f:
        labels_dict = json.load(f)
        
    # In train.py: labels = {str(v): k for k, v in class_indices.items()}
    # Dictionary has string integer keys: "0": "A", "1": "B", etc.
    # Convert back to list ordered by index
    num_classes = len(labels_dict)
    class_names = [labels_dict[str(i)] for i in range(num_classes)]
    print(f"Loaded {num_classes} classes.")
    
    print("Setting up validation data generator...")
    val_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        validation_split=0.2
    )

    try:
        val_generator = val_datagen.flow_from_directory(
            "dataset",
            target_size=(224, 224),
            batch_size=32,
            class_mode="categorical",
            subset="validation",
            shuffle=False
        )
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    if val_generator.samples == 0:
        print("No validation samples found in 'dataset/'. Cannot generate matrix.")
        return

    print("Running predictions on validation data...")
    y_pred_probs = model.predict(val_generator)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = val_generator.classes
    
    print("Computing confusion matrix...")
    cm = confusion_matrix(y_true, y_pred)
    
    # Plotting
    plt.figure(figsize=(14, 12))
    
    # If 26 classes, annot=True might be too cluttered, but we can try if numbers aren't huge.
    sns.heatmap(
        cm, 
        annot=True, 
        fmt="d", 
        cmap="Blues",
        xticklabels=class_names, 
        yticklabels=class_names
    )
    
    plt.title("Confusion Matrix (MobileNetV2 ISL Model)")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    
    # Rotate x labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=300)
    print("✅ Success! Confusion matrix saved to 'confusion_matrix.png'")

if __name__ == "__main__":
    generate_confusion_matrix()
