---
title: ISL Alphabet Recognizer
emoji: 🤟
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---

# ISL Alphabet Recognizer

This application recognizes supported Indian Sign Language alphabet poses from
a guided camera crop. It is a fingerspelling demonstrator, not yet continuous
word- or sentence-level ISL translation.

## Reliability Changes

- Low-confidence or ambiguous frames are rejected instead of written as text.
- A letter is added only after stable recognition across several frames.
- The signer must briefly return to a neutral/no-sign state between letters.
- The model receives the centered guide-box crop rather than the full room.
- Training exports a lightweight ONNX MobileNetV3 model and uses a held-out validation split.

## Collect Webcam Training Samples

A model trained only on `a` through `z` is forced to guess a letter for every
camera frame. Before retraining, collect examples of empty backgrounds,
resting hands, partial hands, movement between letters, and unsupported poses
under `no_sign`. Also collect labeled `A-Z` examples in the same camera,
lighting, clothing, distance, and guide box used during the demo.

In PowerShell:

```powershell
$env:ISL_COLLECTION_MODE="1"
python app.py
```

Open `http://127.0.0.1:7860`, enable the camera, select the label visible in
the guide box, and use `Capture Training Sample`. Samples are stored in the
matching directory under `dataset/`. Collect plenty of `no_sign` samples and
at least 50 varied webcam samples for each alphabet pose you want to demo.

## Prepare And Train

The source images have black backgrounds. Generate camera-like composites and
an explicit `no_sign` class before training:

```powershell
python prepare_dataset.py
```

```powershell
pip install -r requirements-dev.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
python train.py --epochs 8
```

## Evaluate

```powershell
pip install -r requirements-dev.txt
python plot_confusion_matrix.py
```

This prints measured validation accuracy and a classification report, then
writes `confusion_matrix.png`. `plot_quick_cm.py` now invokes the same real
evaluation rather than generating invented results.

## Continuous Translation Roadmap

True ISL translation needs labeled video clips or landmark sequences for words
and phrases, including motion, neutral transitions, multiple signers, and
real webcam conditions. The current image classifier can be a reliable
alphabet module after retraining, but cannot infer a vocabulary it was never
trained to recognize.
