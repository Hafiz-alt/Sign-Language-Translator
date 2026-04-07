<p align="center">
  <img src="https://img.shields.io/badge/🤟-ISL_Translator-blueviolet?style=for-the-badge&logoColor=white" alt="ISL Translator" />
</p>

<h1 align="center">🤖 Indian Sign Language Translator</h1>

<p align="center">
  <b>Real-time AI-powered Indian Sign Language → Text & Speech translation</b><br/>
  <sub>Powered by MobileNetV2 deep learning, Flask, and a scroll-linked Apple-style UI</sub>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/TensorFlow-2.10+-FF6F00?style=flat-square&logo=tensorflow&logoColor=white" />
  <img src="https://img.shields.io/badge/Flask-2.3+-000000?style=flat-square&logo=flask&logoColor=white" />
  <img src="https://img.shields.io/badge/OpenCV-4.7+-5C3EE8?style=flat-square&logo=opencv&logoColor=white" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/A--Z_Alphabet-26_Classes-00e5ff?style=flat-square" />
  <img src="https://img.shields.io/badge/Inference-Real_Time-a855f7?style=flat-square" />
  <img src="https://img.shields.io/badge/TTS-Offline_Speech-ec4899?style=flat-square" />
</p>

---

## ✨ What is this?

**ISL Translator** turns your webcam into a real-time Indian Sign Language interpreter. Show a hand sign → the AI recognizes it → you see the letter on screen → and hear it spoken aloud. No cloud APIs, everything runs **locally** on your machine.

<br/>

## 🎯 Key Features

| Feature | Description |
|:---|:---|
| 🔤 **A–Z Recognition** | Recognizes all 26 ISL alphabet signs with live confidence scores |
| 📹 **Live MJPEG Stream** | Smooth, low-latency webcam feed streamed directly to the browser |
| 🗣️ **Offline Text-to-Speech** | Automatic spoken feedback via `pyttsx3` — no internet required |
| 🎨 **Glassmorphism UI** | Frosted-glass dark theme with cyan/purple accents and micro-animations |
| 🎬 **Apple-Style Scroll Animation** | 120-frame scroll-linked canvas background with lerp smoothing |
| ⚡ **Background Inference** | ML runs on a separate thread — video never freezes while predicting |
| 📊 **Live Confidence Bar** | Color-coded confidence indicator (🟢 high · 🟡 mid · 🔴 low) |

<br/>

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        BROWSER (UI)                         │
│                                                             │
│  ┌─────────────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │  Scroll-Linked  │  │  Live MJPEG  │  │   Side Panel  │  │
│  │  Canvas BG      │  │  Video Feed  │  │  • Text       │  │
│  │  (120 frames)   │  │              │  │  • Confidence  │  │
│  └─────────────────┘  └──────┬───────┘  │  • Prediction  │  │
│                              │          └───────┬────────┘  │
│                              │                  │           │
└──────────────────────────────┼──────────────────┼───────────┘
                               │ /video_feed      │ /prediction
                               │ (MJPEG)          │ (JSON poll)
┌──────────────────────────────┼──────────────────┼───────────┐
│                        FLASK SERVER (app.py)                │
│                              │                  │           │
│  ┌───────────────────────────┴──────────────────┴────────┐  │
│  │                  camera.py (ISLCamera)                 │  │
│  │                                                       │  │
│  │   Main Thread          Background Thread              │  │
│  │   ┌──────────┐         ┌──────────────────┐           │  │
│  │   │ Capture  │────────▶│ MobileNetV2      │           │  │
│  │   │ + Draw   │         │ Inference        │           │  │
│  │   │ Overlay  │◀────────│ (every 0.8s)     │           │  │
│  │   └──────────┘ cached  └──────────────────┘           │  │
│  │                results                                │  │
│  └───────────────────────────────────────────────────────┘  │
│                              │                              │
│                  ┌───────────┴───────────┐                  │
│                  │  pyttsx3 TTS Engine   │                  │
│                  │  (daemon thread)      │                  │
│                  └──────────────────────-┘                  │
└─────────────────────────────────────────────────────────────┘
```

<br/>

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|:---|:---|:---|
| **ML Model** | MobileNetV2 (Transfer Learning) | Feature extraction from hand images |
| **Framework** | TensorFlow / Keras | Model training & inference |
| **Computer Vision** | OpenCV | Webcam capture & frame processing |
| **Backend** | Flask | Web server, MJPEG streaming, JSON API |
| **Speech** | pyttsx3 | Offline text-to-speech |
| **Frontend** | HTML5 Canvas + CSS3 + Vanilla JS | Glassmorphism UI & scroll animation |

<br/>

## 📂 Project Structure

```
ISL-Translator/
├── app.py                  # Flask server — routes, MJPEG stream, TTS
├── camera.py               # Webcam + MobileNetV2 inference engine
├── train.py                # Transfer learning training script
├── class_labels.json       # A–Z label mapping (26 classes)
├── isl_model.h5            # Trained model weights (not in repo)
├── requirements.txt        # Python dependencies
│
├── templates/
│   └── index.html          # Main UI — hero, app section, scroll canvas
│
├── static/
│   ├── style.css           # Dark theme + glassmorphism styles
│   └── frames/             # 120 JPEGs for scroll-linked animation
│       ├── ezgif-frame-001.jpg
│       ├── ezgif-frame-002.jpg
│       └── ... → 120.jpg
│
└── dataset/                # Training images (not in repo)
    ├── a/
    ├── b/
    └── ... → z/
```

<br/>

## 🚀 Quick Start

### Prerequisites
- **Python 3.10+**
- A working **webcam**
- The trained model file `isl_model.h5` (place in project root)

### 1. Clone the repo
```bash
git clone https://github.com/Hafiz-alt/Sign-Language-Translator.git
cd Sign-Language-Translator
```

### 2. Create a virtual environment
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the server
```bash
python app.py
```

### 5. Open in your browser
```
🌐  http://127.0.0.1:5000
```

<br/>

## 🧠 Training Your Own Model

If you want to train from scratch:

1. Organize your dataset:
   ```
   dataset/
   ├── a/   (images of sign "A")
   ├── b/   (images of sign "B")
   └── ...
   ```

2. Run the training script:
   ```bash
   python train.py
   ```

3. The script will output `isl_model.h5` and `class_labels.json`.

> **Note:** The default `train.py` is set to prototype mode (1 epoch, 2 steps). For production training, increase `EPOCHS` to 15–25 and remove the `steps_per_epoch` limit.

<br/>

## 🎬 Scroll Animation

The website features an **Apple-style scroll-linked background animation**:

- **120 frames** are preloaded and drawn on a fixed `<canvas>` behind the UI
- As you scroll, the frame index is mapped to your scroll position
- **Lerp smoothing** (`LERP_SPEED = 0.08`) creates buttery-smooth transitions
- **HiDPI-aware** rendering with `devicePixelRatio` scaling
- The glassmorphism cards float on top, creating a stunning depth effect

<br/>

## 🤝 Contributing

Contributions are welcome! Here are some ideas:

- 🖐️ Add word/phrase recognition (beyond single alphabets)
- 📱 Make the UI fully responsive for mobile
- 🎯 Improve model accuracy with a larger dataset
- 🌐 Add support for other sign languages (ASL, BSL)
- 🎙️ Add speech-to-sign reverse translation

<br/>

## 📜 License

This project is open source and available under the [MIT License](LICENSE).

---

<p align="center">
  Built with ❤️ using <b>TensorFlow</b>, <b>MobileNetV2</b> & <b>Flask</b>
</p>

<p align="center">
  <a href="https://github.com/Hafiz-alt/Sign-Language-Translator">
    <img src="https://img.shields.io/badge/⭐_Star_this_repo-if_you_found_it_useful!-yellow?style=for-the-badge" />
  </a>
</p>
