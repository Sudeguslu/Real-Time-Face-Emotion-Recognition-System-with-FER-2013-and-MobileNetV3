# Real-Time Full-Stack Emotion Recognition System

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-00B294?logo=google&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![Next.js](https://img.shields.io/badge/Next.js-000000?logo=next.js&logoColor=white)

A high-performance, full-stack web application designed to detect and classify 7 basic human emotions in real-time. Built with a decoupled architecture, it leverages a highly concurrent FastAPI WebSocket backend for AI inference and a Next.js frontend for a seamless, minimalist user experience.

> **Objective:** To achieve low-latency, real-time facial expression analysis through a robust web architecture, mitigating flickering and class imbalance issues prevalent in standard emotion recognition models.

---

## ✨ Key Features

- **🏗️ Modern Decoupled Architecture:** Clean separation of concerns between the AI inference engine (Backend) and the client interface (Frontend).
- **⚡ Ultra-Fast Face Detection:** Replaced traditional Haar Cascades with the **MediaPipe Tasks API (BlazeFace)** for millisecond-level, CPU-friendly facial localization.
- **🔄 Temporal Smoothing (Hysteresis):** A custom-built `EmotionSmoother` algorithm acts as a low-pass filter over the last N frames, eliminating UI flickering and providing highly stable emotion outputs.
- **📡 Real-Time WebSockets:** Zero-latency bidirectional data streaming between the Next.js client and FastAPI server, completely bypassing the overhead of standard HTTP requests.
- **⚖️ Balanced Learning:** Tackled the inherent class imbalance of the FER-2013 dataset (e.g., scarce "Disgust" samples) by implementing advanced *Class Weighting* during the PyTorch training phase.

---

## 🛠️ Technology Stack

| Category | Technology | Description |
| :--- | :--- | :--- |
| **Artificial Intelligence** | PyTorch, MobileNetV3 | The core deep learning model for 7-class emotion classification. |
| **Computer Vision** | MediaPipe, OpenCV | Real-time image preprocessing and facial region-of-interest (ROI) cropping. |
| **Backend API** | Python, FastAPI, WebSockets | Asynchronous, high-performance backend handling the AI pipeline. |
| **Frontend** |  Next.js | Client-side application handling the webcam stream and UI rendering. |

---

## 🚀 Installation & Getting Started

Follow these steps to run the project on your local machine.

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git
cd YOUR_REPOSITORY_NAME
```

### 2. Backend Setup

```bash
# Create and activate a virtual environment
python -m venv env

# Windows
.\env\Scripts\activate

# macOS / Linux
source env/bin/activate

# Install all dependencies
pip install -r requirements.txt
```

### 3. Running the Inference Server

```bash
# Start the FastAPI WebSocket server (development mode with hot-reload)
.\env\Scripts\python.exe -m uvicorn main:app --reload
```

> Server will be available at `http://localhost:8000`. WebSocket endpoint: `ws://localhost:8000/ws`

### 4. Running the Live Inference Client

```bash
# Launch the real-time emotion recognition pipeline
python live_inference.py
```

> This opens a local webcam window with live face detection and emotion overlay. Press `q` to quit.

### 5. Frontend Setup

```bash
cd frontend

npm install

npm run dev
```

> Navigate to `http://localhost:3000` to access the UI.

---

## 🗺️ Roadmap

### Phase 1 — Core System ✅

Real-time WebSocket pipeline, MediaPipe face detection, MobileNetV3 classifier, temporal smoothing.

### Phase 2 — Analytics Dashboard 🔄 *(in progress)*

An event- and session-based reporting dashboard is currently in development. Planned features include:

- **Session selector** — filter and replay any recorded inference session
- **Emotion timeline chart** — confidence scores plotted over time per session
- **Distribution breakdown** — bar charts showing the ratio of each of the 7 emotion classes
- **Event log table** — searchable, exportable log of all detected emotion transitions
- **PDF / CSV report export** — generate shareable session reports with a single click
