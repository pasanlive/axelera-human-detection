# Multi-Camera Human Detection, Pose Estimation & Face Recognition System for Axelera Metis 111C

This repository contains a high-performance computer vision system designed for the **Axelera Metis 111C** AI computer board using the **Axelera Voyager SDK** and **YOLO models**.

The system enables simultaneous processing of multiple camera feeds (RTSP streams, webcams, or video files) with multi-modal AI capabilities:
- 🚶 **Human Detection** (YOLOv8 / YOLOv11 person class detection)
- 🧘 **Pose Estimation** (17 skeletal body keypoints)
- 👤 **Face Recognition** (Face detection + ArcFace 512-d feature vector database matching)
- 🎯 **Multi-Object Tracking** (ByteTrack persistent ID tracking per camera feed)
- 🖥️ **Multi-Stream Display** (Tiled camera grid dashboard with real-time FPS overlay)

---

## 🛠️ System Architecture

```
                          ┌───────────────────────────┐
                          │ Multi-Camera RTSP / USB   │
                          └─────────────┬─────────────┘
                                        │
                                        ▼
                          ┌───────────────────────────┐
                          │ Stream Manager Threads    │
                          └─────────────┬─────────────┘
                                        │
                                        ▼
       ┌─────────────────────────────────────────────────────────────────┐
       │                 Axelera Metis 111C AIPU Accelerator             │
       │                                                                 │
       │  ┌───────────────────────┐  ┌────────────────────────────────┐  │
       │  │ Human Detection (.axm)│  │ Pose Estimation 17-kpt (.axm) │  │
       │  └───────────┬───────────┘  └───────────────┬────────────────┘  │
       └──────────────┼──────────────────────────────┼───────────────────┘
                      │                              │
                      ▼                              ▼
       ┌─────────────────────────────────────────────────────────────────┐
       │                Fusion & Multi-Object Tracking                   │
       │                                                                 │
       │  ┌───────────────────────┐  ┌────────────────────────────────┐  │
       │  │ ByteTrack ID Matcher  │  │ Face Identity Vector Gallery   │  │
       │  └───────────────────────┘  └────────────────────────────────┘  │
       └────────────────────────────────┬────────────────────────────────┘
                                        │
                                        ▼
                          ┌───────────────────────────┐
                          │ Real-Time Tiled Dashboard │
                          └───────────────────────────┘
```

---

## 📋 Prerequisites & Installation

### 1. Requirements
- Linux / macOS host system connected to Axelera Metis 111C card (or CPU fallback mode)
- Python 3.9+
- Axelera Voyager SDK installed (or Voyager Docker container)

### 2. Setup Python Environment
```bash
# Clone repository
git clone https://github.com/your-username/axelera-human-detection.git
cd axelera-human-detection

# Install Python dependencies
pip install -r requirements.txt
```

---

## ⚙️ Model Export & Compilation (.axm)

Models must be compiled to Axelera's `.axm` hardware format using the `axelera-compiler` tool provided in the Voyager SDK.

Run the automatic exporter script:
```bash
python models/export_models.py --target metis-111c
```

Or manually compile using the Axelera Voyager CLI:
```bash
axelera-compiler \
    --input models/onnx/yolov8n.onnx \
    --output models/axm/yolov8n.axm \
    --target metis-111c \
    --quantization int8
```

---

## 🚀 Usage Guide

### 1. Configuration
Configure camera streams, model weights, and thresholds in `config/config.yaml`:

```yaml
cameras:
  - id: "cam_01"
    name: "Front Entrance"
    source: "rtsp://admin:password@192.168.1.100:554/stream1"
    enabled: true
  - id: "cam_02"
    name: "Lobby"
    source: "0"  # USB webcam index
    enabled: true
```

### 2. Enroll New Face Identity
To enroll a face profile into the identity gallery database:
```bash
python app.py --enroll "John Doe" --image path/to/john_photo.jpg
```

To list all enrolled identities:
```bash
python app.py --list-faces
```

### 3. Run Real-Time Multi-Camera System
```bash
# Interactive GUI mode
python app.py

# Headless mode (for servers / background deployments)
python app.py --headless
```

---

## 🧪 Verification & Unit Tests

Run unit tests to verify tensor preprocessing, bounding box decoding, face matching, and pose estimation logic:
```bash
python -m unittest discover tests
```

---

## 📄 License
MIT License. Built for Axelera Metis AIPU platform & Voyager SDK.
