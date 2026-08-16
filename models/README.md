# Model Directory & Axelera Voyager Compilation Guide

This directory stores deep learning model weights used by the multi-camera detection system on the **Axelera Metis 111C** AIPU.

## Directory Structure

```
models/
├── export_models.py       # Exporter and Axelera Compiler script
├── onnx/                  # Intermediate ONNX files
│   ├── yolov8n.onnx       # Human detection
│   ├── yolov8n-pose.onnx  # Pose estimation
│   └── arcface_mobilefacenet.onnx
└── axm/                   # Compiled Axelera NPU binaries (.axm)
    ├── yolov8n.axm
    ├── yolov8n-pose.axm
    └── arcface_mobilefacenet.axm
```

## Compiling PyTorch / ONNX to Axelera Metis `.axm`

1. **Step 1: Install Voyager SDK**
   Follow Axelera AI official guide to install the Voyager SDK or launch the Voyager Docker container environment.

2. **Step 2: Export PyTorch to ONNX & AXM**
   Run the export script:
   ```bash
   python models/export_models.py --target metis-111c
   ```

3. **Step 3: Manual Compilation via Axelera CLI (Optional)**
   If using custom ONNX models, compile directly with `axcompile`:
   ```bash
   axcompile \
       --input models/onnx/yolov8n.onnx \
       --output models/axm/yolov8n.axm \
       --overwrite
   ```

4. **Step 4: Ultralytics Integration**
   Ultralytics YOLO v8/v11 models can also be directly exported using:
   ```python
   from ultralytics import YOLO
   model = YOLO("yolov8n.pt")
   model.export(format="axelera")  # Generates .axm model
   ```
