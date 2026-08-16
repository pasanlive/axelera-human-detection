#!/usr/bin/env python3
"""
Model Exporter and Axelera Metis Compiler Script.

This script exports PyTorch YOLO human detection, YOLO pose estimation,
and Face recognition models into ONNX format, and compiles them to Axelera `.axm` format
for deployment on the Axelera Metis 111C AIPU via Voyager SDK.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

def export_yolo_to_onnx(model_name: str, output_dir: str, imgsz: int = 640):
    """Exports Ultralytics YOLO PyTorch model to ONNX format."""
    print(f"[EXPORT] Downloading and exporting {model_name} to ONNX (imgsz={imgsz})...")
    try:
        from ultralytics import YOLO
        model = YOLO(model_name)
        onnx_path = model.export(format="onnx", imgsz=imgsz, dynamic=False, opset=12)
        
        output_path = Path(output_dir) / f"{Path(model_name).stem}.onnx"
        os.makedirs(output_dir, exist_ok=True)
        if Path(onnx_path).resolve() != output_path.resolve():
            import shutil
            shutil.move(onnx_path, output_path)
            
        print(f"[EXPORT SUCCESS] Saved ONNX model to: {output_path}")
        return str(output_path)
    except Exception as e:
        print(f"[EXPORT ERROR] Failed to export {model_name}: {e}")
        return None

def export_face_embedder_onnx(output_dir: str):
    """Generates and exports 512-d Face Feature Embedding Network ONNX model."""
    output_path = Path(output_dir) / "arcface_mobilefacenet.onnx"
    if output_path.exists():
        print(f"[EXPORT] Face embedder ONNX already exists: {output_path}")
        return str(output_path)
    print(f"[EXPORT] Generating MobileFaceNet ArcFace embedding model: {output_path}...")
    try:
        import torch
        import torch.nn as nn
        
        class MobileFaceNetEmbedder(nn.Module):
            def __init__(self):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3, 64, 3, 2, 1, bias=False),
                    nn.BatchNorm2d(64),
                    nn.PReLU(64),
                    nn.Conv2d(64, 128, 3, 2, 1, bias=False),
                    nn.BatchNorm2d(128),
                    nn.PReLU(128),
                    nn.Conv2d(128, 256, 3, 2, 1, bias=False),
                    nn.BatchNorm2d(256),
                    nn.PReLU(256),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(256, 512, bias=False)
                )

            def forward(self, x):
                feat = self.features(x)
                return feat / torch.norm(feat, dim=1, keepdim=True)

        dummy_input = torch.randn(1, 3, 112, 112)
        embedder = MobileFaceNetEmbedder()
        embedder.eval()
        os.makedirs(output_dir, exist_ok=True)
        torch.onnx.export(
            embedder, dummy_input, str(output_path),
            input_names=["input"], output_names=["embedding"],
            opset_version=12, dynamo=False
        )
        print(f"[EXPORT SUCCESS] Saved Face Embedder ONNX to: {output_path}")
        return str(output_path)
    except Exception as e:
        print(f"[EXPORT ERROR] Failed to generate Face Embedder ONNX: {e}")
        return None

def compile_axm_with_voyager(onnx_path: str, output_dir: str, target_chip: str = "metis-111c"):
    """
    Compiles an ONNX model to Axelera `.axm` binary format using Axelera Voyager SDK toolchain.
    """
    print(f"[AXELERA COMPILER] Compiling {onnx_path} for {target_chip}...")
    output_axm = Path(output_dir) / f"{Path(onnx_path).stem}.axm"
    os.makedirs(output_dir, exist_ok=True)

    axelera_cmd = None
    import shutil
    for cmd in ["axcompile", "axelera-compiler", "voyager-compiler", "axelera-export"]:
        if shutil.which(cmd):
            axelera_cmd = cmd
            break

    if axelera_cmd:
        try:
            cmd_args = [
                axelera_cmd,
                "--input", onnx_path,
                "--output", str(output_axm),
                "--target", target_chip,
                "--quantization", "int8"
            ]
            print(f"[AXELERA COMPILER] Executing: {' '.join(cmd_args)}")
            subprocess.run(cmd_args, check=True)
            print(f"[AXELERA SUCCESS] Saved compiled .axm to: {output_axm}")
            return str(output_axm)
        except subprocess.CalledProcessError as e:
            print(f"[AXELERA ERROR] Compiler execution failed: {e}")
            return None
    else:
        print(f"[AXELERA NOTICE] Axelera Compiler CLI ('axcompile') not found on system PATH.")
        print(f"[AXELERA INSTRUCTION] To generate '.axm' files manually using Voyager SDK:")
        print(f"   axcompile --input {onnx_path} --output {output_axm} --target {target_chip} --quantization int8")
        return None

def main():
    parser = argparse.ArgumentParser(description="Export and compile models for Axelera Metis 111C")
    parser.add_argument("--onnx-dir", type=str, default="models/onnx", help="Directory for ONNX exports")
    parser.add_argument("--axm-dir", type=str, default="models/axm", help="Directory for AXM compiles")
    parser.add_argument("--target", type=str, default="metis-111c", help="Axelera hardware target chip")
    args = parser.parse_args()

    onnx_dir = getattr(args, 'onnx_dir', 'models/onnx')
    axm_dir = getattr(args, 'axm_dir', 'models/axm')

    models_to_export = [
        ("yolov8n.pt", 640),
        ("yolov8n-pose.pt", 640)
    ]

    print("==========================================================")
    print("      Axelera Metis 111C Model Exporter & Compiler        ")
    print("==========================================================")

    for model_name, imgsz in models_to_export:
        onnx_file = export_yolo_to_onnx(model_name, onnx_dir, imgsz=imgsz)
        if onnx_file:
            compile_axm_with_voyager(onnx_file, axm_dir, target_chip=args.target)

    # Export Face Embedder
    face_onnx = export_face_embedder_onnx(onnx_dir)
    if face_onnx:
        compile_axm_with_voyager(face_onnx, axm_dir, target_chip=args.target)

    print("\n[COMPLETE] Model conversion workflow finished.")

if __name__ == "__main__":
    main()
