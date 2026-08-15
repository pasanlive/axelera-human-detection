"""
YOLODetector: Human / Object Detection Module using Axelera Metis or Ultralytics.
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from src.inference.voyager_engine import VoyagerEngine

class YOLODetector:
    """Human (Person) Detector powered by YOLO and Voyager SDK / Ultralytics."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.conf_thresh = config.get("conf_threshold", 0.45)
        self.iou_thresh = config.get("iou_threshold", 0.45)
        self.input_size = tuple(config.get("input_size", [640, 640]))
        self.person_class_id = config.get("person_class_id", 0)
        self.model_name = config.get("model_name", "yolov8n.pt")

        self.engine = VoyagerEngine(
            axm_path=config.get("axm_path"),
            onnx_path=config.get("onnx_path")
        )
        
        # High-level Ultralytics PyTorch fallback if available and engine is virtual
        self.ultralytics_model = None
        if self.engine.get_backend() == "virtual":
            try:
                from ultralytics import YOLO
                print(f"[YOLO DETECTOR] Loading PyTorch model '{self.model_name}' via Ultralytics engine...")
                self.ultralytics_model = YOLO(self.model_name)
            except Exception as e:
                print(f"[YOLO DETECTOR NOTICE] Ultralytics PyTorch load: {e}")

    def preprocess(self, frame: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """Letterbox resize image to target input size."""
        h_orig, w_orig = frame.shape[:2]
        w_target, h_target = self.input_size

        scale = min(w_target / w_orig, h_target / h_orig)
        nw, nh = int(w_orig * scale), int(h_orig * scale)

        resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((h_target, w_target, 3), 114, dtype=np.uint8)

        pad_x = (w_target - nw) // 2
        pad_y = (h_target - nh) // 2
        canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized

        # Convert BGR -> RGB, normalize [0, 1], transpose [H,W,C] -> [1,C,H,W]
        input_tensor = canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        input_tensor = np.expand_dims(input_tensor, axis=0)

        return input_tensor, scale, (pad_x, pad_y)

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Runs human detection on input image frame.
        :return: List of dicts containing bbox [x1, y1, x2, y2], confidence, class_id, label
        """
        if self.ultralytics_model is not None:
            # Ultralytics native inference
            results = self.ultralytics_model(frame, conf=self.conf_thresh, verbose=False)[0]
            detections = []
            if results.boxes is not None:
                for box in results.boxes:
                    cls_id = int(box.cls[0].cpu().numpy())
                    conf = float(box.conf[0].cpu().numpy())
                    if cls_id == self.person_class_id:
                        xyxy = box.xyxy[0].cpu().numpy()
                        detections.append({
                            "bbox": [float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])],
                            "confidence": conf,
                            "class_id": cls_id,
                            "label": "Person"
                        })
            return detections

        # Voyager / ONNX tensor pathway
        h_orig, w_orig = frame.shape[:2]
        input_tensor, scale, (pad_x, pad_y) = self.preprocess(frame)
        outputs = self.engine.run(input_tensor)

        # Parse YOLO raw output tensors [1, 84, 8400] -> [boxes, scores]
        detections = self._postprocess(outputs[0], scale, pad_x, pad_y, w_orig, h_orig)
        return detections

    def _postprocess(self, output: np.ndarray, scale: float, pad_x: int, pad_y: int, w_orig: int, h_orig: int) -> List[Dict[str, Any]]:
        """Parses YOLO raw outputs and applies Non-Maximum Suppression (NMS)."""
        if len(output.shape) == 3:
            output = output[0]  # [84, 8400]
        
        # Transpose if output is [84, 8400] -> [8400, 84]
        if output.shape[0] < output.shape[1]:
            output = output.T

        boxes = []
        confidences = []
        class_ids = []

        for row in output:
            scores = row[4:]
            cls_id = np.argmax(scores)
            max_score = scores[cls_id]

            if cls_id == self.person_class_id and max_score >= self.conf_thresh:
                cx, cy, w, h = row[0:4]
                # Convert from padded canvas space back to original image space
                x1 = (cx - w / 2 - pad_x) / scale
                y1 = (cy - h / 2 - pad_y) / scale
                x2 = (cx + w / 2 - pad_x) / scale
                y2 = (cy + h / 2 - pad_y) / scale

                # Clip boundaries
                x1 = max(0.0, min(w_orig, x1))
                y1 = max(0.0, min(h_orig, y1))
                x2 = max(0.0, min(w_orig, x2))
                y2 = max(0.0, min(h_orig, y2))

                boxes.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])
                confidences.append(float(max_score))
                class_ids.append(int(cls_id))

        if len(boxes) == 0:
            return []

        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_thresh, self.iou_thresh)
        results = []
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                results.append({
                    "bbox": [float(x), float(y), float(x + w), float(y + h)],
                    "confidence": confidences[i],
                    "class_id": class_ids[i],
                    "label": "Person"
                })
        return results
