"""
YOLODetector: Human / Object Detection Module using Axelera Metis or Ultralytics.
"""

import os
import cv2
import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Union
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
            onnx_path=config.get("onnx_path"),
            chip_id=config.get("chip_id", 0),
            num_cores=config.get("num_cores", 4)
        )
        
        # High-level official Ultralytics YOLO inference engine (loads ONNX / PyTorch format)
        self.ultralytics_model = None
        try:
            from ultralytics import YOLO
            model_candidates = [config.get("onnx_path"), self.model_name]
            for model_src in model_candidates:
                if model_src and (os.path.exists(str(model_src)) or str(model_src).endswith('.pt')):
                    try:
                        print(f"[YOLO DETECTOR] Loading YOLO model '{model_src}' via Ultralytics engine...")
                        self.ultralytics_model = YOLO(model_src)
                        print(f"[YOLO DETECTOR SUCCESS] Active detector engine loaded using '{model_src}'.")
                        break
                    except Exception as e:
                        print(f"[YOLO DETECTOR NOTICE] Candidate '{model_src}' load notice: {e}")
                        continue
        except Exception as e:
            print(f"[YOLO DETECTOR NOTICE] Ultralytics engine load notice: {e}")

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
        if self.engine.get_backend() == "axelera_voyager":
            # Direct Axelera Metis AIPU NPU Execution Pathway
            h_orig, w_orig = frame.shape[:2]
            input_tensor, scale, (pad_x, pad_y) = self.preprocess(frame)
            outputs = self.engine.run(input_tensor)
            detections = self._postprocess(outputs, scale, pad_x, pad_y, w_orig, h_orig)
            return detections

        if self.ultralytics_model is not None:
            # Fallback PyTorch / ONNX inference pathway
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

        h_orig, w_orig = frame.shape[:2]
        input_tensor, scale, (pad_x, pad_y) = self.preprocess(frame)
        outputs = self.engine.run(input_tensor)
        detections = self._postprocess(outputs, scale, pad_x, pad_y, w_orig, h_orig)
        return detections

    def _postprocess(self, outputs: Union[List[np.ndarray], np.ndarray], scale: float, pad_x: int, pad_y: int, w_orig: int, h_orig: int) -> List[Dict[str, Any]]:
        """Parses YOLO raw outputs (supporting float32, int8, uint8, and multi-head NPU shapes) and applies NMS."""
        if outputs is None:
            return []

        if not isinstance(outputs, list):
            output_list = [outputs]
        else:
            output_list = outputs

        output_list = [o for o in output_list if o is not None and o.size > 0]
        if len(output_list) == 0:
            return []

        if not hasattr(self, '_postprocess_logged'):
            self._postprocess_logged = True
            print(f"[YOLO DETECTOR DEBUG] output_list count={len(output_list)}, shapes={[o.shape for o in output_list]}")

        w_target, h_target = self.input_size

        # =====================================================================
        # Branch 1: Axelera Metis Multi-Head FPN Structure (>= 6 outputs)
        # =====================================================================
        if len(output_list) >= 6:
            boxes = []
            confidences = []
            class_ids = []

            # Group outputs by spatial grid dimensions (gh, gw)
            # DFL head has 64 channels (4 coords * 16 DFL bins)
            # Cls head has 80 (or num_classes) channels
            dfl_heads = []
            cls_heads = []

            for raw_t in output_list:
                t = np.array(raw_t)
                while len(t.shape) > 3 and t.shape[0] == 1:
                    t = t[0]
                t = np.squeeze(t)
                if len(t.shape) != 3:
                    continue

                # Ensure layout is (H, W, C)
                if t.shape[0] in [64, 80] and t.shape[0] != t.shape[1]:
                    t = np.transpose(t, (1, 2, 0))

                c_dim = t.shape[-1]
                if c_dim == 64:
                    dfl_heads.append(t)
                elif c_dim in [80, 84, 85] or (c_dim >= 1 and c_dim != 64 and c_dim != 51):
                    cls_heads.append(t)

            # Sort heads by spatial resolution descending (stride 8: 64x64, stride 16: 32x32, stride 32: 16x16)
            dfl_heads.sort(key=lambda x: x.shape[0] * x.shape[1], reverse=True)
            cls_heads.sort(key=lambda x: x.shape[0] * x.shape[1], reverse=True)

            num_scales = min(len(dfl_heads), len(cls_heads))
            for i in range(num_scales):
                dfl = dfl_heads[i].astype(np.float32)
                cls = cls_heads[i].astype(np.float32)

                gh, gw = dfl.shape[0], dfl.shape[1]
                if cls.shape[0] != gh or cls.shape[1] != gw:
                    continue

                stride = float(w_target) / float(gw) if gw > 0 else 8.0

                # Check if quantized
                if dfl_heads[i].dtype in [np.int8, np.int16]:
                    dfl = dfl / 12.8
                if cls_heads[i].dtype in [np.int8, np.int16]:
                    cls = cls / 12.8

                cls_min = float(np.min(cls))
                cls_max = float(np.max(cls))

                # If values are already in [0, 1] range, DO NOT apply sigmoid!
                # Applying sigmoid to 0.0 gives 0.50, which causes false detections everywhere!
                if cls_min >= 0.0 and cls_max <= 1.05:
                    cls_prob = cls
                else:
                    cls_prob = 1.0 / (1.0 + np.exp(-np.clip(cls, -20.0, 20.0)))

                # 2. Decode DFL box predictions
                dfl_reshaped = dfl.reshape(gh, gw, 4, 16)
                dfl_softmax = np.exp(dfl_reshaped - np.max(dfl_reshaped, axis=-1, keepdims=True))
                dfl_softmax = dfl_softmax / np.sum(dfl_softmax, axis=-1, keepdims=True)
                dfl_val = np.sum(dfl_softmax * np.arange(16, dtype=np.float32), axis=-1)  # shape (gh, gw, 4)

                # Softmax confidence sanity check: flat uniform noise gives max prob = 1/16 = 0.0625
                dfl_max_prob = np.max(dfl_softmax, axis=-1)  # shape (gh, gw, 4)
                dfl_is_peaked = np.min(dfl_max_prob, axis=-1) > 0.10  # Must have a real boundary peak

                num_classes = cls_prob.shape[-1]
                target_cls = min(self.person_class_id, num_classes - 1)

                for r in range(gh):
                    for c in range(gw):
                        # Multi-class checking
                        if num_classes > 1:
                            best_class = int(np.argmax(cls_prob[r, c, :]))
                            person_score = float(cls_prob[r, c, target_cls])
                            # If all classes are identical (flat zero activations), skip
                            if np.all(cls[r, c, :] == cls[r, c, 0]):
                                continue
                            if best_class != target_cls:
                                if person_score < self.conf_thresh or person_score < float(cls_prob[r, c, best_class]) * 0.8:
                                    continue
                            score = max(person_score, float(cls_prob[r, c, best_class])) if best_class == target_cls else person_score
                        else:
                            score = float(cls_prob[r, c, 0])

                        # Threshold check
                        if score < self.conf_thresh:
                            continue

                        # DFL distribution sanity check (must not be flat uniform noise)
                        if not dfl_is_peaked[r, c]:
                            continue

                        l_d, t_d, r_d, b_d = dfl_val[r, c, 0], dfl_val[r, c, 1], dfl_val[r, c, 2], dfl_val[r, c, 3]

                        # Skip degenerate box predictions (e.g. 7.5 on all sides)
                        if abs(l_d - r_d) < 0.05 and abs(t_d - b_d) < 0.05 and abs(l_d - 7.5) < 0.25:
                            continue

                        cx = (c + 0.5 + (r_d - l_d) / 2.0) * stride
                        cy = (r + 0.5 + (b_d - t_d) / 2.0) * stride
                        w = (l_d + r_d) * stride
                        h = (t_d + b_d) * stride

                        # Filter out invalid box dimensions
                        if w < 12 or h < 16 or w > w_target * 1.5 or h > h_target * 1.5:
                            continue

                        x1 = (cx - w / 2.0 - pad_x) / scale
                        y1 = (cy - h / 2.0 - pad_y) / scale
                        x2 = (cx + w / 2.0 - pad_x) / scale
                        y2 = (cy + h / 2.0 - pad_y) / scale

                        x1 = max(0.0, min(float(w_orig), x1))
                        y1 = max(0.0, min(float(h_orig), y1))
                        x2 = max(0.0, min(float(w_orig), x2))
                        y2 = max(0.0, min(float(h_orig), y2))

                        if (x2 - x1) < 10 or (y2 - y1) < 15:
                            continue

                        boxes.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])
                        confidences.append(float(score))
                        class_ids.append(0)

            if len(boxes) > 0:
                indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_thresh, self.iou_thresh)
                results = []
                if len(indices) > 0:
                    for i in indices.flatten():
                        x, y, w, h = boxes[i]
                        results.append({
                            "bbox": [float(x), float(y), float(x + w), float(y + h)],
                            "confidence": float(confidences[i]),
                            "class_id": int(class_ids[i]),
                            "label": "Person"
                        })
                return results
            return []

        # =====================================================================
        # Branch 2: Standard Concatenated YOLO Output (1 tensor, e.g. [1, 84, 8400])
        # =====================================================================
        target_tensor = None
        for o in output_list:
            s = np.squeeze(o).shape
            if len(s) >= 2 and (84 in s or 85 in s or 80 in s):
                target_tensor = o
                break
        if target_tensor is None:
            target_tensor = output_list[0]

        output = np.array(target_tensor, dtype=np.float32)
        while len(output.shape) > 2 and output.shape[0] == 1:
            output = output[0]
        output = np.squeeze(output)

        if len(output.shape) == 3:
            s0, s1, s2 = output.shape
            if s0 in [84, 85, 80] or s0 < min(s1, s2):
                output = output.reshape(s0, -1).T
            else:
                output = output.reshape(-1, s2)

        if len(output.shape) != 2:
            return []

        d0, d1 = output.shape
        if d0 in [84, 85, 80] or (d0 < d1 and d0 < 100):
            output = output.T

        boxes = []
        confidences = []
        class_ids = []

        for row in output:
            if len(row) < 5:
                continue
            scores = row[4:]
            if len(scores) == 0:
                continue

            # Class score determination: do NOT apply sigmoid if already in [0, 1]!
            cls_id = int(np.argmax(scores))
            raw_max = float(scores[cls_id])

            if 0.0 <= raw_max <= 1.05:
                max_score = raw_max
            else:
                max_score = 1.0 / (1.0 + np.exp(-np.clip(raw_max, -20.0, 20.0)))

            if (cls_id == self.person_class_id or len(scores) == 1) and max_score >= self.conf_thresh:
                # Check for uninitialized / flat zero rows
                if np.all(scores == scores[0]):
                    continue

                cx, cy, w, h = row[0:4]

                # Coordinates: only scale if normalized in [0, 1]
                if max(abs(cx), abs(cy), abs(w), abs(h)) <= 2.0:
                    cx *= w_target
                    cy *= h_target
                    w *= w_target
                    h *= h_target

                x1 = (cx - w / 2 - pad_x) / scale
                y1 = (cy - h / 2 - pad_y) / scale
                x2 = (cx + w / 2 - pad_x) / scale
                y2 = (cy + h / 2 - pad_y) / scale

                x1 = max(0.0, min(float(w_orig), x1))
                y1 = max(0.0, min(float(h_orig), y1))
                x2 = max(0.0, min(float(w_orig), x2))
                y2 = max(0.0, min(float(h_orig), y2))

                if (x2 - x1) < 10 or (y2 - y1) < 15:
                    continue

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
                    "confidence": float(confidences[i]),
                    "class_id": int(class_ids[i]),
                    "label": "Person"
                })
        return results
