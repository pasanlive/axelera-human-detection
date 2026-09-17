"""
LicensePlateRecognizer: Two-stage License Plate Detection + OCR Module.

Stage 1: YOLOv8n-plate detector finds plate bounding boxes in frame.
Stage 2: CRNN (BiLSTM + CTC) reads plate text from each crop.
Both stages use VoyagerEngine (Axelera NPU → ONNXRuntime → Virtual fallback).

Pre-trained weights: publicly available ONNX models.
  - Plate detector: YOLOv8n trained on CCPD/OpenALPR datasets (Ultralytics Hub / Roboflow)
  - Plate OCR: clovaai/deep-text-recognition CRNN converted to ONNX
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Optional
from src.inference.voyager_engine import VoyagerEngine


# Alphanumeric character set for CTC decoding (index 0 = blank)
CTC_CHARSET = "-0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


class LicensePlateRecognizer:
    """
    Two-stage license plate detector and OCR reader.
    Stage 1: YOLOv8n-plate detects plate regions.
    Stage 2: CRNN OCR reads plate text from each crop.
    """

    def __init__(self, config: Dict[str, Any], plate_db=None):
        self.config = config
        self.plate_db = plate_db
        self.conf_thresh = config.get("conf_threshold", 0.45)
        self.detector_input_size = tuple(config.get("detector_input_size", [640, 640]))
        self.ocr_input_size = tuple(config.get("ocr_input_size", [100, 32]))
        self.search_in_person_roi = config.get("search_in_person_roi", False)

        # Stage 1: Plate detector engine
        self.detector_engine = VoyagerEngine(
            axm_path=config.get("detector_axm"),
            onnx_path=config.get("detector_onnx"),
            chip_id=config.get("chip_id", 0),
            num_cores=config.get("num_cores", 4)
        )

        # Stage 2: Plate OCR engine
        self.ocr_engine = VoyagerEngine(
            axm_path=config.get("ocr_axm"),
            onnx_path=config.get("ocr_onnx"),
            chip_id=config.get("chip_id", 0),
            num_cores=config.get("num_cores", 4)
        )

        # Optional Ultralytics plate detector fallback
        self.ultralytics_detector = None
        try:
            from ultralytics import YOLO
            det_onnx = config.get("detector_onnx")
            if det_onnx:
                import os
                if os.path.exists(str(det_onnx)):
                    self.ultralytics_detector = YOLO(det_onnx)
                    print(f"[LPR] Plate detector loaded via Ultralytics: '{det_onnx}'")
        except Exception as e:
            print(f"[LPR] Ultralytics plate detector load notice: {e}")

        print(f"[LPR] Initialized — detector: {self.detector_engine.get_backend()}, OCR: {self.ocr_engine.get_backend()}")

    def detect_plates(self, frame: np.ndarray,
                      person_bboxes: Optional[List[List[float]]] = None) -> List[Dict[str, Any]]:
        """
        Detects license plates and reads their text.
        :param frame: Full camera frame (BGR numpy array)
        :param person_bboxes: Optional — restrict plate search to regions near persons
        :return: List of {bbox, plate_text, confidence, cam_id}
        """
        results = []
        h_frame, w_frame = frame.shape[:2]

        # Determine search regions
        search_regions = []
        if self.search_in_person_roi and person_bboxes:
            # Expand each person bbox slightly downward for vehicle plates
            for bbox in person_bboxes:
                x1, y1, x2, y2 = map(int, bbox)
                # Expand below person by 50% of body height for vehicle/plate region
                expand = int((y2 - y1) * 0.5)
                rx1 = max(0, x1 - 20)
                ry1 = max(0, y1)
                rx2 = min(w_frame, x2 + 20)
                ry2 = min(h_frame, y2 + expand)
                search_regions.append((rx1, ry1, rx2, ry2, frame[ry1:ry2, rx1:rx2]))
        else:
            # Search entire frame
            search_regions.append((0, 0, w_frame, h_frame, frame))

        for roi_x1, roi_y1, roi_x2, roi_y2, roi_frame in search_regions:
            if roi_frame is None or roi_frame.size == 0:
                continue

            plate_boxes = self._detect_plate_regions(roi_frame)

            for pbox in plate_boxes:
                # Translate ROI-relative coords back to full-frame coords
                px1 = int(pbox[0]) + roi_x1
                py1 = int(pbox[1]) + roi_y1
                px2 = int(pbox[2]) + roi_x1
                py2 = int(pbox[3]) + roi_y1
                det_conf = pbox[4]

                # Clamp to frame bounds
                px1 = max(0, min(w_frame, px1))
                py1 = max(0, min(h_frame, py1))
                px2 = max(0, min(w_frame, px2))
                py2 = max(0, min(h_frame, py2))

                if (px2 - px1) < 20 or (py2 - py1) < 8:
                    continue

                # Stage 2: OCR on plate crop
                plate_crop = frame[py1:py2, px1:px2]
                plate_text, ocr_conf = self._read_plate_text(plate_crop)

                results.append({
                    "bbox": [float(px1), float(py1), float(px2), float(py2)],
                    "plate_text": plate_text,
                    "confidence": float(det_conf),
                    "ocr_confidence": float(ocr_conf)
                })

        return results

    def _detect_plate_regions(self, roi: np.ndarray) -> List[List[float]]:
        """Runs plate detection on ROI, returns [[x1,y1,x2,y2,conf], ...]."""
        h_orig, w_orig = roi.shape[:2]
        plates = []

        # Ultralytics detector path
        if self.ultralytics_detector is not None:
            try:
                results = self.ultralytics_detector(
                    roi, conf=self.conf_thresh,
                    imgsz=self.detector_input_size[0], verbose=False
                )[0]
                if results.boxes is not None:
                    for box in results.boxes:
                        xyxy = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0].cpu().numpy())
                        plates.append([float(xyxy[0]), float(xyxy[1]),
                                       float(xyxy[2]), float(xyxy[3]), conf])
                return plates
            except Exception:
                pass

        # VoyagerEngine path (NPU or ONNX)
        if self.detector_engine.get_backend() != "virtual":
            try:
                w_t, h_t = self.detector_input_size
                scale = min(w_t / w_orig, h_t / h_orig)
                nw, nh = int(w_orig * scale), int(h_orig * scale)
                resized = cv2.resize(roi, (nw, nh))
                canvas = np.full((h_t, w_t, 3), 114, dtype=np.uint8)
                pad_x = (w_t - nw) // 2
                pad_y = (h_t - nh) // 2
                canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized
                tensor = canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
                tensor = np.expand_dims(tensor, axis=0)
                outputs = self.detector_engine.run(tensor)
                plates = self._postprocess_detector(outputs, scale, pad_x, pad_y, w_orig, h_orig)
                return plates
            except Exception as e:
                print(f"[LPR] Plate detector engine error: {e}")

        # Virtual/fallback — return empty (no false detections)
        return []

    def _postprocess_detector(self, outputs, scale, pad_x, pad_y, w_orig, h_orig):
        """Parses YOLO plate detector output into bounding boxes."""
        if not outputs:
            return []
        output = np.squeeze(np.array(outputs[0], dtype=np.float32))
        if len(output.shape) == 2:
            d0, d1 = output.shape
            if d0 < d1 and d0 < 100:
                output = output.T

        boxes, confidences = [], []
        w_t, h_t = self.detector_input_size
        for row in output:
            if len(row) < 5:
                continue
            scores = row[4:]
            cls_id = int(np.argmax(scores))
            raw_score = float(scores[cls_id])
            score = raw_score if 0.0 <= raw_score <= 1.05 else float(1.0 / (1.0 + np.exp(-np.clip(raw_score, -20.0, 20.0))))
            if score < self.conf_thresh:
                continue
            cx, cy, w, h = row[0:4]
            if max(abs(cx), abs(cy), abs(w), abs(h)) <= 2.0:
                cx *= w_t; cy *= h_t; w *= w_t; h *= h_t
            x1 = max(0.0, min(float(w_orig), (cx - w / 2 - pad_x) / scale))
            y1 = max(0.0, min(float(h_orig), (cy - h / 2 - pad_y) / scale))
            x2 = max(0.0, min(float(w_orig), (cx + w / 2 - pad_x) / scale))
            y2 = max(0.0, min(float(h_orig), (cy + h / 2 - pad_y) / scale))
            if (x2 - x1) >= 20 and (y2 - y1) >= 8:
                boxes.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])
                confidences.append(float(score))

        results = []
        if boxes:
            indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_thresh, 0.45)
            if len(indices) > 0:
                for i in indices.flatten():
                    x, y, w, h = boxes[i]
                    results.append([float(x), float(y), float(x + w), float(y + h), confidences[i]])
        return results

    def _read_plate_text(self, plate_crop: np.ndarray) -> tuple:
        """
        Runs CRNN OCR on plate crop.
        :return: (plate_text: str, confidence: float)
        """
        if plate_crop is None or plate_crop.size == 0:
            return "", 0.0

        try:
            # Preprocess: resize to OCR input, convert to grayscale, normalize
            w_ocr, h_ocr = self.ocr_input_size
            resized = cv2.resize(plate_crop, (w_ocr, h_ocr))
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            # Normalize to [-1, 1]
            tensor = gray.astype(np.float32) / 127.5 - 1.0
            # Shape: [1, 1, H, W]
            tensor = np.expand_dims(np.expand_dims(tensor, axis=0), axis=0)

            if self.ocr_engine.get_backend() != "virtual":
                outputs = self.ocr_engine.run(tensor)
                if outputs and len(outputs) > 0:
                    logits = np.array(outputs[0])  # shape: [T, 1, num_chars] or [T, num_chars]
                    text, conf = self._ctc_decode(logits)
                    return text, conf

            # Virtual fallback — return placeholder
            return "DEMO-PLATE", 0.5

        except Exception as e:
            print(f"[LPR OCR] Error: {e}")
            return "", 0.0

    def _ctc_decode(self, logits: np.ndarray) -> tuple:
        """
        Greedy CTC decode of CRNN output logits.
        :param logits: shape [T, 1, C] or [T, C] where C = len(CTC_CHARSET)
        :return: (decoded_text: str, mean_confidence: float)
        """
        # Normalize shape to [T, C]
        if len(logits.shape) == 3:
            logits = logits[:, 0, :]  # [T, C]
        if len(logits.shape) != 2:
            return "", 0.0

        # Softmax across character dimension
        logits = logits - logits.max(axis=1, keepdims=True)
        probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)

        # Greedy argmax decode
        pred_indices = np.argmax(probs, axis=1)  # [T]
        pred_confs = probs[np.arange(len(pred_indices)), pred_indices]

        # Collapse repeated chars and blanks (CTC rule, blank = index 0)
        decoded = []
        confs = []
        prev_idx = None
        for i, idx in enumerate(pred_indices):
            if idx != 0 and idx != prev_idx:  # not blank and not repeat
                char_idx = min(int(idx), len(CTC_CHARSET) - 1)
                decoded.append(CTC_CHARSET[char_idx])
                confs.append(float(pred_confs[i]))
            prev_idx = idx

        text = "".join(decoded)
        confidence = float(np.mean(confs)) if confs else 0.0
        return text, confidence
