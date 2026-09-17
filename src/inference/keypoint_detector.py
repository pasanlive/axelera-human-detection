"""
KeypointDetector: YOLO11n-Pose Enhanced Keypoint Detection Module.

Model zoo alternative to PoseEstimator (YOLOv8n-Pose).
Mutually exclusive with pose_estimator — enabling this auto-disables pose_estimator.
Outputs same {bbox, confidence, keypoints: np.ndarray (17,3)} format for drop-in
compatibility with the existing Visualizer.
"""

import os
import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Union
from src.inference.voyager_engine import VoyagerEngine

# COCO 17 Keypoint Labels
COCO_KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]


class KeypointDetector:
    """
    YOLO11n-Pose Enhanced Keypoint Detector.
    Model zoo alternative to the default YOLOv8n-Pose estimator.
    Outputs identical format to PoseEstimator for visualizer compatibility.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.conf_thresh = config.get("conf_threshold", 0.50)
        self.iou_thresh = config.get("iou_threshold", 0.45)
        self.input_size = tuple(config.get("input_size", [512, 512]))
        self.num_keypoints = config.get("num_keypoints", 17)
        self.model_name = config.get("model_name", "yolo11n-pose.pt")

        self.engine = VoyagerEngine(
            axm_path=config.get("axm_path"),
            onnx_path=config.get("onnx_path"),
            chip_id=config.get("chip_id", 0),
            num_cores=config.get("num_cores", 4)
        )

        # Ultralytics YOLO11n-Pose fallback
        self.ultralytics_model = None
        try:
            from ultralytics import YOLO
            model_candidates = [config.get("onnx_path"), self.model_name]
            for model_src in model_candidates:
                if model_src and (os.path.exists(str(model_src)) or str(model_src).endswith('.pt')):
                    try:
                        print(f"[KEYPOINT DETECTOR] Loading YOLO11n-Pose model '{model_src}' via Ultralytics...")
                        self.ultralytics_model = YOLO(model_src)
                        print(f"[KEYPOINT DETECTOR SUCCESS] YOLO11n-Pose loaded using '{model_src}'.")
                        break
                    except Exception as e:
                        print(f"[KEYPOINT DETECTOR NOTICE] Candidate '{model_src}': {e}")
                        continue
        except Exception as e:
            print(f"[KEYPOINT DETECTOR NOTICE] Ultralytics load notice: {e}")

    def estimate_pose(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Runs YOLO11n-Pose keypoint estimation across the frame.
        :return: List of dicts with 'bbox', 'confidence', 'keypoints' np.ndarray (17, 3) [x, y, conf]
        Compatible with PoseEstimator output format for drop-in Visualizer use.
        """
        if self.engine.get_backend() == "axelera_voyager":
            h_orig, w_orig = frame.shape[:2]
            input_tensor, scale, (pad_x, pad_y) = self._preprocess(frame)
            outputs = self.engine.run(input_tensor)
            return self._postprocess(outputs, scale, pad_x, pad_y, w_orig, h_orig)

        if self.ultralytics_model is not None:
            results = self.ultralytics_model(frame, conf=self.conf_thresh, verbose=False)[0]
            poses = []
            if results.keypoints is not None and results.boxes is not None:
                boxes = results.boxes.xyxy.cpu().numpy()
                confs = results.boxes.conf.cpu().numpy()
                kpts_data = results.keypoints.data.cpu().numpy()  # [N, 17, 3]
                for i in range(len(boxes)):
                    poses.append({
                        "bbox": boxes[i].tolist(),
                        "confidence": float(confs[i]),
                        "keypoints": kpts_data[i]
                    })
            return poses

        h_orig, w_orig = frame.shape[:2]
        input_tensor, scale, (pad_x, pad_y) = self._preprocess(frame)
        outputs = self.engine.run(input_tensor)
        return self._postprocess(outputs, scale, pad_x, pad_y, w_orig, h_orig)

    def _preprocess(self, frame: np.ndarray):
        """Letterbox resize to target input size."""
        h_orig, w_orig = frame.shape[:2]
        w_target, h_target = self.input_size

        scale = min(w_target / w_orig, h_target / h_orig)
        nw, nh = int(w_orig * scale), int(h_orig * scale)

        resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((h_target, w_target, 3), 114, dtype=np.uint8)

        pad_x = (w_target - nw) // 2
        pad_y = (h_target - nh) // 2
        canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized

        input_tensor = canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        input_tensor = np.expand_dims(input_tensor, axis=0)
        return input_tensor, scale, (pad_x, pad_y)

    def _postprocess(self, outputs: Union[List[np.ndarray], np.ndarray],
                     scale: float, pad_x: int, pad_y: int,
                     w_orig: int, h_orig: int) -> List[Dict[str, Any]]:
        """
        Parses YOLO11n-Pose outputs into bounding boxes + keypoints.
        Supports multi-head NPU structure (9 outputs) and single tensor [1, 56, 8400].
        """
        if outputs is None:
            return []
        if not isinstance(outputs, list):
            output_list = [outputs]
        else:
            output_list = outputs
        output_list = [o for o in output_list if o is not None and o.size > 0]
        if not output_list:
            return []

        if not hasattr(self, '_postprocess_logged'):
            self._postprocess_logged = True
            print(f"[KEYPOINT DETECTOR DEBUG] outputs={len(output_list)}, shapes={[o.shape for o in output_list]}")

        w_target, h_target = self.input_size

        # --- Branch 1: 9-head FPN NPU (DFL 64ch + Score 1ch + Kpt 51ch × 3 scales) ---
        dfl_heads, score_heads, kpt_heads = [], [], []
        for raw_t in output_list:
            t = np.squeeze(np.array(raw_t))
            if len(t.shape) != 3:
                continue
            if t.shape[0] in [64, 1, 51] and t.shape[0] != t.shape[1]:
                t = np.transpose(t, (1, 2, 0))
            c_dim = t.shape[-1]
            if c_dim == 64:
                dfl_heads.append(t)
            elif c_dim == 1:
                score_heads.append(t)
            elif c_dim == 51:
                kpt_heads.append(t)

        dfl_heads.sort(key=lambda x: x.shape[0] * x.shape[1], reverse=True)
        score_heads.sort(key=lambda x: x.shape[0] * x.shape[1], reverse=True)
        kpt_heads.sort(key=lambda x: x.shape[0] * x.shape[1], reverse=True)

        num_scales = min(len(dfl_heads), len(score_heads), len(kpt_heads))
        if num_scales >= 3:
            boxes, confidences, keypoints_list = [], [], []
            for idx in range(num_scales):
                dfl = dfl_heads[idx].astype(np.float32)
                score = score_heads[idx].astype(np.float32)
                kpt = kpt_heads[idx].astype(np.float32)
                gh, gw = dfl.shape[0], dfl.shape[1]
                if score.shape[0] != gh or kpt.shape[0] != gh:
                    continue
                stride = float(w_target) / float(gw) if gw > 0 else 8.0
                if dfl_heads[idx].dtype in [np.int8, np.int16]:
                    dfl /= 12.8
                if score_heads[idx].dtype in [np.int8, np.int16]:
                    score /= 12.8
                if kpt_heads[idx].dtype in [np.int8, np.int16]:
                    kpt /= 12.8
                score_val = score[:, :, 0]
                if score_val.min() >= 0.0 and score_val.max() <= 1.05:
                    score_prob = score_val
                else:
                    score_prob = 1.0 / (1.0 + np.exp(-np.clip(score_val, -20.0, 20.0)))
                dfl_rs = dfl.reshape(gh, gw, 4, 16)
                dfl_sm = np.exp(dfl_rs - np.max(dfl_rs, axis=-1, keepdims=True))
                dfl_sm /= np.sum(dfl_sm, axis=-1, keepdims=True)
                dfl_val = np.sum(dfl_sm * np.arange(16, dtype=np.float32), axis=-1)
                dfl_peaked = np.min(np.max(dfl_sm, axis=-1), axis=-1) > 0.10
                for r in range(gh):
                    for c in range(gw):
                        if float(score_prob[r, c]) < self.conf_thresh:
                            continue
                        if not dfl_peaked[r, c]:
                            continue
                        l_d, t_d, r_d, b_d = dfl_val[r, c]
                        if abs(l_d - r_d) < 0.05 and abs(t_d - b_d) < 0.05:
                            continue
                        cx = (c + 0.5 + (r_d - l_d) / 2.0) * stride
                        cy = (r + 0.5 + (b_d - t_d) / 2.0) * stride
                        w = (l_d + r_d) * stride
                        h = (t_d + b_d) * stride
                        if w < 12 or h < 16 or w > w_target * 1.5:
                            continue
                        x1 = max(0.0, min(float(w_orig), (cx - w / 2.0 - pad_x) / scale))
                        y1 = max(0.0, min(float(h_orig), (cy - h / 2.0 - pad_y) / scale))
                        x2 = max(0.0, min(float(w_orig), (cx + w / 2.0 - pad_x) / scale))
                        y2 = max(0.0, min(float(h_orig), (cy + h / 2.0 - pad_y) / scale))
                        if (x2 - x1) < 10 or (y2 - y1) < 15:
                            continue
                        kpts_raw = kpt[r, c, :51].reshape(17, 3)
                        kpts_scaled = np.zeros((17, 3), dtype=np.float32)
                        for k in range(17):
                            kx_rel, ky_rel, kc_raw = kpts_raw[k]
                            kx = ((c + 0.5 + kx_rel) * stride - pad_x) / scale
                            ky = ((r + 0.5 + ky_rel) * stride - pad_y) / scale
                            kc = float(kc_raw) if -0.05 <= kc_raw <= 1.05 else float(1.0 / (1.0 + np.exp(-np.clip(kc_raw, -20.0, 20.0))))
                            kpts_scaled[k] = [kx, ky, max(0.0, min(1.0, kc))]
                        boxes.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])
                        confidences.append(float(score_prob[r, c]))
                        keypoints_list.append(kpts_scaled)
            if boxes:
                indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_thresh, self.iou_thresh)
                if len(indices) > 0:
                    return [{"bbox": [float(boxes[i][0]), float(boxes[i][1]),
                                      float(boxes[i][0] + boxes[i][2]), float(boxes[i][1] + boxes[i][3])],
                             "confidence": float(confidences[i]),
                             "keypoints": keypoints_list[i]} for i in indices.flatten()]
            return []

        # --- Branch 2: Single concatenated tensor [1, 56, 8400] ---
        target_tensor = next(
            (o for o in output_list if len(np.squeeze(o).shape) >= 2 and any(s in np.squeeze(o).shape for s in [56, 57])),
            output_list[0]
        )
        output = np.squeeze(target_tensor).astype(np.float32)
        is_quantized = target_tensor.dtype in [np.int8, np.int16, np.uint8]
        if is_quantized:
            output = (output.astype(np.float32) + 128.0) / 255.0 if target_tensor.dtype in [np.int8, np.int16] else output.astype(np.float32) / 255.0

        if len(output.shape) == 2:
            d0, d1 = output.shape
            if d0 in [56, 57] or (d0 < d1 and d0 < 100):
                output = output.T

        if len(output.shape) != 2:
            return []

        boxes, confidences, keypoints_list = [], [], []
        for row in output:
            if len(row) < 56:
                continue
            raw_score = float(row[4])
            box_score = float(raw_score) if -0.05 <= raw_score <= 1.05 else float(1.0 / (1.0 + np.exp(-np.clip(raw_score, -20.0, 20.0))))
            if box_score < self.conf_thresh:
                continue
            cx, cy, w, h = row[0:4]
            if max(abs(cx), abs(cy), abs(w), abs(h)) <= 2.0 or is_quantized:
                cx *= w_target; cy *= h_target; w *= w_target; h *= h_target
            if w < 12 or h < 16 or w > w_target * 1.5:
                continue
            x1 = max(0.0, min(float(w_orig), (cx - w / 2.0 - pad_x) / scale))
            y1 = max(0.0, min(float(h_orig), (cy - h / 2.0 - pad_y) / scale))
            x2 = max(0.0, min(float(w_orig), (cx + w / 2.0 - pad_x) / scale))
            y2 = max(0.0, min(float(h_orig), (cy + h / 2.0 - pad_y) / scale))
            if (x2 - x1) < 10 or (y2 - y1) < 15:
                continue
            kpts_raw = row[5:56].reshape(17, 3)
            kpts_scaled = np.zeros((17, 3), dtype=np.float32)
            for k in range(17):
                kx_raw, ky_raw, kc_raw = kpts_raw[k]
                if max(abs(kx_raw), abs(ky_raw)) <= 2.0 or is_quantized:
                    kx_raw *= w_target; ky_raw *= h_target
                kx = (kx_raw - pad_x) / scale
                ky = (ky_raw - pad_y) / scale
                kc = float(kc_raw) if -0.05 <= kc_raw <= 1.05 else float(1.0 / (1.0 + np.exp(-np.clip(kc_raw, -20.0, 20.0))))
                kpts_scaled[k] = [kx, ky, max(0.0, min(1.0, kc))]
            boxes.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])
            confidences.append(float(box_score))
            keypoints_list.append(kpts_scaled)

        if not boxes:
            return []
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_thresh, self.iou_thresh)
        if len(indices) > 0:
            return [{"bbox": [float(boxes[i][0]), float(boxes[i][1]),
                               float(boxes[i][0] + boxes[i][2]), float(boxes[i][1] + boxes[i][3])],
                     "confidence": float(confidences[i]),
                     "keypoints": keypoints_list[i]} for i in indices.flatten()]
        return []
