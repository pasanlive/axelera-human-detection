"""
PoseEstimator: 17-Keypoint Body Pose Estimation Module for Axelera Metis.
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

class PoseEstimator:
    """YOLO Pose Estimator extracting 17 skeletal keypoints per person."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.conf_thresh = config.get("conf_threshold", 0.50)
        self.iou_thresh = config.get("iou_threshold", 0.45)
        self.input_size = tuple(config.get("input_size", [640, 640]))
        self.num_keypoints = config.get("num_keypoints", 17)
        self.model_name = config.get("model_name", "yolov8n-pose.pt")

        self.engine = VoyagerEngine(
            axm_path=config.get("axm_path"),
            onnx_path=config.get("onnx_path"),
            chip_id=config.get("chip_id", 0),
            num_cores=config.get("num_cores", 4)
        )

        # High-level official Ultralytics YOLO pose inference engine (loads ONNX / PyTorch format)
        self.ultralytics_model = None
        try:
            from ultralytics import YOLO
            model_candidates = [config.get("onnx_path"), self.model_name]
            for model_src in model_candidates:
                if model_src and (os.path.exists(str(model_src)) or str(model_src).endswith('.pt')):
                    try:
                        print(f"[POSE ESTIMATOR] Loading YOLO pose model '{model_src}' via Ultralytics engine...")
                        self.ultralytics_model = YOLO(model_src)
                        print(f"[POSE ESTIMATOR SUCCESS] Active pose engine loaded using '{model_src}'.")
                        break
                    except Exception as e:
                        print(f"[POSE ESTIMATOR NOTICE] Candidate '{model_src}' load notice: {e}")
                        continue
        except Exception as e:
            print(f"[POSE ESTIMATOR NOTICE] Ultralytics engine load notice: {e}")

    def estimate_pose(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Runs pose estimation across the entire frame or crops.
        :return: List of dicts with 'bbox', 'confidence', and 'keypoints' np.ndarray (17, 3) [x, y, conf]
        """
        if self.engine.get_backend() == "axelera_voyager":
            # Direct Axelera Metis AIPU NPU Execution Pathway
            h_orig, w_orig = frame.shape[:2]
            input_tensor, scale, (pad_x, pad_y) = self._preprocess(frame)
            outputs = self.engine.run(input_tensor)
            poses = self._postprocess(outputs, scale, pad_x, pad_y, w_orig, h_orig)
            return poses

        if self.ultralytics_model is not None:
            # Fallback PyTorch / ONNX inference pathway
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
                        "keypoints": kpts_data[i]  # shape (17, 3) -> x, y, conf
                    })
            return poses

        h_orig, w_orig = frame.shape[:2]
        input_tensor, scale, (pad_x, pad_y) = self._preprocess(frame)
        outputs = self.engine.run(input_tensor)
        poses = self._postprocess(outputs, scale, pad_x, pad_y, w_orig, h_orig)
        return poses

    def _preprocess(self, frame: np.ndarray):
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

    def _postprocess(self, outputs: Union[List[np.ndarray], np.ndarray], scale: float, pad_x: int, pad_y: int, w_orig: int, h_orig: int) -> List[Dict[str, Any]]:
        """Parses YOLO-Pose tensor outputs into bounding boxes and keypoints (supports float32, int8, uint8, and multi-head NPU shapes)."""
        if outputs is None:
            return []

        if not isinstance(outputs, list):
            output_list = [outputs]
        else:
            output_list = outputs

        if len(output_list) == 0:
            return []

        if not hasattr(self, '_postprocess_logged'):
            self._postprocess_logged = True
            print(f"[POSE ESTIMATOR DEBUG] output_list count={len(output_list)}, shapes={[o.shape for o in output_list if o is not None]}")

        # Handle Axelera 9-output multi-head FPN NPU structure (3 scales DFL + 3 scales Box Score + 3 scales Keypoints)
        if len(output_list) == 9:
            w_target, h_target = self.input_size
            boxes = []
            confidences = []
            keypoints_list = []

            for idx in range(3):
                dfl_raw = output_list[idx]
                score_raw = output_list[idx + 3]
                kpt_raw = output_list[idx + 6]
                if dfl_raw is None or score_raw is None or kpt_raw is None:
                    continue

                dfl = dfl_raw.astype(np.float32) / 12.8 if dfl_raw.dtype in [np.int8, np.int16] else dfl_raw.astype(np.float32)
                score = score_raw.astype(np.float32) / 12.8 if score_raw.dtype in [np.int8, np.int16] else score_raw.astype(np.float32)
                kpt = kpt_raw.astype(np.float32) / 12.8 if kpt_raw.dtype in [np.int8, np.int16] else kpt_raw.astype(np.float32)

                dfl = np.squeeze(dfl)
                score = np.squeeze(score)
                kpt = np.squeeze(kpt)

                if len(dfl.shape) == 3 and len(score.shape) == 3 and len(kpt.shape) == 3:
                    gh, gw = dfl.shape[0], dfl.shape[1]
                    stride = float(w_target) / float(gw) if gw > 0 else 8.0

                    dfl_reshaped = dfl.reshape(gh, gw, 4, 16)
                    dfl_softmax = np.exp(dfl_reshaped - np.max(dfl_reshaped, axis=-1, keepdims=True))
                    dfl_softmax = dfl_softmax / np.sum(dfl_softmax, axis=-1, keepdims=True)
                    dfl_val = np.sum(dfl_softmax * np.arange(16), axis=-1)

                    score_prob = 1.0 / (1.0 + np.exp(-score[:, :, 0]))

                    for r in range(gh):
                        for c in range(gw):
                            box_score = float(score_prob[r, c])

                            if box_score >= self.conf_thresh:
                                l_d, t_d, r_d, b_d = dfl_val[r, c, 0], dfl_val[r, c, 1], dfl_val[r, c, 2], dfl_val[r, c, 3]
                                cx = (c + 0.5 + (r_d - l_d) / 2.0) * stride
                                cy = (r + 0.5 + (b_d - t_d) / 2.0) * stride
                                w = (l_d + r_d) * stride
                                h = (t_d + b_d) * stride

                                x1 = (cx - w / 2.0 - pad_x) / scale
                                y1 = (cy - h / 2.0 - pad_y) / scale
                                x2 = (cx + w / 2.0 - pad_x) / scale
                                y2 = (cy + h / 2.0 - pad_y) / scale

                                x1 = max(0.0, min(w_orig, x1))
                                y1 = max(0.0, min(h_orig, y1))
                                x2 = max(0.0, min(w_orig, x2))
                                y2 = max(0.0, min(h_orig, y2))

                                kpts_raw = kpt[r, c, 0:51].reshape(17, 3)
                                kpts_scaled = np.zeros((17, 3), dtype=np.float32)

                                for k in range(17):
                                    kx_rel, ky_rel, kc_raw = kpts_raw[k]
                                    kx = ((c + 0.5 + kx_rel) * stride - pad_x) / scale
                                    ky = ((r + 0.5 + ky_rel) * stride - pad_y) / scale
                                    kc = 1.0 / (1.0 + np.exp(-kc_raw))
                                    kpts_scaled[k] = [kx, ky, kc]

                                boxes.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])
                                confidences.append(float(box_score))
                                keypoints_list.append(kpts_scaled)

            if len(boxes) > 0:
                indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_thresh, self.iou_thresh)
                results = []
                if len(indices) > 0:
                    for i in indices.flatten():
                        x, y, w, h = boxes[i]
                        results.append({
                            "bbox": [float(x), float(y), float(x + w), float(y + h)],
                            "confidence": float(confidences[i]),
                            "keypoints": keypoints_list[i]
                        })
                return results

        # Search for tensor containing expected YOLO pose feature channels (56, 57, 17)
        target_tensor = None
        for o in output_list:
            if o is None:
                continue
            s = np.squeeze(o).shape
            if len(s) >= 2 and (56 in s or 57 in s or 17 in s):
                target_tensor = o
                break

        if target_tensor is None:
            target_tensor = output_list[0]

        output = target_tensor

        is_quantized = False
        if output.dtype in [np.int8, np.int16]:
            output = (output.astype(np.float32) + 128.0) / 255.0
            is_quantized = True
        elif output.dtype == np.uint8:
            output = output.astype(np.float32) / 255.0
            is_quantized = True
        # Strip batch dimensions [1, C, N] -> [C, N]
        while len(output.shape) > 2 and output.shape[0] == 1:
            output = output[0]

        output = np.squeeze(output)

        # Handle 3D output shapes from NPU (e.g. [56, H, W] or [H, W, 56])
        if len(output.shape) == 3:
            s0, s1, s2 = output.shape
            if s0 in [56, 57, 17] or s0 < min(s1, s2):
                output = output.reshape(s0, -1).T
            elif s2 in [56, 57, 17] or s2 < min(s0, s1):
                output = output.reshape(-1, s2)
            else:
                output = output.reshape(-1, s2)

        if len(output.shape) != 2:
            return []

        d0, d1 = output.shape
        if d0 in [56, 57, 17] or (d0 < d1 and d0 < 100):
            output = output.T

        w_target, h_target = self.input_size
        boxes = []
        confidences = []
        keypoints_list = []

        for row in output:
            if len(row) < 56:
                continue

            raw_score = float(row[4])
            box_score = 1.0 / (1.0 + np.exp(-raw_score)) if (raw_score > 1.0 or raw_score < 0.0) else raw_score
            if box_score < self.conf_thresh:
                continue

            cx, cy, w, h = row[0:4]
            if max(abs(cx), abs(cy), abs(w), abs(h)) <= 2.0 or is_quantized:
                cx *= w_target
                cy *= h_target
                w *= w_target
                h *= h_target

            x1 = (cx - w / 2 - pad_x) / scale
            y1 = (cy - h / 2 - pad_y) / scale
            x2 = (cx + w / 2 - pad_x) / scale
            y2 = (cy + h / 2 - pad_y) / scale

            # Extract 17 keypoints (51 elements starting from index 5)
            kpts_raw = row[5:56].reshape(17, 3)
            kpts_scaled = np.zeros((17, 3), dtype=np.float32)

            for k in range(17):
                kx_raw, ky_raw, kc_raw = kpts_raw[k]
                if max(abs(kx_raw), abs(ky_raw)) <= 2.0 or is_quantized:
                    kx_raw *= w_target
                    ky_raw *= h_target
                kx = (kx_raw - pad_x) / scale
                ky = (ky_raw - pad_y) / scale
                kc = 1.0 / (1.0 + np.exp(-kc_raw)) if (kc_raw > 1.0 or kc_raw < 0.0) else kc_raw
                kpts_scaled[k] = [kx, ky, kc]

            boxes.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])
            confidences.append(float(box_score))
            keypoints_list.append(kpts_scaled)

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
                    "keypoints": keypoints_list[i]
                })

        return results
