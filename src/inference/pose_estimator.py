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

        output_list = [o for o in output_list if o is not None and o.size > 0]
        if len(output_list) == 0:
            return []

        if not hasattr(self, '_postprocess_logged'):
            self._postprocess_logged = True
            print(f"[POSE ESTIMATOR DEBUG] output_list count={len(output_list)}, shapes={[o.shape for o in output_list]}")

        w_target, h_target = self.input_size

        # =====================================================================
        # Branch 1: Axelera 9-output multi-head FPN NPU structure
        # (3 scales DFL 64ch + 3 scales Box Score 1ch + 3 scales Keypoints 51ch)
        # =====================================================================
        dfl_heads = []
        score_heads = []
        kpt_heads = []

        for raw_t in output_list:
            t = np.array(raw_t)
            while len(t.shape) > 3 and t.shape[0] == 1:
                t = t[0]
            t = np.squeeze(t)
            if len(t.shape) != 3:
                continue

            # Ensure layout is (H, W, C)
            if t.shape[0] in [64, 1, 51] and t.shape[0] != t.shape[1]:
                t = np.transpose(t, (1, 2, 0))

            c_dim = t.shape[-1]
            if c_dim == 64:
                dfl_heads.append(t)
            elif c_dim == 1:
                score_heads.append(t)
            elif c_dim == 51:
                kpt_heads.append(t)

        # Sort heads by spatial resolution descending (stride 8: 64x64, stride 16: 32x32, stride 32: 16x16)
        dfl_heads.sort(key=lambda x: x.shape[0] * x.shape[1], reverse=True)
        score_heads.sort(key=lambda x: x.shape[0] * x.shape[1], reverse=True)
        kpt_heads.sort(key=lambda x: x.shape[0] * x.shape[1], reverse=True)

        num_scales = min(len(dfl_heads), len(score_heads), len(kpt_heads))

        if num_scales >= 3:
            boxes = []
            confidences = []
            keypoints_list = []

            for idx in range(num_scales):
                dfl = dfl_heads[idx].astype(np.float32)
                score = score_heads[idx].astype(np.float32)
                kpt = kpt_heads[idx].astype(np.float32)

                gh, gw = dfl.shape[0], dfl.shape[1]
                if score.shape[0] != gh or score.shape[1] != gw or kpt.shape[0] != gh or kpt.shape[1] != gw:
                    continue

                stride = float(w_target) / float(gw) if gw > 0 else 8.0

                # Quantization de-quantize if necessary
                if dfl_heads[idx].dtype in [np.int8, np.int16]:
                    dfl = dfl / 12.8
                if score_heads[idx].dtype in [np.int8, np.int16]:
                    score = score / 12.8
                if kpt_heads[idx].dtype in [np.int8, np.int16]:
                    kpt = kpt / 12.8

                score_val = score[:, :, 0]
                score_min = float(np.min(score_val))
                score_max = float(np.max(score_val))

                # If values are already in [0, 1] range, DO NOT apply sigmoid!
                # Applying sigmoid to 0.0 gives 0.50, which causes false detections everywhere!
                if score_min >= 0.0 and score_max <= 1.05:
                    score_prob = score_val
                else:
                    score_prob = 1.0 / (1.0 + np.exp(-np.clip(score_val, -20.0, 20.0)))

                # Decode DFL
                dfl_reshaped = dfl.reshape(gh, gw, 4, 16)
                dfl_softmax = np.exp(dfl_reshaped - np.max(dfl_reshaped, axis=-1, keepdims=True))
                dfl_softmax = dfl_softmax / np.sum(dfl_softmax, axis=-1, keepdims=True)
                dfl_val = np.sum(dfl_softmax * np.arange(16, dtype=np.float32), axis=-1)

                # Softmax confidence sanity check: flat uniform noise gives max prob = 1/16 = 0.0625
                dfl_max_prob = np.max(dfl_softmax, axis=-1)
                dfl_is_peaked = np.min(dfl_max_prob, axis=-1) > 0.10

                for r in range(gh):
                    for c in range(gw):
                        box_score = float(score_prob[r, c])

                        if box_score < self.conf_thresh:
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

                        kpts_raw = kpt[r, c, 0:51].reshape(17, 3)
                        kpts_scaled = np.zeros((17, 3), dtype=np.float32)

                        for k in range(17):
                            kx_rel, ky_rel, kc_raw = kpts_raw[k]
                            kx = ((c + 0.5 + kx_rel) * stride - pad_x) / scale
                            ky = ((r + 0.5 + ky_rel) * stride - pad_y) / scale
                            if kc_raw > 1.05 or kc_raw < -0.05:
                                kc = 1.0 / (1.0 + np.exp(-np.clip(kc_raw, -20.0, 20.0)))
                            else:
                                kc = float(kc_raw)
                            kc = max(0.0, min(1.0, float(kc)))
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

        # =====================================================================
        # Branch 2: Standard Single Output Tensor [1, 56, 8400] or [1, 8400, 56]
        # =====================================================================
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

        boxes = []
        confidences = []
        keypoints_list = []

        for row in output:
            if len(row) < 56:
                continue

            raw_score = float(row[4])
            if raw_score > 1.05 or raw_score < -0.05:
                box_score = 1.0 / (1.0 + np.exp(-np.clip(raw_score, -20.0, 20.0)))
            else:
                box_score = float(raw_score)

            if box_score < self.conf_thresh:
                continue

            cx, cy, w, h = row[0:4]
            if max(abs(cx), abs(cy), abs(w), abs(h)) <= 2.0 or is_quantized:
                cx *= w_target
                cy *= h_target
                w *= w_target
                h *= h_target

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
                if kc_raw > 1.05 or kc_raw < -0.05:
                    kc = 1.0 / (1.0 + np.exp(-np.clip(kc_raw, -20.0, 20.0)))
                else:
                    kc = float(kc_raw)
                kc = max(0.0, min(1.0, float(kc)))
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
