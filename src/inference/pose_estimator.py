"""
PoseEstimator: 17-Keypoint Body Pose Estimation Module for Axelera Metis.
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Optional
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
        self.input_size = tuple(config.get("input_size", [640, 640]))
        self.num_keypoints = config.get("num_keypoints", 17)
        self.model_name = config.get("model_name", "yolov8n-pose.pt")

        self.engine = VoyagerEngine(
            axm_path=config.get("axm_path"),
            onnx_path=config.get("onnx_path")
        )

        self.ultralytics_model = None
        if self.engine.get_backend() == "virtual":
            try:
                from ultralytics import YOLO
                print(f"[POSE ESTIMATOR] Loading PyTorch pose model '{self.model_name}' via Ultralytics engine...")
                self.ultralytics_model = YOLO(self.model_name)
            except Exception as e:
                print(f"[POSE ESTIMATOR NOTICE] Ultralytics PyTorch load: {e}")

    def estimate_pose(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Runs pose estimation across the entire frame or crops.
        :return: List of dicts with 'bbox', 'confidence', and 'keypoints' np.ndarray (17, 3) [x, y, conf]
        """
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
                        "keypoints": kpts_data[i]  # shape (17, 3) -> x, y, conf
                    })
            return poses

        # Voyager tensor execution path
        h_orig, w_orig = frame.shape[:2]
        input_tensor, scale, (pad_x, pad_y) = self._preprocess(frame)
        outputs = self.engine.run(input_tensor)

        poses = self._postprocess(outputs[0], scale, pad_x, pad_y, w_orig, h_orig)
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

    def _postprocess(self, output: np.ndarray, scale: float, pad_x: int, pad_y: int, w_orig: int, h_orig: int) -> List[Dict[str, Any]]:
        """Parses YOLO-Pose tensor outputs [1, 56, 8400] into bounding boxes and keypoints."""
        if len(output.shape) == 3:
            output = output[0]  # [56, 8400]
        if output.shape[0] < output.shape[1]:
            output = output.T    # [8400, 56]

        poses = []
        for row in output:
            box_score = row[4]
            if box_score < self.conf_thresh:
                continue

            cx, cy, w, h = row[0:4]
            x1 = (cx - w / 2 - pad_x) / scale
            y1 = (cy - h / 2 - pad_y) / scale
            x2 = (cx + w / 2 - pad_x) / scale
            y2 = (cy + h / 2 - pad_y) / scale

            # Extract 17 keypoints (51 elements starting from index 5)
            kpts_raw = row[5:56].reshape(17, 3)
            kpts_scaled = np.zeros((17, 3), dtype=np.float32)

            for k in range(17):
                kx = (kpts_raw[k, 0] - pad_x) / scale
                ky = (kpts_raw[k, 1] - pad_y) / scale
                kc = kpts_raw[k, 2]
                kpts_scaled[k] = [kx, ky, kc]

            poses.append({
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "confidence": float(box_score),
                "keypoints": kpts_scaled
            })

        return poses
