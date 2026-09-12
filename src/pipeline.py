"""
Pipeline: End-to-End Multi-Camera Detection, Pose, and Face Recognition System Orchestrator.
"""

import time
import cv2
import numpy as np
from typing import Dict, Any, List

from src.camera.stream_manager import StreamManager
from src.inference.yolo_detector import YOLODetector
from src.inference.pose_estimator import PoseEstimator
from src.inference.face_recognizer import FaceRecognizer
from src.utils.face_db import FaceDatabase
from src.tracking.byte_tracker import ByteTracker
from src.utils.visualization import Visualizer

class MultiCameraPipeline:
    """Orchestrates multi-camera acquisition, Axelera Metis model inference, tracking, and visualization."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

        print("==========================================================")
        print("  Initializing Axelera Metis Multi-Camera Pipeline System ")
        print("==========================================================")

        # Performance settings
        perf_cfg = config.get("performance", {})
        self.pose_interval = perf_cfg.get("pose_interval", 2)   # Run pose every N frames
        self.face_interval = perf_cfg.get("face_interval", 3)   # Run face recognition every N frames
        self.frame_counters: Dict[str, int] = {}                # cam_id -> frame_count

        # Track ID caching dictionaries
        self.cached_poses: Dict[str, List[Dict[str, Any]]] = {}   # cam_id -> last poses
        self.face_cache: Dict[str, Dict[int, Dict[str, Any]]] = {} # cam_id -> {track_id: face_info}

        # Model enablement state
        models_cfg = config.get("models", {})
        self.enabled_models: Dict[str, bool] = {
            "human_detector": models_cfg.get("human_detector", {}).get("enabled", True),
            "pose_estimator": models_cfg.get("pose_estimator", {}).get("enabled", True),
            "face_recognizer": models_cfg.get("face_recognizer", {}).get("enabled", True),
        }
        self.camera_model_overrides: Dict[str, Dict[str, bool]] = {}

        # 1. Initialize Face Database
        db_path = config.get("face_db", {}).get("path", "data/face_db.json")
        self.face_db = FaceDatabase(db_path=db_path)

        # 2. Initialize Models
        print("[PIPELINE] Initializing YOLO Human Detector...")
        self.detector = YOLODetector(config["models"]["human_detector"])

        print("[PIPELINE] Initializing YOLO Pose Estimator...")
        self.pose_estimator = PoseEstimator(config["models"]["pose_estimator"])

        print("[PIPELINE] Initializing Face Recognizer...")
        self.face_recognizer = FaceRecognizer(config["models"]["face_recognizer"], self.face_db)

        # 3. Trackers per camera stream
        self.trackers: Dict[str, ByteTracker] = {}

        # 4. Initialize Multi-Camera Stream Manager
        self.stream_manager = StreamManager(config.get("cameras", []))
        self.stream_manager.initialize_cameras()

        # 5. Visualizer
        self.visualizer = Visualizer(config)

        self.is_running = False

    def is_model_enabled(self, model_name: str, cam_id: Any = None) -> bool:
        """Checks whether a specific model is enabled globally or for a specific camera."""
        if cam_id and cam_id in self.camera_model_overrides:
            cam_overrides = self.camera_model_overrides[cam_id]
            if model_name in cam_overrides:
                return cam_overrides[model_name]
        return self.enabled_models.get(model_name, True)

    def set_model_enabled(self, model_name: str, enabled: bool, cam_id: Any = None):
        """Enables or disables model inference globally or for a specific camera."""
        if cam_id and cam_id != "all":
            if cam_id not in self.camera_model_overrides:
                self.camera_model_overrides[cam_id] = {}
            self.camera_model_overrides[cam_id][model_name] = enabled
            print(f"[PIPELINE ADMIN] Model '{model_name}' on stream '{cam_id}' set to: {'ENABLED' if enabled else 'DISABLED'}")
        else:
            self.enabled_models[model_name] = enabled
            # Clear per-cam overrides for this model to enforce global setting
            for c_id in self.camera_model_overrides:
                if model_name in self.camera_model_overrides[c_id]:
                    del self.camera_model_overrides[c_id][model_name]
            print(f"[PIPELINE ADMIN] Model '{model_name}' globally set to: {'ENABLED' if enabled else 'DISABLED'}")

        # Clear cached data when disabling to immediately release stale inference
        if not enabled:
            if model_name == "pose_estimator":
                if cam_id and cam_id != "all":
                    self.cached_poses[cam_id] = []
                else:
                    for cid in self.cached_poses:
                        self.cached_poses[cid] = []
            elif model_name == "face_recognizer":
                if cam_id and cam_id != "all":
                    self.face_cache[cam_id] = {}
                else:
                    for cid in self.face_cache:
                        self.face_cache[cid] = {}

    def get_models_status(self) -> List[Dict[str, Any]]:
        """Returns status, specs, and telemetry of all registered AI inference models."""
        models_cfg = self.config.get("models", {})
        det_cfg = models_cfg.get("human_detector", {})
        pose_cfg = models_cfg.get("pose_estimator", {})
        face_cfg = models_cfg.get("face_recognizer", {})

        det_enabled = self.is_model_enabled("human_detector")
        pose_enabled = self.is_model_enabled("pose_estimator")
        face_enabled = self.is_model_enabled("face_recognizer")

        det_backend = getattr(getattr(self.detector, 'engine', None), 'backend', 'Unknown')
        pose_backend = getattr(getattr(self.pose_estimator, 'engine', None), 'backend', 'Unknown')
        face_backend = getattr(getattr(self.face_recognizer, 'embedder_engine', None), 'backend', 'Unknown')

        return [
            {
                "id": "human_detector",
                "name": "YOLOv8 Human Detector",
                "task": "Object Detection",
                "description": "Detects persons with bounding box localization & confidence scoring",
                "enabled": det_enabled,
                "inferencing": det_enabled and self.is_running,
                "backend": det_backend,
                "weights": det_cfg.get("axm_path") or det_cfg.get("model_name", "yolov8n.pt"),
                "input_size": det_cfg.get("input_size", [512, 512]),
                "conf_threshold": det_cfg.get("conf_threshold", 0.45),
                "fps_cadence": "Every frame (1:1)",
                "icon": "user-check"
            },
            {
                "id": "pose_estimator",
                "name": "YOLOv8-Pose Estimator",
                "task": "Pose Estimation",
                "description": "17-Keypoint skeletal joint tracking and action/posture analysis",
                "enabled": pose_enabled,
                "inferencing": pose_enabled and self.is_running,
                "backend": pose_backend,
                "weights": pose_cfg.get("axm_path") or pose_cfg.get("model_name", "yolov8n-pose.pt"),
                "input_size": pose_cfg.get("input_size", [512, 512]),
                "conf_threshold": pose_cfg.get("conf_threshold", 0.50),
                "fps_cadence": f"Every {self.pose_interval} frames",
                "icon": "activity"
            },
            {
                "id": "face_recognizer",
                "name": "ArcFace Facial Recognizer",
                "task": "Biometric Identification",
                "description": "512-dimensional facial embedding extraction and identity matching",
                "enabled": face_enabled,
                "inferencing": face_enabled and self.is_running,
                "backend": face_backend,
                "weights": face_cfg.get("embedder_axm") or "arcface_mobilefacenet.axm",
                "input_size": face_cfg.get("input_size", [112, 112]),
                "match_threshold": face_cfg.get("match_threshold", 0.60),
                "fps_cadence": f"Every {self.face_interval} frames",
                "icon": "smile"
            }
        ]

    def start(self):
        """Starts multi-camera capture and processing loop."""
        self.stream_manager.start()
        self.is_running = True

        for cam_id in self.stream_manager.cameras.keys():
            self.trackers[cam_id] = ByteTracker(
                track_thresh=self.config.get("tracking", {}).get("track_thresh", 0.5)
            )
            self.frame_counters[cam_id] = 0
            self.cached_poses[cam_id] = []
            self.face_cache[cam_id] = {}

        print("[PIPELINE SUCCESS] System initialized and actively processing streams.")

    def process_step(self) -> Dict[str, np.ndarray]:
        """
        Executes one processing step across all active camera streams with high-FPS caching optimizations.
        Only models that are enabled will perform neural network inference.
        :return: Dict of camera_id -> rendered frame numpy array
        """
        frames = self.stream_manager.get_all_frames()
        output_frames = {}

        for cam_id, frame in frames.items():
            cam_obj = self.stream_manager.cameras.get(cam_id)
            cam_name = cam_obj.name if cam_obj else cam_id
            fps = cam_obj.fps if cam_obj else 0.0

            self.frame_counters[cam_id] = self.frame_counters.get(cam_id, 0) + 1
            curr_frame_idx = self.frame_counters[cam_id]

            # Step A: Human Detection (Only execute inferencing if model is enabled)
            if self.is_model_enabled("human_detector", cam_id):
                detections = self.detector.detect(frame)
            else:
                detections = []

            # Step B: Object Tracking (ByteTrack)
            if cam_id in self.trackers:
                if detections:
                    detections = self.trackers[cam_id].update(detections)
                else:
                    detections = []

            # Step C: Pose Estimation (Only execute inferencing if model is enabled)
            if self.is_model_enabled("pose_estimator", cam_id):
                if curr_frame_idx % self.pose_interval == 0 or not self.cached_poses.get(cam_id):
                    poses = self.pose_estimator.estimate_pose(frame)
                    self.cached_poses[cam_id] = poses
                else:
                    poses = self.cached_poses.get(cam_id, [])
            else:
                poses = []
                self.cached_poses[cam_id] = []

            # Step D: Face Recognition (Only execute inferencing if model is enabled)
            active_faces = []
            if self.is_model_enabled("face_recognizer", cam_id):
                if self.is_model_enabled("human_detector", cam_id):
                    person_boxes = []
                    uncached_person_boxes = []
                    
                    for d in detections:
                        bbox = d["bbox"]
                        track_id = d.get("track_id")
                        person_boxes.append(bbox)
                        
                        # Check if face identity for this track_id is already cached
                        if track_id is not None and track_id in self.face_cache.get(cam_id, {}):
                            pass
                        else:
                            uncached_person_boxes.append(bbox)

                    # Execute face feature extraction for uncached or periodic re-check frames
                    if curr_frame_idx % self.face_interval == 0 or uncached_person_boxes:
                        target_boxes = uncached_person_boxes if (curr_frame_idx % self.face_interval != 0) else person_boxes
                        if target_boxes:
                            new_faces = self.face_recognizer.recognize_faces_in_frame(frame, person_bboxes=target_boxes)
                            # Update face cache by matching bbox position to track_id
                            for face in new_faces:
                                f_box = face["bbox"]
                                for d in detections:
                                    d_box = d["bbox"]
                                    if self._bbox_overlap(f_box, d_box) > 0.3:
                                        tid = d.get("track_id")
                                        if tid:
                                            self.face_cache[cam_id][tid] = face

                    # Assemble face list for renderer (combining active tracks & face cache)
                    for d in detections:
                        tid = d.get("track_id")
                        if tid and tid in self.face_cache.get(cam_id, {}):
                            face_info = self.face_cache[cam_id][tid].copy()
                            # Align face bbox with current moving person bbox head region
                            x1, y1, x2, y2 = d["bbox"]
                            head_h = int((y2 - y1) * 0.35)
                            face_info["bbox"] = [x1, y1, x2, y1 + head_h]
                            active_faces.append(face_info)
                else:
                    # Human detector is disabled: run direct full-frame face recognition
                    if curr_frame_idx % self.face_interval == 0:
                        active_faces = self.face_recognizer.recognize_faces_in_frame(frame, person_bboxes=None)
            else:
                self.face_cache[cam_id] = {}

            # Step E: Visualization / Rendering
            rendered = self.visualizer.draw_frame(
                frame=frame,
                detections=detections,
                poses=poses,
                faces=active_faces,
                stream_title=f"{cam_name}",
                fps=fps
            )

            output_frames[cam_id] = rendered

        return output_frames

    @staticmethod
    def _bbox_overlap(b1: List[float], b2: List[float]) -> float:
        """Calculates area overlap ratio between face box and person box."""
        x1 = max(b1[0], b2[0])
        y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2])
        y2 = min(b1[3], b2[3])

        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        return inter / area1 if area1 > 0 else 0.0

    def compose_grid(self, frames_dict: Dict[str, np.ndarray], grid_w: int = 1280, grid_h: int = 720) -> np.ndarray:
        """Combines multiple camera outputs into a single tiled grid layout."""
        if not frames_dict:
            blank = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
            cv2.putText(blank, "No Active Camera Streams", (grid_w // 2 - 200, grid_h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            return blank

        frames_list = list(frames_dict.values())
        num_streams = len(frames_list)

        if num_streams == 1:
            return cv2.resize(frames_list[0], (grid_w, grid_h))

        cols = 2 if num_streams <= 4 else 3
        rows = int(np.ceil(num_streams / cols))

        tile_w = grid_w // cols
        tile_h = grid_h // rows

        grid = np.zeros((tile_h * rows, tile_w * cols, 3), dtype=np.uint8)

        for idx, frame in enumerate(frames_list):
            r = idx // cols
            c = idx % cols
            tile = cv2.resize(frame, (tile_w, tile_h))
            grid[r * tile_h:(r + 1) * tile_h, c * tile_w:(c + 1) * tile_w] = tile

        return cv2.resize(grid, (grid_w, grid_h))

    def stop(self):
        """Stops the processing pipeline."""
        self.is_running = False
        self.stream_manager.stop()
