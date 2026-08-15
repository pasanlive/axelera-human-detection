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

            # Step A: Human Detection (Runs on every frame for high responsiveness)
            detections = self.detector.detect(frame)

            # Step B: Object Tracking (ByteTrack)
            if cam_id in self.trackers:
                detections = self.trackers[cam_id].update(detections)

            # Step C: Pose Estimation (Runs on cadence every N frames)
            if curr_frame_idx % self.pose_interval == 0 or not self.cached_poses.get(cam_id):
                poses = self.pose_estimator.estimate_pose(frame)
                self.cached_poses[cam_id] = poses
            else:
                poses = self.cached_poses.get(cam_id, [])

            # Step D: Face Recognition (Optimized with Track ID caching)
            person_boxes = []
            uncached_person_boxes = []
            
            for d in detections:
                bbox = d["bbox"]
                track_id = d.get("track_id")
                person_boxes.append(bbox)
                
                # Check if face identity for this track_id is already cached
                if track_id is not None and track_id in self.face_cache.get(cam_id, {}):
                    # Use cached face identity
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
            active_faces = []
            for d in detections:
                tid = d.get("track_id")
                if tid and tid in self.face_cache.get(cam_id, {}):
                    face_info = self.face_cache[cam_id][tid].copy()
                    # Align face bbox with current moving person bbox head region
                    x1, y1, x2, y2 = d["bbox"]
                    head_h = int((y2 - y1) * 0.35)
                    face_info["bbox"] = [x1, y1, x2, y1 + head_h]
                    active_faces.append(face_info)

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
