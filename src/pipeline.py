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

        # Initialize trackers for active cameras
        for cam_id in self.stream_manager.cameras.keys():
            self.trackers[cam_id] = ByteTracker(
                track_thresh=self.config.get("tracking", {}).get("track_thresh", 0.5)
            )

        print("[PIPELINE SUCCESS] System initialized and actively processing streams.")

    def process_step(self) -> Dict[str, np.ndarray]:
        """
        Executes one processing step across all active camera streams.
        :return: Dict of camera_id -> rendered frame numpy array
        """
        frames = self.stream_manager.get_all_frames()
        output_frames = {}

        for cam_id, frame in frames.items():
            cam_obj = self.stream_manager.cameras.get(cam_id)
            cam_name = cam_obj.name if cam_obj else cam_id
            fps = cam_obj.fps if cam_obj else 0.0

            # Step A: Human Detection
            detections = self.detector.detect(frame)

            # Step B: Object Tracking
            if cam_id in self.trackers:
                detections = self.trackers[cam_id].update(detections)

            # Step C: Pose Estimation
            poses = self.pose_estimator.estimate_pose(frame)

            # Step D: Face Recognition (utilizing detected human bboxes for targeted face cropping)
            person_boxes = [d["bbox"] for d in detections]
            faces = self.face_recognizer.recognize_faces_in_frame(frame, person_bboxes=person_boxes)

            # Step E: Visualization / Rendering
            rendered = self.visualizer.draw_frame(
                frame=frame,
                detections=detections,
                poses=poses,
                faces=faces,
                stream_title=f"{cam_name}",
                fps=fps
            )

            output_frames[cam_id] = rendered

        return output_frames

    def compose_grid(self, frames_dict: Dict[str, np.ndarray], grid_w: int = 1280, grid_h: int = 720) -> np.ndarray:
        """
        Combines multiple camera outputs into a single tiled grid layout.
        """
        if not frames_dict:
            blank = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
            cv2.putText(blank, "No Active Camera Streams", (grid_w // 2 - 200, grid_h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            return blank

        frames_list = list(frames_dict.values())
        num_streams = len(frames_list)

        if num_streams == 1:
            return cv2.resize(frames_list[0], (grid_w, grid_h))

        # Tile grid dimensions
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
