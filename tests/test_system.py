"""
Unit tests for Axelera Metis Multi-Camera System components.
"""

import unittest
import numpy as np
import os
import tempfile
import yaml

from src.camera.base_camera import SyntheticCamera
from src.utils.face_db import FaceDatabase
from src.inference.face_recognizer import FaceRecognizer
from src.inference.yolo_detector import YOLODetector
from src.inference.pose_estimator import PoseEstimator
from src.tracking.byte_tracker import ByteTracker
from src.utils.visualization import Visualizer
from src.pipeline import MultiCameraPipeline

class TestAxeleraSystem(unittest.TestCase):

    def test_synthetic_camera(self):
        cam = SyntheticCamera("test_cam", "Test Stream", fps_limit=30, width=640, height=480)
        self.assertTrue(cam.open())
        ret, frame = cam.read_frame()
        self.assertTrue(ret)
        self.assertIsNotNone(frame)
        self.assertEqual(frame.shape, (480, 640, 3))
        cam.close()

    def test_face_db_enroll_and_match(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_face_db.json")
            db = FaceDatabase(db_path=db_path)

            # Create synthetic embedding
            vec_john = np.random.randn(512).astype(np.float32)
            vec_john /= np.linalg.norm(vec_john)

            db.add_identity("John Doe", vec_john)

            # Match identical vector
            name, score = db.match(vec_john, threshold=0.5)
            self.assertEqual(name, "John Doe")
            self.assertGreaterEqual(score, 0.99)

            # Match orthogonal vector (Unknown)
            vec_unknown = np.random.randn(512).astype(np.float32)
            name_u, score_u = db.match(vec_unknown, threshold=0.99)
            self.assertEqual(name_u, "Unknown")

    def test_byte_tracker(self):
        tracker = ByteTracker(track_thresh=0.5)
        dets = [
            {"bbox": [100.0, 100.0, 200.0, 200.0], "confidence": 0.9, "label": "Person"}
        ]
        tracked_dets = tracker.update(dets)
        self.assertEqual(len(tracked_dets), 1)
        self.assertIn("track_id", tracked_dets[0])
        self.assertEqual(tracked_dets[0]["track_id"], 1)

    def test_visualizer(self):
        vis = Visualizer({})
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = [{"bbox": [50, 50, 150, 150], "confidence": 0.88, "track_id": 1}]
        poses = [{"keypoints": np.zeros((17, 3))}]
        faces = [{"bbox": [60, 60, 100, 100], "name": "Alice", "similarity": 0.95}]

        out = vis.draw_frame(frame, detections, poses, faces, "Cam 1", 30.0)
        self.assertIsNotNone(out)
        self.assertEqual(out.shape, (480, 640, 3))

    def test_pipeline_step(self):
        config = {
            "hardware": {"device": "virtual"},
            "cameras": [
                {"id": "cam_01", "name": "Test Cam 1", "source": "synthetic", "enabled": True},
                {"id": "cam_02", "name": "Test Cam 2", "source": "synthetic", "enabled": True}
            ],
            "models": {
                "human_detector": {"conf_threshold": 0.4, "input_size": [640, 640], "person_class_id": 0},
                "pose_estimator": {"conf_threshold": 0.4, "input_size": [640, 640]},
                "face_recognizer": {"match_threshold": 0.6, "input_size": [112, 112]}
            },
            "tracking": {"enabled": True},
            "face_db": {"path": "data/test_db.json"},
            "visualization": {"draw_fps": True}
        }
        pipeline = MultiCameraPipeline(config)
        pipeline.start()
        import time
        time.sleep(0.1)
        
        # Run 2 steps
        step_frames = pipeline.process_step()
        self.assertIn("cam_01", step_frames)
        self.assertIn("cam_02", step_frames)

        grid = pipeline.compose_grid(step_frames)
        self.assertEqual(grid.shape, (720, 1280, 3))

        pipeline.stop()

if __name__ == "__main__":
    unittest.main()
