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
from src.web.server import WebServer

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

    def test_pipeline_model_enabling_disabling(self):
        """Verifies that only enabled models use inferencing and disabled models are strictly bypassed."""
        from unittest.mock import MagicMock
        config = {
            "hardware": {"device": "virtual"},
            "cameras": [
                {"id": "cam_01", "name": "Test Cam 1", "source": "synthetic", "enabled": True}
            ],
            "models": {
                "human_detector": {"enabled": True, "conf_threshold": 0.4, "input_size": [640, 640]},
                "pose_estimator": {"enabled": True, "conf_threshold": 0.4, "input_size": [640, 640]},
                "face_recognizer": {"enabled": True, "match_threshold": 0.6, "input_size": [112, 112]}
            },
            "performance": {"pose_interval": 1, "face_interval": 1},
            "tracking": {"enabled": True},
            "face_db": {"path": "data/test_db.json"},
            "visualization": {"draw_fps": True}
        }
        pipeline = MultiCameraPipeline(config)
        pipeline.start()
        import time
        time.sleep(0.1)

        # Check initial models status
        models_status = pipeline.get_models_status()
        self.assertEqual(len(models_status), 3)
        for m in models_status:
            self.assertTrue(m["enabled"])
            self.assertTrue(m["inferencing"])

        # Mock the underlying inferencing methods to track calls
        mock_det = [{"bbox": [50.0, 50.0, 150.0, 200.0], "confidence": 0.9, "label": "Person"}]
        pipeline.detector.detect = MagicMock(return_value=mock_det)
        pipeline.pose_estimator.estimate_pose = MagicMock(return_value=[])
        pipeline.face_recognizer.recognize_faces_in_frame = MagicMock(return_value=[])

        # Step 1: All enabled -> all should be called
        pipeline.process_step()
        self.assertEqual(pipeline.detector.detect.call_count, 1)
        self.assertEqual(pipeline.pose_estimator.estimate_pose.call_count, 1)
        self.assertEqual(pipeline.face_recognizer.recognize_faces_in_frame.call_count, 1)

        # Reset mocks
        pipeline.detector.detect.reset_mock()
        pipeline.pose_estimator.estimate_pose.reset_mock()
        pipeline.face_recognizer.recognize_faces_in_frame.reset_mock()

        # Step 2: Disable all models -> NO inferencing should be executed!
        pipeline.set_model_enabled("human_detector", False)
        pipeline.set_model_enabled("pose_estimator", False)
        pipeline.set_model_enabled("face_recognizer", False)

        pipeline.process_step()
        self.assertEqual(pipeline.detector.detect.call_count, 0, "Detector inference should NOT run when disabled!")
        self.assertEqual(pipeline.pose_estimator.estimate_pose.call_count, 0, "Pose inference should NOT run when disabled!")
        self.assertEqual(pipeline.face_recognizer.recognize_faces_in_frame.call_count, 0, "Face recognizer inference should NOT run when disabled!")

        # Step 3: Selectively enable ONLY pose_estimator
        pipeline.set_model_enabled("pose_estimator", True)
        pipeline.process_step()
        self.assertEqual(pipeline.detector.detect.call_count, 0, "Detector should still NOT run")
        self.assertEqual(pipeline.pose_estimator.estimate_pose.call_count, 1, "Pose inference should run when enabled")
        self.assertEqual(pipeline.face_recognizer.recognize_faces_in_frame.call_count, 0, "Face inference should still NOT run")

        pipeline.stop()

    def test_web_server_model_api(self):
        """Tests the WebServer /api/models and /api/models/toggle REST endpoints."""
        from src.web.server import WebServer
        config = {
            "hardware": {"device": "virtual"},
            "cameras": [],
            "models": {
                "human_detector": {"enabled": True, "conf_threshold": 0.45, "input_size": [512, 512]},
                "pose_estimator": {"enabled": True, "conf_threshold": 0.50, "input_size": [512, 512]},
                "face_recognizer": {"enabled": True, "match_threshold": 0.60, "input_size": [112, 112]}
            },
            "face_db": {"path": "data/test_db.json"}
        }
        pipeline = MultiCameraPipeline(config)
        web_server = WebServer(pipeline, host="127.0.0.1", port=8000, use_https=False)

        if web_server.app:
            client = web_server.app.test_client()

            # 1. GET /api/models
            res = client.get('/api/models')
            self.assertEqual(res.status_code, 200)
            data = res.get_json()
            self.assertIn('models', data)
            self.assertEqual(len(data['models']), 3)

            # 2. POST /api/models/toggle -> disable human_detector
            toggle_res = client.post('/api/models/toggle', json={"model_id": "human_detector", "enabled": False})
            self.assertEqual(toggle_res.status_code, 200)
            toggle_data = toggle_res.get_json()
            self.assertTrue(toggle_data['success'])
            self.assertFalse(pipeline.is_model_enabled("human_detector"))

            # 3. GET /api/status contains updated models
            status_res = client.get('/api/status')
            self.assertEqual(status_res.status_code, 200)
            status_data = status_res.get_json()
            self.assertIn('models', status_data)
            det_model = next(m for m in status_data['models'] if m['id'] == 'human_detector')
            self.assertFalse(det_model['enabled'])

            # 4. POST /api/models/toggle -> re-enable human_detector
            client.post('/api/models/toggle', json={"model_id": "human_detector", "enabled": True})
            self.assertTrue(pipeline.is_model_enabled("human_detector"))

    def test_auto_updater_and_web_api(self):
        """Tests AutoUpdater configuration, intervals, git check mocks, and WebServer update endpoints."""
        from unittest.mock import patch
        from src.utils.auto_updater import AutoUpdater, CHECK_INTERVALS_SEC
        from src.web.server import WebServer

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_cfg_path = os.path.join(tmpdir, "test_config.yaml")
            initial_cfg = {
                "hardware": {"device": "virtual"},
                "cameras": [],
                "models": {
                    "human_detector": {"enabled": True, "conf_threshold": 0.45, "input_size": [512, 512]},
                    "pose_estimator": {"enabled": True, "conf_threshold": 0.50, "input_size": [512, 512]},
                    "face_recognizer": {"enabled": True, "match_threshold": 0.60, "input_size": [112, 112]}
                },
                "face_db": {"path": "data/test_db.json"},
                "auto_update": {
                    "enabled": True,
                    "branch": "live",
                    "check_interval": "daily",
                    "install_time": "immediately",
                    "remote": "origin"
                }
            }
            with open(tmp_cfg_path, "w") as f:
                yaml.safe_dump(initial_cfg, f)

            restart_called = []
            def dummy_restart():
                restart_called.append(True)

            updater = AutoUpdater(initial_cfg, config_path=tmp_cfg_path, restart_callback=dummy_restart)
            self.assertTrue(updater.enabled)
            self.assertEqual(updater.branch, "live")
            self.assertEqual(updater.check_interval, "daily")
            self.assertEqual(updater.install_time, "immediately")

            # Test check intervals mapping
            self.assertEqual(CHECK_INTERVALS_SEC["hourly"], 3600)
            self.assertEqual(CHECK_INTERVALS_SEC["daily"], 86400)
            self.assertEqual(CHECK_INTERVALS_SEC["weekly"], 604800)

            # Test update_config
            updater.update_config({
                "enabled": False,
                "check_interval": "weekly",
                "install_time": "03:00",
                "branch": "release-v1"
            })
            self.assertFalse(updater.enabled)
            self.assertEqual(updater.check_interval, "weekly")
            self.assertEqual(updater.install_time, "03:00")
            self.assertEqual(updater.branch, "release-v1")

            # Verify persisted to YAML
            with open(tmp_cfg_path, "r") as f:
                saved_yaml = yaml.safe_load(f)
            self.assertFalse(saved_yaml["auto_update"]["enabled"])
            self.assertEqual(saved_yaml["auto_update"]["check_interval"], "weekly")
            self.assertEqual(saved_yaml["auto_update"]["install_time"], "03:00")
            self.assertEqual(saved_yaml["auto_update"]["branch"], "release-v1")

            # Test get_status
            status = updater.get_status()
            self.assertFalse(status["enabled"])
            self.assertEqual(status["check_interval"], "weekly")
            self.assertEqual(status["install_time"], "03:00")
            self.assertEqual(status["branch"], "release-v1")
            self.assertIn("current_commit", status)

            # Test WebServer REST integration
            pipeline = MultiCameraPipeline(initial_cfg)
            web_server = WebServer(pipeline, auto_updater=updater, host="127.0.0.1", port=8000, use_https=False)
            if web_server.app:
                client = web_server.app.test_client()

                # GET /api/update/status
                res = client.get('/api/update/status')
                self.assertEqual(res.status_code, 200)
                data = res.get_json()
                self.assertEqual(data["branch"], "release-v1")
                self.assertEqual(data["check_interval"], "weekly")

                # POST /api/update/config
                post_res = client.post('/api/update/config', json={
                    "enabled": True,
                    "check_interval": "hourly",
                    "install_time": "02:00"
                })
                self.assertEqual(post_res.status_code, 200)
                self.assertTrue(updater.enabled)
                self.assertEqual(updater.check_interval, "hourly")
                self.assertEqual(updater.install_time, "02:00")

                # POST /api/update/check with mock
                with patch.object(updater, 'check_for_updates', return_value={"success": True, "branch": "release-v1", "update_pending": False}):
                    chk_res = client.post('/api/update/check')
                    self.assertEqual(chk_res.status_code, 200)
                    chk_data = chk_res.get_json()
                    self.assertTrue(chk_data["success"])

                # POST /api/update/install with mock
                with patch.object(updater, 'apply_update_and_restart', return_value={"success": True, "message": "Restarting"}):
                    inst_res = client.post('/api/update/install')
                    self.assertEqual(inst_res.status_code, 200)
                    inst_data = inst_res.get_json()
                    self.assertTrue(inst_data["success"])

            updater.stop()

    def test_ultra_light_detector_and_cadence(self):
        """Tests ultra-light human detection model reload, 320x320 resolution, and cadence optimization."""
        config = {
            "hardware": {"device": "virtual"},
            "cameras": [
                {"id": "cam_01", "name": "Test Cam 1", "source": "synthetic", "enabled": True}
            ],
            "performance": {"detect_interval": 2},
            "models": {
                "human_detector": {
                    "model_profile": "ultra_light",
                    "model_name": "yolo11n.pt",
                    "onnx_path": "models/onnx/yolo11n_320.onnx",
                    "input_size": [320, 320],
                    "conf_threshold": 0.45
                },
                "pose_estimator": {"enabled": False},
                "face_recognizer": {"enabled": False}
            },
            "tracking": {"enabled": True},
            "face_db": {"path": "data/test_db.json"}
        }

        pipeline = MultiCameraPipeline(config)
        pipeline.start()
        self.assertEqual(pipeline.detect_interval, 2)
        self.assertEqual(pipeline.detector.input_size, (320, 320))

        # Test reload_detector_model
        pipeline.reload_detector_model(
            model_profile="ultra_light",
            model_name="yolo11n.pt",
            input_size=[320, 320],
            detect_interval=3
        )
        self.assertEqual(pipeline.detect_interval, 3)
        self.assertEqual(pipeline.detector.input_size, (320, 320))

        # Test WebServer configure_detector REST endpoint
        web_server = WebServer(pipeline, host="127.0.0.1", port=8000, use_https=False)
        if web_server.app:
            client = web_server.app.test_client()
            res = client.post('/api/models/configure_detector', json={
                "profile": "ultra_light",
                "detect_interval": 2
            })
            self.assertEqual(res.status_code, 200)
            data = res.get_json()
            self.assertTrue(data["success"])
            self.assertEqual(data["profile"], "ultra_light")
            self.assertEqual(pipeline.detect_interval, 2)

        pipeline.stop()

if __name__ == "__main__":
    unittest.main()

