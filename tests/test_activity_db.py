"""
Unit tests for ActivityDatabase and WebServer activity endpoints.
"""

import json
import os
import shutil
import tempfile
import time
import unittest
import numpy as np

from src.utils.activity_db import ActivityDatabase
from src.web.server import WebServer


class TestActivityDatabase(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmp_dir, "test_activities.db")
        self.activity_db = ActivityDatabase(db_path=self.db_path, incident_timeout_s=1.0)

    def tearDown(self):
        self.activity_db.close()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_init_and_schema(self):
        """Verify database and activities table are properly created."""
        self.assertTrue(os.path.exists(self.db_path))
        stats = self.activity_db.get_stats()
        self.assertEqual(stats["total_activities"], 0)
        self.assertEqual(stats["face_incidents"], 0)
        self.assertEqual(stats["plate_incidents"], 0)

    def test_encode_thumb_and_crop(self):
        """Verify image encoding and bounding box cropping utilities."""
        dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
        dummy_img[20:60, 20:60] = [255, 0, 0]

        # Crop ROI
        crop = self.activity_db.crop_roi(dummy_img, [0.2, 0.2, 0.6, 0.6], margin_pct=0.0)
        self.assertIsNotNone(crop)
        self.assertEqual(crop.shape[:2], (40, 40))

        # Encode thumbnail
        thumb = self.activity_db.encode_thumb(crop, max_dim=32)
        self.assertIsNotNone(thumb)
        self.assertTrue(thumb.startswith("data:image/jpeg;base64,"))

        # Test invalid inputs
        self.assertIsNone(self.activity_db.encode_thumb(None))
        self.assertIsNone(self.activity_db.crop_roi(None, [0, 0, 10, 10]))
        self.assertIsNone(self.activity_db.crop_roi(dummy_img, []))

    def test_record_single_incident_flush(self):
        """Verify single frame detection starts incident and flushes correctly."""
        t0 = 1700000000.0
        frame = np.zeros((240, 320, 3), dtype=np.uint8)

        persons = [{"track_id": 1, "box": [10, 10, 50, 100]}]
        faces = [{"name": "Alice", "similarity": 0.88, "bbox": [15, 15, 30, 35]}]
        plates = [{"plate_text": "ABC-1234", "confidence": 0.95, "bbox": [100, 150, 180, 180]}]

        self.activity_db.record_frame_activity(
            camera_id="cam_01",
            now=t0,
            persons=persons,
            faces=faces,
            plates=plates,
            get_frame=lambda: frame,
        )

        # Before timeout, incident is active in memory
        stats_pre = self.activity_db.get_stats()
        self.assertEqual(stats_pre["active_in_progress"], 1)

        # Force flush / simulate passage of time past incident_timeout_s (1.0s)
        self.activity_db._periodic_flush(now=t0 + 2.0, force=False)

        activities = self.activity_db.get_activities(limit=10)
        self.assertEqual(len(activities), 1)

        act = activities[0]
        self.assertEqual(act["primary_camera_id"], "cam_01")
        self.assertIn("cam_01", act["camera_ids"])
        self.assertIn("human", act["detection_type"])
        self.assertIn("face", act["detection_type"])
        self.assertIn("vehicle_plate", act["detection_type"])
        self.assertEqual(act["num_humans"], 1)
        self.assertIn("Alice", act["identified_faces"])
        self.assertIn("ABC-1234", act["identified_plates"])
        self.assertIn("Alice", act["summary"])
        self.assertIn("ABC-1234", act["summary"])

        # Check evidence structure
        ev = act["evidence"]
        self.assertIsNotNone(ev.get("screenshot"))
        self.assertEqual(len(ev.get("faces", [])), 1)
        self.assertEqual(ev["faces"][0]["name"], "Alice")
        self.assertEqual(len(ev.get("plates", [])), 1)
        self.assertEqual(ev["plates"][0]["text"], "ABC-1234")

    def test_incident_continuation_and_duration(self):
        """Verify continuous detections extend incident and correctly calculate duration."""
        t0 = 1700000000.0
        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        # First observation at t0
        self.activity_db.record_frame_activity(
            camera_id="cam_front",
            now=t0,
            persons=[{"track_id": 5}],
            faces=[],
            plates=[],
            get_frame=lambda: frame,
        )

        # Continuation at t0 + 0.4s
        self.activity_db.record_frame_activity(
            camera_id="cam_front",
            now=t0 + 0.4,
            persons=[{"track_id": 5}, {"track_id": 6}],
            faces=[{"name": "Bob", "similarity": 0.82}],
            plates=[],
            get_frame=lambda: frame,
        )

        # Final observation in same sequence at t0 + 0.8s
        self.activity_db.record_frame_activity(
            camera_id="cam_front",
            now=t0 + 0.8,
            persons=[{"track_id": 6}],
            faces=[{"name": "Bob", "similarity": 0.85}],
            plates=[{"plate_text": "K-9900", "confidence": 0.9}],
            get_frame=lambda: frame,
        )

        # Flush completed incident after timeout
        self.activity_db._periodic_flush(now=t0 + 3.0, force=True)

        activities = self.activity_db.get_activities()
        self.assertEqual(len(activities), 1)

        act = activities[0]
        self.assertEqual(act["first_seen"], t0)
        self.assertEqual(act["last_seen"], t0 + 0.8)
        self.assertAlmostEqual(act["duration_s"], 0.8, places=1)
        self.assertEqual(act["num_humans"], 2)  # Max concurrent humans observed
        self.assertIn("Bob", act["identified_faces"])
        self.assertIn("K-9900", act["identified_plates"])

    def test_activity_filters(self):
        """Verify get_activities filter conditions."""
        t0 = 1700000000.0
        # Incident 1 on cam_01 (human only)
        self.activity_db.record_frame_activity(
            camera_id="cam_01",
            now=t0,
            persons=[{"track_id": 1}],
            faces=[],
            plates=[],
        )
        self.activity_db._periodic_flush(now=t0 + 5.0, force=True)

        # Incident 2 on cam_02 (plate only)
        self.activity_db.record_frame_activity(
            camera_id="cam_02",
            now=t0 + 10.0,
            persons=[],
            faces=[],
            plates=[{"plate_text": "XYZ-999", "confidence": 0.99}],
        )
        self.activity_db._periodic_flush(now=t0 + 15.0, force=True)

        # Filter by camera
        cam1_res = self.activity_db.get_activities(camera_id="cam_01")
        self.assertEqual(len(cam1_res), 1)
        self.assertEqual(cam1_res[0]["primary_camera_id"], "cam_01")

        # Filter by type
        plate_res = self.activity_db.get_activities(detection_type="vehicle_plate")
        self.assertEqual(len(plate_res), 1)
        self.assertIn("XYZ-999", plate_res[0]["identified_plates"])

        # Filter by search keyword
        search_res = self.activity_db.get_activities(search="XYZ")
        self.assertEqual(len(search_res), 1)
        self.assertEqual(search_res[0]["primary_camera_id"], "cam_02")

        # Clear
        self.activity_db.clear()
        self.assertEqual(len(self.activity_db.get_activities()), 0)


class MockPipelineForActivity:
    def __init__(self, activity_db):
        self.activity_db = activity_db
        self.cameras = {}
        self.camera_channels = {}
        self.face_db = None
        self.face_history_db = None
        self.plate_db = None
        self.capacity = None
        self.visualizer = None


class TestWebServerActivityAPI(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmp_dir, "api_activities.db")
        self.activity_db = ActivityDatabase(db_path=self.db_path, incident_timeout_s=1.0)
        self.pipeline = MockPipelineForActivity(self.activity_db)
        self.server = WebServer(self.pipeline, port=8999)

    def tearDown(self):
        self.activity_db.close()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_api_activities_endpoints(self):
        """Test GET /api/activities, GET /api/activities/stats, and POST /api/activities/clear."""
        # Seed an incident
        t0 = 1700000000.0
        self.activity_db.record_frame_activity(
            camera_id="cam_gate",
            now=t0,
            persons=[{"track_id": 101}],
            faces=[{"name": "SecurityOfficer", "similarity": 0.91}],
            plates=[{"plate_text": "PATROL-1", "confidence": 0.98}],
        )
        self.activity_db._periodic_flush(now=t0 + 2.0, force=True)

        # GET /api/activities
        status, resp = self.server.handle_api("GET", "/api/activities", {}, {})
        self.assertEqual(status, 200)
        self.assertIn("activities", resp)
        self.assertEqual(resp["count"], 1)
        self.assertEqual(resp["activities"][0]["primary_camera_id"], "cam_gate")

        # GET /api/activities/stats
        status, stats_resp = self.server.handle_api("GET", "/api/activities/stats", {}, {})
        self.assertEqual(status, 200)
        self.assertEqual(stats_resp["total_activities"], 1)
        self.assertEqual(stats_resp["face_incidents"], 1)
        self.assertEqual(stats_resp["plate_incidents"], 1)

        # POST /api/activities/clear
        status, clear_resp = self.server.handle_api("POST", "/api/activities/clear", {}, {})
        self.assertEqual(status, 200)
        self.assertTrue(clear_resp["success"])

        # Check empty
        status, resp_empty = self.server.handle_api("GET", "/api/activities", {}, {})
        self.assertEqual(resp_empty["count"], 0)


if __name__ == "__main__":
    unittest.main()
