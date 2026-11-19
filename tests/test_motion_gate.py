"""
Unit tests for MotionGate (MOG2 Background Subtraction) and Person Confidence Gating:
1. Detections below 0.55 confidence threshold are dropped.
2. A static bounding box on an unchanging background is never confirmed.
3. A moving bounding box is confirmed once motion_ratio >= 0.03 and stays confirmed even after motion stops.
4. Expired track_ids are cleaned up from confirmed_tracks.
5. Per-camera stream isolation (background models and confirmed tracks do not cross).
"""

import unittest
import numpy as np
import cv2

from src.analytics.motion_gate import MotionGate
from src.analytics.human_filter import HumanDetectionFilter


class TestMotionGate(unittest.TestCase):

    def test_confidence_threshold_gating(self):
        """Verify detections with confidence below 0.55 are dropped."""
        hf = HumanDetectionFilter(
            conf_threshold=0.55,
            min_consecutive_frames=1,
            motion_gate_enabled=False,
            min_box_area=100.0,
            min_height=20.0,
        )
        cam = "cam_test"

        low_conf_det = {
            "track_id": 1,
            "bbox": [100.0, 100.0, 160.0, 250.0],
            "confidence": 0.52,  # < 0.55 -> drop
        }
        high_conf_det = {
            "track_id": 2,
            "bbox": [100.0, 100.0, 160.0, 250.0],
            "confidence": 0.65,  # >= 0.55 -> keep
        }

        # 1. Low confidence detection is dropped
        res_low = hf.filter_persons(cam, [low_conf_det], (1920, 1080), now=100.0)
        self.assertEqual(len(res_low), 0)

        # 2. High confidence detection survives
        res_high = hf.filter_persons(cam, [high_conf_det], (1920, 1080), now=100.0)
        self.assertEqual(len(res_high), 1)
        self.assertEqual(res_high[0]["track_id"], 2)

    def test_static_bounding_box_never_confirmed(self):
        """Verify a static bounding box on an unchanging background is never confirmed."""
        gate = MotionGate(
            min_motion_ratio=0.03,
            motion_warmup_frames=5,
            track_ttl_s=5.0,
        )
        # Background is a constant gray image
        bg_frame = np.ones((480, 640, 3), dtype=np.uint8) * 120
        bbox = (100, 100, 200, 300)  # static object / cupboard
        tid = 101

        # Feed 15 unchanging frames (past the 5 warmup frames)
        for i in range(15):
            gate.update(bg_frame)
            confirmed = gate.check_track(tid, bbox, 640, 480, now=100.0 + i * 0.1)
            # Must never confirm a completely static object
            self.assertFalse(confirmed)
            self.assertNotIn(tid, gate.confirmed_tracks)

    def test_moving_bounding_box_confirmed_and_persists_when_stopped(self):
        """
        Verify a moving bounding box is confirmed once motion_ratio >= 0.03,
        and stays confirmed even after the person stops moving.
        """
        gate = MotionGate(
            min_motion_ratio=0.03,
            motion_warmup_frames=5,
            track_ttl_s=5.0,
        )
        base_frame = np.ones((480, 640, 3), dtype=np.uint8) * 50
        bbox = (100, 100, 200, 300)
        tid = 42

        # 1. Warmup period (5 frames)
        for i in range(5):
            gate.update(base_frame)
            gate.check_track(tid, bbox, 640, 480, now=100.0 + i * 0.1)

        self.assertTrue(gate.is_warmed_up())

        # 2. Introduce movement inside the bounding box (moving person)
        moving_frame = base_frame.copy()
        # Change pixel intensity inside ROI to create substantial foreground motion (> 3%)
        moving_frame[110:250, 110:190] = 230
        gate.update(moving_frame)

        is_confirmed = gate.check_track(tid, bbox, 640, 480, now=100.6)
        self.assertTrue(is_confirmed)
        self.assertIn(tid, gate.confirmed_tracks)

        # 3. Person stops moving completely (frames revert to static moving_frame)
        # Run 10 subsequent frames where no new pixels move
        for i in range(10):
            gate.update(moving_frame)
            # Even if motion inside bbox is now 0, track stays confirmed!
            still_confirmed = gate.check_track(tid, bbox, 640, 480, now=101.0 + i * 0.1)
            self.assertTrue(still_confirmed)
            self.assertIn(tid, gate.confirmed_tracks)

    def test_expired_tracks_cleaned_up(self):
        """Verify expired track_ids are pruned from confirmed_tracks upon TTL expiration."""
        gate = MotionGate(
            min_motion_ratio=0.03,
            motion_warmup_frames=1,
            track_ttl_s=2.0,
        )
        frame1 = np.ones((480, 640, 3), dtype=np.uint8) * 40
        gate.update(frame1)

        # Force confirm track 55
        frame2 = frame1.copy()
        frame2[100:200, 100:200] = 255
        gate.update(frame2)
        gate.check_track(55, (100, 100, 200, 200), 640, 480, now=10.0)
        self.assertIn(55, gate.confirmed_tracks)

        # Active tracks include 55 within TTL -> retained
        gate.prune_stale_tracks(active_track_ids=[55], now=11.0)
        self.assertIn(55, gate.confirmed_tracks)

        # Track 55 is not active and time exceeds TTL (2.0s):
        gate.prune_stale_tracks(active_track_ids=[], now=13.5)
        self.assertNotIn(55, gate.confirmed_tracks)

    def test_per_stream_isolation(self):
        """Verify camera streams have separate MotionGate instances without cross-talk."""
        hf = HumanDetectionFilter(
            conf_threshold=0.55,
            min_consecutive_frames=1,
            motion_gate_enabled=True,
            motion_warmup_frames=2,
            min_motion_ratio=0.03,
            min_box_area=100.0,
            min_height=20.0,
        )
        cam1 = "Living_Room_Cam"
        cam2 = "RPi_Cam"

        gate1 = hf.get_motion_gate(cam1)
        gate2 = hf.get_motion_gate(cam2)
        self.assertIsNot(gate1, gate2)

        # Confirm track 10 on cam1
        f_base = np.zeros((300, 300, 3), dtype=np.uint8)
        gate1.update(f_base)
        gate1.update(f_base)
        f_move = f_base.copy()
        f_move[50:150, 50:150] = 255
        gate1.update(f_move)
        ok1 = gate1.check_track(10, (50, 50, 150, 150), 300, 300, now=5.0)
        self.assertTrue(ok1)

        # cam2 should NOT have track 10 confirmed
        self.assertIn(10, gate1.confirmed_tracks)
        self.assertNotIn(10, gate2.confirmed_tracks)


if __name__ == "__main__":
    unittest.main()
