"""
Unit tests for Human Detection False Positive Reduction Filter:
- Bounding box geometry filtering (min_box_area, min_height, max_aspect_ratio)
- Exclusion zone polygon filtering (bottom-center footpoint test)
- Tracker temporal persistence gating (min_consecutive_frames)
- Hard-negative logging when enabled
"""

import os
import shutil
import tempfile
import unittest
import numpy as np

from src.analytics.human_filter import HumanDetectionFilter


class TestHumanFiltering(unittest.TestCase):

    def test_geometry_filtering(self):
        """Verify geometry checks reject wide, tiny, or sliver boxes."""
        hf = HumanDetectionFilter(
            min_box_area=1000.0,
            min_height=40.0,
            max_aspect_ratio=1.5,
            min_aspect_ratio=0.1,
            min_consecutive_frames=1,
        )

        # 1. Normal upright human (w=60, h=150, area=9000, aspect=0.4) -> Valid
        box_valid = (100.0, 100.0, 160.0, 250.0)
        ok, reason = hf.check_geometry(box_valid)
        self.assertTrue(ok)
        self.assertIsNone(reason)

        # 2. Wide box / shadow (w=200, h=80, aspect=2.5 > 1.5) -> Rejected
        box_wide = (100.0, 100.0, 300.0, 180.0)
        ok, reason = hf.check_geometry(box_wide)
        self.assertFalse(ok)
        self.assertIn("aspect_ratio_too_wide", reason)

        # 3. Tiny noise box (w=15, h=20, area=300 < 1000) -> Rejected
        box_tiny = (100.0, 100.0, 115.0, 120.0)
        ok, reason = hf.check_geometry(box_tiny)
        self.assertFalse(ok)
        self.assertIn("area_too_small", reason)

        # 4. Short box (w=50, h=30, h < 40) -> Rejected
        box_short = (100.0, 100.0, 150.0, 130.0)
        ok, reason = hf.check_geometry(box_short)
        self.assertFalse(ok)
        self.assertIn("height_too_small", reason)

    def test_exclusion_zone_filtering(self):
        """Verify polygon exclusion zones drop detections based on bottom-center footpoint."""
        # Define a square exclusion zone on cam_01: x from 200 to 400, y from 300 to 500
        exclusion_zones = {
            "cam_01": [
                [[200.0, 300.0], [400.0, 300.0], [400.0, 500.0], [200.0, 500.0]]
            ]
        }
        hf = HumanDetectionFilter(
            exclusion_zones=exclusion_zones,
            min_consecutive_frames=1,
        )

        # Person whose footpoint (center-x, y2) falls inside [300, 450] -> Inside zone
        # x1=250, x2=350 (cx=300), y1=200, y2=450
        box_inside = (250.0, 200.0, 350.0, 450.0)
        self.assertTrue(hf.is_in_exclusion_zone("cam_01", box_inside, 1920, 1080))

        # Person whose footpoint is outside zone -> Not in zone
        box_outside = (50.0, 50.0, 120.0, 200.0)
        self.assertFalse(hf.is_in_exclusion_zone("cam_01", box_outside, 1920, 1080))

        # Other camera with no configured zones -> Not in zone
        self.assertFalse(hf.is_in_exclusion_zone("cam_02", box_inside, 1920, 1080))

    def test_tracker_consecutive_frame_gating(self):
        """Verify detections are only confirmed once continuously detected for min_consecutive_frames (4)."""
        hf = HumanDetectionFilter(
            min_consecutive_frames=4,
            min_box_area=500.0,
            min_height=30.0,
            max_aspect_ratio=1.5,
        )
        cam = "cam_01"
        t0 = 100.0

        person_track = {
            "track_id": 42,
            "bbox": [100.0, 100.0, 160.0, 250.0],
            "confidence": 0.9,
        }

        # Frame 1: Hits = 1 (< 4) -> Filtered out (not confirmed)
        out1 = hf.filter_persons(cam, [dict(person_track)], (1920, 1080), now=t0)
        self.assertEqual(len(out1), 0)

        # Frame 2: Hits = 2 (< 4) -> Filtered out
        out2 = hf.filter_persons(cam, [dict(person_track)], (1920, 1080), now=t0 + 0.1)
        self.assertEqual(len(out2), 0)

        # Frame 3: Hits = 3 (< 4) -> Filtered out
        out3 = hf.filter_persons(cam, [dict(person_track)], (1920, 1080), now=t0 + 0.2)
        self.assertEqual(len(out3), 0)

        # Frame 4: Hits = 4 (>= 4) -> Confirmed!
        out4 = hf.filter_persons(cam, [dict(person_track)], (1920, 1080), now=t0 + 0.3)
        self.assertEqual(len(out4), 1)
        self.assertEqual(out4[0]["track_id"], 42)
        self.assertEqual(out4[0]["consecutive_hits"], 4)

        # Frame 5: Hits = 5 -> Still confirmed
        out5 = hf.filter_persons(cam, [dict(person_track)], (1920, 1080), now=t0 + 0.4)
        self.assertEqual(len(out5), 1)
        self.assertEqual(out5[0]["consecutive_hits"], 5)

    def test_hard_negative_logging_toggle(self):
        """Verify crops of rejected detections are saved when log_rejected_detections is True."""
        tmp_dir = tempfile.mkdtemp()
        try:
            hf = HumanDetectionFilter(
                min_box_area=2000.0,  # enforce rejection
                log_rejected_detections=True,
                rejected_dir=tmp_dir,
            )
            # Synthetic frame
            frame = np.ones((400, 400, 3), dtype=np.uint8) * 128
            small_person = {
                "track_id": 99,
                "bbox": [50.0, 50.0, 80.0, 90.0],  # area = 1200 < 2000
                "confidence": 0.8,
            }

            out = hf.filter_persons("cam_test", [small_person], (400, 400), now=200.0, get_frame=lambda: frame)
            self.assertEqual(len(out), 0)

            # Check that a crop was saved to tmp_dir
            saved_files = os.listdir(tmp_dir)
            self.assertGreater(len(saved_files), 0)
            self.assertTrue(any(f.endswith(".jpg") for f in saved_files))
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
