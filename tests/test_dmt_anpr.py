"""
Unit tests for Sri Lankan DMT ANPR Pipeline:
- Perspective deskew and tight padding
- 2-line square plate splitting and side-by-side stitching
- DMT character cleanup, province prefixes, and positional letter/digit correction
- DMT regex validation and invalid string rejection
- Multi-frame temporal consensus voting
"""

import os
import shutil
import tempfile
import unittest
import numpy as np
import cv2

from src.analytics.plate_dmt import (
    SRI_LANKA_PROVINCES,
    DMT_PLATE_REGEX,
    clean_and_correct_dmt_plate,
    deskew_plate_crop,
    split_and_stitch_two_line_plate,
    preprocess_plate_crop,
)
from src.utils.plate_db import PlateDatabase


class TestSriLankanDMTPlateANPR(unittest.TestCase):

    def test_provinces_set(self):
        """Verify all 9 official Sri Lankan provinces are supported."""
        expected = {"WP", "CP", "SP", "NW", "NC", "UP", "SG", "EP", "NP"}
        self.assertEqual(SRI_LANKA_PROVINCES, expected)

    def test_perspective_deskew(self):
        """Verify perspective deskew/homography transforms skewed quad into rectified rectangle."""
        # Create a synthetic image with a white rectangle on black background
        img = np.zeros((200, 400, 3), dtype=np.uint8)
        corners = np.array([[30, 40], [350, 20], [370, 150], [40, 170]], dtype=np.float32)
        cv2.fillPoly(img, [corners.astype(np.int32)], (255, 255, 255))

        deskewed = deskew_plate_crop(img, corners)
        self.assertIsNotNone(deskewed)
        self.assertGreater(deskewed.shape[0], 50)
        self.assertGreater(deskewed.shape[1], 150)

    def test_two_line_square_plate_split_and_stitch(self):
        """Verify 2-line square plate (aspect ratio < 2.0) is split and concatenated horizontally."""
        # Square plate: 150 height, 180 width -> aspect ratio 1.2 (< 2.0)
        square_crop = np.zeros((150, 180, 3), dtype=np.uint8)
        # Top half red, bottom half blue
        square_crop[:75, :] = (0, 0, 255)
        square_crop[75:, :] = (255, 0, 0)

        stitched = split_and_stitch_two_line_plate(square_crop, two_line_aspect_thresh=2.0, target_height=48)
        self.assertEqual(stitched.shape[0], 48)
        # Stitched width should be approximately sum of resized strip widths (> 180)
        self.assertGreater(stitched.shape[1], square_crop.shape[1])
        # Single-line rectangular plate should not be split
        rect_crop = np.zeros((50, 250, 3), dtype=np.uint8)
        res_rect = split_and_stitch_two_line_plate(rect_crop, two_line_aspect_thresh=2.0)
        self.assertEqual(res_rect.shape, rect_crop.shape)

    def test_dmt_clean_and_correct_valid_plates(self):
        """Test standard Sri Lankan plates (with and without province)."""
        # Modern 3-letter plates
        res1 = clean_and_correct_dmt_plate("WP CAA-1234")
        self.assertIsNotNone(res1)
        self.assertEqual(res1[0], "WP CAA-1234")

        res2 = clean_and_correct_dmt_plate("CAD-5678")
        self.assertIsNotNone(res2)
        self.assertEqual(res2[0], "CAD-5678")

        # 2-letter vintage / commercial plates
        res3 = clean_and_correct_dmt_plate("SP GA-5678")
        self.assertIsNotNone(res3)
        self.assertEqual(res3[0], "SP GA-5678")

        res4 = clean_and_correct_dmt_plate("CP-HA-9012")
        self.assertIsNotNone(res4)
        self.assertEqual(res4[0], "CP HA-9012")

    def test_dmt_positional_character_correction(self):
        """Verify positional digit/letter confusion corrections."""
        # Last 4 characters must be digits (O/Q -> 0, I/L -> 1, Z -> 2, S -> 5, B -> 8, G -> 6)
        res_digits = clean_and_correct_dmt_plate("WP CAA-OIZS")
        self.assertIsNotNone(res_digits)
        self.assertEqual(res_digits[0], "WP CAA-0125")

        # Preceding letters must be letters (0 -> O, 1 -> I, 2 -> Z, 5 -> S, 8 -> B)
        res_letters = clean_and_correct_dmt_plate("WP 012-1234")
        self.assertIsNotNone(res_letters)
        self.assertEqual(res_letters[0], "WP OIZ-1234")

        # Combined noise: "CP-8G-OIZS" -> 8G -> BG (letters), OIZS -> 0125 (digits)
        res_combo = clean_and_correct_dmt_plate("CP-86-OIZS")
        self.assertIsNotNone(res_combo)
        self.assertEqual(res_combo[0], "CP BG-0125")

    def test_dmt_invalid_strings_rejected(self):
        """Verify invalid or garbage reads are rejected."""
        # Too short
        self.assertIsNone(clean_and_correct_dmt_plate("WP-1"))
        # Invalid province code
        self.assertIsNone(clean_and_correct_dmt_plate("ZZ CAA-1234"))
        # Garbage text
        self.assertIsNone(clean_and_correct_dmt_plate("GARBAGE1234567"))
        self.assertIsNone(clean_and_correct_dmt_plate("???---"))

    def test_multi_frame_consensus_voting_dmt(self):
        """Verify temporal consensus voting in PlateDatabase produces validated DMT string."""
        tmp_dir = tempfile.mkdtemp()
        db_path = os.path.join(tmp_dir, "test_dmt_plates.json")
        try:
            db = PlateDatabase(db_path=db_path, flush_interval_s=1.0)
            cam = "cam_gate"
            t0 = 100.0

            # Feed noisy reads of the same car:
            # 1. "WP CA0-1234" (0 in letters)
            # 2. "WP CAA-I234" (I in digits)
            # 3. "WP CAA-1234" (clean)
            f1, _ = db.fuse_plate(cam, "WP CA0-1234", 0.85, now=t0, sri_lanka_dmt=True)
            f2, _ = db.fuse_plate(cam, "WP CAA-I234", 0.90, now=t0 + 0.1, sri_lanka_dmt=True)
            f3, conf3 = db.fuse_plate(cam, "WP CAA-1234", 0.95, now=t0 + 0.2, sri_lanka_dmt=True)

            self.assertEqual(f3, "WP CAA-1234")
            self.assertGreater(conf3, 0.8)
            self.assertEqual(db.get_vote_count(cam), 3)
            db.close()
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
