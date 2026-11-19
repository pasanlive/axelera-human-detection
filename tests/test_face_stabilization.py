"""
Unit tests for Face Recognizer Stability & Anti-Flickering:
- Face quality gate (min_face_size: 64 px, min_blur_score: 50.0)
- 5-point landmark alignment via cv2.estimateAffinePartial2D against ArcFace 112x112
- Track-level EMA embedding smoothing (alpha: 0.3)
- Dual-threshold hysteresis state transitions (match_high_thresh: 0.55, match_low_thresh: 0.45)
"""

import unittest
import numpy as np
import cv2

from src.analytics.face_align import (
    ARCFACE_REF_LANDMARKS_112,
    align_face_5point,
    check_face_quality,
    compute_blur_score,
)
from src.analytics.identity import TrackIdentityResolver, UNKNOWN
from src.utils.face_db import FaceDatabase


class TestFaceStabilization(unittest.TestCase):

    def test_face_quality_gate_size(self):
        """Verify face crops smaller than min_face_size (64px) are rejected."""
        # 50x50 crop -> too small
        small_crop = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        ok, reason, _ = check_face_quality(small_crop, min_face_size=64, min_blur_score=10.0)
        self.assertFalse(ok)
        self.assertIn("size_too_small", reason)

        # 80x80 sharp crop -> passes size
        sharp_crop = np.random.randint(0, 255, (80, 80, 3), dtype=np.uint8)
        ok_sharp, _, _ = check_face_quality(sharp_crop, min_face_size=64, min_blur_score=10.0)
        self.assertTrue(ok_sharp)

    def test_face_quality_gate_blur(self):
        """Verify blurry face crops (Laplacian var < 50.0) are rejected."""
        # Highly blurred image
        sharp_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        blurry_crop = cv2.GaussianBlur(sharp_img, (25, 25), 10.0)

        ok, reason, blur_score = check_face_quality(blurry_crop, min_face_size=64, min_blur_score=50.0)
        self.assertFalse(ok)
        self.assertIn("blurry", reason)
        self.assertLess(blur_score, 50.0)

    def test_5point_arcface_alignment(self):
        """Verify affine similarity alignment transforms landmarks to canonical 112x112 coordinates."""
        # Create a synthetic face image
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        # Sligthly perturbed landmarks
        src_landmarks = ARCFACE_REF_LANDMARKS_112 + np.array([20.0, 30.0], dtype=np.float32)

        aligned = align_face_5point(img, src_landmarks, output_size=(112, 112))
        self.assertEqual(aligned.shape, (112, 112, 3))

    def test_track_level_ema_embedding_smoothing(self):
        """Verify EMA vector smoothing updates track embedding with L2 normalization."""
        resolver = TrackIdentityResolver(ema_alpha=0.3)
        cam = "cam_01"
        tid = 10
        t0 = 100.0

        # Feed embedding 1
        np.random.seed(123)
        v1 = np.random.randn(512).astype(np.float32)
        v1 /= np.linalg.norm(v1)

        resolver.update(cam, tid, "PersonA", 0.7, now=t0, embedding=v1)
        st = resolver._state[(cam, tid)]
        self.assertIsNotNone(st["smoothed_embedding"])
        np.testing.assert_allclose(st["smoothed_embedding"], v1, atol=1e-5)

        # Feed embedding 2
        v2 = np.random.randn(512).astype(np.float32)
        v2 /= np.linalg.norm(v2)

        resolver.update(cam, tid, "PersonA", 0.7, now=t0 + 0.1, embedding=v2)
        smoothed = st["smoothed_embedding"]

        # Expected EMA: normalize((1 - 0.3) * v1 + 0.3 * v2)
        expected = 0.7 * v1 + 0.3 * v2
        expected /= np.linalg.norm(expected)
        np.testing.assert_allclose(smoothed, expected, atol=1e-5)
        self.assertAlmostEqual(float(np.linalg.norm(smoothed)), 1.0, places=5)

    def test_hysteresis_state_transitions(self):
        """
        Verify dual-threshold hysteresis:
        - match_high_thresh (0.55) required to establish new identity.
        - match_low_thresh (0.45) sufficient to maintain locked identity.
        """
        resolver = TrackIdentityResolver(
            match_high_thresh=0.55,
            match_low_thresh=0.45,
            min_votes=0.5,
        )
        cam = "cam_01"
        tid = 5
        t0 = 200.0

        # Step 1: Initial observation has similarity 0.50 (< match_high_thresh 0.55)
        # Should NOT establish identity (remains Unknown)
        name1, _ = resolver.update(cam, tid, "Alice", 0.50, now=t0)
        self.assertEqual(name1, UNKNOWN)

        # Step 2: High confidence observation (0.75 >= 0.55)
        # Locks identity to Alice
        name2, _ = resolver.update(cam, tid, "Alice", 0.75, now=t0 + 0.1)
        self.assertEqual(name2, "Alice")

        # Step 3: Head turn / lighting drop (similarity 0.48 >= match_low_thresh 0.45)
        # Maintains locked Alice identity despite being below high threshold!
        name3, sim3 = resolver.update(cam, tid, "Alice", 0.48, now=t0 + 0.2)
        self.assertEqual(name3, "Alice")
        self.assertEqual(sim3, 0.48)

        # Step 4: Challenging identity with insufficient margin or low score -> does not flip
        name4, _ = resolver.update(cam, tid, "Bob", 0.50, now=t0 + 0.3)
        self.assertEqual(name4, "Alice")

        # Step 5: Purge stale track
        resolver.purge(cam, [], now=t0 + 20.0)
        self.assertIsNone(resolver.get(cam, tid))


if __name__ == "__main__":
    unittest.main()
