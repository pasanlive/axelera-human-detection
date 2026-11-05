"""
Unit tests for RAM optimization components:
1. PlateDatabase in-memory dirty buffer, debounced disk flush, and temporal character-level fusion.
2. TrackIdentityResolver temporal face embedding feature bank and multi-sample gallery fusion.
3. FaceHistoryDatabase in-memory debounce and SQLite WAL PRAGMAs.
"""

import os
import shutil
import tempfile
import time
import unittest
import numpy as np

from src.analytics.identity import TrackIdentityResolver, UNKNOWN
from src.utils.face_db import FaceDatabase
from src.utils.face_history_db import FaceHistoryDatabase
from src.utils.plate_db import PlateDatabase


class TestPlateDatabaseRAM(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmp_dir, "test_plates.json")
        self.plate_db = PlateDatabase(db_path=self.db_path, max_events=100, flush_interval_s=1.0)

    def tearDown(self):
        self.plate_db.close()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_in_memory_logging_and_debounced_save(self):
        """Verify plate logging updates RAM immediately while debouncing disk I/O."""
        self.plate_db.log_plate("K-1234", "cam_01", confidence=0.92, ocr_confidence=0.95, now=100.0)
        self.assertEqual(len(self.plate_db.events), 1)
        self.assertIn("K-1234", self.plate_db.unique_plates)
        self.assertTrue(self.plate_db._dirty)

        # Before interval expires, disk file is not yet flushed (or clean on initial save)
        # Flush explicitly / test close
        self.plate_db.save(force=True)
        self.assertFalse(self.plate_db._dirty)
        self.assertTrue(os.path.exists(self.db_path))

    def test_temporal_character_voting(self):
        """Verify multi-frame character-level consensus in RAM resolves OCR flicker."""
        cam = "cam_gate"
        t0 = 200.0

        # Feed noisy observations over 0.5s:
        # 'AB0-999' (conf 0.70)
        # 'ABC-999' (conf 0.92)
        # 'ABC-999' (conf 0.95)
        # 'ABC-990' (conf 0.60)
        f1, c1 = self.plate_db.fuse_plate(cam, "AB0-999", 0.70, now=t0)
        self.assertEqual(f1, "AB0-999")  # First observation

        f2, c2 = self.plate_db.fuse_plate(cam, "ABC-999", 0.92, now=t0 + 0.1)
        f3, c3 = self.plate_db.fuse_plate(cam, "ABC-999", 0.95, now=t0 + 0.2)
        f4, c4 = self.plate_db.fuse_plate(cam, "ABC-990", 0.60, now=t0 + 0.3)

        # Winner should be ABC-999 because:
        # pos 2: 'C' has 0.92 + 0.95 = 1.87 vs '0' has 0.70
        # pos 6: '9' has 0.70 + 0.92 + 0.95 = 2.57 vs '0' has 0.60
        self.assertEqual(f4, "ABC-999")
        self.assertGreater(c4, 0.75)


class TestIdentityFeatureBank(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.face_db_path = os.path.join(self.tmp_dir, "face_db.json")
        self.face_db = FaceDatabase(db_path=self.face_db_path, quiet=True)

        # Enroll Alice with an arbitrary 512-dim unit vector
        np.random.seed(42)
        v_alice = np.random.randn(512).astype(np.float32)
        v_alice /= np.linalg.norm(v_alice)
        self.face_db.identities["Alice"] = [v_alice]
        self.face_db._rebuild_index()
        self.v_alice = v_alice

        self.resolver = TrackIdentityResolver(min_votes=0.8, max_bank_size=5)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_feature_bank_and_temporal_fusion(self):
        """Verify track feature bank in RAM fuses temporal embeddings against face gallery."""
        cam = "cam_01"
        tid = 7
        t0 = 500.0

        # Step 1: Enrolled face seen clearly (high similarity 0.85) -> stores in feature bank
        name, sim = self.resolver.update(
            cam_id=cam,
            track_id=tid,
            name="Alice",
            similarity=0.85,
            now=t0,
            embedding=self.v_alice,
            face_db=self.face_db,
        )
        self.assertEqual(name, "Alice")
        st = self.resolver._state.get((cam, tid))
        self.assertIsNotNone(st)
        self.assertEqual(len(st["feature_bank"]), 1)

        # Step 2: Next frame, head turned, single frame detector returns Unknown (sim 0.35)
        # Slightly perturbed vector representing head turn
        v_noisy = self.v_alice + 0.1 * np.random.randn(512).astype(np.float32)
        v_noisy /= np.linalg.norm(v_noisy)

        name2, sim2 = self.resolver.update(
            cam_id=cam,
            track_id=tid,
            name=UNKNOWN,
            similarity=0.35,
            now=t0 + 0.1,
            embedding=v_noisy,
            face_db=self.face_db,
        )

        # Temporal feature bank fusion rescues Alice from being dropped to Unknown!
        self.assertEqual(name2, "Alice")
        self.assertGreater(sim2, 0.70)


class TestFaceHistoryRAM(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmp_dir, "face_hist.db")
        self.hist_db = FaceHistoryDatabase(db_path=self.db_path, cooldown_s=10.0)

    def tearDown(self):
        self.hist_db.close()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_in_memory_cooldown_debounce(self):
        """Verify repeated sightings within cooldown update RAM active state without redundant SQL transactions."""
        t0 = 1000.0

        # Frame 1: New sighting -> inserted
        is_new, count, rec = self.hist_db.log_sighting(
            camera_id="cam_01",
            name="Bob",
            is_identified=True,
            similarity=0.88,
            confidence=0.90,
            track_id=1,
            timestamp=t0,
        )
        self.assertTrue(is_new)
        self.assertEqual(count, 1)

        # Frame 2 to 10 within 0.5s -> fast in-memory debounce
        for i in range(2, 11):
            is_new_sub, count_sub, rec_sub = self.hist_db.log_sighting(
                camera_id="cam_01",
                name="Bob",
                is_identified=True,
                similarity=0.89,
                confidence=0.92,
                track_id=1,
                timestamp=t0 + i * 0.05,
            )
            self.assertFalse(is_new_sub)
            self.assertEqual(count_sub, i)

        # Query stats: triggers _sync_active_to_db automatically
        stats = self.hist_db.get_stats()
        self.assertEqual(stats["total_detections"], 10)
        self.assertEqual(stats["events_count"], 1)


if __name__ == "__main__":
    unittest.main()
