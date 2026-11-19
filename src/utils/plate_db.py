"""
PlateDatabase: High-performance plate text event log, unique plate registry,
and temporal character-level fusion engine.
Persists plate sightings to data/plate_db.json using in-memory buffering.
"""

from __future__ import annotations

import collections
import json
import os
import threading
import time
from typing import Any, Dict, List, Optional, Tuple


class PlateDatabase:
    """
    Logs recognized license plate text with timestamps, camera ID, and confidence.
    Maintains a rolling event log, unique plate registry, and multi-frame temporal voting in RAM.
    Flushes to disk asynchronously/periodically to eliminate disk I/O bottlenecks.
    """

    def __init__(
        self,
        db_path: str = "data/plate_db.json",
        max_events: int = 5000,
        flush_interval_s: float = 5.0,
    ):
        self.db_path = db_path
        self.max_events = max_events
        self.flush_interval_s = flush_interval_s
        self._lock = threading.RLock()

        # In-memory registries
        self.unique_plates: Dict[str, Dict[str, Any]] = {}
        self.events: List[Dict[str, Any]] = []

        # Temporal character-level voting buffer: (cam_id, cluster_id) -> list of (timestamp, text, conf)
        self._temporal_reads: Dict[str, List[Tuple[float, str, float]]] = collections.defaultdict(list)

        self._dirty = False
        self._last_flush = time.time()

        self.load()

    def load(self):
        """Loads plate log from disk into RAM."""
        with self._lock:
            if not os.path.exists(self.db_path):
                os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
                self.save(force=True)
                return
            try:
                with open(self.db_path, "r") as f:
                    data = json.load(f)
                self.unique_plates = data.get("unique_plates", {})
                self.events = data.get("events", [])[-self.max_events:]
                print(f"[PLATE DB] Loaded {len(self.unique_plates)} unique plates, {len(self.events)} events.")
            except Exception as e:
                print(f"[PLATE DB ERROR] Failed to load: {e}")

    def save(self, force: bool = False):
        """Persists plate log to disk if dirty or forced."""
        with self._lock:
            now = time.time()
            if not force and not self._dirty:
                return
            if not force and (now - self._last_flush < self.flush_interval_s):
                return

            try:
                os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
                tmp_path = f"{self.db_path}.tmp"
                with open(tmp_path, "w") as f:
                    json.dump({
                        "unique_plates": self.unique_plates,
                        "events": self.events[-self.max_events:]
                    }, f, indent=2)
                os.replace(tmp_path, self.db_path)
                self._dirty = False
                self._last_flush = now
            except Exception as e:
                print(f"[PLATE DB SAVE ERROR] Failed to persist plate database: {e}")

    def fuse_plate(
        self,
        cam_id: str,
        plate_text: str,
        confidence: float,
        now: Optional[float] = None,
        window_s: float = 1.8,
        vehicle_id: Optional[Union[str, int]] = None,
        sri_lanka_dmt: bool = False,
        min_consensus_votes: int = 1,
    ) -> Tuple[str, float]:
        """
        Temporal character-level voting in RAM with optional Sri Lankan DMT correction.
        Aggregates successive noisy OCR reads of the same vehicle/camera within a short window,
        resolving character flicker (e.g. '0' vs 'O', '8' vs 'B') using confidence-weighted voting.
        """
        if not plate_text or len(plate_text.strip()) < 2:
            return plate_text, confidence

        now = now or time.time()
        plate_text = plate_text.strip().upper()
        cluster_key = f"{cam_id}_{vehicle_id}" if vehicle_id is not None else cam_id

        with self._lock:
            reads = self._temporal_reads[cluster_key]
            # Prune stale observations older than window_s
            cutoff = now - window_s
            reads = [r for r in reads if r[0] >= cutoff]
            reads.append((now, plate_text, max(0.1, float(confidence))))
            self._temporal_reads[cluster_key] = reads

            if len(reads) == 1 and not sri_lanka_dmt:
                return plate_text, confidence

            # 1. Determine most probable string length by weighted voting
            len_votes: Dict[int, float] = collections.defaultdict(float)
            for _, text, conf in reads:
                clean_t = "".join(c for c in text if c.isalnum()) if sri_lanka_dmt else text
                len_votes[len(clean_t)] += conf
            target_len = max(len_votes.items(), key=lambda kv: kv[1])[0]

            # 2. Filter reads matching target length
            matching_reads = []
            for _, text, conf in reads:
                clean_t = "".join(c for c in text if c.isalnum()) if sri_lanka_dmt else text
                if len(clean_t) == target_len:
                    matching_reads.append((clean_t, conf))

            if not matching_reads:
                matching_reads = [(plate_text, max(0.1, float(confidence)))]
                target_len = len(plate_text)

            # 3. Positional character consensus
            fused_chars = []
            char_confs = []
            for i in range(target_len):
                char_weights: Dict[str, float] = collections.defaultdict(float)
                total_pos_weight = 0.0
                for text, conf in matching_reads:
                    ch = text[i]
                    char_weights[ch] += conf
                    total_pos_weight += conf
                best_char, best_weight = max(char_weights.items(), key=lambda kv: kv[1])
                fused_chars.append(best_char)
                char_confs.append(best_weight / max(0.01, total_pos_weight))

            fused_text = "".join(fused_chars)
            fused_conf = round(float(sum(char_confs) / max(1, len(char_confs))), 4)

            # 4. Sri Lankan DMT validation & correction
            if sri_lanka_dmt:
                from src.analytics.plate_dmt import clean_and_correct_dmt_plate
                dmt_result = clean_and_correct_dmt_plate(fused_text)
                if dmt_result is not None:
                    fused_text, conf_mult = dmt_result
                    fused_conf = round(float(fused_conf * conf_mult), 4)
                else:
                    raw_result = clean_and_correct_dmt_plate(plate_text)
                    if raw_result is not None:
                        fused_text, conf_mult = raw_result
                        fused_conf = round(float(confidence * conf_mult), 4)

            return fused_text, fused_conf

    def get_vote_count(self, cam_id: str, vehicle_id: Optional[Union[str, int]] = None) -> int:
        """Returns the number of active observations in the current temporal window."""
        cluster_key = f"{cam_id}_{vehicle_id}" if vehicle_id is not None else cam_id
        with self._lock:
            return len(self._temporal_reads.get(cluster_key, []))

    def log_plate(
        self,
        plate_text: str,
        cam_id: str,
        confidence: float = 0.0,
        ocr_confidence: float = 0.0,
        now: Optional[float] = None,
    ):
        """
        Logs a plate sighting into RAM. Updates unique plate registry and rolling event log.
        Debounces disk persistence to eliminate video processing stalls.
        """
        if not plate_text or len(plate_text.strip()) < 2:
            return

        ts = now if now is not None else time.time()
        plate_text = plate_text.strip().upper()

        with self._lock:
            # Update unique plate registry in RAM
            if plate_text not in self.unique_plates:
                self.unique_plates[plate_text] = {
                    "first_seen": ts,
                    "last_seen": ts,
                    "count": 1,
                    "cameras": [cam_id]
                }
            else:
                entry = self.unique_plates[plate_text]
                entry["last_seen"] = ts
                entry["count"] += 1
                if cam_id not in entry["cameras"]:
                    entry["cameras"].append(cam_id)

            # Append to rolling event log in RAM
            self.events.append({
                "plate_text": plate_text,
                "cam_id": cam_id,
                "confidence": round(confidence, 4),
                "ocr_confidence": round(ocr_confidence, 4),
                "timestamp": ts
            })

            # Trim to max_events in RAM
            if len(self.events) > self.max_events:
                self.events = self.events[-self.max_events:]

            self._dirty = True

            # Periodic disk flush if interval elapsed
            if ts - self._last_flush >= self.flush_interval_s:
                self.save(force=False)

    def get_recent_events(
        self,
        cam_id: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Returns most recent plate events from RAM, optionally filtered by camera."""
        with self._lock:
            events = self.events
            if cam_id:
                events = [e for e in events if e.get("cam_id") == cam_id]
            return list(reversed(events[-limit:]))

    def list_unique_plates(self) -> List[Dict[str, Any]]:
        """Returns all unique plates from RAM sorted by most recently seen."""
        with self._lock:
            result = []
            for plate_text, info in self.unique_plates.items():
                result.append({
                    "plate_text": plate_text,
                    "first_seen": info["first_seen"],
                    "last_seen": info["last_seen"],
                    "count": info["count"],
                    "cameras": list(info["cameras"])
                })
            return sorted(result, key=lambda x: x["last_seen"], reverse=True)

    def remove_plate(self, plate_text: str) -> bool:
        """Removes a plate from the unique registry."""
        plate_text = plate_text.strip().upper()
        with self._lock:
            if plate_text in self.unique_plates:
                del self.unique_plates[plate_text]
                self.save(force=True)
                return True
            return False

    def close(self):
        """Flushes in-memory data to disk on shutdown."""
        self.save(force=True)
