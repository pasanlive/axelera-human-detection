"""
PlateDatabase: Plate text event log and unique plate registry.
Mirrors the structure of FaceDatabase for consistency.
Persists plate sightings to data/plate_db.json.
"""

import json
import os
import time
from typing import List, Dict, Any, Optional


class PlateDatabase:
    """
    Logs recognized license plate text with timestamps, camera ID, and confidence.
    Maintains a rolling event log and unique plate registry.
    """

    def __init__(self, db_path: str = "data/plate_db.json", max_events: int = 1000):
        self.db_path = db_path
        self.max_events = max_events
        # unique_plates: {plate_text -> {first_seen, last_seen, count, cameras}}
        self.unique_plates: Dict[str, Dict[str, Any]] = {}
        # events: rolling list of {plate_text, cam_id, confidence, ocr_confidence, timestamp}
        self.events: List[Dict[str, Any]] = []
        self.load()

    def load(self):
        """Loads plate log from disk."""
        if not os.path.exists(self.db_path):
            os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
            self.save()
            return
        try:
            with open(self.db_path, "r") as f:
                data = json.load(f)
            self.unique_plates = data.get("unique_plates", {})
            self.events = data.get("events", [])[-self.max_events:]
            print(f"[PLATE DB] Loaded {len(self.unique_plates)} unique plates, {len(self.events)} events.")
        except Exception as e:
            print(f"[PLATE DB ERROR] Failed to load: {e}")

    def save(self):
        """Persists plate log to disk."""
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        with open(self.db_path, "w") as f:
            json.dump({
                "unique_plates": self.unique_plates,
                "events": self.events[-self.max_events:]
            }, f, indent=2)

    def log_plate(self, plate_text: str, cam_id: str,
                  confidence: float = 0.0, ocr_confidence: float = 0.0):
        """
        Logs a plate sighting. Updates unique plate registry and event log.
        :param plate_text: Decoded plate string
        :param cam_id: Camera ID that detected the plate
        :param confidence: Plate detector confidence
        :param ocr_confidence: OCR read confidence
        """
        if not plate_text or len(plate_text.strip()) < 2:
            return

        ts = time.time()
        plate_text = plate_text.strip().upper()

        # Update unique plate registry
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

        # Append to rolling event log
        self.events.append({
            "plate_text": plate_text,
            "cam_id": cam_id,
            "confidence": round(confidence, 4),
            "ocr_confidence": round(ocr_confidence, 4),
            "timestamp": ts
        })

        # Trim to max_events
        if len(self.events) > self.max_events:
            self.events = self.events[-self.max_events:]

        self.save()

    def get_recent_events(self, cam_id: Optional[str] = None,
                          limit: int = 50) -> List[Dict[str, Any]]:
        """Returns most recent plate events, optionally filtered by camera."""
        events = self.events
        if cam_id:
            events = [e for e in events if e.get("cam_id") == cam_id]
        return list(reversed(events[-limit:]))

    def list_unique_plates(self) -> List[Dict[str, Any]]:
        """Returns all unique plates sorted by most recently seen."""
        result = []
        for plate_text, info in self.unique_plates.items():
            result.append({
                "plate_text": plate_text,
                "first_seen": info["first_seen"],
                "last_seen": info["last_seen"],
                "count": info["count"],
                "cameras": info["cameras"]
            })
        return sorted(result, key=lambda x: x["last_seen"], reverse=True)

    def remove_plate(self, plate_text: str) -> bool:
        """Removes a plate from the unique registry."""
        plate_text = plate_text.strip().upper()
        if plate_text in self.unique_plates:
            del self.unique_plates[plate_text]
            self.save()
            return True
        return False
