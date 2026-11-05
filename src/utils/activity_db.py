"""
ActivityDatabase: Comprehensive Activity & Incident Tracking Engine.

Tracks, correlates, and persists multi-modal activity incidents across cameras:
- Human presence / crowd tracking
- Face recognition sightings (known & unidentified)
- License plate detections (LPR)
- Zone crossing events
- Incident duration calculation (first_seen to last_seen continuous observation)
- Multi-modal image evidence (face crops, plate crops, full scene movement screenshots)
"""

from __future__ import annotations

import base64
import json
import os
import sqlite3
import threading
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


class ActivityDatabase:
    """
    SQLite-backed activity and incident tracking database.
    Correlates continuous actions into incidents with duration and multi-modal image evidence.
    """

    def __init__(self, db_path: str = "data/activity_history.db", incident_timeout_s: float = 12.0):
        self.db_path = db_path
        self.incident_timeout_s = incident_timeout_s
        self._lock = threading.RLock()
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_db()

        # In-memory incident aggregators keyed by (camera_id, activity_type, correlation_key)
        # e.g. key = (cam_id, "person", track_id) or (cam_id, "group", "general")
        self._active_incidents: Dict[str, Dict[str, Any]] = {}
        self._last_flush_ts = 0.0

    def _init_db(self):
        with self._lock, self._conn:
            # High-performance in-memory cache and WAL settings
            self._conn.execute("PRAGMA journal_mode = WAL;")
            self._conn.execute("PRAGMA synchronous = NORMAL;")
            self._conn.execute("PRAGMA cache_size = -64000;")  # 64 MB RAM cache
            self._conn.execute("PRAGMA temp_store = MEMORY;")
            self._conn.execute("PRAGMA mmap_size = 268435456;")  # 256 MB memory map

            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS activities (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    activity_id TEXT UNIQUE NOT NULL,
                    date TEXT NOT NULL,
                    start_time TEXT NOT NULL,
                    end_time TEXT NOT NULL,
                    first_seen REAL NOT NULL,
                    last_seen REAL NOT NULL,
                    duration_s REAL NOT NULL,
                    camera_ids TEXT NOT NULL,
                    primary_camera_id TEXT NOT NULL,
                    detection_type TEXT NOT NULL,
                    num_humans INTEGER DEFAULT 0,
                    identified_faces TEXT,
                    unidentified_faces INTEGER DEFAULT 0,
                    identified_plates TEXT,
                    track_ids TEXT,
                    summary TEXT NOT NULL,
                    details_json TEXT,
                    image_evidence_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_act_date ON activities(date)")
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_act_type ON activities(detection_type)")
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_act_cam ON activities(primary_camera_id)")
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_act_last_seen ON activities(last_seen)")

    # ─────────────────────────────────────────────────────────────────────────
    # Image Evidence Helper
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def encode_thumb(img: np.ndarray, max_dim: int = 160, quality: int = 75) -> Optional[str]:
        """Encodes an image numpy array into a compact Base64 JPEG data URI."""
        if img is None or img.size == 0:
            return None
        try:
            h, w = img.shape[:2]
            scale = min(1.0, max_dim / max(h, w))
            if scale < 1.0:
                img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            _, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")
        except Exception:
            return None

    @staticmethod
    def crop_roi(frame: np.ndarray, bbox: List[float], margin_pct: float = 0.15) -> Optional[np.ndarray]:
        """Extracts bounding box crop from full image with optional padding."""
        if frame is None or frame.size == 0 or not bbox or len(bbox) < 4:
            return None
        try:
            fh, fw = frame.shape[:2]
            b = bbox
            bx = [b[0] * fw, b[1] * fh, b[2] * fw, b[3] * fh] if max(b) <= 1.05 and fw > 0 else b
            x1, y1, x2, y2 = map(int, bx)
            pw, ph = int((x2 - x1) * margin_pct), int((y2 - y1) * margin_pct)
            cx1, cy1 = max(0, x1 - pw), max(0, y1 - ph)
            cx2, cy2 = min(fw, x2 + pw), min(fh, y2 + ph)
            if cx2 > cx1 and cy2 > cy1:
                return frame[cy1:cy2, cx1:cx2]
        except Exception:
            pass
        return None

    # ─────────────────────────────────────────────────────────────────────────
    # Live Pipeline Observation Ingestion
    # ─────────────────────────────────────────────────────────────────────────

    def record_frame_activity(
        self,
        camera_id: str,
        now: float,
        persons: List[Dict[str, Any]],
        faces: List[Dict[str, Any]],
        plates: List[Dict[str, Any]],
        get_frame: Any = None,
    ):
        """
        Ingests frame detections, updating active activity incidents or starting new ones.
        Continuous detections within `incident_timeout_s` extend existing incidents.
        """
        has_humans = bool(persons)
        has_faces = bool(faces)
        has_plates = bool(plates)

        if not (has_humans or has_faces or has_plates):
            self._periodic_flush(now)
            return

        with self._lock:
            # 1. Determine detection type classification
            types = []
            if has_humans:
                types.append("human")
            if has_faces:
                types.append("face")
            if has_plates:
                types.append("vehicle_plate")
            det_type = "+".join(types)

            # Unique key for tracking continuity on this camera
            # If tracked humans exist, correlate by track IDs; otherwise group by camera
            track_keys = sorted([str(p.get("track_id")) for p in persons if p.get("track_id") is not None])
            inc_key = f"{camera_id}_{det_type}_{'_'.join(track_keys[:3]) if track_keys else 'cluster'}"

            # Check if there is an active incident for this camera and activity
            incident = self._active_incidents.get(inc_key)
            if incident is None:
                # Also check for any recently updated incident on this camera within timeout
                for ik, inc in list(self._active_incidents.items()):
                    if inc["primary_camera_id"] == camera_id and (now - inc["last_seen"]) <= self.incident_timeout_s:
                        # Continue existing action if overlaps
                        inc_key = ik
                        incident = inc
                        break

            full_frame = None

            def _get_full_frame():
                nonlocal full_frame
                if full_frame is None and get_frame is not None and callable(get_frame):
                    try:
                        full_frame = get_frame()
                    except Exception:
                        pass
                return full_frame

            if incident is None:
                # Start new incident
                act_id = f"act_{int(now)}_{uuid.uuid4().hex[:6]}"
                incident = {
                    "activity_id": act_id,
                    "first_seen": now,
                    "last_seen": now,
                    "camera_ids": {camera_id},
                    "primary_camera_id": camera_id,
                    "detection_type": det_type,
                    "max_humans": len(persons),
                    "identified_faces": set(),
                    "unidentified_count": 0,
                    "identified_plates": set(),
                    "track_ids": set(track_keys),
                    "evidence": {
                        "screenshot": None,
                        "faces": [],
                        "plates": [],
                    },
                }
                # Capture initial full scene screenshot
                ff = _get_full_frame()
                if ff is not None:
                    incident["evidence"]["screenshot"] = self.encode_thumb(ff, max_dim=480, quality=70)
                self._active_incidents[inc_key] = incident
            else:
                incident["last_seen"] = now
                incident["camera_ids"].add(camera_id)
                incident["max_humans"] = max(incident["max_humans"], len(persons))
                incident["track_ids"].update(track_keys)
                if det_type not in incident["detection_type"]:
                    incident["detection_type"] = f"{incident['detection_type']}+{det_type}"

            # Update faces evidence
            for f in faces:
                name = f.get("name")
                sim = f.get("similarity", 0.0)
                thumb = f.get("thumbnail")
                if not thumb and "bbox" in f:
                    ff = _get_full_frame()
                    if ff is not None:
                        crop = self.crop_roi(ff, f["bbox"])
                        thumb = self.encode_thumb(crop, max_dim=120)

                if name and name != "Unknown":
                    incident["identified_faces"].add(name)
                    # Keep best image evidence for this face
                    if thumb and not any(ef.get("name") == name for ef in incident["evidence"]["faces"]):
                        incident["evidence"]["faces"].append({"name": name, "similarity": round(sim, 2), "thumb": thumb})
                else:
                    incident["unidentified_count"] += 1
                    if thumb and len(incident["evidence"]["faces"]) < 5:
                        incident["evidence"]["faces"].append({"name": "Unidentified", "similarity": 0.0, "thumb": thumb})

            # Update plates evidence
            for p in plates:
                text = p.get("plate_text", "")
                if text and len(text) >= 2:
                    incident["identified_plates"].add(text)
                    if not any(ep.get("text") == text for ep in incident["evidence"]["plates"]):
                        plate_thumb = None
                        if "bbox" in p:
                            ff = _get_full_frame()
                            if ff is not None:
                                crop = self.crop_roi(ff, p["bbox"])
                                plate_thumb = self.encode_thumb(crop, max_dim=160)
                        incident["evidence"]["plates"].append({
                            "text": text,
                            "confidence": round(p.get("confidence", 0.0), 2),
                            "thumb": plate_thumb,
                        })

            self._periodic_flush(now)

    def _periodic_flush(self, now: float, force: bool = False):
        """Flushes completed incidents (inactive beyond incident_timeout_s) into SQLite."""
        if not force and (0 <= now - self._last_flush_ts < 2.0):
            return
        self._last_flush_ts = now

        to_remove = []
        for key, inc in self._active_incidents.items():
            if force or (now - inc["last_seen"] >= self.incident_timeout_s):
                self._persist_incident(inc)
                to_remove.append(key)

        for k in to_remove:
            self._active_incidents.pop(k, None)

    def _persist_incident(self, inc: Dict[str, Any]):
        """Writes an incident record to SQLite."""
        duration_s = max(0.1, round(inc["last_seen"] - inc["first_seen"], 1))
        start_dt = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(inc["first_seen"]))
        end_dt = time.strftime("%H:%M:%S", time.localtime(inc["last_seen"]))
        date_str = time.strftime("%Y-%m-%d", time.localtime(inc["first_seen"]))

        # Human summary building
        parts = []
        if inc["max_humans"] > 0:
            parts.append(f"{inc['max_humans']} person{'s' if inc['max_humans'] > 1 else ''}")
        if inc["identified_faces"]:
            names_str = ", ".join(sorted(inc["identified_faces"]))
            parts.append(f"identified: {names_str}")
        elif inc["unidentified_count"] > 0:
            parts.append("unidentified faces")
        if inc["identified_plates"]:
            plates_str = ", ".join(sorted(inc["identified_plates"]))
            parts.append(f"plates: {plates_str}")

        summary = "; ".join(parts) if parts else "Activity detected"

        details = {
            "num_humans": inc["max_humans"],
            "identified_faces": sorted(list(inc["identified_faces"])),
            "unidentified_faces": inc["unidentified_count"],
            "identified_plates": sorted(list(inc["identified_plates"])),
            "track_ids": sorted(list(inc["track_ids"])),
        }

        try:
            with self._conn:
                self._conn.execute("""
                    INSERT OR REPLACE INTO activities (
                        activity_id, date, start_time, end_time,
                        first_seen, last_seen, duration_s, camera_ids,
                        primary_camera_id, detection_type, num_humans,
                        identified_faces, unidentified_faces, identified_plates,
                        track_ids, summary, details_json, image_evidence_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    inc["activity_id"],
                    date_str,
                    start_dt,
                    end_dt,
                    inc["first_seen"],
                    inc["last_seen"],
                    duration_s,
                    json.dumps(sorted(list(inc["camera_ids"]))),
                    inc["primary_camera_id"],
                    inc["detection_type"],
                    inc["max_humans"],
                    json.dumps(sorted(list(inc["identified_faces"]))),
                    inc["unidentified_count"],
                    json.dumps(sorted(list(inc["identified_plates"]))),
                    json.dumps(sorted(list(inc["track_ids"]))),
                    summary,
                    json.dumps(details),
                    json.dumps(inc["evidence"]),
                ))
        except Exception as e:
            print(f"[ACTIVITY DB ERROR] Failed to persist incident {inc.get('activity_id')}: {e}")

    # ─────────────────────────────────────────────────────────────────────────
    # Queries & Reporting
    # ─────────────────────────────────────────────────────────────────────────

    def get_activities(
        self,
        limit: int = 50,
        offset: int = 0,
        camera_id: Optional[str] = None,
        detection_type: Optional[str] = None,
        date: Optional[str] = None,
        search: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieves paginated and filtered activity records."""
        with self._lock:
            # First flush any recently ended incidents
            self._periodic_flush(time.time(), force=False)

            sql = "SELECT * FROM activities WHERE 1=1"
            params: List[Any] = []

            if camera_id and camera_id != "all":
                sql += " AND (primary_camera_id = ? OR camera_ids LIKE ?)"
                params.extend([camera_id, f'%"{camera_id}"%'])

            if detection_type and detection_type != "all":
                sql += " AND detection_type LIKE ?"
                params.append(f"%{detection_type}%")

            if date:
                sql += " AND date = ?"
                params.append(date)

            if search:
                sql += " AND (summary LIKE ? OR identified_faces LIKE ? OR identified_plates LIKE ?)"
                kw = f"%{search}%"
                params.extend([kw, kw, kw])

            sql += " ORDER BY last_seen DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])

            cur = self._conn.execute(sql, params)
            rows = cur.fetchall()

            results = []
            for r in rows:
                item = dict(r)
                item["camera_ids"] = json.loads(item.get("camera_ids") or "[]")
                item["identified_faces"] = json.loads(item.get("identified_faces") or "[]")
                item["identified_plates"] = json.loads(item.get("identified_plates") or "[]")
                item["track_ids"] = json.loads(item.get("track_ids") or "[]")
                item["details"] = json.loads(item.get("details_json") or "{}")
                item["evidence"] = json.loads(item.get("image_evidence_json") or "{}")
                results.append(item)
            return results

    def get_stats(self) -> Dict[str, Any]:
        """Calculates high-level activity telemetry."""
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("SELECT COUNT(*) FROM activities")
            total = cur.fetchone()[0]

            today = time.strftime("%Y-%m-%d", time.localtime())
            cur.execute("SELECT COUNT(*) FROM activities WHERE date = ?", (today,))
            total_today = cur.fetchone()[0]

            cur.execute("SELECT AVG(duration_s) FROM activities")
            avg_dur = cur.fetchone()[0] or 0.0

            cur.execute("SELECT COUNT(*) FROM activities WHERE identified_faces != '[]'")
            face_incidents = cur.fetchone()[0]

            cur.execute("SELECT COUNT(*) FROM activities WHERE identified_plates != '[]'")
            plate_incidents = cur.fetchone()[0]

            return {
                "total_activities": total,
                "today_activities": total_today,
                "avg_duration_s": round(avg_dur, 1),
                "face_incidents": face_incidents,
                "plate_incidents": plate_incidents,
                "active_in_progress": len(self._active_incidents),
            }

    def clear(self):
        """Purges activity log."""
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM activities")
            self._active_incidents.clear()

    def close(self):
        with self._lock:
            self._periodic_flush(time.time(), force=True)
            self._conn.close()
