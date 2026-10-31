"""
FaceHistoryDatabase: SQLite database and analytics engine for facial detection
and recognition history.

Persists timestamps, camera IDs, detection counts, confidence/similarity scores,
cropped thumbnails, and embeddings for both recognized identities and unidentified faces.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple


class FaceHistoryDatabase:
    """Thread-safe SQLite database storing face detection & recognition history."""

    def __init__(self, db_path: str = "data/face_history.db", cooldown_s: float = 30.0):
        self.db_path = db_path
        self.cooldown_s = cooldown_s
        self._lock = threading.RLock()
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self):
        with self._lock, self._conn:
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS face_sightings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    camera_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    is_identified INTEGER NOT NULL,
                    similarity REAL DEFAULT 0.0,
                    confidence REAL DEFAULT 0.0,
                    track_id INTEGER,
                    bbox TEXT,
                    thumbnail TEXT,
                    face_id TEXT,
                    detection_count INTEGER DEFAULT 1,
                    last_seen REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_face_sightings_name ON face_sightings(name)")
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_face_sightings_cam ON face_sightings(camera_id)")
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_face_sightings_ts ON face_sightings(last_seen)")
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_face_sightings_face_id ON face_sightings(face_id)")

            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS unidentified_faces (
                    face_id TEXT PRIMARY KEY,
                    camera_id TEXT NOT NULL,
                    first_seen REAL NOT NULL,
                    last_seen REAL NOT NULL,
                    detection_count INTEGER DEFAULT 1,
                    thumbnail TEXT,
                    embedding TEXT,
                    bbox TEXT,
                    track_id INTEGER,
                    status TEXT DEFAULT 'pending',
                    enrolled_name TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_unidentified_status ON unidentified_faces(status)")
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_unidentified_ts ON unidentified_faces(last_seen)")

    def log_sighting(
        self,
        camera_id: str,
        name: str,
        is_identified: bool,
        similarity: float = 0.0,
        confidence: float = 0.0,
        track_id: Optional[int] = None,
        bbox: Optional[List[float]] = None,
        thumbnail: Optional[str] = None,
        embedding: Optional[Any] = None,
        timestamp: Optional[float] = None,
    ) -> Tuple[bool, int, Dict[str, Any]]:
        """
        Logs a face observation. Deduplicates sightings within ``cooldown_s``.

        Returns:
            (is_new, detection_count, record_dict)
        """
        now = timestamp if timestamp is not None else time.time()
        bbox_json = json.dumps(bbox) if bbox is not None else None

        with self._lock:
            cur = self._conn.cursor()

            if is_identified:
                # Find recent sighting of this person on this camera within cooldown
                cur.execute("""
                    SELECT id, detection_count, thumbnail
                    FROM face_sightings
                    WHERE camera_id = ? AND name = ? AND is_identified = 1 AND (? - last_seen) <= ?
                    ORDER BY last_seen DESC LIMIT 1
                """, (camera_id, name, now, self.cooldown_s))
                row = cur.fetchone()

                if row:
                    row_id = row["id"]
                    new_count = row["detection_count"] + 1
                    thumb = thumbnail or row["thumbnail"]
                    cur.execute("""
                        UPDATE face_sightings
                        SET detection_count = ?, last_seen = ?, similarity = MAX(similarity, ?),
                            confidence = MAX(confidence, ?), thumbnail = COALESCE(?, thumbnail)
                        WHERE id = ?
                    """, (new_count, now, similarity, confidence, thumb, row_id))
                    self._conn.commit()
                    return False, new_count, {
                        "id": row_id, "camera_id": camera_id, "name": name,
                        "is_identified": True, "similarity": similarity, "confidence": confidence,
                        "track_id": track_id, "bbox": bbox, "thumbnail": thumb,
                        "detection_count": new_count, "last_seen": now,
                    }
                else:
                    cur.execute("""
                        INSERT INTO face_sightings
                        (timestamp, camera_id, name, is_identified, similarity, confidence,
                         track_id, bbox, thumbnail, face_id, detection_count, last_seen)
                        VALUES (?, ?, ?, 1, ?, ?, ?, ?, ?, NULL, 1, ?)
                    """, (now, camera_id, name, similarity, confidence, track_id, bbox_json, thumbnail, now))
                    new_id = cur.lastrowid
                    self._conn.commit()
                    return True, 1, {
                        "id": new_id, "camera_id": camera_id, "name": name,
                        "is_identified": True, "similarity": similarity, "confidence": confidence,
                        "track_id": track_id, "bbox": bbox, "thumbnail": thumbnail,
                        "detection_count": 1, "last_seen": now,
                    }
            else:
                # Unidentified face
                # Check recent active unidentified sighting by track_id or camera within cooldown
                row = None
                if track_id is not None:
                    cur.execute("""
                        SELECT id, face_id, detection_count, thumbnail
                        FROM face_sightings
                        WHERE camera_id = ? AND track_id = ? AND is_identified = 0 AND (? - last_seen) <= ?
                        ORDER BY last_seen DESC LIMIT 1
                    """, (camera_id, track_id, now, self.cooldown_s))
                    row = cur.fetchone()

                if row:
                    row_id = row["id"]
                    fid = row["face_id"]
                    new_count = row["detection_count"] + 1
                    thumb = thumbnail or row["thumbnail"]
                    cur.execute("""
                        UPDATE face_sightings
                        SET detection_count = ?, last_seen = ?, confidence = MAX(confidence, ?),
                            thumbnail = COALESCE(?, thumbnail)
                        WHERE id = ?
                    """, (new_count, now, confidence, thumb, row_id))

                    if fid:
                        cur.execute("""
                            UPDATE unidentified_faces
                            SET detection_count = ?, last_seen = ?, thumbnail = COALESCE(?, thumbnail)
                            WHERE face_id = ?
                        """, (new_count, now, thumb, fid))

                    self._conn.commit()
                    return False, new_count, {
                        "id": row_id, "face_id": fid, "camera_id": camera_id, "name": "Unidentified",
                        "is_identified": False, "similarity": similarity, "confidence": confidence,
                        "track_id": track_id, "bbox": bbox, "thumbnail": thumb,
                        "detection_count": new_count, "last_seen": now,
                    }
                else:
                    fid = "face_" + uuid.uuid4().hex[:10]
                    emb_json = None
                    if embedding is not None:
                        emb_arr = getattr(embedding, "tolist", lambda: list(embedding))()
                        emb_json = json.dumps(emb_arr)

                    cur.execute("""
                        INSERT INTO unidentified_faces
                        (face_id, camera_id, first_seen, last_seen, detection_count, thumbnail,
                         embedding, bbox, track_id, status)
                        VALUES (?, ?, ?, ?, 1, ?, ?, ?, ?, 'pending')
                    """, (fid, camera_id, now, now, thumbnail, emb_json, bbox_json, track_id))

                    cur.execute("""
                        INSERT INTO face_sightings
                        (timestamp, camera_id, name, is_identified, similarity, confidence,
                         track_id, bbox, thumbnail, face_id, detection_count, last_seen)
                        VALUES (?, ?, 'Unidentified', 0, ?, ?, ?, ?, ?, ?, 1, ?)
                    """, (now, camera_id, similarity, confidence, track_id, bbox_json, thumbnail, fid, now))

                    new_id = cur.lastrowid
                    self._conn.commit()
                    return True, 1, {
                        "id": new_id, "face_id": fid, "camera_id": camera_id, "name": "Unidentified",
                        "is_identified": False, "similarity": similarity, "confidence": confidence,
                        "track_id": track_id, "bbox": bbox, "thumbnail": thumbnail,
                        "detection_count": 1, "last_seen": now,
                    }

    def get_history(
        self,
        limit: int = 50,
        name: Optional[str] = None,
        camera_id: Optional[str] = None,
        is_identified: Optional[bool] = None,
        since_ts: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Queries face detection & recognition history ordered by most recent."""
        query = "SELECT * FROM face_sightings WHERE 1=1"
        params: List[Any] = []

        if name:
            query += " AND name LIKE ?"
            params.append(f"%{name}%")
        if camera_id and camera_id != "all":
            query += " AND camera_id = ?"
            params.append(camera_id)
        if is_identified is not None:
            query += " AND is_identified = ?"
            params.append(1 if is_identified else 0)
        if since_ts is not None:
            query += " AND last_seen >= ?"
            params.append(since_ts)

        query += " ORDER BY last_seen DESC LIMIT ?"
        params.append(limit)

        with self._lock:
            cur = self._conn.cursor()
            cur.execute(query, params)
            rows = cur.fetchall()

        out = []
        for r in rows:
            d = dict(r)
            if d.get("bbox"):
                try:
                    d["bbox"] = json.loads(d["bbox"])
                except Exception:
                    pass
            out.append(d)
        return out

    def get_history_grouped_by_label(self, limit_per_label: int = 8, max_groups: int = 50) -> List[Dict[str, Any]]:
        """
        Groups face sightings by identity label (e.g., 'John Doe', 'Jane', 'Unidentified').
        Returns aggregate metrics (total sightings, detection count, last seen camera/time)
        and the recent sighting thumbnails for each group.
        """
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("""
                SELECT
                    name,
                    is_identified,
                    COUNT(*) as events_count,
                    COALESCE(SUM(detection_count), 0) as total_detections,
                    MAX(last_seen) as last_seen,
                    MIN(timestamp) as first_seen,
                    MAX(similarity) as max_similarity,
                    MAX(confidence) as max_confidence
                FROM face_sightings
                GROUP BY name, is_identified
                ORDER BY last_seen DESC
                LIMIT ?
            """, (max_groups,))
            group_rows = cur.fetchall()

            groups: List[Dict[str, Any]] = []
            for g in group_rows:
                g_dict = dict(g)
                cur.execute("""
                    SELECT id, timestamp, last_seen, camera_id, name, is_identified,
                           similarity, confidence, thumbnail, detection_count, bbox
                    FROM face_sightings
                    WHERE name = ? AND is_identified = ?
                    ORDER BY last_seen DESC
                    LIMIT ?
                """, (g["name"], g["is_identified"], limit_per_label))
                s_rows = cur.fetchall()
                sightings = []
                primary_thumbnail = None
                last_camera = None
                for sr in s_rows:
                    sd = dict(sr)
                    if sd.get("thumbnail") and not primary_thumbnail:
                        primary_thumbnail = sd["thumbnail"]
                    if not last_camera and sd.get("camera_id"):
                        last_camera = sd["camera_id"]
                    if sd.get("bbox"):
                        try:
                            sd["bbox"] = json.loads(sd["bbox"])
                        except Exception:
                            pass
                    sightings.append(sd)

                g_dict["is_identified"] = bool(g_dict["is_identified"])
                g_dict["primary_thumbnail"] = primary_thumbnail
                g_dict["last_camera"] = last_camera
                g_dict["recent_sightings"] = sightings
                groups.append(g_dict)

            return groups

    def get_person_stats(self, name: str) -> Dict[str, Any]:
        """Returns sighting statistics for a specific enrolled identity name."""
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("""
                SELECT
                    COUNT(*) as events_count,
                    COALESCE(SUM(detection_count), 0) as total_detections,
                    MAX(last_seen) as last_seen,
                    MIN(timestamp) as first_seen,
                    MAX(similarity) as max_similarity,
                    (SELECT camera_id FROM face_sightings WHERE name = ? ORDER BY last_seen DESC LIMIT 1) as last_camera,
                    (SELECT thumbnail FROM face_sightings WHERE name = ? AND thumbnail IS NOT NULL ORDER BY last_seen DESC LIMIT 1) as latest_thumbnail
                FROM face_sightings
                WHERE name = ? AND is_identified = 1
            """, (name, name, name))
            row = cur.fetchone()
            if row and row["events_count"]:
                return dict(row)
            return {"events_count": 0, "total_detections": 0, "last_seen": None, "first_seen": None,
                    "max_similarity": 0.0, "last_camera": None, "latest_thumbnail": None}

    def get_unidentified_faces(self, limit: int = 50, status: str = "pending") -> List[Dict[str, Any]]:
        """Returns unidentified faces that can be labeled and enrolled."""
        with self._lock:
            cur = self._conn.cursor()
            if status == "all":
                cur.execute("SELECT * FROM unidentified_faces ORDER BY last_seen DESC LIMIT ?", (limit,))
            else:
                cur.execute("""
                    SELECT * FROM unidentified_faces
                    WHERE status = ?
                    ORDER BY last_seen DESC LIMIT ?
                """, (status, limit))
            rows = cur.fetchall()

        out = []
        for r in rows:
            d = dict(r)
            if d.get("bbox"):
                try:
                    d["bbox"] = json.loads(d["bbox"])
                except Exception:
                    pass
            if d.get("embedding"):
                try:
                    d["has_embedding"] = True
                except Exception:
                    d["has_embedding"] = False
            out.append(d)
        return out

    def get_unidentified_embedding(self, face_id: str) -> Optional[List[float]]:
        """Retrieves raw float vector embedding for an unidentified face."""
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("SELECT embedding FROM unidentified_faces WHERE face_id = ?", (face_id,))
            row = cur.fetchone()
            if row and row["embedding"]:
                try:
                    return json.loads(row["embedding"])
                except Exception:
                    return None
            return None

    def get_unidentified_face(self, face_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves raw record (thumbnail, bbox, camera, etc.) for an unidentified face."""
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("SELECT * FROM unidentified_faces WHERE face_id = ?", (face_id,))
            row = cur.fetchone()
            if row:
                d = dict(row)
                if d.get("bbox"):
                    try:
                        d["bbox"] = json.loads(d["bbox"])
                    except Exception:
                        pass
                return d
            return None

    def enroll_unidentified_face(self, face_id: str, name: str) -> bool:
        """Marks an unidentified face as enrolled under ``name``."""
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("""
                UPDATE unidentified_faces
                SET status = 'enrolled', enrolled_name = ?
                WHERE face_id = ?
            """, (name, face_id))
            cur.execute("""
                UPDATE face_sightings
                SET name = ?, is_identified = 1
                WHERE face_id = ?
            """, (name, face_id))
            self._conn.commit()
            return cur.rowcount > 0

    def delete_sighting(self, sighting_id: int) -> bool:
        """Deletes a sighting record from history."""
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("DELETE FROM face_sightings WHERE id = ?", (sighting_id,))
            self._conn.commit()
            return cur.rowcount > 0

    def clear_history(self) -> None:
        """Clears all sightings and unidentified faces."""
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("DELETE FROM face_sightings")
            cur.execute("DELETE FROM unidentified_faces")
            self._conn.commit()

    def get_stats(self) -> Dict[str, Any]:
        """Calculates face recognition summary metrics."""
        now = time.time()
        day_ago = now - 86400

        with self._lock:
            cur = self._conn.cursor()
            cur.execute("SELECT COUNT(*), COALESCE(SUM(detection_count), 0) FROM face_sightings")
            events_count, total_detections = cur.fetchone()

            cur.execute("SELECT COUNT(DISTINCT name) FROM face_sightings WHERE is_identified = 1")
            unique_identified = cur.fetchone()[0]

            cur.execute("SELECT COUNT(*) FROM unidentified_faces WHERE status = 'pending'")
            pending_unidentified = cur.fetchone()[0]

            cur.execute("SELECT COUNT(*) FROM face_sightings WHERE last_seen >= ?", (day_ago,))
            recent_24h = cur.fetchone()[0]

            cur.execute("""
                SELECT name, SUM(detection_count) as total_count, MAX(last_seen) as last_seen
                FROM face_sightings
                WHERE is_identified = 1
                GROUP BY name
                ORDER BY total_count DESC LIMIT 5
            """)
            top_identities = [{"name": r["name"], "count": r["total_count"], "last_seen": r["last_seen"]}
                              for r in cur.fetchall()]

        return {
            "total_detections": total_detections,
            "events_count": events_count,
            "unique_identified": unique_identified,
            "pending_unidentified": pending_unidentified,
            "recent_24h": recent_24h,
            "top_identities": top_identities,
        }

    def close(self):
        with self._lock:
            if self._conn:
                self._conn.close()
