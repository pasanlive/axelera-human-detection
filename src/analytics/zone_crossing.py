"""
ZoneCrossingDetector: Rule-based virtual zone and line crossing detection.

No ML model required — operates purely on ByteTracker centroids.

Features:
  - Line zones:    tripwire crossing with IN/OUT direction detection
  - Polygon zones: area entry/exit detection

Key design decisions:
  - Zone coordinates are ABSOLUTE PIXELS matching each stream's native capture resolution.
  - Each zone MUST have an explicit camera_id — no wildcard zones.
  - If a stream has no zones configured, update() returns [] immediately (zero overhead).
  - Crossing events stored in a rolling thread-safe buffer.
"""

import time
import threading
import numpy as np
import cv2
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class ZoneEvent:
    """A single zone crossing event."""
    zone_id: str
    zone_name: str
    track_id: int
    direction: str        # "IN"/"OUT" for line; "ENTER"/"EXIT" for polygon
    timestamp: float
    cam_id: str
    centroid: Tuple[float, float]


class ZoneCrossingDetector:
    """
    Detects when tracked objects cross virtual line or polygon zones.

    Zone coordinate system: absolute pixel coordinates at the stream's native resolution.
    Zones are camera-specific — each zone's camera_id must match a configured camera.
    If a camera has no zones, all update() calls for that camera are no-ops.
    """

    def __init__(self, zone_configs: List[Dict[str, Any]], max_events: int = 500):
        """
        :param zone_configs: List of zone definition dicts from config.yaml
        :param max_events: Max events to keep in rolling buffer
        """
        self.max_events = max_events
        self._lock = threading.Lock()

        # zones_by_camera: {cam_id -> [zone_def, ...]}
        self.zones_by_camera: Dict[str, List[Dict[str, Any]]] = {}
        # crossing_counts: {zone_id -> {"IN": int, "OUT": int} or {"ENTER": int, "EXIT": int}}
        self.crossing_counts: Dict[str, Dict[str, int]] = {}
        # track_history: {cam_id -> {track_id -> [centroid, ...]}}  (last 3 positions)
        self._track_history: Dict[str, Dict[int, List[Tuple[float, float]]]] = {}
        # prev_poly_state: {cam_id -> {track_id -> {zone_id -> bool}}}  (inside polygon?)
        self._prev_poly_state: Dict[str, Dict[int, Dict[str, bool]]] = {}
        # Rolling event log
        self._events: List[ZoneEvent] = []

        self._load_zones(zone_configs)

    def _load_zones(self, zone_configs: List[Dict[str, Any]]):
        """Parses zone config and organizes by camera."""
        for zone in zone_configs:
            cam_id = zone.get("camera_id")
            zone_id = zone.get("id", f"zone_{len(self.zones_by_camera)}")
            zone_type = zone.get("type", "line").lower()
            points = zone.get("points", [])

            if not cam_id:
                print(f"[ZONE CROSSING] WARNING: Zone '{zone_id}' has no camera_id — skipped.")
                continue
            if len(points) < 2:
                print(f"[ZONE CROSSING] WARNING: Zone '{zone_id}' has insufficient points — skipped.")
                continue

            zone_def = {
                "id": zone_id,
                "name": zone.get("name", zone_id),
                "type": zone_type,
                "camera_id": cam_id,
                "points": [[int(p[0]), int(p[1])] for p in points],
                "direction_labels": zone.get("direction_labels", ["IN", "OUT"]),
                "color": zone.get("color", [0, 255, 180])  # BGR
            }

            if cam_id not in self.zones_by_camera:
                self.zones_by_camera[cam_id] = []
            self.zones_by_camera[cam_id].append(zone_def)

            # Initialize crossing counts
            self.crossing_counts[zone_id] = {
                zone_def["direction_labels"][0]: 0,
                zone_def["direction_labels"][1] if len(zone_def["direction_labels"]) > 1
                else ("EXIT" if zone_type == "polygon" else "OUT"): 0
            }

        total_zones = sum(len(v) for v in self.zones_by_camera.values())
        print(f"[ZONE CROSSING] Loaded {total_zones} zones across {len(self.zones_by_camera)} camera(s).")

    def update(self, cam_id: str, track_id: int,
               centroid: Tuple[float, float]) -> List[ZoneEvent]:
        """
        Updates track position and checks for zone crossings.
        :param cam_id: Camera stream ID
        :param track_id: ByteTracker track ID
        :param centroid: (cx, cy) in absolute pixel coords
        :return: List of ZoneEvents fired this frame (empty if no zones for this camera)
        """
        # Fast early exit if no zones for this camera
        if cam_id not in self.zones_by_camera:
            return []

        events = []

        # Update position history for this track (keep last 3 positions)
        if cam_id not in self._track_history:
            self._track_history[cam_id] = {}
        history = self._track_history[cam_id]
        if track_id not in history:
            history[track_id] = []
        history[track_id].append(centroid)
        if len(history[track_id]) > 3:
            history[track_id].pop(0)

        # Need at least 2 positions to detect crossing
        if len(history[track_id]) < 2:
            return []

        prev_pt = history[track_id][-2]
        curr_pt = history[track_id][-1]

        for zone in self.zones_by_camera[cam_id]:
            zone_id = zone["id"]
            zone_type = zone["type"]
            pts = zone["points"]
            labels = zone["direction_labels"]

            if zone_type == "line":
                event = self._check_line_crossing(
                    zone, prev_pt, curr_pt, cam_id, track_id, labels
                )
                if event:
                    events.append(event)

            elif zone_type == "polygon":
                event = self._check_polygon_crossing(
                    zone, curr_pt, cam_id, track_id
                )
                if event:
                    events.append(event)

        # Store events in rolling buffer
        if events:
            with self._lock:
                self._events.extend(events)
                if len(self._events) > self.max_events:
                    self._events = self._events[-self.max_events:]

        return events

    def _check_line_crossing(self, zone: Dict, prev_pt: Tuple, curr_pt: Tuple,
                              cam_id: str, track_id: int,
                              labels: List[str]) -> Optional[ZoneEvent]:
        """Detects line segment crossing using cross-product sign change."""
        pts = zone["points"]
        if len(pts) < 2:
            return None

        lx1, ly1 = float(pts[0][0]), float(pts[0][1])
        lx2, ly2 = float(pts[1][0]), float(pts[1][1])

        # Cross product of line direction with vectors to prev and curr point
        def cross(ax, ay, bx, by, px, py):
            return (bx - ax) * (py - ay) - (by - ay) * (px - ax)

        sign_prev = cross(lx1, ly1, lx2, ly2, prev_pt[0], prev_pt[1])
        sign_curr = cross(lx1, ly1, lx2, ly2, curr_pt[0], curr_pt[1])

        # Crossing detected when signs differ
        if (sign_prev > 0) == (sign_curr > 0) or sign_prev == 0 or sign_curr == 0:
            return None

        # Determine direction from sign change
        direction = labels[0] if sign_prev > 0 else (labels[1] if len(labels) > 1 else "OUT")

        # Update counts
        if zone["id"] in self.crossing_counts:
            self.crossing_counts[zone["id"]][direction] = \
                self.crossing_counts[zone["id"]].get(direction, 0) + 1

        return ZoneEvent(
            zone_id=zone["id"],
            zone_name=zone["name"],
            track_id=track_id,
            direction=direction,
            timestamp=time.time(),
            cam_id=cam_id,
            centroid=curr_pt
        )

    def _check_polygon_crossing(self, zone: Dict, curr_pt: Tuple,
                                 cam_id: str, track_id: int) -> Optional[ZoneEvent]:
        """Detects polygon entry/exit via pointPolygonTest."""
        pts = zone["points"]
        if len(pts) < 3:
            return None

        poly = np.array(pts, dtype=np.int32)
        px, py = float(curr_pt[0]), float(curr_pt[1])
        inside_now = cv2.pointPolygonTest(poly, (px, py), False) >= 0

        # Initialize state tracking
        if cam_id not in self._prev_poly_state:
            self._prev_poly_state[cam_id] = {}
        if track_id not in self._prev_poly_state[cam_id]:
            self._prev_poly_state[cam_id][track_id] = {}

        zone_id = zone["id"]
        prev_inside = self._prev_poly_state[cam_id][track_id].get(zone_id)
        self._prev_poly_state[cam_id][track_id][zone_id] = inside_now

        # Only fire on state change (not on first observation)
        if prev_inside is None:
            return None
        if inside_now == prev_inside:
            return None

        direction = "ENTER" if inside_now else "EXIT"
        labels = zone.get("direction_labels", ["ENTER", "EXIT"])
        direction_label = labels[0] if inside_now else (labels[1] if len(labels) > 1 else "EXIT")

        if zone_id in self.crossing_counts:
            self.crossing_counts[zone_id][direction_label] = \
                self.crossing_counts[zone_id].get(direction_label, 0) + 1

        return ZoneEvent(
            zone_id=zone_id,
            zone_name=zone["name"],
            track_id=track_id,
            direction=direction_label,
            timestamp=time.time(),
            cam_id=cam_id,
            centroid=curr_pt
        )

    def purge_stale_tracks(self, cam_id: str, active_track_ids: List[int]):
        """Removes history for tracks that are no longer active."""
        if cam_id in self._track_history:
            stale = [tid for tid in self._track_history[cam_id] if tid not in active_track_ids]
            for tid in stale:
                del self._track_history[cam_id][tid]
        if cam_id in self._prev_poly_state:
            stale = [tid for tid in self._prev_poly_state[cam_id] if tid not in active_track_ids]
            for tid in stale:
                del self._prev_poly_state[cam_id][tid]

    def get_events(self, cam_id: Optional[str] = None,
                   since_ts: Optional[float] = None,
                   limit: int = 100) -> List[Dict[str, Any]]:
        """Returns recent crossing events, optionally filtered by camera or timestamp."""
        with self._lock:
            events = list(self._events)
        if cam_id:
            events = [e for e in events if e.cam_id == cam_id]
        if since_ts:
            events = [e for e in events if e.timestamp >= since_ts]
        events = events[-limit:]
        return [
            {
                "zone_id": e.zone_id,
                "zone_name": e.zone_name,
                "track_id": e.track_id,
                "direction": e.direction,
                "timestamp": e.timestamp,
                "cam_id": e.cam_id,
                "centroid": list(e.centroid)
            }
            for e in reversed(events)
        ]

    def get_zone_counts(self, cam_id: Optional[str] = None) -> Dict[str, Any]:
        """Returns crossing counts per zone, optionally filtered by camera."""
        result = {}
        zones_to_include = []
        if cam_id and cam_id in self.zones_by_camera:
            zones_to_include = self.zones_by_camera[cam_id]
        else:
            for zones in self.zones_by_camera.values():
                zones_to_include.extend(zones)

        for zone in zones_to_include:
            zid = zone["id"]
            result[zid] = {
                "name": zone["name"],
                "type": zone["type"],
                "camera_id": zone["camera_id"],
                "counts": dict(self.crossing_counts.get(zid, {})),
                "points": zone["points"],
                "color": zone.get("color", [0, 255, 180])
            }
        return result

    def get_zones_for_camera(self, cam_id: str) -> List[Dict[str, Any]]:
        """Returns zone definitions for a specific camera."""
        return list(self.zones_by_camera.get(cam_id, []))

    def has_zones_for_camera(self, cam_id: str) -> bool:
        """Returns True if this camera has at least one configured zone."""
        return bool(self.zones_by_camera.get(cam_id))
