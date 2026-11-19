"""
Human Detection False Positive Reduction Filter.

Implements:
1. Geometry & ROI Filtering:
   - min_box_area, min_height, max_aspect_ratio (width / height > 1.5 rejected).
   - Configurable polygon exclusion zones per camera stream (bottom-center footpoint test).
2. Temporal Persistence (Tracker Gating):
   - Confirms detections only after track_id is continuously detected for min_consecutive_frames (default: 4).
   - Built-in lightweight tracker fallback for unassigned track_ids.
3. Hard-Negative Logging:
   - Optional toggle (log_rejected_detections: false) to save crops of rejected detections for mining.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np


def iou_boxes(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Computes IoU between two [x1, y1, x2, y2] boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class HumanDetectionFilter:
    """
    Filters human detections to reduce false positives from shadows, foliage,
    reflections, and NPU quantization noise.
    """

    def __init__(
        self,
        min_consecutive_frames: int = 4,
        min_box_area: float = 1000.0,
        min_height: float = 40.0,
        max_aspect_ratio: float = 1.5,
        min_aspect_ratio: float = 0.1,
        exclusion_zones: Optional[Dict[str, List[List[List[float]]]]] = None,
        log_rejected_detections: bool = False,
        rejected_dir: str = "data/rejected_detections",
        track_ttl_s: float = 2.0,
    ):
        self.min_consecutive_frames = int(min_consecutive_frames)
        self.min_box_area = float(min_box_area)
        self.min_height = float(min_height)
        self.max_aspect_ratio = float(max_aspect_ratio)
        self.min_aspect_ratio = float(min_aspect_ratio)
        self.exclusion_zones = exclusion_zones or {}
        self.log_rejected_detections = bool(log_rejected_detections)
        self.rejected_dir = rejected_dir
        self.track_ttl_s = float(track_ttl_s)

        # (cam_id, track_id) -> hit_count
        self._consecutive_hits: Dict[Tuple[str, int], int] = {}
        # (cam_id, track_id) -> last_seen_timestamp
        self._last_seen: Dict[Tuple[str, int], float] = {}

        # Fallback tracker state: cam_id -> {next_id, tracks: {id: [x1, y1, x2, y2]}}
        self._fallback_trackers: Dict[str, Dict[str, Any]] = {}

        if self.log_rejected_detections:
            os.makedirs(self.rejected_dir, exist_ok=True)

    def _get_pixel_box(
        self,
        box: List[float],
        frame_w: int,
        frame_h: int,
    ) -> Tuple[float, float, float, float]:
        """Normalizes box coordinates to absolute frame pixel coordinates."""
        if max(box) <= 1.05 and frame_w > 0 and frame_h > 0:
            return (
                float(box[0] * frame_w),
                float(box[1] * frame_h),
                float(box[2] * frame_w),
                float(box[3] * frame_h),
            )
        return float(box[0]), float(box[1]), float(box[2]), float(box[3])

    def check_geometry(
        self,
        pixel_box: Tuple[float, float, float, float],
    ) -> Tuple[bool, Optional[str]]:
        """
        Validates bounding box geometry:
        - min_box_area
        - min_height
        - aspect_ratio (width / height <= max_aspect_ratio, e.g. 1.5)
        """
        x1, y1, x2, y2 = pixel_box
        w = max(0.0, x2 - x1)
        h = max(0.0, y2 - y1)
        area = w * h

        if area < self.min_box_area:
            return False, f"area_too_small ({area:.1f} < {self.min_box_area})"

        if h < self.min_height:
            return False, f"height_too_small ({h:.1f} < {self.min_height})"

        aspect_ratio = w / max(1e-5, h)
        if aspect_ratio > self.max_aspect_ratio:
            return False, f"aspect_ratio_too_wide ({aspect_ratio:.2f} > {self.max_aspect_ratio})"

        if aspect_ratio < self.min_aspect_ratio:
            return False, f"aspect_ratio_too_narrow ({aspect_ratio:.2f} < {self.min_aspect_ratio})"

        return True, None

    def is_in_exclusion_zone(
        self,
        cam_id: str,
        pixel_box: Tuple[float, float, float, float],
        frame_w: int,
        frame_h: int,
    ) -> bool:
        """
        Tests if bottom-center footpoint ((x1 + x2)/2, y2) falls inside any
        configured exclusion polygon for cam_id.
        """
        zones = self.exclusion_zones.get(cam_id)
        if not zones:
            return False

        x1, y1, x2, y2 = pixel_box
        foot_x = (x1 + x2) / 2.0
        foot_y = y2

        for polygon in zones:
            if len(polygon) < 3:
                continue
            poly_pts = []
            for pt in polygon:
                px = float(pt[0] * frame_w) if pt[0] <= 1.05 and frame_w > 0 else float(pt[0])
                py = float(pt[1] * frame_h) if pt[1] <= 1.05 and frame_h > 0 else float(pt[1])
                poly_pts.append([px, py])

            contour = np.array(poly_pts, dtype=np.int32).reshape((-1, 1, 2))
            # cv2.pointPolygonTest returns >= 0 if inside or on boundary
            if cv2.pointPolygonTest(contour, (float(foot_x), float(foot_y)), False) >= 0:
                return True

        return False

    def _assign_fallback_tracks(
        self,
        cam_id: str,
        detections: List[Dict[str, Any]],
        frame_w: int,
        frame_h: int,
    ):
        """Lightweight IoU tracker for detections without upstream track_id."""
        tracker_state = self._fallback_trackers.setdefault(cam_id, {"next_id": 1000, "tracks": {}})
        existing_tracks: Dict[int, np.ndarray] = tracker_state["tracks"]

        unassigned_dets = []
        for det in detections:
            if det.get("track_id") is None:
                unassigned_dets.append(det)

        if not unassigned_dets:
            return

        det_boxes = [
            np.array(self._get_pixel_box(d["bbox"], frame_w, frame_h), dtype=np.float32)
            for d in unassigned_dets
        ]

        assigned_track_ids = set()
        for det, d_box in zip(unassigned_dets, det_boxes):
            best_iou = 0.3
            best_tid = None
            for tid, t_box in existing_tracks.items():
                if tid in assigned_track_ids:
                    continue
                score = iou_boxes(d_box, t_box)
                if score > best_iou:
                    best_iou = score
                    best_tid = tid

            if best_tid is not None:
                det["track_id"] = best_tid
                existing_tracks[best_tid] = d_box
                assigned_track_ids.add(best_tid)
            else:
                new_tid = tracker_state["next_id"]
                tracker_state["next_id"] += 1
                det["track_id"] = new_tid
                existing_tracks[new_tid] = d_box
                assigned_track_ids.add(new_tid)

    def log_rejected_crop(
        self,
        crop: np.ndarray,
        cam_id: str,
        reason: str,
        now: float,
    ) -> None:
        """Saves a cropped image of a rejected detection for hard-negative mining."""
        if not self.log_rejected_detections or crop is None or crop.size == 0:
            return
        try:
            filename = f"{cam_id}_{int(now * 1000)}_{reason}.jpg"
            path = os.path.join(self.rejected_dir, filename)
            cv2.imwrite(path, crop)
        except Exception:
            pass

    def filter_persons(
        self,
        cam_id: str,
        persons: List[Dict[str, Any]],
        frame_size: Tuple[int, int],
        now: Optional[float] = None,
        get_frame: Optional[Callable[[], Optional[np.ndarray]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Filters detections against geometry checks, exclusion zones, and tracker gating.
        Returns only confirmed, valid human detections.
        """
        now = now or time.time()
        fw, fh = frame_size if (frame_size and frame_size[0] > 0) else (1920, 1080)

        # 1. Fallback tracking if track_ids are missing
        self._assign_fallback_tracks(cam_id, persons, fw, fh)

        surviving_persons: List[Dict[str, Any]] = []
        active_tids_this_frame = set()

        for person in persons:
            b = person.get("bbox", [0, 0, 0, 0])
            pixel_box = self._get_pixel_box(b, fw, fh)

            # Check geometry
            is_geom_valid, reason = self.check_geometry(pixel_box)
            if not is_geom_valid:
                if self.log_rejected_detections and get_frame is not None:
                    fr = get_frame()
                    if fr is not None:
                        x1, y1, x2, y2 = map(int, pixel_box)
                        crop = fr[max(0, y1) : min(fh, y2), max(0, x1) : min(fw, x2)]
                        self.log_rejected_crop(crop, cam_id, reason or "geom", now)
                continue

            # Check exclusion zone
            if self.is_in_exclusion_zone(cam_id, pixel_box, fw, fh):
                if self.log_rejected_detections and get_frame is not None:
                    fr = get_frame()
                    if fr is not None:
                        x1, y1, x2, y2 = map(int, pixel_box)
                        crop = fr[max(0, y1) : min(fh, y2), max(0, x1) : min(fw, x2)]
                        self.log_rejected_crop(crop, cam_id, "exclusion_zone", now)
                continue

            # Tracker temporal persistence gating
            tid = person.get("track_id")
            if tid is not None:
                active_tids_this_frame.add(tid)
                key = (cam_id, tid)
                last_t = self._last_seen.get(key, 0.0)
                # If seen recently (within TTL), increment streak
                if now - last_t <= self.track_ttl_s:
                    self._consecutive_hits[key] = self._consecutive_hits.get(key, 0) + 1
                else:
                    self._consecutive_hits[key] = 1
                self._last_seen[key] = now

                hits = self._consecutive_hits[key]
                person["consecutive_hits"] = hits

                if hits >= self.min_consecutive_frames:
                    surviving_persons.append(person)
                else:
                    # Unconfirmed single-frame noise / short-lived tentative detection
                    if self.log_rejected_detections and get_frame is not None:
                        fr = get_frame()
                        if fr is not None:
                            x1, y1, x2, y2 = map(int, pixel_box)
                            crop = fr[max(0, y1) : min(fh, y2), max(0, x1) : min(fw, x2)]
                            self.log_rejected_crop(crop, cam_id, f"tentative_hits_{hits}", now)
            else:
                # If no tracker is present and min_consecutive_frames is 1, let through
                if self.min_consecutive_frames <= 1:
                    surviving_persons.append(person)

        # Cleanup expired tracks from memory
        self.purge_stale_tracks(cam_id, active_tids_this_frame, now)

        return surviving_persons

    def purge_stale_tracks(
        self,
        cam_id: str,
        active_tids: Any,
        now: float,
    ) -> None:
        """Purges tracks that have not been seen for longer than track_ttl_s."""
        active = set(active_tids)
        stale_keys = [
            k
            for k in list(self._last_seen)
            if k[0] == cam_id and k[1] not in active and (now - self._last_seen[k] > self.track_ttl_s)
        ]
        for k in stale_keys:
            self._last_seen.pop(k, None)
            self._consecutive_hits.pop(k, None)

        tracker_state = self._fallback_trackers.get(cam_id)
        if tracker_state:
            tracks = tracker_state.get("tracks", {})
            for tid in list(tracks):
                if tid not in active and (cam_id, tid) not in self._last_seen:
                    tracks.pop(tid, None)
