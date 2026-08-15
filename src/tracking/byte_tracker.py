"""
ByteTracker: Multi-Object Tracking Module for Consistent Person Tracking Across Frames.
"""

import numpy as np
from typing import List, Dict, Any

class Track:
    """Represents an active object track."""

    def __init__(self, track_id: int, bbox: List[float], score: float):
        self.track_id = track_id
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.score = score
        self.hits = 1
        self.time_since_update = 0

    def update(self, bbox: List[float], score: float):
        self.bbox = bbox
        self.score = score
        self.hits += 1
        self.time_since_update = 0

class ByteTracker:
    """Lightweight IOU ByteTrack implementation for object tracking."""

    def __init__(self, track_thresh: float = 0.5, max_age: int = 30):
        self.track_thresh = track_thresh
        self.max_age = max_age
        self.tracks: List[Track] = []
        self.next_id = 1

    def update(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Updates tracks with new frame detections.
        :param detections: List of dicts with 'bbox' [x1, y1, x2, y2] and 'confidence'
        :return: Detections updated with assigned 'track_id'
        """
        # Increment age for all active tracks
        for t in self.tracks:
            t.time_since_update += 1

        if len(detections) == 0:
            # Remove stale tracks
            self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
            return []

        det_boxes = [d["bbox"] for d in detections]

        if len(self.tracks) == 0:
            # Initialize new tracks for all detections
            for d in detections:
                t = Track(self.next_id, d["bbox"], d["confidence"])
                self.next_id += 1
                self.tracks.append(t)
                d["track_id"] = t.track_id
            return detections

        # Compute IOU matrix between existing tracks and new detections
        track_boxes = [t.bbox for t in self.tracks]
        iou_matrix = self._compute_iou_matrix(track_boxes, det_boxes)

        matched_tracks = set()
        matched_dets = set()

        # Greedy match high IOU pairs
        if iou_matrix.size > 0:
            for t_idx in range(len(self.tracks)):
                d_idx = np.argmax(iou_matrix[t_idx])
                max_iou = iou_matrix[t_idx, d_idx]
                if max_iou >= 0.3 and d_idx not in matched_dets:
                    self.tracks[t_idx].update(detections[d_idx]["bbox"], detections[d_idx]["confidence"])
                    detections[d_idx]["track_id"] = self.tracks[t_idx].track_id
                    matched_tracks.add(t_idx)
                    matched_dets.add(d_idx)

        # Create new tracks for unmatched detections above threshold
        for d_idx, d in enumerate(detections):
            if d_idx not in matched_dets and d["confidence"] >= self.track_thresh:
                t = Track(self.next_id, d["bbox"], d["confidence"])
                self.next_id += 1
                self.tracks.append(t)
                d["track_id"] = t.track_id

        # Remove stale tracks
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        return detections

    def _compute_iou_matrix(self, boxes1: List[List[float]], boxes2: List[List[float]]) -> np.ndarray:
        """Calculates pairwise IOU matrix between two lists of bounding boxes [x1, y1, x2, y2]."""
        matrix = np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)
        for i, b1 in enumerate(boxes1):
            for j, b2 in enumerate(boxes2):
                matrix[i, j] = self._iou(b1, b2)
        return matrix

    @staticmethod
    def _iou(b1: List[float], b2: List[float]) -> float:
        x1 = max(b1[0], b2[0])
        y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2])
        y2 = min(b1[3], b2[3])

        inter_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        b1_area = (b1[2] - b1[0]) * (b1[3] - b1[1])
        b2_area = (b2[2] - b2[0]) * (b2[3] - b2[1])

        union_area = b1_area + b2_area - inter_area
        return inter_area / union_area if union_area > 0 else 0.0
