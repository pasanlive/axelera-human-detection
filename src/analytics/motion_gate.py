"""
Per-Stream MotionGate Background Subtraction Filter.

Implements MOG2 background subtraction to prevent stationary false positives
(furniture, cupboards, static shadows) from appearing or triggering alerts.
Once a track exhibits motion (motion_ratio >= min_motion_ratio), it is confirmed
and remains confirmed even if the person subsequently stands or sits still.
"""

from __future__ import annotations

import time
from typing import Dict, Iterable, List, Optional, Set, Tuple, Union
import cv2
import numpy as np


class MotionGate:
    """
    Per-stream background subtraction motion gate using MOG2.
    
    Attributes:
        history: Number of frames used to model the background (default: 300).
        var_threshold: Mahalanobis distance threshold to decide whether a pixel is well described
                       by the background model (default: 25.0).
        detect_shadows: Whether to detect and mark shadows (default: False).
        min_motion_ratio: Minimum ratio of moving pixels in the bbox ROI to confirm motion (default: 0.03).
        motion_warmup_frames: Number of initial frames for background subtractor to settle before gating (default: 30).
        track_ttl_s: Duration after which unobserved confirmed tracks are pruned (default: 5.0s).
    """

    def __init__(
        self,
        history: int = 300,
        var_threshold: float = 25.0,
        detect_shadows: bool = False,
        min_motion_ratio: float = 0.03,
        motion_warmup_frames: int = 30,
        track_ttl_s: float = 5.0,
    ):
        self.history = int(history)
        self.var_threshold = float(var_threshold)
        self.detect_shadows = bool(detect_shadows)
        self.min_motion_ratio = float(min_motion_ratio)
        self.motion_warmup_frames = int(motion_warmup_frames)
        self.track_ttl_s = float(track_ttl_s)

        self.subtractor = cv2.createBackgroundSubtractorMOG2(
            history=self.history,
            varThreshold=self.var_threshold,
            detectShadows=self.detect_shadows,
        )
        self.confirmed_tracks: Set[int] = set()
        self._last_seen: Dict[int, float] = {}
        self.frame_count: int = 0
        self.current_fg_mask: Optional[np.ndarray] = None

    def update(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Updates the background model with the current frame and returns the foreground mask.
        """
        if frame is None or frame.size == 0:
            return None

        # Convert to gray if BGR to accelerate MOG2 processing
        if frame.ndim == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        self.current_fg_mask = self.subtractor.apply(gray)
        self.frame_count += 1
        return self.current_fg_mask

    def is_warmed_up(self) -> bool:
        """Returns True if the background subtractor has processed enough warmup frames."""
        return self.frame_count >= self.motion_warmup_frames

    def check_track(
        self,
        track_id: Optional[int],
        bbox: Union[List[float], Tuple[float, ...]],
        frame_w: int = 0,
        frame_h: int = 0,
        now: Optional[float] = None,
    ) -> bool:
        """
        Evaluates whether a tracked person passes the motion gate.

        Logic:
        1. If track_id is already in confirmed_tracks, immediately returns True.
        2. During warmup (frame_count < motion_warmup_frames), returns False to allow
           the background model to settle without falsely confirming static noise.
        3. Computes the ratio of non-zero foreground pixels within the bounding box ROI.
        4. If motion_ratio >= min_motion_ratio (0.03), adds track_id to confirmed_tracks and returns True.
        5. Otherwise, returns False (suppresses the static detection).
        """
        now = now or time.time()

        # 1. Once confirmed, a person who sits or stands still stays confirmed
        if track_id is not None and track_id in self.confirmed_tracks:
            self._last_seen[track_id] = now
            return True

        # 2. Warmup check: background model must settle before confirming new tracks
        if not self.is_warmed_up():
            return False

        # 3. Check foreground mask availability
        if self.current_fg_mask is None or self.current_fg_mask.size == 0:
            return False

        h_mask, w_mask = self.current_fg_mask.shape[:2]
        fw = frame_w if frame_w > 0 else w_mask
        fh = frame_h if frame_h > 0 else h_mask

        # Normalize coordinates to pixel dimensions
        if max(bbox) <= 1.05 and fw > 0 and fh > 0:
            bx1 = int(bbox[0] * fw)
            by1 = int(bbox[1] * fh)
            bx2 = int(bbox[2] * fw)
            by2 = int(bbox[3] * fh)
        else:
            bx1, by1, bx2, by2 = map(int, bbox)

        # Scale coordinates if fg_mask size differs from frame size
        if (w_mask, h_mask) != (fw, fh) and fw > 0 and fh > 0:
            sx = float(w_mask) / float(fw)
            sy = float(h_mask) / float(fh)
            x1, y1 = int(bx1 * sx), int(by1 * sy)
            x2, y2 = int(bx2 * sx), int(by2 * sy)
        else:
            x1, y1, x2, y2 = bx1, by1, bx2, by2

        x1 = max(0, min(w_mask, x1))
        y1 = max(0, min(h_mask, y1))
        x2 = max(0, min(w_mask, x2))
        y2 = max(0, min(h_mask, y2))

        if x2 <= x1 or y2 <= y1:
            return False

        roi = self.current_fg_mask[y1:y2, x1:x2]
        total_pixels = roi.size
        if total_pixels == 0:
            return False

        non_zero_pixels = cv2.countNonZero(roi)
        motion_ratio = non_zero_pixels / float(total_pixels)

        if motion_ratio >= self.min_motion_ratio:
            if track_id is not None:
                self.confirmed_tracks.add(track_id)
                self._last_seen[track_id] = now
            return True

        return False

    def prune_stale_tracks(
        self,
        active_track_ids: Iterable[int],
        now: Optional[float] = None,
    ) -> None:
        """
        Removes expired tracks from confirmed_tracks to prevent unbounded memory growth.
        """
        now = now or time.time()
        active = set(active_track_ids)
        for tid in list(self.confirmed_tracks):
            if tid not in active and (now - self._last_seen.get(tid, 0.0) > self.track_ttl_s):
                self.confirmed_tracks.discard(tid)
                self._last_seen.pop(tid, None)

    def reset(self) -> None:
        """Resets background model and track registries."""
        self.subtractor = cv2.createBackgroundSubtractorMOG2(
            history=self.history,
            varThreshold=self.var_threshold,
            detectShadows=self.detect_shadows,
        )
        self.confirmed_tracks.clear()
        self._last_seen.clear()
        self.frame_count = 0
        self.current_fg_mask = None
