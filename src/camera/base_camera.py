"""
Base Camera Abstraction for OpenCV / RTSP / Synthetic Video Feeds.
"""

import time
import cv2
import numpy as np
from typing import Tuple, Optional

class BaseCamera:
    """Abstract Base Class for Camera Streams."""

    def __init__(self, camera_id: str, name: str, source: str, fps_limit: int = 30):
        self.camera_id = camera_id
        self.name = name
        self.source = source
        self.fps_limit = fps_limit
        self.is_running = False
        self.frame_count = 0
        self.fps = 0.0
        self._last_time = time.time()

    def open(self) -> bool:
        raise NotImplementedError

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        raise NotImplementedError

    def close(self):
        raise NotImplementedError

    def _update_fps(self):
        self.frame_count += 1
        curr_time = time.time()
        elapsed = curr_time - self._last_time
        if elapsed >= 1.0:
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self._last_time = curr_time

class OpenCVCamera(BaseCamera):
    """OpenCV VideoCapture Implementation for Webcams, RTSP streams, and Video Files."""

    def __init__(self, camera_id: str, name: str, source: str, fps_limit: int = 30):
        super().__init__(camera_id, name, source, fps_limit)
        self.cap: Optional[cv2.VideoCapture] = None

    def open(self) -> bool:
        src = int(self.source) if self.source.isdigit() else self.source
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            print(f"[CAMERA ERROR] Failed to open camera stream {self.camera_id} ({self.source})")
            self.is_running = False
            return False
        
        self.is_running = True
        print(f"[CAMERA OPENED] Camera '{self.name}' ({self.camera_id}) opened successfully.")
        return True

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        if not self.is_running or self.cap is None:
            return False, None

        ret, frame = self.cap.read()
        if not ret or frame is None:
            # Handle end of video file / stream drop reconnect
            if isinstance(self.source, str) and not self.source.isdigit() and not self.source.startswith("rtsp"):
                # Loop video file
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
            
            if not ret or frame is None:
                return False, None

        self._update_fps()
        return True, frame

    def close(self):
        self.is_running = False
        if self.cap:
            self.cap.release()
            self.cap = None

class SyntheticCamera(BaseCamera):
    """Synthetic Camera generator for testing/demo when no physical camera is attached."""

    def __init__(self, camera_id: str, name: str, source: str = "synthetic", fps_limit: int = 30, width: int = 1280, height: int = 720):
        super().__init__(camera_id, name, source, fps_limit)
        self.width = width
        self.height = height
        self.angle = 0.0

    def open(self) -> bool:
        self.is_running = True
        print(f"[SYNTHETIC CAMERA] Opened synthetic test stream '{self.name}'.")
        return True

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        if not self.is_running:
            return False, None

        # Create dynamic animated test frame with synthetic human shape for testing
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        # Background gradient
        frame[:, :] = (30, 30, 40)

        # Draw grid
        for x in range(0, self.width, 80):
            cv2.line(frame, (x, 0), (x, self.height), (45, 45, 55), 1)
        for y in range(0, self.height, 80):
            cv2.line(frame, (0, y), (self.width, y), (45, 45, 55), 1)

        # Draw synthetic moving person figure
        self.angle += 0.05
        cx = int(self.width / 2 + np.sin(self.angle) * 300)
        cy = int(self.height / 2 + np.cos(self.angle * 0.7) * 100)

        # Head
        cv2.circle(frame, (cx, cy - 120), 30, (220, 200, 180), -1)
        # Eyes
        cv2.circle(frame, (cx - 10, cy - 125), 4, (20, 20, 20), -1)
        cv2.circle(frame, (cx + 10, cy - 125), 4, (20, 20, 20), -1)
        # Body / Torso
        cv2.rectangle(frame, (cx - 40, cy - 90), (cx + 40, cy + 50), (180, 100, 50), -1)
        # Arms
        cv2.line(frame, (cx - 40, cy - 80), (cx - 80, cy - 10), (180, 100, 50), 12)
        cv2.line(frame, (cx + 40, cy - 80), (cx + 80, cy - 10), (180, 100, 50), 12)
        # Legs
        cv2.line(frame, (cx - 20, cy + 50), (cx - 30, cy + 160), (50, 50, 180), 14)
        cv2.line(frame, (cx + 20, cy + 50), (cx + 30, cy + 160), (50, 50, 180), 14)

        # Label stream
        cv2.putText(frame, f"{self.name} ({self.camera_id}) - SYNTHETIC STREAM", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        time.sleep(1.0 / self.fps_limit)
        self._update_fps()
        return True, frame

    def close(self):
        self.is_running = False
